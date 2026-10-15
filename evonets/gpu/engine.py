"""Il motore di simulazione su GPU (JAX).

Fa esattamente ciò che fanno `World.step`, `NeuralController.act`,
`Prepared.respond` e `Brains.step` (il motore NumPy, che resta come
riferimento nei test), ma:

- tutto il passo è compilato in poche istruzioni per la GPU;
- un'intera generazione gira sulla GPU senza tornare su Python;
- più mondi (arene, semi) girano insieme nello stesso lotto (`jax.vmap`).

Differenze volute rispetto al motore NumPy:

- numeri a 32 bit (le schede video sono molto più lente a 64 bit);
- numeri casuali di JAX: ripetibili con lo stesso seme, ma diversi da quelli
  del motore NumPy;
- tutti gli agenti si calcolano sempre e chi non è in gara resta fermo grazie
  a una maschera (il motore NumPy invece salta le sue righe): per chi è in gara
  il risultato è lo stesso.

La genetica, la compilazione dei geni (`Brains.compile`, `Prepared`) e la
preparazione delle arene restano su CPU, una volta per generazione: qui
arrivano già come tabelle di numeri.

Struttura: `params` sono le costanti di una generazione (geni compilati,
arena), `phys` le costanti della fisica (modificabili dal vivo senza
ricompilare), `state` lo stato che cambia a ogni passo. Tutti sono dizionari
di array; la prima dimensione, se c'è il lotto, è il mondo.
"""

import jax
import jax.numpy as jnp
import numpy as np

F = jnp.float32

# Costanti della configurazione usate dal passo: passano come numeri (non
# compilati dentro), così cambiarle dal vivo non richiede di ricompilare.
PHYS = (
    "arena_w", "arena_h", "dt", "agent_radius", "thrust_accel", "reverse_factor", "lin_drag", "lat_drag",
    "turn_accel", "ang_drag", "wall_restitution", "energy_init", "metabolism_basal", "metabolism_move",
    "metabolism_turn", "metabolism_exponent", "light_attenuation", "hazard_drain", "hazard_falloff", "hazard_pulse", "hazard_speed",
    "toxin_molecule", "hazard_smell", "target_molecule", "target_smell", "smell_range", "vibration_range",
    "target_vibration", "target_vibration_freq", "hazard_vibration", "hazard_vibration_freq",
    "ambient_temperature", "heat_range", "target_heat", "hazard_heat", "receptor_dark", "receptor_saturation",
    "sensor_noise", "synapse_max", "spike_gain", "life_max",
)


def phys_of(cfg) -> dict:
    p = {k: np.float32(getattr(cfg, k)) for k in PHYS}
    a = np.radians(cfg.magnetic_angle_deg)
    p["magnetic"] = (cfg.magnetic_field * np.array([np.cos(a), np.sin(a)])).astype(np.float32)
    return p


def _tuning(q, mu, sigma):
    return jnp.exp(-0.5 * ((q - mu) / sigma) ** 2)


def _rotate(vx, vy, c, s):
    return vx * c - vy * s, vx * s + vy * c


# --- Sensi -----------------------------------------------------------------


def light(ph, ox, oy, ang, hz_pos, hz_r, target, target_r, has_target):
    """Come `signals.light`: per ogni recettore (K) e direzione campione (M),
    luce delle pareti, del bersaglio e del veleno, mediata sulle direzioni."""
    W, H, L = ph["arena_w"], ph["arena_h"], ph["light_attenuation"]
    dx, dy = jnp.cos(ang), jnp.sin(ang)  # (K, M)
    ox, oy = ox[:, None], oy[:, None]
    safe_dx = jnp.where(dx != 0, dx, 1.0)
    safe_dy = jnp.where(dy != 0, dy, 1.0)
    tx = jnp.where(dx > 0, (W - ox) / safe_dx, jnp.where(dx < 0, -ox / safe_dx, jnp.inf))
    ty = jnp.where(dy > 0, (H - oy) / safe_dy, jnp.where(dy < 0, -oy / safe_dy, jnp.inf))
    d_hit = jnp.maximum(jnp.minimum(tx, ty), 0.0)
    fx, fy = target[0] - ox, target[1] - oy
    proj = fx * dx + fy * dy
    disc = target_r * target_r - (fx * fx + fy * fy) + proj * proj
    half = jnp.sqrt(jnp.maximum(disc, 0.0))
    d_t = jnp.where((disc >= 0) & (proj + half >= 0) & (has_target > 0), jnp.maximum(proj - half, 0.0), jnp.inf)
    hit_t = d_t < d_hit
    d_hit = jnp.minimum(d_hit, d_t)
    fade = jnp.exp(-d_hit / L)
    # Nubi di veleno: (Hz, K, M).
    cx = hz_pos[:, 0, None, None] - ox[None]
    cy = hz_pos[:, 1, None, None] - oy[None]
    pr = cx * dx[None] + cy * dy[None]
    dc = hz_r[:, None, None] ** 2 - (cx * cx + cy * cy) + pr * pr
    hh = jnp.sqrt(jnp.maximum(dc, 0.0))
    s_in = jnp.minimum(jnp.maximum(pr - hh, 0.0), d_hit[None])
    s_out = jnp.minimum(pr + hh, d_hit[None])
    ok = (dc >= 0) & (pr + hh >= 0) & (s_out > s_in)
    glow = jnp.where(ok, L * (jnp.exp(-s_in / L) - jnp.exp(-s_out / L)), 0.0).sum(0)
    return (jnp.where(hit_t, 0.0, fade).mean(1), jnp.where(hit_t, fade, 0.0).mean(1), glow.mean(1))


def quanta(ph, P, s):
    """Come `Prepared.quanta`: (N, R) quanti attesi di ogni recettore."""
    N, R = P["active"].shape
    c, sn = jnp.cos(s["heading"])[:, None], jnp.sin(s["heading"])[:, None]
    nx, ny = _rotate(P["body"][..., 0], P["body"][..., 1], c, sn)
    r = ph["agent_radius"]
    px = (s["pos"][:, 0, None] + r * nx).reshape(-1)
    py = (s["pos"][:, 1, None] + r * ny).reshape(-1)
    hz_pos, hz_r = hazard_shape(ph, P, s)
    ang = (s["heading"][:, None, None] + P["angles"]).reshape(N * R, -1)
    wall, targ, glow = light(ph, px, py, ang, hz_pos, hz_r, s["target"], P["target_r"], P["has_target"])
    out = (P["c_wall"] * wall.reshape(N, R) + P["c_target"] * targ.reshape(N, R)
           + P["c_glow"] * glow.reshape(N, R))
    # Sostanze: tossina (dentro: intensità; fuori: odore se c'è), odore del bersaglio.
    ex = px[:, None] - hz_pos[None, :, 0]
    ey = py[:, None] - hz_pos[None, :, 1]
    edge = jnp.sqrt(ex * ex + ey * ey) - hz_r[None]
    inside = jnp.clip(-edge / hz_r[None], 0.0, 1.0) ** ph["hazard_falloff"]
    outside = ph["hazard_smell"] * jnp.exp(-jnp.maximum(edge, 0.0) / ph["smell_range"])
    tox = jnp.where(edge < 0, inside, jnp.where(ph["hazard_smell"] > 0, outside, 0.0)).sum(1).reshape(N, R)
    out += P["g_c"] * _tuning(ph["toxin_molecule"], P["mu_c"], P["sg_c"]) * tox
    dt_ = jnp.sqrt((px - s["target"][0]) ** 2 + (py - s["target"][1]) ** 2) - P["target_r"]
    smell = ph["target_smell"] * jnp.exp(-jnp.maximum(dt_, 0.0) / ph["smell_range"]) * P["has_target"]
    out += P["g_c"] * _tuning(ph["target_molecule"], P["mu_c"], P["sg_c"]) * smell.reshape(N, R)
    # Pressione sui lati che toccano una parete.
    lo = s["pos"] <= r
    hi = s["pos"] >= jnp.stack([ph["arena_w"] - r, ph["arena_h"] - r])
    press = (lo[:, None, 0] * jnp.maximum(0.0, -nx) + hi[:, None, 0] * jnp.maximum(0.0, nx)
             + lo[:, None, 1] * jnp.maximum(0.0, -ny) + hi[:, None, 1] * jnp.maximum(0.0, ny))
    out += P["c_press"] * press
    # Vibrazioni (bersaglio e veleno), dentro l'apertura del recettore.
    centre = s["heading"][:, None] + P["centre"]

    def heard(cx, cy, amp, freq):
        vx, vy = cx - px.reshape(N, R), cy - py.reshape(N, R)
        d = jnp.sqrt(vx * vx + vy * vy)
        off = jnp.abs((jnp.arctan2(vy, vx) - centre + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        return P["g_m"] * _tuning(freq, P["mu_m"], P["sg_m"]) * amp * jnp.exp(-d / ph["vibration_range"]) * (
            off <= P["half"])

    out += heard(s["target"][0], s["target"][1], ph["target_vibration"] * P["has_target"],
                 ph["target_vibration_freq"])
    hv = jax.vmap(lambda c: heard(c[0], c[1], ph["hazard_vibration"], ph["hazard_vibration_freq"]))(hz_pos)
    out += hv.sum(0)
    # Temperatura: ambiente più le sorgenti di calore.
    temp = ph["ambient_temperature"] + ph["target_heat"] * jnp.exp(-jnp.maximum(dt_, 0.0) / ph["heat_range"]) * \
        P["has_target"]
    temp = temp + (ph["hazard_heat"] * jnp.exp(-jnp.maximum(edge, 0.0) / ph["heat_range"])).sum(1)
    temp = jnp.clip(temp, 0.0, 1.0).reshape(N, R)
    out += P["g_t"] * _tuning(temp, P["mu_t"], P["sg_t"])
    # Magnetismo: bussola.
    out += P["g_mag"] * jnp.maximum(0.0, jnp.cos(centre) * ph["magnetic"][0] + jnp.sin(centre) * ph["magnetic"][1])
    # Interno: le grandezze del corpo.
    out += jnp.einsum("nrk,nk->nr", P["c_body"], body(ph, s))
    return out


def body(ph, s):
    """Come `signals.body`: (N, 7) grandezze del corpo in [0, 1]."""
    c, sn = jnp.cos(s["heading"]), jnp.sin(s["heading"])
    vx, vy = s["vel"][:, 0], s["vel"][:, 1]
    v_max = ph["thrust_accel"] / ph["lin_drag"]
    w_max = ph["turn_accel"] / ph["ang_drag"]
    fwd, lat, rot = (vx * c + vy * sn) / v_max, (vy * c - vx * sn) / v_max, s["ang_vel"] / w_max

    def pos(x):
        return jnp.clip(x, 0.0, 1.0)

    return jnp.stack([jnp.clip(s["energy"] / ph["energy_init"], 0.0, 1.0),
                      pos(fwd), pos(-fwd), pos(lat), pos(-lat), pos(rot), pos(-rot)], 1)


def respond(ph, P, s, key, racing):
    """Come `Prepared.respond` + il disturbo `sensor_noise`: risposte con
    rumore, adattamento e saturazione. Chi non è in gara tiene risposta e
    media di prima."""
    n = quanta(ph, P, s)
    k1, k2 = jax.random.split(key)
    noisy = jnp.maximum(n + jnp.sqrt(n + ph["receptor_dark"]) * jax.random.normal(k1, n.shape, F), 0.0)
    m = jnp.where(jnp.isnan(s["m"]), noisy, s["m"])
    r = jnp.clip((noisy - P["sub"] * m) / (ph["receptor_saturation"] + P["div"] * m), 0.0, 1.0) * P["active"]
    r = jnp.where(ph["sensor_noise"] > 0,
                  jnp.clip(r + ph["sensor_noise"] * jax.random.normal(k2, r.shape, F), -1.0, 1.0), r)
    m = P["keep"] * m + (1.0 - P["keep"]) * noisy
    on = racing[:, None] > 0
    return jnp.where(on, r, s["obs"]), jnp.where(on, m, s["m"])


# --- Reti ------------------------------------------------------------------


def learn(ph, P, w, pre, post, after_hidden, racing):
    """Come `Brains._learn`: plasticità modulata, solo per chi impara ed è in gara."""
    M = jnp.einsum("nth,nh->nt", P["kmod"], after_hidden * P["mod_sign"])
    rule = P["rule"]
    drive = ((rule[..., 0] * post)[:, :, None] * pre[:, None, :] + rule[..., 1, None] * pre[:, None, :]
             + (rule[..., 2] * post + rule[..., 3])[:, :, None])
    nw = jnp.clip(w + (P["eta"] * M)[:, :, None] * drive * P["links"], -ph["synapse_max"], ph["synapse_max"])
    sign = P["dale"][:, None, :]
    nw = jnp.where(sign > 0, jnp.maximum(nw, 0.0), jnp.where(sign < 0, jnp.minimum(nw, 0.0), nw))
    on = (P["learns"] * racing)[:, None, None] > 0
    return jnp.where(on, nw, w)


def rate_brain(ph, P, s, obs, racing, n_hidden):
    """Come `Brains.step`: rete continua più plasticità."""
    before = s["state"]
    x = jnp.concatenate([obs, before], 1)
    after = jnp.tanh(jnp.einsum("nts,ns->nt", s["w"], x) + P["bias"]) * P["target_on"]
    on = racing[:, None] > 0
    after = jnp.where(on, after, before)
    w = learn(ph, P, s["w"], x, after, after[:, :n_hidden], racing)
    return dict(state=after, w=w), after[:, n_hidden:]


def spike_brain(ph, P, s, obs, racing, n_hidden, substeps):
    """Come `SpikingBrains.step`: neuroni a impulsi più plasticità."""
    h = n_hidden
    keep, drive = P["keep_n"], 1.0 - P["keep_n"]
    act = P["active_n"]
    charge, spikes, state = s["charge"], s["spikes"], s["state"]
    for _ in range(substeps):
        x = jnp.concatenate([obs, spikes, state[:, h:]], 1)
        current = jnp.einsum("nts,ns->nt", s["w"], x) + P["bias"]
        current = current.at[:, :h].multiply(ph["spike_gain"])
        charge = jnp.maximum(keep * charge + drive * current, -1.0)
        inner = charge[:, :h]
        fired = (inner >= 1.0) & (act > 0)
        inner = jnp.where(fired | (act <= 0), 0.0, inner)
        charge = charge.at[:, :h].set(inner)
        spikes = fired.astype(F)
        state = state.at[:, :h].set(0.8 * state[:, :h] + 0.2 * spikes)
        state = state.at[:, h:].set(jnp.tanh(jnp.maximum(charge[:, h:], 0.0)))
    on = racing[:, None] > 0
    charge = jnp.where(on, charge, s["charge"])
    spikes = jnp.where(on, spikes, s["spikes"])
    state = jnp.where(on, state, s["state"])
    pre = jnp.concatenate([obs, state], 1)
    w = learn(ph, P, s["w"], pre, state, state[:, :h], racing)
    m = state[:, h:]
    out = jnp.stack([m[:, 0] - m[:, 1], m[:, 2] - m[:, 3]], 1)
    return dict(charge=charge, spikes=spikes, state=state, w=w), out


# --- Mondo -----------------------------------------------------------------


def hazard_shape(ph, P, s):
    pos = P["anchor"] + P["amp"] * jnp.sin(s["phase"])
    rad = P["base_radius"] * (1.0 + ph["hazard_pulse"] * jnp.sin(s["pulse_phase"]))
    return pos, rad


def physics(ph, P, s, actions, upkeep):
    """Come `World.step`: movimento, pareti, veleno, energia, compito."""
    dt, r = ph["dt"], ph["agent_radius"]
    act = jnp.clip(actions, -1.0, 1.0)
    thrust = jnp.where(act[:, 0] < 0.0, act[:, 0] * ph["reverse_factor"], act[:, 0])
    turn = act[:, 1]
    ang_vel = (s["ang_vel"] + turn * ph["turn_accel"] * dt) * jnp.exp(-ph["ang_drag"] * dt)
    heading = s["heading"] + ang_vel * dt
    heading = (heading + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    fx, fy = jnp.cos(heading), jnp.sin(heading)
    vel = s["vel"]
    v_fwd = vel[:, 0] * fx + vel[:, 1] * fy
    lx, ly = vel[:, 0] - v_fwd * fx, vel[:, 1] - v_fwd * fy
    v_fwd = (v_fwd + thrust * ph["thrust_accel"] * dt) * jnp.exp(-ph["lin_drag"] * dt)
    k = jnp.exp(-ph["lat_drag"] * dt)
    vel = jnp.stack([v_fwd * fx + lx * k, v_fwd * fy + ly * k], 1)
    active = (s["alive"] > 0) & (s["done"] <= 0)
    vel = jnp.where(active[:, None], vel, 0.0)
    ang_vel = jnp.where(active, ang_vel, 0.0)
    pos = s["pos"] + vel * dt
    lo = jnp.stack([r, r])
    hi = jnp.stack([ph["arena_w"] - r, ph["arena_h"] - r])
    hit = ((pos < lo) & (vel < 0.0)) | ((pos > hi) & (vel > 0.0))
    vel = jnp.where(hit, vel * -ph["wall_restitution"], vel)
    pos = jnp.clip(pos, lo, hi)
    bump = ((pos <= lo) | (pos >= hi)).any(1)
    # Veleno: si muove, poi danneggia.
    k_h = ph["hazard_speed"] * dt
    phase = s["phase"] + P["omega"] * k_h
    pulse = s["pulse_phase"] + P["pulse_omega"] * k_h
    hz_pos = P["anchor"] + P["amp"] * jnp.sin(phase)
    hz_r = P["base_radius"] * (1.0 + ph["hazard_pulse"] * jnp.sin(pulse))
    d = jnp.sqrt(((pos[:, None, :] - hz_pos[None]) ** 2).sum(-1)) - hz_r[None]
    toxicity = (jnp.clip(-d / hz_r[None], 0.0, 1.0) ** ph["hazard_falloff"]).sum(1) * active
    effort = (ph["metabolism_basal"] + ph["metabolism_move"] * jnp.abs(act[:, 0]) ** ph["metabolism_exponent"]
              + ph["metabolism_turn"] * jnp.abs(turn) ** ph["metabolism_exponent"] + upkeep) * active
    energy = jnp.maximum(s["energy"] - (effort + ph["hazard_drain"] * toxicity) * dt, 0.0)
    alive = (s["alive"] > 0) & (energy > 0.0)
    # Il bersaglio (se si muove) e chi lo raggiunge.
    target, tvel = s["target"] + s["target_vel"] * dt, s["target_vel"]
    tr = P["target_r"]
    t_hi = jnp.stack([ph["arena_w"] - tr, ph["arena_h"] - tr])
    out = (target < tr) | (target > t_hi)
    tvel = jnp.where(out, -tvel, tvel)
    target = jnp.clip(target, tr, t_hi)
    reached = jnp.sqrt(((pos - target) ** 2).sum(1)) <= tr + r
    done = (s["done"] > 0) | (reached & alive & (P["has_target"] > 0))
    t = s["t"] + dt
    done_t = jnp.where(done & (s["done"] <= 0), t, s["done_t"])  # quando ha completato
    return dict(pos=pos, vel=vel, heading=heading, ang_vel=ang_vel, energy=energy, alive=alive.astype(F),
                done=done.astype(F), bump=bump.astype(F), pain=jnp.minimum(toxicity, 1.0), phase=phase,
                pulse_phase=pulse, target=target, target_vel=tvel, t=t, done_t=done_t)


def make_step(net_type: str, n_hidden: int, substeps: int):
    """Un passo di un mondo: sensi, rete, fisica. `manual` = (agente, comandi,
    attivo) sostituisce i comandi di un agente (guida a mano)."""

    def step(ph, P, s, manual):
        racing = ((s["alive"] > 0) & (s["done"] <= 0)).astype(F)
        key, sub = jax.random.split(s["key"])
        obs, m = respond(ph, P, s, sub, racing)
        if net_type == "rate":
            brain, out = rate_brain(ph, P, s, obs, racing, n_hidden)
        else:
            brain, out = spike_brain(ph, P, s, obs, racing, n_hidden, substeps)
        actions = out * racing[:, None]
        idx, cmd, on = manual
        actions = jnp.where((on > 0) & (jnp.arange(actions.shape[0])[:, None] == idx), cmd[None, :], actions)
        world = physics(ph, P, s, actions, P["upkeep"])
        return {**s, **world, **brain, "obs": obs, "m": m, "key": key}

    return step


def running(ph, s):
    """Il mondo è ancora in corso? (qualcuno in gara e tempo non finito)."""
    return ((s["alive"] > 0) & (s["done"] <= 0)).any() & (s["t"] < ph["life_max"])
