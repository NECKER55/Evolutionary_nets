"""L'arena delle pile e i suoi cervelli sulla GPU (JAX): lo stesso algoritmo di
`sim/pile.py` e `net/sections.py`, scritto come funzioni pure e compilato,
con molti passi alla volta (`lax.scan`). I numeri casuali sono diversi da
quelli della versione NumPy, quindi le corse non coincidono passo per passo:
coincidono le regole (tests/test_pile.py le confronta pezzo per pezzo e
nelle medie).

Stato: un dizionario di array; quello del mondo ha come prime dimensioni
(recinto, essere), quello del cervello (essere,) con gli esseri in fila
(recinto · n + essere), come nella versione NumPy.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..net import sections as S
from ..sim import pile as P

F = jnp.float32


def _wrap(a):
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


# --- Il mondo ------------------------------------------------------------------------


def init_world(cfg: P.PileConfig, B: int, seed: int) -> dict:
    """Lo stato di partenza, preso dalla versione NumPy (stesso seme, stessa partenza)."""
    w = P.PileWorld(cfg, B, seed)
    n, K = w.n, cfg.batteries
    return dict(pos=w.pos, heading=w.heading, speed=w.speed, energy=w.energy, age=w.age, charging=w.charging,
                taste=w.taste, pain=w.pain, good=w.good.astype(np.int32), zone=w.zone.astype(np.int32),
                bpos=w.bpos, bcolor=w.bcolor.astype(np.int32), bbig=w.bbig.astype(np.int32), bon=w.bon,
                bwait=w.bwait, bage=w.bage,
                t=np.float32(0.0), next_inv=np.float32(cfg.inversion_every), next_zone=np.float32(cfg.zone_every),
                pair_time=w.pair_time, rest=w.rest, ghost=w.ghost, ghost_hide=w.ghost_hide,
                wander=np.zeros((B, n)), escape=np.zeros((B, n)),
                f_in=np.zeros((B, n, P.N_FIN)), f_act=np.zeros((B, n, P.N_FOUT)))


def zone_centres(cfg):
    m, s = cfg.zone_margin, cfg.size
    return jnp.array([[m, m], [s - m, m], [m, s - m], [s - m, s - m]], F)


def senses(cfg: P.PileConfig, w: dict) -> jnp.ndarray:
    """(B, n, N_SENSES): la stessa funzione della versione NumPy (`sim.pile.sense`)."""
    return P.sense(cfg, w["pos"], w["heading"], w["bpos"], w["bcolor"], w["bon"], w["energy"], w["taste"], w["pain"],
                   xp=jnp, ghost=w["ghost"], ghost_hide=w["ghost_hide"])


def motor(program, raw, has, near, ang, wander, escape, W, key):
    """F: lo stesso circuito della versione NumPy (`sim.pile.motor_net`).
    Restituisce (spinta, sterzo, deriva, fuga, ingressi, attività)."""
    wander = 0.9 * wander + 0.3 * jax.random.normal(key, wander.shape, F)
    rays = raw[..., P.RAY]
    escape = P.escape_reflex(rays, escape, jnp)
    x = P.motor_inputs(program, has, near, ang, wander, escape, rays, jnp)
    thrust, steer, act = P.motor_net(x, W, jnp)
    return thrust, steer, wander, escape, x, act


def objects(cfg, w):
    good = P.battery_good(cfg, w["bcolor"], w["bbig"], w["good"])
    return P.objects_of(w["pos"], w["heading"], w["speed"], w["bpos"], w["bcolor"], w["bbig"], w["bon"], good, cfg,
                        xp=jnp, ghost=w["ghost"], ghost_hide=w["ghost_hide"])


def world_step(cfg: P.PileConfig, w: dict, thrust, steer, bite, key):
    """Un passo del mondo: come `PileWorld.step`. Restituisce (stato, uscite)."""
    dt = cfg.dt
    B, n = thrust.shape
    K = cfg.batteries
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    w = dict(w)
    charging = w["charging"]
    thrust = jnp.where(charging > 0.0, 0.0, jnp.clip(thrust, -1.0, 1.0))
    heading = _wrap(w["heading"] + jnp.clip(steer, -1.0, 1.0) * cfg.turn * dt)
    speed = thrust * cfg.speed * jnp.where(thrust < 0, cfg.reverse, 1.0)
    u = jnp.stack([jnp.cos(heading), jnp.sin(heading)], -1)
    pos = jnp.clip(w["pos"] + u * speed[..., None] * dt, cfg.radius, cfg.size - cfg.radius)
    age = w["age"] + dt
    spent = (cfg.drain + cfg.move_cost * jnp.abs(thrust) + cfg.bite_cost * bite) * dt
    energy = w["energy"] - spent
    fade = np.exp(-dt / cfg.taste_tau)
    taste, pain = w["taste"] * fade, w["pain"] * fade
    moving = jnp.abs(speed) > cfg.take_speed * cfg.speed
    charging = jnp.where(moving, 0.0, charging)
    gain = jnp.minimum(charging, cfg.good_energy * dt / cfg.charge_time)
    charging = charging - gain
    energy = jnp.minimum(energy + gain, cfg.energy_max)
    pleasure = gain / cfg.good_energy
    taste = jnp.minimum(taste + gain / cfg.good_energy * 4.0, 1.0)

    # Pile: scadono, ricompaiono nella zona attiva.
    bon, bwait, bage = w["bon"], w["bwait"] - dt, w["bage"] + dt
    ghost = w["ghost"]
    ghost_hide = jnp.maximum(w["ghost_hide"] - dt, 0.0)
    old = bon & (bage >= cfg.battery_life)
    bon = bon & ~old
    bwait = jnp.where(old, cfg.respawn, bwait)
    spawn = ~bon & (bwait <= 0.0)
    centre = zone_centres(cfg)[w["zone"]][:, None, :]
    lo, hi = cfg.battery_radius, cfg.size - cfg.battery_radius
    new_pos = jnp.clip(centre + jax.random.uniform(k1, (B, K, 2), F, -cfg.zone_half, cfg.zone_half), lo, hi)
    bpos = jnp.where(spawn[..., None], new_pos, w["bpos"])
    k2a, k2b = jax.random.split(k2)
    bcolor = jnp.where(spawn, jax.random.randint(k2a, (B, K), 0, 2), w["bcolor"])
    bbig = jnp.where(spawn, jax.random.randint(k2b, (B, K), 0, 2), w["bbig"])
    bage = jnp.where(spawn, jax.random.uniform(k3, (B, K), F, 0.0, 0.5 * cfg.battery_life), bage)
    bon = bon | spawn
    ghost_hide = jnp.where(spawn, 0.0, ghost_hide)  # una pila nuova: la vede anche il fantasma

    # Prese: ogni pila al primo essere che può prenderla, ogni essere al più una pila.
    d = bpos[:, None] - pos[:, :, None]
    front = jnp.abs(_wrap(jnp.arctan2(d[..., 1], d[..., 0]) - heading[:, :, None])) < np.radians(cfg.mouth)
    bon_each, _ = P.ghost_view(bon, ghost, ghost_hide, jnp)
    close = (jnp.hypot(d[..., 0], d[..., 1]) < cfg.radius + cfg.battery_radius + cfg.bite_reach) \
        & bon_each & front
    free = ~moving & (charging <= 0.0) & bite
    # Il fantasma mangia la sua copia: la pila sparisce solo per lui.
    g_can = close & (free & ghost)[..., None]
    g_k = jnp.argmax(g_can, axis=2)
    g_took = g_can.any(2)
    ghost_hide = jnp.where((g_can & (jnp.arange(K)[None, None] == g_k[..., None])).any(1), cfg.battery_life,
                           ghost_hide)
    can = close & (free & ~ghost)[..., None]
    first = jnp.argmax(can, axis=1)
    claim = (jnp.arange(n)[None, :, None] == first[:, None, :]) & can.any(1)[:, None, :]
    take = claim & (jnp.cumsum(claim, axis=2) == 1)
    took = take.any(2) | g_took
    kk = jnp.where(g_took, g_k, jnp.argmax(take, axis=2))
    gone = take.any(1)
    bon = bon & ~gone
    bwait = jnp.where(gone, cfg.respawn, bwait)
    ok = jnp.take_along_axis(P.battery_good(cfg, bcolor, bbig, w["good"]), kk, axis=1)
    charging = jnp.where(took & ok, cfg.good_energy, charging)
    bad = took & ~ok
    energy = jnp.where(bad, energy - cfg.bad_energy, energy)
    pain = jnp.where(bad, 1.0, pain)
    hurt = jnp.where(bad, cfg.bad_energy / cfg.good_energy, 0.0)

    # Accoppiamento: coppie che restano muso contro muso; ognuno con il suo
    # compagno preferito (il tempo più lungo), solo se è reciproco.
    rest, pair_time = w["rest"], w["pair_time"]
    mated = jnp.zeros((B, n), bool)
    partner = jnp.zeros((B, n), jnp.int32)
    if cfg.mating:
        rest = jnp.maximum(rest - dt, 0.0)
        dm = pos[:, None, :, :] - pos[:, :, None, :]
        dist = jnp.hypot(dm[..., 0], dm[..., 1])
        facing = jnp.abs(_wrap(jnp.arctan2(dm[..., 1], dm[..., 0]) - heading[:, :, None])) < np.radians(cfg.mate_angle)
        ready = (energy >= cfg.mate_energy) & (rest <= 0.0) & (jnp.abs(speed) <= cfg.take_speed * cfg.speed) & ~ghost
        okm = facing & jnp.swapaxes(facing, 1, 2) & (dist < cfg.mate_distance) & ready[:, :, None] & ready[:, None, :]
        okm = okm & ~jnp.eye(n, dtype=bool)
        pair_time = jnp.where(okm, pair_time + dt, 0.0)
        cand = jnp.where(pair_time >= cfg.mate_time, pair_time, 0.0)
        partner = jnp.argmax(cand, axis=2).astype(jnp.int32)
        has = cand.max(2) > 0
        back = jnp.take_along_axis(partner, partner, axis=1)
        mated = has & (back == jnp.arange(n)[None])
        energy = jnp.where(mated, energy - cfg.mate_cost, energy)
        rest = jnp.where(mated, cfg.mate_rest, rest)
        pair_time = jnp.where(mated[:, :, None] | mated[:, None, :], 0.0, pair_time)

    # Inversioni e zone (in tutti i recinti insieme).
    t = w["t"] + dt
    inv = t >= w["next_inv"]
    good = jnp.where(inv, 1 - w["good"], w["good"])
    next_inv = w["next_inv"] + inv * cfg.inversion_every
    zc = t >= w["next_zone"]
    zone = jnp.where(zc, (w["zone"] + jax.random.randint(k4, (B,), 1, 4)) % 4, w["zone"])
    next_zone = w["next_zone"] + zc * cfg.zone_every

    # Morti e rinascite.
    starved = energy <= 0.0
    aged = age >= cfg.age_max
    dead = starved | aged
    pos = jnp.where(dead[..., None], jax.random.uniform(k5, (B, n, 2), F, cfg.radius, cfg.size - cfg.radius), pos)
    heading = jnp.where(dead, jax.random.uniform(k6, (B, n), F, -jnp.pi, jnp.pi), heading)
    energy = jnp.where(dead, cfg.energy_start, energy)
    for name, x in (("age", age), ("charging", charging), ("taste", taste), ("pain", pain), ("rest", rest)):
        w[name] = jnp.where(dead, 0.0, x)
    pair_time = jnp.where(dead[:, :, None] | dead[:, None, :], 0.0, pair_time)
    w.update(ghost_hide=ghost_hide)
    w.update(pos=pos, heading=heading, speed=speed, energy=energy, bpos=bpos, bcolor=bcolor, bbig=bbig, bon=bon,
             bwait=bwait,
             bage=bage, t=t, good=good, next_inv=next_inv, zone=zone, next_zone=next_zone, pair_time=pair_time)
    outs = dict(pleasure=pleasure, pain=hurt, spent=spent / cfg.good_energy, dead=dead, mated=mated, partner=partner, good_each=took & ok, bad_each=bad,
                good_taken=(took & ok).sum(), bad_taken=bad.sum(), starved=starved.sum(), aged=(aged & ~starved).sum(),
                matings=mated.sum() // 2, inversion=inv, zone_change=zc)
    return w, outs


# --- I cervelli ----------------------------------------------------------------------

ATT_KEYS = ("att_target", "att_inhib", "att_inhib_t", "att_prev_vis", "att_novelty", "att_new", "att_prob", "att_has",
            "att_parts")


def init_brain(agent: str, N: int, dt: float, seed: int, cfg: P.PileConfig, genomes=None,
               sizes: S.Sizes | None = None) -> tuple[dict, dict]:
    """Lo stato di partenza, dalla versione NumPy (stessa partenza), e dove
    stanno nel genoma i pesi e le connessioni di partenza (`Agent.blocks`)."""
    from ..evo.pile_test import brain_kwargs

    kw = brain_kwargs(agent, cfg, sizes.lesions if sizes else ())
    if agent == "sezioni" and sizes is not None:
        kw["sizes"] = sizes
    b = S.AGENTS[agent](N, dt, seed, genomes=genomes, **kw)
    at = b.att
    out = {"genome": b.genome, "clock": b.clock, "program": b.program.astype(np.int32), "pi": b.pi,
           "bite": b.bite, "p_bite": b.p_bite,
           "att_target": at.target.astype(np.int32), "att_inhib": at.inhib.astype(np.int32),
           "att_prob": at.prob, "att_has": at.has_pos, "att_parts": at.parts,
           "att_inhib_t": at.inhib_t, "att_prev_vis": at.prev_vis, "att_novelty": at.novelty, "att_new": at.new}
    if agent == "guidato":
        return out, b.blocks
    out.update(a_slow=b.a.slow, w_e=b.w_e, e_e=b.e_e, w_v=b.w_v, e_v=b.e_v, v_prev=b.v_prev, value=b.value,
               trust=b.trust,
               fresh=b.fresh,
               delta=b.delta, teach=b.teach)
    if agent == "sezioni":
        out.update(proto=b.proto, pred=b.pred, win_rate=b.win_rate, state_b=b.state_b, prev_b=b.prev_b, mem=b.mem,
                   w_gate=b.w_gate, e_gate=b.e_gate, c_in=b.c_in, u_c=b.u_c, e_c=b.e_c, u_l=b.u_l, code=b.code,
                   d_fast=b.d_fast, d_slow=b.d_slow, t_slow=b.t_slow, unc=b.unc, counts=np.zeros((N, b.nb, 4)),
                   teacher=b.teacher, obs_err=b.obs_err, o_pl=b.o_pl, o_pa=b.o_pa, o_assoc=b.o_assoc, o_e=b.o_e,
                   ep_on=b.ep_on, ep_pl=b.ep_pl, ep_pa=b.ep_pa, ep_e=b.ep_e, o_err=b.o_err, o_out=b.o_out)
    return out, b.blocks


def decode(genome):
    return S.decode(genome, xp=jnp)


def _attend(br, objs, decide, p, dt, key, extra=None, learned=None, o=None):
    """L'attenzione: come `sections.Attention.update` (la mappa di priorità)."""
    vis = objs["vis"]
    n, m = vis.shape
    idx = jnp.arange(n)
    br["att_novelty"] = br["att_novelty"] * jnp.exp(-dt / p["novelty_tau"])[:, None] + (vis & ~br["att_prev_vis"])
    br["att_prev_vis"] = vis
    br["att_inhib_t"] = jnp.maximum(br["att_inhib_t"] - dt, 0.0)
    target = br["att_target"]
    lost = (target >= 0) & ~vis[idx, jnp.maximum(target, 0)]
    target = jnp.where(lost, -1, target)
    br["att_new"] = br["att_new"] * jnp.exp(-dt / p["new_tau"])
    has, obj, feat, near = S.priority_map(objs, S.MAP, xp=jnp)
    br["att_has"] = has
    choose = (decide | (target < 0)) & has.any(1)
    nov = jnp.take_along_axis(br["att_novelty"], obj, axis=1)
    innate = (p["sal_near"][:, None] * near + p["sal_novelty"][:, None] * nov
              + p["sal_stick"][:, None] * (obj == target[:, None])
              - p["sal_inhibit"][:, None] * ((obj == br["att_inhib"][:, None]) & (br["att_inhib_t"] > 0)[:, None]))
    value = imp = unc = jnp.zeros_like(innate)
    if o is not None:
        value, imp, unc = S.o_read(o[0], o[1], o[2], feat, o[3], p, xp=jnp)
        trust = jnp.ones(n) if learned is None else learned
        value = trust[:, None] * p["sal_value"][:, None] * value
        imp = trust[:, None] * p["sal_import"][:, None] * imp
        unc = p["sal_unc"][:, None] * unc
    br["att_parts"] = jnp.stack([innate, value, imp, unc], -1) * has[:, :, None]
    score = innate + value + imp + unc
    if extra is not None:
        score = score + jnp.take_along_axis(extra, obj, axis=1)
    z = jnp.where(has, score / p["sal_temperature"][:, None], -1e9)
    prob = jax.nn.softmax(z, axis=1) * has
    prob = prob / jnp.maximum(prob.sum(1, keepdims=True), 1e-12)
    br["att_prob"] = prob
    place = jax.random.categorical(key, z, axis=1)
    sel = jnp.where(choose, obj[idx, place], target).astype(jnp.int32)
    changed = choose & (sel != target)
    br["att_new"] = jnp.where(changed, 1.0, br["att_new"])
    br["att_target"] = sel


def _seen(br, objs):
    idx, t = jnp.arange(br["att_target"].shape[0]), jnp.maximum(br["att_target"], 0)
    return br["att_target"] >= 0, objs["feat"][idx, t], objs["near"][idx, t], objs["ang"][idx, t]


def _leave(br, mask, p):
    br["att_inhib"] = jnp.where(mask, br["att_target"], br["att_inhib"])
    br["att_inhib_t"] = jnp.where(mask, p["inhibit_time"], br["att_inhib_t"])
    br["att_target"] = jnp.where(mask, -1, br["att_target"])


def _interface(br, raw, objs, p, dt):
    keep = jnp.exp(-dt / p["tau_a"])[:, None]
    x = raw[:, list(S.FAST)]
    fast = jnp.clip(x - br["a_slow"], 0.0, 1.0)
    br["a_slow"] = keep * br["a_slow"] + (1.0 - keep) * x
    has, feat, near, ang = _seen(br, objs)
    fov = S.fovea_of(feat, near, ang, has, br["att_new"], xp=jnp)
    return jnp.concatenate([raw, fast, fov], 1)


def _td(br, p, f_v, reward, horizons, eta, dt, use_critic=True):
    """Come `_Learner._td`: K critici (orizzonti `horizons` (N, K)), δ = media."""
    gamma = jnp.exp(-dt / horizons)
    value = (br["w_v"] * f_v[:, None, :]).sum(-1)
    if use_critic:
        deltas = jnp.where(br["fresh"][:, None], 0.0, reward[:, None] + gamma * value - br["v_prev"])
    else:
        deltas = jnp.repeat(reward[:, None], horizons.shape[1], 1)
    deltas = jnp.clip(deltas, -S.DELTA_MAX, S.DELTA_MAX)
    norm = 1.0 + (f_v**2).sum(1)[:, None] + (br["e_v"] * f_v[:, None, :]).sum(-1)
    br["w_v"] = br["w_v"] + (deltas / norm)[:, :, None] * eta[:, None, :] * br["e_v"]
    br["e_v"] = (gamma * jnp.exp(-dt / p["trace_critic"])[:, None])[:, :, None] * br["e_v"] + f_v[:, None, :]
    br["v_prev"] = (br["w_v"] * f_v[:, None, :]).sum(-1)
    br["fresh"] = jnp.zeros_like(br["fresh"])
    br["value"] = br["v_prev"].mean(1)
    full = deltas.mean(1)
    br["trust"] = p["critic_mix"] if use_critic else jnp.zeros_like(full)
    br["delta"] = reward + br["trust"] * (full - reward)
    return br["delta"]


def _act(br, logits, f_e, teach, eta, temp, p, dt, decide, key, f_tag=None, code=None):
    """`decide` (N,): chi decide in questo passo."""
    np_ = S.N_PROGRAMS
    k_prog, k_bite = jax.random.split(key)
    norm = 1.0 + (f_e**2).sum(1)
    br["w_e"] = jnp.clip(br["w_e"] + (eta * teach / norm)[:, None, None] * br["e_e"], -S.WEIGHT_MAX, S.WEIGHT_MAX)
    br["e_e"] = br["e_e"] * jnp.exp(-dt / p["trace_actor"])[:, None, None]
    z = (logits[:, :np_] + p["stick"][:, None] * jax.nn.one_hot(br["program"], np_)) / temp[:, None]
    pi = jax.nn.softmax(z, axis=1)
    choice = jax.random.categorical(k_prog, z, axis=1).astype(jnp.int32)
    chosen = jax.nn.one_hot(choice, np_)
    p_bite = jax.nn.sigmoid(jnp.clip(logits[:, S.BITE] / temp, -30, 30))
    bite = jax.random.uniform(k_bite, p_bite.shape) < p_bite
    br["program"] = jnp.where(decide, choice, br["program"])
    br["pi"] = jnp.where(decide[:, None], pi, br["pi"])
    br["bite"] = jnp.where(decide, bite, br["bite"])
    br["p_bite"] = jnp.where(decide, p_bite, br["p_bite"])
    dev = jnp.concatenate([chosen - pi, (bite.astype(F) - p_bite)[:, None]], 1) * decide[:, None]
    tag = f_e if f_tag is None else f_tag
    br["e_e"] = br["e_e"] + dev[:, :, None] * tag[:, None, :]
    if code is not None:
        br["e_c"] = br["e_c"] + dev[:, :, None] * code[:, None, :]
    _leave(br, decide & (br["program"] == S.LEAVE), p)
    return br["program"]


def _sections(br, p, raw, objs, reward, dt, sz: S.Sizes, les: frozenset, decide, key, group: int):
    k_att, k_act = jax.random.split(key)
    arbitrate = "R" not in les and "U" not in les
    k = jnp.clip(br["unc"] / 3.0, 0.0, 1.0) if arbitrate else jnp.zeros_like(br["unc"])
    teacher = br["teacher"]
    n = raw.shape[0]
    master = jnp.arange(n) // group * group
    on = ~teacher & teacher[master]
    br.update(S.o_step(br, reward, raw[:, P.ENERGY], master, on, p, les, xp=jnp))  # O: gli episodi finiti insegnano
    extra = jnp.where(teacher[:, None], jnp.where(objs["good"], 10.0, -10.0), 0.0)  # il maestro sa quali sono buone
    o = None if "S" in les else (br["o_pl"], br["o_pa"], br["o_assoc"], raw[:, P.ENERGY])
    _attend(br, objs, decide, p, dt, k_att, extra=extra, learned=1.0 - p["att_shift"] * k, o=o)
    a = _interface(br, raw, objs, p, dt)
    fov = a[:, -len(S.FOVEA):]
    r = S.value_of(reward, raw[:, P.ENERGY], p, xp=jnp) + (p["innate"] * raw).sum(1) * dt
    teach_prev = br["teach"]
    # B
    if "B" in les:
        b = a
        br["state_b"] = a
    else:
        d = ((br["proto"] - a[:, None, :]) ** 2).sum(-1) - p["conscience"][:, None] * (1.0 / sz.b - br["win_rate"])
        kth = jnp.sort(d, axis=1)[:, sz.b_winners: sz.b_winners + 1]
        y = jnp.clip(1.0 - d / jnp.maximum(kth, 1e-9), 0.0, 1.0)
        y = y / jnp.maximum(y.max(1, keepdims=True), 1e-9)
        err = a - jnp.einsum("nab,nb->na", br["pred"], br["prev_b"])
        br["pred"] = br["pred"] + (p["eta_predict"] * dt)[:, None, None] * err[:, :, None] * br["prev_b"][:, None, :]
        rate = (p["eta_b"] * dt * (1.0 + p["surprise_boost"] * jnp.minimum(jnp.abs(br["delta"]), 1.0)))[:, None, None]
        br["proto"] = br["proto"] + rate * y[:, :, None] * ((a[:, None, :] - br["proto"])
                                                            + p["predict"][:, None, None] * err[:, None, :])
        br["win_rate"] = br["win_rate"] + (dt / p["win_window"])[:, None] * ((y > 0) / sz.b_winners - br["win_rate"])
        br["prev_b"] = br["state_b"]
        br["state_b"] = br["state_b"] + (dt / p["tau_b"])[:, None] * (y - br["state_b"])
        b = br["state_b"]
    # D
    if "D" in les:
        m = jnp.zeros_like(fov)
    else:
        br["w_gate"] = jnp.clip(br["w_gate"] + (p["eta_gate"] * teach_prev)[:, None] * br["e_gate"],
                                -2 * S.WEIGHT_MAX, 2 * S.WEIGHT_MAX)
        z = p["gate_event"] * fov[:, -1] + (br["w_gate"] * fov).sum(1) + p["gate_bias"]
        g = jax.nn.sigmoid(jnp.clip(z, -30, 30))
        tagged = jnp.where(fov > p["tag"][:, None], fov, 0.0)
        br["e_gate"] = jnp.exp(-dt / p["trace_actor"])[:, None] * br["e_gate"] + (g * (1.0 - g))[:, None] * tagged
        keep = jnp.exp(-dt / p["tau_d"])[:, None]
        br["mem"] = (1.0 - g)[:, None] * keep * br["mem"] + g[:, None] * fov
        m = br["mem"]
    # C
    if "C" in les:
        c = jnp.zeros((n, sz.c), F)
    else:
        h = jnp.concatenate([fov[:, :S.IDENTITY], p["memory"][:, None] * m[:, :S.IDENTITY],
                             p["context"][:, None] * b], 1)
        z = jnp.einsum("ncj,nj->nc", br["c_in"], h)
        c = (z >= S.c_threshold(fov, p, xp=jnp)[:, None]).astype(F)
    br["code"] = c
    if "O" not in les:
        br["o_e"] = S.o_trace(br["o_e"], fov, m, p, dt, xp=jnp)
        ids = jnp.stack([fov[:, :S.IDENTITY], m[:, :S.IDENTITY]], 1)
        br["o_out"] = S.o_read(br["o_pl"], br["o_pa"], br["o_assoc"], ids, raw[:, P.ENERGY], p, xp=jnp)[0]
    vo = br["o_out"]
    one = jnp.ones((n, 1), F)
    f_v = jnp.concatenate([fov, m, c, raw[:, P.ENERGY:P.ENERGY + 1], p["critic_b"][:, None] * b, vo, one], 1)
    nf = fov.shape[1]
    is_c = (jnp.arange(f_v.shape[1]) >= 2 * nf) & (jnp.arange(f_v.shape[1]) < 2 * nf + sz.c)
    eta_v = jnp.where(is_c[None], p["eta_vc"][:, None], p["eta_v"][:, None])  # la parte su C: in fretta
    horizons = jnp.stack([p["h_short"], p["h_mid"], p["h_long"]], 1)
    delta = _td(br, p, f_v, r, horizons, eta_v, dt, use_critic="V" not in les)
    br["d_slow"] = br["d_slow"] + (dt / p["u_slow"]) * (jnp.abs(delta) - br["d_slow"])
    teach = jnp.sign(delta) * jnp.maximum(jnp.abs(delta) - p["floor"] * br["d_slow"], 0.0)
    br["teach"] = teach
    br["d_fast"] = br["d_fast"] + (dt / p["u_fast"]) * (jnp.abs(teach) - br["d_fast"])
    br["t_slow"] = br["t_slow"] + (dt / p["u_slow"]) * (jnp.abs(teach) - br["t_slow"])
    if "U" not in les:
        br["unc"] = jnp.clip(br["d_fast"] / jnp.maximum(br["t_slow"], 1e-4) - 1.0, 0.0, 3.0)
    boost = 1.0 + p["surprise_rate"] * br["unc"]
    step = p["eta_c"] * boost * teach / (1.0 + c.sum(1))
    cap = p["c_max"][:, None, None]
    br["u_c"] = jnp.clip(br["u_c"] + step[:, None, None] * br["e_c"] - br["u_c"] * (dt / p["forget"])[:, None, None],
                         -cap, cap)
    br["e_c"] = br["e_c"] * jnp.exp(-dt / p["trace_actor"])[:, None, None]
    if "L" not in les:
        rest = (raw[:, P.ENERGY] > p["rest_energy"]) & (raw[:, P.TASTE] > 0.1)
        br["u_l"] = br["u_l"] + (p["eta_l"] * dt * rest)[:, None, None] * (br["u_c"] - br["u_l"])
    slow_boost = jnp.ones_like(boost) if arbitrate else boost
    f_e = jnp.concatenate([b, m, vo, one], 1)
    tag = p["tag"][:, None]
    f_tag = jnp.concatenate([jnp.where(b > tag, b, 0.0), jnp.where(m > tag, m, 0.0), vo, one], 1)
    habit = (br["w_e"] * f_e[:, None, :]).sum(-1)
    if "L" not in les:
        habit = habit + jnp.einsum("npc,nc->np", br["u_l"], c)
    fast = jnp.einsum("npc,nc->np", br["u_c"], c)
    logits = (1.0 - p["arb_slow"] * k)[:, None] * habit + (1.0 + p["arb_fast"] * k)[:, None] * fast
    temp = p["temperature"] * (1.0 + (0.0 if arbitrate else p["surprise_explore"]) * br["unc"])
    prog = _act(br, logits, f_e, teach, p["eta_e"] * slow_boost, temp, p, dt, decide, k_act, f_tag=f_tag, code=c)
    # Il maestro decide da sé.
    has, _, near, _ = _seen(br, objs)
    ok = has & objs["good"][jnp.arange(n), jnp.maximum(br["att_target"], 0)]
    prog, br["bite"], leave = S.teach_override(prog, br["bite"], has, ok, near, teacher, xp=jnp)
    prog = prog.astype(jnp.int32)
    br["p_bite"] = jnp.where(teacher, br["bite"].astype(F), br["p_bite"])
    _leave(br, leave, p)
    br["program"] = prog
    eye = raw[:, P.EYES].reshape(n, 3, 3).max(-1)
    seen = jnp.where(eye.max(1) > 0.2, eye.argmax(1), 3)
    br["counts"] = br["counts"].at[jnp.arange(n), br["state_b"].argmax(1), seen].add(1.0)
    return prog


def _actor_critic(br, p, raw, objs, reward, dt, decide, key):
    k_att, k_act = jax.random.split(key)
    _attend(br, objs, decide, p, dt, k_att)
    a = _interface(br, raw, objs, p, dt)
    f = jnp.concatenate([a, jnp.ones((raw.shape[0], 1), F)], 1)
    r = S.value_of(reward, raw[:, P.ENERGY], p, xp=jnp) + (p["innate"] * raw).sum(1) * dt
    delta = _td(br, p, f, r, p["horizon"][:, None], jnp.repeat(p["eta_v"][:, None], f.shape[1], 1), dt)
    br["teach"] = delta
    logits = (br["w_e"] * f[:, None, :]).sum(-1)
    return _act(br, logits, f, delta, p["eta_e"], p["temperature"], p, dt, decide, k_act)


def _guided(br, objs, decide, p, dt, key):
    _attend(br, objs, decide, p, dt, key, extra=jnp.where(objs["good"], 10.0, -10.0))
    has, _, near, _ = _seen(br, objs)
    ok = has & objs["good"][jnp.arange(has.shape[0]), jnp.maximum(br["att_target"], 0)]
    prog = jnp.where(ok, S.APPROACH, S.EXPLORE).astype(jnp.int32)
    br["bite"] = ok & (near > 0.9)
    br["p_bite"] = br["bite"].astype(F)
    _leave(br, has & ~ok, p)
    br["program"], br["pi"] = prog, jax.nn.one_hot(prog, S.N_PROGRAMS)
    return prog


def _reborn(br, dead, child, agent, sz, key, blocks):
    """I cervelli dei corpi rinati: ciò che si era imparato sparisce, e si
    riparte dai pesi e dalle connessioni di partenza del (nuovo) genoma."""
    br = dict(br)
    if child is not None:
        br["genome"] = jnp.where(dead[:, None], child, br["genome"])

    def zero(x):
        return jnp.where(dead.reshape((-1,) + (1,) * (x.ndim - 1)), jnp.zeros_like(x), x)

    for k in ("att_inhib_t", "att_prev_vis", "att_novelty", "att_new", "att_prob", "att_has", "att_parts"):
        br[k] = zero(br[k])
    br["att_target"] = jnp.where(dead, -1, br["att_target"])
    br["att_inhib"] = jnp.where(dead, -1, br["att_inhib"])
    br["clock"] = jnp.where(dead, 0.0, br["clock"])
    br["bite"] = br["bite"] & ~dead
    if agent == "guidato":
        return br
    for k in ("a_slow", "w_e", "e_e", "w_v", "e_v", "v_prev", "value", "delta", "teach", "trust"):
        br[k] = zero(br[k])

    br["fresh"] = br["fresh"] | dead
    if agent == "sezioni":
        for k in ("pred", "state_b", "prev_b", "mem", "w_gate", "e_gate", "u_c", "e_c", "u_l", "code", "d_fast",
                  "unc", "counts", "o_e", "ep_pl", "ep_pa", "ep_e", "o_err", "o_out", "obs_err"):
            br[k] = zero(br[k])
        br["o_assoc"] = jnp.where(dead[:, None], 1.0, br["o_assoc"])
        br["ep_on"] = br["ep_on"] & ~dead
        br["win_rate"] = jnp.where(dead[:, None], 1.0 / sz.b, br["win_rate"])
        br["d_slow"] = jnp.where(dead, 1e-3, br["d_slow"])
        br["t_slow"] = jnp.where(dead, 1e-4, br["t_slow"])
    n = dead.shape[0]
    start = lambda name: br["genome"][:, blocks[name][0]].reshape((n,) + blocks[name][1])
    born = {"e0": "w_e", "v0": "w_v", "proto": "proto"}
    for name, key_ in born.items():
        if name in blocks:
            br[key_] = jnp.where(dead.reshape((-1,) + (1,) * (br[key_].ndim - 1)), start(name), br[key_])
    if "o0" in blocks:  # i valori innati delle cose
        o0 = start("o0")
        br["o_pl"] = jnp.where(dead[:, None], o0[:, 0], br["o_pl"])
        br["o_pa"] = jnp.where(dead[:, None], o0[:, 1], br["o_pa"])
    if "c_in" in blocks:
        br["c_in"] = jnp.where(dead[:, None, None], (start("c_in") > 0.5).astype(F), br["c_in"])
    return br


def make_run(cfg: P.PileConfig, agent: str, phase: str, sizes: S.Sizes, blocks: dict):
    """La funzione compilata: (stato, chiave, k) -> (stato, serie per passo)."""
    les = frozenset(sizes.lesions)
    dt = cfg.dt
    centres = zone_centres(cfg)

    def step(state, key):
        w, br, misc = state["world"], dict(state["brain"]), dict(state["misc"])
        B, n = w["heading"].shape
        N = B * n
        k_brain, k_motor, k_world, k_child, k_reborn = jax.random.split(key, 5)
        raw = senses(cfg, w)
        flat = raw.reshape(N, -1)
        objs = {k: v.reshape((N,) + v.shape[2:]) for k, v in objects(cfg, w).items()}
        p = decode(br["genome"])
        reward = misc["reward"]
        clock = br["clock"] + dt
        decide = clock >= p["decide_every"] - 1e-6  # ognuno al suo ritmo
        br["clock"] = jnp.where(decide, 0.0, clock)
        if agent == "guidato":
            prog = _guided(br, objs, decide, p, dt, k_brain)
        elif agent == "attore-critico":
            prog = _actor_critic(br, p, flat, objs, reward, dt, decide, k_brain)
        else:
            prog = _sections(br, p, flat, objs, reward, dt, sizes, les, decide, k_brain, cfg.beings)
        has, _, near, ang = _seen(br, objs)
        shape = (B, n)
        W = p["motor"].reshape(B, n, P.N_FOUT, P.N_FIN)
        thrust, steer, wander, escape, f_in, f_act = motor(prog.reshape(shape), raw, has.reshape(shape),
                                                           near.reshape(shape), ang.reshape(shape), w["wander"],
                                                           w["escape"], W, k_motor)
        if agent in ("guidato", "sezioni"):  # il guidato (e il maestro) sa anche dov'è la zona
            guided = jnp.full(shape, True) if agent == "guidato" else br["teacher"].reshape(shape)
            to = centres[w["zone"]][:, None, :] - w["pos"]
            zang = _wrap(jnp.arctan2(to[..., 1], to[..., 0]) - w["heading"])
            go = guided & (prog.reshape(shape) == S.EXPLORE) & (jnp.hypot(to[..., 0], to[..., 1]) > cfg.zone_half)
            steer = jnp.where(go, jnp.clip(2.0 * zang, -1.0, 1.0), steer)
        w = dict(w, wander=wander, escape=escape, f_in=f_in, f_act=f_act)
        w, out = world_step(cfg, w, thrust, steer, br["bite"].reshape(shape), k_world)
        misc["reward"] = jnp.stack([out["pleasure"], out["pain"], out["spent"]], -1).reshape(N, 3)
        misc["usage"] = misc["usage"] + jnp.bincount(prog, length=S.N_PROGRAMS)
        misc["good_by"] = misc["good_by"] + out["good_each"].reshape(N)  # per essere: per provare le reti una a una
        misc["bad_by"] = misc["bad_by"] + out["bad_each"].reshape(N)
        misc["deaths_by"] = misc["deaths_by"] + out["dead"].reshape(N)
        dead = out["dead"].reshape(N)
        child = None
        if phase == "B":
            g = br["genome"].reshape(B, n, -1)
            mated = out["mated"]
            first = jnp.argmax(mated, axis=1)  # il primo accoppiato di ogni recinto
            mate = jnp.take_along_axis(out["partner"], first[:, None], 1)[:, 0]
            any_m = mated.any(1)
            ga = jnp.take_along_axis(g, first[:, None, None], 1)[:, 0]
            gb = jnp.take_along_axis(g, mate[:, None, None], 1)[:, 0]
            misc["pair"] = jnp.where(any_m[:, None, None], jnp.stack([ga, gb], 1), misc["pair"])
            misc["has_pair"] = misc["has_pair"] | any_m
            kc1, kc2, kc3 = jax.random.split(k_child, 3)
            pick = jax.random.randint(kc1, (B, n, 2), 0, n)  # se non c'è ancora una coppia: due a caso del recinto
            ra = jnp.take_along_axis(g, pick[..., 0:1], 1)
            rb = jnp.take_along_axis(g, pick[..., 1:2], 1)
            pa = jnp.where(misc["has_pair"][:, None, None], misc["pair"][:, 0:1], ra)
            pb = jnp.where(misc["has_pair"][:, None, None], misc["pair"][:, 1:2], rb)
            mix = jnp.where(jax.random.uniform(kc2, pa.shape[:2] + (1,)) < 0.5, pa, pb)  # un genitore, intero
            mix = mix + 0.1 * jax.random.normal(kc3, mix.shape, F)
            mix = mix.at[..., S.INNATE].set(jnp.clip(mix[..., S.INNATE], -S.INNATE_CAP, S.INNATE_CAP))
            child = mix.reshape(N, -1)
        br = _reborn(br, dead, child, agent, sizes, k_reborn, blocks)
        misc["reward"] = jnp.where(dead[:, None], 0.0, misc["reward"])
        series = {k: out[k] for k in ("starved", "aged", "matings", "inversion", "zone_change")}
        series["good_taken"] = (out["good_each"].reshape(N) * misc["counted"]).sum()  # il maestro non si conta
        series["bad_taken"] = (out["bad_each"].reshape(N) * misc["counted"]).sum()
        return {"world": w, "brain": br, "misc": misc}, series

    def run(state, key, k):
        keys = jax.random.split(key, k)
        return jax.lax.scan(step, state, keys)

    return jax.jit(run, static_argnums=2)


class PileGpu:
    """Una corsa nell'arena delle pile sulla GPU."""

    def __init__(self, cfg: P.PileConfig, agent: str, phase: str, enclosures: int, seed: int = 0,
                 lesions=(), genomes=None, teacher: str = ""):
        cfg.mating = phase == "B"
        self.cfg, self.agent, self.phase = cfg, agent, phase
        self.lesions, self.seed = tuple(lesions), seed
        self.sizes = S.Sizes(lesions=frozenset(lesions))
        B, n = enclosures, cfg.beings
        self.B, self.n = B, n
        world = init_world(cfg, B, seed)
        brain, self.blocks = init_brain(agent, B * n, cfg.dt, seed, cfg, genomes, self.sizes)
        misc = dict(reward=np.zeros((B * n, 3)), usage=np.zeros(S.N_PROGRAMS), pair=np.zeros((B, 2, brain["genome"].shape[1])),
                    has_pair=np.zeros(B, bool), good_by=np.zeros(B * n), bad_by=np.zeros(B * n),
                    deaths_by=np.zeros(B * n), counted=np.ones(B * n))
        to = lambda x: jnp.asarray(x, F) if np.asarray(x).dtype.kind == "f" else jnp.asarray(x)
        self.state = jax.tree_util.tree_map(to, {"world": world, "brain": brain, "misc": misc})
        self.key = jax.random.PRNGKey(seed)
        self._run = make_run(cfg, agent, phase, self.sizes, self.blocks)
        self.series = {k: [] for k in ("good_taken", "bad_taken", "starved", "aged", "matings", "inversion",
                                       "zone_change")}
        self.t = 0.0
        self.teacher_kind = teacher or "fantasma"
        if teacher:
            self.set_teacher(True)

    def set_teacher(self, on: bool) -> None:
        """Come `NumpyRun.set_teacher`."""
        if self.agent != "sezioni":
            raise ValueError("il maestro c'è solo con il cervello a sezioni")
        N = self.B * self.n
        first = S.teacher_mask(N, self.n)
        br, misc = dict(self.state["brain"]), dict(self.state["misc"])
        gone = br["teacher"] & ~jnp.asarray(first & on)
        br["teacher"] = jnp.asarray(first & on)
        self.key, sub = jax.random.split(self.key)
        br = _reborn(br, gone, None, self.agent, self.sizes, sub, self.blocks)  # al posto del maestro, un giovane
        misc["counted"] = jnp.where(jnp.asarray(first), 0.0, misc["counted"])
        world = dict(self.state["world"])
        world["ghost"] = jnp.asarray((first & on & (self.teacher_kind == "fantasma")).reshape(self.B, self.n))
        world["ghost_hide"] = jnp.zeros_like(world["ghost_hide"])
        self.state = {**self.state, "world": world, "brain": br, "misc": misc}

    def teaching(self) -> tuple[bool, float]:
        """Come `NumpyRun.teaching`."""
        br = self.state["brain"]
        if "teacher" not in br:
            return False, 0.0
        t = np.asarray(br["teacher"])
        if not t.any():
            return False, 0.0
        on = ~t & t[np.arange(len(t)) // self.n * self.n]
        return True, float(np.asarray(br["obs_err"])[on].mean()) if on.any() else 0.0

    def view(self, b: int, i: int) -> tuple[dict, dict, np.ndarray]:
        """Ciò che serve allo schermo in un solo trasferimento: il mondo del
        recinto `b`, il cervello dell'essere `i` (in fila), i sensi del recinto."""
        if not hasattr(self, "_view"):
            B, N, cfg = self.B, self.B * self.n, self.cfg

            def pick(state, b, i):
                w = state["world"]
                world = {k: (v[b] if v.ndim and v.shape[0] == B else v) for k, v in w.items()}
                brain = {k: v[i] for k, v in state["brain"].items() if v.ndim and v.shape[0] == N}
                return world, brain, senses(cfg, w)[b]

            self._view = jax.jit(pick)
        world, brain, raw = jax.device_get(self._view(self.state, b, i))
        world = {k: np.asarray(v) for k, v in world.items()}
        for k in ("t", "next_inv", "next_zone"):
            world[k] = float(world[k])
        return world, {k: np.asarray(v) for k, v in brain.items()}, np.asarray(raw)

    def advance(self, k: int) -> None:
        self.key, sub = jax.random.split(self.key)
        self.state, ys = self._run(self.state, sub, int(k))
        for name, v in jax.device_get(ys).items():
            self.series[name].append(np.asarray(v))
        self.t = float(self.state["world"]["t"])

    def world_of(self, b: int) -> dict:
        """Il mondo del recinto `b`, sul processore."""
        w = self.state["world"]
        out = {k: np.asarray(v[b]) for k, v in w.items() if getattr(v, "ndim", 0) and v.shape[0] == self.B}
        out["t"] = float(w["t"])
        out["next_inv"], out["next_zone"] = float(w["next_inv"]), float(w["next_zone"])
        return out

    def brain_of(self, i: int) -> dict:
        """Il cervello dell'essere `i` (in fila), sul processore."""
        br = self.state["brain"]
        N = self.B * self.n
        return {k: np.asarray(v[i]) for k, v in br.items() if getattr(v, "ndim", 0) and v.shape[0] == N}

    def senses_of(self, b: int) -> np.ndarray:
        return np.asarray(senses(self.cfg, self.state["world"])[b])

    def flat(self, name):
        return np.concatenate(self.series[name]) if self.series[name] else np.zeros(0)
