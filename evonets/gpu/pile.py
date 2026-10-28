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
                bpos=w.bpos, bcolor=w.bcolor.astype(np.int32), bon=w.bon, bwait=w.bwait, bage=w.bage,
                t=np.float32(0.0), next_inv=np.float32(cfg.inversion_every), next_zone=np.float32(cfg.zone_every),
                pair_time=w.pair_time, rest=w.rest, wander=np.zeros((B, n)), escape=np.zeros((B, n)))


def zone_centres(cfg):
    m, s = cfg.zone_margin, cfg.size
    return jnp.array([[m, m], [s - m, m], [m, s - m], [s - m, s - m]], F)


def senses(cfg: P.PileConfig, w: dict) -> jnp.ndarray:
    """(B, n, N_SENSES): come `PileWorld.senses`."""
    pos, heading = w["pos"], w["heading"]
    B, n = heading.shape
    out = []
    eyes = jnp.zeros((B, n, 3, 3), F)  # colore, settore
    secs = jnp.asarray(P.SECTORS, F)
    face = jnp.full((B, n), P.BLUE, jnp.int32)
    for k, (opos, ocol, oon) in enumerate(((w["bpos"], w["bcolor"], w["bon"]), (pos, face, jnp.ones((B, n), bool)))):
        d = opos[:, None, :, :] - pos[:, :, None, :]
        dist = jnp.hypot(d[..., 0], d[..., 1])
        ang = _wrap(jnp.arctan2(d[..., 1], d[..., 0]) - heading[:, :, None])
        near = jnp.clip(1.0 - dist / cfg.eye_range, 0.0, 1.0) * oon[:, None, :]
        if k == 1:
            near = near * (1.0 - jnp.eye(n, dtype=F))[None]
            back = _wrap(jnp.arctan2(-d[..., 1], -d[..., 0]) - heading[:, None, :])
            near = near * (jnp.abs(back) < np.radians(cfg.face_visible))
        for s in range(3):
            inside = (ang >= secs[s]) & (ang < secs[s + 1])
            for c in ((P.RED, P.YELLOW) if k == 0 else (P.BLUE,)):
                v = jnp.where(inside & (ocol[:, None, :] == c), near, 0.0).max(axis=2)
                eyes = eyes.at[:, :, c, s].max(v)
    out.append(eyes.reshape(B, n, 9))
    rays = []
    for a in P.RAYS:
        ux, uy = jnp.cos(heading + a), jnp.sin(heading + a)
        t = jnp.full((B, n), jnp.inf, F)
        for comp, p in ((ux, pos[..., 0]), (uy, pos[..., 1])):
            hit = jnp.where(comp > 0, (cfg.size - p) / jnp.where(comp > 0, comp, 1.0),
                            jnp.where(comp < 0, -p / jnp.where(comp < 0, comp, -1.0), jnp.inf))
            t = jnp.minimum(t, hit)
        rays.append(jnp.clip(1.0 - (t - cfg.radius) / cfg.ray_range, 0.0, 1.0))
    out.append(jnp.stack(rays, -1))
    wall = (pos.min(-1) < cfg.radius + 2.0) | (pos.max(-1) > cfg.size - cfg.radius - 2.0)
    dd = jnp.hypot(*jnp.moveaxis(pos[:, :, None] - pos[:, None], -1, 0)) + jnp.eye(n, dtype=F) * 1e9
    other = dd.min(-1) < 2.0 * cfg.radius
    contact = (wall | other | (w["charging"] > 0)).astype(F)
    d = pos[:, :, None, :] - pos[:, None, :, :]
    dist = jnp.hypot(d[..., 0], d[..., 1])
    ang = _wrap(jnp.arctan2(d[..., 1], d[..., 0]) - heading[:, None, :])
    look = (jnp.abs(ang) < np.radians(cfg.watch_angle)) & (dist < cfg.watch_range) & ~jnp.eye(n, dtype=bool)
    watched = look.any(-1).astype(F)
    out.append(jnp.stack([contact, w["energy"] / cfg.energy_max, watched, w["taste"], w["pain"]], -1))
    return jnp.concatenate(out, -1)


def motor(cfg, program, raw, wander, escape, key):
    """F: come `sim.pile.Motor`."""
    eye = raw[..., P.EYES].reshape(raw.shape[:-1] + (3, 3))
    thrust = jnp.zeros(program.shape, F)
    steer = jnp.zeros(program.shape, F)
    for c in range(3):
        e = eye[..., c, :]
        seen = e.max(-1) > 0.0
        near = e.max(-1)
        centred = e[..., 1] >= jnp.maximum(e[..., 0], e[..., 2])
        turn = jnp.clip(jnp.where(centred, 0.5, 4.0) * (e[..., 2] - e[..., 0]), -1.0, 1.0)
        th = jnp.where(seen, jnp.clip(1.2 * (1.0 - near), 0.15, 1.0), 0.2)
        st = jnp.where(seen, turn, 0.8)
        m = program == c
        thrust = jnp.where(m, th, thrust)
        steer = jnp.where(m, st, steer)
    wander = 0.9 * wander + 0.3 * jax.random.normal(key, wander.shape, F)
    m = program == 3
    thrust = jnp.where(m, 0.8, thrust)
    steer = jnp.where(m, jnp.clip(wander, -1.0, 1.0), steer)
    lr = eye.sum(-2)
    m = program == 4
    thrust = jnp.where(m, 0.6, thrust)
    steer = jnp.where(m, jnp.where(lr[..., 0] > lr[..., 2], 1.0, -1.0), steer)
    ray = raw[..., 9:12]
    wall = ray.max(-1) > 0.5
    free_side = jnp.where(ray[..., 0] < ray[..., 2], -1.0, 1.0)
    escape = jnp.where(wall, jnp.where(escape == 0.0, free_side, escape), 0.0)
    steer = jnp.where(wall, escape, steer)
    thrust = thrust * (1.0 - 0.7 * jnp.clip(ray[..., 1] - 0.3, 0.0, 1.0))
    return thrust, steer, wander, escape


def world_step(cfg: P.PileConfig, w: dict, thrust, steer, key):
    """Un passo del mondo: come `PileWorld.step`. Restituisce (stato, uscite)."""
    dt = cfg.dt
    B, n = thrust.shape
    K = cfg.batteries
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    w = dict(w)
    charging = w["charging"]
    thrust = jnp.where(charging > 0.0, 0.0, jnp.clip(thrust, 0.0, 1.0))
    heading = _wrap(w["heading"] + jnp.clip(steer, -1.0, 1.0) * cfg.turn * dt)
    speed = thrust * cfg.speed
    u = jnp.stack([jnp.cos(heading), jnp.sin(heading)], -1)
    pos = jnp.clip(w["pos"] + u * speed[..., None] * dt, cfg.radius, cfg.size - cfg.radius)
    age = w["age"] + dt
    energy = w["energy"] - (cfg.drain + cfg.move_cost * thrust) * dt
    fade = np.exp(-dt / cfg.taste_tau)
    taste, pain = w["taste"] * fade, w["pain"] * fade
    moving = speed > cfg.take_speed * cfg.speed
    charging = jnp.where(moving, 0.0, charging)
    gain = jnp.minimum(charging, cfg.good_energy * dt / cfg.charge_time)
    charging = charging - gain
    energy = jnp.minimum(energy + gain, cfg.energy_max)
    pleasure = gain / cfg.good_energy
    taste = jnp.minimum(taste + gain / cfg.good_energy * 4.0, 1.0)

    # Pile: scadono, ricompaiono nella zona attiva.
    bon, bwait, bage = w["bon"], w["bwait"] - dt, w["bage"] + dt
    old = bon & (bage >= cfg.battery_life)
    bon = bon & ~old
    bwait = jnp.where(old, cfg.respawn, bwait)
    spawn = ~bon & (bwait <= 0.0)
    centre = zone_centres(cfg)[w["zone"]][:, None, :]
    lo, hi = cfg.battery_radius, cfg.size - cfg.battery_radius
    new_pos = jnp.clip(centre + jax.random.uniform(k1, (B, K, 2), F, -cfg.zone_half, cfg.zone_half), lo, hi)
    bpos = jnp.where(spawn[..., None], new_pos, w["bpos"])
    bcolor = jnp.where(spawn, jax.random.randint(k2, (B, K), 0, 2), w["bcolor"])
    bage = jnp.where(spawn, jax.random.uniform(k3, (B, K), F, 0.0, 0.5 * cfg.battery_life), bage)
    bon = bon | spawn

    # Prese: ogni pila al primo essere che può prenderla, ogni essere al più una pila.
    d = bpos[:, None] - pos[:, :, None]
    front = jnp.abs(_wrap(jnp.arctan2(d[..., 1], d[..., 0]) - heading[:, :, None])) < np.radians(cfg.mouth)
    close = (jnp.hypot(d[..., 0], d[..., 1]) < cfg.radius + cfg.battery_radius) & bon[:, None] & front
    free = ~moving & (charging <= 0.0)
    can = close & free[..., None]
    first = jnp.argmax(can, axis=1)
    claim = (jnp.arange(n)[None, :, None] == first[:, None, :]) & can.any(1)[:, None, :]
    take = claim & (jnp.cumsum(claim, axis=2) == 1)
    took = take.any(2)
    kk = jnp.argmax(take, axis=2)
    gone = take.any(1)
    bon = bon & ~gone
    bwait = jnp.where(gone, cfg.respawn, bwait)
    ok = jnp.take_along_axis(bcolor, kk, axis=1) == w["good"][:, None]
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
        ready = (energy >= cfg.mate_energy) & (rest <= 0.0) & (speed <= cfg.take_speed * cfg.speed)
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
    w.update(pos=pos, heading=heading, speed=speed, energy=energy, bpos=bpos, bcolor=bcolor, bon=bon, bwait=bwait,
             bage=bage, t=t, good=good, next_inv=next_inv, zone=zone, next_zone=next_zone, pair_time=pair_time)
    outs = dict(pleasure=pleasure, pain=hurt, dead=dead, mated=mated, partner=partner,
                good_taken=(took & ok).sum(), bad_taken=bad.sum(), starved=starved.sum(), aged=(aged & ~starved).sum(),
                matings=mated.sum() // 2, inversion=inv, zone_change=zc)
    return w, outs


# --- I cervelli ----------------------------------------------------------------------

LEARNER_KEYS = ("a_slow", "w_e", "e_e", "w_v", "e_v", "v_prev", "fresh", "program", "pi", "delta")
SECTION_KEYS = ("proto", "pred", "win_rate", "state_b", "prev_b", "mem", "w_gate", "e_gate", "c_in", "u_c", "e_c",
                "u_l", "code", "d_fast", "d_slow", "unc", "counts")


def init_brain(agent: str, N: int, dt: float, seed: int, genomes=None, sizes: S.Sizes | None = None) -> dict:
    """Lo stato di partenza, dalla versione NumPy (stessa partenza)."""
    kw = {"sizes": sizes} if agent == "sezioni" else {}
    b = S.AGENTS[agent](N, dt, seed, genomes=genomes, **kw)
    out = {"genome": b.genome, "clock": np.float32(0.0)}
    if agent == "guidato":  # solo il programma in corso (per lo schermo)
        out.update(program=np.zeros(N, np.int32), pi=np.zeros((N, S.N_PROGRAMS)))
        return out
    out.update(a_slow=b.a.slow, w_e=b.w_e, e_e=b.e_e, w_v=b.w_v, e_v=b.e_v, v_prev=b.v_prev, fresh=b.fresh,
               program=b.program.astype(np.int32), pi=b.pi, delta=b.delta)
    if agent == "sezioni":
        out.update(proto=b.proto, pred=b.pred, win_rate=b.win_rate, state_b=b.state_b, prev_b=b.prev_b, mem=b.mem,
                   w_gate=b.w_gate, e_gate=b.e_gate, c_in=b.c_in, u_c=b.u_c, e_c=b.e_c, u_l=b.u_l, code=b.code,
                   d_fast=b.d_fast, d_slow=b.d_slow, unc=b.unc, counts=np.zeros((N, b.nb, 4)))
    return out


def decode(genome):
    names = S.GENE_NAMES
    out = {}
    for k, name in enumerate(names):
        v = genome[:, k]
        out[name] = jnp.exp(v) if name in S.Genes.LOG else jnp.maximum(v, 0.0)
    out["innate"] = jnp.clip(genome[:, len(names):], -S.INNATE_CAP, S.INNATE_CAP)
    return out


def _interface(br, raw, dt):
    keep = np.exp(-dt / 0.3)
    x = raw[:, list(S.FAST)]
    fast = jnp.clip(x - br["a_slow"], 0.0, 1.0)
    br["a_slow"] = keep * br["a_slow"] + (1.0 - keep) * x
    return jnp.concatenate([raw, fast], 1)


def _td(br, p, f_v, reward, dt, use_critic=True):
    gamma = jnp.exp(-dt / p["horizon"])
    value = (br["w_v"] * f_v).sum(1)
    delta = jnp.where(br["fresh"], 0.0, reward + gamma * value - br["v_prev"]) if use_critic else reward
    norm = 1.0 + (f_v**2).sum(1)
    br["w_v"] = br["w_v"] + (p["eta_v"] * delta / norm)[:, None] * br["e_v"]
    br["e_v"] = (gamma * np.exp(-dt / 1.0))[:, None] * br["e_v"] + f_v
    br["v_prev"] = (br["w_v"] * f_v).sum(1)
    br["fresh"] = jnp.zeros_like(br["fresh"])
    br["delta"] = delta
    return delta


def _act(br, logits, f_e, delta, eta, temp, dt, key, code=None):
    np_ = S.N_PROGRAMS
    norm = 1.0 + (f_e**2).sum(1)
    br["w_e"] = br["w_e"] + (eta * delta / norm)[:, None, None] * br["e_e"]
    br["e_e"] = br["e_e"] * np.exp(-dt / 2.0)
    clock = br["clock"] + dt
    decide = clock >= S._Learner.decide_every - 1e-6
    br["clock"] = jnp.where(decide, 0.0, clock)
    z = (logits + S._Learner.stick * jax.nn.one_hot(br["program"], np_)) / temp[:, None]
    pi = jax.nn.softmax(z, axis=1)
    choice = jax.random.categorical(key, z, axis=1).astype(jnp.int32)
    chosen = jax.nn.one_hot(choice, np_)
    br["program"] = jnp.where(decide, choice, br["program"])
    br["pi"] = jnp.where(decide, pi, br["pi"])
    dev = (chosen - pi) * decide
    br["e_e"] = br["e_e"] + dev[:, :, None] * f_e[:, None, :]
    if code is not None:
        br["e_c"] = br["e_c"] + dev[:, :, None] * code[:, None, :]
    return br["program"]


def _sections(br, p, raw, reward, dt, sz: S.Sizes, les: frozenset, key):
    a = _interface(br, raw, dt)
    r = reward + (p["innate"] * raw).sum(1) * dt
    dprev = br["delta"]
    n = raw.shape[0]
    # B
    if "B" in les:
        b = a
        br["state_b"] = a
    else:
        d = ((br["proto"] - a[:, None, :]) ** 2).sum(-1) - sz.conscience * (1.0 / sz.b - br["win_rate"])
        kth = jnp.sort(d, axis=1)[:, sz.b_winners: sz.b_winners + 1]
        y = jnp.clip(1.0 - d / jnp.maximum(kth, 1e-9), 0.0, 1.0)
        y = y / jnp.maximum(y.max(1, keepdims=True), 1e-9)
        err = a - jnp.einsum("nab,nb->na", br["pred"], br["prev_b"])
        br["pred"] = br["pred"] + sz.eta_predict * dt * err[:, :, None] * br["prev_b"][:, None, :]
        rate = (p["eta_b"] * dt * (1.0 + sz.surprise_boost * jnp.abs(dprev)))[:, None, None]
        br["proto"] = br["proto"] + rate * y[:, :, None] * ((a[:, None, :] - br["proto"]) + sz.predict * err[:, None, :])
        br["win_rate"] = br["win_rate"] + (dt / 30.0) * ((y > 0) / sz.b_winners - br["win_rate"])
        br["prev_b"] = br["state_b"]
        br["state_b"] = br["state_b"] + (dt / sz.tau_b) * (y - br["state_b"])
        b = br["state_b"]
    # D
    if "D" in les:
        m = jnp.zeros_like(b)
    else:
        event = a[:, P.N_SENSES:].max(1)
        br["w_gate"] = br["w_gate"] + (sz.eta_gate * dprev)[:, None] * br["e_gate"]
        g = jax.nn.sigmoid(sz.gate_event * event + (br["w_gate"] * b).sum(1) + sz.gate_bias)
        br["e_gate"] = np.exp(-dt / 2.0) * br["e_gate"] + (g * (1.0 - g))[:, None] * b
        keep = jnp.exp(-dt / p["tau_d"])[:, None]
        br["mem"] = (1.0 - g)[:, None] * keep * br["mem"] + g[:, None] * b
        m = br["mem"]
    # C
    if "C" in les:
        c = jnp.zeros((n, sz.c), F)
    else:
        h = jnp.concatenate([b, m], 1)
        z = jnp.einsum("ncj,nj->nc", br["c_in"], h)
        _, top = jax.lax.top_k(z, sz.c_winners)
        c = jax.nn.one_hot(top, sz.c).sum(1) * (z > 1e-6)
    br["code"] = c
    one = jnp.ones((n, 1), F)
    delta = _td(br, p, jnp.concatenate([b, m, c, one], 1), r, dt, use_critic="V" not in les)
    if "U" not in les:
        br["d_fast"] = br["d_fast"] + (dt / 20.0) * (jnp.abs(delta) - br["d_fast"])
        br["d_slow"] = br["d_slow"] + (dt / 200.0) * (jnp.abs(delta) - br["d_slow"])
        br["unc"] = jnp.clip(br["d_fast"] / jnp.maximum(br["d_slow"], 1e-6) - 1.0, 0.0, 3.0)
    boost = 1.0 + p["surprise_rate"] * br["unc"]
    br["u_c"] = br["u_c"] + (p["eta_c"] * boost * delta)[:, None, None] * br["e_c"] - br["u_c"] * (dt / sz.forget)
    br["e_c"] = br["e_c"] * np.exp(-dt / 2.0)
    if "L" not in les:
        rest = (raw[:, 13] > sz.rest_energy) & (raw[:, 15] > 0.1)
        br["u_l"] = br["u_l"] + (p["eta_l"] * dt * rest)[:, None, None] * (br["u_c"] - br["u_l"])
    f_e = jnp.concatenate([b, m, one], 1)
    logits = (br["w_e"] * f_e[:, None, :]).sum(-1) + jnp.einsum("npc,nc->np", br["u_c"], c)
    if "L" not in les:
        logits = logits + jnp.einsum("npc,nc->np", br["u_l"], c)
    temp = p["temperature"] * (1.0 + p["surprise_explore"] * br["unc"])
    prog = _act(br, logits, f_e, delta, p["eta_e"] * boost, temp, dt, key, code=c)
    eye = raw[:, P.EYES].reshape(n, 3, 3).max(-1)
    seen = jnp.where(eye.max(1) > 0.2, eye.argmax(1), 3)
    br["counts"] = br["counts"].at[jnp.arange(n), br["state_b"].argmax(1), seen].add(1.0)
    return prog


def _actor_critic(br, p, raw, reward, dt, key):
    a = _interface(br, raw, dt)
    f = jnp.concatenate([a, jnp.ones((raw.shape[0], 1), F)], 1)
    delta = _td(br, p, f, reward + (p["innate"] * raw).sum(1) * dt, dt)
    logits = (br["w_e"] * f[:, None, :]).sum(-1)
    return _act(br, logits, f, delta, p["eta_e"], p["temperature"], dt, key)


def _guided(raw, good):
    eye = raw[:, P.EYES].reshape(-1, 3, 3)
    idx = jnp.arange(raw.shape[0])
    g, bd = eye[idx, good], eye[idx, 1 - good]
    bad_ahead = (bd[:, 1] > 0.6) & (bd[:, 1] > g[:, 1])
    return jnp.where(bad_ahead, 4, jnp.where(g.max(-1) > 0.0, good, 3)).astype(jnp.int32)


def _reborn(br, dead, child, agent, sz, key):
    """I cervelli dei corpi rinati: ciò che si era imparato sparisce."""
    br = dict(br)
    if child is not None:
        br["genome"] = jnp.where(dead[:, None], child, br["genome"])
    if agent == "guidato":
        return br
    def zero(x):
        return jnp.where(dead.reshape((-1,) + (1,) * (x.ndim - 1)), jnp.zeros_like(x), x)
    for k in ("a_slow", "w_e", "e_e", "w_v", "e_v", "v_prev"):
        br[k] = zero(br[k])
    br["fresh"] = br["fresh"] | dead
    if agent == "sezioni":
        for k in ("pred", "state_b", "prev_b", "mem", "w_gate", "e_gate", "u_c", "e_c", "u_l", "code", "d_fast",
                  "unc", "counts"):
            br[k] = zero(br[k])
        fresh = jax.random.uniform(key, br["proto"].shape, F) * 0.3
        br["proto"] = jnp.where(dead[:, None, None], fresh, br["proto"])
        br["win_rate"] = jnp.where(dead[:, None], 1.0 / sz.b, br["win_rate"])
        br["d_slow"] = jnp.where(dead, 1e-3, br["d_slow"])
    return br


def make_run(cfg: P.PileConfig, agent: str, phase: str, sizes: S.Sizes):
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
        p = decode(br["genome"])
        reward = misc["reward"]
        if agent == "guidato":
            prog = _guided(flat, jnp.repeat(w["good"], n))
            br["program"], br["pi"] = prog, jax.nn.one_hot(prog, S.N_PROGRAMS)
        elif agent == "attore-critico":
            prog = _actor_critic(br, p, flat, reward, dt, k_brain)
        else:
            prog = _sections(br, p, flat, reward, dt, sizes, les, k_brain)
        thrust, steer, wander, escape = motor(cfg, prog.reshape(B, n), raw, w["wander"], w["escape"], k_motor)
        if agent == "guidato":  # sa anche dov'è la zona
            to = centres[w["zone"]][:, None, :] - w["pos"]
            ang = _wrap(jnp.arctan2(to[..., 1], to[..., 0]) - w["heading"])
            go = (prog.reshape(B, n) == 3) & (jnp.hypot(to[..., 0], to[..., 1]) > cfg.zone_half)
            steer = jnp.where(go, jnp.clip(2.0 * ang, -1.0, 1.0), steer)
        w = dict(w, wander=wander, escape=escape)
        w, out = world_step(cfg, w, thrust, steer, k_world)
        misc["reward"] = (out["pleasure"] - out["pain"]).reshape(N)
        misc["usage"] = misc["usage"] + jnp.bincount(prog, length=S.N_PROGRAMS)
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
            mix = jnp.where(jax.random.uniform(kc2, pa.shape) < 0.5, pa, pb)
            mix = mix + 0.1 * jax.random.normal(kc3, mix.shape, F)
            ng = len(S.GENE_NAMES)
            mix = mix.at[..., ng:].set(jnp.clip(mix[..., ng:], -S.INNATE_CAP, S.INNATE_CAP))
            child = mix.reshape(N, -1)
        br = _reborn(br, dead, child, agent, sizes, k_reborn)
        misc["reward"] = jnp.where(dead, 0.0, misc["reward"])
        series = {k: out[k] for k in ("good_taken", "bad_taken", "starved", "aged", "matings", "inversion",
                                       "zone_change")}
        return {"world": w, "brain": br, "misc": misc}, series

    def run(state, key, k):
        keys = jax.random.split(key, k)
        return jax.lax.scan(step, state, keys)

    return jax.jit(run, static_argnums=2)


class PileGpu:
    """Una corsa nell'arena delle pile sulla GPU."""

    def __init__(self, cfg: P.PileConfig, agent: str, phase: str, enclosures: int, seed: int = 0,
                 lesions=(), genomes=None):
        cfg.mating = phase == "B"
        self.cfg, self.agent, self.phase = cfg, agent, phase
        self.sizes = S.Sizes(lesions=frozenset(lesions))
        B, n = enclosures, cfg.beings
        self.B, self.n = B, n
        world = init_world(cfg, B, seed)
        brain = init_brain(agent, B * n, cfg.dt, seed, genomes, self.sizes)
        misc = dict(reward=np.zeros(B * n), usage=np.zeros(S.N_PROGRAMS), pair=np.zeros((B, 2, S.N_GENOME)),
                    has_pair=np.zeros(B, bool))
        to = lambda x: jnp.asarray(x, F) if np.asarray(x).dtype.kind == "f" else jnp.asarray(x)
        self.state = jax.tree_util.tree_map(to, {"world": world, "brain": brain, "misc": misc})
        self.key = jax.random.PRNGKey(seed)
        self._run = make_run(cfg, agent, phase, self.sizes)
        self.series = {k: [] for k in ("good_taken", "bad_taken", "starved", "aged", "matings", "inversion",
                                       "zone_change")}
        self.t = 0.0

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
