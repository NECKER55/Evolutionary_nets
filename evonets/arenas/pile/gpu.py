"""L'arena delle pile sulla GPU (JAX): lo stesso algoritmo di `world.py`, scritto
come funzioni pure e compilato insieme ai cervelli (`brains/pile/gpu.py`),
con molti passi alla volta (`lax.scan`). I numeri casuali sono diversi da
quelli della versione NumPy, quindi le corse non coincidono passo per passo:
coincidono le regole (tests/test_pile.py le confronta pezzo per pezzo e
nelle medie).

Stato: un dizionario di array; quello del mondo ha come prime dimensioni
(recinto, essere), quello del cervello (essere,) con gli esseri in fila
(recinto · n + essere), come nella versione NumPy.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ...brains import pile as S
from ...brains.pile import gpu as BG
from . import world as P

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
    starved = energy <= 0.0  # solo per fame: nessun tetto di età
    dead = starved
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
                good_taken=(took & ok).sum(), bad_taken=bad.sum(), starved=starved.sum(),
                matings=mated.sum() // 2, inversion=inv, zone_change=zc)
    return w, outs


# --- I cervelli ----------------------------------------------------------------------


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
        p = BG.decode(br["genome"])
        reward = misc["reward"]
        clock = br["clock"] + dt
        decide = clock >= p["decide_every"] - 1e-6  # ognuno al suo ritmo
        br["clock"] = jnp.where(decide, 0.0, clock)
        if agent == "guidato":
            prog = BG.guided(br, objs, decide, p, dt, k_brain)
        elif agent == "attore-critico":
            prog = BG.actor_critic(br, p, flat, objs, reward, dt, decide, k_brain)
        else:
            prog = BG.sections(br, p, flat, objs, reward, dt, sizes, les, decide, k_brain, cfg.beings)
        has, _, near, ang = BG.seen_of(br, objs)
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
        br = BG.reborn(br, dead, child, agent, sizes, k_reborn, blocks)
        misc["reward"] = jnp.where(dead[:, None], 0.0, misc["reward"])
        series = {k: out[k] for k in ("starved", "matings", "inversion", "zone_change")}
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
        brain, self.blocks = BG.init_brain(agent, B * n, cfg.dt, seed, cfg, genomes, self.sizes)
        misc = dict(reward=np.zeros((B * n, 3)), usage=np.zeros(S.N_PROGRAMS), pair=np.zeros((B, 2, brain["genome"].shape[1])),
                    has_pair=np.zeros(B, bool), good_by=np.zeros(B * n), bad_by=np.zeros(B * n),
                    deaths_by=np.zeros(B * n), counted=np.ones(B * n))
        to = lambda x: jnp.asarray(x, F) if np.asarray(x).dtype.kind == "f" else jnp.asarray(x)
        self.state = jax.tree_util.tree_map(to, {"world": world, "brain": brain, "misc": misc})
        self.key = jax.random.PRNGKey(seed)
        self._run = make_run(cfg, agent, phase, self.sizes, self.blocks)
        self.series = {k: [] for k in ("good_taken", "bad_taken", "starved", "matings", "inversion", "zone_change")}
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
        br = BG.reborn(br, gone, None, self.agent, self.sizes, sub, self.blocks)  # al posto del maestro, un giovane
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
