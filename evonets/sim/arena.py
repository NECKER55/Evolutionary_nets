"""L'arena continua (docs/SPEC_ARENA.md): tutti gli esseri in un mondo unico,
grande e senza bordi (toroidale: chi esce da un lato rientra dall'altro).

Il mondo ha un numero fisso di posti (`arena_cap`): un posto è vivo o
libero. Nascere è occupare un posto libero (lo fa l'ecologia, evo/ecology.py),
morire è liberarlo: di fame (energia a zero) o di vecchiaia (probabilità che
cresce con l'età, legge di Gompertz). Non ci sono compiti né generazioni.

Il cibo ricresce a macchie fertili che si spostano piano e cambiano con le
stagioni (`Patches`); chi lo tocca per primo lo mangia. Gli esseri si vedono
(dischi opachi del loro colore), si annusano (una molecola loro) e si urtano.

I sensi sono quelli di sempre (net/receptors.py): per la vista, la pressione
e gli odori il mondo offre `light_of`, `pressure_of` e `ArenaFood.odors`, che
guardano solo gli oggetti vicini grazie a una griglia di celle. La luce si
spegne in poche decine di unità (exp(-d / light_attenuation)): oltre le celle
vicine non arriva quasi niente. Nessuna parete: un raggio che non incontra
niente non porta luce.
"""

import numpy as np
from numba import njit, prange

from .hazards import HazardField
from .world import World


# --- La griglia dei vicini (compilata) -----------------------------------------

@njit(cache=True)
def _wrap(d, size):
    """La differenza d più corta sul toro di lato `size`."""
    return d - size * np.floor(d / size + 0.5)


@njit(cache=True)
def grid(pos, use, cw, ch, ncx, ncy):
    """Gli oggetti `use` raggruppati per cella: `start` (celle + 1) e
    `order`, gli indici ordinati per cella (quelli della cella c stanno in
    order[start[c]:start[c + 1]])."""
    n = pos.shape[0]
    count = np.zeros(ncx * ncy + 1, dtype=np.int64)
    cell = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if use[i]:
            cx = int(np.floor(pos[i, 0] / cw)) % ncx
            cy = int(np.floor(pos[i, 1] / ch)) % ncy
            c = cy * ncx + cx
            cell[i] = c
            count[c + 1] += 1
    start = np.cumsum(count)
    order = np.empty(start[-1], dtype=np.int64)
    fill = start[:-1].copy()
    for i in range(n):
        c = cell[i]
        if c >= 0:
            order[fill[c]] = i
            fill[c] += 1
    return start, order


@njit(cache=True)
def _ray_disc(ox, oy, dx, dy, cx, cy, r):
    """Distanza lungo il raggio (o, d) fino al disco (c, r) già portato vicino
    a o; infinito se non lo colpisce."""
    proj = cx * dx + cy * dy
    disc = r * r - (cx * cx + cy * cy) + proj * proj
    if disc < 0.0:
        return np.inf
    half = np.sqrt(disc)
    if proj + half < 0.0:
        return np.inf
    return max(proj - half, 0.0)


@njit(cache=True, parallel=True)
def light_kernel(origins, angles, agent, W, H, cw, ch, ncx, ncy, L,
                 food, food_r, f_start, f_order, beings, b_r, b_start, b_order):
    """Luce di cibo e di esseri (K,) che arriva a K recettori, mediata sulle
    loro direzioni campione: per ogni raggio conta solo il primo oggetto
    opaco colpito, attenuato come exp(-d / L). L'essere a cui appartiene il
    recettore (`agent`) non si vede."""
    k_all, m = angles.shape
    out_food = np.zeros(k_all)
    out_being = np.zeros(k_all)
    for k in prange(k_all):
        ox = origins[k, 0] % W
        oy = origins[k, 1] % H
        a = agent[k]
        cx0 = int(np.floor(ox / cw)) % ncx
        cy0 = int(np.floor(oy / ch)) % ncy
        for j in range(m):
            dx, dy = np.cos(angles[k, j]), np.sin(angles[k, j])
            best = np.inf
            kind = -1
            for iy in range(-1, 2):
                for ix in range(-1, 2):
                    c = ((cy0 + iy) % ncy) * ncx + (cx0 + ix) % ncx
                    for q in range(f_start[c], f_start[c + 1]):
                        o = f_order[q]
                        d = _ray_disc(ox, oy, dx, dy, _wrap(food[o, 0] - ox, W), _wrap(food[o, 1] - oy, H), food_r)
                        if d < best:
                            best, kind = d, 0
                    for q in range(b_start[c], b_start[c + 1]):
                        o = b_order[q]
                        if o == a:
                            continue
                        d = _ray_disc(ox, oy, dx, dy, _wrap(beings[o, 0] - ox, W), _wrap(beings[o, 1] - oy, H), b_r)
                        if d < best:
                            best, kind = d, 1
            if kind == 0:
                out_food[k] += np.exp(-best / L)
            elif kind == 1:
                out_being[k] += np.exp(-best / L)
        out_food[k] /= m
        out_being[k] /= m
    return out_food, out_being


@njit(cache=True, parallel=True)
def odor_kernel(points, agent, W, H, cw, ch, ncx, ncy, R,
                food, food_r, f_start, f_order, beings, b_r, b_start, b_order):
    """Concentrazione dell'odore del cibo e degli esseri nei punti (K, 2):
    somma di exp(-d / R), d dal bordo, sugli oggetti delle celle vicine
    (oltre, l'odore è già sotto il 4%). Il proprio odore non si sente."""
    k_all = points.shape[0]
    c_food = np.zeros(k_all)
    c_being = np.zeros(k_all)
    for k in prange(k_all):
        px = points[k, 0] % W
        py = points[k, 1] % H
        a = agent[k]
        cx0 = int(np.floor(px / cw)) % ncx
        cy0 = int(np.floor(py / ch)) % ncy
        for iy in range(-1, 2):
            for ix in range(-1, 2):
                c = ((cy0 + iy) % ncy) * ncx + (cx0 + ix) % ncx
                for q in range(f_start[c], f_start[c + 1]):
                    o = f_order[q]
                    d = np.hypot(_wrap(food[o, 0] - px, W), _wrap(food[o, 1] - py, H)) - food_r
                    c_food[k] += np.exp(-max(d, 0.0) / R)
                for q in range(b_start[c], b_start[c + 1]):
                    o = b_order[q]
                    if o == a:
                        continue
                    d = np.hypot(_wrap(beings[o, 0] - px, W), _wrap(beings[o, 1] - py, H)) - b_r
                    c_being[k] += np.exp(-max(d, 0.0) / R)
    return c_food, c_being


@njit(cache=True)
def touch_kernel(pos, agent, normals, W, H, cw, ch, ncx, ncy, r, b_start, b_order):
    """Pressione (K,) sui recettori la cui normale guarda un altro essere che
    il loro sta toccando."""
    k_all = agent.shape[0]
    out = np.zeros(k_all)
    for k in range(k_all):
        a = agent[k]
        px, py = pos[a, 0], pos[a, 1]
        cx0 = int(np.floor(px / cw)) % ncx
        cy0 = int(np.floor(py / ch)) % ncy
        for iy in range(-1, 2):
            for ix in range(-1, 2):
                c = ((cy0 + iy) % ncy) * ncx + (cx0 + ix) % ncx
                for q in range(b_start[c], b_start[c + 1]):
                    o = b_order[q]
                    if o == a:
                        continue
                    dx, dy = _wrap(pos[o, 0] - px, W), _wrap(pos[o, 1] - py, H)
                    d = np.hypot(dx, dy)
                    if 1e-9 < d <= 2.0 * r + 1e-6:
                        out[k] += max(0.0, (dx * normals[k, 0] + dy * normals[k, 1]) / d)
    return out


@njit(cache=True)
def collide_kernel(pos, vel, alive, W, H, cw, ch, ncx, ncy, r, b_start, b_order):
    """Gli esseri non si sovrappongono: due dischi che si compenetrano si
    separano di metà ciascuno, e la velocità con cui si avvicinavano si
    annulla (urto anelastico). Restituisce chi tocca un altro essere."""
    n = pos.shape[0]
    push = np.zeros((n, 2))
    touching = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if not alive[i]:
            continue
        cx0 = int(np.floor(pos[i, 0] / cw)) % ncx
        cy0 = int(np.floor(pos[i, 1] / ch)) % ncy
        for iy in range(-1, 2):
            for ix in range(-1, 2):
                c = ((cy0 + iy) % ncy) * ncx + (cx0 + ix) % ncx
                for q in range(b_start[c], b_start[c + 1]):
                    j = b_order[q]
                    if j == i:
                        continue
                    dx, dy = _wrap(pos[j, 0] - pos[i, 0], W), _wrap(pos[j, 1] - pos[i, 1], H)
                    d = np.hypot(dx, dy)
                    if d < 2.0 * r:
                        touching[i] = True
                        if d < 1e-9:
                            dx, dy, d = 1.0, 0.0, 1.0
                        ux, uy = dx / d, dy / d
                        overlap = 2.0 * r - d
                        push[i, 0] -= 0.5 * overlap * ux
                        push[i, 1] -= 0.5 * overlap * uy
                        closing = vel[i, 0] * ux + vel[i, 1] * uy
                        if closing > 0.0:
                            vel[i, 0] -= closing * ux
                            vel[i, 1] -= closing * uy
    for i in range(n):
        pos[i, 0] = (pos[i, 0] + push[i, 0]) % W
        pos[i, 1] = (pos[i, 1] + push[i, 1]) % H
    return touching


@njit(cache=True)
def eat_kernel(food, food_on, pos, alive, W, H, cw, ch, ncx, ncy, reach, b_start, b_order):
    """Per ogni cibo, chi lo mangia in questo passo: l'essere vivo più vicino
    tra quelli che lo toccano (-1 nessuno)."""
    f = food.shape[0]
    eater = np.full(f, -1, dtype=np.int64)
    for o in range(f):
        if not food_on[o]:
            continue
        cx0 = int(np.floor(food[o, 0] / cw)) % ncx
        cy0 = int(np.floor(food[o, 1] / ch)) % ncy
        best = reach
        for iy in range(-1, 2):
            for ix in range(-1, 2):
                c = ((cy0 + iy) % ncy) * ncx + (cx0 + ix) % ncx
                for q in range(b_start[c], b_start[c + 1]):
                    a = b_order[q]
                    if not alive[a]:
                        continue
                    d = np.hypot(_wrap(pos[a, 0] - food[o, 0], W), _wrap(pos[a, 1] - food[o, 1], H))
                    if d <= best:
                        best, eater[o] = d, a
    return eater


# --- Il cibo -----------------------------------------------------------------

class Patches:
    """Le macchie fertili: centri che derivano (passeggiata casuale lenta,
    `arena_patch_speed`, direzione che cambia di `arena_patch_turn` rad/√s) e
    fertilità che oscilla con le stagioni, ogni macchia con la sua fase. La
    fertilità nel punto x è arena_background + Σ F_k · exp(-|x - c_k|² / 2σ²)."""

    def __init__(self, cfg, W: float, H: float, rng: np.random.Generator):
        k = cfg.arena_patches
        self.cfg, self.W, self.H = cfg, W, H
        self.centre = rng.uniform([0.0, 0.0], [W, H], (k, 2))
        self.heading = rng.uniform(-np.pi, np.pi, k)
        self.phase = rng.uniform(0.0, 2.0 * np.pi, k)

    def fertility(self, t: float) -> np.ndarray:
        """(k,) la fertilità di ogni macchia all'istante t, tra arena_season_low e 1."""
        cfg = self.cfg
        wave = 0.5 + 0.5 * np.sin(2.0 * np.pi * t / cfg.arena_season + self.phase)
        return cfg.arena_season_low + (1.0 - cfg.arena_season_low) * wave

    def step(self, dt: float, rng: np.random.Generator) -> None:
        cfg = self.cfg
        self.heading += cfg.arena_patch_turn * np.sqrt(dt) * rng.standard_normal(len(self.heading))
        move = cfg.arena_patch_speed * dt * np.stack([np.cos(self.heading), np.sin(self.heading)], axis=1)
        self.centre = (self.centre + move) % [self.W, self.H]

    def sample(self, n: int, t: float, rng: np.random.Generator) -> np.ndarray:
        """(n, 2) punti estratti con probabilità proporzionale alla fertilità."""
        cfg = self.cfg
        sigma = cfg.arena_patch_sigma
        mass = np.append(self.fertility(t) * 2.0 * np.pi * sigma ** 2, cfg.arena_background * self.W * self.H)
        which = rng.choice(len(mass), n, p=mass / mass.sum())
        out = rng.uniform([0.0, 0.0], [self.W, self.H], (n, 2))  # il fondo: ovunque
        inside = which < len(self.centre)
        out[inside] = self.centre[which[inside]] + sigma * rng.standard_normal((int(inside.sum()), 2))
        return out % [self.W, self.H]


class ArenaFood:
    """Il "compito" dell'arena: nessuno, si vive. Tiene il cibo (posti fissi,
    acceso o no), lo fa ricrescere dalle macchie e fa mangiare chi lo tocca.
    Per i sensi offre gli odori; la luce la dà il mondo (`ArenaWorld.light_of`)."""

    name = "arena continua"

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_pred = 0
        self.n_rock = 0

    def reset(self, world) -> None:
        cfg = self.cfg
        self.food_r = cfg.arena_food_radius
        self.patches = Patches(cfg, world.W, world.H, world.rng)
        k = cfg.arena_food_max
        self.food = np.zeros((k, 2))
        self.food_on = np.zeros(k, dtype=bool)
        self.eaten = np.zeros(world.n, dtype=int)  # cibi mangiati da ogni posto (da quando è nato)
        self.eaten_total = 0
        self._owed = 0.0
        self.grow(world, k // 2)  # si parte con metà del cibo possibile

    def grow(self, world, n: int) -> None:
        """Fino a `n` cibi nuovi nei posti liberi, dove la fertilità è alta."""
        free = np.flatnonzero(~self.food_on)[:n]
        if len(free):
            self.food[free] = self.patches.sample(len(free), world.t, world.rng)
            self.food_on[free] = True

    def step(self, world) -> None:
        cfg = self.cfg
        dt = cfg.dt
        self.patches.step(dt, world.rng)
        self._owed += cfg.arena_food_rate * dt
        new = int(self._owed)
        if new:
            self._owed -= new
            self.grow(world, new)
        g = world.grids()
        eater = eat_kernel(self.food, self.food_on, world.pos, world.alive, world.W, world.H, *g["cells"],
                           self.food_r + cfg.agent_radius, *g["beings"])
        ate = np.flatnonzero(eater >= 0)
        if len(ate):
            who = eater[ate]
            self.food_on[ate] = False
            n_eat = np.bincount(who, minlength=world.n)
            self.eaten += n_eat
            self.eaten_total += len(ate)
            world.taste = np.minimum(world.taste + n_eat, 1.0)  # il gusto del cibo
            world.energy = np.minimum(world.energy + cfg.food_energy * n_eat, cfg.energy_max)
            world.grids_dirty = True

    def odors(self, world, points: np.ndarray, agent: np.ndarray) -> list:
        cfg = self.cfg
        g = world.grids()
        c_food, c_being = odor_kernel(np.ascontiguousarray(points, dtype=float), np.ascontiguousarray(agent),
                                      world.W, world.H, *g["cells"], cfg.odor_range, self.food, self.food_r,
                                      *g["food"], world.pos, cfg.agent_radius, *g["beings"])
        return [(cfg.food_molecule, cfg.food_smell * c_food), (cfg.being_molecule, cfg.being_smell * c_being)]

    def completed(self, world) -> np.ndarray:
        return np.zeros(world.n, dtype=bool)


# --- Il mondo ------------------------------------------------------------------

class ArenaWorld(World):
    """Il mondo continuo: `n` posti (la riga i è il posto i), di cui vivi
    quelli con `alive`. Si muove come `World` ma senza pareti (toroidale) e
    con gli urti tra esseri; si muore di fame e di vecchiaia. Le nascite le
    decide l'ecologia (`spawn`)."""

    def __init__(self, cfg, n_slots: int, n_initial: int, seed: int = 0):
        self.n_initial = min(n_initial, n_slots)
        self.W, self.H = float(cfg.arena_size_w), float(cfg.arena_size_h)
        ncx = max(3, int(self.W // cfg.arena_cell))
        ncy = max(3, int(self.H // cfg.arena_cell))
        self.cells = (self.W / ncx, self.H / ncy, ncx, ncy)
        super().__init__(cfg, n_slots, seed, task=ArenaFood(cfg))

    def reset(self) -> None:
        cfg = self.cfg
        self.rng = np.random.default_rng(self.seed)
        self.t = 0.0
        self.hazards = HazardField(_no_hazards(cfg), self.rng)
        self.keep_living = True
        self.grids_dirty = True
        self._grids = None
        n = self.n
        self.alive = np.zeros(n, dtype=bool)
        self.alive[: self.n_initial] = True
        self.pos = np.zeros((n, 2))
        self.heading = self.rng.uniform(-np.pi, np.pi, n)
        self.vel = np.zeros((n, 2))
        self.ang_vel = np.zeros(n)
        self.energy = np.where(self.alive, cfg.energy_init, 0.0)
        self.age = np.zeros(n)  # s da quando il posto è nato
        self.done = np.zeros(n, dtype=bool)
        self.pain = np.zeros(n)
        self.taste = np.zeros(n)
        self.bump = np.zeros(n, dtype=bool)  # a contatto con un altro essere
        self.deaths = {"fame": 0, "vecchiaia": 0}  # morti da quando il mondo è nato
        self.task.reset(self)
        self.pos[self.alive] = self.task.patches.sample(int(self.alive.sum()), 0.0, self.rng)

    def spawn(self, slots: np.ndarray, pos: np.ndarray, energy) -> None:
        """Nascono esseri nei posti `slots`, in `pos`, con `energy`."""
        self.alive[slots] = True
        self.pos[slots] = pos % [self.W, self.H]
        self.heading[slots] = self.rng.uniform(-np.pi, np.pi, len(slots))
        self.vel[slots] = 0.0
        self.ang_vel[slots] = 0.0
        self.energy[slots] = energy
        self.age[slots] = 0.0
        self.taste[slots] = 0.0
        self.pain[slots] = 0.0
        self.task.eaten[slots] = 0
        self.grids_dirty = True

    def grids(self) -> dict:
        """Le griglie dei vicini (cibo ed esseri vivi), rifatte quando
        qualcosa si è mosso, è nato, è morto o è stato mangiato."""
        if self.grids_dirty or self._grids is None:
            cells = self.cells
            task = self.task
            self._grids = {"cells": cells,
                           "food": grid(task.food, task.food_on, *cells),
                           "beings": grid(self.pos, self.alive, *cells)}
            self.grids_dirty = False
        return self._grids

    def step(self, actions: np.ndarray, upkeep: np.ndarray | None = None) -> None:
        """Come `World.step`, senza pareti e con gli urti; poi si mangia, si
        invecchia e si muore."""
        cfg = self.cfg
        dt = cfg.dt
        active = self.alive
        act = np.clip(actions, -1.0, 1.0) * active[:, None]
        thrust = np.where(act[:, 0] < 0.0, act[:, 0] * cfg.reverse_factor, act[:, 0])
        turn = act[:, 1]
        self.ang_vel += turn * cfg.turn_accel * dt
        self.ang_vel *= np.exp(-cfg.ang_drag * dt)
        self.heading += self.ang_vel * dt
        self.heading = (self.heading + np.pi) % (2.0 * np.pi) - np.pi
        fwd = np.stack([np.cos(self.heading), np.sin(self.heading)], axis=1)
        v_fwd = np.einsum("ij,ij->i", self.vel, fwd)
        v_lat = self.vel - v_fwd[:, None] * fwd
        v_fwd = (v_fwd + thrust * cfg.thrust_accel * dt) * np.exp(-cfg.lin_drag * dt)
        v_lat *= np.exp(-cfg.lat_drag * dt)
        self.vel = v_fwd[:, None] * fwd + v_lat
        self.vel[~active] = 0.0
        self.ang_vel[~active] = 0.0
        self.pos = (self.pos + self.vel * dt) % [self.W, self.H]
        self.grids_dirty = True
        g = self.grids()
        self.bump = collide_kernel(self.pos, self.vel, self.alive, self.W, self.H, *g["cells"], cfg.agent_radius,
                                   *g["beings"])
        self.grids_dirty = True

        self.pain = np.zeros(self.n)
        self.taste *= np.exp(-dt / cfg.taste_tau)
        mass = 1.0 + cfg.reserve_mass * np.maximum(self.energy - cfg.energy_init, 0.0) / cfg.energy_init
        effort = (
            cfg.metabolism_basal
            + mass * cfg.metabolism_move * np.abs(act[:, 0]) ** cfg.metabolism_exponent
            + mass * cfg.metabolism_turn * np.abs(turn) ** cfg.metabolism_exponent
            + (0.0 if upkeep is None else upkeep)
        ) * active
        self.energy = np.maximum(self.energy - effort * dt, 0.0)
        self.task.step(self)

        starved = active & (self.energy <= 0.0)
        self.age += dt * self.alive
        hazard = cfg.arena_aging_h0 * np.exp(self.age / cfg.arena_aging_tau)
        old = self.alive & ~starved & (self.rng.random(self.n) < hazard * dt)
        self.deaths["fame"] += int(starved.sum())
        self.deaths["vecchiaia"] += int(old.sum())
        self.alive &= ~(starved | old)
        self.energy[~self.alive] = 0.0
        self.t += dt

    def light_of(self, origins: np.ndarray, angles: np.ndarray, agent: np.ndarray | None) -> dict:
        """Come `signals.light`, nel mondo continuo: cibo ed esseri vicini,
        nessuna parete, nessun bersaglio, nessun veleno."""
        cfg = self.cfg
        k = len(origins)
        if agent is None:
            agent = np.full(k, -1, dtype=np.int64)
        g = self.grids()
        food, being = light_kernel(np.ascontiguousarray(origins, dtype=float),
                                   np.ascontiguousarray(angles, dtype=float), np.ascontiguousarray(agent),
                                   self.W, self.H, *g["cells"], float(cfg.light_attenuation),
                                   self.task.food, self.task.food_r, *g["food"], self.pos, cfg.agent_radius,
                                   *g["beings"])
        zero = np.zeros(k)
        return {"parete": zero, "bersaglio": zero, "veleno": zero, "cibo": food, "predatore": zero,
                "roccia": zero, "frutto": zero, "essere": being}

    def pressure_of(self, agent: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Come `signals.pressure`: nessuna parete, si sentono gli altri esseri toccati."""
        g = self.grids()
        return touch_kernel(self.pos, np.ascontiguousarray(agent), np.ascontiguousarray(normals, dtype=float),
                            self.W, self.H, *g["cells"], self.cfg.agent_radius, *g["beings"])

    @property
    def racing(self) -> np.ndarray:
        return self.alive

    @property
    def over(self) -> bool:
        return False


def _no_hazards(cfg):
    """La stessa configurazione senza zone di veleno (l'arena non ne ha)."""
    import dataclasses

    return dataclasses.replace(cfg, n_hazards=0)
