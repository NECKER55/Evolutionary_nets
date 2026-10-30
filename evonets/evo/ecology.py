"""La vita nell'arena continua (docs/SPEC_ARENA.md): niente generazioni. Il
mondo (sim/arena.py) scorre; ogni `arena_tick` secondi si contano le nascite:
chi ha abbastanza energia, è adulto e non ha appena partorito fa un figlio,
da solo o, se un compatibile è vicino, con un partner. Il figlio occupa un
posto libero, nasce accanto al genitore e passa dal grembo se il genoma è
"sviluppo". Ogni `arena_stats_every` secondi una riga di storia, il campione
(il più anziano vivo) e, se è il momento, l'esame fisso su adulti presi a
caso. I salvataggi sono propri (`kind` = "arena") e si riprendono con le
regole più recenti, come gli altri.
"""

import dataclasses
import json
import os
from pathlib import Path

import numpy as np

from ..config import Config
from ..net import womb
from ..net.brain import META, Genome, fit_traits
from ..net.controllers import NeuralController
from ..net.development import develop, population_stats
from ..net.geometry import brain_side, connect_prob, distances
from ..sim.arena import ArenaWorld, Pools, temperature
from .exam import Exam
from .genetics import crossover, mutate as _mutate_genes
from .saves import FORMAT, RUN_STATE, code_version

# Lo stato di una corsa nell'arena che viene dal salvataggio (oltre a
# RUN_STATE): le dimensioni del mondo e dei posti.
ARENA_STATE = ("arena_cap", "arena_size_w", "arena_size_h")
# Le manopole del mondo che si complica, nell'ordine in cui il mondo le alza
# (docs/SPEC_ARENA.md, sezione 15). Vengono dal salvataggio: sono lo stato del
# mondo, non regole.
DIALS = ("arena_scarcity", "arena_thirst", "arena_pool_distance", "arena_depletion", "arena_fruits", "arena_reversal", "arena_predation",
         "arena_extraction", "arena_night", "arena_drought")
DIAL_NAMES = ("cibo raro", "sete", "pozze lontane", "macchie che si esauriscono", "frutti cattivi", "inversione dei colori", "predazione",
              "cibo da aprire", "notte", "siccità")
# Le regole che esistono solo con il motore JAX (con NumPy si spengono).
ONLY_GPU = ("arena_temperature", "arena_water")
EXTRA_MOTORS = 4  # mordere, luce, odore, nutrire
ADULT = 60.0  # s: età minima per l'esame
# Lo stato del mondo nei salvataggi, con il prefisso "w_" (pos è anche un
# campo del genoma: la posizione dei neuroni nel cervello).
WORLD = ("pos", "heading", "vel", "ang_vel", "energy", "age", "taste")


class Ecology:
    """Una corsa nell'arena continua: `world` (i posti, il cibo), `ctrl` (un
    cervello per posto), la storia, il registro, gli esami, i campioni."""

    def __init__(self, cfg: Config, seed: int = 0, initial: int | None = None):
        cfg.task, cfg.n_hazards, cfg.ladder = "arena", 0, ""
        cfg.population = cfg.arena_cap
        self.cfg, self.seed = cfg, seed
        initial = cfg.arena_cap // 4 if initial is None else min(initial, cfg.arena_cap)
        self.world = ArenaWorld(cfg, cfg.arena_cap, initial, seed)
        self.ctrl = NeuralController(cfg, cfg.arena_cap, seed, EXTRA_MOTORS)
        self.rng = np.random.default_rng([seed, 11])  # nascite, partner, esami
        if cfg.arena_hidden_initial > 0:
            g = self.ctrl.brains.genome
            self._primordial(g)
            self.ctrl.brains.set_genome(g)
        n = cfg.arena_cap
        # Il corpo (geni dell'arena, fuori dal genoma del cervello): taglia (in
        # scala logaritmica), colore della luce, molecola dell'odore; e chi è il
        # genitore (numero, -1 nessuno), per le cure.
        self.size_gene = self.rng.normal(0.0, cfg.arena_size_std, n)
        self.sig_color = self.rng.random(n)
        self.sig_mol = self.rng.random(n)
        self.parent = np.full(n, -1, dtype=np.int64)
        self.ids = np.arange(n, dtype=np.int64)
        self.next_id = n
        self.last_birth = np.full(n, -np.inf)
        self.totals = {"nati": 0, "con partner": 0, "colonizzatori": 0}
        # I migliori tra i vissuti (cibi mangiati, genoma, corpo), per i colonizzatori.
        self.archive: list[tuple[int, Genome, np.ndarray]] = []
        self._food_seen = self.world.task.food_on.copy()  # per sapere quali cibi sono stati mangiati
        self.next_reversal = cfg.arena_reversal_every
        self._dial_at = None  # minuto dell'ultimo cambio di una manopola del mondo
        self._probe = None  # (genomi, numeri) dell'ultimo esame: la prova senza plasticità
        self._was_alive = self.world.alive.copy()
        self.history: list[dict] = []
        self.log: list[dict] = [{"time": 0.0, "event": "inizio", "code": code_version()}]
        self.exams: list[dict] = []
        self.champions: list[tuple] = []
        self.exam: Exam | None = None
        self._last = {"nati": 0, "con partner": 0, "colonizzatori": 0, "fame": 0, "vecchiaia": 0, "mangiati": 0}
        self.next_tick = cfg.arena_tick
        self.next_stats = cfg.arena_stats_every
        self._food_sum, self._food_n = 0, 0  # cibo presente, sommato a ogni secondo del minuto
        if self.ctrl.brains.genome.n_markers:
            self._womb(np.flatnonzero(self.world.alive))
        self.gpu = None  # l'arena sulla GPU (gpu/arena.py), se c'è
        self._stale = False  # cervelli compilati sul processore da rifare (con la GPU)
        water = cfg.arena_water and cfg.engine == "gpu"
        # L'acqua: la riserva di ognuno e le pozze (con un generatore proprio).
        w = self.world
        self.pool_rng = np.random.default_rng([seed, 13])
        w.water = np.where(w.alive, self._full_water(np.arange(n)), 0.0)
        w.drunk_total = 0.0
        w.deaths.setdefault("sete", 0)
        w.pools = Pools(cfg, w.W, w.H, self.pool_rng, w.task.patches, start=water)
        self._drinks_reset()
        if cfg.engine == "gpu":
            self._start_gpu()
        if self.gpu is None:  # il mondo che si complica, il clima e l'acqua ci sono solo con il motore JAX
            for k in DIALS:
                setattr(cfg, k, 0.0)
            cfg.arena_temperature = 0.0
            cfg.arena_water = False
            w.pools.volume[:] = 0.0

    def _start_gpu(self, key=None) -> None:
        from ..gpu import available

        if not available():
            return
        from ..gpu.arena import ArenaGpu

        self.gpu = ArenaGpu(self.world, self.ctrl, self.seed, self.bodies())
        self.world.task.patches.climate = self.cfg.arena_temperature > 0
        if key is not None:
            import jax.numpy as jnp

            self.gpu.s["key"] = jnp.asarray(key, dtype=jnp.uint32)

    # --- La misura delle bevute ------------------------------------------------
    # Una volta al secondo: chi sta bevendo (l'acqua è salita). Una bevuta
    # comincia quando si inizia a bere e finisce quando si smette; l'intervallo
    # è il tempo dalla fine di una bevuta (o dalla nascita) all'inizio della
    # successiva. Per ogni essere si tiene il primo intervallo e la media dei
    # successivi: se i successivi sono più brevi, torna alla pozza più in
    # fretta di quando l'ha trovata la prima volta (impara dove sta).

    def _drinks_reset(self) -> None:
        n = self.world.n
        self._prev_water = self.world.water.copy()
        self._drinking = np.zeros(n, bool)
        self._dry_since = np.zeros(n)  # fine dell'ultima bevuta (o nascita)
        self._first_gap = np.full(n, np.nan)
        self._later_sum, self._later_n = np.zeros(n), np.zeros(n, int)
        self._gaps: list[float] = []  # intervalli conclusi in questo minuto

    def _born(self, slots) -> None:
        """Le misure delle bevute ripartono per chi nasce nei posti `slots`."""
        self._drinking[slots] = False
        self._dry_since[slots] = self.world.t
        self._first_gap[slots] = np.nan
        self._later_sum[slots], self._later_n[slots] = 0.0, 0
        self._prev_water[slots] = self.world.water[slots]

    def _watch_drinks(self) -> None:
        w = self.world
        now = w.alive & (w.water > self._prev_water + 1e-6)
        start = np.flatnonzero(now & ~self._drinking)
        gap = w.t - self._dry_since[start]
        self._gaps.extend(gap.tolist())
        first = np.isnan(self._first_gap[start])
        self._first_gap[start[first]] = gap[first]
        self._later_sum[start[~first]] += gap[~first]
        self._later_n[start[~first]] += 1
        self._dry_since[self._drinking & ~now] = w.t  # chi ha smesso di bere
        self._drinking = now
        self._prev_water = w.water.copy()

    def drink_stats(self) -> dict:
        """Intervallo medio tra le bevute nel minuto, e quanto più in fretta
        tornano a bere i vivi che hanno già bevuto almeno due volte dopo la
        prima (mediana di intervalli successivi / primo intervallo)."""
        alive = np.flatnonzero(self.world.alive & (self._later_n >= 2) & (self._first_gap > 0))
        ratio = self._later_sum[alive] / self._later_n[alive] / self._first_gap[alive]
        out = {"drink_gap": round(float(np.mean(self._gaps)), 1) if self._gaps else None,
               "drink_bouts": len(self._gaps),
               "return_ratio": round(float(np.median(ratio)), 2) if len(alive) >= 5 else None,
               "return_n": len(alive)}
        self._gaps = []
        return out

    def _full_water(self, rows) -> np.ndarray:
        """La riserva d'acqua piena dei posti `rows` (dipende dalla taglia)."""
        return self.cfg.arena_water_max * self.sizes()[rows]

    def _mutate(self, kids: Genome) -> None:
        """Le mutazioni dei figli; poi (arena_wire_new, genoma diretto) i
        neuroni interni appena comparsi crescono i loro collegamenti con la
        regola della nascita: probabilità che cala con la distanza nel cervello
        e con la portata dell'individuo, in entrata e in uscita."""
        cfg = self.cfg
        before = kids.active.copy()
        _mutate_genes(kids, cfg, self.rng)
        if cfg.arena_wire_new:
            self._wire(kids, kids.active & ~before)

    def _primordial(self, g: Genome) -> None:
        """Gli esseri primordiali: arena_hidden_initial neuroni interni
        accesi (i primi posti). Con il genoma diretto si collegano con la
        regola della nascita; con lo sviluppo i collegamenti nascono dai
        marcatori, quindi si rifà lo sviluppo."""
        k = min(self.cfg.arena_hidden_initial, g.n_hidden)
        if k <= 0:
            return
        new = np.zeros_like(g.active)
        new[:, :k] = ~g.active[:, :k]
        g.active[:, :k] = True
        if g.n_markers:
            g.unborn()
            develop(g, self.cfg, self.rng, self.ctrl.brains.GAIN)
        else:
            self._wire(g, new)

    def _wire(self, kids: Genome, new: np.ndarray) -> None:
        """I neuroni interni `new` (N, H) fanno crescere dendriti e assone."""
        cfg = self.cfg
        if kids.n_markers or not new.any():
            return
        rows = np.flatnonzero(new.any(axis=1))
        g, R = kids, kids.n_rec
        dist = distances(cfg, g.pos[rows], g.r_y[rows], g.n_out, brain_side(cfg, g.r_active[rows], g.active[rows],
                                                                              g.n_out))
        reach = np.exp(g.meta[rows, META.index("conn_range")])
        p = connect_prob(cfg, dist, R, reach)  # (n, T, S)
        grow = self.rng.random(p.shape) < p
        for j, i in enumerate(rows):
            for k in np.flatnonzero(new[i]):
                g.conn[i, k, :] = grow[j, k, :]  # dendriti: da chi riceve
                g.conn[i, :, R + k] = grow[j, :, R + k]  # assone: a chi manda

    def sizes(self) -> np.ndarray:
        cfg = self.cfg
        return np.clip(np.exp(self.size_gene), cfg.arena_size_min, cfg.arena_size_max)

    def bodies(self) -> dict:
        """Il corpo di ogni posto, per la GPU."""
        w = self.world
        return {"size": self.sizes(), "sig_color": self.sig_color, "sig_mol": self.sig_mol,
                "uid": self.ids.astype(np.float64), "parent": self.parent.astype(np.float64),
                "young": (w.age < self.cfg.arena_maturity) & w.alive}

    def _inherit(self, slots, mothers, fathers) -> None:
        """Il corpo dei figli: ogni gene da uno dei due genitori, poi mutato
        con la probabilità del genitore (`weight_prob`); la taglia di
        `arena_size_std`, colori e molecole di 0,05."""
        cfg, rng = self.cfg, self.rng
        n = len(slots)
        pick = np.where(rng.random((n, 3)) < 0.5, mothers[:, None], fathers[:, None])
        genes = np.stack([self.size_gene, self.sig_color, self.sig_mol], axis=1)
        child = genes[pick, np.arange(3)]
        prob = self.ctrl.brains.genome.rate("weight_prob")[mothers]
        hit = rng.random((n, 3)) < prob[:, None]
        child += hit * rng.standard_normal((n, 3)) * np.array([cfg.arena_size_std, 0.05, 0.05])
        lo, hi = np.log(cfg.arena_size_min), np.log(cfg.arena_size_max)
        self.size_gene[slots] = np.clip(child[:, 0], lo, hi)
        self.sig_color[slots] = np.clip(child[:, 1], 0.0, 1.0)
        self.sig_mol[slots] = np.clip(child[:, 2], 0.0, 1.0)

    # --- Il tempo -------------------------------------------------------------

    @property
    def generation(self) -> int:
        """I minuti di mondo trascorsi (al posto delle generazioni)."""
        return int(self.world.t // 60.0)

    @property
    def alive(self) -> np.ndarray:
        return np.flatnonzero(self.world.alive)

    def step(self) -> None:
        """Un passo di mondo (1/60 s)."""
        self.advance(1)

    def advance(self, k: int) -> None:
        """k passi di mondo, con nascite e misure quando toccano. Sulla GPU i
        passi tra un secondo e l'altro girano tutti insieme."""
        if self.gpu is None:
            for _ in range(k):
                self._cpu_step()
            return
        dt = self.cfg.dt
        w = self.world
        while k > 0:
            n = min(k, max(1, int(round((self.next_tick - w.t) / dt))))
            self.gpu.advance(n)
            k -= n
            if w.t >= self.next_tick - 1e-6:
                self._gpu_tick()

    def run(self, seconds: float) -> None:
        """Avanza di `seconds` secondi di mondo."""
        self.advance(int(round(seconds / self.cfg.dt)))

    def _cpu_step(self) -> None:
        w = self.world
        w.step(self.ctrl.act(w), self.ctrl.upkeep)
        died = np.flatnonzero(self._was_alive & ~w.alive)
        if len(died):
            self._remember(died)
        if w.t >= self.next_tick:
            self.next_tick += self.cfg.arena_tick
            self._tick()
        self._was_alive = w.alive.copy()
        if w.t >= self.next_stats:
            self.next_stats += self.cfg.arena_stats_every
            self.measure()

    def _tick(self) -> None:
        self._births()
        self._colonists()
        self._poll_exam()

    def _gpu_tick(self) -> None:
        """Una volta al secondo, sul processore: vecchiaia, cibo nuovo e
        macchie, nascite, colonizzatori; poi il mondo torna sulla GPU."""
        cfg, w, rng = self.cfg, self.world, self.world.rng
        tick = cfg.arena_tick
        self.next_tick += tick
        self._food_sum += int(w.task.food_on.sum())  # per la media del minuto (prima dell'ondata)
        self._food_n += 1
        if cfg.arena_water:
            self._watch_drinks()
        w.age += tick * w.alive
        hazard = cfg.arena_aging_h0 * np.exp(w.age / cfg.arena_aging_tau)
        old = w.alive & (rng.random(w.n) < 1.0 - np.exp(-hazard * tick))
        w.deaths["vecchiaia"] += int(old.sum())
        w.alive &= ~old
        w.energy[~w.alive] = 0.0
        task = w.task
        task.patches.step(tick, rng)
        self._wear(tick)
        task._owed += task.rate() * tick
        every = cfg.arena_food_every
        if every <= 0 or int(w.t // every) > int((w.t - tick) // every):  # a ondate: solo allo scoccare
            new = int(task._owed)
            task._owed -= new
            task.grow(w, new)
        self._food_seen = task.food_on.copy()
        if cfg.arena_water:
            w.pools.step(tick, self.pool_rng, task.patches)
        died = np.flatnonzero(self._was_alive & ~w.alive)
        if len(died):
            self._remember(died)
        self._tick()
        self._was_alive = w.alive.copy()
        if w.t >= self.next_stats - 1e-6:
            self.next_stats += cfg.arena_stats_every
            self.measure()
        self.gpu.push_world(self.bodies())

    def _wear(self, tick: float) -> None:
        """Le macchie: si consumano dove si è mangiato (manopola
        arena_depletion), si riprendono in arena_patch_recovery secondi, e ogni
        arena_reversal_every secondi ognuna può invertire il colore buono
        (con probabilità arena_reversal / 2)."""
        cfg, task, rng = self.cfg, self.world.task, self.world.rng
        patches = task.patches
        eaten = self._food_seen & ~task.food_on
        if eaten.any() and cfg.arena_depletion > 0:
            near = patches.nearest(task.food[eaten])
            hits = np.bincount(near[near >= 0], minlength=len(patches.health))
            patches.health -= cfg.arena_patch_wear * cfg.arena_depletion * hits
        patches.health = np.clip(patches.health + tick / cfg.arena_patch_recovery, 0.05, 1.0)
        if self.world.t >= self.next_reversal:
            self.next_reversal += cfg.arena_reversal_every
            patches.good = (rng.random(len(patches.good)) < 0.5 * cfg.arena_reversal).astype(int)

    # --- Nascite --------------------------------------------------------------

    def _identities(self, i: int) -> set:
        g = self.ctrl.brains.genome
        return ({("r", int(x)) for x in g.r_id[i][g.r_active[i]]}
                | {("h", int(x)) for x in g.h_id[i][g.active[i]]})

    def compatible(self, i: int, j: int) -> bool:
        """Due esseri si incrociano se condividono almeno `arena_mate_share`
        delle identità dei loro geni (recettori e neuroni interni accesi)."""
        a, b = self._identities(i), self._identities(j)
        union = a | b
        return not union or len(a & b) / len(union) >= self.cfg.arena_mate_share

    def _partner(self, m: int, adults: np.ndarray) -> int:
        """Il compatibile più vicino entro `arena_mate_range`, o m stessa."""
        w, cfg = self.world, self.cfg
        d = np.hypot(*(((w.pos[adults] - w.pos[m] + [w.W / 2, w.H / 2]) % [w.W, w.H] - [w.W / 2, w.H / 2]).T))
        near = (d <= cfg.arena_mate_range) & (adults != m)
        for j in adults[near][np.argsort(d[near])]:
            if self.compatible(m, int(j)):
                return int(j)
        return m

    def _births(self) -> None:
        w, cfg, rng = self.world, self.cfg, self.rng
        ready = (w.alive & (w.energy >= cfg.arena_birth_energy) & (w.age >= cfg.arena_maturity)
                 & (w.t - self.last_birth >= cfg.arena_birth_interval))
        if cfg.arena_water:  # serve anche l'acqua da dare al figlio, e restarne un po'
            ready &= w.water >= cfg.arena_birth_water + cfg.arena_birth_water_spare * self._full_water(np.arange(w.n))
        mothers = np.flatnonzero(ready)
        free = np.flatnonzero(~w.alive)
        if not len(mothers) or not len(free):
            return
        if len(mothers) > len(free):  # posti finiti: nasce solo chi trova posto
            mothers = np.sort(rng.choice(mothers, len(free), replace=False))
        brains = self.ctrl.brains
        g = brains.genome
        fathers = mothers.copy()
        adults = np.flatnonzero(w.alive & (w.age >= cfg.arena_maturity))
        sex = g.rate("sex")
        for k, m in enumerate(mothers):
            if rng.random() < sex[m]:
                fathers[k] = self._partner(int(m), adults)
        children = crossover(g, mothers, fathers, rng, cfg.receptor_dup_split)
        self._mutate(children)
        children.unborn()
        if children.n_markers:
            develop(children, cfg, rng, brains.GAIN)
        slots = free[: len(mothers)]
        self._place(slots, children)
        self._inherit(slots, mothers, fathers)
        self.parent[slots] = self.ids[mothers]
        angle = rng.uniform(-np.pi, np.pi, len(slots))
        pos = w.pos[mothers] + cfg.arena_birth_distance * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        w.spawn(slots, pos, cfg.arena_child_energy)
        w.energy[mothers] -= cfg.arena_birth_cost
        if cfg.arena_water:
            w.water[mothers] -= cfg.arena_birth_water
            w.water[slots] = self._full_water(slots)  # il piccolo nasce idratato
        self._born(slots)
        self.last_birth[mothers] = w.t
        self.last_birth[slots] = -np.inf
        self.ids[slots] = np.arange(self.next_id, self.next_id + len(slots))
        self.next_id += len(slots)
        self.ctrl.obs[slots] = 0.0
        self.ctrl.adapt[slots] = np.nan
        self.totals["nati"] += len(slots)
        self.totals["con partner"] += int((fathers != mothers).sum())

    def _remember(self, died: np.ndarray) -> None:
        """Chi muore entra nell'archivio dei migliori se ha mangiato più dei
        peggiori che ci sono."""
        eaten = self.world.task.eaten
        g = self.ctrl.brains.genome
        for i in died:
            if eaten[i] > 0:
                body = np.array([self.size_gene[i], self.sig_color[i], self.sig_mol[i]])
                self.archive.append((int(eaten[i]), g.take([i]), body))
        self.archive.sort(key=lambda x: -x[0])
        del self.archive[self.cfg.arena_archive:]

    def _colonists(self) -> None:
        """Isola e continente: sotto `arena_min_alive` vivi arrivano esseri
        nuovi, sparsi nelle macchie. Sono copie mutate dei migliori
        dell'archivio, scelti con probabilità proporzionale al cibo mangiato,
        o reti nuove (primordiali) se l'archivio è vuoto."""
        w, cfg, rng = self.world, self.cfg, self.rng
        missing = cfg.arena_min_alive - int(w.alive.sum())
        free = np.flatnonzero(~w.alive)
        n = min(missing, len(free))
        if n <= 0:
            return
        slots = free[:n]
        brains = self.ctrl.brains
        if self.archive:
            weight = np.array([e for e, _, _ in self.archive], dtype=float)
            pick = rng.choice(len(self.archive), n, p=weight / weight.sum())
            pool = Genome.concat([g for _, g, _ in self.archive])
            body = np.array([b for _, _, b in self.archive])[pick]
            kids = crossover(pool, pick, pick, rng, cfg.receptor_dup_split)
            self._mutate(kids)
            kids.unborn()
            if kids.n_markers:
                develop(kids, cfg, rng, brains.GAIN)
        else:
            fresh = type(brains)(cfg, n, brains.n_out, int(rng.integers(2**31)))
            kids = fresh.genome
            self._primordial(kids)
            body = np.stack([rng.normal(0.0, cfg.arena_size_std, n), rng.random(n), rng.random(n)], axis=1)
        self._place(slots, kids)
        self.size_gene[slots], self.sig_color[slots], self.sig_mol[slots] = body.T
        self.parent[slots] = -1
        w.spawn(slots, w.task.patches.sample(n, w.t, rng), cfg.energy_init)
        w.water[slots] = self._full_water(slots)
        self._born(slots)
        self.last_birth[slots] = -np.inf
        self.ids[slots] = np.arange(self.next_id, self.next_id + n)
        self.next_id += n
        self.ctrl.obs[slots] = 0.0
        self.ctrl.adapt[slots] = np.nan
        self.totals["colonizzatori"] += n

    def _place(self, slots: np.ndarray, children: Genome) -> None:
        """I genomi `children` nascono nei posti `slots` (con il grembo se il
        genoma è "sviluppo"). Sul processore il cervello si ricompila tenendo
        gli altri come sono; sulla GPU i neonati arrivano riga per riga."""
        if self.gpu is None:
            self.ctrl.brains.replace(slots, children)
            if children.n_markers:
                self._womb(slots)
            return
        from ..gpu.arena import pack_brain_state, pack_params

        brains = self.ctrl.brains
        cls = type(brains)
        tmp = cls.__new__(cls)  # i soli neonati, compilati a parte
        tmp.cfg, tmp.n_out = brains.cfg, brains.n_out
        for k in ("substeps", "gain", "sensors_excitatory_only"):
            if hasattr(brains, k):
                setattr(tmp, k, getattr(brains, k))
        tmp.set_genome(children)
        if children.n_markers:
            self._womb(np.arange(len(children)), tmp)
        born = tmp.genome
        g = brains.genome
        for f in Genome.FIELDS:
            getattr(g, f)[slots] = getattr(born, f)
        self._stale = True
        n = len(slots)
        state = pack_brain_state(tmp, np.zeros((n, tmp.n_in)), np.full((n, tmp.n_in), np.nan))
        self.gpu.set_rows(slots, pack_params(tmp), state)

    def _womb(self, rows: np.ndarray, brains=None) -> None:
        """I neonati `rows` passano dal grembo (net/womb.py); il loro cervello
        rifinito diventa il fenotipo alla nascita. Gli altri non cambiano."""
        if not len(rows):
            return
        b, cfg = brains or self.ctrl.brains, self.cfg
        g = b.genome
        obs = womb.waves(cfg, g.r_y[rows], g.r_active[rows], self.rng)
        tau, eta = cfg.homeo_tau, b.eta
        b.eta = np.zeros_like(eta)  # niente ricompensa nel grembo
        b.avg[rows], b.clock[rows] = 0.0, 0.0
        try:
            for k in range(len(obs)):
                cfg.homeo_tau = min(tau, (k + 1) * cfg.dt)
                b.step_some(obs[k], rows)
        finally:
            cfg.homeo_tau, b.eta = tau, eta
        born = g.take(rows)
        born.weights, born.born_avg, born.born_bias = b.w_eff[rows].copy(), b.avg[rows].copy(), b.bias_now[rows].copy()
        b.replace(rows, born)

    # --- Misure ---------------------------------------------------------------

    def species(self, sample: int = 200) -> int:
        """Gruppi di compatibili (quasi specie), su un campione di vivi: ognuno
        entra nel primo gruppo con il cui fondatore è compatibile."""
        alive = self.alive
        if len(alive) > sample:
            alive = self.rng.choice(alive, sample, replace=False)
        founders: list[int] = []
        for i in alive:
            if not any(self.compatible(int(i), f) for f in founders):
                founders.append(int(i))
        return len(founders)

    def measure(self) -> dict:
        """Una riga di storia (e il campione, e l'esame se è il momento)."""
        from .evolution import receptor_stats

        w, b, cfg = self.world, self.ctrl.brains, self.cfg
        self._fresh()
        alive = self.alive
        task = w.task
        now = {"nati": self.totals["nati"], "con partner": self.totals["con partner"],
               "colonizzatori": self.totals["colonizzatori"],
               "fame": w.deaths["fame"], "vecchiaia": w.deaths["vecchiaia"], "mangiati": task.eaten_total,
               "predati": w.deaths.get("predati", 0), "cattivi": task.bad_total, "cure": task.fed_total,
               "sete": w.deaths.get("sete", 0), "bevuta": w.drunk_total}
        delta = {k: now[k] - self._last.get(k, 0) for k in now}
        self._last = now
        food = round(self._food_sum / self._food_n) if self._food_n else int(task.food_on.sum())
        self._food_sum, self._food_n = 0, 0
        row = {"time": round(w.t, 3), "minute": self.generation, "alive": len(alive),
               "births": delta["nati"], "mates": delta["con partner"], "colonists": delta["colonizzatori"],
               "starved": delta["fame"],
               "old_age": delta["vecchiaia"], "eaten": delta["mangiati"],
               "food": food,
               "killed": delta.get("predati", 0), "bad_eaten": delta.get("cattivi", 0),
               "fed": round(float(delta.get("cure", 0.0)), 1),
               "dials": {k: round(getattr(cfg, k), 3) for k in DIALS},
               "patch_health": round(float(task.patches.health.mean()), 3),
               "thirst": delta.get("sete", 0), "drunk": round(float(delta.get("bevuta", 0.0)), 1),
               "pools": int(w.pools.on.sum()), **(self.drink_stats() if cfg.arena_water else {}),
               "pool_water": round(float(w.pools.volume.sum() / max(cfg.arena_pool_volume, 1.0)), 3)}
        if len(alive):
            g = b.genome.take(alive)
            row.update({
                "age_mean": float(w.age[alive].mean()), "age_max": float(w.age[alive].max()),
                "energy_mean": float(w.energy[alive].mean()),
                "water_mean": float((w.water[alive] / self._full_water(alive)).mean()) if cfg.arena_water else 1.0,
                "neurons": float(b.active[alive].sum(axis=1).mean()),
                "links": float(b.links[alive].sum(axis=(1, 2)).mean()),
                **receptor_stats(g, b.links[alive]),
                "upkeep": float(b.upkeep[alive].mean()),
                **population_stats(g),
                "mutation": {k: float(np.exp(np.median(g.meta[:, i]))) for i, k in enumerate(META)},
                "species": self.species(),
                "temp_mean": float(temperature(cfg, w.pos[alive], w.t).mean()) if cfg.arena_temperature > 0 else 0.5,
                "size_mean": float(self.sizes()[alive].mean()), "size_sd": float(self.sizes()[alive].std()),
                "bite_mean": float(getattr(w, "bite", np.zeros(w.n))[alive].mean()),
                "glow_mean": float(getattr(w, "glow", np.zeros(w.n))[alive].mean()),
                "smell_mean": float(getattr(w, "smell", np.zeros(w.n))[alive].mean()),
            })
            oldest = alive[np.argmax(w.age[alive])]
            self.champions.append((self.generation, int(w.age[oldest]), b.genome.take([oldest]),
                                   int(self.ids[oldest])))
        self.history.append(row)
        self._world_ladder()
        every = self.cfg.exam_every
        if every > 0 and self.generation > 0 and self.generation % every == 0 and self.exam is None:
            self.start_exam()
        return row

    def _world_ladder(self) -> None:
        """La scala del mondo (sezione 15): se la popolazione si regge da sola
        (nessun colonizzatore) da arena_dial_calm minuti, la prima manopola non
        ancora al massimo sale di arena_dial_step; se ha avuto bisogno di
        colonizzatori in ognuno degli ultimi arena_dial_calm minuti, l'ultima
        manopola alzata scende. Un passo al minuto al massimo; ogni cambio va
        nel registro."""
        cfg = self.cfg
        if not cfg.arena_world_ladder or self.gpu is None:
            return
        recent = self.history[-cfg.arena_dial_calm:]
        if len(recent) < cfg.arena_dial_calm or (self._dial_at is not None
                                                 and self.generation - self._dial_at < cfg.arena_dial_calm):
            return
        calm = all(r["colonists"] == 0 and r["alive"] >= cfg.arena_min_alive for r in recent)
        troubled = all(r["colonists"] > 0 for r in recent)
        k, step = None, 0.0
        if calm:
            k = next((d for d in DIALS if getattr(cfg, d) < 1.0), None)
            step = cfg.arena_dial_step
        elif troubled:
            k = next((d for d in reversed(DIALS) if getattr(cfg, d) > 0.0), None)
            step = -cfg.arena_dial_step
        if k is None:
            return
        old = getattr(cfg, k)
        setattr(cfg, k, round(float(np.clip(old + step, 0.0, 1.0)), 4))
        self._dial_at = self.generation
        self.log.append({"time": round(self.world.t, 3), "minute": self.generation, "event": "mondo",
                         "dial": k, "name": DIAL_NAMES[DIALS.index(k)], "from": old, "to": getattr(cfg, k),
                         "why": "si regge da sola" if step > 0 else "ha bisogno di colonizzatori"})
        self.gpu.set_phys()

    def _fresh(self) -> None:
        """Con la GPU: i cervelli compilati sul processore (per misure e
        salvataggi) dopo le nascite."""
        if self._stale:
            self.ctrl.brains.compile()
            self._stale = False

    def start_exam(self, k: int = 20, lesion: str | None = None) -> bool:
        """L'esame fisso (i livelli del mondo del cibo) su `k` adulti presi a
        caso. Gira in sottofondo; il risultato va in `exams`."""
        w = self.world
        adults = np.flatnonzero(w.alive & (w.age >= ADULT))
        if not len(adults) or self.exam is not None:
            return False
        idx = np.sort(self.rng.choice(adults, min(k, len(adults)), replace=False))
        self.exam = Exam(self.cfg, self.ctrl.brains.genome.take(idx), self.generation, ids=self.ids[idx],
                         lesion=lesion)
        self._probe = (self.ctrl.brains.genome.take(idx), self.ids[idx]) if lesion is None else None
        return True

    def _poll_exam(self) -> None:
        if self.exam is None:
            return
        self.exam.run(0.02)
        if self.exam.finished:
            r = self.exam.result()
            r["time"] = round(self.world.t, 3)
            self.exams.append(r)
            self.exam.close()
            self.exam = None
            if self._probe is not None:  # la prova dell'apprendimento: gli stessi, senza plasticità
                genomes, ids = self._probe
                self._probe = None
                self.exam = Exam(self.cfg, genomes, r["generation"], ids=ids, lesion="plasticità")

    # --- Salvataggi -----------------------------------------------------------

    LIVE = ("w_eff", "early", "elig", "critic", "bias_now", "avg", "clock", "state", "charge", "spikes")

    def save(self, path) -> None:
        """Salva tutto ciò che serve per riprendere: genomi di tutti i posti,
        stato dei cervelli (ciò che si è imparato), mondo, cibo, macchie,
        storia. Scrittura sicura come negli altri salvataggi."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        w, b = self.world, self.ctrl.brains
        extra = {}
        if self.gpu is not None:  # lo stato dei cervelli vive sulla GPU
            self._fresh()
            live = self.gpu.brain_state()
            for k in self.LIVE:
                name = "w" if k == "w_eff" else k
                if name in live and hasattr(b, k):
                    setattr(b, k, live[name])
            self.ctrl.obs, self.ctrl.adapt = live["obs"], live["m"]
            b.learned, b._subset = True, None
            extra["gpu_key"] = np.asarray(self.gpu.s["key"])
        g, task = b.genome, w.task
        data = {f: getattr(g, f) for f in Genome.FIELDS}
        data.update({"live_" + k: getattr(b, k) for k in self.LIVE if hasattr(b, k)})
        data.update(
            kind="arena", format=FORMAT, generation=self.generation, net_type=self.cfg.net_type, seed=self.seed,
            population=self.cfg.arena_cap, world_seed=w.seed, t=w.t,
            **{"w_" + k: getattr(w, k) for k in WORLD}, alive=w.alive, eaten=task.eaten,
            food=task.food, food_on=task.food_on, food_owed=task._owed, eaten_total=task.eaten_total,
            patch_centre=task.patches.centre, patch_heading=task.patches.heading, patch_phase=task.patches.phase,
            ids=self.ids, next_id=self.next_id, last_birth=self.last_birth, next_tick=self.next_tick,
            next_stats=self.next_stats, next_reversal=self.next_reversal,
            size_gene=self.size_gene, sig_color=self.sig_color, sig_mol=self.sig_mol, parent=self.parent,
            food_color=task.color, food_bad=task.bad, food_kind=task.kind, food_radius=task.radius,
            food_seen=self._food_seen, patch_health=task.patches.health, patch_good=task.patches.good,
            bad_total=task.bad_total, fed_total=task.fed_total,
            water=w.water, drunk_total=w.drunk_total, pool_centre=w.pools.centre, pool_volume=w.pools.volume,
            deaths=json.dumps(w.deaths), totals=json.dumps(self.totals), last=json.dumps(self._last),
            rng=json.dumps({"world": w.rng.bit_generator.state, "life": self.rng.bit_generator.state,
                            "noise": self.ctrl.noise_rng.bit_generator.state,
                            "pools": self.pool_rng.bit_generator.state}),
            obs=self.ctrl.obs, adapt=self.ctrl.adapt,
            history=json.dumps(self.history), log=json.dumps(self.log), exams=json.dumps(self.exams),
            config=json.dumps(dataclasses.asdict(self.cfg)), **extra)
        if self.archive:
            scores, genomes, bodies = zip(*self.archive)
            ar = Genome.concat(list(genomes))
            data.update(ar_eaten=np.array(scores), ar_body=np.array(bodies),
                        **{"ar_" + f: getattr(ar, f) for f in Genome.FIELDS})
        if self.champions:
            gens, ages, genomes, ids = zip(*self.champions)
            ch = Genome.concat(list(genomes))
            data.update(ch_generation=np.array(gens), ch_age=np.array(ages), ch_id=np.array(ids),
                        **{"ch_" + f: getattr(ch, f) for f in Genome.FIELDS})
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, cfg: Config, path) -> "Ecology":
        """Riprende un'arena salvata con le regole più recenti: dal file
        vengono i genomi, lo stato e le dimensioni (RUN_STATE e ARENA_STATE);
        le altre costanti sono quelle del codice. Le regole cambiate finiscono
        nel registro."""
        with np.load(path) as d:
            if "kind" not in d or str(d["kind"]) != "arena":
                raise ValueError(f"{path}: non è un salvataggio dell'arena continua")
            saved = json.loads(str(d["config"]))
            for k in RUN_STATE + ARENA_STATE + DIALS:
                if k in saved:
                    setattr(cfg, k, saved[k])
            fields = {f.name for f in dataclasses.fields(cfg)}
            rules = {k: [saved.get(k), getattr(cfg, k)] for k in sorted(fields - set(RUN_STATE) - set(ARENA_STATE)
                                                                         - set(DIALS) - {"engine", "exam_every"})
                     if saved.get(k) != getattr(cfg, k)}
            engine = cfg.engine
            # Il motore NumPy di passaggio spegnerebbe manopole, clima e acqua.
            keep = {k: getattr(cfg, k) for k in DIALS + ONLY_GPU}
            cfg.engine = "numpy"  # la GPU parte dopo, dallo stato ripreso
            e = cls(cfg, int(d["seed"]), initial=0)
            cfg.engine = engine
            for k, v in keep.items():
                setattr(cfg, k, v)
            w, b, task = e.world, e.ctrl.brains, e.world.task
            b.set_genome(fit_traits(Genome(*(d[f] for f in Genome.FIELDS)), b.N_TRAITS))
            for k in cls.LIVE:
                if "live_" + k in d and hasattr(b, k) and getattr(b, k).shape == d["live_" + k].shape:
                    setattr(b, k, d["live_" + k].copy())
            b.learned, b._subset = True, None
            w.t = float(d["t"])
            for k in WORLD:
                setattr(w, k, d["w_" + k].copy())
            w.alive = d["alive"].copy()
            task.eaten, task.food, task.food_on = d["eaten"].copy(), d["food"].copy(), d["food_on"].copy()
            task._owed, task.eaten_total = float(d["food_owed"]), int(d["eaten_total"])
            task.patches.centre, task.patches.heading = d["patch_centre"].copy(), d["patch_heading"].copy()
            task.patches.phase = d["patch_phase"].copy()
            w.deaths = json.loads(str(d["deaths"]))
            w.grids_dirty = True
            e.ids, e.next_id, e.last_birth = d["ids"].copy(), int(d["next_id"]), d["last_birth"].copy()
            e.next_tick, e.next_stats = float(d["next_tick"]), float(d["next_stats"])
            if "size_gene" in d:
                e.next_reversal = float(d["next_reversal"])
                e.size_gene, e.sig_color, e.sig_mol = d["size_gene"].copy(), d["sig_color"].copy(), d["sig_mol"].copy()
                e.parent = d["parent"].copy()
                task.color, task.bad, task.kind = d["food_color"].copy(), d["food_bad"].copy(), d["food_kind"].copy()
                task.radius, e._food_seen = d["food_radius"].copy(), d["food_seen"].copy()
                task.patches.health, task.patches.good = d["patch_health"].copy(), d["patch_good"].copy()
                task.bad_total, task.fed_total = int(d["bad_total"]), float(d["fed_total"])
            if "water" in d:  # dal salvataggio; prima dell'acqua: tutti pieni, pozze nuove
                w.water, w.drunk_total = d["water"].copy(), float(d["drunk_total"])
            else:
                w.water = np.where(w.alive, e._full_water(np.arange(w.n)), 0.0)
            e._drinks_reset()
            if cfg.arena_water and "pool_volume" in d and len(d["pool_volume"]) == len(w.pools.volume):
                w.pools.centre, w.pools.volume = d["pool_centre"].copy(), d["pool_volume"].copy()
            elif cfg.arena_water and cfg.engine == "gpu":
                for _ in range(min(cfg.arena_pools, len(w.pools.volume))):
                    w.pools.add(e.pool_rng, task.patches)
            e.totals, e._last = json.loads(str(d["totals"])), json.loads(str(d["last"]))
            states = json.loads(str(d["rng"]))
            w.rng.bit_generator.state = states["world"]
            e.rng.bit_generator.state = states["life"]
            e.ctrl.noise_rng.bit_generator.state = states["noise"]
            if "pools" in states:
                e.pool_rng.bit_generator.state = states["pools"]
            e.ctrl.obs, e.ctrl.adapt = d["obs"].copy(), d["adapt"].copy()
            e.history, e.log = json.loads(str(d["history"])), json.loads(str(d["log"]))
            e.exams = json.loads(str(d["exams"]))
            if "ar_weights" in d:
                ar = fit_traits(Genome(*(d["ar_" + f] for f in Genome.FIELDS)), b.N_TRAITS)
                body = d["ar_body"] if "ar_body" in d else np.zeros((len(d["ar_eaten"]), 3))
                e.archive = [(int(x), ar.take([k]), body[k].copy()) for k, x in enumerate(d["ar_eaten"])]
            e._was_alive = w.alive.copy()
            if "ch_weights" in d:
                ch = fit_traits(Genome(*(d["ch_" + f] for f in Genome.FIELDS)), b.N_TRAITS)
                e.champions = [(int(gen), int(a), ch.take([k]), int(i)) for k, (gen, a, i)
                               in enumerate(zip(d["ch_generation"], d["ch_age"], d["ch_id"]))]
            key = d["gpu_key"].copy() if "gpu_key" in d else None
        if cfg.engine == "gpu":
            e._start_gpu(key)
        resumed = {"time": round(e.world.t, 3), "event": "ripresa", "code": code_version()}
        if rules:
            resumed["changes"] = rules
        e.log.append(resumed)
        return e
