import numpy as np


class ReachTarget:
    """Primo compito: toccare il bersaglio. Uno solo, condiviso e non consumabile:
    ogni agente affronta la stessa prova senza competere con gli altri.

    Un compito è un oggetto con `name`, `reset(world)` e
    `completed(world) -> (N,) bool`; per renderlo più arduo se ne scrive un altro
    con la stessa interfaccia e lo si passa a `World`. Facoltativi:
    `spawn_ok(world, pos) -> (N,) bool` (dove gli agenti possono nascere) e
    `step(world)` (il compito cambia durante la vita), `spawned(world)` (dopo
    la nascita degli agenti).
    """

    name = "raggiungi il bersaglio"

    def reset(self, world) -> None:
        cfg = world.cfg
        r = cfg.target_radius
        self.radius = r
        self.target = world.rng.uniform([r, r], [cfg.arena_w - r, cfg.arena_h - r])

    def completed(self, world) -> np.ndarray:
        d = np.linalg.norm(world.pos - self.target, axis=1)
        return d <= self.radius + world.cfg.agent_radius


class FarTarget(ReachTarget):
    """Il bersaglio nasce lontano da tutti: almeno `min_dist` da ogni agente.
    Misura l'esplorazione."""

    name = "bersaglio lontano"

    def __init__(self, min_dist: float = 90.0):
        self.min_dist = min_dist

    def spawn_ok(self, world, pos: np.ndarray) -> np.ndarray:
        return np.linalg.norm(pos - self.target, axis=1) >= self.min_dist


class MovingTarget(ReachTarget):
    """Il bersaglio si muove a velocità costante e rimbalza sulle pareti:
    bisogna inseguirlo, non solo andare dove lo si è visto."""

    name = "bersaglio in movimento"

    def __init__(self, speed: float = 5.0):
        self.speed = speed

    def reset(self, world) -> None:
        super().reset(world)
        a = world.rng.uniform(-np.pi, np.pi)
        self.vel = self.speed * np.array([np.cos(a), np.sin(a)])

    def step(self, world) -> None:
        cfg, r = world.cfg, self.radius
        self.target = self.target + self.vel * cfg.dt
        hi = np.array([cfg.arena_w - r, cfg.arena_h - r])
        out = (self.target < r) | (self.target > hi)
        self.vel = np.where(out, -self.vel, self.vel)
        self.target = np.clip(self.target, r, hi)


FORMED = 1e6  # s: una nube in equilibrio da sempre (o una traccia svanita da sempre)


def lerp(a, b, t):
    return a + (b - a) * t


class Foraging:
    """Nutrirsi (docs/SPEC_MONDO.md): si riproduce chi mangia almeno N cibi.

    Ogni agente ha la sua copia del mondo: cibo e predatori partono uguali per
    tutti, poi ognuno li cambia solo nella sua copia (mangiando, facendo
    fuggire le prede, attirando i predatori). Le manopole della
    configurazione (`forage_*`) decidono quanti cibi, quanto grandi, quanti ne
    servono, se fuggono, quanti predatori.

    Il cibo mangiato ricompare nella posizione successiva di una coda estratta
    a sorte all'inizio dell'arena (comune a tutti, ognuno la scorre per conto
    suo): niente numeri casuali durante la vita, così i due motori (NumPy e
    GPU) fanno esattamente la stessa cosa.

    Odori: cibo e predatori rilasciano una molecola loro (`odors`). La nube di
    un cibo appena comparso si forma in τ secondi e quella di un cibo mangiato
    resta nel punto dove era e svanisce nello stesso tempo; τ cresce con la
    manopola `forage_scent` (0 = nube sempre in equilibrio).
    """

    name = "nutrirsi"
    QUEUE = 4096  # posizioni di ricomparsa del cibo

    def reset(self, world) -> None:
        cfg, rng, n = world.cfg, world.rng, world.n
        self.n_food = int(round(lerp(cfg.food_count_easy, cfg.food_count_hard, cfg.forage_abundance)))
        self.food_r = lerp(cfg.food_radius_easy, cfg.food_radius_hard, cfg.forage_abundance)
        self.need = int(round(lerp(cfg.food_need_easy, cfg.food_need_hard, cfg.forage_need)))
        self.prey_speed = cfg.prey_speed_max * cfg.forage_prey
        q = cfg.forage_predators
        self.n_pred = int(np.ceil(cfg.predator_count_max * q)) if q > 0 else 0
        self.pred_speed = cfg.predator_speed_max * q
        K, P = cfg.food_count_easy, cfg.predator_count_max  # posti fissi: forma uguale a ogni livello
        lo = np.array([self.food_r, self.food_r])
        hi = np.array([cfg.arena_w - self.food_r, cfg.arena_h - self.food_r])
        food = rng.uniform(lo, hi, (K, 2))
        self.food = np.broadcast_to(food, (n, K, 2)).copy()
        self.food_on = np.zeros((n, K), dtype=bool)
        self.food_on[:, : self.n_food] = True
        self.queue = rng.uniform(lo, hi, (self.QUEUE, 2))
        self.ptr = np.zeros(n, dtype=int)
        r = cfg.predator_radius
        pred = rng.uniform([r, r], [cfg.arena_w - r, cfg.arena_h - r], (P, 2))
        self.pred = np.broadcast_to(pred, (n, P, 2)).copy()
        self.pred_on = np.zeros((n, P), dtype=bool)
        self.pred_on[:, : self.n_pred] = True
        self.eaten = np.zeros(n, dtype=int)
        self.scent_tau = cfg.forage_scent * cfg.scent_time_max
        self.age = np.full((n, K), FORMED)  # da quanto c'è il cibo di ogni posto (s)
        self.ghost = self.food.copy()  # dove era il cibo mangiato l'ultima volta
        self.ghost_t = np.full((n, K), FORMED)  # da quanto è stato mangiato (s)

    def spawn_ok(self, world, pos: np.ndarray) -> np.ndarray:
        cfg = world.cfg
        on = self.food_on[0]
        d_food = np.linalg.norm(pos[:, None, :] - self.food[0][None, on], axis=2)
        ok = (d_food >= cfg.food_min_dist).all(axis=1)
        if self.n_pred:
            d_pred = np.linalg.norm(pos[:, None, :] - self.pred[0][None, self.pred_on[0]], axis=2)
            ok &= (d_pred >= cfg.predator_min_dist).all(axis=1)
        return ok

    def spawned(self, world) -> None:
        """Dopo la nascita: distanza di ognuno dal cibo più vicino (per
        misurare quanto pesa la fortuna)."""
        on = self.food_on[0]
        self.start_dist = np.linalg.norm(world.pos[:, None, :] - self.food[0][None, on], axis=2).min(axis=1)

    def discs(self, world):
        """Gli oggetti che si vedono, per agente: posizioni (N, O, 2), raggi
        (N, O) (0 = non c'è) e tipo di ciascuno (O,): 0 cibo, 1 predatore."""
        cfg = world.cfg
        pos = np.concatenate([self.food, self.pred], axis=1)
        radius = np.concatenate([np.where(self.food_on, self.food_r, 0.0),
                                 np.where(self.pred_on, cfg.predator_radius, 0.0)], axis=1)
        kinds = np.r_[np.zeros(self.food.shape[1], dtype=int), np.ones(self.pred.shape[1], dtype=int)]
        return pos, radius, kinds

    def scent(self) -> tuple[np.ndarray, np.ndarray]:
        """Forza della nube di ogni cibo e della sua traccia (N, K), da 0 a 1."""
        if self.scent_tau <= 0.0:
            return np.ones_like(self.age), np.zeros_like(self.ghost_t)
        return 1.0 - np.exp(-self.age / self.scent_tau), np.exp(-self.ghost_t / self.scent_tau)

    def odors(self, world, points: np.ndarray, agent: np.ndarray) -> list[tuple[float, np.ndarray]]:
        """Le sostanze nei punti (M, 2), ciascuno del corpo dell'agente
        `agent` (M,): [(molecola, concentrazione (M,))]. La nube cala come
        exp(-d / odor_range), con d dal bordo dell'oggetto."""
        cfg = world.cfg
        lam = cfg.odor_range
        out = []
        if cfg.food_smell > 0.0:
            on, trace = self.scent()
            p = points[:, None, :]
            d = np.maximum(np.linalg.norm(p - self.food[agent], axis=2) - self.food_r, 0.0)
            dg = np.maximum(np.linalg.norm(p - self.ghost[agent], axis=2) - self.food_r, 0.0)
            c = (on[agent] * np.exp(-d / lam) + trace[agent] * np.exp(-dg / lam)) * self.food_on[agent]
            out.append((cfg.food_molecule, cfg.food_smell * c.sum(axis=1)))
        if cfg.predator_smell > 0.0 and self.n_pred:
            d = np.maximum(np.linalg.norm(points[:, None, :] - self.pred[agent], axis=2) - cfg.predator_radius, 0.0)
            c = np.exp(-d / lam) * self.pred_on[agent]
            out.append((cfg.predator_molecule, cfg.predator_smell * c.sum(axis=1)))
        return out

    def step(self, world) -> None:
        """Prede e predatori si muovono; chi tocca il cibo lo mangia; chi è
        toccato da un predatore perde energia. Solo per chi è in gara."""
        cfg, dt = world.cfg, world.cfg.dt
        self.age += dt
        self.ghost_t += dt
        active = world.alive & ~world.done & (world.energy > 0.0)  # chi è morto ora non mangia più
        me = world.pos[:, None, :]
        # Prede: fuggono dall'agente quando è vicino, rimbalzando sulle pareti.
        if self.prey_speed > 0:
            away = self.food - me
            d = np.linalg.norm(away, axis=2, keepdims=True)
            flee = (d < cfg.prey_flee_range) & self.food_on[..., None] & active[:, None, None]
            step = np.where(flee, away / np.maximum(d, 1e-9) * self.prey_speed * dt, 0.0)
            r = self.food_r
            self.food = np.clip(self.food + step, [r, r], [cfg.arena_w - r, cfg.arena_h - r])
        # Predatori: inseguono l'agente quando è abbastanza vicino.
        if self.n_pred:
            toward = me - self.pred
            d = np.linalg.norm(toward, axis=2, keepdims=True)
            chase = (d < cfg.predator_range) & self.pred_on[..., None] & active[:, None, None]
            step = np.where(chase, toward / np.maximum(d, 1e-9) * np.minimum(self.pred_speed * dt, d), 0.0)
            r = cfg.predator_radius
            self.pred = np.clip(self.pred + step, [r, r], [cfg.arena_w - r, cfg.arena_h - r])
            d = np.linalg.norm(world.pos[:, None, :] - self.pred, axis=2)
            bitten = ((d <= cfg.predator_radius + cfg.agent_radius) & self.pred_on).sum(axis=1)
            world.energy = np.maximum(world.energy - cfg.predator_damage * dt * bitten * active, 0.0)
            active &= world.energy > 0.0  # chi è appena morto per un morso non mangia
        # Mangiare: il cibo toccato sparisce e ricompare altrove (dalla coda).
        d = np.linalg.norm(world.pos[:, None, :] - self.food, axis=2)
        eat = (d <= self.food_r + cfg.agent_radius) & self.food_on & active[:, None]
        if eat.any():
            for i, k in np.argwhere(eat):
                self.ghost[i, k], self.ghost_t[i, k], self.age[i, k] = self.food[i, k], 0.0, 0.0
                a, b = self.queue[self.ptr[i] % self.QUEUE], self.queue[(self.ptr[i] + 1) % self.QUEUE]
                far = np.linalg.norm(a - world.pos[i]) >= cfg.food_min_dist
                self.food[i, k] = a if far else b
                self.ptr[i] += 2
            n_eat = eat.sum(axis=1)
            self.eaten += n_eat
            world.energy = np.minimum(world.energy + cfg.food_energy * n_eat, cfg.energy_init)

    def completed(self, world) -> np.ndarray:
        return self.eaten >= self.need


TASKS = {"bersaglio": ReachTarget, "cibo": Foraging}
