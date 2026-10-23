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


PUSH_PASSES = 2  # le rocce non si toccano (`rock_gap`): una passata basta, la seconda è per sicurezza


def push_out(pos, rocks, rad, body: float, vel=None, restitution: float = 0.0):
    """Fuori dalle rocce: ogni punto (o corpo di raggio `body`) che sta dentro
    una roccia viene spinto sul suo bordo; la velocità verso la roccia, se
    data, rimbalza come su una parete. pos (N, ..., 2) con le rocce
    dell'individuo: rocks (N, Kr, 2), rad (N, Kr) (0 = non c'è).
    Restituisce (pos, vel, a contatto con una roccia (N, ...))."""
    extra = pos.ndim - 2  # assi in più tra individuo e coordinate (oggetti)
    touch = np.zeros(pos.shape[:-1], dtype=bool)
    # Più passate: due rocce che si toccano possono rimandarsi un oggetto.
    for k in list(range(rocks.shape[1])) * PUSH_PASSES:
        c = rocks[:, k].reshape((len(rocks),) + (1,) * extra + (2,))
        r = rad[:, k].reshape((len(rocks),) + (1,) * extra)
        d = pos - c
        dist = np.linalg.norm(d, axis=-1)
        lim = r + body
        inside = (r > 0.0) & (dist < lim)
        unit = np.where(dist[..., None] > 1e-9, d / np.maximum(dist, 1e-9)[..., None], np.array([1.0, 0.0]))
        pos = np.where(inside[..., None], c + unit * lim[..., None], pos)
        if vel is not None:
            vn = (vel * unit).sum(-1)
            vel = np.where((inside & (vn < 0.0))[..., None], vel - ((1.0 + restitution) * vn)[..., None] * unit, vel)
        touch |= (r > 0.0) & (dist <= lim + 1e-6)
    return pos, vel, touch


def lerp(a, b, t):
    return a + (b - a) * t


class Foraging:
    """Nutrirsi (docs/SPEC_MONDO.md): si riproduce chi mangia almeno N cibi.

    Ogni agente ha la sua copia del mondo. Con `forage_own_arena` ogni copia
    è un'arena diversa (cibo, predatori, ricomparse); senza, partono uguali per
    tutti. Poi ognuno li cambia solo nella sua copia (mangiando, facendo
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
    keeps_living = True  # chi è maturo continua a vivere (se `mature_keep_living`)

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
        own = cfg.forage_own_arena  # ognuno la sua arena, o la stessa per tutti
        food = rng.uniform(lo, hi, (n if own else 1, K, 2))
        self.food = np.broadcast_to(food, (n, K, 2)).copy()
        self.food_on = np.zeros((n, K), dtype=bool)
        self.food_on[:, : self.n_food] = True
        # Frutti: in ogni vita uno dei due colori è buono (`good`, 0 rosso o 1
        # giallo, a caso per ognuno) e una quota dei posti (fino a metà, con
        # la manopola) ha frutti cattivi, dell'altro colore. Ogni posto tiene
        # il suo tipo anche quando il frutto ricompare.
        self.bad = np.zeros((n, K), dtype=bool)
        self.good = np.zeros(n, dtype=int)
        n_bad = int(round(self.n_food * 0.5 * cfg.forage_variety))
        if n_bad:  # estratti solo se servono: senza frutti cattivi il mondo resta quello di prima
            self.good = rng.integers(0, 2, n)
            for i in range(n):
                self.bad[i, rng.choice(self.n_food, n_bad, replace=False)] = True
        self.color = np.where(self.bad, 1 - self.good[:, None], self.good[:, None])  # (N, K): 0 rosso, 1 giallo
        self.bad_eaten = np.zeros(n, dtype=int)
        self.queue = rng.uniform(lo, hi, (self.QUEUE, 2))
        # Con l'arena propria ognuno scorre la coda da un punto suo: il cibo
        # ricompare in posti diversi per ciascuno (la coda resta comune, così
        # il motore GPU non cambia).
        self.ptr = 2 * rng.integers(0, self.QUEUE // 2, n) if own else np.zeros(n, dtype=int)
        r = cfg.predator_radius
        pred = rng.uniform([r, r], [cfg.arena_w - r, cfg.arena_h - r], (n if own else 1, P, 2))
        self.pred = np.broadcast_to(pred, (n, P, 2)).copy()
        self.pred_on = np.zeros((n, P), dtype=bool)
        self.pred_on[:, : self.n_pred] = True
        # Rocce: fisse per tutta l'arena, in posti diversi per ognuno (come il cibo).
        q = cfg.forage_obstacles
        self.n_rock = int(np.ceil(cfg.rock_count_max * q)) if q > 0 else 0
        Kr = cfg.rock_count_max
        rmax = cfg.rock_radius_max
        self.rock, self.rock_r = np.zeros((n, Kr, 2)), np.zeros((n, Kr))  # posti fissi: forma uguale a ogni livello
        if self.n_rock:  # estratte solo se ci sono: senza rocce il mondo resta quello di prima
            rocks, rad = self._place_rocks(cfg, rng, n if own else 1, Kr)
            self.rock[:] = rocks
            self.rock_r[:] = rad
            self.rock_r[:, self.n_rock :] = 0.0  # le rocce che non ci sono
        self.food = self._outside(self.food, self.food_r)
        self.pred = self._outside(self.pred, cfg.predator_radius)
        self.eaten = np.zeros(n, dtype=int)
        self.scent_tau = cfg.forage_scent * cfg.scent_time_max
        self.age = np.full((n, K), FORMED)  # da quanto c'è il cibo di ogni posto (s)
        self.ghost = self.food.copy()  # dove era il cibo mangiato l'ultima volta
        self.ghost_t = np.full((n, K), FORMED)  # da quanto è stato mangiato (s)

    @staticmethod
    def _place_rocks(cfg, rng, copies: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        """k rocce per copia di mondo, a caso, che non si sovrappongono e
        lasciano tra loro almeno `rock_gap` di passaggio (se non c'è posto,
        una roccia resta più piccola)."""
        rmax = cfg.rock_radius_max
        pos, rad = np.zeros((copies, k, 2)), np.zeros((copies, k))
        for j in range(k):  # una roccia alla volta, in tutte le copie insieme
            r = rng.uniform(cfg.rock_radius_min, rmax, copies)
            todo = np.ones(copies, dtype=bool)
            for _ in range(100):
                p = rng.uniform([rmax, rmax], [cfg.arena_w - rmax, cfg.arena_h - rmax], (copies, 2))
                gaps = np.linalg.norm(pos[:, :j] - p[:, None], axis=2) - rad[:, :j] - r[:, None]
                ok = todo & (gaps >= cfg.rock_gap).all(axis=1)
                pos[ok, j], rad[ok, j] = p[ok], r[ok]
                todo &= ~ok
                if not todo.any():
                    break
            # chi non ha trovato posto resta senza questa roccia (raggio 0)
        return pos, rad

    def _outside(self, pos: np.ndarray, body: float) -> np.ndarray:
        """Oggetti (N, K, 2) spinti fuori dalle rocce del loro individuo."""
        return push_out(pos, self.rock, self.rock_r, body)[0] if self.n_rock else pos

    def _nearest(self, pos: np.ndarray, obj: np.ndarray, on: np.ndarray) -> np.ndarray:
        """Distanza di ogni agente (riga di `pos`) dall'oggetto acceso più
        vicino nella sua copia di mondo."""
        return np.where(on, np.linalg.norm(pos[:, None, :] - obj, axis=2), np.inf).min(axis=1)

    def spawn_ok(self, world, pos: np.ndarray) -> np.ndarray:
        cfg = world.cfg
        ok = self._nearest(pos, self.food, self.food_on) >= cfg.food_min_dist
        if self.n_pred:
            ok &= self._nearest(pos, self.pred, self.pred_on) >= cfg.predator_min_dist
        if self.n_rock:  # non si nasce dentro una roccia
            edge = np.linalg.norm(pos[:, None, :] - self.rock, axis=2) - self.rock_r
            ok &= np.where(self.rock_r > 0, edge, np.inf).min(axis=1) >= cfg.agent_radius + 2.0
        return ok

    def spawned(self, world) -> None:
        """Dopo la nascita: distanza di ognuno dal cibo più vicino (per
        misurare quanto pesa la fortuna)."""
        self.start_dist = self._nearest(world.pos, self.food, self.food_on)

    def discs(self, world):
        """Gli oggetti che si vedono, per agente: posizioni (N, O, 2), raggi
        (N, O) (0 = non c'è) e tipo di ciascuno (O,): 0 cibo rosso, 1
        predatore, 2 roccia, 3 frutto giallo (ogni cibo compare due volte, una
        per colore, con raggio 0 nel colore che non ha)."""
        cfg = world.cfg
        K = self.food.shape[1]
        pos = np.concatenate([self.food, self.pred, self.rock, self.food], axis=1)
        radius = np.concatenate([np.where(self.food_on & (self.color == 0), self.food_r, 0.0),
                                 np.where(self.pred_on, cfg.predator_radius, 0.0), self.rock_r,
                                 np.where(self.food_on & (self.color == 1), self.food_r, 0.0)], axis=1)
        kinds = np.r_[np.zeros(K, dtype=int), np.ones(self.pred.shape[1], dtype=int),
                      np.full(self.rock.shape[1], 2, dtype=int), np.full(K, 3, dtype=int)]
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
            red = self.color[agent] == 0  # ogni frutto profuma della molecola del suo colore
            out.append((cfg.food_molecule, cfg.food_smell * (c * red).sum(axis=1)))
            if self.bad.any():
                out.append((cfg.fruit_molecule, cfg.food_smell * (c * ~red).sum(axis=1)))
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
        active = world.racing & (world.energy > 0.0)  # chi è morto ora non mangia più
        if self.n_rock:  # le rocce fermano l'agente, come una parete
            world.pos, world.vel, touch = push_out(world.pos, self.rock, self.rock_r, cfg.agent_radius,
                                                   world.vel, cfg.wall_restitution)
            world.bump |= touch
        me = world.pos[:, None, :]
        # Prede: fuggono dall'agente quando è vicino, rimbalzando sulle pareti.
        if self.prey_speed > 0:
            away = self.food - me
            d = np.linalg.norm(away, axis=2, keepdims=True)
            flee = (d < cfg.prey_flee_range) & self.food_on[..., None] & active[:, None, None]
            step = np.where(flee, away / np.maximum(d, 1e-9) * self.prey_speed * dt, 0.0)
            r = self.food_r
            self.food = self._outside(np.clip(self.food + step, [r, r], [cfg.arena_w - r, cfg.arena_h - r]), r)
        # Predatori: inseguono l'agente quando è abbastanza vicino.
        if self.n_pred:
            toward = me - self.pred
            d = np.linalg.norm(toward, axis=2, keepdims=True)
            chase = (d < cfg.predator_range) & self.pred_on[..., None] & active[:, None, None]
            step = np.where(chase, toward / np.maximum(d, 1e-9) * np.minimum(self.pred_speed * dt, d), 0.0)
            r = cfg.predator_radius
            self.pred = self._outside(np.clip(self.pred + step, [r, r], [cfg.arena_w - r, cfg.arena_h - r]), r)
            d = np.linalg.norm(world.pos[:, None, :] - self.pred, axis=2)
            bitten = ((d <= cfg.predator_radius + cfg.agent_radius) & self.pred_on).sum(axis=1)
            world.energy = np.maximum(world.energy - cfg.predator_damage * dt * bitten * active, 0.0)
            world.pain = np.minimum(world.pain + bitten * active, 1.0)  # un morso fa male
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
            self.food = self._outside(self.food, self.food_r)  # il cibo non ricompare dentro una roccia
            n_eat = (eat & ~self.bad).sum(axis=1)  # i frutti buoni nutrono e hanno gusto
            n_bad = (eat & self.bad).sum(axis=1)  # i cattivi fanno male e tolgono energia
            self.eaten += n_eat
            self.bad_eaten += n_bad
            world.taste = np.minimum(world.taste + n_eat, 1.0)  # il gusto del cibo
            world.pain = np.minimum(world.pain + n_bad, 1.0)
            world.energy = np.clip(world.energy + cfg.food_energy * n_eat - cfg.bad_fruit_damage * n_bad, 0.0,
                                   cfg.energy_max)

    def completed(self, world) -> np.ndarray:
        return self.eaten >= self.need


TASKS = {"bersaglio": ReachTarget, "cibo": Foraging}
