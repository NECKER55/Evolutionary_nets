"""L'arena delle pile (docs/SPEC_ARENA_PILE.md), ispirata al Cyber Rodent
project: in un recinto chiuso pochi esseri cercano pile per ricaricarsi e si
riproducono incontrandosi muso contro muso. Molti recinti girano insieme
(prima dimensione degli array: il recinto, seconda: l'essere).

Due tipi di pile, rosse e gialle: una ricarica, l'altra scarica un po' e fa
male; ogni `inversion_every` secondi i ruoli si scambiano. Le pile compaiono
in una di quattro zone (gli angoli), e ogni `zone_every` secondi la zona
cambia. I sensi sono fissi (`SENSES`); il corpo ha due ruote (spinta e
sterzo); prendere una pila è automatico a contatto e quasi fermi.

Il mondo non sa niente dei cervelli: riceve spinta e sterzo, restituisce i
sensi e ciò che è successo (piacere, dolore, morti, accoppiamenti). Chi muore
rinasce subito altrove; con che cervello lo decide chi guida gli esseri.
"""

from dataclasses import dataclass

import numpy as np

RED, YELLOW, BLUE = 0, 1, 2  # i colori: pile rosse, pile gialle, la luce sul muso degli esseri
SECTORS = np.radians([-60.0, -20.0, 20.0, 60.0])  # occhi: sinistra, centro, destra
RAYS = np.radians([-30.0, 0.0, 30.0])  # i raggi di distanza (infrarossi)

# I sensi, nell'ordine dei canali. Occhi: colore × direzione, 1 = oggetto
# addosso, 0 = lontano o assente.
SENSES = tuple(f"occhio {c} {s}" for c in ("rosso", "giallo", "blu") for s in ("sinistra", "centro", "destra")) + (
    "distanza sinistra", "distanza centro", "distanza destra", "contatto", "energia", "mi guardano", "gusto",
    "dolore")
EYES = slice(0, 9)
N_SENSES = len(SENSES)


@dataclass
class PileConfig:
    """Le costanti dell'arena delle pile (proposte da tarare: docs/SPEC_ARENA_PILE.md)."""

    size: float = 500.0  # lato del recinto
    beings: int = 6  # esseri per recinto
    dt: float = 0.05
    radius: float = 10.0
    speed: float = 120.0  # velocità massima (unità/s)
    turn: float = 3.0  # rad/s alla sterzata massima
    batteries: int = 8  # pile presenti al massimo in un recinto
    battery_radius: float = 8.0
    respawn: float = 2.0  # s prima che una pila presa ricompaia
    battery_life: float = 40.0  # s: una pila non presa si spegne e ricompare altrove (le cattive non si accumulano)
    zone_half: float = 100.0  # mezza larghezza di una zona
    zone_margin: float = 110.0  # distanza del centro della zona dai muri
    inversion_every: float = 300.0  # s: rosso e giallo si scambiano i ruoli
    zone_every: float = 200.0  # s: la zona delle pile cambia
    take_speed: float = 0.3  # per prendere una pila: velocità sotto questa quota della massima
    mouth: float = 20.0  # gradi: si prende solo una pila davanti al muso (non quelle che si sfiorano di lato)
    charge_time: float = 2.0  # s per consumare una pila buona (intanto si sta fermi)
    good_energy: float = 25.0
    bad_energy: float = 8.0
    energy_max: float = 100.0
    energy_start: float = 60.0
    drain: float = 0.15  # energia/s da fermi
    move_cost: float = 0.25  # energia/s in più a tutta spinta
    age_max: float = 600.0  # s
    eye_range: float = 350.0
    ray_range: float = 80.0
    watch_angle: float = 30.0  # gradi: "mi guardano" se il muso di un altro punta verso di me entro questo angolo
    watch_range: float = 150.0
    face_visible: float = 90.0  # gradi: la luce sul muso si vede se l'altro è rivolto verso di me entro questo angolo
    # Accoppiamento (solo con mating = True): entrambi sopra la soglia di
    # energia, muso contro muso e vicini per mate_time secondi; costa
    # mate_cost a entrambi; poi mate_rest secondi di riposo.
    mating: bool = False
    mate_energy: float = 50.0
    mate_distance: float = 30.0
    mate_angle: float = 30.0
    mate_time: float = 2.0
    mate_cost: float = 10.0
    mate_rest: float = 20.0
    taste_tau: float = 0.5  # s: il gusto e il dolore svaniscono così


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class PileWorld:
    """`enclosures` recinti con `cfg.beings` esseri ciascuno."""

    def __init__(self, cfg: PileConfig, enclosures: int, seed: int = 0):
        self.cfg, self.B, self.n = cfg, enclosures, cfg.beings
        self.rng = np.random.default_rng(seed)
        B, n, K = self.B, self.n, cfg.batteries
        self.t = 0.0
        self.pos = self._random_pos((B, n))
        self.heading = self.rng.uniform(-np.pi, np.pi, (B, n))
        self.speed = np.zeros((B, n))
        self.energy = np.full((B, n), cfg.energy_start)
        self.age = np.zeros((B, n))
        self.charging = np.zeros((B, n))  # energia ancora da prendere dalla pila in corso
        self.taste = np.zeros((B, n))
        self.pain = np.zeros((B, n))
        # Le pile: posizione, colore, presente, tempo prima di ricomparire.
        self.good = self.rng.integers(0, 2, B)  # il colore buono di ogni recinto
        self.zone = self.rng.integers(0, 4, B)
        self.bpos = np.zeros((B, K, 2))
        self.bcolor = np.zeros((B, K), dtype=int)
        self.bon = np.zeros((B, K), dtype=bool)
        self.bwait = self.rng.uniform(0.0, cfg.respawn, (B, K))
        self.bage = np.zeros((B, K))
        self.next_inversion = cfg.inversion_every
        self.next_zone = cfg.zone_every
        # Accoppiamento.
        self.pair_time = np.zeros((B, n, n))
        self.rest = np.zeros((B, n))
        # Contatori e registro degli eventi (per le misure).
        self.taken = []  # (t, recinto, essere, colore, buona)
        self.deaths = {"fame": 0, "vecchiaia": 0}
        self.matings = []  # (t, recinto, a, b)
        self.inversions = []  # tempi delle inversioni
        self.zone_changes = []  # tempi dei cambi di zona

    # --- Geometria -------------------------------------------------------------------

    def _random_pos(self, shape):
        r = self.cfg.radius
        return self.rng.uniform(r, self.cfg.size - r, shape + (2,))

    def zone_centres(self) -> np.ndarray:
        m, s = self.cfg.zone_margin, self.cfg.size
        return np.array([[m, m], [s - m, m], [m, s - m], [s - m, s - m]])

    def _spawn(self, mask):
        """Pile nuove dove `mask` (B, K): nella zona attiva, colore a caso."""
        if not mask.any():
            return
        b, k = np.nonzero(mask)
        centre = self.zone_centres()[self.zone[b]]
        h = self.cfg.zone_half
        lo, hi = self.cfg.battery_radius, self.cfg.size - self.cfg.battery_radius
        self.bpos[b, k] = np.clip(centre + self.rng.uniform(-h, h, (len(b), 2)), lo, hi)
        self.bcolor[b, k] = self.rng.integers(0, 2, len(b))
        self.bon[b, k] = True
        self.bage[b, k] = self.rng.uniform(0.0, 0.5 * self.cfg.battery_life, len(b))  # non scadono tutte insieme

    # --- I sensi ---------------------------------------------------------------------

    def senses(self) -> np.ndarray:
        """(B, n, N_SENSES), tutti tra 0 e 1."""
        cfg = self.cfg
        B, n = self.B, self.n
        out = np.zeros((B, n, N_SENSES))
        # Gli oggetti visibili: le pile presenti e la luce sul muso degli altri.
        objs = [(self.bpos, self.bcolor, self.bon)]
        face = np.broadcast_to(np.full(n, BLUE), (B, n))
        objs.append((self.pos, face, np.ones((B, n), dtype=bool)))
        for k, (opos, ocol, oon) in enumerate(objs):
            d = opos[:, None, :, :] - self.pos[:, :, None, :]  # (B, n, M, 2)
            dist = np.hypot(d[..., 0], d[..., 1])
            ang = _wrap(np.arctan2(d[..., 1], d[..., 0]) - self.heading[:, :, None])
            near = np.clip(1.0 - dist / cfg.eye_range, 0.0, 1.0) * oon[:, None, :]
            if k == 1:  # gli altri esseri: non se stessi, e la luce si vede solo se sono rivolti verso di me
                near = near * (1.0 - np.eye(n))[None]
                back = _wrap(np.arctan2(-d[..., 1], -d[..., 0]) - self.heading[:, None, :])
                near = near * (np.abs(back) < np.radians(cfg.face_visible))
            for s in range(3):
                inside = (ang >= SECTORS[s]) & (ang < SECTORS[s + 1])
                for c in ((RED, YELLOW) if k == 0 else (BLUE,)):
                    v = np.where(inside & (ocol[:, None, :] == c), near, 0.0).max(axis=2)
                    out[..., 3 * c + s] = np.maximum(out[..., 3 * c + s], v)
        # Distanza dai muri lungo tre raggi.
        for r, a in enumerate(RAYS):
            u = np.stack([np.cos(self.heading + a), np.sin(self.heading + a)], -1)
            t = np.full((B, n), np.inf)
            for axis in (0, 1):
                comp = u[..., axis]
                with np.errstate(divide="ignore", invalid="ignore"):
                    hit = np.where(comp > 0, (cfg.size - self.pos[..., axis]) / comp,
                                   np.where(comp < 0, -self.pos[..., axis] / comp, np.inf))
                t = np.minimum(t, hit)
            out[..., 9 + r] = np.clip(1.0 - (t - cfg.radius) / cfg.ray_range, 0.0, 1.0)
        # Contatto: un muro, un altro essere, una pila.
        wall = (self.pos.min(-1) < cfg.radius + 2.0) | (self.pos.max(-1) > cfg.size - cfg.radius - 2.0)
        dd = np.hypot(*np.moveaxis(self.pos[:, :, None] - self.pos[:, None], -1, 0)) + np.eye(n) * 1e9
        other = dd.min(-1) < 2.0 * cfg.radius
        out[..., 12] = (wall | other | (self.charging > 0)).astype(float)
        out[..., 13] = self.energy / cfg.energy_max
        out[..., 14] = self._watched().astype(float)
        out[..., 15] = self.taste
        out[..., 16] = self.pain
        return out

    def _watched(self):
        """(B, n): un altro essere vicino ha il muso verso di me."""
        cfg = self.cfg
        d = self.pos[:, :, None, :] - self.pos[:, None, :, :]  # da j a i: [b, i, j]
        dist = np.hypot(d[..., 0], d[..., 1])
        ang = _wrap(np.arctan2(d[..., 1], d[..., 0]) - self.heading[:, None, :])
        look = (np.abs(ang) < np.radians(cfg.watch_angle)) & (dist < cfg.watch_range) & ~np.eye(self.n, dtype=bool)
        return look.any(-1)

    # --- Un passo --------------------------------------------------------------------

    def step(self, thrust: np.ndarray, steer: np.ndarray) -> dict:
        """thrust (B, n) in [0, 1], steer (B, n) in [-1, 1]. Restituisce ciò che è
        successo in questo passo: piacere e dolore (B, n), chi è morto (B, n),
        accoppiamenti [(recinto, a, b)]."""
        cfg, dt = self.cfg, self.cfg.dt
        B, n = self.B, self.n
        # Mentre consuma una pila l'essere sta fermo (come il robot attaccato alla pila).
        thrust = np.where(self.charging > 0.0, 0.0, np.clip(thrust, 0.0, 1.0))
        self.heading = _wrap(self.heading + np.clip(steer, -1.0, 1.0) * cfg.turn * dt)
        self.speed = thrust * cfg.speed
        u = np.stack([np.cos(self.heading), np.sin(self.heading)], -1)
        self.pos = np.clip(self.pos + u * self.speed[..., None] * dt, cfg.radius, cfg.size - cfg.radius)
        self.age += dt
        self.energy -= (cfg.drain + cfg.move_cost * thrust) * dt
        fade = np.exp(-dt / cfg.taste_tau)
        self.taste *= fade
        self.pain *= fade
        pleasure = np.zeros((B, n))
        hurt = np.zeros((B, n))

        # Ricarica in corso (si è fermi: vedi sopra).
        moving = self.speed > cfg.take_speed * cfg.speed
        self.charging[moving] = 0.0
        gain = np.minimum(self.charging, cfg.good_energy * dt / cfg.charge_time)
        self.charging -= gain
        self.energy = np.minimum(self.energy + gain, cfg.energy_max)
        pleasure += gain / cfg.good_energy  # una pila intera vale 1
        self.taste = np.minimum(self.taste + gain / cfg.good_energy * 4.0, 1.0)

        # Pile nuove e pile prese.
        self.bwait -= dt
        self.bage += dt
        old = self.bon & (self.bage >= cfg.battery_life)
        self.bon[old] = False
        self.bwait[old] = cfg.respawn
        self._spawn(~self.bon & (self.bwait <= 0.0))
        d = self.bpos[:, None] - self.pos[:, :, None]  # (B, n, K, 2)
        front = np.abs(_wrap(np.arctan2(d[..., 1], d[..., 0]) - self.heading[:, :, None])) < np.radians(cfg.mouth)
        close = (np.hypot(d[..., 0], d[..., 1]) < cfg.radius + cfg.battery_radius) & self.bon[:, None] & front
        free = ~moving & (self.charging <= 0.0)
        for b, i in zip(*np.nonzero(free & close.any(-1))):
            k = int(np.flatnonzero(close[b, i] & self.bon[b])[0]) if (close[b, i] & self.bon[b]).any() else -1
            if k < 0:
                continue  # presa da un altro in questo passo
            self.bon[b, k] = False
            self.bwait[b, k] = cfg.respawn
            ok = self.bcolor[b, k] == self.good[b]
            self.taken.append((self.t, b, i, int(self.bcolor[b, k]), bool(ok)))
            if ok:
                self.charging[b, i] = cfg.good_energy
            else:
                self.energy[b, i] -= cfg.bad_energy
                self.pain[b, i] = 1.0
                hurt[b, i] = cfg.bad_energy / cfg.good_energy  # il dolore pesa quanto il danno (una pila buona vale 1)

        # Accoppiamenti.
        mates = self._mate() if cfg.mating else []

        # Inversioni e cambi di zona (in tutti i recinti insieme: le misure si allineano).
        self.t += dt
        if self.t >= self.next_inversion:
            self.good = 1 - self.good
            self.inversions.append(self.t)
            self.next_inversion += cfg.inversion_every
        if self.t >= self.next_zone:
            self.zone = (self.zone + self.rng.integers(1, 4, B)) % 4  # sempre una zona diversa
            self.zone_changes.append(self.t)
            self.next_zone += cfg.zone_every

        # Morti: per fame o per età; rinascono subito altrove.
        starved = self.energy <= 0.0
        old = self.age >= cfg.age_max
        dead = starved | old
        self.deaths["fame"] += int(starved.sum())
        self.deaths["vecchiaia"] += int((old & ~starved).sum())
        if dead.any():
            self.reborn(dead)
        return {"pleasure": pleasure, "pain": hurt, "dead": dead, "mates": mates}

    def reborn(self, mask):
        """Il corpo degli esseri `mask` (B, n) rinasce: posto e direzione a caso,
        energia di partenza, età zero."""
        cfg = self.cfg
        k = int(mask.sum())
        self.pos[mask] = self.rng.uniform(cfg.radius, cfg.size - cfg.radius, (k, 2))
        self.heading[mask] = self.rng.uniform(-np.pi, np.pi, k)
        self.energy[mask] = cfg.energy_start
        self.age[mask] = 0.0
        self.charging[mask] = 0.0
        self.taste[mask] = 0.0
        self.pain[mask] = 0.0
        self.rest[mask] = 0.0
        b, i = np.nonzero(mask)
        self.pair_time[b, i, :] = 0.0
        self.pair_time[b, :, i] = 0.0

    def _mate(self):
        """Coppie che restano muso contro muso abbastanza a lungo."""
        cfg, dt = self.cfg, self.cfg.dt
        n = self.n
        self.rest = np.maximum(self.rest - dt, 0.0)
        d = self.pos[:, None, :, :] - self.pos[:, :, None, :]  # [b, i, j] = da i a j
        dist = np.hypot(d[..., 0], d[..., 1])
        ang = np.abs(_wrap(np.arctan2(d[..., 1], d[..., 0]) - self.heading[:, :, None]))
        facing = ang < np.radians(cfg.mate_angle)
        ready = (self.energy >= cfg.mate_energy) & (self.rest <= 0.0) & (self.speed <= cfg.take_speed * cfg.speed)
        ok = facing & facing.transpose(0, 2, 1) & (dist < cfg.mate_distance) & ready[:, :, None] & ready[:, None, :]
        ok &= ~np.eye(n, dtype=bool)
        self.pair_time = np.where(ok, self.pair_time + dt, 0.0)
        out = []
        for b, i, j in zip(*np.nonzero(np.triu(self.pair_time >= cfg.mate_time, 1))):
            if self.rest[b, i] > 0 or self.rest[b, j] > 0:
                continue
            self.energy[b, [i, j]] -= cfg.mate_cost
            self.rest[b, [i, j]] = cfg.mate_rest
            self.pair_time[b, i, :] = self.pair_time[b, :, i] = 0.0
            self.pair_time[b, j, :] = self.pair_time[b, :, j] = 0.0
            self.matings.append((self.t, int(b), int(i), int(j)))
            out.append((int(b), int(i), int(j)))
        return out


# --- I programmi di movimento (la sezione F, fissa) ---------------------------------

PROGRAMS = ("verso il rosso", "verso il giallo", "verso il blu", "esplora", "evita")
N_PROGRAMS = len(PROGRAMS)


class Motor:
    """F: dai programmi alle ruote, innato e uguale per tutti. Ogni programma è
    un movimento semplice guidato dai sensi (come le tassie degli animali):
    girarsi verso un colore e avvicinarsi rallentando, vagare, girarsi via da
    ciò che si ha davanti. Un riflesso fisso fa girare via dai muri: sceglie
    da che parte girare e ci resta finché davanti non è libero (senza, in un
    angolo oscillava tra le due parti per sempre, spingendo contro i muri)."""

    def __init__(self, shape, rng: np.random.Generator):
        self.rng = rng
        self.wander = np.zeros(shape)
        self.escape = np.zeros(shape)  # verso della fuga dal muro in corso (0 = nessuna)

    def __call__(self, program: np.ndarray, senses: np.ndarray, dt: float):
        eye = senses[..., EYES].reshape(senses.shape[:-1] + (3, 3))  # colore × (sinistra, centro, destra)
        thrust = np.zeros(program.shape)
        steer = np.zeros(program.shape)
        for c in range(3):
            m = program == c
            if not m.any():
                continue
            e = eye[..., c, :]
            seen = e.max(-1) > 0.0
            near = e.max(-1)
            # Verso il lato che lo vede di più; se è già al centro, solo piccole correzioni.
            centred = e[..., 1] >= np.maximum(e[..., 0], e[..., 2])
            turn = np.clip(np.where(centred, 0.5, 4.0) * (e[..., 2] - e[..., 0]), -1.0, 1.0)
            # Visto: avvicinati, rallentando da vicino (per poter prendere la pila
            # o fermarsi davanti a un compagno). Non visto: gira sul posto e cerca.
            th = np.where(seen, np.clip(1.2 * (1.0 - near), 0.15, 1.0), 0.2)
            st = np.where(seen, turn, 0.8)
            thrust = np.where(m, th, thrust)
            steer = np.where(m, st, steer)
        self.wander = 0.9 * self.wander + 0.3 * self.rng.normal(0.0, 1.0, self.wander.shape)
        m = program == 3
        thrust = np.where(m, 0.8, thrust)
        steer = np.where(m, np.clip(self.wander, -1.0, 1.0), steer)
        m = program == 4
        lr = eye.sum(-2)  # quanto si vede a sinistra, centro, destra (tutti i colori)
        away = np.where(lr[..., 0] > lr[..., 2], 1.0, -1.0)
        thrust = np.where(m, 0.6, thrust)
        steer = np.where(m, away, steer)
        # Riflesso: un muro davanti fa girare dalla parte più libera e frenare.
        ray = senses[..., 9:12]
        wall = ray.max(-1) > 0.5
        free_side = np.where(ray[..., 0] < ray[..., 2], -1.0, 1.0)
        self.escape = np.where(wall, np.where(self.escape == 0.0, free_side, self.escape), 0.0)
        steer = np.where(wall, self.escape, steer)
        thrust = thrust * (1.0 - 0.7 * np.clip(ray[..., 1] - 0.3, 0.0, 1.0))
        return thrust, steer
