"""L'arena delle pile (docs/SPEC_ARENA_PILE.md), ispirata al Cyber Rodent
project: in un recinto chiuso pochi esseri cercano pile per ricaricarsi e si
riproducono incontrandosi muso contro muso. Molti recinti girano insieme
(prima dimensione degli array: il recinto, seconda: l'essere).

Due tipi di pile, rosse e gialle: una ricarica, l'altra scarica un po' e fa
male; ogni `inversion_every` secondi i ruoli si scambiano. Le pile compaiono
in una di quattro zone (gli angoli), e ogni `zone_every` secondi la zona
cambia. I quattro muri hanno colori diversi (punti di riferimento).

I sensi sono fissi (`SENSES`, calcolati da `sense`, uguale per NumPy e JAX):
occhi per colore e direzione (pile, compagni, muri), tre raggi di distanza,
tre sensori d'urto, energia, "mi guardano", gusto, dolore. Il corpo ha
motori elementari: girare, avanti/indietro, mordere. Si mangia solo
mordendo: una pila davanti al muso, vicina, andando piano; mordere costa un
po' di energia.

Il mondo non sa niente dei cervelli: riceve sterzo, spinta e morso,
restituisce i sensi e ciò che è successo (piacere, dolore, morti,
accoppiamenti). Chi muore rinasce subito altrove; con che cervello lo decide
chi guida gli esseri.
"""

from dataclasses import dataclass

import numpy as np

RED, YELLOW, BLUE = 0, 1, 2  # i colori: pile rosse, pile gialle, la luce sul muso degli esseri
SECTORS = np.radians([-60.0, -20.0, 20.0, 60.0])  # occhi: sinistra, centro, destra
SIGHT = np.radians([-40.0, 0.0, 40.0])  # dove guarda ogni settore (per i muri)
RAYS = np.radians([-30.0, 0.0, 30.0])  # i raggi di distanza (infrarossi)
WALLS = ("nord", "est", "sud", "ovest")  # y = 0, x = lato, y = lato, x = 0
WALL_COLOURS = ("verde", "viola", "arancio", "bianco")
SIDES = ("sinistra", "centro", "destra")

# I sensi, nell'ordine dei canali. Occhi: colore × direzione, 1 = oggetto
# addosso, 0 = lontano o assente; per i muri, quanto è vicino il muro di
# quel colore che si vede in quella direzione (almeno 0,1 se si vede).
SENSES = (tuple(f"occhio {c} {s}" for c in ("rosso", "giallo", "blu") for s in SIDES)
          + tuple(f"occhio muro {c} {s}" for c in WALL_COLOURS for s in SIDES)
          + ("distanza sinistra", "distanza centro", "distanza destra", "urto sinistra", "urto davanti",
             "urto destra", "energia", "mi guardano", "gusto", "dolore"))
EYES = slice(0, 9)
WALL_EYES = slice(9, 21)
RAY = slice(21, 24)
BUMP = slice(24, 27)
ENERGY, WATCHED, TASTE, PAIN = 27, 28, 29, 30
N_SENSES = len(SENSES)


def ghost_view(bon, ghost, ghost_hide, xp=np):
    """Che cosa vede ognuno quando c'è un maestro fantasma (`ghost` (B, n)):
    le pile (B, n, K) (quelle che il fantasma ha mangiato spariscono solo per
    lui, `ghost_hide` (B, K) > 0) e gli altri esseri (B, n, n): il fantasma
    non lo vede nessuno."""
    if ghost is None:
        return bon[:, None, :], None
    bon_each = bon[:, None, :] & ~(ghost[:, :, None] & (ghost_hide[:, None, :] > 0.0))
    return bon_each, ghost[:, None, :]


def sense(cfg, pos, heading, bpos, bcolor, bon, energy, taste, pain, xp=np, ghost=None, ghost_hide=None):
    """I sensi di ogni essere, (B, n, N_SENSES), tutti tra 0 e 1: la stessa
    funzione per NumPy e JAX (`xp`). `ghost`, `ghost_hide`: il maestro
    fantasma (vedi `ghost_view`)."""
    wrap = lambda a: (a + np.pi) % (2.0 * np.pi) - np.pi
    B, n = heading.shape
    bon_each, hidden = ghost_view(bon, ghost, ghost_hide, xp)
    eye = xp.eye(n, dtype=bool)[None] if hidden is None else xp.eye(n, dtype=bool)[None] | hidden
    chans = []
    # Occhi: pile (rosso, giallo) e la luce sul muso degli altri (blu), se sono rivolti verso di me.
    db = bpos[:, None, :, :] - pos[:, :, None, :]
    ab = wrap(xp.arctan2(db[..., 1], db[..., 0]) - heading[:, :, None])
    nb = xp.clip(1.0 - xp.hypot(db[..., 0], db[..., 1]) / cfg.eye_range, 0.0, 1.0) * bon_each
    de = pos[:, None, :, :] - pos[:, :, None, :]
    ae = wrap(xp.arctan2(de[..., 1], de[..., 0]) - heading[:, :, None])
    back = wrap(xp.arctan2(-de[..., 1], -de[..., 0]) - heading[:, None, :])
    ne = xp.clip(1.0 - xp.hypot(de[..., 0], de[..., 1]) / cfg.eye_range, 0.0, 1.0)
    ne = ne * ~eye * (xp.abs(back) < np.radians(cfg.face_visible))
    for c in (RED, YELLOW, BLUE):
        for s in range(3):
            if c == BLUE:
                inside = (ae >= SECTORS[s]) & (ae < SECTORS[s + 1])
                chans.append(xp.where(inside, ne, 0.0).max(axis=2))
            else:
                inside = (ab >= SECTORS[s]) & (ab < SECTORS[s + 1]) & (bcolor[:, None, :] == c)
                chans.append(xp.where(inside, nb, 0.0).max(axis=2))

    def wall_hits(angle):
        """Distanza da ognuno dei 4 muri lungo la direzione `angle` (inf se non lo colpisce)."""
        ux, uy = xp.cos(heading + angle), xp.sin(heading + angle)
        x, y = pos[..., 0], pos[..., 1]
        big = 1e9
        north = xp.where(uy < 0, -y / xp.where(uy < 0, uy, -1.0), big)
        east = xp.where(ux > 0, (cfg.size - x) / xp.where(ux > 0, ux, 1.0), big)
        south = xp.where(uy > 0, (cfg.size - y) / xp.where(uy > 0, uy, 1.0), big)
        west = xp.where(ux < 0, -x / xp.where(ux < 0, ux, -1.0), big)
        return xp.stack([north, east, south, west], -1)

    # Occhi per i muri: in ogni direzione si vede il muro più vicino, del suo colore.
    walls = []
    for s in range(3):
        t = wall_hits(SIGHT[s])
        first = t.argmin(-1)
        near = xp.clip(1.0 - t.min(-1) / (2.0 * cfg.size), 0.1, 1.0)
        walls.append([xp.where(first == w, near, 0.0) for w in range(4)])
    chans += [walls[s][w] for w in range(4) for s in range(3)]
    # Raggi di distanza (infrarossi).
    for a in RAYS:
        chans.append(xp.clip(1.0 - (wall_hits(a).min(-1) - cfg.radius) / cfg.ray_range, 0.0, 1.0))
    # Urti: muri, altri esseri, pile, a sinistra, davanti o a destra.
    x, y = pos[..., 0], pos[..., 1]
    touch = cfg.radius + 2.0
    wall_on = xp.stack([y < touch, x > cfg.size - touch, y > cfg.size - touch, x < touch], -1)
    wall_dir = xp.asarray(np.array([-np.pi / 2, 0.0, np.pi / 2, np.pi]))
    a_wall = wrap(wall_dir[None, None, :] - heading[..., None])
    on_e = (xp.hypot(de[..., 0], de[..., 1]) < 2.0 * cfg.radius) & ~eye
    on_b = (xp.hypot(db[..., 0], db[..., 1]) < cfg.radius + cfg.battery_radius) & bon_each
    ang = xp.concatenate([a_wall, ae, ab], -1)
    on = xp.concatenate([wall_on, on_e, on_b], -1)
    q = np.pi / 4
    chans += [(on & (ang < -q)).any(-1).astype(float), (on & (xp.abs(ang) <= q)).any(-1).astype(float),
              (on & (ang > q)).any(-1).astype(float)]
    # Corpo e compagni.
    look = (xp.abs(back) < np.radians(cfg.watch_angle)) & (xp.hypot(de[..., 0], de[..., 1]) < cfg.watch_range) & ~eye
    chans += [energy / cfg.energy_max, look.any(-1).astype(float), taste, pain]
    return xp.stack(chans, -1)


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
    inversion_every: float = 300.0  # s: rosso e giallo (o grande e piccola) si scambiano i ruoli
    # Che cosa rende buona una pila: il colore ("colore") o la grandezza
    # ("grandezza"; allora il colore non conta). Le pile sono rosse o gialle, e
    # grandi o piccole, a caso.
    good_by: str = "colore"
    zone_every: float = 200.0  # s: la zona delle pile cambia
    take_speed: float = 0.3  # per mordere una pila: velocità sotto questa quota della massima
    mouth: float = 25.0  # gradi: si morde solo una pila davanti al muso
    bite_reach: float = 6.0  # la pila deve essere entro raggio + raggio della pila + questo
    bite_cost: float = 0.1  # energia/s mentre si tiene la bocca aperta per mordere
    reverse: float = 0.5  # all'indietro si va a questa quota della velocità massima
    charge_time: float = 2.0  # s per consumare una pila buona (intanto si sta fermi)
    good_energy: float = 25.0
    bad_energy: float = 8.0
    energy_max: float = 100.0
    energy_start: float = 60.0
    drain: float = 0.15  # energia/s da fermi
    move_cost: float = 0.25  # energia/s in più a tutta spinta
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
        self.bbig = np.zeros((B, K), dtype=int)  # 1 = pila grande
        self.bon = np.zeros((B, K), dtype=bool)
        self.bwait = self.rng.uniform(0.0, cfg.respawn, (B, K))
        self.bage = np.zeros((B, K))
        self.next_inversion = cfg.inversion_every
        self.next_zone = cfg.zone_every
        # Accoppiamento.
        self.pair_time = np.zeros((B, n, n))
        self.rest = np.zeros((B, n))
        # Il maestro fantasma (se c'è): non lo vede nessuno, e le pile che mangia
        # spariscono solo per lui (per `ghost_hide` secondi o finché ricompaiono).
        self.ghost = np.zeros((B, n), dtype=bool)
        self.ghost_hide = np.zeros((B, K))
        # Contatori e registro degli eventi (per le misure).
        self.taken = []  # (t, recinto, essere, colore, buona)
        self.deaths = {"fame": 0}
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
        self.bbig[b, k] = self.rng.integers(0, 2, len(b))
        self.bon[b, k] = True
        self.bage[b, k] = self.rng.uniform(0.0, 0.5 * self.cfg.battery_life, len(b))  # non scadono tutte insieme
        self.ghost_hide[b, k] = 0.0  # una pila nuova: la vede anche il fantasma

    # --- I sensi ---------------------------------------------------------------------

    def senses(self) -> np.ndarray:
        """(B, n, N_SENSES), tutti tra 0 e 1 (vedi `sense`)."""
        return sense(self.cfg, self.pos, self.heading, self.bpos, self.bcolor, self.bon, self.energy, self.taste,
                     self.pain, ghost=self.ghost, ghost_hide=self.ghost_hide)

    def objects(self) -> dict:
        """Gli oggetti in vista di ogni essere (vedi `objects_of`)."""
        return objects_of(self.pos, self.heading, self.speed, self.bpos, self.bcolor, self.bbig, self.bon,
                          self.battery_good(), self.cfg, ghost=self.ghost, ghost_hide=self.ghost_hide)

    def battery_good(self) -> np.ndarray:
        """(B, K): quali pile sono buone adesso."""
        return battery_good(self.cfg, self.bcolor, self.bbig, self.good)

    # --- Un passo --------------------------------------------------------------------

    def step(self, thrust: np.ndarray, steer: np.ndarray, bite=None) -> dict:
        """thrust (B, n) in [-1, 1] (indietro più piano), steer (B, n) in [-1, 1],
        bite (B, n) bool: si morde. Restituisce ciò che è successo in questo
        passo: piacere e dolore (B, n), l'energia spesa (B, n, in pile buone:
        una pila buona = 1), chi è morto (B, n), accoppiamenti [(recinto, a, b)]."""
        cfg, dt = self.cfg, self.cfg.dt
        B, n = self.B, self.n
        bite = np.zeros((B, n), dtype=bool) if bite is None else np.asarray(bite, dtype=bool)
        # Mentre consuma una pila l'essere sta fermo (come il robot attaccato alla pila).
        thrust = np.where(self.charging > 0.0, 0.0, np.clip(thrust, -1.0, 1.0))
        self.heading = _wrap(self.heading + np.clip(steer, -1.0, 1.0) * cfg.turn * dt)
        self.speed = thrust * cfg.speed * np.where(thrust < 0, cfg.reverse, 1.0)
        u = np.stack([np.cos(self.heading), np.sin(self.heading)], -1)
        self.pos = np.clip(self.pos + u * self.speed[..., None] * dt, cfg.radius, cfg.size - cfg.radius)
        self.age += dt
        spent = (cfg.drain + cfg.move_cost * np.abs(thrust) + cfg.bite_cost * bite) * dt
        self.energy -= spent
        fade = np.exp(-dt / cfg.taste_tau)
        self.taste *= fade
        self.pain *= fade
        pleasure = np.zeros((B, n))
        hurt = np.zeros((B, n))

        # Ricarica in corso (si è fermi: vedi sopra).
        moving = np.abs(self.speed) > cfg.take_speed * cfg.speed
        self.charging[moving] = 0.0
        gain = np.minimum(self.charging, cfg.good_energy * dt / cfg.charge_time)
        self.charging -= gain
        self.energy = np.minimum(self.energy + gain, cfg.energy_max)
        pleasure += gain / cfg.good_energy  # una pila intera vale 1
        self.taste = np.minimum(self.taste + gain / cfg.good_energy * 4.0, 1.0)

        # Pile nuove e pile prese.
        self.bwait -= dt
        self.bage += dt
        self.ghost_hide = np.maximum(self.ghost_hide - dt, 0.0)
        old = self.bon & (self.bage >= cfg.battery_life)
        self.bon[old] = False
        self.bwait[old] = cfg.respawn
        self._spawn(~self.bon & (self.bwait <= 0.0))
        d = self.bpos[:, None] - self.pos[:, :, None]  # (B, n, K, 2)
        front = np.abs(_wrap(np.arctan2(d[..., 1], d[..., 0]) - self.heading[:, :, None])) < np.radians(cfg.mouth)
        bon_each, _ = ghost_view(self.bon, self.ghost, self.ghost_hide)
        close = (np.hypot(d[..., 0], d[..., 1]) < cfg.radius + cfg.battery_radius + cfg.bite_reach) \
            & bon_each & front
        free = ~moving & (self.charging <= 0.0) & bite
        for b, i in zip(*np.nonzero(free & close.any(-1))):
            avail = close[b, i] & (self.bon[b] if not self.ghost[b, i] else True)
            k = int(np.flatnonzero(avail)[0]) if avail.any() else -1
            if k < 0:
                continue  # presa da un altro in questo passo
            if self.ghost[b, i]:
                self.ghost_hide[b, k] = cfg.battery_life  # sparisce solo per lui
            else:
                self.bon[b, k] = False
                self.bwait[b, k] = cfg.respawn
            ok = bool(self.battery_good()[b, k])
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

        # Morti: solo per fame (nessun tetto di età); rinascono subito altrove.
        dead = self.energy <= 0.0
        self.deaths["fame"] += int(dead.sum())
        if dead.any():
            self.reborn(dead)
        return {"pleasure": pleasure, "pain": hurt, "spent": spent / cfg.good_energy, "dead": dead, "mates": mates}

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
        ready = (self.energy >= cfg.mate_energy) & (self.rest <= 0.0) & (np.abs(self.speed) <= cfg.take_speed * cfg.speed) \
            & ~self.ghost
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


# --- Gli oggetti in vista (per l'attenzione) ----------------------------------------


FEATURES = ("rosso", "giallo", "blu", "grande", "piccolo", "si muove")  # ciò che si vede di un oggetto
N_FEATURES = len(FEATURES)


def battery_good(cfg, bcolor, bbig, good):
    """(B, K): una pila è buona se ha il colore buono (o la grandezza buona)."""
    return (bbig if cfg.good_by == "grandezza" else bcolor) == good[:, None]


def objects_of(pos, heading, speed, bpos, bcolor, bbig, bon, bgood, cfg, xp=np, ghost=None, ghost_hide=None):
    """Gli oggetti che ogni essere vede, per l'attenzione: prima le pile (K),
    poi gli altri esseri (n). Per ognuno: visibile (dentro il ventaglio degli
    occhi e la portata; un essere solo se è rivolto verso di me, come la luce
    sul muso), le sue caratteristiche (`FEATURES`: colore, grandezza, se si
    muove), vicinanza (1 = addosso, 0 = al limite della vista), angolo
    rispetto al muso (rad, positivo = destra), e se è una pila buona (solo
    per il guidato e le misure: un cervello non lo sa). Forme (B, n, K + n).
    `ghost`, `ghost_hide`: il maestro fantasma (vedi `ghost_view`)."""
    B, n = heading.shape
    K = bpos.shape[1]
    opos = xp.concatenate([bpos, pos], axis=1)
    bon_each, hidden = ghost_view(bon, ghost, ghost_hide, xp)
    others = xp.ones((B, n, n), dtype=bool) if hidden is None else ~xp.broadcast_to(hidden, (B, n, n))
    on = xp.concatenate([xp.broadcast_to(bon_each, (B, n, K)), others], axis=2)  # (B, n, M): chi vede che cosa
    f_batt = xp.stack([bcolor == RED, bcolor == YELLOW, xp.zeros((B, K), dtype=bool), bbig == 1, bbig == 0,
                       xp.zeros((B, K), dtype=bool)], -1)
    moving = xp.abs(speed) > 0.05 * cfg.speed
    f_being = xp.stack([xp.zeros((B, n), dtype=bool), xp.zeros((B, n), dtype=bool), xp.ones((B, n), dtype=bool),
                        xp.zeros((B, n), dtype=bool), xp.zeros((B, n), dtype=bool), moving], -1)
    feat = xp.concatenate([f_batt, f_being], 1).astype(float)  # (B, M, F)
    d = opos[:, None, :, :] - pos[:, :, None, :]
    dist = xp.hypot(d[..., 0], d[..., 1])
    ang = (xp.arctan2(d[..., 1], d[..., 0]) - heading[:, :, None] + np.pi) % (2 * np.pi) - np.pi
    near = xp.clip(1.0 - dist / cfg.eye_range, 0.0, 1.0)
    self_or_back = xp.concatenate([xp.zeros((B, n, K), dtype=bool), xp.eye(n, dtype=bool)[None].repeat(B, 0)], 2)
    dk = d[:, :, K:, :]  # verso gli altri esseri: [b, i, j]; l'altro (j) è rivolto verso di me?
    back = (xp.arctan2(-dk[..., 1], -dk[..., 0]) - heading[:, None, :] + np.pi) % (2 * np.pi) - np.pi
    facing_me = xp.concatenate([xp.ones((B, n, K), dtype=bool), xp.abs(back) < np.radians(cfg.face_visible)], 2)
    vis = on & (near > 0.0) & (xp.abs(ang) < SECTORS[-1]) & ~self_or_back & facing_me
    good = xp.concatenate([bgood, xp.zeros((B, n), dtype=bool)], 1)
    shape = vis.shape
    return {"vis": vis, "feat": xp.broadcast_to(feat[:, None], shape + (N_FEATURES,)), "near": near, "ang": ang,
            "good": xp.broadcast_to(good[:, None], shape)}


# --- I programmi di movimento e il circuito motore (la sezione F) -------------------

PROGRAMS = ("avvicinati", "allontanati", "lascialo", "esplora")
N_PROGRAMS = len(PROGRAMS)

# F è un piccolo circuito di neuroni innati (come i circuiti motori del tronco
# encefalico): non impara durante la vita, ma i suoi pesi sono geni (fase B).
# Ingressi: il programma scelto da E combinato con dove sta l'oggetto attento,
# la deriva del vagare, il riflesso dei muri. Uscite: 4 neuroni motori.
F_INPUTS = ("avvicinati: oggetto a sinistra", "avvicinati: oggetto a destra", "avvicinati: oggetto lontano",
            "avvicinati: oggetto vicino", "allontanati: oggetto a sinistra", "allontanati: oggetto a destra",
            "allontanati", "vaga", "vaga: deriva a sinistra", "vaga: deriva a destra", "riflesso: muro, gira a sinistra",
            "riflesso: muro, gira a destra", "muro davanti", "costante")
F_NEURONS = ("gira a sinistra", "gira a destra", "avanti", "indietro")
N_FIN, N_FOUT = len(F_INPUTS), len(F_NEURONS)


def default_motor() -> np.ndarray:
    """I pesi di partenza di F (4 × 14): riproducono i programmi innati.
    Avvicinati: girarsi verso l'oggetto e andarci, rallentando da vicino;
    allontanati: girarsi via e andare; vaga: avanti con una deriva a caso;
    riflesso: un muro davanti fa girare dalla parte libera e frenare."""
    W = np.zeros((N_FOUT, N_FIN))
    L, R, F, B = range(N_FOUT)
    W[L, 0] = W[R, 1] = 3.1
    W[L, 5] = W[R, 4] = 2.0
    W[L, 8] = W[R, 9] = 1.0
    W[L, 10] = W[R, 11] = 4.0
    W[F, 2], W[F, 3], W[F, 6], W[F, 7], W[F, 12] = 1.35, 0.15, 0.7, 0.8, -0.7
    return W


def motor_inputs(program, has, near, ang, wander, escape, rays, xp=np):
    """Gli ingressi di F, (..., 14)."""
    side = SECTORS[-1]
    left, right = xp.clip(-ang / side, 0.0, 1.0), xp.clip(ang / side, 0.0, 1.0)
    go = ((program == 0) & has).astype(float)
    away = ((program == 1) & has).astype(float)
    roam = 1.0 - go - away
    one = xp.ones_like(near)
    return xp.stack([go * left, go * right, go * (1.0 - near), go * near, away * left, away * right, away, roam,
                     roam * xp.maximum(-wander, 0.0), roam * xp.maximum(wander, 0.0),
                     (escape < 0).astype(float), (escape > 0).astype(float),
                     xp.clip(rays[..., 1] - 0.3, 0.0, 1.0), one], -1)


def escape_reflex(rays, escape, xp=np):
    """Il riflesso dei muri: con un muro vicino davanti sceglie da che parte
    girare (la più libera) e ci resta finché davanti non è libero (senza, in
    un angolo oscillava tra le due parti per sempre)."""
    wall = rays.max(-1) > 0.5
    free_side = xp.where(rays[..., 0] < rays[..., 2], -1.0, 1.0)
    return xp.where(wall, xp.where(escape == 0.0, free_side, escape), 0.0)


def motor_net(x, W, xp=np):
    """F: (ingressi, pesi (..., 4, 14)) -> (spinta, sterzo, attività dei 4 neuroni)."""
    a = xp.clip(xp.einsum("...oi,...i->...o", W, x), 0.0, 1.0)
    return a[..., 2] - a[..., 3], a[..., 1] - a[..., 0], a


class Motor:
    """F nella versione NumPy: tiene la deriva del vagare e il verso della fuga
    dal muro di ogni essere."""

    def __init__(self, shape, rng: np.random.Generator):
        self.rng = rng
        self.wander = np.zeros(shape)
        self.escape = np.zeros(shape)
        self.activity = np.zeros(shape + (N_FOUT,))

    def __call__(self, program, senses, has, near, ang, W):
        """program, has, near, ang: (B, n); W: (B, n, 4, 14). -> (spinta, sterzo)."""
        self.wander = 0.9 * self.wander + 0.3 * self.rng.normal(0.0, 1.0, self.wander.shape)
        rays = senses[..., RAY]
        self.escape = escape_reflex(rays, self.escape)
        self.inputs = motor_inputs(program, has, near, ang, self.wander, self.escape, rays)
        thrust, steer, self.activity = motor_net(self.inputs, W)
        return thrust, steer
