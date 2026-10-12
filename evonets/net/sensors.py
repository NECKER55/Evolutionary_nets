import numpy as np

from ..config import Config

# Il canale "consegna" è riservato: resta a 0 finché un compito non definisce
# una zona di consegna, così la forma della rete non cambia tra un compito e l'altro.
CHANNELS = ("parete", "veleno", "bersaglio", "consegna")
BODY = ("vel. avanti", "vel. traverso", "rotazione", "energia", "dolore", "urto")


class Vision:
    """Campo visivo: un ventaglio di raggi davanti all'agente.

    Ogni raggio restituisce, per canale, la *vicinanza* di ciò che incontra:
    0 = niente entro `vision_range`, 1 = a contatto. I canali sono indipendenti
    (il veleno è una nube, non nasconde ciò che sta dietro).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def size(self) -> int:
        """Numero di ingressi che la vista fornisce a una rete."""
        return self.cfg.vision_rays * len(CHANNELS)

    def ray_angles(self, heading: np.ndarray) -> np.ndarray:
        """(N, R): direzione assoluta di ogni raggio, da sinistra a destra."""
        half = np.radians(self.cfg.vision_fov_deg) / 2.0
        return heading[:, None] + np.linspace(-half, half, self.cfg.vision_rays)[None, :]

    def sense(self, world) -> np.ndarray:
        """(N, R, C) in [0, 1], un canale per tipo di cosa vista."""
        cfg = self.cfg
        # Internamente l'asse degli agenti sta per ultimo: è quello lungo, e
        # NumPy è molto più veloce quando il ciclo interno scorre su di esso.
        ang = np.ascontiguousarray(self.ray_angles(world.heading).T)  # (R, N)
        dx, dy = np.cos(ang), np.sin(ang)
        ox, oy = world.pos[:, 0], world.pos[:, 1]
        dist = np.empty((len(CHANNELS), *ang.shape))
        dist[0] = self._walls(ox, oy, dx, dy)
        dist[1] = self._circles(ox, oy, dx, dy, world.hazards.pos, world.hazards.radius)
        for ch, (pos, rad) in ((2, ("target", "radius")), (3, ("goal", "goal_radius"))):
            if hasattr(world.task, pos):
                center, radius = getattr(world.task, pos), getattr(world.task, rad)
                dist[ch] = self._circles(ox, oy, dx, dy, center[None, :], np.array([radius]))
            else:
                dist[ch] = np.inf
        close = np.clip(1.0 - dist / cfg.vision_range, 0.0, 1.0)
        return np.ascontiguousarray(close.transpose(2, 1, 0))

    def _walls(self, ox, oy, dx, dy) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx != 0.0, (np.where(dx > 0.0, self.cfg.arena_w, 0.0) - ox) / dx, np.inf)
            ty = np.where(dy != 0.0, (np.where(dy > 0.0, self.cfg.arena_h, 0.0) - oy) / dy, np.inf)
        return np.minimum(tx, ty)

    @staticmethod
    def _circles(ox, oy, dx, dy, centers, radii) -> np.ndarray:
        """Distanza (R, N) dal cerchio più vicino lungo ogni raggio; inf = nessuno."""
        best = np.full(dx.shape, np.inf)
        # Un cerchio alla volta: i cerchi sono pochi e gli array restano piccoli.
        for (cx, cy), r in zip(centers, radii):
            fx, fy = cx - ox, cy - oy
            proj = fx * dx + fy * dy
            disc = (r * r - fx * fx - fy * fy) + proj * proj
            half_chord = np.sqrt(np.maximum(disc, 0.0))
            # Colpito se la retta taglia il cerchio e l'uscita è davanti; se
            # l'origine è già dentro, l'ingresso è alle spalle -> distanza 0.
            hit = (disc >= 0.0) & (proj + half_chord >= 0.0)
            np.minimum(best, np.where(hit, np.maximum(proj - half_chord, 0.0), np.inf), out=best)
        return best


class Senses:
    """Tutto ciò che la rete riceve: la vista seguita dai sensi del corpo.

    `observe` restituisce (N, size) con valori in [-1, 1]; `labels` dà il nome
    di ogni ingresso, nello stesso ordine.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vision = Vision(cfg)
        self.rays = None  # ultima vista (N, R, C), per il viewer

    @property
    def size(self) -> int:
        return self.vision.size + len(BODY)

    @property
    def labels(self) -> list[str]:
        rays = [f"raggio {r + 1} {c}" for r in range(self.cfg.vision_rays) for c in CHANNELS]
        return rays + list(BODY)

    def observe(self, world) -> np.ndarray:
        cfg = self.cfg
        self.rays = self.vision.sense(world)
        cos, sin = np.cos(world.heading), np.sin(world.heading)
        vx, vy = world.vel[:, 0], world.vel[:, 1]
        v_max = cfg.thrust_accel / cfg.lin_drag
        w_max = cfg.turn_accel / cfg.ang_drag
        body = np.stack(
            [
                np.clip((vx * cos + vy * sin) / v_max, -1.0, 1.0),
                np.clip((vy * cos - vx * sin) / v_max, -1.0, 1.0),
                np.clip(world.ang_vel / w_max, -1.0, 1.0),
                world.energy / cfg.energy_init,
                world.pain,
                world.bump.astype(float),
            ],
            axis=1,
        )
        return np.concatenate([self.rays.reshape(world.n, -1), body], axis=1)
