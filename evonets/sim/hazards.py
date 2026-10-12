import numpy as np

from ..config import Config


class HazardField:
    """H zone tossiche circolari. Ogni centro oscilla attorno a un'ancora con una curva di
    Lissajous (una sinusoide indipendente per asse) e il raggio pulsa."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        h = cfg.n_hazards
        size = np.array([cfg.arena_w, cfg.arena_h])
        self.base_radius = rng.uniform(cfg.hazard_radius_min, cfg.hazard_radius_max, h)
        # Ingombro massimo, così il pericolo non esce mai dall'arena.
        reach = (self.base_radius * (1.0 + cfg.hazard_pulse))[:, None]
        self.anchor = rng.uniform(reach, size - reach, (h, 2))
        room = np.minimum(self.anchor - reach, size - reach - self.anchor)
        self.amp = room * rng.uniform(0.3, 1.0, (h, 2))
        self.omega = rng.uniform(cfg.hazard_omega_min, cfg.hazard_omega_max, (h, 2))
        self.pulse_omega = rng.uniform(0.5, 1.5, h)
        # Le fasi vengono integrate (non calcolate da t) così hazard_speed può
        # cambiare a metà simulazione senza far saltare i pericoli.
        self.phase = rng.uniform(0.0, 2.0 * np.pi, (h, 2))
        self.pulse_phase = rng.uniform(0.0, 2.0 * np.pi, h)
        self._update()

    def _update(self) -> None:
        self.pos = self.anchor + self.amp * np.sin(self.phase)
        self.radius = self.base_radius * (
            1.0 + self.cfg.hazard_pulse * np.sin(self.pulse_phase)
        )

    def step(self, dt: float) -> None:
        k = self.cfg.hazard_speed * dt
        self.phase += self.omega * k
        self.pulse_phase += self.pulse_omega * k
        self._update()

    def _edge_distance(self, points: np.ndarray) -> np.ndarray:
        """(N, H): distanza dal bordo di ogni zona; negativa = dentro."""
        d = np.linalg.norm(points[:, None, :] - self.pos[None, :, :], axis=2)
        return d - self.radius[None, :]

    def clearance(self, points: np.ndarray) -> np.ndarray:
        """Distanza (N,) dal bordo della zona più vicina; negativa = dentro."""
        if len(self.pos) == 0:
            return np.full(len(points), np.inf)
        return self._edge_distance(points).min(axis=1)

    def intensity(self, points: np.ndarray) -> np.ndarray:
        """Tossicità (N,): 0 fuori e sul bordo, 1 al centro di una zona.
        Le zone sovrapposte si sommano."""
        if len(self.pos) == 0:
            return np.zeros(len(points))
        depth = np.clip(-self._edge_distance(points) / self.radius[None, :], 0.0, 1.0)
        return (depth ** self.cfg.hazard_falloff).sum(axis=1)
