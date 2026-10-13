import numpy as np

from .brain import Brains, SpikingBrains

# tipo di rete -> (classe, numero di neuroni motori)
NET_TYPES = {"rate": (Brains, 2), "spike-ei": (SpikingBrains, 4)}
from .sensors import Senses


class RandomWalkController:
    """Controllore mock: comandi casuali ma lisci (processo di Ornstein-Uhlenbeck).

    Serve solo a validare la fisica in batch; dalla Milestone 2 verrà affiancato
    dalle reti, che espongono la stessa interfaccia `act(world) -> (N, 2)`.
    """

    def __init__(self, n: int, seed: int = 0, theta: float = 1.5, sigma: float = 1.2):
        self.rng = np.random.default_rng(seed)
        self.theta = theta
        self.sigma = sigma
        self.mean = np.array([0.6, 0.0])  # tende ad avanzare, sterzata neutra
        self.state = np.tile(self.mean, (n, 1))

    def act(self, world) -> np.ndarray:
        dt = world.cfg.dt
        noise = self.rng.standard_normal(self.state.shape)
        self.state += self.theta * (self.mean - self.state) * dt
        self.state += self.sigma * np.sqrt(dt) * noise
        np.clip(self.state, -1.0, 1.0, out=self.state)
        return self.state


class NeuralController:
    """Una rete per agente: sensi -> rete -> (spinta, sterzata).
    Il tipo di rete è scelto da `cfg.net_type`."""

    def __init__(self, cfg, n: int, seed: int = 0):
        self.senses = Senses(cfg)
        if cfg.net_type not in NET_TYPES:
            raise ValueError(f"net_type sconosciuto: {cfg.net_type!r} (validi: {', '.join(NET_TYPES)})")
        kind, n_motors = NET_TYPES[cfg.net_type]
        self.brains = kind(cfg, n, self.senses.size, n_motors, seed)

    def reset(self) -> None:
        self.brains.reset_state()

    def act(self, world) -> np.ndarray:
        return self.brains.step(self.senses.observe(world))
