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
        self.noise_rng = np.random.default_rng([seed, 7])  # solo per il rumore dei sensi

    def reset(self) -> None:
        self.brains.reset_state()

    @property
    def upkeep(self) -> np.ndarray:
        """(N,) energia/s che ogni rete costa per il suo cablaggio."""
        return self.senses.cfg.wire_cost * self.brains.wire_length

    def act(self, world) -> np.ndarray:
        """Comandi di tutti gli agenti. Si calcolano solo per chi è ancora in
        gara: per i morti e per chi ha completato la fisica ignora i comandi,
        quindi danno 0 e la loro rete resta ferma. Il risultato della corsa non
        cambia, il calcolo si alleggerisce man mano che la gara si sfoltisce."""
        racing = world.alive & ~world.done
        if racing.all():
            return self.brains.step(self._noisy(self.senses.observe(world)))
        actions = np.zeros((world.n, 2))
        rows = np.flatnonzero(racing)
        if len(rows):
            actions[rows] = self.brains.step_some(self._noisy(self.senses.observe(world, rows)), rows)
        return actions

    def _noisy(self, obs: np.ndarray) -> np.ndarray:
        sd = self.senses.cfg.sensor_noise
        if sd <= 0.0:
            return obs
        return np.clip(obs + sd * self.noise_rng.standard_normal(obs.shape), -1.0, 1.0)
