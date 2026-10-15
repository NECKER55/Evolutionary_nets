import numpy as np

from .brain import Brains, SpikingBrains

# tipo di rete -> (classe, numero di neuroni motori)
NET_TYPES = {"rate": (Brains, 2), "spike-ei": (SpikingBrains, 4)}


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
    """Una rete per agente: recettori -> rete -> (spinta, sterzata).
    Il tipo di rete è scelto da `cfg.net_type`; i recettori stanno nel genoma.

    I recettori sono rumorosi: ogni lettura consuma numeri casuali da un
    generatore proprio (`noise_rng`). Riparte dal seme dell'arena a ogni
    `reset(seed)`, così la corsa resta ripetibile anche ripresa da un
    salvataggio. L'ultima lettura resta in `obs`: chi disegna la usa invece di
    rileggere i sensi, che cambierebbe la sequenza casuale e quindi la corsa.
    """

    def __init__(self, cfg, n: int, seed: int = 0):
        if cfg.net_type not in NET_TYPES:
            raise ValueError(f"net_type sconosciuto: {cfg.net_type!r} (validi: {', '.join(NET_TYPES)})")
        self.cfg = cfg
        kind, n_motors = NET_TYPES[cfg.net_type]
        self.brains = kind(cfg, n, n_motors, seed)
        self.noise_rng = np.random.default_rng([seed, 7])  # solo per il rumore dei sensi
        self.obs = np.zeros((n, self.brains.n_in))
        self.adapt = np.full((n, self.brains.n_in), np.nan)  # media recente dei quanti di ogni recettore

    def reset(self, seed: int | None = None) -> None:
        """Arena nuova: stato delle reti azzerato e, con `seed`, rumore dei
        sensi che riparte da quel seme."""
        self.brains.reset_state()
        self.obs = np.zeros((self.brains.n, self.brains.n_in))
        self.adapt = np.full((self.brains.n, self.brains.n_in), np.nan)
        if seed is not None:
            self.noise_rng = np.random.default_rng([seed, 7])

    @property
    def upkeep(self) -> np.ndarray:
        """(N,) energia/s che ogni individuo spende per il suo corpo e il suo
        cervello: cablaggio, neuroni, recettori."""
        return self.brains.upkeep

    def sense(self, world, rows=None) -> np.ndarray:
        """Le risposte dei recettori (con rumore) degli agenti `rows`."""
        prepared = self.brains.prepared(rows)
        if self.adapt.shape != (world.n, self.brains.n_in):  # popolazione o recettori cambiati
            self.adapt = np.full((world.n, self.brains.n_in), np.nan)
        if rows is None:
            obs, self.adapt = prepared.respond(world, np.arange(world.n), self.noise_rng, self.adapt)
        else:
            obs, self.adapt[rows] = prepared.respond(world, rows, self.noise_rng, self.adapt[rows])
        return self._noisy(obs)

    def act(self, world) -> np.ndarray:
        """Comandi di tutti gli agenti. Si calcolano solo per chi è ancora in
        gara: per i morti e per chi ha completato la fisica ignora i comandi,
        quindi danno 0 e la loro rete resta ferma. Il risultato della corsa non
        cambia, il calcolo si alleggerisce man mano che la gara si sfoltisce."""
        racing = world.alive & ~world.done
        if self.obs.shape != (world.n, self.brains.n_in):
            self.obs = np.zeros((world.n, self.brains.n_in))
        if racing.all():
            self.obs = self.sense(world)
            return self.brains.step(self.obs)
        actions = np.zeros((world.n, 2))
        rows = np.flatnonzero(racing)
        if len(rows):
            obs = self.sense(world, rows)
            self.obs[rows] = obs
            actions[rows] = self.brains.step_some(obs, rows)
        return actions

    def _noisy(self, obs: np.ndarray) -> np.ndarray:
        sd = self.cfg.sensor_noise
        if sd <= 0.0:
            return obs
        return np.clip(obs + sd * self.noise_rng.standard_normal(obs.shape), -1.0, 1.0)
