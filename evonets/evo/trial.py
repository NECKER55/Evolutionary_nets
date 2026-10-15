import time

import numpy as np

from ..net.brain import Genome
from ..net.controllers import NeuralController
from ..sim.world import World

def _task_args(task) -> dict:
    """Gli argomenti con cui ricreare un compito dello stesso tipo."""
    return {k: getattr(task, k) for k in ("min_dist", "speed") if hasattr(task, k)}


# Semi delle arene di prova: lontani da quelli dell'evoluzione e sempre gli
# stessi, così reti diverse si confrontano sulle stesse prove.
TRIAL_SEED = 1_000_000


class Trial:
    """Mette alla prova una o più reti (`genomes`, una per riga): `arenas`
    arene, in ciascuna `clones` copie identiche di ogni rete partono da punti
    diversi. Misura quante volte ciascuna completa il compito e in quanto tempo.

    Si avanza a pezzi (`run(budget)`), così l'interfaccia resta reattiva.
    """

    def __init__(self, cfg, genomes: Genome, arenas: int = 20, clones: int = 50, seed: int = TRIAL_SEED,
                 task=None):
        self.cfg, self.arenas, self.clones, self.seed = cfg, arenas, clones, seed
        self.k = len(genomes)
        self.owner = np.repeat(np.arange(self.k), clones)  # di quale rete è ogni copia
        n = len(self.owner)
        self.world = World(cfg, n, seed, task)
        self.ctrl = NeuralController(cfg, n)
        self.ctrl.brains.set_genome(genomes.take(self.owner))
        self.ctrl.reset(seed)
        self.done_at = np.full(n, np.nan)  # quando ciascuna copia ha completato
        self.completed: list[int] = []  # per arena: copie che hanno completato (tutte le reti)
        self.wins = np.zeros(self.k, dtype=int)  # per rete: copie che hanno completato, in tutto
        self.times: list[float] = []  # tempi di completamento, di tutte le arene

    @property
    def finished(self) -> bool:
        return len(self.completed) >= self.arenas

    def step(self) -> None:
        w = self.world
        before = w.done.copy()
        w.step(self.ctrl.act(w), self.ctrl.upkeep)
        self.done_at[w.done & ~before] = w.t
        if (w.alive & ~w.done).any() and w.t < self.cfg.life_max:
            return
        self.completed.append(int(w.done.sum()))
        self.wins += np.bincount(self.owner[w.done], minlength=self.k)
        self.times += list(self.done_at[w.done])
        if not self.finished:
            w.seed = self.seed + len(self.completed)
            w.reset()
            self.ctrl.reset(w.seed)
            self.done_at[:] = np.nan

    def run(self, budget: float) -> None:
        """Avanza per circa `budget` secondi di calcolo. Con il motore GPU fa
        tutte le arene rimaste in un colpo solo, nello stesso lotto."""
        if not self.finished and self._run_on_gpu():
            return
        deadline = time.perf_counter() + budget
        while not self.finished and time.perf_counter() < deadline:
            self.step()

    def _run_on_gpu(self) -> bool:
        if self.cfg.engine != "gpu" or self.completed:
            return False
        from ..gpu import available
        if not available():
            return False
        from ..gpu.sim import GpuWorlds, supported
        if not supported(self.world, self.ctrl):
            return False
        genome = self.ctrl.brains.genome
        pairs = []
        for k in range(self.arenas):
            w = World(self.cfg, len(self.owner), self.seed + k, type(self.world.task)(**_task_args(self.world.task)))
            c = NeuralController(self.cfg, len(self.owner))
            c.brains.set_genome(genome)
            c.reset(self.seed + k)
            pairs.append((w, c))
        GpuWorlds(self.cfg, pairs, [self.seed + k for k in range(self.arenas)]).run()
        for w, _ in pairs:
            self.completed.append(int(w.done.sum()))
            self.wins += np.bincount(self.owner[w.done], minlength=self.k)
            self.times += list(w.done_t[w.done])
        return True

    @property
    def rate(self) -> float:
        """Quota di prove completate finora, da tutte le reti insieme."""
        return sum(self.completed) / max(len(self.completed) * self.clones * self.k, 1)

    def rates(self) -> np.ndarray:
        """Quota di prove completate finora, rete per rete."""
        return self.wins / max(len(self.completed) * self.clones, 1)
