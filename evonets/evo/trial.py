import time

import numpy as np

from ..net.brain import Genome
from ..net.controllers import NeuralController
from ..sim.world import World

# Semi delle arene di prova: lontani da quelli dell'evoluzione e sempre gli
# stessi, così reti diverse si confrontano sulle stesse prove.
TRIAL_SEED = 1_000_000


class Trial:
    """Mette alla prova una sola rete: `arenas` arene, in ciascuna `clones`
    copie identiche della rete partono da punti diversi. Misura quante volte
    completa il compito e in quanto tempo.

    Si avanza a pezzi (`run(budget)`), così l'interfaccia resta reattiva.
    """

    def __init__(self, cfg, genome: Genome, arenas: int = 20, clones: int = 50, seed: int = TRIAL_SEED):
        self.cfg, self.arenas, self.clones, self.seed = cfg, arenas, clones, seed
        self.world = World(cfg, clones, seed)
        self.ctrl = NeuralController(cfg, clones)
        self.ctrl.brains.set_genome(genome.take([0] * clones))
        self.done_at = np.full(clones, np.nan)  # quando ciascuna copia ha completato
        self.completed: list[int] = []  # per arena: copie che hanno completato
        self.times: list[float] = []  # tempi di completamento, di tutte le arene

    @property
    def finished(self) -> bool:
        return len(self.completed) >= self.arenas

    def step(self) -> None:
        w = self.world
        before = w.done.copy()
        w.step(self.ctrl.act(w))
        self.done_at[w.done & ~before] = w.t
        if (w.alive & ~w.done).any() and w.t < self.cfg.life_max:
            return
        self.completed.append(int(w.done.sum()))
        self.times += list(self.done_at[w.done])
        if not self.finished:
            w.seed = self.seed + len(self.completed)
            w.reset()
            self.ctrl.reset()
            self.done_at[:] = np.nan

    def run(self, budget: float) -> None:
        """Avanza per circa `budget` secondi di calcolo."""
        deadline = time.perf_counter() + budget
        while not self.finished and time.perf_counter() < deadline:
            self.step()

    @property
    def rate(self) -> float:
        """Quota di prove completate finora."""
        return sum(self.completed) / max(len(self.completed) * self.clones, 1)
