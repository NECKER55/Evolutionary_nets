"""L'agente guidato a mano: controlla il mondo, non impara."""

import numpy as np

from ...arenas.pile.world import N_PROGRAMS
from .agent import Agent
from .genes import APPROACH, EXPLORE


class Guided(Agent):
    """Guidato a mano, per controllare il mondo: sa quali pile sono buone
    (`objs["good"]`). Si concentra sulla pila buona più
    vicina e ci va; se vede solo pile cattive le lascia ed esplora (chi lo usa
    lo fa andare verso la zona, che sa)."""

    name = "guidato"

    def __init__(self, n, dt, seed=0, genomes=None, objects: int = 1, batteries: int = 0):
        super().__init__(n, dt, seed, genomes, objects)
        self._attach(())
        self.batteries = batteries

    def step(self, raw, reward, objs):
        decide = self._tick()
        self.att.update(objs, decide, self.p, extra=np.where(objs["good"], 10.0, -10.0))
        has, _, near, _ = self.att.seen(objs)
        ok = has & objs["good"][np.arange(self.n), np.maximum(self.att.target, 0)]
        self.program = np.where(ok, APPROACH, EXPLORE)
        self.bite = ok & (near > 0.9)  # morde solo la buona, quando ce l'ha davanti
        self.p_bite = self.bite.astype(float)
        self.att.leave(has & ~ok, self.p)  # non è la buona: la lascia
        self.pi = np.eye(N_PROGRAMS)[self.program]
        return self.program
