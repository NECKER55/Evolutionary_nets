"""Il cervello ragionante, per ora: il motore (sezione 7) con sopra un
comandante finto, che tiene il posto delle sezioni 1-6 che ancora mancano."""

import numpy as np

from ...arenas.pile.world import BUMP
from ..mondo3d.perfect import Perfect
from .motor import BIRTH, RETUNE, Motor

MUTE = 1.0  # s in cui si vuole mordere e la bocca non si apre: il corpo non obbedisce più


class Reasoning:
    """Il comandante finto legge la verità come il perfetto e dice che cosa
    vorrebbe fare; è il motore a doverlo ottenere da un corpo che non conosce."""

    def __init__(self, world, seed: int = 0, boss=None):
        self.w = world
        self.boss = boss or Perfect(world, seed)  # chi dà le intenzioni: il motore non sa chi è
        self.motor = Motor((world.B, world.n), world.wiring.shape[1], world.cfg.dt, seed)
        self.mute = np.zeros((world.B, world.n))  # s di morsi voluti e non riusciti

    def _mute(self, bite) -> np.ndarray:
        """Chi da `MUTE` secondi vuole mordere e non apre la bocca."""
        opened = self.w.felt()[..., 2] >= 0.5
        self.mute = np.where(opened | self.motor.tuning, 0.0, self.mute + bite * self.w.cfg.dt)
        return self.mute > MUTE

    def act(self):
        w, motor = self.w, self.motor
        if w.wiring.shape[1] != motor.K:
            motor.resize(w.wiring.shape[1])
        thrust, steer, bite = self.boss.act()
        lost = motor.lost | self._mute(bite)
        if lost.any():
            motor.tune(RETUNE, lost)
            self.mute[lost] = 0.0
        newborn = w.age <= 0.0
        if newborn.any() and w.t > 0.0:
            motor.tune(BIRTH, newborn)
        intent = np.stack([thrust, steer, bite.astype(float)], -1)
        held = w.senses()[..., BUMP].any(-1) | (w.charging > 0.0)
        return w.wires(motor.step(intent, w.felt(), held))
