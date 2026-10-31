"""L'attore-critico: l'agente di confronto che impara senza sezioni."""

import numpy as np

from ...arenas.pile.world import ENERGY
from .agent import Learner
from .genes import N_A, N_HEADS, Interface, value_of


class ActorCritic(Learner):
    """Il riferimento: attore-critico classico (TD, Doya 2000) che legge
    direttamente l'interfaccia A (con la fovea). Stessi programmi, stessa
    regola di E e V del cervello a sezioni, salienza solo innata; senza B, C,
    D, L e U, senza tracce strette e senza soglia."""

    name = "attore-critico"

    def __init__(self, n, dt, seed=0, genomes=None, objects: int = 1):
        super().__init__(n, dt, seed, genomes, objects)
        self.a = Interface(n)
        self._init_actor_critic(N_A + 1, N_A + 1)
        self._attach((("e0", (N_HEADS, N_A + 1), np.zeros((n, N_HEADS, N_A + 1))),
                      ("v0", (1, N_A + 1), np.zeros((n, 1, N_A + 1)))))
        self._born(slice(None))

    def _born(self, who) -> None:
        """I pesi di partenza, dal genoma."""
        self.w_e[who], self.w_v[who] = self.start("e0", who), self.start("v0", who)

    def step(self, raw, reward, objs):
        p = self.p
        decide = self._tick()
        self.att.update(objs, decide, p)
        a = self.a(raw, self.att.fovea(objs), np.exp(-self.dt / p["tau_a"]))
        f = np.concatenate([a, np.ones((self.n, 1))], 1)
        r = value_of(reward, raw[:, ENERGY], p) + self.innate_reward(raw)
        delta = self._td(f, r, p["horizon"][:, None], np.repeat(p["eta_v"][:, None], f.shape[1], 1))
        self.teach = delta
        logits = (self.w_e * f[:, None, :]).sum(-1)
        return self._act(logits, f, delta, p["eta_e"], p["temperature"], decide)

    def reborn(self, who, genomes=None):
        super().reborn(who, genomes)
        self.a.reset(who)
        self._reset_actor_critic(who)
        self._born(who)
