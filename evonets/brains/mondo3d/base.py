"""L'agente base: vive solo di ciò che vede la sua telecamera."""

import numpy as np

SAFE = 45.0  # cm: sotto questa profondità c'è qualcosa davanti
BITE = 10.0  # cm dalla telecamera: una pila è a portata di morso, qualunque cosa sia


class Base:
    """Vaga, non sa che cosa sia una pila né quale sia buona: legge solo la
    profondità, frena davanti agli ostacoli, morde ciò che ha davanti e se ne va."""

    def __init__(self, world, seed: int = 0):
        self.w = world
        self.rng = np.random.default_rng([seed, 5])
        shape = (world.B, world.n)
        self.drift = np.zeros(shape)
        self.away = np.zeros(shape)  # s ancora da passare a girarsi via
        self.side = np.ones(shape)

    def act(self):
        w = self.w
        shape = (w.B, w.n)
        self.drift = 0.9 * self.drift + 0.15 * self.rng.normal(0.0, 1.0, shape)
        thrust, steer, bite = np.full(shape, 0.7), self.drift.copy(), np.zeros(shape, dtype=bool)
        self.away = np.maximum(self.away - w.cfg.dt, 0.0)
        for b, i in np.ndindex(shape):
            if self.away[b, i] > 0.0:
                thrust[b, i], steer[b, i] = 0.0, self.side[b, i]
                continue
            d = w.view(b, i, rgb=False)["depth"]
            third, mid, unit = d.shape[1] // 3, d.shape[0] // 2, d.shape[0] // 64
            band = d[mid - unit:mid + 3 * unit]  # all'altezza dello sguardo: niente pavimento
            left, front, right = band[:, :third].min(), band[:, third:-third].min(), band[:, -third:].min()
            if front < BITE:
                self.away[b, i] = float(self.rng.uniform(0.5, 1.5))
                self.side[b, i] = 1.0 if left > right else -1.0
                thrust[b, i], bite[b, i] = 0.0, True
            elif front < SAFE:
                thrust[b, i] = 0.1 + 0.3 * front / SAFE
        return thrust, steer, bite
