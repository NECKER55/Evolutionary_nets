"""Il perfetto sceglie, e gli occhi vedono accanto: la percezione lavora e si mostra sullo
schermo, ma nessuna decisione la usa. Serve a provare la percezione con la scelta ferma."""

import numpy as np

from ..mondo3d.perfect import Perfect


class Looking:
    """Il comandante perfetto con sopra gli occhi. `seen` è ciò che gli occhi vedono,
    solo per lo schermo: il perfetto non lo legge."""

    def __init__(self, world, eyes, seed: int = 0):
        self.w, self.eyes = world, eyes
        self.perfect = Perfect(world, seed)
        self.seen = None  # l'ultimo sguardo, come lo vede il vedente

    def _goal(self) -> np.ndarray:
        """Il colore buono di ogni essere: solo per lo schermo."""
        return np.broadcast_to(self.w.good[:, None], (self.w.B, self.w.n))

    def act(self):
        self.seen = self.eyes.see()
        return self.perfect.act()
