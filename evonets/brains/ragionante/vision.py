"""La sezione P, la percezione: dalla telecamera agli oggetti.

`Percept` è l'interfaccia verso il resto del cervello, la stessa per il detector finto
(`FakeDetector`, legge il simulatore) e per quello vero (`ready.ReadyDetector`: FastSAM,
profondità, DINOv2): una lista di cose, ognuna con dove sta e com'è fatta. Nessuno dei
due sa quali pile siano buone: quello si impara, non si vede.

Qui c'è anche la geometria comune: da una maschera e dalla profondità ad angolo,
distanza e larghezza (`locate`).
"""

from dataclasses import dataclass

import numpy as np

from ...arenas.mondo3d.world import Mondo3D
from ...arenas.pile.world import RED, SECTORS

KINDS = Mondo3D.KINDS
MAX_OBJECTS = 32  # cose al massimo in una vista: con lo sfondo, FastSAM ne ha trovate fino a 27
EMBED = len(KINDS) + 4  # che cos'è, grandezza, vicinanza, seno e coseno dell'angolo
VECTOR = 64  # il vettore "com'è fatto" delle reti pronte, come `bank.SIZE`
FEATURES = VECTOR + 5  # più larghezza, seno e coseno dell'angolo, vicinanza, "c'è"
SPREAD = np.tan(SECTORS[-1])  # la telecamera: a distanza 1, metà larghezza dell'immagine copre questa larghezza


@dataclass
class Percept:
    """Tutto (B, n, M), con M = `MAX_OBJECTS`. `kind`: indice in `KINDS`, -1 se non somiglia
    a nessuno; `ang`: rad rispetto al muso, positivo a sinistra; `dist`: cm dal centro
    dell'essere; `size`: larghezza in cm; `box`: (..., 4), sx, su, dx, giù tra 0 e 1, se
    c'è; `vector`: (..., `VECTOR`) com'è fatto, solo con le reti pronte (altrimenti zeri)."""

    on: np.ndarray
    kind: np.ndarray
    ang: np.ndarray
    dist: np.ndarray
    size: np.ndarray
    box: np.ndarray
    vector: np.ndarray

    @classmethod
    def empty(cls, B: int, n: int) -> "Percept":
        s = (B, n, MAX_OBJECTS)
        return cls(np.zeros(s, dtype=bool), np.zeros(s, dtype=int), np.zeros(s), np.zeros(s), np.zeros(s),
                   np.zeros(s + (4,)), np.zeros(s + (VECTOR,), dtype=np.float32))

    def embedding(self, eye_range: float) -> np.ndarray:
        """(B, n, M, `EMBED`): i 7 numeri con il nome, quelli con cui sono addestrate L e W."""
        kind = np.eye(len(KINDS) + 1)[self.kind][..., :len(KINDS)]  # -1, nessun nome: tutti zeri
        rest = np.stack([self.size / 20.0, np.clip(1.0 - self.dist / eye_range, 0.0, 1.0),
                         np.sin(self.ang), np.cos(self.ang)], -1)
        return np.concatenate([kind, rest], -1) * self.on[..., None]

    def features(self, eye_range: float) -> np.ndarray:
        """(B, n, M, `FEATURES`): ciò che esce da P con le reti pronte, senza nomi. Vettore, larghezza
        in m, seno e coseno dell'angolo, vicinanza, "c'è"; un posto vuoto è tutto zeri."""
        rest = np.stack([self.size / 100.0, np.sin(self.ang), np.cos(self.ang),
                         np.clip(1.0 - self.dist / eye_range, 0.0, 1.0), np.ones(self.on.shape)], -1)
        return np.concatenate([self.vector, rest], -1) * self.on[..., None]


class FakeDetector:
    """La verità del simulatore, con la stessa interfaccia: solo ciò che è
    inquadrato e in portata. Non vede gli oggetti nascosti dietro altri."""

    def __init__(self, world):
        self.w = world

    def see(self) -> Percept:
        w, cfg = self.w, self.w.cfg
        o, out = w.objects(), Percept.empty(w.B, w.n)
        K = cfg.batteries
        radius = cfg.battery_radius * np.where(w.bbig, 1.0, cfg.pile_small)
        for b, i in np.ndindex(w.B, w.n):
            seen = np.flatnonzero(o["vis"][b, i])
            seen = seen[np.argsort(-o["near"][b, i, seen])][:MAX_OBJECTS]
            m = len(seen)
            pile = seen < K
            colour = w.bcolor[b, seen.clip(max=K - 1)]
            out.on[b, i, :m] = True
            out.kind[b, i, :m] = np.where(pile, np.where(colour == RED, KINDS.index("pila rossa"), KINDS.index("pila gialla")),
                                          KINDS.index("essere"))
            out.ang[b, i, :m] = o["ang"][b, i, seen]
            out.dist[b, i, :m] = (1.0 - o["near"][b, i, seen]) * cfg.eye_range
            out.size[b, i, :m] = np.where(pile, 2.0 * radius[b, seen.clip(max=K - 1)], 2.0 * cfg.radius)
        return out


def box_of(mask: np.ndarray) -> tuple:
    """(sx, su, dx, giù) in pixel."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1


def locate(mask, depth, cfg) -> tuple:
    """Dalla maschera e dalla profondità a (angolo, distanza dal centro dell'essere, larghezza).
    I pixel sono quadrati: anche in verticale un pixel vale quanto in orizzontale."""
    side = mask.shape[1] / 2.0
    left, _, right, _ = box_of(mask)
    z = float(np.median(depth[mask]))
    wide = (right - left) * z * SPREAD / side
    across = ((left + right) / 2.0 - side) / side * z * SPREAD
    forward = z + cfg.camera_ahead
    return -np.arctan2(across, forward), np.hypot(across, forward) + wide / 2.0, wide
