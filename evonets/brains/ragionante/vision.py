"""La sezione 1, la percezione: dalla telecamera agli oggetti.

`Percept` è l'interfaccia verso il resto del cervello, ed è la stessa per il
detector finto (legge il simulatore) e per quello vero (`Detector`): una lista di
oggetti, ognuno con che cos'è, dov'è e un embedding. Nessuno dei due sa quali pile
siano buone: quello si impara, non si vede.

Il detector vero è una piccola rete convoluzionale, addestrata sulle etichette del
simulatore (`train_vision.py`), che dice che cosa c'è in ogni pixel. Le macchie di
uno stesso tipo sono gli oggetti, e la profondità dice quanto sono lontani.
"""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage

from ...arenas.mondo3d.world import Mondo3D
from ...arenas.pile.world import RED, SECTORS

KINDS = Mondo3D.KINDS
SIDE = 132  # lato dell'immagine che legge la rete: la telecamera, rimpicciolita
MAX_OBJECTS = 12  # oggetti al massimo in una vista
EMBED = len(KINDS) + 4  # che cos'è, grandezza, vicinanza, seno e coseno dell'angolo
FILTERS = (12, 16, 16)  # neuroni per strato; il secondo e il terzo guardano più largo
MIN_PIXELS = 3  # una macchia più piccola è rumore
MIN_FILL = 0.3  # una macchia che riempie meno del suo riquadro è un filo lungo un bordo, non un oggetto
MAX_THIN = 5.0  # e così una macchia molto più alta che larga
WEIGHTS = Path(__file__).parent / "vision.npz"
SPREAD = np.tan(SECTORS[-1])  # la telecamera: a distanza 1, metà immagine copre questa larghezza


@dataclass
class Percept:
    """Tutto (B, n, M), con M = `MAX_OBJECTS`. `kind`: indice in `KINDS`; `ang`: rad
    rispetto al muso, positivo a sinistra; `dist`: cm dal centro dell'essere;
    `size`: larghezza in cm; `box`: (..., 4), sx, su, dx, giù tra 0 e 1, se c'è."""

    on: np.ndarray
    kind: np.ndarray
    ang: np.ndarray
    dist: np.ndarray
    size: np.ndarray
    box: np.ndarray

    @classmethod
    def empty(cls, B: int, n: int) -> "Percept":
        s = (B, n, MAX_OBJECTS)
        return cls(np.zeros(s, dtype=bool), np.zeros(s, dtype=int), np.zeros(s), np.zeros(s), np.zeros(s),
                   np.zeros(s + (4,)))

    def embedding(self, eye_range: float) -> np.ndarray:
        """(B, n, M, `EMBED`): ciò che riceveranno le sezioni dopo."""
        kind = np.eye(len(KINDS))[self.kind]
        rest = np.stack([self.size / 20.0, np.clip(1.0 - self.dist / eye_range, 0.0, 1.0),
                         np.sin(self.ang), np.cos(self.ang)], -1)
        return np.concatenate([kind, rest], -1) * self.on[..., None]


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


def shrink(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Dalla telecamera a ciò che legge la rete: (`SIDE`, `SIDE`, 4), colori e
    logaritmo della distanza (i colpi d'occhio lontani non si perdono in cm)."""
    f = rgb.shape[0] // SIDE
    h = f * SIDE
    small = rgb[:h, :h].reshape(SIDE, f, SIDE, f, 3).mean((1, 3)) / 255.0
    far = np.log(np.clip(depth[:h, :h], 3.0, 1000.0)).reshape(SIDE, f, SIDE, f).mean((1, 3)) / np.log(1000.0)
    return np.concatenate([small, far[..., None]], -1).astype(np.float32)


def network(key, inputs: int = 4) -> list:
    """I pesi: due file per strato, kernel 3×3 e bias, più la lettura finale."""
    params, wide = [], inputs
    for out in FILTERS:
        key, k = jax.random.split(key)
        params.append([jax.nn.initializers.he_normal()(k, (3, 3, wide, out)), jnp.zeros(out)])
        wide = out
    key, k = jax.random.split(key)
    params.append([0.1 * jax.random.normal(k, (1, 1, wide, len(KINDS) + 1)), jnp.zeros(len(KINDS) + 1)])
    return params


def logits(params, x):
    """`x` (N, H, W, 4) -> (N, H, W, classi): 0 lo sfondo, poi `KINDS`."""
    for layer, ((kernel, bias), dilation) in enumerate(zip(params, (1, 2, 3, 1))):
        pad = "SAME"
        x = jax.lax.conv_general_dilated(x, kernel, (1, 1), pad, rhs_dilation=(dilation, dilation),
                                         dimension_numbers=("NHWC", "HWIO", "NHWC")) + bias
        if layer < len(params) - 1:
            x = jax.nn.relu(x)
    return x


def blobs(classes: np.ndarray, depth: np.ndarray, cfg) -> list:
    """Dalle classi per pixel agli oggetti: una macchia di un tipo è un oggetto.
    `depth` è della stessa misura. -> [(tipo, ang, dist, size, box)], dal più vicino."""
    side = classes.shape[0]
    scale = side / 2.0
    found = []
    for k in range(len(KINDS)):
        marks, _ = ndimage.label(classes == k + 1)
        for label, place in enumerate(ndimage.find_objects(marks), 1):
            rows, cols = place
            pixels = marks[place] == label
            high, broad = pixels.shape
            if pixels.sum() < MIN_PIXELS or pixels.mean() < MIN_FILL or high > MAX_THIN * broad:
                continue
            z = float(np.median(depth[place][pixels]))
            wide = (cols.stop - cols.start) * z * SPREAD / scale
            side_x = ((cols.start + cols.stop) / 2.0 - scale) / scale * z * SPREAD
            forward = z + cfg.camera_ahead
            found.append((k, -np.arctan2(side_x, forward), np.hypot(side_x, forward) + wide / 2.0, wide,
                          (cols.start / side, rows.start / side, cols.stop / side, rows.stop / side)))
    found.sort(key=lambda o: o[2])
    return found[:MAX_OBJECTS]


class Detector:
    """Il detector vero: legge solo immagine e profondità della telecamera."""

    def __init__(self, world, weights=WEIGHTS):
        self.w = world
        data = np.load(weights)
        self.params = [[jnp.asarray(data[f"k{i}"]), jnp.asarray(data[f"b{i}"])] for i in range(len(data.files) // 2)]
        self.read = jax.jit(lambda p, x: jnp.argmax(logits(p, x), -1))

    def see(self) -> Percept:
        w = self.w
        out = Percept.empty(w.B, w.n)
        views = [w.view(b, i) for b, i in np.ndindex(w.B, w.n)]
        inputs = np.stack([shrink(v["rgb"], v["depth"]) for v in views])
        classes = np.asarray(self.read(self.params, inputs))
        for (b, i), v, c in zip(np.ndindex(w.B, w.n), views, classes):
            f = v["depth"].shape[0] // SIDE
            for m, (k, ang, dist, wide, box) in enumerate(blobs(c, v["depth"][f // 2::f, f // 2::f][:SIDE, :SIDE], w.cfg)):
                out.on[b, i, m] = True
                out.kind[b, i, m], out.ang[b, i, m], out.dist[b, i, m], out.size[b, i, m] = k, ang, dist, wide
                out.box[b, i, m] = box
        return out
