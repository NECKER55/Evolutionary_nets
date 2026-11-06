"""La sezione P con reti già pronte, da confrontare con la nostra (`vision.py`):

- **SlimSAM** (nielsr/slimsam-77-uniform, Apache 2.0) separa ogni cosa che vede,
  senza sapere che cosa sia e senza etichette;
- **DINOv2 small** (facebook/dinov2-small, Apache 2.0) descrive ogni ritaglio con un
  vettore, a cui si aggiunge il colore;
- `ObjectMemory` conserva i vettori, anche su disco: un vettore vicino a uno noto
  è un oggetto già visto, altrimenti è nuovo e si salva.

Tra un'immagine e l'altra l'oggetto si segue col `Tracker`: se cambia aspetto sotto
gli occhi resta lo stesso, e il suo aspetto nuovo si aggiunge alla stessa voce.

Non sa che cosa sia una pila: per dire "pila gialla" al comandante bastano pochi
esempi (`Prototypes`), non migliaia di immagini. Pesa troppo per ogni passo:
guarda ogni `PERIOD` secondi e nel frattempo tiene ciò che sa, aggiornato col
movimento sentito. Richiede `uv run --group percezione`.
"""

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from transformers import AutoImageProcessor, AutoModel, SamModel, SamProcessor

from .vision import KINDS, MAX_OBJECTS, SPREAD, Percept

SEGMENTER = "nielsr/slimsam-77-uniform"
DESCRIBER = "facebook/dinov2-small"
EMBED = 384 + 3  # i vettori di DINOv2 small, più il colore
HUE = 1.5  # quanto pesa il colore nel vettore: da solo, DINOv2 sui nostri ritagli vede quasi solo la forma
PALE = 0.04  # sotto questa saturazione un oggetto è grigio: il colore non dice niente
FLOOR = 1.5  # cm: più in alto di così non è pavimento
TALL = 20.0  # cm: ciò che sale fin qui è un muro
SAME_WALL = 0.06  # quota di distanza entro cui un pixel basso appartiene allo stesso muro
STEP = 6.0  # cm: un salto di profondità tra due pixel vicini separa due oggetti
MIN_SPOT = 8  # pixel di una macchia di profondità perché meriti una domanda
MIN_SCORE = 0.7  # sotto questa qualità dichiarata da SlimSAM, la maschera si butta
MAX_AREA = 0.2  # più grande di così è pavimento, muro o cielo: sfondo
MIN_AREA = 6  # pixel: più piccola è rumore
OVERLAP = 0.6  # maschere che si coprono così sono lo stesso oggetto
SAME = 0.8  # coseno da cui due vettori sono lo stesso oggetto
VIEWS = 16  # viste ricordate per oggetto: si impara a riconoscerlo da più lati
PERIOD = 0.5  # s tra uno sguardo e l'altro
FOLLOW = 0.3  # sovrapposizione dei riquadri da cui, tra due immagini, è lo stesso oggetto
WINDOW = 40.0  # cm: il lato del ritaglio a scala fissa, più largo dell'oggetto più largo (un essere, 20 cm)
GREY = 128  # fuori dalla maschera: così lo sfondo non conta


class ObjectMemory:
    """Gli oggetti visti: per ognuno fino a `VIEWS` vettori. Sopravvive tra una
    vita e l'altra se la si salva: l'encoder non cambia, quindi i vettori restano validi."""

    def __init__(self, same: float = SAME):
        self.same = same
        self.owner = np.zeros(0, dtype=int)  # di chi è ogni vista
        self.views = np.zeros((0, EMBED), dtype=np.float32)
        self.kinds = 0  # oggetti conosciuti

    def closest(self, vector) -> tuple:
        """-> (oggetto, somiglianza); (-1, -1) se non si conosce niente."""
        if not len(self.views):
            return -1, -1.0
        score = self.views @ vector
        best = int(score.argmax())
        return int(self.owner[best]), float(score[best])

    def add(self, vector, who: int) -> None:
        """Una vista in più di `who`; se ne ha troppe, sostituisce quella più simile a questa."""
        mine = np.flatnonzero(self.owner == who)
        if len(mine) >= VIEWS:
            self.views[mine[(self.views[mine] @ vector).argmax()]] = vector
            return
        self.views = np.concatenate([self.views, vector[None]])
        self.owner = np.append(self.owner, who)

    def know(self, vector) -> tuple:
        """Il cuore: già visto o nuovo? -> (oggetto, se è nuovo)."""
        who, score = self.closest(vector)
        if score >= self.same:
            return who, False
        who, self.kinds = self.kinds, self.kinds + 1
        self.add(vector, who)
        return who, True

    def save(self, path) -> None:
        np.savez_compressed(path, owner=self.owner, views=self.views, kinds=self.kinds)

    @classmethod
    def load(cls, path, same: float = SAME) -> "ObjectMemory":
        data, out = np.load(path), cls(same)
        out.owner, out.views, out.kinds = data["owner"], data["views"], int(data["kinds"])
        return out


def proposals(depth: np.ndarray, cfg) -> np.ndarray:
    """Dove chiedere a SlimSAM. Da solo, con una griglia di punti, si ferma a pavimento e
    muri: gli oggetti sono piccoli e tutti sull'orizzonte. Qui è la profondità a dire che
    cosa sporge dal pavimento e non è un muro. -> (P, 2): riga e colonna, tra 0 e 1."""
    h, w = depth.shape
    up = (h / 2.0 - np.arange(h)[:, None] - 0.5) / (w / 2.0) * SPREAD  # pixel quadrati: si misura sulla larghezza
    height = cfg.camera_height + depth * up
    near = depth < cfg.eye_range
    tall = near & (height > TALL)
    wall = np.where(tall.any(0), np.ma.median(np.ma.array(depth, mask=~tall), axis=0).filled(np.inf), np.inf)
    on_wall = np.abs(depth - wall) < SAME_WALL * depth
    spots = near & (height > FLOOR) & (height <= TALL) & ~on_wall
    step = np.zeros((h, w), dtype=bool)
    step[:, 1:] = np.abs(np.diff(depth, axis=1)) > STEP
    marks, count = ndimage.label(spots & ~step)
    sizes = np.bincount(marks.ravel(), minlength=count + 1)
    rows, cols = np.indices((h, w))
    points = [(np.median(rows[marks == k]) / h, np.median(cols[marks == k]) / w) for k in range(1, count + 1)
              if sizes[k] >= MIN_SPOT]
    return np.array(points).reshape(-1, 2)


def hue(pixels: np.ndarray) -> np.ndarray:
    """Il colore di un oggetto, a prescindere da quanto è illuminato: (3,) di lunghezza 1, o zeri se è grigio."""
    mean = pixels.mean(0) / 255.0
    tint = mean / max(mean.sum(), 1e-6) - 1.0 / 3.0
    size = np.linalg.norm(tint)
    return tint / size if size > PALE else np.zeros(3)


class Eyes:
    """SlimSAM più DINOv2: da un'immagine a [(maschera, vettore)]."""

    def __init__(self, threads: int = 0):
        if threads:
            torch.set_num_threads(threads)
        self.sam = SamModel.from_pretrained(SEGMENTER).eval()
        self.sam_in = SamProcessor.from_pretrained(SEGMENTER)
        self.dino = AutoModel.from_pretrained(DESCRIBER).eval()
        self.dino_in = AutoImageProcessor.from_pretrained(DESCRIBER)

    @torch.no_grad()
    def masks(self, rgb: np.ndarray, depth: np.ndarray, cfg) -> list:
        """Ogni cosa che si stacca dallo sfondo, senza nomi: [(H, W) di bool], dalla migliore."""
        h, w = rgb.shape[:2]
        spots = proposals(depth, cfg)
        if not len(spots):
            return []
        x = self.sam_in(images=rgb, return_tensors="pt")
        codes = self.sam.get_image_embeddings(x["pixel_values"])
        high, wide = (float(v) for v in x["reshaped_input_sizes"][0])  # il lato lungo diventa 1024
        points = torch.tensor(spots[:, ::-1].copy() * (wide, high), dtype=torch.float32)[None, :, None]
        out = self.sam(image_embeddings=codes, input_points=points, multimask_output=False)
        # Le maschere coprono il quadrato da 1024 con i bordi aggiunti: si tolgono, poi si ingrandisce.
        full = self.sam_in.post_process_masks(out.pred_masks, x["original_sizes"], x["reshaped_input_sizes"])[0][:, 0]
        keep, score = full.numpy(), out.iou_scores[0, :, 0].numpy()
        kept: list = []
        for k in np.argsort(-score):
            m = keep[k]
            area = m.sum()
            if score[k] < MIN_SCORE or area < MIN_AREA or area > MAX_AREA * h * w:
                continue
            if all((m & o).sum() < OVERLAP * min(area, o.sum()) for o in kept):
                kept.append(m)
        return kept

    @torch.no_grad()
    def describe(self, rgb: np.ndarray, masks: list, depth: np.ndarray | None = None) -> np.ndarray:
        """Un vettore di lunghezza 1 per maschera; fuori dalla maschera è grigio, così lo sfondo non conta.
        Con `depth` il ritaglio è a scala fissa (`fixed`): la grandezza vera resta nell'immagine."""
        if not masks:
            return np.zeros((0, EMBED), dtype=np.float32)
        if depth is None:
            crops = [np.where(m[..., None], rgb, GREY)[slice(*rows), slice(*cols)].astype(np.uint8)
                     for m in masks for rows, cols in [span(m)]]
            sizes = {}
        else:
            crops = [fixed(rgb, m, depth) for m in masks]
            sizes = {"size": {"height": 224, "width": 224}, "do_center_crop": False}  # tutta la finestra, senza tagli
        x = self.dino_in(images=crops, input_data_format="channels_last", return_tensors="pt", **sizes)  # un ritaglio piccolo sembrerebbe avere i canali davanti
        v = self.dino(**x).last_hidden_state[:, 0].numpy()
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        v = np.concatenate([v, HUE * np.stack([hue(rgb[m]) for m in masks])], -1)
        return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


class Prototypes:
    """Il minimo per dare un nome: pochi vettori di esempio per ogni tipo in `KINDS`."""

    PATH = Path(__file__).parent / "prototypes.npz"

    def __init__(self, path=PATH):
        data = np.load(path)
        self.vectors, self.kinds = data["vectors"], data["kinds"]

    def name(self, vectors: np.ndarray) -> tuple:
        """-> (tipi, somiglianza)."""
        score = vectors @ self.vectors.T
        best = score.argmax(-1)
        return self.kinds[best], score.max(-1)


def span(mask: np.ndarray) -> tuple:
    """((su, giù), (sx, dx)) in pixel, giù e dx esclusi."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return (rows[0], rows[-1] + 1), (cols[0], cols[-1] + 1)


def fixed(rgb: np.ndarray, mask: np.ndarray, depth: np.ndarray, window: float = WINDOW) -> np.ndarray:
    """Un ritaglio quadrato di `window` cm veri attorno all'oggetto, qualunque sia la distanza:
    una pila piccola resta piccola nel ritaglio. La profondità dice quanti pixel fanno `window` cm."""
    z = float(np.median(depth[mask]))
    side = max(4, round(window * mask.shape[1] / 2.0 / SPREAD / z))
    (top, down), (left, right) = span(mask)
    y0, x0 = (top + down - side) // 2, (left + right - side) // 2
    out = np.full((side, side, 3), GREY, dtype=np.uint8)
    seen = np.where(mask[..., None], rgb, GREY)
    ya, xa = max(y0, 0), max(x0, 0)
    yb, xb = min(y0 + side, rgb.shape[0]), min(x0 + side, rgb.shape[1])
    out[ya - y0:yb - y0, xa - x0:xb - x0] = seen[ya:yb, xa:xb]
    return out


def box_of(mask: np.ndarray) -> tuple:
    """(sx, su, dx, giù) in pixel."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1


def overlap(a, b) -> float:
    wide = min(a[2], b[2]) - max(a[0], b[0])
    high = min(a[3], b[3]) - max(a[1], b[1])
    if wide <= 0 or high <= 0:
        return 0.0
    both = wide * high
    return both / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - both)


class Tracker:
    """Gli oggetti dell'immagine di prima: chi resta nello stesso posto è lo stesso."""

    def __init__(self):
        self.last = []  # (riquadro, oggetto)

    def follow(self, boxes: list) -> list:
        """-> per ogni riquadro, l'oggetto che era lì; -1 se non c'era nessuno."""
        out, free = [-1] * len(boxes), list(self.last)
        for k, box in enumerate(boxes):
            if free:
                score = [overlap(box, old[0]) for old in free]
                best = int(np.argmax(score))
                if score[best] >= FOLLOW:
                    out[k] = free.pop(best)[1]
        return out


def locate(mask, depth, cfg) -> tuple:
    """Dalla maschera a (angolo, distanza, larghezza), come in `vision.blobs`."""
    side = mask.shape[1] / 2.0
    left, _, right, _ = box_of(mask)
    z = float(np.median(depth[mask]))
    wide = (right - left) * z * SPREAD / side
    across = ((left + right) / 2.0 - side) / side * z * SPREAD
    forward = z + cfg.camera_ahead
    return -np.arctan2(across, forward), np.hypot(across, forward) + wide / 2.0, wide


class ReadyDetector:
    """La stessa interfaccia del detector nostro (`Percept`), più `ids`, `memory`
    e il nome dell'oggetto che si vede. Lo sguardo dura più di un passo del mondo,
    ma la simulazione lo aspetta: il robot non è penalizzato dal tempo di calcolo,
    solo dal fatto che guarda ogni `PERIOD` secondi."""

    def __init__(self, world, memory: ObjectMemory | None = None, eyes: Eyes | None = None):
        self.w = world
        self.eyes = eyes or Eyes()
        self.memory = memory or ObjectMemory()
        self.names = Prototypes()
        self.tracks = [[Tracker() for _ in range(world.n)] for _ in range(world.B)]
        self.every = max(1, round(PERIOD / world.cfg.dt))
        self.ticks = 0
        self.held = Percept.empty(world.B, world.n)
        self.ids = np.full((world.B, world.n, MAX_OBJECTS), -1)
        self.fresh = np.zeros((world.B, world.n, MAX_OBJECTS), dtype=bool)  # visti ora per la prima volta

    def _look(self, b: int, i: int) -> None:
        w, held = self.w, self.held
        view = w.view(b, i)
        rgb, depth = view["rgb"], view["depth"]
        masks = self.eyes.masks(rgb, depth, w.cfg)
        vectors = self.eyes.describe(rgb, masks)
        for field in (held.on[b, i], self.fresh[b, i]):
            field[:] = False
        if not masks:
            self.tracks[b][i].last = []
            return
        kinds, score = self.names.name(vectors)
        boxes = [box_of(m) for m in masks]
        followed = self.tracks[b][i].follow(boxes)
        tracked = []
        for m, (mask, vector, kind, was, box) in enumerate(zip(masks[:MAX_OBJECTS], vectors, kinds, followed, boxes)):
            who, new = (was, False) if was >= 0 else self.memory.know(vector)
            self.memory.add(vector, who)
            tracked.append((box, who))
            ang, dist, wide = locate(mask, depth, w.cfg)
            held.on[b, i, m], held.kind[b, i, m], held.ang[b, i, m] = True, kind, ang
            held.dist[b, i, m], held.size[b, i, m] = dist, wide
            held.box[b, i, m] = np.array(box) / np.tile(rgb.shape[1::-1], 2)
            self.ids[b, i, m], self.fresh[b, i, m] = who, new
        self.tracks[b][i].last = tracked

    def see(self) -> Percept:
        w = self.w
        if self.ticks % self.every == 0:
            for b, i in np.ndindex(w.B, w.n):
                self._look(b, i)
        else:
            # Fra uno sguardo e l'altro: ciò che so, spostato dal movimento che sento.
            cfg, held = w.cfg, self.held
            speed, spin = w.speed[..., None], w.spin[..., None] * cfg.turn
            x = held.dist * np.cos(held.ang) - speed * cfg.dt
            y = held.dist * np.sin(held.ang)
            held.dist, held.ang = np.hypot(x, y), np.arctan2(y, x) - spin * cfg.dt
        self.ticks += 1
        return self.held
