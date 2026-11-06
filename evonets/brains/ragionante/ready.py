"""La sezione P con reti già pronte, da confrontare con la nostra (`vision.py`):

- **FastSAM-s** (Ultralytics, AGPL-3.0) trova in un colpo solo tutte le cose
  dell'immagine, senza nomi: una maschera per cosa, anche pezzi di muro e pavimento;
- la **profondità** sulla maschera dice angolo, distanza e larghezza vera (`locate`);
- un **ritaglio a scala fissa** (`fixed`) di `WINDOW` cm veri tiene la grandezza nell'immagine;
- **DINOv2 small** (facebook/dinov2-small, Apache 2.0) ne fa un vettore, a cui si
  aggiunge il colore, ridotto a 64 numeri con la riduzione della banca (`bank.py`);
- `ObjectMemory` conserva i vettori, anche su disco: un vettore vicino a uno noto
  è un oggetto già visto, altrimenti è nuovo e si salva.

Tra un'immagine e l'altra la cosa si segue col `Tracker`: se resta nello stesso posto
è la stessa, anche se cambia aspetto.

Non sa che cosa sia una pila: il nome (`KINDS`) serve solo al comandante finto, e
viene dalla banca: la riga più simile, se somiglia abbastanza. Pesa troppo per ogni
passo: guarda ogni `PERIOD` secondi e nel frattempo tiene ciò che sa, aggiornato col
movimento sentito. Richiede `uv run --group percezione`.
"""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from .vision import MAX_OBJECTS, SPREAD, Percept, box_of, locate

FINDER = Path.home() / ".cache" / "evonets" / "FastSAM-s.pt"  # si scarica la prima volta, 23 MB
DESCRIBER = "facebook/dinov2-small"
EMBED = 384 + 3  # i vettori di DINOv2 small, più il colore
HUE = 1.5  # quanto pesa il colore nel vettore: da solo, DINOv2 sui nostri ritagli vede quasi solo la forma
PALE = 0.04  # sotto questa saturazione un oggetto è grigio: il colore non dice niente
CONFIDENCE = 0.4  # FastSAM: sotto questa fiducia una cosa si butta (il loro valore)
SAME_THING = 0.9  # FastSAM: maschere che si coprono così sono la stessa cosa (il loro valore)
SIDE = 1024  # FastSAM legge l'immagine con il lato lungo di tanti pixel
MIN_AREA = 6  # pixel: più piccola è rumore
SAME = 0.8  # coseno da cui due vettori sono lo stesso oggetto
VIEWS = 16  # viste ricordate per oggetto: si impara a riconoscerlo da più lati
PERIOD = 0.5  # s tra uno sguardo e l'altro
FOLLOW = 0.3  # sovrapposizione dei riquadri da cui, tra due immagini, è lo stesso oggetto
WINDOW = 40.0  # cm: il lato del ritaglio a scala fissa, più largo dell'oggetto più largo (un essere, 20 cm)
GREY = 128  # fuori dalla maschera: così lo sfondo non conta
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ObjectMemory:
    """Gli oggetti visti: per ognuno fino a `VIEWS` vettori, di lunghezza 1 e della misura del
    primo che riceve. Sopravvive tra una vita e l'altra se la si salva: l'encoder non cambia,
    quindi i vettori restano validi."""

    def __init__(self, same: float = SAME):
        self.same = same
        self.owner = np.zeros(0, dtype=int)  # di chi è ogni vista
        self.views = np.zeros((0, 0), dtype=np.float32)
        self.kinds = 0  # oggetti conosciuti

    def closest(self, vector, taken=()) -> tuple:
        """-> (oggetto, somiglianza); (-1, -1) se non si conosce niente. Gli oggetti `taken` non contano."""
        if not len(self.views):
            return -1, -1.0
        score = np.where(np.isin(self.owner, list(taken)), -np.inf, self.views @ vector)
        best = int(score.argmax())
        return int(self.owner[best]), float(score[best])

    def add(self, vector, who: int) -> None:
        """Una vista in più di `who`; se ne ha troppe, sostituisce quella più simile a questa."""
        mine = np.flatnonzero(self.owner == who)
        if len(mine) >= VIEWS:
            self.views[mine[(self.views[mine] @ vector).argmax()]] = vector
            return
        self.views = np.concatenate([self.views, vector[None]]) if len(self.views) else vector[None].astype(np.float32)
        self.owner = np.append(self.owner, who)

    def know(self, vector, taken=()) -> tuple:
        """Il cuore: già visto o nuovo? -> (oggetto, se è nuovo). `taken`: gli oggetti già
        nella stessa immagine, che non possono essere anche questo."""
        who, score = self.closest(vector, taken)
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


def hue(pixels: np.ndarray) -> np.ndarray:
    """Il colore di un oggetto, a prescindere da quanto è illuminato: (3,) di lunghezza 1, o zeri se è grigio."""
    mean = pixels.mean(0) / 255.0
    tint = mean / max(mean.sum(), 1e-6) - 1.0 / 3.0
    size = np.linalg.norm(tint)
    return tint / size if size > PALE else np.zeros(3)


class Eyes:
    """FastSAM più DINOv2: da un'immagine a [(maschera, vettore)]."""

    def __init__(self, threads: int = 0):
        if threads:
            torch.set_num_threads(threads)
        self.finder = None  # FastSAM si carica la prima volta che serve: la banca non lo usa
        self.dino = AutoModel.from_pretrained(DESCRIBER).eval().to(DEVICE)
        self.dino_in = AutoImageProcessor.from_pretrained(DESCRIBER)

    def masks(self, rgb: np.ndarray) -> list:
        """Tutte le cose dell'immagine, senza nomi e sfondo compreso: [(H, W) di bool]."""
        if self.finder is None:
            from ultralytics import FastSAM

            FINDER.parent.mkdir(parents=True, exist_ok=True)
            self.finder = FastSAM(str(FINDER))
        found = self.finder(np.ascontiguousarray(rgb[..., ::-1]), device=DEVICE, retina_masks=True, imgsz=SIDE,
                            conf=CONFIDENCE, iou=SAME_THING, verbose=False)[0]  # vuole i colori al contrario
        if found.masks is None:
            return []
        return [m for m in found.masks.data.cpu().numpy() > 0.5 if m.sum() >= MIN_AREA]

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
        v = self.dino(pixel_values=x["pixel_values"].to(DEVICE)).last_hidden_state[:, 0].cpu().numpy()
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        v = np.concatenate([v, HUE * np.stack([hue(rgb[m]) for m in masks])], -1)
        return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


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


class ReadyDetector:
    """La stessa interfaccia del detector nostro (`Percept`), con il vettore, più `ids` e
    `memory`. Lo sguardo dura più di un passo del mondo, ma la simulazione lo aspetta: il
    robot non è penalizzato dal tempo di calcolo, solo dal fatto che guarda ogni `PERIOD` secondi.

    `kind` è il tipo della riga più simile della banca, se è una cosa del mondo (pila rossa,
    pila gialla, essere) e somiglia almeno `NAMED`: serve solo al comandante finto.
    `looks[b][i]`: l'ultimo sguardo intero, per lo schermo; con `truth` anche gli oggetti veri."""

    def __init__(self, world, memory: ObjectMemory | None = None, eyes: Eyes | None = None):
        from .bank import Bank

        self.w = world
        self.eyes = eyes or Eyes()
        self.memory = memory or ObjectMemory()
        self.bank = Bank()
        self.tracks = [[Tracker() for _ in range(world.n)] for _ in range(world.B)]
        self.every = max(1, round(PERIOD / world.cfg.dt))
        self.ticks = 0
        self.held = Percept.empty(world.B, world.n)
        self.ids = np.full((world.B, world.n, MAX_OBJECTS), -1)
        self.fresh = np.zeros((world.B, world.n, MAX_OBJECTS), dtype=bool)  # visti ora per la prima volta
        self.looks = [[None] * world.n for _ in range(world.B)]
        self.truth = False  # se registrare anche gli oggetti veri a ogni sguardo: costa un'immagine in più

    def _look(self, b: int, i: int) -> None:
        w, held = self.w, self.held
        view = w.view(b, i)
        rgb, depth = view["rgb"], view["depth"]
        masks = self.eyes.masks(rgb)
        masks = [masks[k] for k in np.argsort([np.median(depth[m]) for m in masks])][:MAX_OBJECTS]  # dalle più vicine
        raw = self.eyes.describe(rgb, masks, depth)
        vectors = self.bank.shrink(raw)
        kinds, score = self.bank.name(vectors)
        for field in (held.on[b, i], self.fresh[b, i]):
            field[:] = False
        boxes = [box_of(m) for m in masks]
        followed = self.tracks[b][i].follow(boxes)
        tracked = []
        for m, (mask, vector, kind, was, box) in enumerate(zip(masks, vectors, kinds, followed, boxes)):
            # I 64 numeri, non i grezzi: col ritaglio a scala fissa il grigio attorno rende simili tutti i grezzi.
            who, new = (was, False) if was >= 0 else self.memory.know(vector, [w for _, w in tracked] + followed)
            self.memory.add(vector, who)
            tracked.append((box, who))
            ang, dist, wide = locate(mask, depth, w.cfg)
            held.on[b, i, m], held.kind[b, i, m], held.ang[b, i, m] = True, kind, ang
            held.dist[b, i, m], held.size[b, i, m], held.vector[b, i, m] = dist, wide, vector
            held.box[b, i, m] = np.array(box) / np.tile(rgb.shape[1::-1], 2)
            self.ids[b, i, m], self.fresh[b, i, m] = who, new
        self.tracks[b][i].last = tracked
        self.looks[b][i] = {"t": w.t, "rgb": rgb, "depth": depth, "masks": masks, "ids": [who for _, who in tracked],
                            "kind": kinds, "score": score, "dist": held.dist[b, i, :len(masks)].copy(),
                            "size": held.size[b, i, :len(masks)].copy(),
                            "truth": things(w, b, i) if self.truth else None}

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


def things(w, b: int, i: int) -> list:
    """Gli oggetti veri nell'immagine dell'essere `i`, dal simulatore: [(maschera, nome)].
    Solo per controllare la percezione: nessuna sezione lo legge."""
    seg = w.segments(b, i)
    out = []
    for k in np.flatnonzero(w.bon[b]):
        mask = seg == w.pgeom[b, k]
        if mask.sum() >= MIN_AREA:
            out.append((mask, f"pila {('rossa', 'gialla')[int(w.bcolor[b, k])]}"))
    for j in range(w.n):
        if j != i:
            mask = np.isin(seg, w.body_of[b][j])
            if mask.sum() >= MIN_AREA:
                out.append((mask, "essere"))
    return out
