"""La banca di embedding: vettori veri di oggetti visti nel 3D, calcolati una volta sola.

Le reti pronte sono troppo lente per milioni di vite, e un mondo senza immagini non ha
niente da mostrare loro. Così si guarda ogni oggetto da molti punti di vista, si salva il
vettore con ciò che serve a ritrovarlo, e poi si legge dalla tabella: è la lingua comune
delle sezioni, ricavata da P e non scritta a mano. Il ritaglio lo dà il simulatore, quindi
qui SlimSAM non serve. Ci sono anche forme e colori che nel mondo non esistono ancora.

Uso: `uv run --group percezione python -m evonets.brains.ragionante.bank --viste 200`.
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from ...arenas.mondo3d.world import SCALE, Mondo3D, Mondo3DConfig

SIZE = 64  # numeri della lingua comune
RAW = 387  # i vettori di `ready.Eyes`: DINOv2 più il colore
PATH = Path(__file__).parent / "embeddings.npz"
COLOURS = {"rossa": (0.92, 0.31, 0.27), "gialla": (0.94, 0.82, 0.24), "blu": (0.20, 0.35, 0.90),
           "verde": (0.25, 0.75, 0.35), "viola": (0.62, 0.30, 0.85)}
FORMS = {"cilindro": mujoco.mjtGeom.mjGEOM_CYLINDER, "scatola": mujoco.mjtGeom.mjGEOM_BOX,
         "sfera": mujoco.mjtGeom.mjGEOM_SPHERE}
SIZES = {"piccola": 0.6, "grande": 1.0}
BEING = "essere"
FAR = (30.0, 300.0)  # cm tra cui si guarda un oggetto
WIDE = 0.8  # rad: di quanto può stare di lato


def kinds() -> list:
    """Tutti i tipi di oggetto, come (colore, forma, grandezza); l'ultimo è un essere."""
    return [(c, f, s) for c in COLOURS for f in FORMS for s in SIZES] + [(BEING, "", "")]


def name(kind) -> str:
    return " ".join(x for x in kind if x)


def _shape(w, kind) -> int:
    """Veste la pila 0 da `kind`. -> la sua geometria."""
    colour, form, size = kind
    cfg, m, g = w.cfg, w.model, int(w.pgeom[0, 0])
    r, h = cfg.battery_radius * SIZES[size] * SCALE, cfg.pile_height / 2.0 * SCALE
    m.geom_type[g] = FORMS[form]
    m.geom_size[g] = {"cilindro": (r, h, 0.0), "scatola": (r, r, h), "sfera": (h, 0.0, 0.0)}[form]
    m.geom_rbound[g] = float(np.hypot(r, h))
    m.geom_rgba[g] = (*COLOURS[colour], 1.0)
    return g


def _pose(w, rng, kind) -> tuple:
    """Un punto di vista a caso. -> (geometrie dell'oggetto, angolo, distanza)."""
    cfg = w.cfg
    ahead = np.exp(rng.uniform(*np.log(FAR)))
    side = rng.uniform(-WIDE, WIDE)
    w.heading[0, 0] = rng.uniform(-np.pi, np.pi)
    w.pos[0, 0] = cfg.size / 2.0 + rng.uniform(-40.0, 40.0, 2)
    way = w.heading[0, 0] + side
    where = w.pos[0, 0] + ahead * np.array((np.cos(way), np.sin(way)))
    behind = w.pos[0, 0] - 60.0 * np.array((np.cos(w.heading[0, 0]), np.sin(w.heading[0, 0])))
    w.bon[0, 0], w.bpos[0, 0] = kind[0] != BEING, where
    w.pos[0, 1] = where if kind[0] == BEING else behind
    w.heading[0, 1] = rng.uniform(-np.pi, np.pi)
    w.model.light_diffuse[0] = rng.uniform(0.35, 0.65)
    w.sync()
    if kind[0] == BEING:
        owner = [mujoco.mj_id2name(w.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in w.model.geom_bodyid]
        return np.array([g for g, o in enumerate(owner) if o.endswith("0_1")]), side, ahead
    g = _shape(w, kind)
    mujoco.mj_forward(w.model, w.data)
    return np.array([g]), side, ahead


def record(views: int, seed: int = 0) -> dict:
    """`views` viste per tipo. -> vettori grezzi, tipo, angolo e distanza di ogni riga."""
    from .ready import MIN_AREA, Eyes

    eyes = Eyes()
    w = Mondo3D(Mondo3DConfig(beings=2, size=800.0), 1, seed)
    w.bon[:], w.bwait[:] = False, 1e9
    rng = np.random.default_rng(seed)
    out = {"raw": [], "kind": [], "ang": [], "dist": []}
    start = time.time()
    for k, kind in enumerate(kinds()):
        done = 0
        while done < views:
            geoms, ang, dist = _pose(w, rng, kind)
            mask = np.isin(w.segments(0, 0), geoms)
            if mask.sum() < MIN_AREA:
                continue
            out["raw"].append(eyes.describe(w.view(0, 0, depth=False)["rgb"], [mask])[0])
            out["kind"].append(k)
            out["ang"].append(ang)
            out["dist"].append(dist)
            done += 1
        print(f"{name(kind)}: {time.time() - start:.0f} s", flush=True)
    w.close()
    return {k: np.array(v, dtype=np.float32 if k != "kind" else int) for k, v in out.items()}


def fit(raw: np.ndarray, size: int = SIZE) -> tuple:
    """La riduzione, un calcolo e non una rete. -> (media, base (size, `RAW`), quota di varianza tenuta)."""
    mean = raw.mean(0)
    _, values, basis = np.linalg.svd(raw - mean, full_matrices=False)
    return mean, basis[:size], float((values[:size] ** 2).sum() / (values ** 2).sum())


def shrink(raw: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """(..., `RAW`) -> (..., `SIZE`), di lunghezza 1 come i vettori di partenza."""
    v = (raw - mean) @ basis.T
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def recognised(vectors: np.ndarray, kind: np.ndarray, label: np.ndarray | None = None) -> float:
    """Quota di righe il cui vicino più simile ha lo stesso tipo (o la stessa `label`), senza contare se stessa."""
    near = vectors @ vectors.T
    np.fill_diagonal(near, -np.inf)
    label = kind if label is None else label
    return float((label[near.argmax(1)] == label).mean())


class Bank:
    """La tabella già pronta: oggetto e punto di vista -> vettore."""

    def __init__(self, path=PATH):
        data = np.load(path)
        self.vectors, self.kind, self.ang, self.dist = data["vectors"], data["kind"], data["ang"], data["dist"]
        self.mean, self.basis, self.names = data["mean"], data["basis"], [str(n) for n in data["names"]]

    def pick(self, rng, kind: int, ang: float, dist: float, close: int = 8) -> np.ndarray:
        """Una vista vera di `kind`, a caso tra le `close` più simili a quella chiesta."""
        rows = np.flatnonzero(self.kind == kind)
        gap = (self.ang[rows] - ang) ** 2 + np.log(self.dist[rows] / dist) ** 2
        return self.vectors[rows[rng.choice(np.argsort(gap)[:close])]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--viste", type=int, default=200, help="viste per tipo di oggetto")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    data = record(a.viste, a.seed)
    mean, basis, kept = fit(data["raw"])
    vectors = shrink(data["raw"], mean, basis).astype(np.float32)
    all_kinds = kinds()
    np.savez_compressed(PATH, vectors=vectors, kind=data["kind"], ang=data["ang"], dist=data["dist"],
                        mean=mean, basis=basis.astype(np.float32), names=np.array([name(k) for k in all_kinds]))
    colour = np.array([list(COLOURS).index(k[0]) if k[0] != BEING else len(COLOURS) for k in all_kinds])[data["kind"]]
    print(f"{len(vectors)} righe; varianza tenuta {kept:.1%}; riconosciuti 387 numeri "
          f"{recognised(data['raw'], data['kind']):.1%}, 64 numeri {recognised(vectors, data['kind']):.1%}, "
          f"solo colore {recognised(vectors, data['kind'], colour):.1%}", flush=True)


if __name__ == "__main__":
    main()
