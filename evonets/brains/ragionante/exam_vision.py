"""L'esame della sezione P nel mondo delle pile, in mondi mai visti: confronta ogni
detector con la verità del simulatore, sulle stesse immagini.

Una pila conta se se ne vedono almeno `MIN_PILE` pixel e non sta sfumando. È trovata
se c'è un oggetto del suo colore entro `ANGLE` gradi e il 25% di distanza.

Uso: `uv run --group percezione python -m evonets.brains.ragionante.exam_vision`.
"""

import argparse
import json
import time

import numpy as np

from ...arenas.mondo3d.run import Run3D
from ...arenas.mondo3d.world import Mondo3DConfig
from ...arenas.pile.world import RED
from .train_vision import stage
from .vision import SIDE, Detector, Percept

MIN_PILE = 4  # pixel, nell'immagine da `SIDE`
ANGLE = np.radians(6.0)


def truth(w, b, i) -> list:
    """Le pile davvero visibili: [(tipo, angolo, distanza)]."""
    seg = w.segments(b, i)
    f = seg.shape[0] // SIDE
    d = w.bpos[b] - w.pos[b, i]
    dist, ang = np.hypot(*d.T), (np.arctan2(d[:, 1], d[:, 0]) - w.heading[b, i] + np.pi) % (2.0 * np.pi) - np.pi
    out = []
    for k in np.flatnonzero(w.bon[b]):
        g = w.pgeom[b, k]
        if (seg == g).sum() / f / f >= MIN_PILE and w.model.geom_rgba[g, 3] >= 0.5:
            out.append((0 if w.bcolor[b, k] == RED else 1, ang[k], dist[k]))
    return out


def score(p: Percept, b, i, piles: list, tally: dict) -> None:
    used = np.zeros(p.on.shape[-1], dtype=bool)
    for kind, ang, dist in piles:
        free = np.flatnonzero(p.on[b, i] & (p.kind[b, i] == kind) & ~used)
        hit = False
        if len(free):
            j = free[np.argmin(np.abs(p.ang[b, i, free] - ang) + np.abs(p.dist[b, i, free] - dist) / 100.0)]
            hit = abs(p.ang[b, i, j] - ang) < ANGLE and abs(p.dist[b, i, j] - dist) < max(25.0, 0.25 * dist)
            if hit:
                used[j] = True
                tally["ang"].append(abs(p.ang[b, i, j] - ang))
                tally["dist"].append(abs(p.dist[b, i, j] - dist) / dist)
        tally["trovate" if hit else "perse"] += 1
    tally["false"] += int((p.on[b, i] & (p.kind[b, i] < 2) & ~used).sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mondi", type=int, default=3)
    p.add_argument("--viste", type=int, default=12, help="immagini per mondo")
    p.add_argument("--pronto", action="store_true", help="anche SlimSAM e DINOv2")
    p.add_argument("--scene", action="store_true", help="scene costruite: contro i muri e a un passo dalle pile")
    a = p.parse_args()
    names = ["nostra"] + ["pronta"] * a.pronto
    tally = {n: {"trovate": 0, "perse": 0, "false": 0, "ang": [], "dist": [], "s": 0.0, "viste": 0} for n in names}
    shared = None
    for seed in range(7000, 7000 + a.mondi):
        r = Run3D(Mondo3DConfig(beings=3), "perfetto", 1, seed)
        r.advance(100)
        rng = np.random.default_rng(seed)
        eyes = {"nostra": Detector(r.w)}
        if a.pronto:
            from .ready import Eyes, ReadyDetector

            shared = shared or Eyes()
            eyes["pronta"] = ReadyDetector(r.w, eyes=shared)
        for _ in range(a.viste):
            if a.scene:
                stage(r.w, rng)
            else:
                r.advance(40)
            for name, detector in eyes.items():
                start = time.perf_counter()
                detector.ticks = 0
                seen = detector.see()
                tally[name]["s"] += time.perf_counter() - start
                tally[name]["viste"] += r.B * r.n
                for b, i in np.ndindex(r.B, r.n):
                    score(seen, b, i, truth(r.w, b, i), tally[name])
        r.close()
    for name, t in tally.items():
        found, lost, false = t["trovate"], t["perse"], t["false"]
        print(json.dumps({"rete": name, "pile trovate": round(found / max(found + lost, 1), 3), "su": found + lost,
                          "rilevamenti giusti": round(found / max(found + false, 1), 3),
                          "errore angolo (gradi)": round(float(np.degrees(np.mean(t["ang"]))), 2),
                          "errore distanza": round(float(np.mean(t["dist"])), 3),
                          "ms per vista": round(1000.0 * t["s"] / t["viste"], 1)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
