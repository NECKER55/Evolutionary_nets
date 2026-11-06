"""Registra eventi veri dal mondo 3D: un vedente morde pile, a volte quelle sbagliate,
e ogni morso diventa una riga come quelle del mondo a eventi. L'oggetto è ciò che ha
visto il detector finto. Gli eventi di `events_3d.npz` furono registrati con la nostra
piccola rete convoluzionale, poi tolta: stesso formato, oggetti visti con qualche errore.

Uso: `uv run python -m evonets.brains.ragionante.record_events --minuti 15`.
"""

import argparse
from pathlib import Path

import numpy as np

from ...arenas.mondo3d.world import Mondo3D, Mondo3DConfig
from .brain import Reasoning
from .events import BITE, PAIN, PLEASURE
from .seeing import Seeing
from .vision import FakeDetector

MISTAKES = 0.4
RECORDS = Path(__file__).parent / "events_3d.npz"


def bitten(seen, cfg, b: int, i: int, colour: int):
    """L'oggetto che si è morso: il più vicino del suo colore nella vista di prima."""
    on = np.flatnonzero(seen.on[b, i] & (seen.kind[b, i] == colour))
    if not len(on):
        return None
    return seen.embedding(cfg.eye_range)[b, i, on[np.argmin(seen.dist[b, i, on])]]


def record(minutes: float, enclosures: int, seed: int) -> dict:
    cfg = Mondo3DConfig(beings=1)
    w = Mondo3D(cfg, enclosures, seed)
    brain = Reasoning(w, seed, Seeing(w, FakeDetector(w), seed, mistakes=MISTAKES))
    rows = {"life": [], "obj": [], "act": [], "out": [], "need": [], "swaps": []}
    for _ in range(int(minutes * 60.0 / cfg.dt)):
        action = brain.act()
        seen, need, taken = brain.boss.seen, 1.0 - w.energy / cfg.energy_max, len(w.taken)
        w.step(*action)
        for _, b, i, colour, ok in w.taken[taken:]:
            obj = bitten(seen, cfg, b, i, colour)
            if obj is not None:
                rows["life"].append(b * w.n + i)
                rows["obj"].append(obj)
                rows["act"].append(BITE)
                rows["out"].append(PLEASURE if ok else PAIN)
                rows["need"].append(need[b, i])
                rows["swaps"].append(len(w.inversions))
    w.close()
    return {k: np.array(v, dtype=np.float32 if k in ("obj", "need") else int) for k, v in rows.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minuti", type=float, default=15.0)
    p.add_argument("--recinti", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    data = record(a.minuti, a.recinti, a.seed)
    np.savez_compressed(RECORDS, **data)
    print(f"{len(data['out'])} morsi, {(data['out'] == PAIN).sum()} con dolore, {data['swaps'].max()} scambi", flush=True)


if __name__ == "__main__":
    main()
