"""I pochi esempi che danno un nome a ciò che vedono le reti pronte: per ogni
tipo in `KINDS`, qualche ritaglio vero con la sua etichetta del simulatore.

Uso: `uv run --group percezione python -m evonets.brains.ragionante.teach_names`.
"""

import argparse

import numpy as np

from ...arenas.mondo3d.run import Run3D
from ...arenas.mondo3d.world import Mondo3DConfig
from .ready import Eyes, Prototypes
from .vision import KINDS

PURE = 0.7  # quota della maschera che deve essere di un solo tipo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--esempi", type=int, default=5, help="esempi per tipo")
    p.add_argument("--seed", type=int, default=4000)
    a = p.parse_args()
    eyes, vectors, kinds = Eyes(), [], []
    run = Run3D(Mondo3DConfig(beings=3), "perfetto", 1, a.seed)
    run.advance(100)
    while min(kinds.count(k) for k in range(len(KINDS))) < a.esempi:
        run.advance(30)
        for b, i in np.ndindex(run.B, run.n):
            view = run.w.view(b, i)
            rgb = view["rgb"]
            truth = run.w.labels(b, i)
            masks = eyes.masks(rgb, view["depth"], run.w.cfg)
            for m, v in zip(masks, eyes.describe(rgb, masks)):
                count = np.bincount(truth[m], minlength=len(KINDS) + 1)
                k = int(count[1:].argmax())
                if count[k + 1] >= PURE * m.sum() and kinds.count(k) < a.esempi:
                    vectors.append(v)
                    kinds.append(k)
        print({n: kinds.count(k) for k, n in enumerate(KINDS)}, flush=True)
    run.close()
    np.savez_compressed(Prototypes.PATH, vectors=np.stack(vectors), kinds=np.array(kinds))


if __name__ == "__main__":
    main()
