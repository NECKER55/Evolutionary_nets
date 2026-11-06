"""Le prove nel mondo 3D: quante pile prende un agente di controllo.

Uso: `uv run python -m evonets.arenas.mondo3d.run --agente perfetto --minuti 10`;
`--confronto` fa tutti gli agenti con lo stesso seme.
"""

import argparse
import json
import time

import numpy as np

from ...brains.mondo3d import CONTROLS as AGENTS
from .world import Mondo3D, Mondo3DConfig


class Run3D:
    """Un agente in un mondo 3D; stessa idea di `arenas.pile.run.NumpyRun`."""

    def __init__(self, cfg: Mondo3DConfig, agent: str, enclosures: int = 1, seed: int = 0):
        self.cfg, self.agent = cfg, agent
        self.w = Mondo3D(cfg, enclosures, seed)
        self.B, self.n = self.w.B, self.w.n
        self.brain = AGENTS[agent](self.w, seed)

    @property
    def t(self) -> float:
        return self.w.t

    def step(self) -> None:
        self.w.step(*self.brain.act())

    def advance(self, k: int) -> None:
        for _ in range(int(k)):
            self.step()

    def close(self) -> None:
        self.w.close()


def measures(r: Run3D) -> dict:
    """Le misure da confrontare, con le stesse parole di `arenas.pile.run.measures`."""
    w = r.w
    good = sum(1 for x in w.taken if x[4])
    bad = len(w.taken) - good
    minutes = max(w.t / 60.0, 1e-9) * r.B * r.n
    return {"agente": r.agent, "minuti": round(w.t / 60.0, 2), "esseri": r.B * r.n,
            "buone": good, "cattive": bad, "pile/min": round(good / minutes, 2), "cattive/min": round(bad / minutes, 2),
            "scelta": round(good / len(w.taken), 3) if w.taken else None, "morti di fame": w.deaths["fame"],
            "energia media": round(float(w.energy.mean()), 1)}


def run(agent: str, enclosures: int, minutes: float, seed: int, cfg: Mondo3DConfig) -> dict:
    r = Run3D(cfg, agent, enclosures, seed)
    start = time.perf_counter()
    r.advance(int(round(minutes * 60.0 / cfg.dt)))
    seconds = time.perf_counter() - start
    out = measures(r) | {"s di mondo/s": round(r.t / seconds, 1)}
    r.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agente", choices=tuple(AGENTS), default="perfetto")
    ap.add_argument("--confronto", action="store_true", help="tutti gli agenti")
    ap.add_argument("--recinti", type=int, default=4)
    ap.add_argument("--esseri", type=int, default=1, help="esseri per recinto")
    ap.add_argument("--minuti", type=float, default=10.0)
    ap.add_argument("--buone", choices=("colore", "grandezza"), default="colore")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = lambda: Mondo3DConfig(beings=args.esseri, good_by=args.buone)
    for agent in tuple(AGENTS) if args.confronto else (args.agente,):
        print(json.dumps(run(agent, args.recinti, args.minuti, args.seed, cfg()), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
