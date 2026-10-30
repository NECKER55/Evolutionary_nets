"""Il banco di prova fisso dell'arena delle pile (docs/TRAGUARDI_PILE.md):
lo stesso compito per ogni versione, per sapere se il sistema migliora davvero
e se migliorando una cosa ne peggiora un'altra.

Compito: cervello a sezioni (o un altro agente), fase A, 200 recinti da 6
esseri, 15 minuti di mondo, semi 0 e 1, pile buone per colore. Misure (da
`pile_test.measures`): quota di pile buone, pile al minuto (buone e cattive),
pile buone al minuto, morti di fame al minuto, scelta nelle finestre intorno
alle inversioni (−30..0, 0..15, 15..30, 30..60, 60..90 s), ed energia
netta dalle pile al minuto (buone × energia buona − cattive × energia persa).

Uso: `uv run python -m evonets.evo.pile_bench --etichetta "nome versione"`.
Per una versione vecchia: `git worktree add /tmp/v <commit>` e poi
`PYTHONPATH=/tmp/v uv run python evonets/evo/pile_bench.py ...` da qui (lo
script usa solo `make_run` e `measures`, presenti in tutte le versioni).
"""

import argparse
import json
import time

from evonets.evo.pile_test import make_run, measures
from evonets.sim.pile import PileConfig


def bench(agent: str = "sezioni", seeds=(0, 1), minutes: float = 15.0, enclosures: int = 200) -> list[dict]:
    rows = []
    for seed in seeds:
        t0 = time.time()
        r = make_run(agent, "A", enclosures, seed, (), PileConfig(), "gpu")
        r.advance(int(minutes * 60 / r.cfg.dt))
        m = measures(r)
        good = m["pile al minuto"] * m["buone"]
        bad = m["pile al minuto"] - good
        rows.append({"seme": seed, "buone": m["buone"], "pile/min": m["pile al minuto"],
                     "buone/min": round(good, 3),
                     "energia/min": round(good * r.cfg.good_energy - bad * r.cfg.bad_energy, 2),
                     "morti/min": m["morti di fame al minuto"],
                     "inversione": list(m.get("scelta intorno alle inversioni", {}).values()),
                     "s": round(time.time() - t0)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etichetta", default="attuale")
    ap.add_argument("--agente", default="sezioni")
    ap.add_argument("--minuti", type=float, default=15.0)
    args = ap.parse_args()
    for row in bench(args.agente, minutes=args.minuti):
        print(json.dumps({"versione": args.etichetta, **row}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
