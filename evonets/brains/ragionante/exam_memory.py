"""L'esame delle sezioni 3 e 4: vite mai viste, comprese le famiglie di regole escluse
dall'addestramento, contro due riferimenti e una lesione.

Uso: `uv run python -m evonets.brains.ragionante.exam_memory --variante a --livello 2`.
"""

import argparse
import json

import jax
import numpy as np

from . import memory
from .events import BITE, FAMILIES, HELD_OUT, IGNORE, LEVELS, NOTHING, PLEASURE, bite_accuracy, ideal, lives, remembered, rule_name, table
from .record_events import RECORDS
from .train_memory import path_of

LIVES = 2048
SEED = 99


def family(p, name: str, known: tuple) -> dict:
    """Quanto prevede in vite di una famiglia. Il limite conosce anche quella famiglia."""
    data = lives(jax.random.PRNGKey(SEED + FAMILIES.index(name)), LIVES, (name,))
    guess = memory.forecast(p, data)
    limit = ideal(data, tuple(dict.fromkeys(known + (name,))))
    bitten = np.asarray(data["out"]) != NOTHING
    nth = bitten.cumsum(1)  # quanti morsi ha già fatto
    late = lambda q, k: float(((q > 0.5) == (np.asarray(data["out"]) == PLEASURE))[bitten & (nth > k)].mean())
    return {"famiglia": name, "dopo 1 morso": round(late(guess, 1), 3), "dopo 3": round(late(guess, 3), 3), "dopo 5": round(late(guess, 5), 3),
            "limite dopo 5": round(late(limit, 5), 3), "ricordo dopo 5": round(late(remembered(data), 5), 3),
            "senza memoria": round(bite_accuracy(memory.forecast(p, data, blind=True), data), 3)}


def recovery(p, names: tuple) -> dict:
    """Quanti morsi giusti subito dopo uno scambio: il primo non si può indovinare."""
    data = lives(jax.random.PRNGKey(SEED + 50), LIVES, names)
    guess, limit = memory.forecast(p, data), ideal(data, names)
    out, flipped = np.asarray(data["out"]), np.asarray(data["flipped"])
    changed = np.zeros_like(flipped)
    changed[:, 1:] = flipped[:, 1:] != flipped[:, :-1]
    res = {"rete": [0, 0, 0], "limite": [0, 0, 0], "quanti": [0, 0, 0]}
    for i, t in zip(*np.nonzero(changed)):
        k = 0
        for j in np.flatnonzero(out[i, t:] != NOTHING)[:3] + t:
            right = out[i, j] == PLEASURE
            res["rete"][k] += (guess[i, j] > 0.5) == right
            res["limite"][k] += (limit[i, j] > 0.5) == right
            res["quanti"][k] += 1
            k += 1
    return {"morso n. dopo lo scambio": [1, 2, 3],
            "rete": [round(a / b, 3) for a, b in zip(res["rete"], res["quanti"])],
            "limite": [round(a / b, 3) for a, b in zip(res["limite"], res["quanti"])]}


def delay(p, names: tuple, gaps=(0, 5, 15, 30)) -> dict:
    """Ricorda ancora dopo eventi inutili? 12 eventi per capire, poi solo eventi senza esito."""
    out = {}
    base = lives(jax.random.PRNGKey(SEED + 60), LIVES, names)
    clean = ~np.asarray(base["flipped"]).any(1)  # si tengono solo le vite in cui non c'è uno scambio prima
    for gap in gaps:
        act, res = np.array(base["act"]), np.array(base["out"])
        act[:, 12:12 + gap], res[:, 12:12 + gap] = IGNORE, NOTHING
        data = base | {"act": act, "out": res}
        guess = memory.forecast(p, data)
        after = np.zeros_like(res, dtype=bool)
        after[:, 12 + gap:12 + gap + 5] = True
        keep = after & (res != NOTHING) & ~np.asarray(base["flipped"])
        out[f"dopo {gap} vuoti"] = round(float(((guess > 0.5) == (res == PLEASURE))[keep].mean()), 3)
    return out | {"vite senza scambi": int(clean.sum())}


def trust(p, names: tuple) -> dict:
    """La sezione 4 sa quando sbaglia? Confronta il dubbio nei morsi giusti e sbagliati."""
    data = lives(jax.random.PRNGKey(SEED + 70), LIVES, names)
    r = jax.vmap(lambda d: memory.live(p, d))(data)
    doubt, wrong = np.asarray(r["doubt"]), np.asarray(memory.pleasure(r["log"]))
    bitten = np.asarray(data["out"]) != NOTHING
    right = (wrong > 0.5) == (np.asarray(data["out"]) == PLEASURE)
    return {"dubbio se giusto": round(float(doubt[bitten & right].mean()), 3),
            "dubbio se sbagliato": round(float(doubt[bitten & ~right].mean()), 3)}


def real(p) -> dict:
    """Eventi veri registrati nel 3D (`record_events.py`), a rete bloccata."""
    data = np.load(RECORDS)
    hits, firsts = [], []
    for life in np.unique(data["life"]):
        at = data["life"] == life
        one = {k: jax.numpy.asarray(data[k][at]) for k in ("obj", "act", "out", "need")}
        guess = np.asarray(memory.pleasure(memory.live(p, one)["log"]))
        right = (guess > 0.5) == (data["out"][at] == PLEASURE)
        hits.append(right[5:])
        firsts.append(right[:5])
    late, early = np.concatenate(hits), np.concatenate(firsts)
    return {"morsi veri": int(len(data["out"])), "dopo 5 morsi": round(float(late.mean()), 3), "su": int(len(late)),
            "primi 5": round(float(early.mean()), 3), "quota di piacere": round(float((data["out"] == PLEASURE).mean()), 3)}


def show(p, name: str, seed: int, rows: int = 18) -> str:
    """Una vita mai vista, riga per riga, con ciò che prevedeva la rete prima di vedere."""
    data = lives(jax.random.PRNGKey(seed), 1, (name,))
    one = jax.tree_util.tree_map(lambda x: x[0], data)
    guess = memory.forecast(p, data)[0]
    lines = [f"regola nascosta: {rule_name(one)}", "n.  oggetto                     azione    esito     previsione del piacere"]
    for n, colour, size, near, act, out, flipped in table(one)[:rows]:
        mark = "  <- scambio" if flipped and n > 1 and not bool(one["flipped"][n - 2]) else ""
        note = f"{guess[n - 1]:.2f}" if act == "mordo" else "-"
        lines.append(f"{n:2d}  {colour:7s}{size:8s}{near:8s}      {act:9s} {out:9s} {note}{mark}")
    return "\n".join(lines)


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--variante", choices=("a", "b"), default="a")
    a.add_argument("--livello", choices=tuple(LEVELS), default="2")
    a.add_argument("--famiglie", nargs="*", help="regole usate in addestramento, se non sono quelle del livello")
    a.add_argument("--file")
    args = a.parse_args()
    known = tuple(args.famiglie) if args.famiglie else LEVELS[args.livello]
    p = memory.load(args.file or path_of(args.variante, f"_{args.livello}"))
    print(json.dumps({"variante": args.variante, "addestrata su": known}, ensure_ascii=False))
    for name in known + tuple(f for f in HELD_OUT if f not in known):
        print(("     " if name in known else "MAI: ") + json.dumps(family(p, name, known), ensure_ascii=False))
    for part in (recovery, delay, trust):
        print(json.dumps(part(p, known), ensure_ascii=False))
    if RECORDS.exists():
        print(json.dumps(real(p), ensure_ascii=False))


if __name__ == "__main__":
    main()
