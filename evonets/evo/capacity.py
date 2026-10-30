"""Il test di capacità del cervello (docs/SPEC_CAPACITA.md): il cervello di
oggi, con la sua plasticità, è in grado di imparare durante la vita?

1. Il meccanismo, sinapsi per sinapsi (`mechanism`): una sinapsi da un senso
   a un motore, i due attivi insieme per un secondo; dopo un ritardo arriva
   piacere (gusto) o dolore. Se la regola distingue il bene dal male, il
   piacere deve rafforzare la sinapsi più del dolore. Si usa la regola vera
   (`brain.plasticity`) con i geni di partenza di molti cervelli.
2. L'evoluzione su un compito che si risolve solo imparando (`evolve`): il
   mondo del cibo con i frutti, metà cattivi, e quale colore è cattivo cambia
   a caso in ogni vita. La misura è la scelta: frutti buoni sul totale dei
   mangiati (50% = a caso), con lo stesso peso per le vite in cui il buono è
   il rosso e quelle in cui è il giallo: altrimenti chi preferisce un colore
   per istinto sembra scegliere bene (nelle vite fortunate mangia di più e
   pesa di più). Ogni tanto la stessa popolazione si prova con e senza
   plasticità (`discrimination`).

Uso: `uv run python -m evonets.evo.capacity --part meccanismo` (o `evoluzione`,
o `tutto`); `--help` per le opzioni.
"""

import argparse
import dataclasses
import json
import time

import numpy as np

from ..config import Config
from ..net.brain import PLASTICITY_CONSTANTS, Genome, plasticity
from ..net.controllers import NET_TYPES, NeuralController
from ..sim.world import World

DELAYS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
FIELDS = ("w_eff", "early", "elig", "critic", "bias_now", "avg", "sign", "links", "target_act", "homeo", "intrinsic", "hebb",
          "eta", "rule", "early_keep", "elig_keep", "boost", "closure", "clock", "value")


# --- 1. Il meccanismo ---------------------------------------------------------------

def mechanism(cfg: Config, n: int = 400, seed: int = 0, delays=DELAYS, hold: float = 1.0,
              after: float = 20.0, rule=None, genomes=None) -> list[dict]:
    """Per ogni ritardo: di quanto il piacere e il dolore cambiano la sinapsi
    senso -> motore rispetto a nessun segnale, in media su `n` cervelli con i
    geni di partenza (o `genomes`, per esempio quelli evoluti). `rule`
    impone a tutti la stessa regola modulata (4 coefficienti: pre·post, pre,
    post, costante). `valence` è la quota di cervelli in cui il piacere la
    rafforza più del dolore (0,5 = non distingue)."""
    cfg = dataclasses.replace(cfg, plasticity=True)
    kind, n_out = NET_TYPES[cfg.net_type]
    if genomes is not None:
        n_out = genomes.n_out
        brains = kind(cfg, len(genomes), n_out, seed)
        brains.set_genome(genomes)
        n = len(genomes)
    else:
        brains = kind(cfg, n, n_out, seed)
    H, R = brains.n_hidden, brains.n_in
    t, s = H, 0  # la sinapsi: dal primo recettore al primo motore
    base = {k: np.array(getattr(brains, k), dtype=float) for k in FIELDS}
    base["links"] = np.zeros_like(base["links"])
    base["links"][:, t, s] = 1.0
    base["sign"][:, t, s] = 1.0
    base["w_eff"][:, t, s] = 1.0
    base["early"] = np.zeros_like(base["early"])
    base["elig"] = np.zeros_like(base["elig"])
    base["critic"] = np.zeros_like(base["critic"])
    base["clock"] = np.full(n, 60.0)  # adulti: il periodo critico è finito
    if rule is not None:
        base["rule"][:] = np.asarray(rule, dtype=float)
    consts = {k: getattr(cfg, k) for k in PLASTICITY_CONSTANTS}
    T, S = base["w_eff"].shape[1:]
    dt = cfg.dt
    out = []
    for delay in delays:
        final = {}
        for signal in ("niente", "piacere", "dolore"):
            p = {k: v.copy() for k, v in base.items()}
            t0 = hold / 2 if delay == 0 else hold + delay  # con ritardo 0 il segnale arriva mentre sono attivi
            for k in range(int(round((hold + delay + after) / dt))):
                now = k * dt
                pre, post = np.zeros((n, S)), np.zeros((n, T))
                if now < hold:
                    pre[:, s], post[:, t] = 1.0, 1.0
                sig = np.zeros((n, 2))
                if signal != "niente" and now >= t0:
                    sig[:, 0 if signal == "piacere" else 1] = np.exp(-(now - t0) / cfg.taste_tau)
                p["signal"] = sig
                w, early, bias, avg, elig, critic = plasticity(consts, np, p, pre, post, np.zeros((n, T)), R,
                                                               brains.CENTRED)
                p["w_eff"], p["early"], p["bias_now"], p["avg"], p["elig"], p["critic"] = w, early, bias, avg, elig, critic
            final[signal] = p["w_eff"][:, t, s].copy()
        good, bad = final["piacere"] - final["niente"], final["dolore"] - final["niente"]
        out.append({"delay": delay, "pleasure": float(good.mean()), "pain": float(bad.mean()),
                    "valence": float((good > bad + 1e-12).mean()), "same": float((np.abs(good - bad) <= 1e-12).mean()),
                    "hebb_trace": float((final["niente"] - 1.0).mean())})
    return out


# --- 2. L'evoluzione ----------------------------------------------------------------

def task_config(cfg: Config, need: int = 4) -> Config:
    """Il compito: il mondo del cibo con metà frutti cattivi (il colore cattivo
    cambia in ogni vita), nient'altro; bisogna mangiare `need` frutti buoni."""
    return dataclasses.replace(cfg, task="cibo", ladder="", forage_variety=1.0, forage_abundance=0.0,
                               forage_need=(need - cfg.food_need_easy) / (cfg.food_need_hard - cfg.food_need_easy),
                               forage_prey=0.0, forage_scent=0.0, forage_predators=0.0, forage_obstacles=0.0,
                               exam_every=0)


def choice(task) -> tuple[float, list[float]]:
    """La scelta: frutti buoni sul totale, separatamente per le vite in cui il
    buono è il rosso e in cui è il giallo, e la media dei due (50% = a caso,
    anche per chi preferisce sempre un colore)."""
    parts = []
    for color in (0, 1):
        m = task.good == color
        good, bad = task.eaten[m].sum(), task.bad_eaten[m].sum()
        parts.append(float(good / max(good + bad, 1)))
    return float(np.mean(parts)), parts


def discrimination(cfg: Config, genomes, seed: int, plastic: bool = True) -> dict:
    """Le reti `genomes` (una vita ciascuna, nella stessa arena di prova), con
    o senza plasticità: frutti buoni e cattivi mangiati, la scelta (vedi
    `choice`) e la quota di riusciti."""
    cfg = dataclasses.replace(cfg, plasticity=plastic)
    n = len(genomes)
    w = World(cfg, n, seed)
    c = NeuralController(cfg, n)
    c.brains.set_genome(genomes)
    c.reset(seed)
    done = False
    if cfg.engine == "gpu":
        from ..gpu import available

        if available():
            from ..gpu.sim import GpuWorlds, supported

            if supported(w, c):
                GpuWorlds(cfg, [(w, c)], [seed]).run()
                done = True
    while not done and not w.over:
        w.step(c.act(w), c.upkeep)
    both, parts = choice(w.task)
    return {"good": int(w.task.eaten.sum()), "bad": int(w.task.bad_eaten.sum()), "choice": both,
            "red_good": parts[0], "yellow_good": parts[1], "completed": float(w.done.mean())}


def evolve(cfg: Config, generations: int, seed: int = 0, every: int = 25, sample: int = 300, log=print,
           rule=None, save=None) -> list[dict]:
    """`generations` generazioni sul compito; a ogni generazione la scelta
    della popolazione, ogni `every` la prova con e senza plasticità su
    `sample` individui (le stesse arene per le due prove). Con `rule` la
    regola modulata di partenza è quella per tutti (un istinto a imparare
    ereditato), poi muta come gli altri geni."""
    from .evolution import Evolution

    evo = Evolution(cfg, seed)
    if rule is not None:
        b = evo.ctrl.brains
        g = b.genome
        g.traits[:, :, b.BASE_TRAITS + 1:b.BASE_TRAITS + 5] = np.asarray(rule, dtype=float)
        b.set_genome(g)
    rng = np.random.default_rng([seed, 5])
    rows = []
    t0 = time.time()
    for g in range(1, generations + 1):
        st = evo.run_generation()
        good, bad = st.get("eaten", 0.0), st.get("bad_eaten", 0.0)
        # La quota grezza della popolazione (pesata da chi mangia di più): solo indicativa.
        row = {"generation": g, "raw": round(good / max(good + bad, 1e-9), 3), "completed": st["completed"],
               "neurons": round(st["neurons"], 2), "links": round(st["links"], 1), "used": round(st.get("used", 0), 2),
               "s": round(time.time() - t0)}
        if g % every == 0 or g == generations:
            genome = evo.ctrl.brains.genome
            idx = np.sort(rng.choice(len(genome), min(sample, len(genome)), replace=False))
            test = genome.take(idx)
            row["with"] = discrimination(cfg, test, 7_000 + g, True)
            row["without"] = discrimination(cfg, test, 7_000 + g, False)
        rows.append(row)
        log(json.dumps(row))
    if save:  # la popolazione finale, per esaminarla dopo
        g = evo.ctrl.brains.genome
        np.savez_compressed(save, **{f: getattr(g, f) for f in Genome.FIELDS})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("meccanismo", "evoluzione", "tutto"), default="tutto")
    ap.add_argument("--net", choices=tuple(NET_TYPES), default="spike-ei")
    ap.add_argument("--genome", choices=("diretto", "sviluppo"), default="diretto")
    ap.add_argument("--mode", choices=("evoluzione", "cervello"), default="evoluzione",
                    help="cervello = sensi fissi (occhi rosso e giallo, gusto, dolore...): evolve solo il cervello")
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--population", type=int, default=None)
    ap.add_argument("--need", type=int, default=4, help="frutti buoni da mangiare per riuscire")
    ap.add_argument("--no-plasticity", action="store_true", help="controllo: evoluzione senza plasticità")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="velocità di apprendimento di partenza, in volte quella normale (plasticity_initial_rate)")
    ap.add_argument("--rule", action="store_true",
                    help="regola di partenza già orientata: pre·post·M per tutti (istinto a imparare)")
    ap.add_argument("--surprise", type=float, default=None,
                    help="sorpresa: 0 = piacere grezzo, 1 = piacere meno quello atteso (config.surprise)")
    ap.add_argument("--save", default=None, help="salva qui (.npz) i genomi finali")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", choices=("gpu", "numpy"), default="gpu")
    args = ap.parse_args()
    cfg = Config(net_type=args.net, genome=args.genome, engine=args.engine, sim_mode=args.mode)
    if args.surprise is not None:
        cfg.surprise = args.surprise
    if args.population:
        cfg.population = args.population
    if args.part in ("meccanismo", "tutto"):
        print("# meccanismo: variazione della sinapsi senso -> motore rispetto a nessun segnale")
        for r in mechanism(cfg, seed=args.seed):
            print(json.dumps(r), flush=True)
    if args.part in ("evoluzione", "tutto"):
        task = task_config(cfg, args.need)
        if args.no_plasticity:
            task.plasticity = False
        task.plasticity_initial_rate *= args.rate
        print(f"# evoluzione: {args.generations} generazioni, plasticità {'spenta' if args.no_plasticity else 'accesa'}")
        evolve(task, args.generations, args.seed, log=lambda s: print(s, flush=True),
               rule=(1.0, 0.0, 0.0, 0.0) if args.rule else None, save=args.save)


if __name__ == "__main__":
    main()
