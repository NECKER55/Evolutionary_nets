"""Evoluzione di reti che guidano agenti in un'arena 2D.

    uv run main.py                          # interfaccia grafica: menu, corsa, esploratore delle reti
    uv run main.py --load runs/pop.npz      # il menu parte con quel salvataggio già scelto
    uv run main.py --evolve 300             # 300 generazioni nel terminale, poi salva
    uv run main.py --evolve 300 --select 5  # ... poi i sopravvissuti rifanno la prova 5 volte
    uv run main.py --exam runs/pop.npz      # esame fisso dei più anziani di un salvataggio
    uv run main.py --evolve 500 --seeds 4   # 4 semi insieme sulla GPU: runs/pop-s<seme>.npz
    uv run main.py --task cibo --ladder --evolve 2000          # mondo del cibo, scala adattiva
    uv run main.py --task cibo --ladder --controls --seeds 4 --evolve 2000
                                            # scala + controlli (sempre basico, sempre complesso), 4 semi
    uv run main.py --exam runs/pop.npz --lesion neuroni   # esame con i neuroni interni tolti
    uv run main.py --bench                  # throughput della simulazione senza grafica

Senza --evolve né --bench le altre opzioni precompilano il menu.
"""

import argparse
import secrets
import time
from pathlib import Path

import numpy as np

from evonets import Config, World
from evonets.evo import Evolution
from evonets.evo.evolution import describe_change
from evonets.evo.exam import LESIONS, battery
from evonets.net import NeuralController


SAVE_EVERY = 25  # generazioni tra un salvataggio e l'altro, con --evolve


def stop_requested() -> list:
    """Una lista che diventa non vuota quando arriva Ctrl+C o una richiesta di
    terminare: la corsa finisce la generazione, salva ed esce."""
    import signal

    flag: list = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: flag.append(True))
    return flag


def bench(cfg: Config, n_agents: int, steps: int, seed: int) -> None:
    world = World(cfg, n_agents, seed)
    ctrl = NeuralController(cfg, n_agents, seed)
    t0 = time.perf_counter()
    for _ in range(steps):
        world.step(ctrl.act(world), ctrl.upkeep)
    el = time.perf_counter() - t0
    print(
        f"{n_agents} agenti x {steps} step in {el:.2f}s -> "
        f"{steps / el:,.0f} step/s, {n_agents * steps / el:,.0f} agenti-step/s "
        f"({steps * cfg.dt / el:,.0f}x tempo reale), "
        f"completato: {int(world.done.sum())}, morti: {int((~world.alive).sum())}"
    )


def start(cfg: Config, seed: int, load: str | None) -> Evolution:
    if load:
        evo = Evolution.load(cfg, load)
        print(f"ripreso {load}: generazione {evo.generation}, {cfg.population} individui, rete {cfg.net_type}")
        for e in evo.log:
            if e["event"] == "cambio":
                print(f"  gen. {e['generation']}: {describe_change(e['changes'])}")
            if "note" in e:
                print(f"  nota: {e['note']}")
        return evo
    return Evolution(cfg, seed)


def print_exam(r: dict) -> None:
    scores = "  ".join(f"{k} {v:.0%}" for k, v in r["scores"].items())
    lesion = f", senza {r['lesion']}" if r.get("lesion") else ""
    print(f"  ESAME v{r['version']} gen. {r['generation']} ({r['who']} più anziani{lesion}, {r['seconds']:.0f}s): "
          f"media {r['score']:.0%}, migliore {r['best']:.0%} | {scores}")


def exam(cfg: Config, path: str, lesion: str | None = None) -> None:
    evo = Evolution.load(cfg, path)
    tests = battery(cfg)
    print(f"esame di {path}, generazione {evo.generation}: {len(tests)} prove, "
          f"{'; '.join(f'{t.name} ({t.what})' for t in tests)}" + (f"; lesione: {lesion}" if lesion else ""))
    evo.start_exam(lesion=lesion).run()
    print_exam(evo.collect_exam())
    evo.save(path)
    print(f"risultato aggiunto al salvataggio ({len(evo.exams)} esami in tutto)")


def evolve_many(cfg: Config, generations: int, seed: int, k: int, save: str, exam_every: int | None,
                controls: bool = False) -> None:
    """K semi in parallelo, nello stesso lotto sulla GPU. Con `controls`, per
    ogni seme anche le due popolazioni di controllo della scala (sempre
    basico, sempre complesso), con lo stesso seme di partenza."""
    import dataclasses

    from evonets.evo.evolution import run_together

    modes = ("scala", "basico", "complesso") if controls else (cfg.ladder,)
    evos = [Evolution(dataclasses.replace(cfg, ladder=m), seed + i) for i in range(k) for m in modes]
    if exam_every is not None:
        for e in evos:
            e.cfg.exam_every = exam_every
    path = Path(save)
    paths = [path.with_name(f"{path.stem}-s{e.seed}{'-' + e.cfg.ladder if controls else ''}{path.suffix}")
             for e in evos]
    names = [f"s{e.seed}" + (f"-{e.cfg.ladder}" if controls else "") for e in evos]
    print(f"{len(evos)} popolazioni ({', '.join(names)}), {cfg.population} individui ciascuna, rete {cfg.net_type}")
    forage = cfg.task == "cibo"
    print(" gen  completati per popolazione" + ("  (posizione nella scala 0-3)" if forage else "")
          + "  neuroni  recett.")
    t0 = time.perf_counter()
    stop = stop_requested()
    for _ in range(generations):
        if stop:
            print("fermata richiesta: salvo e mi fermo")
            break
        stats = run_together(evos)
        if evos[0].generation % SAVE_EVERY == 0:
            for e, p in zip(evos, paths):
                e.save(p)
        g = stats[0]["generation"]
        print(f"{g:4d}  " + " ".join(f"{s['completed']:4d}" + (f" ({s['position']:.2f})" if forage else "")
                                     for s in stats)
              + f"  {np.mean([s['neurons'] for s in stats]):7.1f}  {np.mean([s['receptors'] for s in stats]):7.1f}")
        for e, name in zip(evos, names):
            if e.exam_due():
                e.start_exam().run()
                r = e.collect_exam()
                print(f"  {name}: ", end="")
                print_exam(r)
    el = time.perf_counter() - t0
    for e, p in zip(evos, paths):
        e.save(p)
    print(f"{generations} generazioni di {len(evos)} popolazioni in {el:.0f}s ({el / generations:.2f}s l'una), salvati in "
          + ", ".join(str(p) for p in paths))


def evolve(cfg: Config, generations: int, seed: int, load: str | None, save: str, select: int = 0,
           exam_every: int | None = None) -> None:
    evo = start(cfg, seed, load)
    if exam_every is not None:
        cfg.exam_every = exam_every
    seed_text = f"seme {evo.seed}" if evo.seed is not None else "seme sconosciuto (salvataggio vecchio)"
    print(f"{cfg.population} individui, rete {cfg.net_type}, {seed_text}, compito: {evo.world.task.name}")
    print(" gen  completato  sopravv.  recett.  neuroni  collegam.  vita(s)  inibitori  evento")
    t0 = time.perf_counter()
    stop = stop_requested()
    done = 0
    for _ in range(generations):
        if stop:
            print("fermata richiesta: salvo e mi fermo")
            break
        s = evo.run_generation()
        done += 1
        if evo.generation % SAVE_EVERY == 0:
            evo.save(save)  # così una corsa fermata non si perde
        print(
            f"{s['generation']:4d}  {s['completed']:4d} {s['completed'] / cfg.population:5.0%}  "
            f"{s['survivors']:8d}  {s['receptors']:7.1f}  {s['neurons']:7.1f}  {s['links']:9.1f}  {s['duration']:7.1f}  "
            f"{format(s['inhibitory'], '9.0%') if 'inhibitory' in s else '        -'}  "
            f"{'' if s['event'] == 'normale' else s['event']}"
            + (f"  scala {s['position']:.2f}, mangiati {s['eaten']:.1f}" if "position" in s else "")
        )
        if evo.exam_due():
            evo.start_exam().run()
            print_exam(evo.collect_exam())
    el = time.perf_counter() - t0
    evo.save(save)
    print(f"{done} generazioni in {el:.0f}s ({el / max(done, 1):.2f}s l'una), salvato in {save}")
    if select and not stop:
        n = evo.start_selection(select)
        if not n:
            print("selezione finale: nessun sopravvissuto da selezionare")
            return
        print(f"selezione finale: {n} sopravvissuti, {select} prove, nessun figlio")
        while evo.selecting:
            while not evo.finished:
                evo.step()
            r = evo.next_selection_round()
            print(f"  prova {r['round']}: {r['entered']} in gara, {r['passed']} superano")
        if evo.selection["note"]:
            print(f"  {evo.selection['note']}")
        path = Path(save)
        path = path.with_name(path.stem + "-selezionati" + path.suffix)
        evo.save(path)
        print(f"restano {len(evo.ids)}: {', '.join(f'#{i}' for i in evo.ids)}; salvati in {path}")


def main() -> None:
    import sys

    sys.stdout.reconfigure(line_buffering=True)  # il registro si legge mentre la corsa va
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=None, help="individui (default: quelli della configurazione, 1000)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seme della corsa (default: casuale, mostrato a schermo per poterla rifare)")
    ap.add_argument("--net", choices=("rate", "spike-ei"), default="rate",
                    help="tipo di rete: rate = neuroni continui, "
                         "spike-ei = a impulsi con neuroni eccitatori o inibitori")
    ap.add_argument("--evolve", type=int, metavar="G", help="fa evolvere G generazioni senza grafica")
    ap.add_argument("--select", type=int, default=0, metavar="K",
                    help="con --evolve: alla fine i sopravvissuti rifanno la prova K volte, senza figli")
    ap.add_argument("--exam-every", type=int, metavar="K",
                    help="con --evolve: esame fisso ogni K generazioni (default 50; 0 = mai)")
    ap.add_argument("--exam", metavar="FILE", help="esame fisso dei più anziani di un salvataggio")
    ap.add_argument("--seeds", type=int, default=1, metavar="K",
                    help="con --evolve: K semi (dal seme dato in su) insieme nello stesso lotto sulla GPU")
    ap.add_argument("--engine", choices=("gpu", "numpy"), default="gpu",
                    help="motore di simulazione: gpu = compilato con JAX (default), numpy = l'originale")
    ap.add_argument("--task", choices=("bersaglio", "cibo"), default="bersaglio",
                    help="compito: bersaglio = toccarlo una volta (il mondo di prima), "
                         "cibo = nutrirsi (docs/SPEC_MONDO.md; senza veleno)")
    ap.add_argument("--ladder", action="store_true",
                    help="con --task cibo: la scala di difficoltà adattiva (evo/ladder.py)")
    ap.add_argument("--controls", action="store_true",
                    help="con --ladder e --evolve: per ogni seme anche le popolazioni di controllo "
                         "(sempre basico, sempre complesso) nello stesso lotto")
    ap.add_argument("--lesion", choices=LESIONS, help="con --exam: esame con quella parte tolta")
    ap.add_argument("--load", metavar="FILE", help="riprende da una popolazione salvata")
    ap.add_argument("--save", metavar="FILE",
                    help="dove salvare (default: runs/pop.npz con --evolve, un nome nuovo nel menu)")
    ap.add_argument("--bench", action="store_true", help="misura il throughput senza grafica")
    ap.add_argument("--steps", type=int, default=5000, help="step per --bench")
    args = ap.parse_args()
    cfg = Config(net_type=args.net, engine=args.engine, task=args.task)
    if args.task == "cibo":
        cfg.n_hazards = 0  # il veleno resta spento in tutta la scala
    if args.ladder or args.controls:
        if args.task != "cibo":
            ap.error("--ladder e --controls valgono solo con --task cibo")
        cfg.ladder = "scala"
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    if args.bench:
        bench(cfg, args.agents or 1000, args.steps, seed)
        return
    if args.agents:
        cfg.population = args.agents
    if args.exam:
        exam(cfg, args.exam, args.lesion)
    elif args.evolve and (args.seeds > 1 or args.controls):
        evolve_many(cfg, args.evolve, seed, args.seeds, args.save or "runs/pop.npz", args.exam_every, args.controls)
    elif args.evolve:
        evolve(cfg, args.evolve, seed, args.load, args.save or "runs/pop.npz", args.select, args.exam_every)
    else:
        from evonets.viz.app import App

        App(dict(net=args.net, agents=args.agents, seed=args.seed, load=args.load, save=args.save)).run()


if __name__ == "__main__":
    main()
