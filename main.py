"""Evoluzione di reti che guidano agenti in un'arena 2D.

    uv run main.py                          # interfaccia grafica: menu, corsa, esploratore delle reti
    uv run main.py --load runs/pop.npz      # il menu parte con quel salvataggio già scelto
    uv run main.py --evolve 300             # 300 generazioni nel terminale, poi salva
    uv run main.py --evolve 300 --select 5  # ... poi i sopravvissuti rifanno la prova 5 volte
    uv run main.py --bench                  # throughput della simulazione senza grafica

Senza --evolve né --bench le altre opzioni precompilano il menu.
"""

import argparse
import secrets
import time
from pathlib import Path

from evonets import Config, World
from evonets.evo import Evolution
from evonets.net import NeuralController


def bench(cfg: Config, n_agents: int, steps: int, seed: int) -> None:
    world = World(cfg, n_agents, seed)
    ctrl = NeuralController(cfg, n_agents, seed)
    t0 = time.perf_counter()
    for _ in range(steps):
        world.step(ctrl.act(world))
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
        return evo
    return Evolution(cfg, seed)


def evolve(cfg: Config, generations: int, seed: int, load: str | None, save: str, select: int = 0) -> None:
    evo = start(cfg, seed, load)
    seed_text = f"seme {evo.seed}" if evo.seed is not None else "seme sconosciuto (salvataggio vecchio)"
    print(f"{cfg.population} individui, rete {cfg.net_type}, {seed_text}, compito: {evo.world.task.name}")
    print(" gen  completato  sopravv.  neuroni  collegam.  vita(s)  inibitori  evento")
    t0 = time.perf_counter()
    for _ in range(generations):
        s = evo.run_generation()
        print(
            f"{s['generation']:4d}  {s['completed']:4d} {s['completed'] / cfg.population:5.0%}  "
            f"{s['survivors']:8d}  {s['neurons']:7.1f}  {s['links']:9.1f}  {s['duration']:7.1f}  "
            f"{format(s['inhibitory'], '9.0%') if 'inhibitory' in s else '        -'}  "
            f"{'' if s['event'] == 'normale' else s['event']}"
        )
    el = time.perf_counter() - t0
    evo.save(save)
    print(f"{generations} generazioni in {el:.0f}s ({el / generations:.2f}s l'una), salvato in {save}")
    if select:
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=None, help="individui (default: 200; 1000 in --bench)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seme della corsa (default: casuale, mostrato a schermo per poterla rifare)")
    ap.add_argument("--net", choices=("rate", "spike-ei"), default="rate",
                    help="tipo di rete: rate = neuroni continui, "
                         "spike-ei = a impulsi con neuroni eccitatori o inibitori")
    ap.add_argument("--evolve", type=int, metavar="G", help="fa evolvere G generazioni senza grafica")
    ap.add_argument("--select", type=int, default=0, metavar="K",
                    help="con --evolve: alla fine i sopravvissuti rifanno la prova K volte, senza figli")
    ap.add_argument("--load", metavar="FILE", help="riprende da una popolazione salvata")
    ap.add_argument("--save", metavar="FILE",
                    help="dove salvare (default: runs/pop.npz con --evolve, un nome nuovo nel menu)")
    ap.add_argument("--bench", action="store_true", help="misura il throughput senza grafica")
    ap.add_argument("--steps", type=int, default=5000, help="step per --bench")
    args = ap.parse_args()
    cfg = Config(net_type=args.net)
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    if args.bench:
        bench(cfg, args.agents or 1000, args.steps, seed)
        return
    if args.agents:
        cfg.population = args.agents
    if args.evolve:
        evolve(cfg, args.evolve, seed, args.load, args.save or "runs/pop.npz", args.select)
    else:
        from evonets.viz.app import App

        App(dict(net=args.net, agents=args.agents, seed=args.seed, load=args.load, save=args.save)).run()


if __name__ == "__main__":
    main()
