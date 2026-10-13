"""Evoluzione di reti che guidano agenti in un'arena 2D.

    uv run main.py                          # evoluzione dal vivo nel viewer
    uv run main.py --net spike-ei           # a impulsi, neuroni eccitatori o inibitori
    uv run main.py --load runs/pop.npz      # riprende da un salvataggio
    uv run main.py --evolve 300             # 300 generazioni senza grafica, poi salva
    uv run main.py --bench                  # throughput della simulazione senza grafica
"""

import argparse
import time

import numpy as np

from evonets import Config, World
from evonets.config import TUNABLES
from evonets.evo import Evolution
from evonets.net import NeuralController

SPEEDS = (1, 2, 4, 8, 16)
TURBO_BUDGET = 0.08  # s di calcolo per fotogramma in modalità turbo


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


def evolve(cfg: Config, generations: int, seed: int, load: str | None, save: str) -> None:
    evo = start(cfg, seed, load)
    print(f"{cfg.population} individui, rete {cfg.net_type}, compito: {evo.world.task.name}")
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


def view(cfg: Config, seed: int, load: str | None, save: str) -> None:
    import pygame

    from evonets.viz.render import DIM, PLAYER, TEXT, Renderer

    pygame.init()
    evo = start(cfg, seed, load)
    world = evo.world
    renderer = Renderer(world)
    clock = pygame.time.Clock()
    focus = int(np.argmax(evo.age))
    manual = paused = turbo = show_consts = False
    show_rays = True
    speed_i, tun_i, saved_at = 0, 0, None
    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and renderer.arena.collidepoint(ev.pos):
                click = np.array(ev.pos) / renderer.scale
                focus = int(np.argmin(np.linalg.norm(world.pos - click, axis=1)))
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_t:
                    turbo = not turbo
                elif ev.key == pygame.K_v:
                    show_rays = not show_rays
                elif ev.key == pygame.K_m:
                    manual = not manual
                elif ev.key == pygame.K_c:
                    show_consts = not show_consts
                elif ev.key == pygame.K_s:
                    evo.save(save)
                    saved_at = evo.generation
                elif ev.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                    speed_i = min(speed_i + 1, len(SPEEDS) - 1)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed_i = max(speed_i - 1, 0)
                elif show_consts and ev.key == pygame.K_TAB:
                    tun_i = (tun_i + 1) % len(TUNABLES)
                elif show_consts and ev.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET,
                                                pygame.K_COMMA, pygame.K_PERIOD):
                    up = ev.key in (pygame.K_RIGHTBRACKET, pygame.K_PERIOD)
                    name = TUNABLES[tun_i]
                    setattr(cfg, name, round(getattr(cfg, name) * (1.1 if up else 1 / 1.1), 4))

        keys = pygame.key.get_pressed()
        hand = np.array(
            [
                (keys[pygame.K_UP] or keys[pygame.K_w]) - (keys[pygame.K_DOWN] or keys[pygame.K_s]),
                (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a]),
            ],
            dtype=float,
        )

        def advance() -> None:
            nonlocal focus
            if evo.finished:
                evo.next_generation()
                focus = int(np.argmax(evo.age))  # il sopravvissuto più anziano, se c'è
            else:
                evo.step((focus, hand) if manual else None)

        if not paused:
            if turbo:
                deadline = time.perf_counter() + TURBO_BUDGET
                while time.perf_counter() < deadline:
                    advance()
            else:
                for _ in range(SPEEDS[speed_i]):
                    advance()

        if world.done[focus]:
            status = "COMPITO COMPLETATO"
        elif world.alive[focus]:
            status = "vivo"
        else:
            status = "MORTO"
        pace = "TURBO" if turbo else f"x{SPEEDS[speed_i]}"
        hud = [
            (f"generazione {evo.generation}   t={world.t:5.1f}s   {pace}   {clock.get_fps():3.0f} fps"
             f"{'   PAUSA' if paused else ''}", TEXT),
            (f"compito: {world.task.name}", TEXT),
            (f"completato {int(world.done.sum())}/{world.n}   morti {int((~world.alive).sum())}   "
             f"sopravvissuti in gara {int((evo.age > 0).sum())}", TEXT),
        ]
        if evo.history:
            h = evo.history[-1]
            note = "" if h["event"] == "normale" else f"   [{h['event']}]"
            hud.append((f"gen. precedente: {h['completed']}/{world.n} completati, "
                        f"{h['neurons']:.1f} neuroni e {h['links']:.0f} collegamenti in media{note}", DIM))
        hud += [
            ("", TEXT),
            (f"agente {focus} ({'guidi tu' if manual else 'guida la rete'}), "
             f"età {evo.age[focus]}: {status}", PLAYER),
            (f"energia {world.energy[focus]:5.1f}   v={world.speed[focus]:5.1f} u/s   "
             f"w={world.ang_vel[focus]:+5.2f} rad/s", PLAYER),
            ("", TEXT),
        ]
        if show_consts:
            for i, name in enumerate(TUNABLES):
                mark = ">" if i == tun_i else " "
                hud.append((f"{mark} {name:17s} {getattr(cfg, name):8.3f}", TEXT if i == tun_i else DIM))
            hud.append(("Tab scegli | , . (o [ ]) regola | C chiudi", DIM))
        else:
            hud.append(("+/- velocita' | T turbo | Spazio pausa | S salva | C costanti | Esc esci", DIM))
            hud.append(("clic segui un agente | M guida manuale (WASD/frecce) | V raggi", DIM))
        if saved_at is not None:
            hud.append((f"salvata la generazione {saved_at} in {save}"[:100], DIM))
        renderer.draw(focus, hud, evo.ctrl, show_rays, evo.age, evo.history)
        clock.tick(60)

    pygame.quit()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=None, help="individui (default: 200; 1000 in --bench)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--net", choices=("rate", "spike-ei"), default="rate",
                    help="tipo di rete: rate = neuroni continui, "
                         "spike-ei = a impulsi con neuroni eccitatori o inibitori")
    ap.add_argument("--evolve", type=int, metavar="G", help="fa evolvere G generazioni senza grafica")
    ap.add_argument("--load", metavar="FILE", help="riprende da una popolazione salvata")
    ap.add_argument("--save", metavar="FILE", default="runs/pop.npz",
                    help="dove salvare (tasto S nel viewer, fine di --evolve)")
    ap.add_argument("--bench", action="store_true", help="misura il throughput senza grafica")
    ap.add_argument("--steps", type=int, default=5000, help="step per --bench")
    args = ap.parse_args()
    cfg = Config(net_type=args.net)
    if args.bench:
        bench(cfg, args.agents or 1000, args.steps, args.seed)
        return
    if args.agents:
        cfg.population = args.agents
    if args.evolve:
        evolve(cfg, args.evolve, args.seed, args.load, args.save)
    else:
        view(cfg, args.seed, args.load, args.save)


if __name__ == "__main__":
    main()
