"""Arena, sensi e reti casuali: ogni agente è guidato dalla propria rete.

    uv run main.py                 # viewer interattivo
    uv run main.py --evolve 100    # 100 generazioni senza grafica, con statistiche
    uv run main.py --bench         # throughput della simulazione senza grafica
"""

import argparse
import time

import numpy as np

from evonets import Config, World
from evonets.config import TUNABLES
from evonets.evo import Evolution
from evonets.net import NeuralController

SPEEDS = (1, 2, 4, 8, 16)


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


def evolve(cfg: Config, generations: int, seed: int) -> None:
    evo = Evolution(cfg, seed)
    print(f"{cfg.population} individui, compito: {evo.world.task.name}")
    print(" gen  completato  sopravv.  neuroni  collegam.  vita(s)  evento")
    t0 = time.perf_counter()
    for _ in range(generations):
        s = evo.run_generation()
        print(
            f"{s['generation']:4d}  {s['completed']:4d} {s['completed'] / cfg.population:5.0%}  "
            f"{s['survivors']:8d}  {s['neurons']:7.1f}  {s['links']:9.1f}  {s['duration']:7.1f}  "
            f"{'' if s['event'] == 'normale' else s['event']}"
        )
    el = time.perf_counter() - t0
    print(f"{generations} generazioni in {el:.0f}s ({el / generations:.2f}s l'una)")


def view(cfg: Config, n_agents: int, seed: int) -> None:
    import pygame

    from evonets.viz.render import DIM, PLAYER, TEXT, Renderer

    pygame.init()
    world = World(cfg, n_agents, seed)
    ctrl = NeuralController(cfg, n_agents, seed)
    renderer = Renderer(world)
    clock = pygame.time.Clock()
    focus, manual, paused, show_rays = 0, False, False, True
    speed_i, tun_i, pop_seed = 0, 0, seed
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
                elif ev.key == pygame.K_v:
                    show_rays = not show_rays
                elif ev.key == pygame.K_m:
                    manual = not manual
                elif ev.key == pygame.K_r:
                    world.reset()
                    ctrl.reset()
                elif ev.key == pygame.K_n:  # nuova disposizione di pericoli e spawn
                    world.seed += 1
                    world.reset()
                    ctrl.reset()
                elif ev.key == pygame.K_g:  # nuova popolazione di reti casuali
                    pop_seed += 1
                    ctrl = NeuralController(cfg, n_agents, pop_seed)
                    world.reset()
                elif ev.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                    speed_i = min(speed_i + 1, len(SPEEDS) - 1)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed_i = max(speed_i - 1, 0)
                elif ev.key == pygame.K_TAB:
                    tun_i = (tun_i + 1) % len(TUNABLES)
                elif ev.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET, pygame.K_COMMA, pygame.K_PERIOD):
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
        if not paused:
            for _ in range(SPEEDS[speed_i]):
                actions = ctrl.act(world).copy()
                if manual:
                    actions[focus] = hand
                world.step(actions)

        if world.done[focus]:
            status = "COMPITO COMPLETATO"
        elif world.alive[focus]:
            status = "vivo"
        else:
            status = "MORTO"
        hud = [
            (f"t={world.t:6.1f}s  x{SPEEDS[speed_i]}  {clock.get_fps():4.0f} fps"
             f"{'  PAUSA' if paused else ''}", TEXT),
            (f"compito: {world.task.name}   completato {int(world.done.sum())}/{world.n}   "
             f"morti {int((~world.alive).sum())}", TEXT),
            (f"agente {focus} ({'guidi tu' if manual else 'guida la rete'}): {status}", PLAYER),
            (f"energia {world.energy[focus]:5.1f}   v={world.speed[focus]:5.1f} u/s   "
             f"w={world.ang_vel[focus]:+5.2f} rad/s", PLAYER),
            ("", TEXT),
        ]
        for i, name in enumerate(TUNABLES):
            mark = ">" if i == tun_i else " "
            hud.append((f"{mark} {name:17s} {getattr(cfg, name):8.3f}", TEXT if i == tun_i else DIM))
        hud.append(("", TEXT))
        hud.append(("clic segui un agente | M guida manuale (WASD/frecce) | V raggi", DIM))
        hud.append(("G nuove reti | N nuova arena | R reset | Spazio pausa | +/- velocita'", DIM))
        hud.append(("Tab e ,/. (o [ ]) regola una costante | Esc esci", DIM))
        renderer.draw(focus, hud, ctrl, show_rays)
        clock.tick(60)

    pygame.quit()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=None, help="numero di agenti (default: 20, 1000 in --bench)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evolve", type=int, metavar="G", help="fa evolvere G generazioni senza grafica")
    ap.add_argument("--bench", action="store_true", help="misura il throughput senza grafica")
    ap.add_argument("--steps", type=int, default=5000, help="step per --bench")
    args = ap.parse_args()
    cfg = Config()
    if args.evolve:
        if args.agents:
            cfg.population = args.agents
        evolve(cfg, args.evolve, args.seed)
    elif args.bench:
        bench(cfg, args.agents or 1000, args.steps, args.seed)
    else:
        view(cfg, args.agents or 20, args.seed)


if __name__ == "__main__":
    main()
