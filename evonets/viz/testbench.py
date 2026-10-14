"""Banco di prova: una sola rete, da sola nell'arena.

Si guarda l'agente arena dopo arena (le arene sono sempre le stesse per tutte
le reti, così si confrontano alla pari) e, a richiesta, si lancia una
valutazione: la stessa rete in 20 arene, 50 copie per arena, per una
percentuale di successo che non dipende dalla fortuna di una prova sola.
"""

import numpy as np
import pygame

from ..evo.trial import TRIAL_SEED, Trial
from ..net.controllers import NeuralController
from ..sim.world import World
from .render import BORDER, DEAD, DIM, PLAYER, TARGET, TEXT, WALL, Renderer

SPEEDS = (1, 2, 4, 8, 16, 64)
HOLD = 45  # fotogrammi di attesa a fine arena prima della successiva
TRIAL_BUDGET = 0.03  # s di calcolo per fotogramma dedicati alla valutazione


class TestScreen:
    def __init__(self, app, cfg, selection, back):
        self.app, self.cfg, self.back = app, cfg, back
        self.genome, self.name, self.title = selection
        self.world = World(cfg, 1, TRIAL_SEED)
        self.ctrl = NeuralController(cfg, 1)
        self.ctrl.brains.set_genome(self.genome.take([0]))
        self.renderer = Renderer(self.world, app.screen)
        self.arena = 0
        self.results: list[tuple[bool, float]] = []  # per arena vista: (completato, tempo)
        self.recorded = False
        self.speed_i, self.hold = 0, 0
        self.paused = self.manual = False
        self.auto = self.show_rays = True
        self.trial: Trial | None = None

    def _new_arena(self, k: int) -> None:
        self.arena = k
        self.world.seed = TRIAL_SEED + k
        self.world.reset()
        self.ctrl.reset()
        self.recorded, self.hold = False, 0

    @property
    def over(self) -> bool:
        w = self.world
        return not (w.alive & ~w.done).any() or w.t >= self.cfg.life_max

    def handle(self, ev) -> None:
        if ev.type != pygame.KEYDOWN:
            return
        k = ev.key
        if k == pygame.K_ESCAPE:
            self.app.show(self.back)
        elif k == pygame.K_SPACE:
            self.paused = not self.paused
        elif k == pygame.K_n:
            self._new_arena(self.arena + 1)
        elif k == pygame.K_r:
            self._new_arena(self.arena)
        elif k == pygame.K_a and not self.manual:  # con la guida manuale A sterza
            self.auto = not self.auto
        elif k == pygame.K_v:
            self.show_rays = not self.show_rays
        elif k == pygame.K_m:
            self.manual = not self.manual
        elif k == pygame.K_e:
            self.trial = Trial(self.cfg, self.genome)
        elif k in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
            self.speed_i = min(self.speed_i + 1, len(SPEEDS) - 1)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed_i = max(self.speed_i - 1, 0)

    def update(self) -> None:
        if self.trial is not None and not self.trial.finished:
            self.trial.run(TRIAL_BUDGET)
        if self.paused:
            return
        if self.over:
            if not self.recorded:
                self.results.append((bool(self.world.done[0]), self.world.t))
                self.recorded, self.hold = True, HOLD
            elif self.hold > 0:
                self.hold -= 1
            elif self.auto:
                self._new_arena(self.arena + 1)
            return
        keys = pygame.key.get_pressed()
        hand = np.array([[(keys[pygame.K_UP] or keys[pygame.K_w]) - (keys[pygame.K_DOWN] or keys[pygame.K_s]),
                          (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])]],
                        dtype=float)
        for _ in range(SPEEDS[self.speed_i]):
            actions = self.ctrl.act(self.world)
            self.world.step(hand if self.manual else actions)
            if self.over:
                break

    def close(self) -> None:
        self.back.close()

    def draw(self, s) -> None:
        w = self.world
        if w.done[0]:
            status = f"COMPITO COMPLETATO in {w.t:.1f}s"
        elif not w.alive[0]:
            status = f"MORTO dopo {w.t:.1f}s"
        else:
            status = "in corsa"
        hud = [
            (f"PROVA: {self.title}", PLAYER),
            (f"arena {self.arena}   t={w.t:5.1f}s   x{SPEEDS[self.speed_i]}"
             f"{'   PAUSA' if self.paused else ''}{'   guidi tu' if self.manual else ''}", TEXT),
            (f"{status}   energia {w.energy[0]:5.1f}   v={w.speed[0]:5.1f} u/s", PLAYER),
            ("", TEXT),
            ("Spazio pausa | +/- velocita' | N arena successiva | R rifai questa", DIM),
            (f"A passa da sola alla successiva ({'si' if self.auto else 'no'}) | V raggi | M guida manuale", DIM),
            ("E valutazione: 20 arene x 50 copie | Esc indietro", DIM),
        ]
        status = self.app.run_status()
        if status:
            hud.append((status, DIM))
        self.renderer.draw(0, hud, self.ctrl, self.show_rays)
        self._draw_results(s)

    def _draw_results(self, s) -> None:
        r = self.renderer
        small = self.app.small
        x, y = r.panel.x + 10, r.net_h + 8
        right = r.panel.right - 10
        pygame.draw.line(s, BORDER, (r.panel.x, r.net_h), (r.panel.right, r.net_h), 1)
        done = sum(ok for ok, _ in self.results)
        n = len(self.results)
        line = f"arene viste: {n}, completate {done}" + (f" ({done / n:.0%})" if n else "")
        s.blit(small.render(line, True, TEXT), (x, y))
        for k, (ok, _) in enumerate(self.results[-48:]):  # un quadratino per arena
            pygame.draw.rect(s, TARGET if ok else DEAD, (x + 9 * k, y + 18, 7, 7))

        y += 40
        t = self.trial
        if t is None:
            s.blit(small.render("E: valutazione su 20 arene fisse, 50 copie per arena", True, DIM), (x, y))
            s.blit(small.render("(le stesse per ogni rete: confronto alla pari)", True, DIM), (x, y + 16))
            return
        state = "fatto" if t.finished else f"in corso, arena {len(t.completed) + 1}/{t.arenas}"
        s.blit(small.render(f"valutazione ({state}): completa {t.rate:.0%} delle prove", True, TEXT), (x, y))
        if t.times:
            s.blit(small.render(f"tempo medio per arrivare: {np.mean(t.times):.1f}s", True, DIM), (x, y + 16))
        # Una barra per arena: quota di copie che hanno completato.
        box = pygame.Rect(x, y + 40, right - x, s.get_height() - y - 60)
        pygame.draw.line(s, BORDER, (box.x, box.bottom), (box.right, box.bottom), 1)
        bw = box.w / t.arenas
        for k, c in enumerate(t.completed):
            h = box.h * c / t.clones
            pygame.draw.rect(s, WALL, (box.x + k * bw + 1, box.bottom - h, bw - 2, h))
        s.blit(small.render("successo per arena", True, DIM), (box.x, box.bottom + 2))
