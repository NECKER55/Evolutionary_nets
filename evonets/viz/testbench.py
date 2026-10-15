"""Banco di prova: una rete sola, o un gruppo (per esempio i sopravvissuti),
nell'arena fuori dall'evoluzione.

Si guardano gli agenti arena dopo arena: le arene sono sempre le stesse per
tutte le reti, così si confrontano alla pari. Non nasce nessuno e nessuno viene
eliminato: si osserva soltanto. A richiesta si lancia una valutazione, la
stessa prova su 20 arene con più copie di ogni rete, per una percentuale di
successo che non dipende dalla fortuna di una prova sola.
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
    """`genomes`: una rete per riga; `labels`: il nome di ciascuna (per
    esempio "#723"); `title`: cosa si sta provando, in parole."""

    def __init__(self, app, cfg, genomes, labels: list[str], title: str, back):
        self.app, self.cfg, self.back = app, cfg, back
        self.genomes, self.labels, self.title = genomes, labels, title
        self.k = len(genomes)
        self.world = World(cfg, self.k, TRIAL_SEED)
        self.ctrl = NeuralController(cfg, self.k)
        self.ctrl.brains.set_genome(genomes.take(np.arange(self.k)))
        self.ctrl.reset(TRIAL_SEED)
        self.renderer = Renderer(self.world, app.screen)
        self.renderer.name_of = lambda i: self.labels[i]
        self.arena = 0
        self.focus = 0
        self.wins = np.zeros(self.k, dtype=int)  # arene viste completate, per rete
        self.seen = 0  # arene viste fino in fondo
        self.history: list[int] = []  # per arena vista: quanti hanno completato
        self.recorded = False
        self.speed_i, self.hold = 0, 0
        self.paused = self.manual = False
        self.auto = self.show_rays = True
        self.trial: Trial | None = None

    def _new_arena(self, k: int) -> None:
        self.arena = k
        self.world.seed = TRIAL_SEED + k
        self.world.reset()
        self.ctrl.reset(self.world.seed)
        self.recorded, self.hold = False, 0

    @property
    def over(self) -> bool:
        w = self.world
        return not (w.alive & ~w.done).any() or w.t >= self.cfg.life_max

    def handle(self, ev) -> None:
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.renderer.arena.collidepoint(ev.pos):
            click = np.array(ev.pos) / self.renderer.scale
            self.focus = int(np.argmin(np.linalg.norm(self.world.pos - click, axis=1)))
            return
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
        elif k == pygame.K_TAB:
            self.focus = (self.focus + 1) % self.k
        elif k == pygame.K_e:
            # Con tante reti bastano meno copie a testa: il tempo resta ragionevole.
            self.trial = Trial(self.cfg, self.genomes, clones=50 if self.k == 1 else 10)
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
                self.wins += self.world.done
                self.seen += 1
                self.history.append(int(self.world.done.sum()))
                self.recorded, self.hold = True, HOLD
            elif self.hold > 0:
                self.hold -= 1
            elif self.auto:
                self._new_arena(self.arena + 1)
            return
        keys = pygame.key.get_pressed()
        hand = np.array([(keys[pygame.K_UP] or keys[pygame.K_w]) - (keys[pygame.K_DOWN] or keys[pygame.K_s]),
                         (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])],
                        dtype=float)
        for _ in range(SPEEDS[self.speed_i]):
            actions = self.ctrl.act(self.world)
            if self.manual:
                actions = actions.copy()
                actions[self.focus] = hand
            self.world.step(actions, self.ctrl.upkeep)
            if self.over:
                break

    def close(self) -> None:
        self.back.close()

    def draw(self, s) -> None:
        w, i = self.world, self.focus
        if w.done[i]:
            status = "COMPITO COMPLETATO"
        elif not w.alive[i]:
            status = "MORTO"
        else:
            status = "in corsa"
        hud = [
            (f"PROVA: {self.title}", PLAYER),
            (f"arena di prova {self.arena}   t={w.t:5.1f}s   x{SPEEDS[self.speed_i]}"
             f"{'   PAUSA' if self.paused else ''}{'   guidi tu' if self.manual else ''}", TEXT),
        ]
        if self.k > 1:
            hud.append((f"in questa arena: completato {int(w.done.sum())}/{self.k}   "
                        f"morti {int((~w.alive).sum())}   in corsa {int((w.alive & ~w.done).sum())}", TEXT))
        hud += [
            (f"seguito: {self.labels[i]}: {status}   energia {w.energy[i]:5.1f}   v={w.speed[i]:5.1f} u/s", PLAYER),
            ("nessun figlio, nessuna eliminazione: si osserva soltanto", DIM),
            ("", TEXT),
            ("Spazio pausa | +/- velocita' | N arena successiva | R rifai questa", DIM),
            (f"A passa da sola alla successiva ({'si' if self.auto else 'no'}) | V raggi | M guida manuale", DIM),
            (("clic o Tab: segui un altro | " if self.k > 1 else "") + "E valutazione su 20 arene | Esc indietro", DIM),
        ]
        status = self.app.run_status()
        if status:
            hud.append((status, DIM))
        self.renderer.draw(i, hud, self.ctrl, self.show_rays)
        self._draw_results(s)

    def _draw_results(self, s) -> None:
        r, small = self.renderer, self.app.small
        x, y = r.panel.x + 10, r.net_h + 8
        right = r.panel.right - 10
        pygame.draw.line(s, BORDER, (r.panel.x, r.net_h), (r.panel.right, r.net_h), 1)
        if self.k == 1:
            self._draw_single(s, x, y, right)
            return
        # Gruppo: una riga per rete, le migliori in alto.
        t = self.trial
        rates = t.rates() if t is not None else None
        state = "" if t is None else (" fatta" if t.finished else f" in corso ({len(t.completed)}/{t.arenas})")
        head = f"{'rete':>6} {'arene viste':>11}" + (f" {'valutaz.':>9}" if t else "")
        s.blit(small.render(f"risultati ({self.seen} arene viste{', valutazione' + state if t else ''})",
                            True, TEXT), (x, y))
        s.blit(small.render(head, True, DIM), (x, y + 16))
        note = "arene viste: completate/viste" + ("; valutaz.: % di 20 arene x 10 copie" if t else "")
        s.blit(small.render(note, True, BORDER), (x, s.get_height() - 34))
        key = rates if rates is not None else self.wins
        order = np.argsort(-key, kind="stable")
        rows = (s.get_height() - y - 56) // 15
        for j, k in enumerate(order[:rows]):
            seen = f"{self.wins[k]}/{self.seen}"
            line = f"{self.labels[k]:>6} {seen:>11}" + (f" {rates[k]:>9.0%}" if t else "")
            color = PLAYER if k == self.focus else TEXT
            s.blit(small.render(line, True, color), (x, y + 34 + 15 * j))
        if self.k > rows:
            s.blit(small.render(f"... e altre {self.k - rows}", True, DIM), (x, y + 34 + 15 * rows))
        if t is None:
            s.blit(small.render("E: valutazione, 20 arene x 10 copie a testa", True, DIM), (x, s.get_height() - 18))

    def _draw_single(self, s, x, y, right) -> None:
        small = self.app.small
        done = int(self.wins[0])
        line = f"arene viste: {self.seen}, completate {done}" + (f" ({done / self.seen:.0%})" if self.seen else "")
        s.blit(small.render(line, True, TEXT), (x, y))
        for k, c in enumerate(self.history[-48:]):  # un quadratino per arena
            pygame.draw.rect(s, TARGET if c else DEAD, (x + 9 * k, y + 18, 7, 7))
        y += 40
        t = self.trial
        if t is None:
            s.blit(small.render("E: valutazione su 20 arene fisse, 50 copie per arena", True, DIM), (x, y))
            s.blit(small.render("(le stesse per ogni rete: confronto alla pari)", True, DIM), (x, y + 16))
            return
        state = "fatta" if t.finished else f"in corso, arena {len(t.completed) + 1}/{t.arenas}"
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
