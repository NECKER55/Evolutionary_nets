"""La schermata dell'arena continua (docs/SPEC_ARENA.md, sezione 10): una
vista che segue un essere, con zoom e scia, la mappa intera del mondo e le
misure al minuto. Il mondo è un toro: la vista e la mappa mostrano gli
oggetti dalla parte più vicina."""

import time
from pathlib import Path

import numpy as np
import pygame

from .launcher import Screen
from .render import BG, BORDER, DIM, FOOD, PLAYER, TARGET, TEXT

PANEL_W = 380
BEING = (110, 200, 120)  # gli esseri: verdi (0,25 sull'asse della luce)
PATCH = (60, 90, 50)
SPEEDS = (1, 4, 16)  # passi per fotogramma: tempo reale, x4, x16
TURBO_BUDGET = 0.08  # s di calcolo per fotogramma in turbo
BACKGROUND_STEPS = 6  # passi per fotogramma mentre si guarda un'altra schermata
TRAIL = 600


class ArenaScreen(Screen):
    """La corsa nell'arena continua. `evo` è l'ecologia (evo/ecology.py)."""

    def __init__(self, app, eco, save: str, minutes: int = 0):
        self.app, self.evo, self.cfg, self.save_path = app, eco, eco.cfg, save
        self.world = eco.world
        self.goal = eco.world.t + 60.0 * minutes if minutes else None
        self.paused = False
        self.turbo = False
        self.speed_i = 0
        self.zoom = 3.0  # pixel per unità
        self.follow = self._oldest()
        self.trail: list[tuple[float, float]] = []
        self.message = ""
        self.saved_t = eco.world.t
        self.saved_exams = len(eco.exams)
        self.busy = 0.0

    # --- il tempo -------------------------------------------------------------

    def _oldest(self) -> int:
        alive = np.flatnonzero(self.world.alive)
        return int(alive[np.argmax(self.world.age[alive])]) if len(alive) else -1

    def _steps(self, k: int = 0, budget: float = 0.0) -> None:
        """k passi, o quanti ne stanno in `budget` secondi."""
        eco, start = self.evo, time.perf_counter()
        n = 0
        while (n < k or time.perf_counter() - start < budget) and not self._finished():
            eco.step()
            n += 1
            if k and n >= k and not budget:
                break
        self.busy += time.perf_counter() - start
        if self.follow < 0 or not self.world.alive[self.follow]:
            self.follow = self._oldest()
            self.trail = []

    def _finished(self) -> bool:
        if self.goal is not None and self.world.t >= self.goal - 1e-9:
            if not self.paused:
                self.paused = True
                self._save()
                self.message = f"minuti finiti: salvato in {self.save_path}"
            return True
        return False

    def background(self, budget: float) -> None:
        if not self.paused:
            self._steps(BACKGROUND_STEPS)

    def update(self) -> None:
        if self.paused:
            return
        if self.turbo:
            self._steps(budget=TURBO_BUDGET)
        else:
            self._steps(SPEEDS[self.speed_i])
        if self.follow >= 0:
            p = self.world.pos[self.follow]
            self.trail.append((float(p[0]), float(p[1])))
            del self.trail[:-TRAIL]

    # --- salvataggi e uscita --------------------------------------------------

    def _save(self) -> None:
        self.evo.save(self.save_path)
        self.saved_t, self.saved_exams = self.world.t, len(self.evo.exams)
        self.message = f"salvato al minuto {self.evo.generation} in {self.save_path}"

    def _leave(self) -> str:
        self.app.active_run = None
        if self.evo.exam is not None:
            self.evo.exam.close()
            self.evo.exam = None
        if self.world.t != self.saved_t or len(self.evo.exams) != self.saved_exams:
            self._save()
            return self.message
        return ""

    def close(self) -> None:
        self._leave()

    # --- comandi --------------------------------------------------------------

    def handle(self, ev) -> None:
        if ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_ESCAPE:
                self.app.launcher.returned(self._leave())
            elif k == pygame.K_SPACE:
                self.paused = not self.paused
            elif k == pygame.K_s:
                self._save()
            elif k == pygame.K_z:
                self.turbo = not self.turbo
            elif k in (pygame.K_1, pygame.K_2, pygame.K_3):
                self.speed_i, self.turbo = k - pygame.K_1, False
            elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.zoom = min(self.zoom * 1.25, 20.0)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.zoom = max(self.zoom / 1.25, 0.3)
            elif k == pygame.K_TAB:
                self._next()
            elif k == pygame.K_o:
                self.follow, self.trail = self._oldest(), []
            elif k == pygame.K_e:
                if self.evo.start_exam():
                    self.message = "esame avviato su 20 adulti a caso (in sottofondo)"
                else:
                    self.message = "esame già in corso, o nessun adulto"
        elif ev.type == pygame.MOUSEWHEEL:
            self.zoom = float(np.clip(self.zoom * (1.25 ** ev.y), 0.3, 20.0))
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and ev.pos[0] < self._view_w():
            self._pick(ev.pos)

    def _next(self) -> None:
        """Il prossimo essere vivo, dal più anziano in giù."""
        alive = np.flatnonzero(self.world.alive)
        if not len(alive):
            return
        order = alive[np.argsort(-self.world.age[alive])]
        here = np.flatnonzero(order == self.follow)
        self.follow = int(order[(here[0] + 1) % len(order)] if len(here) else order[0])
        self.trail = []

    def _pick(self, pos) -> None:
        """Segue l'essere vivo più vicino al punto cliccato."""
        w = self.world
        alive = np.flatnonzero(w.alive)
        if not len(alive):
            return
        world_pt = self._unproject(np.array(pos, dtype=float))
        d = np.hypot(*self._near(w.pos[alive] - world_pt).T)
        self.follow, self.trail = int(alive[np.argmin(d)]), []

    # --- disegno --------------------------------------------------------------

    def _view_w(self) -> int:
        return self.app.screen.get_width() - PANEL_W

    def _centre(self) -> np.ndarray:
        if self.follow >= 0:
            return self.world.pos[self.follow].copy()
        return np.array([self.world.W / 2, self.world.H / 2])

    def _near(self, d: np.ndarray) -> np.ndarray:
        """Differenze riportate alla parte più vicina del toro."""
        size = np.array([self.world.W, self.world.H])
        return d - size * np.floor(d / size + 0.5)

    def _project(self, pts: np.ndarray) -> np.ndarray:
        h = self.app.screen.get_height()
        rel = self._near(pts - self._centre())
        return rel * self.zoom + [self._view_w() / 2, h / 2]

    def _unproject(self, px: np.ndarray) -> np.ndarray:
        h = self.app.screen.get_height()
        return self._centre() + (px - [self._view_w() / 2, h / 2]) / self.zoom

    def draw(self, s) -> None:
        app, w, eco = self.app, self.world, self.evo
        task = w.task
        s.fill(BG)
        vw, h = self._view_w(), s.get_height()
        view = pygame.Rect(0, 0, vw, h)
        s.set_clip(view)
        # Le macchie fertili: ombre larghe quanto il loro raggio tipico.
        fert = task.patches.fertility(w.t)
        for c, f in zip(self._project(task.patches.centre), fert):
            r = int(self.cfg.arena_patch_sigma * self.zoom)
            if -r < c[0] < vw + r and -r < c[1] < h + r:
                shade = tuple(int(BG[i] + (PATCH[i] - BG[i]) * f) for i in range(3))
                pygame.draw.circle(s, shade, c.astype(int), r)
        # Il cibo.
        on = np.flatnonzero(task.food_on)
        px = self._project(task.food[on])
        inside = (px[:, 0] > -5) & (px[:, 0] < vw + 5) & (px[:, 1] > -5) & (px[:, 1] < h + 5)
        rf = max(1, int(task.food_r * self.zoom))
        for p in px[inside]:
            pygame.draw.circle(s, FOOD, p.astype(int), rf)
        # La scia di chi si segue.
        if len(self.trail) > 1:
            pts = self._project(np.array(self.trail))
            for a, b in zip(pts[:-1], pts[1:]):
                if np.abs(a - b).max() < 50:  # non attraverso il bordo del toro
                    pygame.draw.line(s, (90, 80, 40), a, b, 1)
        # Gli esseri.
        alive = np.flatnonzero(w.alive)
        px = self._project(w.pos[alive])
        inside = (px[:, 0] > -5) & (px[:, 0] < vw + 5) & (px[:, 1] > -5) & (px[:, 1] < h + 5)
        rb = max(2, int(self.cfg.agent_radius * self.zoom))
        for i, p in zip(alive[inside], px[inside]):
            color = PLAYER if i == self.follow else BEING
            c = p.astype(int)
            pygame.draw.circle(s, color, c, rb)
            nose = c + (rb + 3) * np.array([np.cos(w.heading[i]), np.sin(w.heading[i])])
            pygame.draw.line(s, color, c, nose.astype(int), 1)
        s.set_clip(None)
        pygame.draw.line(s, BORDER, (vw, 0), (vw, h))
        self._panel(s, vw + 14, app)

    def _panel(self, s, x: int, app) -> None:
        w, eco, cfg = self.world, self.evo, self.cfg
        # La mappa intera.
        mw = PANEL_W - 28
        mh = int(mw * w.H / w.W)
        box = pygame.Rect(x, 12, mw, mh)
        pygame.draw.rect(s, (18, 20, 26), box)
        k = mw / w.W
        for c, f in zip(w.task.patches.centre, w.task.patches.fertility(w.t)):
            pygame.draw.circle(s, tuple(int(BG[i] + (PATCH[i] - BG[i]) * f) for i in range(3)),
                               (int(x + c[0] * k), int(12 + c[1] * k)), max(2, int(cfg.arena_patch_sigma * k)))
        for p in w.pos[w.alive]:
            s.set_at((int(x + p[0] * k), int(12 + p[1] * k)), BEING)
        if self.follow >= 0:
            p = w.pos[self.follow]
            half = np.array([self._view_w(), s.get_height()]) / self.zoom / 2 * k
            pygame.draw.rect(s, PLAYER, (x + p[0] * k - half[0], 12 + p[1] * k - half[1], 2 * half[0], 2 * half[1]),
                             1)
        pygame.draw.rect(s, BORDER, box, 1)
        y = 12 + mh + 12

        def line(text, color=TEXT, font=None):
            nonlocal y
            s.blit((font or app.small).render(text, True, color), (x, y))
            y += 17

        state = "IN PAUSA" if self.paused else ("TURBO" if self.turbo else f"x{SPEEDS[self.speed_i]}")
        minutes, secs = divmod(int(w.t), 60)
        line(f"arena continua   {minutes}:{secs:02d} di mondo   {state}", PLAYER, app.font)
        alive = np.flatnonzero(w.alive)
        line(f"vivi {len(alive)} su {cfg.arena_cap}   cibo {int(w.task.food_on.sum())}")
        line(f"nati {eco.totals['nati']} (con partner {eco.totals['con partner']}), "
             f"colonizzatori {eco.totals['colonizzatori']}")
        line(f"morti di fame {w.deaths['fame']}, di vecchiaia {w.deaths['vecchiaia']}")
        if len(alive):
            line(f"età media {w.age[alive].mean():.0f} s, massima {w.age[alive].max():.0f} s")
        if eco.history:
            r = eco.history[-1]
            line(f"ultimo minuto: {r['births']} nati, {r['eaten']} cibi mangiati", DIM)
            if "species" in r:
                line(f"sensi usati {r.get('used', 0):.2f}, neuroni {r.get('neurons', 0):.2f}, "
                     f"gruppi {r['species']}", DIM)
        if eco.exam is not None:
            line(f"esame in corso ({eco.exam.progress:.0%})", TARGET)
        elif eco.exams:
            x = eco.exams[-1]
            line(f"ultimo esame (min. {x['generation']}): media {x['score']:.0%}", TARGET)
        y += 8
        if self.follow >= 0:
            i = self.follow
            line(f"seguito: #{eco.ids[i]}", PLAYER, app.font)
            line(f"età {w.age[i]:.0f} s   energia {w.energy[i]:.0f}   cibi mangiati {w.task.eaten[i]}")
            b = eco.ctrl.brains
            line(f"recettori {int(b.genome.r_active[i].sum())}   neuroni {int(b.active[i].sum())}   "
                 f"collegamenti {int(b.links[i].sum())}")
        y += 8
        for text in ("Spazio pausa   S salva   Esc menu (salva)",
                     "1/2/3 velocità   Z turbo   E esame",
                     "rotella o +/- zoom   Tab prossimo   O più anziano",
                     "clic su un essere: seguilo"):
            line(text, DIM)
        if self.message:
            y += 6
            line(self.message[:60], TARGET)
