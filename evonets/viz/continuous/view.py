"""La schermata dell'arena continua (docs/SPEC_ARENA.md, sezione 10): una
vista che segue un essere, con zoom e scia, oppure libera (si sposta con le
frecce o cliccando sulla mappa intera), la mappa intera del mondo e le
misure al minuto. Il mondo è un toro: la vista e la mappa mostrano gli
oggetti dalla parte più vicina."""

import time
from pathlib import Path

import numpy as np
import pygame

from ...arenas.continuous.world import temperature
from ...brains.freeform.genome import live_links
from ..launcher import Screen
from ..render import BG, BORDER, DIM, FOOD, FRUIT, PLAYER, TARGET, TEXT

PANEL_W = 380
BEING = (110, 200, 120)  # gli esseri: verdi (0,25 sull'asse della luce)
PATCH = (60, 90, 50)
WATER = (60, 110, 220)  # le pozze
# I nomi brevi delle manopole, per il pannello (nell'ordine di ecology.DIALS).
DIAL_SHORT = ("cibo raro", "sete", "pozze lontane", "macchie esaurite", "frutti cattivi", "inversione", "predazione", "cibo da aprire",
              "notte", "siccità")
SPEEDS = (1, 2, 4, 8, 16)  # passi per fotogramma (a 60 fotogrammi al secondo: x1 = tempo reale)
TURBO_BUDGET = 0.08  # s di calcolo per fotogramma in turbo
MAX_BUDGET = 0.5  # s di calcolo per fotogramma alla massima velocità (si ridisegna 2 volte al secondo)
BACKGROUND_STEPS = 6  # passi per fotogramma mentre si guarda un'altra schermata
TRAIL = 600
PAN = 700.0  # pixel al secondo con le frecce (x3 con Maiusc)
MAP_TOP = 52  # sopra, i bottoni delle schede
TABS = (("mondo", "Mondo"), ("panoramica", "Panoramica"), ("reti", "Reti"))

def warmth(temp: float) -> tuple:
    """Lo sfondo per una temperatura: blu scuro al freddo, rosso scuro al caldo."""
    cold, hot = np.array([16, 22, 40]), np.array([44, 18, 16])
    return tuple(int(v) for v in BG + (np.where(temp < 0.5, cold, hot) - BG) * abs(temp - 0.5) * 2)


def light_rgb(q: float) -> tuple:
    """Il colore sullo schermo di un punto dell'asse della luce (0 blu, 0,25
    verde, 0,65 giallo, 0,85 rosso)."""
    stops = ((0.0, (70, 90, 255)), (0.25, (110, 220, 120)), (0.45, (200, 200, 200)), (0.65, (240, 210, 60)),
             (0.85, (240, 80, 70)), (1.0, (200, 40, 160)))
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if q <= b:
            t = (q - a) / (b - a)
            return tuple(int(ca[i] + (cb[i] - ca[i]) * t) for i in range(3))
    return stops[-1][1]


class ArenaScreen(Screen):
    """La corsa nell'arena continua. `evo` è l'ecologia (arenas/continuous/ecology.py)."""

    def __init__(self, app, eco, save: str, minutes: int = 0):
        self.app, self.evo, self.cfg, self.save_path = app, eco, eco.cfg, save
        self.world = eco.world
        self.goal = eco.world.t + 60.0 * minutes if minutes else None
        self.paused = False
        self.turbo = False
        self.full = False  # massima velocità
        self.show_help = False  # H: che cos'è ogni cosa
        self.rate = 0.0  # secondi di mondo per secondo vero (media mobile)
        self.tab = "mondo"  # V o clic: mondo, panoramica, reti
        self.tab_rects: dict = {}
        self.overview = self.explorer = None  # si creano la prima volta che servono
        self.speed_i = 0
        self.zoom = 3.0  # pixel per unità
        self.follow = self._oldest()
        self.camera: np.ndarray | None = None  # None = segue `follow`; altrimenti il centro della visuale libera
        self.map_rect = pygame.Rect(0, 0, 0, 0)  # la mappa intera, fissata da `draw`
        self.dragging = False
        self.last_frame = time.perf_counter()
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
        """k passi (insieme: sulla GPU girano in una volta), o quanti ne stanno
        in `budget` secondi, a blocchi di un secondo di mondo."""
        eco, start = self.evo, time.perf_counter()
        if budget:
            block = int(round(self.cfg.arena_tick / self.cfg.dt))
            while time.perf_counter() - start < budget and not self._finished():
                eco.advance(block)
        elif not self._finished():
            if self.goal is not None:
                k = max(1, min(k, int(round((self.goal - self.world.t) / self.cfg.dt))))
            eco.advance(k)
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
        now = time.perf_counter()
        real, self.last_frame = now - self.last_frame, now  # dal fotogramma prima
        self._pan(min(real, 0.1))
        if self.paused:
            self.rate = 0.0
            return
        t0 = self.world.t
        if self.full:
            self._steps(budget=MAX_BUDGET)
        elif self.turbo:
            self._steps(budget=TURBO_BUDGET)
        else:
            self._steps(SPEEDS[self.speed_i])
        self.rate = 0.8 * self.rate + 0.2 * (self.world.t - t0) / max(real, 1e-6)
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

    def _pages(self):
        from .pages import ArenaExplorer, ArenaOverview

        if self.overview is None:
            self.overview, self.explorer = ArenaOverview(self.app), ArenaExplorer(self.evo)
        return self.overview, self.explorer

    def handle(self, ev) -> None:
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for key, r in self.tab_rects.items():
                if r.collidepoint(ev.pos):
                    self.tab = key
                    return
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_v:
            keys = [k for k, _ in TABS]
            self.tab = keys[(keys.index(self.tab) + 1) % len(keys)]
            return
        if self.tab == "reti":
            ex = self._pages()[1]
            if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                slot = ex.chosen_slot()
                if slot is None:
                    self.message = "quell'essere non è più vivo"
                else:
                    self.follow, self.trail, self.camera, self.tab = slot, [], None, "mondo"
                return
            if ex.handle(ev):
                return
        if ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_ESCAPE:
                self.app.launcher.returned(self._leave())
            elif k == pygame.K_SPACE:
                self.paused = not self.paused
            elif k == pygame.K_s and ev.mod & pygame.KMOD_CTRL:  # S da sola sposta la visuale (WASD)
                self._save()
            elif k == pygame.K_z:  # come nelle altre corse: massima velocità
                self.full, self.turbo = not self.full, False
            elif k == pygame.K_t:
                self.turbo, self.full = not self.turbo, False
            elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.turbo = self.full = False
                self.speed_i = min(self.speed_i + 1, len(SPEEDS) - 1)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.turbo = self.full = False
                self.speed_i = max(self.speed_i - 1, 0)
            elif k == pygame.K_PAGEUP:
                self.zoom = min(self.zoom * 1.25, 20.0)
            elif k == pygame.K_PAGEDOWN:
                self.zoom = max(self.zoom / 1.25, 0.3)
            elif k == pygame.K_h:
                self.show_help = not self.show_help
            elif k == pygame.K_TAB and self.tab == "mondo":
                self._next()
                self.camera = None
            elif k == pygame.K_o:
                self.follow, self.trail, self.camera = self._oldest(), [], None
            elif k == pygame.K_f:
                self.camera = None  # di nuovo dietro all'essere scelto
            elif k == pygame.K_e:
                if self.evo.start_exam():
                    self.message = "esame avviato su 20 adulti a caso (in sottofondo)"
                else:
                    self.message = "esame già in corso, o nessun adulto"
        elif ev.type == pygame.MOUSEWHEEL:
            self.zoom = float(np.clip(self.zoom * (1.25 ** ev.y), 0.3, 20.0))
        elif self.tab != "mondo":
            return
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.map_rect.collidepoint(ev.pos):
            self.dragging = True
            self._look_at_map(ev.pos)
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            self._look_at_map(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.dragging = False

    def _look_at_map(self, px) -> None:
        """La visuale libera si centra sul punto cliccato della mappa intera."""
        r, w = self.map_rect, self.world
        x = (min(max(px[0], r.left), r.right) - r.left) / r.w * w.W
        y = (min(max(px[1], r.top), r.bottom) - r.top) / r.h * w.H
        self.camera = np.array([x, y]) % [w.W, w.H]

    def _pan(self, elapsed: float) -> None:
        """Frecce o WASD: la visuale diventa libera e si sposta."""
        if self.tab != "mondo":
            return
        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
        dy = (keys[pygame.K_DOWN] or keys[pygame.K_s]) - (keys[pygame.K_UP] or keys[pygame.K_w])
        if not dx and not dy or pygame.key.get_mods() & pygame.KMOD_CTRL:
            return
        fast = 3.0 if pygame.key.get_mods() & pygame.KMOD_SHIFT else 1.0
        if self.camera is None:
            self.camera = self._centre()
        step = PAN * fast * elapsed / self.zoom
        self.camera = (self.camera + step * np.array([dx, dy], dtype=float)) % [self.world.W, self.world.H]

    def _next(self) -> None:
        """Il prossimo essere vivo, dal più anziano in giù."""
        alive = np.flatnonzero(self.world.alive)
        if not len(alive):
            return
        order = alive[np.argsort(-self.world.age[alive])]
        here = np.flatnonzero(order == self.follow)
        self.follow = int(order[(here[0] + 1) % len(order)] if len(here) else order[0])
        self.trail = []

    # --- disegno --------------------------------------------------------------

    def _view_w(self) -> int:
        return self.app.screen.get_width() - PANEL_W

    def _centre(self) -> np.ndarray:
        if self.camera is not None:
            return self.camera.copy()
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
        if self.tab == "mondo":
            self._draw_world(s)
        else:
            self._draw_page(s)
        self._draw_tabs(s)
        if self.show_help:
            self._help(s, self.app)

    def _draw_tabs(self, s) -> None:
        """I bottoni delle schede, in alto a destra."""
        app = self.app
        self.tab_rects = {}
        x = s.get_width() - 14
        for key, label in reversed(TABS):
            img = app.font.render(label, True, BG if self.tab == key else TEXT)
            r = pygame.Rect(0, 10, img.get_width() + 22, 30)
            r.right = x
            pygame.draw.rect(s, PLAYER if self.tab == key else (32, 35, 46), r, border_radius=4)
            pygame.draw.rect(s, PLAYER, r, 1, border_radius=4)
            s.blit(img, img.get_rect(center=r.center))
            self.tab_rects[key] = r
            x = r.x - 8

    def _draw_page(self, s) -> None:
        """Panoramica o reti, a tutto schermo, con la riga di stato in alto."""
        app, w = self.app, self.world
        W, H = s.get_size()
        s.fill(BG)
        minutes, secs = divmod(int(w.t), 60)
        pace = "IN PAUSA" if self.paused else ("MASSIMA" if self.full else "TURBO" if self.turbo
                                               else f"x{SPEEDS[self.speed_i]}") + f" ({self.rate:.1f} s/s)"
        s.blit(app.big.render("Arena continua", True, PLAYER), (16, 12))
        s.blit(app.small.render(f"{minutes}:{secs:02d} di mondo   {pace}   {int(w.alive.sum())} vivi su "
                                f"{self.cfg.arena_cap}   rete {self.cfg.net_type}, genoma {self.cfg.genome}"
                                + ("   GPU" if self.evo.gpu is not None else ""), True, TEXT), (250, 20))
        pygame.draw.line(s, BORDER, (0, 50), (W, 50), 1)
        overview, ex = self._pages()
        if self.tab == "panoramica":
            overview.draw(s, pygame.Rect(12, 60, W - 24, H - 60 - 40), self.evo)
            hints = "V schede | E esame | H aiuto | Spazio pausa | +/- T Z velocità | Ctrl+S salva | Esc menu"
        else:
            ex.refresh()
            ex.place((16, 58), pygame.Rect(10, 84, 300, H - 84 - 44), pygame.Rect(320, 56, 760, H - 56 - 40),
                     pygame.Rect(1100, 106, W - 1106, H - 106 - 50))
            pygame.draw.line(s, BORDER, (314, 82), (314, H - 40), 1)
            pygame.draw.line(s, BORDER, (1090, 56), (1090, H - 40), 1)
            ex.draw(s, app.font, app.small)
            hints = ("V schede | 1/2/3 elenchi | R aggiorna | Invio segui nel mondo | P rami morti | "
                     "clic su un neurone: i suoi collegamenti | Spazio pausa | Esc menu")
        pygame.draw.line(s, BORDER, (0, H - 30), (W, H - 30), 1)
        s.blit(app.small.render(hints, True, DIM), (12, H - 22))
        if self.message:
            s.blit(app.small.render(self.message[:150], True, TARGET), (W // 2, H - 22))

    def _draw_world(self, s) -> None:
        app, w, eco = self.app, self.world, self.evo
        task = w.task
        s.fill(BG)
        vw, h = self._view_w(), s.get_height()
        view = pygame.Rect(0, 0, vw, h)
        s.set_clip(view)
        if self.cfg.arena_temperature > 0:  # il clima: blu dove fa freddo, rosso dove fa caldo
            for row in range(0, h, 8):
                y = self._unproject(np.array([0.0, row + 4.0]))[1] % w.H
                temp = float(temperature(self.cfg, np.array([0.0, y]), w.t))
                pygame.draw.rect(s, warmth(temp), (0, row, vw, 8))
        # Le macchie fertili: ombre larghe quanto il loro raggio tipico.
        fert = task.patches.fertility(w.t)
        for c, f in zip(self._project(task.patches.centre), fert):
            r = int(self.cfg.arena_patch_sigma * self.zoom)
            if -r < c[0] < vw + r and -r < c[1] < h + r:
                shade = tuple(int(BG[i] + (PATCH[i] - BG[i]) * f) for i in range(3))
                pygame.draw.circle(s, shade, c.astype(int), r)
        # Le pozze: il raggio cala man mano che si beve.
        pools = w.pools
        for c, r in zip(self._project(pools.centre[pools.on]), pools.radius()[pools.on] * self.zoom):
            if -r < c[0] < vw + r and -r < c[1] < h + r:
                pygame.draw.circle(s, WATER, c.astype(int), max(2, int(r)))
                pygame.draw.circle(s, (150, 190, 250), c.astype(int), max(2, int(r)), 1)
        # Il cibo: rosso o giallo; per chi guarda, un anello scuro sui frutti
        # cattivi, un bordo spesso sul cibo duro; il cibo grande è più grosso.
        on = np.flatnonzero(task.food_on)
        px = self._project(task.food[on])
        inside = (px[:, 0] > -8) & (px[:, 0] < vw + 8) & (px[:, 1] > -8) & (px[:, 1] < h + 8)
        for k, p in zip(on[inside], px[inside]):
            c = p.astype(int)
            rf = max(1, int(task.radius[k] * self.zoom))
            pygame.draw.circle(s, FRUIT if task.color[k] else FOOD, c, rf)
            if task.kind[k] == 1:
                pygame.draw.circle(s, (250, 250, 250), c, rf, max(1, rf // 3))
            elif task.kind[k] == 2:
                pygame.draw.circle(s, (250, 250, 250), c, rf, 1)
            if task.bad[k]:
                pygame.draw.circle(s, (40, 20, 20), c, rf + 2, 2)
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
        sizes = eco.sizes()
        glow = getattr(w, "glow", np.zeros(w.n))
        bite = getattr(w, "bite", np.zeros(w.n))
        halo = pygame.Surface((vw, h), pygame.SRCALPHA)
        for i, p in zip(alive[inside], px[inside]):
            c = p.astype(int)
            rb = max(2, int(self.cfg.agent_radius * np.sqrt(sizes[i]) * self.zoom))
            if glow[i] > 0.05:  # la luce del segnale, del suo colore
                pygame.draw.circle(halo, (*light_rgb(eco.sig_color[i]), int(160 * glow[i])), c, rb * 3)
            color = PLAYER if i == self.follow else BEING
            pygame.draw.circle(s, color, c, rb)
            nose = c + (rb + 3) * np.array([np.cos(w.heading[i]), np.sin(w.heading[i])])
            pygame.draw.line(s, (240, 60, 60) if bite[i] > 0.5 else color, c, nose.astype(int), 2 if bite[i] > 0.5 else 1)
        s.blit(halo, (0, 0))
        day = 1.0 - 0.85 * self.cfg.arena_night * (0.5 - 0.5 * np.cos(2 * np.pi * w.t / self.cfg.arena_day))
        if day < 0.99:  # la notte
            dark = pygame.Surface((vw, h), pygame.SRCALPHA)
            dark.fill((0, 0, 20, int(200 * (1.0 - day))))
            s.blit(dark, (0, 0))
        s.set_clip(None)
        pygame.draw.line(s, BORDER, (vw, 0), (vw, h))
        self._panel(s, vw + 14, app)

    def _panel(self, s, x: int, app) -> None:
        w, eco, cfg = self.world, self.evo, self.cfg
        # La mappa intera.
        mw = PANEL_W - 28
        mh = int(mw * w.H / w.W)
        box = pygame.Rect(x, MAP_TOP, mw, mh)
        self.map_rect = box
        pygame.draw.rect(s, (18, 20, 26), box)
        s.set_clip(box)  # niente esce dalla mappa
        k = mw / w.W
        if cfg.arena_temperature > 0:
            for row in range(0, mh, 4):
                temp = float(temperature(cfg, np.array([0.0, (row + 2) / mh * w.H]), w.t))
                pygame.draw.rect(s, warmth(temp), (x, MAP_TOP + row, mw, 4))
        for c, f in zip(w.task.patches.centre, w.task.patches.fertility(w.t)):
            pygame.draw.circle(s, tuple(int(BG[i] + (PATCH[i] - BG[i]) * f) for i in range(3)),
                               (int(x + c[0] * k), int(MAP_TOP + c[1] * k)), max(2, int(cfg.arena_patch_sigma * k)))
        for c, r in zip(w.pools.centre[w.pools.on], w.pools.radius()[w.pools.on]):
            pygame.draw.circle(s, WATER, (int(x + c[0] * k), int(MAP_TOP + c[1] * k)), max(2, int(r * k)))
        for p in w.pos[w.alive]:
            s.set_at((int(x + p[0] * k), int(MAP_TOP + p[1] * k)), BEING)
        if self.follow >= 0 or self.camera is not None:
            p = self._centre()
            half = np.array([self._view_w(), s.get_height()]) / self.zoom / 2 * k
            pygame.draw.rect(s, PLAYER, (x + p[0] * k - half[0], MAP_TOP + p[1] * k - half[1], 2 * half[0], 2 * half[1]),
                             1)
        s.set_clip(None)
        pygame.draw.rect(s, BORDER, box, 1)
        y = MAP_TOP + mh + 10

        def line(text, color=TEXT, font=None):
            nonlocal y
            s.blit((font or app.small).render(text, True, color), (x, y))
            y += 17

        if self.paused:
            state = "IN PAUSA"
        else:
            pace = "MASSIMA" if self.full else "TURBO" if self.turbo else f"x{SPEEDS[self.speed_i]}"
            state = f"{pace} ({self.rate:.1f} s/s)"
        if self.evo.gpu is not None:
            state += " GPU"
        minutes, secs = divmod(int(w.t), 60)
        line(f"{minutes}:{secs:02d} di mondo   {state}", PLAYER, app.font)
        alive = np.flatnonzero(w.alive)
        line(f"vivi {len(alive)} su {cfg.arena_cap}   cibo {int(w.task.food_on.sum())}"
             + (f"   pozze {int(w.pools.on.sum())}" if cfg.arena_water else ""))
        line(f"nati {eco.totals['nati']} (con partner {eco.totals['con partner']}), "
             f"colonizzatori {eco.totals['colonizzatori']}")
        line(f"morti: fame {w.deaths['fame']}, sete {w.deaths.get('sete', 0)}, "
             f"vecchiaia {w.deaths['vecchiaia']}, predati {w.deaths.get('predati', 0)}")
        if eco.history:
            r = eco.history[-1]
            age = f", età media {w.age[alive].mean():.0f} s" if len(alive) else ""
            line(f"ultimo minuto: {r['births']} nati, {r['eaten']} mangiati{age}", DIM)
            if "species" in r:
                line(f"sensi usati {r.get('used', 0):.2f}, neuroni {r.get('neurons', 0):.2f}, "
                     f"gruppi {r['species']}", DIM)
        # Le manopole del mondo che si complica.
        if eco.gpu is not None:
            y += 4
            from ...arenas.continuous.ecology import DIALS

            half = (PANEL_W - 28) // 2
            for j, (k, label) in enumerate(zip(DIALS, DIAL_SHORT)):  # su due colonne
                v = getattr(cfg, k)
                cx, cy = x + (j % 2) * half, y + (j // 2) * 15
                pygame.draw.rect(s, (40, 44, 56), (cx, cy + 4, 24, 8))
                pygame.draw.rect(s, TARGET if v > 0 else (40, 44, 56), (cx, cy + 4, int(24 * v), 8))
                s.blit(app.small.render(f"{label} {v:.0%}", True, TEXT if v > 0 else DIM), (cx + 28, cy))
            y += 15 * ((len(DIALS) + 1) // 2) + 2
        if eco.exam is not None:
            line(f"esame in corso ({eco.exam.progress:.0%})", TARGET)
        elif eco.exams:
            normal = [r for r in eco.exams if r.get("lesion") is None]
            if normal:
                last = normal[-1]  # (non `x`: è la colonna dove si scrive)
                probe = [r for r in eco.exams if r.get("lesion") == "plasticità" and r["generation"] == last["generation"]]
                extra = f", senza plasticità {probe[-1]['score']:.0%}" if probe else ""
                line(f"esame (min. {last['generation']}): {last['score']:.0%}{extra}", TARGET)
        y += 8
        if self.camera is not None:
            line("visuale libera: F per tornare a seguire", TARGET)
        if self.follow >= 0:
            i = self.follow
            line(f"seguito: #{eco.ids[i]}" + ("" if self.camera is None else " (fuori visuale)"), PLAYER, app.font)
            water = (f"   acqua {w.water[i] / (cfg.arena_water_max * eco.sizes()[i]):.0%}" if cfg.arena_water else "")
            line(f"età {w.age[i]:.0f} s   energia {w.energy[i]:.0f}{water}   cibi {w.task.eaten[i]}")
            if cfg.arena_temperature > 0:
                temp = float(temperature(cfg, w.pos[i], w.t))
                line(f"temperatura {temp:.2f}: consuma x{cfg.arena_temp_q ** (temp - 0.5):.2f}")
            g = eco.ctrl.brains.genome  # dal genoma: con la GPU il cervello compilato qui è di prima
            links = live_links(g.conn[i:i + 1], g.active[i:i + 1], g.r_active[i:i + 1]).sum()
            line(f"recettori {int(g.r_active[i].sum())}   neuroni {int(g.active[i].sum())}   "
                 f"collegamenti {int(links)}")
        y += 8
        # Legenda: che cosa si vede.
        for color, text in ((BEING, "essere (giallo: il seguito; muso rosso: morde)"),
                            (FOOD, "cibo rosso"), (FRUIT, "frutto giallo (anello scuro: cattivo)"),
                            (PATCH, "macchia fertile: più chiara, più cibo"),
                            (WATER, "pozza d'acqua (si restringe bevendo)"),
                            ((150, 60, 50), "sfondo rosso caldo, blu freddo")):
            pygame.draw.circle(s, color, (x + 5, y + 7), 4)
            s.blit(app.small.render(text, True, DIM), (x + 16, y))
            y += 16
        y += 6
        line("H: tasti e che cos'è ogni cosa   V: schede", DIM)
        if self.message:
            y += 6
            line(self.message[:60], TARGET)

    HELP = (
        ("Tasti", None),
        ("V schede (mondo, panoramica, reti)   Spazio pausa   +/- velocità   T turbo   Z massima", ""),
        ("frecce o WASD: sposta la visuale (Maiusc: veloce)   clic o trascina sulla mappa: lì", ""),
        ("rotella o Pag: zoom   F segui   Tab prossimo   O più anziano   E esame   Ctrl+S salva", ""),
        ("", ""),
        ("Che cosa si vede", None),
        ("Il mondo è un toro: chi esce da un lato rientra dall'altro. La vista e la mappa", ""),
        ("mostrano ogni cosa dalla parte più vicina.", ""),
        ("Essere (verde): un agente con il suo cervello. Il trattino indica dove guarda.", ""),
        ("Seguito (giallo): l'essere scelto, con la scia del suo percorso recente.", ""),
        ("Cibo (rosso): dà energia a chi lo tocca per primo. Ricresce soprattutto nelle macchie.", ""),
        ("Macchie fertili (ombre verdi): dove ricresce il cibo. Si spostano piano e cambiano", ""),
        ("con le stagioni: più chiare = più fertili adesso.", ""),
        ("Mappa intera (in alto a destra): tutto il mondo; il riquadro giallo è la vista.", ""),
        ("Grandezza di un essere: la sua taglia. Alone colorato: accende la sua luce (segnale).", ""),
        ("Muso rosso: sta mordendo. Cibo con bordo bianco spesso: duro (va morso a lungo);", ""),
        ("grande con bordo sottile: va morso in due. Anello scuro: frutto cattivo (fa male).", ""),
        ("Velo scuro: è notte (la luce del mondo cala, quella dei segnali no).", ""),
        ("Sfondo: il clima. Rosso = caldo (si consuma di più, le macchie producono di più),", ""),
        ("blu = freddo (si consuma meno, si produce meno). Fascia calda al centro del mondo.", ""),
        ("Pozze (blu): l'acqua. Si beve stando dentro; ogni sorso la restringe, e vuota", ""),
        ("sparisce. Quando ne restano poche ne nasce una nuova, lontana dal cibo ma non troppo.", ""),
        ("L'acqua del corpo cala sempre (di più al caldo e correndo): a zero si muore di sete.", ""),
        ("Si vede (blu), si annusa da lontano (umidità) e si sente da dentro (la sete).", ""),
        ("", "col"),
        ("Le misure", None),
        ("min:s di mondo: il tempo del mondo. s/s: secondi di mondo per secondo vero.", ""),
        ("vivi su posti: esseri vivi e tetto (Tetto nel menu).", ""),
        ("cibo: quanti cibi ci sono adesso (al massimo 1500).", ""),
        ("nati: figli di esseri del mondo (con partner: con due genitori).", ""),
        ("colonizzatori: arrivati da fuori perché i vivi erano sotto il minimo; sono copie", ""),
        ("mutate di chi ha mangiato di più (spariscono quando il mondo si regge da solo).", ""),
        ("morti di fame: energia finita. di sete: acqua finita. di vecchiaia: per età.", ""),
        ("pozze: quante ce ne sono adesso. Acqua (seguito): la sua riserva, 0% = muore di sete.", ""),
        ("sensi usati: recettori collegati al cervello, in media. neuroni: neuroni interni.", ""),
        ("gruppi: famiglie che non si incrociano più tra loro (quasi specie).", ""),
        ("esame: la prova fissa del mondo del cibo su 20 adulti a caso, ogni 10 minuti,", ""),
        ("poi la stessa senza plasticità: la differenza è quanto imparano durante la vita.", ""),
        ("manopole del mondo: le novità, che il mondo accende una alla volta quando la", ""),
        ("popolazione si regge da sola (e abbassa se torna ad aver bisogno di colonizzatori).", ""),
        ("seguito: età, energia (250 = può fare un figlio), acqua, cibi mangiati, cervello.", ""),
        ("", ""),
        ("H chiude questa pagina.", ""),
    )

    def _help(self, s, app) -> None:
        w, h = s.get_size()
        shade = pygame.Surface((w, h), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 190))
        s.blit(shade, (0, 0))
        box = pygame.Rect(60, 40, w - 120, h - 80)
        pygame.draw.rect(s, (20, 22, 30), box)
        pygame.draw.rect(s, BORDER, box, 1)
        x, y = box.x + 24, box.y + 18
        for text, kind in self.HELP:
            if kind == "col":  # seconda colonna
                x, y = box.x + box.w // 2 + 12, box.y + 18
            elif kind is None:
                s.blit(app.big.render(text, True, PLAYER), (x, y))
                y += 30
            else:
                s.blit(app.small.render(text, True, TEXT), (x, y))
                y += 18
