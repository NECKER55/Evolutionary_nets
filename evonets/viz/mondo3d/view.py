"""La schermata del mondo 3D: a sinistra il recinto, visto da fuori; a destra
ciò che riceve l'essere seguito (immagine e profondità, gli stessi pixel che
avrà un cervello) e come va. MuJoCo disegna senza finestre, qui si mostra.
La vista da fuori è solo per noi."""

import time

import numpy as np
import pygame

from ...arenas.mondo3d.bodies import BODIES
from ...arenas.mondo3d.run import measures
from ...brains.ragionante import RETUNE
from ..launcher import Screen
from ..render import BG, BORDER, DIM, PLAYER, TEXT
from ..widgets import Button
from .motor_panel import MotorPanel

SPEEDS = (1, 3, 6, 12, 24, 48)  # volte il tempo reale, come nell'arena delle pile
# Lato della vista da fuori e di immagine e profondità, in pixel; col cervello servono più spazio per la sezione.
WORLD_SIDE, EYE_SIDE = 560, 320
WORLD_COMPACT, EYE_COMPACT = 400, 196
DEPTH_NEAR, DEPTH_FAR = 5.0, 700.0  # cm: la scala dei colori, logaritmica
SKY = 1900.0  # cm: da qui in poi la telecamera non vede niente, solo cielo
SKY_C = (12, 12, 16)
# Dal vicino al lontano: giallo, verde, azzurro, viola.
DEPTH_STOPS = np.array([(253, 231, 37), (53, 183, 121), (49, 104, 142), (68, 1, 84)], dtype=float)
TICKS = (10, 25, 50, 100, 200, 400)  # cm segnati sulla scala
GOOD_C, BAD_C = (90, 220, 120), (240, 90, 80)
RED_C, YELLOW_C = (235, 80, 70), (240, 210, 60)
FLASH = 4.0  # s veri per cui si annuncia uno scambio
HELP = "Spazio pausa   +/- velocità   T turbo   Tab o frecce </> essere   frecce su/giù recinto   V da dietro   Esc menu"
NO_NET = "La rete si vede solo con gli agenti 'cervello ragionante' e 'vedente': questo non ha neuroni. Esc per tornare al menu."
KIND_C = ((235, 80, 70), (240, 210, 60), (120, 170, 255))  # i riquadri degli oggetti trovati, come in `vision.KINDS`
MESSAGE = 3.0  # s veri per cui resta un avviso
BRAIN_HELP = "   R tara   C cambia corpo"


def surface(rgb: np.ndarray) -> pygame.Surface:
    """Un'immagine (H, W, 3) di byte come superficie pygame."""
    return pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))


def depth_scale(depth) -> np.ndarray:
    """Da cm a [0, 1], in scala logaritmica: quasi tutti i pixel sono pavimento
    vicino, e su una scala lineare sarebbero tutti dello stesso colore."""
    return np.clip(np.log(np.maximum(depth, 1e-3) / DEPTH_NEAR) / np.log(DEPTH_FAR / DEPTH_NEAR), 0.0, 1.0)


def depth_colour(t) -> np.ndarray:
    x = np.linspace(0.0, 1.0, len(DEPTH_STOPS))
    return np.stack([np.interp(t, x, DEPTH_STOPS[:, c]) for c in range(3)], -1)


def depth_image(depth: np.ndarray) -> np.ndarray:
    """La profondità a colori: giallo vicino, viola lontano, nero il cielo."""
    return np.where((depth >= SKY)[..., None], np.array(SKY_C, dtype=float),
                    depth_colour(depth_scale(depth))).astype(np.uint8)


class Mondo3DScreen(Screen):
    def __init__(self, app, run, minutes: int = 0):
        self.app, self.run = app, run
        self.goal = 60.0 * minutes if minutes else None
        self.encl, self.follow = 0, 0
        self.paused = self.turbo = self.behind = False
        self.speed_i = 1
        self.frames = 0
        self.rate = 0.0
        self.last = time.perf_counter()
        self.mouse = None  # dov'è il mouse dentro i pixel della telecamera
        self.boxes = []
        self.brain = run.brain if hasattr(run.brain, "motor") else None
        self.panel = MotorPanel()
        self.buttons = [Button("Tara i sensi  [R]", self._tune, primary=True),
                        Button("Cambia corpo  [C]", self._next_body)] if self.brain else []
        self.pointer = None
        self.note, self.note_until = "", 0.0  # un avviso sui tasti
        self.swaps = 0  # scambi già annunciati
        self.flash_until = 0.0

    def _finished(self) -> bool:
        if self.goal is not None and self.run.t >= self.goal - 1e-9:
            self.paused = True
            return True
        return False

    def update(self) -> None:
        now = time.perf_counter()
        real, self.last = now - self.last, now
        t0 = self.run.t
        if not self.paused and not self._finished():
            if self.turbo:
                start = time.perf_counter()
                while time.perf_counter() - start < 0.08 and not self._finished():
                    self.run.step()
            else:
                per_second = SPEEDS[self.speed_i] / self.run.cfg.dt
                self.frames += 1
                self.run.advance(int(self.frames * per_second / 60.0) - int((self.frames - 1) * per_second / 60.0))
        self.rate = 0.8 * self.rate + 0.2 * (self.run.t - t0) / max(real, 1e-6)
        if len(self.run.w.inversions) > self.swaps:
            self.swaps = len(self.run.w.inversions)
            self.flash_until = now + FLASH

    def _say(self, text: str) -> None:
        self.note, self.note_until = text, time.perf_counter() + MESSAGE

    def _follow(self, step: int) -> None:
        if self.run.n == 1:
            self._say("c'è un solo essere: alza 'Esseri per recinto' nel menu")
        self.follow = (self.follow + step) % self.run.n

    def _enclosure(self, step: int) -> None:
        if self.run.B == 1:
            self._say("c'è un solo recinto: alza 'Recinti' nel menu")
        self.encl = (self.encl + step) % self.run.B

    def _tune(self) -> None:
        self.brain.motor.tune(RETUNE)

    def _next_body(self) -> None:
        """Cambia il corpo senza avvisare il cervello: se ne accorgerà da solo."""
        names, w = tuple(BODIES), self.run.w
        w.set_body(names[(names.index(w.body) + 1) % len(names)])

    def handle(self, ev) -> None:
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for b in self.buttons:
                b.click(ev.pos)
        if ev.type == pygame.MOUSEMOTION:
            self.pointer = ev.pos
            box = next((b for b in self.boxes if b.collidepoint(ev.pos)), None)
            self.mouse = None if box is None else ((ev.pos[1] - box.y) / box.h, (ev.pos[0] - box.x) / box.w)
        if ev.type != pygame.KEYDOWN:
            return
        k = ev.key
        if k == pygame.K_ESCAPE:
            self.close()
            self.app.launcher.returned("mondo 3D chiuso")
        elif k == pygame.K_SPACE:
            self.paused = not self.paused
        elif k == pygame.K_t:
            self.turbo = not self.turbo
        elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.turbo, self.speed_i = False, min(self.speed_i + 1, len(SPEEDS) - 1)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.turbo, self.speed_i = False, max(self.speed_i - 1, 0)
        elif k in (pygame.K_TAB, pygame.K_RIGHT):
            self._follow(1)
        elif k == pygame.K_LEFT:
            self._follow(-1)
        elif k in (pygame.K_PAGEDOWN, pygame.K_DOWN):
            self._enclosure(1)
        elif k in (pygame.K_PAGEUP, pygame.K_UP):
            self._enclosure(-1)
        elif k == pygame.K_v:
            self.behind = not self.behind
        elif k == pygame.K_r and self.brain:
            self._tune()
        elif k == pygame.K_c and self.brain:
            self._next_body()

    def close(self) -> None:
        self.run.close()

    def draw(self, s) -> None:
        app, r, w = self.app, self.run, self.run.w
        s.fill(BG)
        head = (f"Mondo 3D — agente {r.agent} — recinto {self.encl + 1}/{r.B} — essere {self.follow + 1}/{r.n}")
        s.blit(app.font.render(head, True, PLAYER), (16, 12))
        speed = "turbo" if self.turbo else f"x{SPEEDS[self.speed_i]}"
        info = f"t {r.t / 60:5.1f} min   {speed} ({self.rate:,.0f} s di mondo/s)" + ("   PAUSA" if self.paused else "")
        s.blit(app.small.render(info, True, DIM), (640, 16))
        side, eye = (WORLD_COMPACT, EYE_COMPACT) if self.brain else (WORLD_SIDE, EYE_SIDE)
        left = pygame.Rect(16, 44, side, side)
        s.blit(surface(w.overview(self.encl, side, side, self.follow, self.behind)), left)
        pygame.draw.rect(s, BORDER, left, 1)
        if time.perf_counter() < self.flash_until:
            self._flash(s, left)
        view = w.view(self.encl, self.follow)
        x, y = (16, left.bottom + 24) if self.brain else (left.right + 24, 62)
        gap = 8 if self.brain else 16
        self.boxes = []
        for name, image, dx in (("immagine", view["rgb"], 0), ("profondità", depth_image(view["depth"]), eye + gap)):
            s.blit(app.small.render(f"{name} ({w.cfg.image}×{w.cfg.image})", True, TEXT), (x + dx, y - 18))
            box = pygame.Rect(x + dx, y, eye, eye)
            s.blit(pygame.transform.smoothscale(surface(image), box.size), box)
            pygame.draw.rect(s, BORDER, box, 1)
            self.boxes.append(box)
        if self.brain:
            self._found(s, self.boxes[0], x, y - 18, eye)
            self._depth_reading(s, x, y + eye + 4, view["depth"])
            self._swaps(s, left.right + 24, 44)
            self._state(s, 1010, 44)
            self._motor(s, left.right + 24, 150)
        else:
            self._depth_legend(s, x + eye + gap, y + eye + 8, view["depth"], eye, gap)
            self._swaps(s, x, y + eye + 50)
            self._state(s, x, y + eye + 138)
        s.blit(app.small.render(HELP + (BRAIN_HELP if self.brain else ""), True, DIM), (16, 684 if self.brain else left.bottom + 12))
        if not self.brain:
            s.blit(app.small.render(NO_NET, True, TEXT), (16, left.bottom + 30))
        if time.perf_counter() < self.note_until:
            s.blit(app.font.render(self.note, True, PLAYER), (left.x + 8, left.y + 8))

    def _found(self, s, box, x, y, width) -> None:
        """La sezione 1: gli oggetti trovati, sopra l'immagine, con la loro distanza."""
        seen = getattr(self.brain.boss, "seen", None)
        if seen is None:
            return
        on = np.flatnonzero(seen.on[self.encl, self.follow])
        for m in on:
            left, top, right, down = seen.box[self.encl, self.follow, m]
            if right <= left:  # il detector finto non ha riquadri
                continue
            kind = seen.kind[self.encl, self.follow, m]
            rect = pygame.Rect(box.x + left * box.w - 1, box.y + top * box.h - 1,
                               (right - left) * box.w + 2, (down - top) * box.h + 2)
            pygame.draw.rect(s, KIND_C[kind], rect, 1)
            label = f"{seen.dist[self.encl, self.follow, m]:.0f}"
            s.blit(self.app.small.render(label, True, KIND_C[kind]), (rect.x, max(box.y, rect.y - 13)))
        text = f"vede {len(on)} oggetti"
        s.blit(self.app.small.render(text, True, TEXT), (box.x + 6, box.y + 4))

    def _motor(self, s, x, y) -> None:
        """La sezione 7 dell'essere seguito e i suoi pulsanti."""
        box = pygame.Rect(x, y, s.get_width() - x - 16, 520)
        self.panel.draw(s, self.app, box, self.run.w, self.brain.motor, self.encl, self.follow, self.pointer)
        bx = box.right - 12
        for b in self.buttons:
            width = self.app.font.size(b.label)[0] + 32
            bx -= width + 10
            b.draw(s, self.app.font, bx, box.y + 8)

    def _depth_reading(self, s, x, y, depth) -> None:
        """Il minimo della legenda, quando lo spazio serve alla sezione: la distanza sotto il mouse."""
        text = self._reading(depth) or "mouse sulle immagini: distanza del punto"
        s.blit(self.app.small.render(text, True, TEXT if self.mouse else DIM), (x, y))

    def _reading(self, depth) -> str:
        if self.mouse is None:
            return ""
        n = depth.shape[0]
        d = float(depth[min(int(self.mouse[0] * n), n - 1), min(int(self.mouse[1] * n), n - 1)])
        return "sotto il mouse: cielo, nessuna superficie" if d >= SKY else f"sotto il mouse: {d:.0f} cm"

    def _depth_legend(self, s, x, y, depth, side, gap) -> None:
        """La scala dei colori, in distanze vere, e quanto dista il punto sotto il mouse."""
        app = self.app
        for px in range(side):
            pygame.draw.line(s, depth_colour(px / (side - 1)).astype(int).tolist(), (x + px, y), (x + px, y + 8))
        for cm in TICKS:
            px = x + int(float(depth_scale(float(cm))) * (side - 1))
            pygame.draw.line(s, TEXT, (px, y + 8), (px, y + 12))
            label = f"{cm} cm" if cm < 100 else f"{cm / 100:g} m"
            s.blit(app.small.render(label, True, DIM), (px - 12, y + 14))
        text = self._reading(depth)
        s.blit(app.small.render(text or "passa il mouse per leggere la distanza", True, TEXT if text else DIM),
               (x - side - gap, y + 14))

    def _names(self):
        """(colore, parola) delle pile buone e delle cattive adesso nel recinto guardato."""
        w = self.run.w
        if w.cfg.good_by == "grandezza":
            big = ((240, 240, 240), "GRANDI")
            small = ((170, 170, 170), "PICCOLE")
            return (big, small) if w.good[self.encl] else (small, big)
        red, yellow = (RED_C, "ROSSE"), (YELLOW_C, "GIALLE")
        return (yellow, red) if w.good[self.encl] else (red, yellow)

    def _flash(self, s, box) -> None:
        """L'annuncio di uno scambio, sopra il mondo."""
        (col, word), _ = self._names()
        band = pygame.Rect(box.x, box.centery - 42, box.w, 84)
        pygame.draw.rect(s, (15, 15, 20), band)
        pygame.draw.rect(s, col, band, 3)
        text = self.app.big.render(f"SCAMBIO: ora sono buone le {word}", True, col)
        s.blit(text, text.get_rect(center=band.center))

    def _swaps(self, s, x, y) -> None:
        """Quali pile sono buone adesso e quanto manca ai prossimi cambi. Solo per
        noi: gli esseri non vengono avvisati, lo scoprono mordendo."""
        app, w = self.app, self.run.w
        (good_c, good), (bad_c, bad) = self._names()
        s.blit(app.font.render("buone:", True, TEXT), (x, y))
        pygame.draw.circle(s, good_c, (x + 76, y + 8), 8)
        s.blit(app.font.render(good, True, good_c), (x + 92, y))
        s.blit(app.font.render("cattive:", True, TEXT), (x + 210, y))
        pygame.draw.circle(s, bad_c, (x + 300, y + 8), 8)
        s.blit(app.font.render(bad, True, bad_c), (x + 316, y))
        for row, (name, due, every) in enumerate((("scambio buone/cattive", w.next_inversion, w.cfg.inversion_every),
                                                  ("cambio di zona", w.next_zone, w.cfg.zone_every))):
            yy = y + 26 + 20 * row
            left = max(due - w.t, 0.0)
            s.blit(app.small.render(f"{name} tra {left:3.0f} s", True, DIM), (x, yy))
            bar = pygame.Rect(x + 240, yy + 3, 300, 8)
            pygame.draw.rect(s, BORDER, bar, 1)
            pygame.draw.rect(s, TEXT, (bar.x, bar.y, int(bar.w * left / every), bar.h))
        s.blit(app.small.render("nel mondo: verde = buone, rosso = cattive, riquadro = zona (solo per noi)", True, DIM),
               (x, y + 68))

    def _state(self, s, x, y) -> None:
        app, w, b, i = self.app, self.run.w, self.encl, self.follow
        m, cfg = measures(self.run), w.cfg
        s.blit(app.font.render(f"energia {w.energy[b, i]:5.1f} / {cfg.energy_max:.0f}", True, TEXT), (x, y))
        bar = pygame.Rect(x + 230, y + 3, 180, 10)
        pygame.draw.rect(s, BORDER, bar, 1)
        pygame.draw.rect(s, GOOD_C, (bar.x, bar.y, int(bar.w * max(w.energy[b, i], 0.0) / cfg.energy_max), bar.h))
        lines = [(f"prese da tutti: {m['buone']} buone, {m['cattive']} cattive", TEXT),
                 (f"al minuto per essere: {m['pile/min']:.2f} buone, {m['cattive/min']:.2f} cattive", TEXT),
                 (f"morti di fame: {m['morti di fame']}", TEXT)]
        if w.charging[b, i] > 0.0:
            lines.append(("sta ricaricando", GOOD_C))
        if w.pain[b, i] > 0.2:
            lines.append(("dolore: pila cattiva", BAD_C))
        for k, (text, col) in enumerate(lines):
            s.blit(app.small.render(text, True, col), (x, y + 26 + 18 * k))
