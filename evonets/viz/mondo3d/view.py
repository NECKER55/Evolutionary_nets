"""La schermata del mondo 3D: a sinistra il recinto, visto da fuori; a destra
ciò che riceve l'essere seguito (immagine e profondità, gli stessi pixel che
avrà un cervello) e come va. MuJoCo disegna senza finestre, qui si mostra.
La vista da fuori è solo per noi."""

import time

import numpy as np
import pygame

from ...arenas.mondo3d.run import measures
from ..launcher import Screen
from ..render import BG, BORDER, DIM, PLAYER, TEXT

SPEEDS = (1, 3, 6, 12, 24, 48)  # volte il tempo reale, come nell'arena delle pile
WORLD_SIDE = 560  # lato della vista da fuori, in pixel
EYE_SIDE = 320  # lato di immagine e profondità sullo schermo
DEPTH_NEAR, DEPTH_FAR = 5.0, 700.0  # cm: la scala dei colori, logaritmica
SKY = 1900.0  # cm: da qui in poi la telecamera non vede niente, solo cielo
SKY_C = (12, 12, 16)
# Dal vicino al lontano: giallo, verde, azzurro, viola.
DEPTH_STOPS = np.array([(253, 231, 37), (53, 183, 121), (49, 104, 142), (68, 1, 84)], dtype=float)
TICKS = (10, 25, 50, 100, 200, 400)  # cm segnati sulla scala
GOOD_C, BAD_C = (90, 220, 120), (240, 90, 80)
RED_C, YELLOW_C = (235, 80, 70), (240, 210, 60)
FLASH = 4.0  # s veri per cui si annuncia uno scambio
HELP = "Spazio pausa   +/- velocità   T turbo   Tab essere   PagGiù recinto   V vista da dietro   Esc menu"


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

    def handle(self, ev) -> None:
        if ev.type == pygame.MOUSEMOTION:
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
        elif k == pygame.K_TAB:
            self.follow = (self.follow + 1) % self.run.n
        elif k == pygame.K_PAGEDOWN:
            self.encl = (self.encl + 1) % self.run.B
        elif k == pygame.K_PAGEUP:
            self.encl = (self.encl - 1) % self.run.B
        elif k == pygame.K_v:
            self.behind = not self.behind

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
        left = pygame.Rect(16, 44, WORLD_SIDE, WORLD_SIDE)
        s.blit(surface(w.overview(self.encl, WORLD_SIDE, WORLD_SIDE, self.follow, self.behind)), left)
        pygame.draw.rect(s, BORDER, left, 1)
        if time.perf_counter() < self.flash_until:
            self._flash(s, left)
        view = w.view(self.encl, self.follow)
        x = left.right + 24
        self.boxes = []
        for name, image, dx in (("immagine", view["rgb"], 0), ("profondità", depth_image(view["depth"]), EYE_SIDE + 16)):
            s.blit(app.small.render(f"{name} ({w.cfg.image}×{w.cfg.image})", True, TEXT), (x + dx, 44))
            box = pygame.Rect(x + dx, 62, EYE_SIDE, EYE_SIDE)
            s.blit(pygame.transform.smoothscale(surface(image), box.size), box)
            pygame.draw.rect(s, BORDER, box, 1)
            self.boxes.append(box)
        self._depth_legend(s, x + EYE_SIDE + 16, 62 + EYE_SIDE + 8, view["depth"])
        self._swaps(s, x, 62 + EYE_SIDE + 50)
        self._state(s, x, 62 + EYE_SIDE + 138)
        s.blit(app.small.render(HELP, True, DIM), (16, left.bottom + 12))

    def _depth_legend(self, s, x, y, depth) -> None:
        """La scala dei colori, in distanze vere, e quanto dista il punto sotto il mouse."""
        app = self.app
        for px in range(EYE_SIDE):
            pygame.draw.line(s, depth_colour(px / (EYE_SIDE - 1)).astype(int).tolist(), (x + px, y), (x + px, y + 8))
        for cm in TICKS:
            px = x + int(float(depth_scale(float(cm))) * (EYE_SIDE - 1))
            pygame.draw.line(s, TEXT, (px, y + 8), (px, y + 12))
            label = f"{cm} cm" if cm < 100 else f"{cm / 100:g} m"
            s.blit(app.small.render(label, True, DIM), (px - 12, y + 14))
        if self.mouse is not None:
            n = depth.shape[0]
            d = float(depth[min(int(self.mouse[0] * n), n - 1), min(int(self.mouse[1] * n), n - 1)])
            text = "sotto il mouse: cielo, nessuna superficie" if d >= SKY else f"sotto il mouse: {d:.0f} cm"
            s.blit(app.small.render(text, True, TEXT), (x - EYE_SIDE - 16, y + 14))
        else:
            s.blit(app.small.render("passa il mouse per leggere la distanza", True, DIM), (x - EYE_SIDE - 16, y + 14))

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
        s.blit(app.small.render("nel mondo: verde sotto le buone, rosso sotto le cattive, riquadro = zona (solo per noi)",
                                True, DIM), (x, y + 68))

    def _state(self, s, x, y) -> None:
        app, w, b, i = self.app, self.run.w, self.encl, self.follow
        m, cfg = measures(self.run), w.cfg
        s.blit(app.font.render(f"energia {w.energy[b, i]:5.1f} / {cfg.energy_max:.0f}", True, TEXT), (x, y))
        bar = pygame.Rect(x + 230, y + 3, 200, 10)
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
