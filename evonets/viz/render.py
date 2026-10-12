import numpy as np
import pygame

from ..net.sensors import BODY, CHANNELS
from ..sim.world import World

BG = (14, 16, 22)
ARENA = (24, 27, 36)
BORDER = (70, 76, 96)
HAZARD = (120, 220, 60)
MOCK = (110, 150, 220)
PLAYER = (255, 200, 60)
DEAD = (90, 90, 100)
TEXT = (220, 224, 235)
DIM = (140, 146, 165)
TARGET = (80, 200, 230)
DONE = (235, 240, 250)
RAY = (60, 66, 84)
WALL = (200, 204, 215)
# Colore del punto d'impatto per canale visivo: parete, veleno, bersaglio.
GOAL = (230, 120, 220)
HIT = (WALL, HAZARD, TARGET, GOAL)
EXCITE = (255, 160, 50)
INHIBIT = (70, 150, 255)
IDLE = (52, 56, 70)
PANEL_W = 440
GRAPH_H = 190
BODY_SHORT = ("avanti", "traverso", "rotaz.", "energia", "dolore", "urto")
RINGS = 10


class Renderer:
    def __init__(self, world: World, scale: float = 10.0):
        self.world = world
        self.scale = scale
        cfg = world.cfg
        size = (int(cfg.arena_w * scale), int(cfg.arena_h * scale))
        self.arena = pygame.Rect(0, 0, *size)
        self.panel = pygame.Rect(size[0], 0, PANEL_W, size[1])
        self.screen = pygame.display.set_mode((size[0] + PANEL_W, size[1]))
        pygame.display.set_caption("Evolutionary nets")
        self.overlay = pygame.Surface(size, pygame.SRCALPHA)
        self.net_h = self.panel.h - GRAPH_H  # sopra la rete, sotto il grafico
        self.net_layer = pygame.Surface((PANEL_W, self.net_h), pygame.SRCALPHA)
        self.font = pygame.font.SysFont("monospace", 15)
        self.small = pygame.font.SysFont("monospace", 11)

    def _px(self, p) -> tuple[int, int]:
        return int(p[0] * self.scale), int(p[1] * self.scale)

    def _draw_hazards(self) -> None:
        hz = self.world.hazards
        self.overlay.fill((0, 0, 0, 0))
        falloff = self.world.cfg.hazard_falloff
        for p, r in zip(hz.pos, hz.radius):
            c, rad = self._px(p), int(r * self.scale)
            # Anelli concentrici: l'opacità segue lo stesso profilo del danno.
            for k in range(RINGS, 0, -1):
                depth = 1.0 - (k - 0.5) / RINGS
                alpha = int(15 + 185 * depth ** falloff)
                pygame.draw.circle(self.overlay, (*HAZARD, alpha), c, max(1, rad * k // RINGS))
            pygame.draw.circle(self.overlay, (*HAZARD, 120), c, rad, 1)
        self.screen.blit(self.overlay, (0, 0))

    def _draw_task(self) -> None:
        task = self.world.task
        if hasattr(task, "target"):
            c, rad = self._px(task.target), int(task.radius * self.scale)
            pygame.draw.circle(self.screen, TARGET, c, rad, 2)
            pygame.draw.circle(self.screen, TARGET, c, 3)

    def _draw_vision(self, i: int, vision, obs: np.ndarray) -> None:
        """Raggi dell'agente i: linea fino al limite visivo, un punto per ogni
        cosa vista (colore = canale)."""
        w = self.world
        rng = w.cfg.vision_range
        o = w.pos[i]
        for ang, seen in zip(vision.ray_angles(w.heading[i : i + 1])[0], obs[i]):
            d = np.array([np.cos(ang), np.sin(ang)])
            # Il raggio si ferma alla parete, se la vede.
            pygame.draw.line(self.screen, RAY, self._px(o), self._px(o + d * (1.0 - seen[0]) * rng), 1)
            for close, color in zip(seen, HIT):
                if close > 0.0:
                    pygame.draw.circle(self.screen, color, self._px(o + d * (1.0 - close) * rng), 4)

    def _draw_agents(self, player: int | None, ages=None) -> None:
        w = self.world
        rad = max(2, int(w.cfg.agent_radius * self.scale))
        for i in range(w.n):
            if ages is not None and ages[i] > 0 and w.alive[i]:
                # Anello: è un sopravvissuto delle generazioni precedenti.
                pygame.draw.circle(self.screen, TEXT, self._px(w.pos[i]), rad + 3, 1)
            if i == player:
                continue
            self._draw_agent(i, self._tint(MOCK, i), rad)
        if player is not None:
            self._draw_agent(player, self._tint(PLAYER, player), rad)
            if w.alive[player] and not w.done[player]:
                # Vettore velocità: rende visibile la derapata rispetto al muso.
                tip = w.pos[player] + w.vel[player] * 0.25
                pygame.draw.line(self.screen, TEXT, self._px(w.pos[player]), self._px(tip), 1)

    def _tint(self, color, i: int):
        """Il colore sbiadisce verso il grigio man mano che l'energia cala."""
        w = self.world
        if not w.alive[i]:
            return DEAD
        if w.done[i]:
            return DONE
        f = 0.25 + 0.75 * min(1.0, w.energy[i] / w.cfg.energy_init)
        return tuple(int(d + (c - d) * f) for c, d in zip(color, DEAD))

    def _draw_agent(self, i: int, color, rad: int) -> None:
        w = self.world
        c = self._px(w.pos[i])
        nose = w.pos[i] + w.cfg.agent_radius * 1.8 * np.array(
            [np.cos(w.heading[i]), np.sin(w.heading[i])]
        )
        pygame.draw.circle(self.screen, color, c, rad)
        pygame.draw.line(self.screen, color, c, self._px(nose), 2)

    def _draw_hud(self, lines: list[tuple[str, tuple]]) -> None:
        # Fondo scuro semitrasparente: il testo resta leggibile sopra gli agenti.
        images = [self.font.render(text, True, color) for text, color in lines]
        width = min(max(img.get_width() for img in images) + 20, self.arena.w)
        back = pygame.Surface((width, 18 * len(images) + 12), pygame.SRCALPHA)
        back.fill((*BG, 190))
        self.screen.blit(back, (0, 0))
        for k, img in enumerate(images):
            self.screen.blit(img, (10, 8 + 18 * k))

    @staticmethod
    def _signal(value: float):
        """Colore di un'attivazione: caldo se positiva, freddo se negativa."""
        base = EXCITE if value >= 0.0 else INHIBIT
        f = min(1.0, abs(float(value)))
        return tuple(int(a + (b - a) * f) for a, b in zip(IDLE, base))

    def _net_layout(self, brains, i: int) -> dict[int, tuple[int, int]]:
        """Posizione nel pannello di ogni neurone esistente: ingressi in colonna
        a sinistra (raggruppati per raggio, poi il corpo), interni in cerchio,
        uscite a destra."""
        n_ch, n_vis = len(CHANNELS), brains.n_in - len(BODY)
        h_px = self.net_h
        pos, y = {}, 44.0
        body_step = 12.0  # i sensi del corpo hanno un'etichetta ciascuno
        n_body = brains.n_in - n_vis
        step = (h_px - 60.0 - 3.0 * (n_vis // n_ch) - 12.0 - body_step * n_body) / n_vis
        for k in range(brains.n_in):
            if k and k <= n_vis and k % n_ch == 0:
                y += 3.0 if k < n_vis else 12.0
            pos[k] = (78, int(y))
            y += step if k < n_vis else body_step
        hidden = np.flatnonzero(brains.active[i])
        cx, cy, rad = 250, h_px // 2 + 10, 105
        for j, hid in enumerate(hidden):
            a = 2.0 * np.pi * j / len(hidden) - np.pi / 2.0
            pos[brains.n_in + hid] = (int(cx + rad * np.cos(a)), int(cy + rad * np.sin(a)))
        for o in range(brains.n_out):
            pos[brains.n_in + brains.n_hidden + o] = (PANEL_W - 70, cy - 50 + 100 * o)
        return pos

    def _draw_net(self, brains, i: int, obs: np.ndarray) -> None:
        """Rete dell'agente i dal vivo. Linee: collegamenti (caldo = eccita,
        freddo = inibisce), tanto più visibili quanto più segnale sta passando.
        Pallini: attivazione del neurone in questo istante."""
        layer = self.net_layer
        layer.fill((0, 0, 0, 0))
        pos = self._net_layout(brains, i)
        values = np.concatenate([obs, brains.state[i]])
        first_target = brains.n_in
        w = brains.w_eff[i]
        links = np.argwhere(brains.links[i])
        for t, s in links:
            flow = w[t, s] * values[s]
            color = EXCITE if w[t, s] > 0.0 else INHIBIT
            alpha = int(28 + 227 * min(1.0, abs(flow)))
            a, b = pos[s], pos[first_target + t]
            if s == first_target + t:  # il neurone parla a se stesso
                pygame.draw.circle(layer, (*color, alpha), (a[0], a[1] - 8), 6, 1)
            else:
                pygame.draw.line(layer, (*color, alpha), a, b, 1)

        n_ch, n_vis = len(CHANNELS), brains.n_in - len(BODY)
        for k, p in pos.items():
            pygame.draw.circle(layer, self._signal(values[k]), p, 4)
            if k < n_vis:  # il bordo dice che tipo di cosa guarda quel neurone
                pygame.draw.circle(layer, HIT[k % n_ch], p, 4, 1)
                if k % n_ch == 0:
                    layer.blit(self.small.render(f"raggio {k // n_ch + 1}", True, DIM), (8, p[1] - 4))
            elif k < brains.n_in:
                layer.blit(self.small.render(BODY_SHORT[k - n_vis], True, DIM), (8, p[1] - 6))
            else:
                pygame.draw.circle(layer, TEXT, p, 4, 1)
        for o, name in enumerate(("spinta", "sterzo")):
            p = pos[brains.n_in + brains.n_hidden + o]
            v = brains.state[i, brains.n_hidden + o]
            layer.blit(self.small.render(f"{name} {v:+.2f}", True, TEXT), (p[0] - 26, p[1] + 10))

        n_hidden = int(brains.active[i].sum())
        head = f"rete dell'agente {i}: {n_hidden} neuroni interni, {len(links)} collegamenti"
        layer.blit(self.small.render(head, True, TEXT), (8, 8))
        for x, title in ((60, "sensi"), (228, "interni"), (PANEL_W - 88, "motori")):
            layer.blit(self.small.render(title, True, DIM), (x, 24))
        legend_y = self.net_h - 16
        x = 8
        for text, color in (("eccita", EXCITE), ("inibisce", INHIBIT)) + tuple(zip(CHANNELS, HIT)):
            img = self.small.render(text, True, color)
            layer.blit(img, (x, legend_y))
            x += img.get_width() + 8
        self.screen.blit(layer, self.panel.topleft)

    def _draw_graph(self, history: list[dict]) -> None:
        """Percentuale di successi per generazione: linea sottile = valore di
        ogni generazione, linea spessa = media delle ultime 10."""
        box = pygame.Rect(self.panel.x + 44, self.net_h + 30, PANEL_W - 64, GRAPH_H - 58)
        pygame.draw.line(self.screen, BORDER, (self.panel.x, self.net_h), (self.panel.right, self.net_h), 1)
        title = "successi (%) - sottile: ogni gen., spessa: media di 10"
        self.screen.blit(self.small.render(title, True, DIM), (self.panel.x + 8, self.net_h + 8))
        for pct in (0, 50, 100):
            y = box.bottom - box.h * pct // 100
            pygame.draw.line(self.screen, IDLE, (box.x, y), (box.right, y), 1)
            self.screen.blit(self.small.render(f"{pct:3d}", True, DIM), (self.panel.x + 14, y - 6))
        if not history:
            return
        pct = np.array([h["completed"] for h in history]) * 100.0 / self.world.n
        csum = np.cumsum(np.insert(pct, 0, 0.0))
        k = np.arange(1, len(pct) + 1)
        start = np.maximum(k - 10, 0)
        avg = (csum[k] - csum[start]) / (k - start)
        xs = box.x + (k - 1) * box.w / max(len(pct) - 1, 1)
        for values, color, width in ((pct, WALL, 1), (avg, TARGET, 2)):
            pts = [(int(x), int(box.bottom - box.h * v / 100.0)) for x, v in zip(xs, values)]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, color, False, pts, width)
            else:
                pygame.draw.circle(self.screen, color, pts[0], 2)
        self.screen.blit(self.small.render("gen. 0", True, DIM), (box.x, box.bottom + 6))
        last = self.small.render(f"gen. {history[-1]['generation']}: {pct[-1]:.0f}%  media {avg[-1]:.0f}%", True, TEXT)
        self.screen.blit(last, (box.right - last.get_width(), box.bottom + 6))

    def draw(self, focus: int, hud: list[tuple[str, tuple]], ctrl=None, show_rays: bool = True,
             ages=None, history=None) -> None:
        """`ctrl` (un NeuralController) abilita raggi e pannello della rete;
        `ages` segna i sopravvissuti, `history` disegna il grafico dei successi."""
        self.screen.fill(BG)
        pygame.draw.rect(self.screen, ARENA, self.arena)
        self._draw_hazards()
        self._draw_task()
        obs = ctrl.senses.observe(self.world) if ctrl is not None else None
        if ctrl is not None and show_rays and self.world.alive[focus]:
            self._draw_vision(focus, ctrl.senses.vision, ctrl.senses.rays)
        self._draw_agents(focus, ages)
        pygame.draw.rect(self.screen, BG, self.panel)
        pygame.draw.rect(self.screen, BORDER, self.arena, 2)
        if ctrl is not None:
            self._draw_net(ctrl.brains, focus, obs[focus])
        if history is not None:
            self._draw_graph(history)
        self._draw_hud(hud)
        pygame.display.flip()
