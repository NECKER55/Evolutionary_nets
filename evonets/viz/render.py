import numpy as np
import pygame

from ..net import receptors
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
# Colore di un recettore secondo il fenomeno a cui è più sensibile (bianco se
# generalista): luce, sostanze, meccanica, temperatura, magnetismo, interno.
PHENOMENON_COLOR = ((250, 230, 110), (120, 220, 60), (235, 150, 70), (240, 90, 90), (180, 130, 245),
                    (150, 160, 185))
GENERALIST = TEXT
MODULATOR = (200, 120, 255)  # neuroni modulatori: anello viola
EXCITE = (255, 160, 50)
INHIBIT = (70, 150, 255)
IDLE = (52, 56, 70)
PANEL_W = 440
GRAPH_H = 190
RINGS = 10


def receptor_color(genes, i: int, k: int):
    """Colore del recettore k dell'individuo i: il suo fenomeno dominante se è
    specializzato (almeno l'80% della sensibilità), altrimenti bianco."""
    p, share = receptors.dominant(genes, i, k)
    return PHENOMENON_COLOR[p] if share >= 0.8 else GENERALIST


def success_series(history: list[dict], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Percentuale di successi per generazione e la sua media sulle ultime 10."""
    pct = np.array([h["completed"] for h in history]) * 100.0 / n
    csum = np.cumsum(np.insert(pct, 0, 0.0))
    k = np.arange(1, len(pct) + 1)
    start = np.maximum(k - 10, 0)
    return pct, (csum[k] - csum[start]) / (k - start)


def plot(surface, box: pygame.Rect, font, series, top: float, ticks=(0.0, 0.5, 1.0), fmt="{:3.0f}") -> None:
    """Grafico a linee in `box`, da 0 a `top`. `series`: (valori, colore, spessore);
    `ticks` sono frazioni di `top` con una riga di griglia ciascuna."""
    top = max(top, 1e-9)
    for f in ticks:
        y = box.bottom - box.h * f
        pygame.draw.line(surface, IDLE, (box.x, y), (box.right, y), 1)
        img = font.render(fmt.format(top * f), True, DIM)
        surface.blit(img, (box.x - img.get_width() - 6, y - 6))
    for values, color, width in series:
        if not len(values):
            continue
        xs = box.x + np.arange(len(values)) * box.w / max(len(values) - 1, 1)
        pts = [(int(x), int(box.bottom - box.h * min(v, top) / top)) for x, v in zip(xs, values)]
        if len(pts) > 1:
            pygame.draw.lines(surface, color, False, pts, width)
        else:
            pygame.draw.circle(surface, color, pts[0], 2)


class Renderer:
    @staticmethod
    def window_size(cfg, arena_px: int = 1000) -> tuple[int, int]:
        scale = arena_px / cfg.arena_w
        return int(cfg.arena_w * scale) + PANEL_W, int(cfg.arena_h * scale)

    def __init__(self, world: World, screen: pygame.Surface, arena_px: int = 1000):
        self.world = world
        cfg = world.cfg
        self.scale = scale = arena_px / cfg.arena_w  # pixel per unità: l'arena occupa sempre la stessa larghezza
        size = (int(cfg.arena_w * scale), int(cfg.arena_h * scale))
        self.arena = pygame.Rect(0, 0, *size)
        self.panel = pygame.Rect(size[0], 0, PANEL_W, size[1])
        self.screen = screen
        self.name_of = lambda i: f"agente {i}"  # come chiamare l'agente i nel pannello della rete
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

    def _draw_receptors(self, i: int, genes, obs: np.ndarray) -> None:
        """I recettori dell'agente i: per ognuno il suo campo (due linee ai
        bordi dell'apertura) e una linea al centro tanto più luminosa quanto
        più risponde. Colore = fenomeno a cui è sensibile."""
        w = self.world
        h = w.heading[i]
        c, s_ = np.cos(h), np.sin(h)
        reach = 1.5 * w.cfg.light_attenuation
        for k in np.flatnonzero(genes.active[i]):
            bx, by = genes.body[i, k]
            o = w.pos[i] + w.cfg.agent_radius * np.array([bx * c - by * s_, bx * s_ + by * c])
            lx, ly = genes.look[i, k]
            centre = h + np.arctan2(ly, lx)
            half = genes.aperture_rad[i, k] / 2.0
            color = receptor_color(genes, i, k)
            if half < np.pi * 0.98:
                for edge in (centre - half, centre + half):
                    tip = o + reach * np.array([np.cos(edge), np.sin(edge)])
                    pygame.draw.line(self.screen, RAY, self._px(o), self._px(tip), 1)
            f = float(np.clip(obs[k], 0.0, 1.0)) ** 0.5
            shade = tuple(int(a + (b - a) * f) for a, b in zip(RAY, color))
            tip = o + reach * (0.3 + 0.7 * f) * np.array([np.cos(centre), np.sin(centre)])
            pygame.draw.line(self.screen, shade, self._px(o), self._px(tip), 2)
            pygame.draw.circle(self.screen, color, self._px(o), 2)

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
        """Posizione nel pannello di ogni neurone esistente, come nel cervello
        (net/geometry.py): recettori sul bordo sinistro all'altezza del loro
        nervo, interni nella loro posizione, motori sul bordo destro."""
        h_px = self.net_h
        left, right, top, bottom = 78, PANEL_W - 70, 44, h_px - 26
        g = brains.genome
        pos = {}
        for k in np.flatnonzero(g.r_active[i]):
            pos[k] = (left, int(top + g.r_y[i, k] * (bottom - top)))
        for hid in np.flatnonzero(brains.active[i]):
            x, y = brains.pos[i, hid]
            pos[brains.n_in + hid] = (int(left + 22 + x * (right - left - 44)), int(top + y * (bottom - top)))
        for o in range(brains.n_out):
            y = (o + 0.5) / brains.n_out
            pos[brains.n_in + brains.n_hidden + o] = (right, int(top + y * (bottom - top)))
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

        genes = brains.receptors
        for k, p in pos.items():
            pygame.draw.circle(layer, self._signal(values[k]), p, 4)
            if k < brains.n_in:  # il bordo dice a cosa è sensibile il recettore
                pygame.draw.circle(layer, receptor_color(genes, i, k), p, 5, 1)
                name = receptors.label(genes, i, k).split(" ", 1)[1]
                layer.blit(self.small.render(name[:11], True, DIM), (4, p[1] - 6))
            else:
                # Con i tipi fissi il bordo dice se il neurone eccita o inibisce.
                rim, hid = TEXT, k - brains.n_in
                if hasattr(brains, "inhibitory") and hid < brains.n_hidden:
                    rim = INHIBIT if brains.inhibitory[i, hid] else EXCITE
                if hid < brains.n_hidden and brains.modulatory[i, hid]:
                    rim = MODULATOR
                pygame.draw.circle(layer, rim, p, 5 if rim is not TEXT else 4, 1)
        for o, name in enumerate(brains.OUT_LABELS):
            p = pos[brains.n_in + brains.n_hidden + o]
            v = brains.state[i, brains.n_hidden + o]
            layer.blit(self.small.render(f"{name} {v:+.2f}", True, TEXT), (p[0] - 34, p[1] + 10))

        n_hidden = int(brains.active[i].sum())
        n_rec = int(brains.genome.r_active[i].sum())
        kind = {"rate": "continua", "spike-ei": "a impulsi E/I"}[brains.KIND]
        head = f"{self.name_of(i)}, rete {kind}: {n_rec} recettori, {n_hidden} neuroni, {len(links)} colleg."
        if hasattr(brains, "inhibitory"):
            n_inh = int((brains.inhibitory[i] & brains.active[i]).sum())
            head = (f"{self.name_of(i)}, {kind}: {n_rec} rec., {n_hidden - n_inh} ecc. + {n_inh} inib., "
                    f"{len(links)} coll.")
        layer.blit(self.small.render(head, True, TEXT), (8, 8))
        layer.blit(self.small.render(f"costo corpo e cervello {brains.upkeep[i]:.2f} energia/s", True, DIM),
                   (8, 22))
        for x, title in ((52, "recettori"), (200, "interni"), (PANEL_W - 88, "motori")):
            layer.blit(self.small.render(title, True, DIM), (x, 34))
        legend_y = self.net_h - 16
        x = 8
        short = ("luce", "sost.", "mecc.", "temp.", "magn.", "int.")
        for text, color in (("eccita", EXCITE), ("inibisce", INHIBIT)) + tuple(zip(short, PHENOMENON_COLOR)):
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
        if not history:
            plot(self.screen, box, self.small, (), 100.0)
            return
        pct, avg = success_series(history, self.world.cfg.population)
        plot(self.screen, box, self.small, ((pct, WALL, 1), (avg, TARGET, 2)), 100.0)
        self.screen.blit(self.small.render(f"gen. {history[0]['generation']}", True, DIM), (box.x, box.bottom + 6))
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
        # L'ultima lettura dei sensi, senza rileggerli: rileggerli consumerebbe
        # il rumore dei recettori e cambierebbe la corsa.
        obs = ctrl.obs if ctrl is not None and len(ctrl.obs) == self.world.n else None
        if obs is None and ctrl is not None:
            obs = np.zeros((self.world.n, ctrl.brains.n_in))
        if ctrl is not None and show_rays and self.world.alive[focus]:
            self._draw_receptors(focus, ctrl.brains.receptors, obs[focus])
        self._draw_agents(focus, ages)
        pygame.draw.rect(self.screen, BG, self.panel)
        pygame.draw.rect(self.screen, BORDER, self.arena, 2)
        if ctrl is not None:
            self._draw_net(ctrl.brains, focus, obs[focus])
        if history is not None:
            self._draw_graph(history)
        self._draw_hud(hud)
