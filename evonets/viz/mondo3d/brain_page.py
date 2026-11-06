"""Il cervello a tutto schermo, mentre la prova gira: lo schema delle sezioni con i neuroni
veri di quelle che esistono. Un colore dice che cosa è ogni blocco: vero, finto,
che guarda soltanto, o non ancora costruito. Passando sopra a un blocco si legge
che cosa fa. Si legge da sinistra a destra, come il segnale."""

import numpy as np
import pygame

from ...arenas.mondo3d.bodies import FELT
from ...brains.ragionante import HIDDEN, INTENTS
from ...brains.ragionante.events import OUTCOMES
from ...brains.ragionante.vision import KINDS
from ..render import BORDER, DIM, PLAYER, TEXT
from ..widgets import wrap
from .motor_panel import POS, NEG, _shade

REAL, FAKE, SIDE, NONE = (90, 220, 120), (255, 170, 70), (110, 170, 255), (85, 90, 105)
STATES = {REAL: "vera", FAKE: "finta", SIDE: "guarda, non guida", NONE: "non esiste"}
KIND_C = ((235, 80, 70), (240, 210, 60), (120, 170, 255))
RING = 6
LINKS = 30  # sinapsi disegnate tra due strati: solo le più forti, o sarebbe un muro di linee
ABOUT = {
    "P": "Percezione: dall'immagine agli oggetti, ognuno con che cos'è, angolo, distanza.",
    "A": "Attenzione: sceglie un oggetto. Oggi è una regola scritta nel comandante, senza neuroni.",
    "L": "Memoria di lavoro: 64 neuroni GRU. Cambia solo quando c'è un morso. Oggi guarda e basta.",
    "M": "Memoria spaziale: dove sono e dove stanno le cose. Non esiste ancora: chi non vede una pila non sa dov'era.",
    "W": "Previsione: dato ciò che ha in memoria e la pila più vicina, se la mordo piacere o dolore? Oggi guarda e basta.",
    "R": "Ricordi: fatti e luoghi a lungo termine. Non esiste ancora: alla morte non resta niente.",
    "D": "Decisione: oggi sono regole scritte a mano, e sanno quale colore è buono senza impararlo.",
    "C": "Comandi: traduce l'intenzione per un corpo che non conosce. Si tara da solo col gradiente.",
    "B": "Bisogni: non esiste ancora. L'energia entra direttamente nella memoria e nella previsione.",
}


class BrainPage:
    def __init__(self):
        self.spots = []  # (riquadro, spiegazione)

    def _block(self, s, app, box, key, title, state, note="") -> None:
        pygame.draw.rect(s, (20, 23, 31), box, border_radius=6)
        pygame.draw.rect(s, state, box, 2, border_radius=6)
        s.blit(app.font.render(title, True, state if state != NONE else DIM), (box.x + 8, box.y + 6))
        s.blit(app.small.render(STATES[state] + note, True, DIM), (box.x + 8, box.y + 24))
        self.spots.append((box, ABOUT[key]))

    def _arrow(self, s, a, b, on: bool, dead: bool = False) -> None:
        """Da un punto a un altro, ad angolo. Spenta se non passa niente, a tratti se manca un capo."""
        col = POS if on else (60, 64, 78)
        mid = (a[0] + b[0]) // 2
        for p, q in (((a[0], a[1]), (mid, a[1])), ((mid, a[1]), (mid, b[1])), ((mid, b[1]), (b[0], b[1]))):
            n = max(int(np.hypot(q[0] - p[0], q[1] - p[1]) / 8), 1)
            for k in range(n):
                if dead and k % 2:
                    continue
                u, v = k / n, (k + 1) / n
                pygame.draw.line(s, col, (p[0] + u * (q[0] - p[0]), p[1] + u * (q[1] - p[1])),
                                 (p[0] + v * (q[0] - p[0]), p[1] + v * (q[1] - p[1])), 2 if on else 1)
        pygame.draw.polygon(s, col, [b, (b[0] - 8, b[1] - 4), (b[0] - 8, b[1] + 4)])

    def _cells(self, s, x, y, values, per_row, side) -> list:
        """Neuroni in griglia, verdi o rossi. -> i centri."""
        centres = []
        for k, v in enumerate(values):
            c = (x + (k % per_row) * side + side // 2, y + (k // per_row) * side + side // 2)
            pygame.draw.circle(s, _shade(float(v)), c, max(side // 2 - 1, 1))
            centres.append(c)
        return centres

    def _strong(self, s, table, a, b, tick: bool) -> None:
        """Le sinapsi più forti tra due gruppi: `table` (len(b), len(a))."""
        order = np.argsort(-np.abs(table), axis=None)[:LINKS]
        for k, i in zip(*np.unravel_index(order, table.shape)):
            col = POS if table[k, i] > 0 else NEG
            pygame.draw.line(s, tuple(c // (2 if tick else 3) for c in col), a[i], b[k], 1)

    def _column(self, x, y, h, n) -> list:
        return [(x, int(y + h * (k + 0.5) / n)) for k in range(n)]

    def _synapses(self, s, a, b, table, dim: bool) -> None:
        for i, p in enumerate(a):
            for k, q in enumerate(b):
                w = table[k, i]
                if abs(w) >= 0.03:
                    col = POS if w > 0 else NEG
                    pygame.draw.line(s, tuple(c // 2 for c in col) if dim else col, p, q, max(1, min(3, int(1 + abs(w)))))

    def _neurons(self, s, app, pts, values, names, left) -> None:
        for p, v, name in zip(pts, values, names):
            pygame.draw.circle(s, _shade(float(v)), p, RING)
            pygame.draw.circle(s, TEXT, p, RING, 1)
            self.spots.append((pygame.Rect(p[0] - 8, p[1] - 8, 16, 16), f"{name}: {float(v):+.2f}"))
            if name:
                img = app.small.render(name[:9], True, DIM)
                s.blit(img, (p[0] - 10 - img.get_width(), p[1] - 7) if left else (p[0] + 10, p[1] - 7))

    def _percept(self, s, app, box, run, b, i, boss) -> None:
        """L'immagine vera con gli oggetti trovati."""
        from .view import surface

        seen = getattr(boss, "seen", None)
        side = min(box.w - 16, box.h - 60)
        pic = pygame.Rect(box.x + 8, box.y + 44, side, side)
        s.blit(pygame.transform.smoothscale(surface(run.w.view(b, i, depth=False)["rgb"]), pic.size), pic)
        pygame.draw.rect(s, BORDER, pic, 1)
        if seen is None:
            return
        for m in np.flatnonzero(seen.on[b, i]):
            left, top, right, down = seen.box[b, i, m]
            if right > left:
                pygame.draw.rect(s, KIND_C[seen.kind[b, i, m]], (pic.x + left * pic.w, pic.y + top * pic.h,
                                                                 (right - left) * pic.w + 2, (down - top) * pic.h + 2), 1)

    def _chosen(self, s, app, box, boss, b, i) -> None:
        """Gli oggetti che arrivano, col più vicino del colore buono: è la regola che fa da attenzione."""
        seen = boss.seen
        if seen is None:
            return
        on = np.flatnonzero(seen.on[b, i])
        goal = boss._goal()[b, i]
        wanted = on[seen.kind[b, i, on] == goal]
        pick = wanted[np.argmin(seen.dist[b, i, wanted])] if len(wanted) else -1
        for row, m in enumerate(on[:6]):
            kind = seen.kind[b, i, m]
            text = f"{'>' if m == pick else ' '} {KINDS[kind]} {seen.dist[b, i, m]:.0f} cm"
            s.blit(app.small.render(text[:24], True, KIND_C[kind]), (box.x + 8, box.y + 44 + 15 * row))

    def _wish(self, s, app, box, motor, b, i) -> None:
        """L'intenzione che la decisione manda ai comandi."""
        for row, name in enumerate(INTENTS):
            value, y = float(motor.intent[b, i, row]), box.y + 46 + 22 * row
            s.blit(app.small.render(name, True, TEXT), (box.x + 8, y))
            mid, width = box.x + box.w * 3 // 5, box.w // 3
            pygame.draw.line(s, BORDER, (mid - width // 2, y + 7), (mid + width // 2, y + 7))
            pygame.draw.line(s, POS if value >= 0 else NEG, (mid, y + 7), (mid + int(value * width / 2), y + 7), 5)

    def _motor(self, s, app, box, run, b, i) -> None:
        m, w = run.brain.motor, run.w
        top, h = box.y + 62, box.h - 74
        xs = [box.x + int(box.w * f) for f in (0.3, 0.55, 0.75)]
        tuning = bool(m.tuning[b, i])
        a, c, o = self._column(xs[0], top, h, 3), self._column(xs[1], top, h, HIDDEN), self._column(xs[2], top, h, m.K)
        self._synapses(s, a, c, m.params[0][b, i], tuning)
        self._synapses(s, c, o, m.params[2][b, i], tuning)
        self._neurons(s, app, a, m.intent[b, i], INTENTS, True)
        self._neurons(s, app, c, m.hidden[b, i], [""] * HIDDEN, False)
        self._neurons(s, app, o, m.u[b, i], w.outputs, False)
        if tuning:
            s.blit(app.small.render(f"si tara: {m.left[b, i]:.0f} s", True, FAKE), (box.x + 100, box.y + 6))

    def _memory(self, s, app, box, watch, b, i) -> list:
        h = watch.h[b, i]
        s.blit(app.small.render(f"eventi scritti: {watch.reads[b, i]}", True, DIM), (box.x + 8, box.y + 42))
        side = min((box.w - 24) // 8, (box.h - 70) // 8)
        return self._cells(s, box.x + 12, box.y + 60, h, 8, side)

    def _model(self, s, app, box, run, watch, b, i, cells) -> None:
        looks, first, second, odds = watch.looks(b, i)
        p = watch.p
        side = max(min(box.h - 96, 300) // 16, 3)
        x1, x2, y = box.x + 10, box.x + 10 + 8 * side + 14, box.y + 46
        a = self._cells(s, x1, y, first, 8, side)
        c = self._cells(s, x2, y, second, 8, side)
        self._strong(s, p["m2"], a, c, True)
        o = self._column(x2 + 8 * side + 34, y, 16 * side, len(OUTCOMES))
        self._neurons(s, app, o, odds, OUTCOMES, False)
        self._strong(s, p["out"][:len(OUTCOMES)], c, o, True)
        yy = y + 16 * side + 6
        for kind, dist, good, doubt in looks[:3]:
            text = f"{KINDS[kind]} {dist:.0f} cm: piacere {good:.2f}, dubbio {doubt:.2f}"
            s.blit(app.small.render(text, True, KIND_C[kind]), (box.x + 8, yy))
            yy += 15
        if not looks:
            s.blit(app.small.render("nessuna pila in vista: niente da prevedere", True, DIM), (box.x + 8, yy))

    def _events(self, s, app, x, y, watch, b, i) -> None:
        s.blit(app.small.render("ultimi morsi: oggetto, esito vero, piacere previsto prima di mordere", True, DIM), (x, y))
        for k, (label, out, good, _) in enumerate(reversed(watch.rows[b, i])):
            right = (good > 0.5) == (out == "piacere")
            col = POS if right else NEG
            s.blit(app.small.render(f"{label:7s} {out:8s} {good:.2f}  {'giusto' if right else 'sbagliato'}", True, col),
                   (x, y + 16 + 15 * k))

    def _doing(self, run, b, i) -> str:
        """Che cosa sta facendo, in una riga."""
        w, m = run.w, run.brain.motor
        if m.tuning[b, i]:
            return "Il motore si sta tarando: prova comandi a caso e impara dal corpo. Il resto aspetta."
        if w.charging[b, i] > 0.0:
            return "Sta consumando una pila buona: resta fermo."
        if w.pain[b, i] > 0.2:
            return "Ha appena morso una pila cattiva: dolore."
        seen = getattr(run.brain.boss, "seen", None)
        if seen is None:
            return "Il comandante finto legge la verità dal simulatore: la telecamera non serve a nessuno."
        if seen.on[b, i].any():
            return f"Vede {int(seen.on[b, i].sum())} oggetti, e il comandante punta il più vicino del colore buono."
        return "Non vede niente e non ricorda dov'erano le pile: cerca girando in tondo. M servirebbe qui."

    def draw(self, s, app, run, b, i, mouse, paused) -> None:
        self.spots = []
        W, H = s.get_size()
        boss, watch = run.brain.boss, run.watch
        eyes = hasattr(boss, "eyes")
        s.blit(app.font.render(f"Il cervello dell'essere {i + 1}, recinto {b + 1}   t {run.t / 60:.1f} min"
                               + ("   PAUSA" if paused else ""), True, PLAYER), (16, 10))
        x = 16
        for state, name in STATES.items():
            pygame.draw.circle(s, state, (x + 6, 40), 6)
            s.blit(app.small.render(name, True, DIM), (x + 16, 33))
            x += 40 + 8 * len(name)
        s.blit(app.small.render("frecce: verde passa il segnale, grigio no, a tratti manca un capo", True, DIM), (x + 10, 33))
        gap, top = 30, 56
        low = (H - top - 130) * 11 // 20
        wide = [(W - 32 - 4 * gap) * f // 100 for f in (21, 15, 18, 24, 22)]
        xs = [16]
        for k in wide[:-1]:
            xs.append(xs[-1] + k + gap)
        high = H - top - 130 - low - gap
        y2 = top + low + gap
        box = {"P": pygame.Rect(xs[0], top, wide[0], low), "A": pygame.Rect(xs[1], top, wide[1], low // 2),
               "L": pygame.Rect(xs[2], top, wide[2], low), "W": pygame.Rect(xs[3], top, wide[3], low),
               "D": pygame.Rect(xs[4], top, wide[4], low // 2), "B": pygame.Rect(xs[0], y2, wide[0], high // 2),
               "M": pygame.Rect(xs[2], y2, wide[2], high), "R": pygame.Rect(xs[3], y2, wide[3], high),
               "C": pygame.Rect(xs[4], top + low // 2 + gap, wide[4], low // 2 + high)}
        centre = lambda k, f=0.5: (box[k].x, box[k].y + int(box[k].h * f))
        right = lambda k, f=0.5: (box[k].right, box[k].y + int(box[k].h * f))
        live = watch.reads[b, i] > 0
        self._arrow(s, right("P", 0.3), centre("A"), eyes)
        self._arrow(s, right("P", 0.9), centre("L", 0.85), False, True)
        self._arrow(s, right("A"), centre("L", 0.3), eyes, True)
        self._arrow(s, right("L"), centre("W"), live)
        self._arrow(s, right("A", 0.3), centre("D", 0.2), eyes)
        self._arrow(s, right("W", 0.3), centre("D", 0.5), False, True)
        self._arrow(s, (box["L"].centerx, box["M"].y), (box["L"].centerx + 1, box["L"].bottom), False, True)
        self._arrow(s, right("M", 0.3), centre("R", 0.3), False, True)
        self._arrow(s, right("R", 0.3), centre("C", 0.75), False, True)
        self._arrow(s, right("B"), centre("M", 0.5), False, True)
        down = (box["D"].centerx, box["D"].bottom)
        pygame.draw.line(s, POS, down, (down[0], box["C"].y), 2)
        pygame.draw.polygon(s, POS, [(down[0], box["C"].y), (down[0] - 4, box["C"].y - 8), (down[0] + 4, box["C"].y - 8)])
        self._block(s, app, box["P"], "P", "P. Percezione", REAL if eyes else NONE,
                    f": {type(boss.eyes).__name__}" if eyes else ": spenta")
        self._percept(s, app, box["P"], run, b, i, boss)
        self._block(s, app, box["A"], "A", "A. Attenzione", FAKE if eyes else NONE, ": una regola" if eyes else "")
        if eyes:
            self._chosen(s, app, box["A"], boss, b, i)
        self._block(s, app, box["L"], "L", "L. Memoria lavoro", SIDE)
        cells = self._memory(s, app, box["L"], watch, b, i)
        self._block(s, app, box["W"], "W", "W. Previsione", SIDE, ": esito, non il mondo")
        self._model(s, app, box["W"], run, watch, b, i, cells)
        self._block(s, app, box["D"], "D", "D. Decisione", FAKE, f": {type(boss).__name__}")
        self._wish(s, app, box["D"], run.brain.motor, b, i)
        self._block(s, app, box["C"], "C", "C. Comandi", REAL, f": {run.w.body}")
        self._motor(s, app, box["C"], run, b, i)
        for key, name in (("M", "M. Memoria spaziale"), ("R", "R. Ricordi"), ("B", "B. Bisogni")):
            self._block(s, app, box[key], key, name, NONE)
        foot = H - 118
        s.blit(app.font.render(self._doing(run, b, i), True, TEXT), (16, foot))
        self._events(s, app, 16, foot + 24, watch, b, i)
        text = "passa il mouse su un blocco o su un neurone"
        if mouse is not None:
            hit = [t for r, t in self.spots if r.collidepoint(mouse)]
            text = hit[-1] if hit else text
        for k, line in enumerate(wrap(app.small, text, W // 2 - 32)[:3]):
            s.blit(app.small.render(line, True, TEXT), (W // 2, foot + 28 + 15 * k))
        s.blit(app.small.render("N torna al mondo   Spazio pausa   +/- velocità   Tab essere   R tara   C cambia corpo", True, DIM),
               (W // 2, H - 20))
