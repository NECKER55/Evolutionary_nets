"""La sezione C sullo schermo: intenzione, nascosti, uscite, corpo e sensi, con
il cerchio chiuso da cui impara. Si legge da sinistra a destra."""

import numpy as np
import pygame

from ...arenas.mondo3d.bodies import FELT
from ...brains.ragionante import HIDDEN, INTENTS
from ..render import BORDER, DIM, PLAYER, TEXT

POS, NEG = (90, 220, 120), (240, 90, 80)
FIXED = (95, 100, 120)
TUNE_C = (255, 190, 60)
RADIUS = 17
HELP = {
    "intent": "Intenzione: che cosa vuole il comandante finto. È sempre la stessa, qualunque sia il corpo.",
    "hidden": "Neurone nascosto: impara da solo che cosa rappresentare. Non ha un significato stabilito da noi.",
    "out": "Uscita: ciò che il cervello manda al corpo. Durante la taratura è un comando di prova.",
    "body": "Corpo: che cosa succede davvero. Il cablaggio tra uscite e corpo il cervello non lo conosce.",
    "felt": "Senso del corpo: che cosa ho fatto davvero. È ciò che insegna alle sinapsi.",
}


def _shade(value: float) -> tuple:
    """Colore di un neurone: verde positivo, rosso negativo, grigio a riposo."""
    a = min(abs(value), 1.0)
    base = POS if value >= 0 else NEG
    return tuple(int(40 + a * (c - 40)) for c in base)


def _column(x, top, height, n) -> list:
    return [(x, int(top + height * (k + 0.5) / n)) for k in range(n)]


class MotorPanel:
    def __init__(self):
        self.spots = []  # (centro, spiegazione)
        self.history = []  # l'errore della rete, per il grafico

    def _neuron(self, s, font, pos, value, label, left, key, radius=RADIUS) -> None:
        pygame.draw.circle(s, _shade(value), pos, radius)
        pygame.draw.circle(s, TEXT, pos, radius, 1)
        num = font.render(f"{value:+.1f}", True, (240, 240, 240))
        s.blit(num, num.get_rect(center=pos))
        img = font.render(label, True, TEXT)
        s.blit(img, (pos[0] - radius - 8 - img.get_width(), pos[1] - 7) if left else (pos[0] + radius + 8, pos[1] - 7))
        self.spots.append((pos, f"{label}: {value:+.2f}. {HELP[key]}"))

    def _links(self, s, a, b, table, tuning) -> None:
        """Le sinapsi: spessore = forza, colore = segno; spente se comanda il caso."""
        for i, p in enumerate(a):
            for k, q in enumerate(b):
                w = table[k, i]
                if abs(w) < 0.03:
                    continue
                col = POS if w > 0 else NEG
                if tuning:
                    col = tuple(c // 2 for c in col)
                pygame.draw.line(s, col, p, q, max(1, min(4, int(1 + abs(w)))))

    def draw(self, s, app, box, world, motor, b, i, mouse) -> None:
        """`box`: dove; (b, i): chi; `mouse`: dov'è, per le spiegazioni."""
        small, font = app.small, app.font
        self.spots = []
        pygame.draw.rect(s, BORDER, box, 1)
        tuning = bool(motor.tuning[b, i])
        s.blit(font.render("7. Motore", True, PLAYER), (box.x + 12, box.y + 10))
        state = f"TARATURA: prova i comandi, ancora {motor.left[b, i]:.0f} s" if tuning else "in uso: traduce l'intenzione"
        s.blit(font.render(state, True, TUNE_C if tuning else POS), (box.x + 110, box.y + 10))
        s.blit(small.render(f"corpo: {world.body}   tarature: {motor.tunings[b, i]}", True, DIM), (box.x + 12, box.y + 32))
        top, height = box.y + 76, 250
        xs = [box.x + int(box.w * f) for f in (0.11, 0.28, 0.46, 0.68, 0.87)]
        heads = ("intenzione", "nascosti", "uscite", "corpo (cablaggio nascosto)", "sensi del corpo")
        for x, h in zip(xs, heads):
            img = small.render(h, True, TEXT)
            s.blit(img, (x - img.get_width() // 2, top - 22))
        intent, hidden = _column(xs[0], top, height, 3), _column(xs[1], top, height, HIDDEN)
        out, body, felt = _column(xs[2], top, height, motor.K), _column(xs[3], top, height, 3), _column(xs[4], top, height, 3)
        wiring = world.wiring
        self._links(s, intent, hidden, motor.params[0][b, i], tuning)
        self._links(s, hidden, out, motor.params[2][b, i], tuning)
        for k, p in enumerate(out):
            for j, q in enumerate(body):
                if abs(wiring[j, k]) > 0.01:
                    pygame.draw.line(s, FIXED, p, q, 1)
        for p, q in zip(body, felt):
            pygame.draw.line(s, FIXED, p, q, 1)
        # Il cerchio chiuso: ciò che si sente torna a insegnare alle sinapsi.
        y = top + height + 12
        col = TUNE_C if tuning else BORDER
        mid = xs[1]
        pygame.draw.lines(s, col, False, [(xs[4], top + height - 20), (xs[4], y), (mid, y), (mid, top + height - 26)], 2)
        pygame.draw.polygon(s, col, [(mid, top + height - 34), (mid - 5, top + height - 24), (mid + 5, top + height - 24)])
        s.blit(small.render("gradiente: ciò che ho sentito (ingresso) e il comando che ho dato (bersaglio) correggono le sinapsi",
                            True, col), (box.x + 12, y + 6))
        values = world.wiring @ motor.u[b, i]
        for k, p in enumerate(intent):
            self._neuron(s, small, p, motor.intent[b, i, k], INTENTS[k], True, "intent")
        for k, p in enumerate(hidden):
            self._neuron(s, small, p, motor.hidden[b, i, k], "", False, "hidden", 13)
        for k, p in enumerate(out):
            self._neuron(s, small, p, motor.u[b, i, k], world.outputs[k], False, "out")
        for k, p in enumerate(body):
            self._neuron(s, small, p, values[k], ("spinta", "sterzo", "morso")[k], False, "body")
        for k, p in enumerate(felt):
            self._neuron(s, small, p, motor.felt[b, i, k], FELT[k], False, "felt")
        self._table(s, app, box.x + 12, y + 30, motor.response[b, i], world.outputs)
        self._curve(s, app, pygame.Rect(box.x + 560, y + 34, box.w - 576, 70), float(motor.error[b, i]))
        if mouse is not None:
            near = min(self.spots, key=lambda x: np.hypot(x[0][0] - mouse[0], x[0][1] - mouse[1]))
            if np.hypot(near[0][0] - mouse[0], near[0][1] - mouse[1]) < RADIUS + 6:
                s.blit(small.render(near[1][:120], True, TEXT), (box.x + 12, box.y + 50))

    def _table(self, s, app, x, y, W, names) -> None:
        """Che cosa ha imparato, in cifre: quanto cambia ogni uscita per ogni intenzione."""
        s.blit(app.small.render("risposta" + "".join(f"{n[:8]:>10s}" for n in INTENTS), True, DIM), (x, y))
        for k, row in enumerate(W):
            line = f"{names[k][:8]:8s}" + "".join(f"{v:10.2f}" for v in row)
            s.blit(app.small.render(line, True, TEXT), (x, y + 15 * (k + 1)))

    def _curve(self, s, app, box, error) -> None:
        """Quanto la rete sbaglia nello spiegare i comandi dati: scende mentre impara."""
        self.history.append(error)
        del self.history[:-box.w]
        s.blit(app.small.render(f"errore della rete {error:.2f}", True, DIM), (box.x, box.y - 16))
        pygame.draw.rect(s, BORDER, box, 1)
        top = max(max(self.history), 1e-6)
        pts = [(box.x + k, box.bottom - int(box.h * v / top)) for k, v in enumerate(self.history)]
        if len(pts) > 1:
            pygame.draw.lines(s, TUNE_C, False, pts)
