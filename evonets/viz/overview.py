"""La panoramica della corsa: la scheda principale della pagina, pensata per
essere letta, non decifrata. Ogni sezione ha un titolo, i numeri sono scritti
a parole e una frase dice come interpretarli.

- a che punto è la scala (mondo del cibo) o qual è il compito;
- quanti si riproducono, generazione per generazione;
- dove sta la scala nel tempo;
- l'esame: profilo per livello e confronto con gli esami precedenti;
- il cervello medio e quanto conta la fortuna.
"""

import numpy as np
import pygame

from ..evo import ladder
from ..net.anatomy import KIND_NAMES
from .render import (
    BORDER,
    DIM,
    FOOD,
    IDLE,
    MODULATOR,
    PLAYER,
    PREDATOR,
    TARGET,
    TEXT,
    WALL,
    plot,
)

PANEL = (22, 25, 33)
BAND = (46, 52, 38)
THIN = (88, 94, 112)  # la linea di ogni generazione: di sfondo, la media è quella da leggere
WINDOW = 300  # generazioni mostrate nel grafico dei successi
DONE_STAGE = (120, 96, 36)
STAGE_NAMES = ("trovare il cibo", "inseguire il cibo che sfugge", "sfuggire ai predatori")
STAGE_COLORS = (FOOD, (235, 150, 70), PREDATOR)
CHANGE = 0.05  # differenza d'esame sotto la quale si parla di "stabile" (5 punti)


def wrap(text: str, font, width: int) -> list[str]:
    words, lines, line = text.split(" "), [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if font.size(trial)[0] <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = w
    return lines + ([line] if line else [])


def para(s, font, text: str, x: int, y: int, width: int, color=TEXT, step: int = 0) -> int:
    """Un paragrafo che va a capo da solo. Restituisce la y dopo l'ultima riga."""
    step = step or font.get_linesize()
    for line in wrap(text, font, width):
        s.blit(font.render(line, True, color), (x, y))
        y += step
    return y


def pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def num(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


class Overview:
    def __init__(self, app):
        self.app = app
        self.title = pygame.font.SysFont("monospace", 16, bold=True)
        self.oldest_rects: list[pygame.Rect] = []

    # --- struttura ---------------------------------------------------------

    def _section(self, s, rect: pygame.Rect, title: str, note: str = "") -> int:
        pygame.draw.rect(s, PANEL, rect, border_radius=6)
        s.blit(self.title.render(title, True, PLAYER), (rect.x + 14, rect.y + 10))
        if note:
            img = self.app.small.render(note, True, DIM)
            s.blit(img, (rect.right - img.get_width() - 14, rect.y + 13))
        return rect.y + 36

    def draw(self, s, area: pygame.Rect, evo, env: str = "") -> None:
        """`env`: lo stato del registro dell'ambiente, in una riga."""
        self.oldest_rects = []
        gap = 12
        left_w = (area.w - gap) // 2
        left = pygame.Rect(area.x, area.y, left_w, area.h)
        right = pygame.Rect(left.right + gap, area.y, area.w - left_w - gap, area.h)
        forage = evo.cfg.task == "cibo"
        hist = evo.history
        if forage:
            top = pygame.Rect(left.x, left.y, left.w, 186)
            self._ladder(s, top, evo)
            rest = left.h - top.h - gap
            succ = pygame.Rect(left.x, top.bottom + gap, left.w, rest * 3 // 5)
            self._success(s, succ, evo)
            pos = pygame.Rect(left.x, succ.bottom + gap, left.w, left.bottom - succ.bottom - gap)
            self._position(s, pos, hist)
        else:
            top = pygame.Rect(left.x, left.y, left.w, 110)
            self._task(s, top, evo, env)
            self._success(s, pygame.Rect(left.x, top.bottom + gap, left.w, left.bottom - top.bottom - gap), evo)
        exam = pygame.Rect(right.x, right.y, right.w, right.h * 3 // 5)
        self._exam(s, exam, evo)
        self._brain(s, pygame.Rect(right.x, exam.bottom + gap, right.w, right.bottom - exam.bottom - gap), evo)

    # --- scala e compito ---------------------------------------------------

    def _task(self, s, rect, evo, env: str) -> None:
        y = self._section(s, rect, "IL COMPITO")
        task = evo.world.task
        text = (f"{task.name}: si riproduce chi lo completa. Il compito non cambia da solo: "
                "lo rende più difficile chi conduce l'esperimento.")
        y = para(s, self.app.font, text, rect.x + 14, y, rect.w - 28)
        para(s, self.app.small, env, rect.x + 14, y + 4, rect.w - 28, DIM)

    def _ladder(self, s, rect, evo) -> None:
        cfg, task, app = evo.cfg, evo.world.task, self.app
        note = {"scala": "si muove da sola", "basico": "controllo: ferma al basico",
                "complesso": "controllo: ferma al complesso"}.get(cfg.ladder, "manopole ferme")
        y = self._section(s, rect, "A CHE PUNTO È LA SCALA", note)
        x0, w = rect.x + 14, rect.w - 28
        x = ladder.position(cfg)
        positions = np.array([h.get("position", x) for h in evo.history[-100:]])

        # La barra: tre stadi, la parte fatta piena, il tratto percorso nelle
        # ultime 100 generazioni sotto, la posizione come una tacca.
        bar = pygame.Rect(x0, y + 4, w, 26)
        seg = bar.w / 3
        for i, name in enumerate(STAGE_NAMES):
            r = pygame.Rect(int(bar.x + i * seg), bar.y, int(seg) - 3, bar.h)
            pygame.draw.rect(s, IDLE, r, border_radius=4)
            fill = float(np.clip(x - i, 0.0, 1.0))
            if fill > 0:
                pygame.draw.rect(s, DONE_STAGE, (r.x, r.y, int(r.w * fill), r.h), border_radius=4)
            label = app.small.render(f"{i + 1}. {name}", True, TEXT)
            s.blit(label, label.get_rect(center=r.center))
        px = lambda v: int(bar.x + v / ladder.TOP * bar.w)
        if len(positions) > 1:
            lo, hi = px(positions.min()), px(positions.max())
            pygame.draw.rect(s, TARGET, (lo, bar.bottom + 3, max(hi - lo, 2), 4))
        pygame.draw.polygon(s, TEXT, [(px(x), bar.y - 1), (px(x) - 6, bar.y - 9), (px(x) + 6, bar.y - 9)])
        y = bar.bottom + 14

        i, f = ladder.stage(cfg)
        where = (f"Stadio {i + 1} di 3, {STAGE_NAMES[i]}: fatto al {pct(f)} (posizione {num(x, 2)} su 3)."
                 if x < ladder.TOP else "In cima alla scala: il mondo complesso.")
        y = para(s, app.font, where, x0, y, w, PLAYER)
        if hasattr(task, "need"):
            prey = (f"il cibo scappa a {num(task.prey_speed)} u/s" if task.prey_speed > 0 else "il cibo sta fermo")
            if cfg.food_smell > 0:
                prey += (f"; la nube d'odore si forma e svanisce in {num(task.scent_tau)} s" if task.scent_tau > 0
                         else "; la nube d'odore è sempre formata")
            else:
                prey += "; il cibo non ha odore"
            pred = (f"{task.n_pred} {'predatore' if task.n_pred == 1 else 'predatori'} a {num(task.pred_speed)} u/s"
                    if task.n_pred else "nessun predatore")
            y = para(s, app.font, f"Ora: {task.n_food} cibi in arena (raggio {num(task.food_r)}), ne servono "
                                  f"{task.need} per riprodursi; {prey}; {pred}.", x0, y, w, TEXT)
        y = para(s, app.small, self._ladder_verdict(cfg, positions), x0, y + 2, w, TARGET)

    @staticmethod
    def _ladder_verdict(cfg, p: np.ndarray) -> str:
        """Una frase sull'andamento della scala nelle ultime generazioni (la
        barra azzurra sotto gli stadi)."""
        if cfg.ladder != "scala":
            return "Popolazione di controllo: il mondo non cambia mai."
        n = len(p)
        if n < 20:
            return "Troppo presto per dire dove si fermerà (barra azzurra: posizioni delle ultime generazioni)."
        recent = p[-20:]
        if recent[-1] - recent[0] >= 0.1:
            return f"Sta salendo: +{num(recent[-1] - recent[0], 2)} nelle ultime 20 generazioni."
        if recent[0] - recent[-1] >= 0.1:
            return (f"Sta scendendo: -{num(recent[0] - recent[-1], 2)} nelle ultime 20 generazioni; "
                    "il mondo era diventato troppo difficile.")
        if n >= 50 and p.max() - p.min() <= 0.25:
            return (f"Ferma: nelle ultime {n} generazioni è rimasta tra {num(p.min(), 2)} e {num(p.max(), 2)} "
                    "(barra azzurra). Gli agenti non riescono a superare questo punto.")
        return f"Si muove: tra {num(p.min(), 2)} e {num(p.max(), 2)} nelle ultime {n} generazioni (barra azzurra)."

    # --- successi e posizione nel tempo ------------------------------------

    def _x_labels(self, s, box, hist) -> None:
        small = self.app.small
        s.blit(small.render(f"gen. {hist[0]['generation']}", True, DIM), (box.x, box.bottom + 4))
        last = small.render(f"gen. {hist[-1]['generation']}", True, DIM)
        s.blit(last, (box.right - last.get_width(), box.bottom + 4))

    def _success(self, s, rect, evo) -> None:
        cfg, hist, app = evo.cfg, evo.history, self.app
        y = self._section(s, rect, "QUANTI SI RIPRODUCONO", "grigio: ogni generazione | azzurro: media di 10")
        x0, w = rect.x + 14, rect.w - 28
        if not hist:
            para(s, app.font, "Ancora nessuna generazione conclusa.", x0, y, w, DIM)
            return
        n = cfg.population
        done = np.array([h["completed"] for h in hist], dtype=float) * 100 / n
        avg = np.convolve(done, np.ones(10), "full")[: len(done)] / np.minimum(np.arange(1, len(done) + 1), 10)
        text = (f"Ultima generazione: {hist[-1]['completed']} su {n} ({done[-1]:.0f}%); media delle ultime 10: "
                f"{avg[-1]:.0f}%.")
        if cfg.task == "cibo":
            text += f" Cibi mangiati in media: {num(hist[-1].get('eaten', 0.0))}."
        y = para(s, app.font, text, x0, y, w)
        box = pygame.Rect(rect.x + 50, y + 8, rect.w - 70, rect.bottom - y - 30)
        if cfg.ladder == "scala":  # la fascia in cui la scala resta ferma
            y_hi = box.bottom - box.h * cfg.ladder_up
            y_lo = box.bottom - box.h * cfg.ladder_down
            pygame.draw.rect(s, BAND, (box.x, y_hi, box.w, y_lo - y_hi))
        shown = slice(-WINDOW, None)  # le ultime generazioni: con migliaia la linea diventa illeggibile
        plot(s, box, app.small, ((done[shown], THIN, 1), (avg[shown], TARGET, 2)), 100.0, (0, 0.5, 1), "{:3.0f}%")
        if cfg.ladder == "scala":
            label = app.small.render(f"fascia verde: tra {pct(cfg.ladder_down)} e {pct(cfg.ladder_up)} la scala "
                                     "resta ferma; sopra sale, sotto scende", True, TEXT)
            s.blit(label, (box.x + 4, box.y + 2))
        self._x_labels(s, box, hist[shown])

    def _position(self, s, rect, hist) -> None:
        app = self.app
        y = self._section(s, rect, "LA SCALA NEL TEMPO", "posizione da 0 a 3")
        if not hist:
            return
        box = pygame.Rect(rect.x + 50, y + 4, rect.w - 70, rect.bottom - y - 26)
        for i in range(3):  # una fascia per stadio, con il suo nome
            top = box.bottom - box.h * (i + 1) / 3
            img = app.small.render(STAGE_NAMES[i], True, STAGE_COLORS[i])
            s.blit(img, (box.right - img.get_width() - 4, top + 3))
        p = np.array([h.get("position", 0.0) for h in hist])
        plot(s, box, app.small, ((p, PLAYER, 2),), ladder.TOP, (0, 1 / 3, 2 / 3, 1), "{:1.0f}")
        self._x_labels(s, box, hist)

    # --- esame ------------------------------------------------------------

    @staticmethod
    def _level_label(key: str) -> str:
        """Cosa c'è in un livello dell'esame, in poche lettere."""
        from ..evo.exam import LEVELS, level_world

        test = next((t for t in LEVELS if t.key == key), None)
        if test is None:
            return ""
        w = level_world(test.changes)
        if w["predators"]:
            return f"{w['predators']} pred."
        if w["flee"] > 0:
            return f"fuga {w['flee']:.0f}"
        return f"{w['need']} su {w['food']}"

    def _exam(self, s, rect, evo) -> None:
        app = self.app
        # Si confrontano solo esami della stessa versione (prove diverse = punteggi diversi).
        version = evo.exams[-1].get("version") if evo.exams else None
        same = [e for e in evo.exams if e.get("version") == version]
        intact = [e for e in same if not e.get("lesion")]
        lesions = [e for e in same if e.get("lesion")]
        note = f"ogni {evo.cfg.exam_every} generazioni (E: ora)" if evo.cfg.exam_every else "E: fallo ora"
        y = self._section(s, rect, "ESAME: COSA SANNO FARE", note)
        x0, w = rect.x + 14, rect.w - 28
        if evo.exam is not None:
            y = para(s, app.small, f"Esame della generazione {evo.exam.generation} in corso: "
                                   f"{pct(evo.exam.progress)}.", x0, y, w, PLAYER)
        if not intact:
            para(s, app.font, "Nessun esame ancora. L'esame mette i 20 individui più anziani su prove fisse, "
                              "sempre uguali, e dice cosa sanno fare davvero, al di là del mondo di oggi.",
                 x0, y, w, DIM)
            return
        last = intact[-1]
        keys = list(last["scores"])
        older = len(evo.exams) - len(same)
        y = para(s, app.font, f"Ultimo: generazione {last['generation']}, {last['who']} più anziani, media "
                              f"{pct(last['score'])}." + (f" ({older} esami di una versione precedente non "
                                                           "confrontati.)" if older else ""), x0, y, w)

        # Barre grandi, con la percentuale scritta sopra e il livello sotto.
        levels = "L0" in last["scores"]
        box = pygame.Rect(x0 + 6, y + 22, w - 12, rect.bottom - y - 150)
        bw = box.w / len(keys)
        first = intact[0] if len(intact) > 1 else None
        pygame.draw.line(s, BORDER, (box.x, box.bottom), (box.right, box.bottom), 1)
        for k, key in enumerate(keys):
            v = last["scores"][key]
            bx = int(box.x + k * bw + 6)
            h = int(box.h * v)
            pygame.draw.rect(s, PLAYER, (bx, box.bottom - h, int(bw - 12), h), border_radius=3)
            img = app.small.render(pct(v), True, TEXT)
            s.blit(img, (bx + (bw - 12) / 2 - img.get_width() / 2, box.bottom - h - 15))
            if first is not None:
                fy = box.bottom - box.h * first["scores"].get(key, 0.0)
                pygame.draw.line(s, WALL, (bx - 3, fy), (bx + bw - 9, fy), 2)
            if lesions and key in lesions[-1]["scores"]:
                ly = box.bottom - box.h * lesions[-1]["scores"][key]
                pygame.draw.circle(s, MODULATOR, (int(bx + (bw - 12) / 2), int(ly)), 4)
            cx = bx + (bw - 12) / 2
            for j, text in enumerate((key, self._level_label(key) if levels else "")):
                img = app.small.render(text, True, TEXT if j == 0 else DIM)
                s.blit(img, (cx - img.get_width() / 2, box.bottom + 4 + 14 * j))
        y = box.bottom + 36
        legend = ("sotto ogni livello: cibi da mangiare su cibi in arena, velocità di fuga delle prede (u/s, "
                  "con la nube d'odore che sfugge), numero di predatori. " if levels else "")
        legend += (f"Tacca bianca: primo esame (gen. {first['generation']})." if first else "")
        if lesions:
            legend += f" Punto viola: senza {lesions[-1]['lesion']} (gen. {lesions[-1]['generation']})."
        y = para(s, app.small, legend, x0, y, w, DIM)
        for line in self._exam_verdict(intact, levels):
            y = para(s, app.small, line, x0, y + 2, w, TARGET)

    @staticmethod
    def _exam_verdict(intact: list[dict], levels: bool) -> list[str]:
        last = intact[-1]

        def diff(a, b) -> str:
            parts = [f"{k} {'+' if d > 0 else ''}{d * 100:.0f}" for k in a["scores"]
                     if abs(d := a["scores"][k] - b["scores"].get(k, 0.0)) >= CHANGE]
            return ", ".join(parts)

        lines = []
        if len(intact) > 1:
            first = intact[0]
            d = diff(last, first)
            lines.append(f"Dalla generazione {first['generation']}: " + (f"{d} punti." if d else
                                                                        "nessun livello cambiato di più di 5 punti."))
            prev = intact[-2]
            d = diff(last, prev)
            lines.append(f"Dall'esame precedente (gen. {prev['generation']}): " + (f"{d} punti." if d else "stabile."))
        never = [k for k, v in last["scores"].items() if v < 0.02]
        if never:
            span = never[0] if len(never) == 1 else f"{never[0]}–{never[-1]}"
            lines.append(f"Mai superati: {span}" + (" (gli agenti non hanno mai vissuto in quel mondo)."
                                                    if levels else "."))
        return lines

    # --- cervello e popolazione --------------------------------------------

    def _brain(self, s, rect, evo) -> None:
        app, hist = self.app, evo.history
        kind = KIND_NAMES.get(evo.cfg.net_type, evo.cfg.net_type)
        y = self._section(s, rect, "IL CERVELLO MEDIO E LA POPOLAZIONE", f"rete {kind}")
        x0, w = rect.x + 14, rect.w - 28
        if not hist:
            para(s, app.font, "Ancora nessuna generazione conclusa.", x0, y, w, DIM)
            return
        last, first = hist[-1], hist[0]
        names = ("luce", "sostanze", "meccanica", "temperatura", "magnetismo", "interno", "generalisti")
        used = ", ".join(f"{n} {num(v)}" for n, v in zip(names, last.get("used_by", [])) if v >= 0.05)
        lines = [
            f"Neuroni interni: {num(last['neurons'])} (alla gen. {first['generation']}: {num(first['neurons'])}); "
            f"collegamenti: {last['links']:.0f}.",
            f"Recettori usati: {num(last.get('used', 0.0))} su {num(last.get('receptors', 0.0))} accesi"
            + (f" ({used})." if used else "."),
        ]
        if "inhibitory" in last:
            lines.append(f"Neuroni inibitori: {pct(last['inhibitory'])}; modulatori: {num(last.get('modulators', 0.0))}.")
        if "markers" in last:
            lines.append(f"Sviluppo: {num(last['markers'])} marcatori in media, {num(last['types'])} tipi di neuroni "
                         f"interni per individuo (alla gen. {first['generation']}: {num(first.get('types', 0.0))}).")
        value = last.get("value")
        if value is not None:
            lines.append(f"Sistema di valore innato: gusto x{value[0]:.2g}, dolore x{value[1]:.2g} "
                         "(forza media dei due nuclei).")
        sex = last.get("mutation", {}).get("sex")
        if sex is not None:
            first_sex = first.get("mutation", {}).get("sex", sex)
            lines.append(f"Figli con un partner: {pct(sex)} (gene della madre; alla gen. {first['generation']}: "
                         f"{pct(first_sex)}).")
        luck = [h["luck"] for h in hist[-10:] if "luck" in h]
        if luck:
            near = float(np.mean([l["near"] for l in luck]))
            far = float(np.mean([l["far"] for l in luck]))
            verdict = ("la fortuna conta poco" if abs(near - far) < 0.1 else
                       "chi nasce vicino al cibo è avvantaggiato" if near > far else
                       "chi nasce lontano dal cibo riesce meglio")
            lines.append(f"Fortuna: completa il {pct(near)} dei nati vicino al cibo e il {pct(far)} dei nati "
                         f"lontano: {verdict}.")
        for line in lines:
            y = para(s, app.font, line, x0, y, w) + 4
        o = evo.oldest()
        if o is None:
            para(s, app.font, "Nessun sopravvissuto: sono tutti appena nati.", x0, y, w, DIM)
            return
        y = para(s, app.font, f"Il più anziano: #{o['id']}, {o['age']} {'prova vinta' if o['age'] == 1 else 'prove vinte'}"
                              f" di fila (nato alla gen. {o['born']}).", x0, y, w)
        r = s.blit(app.font.render("> la sua rete (clic o O)", True, PLAYER), (x0, y))
        pygame.draw.line(s, PLAYER, r.bottomleft, r.bottomright, 1)
        self.oldest_rects.append(r)
