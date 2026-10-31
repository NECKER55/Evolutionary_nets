"""Le schede dell'arena continua oltre alla vista del mondo: la panoramica
(a che punto siamo, grafici minuto per minuto con una frase che li spiega) e
le reti (l'esploratore, con gli elenchi adatti a un mondo senza generazioni)."""

import numpy as np
import pygame

from ...arenas.continuous.ecology import DIAL_NAMES, DIALS
from ...brains.freeform.anatomy import Anatomy
from ...brains.freeform.genome import live_links
from ..generational.overview import PANEL, para
from ..inspector import HINT, InfoPanel, NetView, RowList
from ..render import DIM, FOOD, FRUIT, MODULATOR, PLAYER, PREDATOR, TARGET, TEXT

WINDOW = 120  # minuti mostrati nei grafici
DEMANDS = ("arrivare al cibo prima degli altri: vederlo da lontano, non sprecare energia",
           "bere spesso: fare la spola tra cibo e acqua, scegliere quale bisogno prima",
           "andare a bere lontano dal cibo e tornare: orientarsi, ricordare dov'è l'acqua",
           "trovare un'altra macchia quando questa si esaurisce, ricordare dove erano",
           "distinguere i colori ed evitare i frutti cattivi",
           "imparare durante la vita quale colore è buono in ogni macchia",
           "cacciare chi è più piccolo e fuggire da chi è più grande",
           "mordere a lungo il cibo duro, mordere in due quello grande",
           "trovare il cibo al buio: olfatto, memoria, segnali",
           "trovare pozze nuove quando le vecchie evaporano, ricordare dove sono")
GREEN = (110, 200, 120)
GREY = (150, 150, 160)
WATER = (90, 140, 240)
DRINK = (170, 210, 255)


LABEL_W = 104  # spazio a destra di ogni grafico per i nomi delle linee


def nice(top: float) -> float:
    """Il primo numero tondo (1, 2, 2,5, 5 per una potenza di 10) da `top` in su."""
    if top <= 0:
        return 1.0
    p = 10 ** np.floor(np.log10(top))
    return float(next(m * p for m in (1, 2, 2.5, 5, 10) if m * p >= top - 1e-9))


def chart(s, box: pygame.Rect, font, title: str, minutes, lines, fmt="{:.0f}", top: float | None = None,
          skip_first: bool = True) -> None:
    """Un grafico chiaro: titolo, asse da 0 a un numero tondo, minuti in
    basso, e ogni linea con il suo nome e il valore attuale scritti alla fine
    (niente legenda da decifrare). `lines`: (nome, valori, colore, spessore);
    le linee sempre a zero non si disegnano. Con `skip_first` la scala non
    tiene conto del primo minuto (la moria iniziale schiaccerebbe il resto:
    la sua linea esce dall'alto)."""
    s.blit(font.render(title, True, TEXT), (box.x, box.y))
    lines = [ln for ln in lines if len(ln[1]) and np.nanmax(ln[1]) > 0]
    area = pygame.Rect(box.x + 40, box.y + 20, box.w - 40 - LABEL_W, box.h - 20 - 16)
    if not lines:
        s.blit(font.render("(ancora niente)", True, DIM), (area.x, area.y + 10))
        return
    start = 1 if skip_first and len(lines[0][1]) > 3 else 0
    if top is None:
        top = nice(max(np.nanmax(v[start:]) for _, v, _, _ in lines) * 1.05)
        if start and any(np.nanmax(v[:start]) > top for _, v, _, _ in lines):
            s.blit(font.render("(primo minuto fuori scala)", True, DIM), (box.x + font.size(title)[0] + 10, box.y))
    top = nice(top)
    for f in (0.0, 0.5, 1.0):
        y = area.bottom - area.h * f
        pygame.draw.line(s, (44, 48, 60), (area.x, y), (area.right, y), 1)
        img = font.render(fmt.format(top * f), True, DIM)
        s.blit(img, (area.x - img.get_width() - 5, y - 7))
    ends = []
    for name, values, color, width in lines:
        n = len(values)
        xs = area.x + np.arange(n) * area.w / max(n - 1, 1)
        pts = [(int(x), int(area.bottom - area.h * min(max(v, 0.0), top) / top)) for x, v in zip(xs, values)
               if np.isfinite(v)]
        if len(pts) > 1:
            pygame.draw.lines(s, color, False, pts, width)
        elif pts:
            pygame.draw.circle(s, color, pts[0], 3)
        if pts:
            ends.append([pts[-1][1], f"{name} {fmt.format(values[-1])}", color])
    ends.sort(key=lambda e: e[0])  # i nomi non si sovrappongono: si scostano di una riga
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 13)
    if ends and ends[-1][0] > area.bottom + 6:  # troppo in basso: il gruppo risale
        ends[-1][0] = area.bottom + 6
        for k in range(len(ends) - 2, -1, -1):
            ends[k][0] = min(ends[k][0], ends[k + 1][0] - 13)
    for y, text, color in ends:
        s.blit(font.render(text, True, color), (area.right + 6, y - 7))
    if len(minutes):
        first, last = font.render(f"min {minutes[0]}", True, DIM), font.render(f"min {minutes[-1]}", True, DIM)
        s.blit(first, (area.x, area.bottom + 3))
        s.blit(last, (area.right - last.get_width(), area.bottom + 3))


class ArenaOverview:
    """La panoramica dell'arena: da leggere, non da decifrare."""

    def __init__(self, app):
        self.app = app
        self.title = pygame.font.SysFont("monospace", 16, bold=True)

    def _section(self, s, rect, title: str, note: str = "") -> int:
        pygame.draw.rect(s, PANEL, rect, border_radius=6)
        s.blit(self.title.render(title, True, PLAYER), (rect.x + 14, rect.y + 10))
        if note:
            img = self.app.small.render(note, True, DIM)
            s.blit(img, (rect.right - img.get_width() - 14, rect.y + 13))
        return rect.y + 34

    def draw(self, s, area: pygame.Rect, eco) -> None:
        gap = 12
        half = (area.w - gap) // 2
        left = pygame.Rect(area.x, area.y, half, area.h)
        right = pygame.Rect(left.right + gap, area.y, area.w - half - gap, area.h)
        top = pygame.Rect(left.x, left.y, left.w, 240)
        self._where(s, top, eco)
        self._population(s, pygame.Rect(left.x, top.bottom + gap, left.w, left.bottom - top.bottom - gap), eco)
        h3 = (right.h - 2 * gap) // 3
        self._brains(s, pygame.Rect(right.x, right.y, right.w, h3), eco)
        self._exam(s, pygame.Rect(right.x, right.y + h3 + gap, right.w, h3), eco)
        self._world(s, pygame.Rect(right.x, right.y + 2 * (h3 + gap), right.w, right.bottom - right.y - 2 * (h3 + gap)),
                    eco)

    # --- a che punto siamo ---------------------------------------------------

    def _where(self, s, rect, eco) -> None:
        cfg, app, hist = eco.cfg, self.app, eco.history
        minutes = eco.world.t / 60.0
        y = self._section(s, rect, "A CHE PUNTO SIAMO", f"{minutes:.0f} minuti di mondo, {int(eco.world.alive.sum())} vivi")
        x0 = rect.x + 14
        if eco.gpu is None:
            para(s, app.font, "Mondo base (motore NumPy): le novità del mondo che si complica girano solo con il "
                              "motore JAX (\"gpu\").", x0, y, rect.w - 28)
            return
        current = next((k for k, d in enumerate(DIALS) if getattr(cfg, d) < 1.0), None)
        for k, (d, name) in enumerate(zip(DIALS, DIAL_NAMES)):
            v = getattr(cfg, d)
            color = TARGET if v >= 1.0 else PLAYER if k == current else DIM
            pygame.draw.rect(s, (40, 44, 56), (x0, y + 4, 120, 9))
            pygame.draw.rect(s, color, (x0, y + 4, int(120 * v), 9))
            state = "fatto" if v >= 1.0 else ("in corso" if k == current else "da venire")
            s.blit(app.small.render(f"{k + 1}. {name}  {v:.0%}  ({state})", True,
                                    TEXT if current is None or k <= current else DIM),
                   (x0 + 130, y))
            y += 16
        recent = hist[-cfg.arena_dial_calm:]
        alone = 0
        for r in reversed(hist):
            if r["colonists"]:
                break
            alone += 1
        if not hist:
            text = "Il primo minuto non è ancora finito."
        elif alone:
            text = f"La popolazione si regge da sola da {alone} minuti (nessun colonizzatore). "
            if current is None:
                text += "Tutte le novità sono accese: il mondo è completo."
            else:
                text += (f"Il mondo alza «{DIAL_NAMES[current]}» di {cfg.arena_dial_step:.0%} ogni "
                         f"{cfg.arena_dial_calm} minuti finché regge. Adesso chiede: {DEMANDS[current]}.")
        else:
            need = sum(r["colonists"] for r in recent)
            text = (f"La popolazione non si regge ancora da sola: negli ultimi minuti sono arrivati {need} "
                    "colonizzatori. Il mondo non si complica finché non ce la fa"
                    + (", e se continua abbassa l'ultima novità." if any(getattr(cfg, d) > 0 for d in DIALS) else "."))
        para(s, app.small, text, x0, y + 4, rect.w - 28)

    # --- grafici -----------------------------------------------------------------

    def _series(self, hist, key, scale: float = 1.0):
        return np.array([float(r.get(key, 0.0) or 0.0) * scale for r in hist[-WINDOW:]])

    def _minutes(self, hist) -> list:
        return [r["minute"] for r in hist[-WINDOW:]]

    def _pair(self, rect, y, text_lines: int = 2):
        """Due grafici affiancati sotto il titolo della sezione, con lo spazio
        per `text_lines` righe di spiegazione sotto."""
        h = rect.bottom - y - 14 * text_lines - 12
        w = (rect.w - 28 - 16) // 2
        return pygame.Rect(rect.x + 14, y, w, h), pygame.Rect(rect.x + 14 + w + 16, y, w, h)

    def _population(self, s, rect, eco) -> None:
        app, hist = self.app, eco.history
        y = self._section(s, rect, "POPOLAZIONE", f"ultimi {min(len(hist), WINDOW)} minuti")
        if not hist:
            para(s, app.font, "Le misure arrivano a ogni minuto di mondo.", rect.x + 14, y, rect.w - 28, DIM)
            return
        mins = self._minutes(hist)
        h = (rect.bottom - y - 50) // 2
        a = pygame.Rect(rect.x + 14, y, rect.w - 28, h)
        b = pygame.Rect(rect.x + 14, y + h + 6, rect.w - 28, h)
        chart(s, a, app.small, "Quanti sono vivi", mins, [("vivi", self._series(hist, "alive"), PLAYER, 2)])
        chart(s, b, app.small, "Ogni minuto: chi nasce, chi muore e perché, chi arriva da fuori", mins,
              [("nati", self._series(hist, "births"), GREEN, 2),
               ("fame", self._series(hist, "starved"), FOOD, 1),
               ("sete", self._series(hist, "thirst"), WATER, 1),
               ("vecchiaia", self._series(hist, "old_age"), GREY, 1),
               ("predati", self._series(hist, "killed"), PREDATOR, 1),
               ("colonizz.", self._series(hist, "colonists"), MODULATOR, 1)])
        r = hist[-1]
        text = (f"Ultimo minuto: {r['births']} nati ({r['mates']} con partner); morti {r['starved']} di fame, "
                f"{r.get('thirst', 0)} di sete, {r['old_age']} di vecchiaia, {r.get('killed', 0)} predati; "
                f"{r['colonists']} colonizzatori. Età media {r.get('age_mean', 0):.0f} s. "
                + ("Si regge da sola quando i colonizzatori vanno a zero." if r["colonists"] else
                   "Nessun colonizzatore: si regge da sola."))
        para(s, app.small, text, rect.x + 14, b.bottom + 6, rect.w - 28)

    def _brains(self, s, rect, eco) -> None:
        app, hist = self.app, eco.history
        y = self._section(s, rect, "CERVELLO E CORPO", "in media sui vivi")
        if not hist:
            return
        mins = self._minutes(hist)
        a, b = self._pair(rect, y)
        chart(s, a, app.small, "Cervello", mins,
              [("sensi", self._series(hist, "used"), TARGET, 2), ("neuroni", self._series(hist, "neurons"), MODULATOR, 2)],
              fmt="{:.1f}")
        chart(s, b, app.small, "Corpo e azioni (0-1)", mins,
              [("taglia", self._series(hist, "size_mean"), GREEN, 1), ("morso", self._series(hist, "bite_mean"), FOOD, 1),
               ("luce", self._series(hist, "glow_mean"), FRUIT, 1), ("odore", self._series(hist, "smell_mean"), GREY, 1)],
              fmt="{:.2f}")
        r, first = hist[-1], hist[0]
        trend = r.get("used", 0) - first.get("used", 0)
        text = (f"Sensi usati = recettori collegati al cervello ({'+' if trend >= 0 else ''}{trend:.2f} dall'inizio); "
                f"collegamenti {r.get('links', 0):.0f}. "
                + ("I sensi calano: il mondo si lascia vivere alla cieca." if trend < -0.3 else
                   "I sensi crescono: vedere e annusare conviene." if trend > 0.3 else "I sensi sono stabili."))
        para(s, app.small, text, rect.x + 14, a.bottom + 4, rect.w - 28)

    def _exam(self, s, rect, eco) -> None:
        app = self.app
        normal = [r for r in eco.exams if r.get("lesion") is None]
        probe = {r["generation"]: r["score"] for r in eco.exams if r.get("lesion") == "plasticità"}
        every = eco.cfg.exam_every
        y = self._section(s, rect, "ESAME", f"ogni {every} minuti, 20 adulti a caso" if every else
                          "esame automatico spento (E lo avvia)")
        if not normal:
            text = ("Nessun esame ancora. L'esame fisso (i livelli del mondo del cibo) si fa a ogni "
                    f"{eco.cfg.exam_every} minuti di mondo, poi lo stesso senza plasticità: la differenza è quanto "
                    "imparano durante la vita.")
            if eco.exam is not None:
                text = f"Esame in corso ({eco.exam.progress:.0%})."
            para(s, app.small, text, rect.x + 14, y, rect.w - 28, DIM)
            return
        a = np.array([r["score"] for r in normal]) * 100
        b = np.array([probe.get(r["generation"], np.nan) for r in normal]) * 100
        box = pygame.Rect(rect.x + 14, y, rect.w - 28, rect.bottom - y - 40)
        chart(s, box, app.small, "Riuscita all'esame (%)", [r["generation"] for r in normal],
              [("normale", a, PLAYER, 2), ("senza plast.", b, MODULATOR, 2)], fmt="{:.0f}%",
              top=max(10.0, np.nanmax(a), np.nanmax(np.nan_to_num(b))), skip_first=False)
        last = normal[-1]
        p = probe.get(last["generation"])
        text = f"Minuto {last['generation']}: {last['score']:.0%}"
        if p is not None:
            d = (last["score"] - p) * 100
            text += (f", senza plasticità {p:.0%}: " + ("imparano durante la vita." if d > 2 else
                                                         "la plasticità peggiora: si confondono." if d < -2 else
                                                         "nessuna differenza, per ora non imparano."))
        para(s, app.small, text, rect.x + 14, box.bottom + 4, rect.w - 28)

    @staticmethod
    def _returns(r) -> str:
        """Imparano dove sta l'acqua? Il ritorno alla pozza rispetto alla prima volta."""
        ratio = r.get("return_ratio")
        if ratio is None:
            return ""
        verdict = ("tornano prima: imparano dove sta l'acqua" if ratio < 0.8 else
                   "ci mettono di più che la prima volta" if ratio > 1.2 else "come la prima volta: non imparano")
        return f" Chi ha già bevuto torna a bere in {ratio:.0%} del tempo della prima volta: {verdict}."

    def _world(self, s, rect, eco) -> None:
        app, hist = self.app, eco.history
        y = self._section(s, rect, "CIBO E ACQUA")
        if not hist:
            return
        mins = self._minutes(hist)
        a, b = self._pair(rect, y)
        chart(s, a, app.small, "Cibo (al minuto)", mins,
              [("presente", self._series(hist, "food"), GREY, 1), ("mangiati", self._series(hist, "eaten"), FOOD, 2),
               ("cattivi", self._series(hist, "bad_eaten"), FRUIT, 1)])
        water = [("nei corpi %", self._series(hist, "water_mean", 100), WATER, 2)]
        if any(r.get("patch_health", 1) < 0.995 for r in hist[-WINDOW:]):
            water.append(("macchie %", self._series(hist, "patch_health", 100), GREEN, 1))
        chart(s, b, app.small, "Acqua", mins, water + [("s tra bevute", self._series(hist, "drink_gap"), DRINK, 1)])
        r = hist[-1]
        text = ((f"{r['pools']} pozze; tra due bevute {r['drink_gap']:.0f} s." if r.get("drink_gap") and "pools" in r
                 else f"{r.get('pools', 0)} pozze." if eco.cfg.arena_water else "Niente acqua (motore NumPy).")
                + (f" Temperatura dove vivono {r['temp_mean']:.2f} (0 freddo, 1 caldo)." if "temp_mean" in r
                   and eco.cfg.arena_temperature > 0 else "")
                + self._returns(r))
        para(s, app.small, text, rect.x + 14, a.bottom + 4, rect.w - 28)


# --- Le reti -----------------------------------------------------------------------

OLDEST, CHAMPIONS, ALL = "anziani", "campioni", "tutti"
ARENA_SOURCES = (OLDEST, CHAMPIONS, ALL)
ARENA_TABS = {OLDEST: "1 anziani", CHAMPIONS: "2 campioni", ALL: "3 tutti"}


class ArenaExplorer:
    """Elenco di reti + rete disegnata + dettagli del neurone, per l'arena:
    i più anziani vivi, i campioni (il più anziano di ogni minuto), tutti i
    vivi. Gli elenchi dei vivi sono fotografie (R le rifà)."""

    def __init__(self, eco):
        self.eco, self.cfg = eco, eco.cfg
        self.source = OLDEST
        self.list, self.net, self.info = RowList(), NetView(), InfoPanel()
        self.genomes: dict = {}
        self.tab_rects: dict = {}
        self.tabs_at = (0, 0)
        self.seen = None
        self.key = None
        self.snapshot()

    def snapshot(self) -> None:
        eco = self.eco
        w = eco.world
        rows, self.genomes = [], {}
        head = f"{'num.':>7} {'età':>4} {'tag':>4} {'cibi':>4} {'neur':>4} {'coll':>4}"
        if self.source == CHAMPIONS:
            for minute, age, g, uid in reversed(eco.champions):
                n, l = int(g.active[0].sum()), int(live_links(g.conn, g.active, g.r_active).sum())
                rows.append((f"{'min ' + str(minute):>7} {age:>4d} {'':>4} {'':>4} {n:>4d} {l:>4d}", ("ch", minute)))
                self.genomes[("ch", minute)] = (lambda g=g: g, f"campione del minuto {minute}: #{uid}", uid, None)
            self.list.header = [f"{'minuto':>7} {'età':>4} {'':>4} {'':>4} {'neur':>4} {'coll':>4}",
                                "campione = il più anziano", "vivo di ogni minuto"]
            self.seen = len(eco.champions)
        else:
            alive = np.flatnonzero(w.alive)
            order = alive[np.argsort(-w.age[alive])]
            if self.source == OLDEST:
                order = order[:100]
            g, sizes = eco.ctrl.brains.genome, eco.sizes()
            links = live_links(g.conn[order], g.active[order], g.r_active[order]).sum(axis=(1, 2))
            for i, l in zip(order, links):
                uid = int(eco.ids[i])
                n = int(g.active[i].sum())
                rows.append((f"{'#' + str(uid):>7} {w.age[i]:>4.0f} {sizes[i]:>4.1f} {w.task.eaten[i]:>4d} "
                             f"{n:>4d} {l:>4d}", ("v", uid)))
                self.genomes[("v", uid)] = (lambda i=i: g.take([i]), f"#{uid}, età {w.age[i]:.0f} s", uid, int(i))
            what = "i 100 più anziani vivi" if self.source == OLDEST else f"tutti i {len(rows)} vivi"
            self.list.header = [head, f"{what} (R aggiorna)", "Invio: seguilo nel mondo"]
            self.seen = None
        follow = self.list.selected == 0
        self.list.rows = rows
        keys = [k for _, k in rows]
        self.list.select(0 if follow or self.key not in keys else keys.index(self.key))
        self._load()

    def refresh(self) -> None:
        if self.source == CHAMPIONS and self.seen != len(self.eco.champions):
            self.snapshot()

    def switch(self, source: str) -> None:
        if source != self.source:
            self.source = source
            self.list.selected = self.list.scroll = 0
            self.snapshot()

    def _load(self) -> None:
        key = self.list.rows[self.list.selected][1] if self.list.rows else None
        if key != self.key:
            self.key = key
            self.net.show(Anatomy(self.cfg, self.genomes[key][0]()) if key is not None else None)

    def chosen_slot(self) -> int | None:
        """Il posto nel mondo dell'essere scelto (se è vivo)."""
        if self.key is None:
            return None
        slot = self.genomes[self.key][3]
        if slot is None:  # un campione: lo si cerca tra i vivi
            uid = self.genomes[self.key][2]
            hit = np.flatnonzero((self.eco.ids == uid) & self.eco.world.alive)
            slot = int(hit[0]) if len(hit) else None
        return slot if slot is not None and self.eco.world.alive[slot] else None

    def place(self, tabs_at, list_rect, net_rect, info_rect) -> None:
        self.tabs_at = tabs_at
        self.list.rect, self.info.rect = list_rect, info_rect
        if net_rect != self.net.rect:
            self.net.rect = net_rect
            self.net.layout()

    def handle(self, ev) -> bool:
        if self.list.handle(ev):
            self._load()
            return True
        self.info.handle(ev)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for source, rect in self.tab_rects.items():
                if rect.collidepoint(ev.pos):
                    self.switch(source)
                    return True
            return self.net.click(ev.pos)
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_TAB:
                self.switch(ARENA_SOURCES[(ARENA_SOURCES.index(self.source) + 1) % len(ARENA_SOURCES)])
            elif ev.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                self.switch(ARENA_SOURCES[ev.key - pygame.K_1])
            elif ev.key == pygame.K_r and self.source != CHAMPIONS:
                self.snapshot()
            elif ev.key == pygame.K_p:
                self.net.toggle_prune()
            else:
                return False
            return True
        return False

    def draw(self, s, font, small) -> None:
        x, y = self.tabs_at
        self.tab_rects = {}
        for source in ARENA_SOURCES:
            img = small.render(ARENA_TABS[source], True, PLAYER if source == self.source else DIM)
            s.blit(img, (x, y))
            self.tab_rects[source] = pygame.Rect(x, y, img.get_width(), img.get_height())
            if source == self.source:
                pygame.draw.line(s, PLAYER, (x, y + img.get_height()), (x + img.get_width(), y + img.get_height()), 1)
            x += img.get_width() + 10
        self.list.draw(s, small)
        if self.net.anat is None:
            s.blit(small.render("nessuna rete da mostrare", True, DIM), (self.net.rect.x + 10, self.net.rect.y + 10))
            return
        focus = self.net.focus()
        self.net.draw(s, small, focus)
        title = self.genomes[self.key][1]
        if self.key[0] == "v" and not self.eco.world.alive[self.genomes[self.key][3]]:
            title += " (morto)"
        s.blit(small.render(title, True, TARGET), (self.info.rect.x, self.info.rect.y - 20))
        lines = list(HINT) if focus is None else self.net.anat.describe(focus)
        self.info.draw(s, small, lines, focus)
