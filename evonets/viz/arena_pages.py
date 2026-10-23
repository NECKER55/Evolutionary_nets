"""Le schede dell'arena continua oltre alla vista del mondo: la panoramica
(a che punto siamo, grafici minuto per minuto con una frase che li spiega) e
le reti (l'esploratore, con gli elenchi adatti a un mondo senza generazioni)."""

import numpy as np
import pygame

from ..evo.ecology import DIAL_NAMES, DIALS
from ..net.anatomy import Anatomy
from ..net.brain import live_links
from .inspector import HINT, InfoPanel, NetView, RowList
from .overview import PANEL, para
from .render import DIM, FOOD, FRUIT, MODULATOR, PLAYER, PREDATOR, TARGET, TEXT, plot

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


def legend(s, font, items, x: int, y: int) -> None:
    """Una riga di legenda: (colore, nome)."""
    for color, name in items:
        pygame.draw.line(s, color, (x, y + 7), (x + 16, y + 7), 2)
        img = font.render(name, True, DIM)
        s.blit(img, (x + 20, y))
        x += img.get_width() + 34


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
        return rect.y + 36

    def draw(self, s, area: pygame.Rect, eco) -> None:
        gap = 12
        half = (area.w - gap) // 2
        left = pygame.Rect(area.x, area.y, half, area.h)
        right = pygame.Rect(left.right + gap, area.y, area.w - half - gap, area.h)
        top = pygame.Rect(left.x, left.y, left.w, 232)
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
            color = TARGET if v >= 1.0 else PLAYER if k == current and v > 0 else PLAYER if k == current else DIM
            pygame.draw.rect(s, (40, 44, 56), (x0, y + 4, 120, 9))
            pygame.draw.rect(s, color, (x0, y + 4, int(120 * v), 9))
            state = "fatto" if v >= 1.0 else ("in corso" if k == current else "da venire")
            s.blit(app.small.render(f"{k + 1}. {name}  {v:.0%}  ({state})", True,
                                    TEXT if current is None or k <= current else DIM),
                   (x0 + 130, y))
            y += 17
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
        para(s, app.font, text, x0, y + 6, rect.w - 28)

    # --- grafici -----------------------------------------------------------------

    def _series(self, hist, key, scale: float = 1.0):
        return np.array([float(r.get(key, 0.0) or 0.0) * scale for r in hist[-WINDOW:]])

    def _plot_box(self, rect, y, legend_rows: int = 1) -> pygame.Rect:
        return pygame.Rect(rect.x + 56, y + 18 * legend_rows + 4, rect.w - 72, rect.bottom - y - 18 * legend_rows - 52)

    def _population(self, s, rect, eco) -> None:
        app, hist = self.app, eco.history
        y = self._section(s, rect, "POPOLAZIONE", f"ultimi {min(len(hist), WINDOW)} minuti")
        if not hist:
            para(s, app.font, "Le misure arrivano a ogni minuto di mondo.", rect.x + 14, y, rect.w - 28, DIM)
            return
        series = [(self._series(hist, "alive"), PLAYER, 2), (self._series(hist, "births"), GREEN, 1),
                  (self._series(hist, "starved"), FOOD, 1), (self._series(hist, "thirst"), WATER, 1),
                  (self._series(hist, "old_age"), GREY, 1),
                  (self._series(hist, "killed"), PREDATOR, 1), (self._series(hist, "colonists"), MODULATOR, 1)]
        legend(s, app.small, [(PLAYER, "vivi"), (GREEN, "nati/min"), (FOOD, "fame/min"), (WATER, "sete"),
                              (GREY, "vecchiaia"), (PREDATOR, "predati"), (MODULATOR, "colonizzatori")],
               rect.x + 14, y)
        box = self._plot_box(rect, y)
        top = max(max(v.max() for v, _, _ in series), 1.0)
        plot(s, box, app.small, series, top, fmt="{:4.0f}")
        r = hist[-1]
        text = (f"Ultimo minuto: {r['alive']} vivi, {r['births']} nati ({r['mates']} con partner), "
                f"{r['starved']} morti di fame, {r.get('thirst', 0)} di sete, {r['old_age']} di vecchiaia, "
                f"{r.get('killed', 0)} predati, "
                f"{r['colonists']} colonizzatori. Età media {r.get('age_mean', 0):.0f} s.")
        para(s, app.small, text, rect.x + 14, box.bottom + 8, rect.w - 28)

    def _brains(self, s, rect, eco) -> None:
        app, hist = self.app, eco.history
        y = self._section(s, rect, "CERVELLO E CORPO", "in media sui vivi")
        if not hist:
            return
        series = [(self._series(hist, "used"), TARGET, 2), (self._series(hist, "neurons"), MODULATOR, 1),
                  (self._series(hist, "links", 0.1), GREY, 1), (self._series(hist, "size_mean"), GREEN, 1),
                  (self._series(hist, "bite_mean"), FOOD, 1), (self._series(hist, "glow_mean"), FRUIT, 1)]
        legend(s, app.small, [(TARGET, "sensi usati"), (MODULATOR, "neuroni"), (GREY, "collegam./10"),
                              (GREEN, "taglia"), (FOOD, "morso"), (FRUIT, "luce")], rect.x + 14, y)
        box = self._plot_box(rect, y)
        top = max(max(v.max() for v, _, _ in series), 1.0)
        plot(s, box, app.small, series, top, fmt="{:4.1f}")
        r, first = hist[-1], hist[0]
        trend = r.get("used", 0) - first.get("used", 0)
        text = (f"Sensi usati {r.get('used', 0):.2f} ({'+' if trend >= 0 else ''}{trend:.2f} dall'inizio), "
                f"neuroni {r.get('neurons', 0):.2f}, taglia {r.get('size_mean', 0):.2f}. "
                + ("I sensi calano: il mondo si lascia vivere alla cieca." if trend < -0.3 else
                   "I sensi crescono: vedere e annusare conviene." if trend > 0.3 else ""))
        para(s, app.small, text, rect.x + 14, box.bottom + 8, rect.w - 28)

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
        legend(s, app.small, [(PLAYER, "normale"), (MODULATOR, "senza plasticità")], rect.x + 14, y)
        box = self._plot_box(rect, y)
        series = [(a, PLAYER, 2)] + ([(b[~np.isnan(b)], MODULATOR, 1)] if (~np.isnan(b)).any() else [])
        plot(s, box, app.small, series, max(10.0, np.nanmax(a) * 1.2, np.nanmax(np.nan_to_num(b)) * 1.2),
             fmt="{:3.0f}%")
        last = normal[-1]
        p = probe.get(last["generation"])
        text = f"Minuto {last['generation']}: {last['score']:.0%}"
        if p is not None:
            d = (last["score"] - p) * 100
            text += (f", senza plasticità {p:.0%}: " + ("imparano durante la vita." if d > 2 else
                                                         "la plasticità peggiora: si confondono." if d < -2 else
                                                         "nessuna differenza, per ora non imparano."))
        para(s, app.small, text, rect.x + 14, box.bottom + 8, rect.w - 28)

    @staticmethod
    def _returns(r) -> str:
        """Imparano dove sta l'acqua? Il ritorno alla pozza rispetto alla prima volta."""
        ratio = r.get("return_ratio")
        if ratio is None:
            return ""
        verdict = ("tornano prima: imparano dove sta l'acqua" if ratio < 0.8 else
                   "ci mettono di più che la prima volta" if ratio > 1.2 else "come la prima volta: non imparano")
        return f"Chi ha già bevuto torna a bere in {ratio:.0%} del tempo della prima volta ({r['return_n']} esseri): {verdict}."

    def _world(self, s, rect, eco) -> None:
        app, hist = self.app, eco.history
        y = self._section(s, rect, "IL MONDO")
        if not hist:
            return
        series = [(self._series(hist, "eaten"), FOOD, 2), (self._series(hist, "bad_eaten"), FRUIT, 1),
                  (self._series(hist, "food", 0.1), GREY, 1), (self._series(hist, "patch_health", 100), GREEN, 1),
                  (self._series(hist, "water_mean", 100), WATER, 1), (self._series(hist, "drink_gap"), DRINK, 1)]
        legend(s, app.small, [(FOOD, "mangiati/min"), (FRUIT, "cattivi"), (GREY, "cibo/10"), (GREEN, "macchie %"),
                              (WATER, "acqua %"), (DRINK, "s tra bevute")], rect.x + 14, y)
        box = self._plot_box(rect, y)
        top = max(max(v.max() for v, _, _ in series), 1.0)
        plot(s, box, app.small, series, top, fmt="{:4.0f}")
        r = hist[-1]
        text = (f"Cibo presente {r['food']}, mangiati {r['eaten']} nell'ultimo minuto"
                + (f", {r.get('bad_eaten', 0)} frutti cattivi" if r.get("bad_eaten") else "")
                + (f"; salute media delle macchie {r.get('patch_health', 1):.0%}" if r.get("patch_health", 1) < 0.995
                   else "")
                + (f"; energia data ai figli {r.get('fed', 0):.0f}" if r.get("fed") else "")
                + (f"; temperatura dove vivono {r['temp_mean']:.2f}" if "temp_mean" in r
                   and eco.cfg.arena_temperature > 0 else "")
                + (f"; {r['pools']} pozze, acqua nei corpi {r.get('water_mean', 1):.0%}" if eco.cfg.arena_water
                   and "pools" in r else "")
                + (f"; tra due bevute {r['drink_gap']:.0f} s" if r.get("drink_gap") else "") + ". "
                + self._returns(r))
        para(s, app.small, text, rect.x + 14, box.bottom + 8, rect.w - 28)


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
