"""Esplorare le reti di una corsa.

`Explorer` mette insieme un elenco di reti (i campioni di ogni generazione o
una fotografia della popolazione), la rete scelta disegnata e i dettagli del
neurone toccato. Lo usano sia il cruscotto della corsa sia `Inspector`, che
aggiunge il genoma come matrice.

La rete è disegnata per distanza dai sensi: sensi a sinistra, poi una colonna
per ogni passo, motori a destra. Gli archi curvi tornano indietro (anelli,
cioè memoria). I rami morti (che non arrivano ai motori) sono in grigio.
"""

from pathlib import Path

import numpy as np
import pygame

from ..net.anatomy import KIND_NAMES, Anatomy, sizes
from ..net.sensors import CHANNELS
from .render import BG, BORDER, DEAD, DIM, EXCITE, HIT, IDLE, INHIBIT, MOCK, PLAYER, TARGET, TEXT

ROW = 20
LINE = 16
EXPORT_DIR = Path("runs") / "genomi"
POPULATION, SURVIVORS, CHAMPIONS = "population", "survivors", "champions"
SOURCES = (CHAMPIONS, SURVIVORS, POPULATION)
TAB_LABELS = {CHAMPIONS: "1 campioni", SURVIVORS: "2 sopravv.", POPULATION: "3 tutti"}


def arrow(surface, color, a, b, size: float = 9.0, back: float = 11.0) -> None:
    """Punta di freccia sulla linea da a a b, appena prima di b (il neurone)."""
    d = np.array(b, dtype=float) - np.array(a, dtype=float)
    n = np.hypot(*d)
    if n < 1e-6:
        return
    d /= n
    tip = np.array(b) - d * back
    side = np.array([-d[1], d[0]]) * size * 0.45
    base = tip - d * size
    pygame.draw.polygon(surface, color, [tuple(tip), tuple(base + side), tuple(base - side)])


class RowList:
    """Elenco scorrevole con una riga scelta. Righe: (testo, chiave).
    `header`: righe fisse in cima che dicono cosa c'è in ogni colonna."""

    def __init__(self):
        self.rows: list[tuple[str, object]] = []
        self.header: list[str] = []
        self.selected, self.scroll = 0, 0
        self.rect = pygame.Rect(0, 0, 0, 0)

    @property
    def top(self) -> int:
        """Dove comincia la prima riga, sotto l'intestazione."""
        return self.rect.y + LINE * len(self.header) + (4 if self.header else 0)

    @property
    def visible(self) -> int:
        return max((self.rect.bottom - self.top) // ROW, 1)

    def select(self, k: int) -> None:
        self.selected = int(np.clip(k, 0, max(len(self.rows) - 1, 0)))
        if self.selected < self.scroll:
            self.scroll = self.selected
        elif self.selected >= self.scroll + self.visible:
            self.scroll = self.selected - self.visible + 1

    def handle(self, ev) -> bool:
        """True se è cambiata la riga scelta."""
        if ev.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            self.scroll = int(np.clip(self.scroll - 3 * ev.y, 0, max(len(self.rows) - self.visible, 0)))
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.rect.collidepoint(ev.pos):
            k = self.scroll + (ev.pos[1] - self.top) // ROW
            if ev.pos[1] >= self.top and k < len(self.rows) and k != self.selected:
                self.select(k)
                return True
        elif ev.type == pygame.KEYDOWN and self.rows:
            step = {pygame.K_DOWN: 1, pygame.K_UP: -1, pygame.K_PAGEDOWN: 20, pygame.K_PAGEUP: -20,
                    pygame.K_HOME: -len(self.rows), pygame.K_END: len(self.rows)}.get(ev.key)
            if step is not None:
                old = self.selected
                self.select(self.selected + step)
                return self.selected != old
        return False

    def draw(self, s, font) -> None:
        for j, text in enumerate(self.header):
            s.blit(font.render(text, True, TEXT if j == 0 else BORDER), (self.rect.x + 6, self.rect.y + LINE * j))
        if self.header:
            y = self.top - 3
            pygame.draw.line(s, BORDER, (self.rect.x, y), (self.rect.right, y), 1)
        for j, k in enumerate(range(self.scroll, min(len(self.rows), self.scroll + self.visible))):
            y = self.top + j * ROW
            if k == self.selected:
                pygame.draw.rect(s, IDLE, (self.rect.x, y, self.rect.w, ROW))
            s.blit(font.render(self.rows[k][0], True, TEXT if k == self.selected else DIM), (self.rect.x + 6, y + 4))
        if len(self.rows) > self.visible:  # barra di scorrimento
            span = self.rect.bottom - self.top
            h = span * self.visible / len(self.rows)
            y = self.top + span * self.scroll / len(self.rows)
            pygame.draw.rect(s, BORDER, (self.rect.right - 3, y, 3, max(h, 8)))


class NetView:
    """Una rete disegnata e toccabile."""

    def __init__(self):
        self.anat: Anatomy | None = None
        self.prune = False
        self.node: int | None = None  # neurone fissato col clic
        self.pos: dict[int, tuple[int, int]] = {}
        self.rect = pygame.Rect(0, 0, 0, 0)

    def show(self, anat: Anatomy | None) -> None:
        self.anat, self.node = anat, None
        self.layout()

    def layout(self) -> None:
        """Ogni neurone dove sta davvero nel cervello (un quadrato di lato 1,
        vedi net/geometry.py): sensi sul bordo sinistro, motori sul destro,
        interni nella loro posizione. Un collegamento lungo sul disegno è
        lungo anche nel cervello, e costa in proporzione."""
        self.pos = {}
        a, G = self.anat, self.rect
        if a is None:
            return
        show = a.exists & (a.useful if self.prune else True)
        top, bottom = G.y + 64, G.bottom - 30
        left, right = G.x + 104, G.right - 72
        for v in range(a.n_nodes):
            if not show[v]:
                continue
            x, y = a.xy[v]
            if a.is_hidden(v):  # un po' di margine dalle colonne di sensi e motori
                px = left + 22 + x * (right - left - 44)
            else:
                px = left + x * (right - left)
            self.pos[v] = (int(px), int(top + y * (bottom - top)))
        self.box = pygame.Rect(left, top, right - left, bottom - top)

    def node_at(self, p) -> int | None:
        best, dist = None, 12.0
        for v, q in self.pos.items():
            d = np.hypot(p[0] - q[0], p[1] - q[1])
            if d < dist:
                best, dist = v, d
        return best

    def focus(self) -> int | None:
        """Il neurone sotto il mouse, altrimenti quello fissato."""
        hover = self.node_at(pygame.mouse.get_pos())
        return hover if hover is not None else self.node

    def click(self, p) -> bool:
        if not self.rect.collidepoint(p):
            return False
        v = self.node_at(p)
        self.node = None if v == self.node else v
        return True

    def toggle_prune(self) -> None:
        self.prune = not self.prune
        self.layout()

    def draw(self, s, font, focus) -> None:
        a, G = self.anat, self.rect
        if a is None:
            return
        sm = a.summary()
        head = (f"rete {KIND_NAMES[a.kind]}: {sm['neurons']} interni ({sm['useful_neurons']} utili), "
                f"{sm['links']} coll. ({sm['useful_links']} utili)")
        s.blit(font.render(head, True, TEXT), (G.x + 8, G.y + 4))
        sub = (f"sensi {sm['senses_used']}/{a.n_in}, anelli {sm['loops']}, "
               f"cablaggio {sm['wire_length']:.0f} ({sm['wire_cost']:.2f} en./s)")
        if "inhibitory" in sm:
            sub += f", {sm['neurons'] - sm['inhibitory']} ecc. + {sm['inhibitory']} inib."
        s.blit(font.render(sub, True, DIM), (G.x + 8, G.y + 20))
        for x, title in ((G.x + 84, "sensi"), (G.centerx - 90, "interni, dove stanno nel cervello"),
                         (G.right - 92, "motori")):
            s.blit(font.render(title, True, DIM), (x, G.y + 40))
        pygame.draw.rect(s, (30, 33, 44), self.box.inflate(10, 10), 1)  # il cervello

        layer = pygame.Surface(G.size, pygame.SRCALPHA)
        ox, oy = G.topleft
        pos = {v: (p[0] - ox, p[1] - oy) for v, p in self.pos.items()}
        edges = [(u, v) for u, v in a.edges() if u in pos and v in pos]
        w_max = max((abs(a.weight(u, v)) for u, v in edges), default=1.0)
        for u, v in edges:
            w = a.weight(u, v)
            dim = not (a.useful[u] and a.useful[v])
            color = DEAD if dim else EXCITE if w > 0 else INHIBIT
            strength = abs(w) / w_max
            alpha = int(30 + 170 * strength)
            if focus is not None:
                alpha = 235 if focus in (u, v) else 12
            width = 2 if strength > 0.5 and not dim else 1
            p, q = pos[u], pos[v]
            if u == v:
                pygame.draw.circle(layer, (*color, alpha), (p[0], p[1] - 12), 7, 1)
                continue
            pts = [p, q]  # dritto: la lunghezza sul disegno è quella nel cervello
            pygame.draw.lines(layer, (*color, alpha), False, pts, width)
            if focus is not None and focus in (u, v):
                arrow(layer, (*color, alpha), pts[-2], q)
        s.blit(layer, G.topleft)

        for v, p in self.pos.items():
            dead = not a.useful[v]
            if v < a.n_in:
                rim = HIT[v % len(CHANNELS)] if v < a.n_vision else TEXT
                pygame.draw.circle(s, IDLE, p, 5)
                pygame.draw.circle(s, DEAD if dead else rim, p, 5, 1)
                img = font.render(a.labels[v], True, DEAD if dead else DIM)
                s.blit(img, (p[0] - img.get_width() - 9, p[1] - 7))
            elif a.is_motor(v):
                pygame.draw.circle(s, MOCK, p, 11)
                s.blit(font.render(a.labels[v], True, TEXT), (p[0] + 15, p[1] - 7))
            else:
                rim = TEXT
                if a.inhibitory is not None:
                    rim = INHIBIT if a.inhibitory[v - a.n_in] else EXCITE
                pygame.draw.circle(s, (34, 38, 50), p, 10)
                pygame.draw.circle(s, DEAD if dead else rim, p, 10, 2)
                img = font.render(str(v - a.n_in), True, DEAD if dead else TEXT)
                s.blit(img, img.get_rect(center=p))
            if v == focus:
                pygame.draw.circle(s, PLAYER, p, 14, 1)
            if v == self.node:
                pygame.draw.circle(s, PLAYER, p, 16, 1)
        legend = [("eccita", EXCITE), ("inibisce", INHIBIT), ("frecce = verso", DIM),
                  ("rami morti " + ("nascosti" if self.prune else "grigi") + " [P]", DEAD)]
        if a.inhibitory is not None:
            legend[2:2] = [("bordo = tipo", TEXT)]
        x = G.x + 8
        for text, color in legend:
            img = font.render(text, True, color)
            s.blit(img, (x, G.bottom - 20))
            x += img.get_width() + 14


class InfoPanel:
    """Testo scorrevole con la rotella: i dettagli del neurone."""

    def __init__(self):
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.scroll, self.key = 0, None

    def handle(self, ev) -> None:
        if ev.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            self.scroll = max(0, self.scroll - 3 * ev.y)

    def draw(self, s, font, lines: list[str], key) -> None:
        if key != self.key:  # neurone nuovo: si riparte dall'inizio
            self.scroll, self.key = 0, key
        fit = max(self.rect.h // LINE - 1, 1)
        self.scroll = min(self.scroll, max(len(lines) - fit, 0))
        for j, text in enumerate(lines[self.scroll : self.scroll + fit]):
            k = j + self.scroll
            link = text.startswith("  ")  # una riga di collegamento: colorata per segno
            color = TEXT if k == 0 else EXCITE if link and " eccita" in text else \
                INHIBIT if link and " inibisce" in text else DIM
            s.blit(font.render(text, True, color), (self.rect.x, self.rect.y + j * LINE))
        if len(lines) > fit:
            more = f"rotella: righe {self.scroll + 1}-{min(self.scroll + fit, len(lines))} di {len(lines)}"
            s.blit(font.render(more, True, BORDER), (self.rect.x, self.rect.y + fit * LINE + 2))


HINT = ["Tocca un neurone per vedere i",
        "suoi collegamenti: da chi riceve",
        "e a chi manda, con peso e segno",
        "(eccita / inibisce).",
        "Il clic lo fissa, un altro clic",
        "lo libera. La rotella scorre."]


class Explorer:
    """Elenco di reti + rete disegnata + dettagli del neurone.

    I campioni si aggiornano da soli mentre la corsa va avanti: se è scelto il
    più recente, si segue l'ultimo arrivato. La popolazione è una fotografia
    (R la rifà), perché cambia tutta a ogni generazione.
    """

    def __init__(self, evo, source: str | None = None):
        self.evo, self.cfg = evo, evo.cfg
        self.n_in = evo.ctrl.brains.n_in
        self.source = source or (CHAMPIONS if evo.champions else POPULATION)
        self.list, self.net, self.info = RowList(), NetView(), InfoPanel()
        self.genomes: dict = {}
        self.tab_rects: dict = {}
        self.tabs_at = (0, 0)
        self.seen = None
        self.key = None
        self.snapshot()

    # --- righe ---------------------------------------------------------------

    def snapshot(self) -> None:
        """Ricostruisce le righe dell'elenco corrente."""
        evo = self.evo
        rows, self.genomes = [], {}
        if self.source in (POPULATION, SURVIVORS):
            genome, age, ids = evo.ctrl.brains.genome, evo.age.copy(), evo.ids.copy()
            neurons, links = sizes(genome, self.n_in)
            members = range(len(genome)) if self.source == POPULATION else np.flatnonzero(age > 0)
            for i in sorted(members, key=lambda i: (-age[i], ids[i])):
                key = (evo.generation, int(i))
                rows.append((f"{'#' + str(ids[i]):>6} {age[i]:>4d} {neurons[i]:>4d} {links[i]:>6d}", key))
                self.genomes[key] = (lambda i=i: genome.take([i]), f"gen{evo.generation:04d}-individuo{ids[i]}",
                                     f"individuo #{ids[i]}, età {age[i]}", f"#{ids[i]}")
            self.seen = None
            if self.source == SURVIVORS:
                what = (f"{len(rows)} rimasti dalla selezione" if evo.selection is not None
                        else f"{len(rows)} vincitori della gen. {evo.generation - 1}")
                self.list.header = [f"{'num.':>6} {'età':>4} {'neur':>4} {'colleg':>6}", what,
                                    "A: tutti insieme in arena"]
            else:
                self.list.header = [f"{'num.':>6} {'età':>4} {'neur':>4} {'colleg':>6}",
                                    f"tutti i {len(rows)} in gara, foto gen. {evo.generation}",
                                    "età 0 = figlio non ancora provato"]
        else:
            done = {h["generation"]: h["completed"] for h in evo.history}
            for gen, age, g, uid in reversed(evo.champions):
                neurons, links = sizes(g, self.n_in)
                pct = f"{done[gen] * 100 / self.cfg.population:3.0f}%" if gen in done else "-"
                rows.append((f"{gen:>5d} {pct:>5} {age:>4d} {neurons[0]:>4d} {links[0]:>6d}", gen))
                self.genomes[gen] = (lambda g=g: g, f"campione-gen{gen:04d}",
                                     f"campione gen. {gen}: #{uid}" if uid >= 0 else f"campione gen. {gen}",
                                     f"#{uid}" if uid >= 0 else f"gen{gen}")
            self.seen = len(evo.champions)
            self.list.header = [f"{'gen':>5} {'succ':>5} {'età':>4} {'neur':>4} {'colleg':>6}",
                                "campione = il vincitore più",
                                "anziano (a pari età il nato",
                                "prima). Non è scelto a caso."]
        follow = self.list.selected == 0
        self.list.rows = rows
        keys = [k for _, k in rows]
        self.list.select(0 if follow or self.key not in keys else keys.index(self.key))
        self._load()

    def refresh(self) -> None:
        """Da chiamare a ogni fotogramma: aggiunge i campioni nuovi."""
        if self.source == CHAMPIONS and self.seen != len(self.evo.champions):
            self.snapshot()

    def show_individual(self, index: int) -> None:
        """Mostra la rete dell'individuo al posto `index` della popolazione
        attuale (fa una foto nuova; tra i sopravvissuti, se lo è)."""
        self.source = SURVIVORS if self.evo.age[index] > 0 else POPULATION
        self.snapshot()
        keys = [k for _, k in self.list.rows]
        key = (self.evo.generation, index)
        if key in keys:
            self.list.select(keys.index(key))
            self._load()

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

    def selected(self):
        """(genoma, nome per i file, descrizione, etichetta breve) della rete
        scelta, o None."""
        if self.key is None:
            return None
        get, name, title, label = self.genomes[self.key]
        return get(), name, title, label

    # --- disposizione e comandi --------------------------------------------

    def place(self, tabs_at, list_rect, net_rect, info_rect) -> None:
        self.tabs_at = tabs_at
        self.list.rect, self.info.rect = list_rect, info_rect
        if net_rect != self.net.rect:
            self.net.rect = net_rect
            self.net.layout()

    def handle(self, ev) -> bool:
        """True se l'evento è stato usato."""
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
                self.switch(SOURCES[(SOURCES.index(self.source) + 1) % len(SOURCES)])
            elif ev.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                self.switch(SOURCES[ev.key - pygame.K_1])
            elif ev.key == pygame.K_r and self.source != CHAMPIONS:
                self.snapshot()
            elif ev.key == pygame.K_p:
                self.net.toggle_prune()
            else:
                return False
            return True
        return False

    # --- disegno -----------------------------------------------------------

    def draw(self, s, font, small) -> int | None:
        """Disegna elenco, rete e dettagli; restituisce il neurone in evidenza."""
        x, y = self.tabs_at
        self.tab_rects = {}
        for source in SOURCES:
            img = small.render(TAB_LABELS[source], True, PLAYER if source == self.source else DIM)
            s.blit(img, (x, y))
            self.tab_rects[source] = pygame.Rect(x, y, img.get_width(), img.get_height())
            if source == self.source:
                pygame.draw.line(s, PLAYER, (x, y + img.get_height()), (x + img.get_width(), y + img.get_height()), 1)
            x += img.get_width() + 10
        self.list.draw(s, small)
        if self.net.anat is None:
            msg = {CHAMPIONS: "nessun campione: serve una generazione con almeno un vincitore",
                   SURVIVORS: "nessun sopravvissuto: l'ultima generazione non ha vinto nessuno",
                   POPULATION: "popolazione vuota"}[self.source]
            s.blit(small.render(msg, True, DIM), (self.net.rect.x + 10, self.net.rect.y + 10))
            return None
        focus = self.net.focus()
        self.net.draw(s, small, focus)
        title = self.genomes[self.key][2]
        if self.source != CHAMPIONS:
            title += " (R aggiorna)"
        s.blit(small.render(title, True, TARGET), (self.info.rect.x, self.info.rect.y - 20))
        lines = list(HINT) if focus is None else self.net.anat.describe(focus)
        self.info.draw(s, small, lines, focus)
        return focus


class Inspector:
    """Schermata intera: `Explorer` più il genoma come matrice. La corsa, se
    c'è, continua in sottofondo."""

    LIST_W, GENOME_W, TOP = 300, 430, 44

    def __init__(self, app, evo, back, title: str = "", source: str | None = None):
        self.app, self.evo, self.back, self.title = app, evo, back, title
        self.explorer = Explorer(evo, source)
        self.message = ""
        self.cell = (1, 1)
        self.heat_rect = pygame.Rect(0, 0, 0, 0)

    def _export(self) -> None:
        sel = self.explorer.selected()
        if sel is None:
            return
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        _, name, title = sel
        path = EXPORT_DIR / f"{name}.txt"
        path.write_text(self.explorer.net.anat.genome_text(f"{self.title}  {title}".strip()))
        self.message = f"genoma scritto in {path}"

    def _cell_at(self, p):
        """(destinatario, sorgente) della cella della matrice sotto il mouse."""
        if self.explorer.net.anat is None or not self.heat_rect.collidepoint(p):
            return None
        return (p[1] - self.heat_rect.y) // self.cell[1], (p[0] - self.heat_rect.x) // self.cell[0]

    def handle(self, ev) -> None:
        ex = self.explorer
        if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_i):
            self.app.show(self.back)
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_x:
            self._export()
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_t:
            self.app.test(ex.selected(), back=self)
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_a:
            self.message = self.app.test_survivors(self.evo, back=self)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and (cell := self._cell_at(ev.pos)):
            ex.net.node = ex.net.anat.n_in + cell[0]
        else:
            ex.handle(ev)

    def update(self) -> None:
        self.explorer.refresh()

    def close(self) -> None:
        self.back.close()

    def draw(self, s) -> None:
        app, ex = self.app, self.explorer
        s.fill(BG)
        W, H = s.get_size()
        lw, gw, top = self.LIST_W, self.GENOME_W, self.TOP
        info_y = top + 400
        ex.place((10, 12), pygame.Rect(0, top, lw - 6, H - top - 4),
                 pygame.Rect(lw, top, W - lw - gw, H - top),
                 pygame.Rect(W - gw + 14, info_y, gw - 20, H - info_y - 30))
        pygame.draw.line(s, BORDER, (0, top - 4), (W, top - 4), 1)
        pygame.draw.line(s, BORDER, (lw - 4, top), (lw - 4, H), 1)
        pygame.draw.line(s, BORDER, (W - gw, top), (W - gw, H), 1)
        hint = ("Su/Giù rete | Tab elenco | clic neurone | P rami morti | T prova la rete | "
                "A sopravvissuti in arena | X esporta | Esc indietro")
        s.blit(app.small.render(hint, True, DIM), (W - app.small.size(hint)[0] - 10, 8))
        focus = ex.draw(s, app.font, app.small)
        if ex.net.anat is not None:
            self._draw_genome(s, focus)
        status = self.message or app.run_status()
        if status:
            img = app.small.render(status, True, TARGET)
            s.blit(img, (W - img.get_width() - 10, 28))

    def _draw_genome(self, s, focus) -> None:
        a, app = self.explorer.net.anat, self.app
        W = s.get_width()
        x0, top = W - self.GENOME_W + 14, self.TOP
        T, S = a.links.shape
        self.cell = cw, ch = max(1, 400 // S), max(2, min(8, 280 // T))
        self.heat_rect = r = pygame.Rect(x0, top + 40, S * cw, T * ch)
        s.blit(app.small.render("genoma: un blocco (riga) per neurone", True, TEXT), (x0, top + 4))
        s.blit(app.small.render("colonne: sensi | interni | motori che lo alimentano", True, DIM), (x0, top + 20))

        rgb = np.empty((S, T, 3), dtype=np.uint8)
        rgb[:] = (24, 27, 36)
        strength = np.clip(np.abs(a.w) / 2.0, 0.0, 1.0).T[:, :, None]
        hot = np.where((a.w > 0).T[:, :, None], np.array(EXCITE), np.array(INHIBIT))
        live = (np.array(IDLE) + (hot - np.array(IDLE)) * strength).astype(np.uint8)
        links, latent = a.links.T, (a.conn & ~a.links).T
        rgb[links] = live[links]
        rgb[latent] = (58, 60, 70)
        rgb[:, ~a.exists[a.n_in:]] //= 2  # righe dei neuroni spenti: più scure
        s.blit(pygame.transform.scale(pygame.surfarray.make_surface(rgb), r.size), r)
        for col in (a.n_in, a.n_in + a.n_hidden):
            pygame.draw.line(s, BORDER, (r.x + col * cw, r.y - 4), (r.x + col * cw, r.bottom + 4), 1)
        pygame.draw.line(s, BORDER, (r.x - 4, r.y + a.n_hidden * ch), (r.right + 4, r.y + a.n_hidden * ch), 1)
        pygame.draw.rect(s, BORDER, r.inflate(2, 2), 1)
        if focus is not None:
            if focus >= a.n_in:  # il suo blocco
                pygame.draw.rect(s, PLAYER, (r.x - 2, r.y + (focus - a.n_in) * ch - 1, r.w + 4, ch + 2), 1)
            pygame.draw.rect(s, PLAYER, (r.x + focus * cw - 1, r.y - 2, cw + 2, r.h + 4), 1)  # dove parla

        cell = self._cell_at(pygame.mouse.get_pos())
        if cell is not None:
            t, src = cell
            gene = "attivo" if a.links[t, src] else "latente: un capo è spento" if a.conn[t, src] else "assente"
            w = a.w[t, src] if a.links[t, src] else a.brains.weights[0, t, src]
            text = f"{a.labels[src]} -> {a.labels[a.n_in + t]}: {w:+.3f} ({gene})"
        else:
            text = "colore: segno e forza | grigio: latente | scuro: spento"
        s.blit(app.small.render(text, True, TEXT if cell else DIM), (x0, r.bottom + 8))
