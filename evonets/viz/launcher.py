"""Il menu: si sceglie la corsa da lanciare (rete, individui, mondo, scala,
seme, generazioni) oppure si apre un salvataggio, per continuarlo o per
esplorarne le reti. `Screen` è la base di tutte le schermate."""

import json
import secrets
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pygame

from ..arenas.generational.evolution import Evolution
from ..arenas.mondo3d.bodies import BODIES
from ..brains.freeform.anatomy import KIND_NAMES
from ..config import Config
from .render import BG, BORDER, DIM, IDLE, PLAYER, TARGET, TEXT
from .widgets import ROW_H, Button, Choice, Number, Picker, Text

RUNS = Path("runs")
DAYS = ("lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica")
MONTHS = ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre",
          "ottobre", "novembre", "dicembre")


class Screen:
    def handle(self, ev) -> None: ...
    def update(self) -> None: ...
    def draw(self, surface) -> None: ...
    def close(self) -> None:
        """La finestra si sta chiudendo."""


def pile_meta(path) -> dict | None:
    """La descrizione di un salvataggio dell'arena delle pile (None se è d'altro tipo)."""
    try:
        with np.load(path) as d:
            if "kind" in d and str(d["kind"]) == "pile":
                return json.loads(str(d["meta"]))
    except Exception:
        pass
    return None


def pile_facts(meta: dict) -> str:
    les = f", senza {','.join(meta['lesions'])}" if meta.get("lesions") else ""
    return (f"arena delle pile · {meta['agent']}{les} · fase {meta['phase']} · {meta['enclosures']} recinti × "
            f"{meta['beings']} · minuto {meta['minutes']:.1f}")


def save_info(path) -> str:
    """Una riga che descrive un salvataggio, senza caricarlo per intero."""
    meta = pile_meta(path)
    if meta is not None:
        return pile_facts(meta)
    try:
        with np.load(path) as d:
            if "kind" in d and str(d["kind"]) == "arena":
                return summarize(path).facts
            if "pos" not in d:
                return "versione vecchia, senza geometria del cervello: non si può aprire"
            net = str(d["net_type"]) if "net_type" in d else "rate"
            n = len(d["weights"])
            text = f"generazione {int(d['generation'])}, {n} individui, rete {KIND_NAMES.get(net, net)}"
            if "seed" in d:
                text += f", seme {int(d['seed'])}"
            if "config" in d:
                saved = json.loads(str(d["config"]))
                if saved.get("task") == "cibo":
                    mode = {"scala": "scala adattiva", "basico": "controllo basico",
                            "complesso": "controllo complesso"}.get(saved.get("ladder"), "manopole ferme")
                    text += f", mondo del cibo ({mode})"
            if "exams" in d:
                exams = json.loads(str(d["exams"]))
                if exams:
                    text += f", {len(exams)} esami (ultimo {exams[-1]['score']:.0%})"
            if "log" in d:
                changes = sum(e["event"] == "cambio" for e in json.loads(str(d["log"])))
                text += (f", ambiente cambiato {changes} {'volta' if changes == 1 else 'volte'}" if changes
                         else ", ambiente invariato")
            else:
                text += ", formato vecchio senza configurazione (si useranno i valori attuali)"
            if "ch_generation" in d:
                text += f", {len(d['ch_generation'])} campioni"
            return text
    except Exception as err:  # file illeggibile o di un altro formato
        return f"illeggibile: {err}"


# --- L'elenco dei salvataggi ------------------------------------------------


@dataclass
class SaveSummary:
    """Ciò che serve per riconoscere un salvataggio in un elenco."""
    path: str
    when: datetime  # ultima modifica del file
    name: str
    facts: str  # generazione, tipo di corsa, esame, a parole brevi
    error: bool = False

    def text(self) -> str:
        """Tutto ciò su cui si può cercare."""
        return f"{self.name} {self.facts}".lower()


_SUMMARIES: dict[tuple[str, float], SaveSummary] = {}


def _rows(path) -> int | None:
    """Quanti individui, dalla forma dei pesi, senza decomprimerli."""
    with zipfile.ZipFile(path) as z, z.open("weights.npy") as f:
        fmt = np.lib.format
        header = fmt.read_array_header_1_0 if fmt.read_magic(f) == (1, 0) else fmt.read_array_header_2_0
        shape = header(f)[0]
    return int(shape[0])


def summarize(path) -> SaveSummary:
    """Il riepilogo di un salvataggio, letto solo dalle parti piccole del file
    e ricordato finché il file non cambia."""
    path = Path(path)
    when = datetime.fromtimestamp(path.stat().st_mtime)
    key = (str(path), when.timestamp())
    if key not in _SUMMARIES and pile_meta(path) is not None:
        _SUMMARIES[key] = SaveSummary(str(path), when, path.name, pile_facts(pile_meta(path)))
    if key not in _SUMMARIES:
        try:
            with np.load(path) as d:
                saved = json.loads(str(d["config"])) if "config" in d else {}
                net = str(d["net_type"]) if "net_type" in d else saved.get("net_type", "rate")
                if "kind" in d and str(d["kind"]) == "arena":
                    facts = [f"arena continua, {int(d['generation'])} min", f"{int(d['alive'].sum())} vivi su "
                             f"{_rows(path)}"]
                else:
                    facts = [f"gen. {int(d['generation'])}", f"{_rows(path)} ind."]
                facts.append("genoma " + (saved.get("genome") or "diretto"))
                facts.append({"rate": "rete continua", "spike-ei": "a impulsi"}.get(net, net))
                facts.append("sensi fissi" if saved.get("sim_mode") == "cervello" else "sensi che evolvono")
                if saved.get("task") == "arena":
                    pass
                elif saved.get("task") == "cibo":
                    where = " 0,00"
                    if "log" in d:
                        steps = [e for e in json.loads(str(d["log"])) if e.get("event") == "scala"]
                        if steps:
                            where = f" {steps[-1]['position']:.2f}".replace(".", ",")
                    facts.append({"scala": f"cibo, scala{where}", "basico": "cibo, controllo basico",
                                  "complesso": "cibo, controllo complesso"}.get(saved.get("ladder"), "cibo"))
                else:
                    facts.append("bersaglio")
                exams = json.loads(str(d["exams"])) if "exams" in d else []
                if exams:
                    facts.append(f"esame {exams[-1]['score']:.0%}")
            _SUMMARIES[key] = SaveSummary(str(path), when, path.name, " · ".join(facts))
        except Exception as err:  # file illeggibile o di un altro formato
            _SUMMARIES[key] = SaveSummary(str(path), when, path.name, f"illeggibile: {err}", error=True)
    return _SUMMARIES[key]


def day_label(day: date, today: date) -> str:
    """"Oggi, 23 ottobre", "Ieri, 22 ottobre", "martedì 20 ottobre" (con
    l'anno se non è quello di oggi)."""
    text = f"{day.day} {MONTHS[day.month - 1]}" + (f" {day.year}" if day.year != today.year else "")
    if day == today:
        return f"Oggi, {text}"
    if day == today - timedelta(days=1):
        return f"Ieri, {text}"
    return f"{DAYS[day.weekday()]} {text}"


def grouped(summaries, today: date, query: str = "") -> list[tuple[str, list[SaveSummary]]]:
    """I salvataggi che contengono tutte le parole di `query`, raggruppati per
    giorno: dal più recente, e nel giorno dal più recente."""
    words = query.lower().split()
    days: dict[date, list[SaveSummary]] = {}
    for x in sorted(summaries, key=lambda x: x.when, reverse=True):
        if all(w in x.text() for w in words):
            days.setdefault(x.when.date(), []).append(x)
    return [(day_label(day, today), items) for day, items in days.items()]


class SavePicker:
    """L'elenco dei salvataggi sopra il menu: raggruppati per giorno, con un
    riepilogo per ciascuno; si scorre con frecce o rotella, si filtra
    scrivendo, Invio o clic sceglie, Esc chiude senza cambiare."""

    ITEM_H, HEAD_H = 40, 28

    def __init__(self, files, current):
        self.summaries = [summarize(p) for p in files]
        self.query = ""
        self.scroll = 0
        self.rows: list[tuple[str, object]] = []  # ("nessuno"/"giorno"/"file", dato), per disegnare
        self.rects: list[tuple[pygame.Rect, str | None]] = []
        self._build()
        paths = [v for kind, v in self.rows if kind != "giorno"]
        self.pick = paths.index(current) if current in paths else 0

    def _build(self) -> None:
        self.rows = [("nessuno", None)] if not self.query else []
        for label, items in grouped(self.summaries, date.today(), self.query):
            self.rows.append(("giorno", label))
            self.rows += [("file", x) for x in items]
        self.choices = [None if kind == "nessuno" else v.path for kind, v in self.rows if kind != "giorno"]
        self.pick = min(getattr(self, "pick", 0), max(len(self.choices) - 1, 0))

    def handle(self, ev):
        """None finché si sceglie; ("scelto", percorso o None) o ("annullato",)."""
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                return ("annullato",)
            if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                return ("scelto", self.choices[self.pick]) if self.choices else None
            step = {pygame.K_DOWN: 1, pygame.K_UP: -1, pygame.K_PAGEDOWN: 8, pygame.K_PAGEUP: -8}.get(ev.key)
            if step:
                self.pick = int(np.clip(self.pick + step, 0, max(len(self.choices) - 1, 0)))
            elif ev.key == pygame.K_HOME:
                self.pick = 0
            elif ev.key == pygame.K_END:
                self.pick = max(len(self.choices) - 1, 0)
            elif ev.key == pygame.K_BACKSPACE:
                self.query = self.query[:-1]
                self._build()
            elif ev.unicode and ev.unicode.isprintable():
                self.query += ev.unicode
                self.pick = 0
                self._build()
        elif ev.type == pygame.MOUSEWHEEL:
            self.scroll = max(0, self.scroll - ev.y * self.ITEM_H)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for rect, path in self.rects:
                if rect.collidepoint(ev.pos):
                    return ("scelto", path)
        return None

    def _keep_visible(self, top: int, height: int) -> None:
        """Scorre l'elenco perché la riga scelta resti a vista."""
        y, k = 0, 0
        for kind, _ in self.rows:
            h = self.HEAD_H if kind == "giorno" else self.ITEM_H
            if kind != "giorno":
                if k == self.pick:
                    self.scroll = min(max(self.scroll, y + h - height), y)
                    if k == 0:
                        self.scroll = 0  # in cima: si vede anche il primo giorno
                    return
                k += 1
            y += h

    def draw(self, s, app, moved: bool) -> None:
        w, h = s.get_size()
        shade = pygame.Surface((w, h), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 170))  # il menu resta dietro, in ombra
        s.blit(shade, (0, 0))
        box = pygame.Rect(60, 30, min(w - 120, 1100), h - 60)
        pygame.draw.rect(s, (20, 22, 30), box)
        pygame.draw.rect(s, BORDER, box, 1)
        s.blit(app.big.render("Scegli un salvataggio", True, PLAYER), (box.x + 20, box.y + 14))
        query = self.query + ("_" if pygame.time.get_ticks() // 500 % 2 else " ")
        hint = f"filtro: {query}" if self.query else "scrivi per filtrare (nome, genoma, rete, sensi...)"
        s.blit(app.font.render(hint, True, TEXT if self.query else DIM), (box.x + 20, box.y + 50))
        s.blit(app.small.render(f"{len(self.summaries)} salvataggi in runs/   ·   Su/Giù, rotella   ·   Invio o clic "
                                f"sceglie   ·   Esc annulla", True, DIM), (box.x + 20, box.y + 72))
        top, bottom = box.y + 96, box.bottom - 10
        if moved:
            self._keep_visible(top, bottom - top)
        clip = s.get_clip()
        s.set_clip(pygame.Rect(box.x, top, box.w, bottom - top))
        self.rects = []
        y, k = top - self.scroll, 0
        mouse = pygame.mouse.get_pos()
        for kind, v in self.rows:
            if kind == "giorno":
                if top - self.HEAD_H < y < bottom:
                    s.blit(app.font.render(v, True, TARGET), (box.x + 20, y + 6))
                    pygame.draw.line(s, BORDER, (box.x + 20, y + self.HEAD_H - 3), (box.right - 20, y + self.HEAD_H - 3))
                y += self.HEAD_H
                continue
            rect = pygame.Rect(box.x + 14, y, box.w - 28, self.ITEM_H - 4)
            if top - self.ITEM_H < y < bottom:
                chosen = k == self.pick
                if chosen or rect.collidepoint(mouse):
                    pygame.draw.rect(s, (70, 58, 24) if chosen else IDLE, rect, border_radius=4)
                if kind == "nessuno":
                    s.blit(app.font.render("Nessuno: reti nuove", True, TEXT), (rect.x + 10, y + 11))
                else:
                    s.blit(app.font.render(f"{v.when:%H:%M}   {v.name}", True, TEXT), (rect.x + 10, y + 3))
                    facts = v.facts
                    while len(facts) > 4 and app.small.size(facts)[0] > rect.w - 80:
                        facts = facts[:-4] + "..."
                    s.blit(app.small.render(facts, True, (255, 110, 90) if v.error else DIM), (rect.x + 70, y + 21))
                self.rects.append((rect, v if kind == "nessuno" else v.path))
            k += 1
            y += self.ITEM_H
        if not self.choices:
            s.blit(app.font.render("nessun salvataggio corrisponde al filtro", True, DIM), (box.x + 20, top + 10))
        s.set_clip(clip)


class Launcher(Screen):
    """Il menu: si sceglie il tipo di corsa e la si lancia, oppure si apre un
    salvataggio per esplorarne le reti."""

    def __init__(self, app, defaults: dict):
        self.app = app
        self.message, self.error = "", False
        self.mode = Choice("Modalità", [("fast", "veloce: statistiche e reti, F per l'arena"),
                                         ("max", "massima velocità: solo statistiche"),
                                         ("live", "dal vivo: arena sempre a schermo")],
                           help="Il risultato non cambia con la modalità: cambia solo quanto in fretta arriva.")
        self.net = Choice("Rete", [("rate", "continua (tanh)"), ("spike-ei", "a impulsi, eccitatori/inibitori")],
                          help="Tipo di neurone. Ignorato se si riprende un salvataggio.")
        self.net.select(defaults.get("net", "rate"))
        self.agents = Number("Individui", defaults.get("agents") or Config().population, step=50, low=2,
                             help="Dimensione della popolazione. Ignorato se si riprende un salvataggio.")
        fixed = defaults.get("seed") is not None
        self.chance = Choice("Casualità", [("random", "seme casuale, nuovo a ogni corsa"),
                                            ("fixed", "seme fisso (riga sotto): corsa ripetibile")],
                             index=1 if fixed else 0,
                             help="Il seme usato viene comunque mostrato e salvato: ogni corsa si può rifare.")
        self.seed = Number("Seme", defaults.get("seed") or 0,
                           help="Stesso seme e stesse impostazioni = stessa corsa.")
        self.sim = Choice("Simulazione", [("evoluzione", "completa: evolvono sensi e cervello"),
                                          ("cervello", "solo cervello: sensi fissi (set naturale)")],
                          help="Solo cervello: tutti hanno gli stessi 10 sensi (brains/freeform/senses.py), che non "
                               "cambiano; evolve solo il cervello. Ignorato se si riprende un salvataggio.")
        self.sim.select(defaults.get("sim") or "evoluzione")
        self.genome = Choice("Genoma", [("diretto", "scrive ogni sinapsi"),
                                        ("sviluppo", "sviluppo: marcatori e affinità")],
                             help="Sviluppo: il cervello nasce dai marcatori delle cellule e da una tabella di "
                                  "affinità (docs/SPEC_SVILUPPO.md). Ignorato se si riprende un salvataggio.")
        self.genome.select(defaults.get("genome") or "diretto")
        self.task = Choice("Mondo", [("bersaglio", "bersaglio e veleno (il mondo di prima)"),
                                     ("cibo", "nutrirsi: cibo, prede e predatori"),
                                     ("arena", "arena continua: tutti insieme, nascite e morti"),
                                     ("pile", "arena delle pile: cervello a sezioni, sensi fissi"),
                                     ("mondo3d", "mondo 3D: pile con MuJoCo, telecamera e profondità")],
                           help="Il compito. Nel mondo del cibo si riproduce chi mangia almeno N cibi; il veleno "
                                "è spento. Nell'arena continua si vive insieme, senza generazioni "
                                "(docs/SPEC_ARENA.md). Pile: docs/SPEC_ARENA_PILE.md. "
                                "Mondo 3D: docs/SPEC_CERVELLO_RAGIONANTE.md.")
        self.task.select(defaults.get("task") or "bersaglio")
        self.ladder = Choice("Scala", [("scala", "adattiva: sale se va bene, scende se va male"),
                                       ("basico", "ferma al mondo basico (controllo)"),
                                       ("complesso", "ferma al mondo complesso (controllo)")],
                             help="Solo nel mondo del cibo: come si muovono le manopole della difficoltà "
                                  "(docs/SPEC_MONDO.md).")
        self.ladder.select(defaults.get("ladder") or "scala")
        self.cap = Number("Tetto", Config().arena_cap, step=500, low=10,
                          help="Solo nell'arena continua: quanti esseri vivi al massimo (i posti). La popolazione "
                               "la decide il cibo; il tetto evita l'esplosione. Individui = quanti all'inizio.")
        self.gens = Number("Generazioni", 0, step=50,
                           help="Dopo quante generazioni fermarsi e salvare. 0 = senza limite.")
        self.exam_every = Number("Esame ogni", Config().exam_every, step=50,
                                 help="Ogni quante generazioni mettere all'esame fisso (7 prove; 8 livelli nel "
                                      "mondo del cibo) i 20 più anziani; non cambia la corsa. 0 = mai.")
        self.select = Number("Selezione finale", 0, step=1,
                             help="Finite le generazioni, i sopravvissuti rifanno la prova questo numero di "
                                  "volte, senza figli: chi fallisce è eliminato. 0 = niente.")
        # L'arena delle pile (docs/SPEC_ARENA_PILE.md): righe proprie.
        self.p_agent = Choice("Cervello", [("sezioni", "a sezioni (A B C D E L U V F)"),
                                           ("attore-critico", "attore-critico classico (riferimento)"),
                                           ("guidato", "guidato a mano: sa colore e zona (controllo)")],
                              help="Chi guida gli esseri: il progetto, il riferimento o il controllo del mondo.")
        self.p_phase = Choice("Fase", [("A", "A: popolazione fissa, si misura l'apprendimento"),
                                       ("B", "B: accoppiamento e rinascita da incrocio (evoluzione)")],
                              help="A: si rinasce col cervello di partenza. B: coi geni di una coppia.")
        self.p_lesion = Choice("Lesione", [("", "nessuna")] + [(k, f"senza {k}") for k in "BDCLUV"],
                               help="Solo per il cervello a sezioni: toglierne una per vedere a cosa serve.")
        self.p_good = Choice("Pile buone per", [("colore", "colore (rosse o gialle)"),
                                                ("grandezza", "grandezza (grandi o piccole; il colore non conta)")],
                             help="Che cosa distingue le pile buone: serve a provare che si impara qualunque cosa.")
        self.p_teacher = Choice("Maestro", [("", "no"),
                                            ("fantasma", "fantasma: nessuno lo vede, mangia la sua copia delle pile"),
                                            ("visibile", "visibile: un compagno vero (si mangia le pile buone)")],
                                help="Solo per il cervello a sezioni: il primo di ogni recinto, guidato a mano; gli "
                                     "altri imparano il valore delle cose (O) anche da ciò che succede a lui. M lo "
                                     "toglie o lo rimette.")
        self.p_encl = Number("Recinti", 200, step=50, low=1,
                             help="Recinti che girano insieme (GPU); se ne guarda uno alla volta.")
        self.p_beings = Number("Esseri per recinto", 6, step=1, low=1, help="Quanti esseri in ogni recinto.")
        # Il mondo 3D (docs/SPEC_CERVELLO_RAGIONANTE.md): per ora solo gli agenti di controllo.
        self.m_agent = Choice("Agente", [("ragionante", "cervello ragionante: per ora solo il motore (sezione 7)"),
                                         ("perfetto", "perfetto: legge la verità dal simulatore (soffitto)"),
                                         ("base", "base: vaga guardando solo la profondità (pavimento)")],
                              help="Il cervello nuovo, e i due controlli tra cui si misura. Non usa il cervello a sezioni.")
        self.m_body = Choice("Corpo", [(k, k) for k in BODIES],
                             help="Come le uscite del cervello muovono il corpo: il cervello non lo sa e deve tararsi.")
        self.m_encl = Number("Recinti", 1, step=1, low=1, help="Recinti nello stesso mondo; se ne guarda uno alla volta.")
        self.m_beings = Number("Esseri per recinto", 3, step=1, low=1, help="Quanti esseri in ogni recinto.")
        load = defaults.get("load")
        self.extra = [load] if load and Path(load).exists() and Path(load).parent.resolve() != RUNS.resolve() else []
        self.picker: SavePicker | None = None
        self._moved = False
        self.load = Picker("Riprendi da", self._open_picker,
                           help="Invio o clic: l'elenco dei salvataggi in runs/, per giorno, da cui continuare "
                                "o da esplorare.")
        self._show_load(None)
        self.save = Text("Salva in", defaults.get("save") or self._fresh_name(),
                         help="Dove salvare: tasto S, fine delle generazioni, uscita dalla corsa.")
        self._task_seen = None
        self.fields = [self.mode, self.sim, self.genome, self.net, self.agents, self.task, self.ladder, self.cap,
                       self.chance, self.seed,
                       self.gens, self.exam_every, self.select, self.load, self.save]
        self.pile_fields = [self.task, self.p_agent, self.p_phase, self.p_lesion, self.p_good, self.p_teacher, self.p_encl,
                            self.p_beings,
                            self.chance, self.seed, self.gens, self.load, self.save]
        self.world3d_fields = [self.task, self.m_agent, self.m_body, self.p_good, self.m_encl, self.m_beings, self.chance,
                               self.seed, self.gens]
        self.focus = 0
        self.start_btn = Button("Avvia la corsa  [Invio]", self.start, primary=True)
        self.inspect_btn = Button("Esplora reti e genomi del salvataggio", self.explore)
        self.quit_btn = Button("Esci", self.quit)
        if load:
            self._choose(load)

    def _visible(self) -> list:
        """Le righe da mostrare: l'arena delle pile ha le sue."""
        return {"pile": self.pile_fields, "mondo3d": self.world3d_fields}.get(self.task.value, self.fields)

    def _files(self) -> list[Path]:
        return sorted(RUNS.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True) + \
            [Path(p) for p in self.extra]

    def _open_picker(self) -> None:
        self.picker = SavePicker(self._files(), self.load.value)
        self._moved = True

    def _show_load(self, path) -> None:
        """La riga "Riprendi da" con il nome e quando è stato salvato."""
        self.load.value = path
        if path is None:
            self.load.shown = "nessuno: reti nuove"
        else:
            x = summarize(path)
            when = day_label(x.when.date(), date.today()).split(",")[0]
            self.load.shown = f"{x.name}   ({when.lower() if when in ('Oggi', 'Ieri') else when} {x.when:%H:%M})"

    def _choose(self, path) -> None:
        self._show_load(path)
        self._loaded(path)

    @staticmethod
    def _fresh_name() -> str:
        return str(RUNS / f"corsa-{datetime.now():%Y%m%d-%H%M%S}.npz")

    def _loaded(self, path) -> None:
        """Scelto un salvataggio (o nessuno). Le righe che vengono dal file si
        disattivano e mostrano i valori del file; tornando a "nessuno" riprendono
        quelli scelti a mano."""
        fields = (self.sim, self.genome, self.net, self.agents, self.task, self.ladder, self.chance, self.seed,
                  self.p_agent, self.p_phase, self.p_lesion, self.p_good, self.p_encl, self.p_beings)
        if path is not None and getattr(self, "_mine", None) is None:
            self._mine = [f.value if isinstance(f, Number) else f.index for f in fields]
        if path is None and getattr(self, "_mine", None) is not None:
            for f, v in zip(fields, self._mine):
                if isinstance(f, Number):
                    f.value = v
                else:
                    f.index = v
            self._mine = None
        if path is not None:
            self._show_saved(path)
        self.net.enabled = self.agents.enabled = self.chance.enabled = self.task.enabled = path is None
        self.sim.enabled = self.genome.enabled = path is None
        for f in (self.p_agent, self.p_phase, self.p_lesion, self.p_good, self.p_teacher, self.p_encl, self.p_beings):
            f.enabled = path is None
        if path is not None:
            self.save.value, self.save.edited = path, False
        elif not self.save.edited:
            self.save.value = self._fresh_name()

    def _show_saved(self, path) -> None:
        """Mette nelle righe disattivate i valori del salvataggio."""
        meta = pile_meta(path)
        if meta is not None:
            self.task.select("pile")
            self.p_agent.select(meta["agent"])
            self.p_phase.select(meta["phase"])
            self.p_lesion.select(meta["lesions"][0] if meta.get("lesions") else "")
            self.p_encl.value, self.p_beings.value = meta["enclosures"], meta["beings"]
            self.p_good.select(meta.get("good_by", "colore"))
            self.chance.select("fixed")
            self.seed.value = meta["seed"]
            return
        try:
            with np.load(path) as d:
                saved = json.loads(str(d["config"])) if "config" in d else {}
                net = str(d["net_type"]) if "net_type" in d else saved.get("net_type", "rate")
                n = len(d["weights"])
                seed = int(d["seed"]) if "seed" in d else None
        except Exception:  # file illeggibile: lo dirà save_info
            return
        for field, value in ((self.net, net), (self.task, saved.get("task") or "bersaglio"),
                             (self.ladder, saved.get("ladder")), (self.sim, saved.get("sim_mode") or "evoluzione"),
                             (self.genome, saved.get("genome") or "diretto")):
            if value in [v for v, _ in field.options]:
                field.select(value)
        self.agents.value = n
        if saved.get("task") == "arena":
            self.cap.value = n
            with np.load(path) as d:
                self.agents.value = int(d["alive"].sum()) if "alive" in d else n
        if seed is not None:
            self.chance.select("fixed")
            self.seed.value = seed

    def _config(self) -> Config:
        cfg = Config(net_type=self.net.value, task=self.task.value, sim_mode=self.sim.value,
                     genome=self.genome.value)
        cfg.population = self.agents.fixed()
        if cfg.task == "cibo":
            cfg.n_hazards = 0  # il veleno resta spento in tutta la scala
            cfg.ladder = self.ladder.value
        if cfg.task == "arena":
            cfg.n_hazards = 0
            cfg.arena_cap = max(self.cap.fixed(), cfg.population)
        return cfg

    def _is_arena(self) -> bool:
        """La corsa da lanciare è nell'arena continua (nuova o ripresa)."""
        if self.load.value:
            try:
                with np.load(self.load.value) as d:
                    return "kind" in d and str(d["kind"]) == "arena"
            except Exception:
                return False
        return self.task.value == "arena"

    def _ecology(self):
        from ..arenas.continuous.ecology import Ecology

        cfg = self._config()
        try:
            cfg.exam_every = self.exam_every.fixed()
            if self.load.value:
                eco = Ecology.load(cfg, self.load.value)
                cfg.exam_every = self.exam_every.fixed()
                return eco
            if self.chance.value == "random":
                self.seed.value = secrets.randbelow(2**31)
            return Ecology(cfg, self.seed.fixed(), initial=self.agents.fixed())
        except Exception as err:
            self.message, self.error = f"non riesco ad aprire l'arena: {err}", True
            return None

    def _evolution(self) -> Evolution | None:
        cfg = self._config()
        try:
            cfg.exam_every = self.exam_every.fixed()
            if self.load.value:
                evo = Evolution.load(cfg, self.load.value)
                cfg.exam_every = self.exam_every.fixed()  # scelta dal menu: se diversa, va nel registro
                return evo
            if self.chance.value == "random":
                self.seed.value = secrets.randbelow(2**31)  # resta visibile nel menu
            return Evolution(cfg, self.seed.fixed())
        except Exception as err:
            self.message, self.error = f"non riesco ad aprire il salvataggio: {err}", True
            return None

    def _pile(self, explore: bool = False) -> None:
        from ..arenas.pile import saves as pile_saves
        from ..arenas.pile.run import make_run
        from ..arenas.pile.world import PileConfig
        from .pile.view import PileScreen

        save = self.save.value.strip() or self._fresh_name()
        if self.load.value:  # riprendere un salvataggio, o provarne le reti una a una
            try:
                run = pile_saves.alone(self.load.value) if explore else pile_saves.load(self.load.value)
            except Exception as err:
                self.message, self.error = f"non riesco ad aprire il salvataggio: {err}", True
                return
            screen = PileScreen(self.app, run, 0 if explore else self.gens.fixed(), save=None if explore else save)
            self.app.active_run = screen
            self.app.show(screen)
            return
        if explore:
            self.message, self.error = "scegli prima un salvataggio da provare", False
            return
        if self.chance.value == "random":
            self.seed.value = secrets.randbelow(2**31)
        cfg = PileConfig(beings=self.p_beings.fixed(), good_by=self.p_good.value)
        lesions = (self.p_lesion.value,) if self.p_lesion.value and self.p_agent.value == "sezioni" else ()
        try:
            run = make_run(self.p_agent.value, self.p_phase.value, self.p_encl.fixed(), self.seed.fixed(), lesions,
                           cfg, teacher=self.p_teacher.value if self.p_agent.value == "sezioni" else "")
        except Exception as err:
            self.message, self.error = f"non riesco ad avviare l'arena delle pile: {err}", True
            return
        screen = PileScreen(self.app, run, self.gens.fixed(), save=save)
        self.app.active_run = screen
        self.app.show(screen)

    def _world3d(self) -> None:
        from ..arenas.mondo3d.run import Run3D
        from ..arenas.mondo3d.world import Mondo3DConfig
        from .mondo3d.view import Mondo3DScreen

        if self.chance.value == "random":
            self.seed.value = secrets.randbelow(2**31)
        cfg = Mondo3DConfig(beings=self.m_beings.fixed(), good_by=self.p_good.value, body=self.m_body.value)
        try:
            run = Run3D(cfg, self.m_agent.value, self.m_encl.fixed(), self.seed.fixed())
        except Exception as err:
            self.message, self.error = f"non riesco ad avviare il mondo 3D: {err}", True
            return
        self.app.show(Mondo3DScreen(self.app, run, self.gens.fixed()))

    def start(self) -> None:
        if self.task.value == "pile":
            self._pile()
            return
        if self.task.value == "mondo3d":
            self._world3d()
            return
        if self._is_arena():
            from .continuous.view import ArenaScreen

            eco = self._ecology()
            if eco is not None:
                run = ArenaScreen(self.app, eco, self.save.value.strip() or self._fresh_name(), self.gens.fixed())
                self.app.active_run = run
                self.app.show(run)
            return
        evo = self._evolution()
        if evo is not None:
            from .generational.run import RunScreen

            save = self.save.value.strip() or self._fresh_name()
            run = RunScreen(self.app, evo, save, self.gens.fixed(), watch=self.mode.value == "live",
                            select=self.select.fixed() if self.gens.value else 0, full=self.mode.value == "max")
            self.app.active_run = run
            self.app.show(run)

    def explore(self) -> None:
        from .inspector import Inspector

        if self.task.value == "pile":
            self._pile(explore=True)  # le reti del salvataggio, ognuna da sola nel suo recinto
            return
        if self._is_arena():
            self.message, self.error = ("l'esploratore delle reti non c'è ancora per l'arena continua: "
                                        "avviala e segui un essere"), False
            return
        evo = self._evolution()
        if evo is not None:
            self.app.show(Inspector(self.app, evo, back=self, title=Path(self.load.value).name))

    def quit(self) -> None:
        self.app.running = False

    def returned(self, message: str) -> None:
        """Si torna al menu da una corsa: i salvataggi possono essere cambiati."""
        self.message, self.error = message, False
        current = self.load.value
        self._choose(current if current and Path(current).exists() else None)
        self.app.show(self)

    def handle(self, ev) -> None:
        if self.picker is not None:
            result = self.picker.handle(ev)
            self._moved = ev.type == pygame.KEYDOWN
            if result is not None:
                if result[0] == "scelto":
                    self._choose(result[1])
                self.picker = None
            return
        fields = self._visible()
        self.focus = min(self.focus, len(fields) - 1)
        field = fields[self.focus]
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for k, f in enumerate(fields):
                if f.rect.collidepoint(ev.pos):
                    self.focus = k
                    f.click(ev.pos)
            for b in (self.start_btn, self.inspect_btn, self.quit_btn):
                b.click(ev.pos)
        elif ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if field is self.load:
                    self._open_picker()  # Invio su "Riprendi da" apre l'elenco
                else:
                    self.start()
            elif ev.key == pygame.K_ESCAPE:
                self.quit()
            elif ev.key in (pygame.K_DOWN, pygame.K_TAB):
                self.focus = (self.focus + 1) % len(fields)
            elif ev.key == pygame.K_UP:
                self.focus = (self.focus - 1) % len(fields)
            elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT) and not isinstance(field, Text):
                field.change(1 if ev.key == pygame.K_RIGHT else -1)
            else:
                field.type(ev)

    def draw(self, s) -> None:
        app = self.app
        arena = self.task.value == "arena"
        if arena != ((self._task_seen or "bersaglio") == "arena"):
            # minuti invece di generazioni: l'esame ogni 10 minuti di mondo, non ogni 50
            if arena and self.exam_every.value == Config().exam_every:
                self.exam_every.value = 10
            elif not arena and self.exam_every.value == 10:
                self.exam_every.value = Config().exam_every
        self._task_seen = self.task.value
        self.gens.label = "Minuti di mondo" if arena else "Generazioni"
        self.exam_every.label = "Esame ogni (min)" if arena else "Esame ogni"
        self.cap.enabled = self.task.enabled and arena
        self.select.enabled = self.gens.value > 0 and not arena  # serve un punto in cui le generazioni finiscono
        self.seed.enabled = self.chance.enabled and self.chance.value == "fixed"
        self.ladder.enabled = self.task.enabled and self.task.value == "cibo"
        s.fill(BG)
        s.blit(app.big.render("Evolutionary nets", True, PLAYER), (60, 40))
        s.blit(app.font.render("nuova corsa, ripresa di un salvataggio o esplorazione delle sue reti", True, DIM),
               (60, 76))
        y = 100
        fields = self._visible()
        self.focus = min(self.focus, len(fields) - 1)
        pile = fields is not self.fields
        self.gens.label = "Minuti di mondo" if pile else self.gens.label
        self.p_lesion.enabled = self.p_agent.value == "sezioni"
        self.m_body.enabled = self.m_agent.value == "ragionante"
        for k, f in enumerate(fields):
            f.draw(s, app.font, 60, y, 175, 540, k == self.focus)
            y += ROW_H  # righe fitte: con il tetto dell'arena sono 15
        field = fields[self.focus]
        help = field.help if field.enabled or field is not self.select else \
            "La selezione finale parte quando le generazioni finiscono: imposta prima Generazioni."
        s.blit(app.small.render(help, True, DIM), (235, y + 2))
        if self.load.value:
            s.blit(app.small.render(save_info(self.load.value), True, TARGET), (235, y + 20))
        y += 44
        x = 60
        self.inspect_btn.enabled = self.load.value is not None and self.task.value != "mondo3d"
        for b in (self.start_btn, self.inspect_btn, self.quit_btn):
            x += b.draw(s, app.font, x, y) + 14
        if self.message:
            s.blit(app.font.render(self.message[:150], True, (255, 110, 90) if self.error else TARGET), (60, y + 50))

        help_x = 830
        lines = [
            ("Come si usa", TEXT),
            ("", TEXT),
            ("Su/Giù o Tab   scegli la riga", DIM),
            ("Sinistra/Destra o clic   cambia il valore", DIM),
            ("numeri e percorso si scrivono da tastiera", DIM),
            ("Invio   avvia      Esc   esci", DIM),
            ("Invio su Riprendi da   elenco dei salvataggi", DIM),
            ("", TEXT),
            ("Durante la corsa", TEXT),
            ("", TEXT),
            ("La pagina principale ha due schede: la", DIM),
            ("panoramica (a che punto siamo, a parole) e", DIM),
            ("le reti dei vincitori.  V   cambia scheda", DIM),
            ("F   apre/chiude l'arena (la corsa continua)", DIM),
            ("T   prova da sola la rete scelta", DIM),
            ("I   reti e genomi a tutto schermo", DIM),
            ("E   esame    L   esame con una lesione", DIM),
            ("Con Generazioni e Selezione finale, alla fine", DIM),
            ("i sopravvissuti rifanno la prova senza figli.", DIM),
            ("S   salva     Spazio   pausa", DIM),
            ("Esc torna qui (salva se ci sono novità)", DIM),
            ("", TEXT),
            ("Le tracce delle reti (un campione per", DIM),
            ("generazione) finiscono nel salvataggio:", DIM),
            ("si possono esplorare anche dopo.", DIM),
        ]
        for k, (text, color) in enumerate(lines):
            s.blit(app.font.render(text, True, color), (help_x, 110 + 20 * k))
        if self.picker is not None:
            self.picker.draw(s, app, self._moved)
            self._moved = False
