"""L'interfaccia grafica: una finestra sola con più schermate.

- `Launcher`: il menu da cui si sceglie che corsa lanciare.
- `RunScreen`: la corsa. La pagina principale mostra le statistiche e, accanto,
  le reti dei vincitori delle generazioni già fatte, da esplorare mentre
  l'evoluzione va avanti; con F si apre l'arena.
- `Inspector` (inspector.py): reti e genomi a tutto schermo.
- `TestScreen` (testbench.py): una sola rete messa alla prova nell'arena.

Una corsa aperta continua in sottofondo anche mentre si guardano le altre
schermate (B la mette in pausa).
"""

import json
import secrets
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame

from ..config import TUNABLES, Config
from ..evo import Evolution
from ..evo import ladder
from ..evo.evolution import describe_change
from ..evo.exam import LESIONS
from ..net.anatomy import KIND_NAMES
from .render import BG, BORDER, DIM, MODULATOR, PLAYER, TARGET, TEXT, Renderer, success_series
from .overview import Overview
from .widgets import ROW_H, Button, Choice, Number, Text

SPEEDS = (1, 2, 4, 8, 16)
TURBO_BUDGET = 0.08  # s di calcolo per fotogramma guardando l'arena in turbo
PAGE_BUDGET = 0.05  # s di calcolo per fotogramma con la pagina principale (resta reattiva)
MAX_BUDGET = 0.5  # s di calcolo per fotogramma alla massima velocità (si ridisegna 2 volte al secondo)
CHUNK = 60  # passi per chiamata quando si calcola a blocchi (un secondo simulato)
BACKGROUND_BUDGET = 0.025  # s per fotogramma alla corsa mentre si guarda un'altra schermata
RUNS = Path("runs")


class App:
    def __init__(self, defaults: dict):
        pygame.init()
        pygame.key.set_repeat(350, 40)
        self.screen = pygame.display.set_mode(Renderer.window_size(Config()))
        pygame.display.set_caption("Evolutionary nets")
        self.font = pygame.font.SysFont("monospace", 15)
        self.small = pygame.font.SysFont("monospace", 12)
        self.big = pygame.font.SysFont("monospace", 24, bold=True)
        self.clock = pygame.time.Clock()
        self.launcher = Launcher(self, defaults)
        self.current = self.launcher
        self.active_run: RunScreen | None = None  # la corsa aperta, se c'è
        self.running = True

    def show(self, screen) -> None:
        self.current = screen

    def test(self, selection, back) -> None:
        """Apre il banco di prova per una rete: (genoma, nome, descrizione, etichetta)."""
        from .testbench import TestScreen

        if selection is not None:
            genome, _, title, label = selection
            cfg = self.active_run.cfg if self.active_run else back.evo.cfg
            self.show(TestScreen(self, cfg, genome, [label], title, back))

    def test_survivors(self, evo, back) -> str:
        """Apre il banco di prova con tutti i sopravvissuti insieme. Restituisce
        un messaggio se non ce ne sono."""
        from .testbench import TestScreen

        idx = np.flatnonzero(evo.age > 0)
        if not len(idx):
            return "nessun sopravvissuto da provare: l'ultima generazione non ha vinto nessuno"
        if evo.selection is not None and not evo.selecting:
            what = ("l'unico rimasto" if len(idx) == 1 else f"i {len(idx)} rimasti") + " dalla selezione finale"
        else:
            what = f"i {len(idx)} sopravvissuti (vincitori della gen. {evo.generation - 1})"
        labels = [f"#{i}" for i in evo.ids[idx]]
        self.show(TestScreen(self, evo.cfg, evo.ctrl.brains.genome.take(idx), labels, what, back))
        return ""

    def run_status(self) -> str:
        """Riga sullo stato della corsa in sottofondo, per le altre schermate."""
        r = self.active_run
        if r is None or self.current is r:
            return ""
        state = "in pausa" if r.paused else "in corso"
        return f"corsa in sottofondo {state}: generazione {r.evo.generation} (B pausa/riprende)"

    def run(self) -> None:
        while self.running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.current.close()
                    self.running = False
                elif (ev.type == pygame.KEYDOWN and ev.key == pygame.K_b and self.active_run
                      and self.current not in (self.active_run, self.launcher)):
                    self.active_run.paused = not self.active_run.paused
                else:
                    self.current.handle(ev)
            if not self.running:
                break
            if self.active_run is not None and self.current is not self.active_run:
                self.active_run.background(BACKGROUND_BUDGET)
            self.current.update()
            self.current.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()


class Screen:
    def handle(self, ev) -> None: ...
    def update(self) -> None: ...
    def draw(self, surface) -> None: ...
    def close(self) -> None:
        """La finestra si sta chiudendo."""


def save_info(path) -> str:
    """Una riga che descrive un salvataggio, senza caricarlo per intero."""
    try:
        with np.load(path) as d:
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


class Launcher(Screen):
    """Il menu: si sceglie il tipo di corsa e la si lancia, oppure si apre un
    salvataggio per esplorarne le reti."""

    def __init__(self, app: App, defaults: dict):
        self.app = app
        self.message, self.error = "", False
        files = sorted(RUNS.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
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
        self.task = Choice("Mondo", [("bersaglio", "bersaglio e veleno (il mondo di prima)"),
                                     ("cibo", "nutrirsi: cibo, prede e predatori")],
                           help="Il compito. Nel mondo del cibo si riproduce chi mangia almeno N cibi; il veleno "
                                "è spento. Ignorato se si riprende un salvataggio.")
        self.task.select(defaults.get("task") or "bersaglio")
        self.ladder = Choice("Scala", [("scala", "adattiva: sale se va bene, scende se va male"),
                                       ("basico", "ferma al mondo basico (controllo)"),
                                       ("complesso", "ferma al mondo complesso (controllo)")],
                             help="Solo nel mondo del cibo: come si muovono le manopole della difficoltà "
                                  "(docs/SPEC_MONDO.md).")
        self.ladder.select(defaults.get("ladder") or "scala")
        self.gens = Number("Generazioni", 0, step=50,
                           help="Dopo quante generazioni fermarsi e salvare. 0 = senza limite.")
        self.exam_every = Number("Esame ogni", Config().exam_every, step=50,
                                 help="Ogni quante generazioni mettere all'esame fisso (7 prove; 8 livelli nel "
                                      "mondo del cibo) i 20 più anziani; non cambia la corsa. 0 = mai.")
        self.select = Number("Selezione finale", 0, step=1,
                             help="Finite le generazioni, i sopravvissuti rifanno la prova questo numero di "
                                  "volte, senza figli: chi fallisce è eliminato. 0 = niente.")
        load = defaults.get("load")
        options = [(None, "nessuno: reti nuove")] + [(str(p), p.name) for p in files]
        if load and load not in [v for v, _ in options]:
            options.insert(1, (load, Path(load).name))
        self.load = Choice("Riprendi da", options, help="Un salvataggio in runs/ da cui continuare o da esplorare.",
                           on_change=self._loaded)
        self.save = Text("Salva in", defaults.get("save") or self._fresh_name(),
                         help="Dove salvare: tasto S, fine delle generazioni, uscita dalla corsa.")
        self.fields = [self.mode, self.net, self.agents, self.task, self.ladder, self.chance, self.seed, self.gens,
                       self.exam_every, self.select, self.load, self.save]
        self.focus = 0
        self.start_btn = Button("Avvia la corsa  [Invio]", self.start, primary=True)
        self.inspect_btn = Button("Esplora reti e genomi del salvataggio", self.explore)
        self.quit_btn = Button("Esci", self.quit)
        if load:
            self.load.select(load)
            self._loaded(load)

    @staticmethod
    def _fresh_name() -> str:
        return str(RUNS / f"corsa-{datetime.now():%Y%m%d-%H%M%S}.npz")

    def _loaded(self, path) -> None:
        """Scelto un salvataggio (o nessuno). Le righe che vengono dal file si
        disattivano e mostrano i valori del file; tornando a "nessuno" riprendono
        quelli scelti a mano."""
        fields = (self.net, self.agents, self.task, self.ladder, self.chance, self.seed)
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
        if path is not None:
            self.save.value, self.save.edited = path, False
        elif not self.save.edited:
            self.save.value = self._fresh_name()

    def _show_saved(self, path) -> None:
        """Mette nelle righe disattivate i valori del salvataggio."""
        try:
            with np.load(path) as d:
                saved = json.loads(str(d["config"])) if "config" in d else {}
                net = str(d["net_type"]) if "net_type" in d else saved.get("net_type", "rate")
                n = len(d["weights"])
                seed = int(d["seed"]) if "seed" in d else None
        except Exception:  # file illeggibile: lo dirà save_info
            return
        for field, value in ((self.net, net), (self.task, saved.get("task") or "bersaglio"),
                             (self.ladder, saved.get("ladder"))):
            if value in [v for v, _ in field.options]:
                field.select(value)
        self.agents.value = n
        if seed is not None:
            self.chance.select("fixed")
            self.seed.value = seed

    def _config(self) -> Config:
        cfg = Config(net_type=self.net.value, task=self.task.value)
        cfg.population = self.agents.fixed()
        if cfg.task == "cibo":
            cfg.n_hazards = 0  # il veleno resta spento in tutta la scala
            cfg.ladder = self.ladder.value
        return cfg

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

    def start(self) -> None:
        evo = self._evolution()
        if evo is not None:
            save = self.save.value.strip() or self._fresh_name()
            run = RunScreen(self.app, evo, save, self.gens.fixed(), watch=self.mode.value == "live",
                            select=self.select.fixed() if self.gens.value else 0, full=self.mode.value == "max")
            self.app.active_run = run
            self.app.show(run)

    def explore(self) -> None:
        from .inspector import Inspector

        evo = self._evolution()
        if evo is not None:
            self.app.show(Inspector(self.app, evo, back=self, title=Path(self.load.value).name))

    def quit(self) -> None:
        self.app.running = False

    def returned(self, message: str) -> None:
        """Si torna al menu da una corsa: i salvataggi possono essere cambiati."""
        self.message, self.error = message, False
        files = sorted(RUNS.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
        current = self.load.value
        self.load.options = [(None, "nessuno: reti nuove")] + [(str(p), p.name) for p in files]
        self.load.index = 0
        if current in [v for v, _ in self.load.options]:
            self.load.select(current)
        self._loaded(self.load.value)
        self.app.show(self)

    def handle(self, ev) -> None:
        field = self.fields[self.focus]
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for k, f in enumerate(self.fields):
                if f.rect.collidepoint(ev.pos):
                    self.focus = k
                    f.click(ev.pos)
            for b in (self.start_btn, self.inspect_btn, self.quit_btn):
                b.click(ev.pos)
        elif ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.start()
            elif ev.key == pygame.K_ESCAPE:
                self.quit()
            elif ev.key in (pygame.K_DOWN, pygame.K_TAB):
                self.focus = (self.focus + 1) % len(self.fields)
            elif ev.key == pygame.K_UP:
                self.focus = (self.focus - 1) % len(self.fields)
            elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT) and not isinstance(field, Text):
                field.change(1 if ev.key == pygame.K_RIGHT else -1)
            else:
                field.type(ev)

    def draw(self, s) -> None:
        app = self.app
        self.select.enabled = self.gens.value > 0  # serve un punto in cui le generazioni finiscono
        self.seed.enabled = self.chance.enabled and self.chance.value == "fixed"
        self.ladder.enabled = self.task.enabled and self.task.value == "cibo"
        s.fill(BG)
        s.blit(app.big.render("Evolutionary nets", True, PLAYER), (60, 40))
        s.blit(app.font.render("nuova corsa, ripresa di un salvataggio o esplorazione delle sue reti", True, DIM),
               (60, 76))
        y = 110
        for k, f in enumerate(self.fields):
            f.draw(s, app.font, 60, y, 175, 540, k == self.focus)
            y += ROW_H + 4
        field = self.fields[self.focus]
        help = field.help if field.enabled or field is not self.select else \
            "La selezione finale parte quando le generazioni finiscono: imposta prima Generazioni."
        s.blit(app.small.render(help, True, DIM), (235, y + 2))
        if self.load.value:
            s.blit(app.small.render(save_info(self.load.value), True, TARGET), (235, y + 20))
        y += 56
        x = 60
        self.inspect_btn.enabled = self.load.value is not None
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


class RunScreen(Screen):
    """Una corsa. Pagina principale: statistiche a sinistra, reti dei vincitori
    a destra, esplorabili mentre l'evoluzione continua. F apre l'arena."""

    def __init__(self, app: App, evo: Evolution, save: str, limit: int = 0, watch: bool = False,
                 select: int = 0, full: bool = False):
        from .inspector import CHAMPIONS, Explorer

        self.app, self.evo, self.cfg, self.save_path = app, evo, evo.cfg, save
        self.world = evo.world
        self.renderer = Renderer(evo.world, app.screen)
        self.renderer.name_of = lambda i: f"#{self.evo.ids[i]}"
        self.explorer = Explorer(evo, CHAMPIONS)
        self.watch = watch
        self.full = full  # massima velocità: niente esploratore, si ridisegna di rado
        self.goal = evo.generation + limit if limit else None
        self.select = select  # prove della selezione finale, a generazioni finite
        self.focus = int(np.argmax(evo.age))
        self.manual = self.paused = self.show_consts = False
        self.turbo = not watch  # aprendo l'arena da una corsa veloce si resta veloci
        self.show_rays = True
        self.speed_i, self.tun_i = 0, 0
        self.saved_gen = evo.generation
        self.message = ""
        self.oldest_rects: list[pygame.Rect] = []  # dove cliccare per vedere la rete del più anziano
        self.start_gen = evo.generation
        self.summary: list[tuple[str, tuple]] | None = None  # riepilogo di fine esperimento
        self.summary_open = False
        self.buttons = [Button("Guarda i sopravvissuti in arena  [A]", self._test_survivors, primary=True),
                        Button("Esplora le loro reti  [2]", self._explore_survivors),
                        Button("Continua l'evoluzione  [Spazio]", self._resume),
                        Button("Chiudi  [Invio]", self._close_summary)]
        self.gen0, self.busy = evo.generation, 0.0
        self.saved_exams = len(evo.exams)
        self.lesion_menu = False  # L: scelta della parte da togliere per l'esame
        self.overview = Overview(app)
        self.tab = "overview"  # scheda della pagina principale: "overview" o "nets"
        self.tab_rects: dict[str, pygame.Rect] = {}
        self.show_help = False  # H nell'arena: elenco dei tasti

    # --- evoluzione --------------------------------------------------------

    def _save(self) -> None:
        self.evo.save(self.save_path)
        self.saved_gen, self.saved_exams = self.evo.generation, len(self.evo.exams)
        self.message = f"salvata la generazione {self.saved_gen} in {self.save_path}"

    def _selected_path(self) -> str:
        p = Path(self.save_path)
        return str(p.with_name(p.stem + "-selezionati" + p.suffix))

    def _advance(self, hand=None, k: int = 1) -> bool:
        """k passi (meno se la generazione finisce); False quando bisogna
        fermarsi (fine delle generazioni o della selezione finale)."""
        evo = self.evo
        if not evo.finished:
            evo.advance(k, (self.focus, hand) if self.manual and hand is not None else None)
            return True
        if evo.selecting:
            evo.next_selection_round()
            self.focus = 0
            if evo.selecting:
                return True
            evo.save(self._selected_path())
            self._finish()
            return False
        evo.next_generation()
        if evo.exam_due():
            evo.start_exam()
        self.focus = int(np.argmax(evo.age))  # il sopravvissuto più anziano, se c'è
        if self.goal is not None and evo.generation >= self.goal:
            self.goal = None
            self._save()
            if self.select:
                n = evo.start_selection(self.select)
                self.select = 0
                if n:
                    self.focus = 0
                    self.message = f"generazioni finite e salvate; selezione finale: {n} sopravvissuti in gara"
                    return True
            self._finish()
            return False
        return True

    # --- fine dell'esperimento -----------------------------------------------

    def _finish(self) -> None:
        """Pausa e riepilogo: quanti sono rimasti e cosa si può fare ora."""
        evo, cfg, hist = self.evo, self.cfg, self.evo.history
        self.paused = True
        self.explorer.switch("survivors")
        self.explorer.snapshot()
        n_surv = int((evo.age > 0).sum())
        lines = [("ESPERIMENTO CONCLUSO", PLAYER), ("", TEXT)]
        lines.append((f"generazioni fatte: {evo.generation - self.start_gen} (dalla {self.start_gen} "
                      f"alla {evo.generation - 1}), seme {evo.seed}", TEXT))
        if hist:
            pct, avg = success_series(hist, cfg.population)
            best = int(np.argmax(pct))
            lines.append((f"ultima generazione: {hist[-1]['completed']}/{cfg.population} hanno completato "
                          f"({pct[-1]:.0f}%); media ultime 10: {avg[-1]:.0f}%; record {pct[best]:.0f}% "
                          f"(gen. {hist[best]['generation']})", TEXT))
        intact = [e for e in evo.exams if not e.get("lesion")]
        if intact:
            r = intact[-1]
            lines.append((f"ultimo esame fisso (gen. {r['generation']}, {r['who']} più anziani): media "
                          f"{r['score']:.0%}, migliore {r['best']:.0%}; "
                          + ", ".join(f"{k} {v:.0%}" for k, v in r["scores"].items()), TEXT))
        if evo.exam is not None:
            lines.append((f"esame della gen. {evo.exam.generation} in corso ({evo.exam.progress:.0%}): "
                          "continua anche in pausa", DIM))
        lines.append(("", TEXT))
        sel = evo.selection
        if sel is not None and not evo.selecting:
            steps = "; ".join(f"prova {e['round']}: {e['passed']} su {e['entered']}" for e in sel["log"])
            lines.append((f"SOPRAVVISSUTI: {n_surv}, rimasti dalla selezione finale", PLAYER))
            lines.append((f"  partiti {sel['entered']} (i vincitori dell'ultima generazione), {sel['rounds']} prove "
                          "senza figli, chi fallisce è eliminato", TEXT))
            lines.append((f"  superano: {steps}", TEXT))
            if sel["note"]:
                lines.append(("  " + sel["note"], DIM))
            lines.append((f"  salvati a parte in {self._selected_path()}", DIM))
        else:
            lines.append((f"SOPRAVVISSUTI: {n_surv}, i vincitori dell'ultima generazione", PLAYER))
            lines.append((f"  gli altri {cfg.population - n_surv} della popolazione sono figli appena nati, "
                          "non ancora messi alla prova", TEXT))
        if n_surv:
            ids = sorted(int(i) for i in evo.ids[evo.age > 0])
            shown = ", ".join(f"#{i}" for i in ids[:16]) + (" ..." if len(ids) > 16 else "")
            lines.append((f"  numeri: {shown}", DIM))
            lines.append(("  " + self._oldest_text(), DIM))
        lines.append(("", TEXT))
        lines.append((f"popolazione salvata in {self.save_path}", DIM))
        self.summary, self.summary_open = lines, True
        self.message = "esperimento concluso: Invio riapre il riepilogo, A guarda i sopravvissuti in arena"

    def _start_exam(self, lesion: str | None = None) -> None:
        if self.evo.exam is not None:
            self.message = "c'è già un esame in corso"
            return
        self.evo.start_exam(lesion=lesion)
        what = f", senza {lesion}," if lesion else ""
        self.message = (f"esame dei {len(self.evo.exam.genomes)} più anziani della gen. {self.evo.generation}{what} "
                        "avviato: va avanti anche in pausa")

    def _test_survivors(self) -> None:
        msg = self.app.test_survivors(self.evo, back=self)
        if msg:
            self.message = msg

    def _explore_survivors(self) -> None:
        self.summary_open = self.full = False
        self.tab = "nets"
        self.explorer.switch("survivors")
        self.explorer.snapshot()

    def _resume(self) -> None:
        self.summary_open = self.paused = False

    def _close_summary(self) -> None:
        self.summary_open = False

    def _compute(self, budget: float, hand=None) -> None:
        start = time.perf_counter()
        deadline = start + budget
        # A blocchi: con il motore GPU ogni chiamata ha un costo fisso. Guidando
        # a mano si va un passo alla volta.
        chunk = 1 if self.manual else CHUNK
        while time.perf_counter() < deadline and self._advance(hand, chunk):
            pass
        self.busy += time.perf_counter() - start

    def _examine(self, budget: float) -> float:
        """Porta avanti l'esame in corso, se c'è, con metà del tempo; il resto
        va all'evoluzione. L'esame va avanti anche in pausa: non è evoluzione.
        Restituisce il tempo rimasto."""
        evo = self.evo
        if evo.exam is None:
            return budget
        start = time.perf_counter()
        evo.exam.run(budget / 2)
        r = evo.collect_exam()
        if r is not None:
            what = f" senza {r['lesion']}" if r.get("lesion") else ""
            self.message = (f"esame della gen. {r['generation']}{what} concluso: media {r['score']:.0%}, "
                            f"migliore {r['best']:.0%} (in {r['seconds']:.0f}s)")
        return max(budget - (time.perf_counter() - start), 0.0)

    def background(self, budget: float) -> None:
        """Avanza mentre si guarda un'altra schermata."""
        budget = self._examine(budget)
        if not self.paused:
            self._compute(budget)

    def update(self) -> None:
        self.explorer.refresh()
        if self.paused:
            self._examine(PAGE_BUDGET)
            return
        if not self.watch:
            self._compute(self._examine(MAX_BUDGET if self.full else PAGE_BUDGET))
            return
        keys = pygame.key.get_pressed()
        hand = np.array(
            [
                (keys[pygame.K_UP] or keys[pygame.K_w]) - (keys[pygame.K_DOWN] or keys[pygame.K_s]),
                (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a]),
            ],
            dtype=float,
        )
        if self.turbo:
            self._compute(self._examine(TURBO_BUDGET), hand)
        else:
            self._examine(PAGE_BUDGET)
            start = time.perf_counter()
            for _ in range(SPEEDS[self.speed_i]):
                if not self._advance(hand):
                    break
            self.busy += time.perf_counter() - start

    def _leave(self) -> str:
        self.app.active_run = None
        if self.evo.exam is not None:  # esame non finito: si perde
            self.evo.exam.close()
            self.evo.exam = None
        if self.evo.generation != self.saved_gen or len(self.evo.exams) != self.saved_exams:
            self._save()
            return self.message
        return ""

    def close(self) -> None:
        self._leave()

    # --- comandi -----------------------------------------------------------

    def handle(self, ev) -> None:
        if self.lesion_menu and ev.type == pygame.KEYDOWN:
            self.lesion_menu = False
            k = ev.key - pygame.K_1
            if 0 <= k < len(LESIONS):
                self._start_exam(LESIONS[k])
            return
        if self.summary_open and not self.watch:
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for b in self.buttons:
                    if b.click(ev.pos):
                        return
                return
            if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._close_summary()
                return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_2:
                self._explore_survivors()
                return
        if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.summary:
            self.summary_open = True
            return
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_a and not self.watch:
            self._test_survivors()
            return
        if ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_ESCAPE:
                self.app.launcher.returned(self._leave())
                return
            if k == pygame.K_SPACE:
                self.paused = not self.paused
                self.summary_open = False
                return
            if k == pygame.K_f:
                self.watch = not self.watch
                return
            if k == pygame.K_i:
                from .inspector import Inspector

                self.app.show(Inspector(self.app, self.evo, back=self, title=self.save_path,
                                        source=self.explorer.source))
                return
            if k == pygame.K_s and not self.manual:
                self._save()
                return
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_o:
            self._show_oldest()
            return
        if self.watch:
            self._handle_arena(ev)
        elif (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
              and any(r.collidepoint(ev.pos) for r in self.oldest_rects)):
            self._show_oldest()
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and any(
                r.collidepoint(ev.pos) for r in self.tab_rects.values()):
            self.tab = next(k for k, r in self.tab_rects.items() if r.collidepoint(ev.pos))
            self.full = False
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_v:
            self.tab = "nets" if self.tab == "overview" else "overview"
            self.full = False
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_t:
            self.app.test(self.explorer.selected(), back=self)
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_e:
            self._start_exam()
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_l:
            if self.evo.exam is not None:
                self.message = "c'è già un esame in corso"
            else:
                self.lesion_menu = True
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_z:
            self.full = not self.full
        elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_1, pygame.K_2, pygame.K_3) and not self.full:
            self.tab = "nets"  # gli elenchi delle reti stanno nell'altra scheda
            self.explorer.handle(ev)
        elif not self.full and self.tab == "nets":
            self.explorer.handle(ev)

    def _show_oldest(self) -> None:
        """La rete del più anziano: nell'arena lo si segue (il pannello a destra
        mostra la sua rete), nella pagina principale lo si apre nell'esploratore."""
        o = self.evo.oldest()
        if o is None:
            self.message = "nessun sopravvissuto: sono tutti appena nati"
            return
        if self.watch:
            self.focus = o["index"]
        else:
            self.full, self.tab = False, "nets"
            self.explorer.show_individual(o["index"])

    def _handle_arena(self, ev) -> None:
        cfg = self.cfg
        if ev.type == pygame.MOUSEBUTTONDOWN and self.renderer.arena.collidepoint(ev.pos):
            click = np.array(ev.pos) / self.renderer.scale
            self.focus = int(np.argmin(np.linalg.norm(self.world.pos - click, axis=1)))
        elif ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_t:
                self.turbo = not self.turbo
            elif k == pygame.K_v:
                self.show_rays = not self.show_rays
            elif k == pygame.K_m:
                self.manual = not self.manual
            elif k == pygame.K_c:
                self.show_consts = not self.show_consts
            elif k == pygame.K_h:
                self.show_help = not self.show_help
            elif k in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                self.turbo, self.speed_i = False, min(self.speed_i + 1, len(SPEEDS) - 1)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.turbo, self.speed_i = False, max(self.speed_i - 1, 0)
            elif self.show_consts and k == pygame.K_TAB:
                self.tun_i = (self.tun_i + 1) % len(TUNABLES)
            elif self.show_consts and k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET,
                                            pygame.K_COMMA, pygame.K_PERIOD):
                up = k in (pygame.K_RIGHTBRACKET, pygame.K_PERIOD)
                name = TUNABLES[self.tun_i]
                setattr(cfg, name, round(getattr(cfg, name) * (1.1 if up else 1 / 1.1), 4))

    # --- disegno -----------------------------------------------------------

    def _oldest_text(self) -> str:
        o = self.evo.oldest()
        if o is None:
            return "nessun sopravvissuto: sono tutti appena nati"
        return (f"il più anziano: #{o['id']}, nato alla gen. {o['born']}, "
                f"ha vinto {o['age']} {'prova' if o['age'] == 1 else 'prove'} di fila")

    def _selection_text(self) -> str:
        sel = self.evo.selection
        if sel is None:
            return ""
        if self.evo.selecting:
            return (f"SELEZIONE FINALE: prova {sel['done'] + 1}/{sel['rounds']}, "
                    f"in gara {self.world.n} (partiti {sel['entered']}), nessun figlio")
        return f"selezione finale conclusa: restano {self.world.n} su {sel['entered']}"

    def _ladder_lines(self) -> list[str]:
        """A che punto è la scala, in due righe corte: lo stadio, e cosa
        trovano gli agenti in questa generazione. Vuote fuori dal mondo del cibo."""
        cfg, task = self.cfg, self.world.task
        if cfg.task != "cibo" or not hasattr(task, "need"):
            return []
        i, f = ladder.stage(cfg)
        if cfg.ladder == "scala":
            head = (f"stadio {i + 1}/3, {ladder.STAGE_NAMES[i]}: {f:.0%}" if ladder.position(cfg) < ladder.TOP
                    else "scala in cima: mondo complesso")
        else:
            head = {"basico": "controllo: sempre il mondo basico",
                    "complesso": "controllo: sempre il mondo complesso"}.get(cfg.ladder, "manopole ferme")
        prey = f"prede a {task.prey_speed:.1f} u/s" if task.prey_speed > 0 else "cibo fermo"
        if task.scent_tau > 0 and cfg.food_smell > 0:
            prey += f", nube in {task.scent_tau:.1f} s"
        pred = (f"{task.n_pred} predatori a {task.pred_speed:.1f} u/s" if task.n_pred
                else "nessun predatore" + (" (dallo stadio 3)" if cfg.ladder == "scala" else ""))
        return [head, f"ora: {task.n_food} cibi r{task.food_r:.1f}, servono {task.need}; {prey}; {pred}"]

    def _env_text(self) -> str:
        """Lo stato del registro dell'ambiente, in una riga."""
        if self.cfg.task == "cibo":
            moves = sum(e["event"] == "scala" for e in self.evo.log)
            other = sum(e["event"] == "cambio" for e in self.evo.log)
            return (f"ambiente: mondo del cibo, la scala si è mossa {moves} "
                    f"{'volta' if moves == 1 else 'volte'}, altri cambi {other}")
        changes = [e for e in self.evo.log if e["event"] == "cambio"]
        code = next((e["code"] for e in reversed(self.evo.log) if "code" in e), "sconosciuta")
        if not changes:
            return f"ambiente: invariato (registrato nel salvataggio)   codice {code}"
        last = changes[-1]
        return (f"ambiente: {len(changes)} {'cambio' if len(changes) == 1 else 'cambi'}, ultimo alla gen. "
                f"{last['generation']}: {describe_change(last['changes'])}"[:150])

    def _rate(self) -> float:
        """Generazioni al minuto di calcolo da quando la corsa è aperta."""
        return (self.evo.generation - self.gen0) * 60.0 / max(self.busy, 1e-9)

    def draw(self, s) -> None:
        if self.watch:
            self._draw_arena()
        else:
            self._draw_page(s)

    def _draw_arena(self) -> None:
        evo, world, focus = self.evo, self.world, self.focus
        if world.done[focus]:
            status = "COMPITO COMPLETATO"
        elif world.alive[focus]:
            status = "vivo"
        else:
            status = "MORTO"
        pace = "TURBO" if self.turbo else f"x{SPEEDS[self.speed_i]}"
        hud = [
            (f"generazione {evo.generation}   t={world.t:5.1f}s   {pace}{'   PAUSA' if self.paused else ''}", TEXT),
            *[(line, PLAYER if k == 0 else TEXT) for k, line in enumerate(self._ladder_lines())],
            *([] if self._ladder_lines() else [(f"compito: {world.task.name}", TEXT)]),
            (f"in questa generazione: completato {int(world.done.sum())} su {world.n}, "
             f"morti {int((~world.alive).sum())}", TEXT),
        ]
        if self._selection_text():
            hud.insert(0, (self._selection_text(), PLAYER))
        hud += [
            ("", TEXT),
            (f"seguito: #{evo.ids[focus]} ({'guidi tu' if self.manual else 'guida la rete'}), "
             f"età {evo.age[focus]}: {status}", PLAYER),
            (f"energia {world.energy[focus]:5.1f}   v={world.speed[focus]:5.1f} u/s   "
             f"w={world.ang_vel[focus]:+5.2f} rad/s", PLAYER),
        ]
        if self.renderer.task_line(focus):
            hud.append((self.renderer.task_line(focus), PLAYER))
            hud.append(("cibo e predatori disegnati: la copia di mondo dell'agente seguito", DIM))
        hud.append(("", TEXT))
        if self.show_consts:
            for i, name in enumerate(TUNABLES):
                mark = ">" if i == self.tun_i else " "
                hud.append((f"{mark} {name:17s} {getattr(self.cfg, name):8.3f}", TEXT if i == self.tun_i else DIM))
            hud.append(("Tab scegli | , . (o [ ]) regola | C chiudi", DIM))
            hud.append(("ogni cambio va nel registro dell'ambiente, salvato con la corsa", DIM))
        elif self.show_help:
            hud.append(("+/- velocita' | T turbo | Spazio pausa | S salva | C costanti", DIM))
            hud.append(("clic segui un agente | M guida manuale (WASD/frecce) | V raggi", DIM))
            hud.append(("O segui il più anziano | F pagina principale | I reti e genomi | Esc menu | H chiudi", DIM))
        else:
            hud.append(("H tasti | F pagina principale", DIM))
        if self.message:
            hud.append((self.message[:100], DIM))
        self.renderer.draw(focus, hud, evo.ctrl, self.show_rays, evo.age, evo.history)

    def _draw_page(self, s) -> None:
        """La pagina principale: intestazione, due schede (panoramica e reti),
        suggerimenti in fondo."""
        app, evo, world, cfg = self.app, self.evo, self.world, self.cfg
        s.fill(BG)
        W, H = s.get_size()
        if self.full:
            self.tab = "overview"
        title = "Selezione finale" if evo.selecting else f"Generazione {evo.generation}"
        s.blit(app.big.render(title, True, PLAYER), (16, 12))
        state = "IN PAUSA" if self.paused else ("MASSIMA VELOCITÀ, " if self.full else "") + f"{self._rate():.1f} gen/min"
        seed = "" if evo.seed is None else f"   seme {evo.seed}"
        head = (f"{state}   rete {KIND_NAMES[cfg.net_type]}, {cfg.population} individui{seed}"
                + (f"   si ferma alla gen. {self.goal}" if self.goal else "")
                + (f", poi {self.select} prove di selezione" if self.goal and self.select else ""))
        s.blit(app.small.render(head, True, TEXT), (250, 10))
        n_surv = int((evo.age > 0).sum())
        if evo.selecting:
            who = self._selection_text()
        elif evo.selection is not None and n_surv == world.n:
            who = f"restano {n_surv} dalla selezione finale"
        else:
            who = f"in gara: {n_surv} sopravvissuti della generazione prima + {world.n - n_surv} figli nuovi"
        s.blit(app.small.render(f"{who}   |   generazione in corso: t={world.t:4.1f}s, completato "
                                f"{int(world.done.sum())}, morti {int((~world.alive).sum())}", True, DIM), (250, 28))

        # Le schede: clic o V.
        self.tab_rects = {}
        x = W - 16
        for key, label in (("nets", "Reti  "), ("overview", "Panoramica")):
            img = app.font.render(label.strip(), True, BG if self.tab == key else TEXT)
            r = pygame.Rect(0, 10, img.get_width() + 24, 30)
            r.right = x
            pygame.draw.rect(s, PLAYER if self.tab == key else (32, 35, 46), r, border_radius=4)
            pygame.draw.rect(s, PLAYER, r, 1, border_radius=4)
            s.blit(img, img.get_rect(center=r.center))
            self.tab_rects[key] = r
            x = r.x - 8
        pygame.draw.line(s, BORDER, (0, 50), (W, 50), 1)

        self.oldest_rects = []
        if self.tab == "overview":
            self.overview.draw(s, pygame.Rect(12, 60, W - 24, H - 60 - 62), evo, self._env_text())
            self.oldest_rects = self.overview.oldest_rects
            hints = ("V reti | F arena | E esame | L lesione | A sopravvissuti in arena | O più anziano | "
                     "Z max velocità | Spazio pausa | S salva | Esc menu")
        else:
            ex = self.explorer
            ex.place((16, 58), pygame.Rect(10, 84, 216, H - 84 - 54),
                     pygame.Rect(236, 56, 700, H - 56 - 50), pygame.Rect(956, 106, W - 962, H - 106 - 60))
            pygame.draw.line(s, BORDER, (230, 82), (230, H - 50), 1)
            pygame.draw.line(s, BORDER, (946, 56), (946, H - 50), 1)
            ex.draw(s, app.font, app.small)
            hints = ("V panoramica | 1/2/3 elenchi | T prova la rete scelta | P rami morti | I genoma | "
                     "F arena | O più anziano | Spazio pausa | S salva | Esc menu")

        if self.summary_open:
            self._draw_summary(s)
        if self.lesion_menu:
            self._draw_lesions(s)
        pygame.draw.line(s, BORDER, (0, H - 50), (W, H - 50), 1)
        pygame.draw.line(s, BORDER, (0, H - 30), (W, H - 30), 1)
        s.blit(app.small.render(hints, True, DIM), (12, H - 22))
        if self.message:
            s.blit(app.small.render(self.message[:190], True, TARGET), (12, H - 44))

    def _draw_summary(self, s) -> None:
        app = self.app
        box = pygame.Rect(480, 90, 940, 40 + 20 * len(self.summary) + 70)
        back = pygame.Surface(box.size, pygame.SRCALPHA)
        back.fill((14, 16, 22, 245))
        s.blit(back, box)
        pygame.draw.rect(s, PLAYER, box, 1, border_radius=4)
        for k, (text, color) in enumerate(self.summary):
            font = app.big if k == 0 else app.small
            s.blit(font.render(text[:130], True, color), (box.x + 20, box.y + 16 + 20 * k + (0 if k == 0 else 8)))
        x, y = box.x + 20, box.bottom - 52
        self.buttons[0].enabled = bool((self.evo.age > 0).any())
        for b in self.buttons:
            x += b.draw(s, app.small, x, y) + 10

    def _draw_lesions(self, s) -> None:
        """Il menu delle lesioni: cosa togliere ai più anziani per l'esame."""
        app = self.app
        names = {"neuroni": "i neuroni interni", "plasticità": "la plasticità (pesi fissi per tutta la vita)"}
        lines = [f"{k + 1}   {names.get(x, f'il senso: {x}')}" for k, x in enumerate(LESIONS)]
        box = pygame.Rect(480, 90, 620, 110 + 20 * len(lines))
        back = pygame.Surface(box.size, pygame.SRCALPHA)
        back.fill((14, 16, 22, 245))
        s.blit(back, box)
        pygame.draw.rect(s, MODULATOR, box, 1, border_radius=4)
        s.blit(app.font.render("ESAME CON LESIONE: cosa togliere?", True, MODULATOR), (box.x + 20, box.y + 14))
        s.blit(app.small.render("ai 20 più anziani, sulle stesse prove dell'esame; gli originali non cambiano",
                                True, DIM), (box.x + 20, box.y + 38))
        for k, line in enumerate(lines):
            s.blit(app.font.render(line, True, TEXT), (box.x + 30, box.y + 64 + 20 * k))
        s.blit(app.small.render("un altro tasto chiude", True, DIM), (box.x + 20, box.bottom - 26))
