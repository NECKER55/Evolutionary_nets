"""Il menu: si sceglie la corsa da lanciare (rete, individui, mondo, scala,
seme, generazioni) oppure si apre un salvataggio, per continuarlo o per
esplorarne le reti. `Screen` è la base di tutte le schermate."""

import json
import secrets
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame

from ..config import Config
from ..evo import Evolution
from ..net.anatomy import KIND_NAMES
from .render import BG, DIM, PLAYER, TARGET, TEXT
from .widgets import ROW_H, Button, Choice, Number, Text

RUNS = Path("runs")


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

    def __init__(self, app, defaults: dict):
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
        self.sim = Choice("Simulazione", [("evoluzione", "completa: evolvono sensi e cervello"),
                                          ("cervello", "solo cervello: sensi fissi (set naturale)")],
                          help="Solo cervello: tutti hanno gli stessi 10 sensi (net/senses.py), che non "
                               "cambiano; evolve solo il cervello. Ignorato se si riprende un salvataggio.")
        self.sim.select(defaults.get("sim") or "evoluzione")
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
        self.fields = [self.mode, self.sim, self.net, self.agents, self.task, self.ladder, self.chance, self.seed,
                       self.gens, self.exam_every, self.select, self.load, self.save]
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
        fields = (self.sim, self.net, self.agents, self.task, self.ladder, self.chance, self.seed)
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
        self.sim.enabled = path is None
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
                             (self.ladder, saved.get("ladder")), (self.sim, saved.get("sim_mode") or "evoluzione")):
            if value in [v for v, _ in field.options]:
                field.select(value)
        self.agents.value = n
        if seed is not None:
            self.chance.select("fixed")
            self.seed.value = seed

    def _config(self) -> Config:
        cfg = Config(net_type=self.net.value, task=self.task.value, sim_mode=self.sim.value)
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
            from .app import (
                RunScreen,  # la corsa importa il menu: qui si importa al momento
            )

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
        y = 100
        for k, f in enumerate(self.fields):
            f.draw(s, app.font, 60, y, 175, 540, k == self.focus)
            y += ROW_H + 2
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
