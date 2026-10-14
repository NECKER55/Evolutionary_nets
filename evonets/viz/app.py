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

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame

from ..config import TUNABLES, Config
from ..evo import Evolution
from ..net.anatomy import KIND_NAMES
from .render import BG, BORDER, DIM, PLAYER, TARGET, TEXT, WALL, Renderer, plot, success_series
from .widgets import ROW_H, Button, Choice, Number, Text

SPEEDS = (1, 2, 4, 8, 16)
TURBO_BUDGET = 0.08  # s di calcolo per fotogramma guardando l'arena in turbo
PAGE_BUDGET = 0.05  # s di calcolo per fotogramma con la pagina principale (resta reattiva)
FULL_BUDGET = 0.25  # s di calcolo per fotogramma in "solo calcolo"
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
        """Apre il banco di prova per una rete: (genoma, nome, descrizione)."""
        from .testbench import TestScreen

        if selection is not None:
            cfg = self.active_run.cfg if self.active_run else back.evo.cfg
            self.show(TestScreen(self, cfg, selection, back))

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
            net = str(d["net_type"]) if "net_type" in d else "rate"
            n = len(d["weights"])
            text = f"generazione {int(d['generation'])}, {n} individui, rete {KIND_NAMES.get(net, net)}"
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
                                         ("live", "dal vivo: arena sempre a schermo")],
                           help="La veloce calcola senza disegnare; l'arena si apre e si chiude con F.")
        self.net = Choice("Rete", [("rate", "continua (tanh)"), ("spike-ei", "a impulsi, eccitatori/inibitori")],
                          help="Tipo di neurone. Ignorato se si riprende un salvataggio.")
        self.net.select(defaults.get("net", "rate"))
        self.agents = Number("Individui", defaults.get("agents") or 200, step=50, low=2,
                             help="Dimensione della popolazione. Ignorato se si riprende un salvataggio.")
        self.seed = Number("Seme", defaults.get("seed", 0),
                           help="Stesso seme e stesse impostazioni = stessa corsa.")
        self.gens = Number("Generazioni", 0, step=50,
                           help="Dopo quante generazioni fermarsi e salvare. 0 = senza limite.")
        load = defaults.get("load")
        options = [(None, "nessuno: reti nuove")] + [(str(p), p.name) for p in files]
        if load and load not in [v for v, _ in options]:
            options.insert(1, (load, Path(load).name))
        self.load = Choice("Riprendi da", options, help="Un salvataggio in runs/ da cui continuare o da esplorare.",
                           on_change=self._loaded)
        self.save = Text("Salva in", defaults.get("save") or self._fresh_name(),
                         help="Dove salvare: tasto S, fine delle generazioni, uscita dalla corsa.")
        self.fields = [self.mode, self.net, self.agents, self.seed, self.gens, self.load, self.save]
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
        self.net.enabled = self.agents.enabled = self.seed.enabled = path is None
        if path is not None:
            self.save.value, self.save.edited = path, False
        elif not self.save.edited:
            self.save.value = self._fresh_name()

    def _config(self) -> Config:
        cfg = Config(net_type=self.net.value)
        cfg.population = self.agents.fixed()
        return cfg

    def _evolution(self) -> Evolution | None:
        cfg = self._config()
        try:
            if self.load.value:
                return Evolution.load(cfg, self.load.value)
            return Evolution(cfg, self.seed.fixed())
        except Exception as err:
            self.message, self.error = f"non riesco ad aprire il salvataggio: {err}", True
            return None

    def start(self) -> None:
        evo = self._evolution()
        if evo is not None:
            save = self.save.value.strip() or self._fresh_name()
            run = RunScreen(self.app, evo, save, self.gens.fixed(), watch=self.mode.value == "live")
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
        s.fill(BG)
        s.blit(app.big.render("Evolutionary nets", True, PLAYER), (60, 40))
        s.blit(app.font.render("nuova corsa, ripresa di un salvataggio o esplorazione delle sue reti", True, DIM),
               (60, 76))
        y = 130
        for k, f in enumerate(self.fields):
            f.draw(s, app.font, 60, y, 150, 560, k == self.focus)
            y += ROW_H + 4
        field = self.fields[self.focus]
        s.blit(app.small.render(field.help, True, DIM), (210, y + 2))
        if self.load.value:
            s.blit(app.small.render(save_info(self.load.value), True, TARGET), (210, y + 20))
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
            ("La pagina principale mostra le statistiche", DIM),
            ("e le reti dei vincitori, esplorabili mentre", DIM),
            ("l'evoluzione continua.", DIM),
            ("F   apre/chiude l'arena (la corsa continua)", DIM),
            ("T   prova da sola la rete scelta", DIM),
            ("I   reti e genomi a tutto schermo", DIM),
            ("S   salva     Spazio   pausa", DIM),
            ("Esc torna qui (salva se ci sono novità)", DIM),
            ("", TEXT),
            ("Le tracce delle reti (un campione per", DIM),
            ("generazione) finiscono nel salvataggio:", DIM),
            ("si possono esplorare anche dopo.", DIM),
        ]
        for k, (text, color) in enumerate(lines):
            s.blit(app.font.render(text, True, color), (help_x, 130 + 20 * k))


class RunScreen(Screen):
    """Una corsa. Pagina principale: statistiche a sinistra, reti dei vincitori
    a destra, esplorabili mentre l'evoluzione continua. F apre l'arena."""

    def __init__(self, app: App, evo: Evolution, save: str, limit: int = 0, watch: bool = False):
        from .inspector import CHAMPIONS, Explorer

        self.app, self.evo, self.cfg, self.save_path = app, evo, evo.cfg, save
        self.world = evo.world
        self.renderer = Renderer(evo.world, app.screen)
        self.explorer = Explorer(evo, CHAMPIONS)
        self.watch = watch
        self.full = False  # "solo calcolo": niente esploratore, massima velocità
        self.goal = evo.generation + limit if limit else None
        self.focus = int(np.argmax(evo.age))
        self.manual = self.paused = self.show_consts = False
        self.turbo = not watch  # aprendo l'arena da una corsa veloce si resta veloci
        self.show_rays = True
        self.speed_i, self.tun_i = 0, 0
        self.saved_gen = evo.generation
        self.message = ""
        self.gen0, self.busy = evo.generation, 0.0

    # --- evoluzione --------------------------------------------------------

    def _save(self) -> None:
        self.evo.save(self.save_path)
        self.saved_gen = self.evo.generation
        self.message = f"salvata la generazione {self.saved_gen} in {self.save_path}"

    def _advance(self, hand=None) -> bool:
        """Un passo; False quando bisogna fermarsi (limite di generazioni)."""
        evo = self.evo
        if evo.finished:
            evo.next_generation()
            self.focus = int(np.argmax(evo.age))  # il sopravvissuto più anziano, se c'è
            if self.goal is not None and evo.generation >= self.goal:
                self.goal, self.paused = None, True
                self._save()
                self.message = f"raggiunte le generazioni richieste: {self.message}. Spazio per continuare"
                return False
        else:
            evo.step((self.focus, hand) if self.manual and hand is not None else None)
        return True

    def _compute(self, budget: float, hand=None) -> None:
        start = time.perf_counter()
        deadline = start + budget
        while time.perf_counter() < deadline and self._advance(hand):
            pass
        self.busy += time.perf_counter() - start

    def background(self, budget: float) -> None:
        """Avanza mentre si guarda un'altra schermata."""
        if not self.paused:
            self._compute(budget)

    def update(self) -> None:
        self.explorer.refresh()
        if self.paused:
            return
        if not self.watch:
            self._compute(FULL_BUDGET if self.full else PAGE_BUDGET)
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
            self._compute(TURBO_BUDGET, hand)
        else:
            start = time.perf_counter()
            for _ in range(SPEEDS[self.speed_i]):
                if not self._advance(hand):
                    break
            self.busy += time.perf_counter() - start

    def _leave(self) -> str:
        self.app.active_run = None
        if self.evo.generation != self.saved_gen:
            self._save()
            return self.message
        return ""

    def close(self) -> None:
        self._leave()

    # --- comandi -----------------------------------------------------------

    def handle(self, ev) -> None:
        if ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_ESCAPE:
                self.app.launcher.returned(self._leave())
                return
            if k == pygame.K_SPACE:
                self.paused = not self.paused
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
        if self.watch:
            self._handle_arena(ev)
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_t:
            self.app.test(self.explorer.selected(), back=self)
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_z:
            self.full = not self.full
        elif not self.full:
            self.explorer.handle(ev)

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
            (f"generazione {evo.generation}   t={world.t:5.1f}s   {pace}   {self.app.clock.get_fps():3.0f} fps"
             f"{'   PAUSA' if self.paused else ''}", TEXT),
            (f"compito: {world.task.name}", TEXT),
            (f"completato {int(world.done.sum())}/{world.n}   morti {int((~world.alive).sum())}   "
             f"sopravvissuti in gara {int((evo.age > 0).sum())}", TEXT),
        ]
        if evo.history:
            h = evo.history[-1]
            note = "" if h["event"] == "normale" else f"   [{h['event']}]"
            hud.append((f"gen. precedente: {h['completed']}/{world.n} completati, "
                        f"{h['neurons']:.1f} neuroni e {h['links']:.0f} collegamenti in media{note}", DIM))
        hud += [
            ("", TEXT),
            (f"agente {focus} ({'guidi tu' if self.manual else 'guida la rete'}), "
             f"età {evo.age[focus]}: {status}", PLAYER),
            (f"energia {world.energy[focus]:5.1f}   v={world.speed[focus]:5.1f} u/s   "
             f"w={world.ang_vel[focus]:+5.2f} rad/s", PLAYER),
            ("", TEXT),
        ]
        if self.show_consts:
            for i, name in enumerate(TUNABLES):
                mark = ">" if i == self.tun_i else " "
                hud.append((f"{mark} {name:17s} {getattr(self.cfg, name):8.3f}", TEXT if i == self.tun_i else DIM))
            hud.append(("Tab scegli | , . (o [ ]) regola | C chiudi", DIM))
        else:
            hud.append(("+/- velocita' | T turbo | Spazio pausa | S salva | C costanti", DIM))
            hud.append(("clic segui un agente | M guida manuale (WASD/frecce) | V raggi", DIM))
            hud.append(("F torna alla pagina principale | I reti e genomi | Esc menu", DIM))
        if self.message:
            hud.append((self.message[:100], DIM))
        self.renderer.draw(focus, hud, evo.ctrl, self.show_rays, evo.age, evo.history)

    def _draw_page(self, s) -> None:
        app, evo, world, cfg = self.app, self.evo, self.world, self.cfg
        s.fill(BG)
        W, H = s.get_size()
        state = "IN PAUSA" if self.paused else f"{self._rate():.1f} gen/min"
        s.blit(app.big.render(f"Generazione {evo.generation}", True, PLAYER), (16, 10))
        head = (f"{state}   rete {KIND_NAMES[cfg.net_type]}, {cfg.population} individui"
                + (f"   si ferma alla gen. {self.goal}" if self.goal else ""))
        s.blit(app.small.render(head, True, TEXT), (250, 12))
        s.blit(app.small.render(
            f"in corso: t={world.t:4.1f}s  completato {int(world.done.sum())}  morti {int((~world.alive).sum())}  "
            f"in gara {int((world.alive & ~world.done).sum())}   compito: {world.task.name}", True, DIM), (250, 30))
        pygame.draw.line(s, BORDER, (0, 54), (W, 54), 1)

        self._draw_stats(s, pygame.Rect(0, 58, 456, H - 92))
        pygame.draw.line(s, BORDER, (460, 54), (460, H - 30), 1)
        if self.full:
            msg = ["Solo calcolo: l'esploratore è nascosto e quasi tutto il tempo va", "all'evoluzione. Z per riaprirlo."]
            for k, text in enumerate(msg):
                s.blit(app.font.render(text, True, DIM), (490, 100 + 22 * k))
        else:
            ex = self.explorer
            ex.place((470, 62), pygame.Rect(464, 86, 216, H - 86 - 34),
                     pygame.Rect(684, 58, 470, H - 58 - 30), pygame.Rect(1164, 108, W - 1170, H - 108 - 40))
            pygame.draw.line(s, BORDER, (680, 84), (680, H - 30), 1)
            pygame.draw.line(s, BORDER, (1156, 58), (1156, H - 30), 1)
            ex.draw(s, app.font, app.small)

        pygame.draw.line(s, BORDER, (0, H - 30), (W, H - 30), 1)
        hints = ("F arena | Su/Giù/clic scegli rete | clic neurone | T prova la rete | Tab campioni/popolazione | "
                 "P rami morti | I genoma | Z solo calcolo | Spazio pausa | S salva | Esc menu")
        s.blit(app.small.render(hints, True, DIM), (12, H - 22))
        if self.message:
            img = app.small.render(self.message[:110], True, TARGET)
            s.blit(img, (W - img.get_width() - 12, 40))

    def _draw_stats(self, s, area: pygame.Rect) -> None:
        app, cfg, hist = self.app, self.cfg, self.evo.history
        x0, y = area.x + 12, area.y + 4
        # Riquadri con i numeri chiave.
        if hist:
            pct, avg = success_series(hist, cfg.population)
            best = int(np.argmax(pct))
            last = hist[-1]
            tiles = [("ultima gen.", f"{pct[-1]:.0f}%"), ("media 10", f"{avg[-1]:.0f}%"),
                     ("record", f"{pct[best]:.0f}% g{hist[best]['generation']}"),
                     ("età max", f"{last['oldest']}"),
                     ("neuroni", f"{last['neurons']:.1f}"), ("colleg.", f"{last['links']:.0f}"),
                     ("durata", f"{last['duration']:.1f}s"), ("campioni", f"{len(self.evo.champions)}")]
        else:
            tiles = [("ultima gen.", "-"), ("media 10", "-"), ("record", "-"), ("età max", "-"),
                     ("neuroni", "-"), ("colleg.", "-"), ("durata", "-"), ("campioni", "0")]
        tw = (area.w - 24) // 4
        for k, (label, value) in enumerate(tiles):
            r = pygame.Rect(x0 + (k % 4) * tw, y + (k // 4) * 40, tw - 6, 36)
            pygame.draw.rect(s, (24, 27, 36), r, border_radius=3)
            s.blit(app.small.render(label, True, DIM), (r.x + 6, r.y + 3))
            s.blit(app.font.render(value, True, TEXT), (r.x + 6, r.y + 17))
        y += 86

        spiking = bool(hist) and "inhibitory" in hist[-1]
        graphs = [
            ("successi (%): sottile ogni gen., spessa media di 10",
             lambda: ((pct, WALL, 1), (avg, TARGET, 2)), lambda: 100.0, (0, 0.5, 1)),
            ("reti: arancio neuroni interni, azzurro collegamenti/10",
             lambda: ((col("neurons"), PLAYER, 2), (col("links") / 10, TARGET, 2)),
             lambda: max(col("neurons").max(), col("links").max() / 10, 1.0), (0, 0.5, 1)),
            ("sopravvissuti (bianco) ed età massima (arancio)",
             lambda: ((col("survivors"), WALL, 1), (col("oldest"), PLAYER, 2)),
             lambda: max(col("survivors").max(), col("oldest").max(), 1.0), (0, 0.5, 1)),
            ("quota di neuroni inibitori (%)" if spiking else "durata della generazione (s)",
             (lambda: ((col("inhibitory") * 100, TARGET, 2),)) if spiking else (lambda: ((col("duration"), TARGET, 2),)),
             (lambda: 100.0) if spiking else (lambda: max(col("duration").max(), 1.0)), (0, 0.5, 1)),
        ]

        def col(name):
            return np.array([h[name] for h in hist], dtype=float)

        gh = (area.bottom - y) // len(graphs)
        for title, series, top, ticks in graphs:
            s.blit(app.small.render(title, True, DIM), (x0, y))
            box = pygame.Rect(area.x + 50, y + 22, area.w - 64, gh - 44)
            if hist:
                plot(s, box, app.small, series(), top(), ticks)
            else:
                plot(s, box, app.small, (), 1.0, ticks)
            y += gh
        if hist:
            s.blit(app.small.render(f"gen. {hist[0]['generation']}-{hist[-1]['generation']}", True, DIM),
                   (area.x + 50, area.bottom - 16))
