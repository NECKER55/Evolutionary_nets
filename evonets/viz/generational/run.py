"""La corsa a generazioni: la pagina principale mostra le statistiche e, accanto, le reti dei
vincitori delle generazioni già fatte, da esplorare mentre l'evoluzione va avanti; con F si apre l'arena.
"""

import time
from pathlib import Path

import numpy as np
import pygame

from ...arenas.generational import ladder
from ...arenas.generational.evolution import Evolution, describe_change
from ...arenas.generational.exam import LESIONS
from ...brains.freeform.anatomy import KIND_NAMES
from ...config import TUNABLES
from ..launcher import Screen
from ..render import BG, BORDER, DIM, MODULATOR, PLAYER, TARGET, TEXT, Renderer, success_series
from ..widgets import Button
from .overview import Overview

SPEEDS = (1, 2, 4, 8, 16)
TURBO_BUDGET = 0.08  # s di calcolo per fotogramma guardando l'arena in turbo
PAGE_BUDGET = 0.05  # s di calcolo per fotogramma con la pagina principale (resta reattiva)
MAX_BUDGET = 0.5  # s di calcolo per fotogramma alla massima velocità (si ridisegna 2 volte al secondo)
CHUNK = 60  # passi per chiamata quando si calcola a blocchi (un secondo simulato)


class RunScreen(Screen):
    """Una corsa. Pagina principale: statistiche a sinistra, reti dei vincitori
    a destra, esplorabili mentre l'evoluzione continua. F apre l'arena."""

    def __init__(self, app, evo: Evolution, save: str, limit: int = 0, watch: bool = False,
                 select: int = 0, full: bool = False):
        from ..inspector import CHAMPIONS, Explorer

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
                from ..inspector import Inspector

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
            head = (f"stadio {i + 1}/{len(ladder.STAGES)}, {ladder.STAGE_NAMES[i]}: {f:.0%}"
                    if ladder.position(cfg) < ladder.TOP
                    else "scala in cima: mondo complesso")
        else:
            head = {"basico": "controllo: sempre il mondo basico",
                    "complesso": "controllo: sempre il mondo complesso"}.get(cfg.ladder, "manopole ferme")
        prey = f"prede a {task.prey_speed:.1f} u/s" if task.prey_speed > 0 else "cibo fermo"
        if task.scent_tau > 0 and cfg.food_smell > 0:
            prey += f", nube in {task.scent_tau:.1f} s"
        pred = (f"{task.n_pred} predatori a {task.pred_speed:.1f} u/s" if task.n_pred
                else "nessun predatore" + (" (dallo stadio 3)" if cfg.ladder == "scala" else ""))
        if task.n_rock:
            pred += f"; {task.n_rock} rocce"
        if task.bad.any():
            pred += f"; {int(task.bad[0].sum())} frutti cattivi"
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
        senses = ", sensi fissi" if cfg.sim_mode == "cervello" else ""
        head = (f"{state}   rete {KIND_NAMES[cfg.net_type]}{senses}, {cfg.population} individui{seed}"
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
