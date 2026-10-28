"""La schermata dell'arena delle pile (docs/SPEC_ARENA_PILE.md): a sinistra un
recinto, con l'essere seguito, la sua vista e i suoi raggi; a destra il suo
cervello, sezione per sezione (docs/SPEC_CERVELLO_SEZIONI.md), con i neuroni
accesi, i collegamenti che contano e una spiegazione di ciò che il mouse
indica.

Il cervello si legge da sinistra a destra, come scorre il segnale: A (i
sensi) -> B (cosa c'è) -> D (cosa è appena successo) -> C (cosa ho imparato
di recente; L la versione lenta) -> E (cosa faccio) -> F (come lo faccio).
V (il critico) guarda B, D, C e manda δ, la sorpresa, alle sezioni che
imparano; U (l'incertezza) alza velocità ed esplorazione dopo molte sorprese.
"""

import time

import numpy as np
import pygame

from ..net.sections import BITE, FAST, FOVEA, MAP, MOTOR, N_A, N_HEADS, fovea_of
from ..sim.pile import (BUMP, ENERGY, F_INPUTS, F_NEURONS, FEATURES, N_PROGRAMS, N_SENSES, PAIN, PROGRAMS, RAY, RAYS,
                        SECTORS, SENSES, TASTE, WALL_COLOURS, WATCHED, battery_good, objects_of)
from .launcher import Screen
from .render import BG, BORDER, DIM, PLAYER, TARGET, TEXT

RED_C, YELLOW_C, BLUE_C = (235, 80, 70), (240, 210, 60), (90, 130, 255)
BEING_C = (150, 200, 160)
ME_C = (110, 230, 255)  # l'essere seguito (non giallo: si confonderebbe con le pile gialle)
TEACHER_C = (255, 190, 60)  # il maestro (oro, con la corona)
POS, NEG = (90, 220, 120), (240, 90, 80)  # rafforza, indebolisce
FIXED = (95, 100, 120)  # collegamenti fissi
DELTA_C = (200, 120, 255)  # la sorpresa
ZONE_C = (40, 46, 34)
SPEEDS = (1, 3, 6, 12, 24, 48)  # volte il tempo reale
SHORT = tuple(n.replace("occhio ", "").replace("sinistra", "sx").replace("centro", "c").replace("destra", "dx")
              for n in SENSES)
WALL_C = ((90, 200, 110), (170, 110, 230), (240, 150, 60), (230, 230, 235))  # i muri: nord, est, sud, ovest
EYE_ROWS = (("rosso", RED_C), ("giallo", YELLOW_C), ("blu", BLUE_C)) + tuple(
    (f"muro {c}", col) for c, col in zip(WALL_COLOURS, WALL_C))
PROG_SHORT = ("avvicinati", "allontanati", "lascialo", "esplora", "mordi")
F_SHORT = ("vai verso", "vai via", "guarda altro", "vaga")
FOVEA_SHORT = tuple(f"fov. {f}" for f in FEATURES) + ("fov. vicino", "fov. sinistra", "fov. destra", "fov. nuovo")

SECTION_HELP = {
    "A": ("A  interfaccia dei sensi (fissa)",
          "Cosa sentono i sensi adesso: barra piena = valore, tacca chiara = quanto è appena salito ('cosa è "
          "cambiato'). Sotto, la fovea: colore, vicinanza e lato del solo oggetto attento. Non impara: è come la "
          "retina."),
    "B": ("B  nucleo stabile: cosa c'è",
          "16 neuroni in gara: vincono i 3 più simili a ciò che arriva. Ognuno si specializza da solo, piano, su "
          "una situazione che si ripete (nessuno nasce 'rosso'). Le linee verso A: a cosa risponde."),
    "D": ("D  memoria di lavoro: che cosa ho appena guardato",
          "Una copia della fovea, scritta quando entra un oggetto nuovo, che resta per qualche secondo anche "
          "quando l'oggetto non si vede più: quando arriva il gusto o il dolore, D sa ancora che cosa si è preso. "
          "Il cancello decide quando scrivere: è l'unica cosa che impara."),
    "C": ("C  memoria veloce (e L, lenta)",
          "64 neuroni con ingressi fissi e casuali, soprattutto dall'oggetto (fovea e D) e un po' dal contesto (B), "
          "come il corpo a fungo degli insetti con gli odori: lo stesso oggetto accende sempre gli stessi. Le loro uscite verso E imparano in un colpo e dimenticano in minuti. L è la "
          "stessa uscita, cento volte più lenta: tiene solo ciò che vale sempre."),
    "E": ("E  selezione: cosa faccio",
          "4 programmi in gara (la barra è la probabilità di sceglierlo; vince uno alla volta, con un po' di caso) "
          "e, a parte, se mordere (si mangia solo mordendo, e mordere costa). Impara con δ: si rafforza ciò che "
          "ha portato meglio del previsto."),
    "F": ("F  motore: come lo faccio (innato)",
          "Un piccolo circuito di neuroni innati, come i circuiti motori del tronco encefalico: dal programma "
          "scelto in E e da dove sta l'oggetto attento a 4 neuroni motori (gira a sinistra, a destra, avanti, "
          "indietro), più il riflesso dei muri. Non impara nella vita; i suoi pesi sono geni (fase B). Il morso "
          "lo decide E."),
    "S": ("S  attenzione: su cosa mi concentro",
          "Una mappa di priorità fatta di posizioni (come il collicolo superiore): 9 neuroni-direzione, da sinistra "
          "a destra, ognuno con l'oggetto più vicino in quella direzione; più è acceso, più è probabile che lo "
          "sguardo vada lì (anello: dove va adesso). S non impara: sotto ogni neurone, le quattro cose che somma: "
          "la salienza innata (grigio: vicino, nuovo, inerzia, inibizione del ritorno) e ciò che O dice "
          "dell'oggetto: il valore (verde se attira, rosso se respinge), l'importanza (arancio: conta, buono o "
          "cattivo che sia) e l'incertezza (viola)."),
    "O": ("O  valore degli oggetti: che cosa fa bene, che cosa fa male",
          "Come la corteccia orbitofrontale: per ogni caratteristica (e lo sfondo), quanto piacere (verde) e quanto "
          "dolore (rosso) porta di solito, e quanto è incerta (viola). Impara quando il corpo sente qualcosa: alla "
          "fine dell'episodio, il merito va a ciò che si aveva in mente prima, con le caratteristiche in "
          "competizione (Rescorla-Wagner); anche da ciò che succede al maestro. Il valore di adesso è piacere × "
          "fame − dolore: da sazi il cibo vale meno senza imparare niente. In alto i due neuroni di uscita, letti "
          "da E e dal critico: quanto vale ciò che guardo, e ciò che ho appena preso."),
    "V": ("V  critico: va meglio o peggio del previsto?",
          "Prevede quanto bene arriverà da B, D, C. δ = ciò che arriva meno ciò che prevedeva: positivo = meglio "
          "del previsto. È il maestro interno: le linee viola lo portano a chi impara."),
    "U": ("U  incertezza",
          "Quante sorprese di recente rispetto al solito. Se il mondo è cambiato (un'inversione) alza la "
          "velocità di apprendimento di C ed E e l'esplorazione."),
}


def lerp(c0, c1, t):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))


def wrap_text(font, text, width):
    words, lines, line = text.split(), [], ""
    for w in words:
        test = (line + " " + w).strip()
        if font.size(test)[0] > width and line:
            lines.append(line)
            line = w
        else:
            line = test
    return lines + ([line] if line else [])


class PileScreen(Screen):
    """La simulazione nell'arena delle pile: `run` è una corsa di
    `evo.pile_test` (NumPy o GPU)."""

    def __init__(self, app, run, minutes: int = 0, save: str | None = None, back=None):
        """`save`: dove salvare (Ctrl+S, e all'uscita se ci sono novità).
        `back`: la schermata a cui tornare con Esc (la prova delle reti una a
        una torna alla corsa); senza, si torna al menu."""
        self.app, self.run = app, run
        self.cfg = run.cfg
        self.save_path, self.back = save, back
        self.testing = hasattr(run, "origin")  # le reti di una corsa, ognuna da sola nel suo recinto
        self.saved_t = run.t
        self.goal = run.t + 60.0 * minutes if minutes else None
        self.encl, self.follow = 0, 0
        self.paused = False
        self.turbo = False
        self.speed_i = 1
        self.frames = 0
        self.show_help = False
        self.hover: tuple | None = None
        self.targets: list = []  # (rettangolo, chiave) per le spiegazioni sotto il mouse
        self.message = ""
        self.trail: list = []
        self.view = run.view(0, 0)
        self.rate = 0.0
        self.last = time.perf_counter()
        self.teach = (False, 0.0)  # (c'è il maestro, quanto imparano ancora da lui)
        self.teach_hist: list = []  # (t, errore), ogni 5 s di mondo
        self.teach_lesson = [0.0, 0.0]

    # --- il tempo -------------------------------------------------------------

    def _steps_per_frame(self) -> int:
        """Passi da fare in questo fotogramma (a 60 fotogrammi al secondo)."""
        per_second = SPEEDS[self.speed_i] / self.cfg.dt
        self.frames += 1
        due = int(self.frames * per_second / 60.0) - int((self.frames - 1) * per_second / 60.0)
        return due

    def _advance(self, k):
        if k <= 0 or self._finished():
            return
        # Blocchi di grandezze fisse: sulla GPU ogni grandezza si compila una volta.
        for size in (64, 16, 4, 1):
            while k >= size:
                self.run.advance(size)
                k -= size

    def _finished(self) -> bool:
        if self.goal is not None and self.run.t >= self.goal - 1e-9:
            if not self.paused:
                self.paused = True
                self.message = "minuti finiti: la corsa è ferma (Esc per tornare al menu)"
            return True
        return False

    def background(self, budget: float) -> None:
        pass  # fuori da questa schermata la corsa si ferma

    def update(self) -> None:
        now = time.perf_counter()
        real, self.last = now - self.last, now
        t0 = self.run.t
        if not self.paused:
            if self.turbo:
                start = time.perf_counter()
                while time.perf_counter() - start < 0.08 and not self._finished():
                    self._advance(64)
            else:
                self._advance(self._steps_per_frame())
        self.rate = 0.8 * self.rate + 0.2 * (self.run.t - t0) / max(real, 1e-6)
        self.view = self.run.view(self.encl, self.encl * self.run.n + self.follow)
        p = self.view[0]["pos"][self.follow]
        self.trail.append((float(p[0]), float(p[1])))
        del self.trail[:-300]
        if hasattr(self.run, "teaching"):
            self.teach = self.run.teaching()
            if self.teach[0] and (not self.teach_hist or self.run.t - self.teach_hist[-1][0] >= 5.0):
                self.teach_hist.append((self.run.t, self.teach[1]))
                del self.teach_hist[:-240]

    # --- comandi --------------------------------------------------------------

    def _choose(self, encl=None, follow=None):
        if encl is not None:
            self.encl = encl % self.run.B
        if follow is not None:
            self.follow = follow % self.run.n
        self.trail = []

    def handle(self, ev) -> None:
        if ev.type == pygame.MOUSEMOTION:
            self.hover = next((key for r, key in self.targets if r.collidepoint(ev.pos)), None)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            box = self._arena_rect()
            if box.collidepoint(ev.pos):  # clic su un essere: lo si segue
                w = self.view[0]
                s = box.w / self.cfg.size
                pts = np.array([box.x, box.y]) + w["pos"] * s
                d = np.hypot(*(pts - np.array(ev.pos)).T)
                if d.min() < 20:
                    self._choose(follow=int(d.argmin()))
        elif ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_ESCAPE:
                if self.back is not None:
                    self.app.show(self.back)
                    return
                self.app.active_run = None
                self.app.launcher.returned(self._leave())
            elif k == pygame.K_s and ev.mod & pygame.KMOD_CTRL:
                self._save()
            elif k == pygame.K_p and not self.testing:
                self._test_networks()
            elif k == pygame.K_m and not self.testing and self.run.agent == "sezioni":
                on = not self.teach[0]
                self.run.set_teacher(on)
                self.teach = self.run.teaching()
                self.message = (f"maestro messo al minuto {self.run.t / 60:.1f}" if on else
                                f"maestro tolto al minuto {self.run.t / 60:.1f}: al suo posto un giovane")
            elif k == pygame.K_SPACE:
                self.paused = not self.paused
            elif k == pygame.K_t:
                self.turbo = not self.turbo
            elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.turbo, self.speed_i = False, min(self.speed_i + 1, len(SPEEDS) - 1)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.turbo, self.speed_i = False, max(self.speed_i - 1, 0)
            elif k == pygame.K_TAB:
                self._choose(follow=self.follow + (-1 if ev.mod & pygame.KMOD_SHIFT else 1))
            elif k in (pygame.K_PAGEDOWN, pygame.K_RIGHTBRACKET):
                self._choose(encl=self.encl + 1)
            elif k in (pygame.K_PAGEUP, pygame.K_LEFTBRACKET):
                self._choose(encl=self.encl - 1)
            elif k == pygame.K_h:
                self.show_help = not self.show_help

    def _save(self) -> None:
        from ..evo.pile_saves import save

        if self.testing or not self.save_path:
            self.message = "la prova delle reti non si salva" if self.testing else "nessun file dove salvare"
            return
        save(self.run, self.save_path)
        self.saved_t = self.run.t
        self.message = f"salvato al minuto {self.run.t / 60:.1f} in {self.save_path}"

    def _leave(self) -> str:
        """Uscendo: si salva se ci sono novità."""
        if not self.testing and self.save_path and self.run.t != self.saved_t:
            self._save()
            return self.message
        return "arena delle pile chiusa"

    def _test_networks(self) -> None:
        """P: le reti di questa corsa, ognuna da sola nel suo recinto (con ciò che
        ha imparato fin qui); Esc torna qui."""
        from ..evo.pile_saves import alone

        engine = "numpy" if type(self.run).__name__ == "NumpyRun" else "gpu"
        test = PileScreen(self.app, alone(self.run, engine=engine), back=self)
        test.encl = self.encl * self.run.n + self.follow  # si parte dalla rete seguita
        self.app.show(test)

    # --- disegno --------------------------------------------------------------

    def _arena_rect(self) -> pygame.Rect:
        return pygame.Rect(16, 44, 500, 500)

    def draw(self, s) -> None:
        s.fill(BG)
        self.targets = []
        self._top(s)
        self._arena(s)
        self._stats(s)
        self._brain(s)
        if self.hover is not None:
            self._tooltip(s)
        if self.show_help:
            self._help(s)

    def _top(self, s):
        app, r, w = self.app, self.run, self.view[0]
        les = getattr(getattr(r, "sizes", None), "lesions", None) or getattr(getattr(r, "brain", None), "les", None)
        les = f", senza {','.join(sorted(les))}" if les else ""
        head = (f"Arena delle pile — {r.agent}{les}, fase {r.phase} — recinto {self.encl + 1}/{r.B} "
                f"— essere {self.follow + 1}/{r.n}")
        if self.teach[0] and self.follow == 0:
            head += " (il maestro)"
        if self.testing:
            head = f"Prova delle reti una a una — {r.agent}{les} — rete {self.encl + 1}/{r.B}, da sola nel suo recinto"
        s.blit(app.font.render(head, True, PLAYER), (16, 12))
        speed = "turbo" if self.turbo else f"x{SPEEDS[self.speed_i]}"
        info = (f"t {w['t'] / 60:5.1f} min   {speed} ({self.rate:,.0f} s di mondo/s)"
                + ("   PAUSA" if self.paused else "") + "   H aiuto")
        s.blit(app.small.render(info, True, DIM), (s.get_width() - 470, 16))

    def _arena(self, s):
        cfg, w, raw = self.cfg, self.view[0], self.view[2]
        box = self._arena_rect()
        sc = box.w / cfg.size
        P = lambda p: (int(box.x + p[0] * sc), int(box.y + p[1] * sc))
        pygame.draw.rect(s, (22, 25, 33), box)
        m, h = cfg.zone_margin, cfg.zone_half
        for z, c in enumerate(((m, m), (cfg.size - m, m), (m, cfg.size - m), (cfg.size - m, cfg.size - m))):
            r = pygame.Rect(0, 0, int(2 * h * sc), int(2 * h * sc))
            r.center = P(c)
            pygame.draw.rect(s, (52, 62, 40) if z == int(w["zone"]) else ZONE_C, r, 0 if z == int(w["zone"]) else 1)
        s.set_clip(box)  # la vista e le scie restano nel recinto
        good_now = battery_good(cfg, np.asarray(w["bcolor"])[None], np.asarray(w["bbig"])[None],
                                np.atleast_1d(np.asarray(w["good"])))[0]
        for k in range(cfg.batteries):
            if not w["bon"][k]:
                continue
            col = RED_C if int(w["bcolor"][k]) == 0 else YELLOW_C
            c = P(w["bpos"][k])
            rad = max(3, int(cfg.battery_radius * sc * (1.4 if int(w["bbig"][k]) else 0.8)))  # grande o piccola
            pygame.draw.circle(s, col, c, rad)
            if bool(good_now[k]):
                pygame.draw.circle(s, (230, 240, 230), c, rad + 2, 1)
        # La vista e i raggi dell'essere seguito.
        i = self.follow
        me, hd = w["pos"][i], float(w["heading"][i])
        # Il cono di vista è semitrasparente: si vede ciò che c'è dentro.
        veil = pygame.Surface(s.get_size(), pygame.SRCALPHA)
        for k_, (a0, a1) in enumerate(zip(SECTORS[:-1], SECTORS[1:])):
            pts = [P(me)] + [P(me + cfg.eye_range * np.array([np.cos(hd + a), np.sin(hd + a)]))
                             for a in np.linspace(a0, a1, 6)]
            pygame.draw.polygon(veil, (120, 150, 220, 22 if k_ != 1 else 34), pts)
            pygame.draw.polygon(veil, (140, 170, 240, 90), pts, 1)
        s.blit(veil, (0, 0))
        for k in range(cfg.batteries):  # le pile di nuovo sopra il velo, nitide
            if w["bon"][k]:
                pygame.draw.circle(s, RED_C if int(w["bcolor"][k]) == 0 else YELLOW_C, P(w["bpos"][k]),
                                   max(3, int(cfg.battery_radius * sc * (1.4 if int(w["bbig"][k]) else 0.8))))
        for r_, a in enumerate(RAYS):
            end = me + (cfg.ray_range + cfg.radius) * np.array([np.cos(hd + a), np.sin(hd + a)])
            pygame.draw.line(s, lerp((60, 66, 84), (255, 150, 60), raw[i, 9 + r_]), P(me), P(end), 1)
        if len(self.trail) > 1:
            pygame.draw.lines(s, (70, 80, 60), False, [P(p) for p in self.trail], 1)
        for j in range(self.run.n):
            c = P(w["pos"][j])
            rad = max(4, int(cfg.radius * sc))
            teacher = self.teach[0] and j == 0
            col = ME_C if j == i else (TEACHER_C if teacher else BEING_C)
            if w["charging"][j] > 0:
                col = lerp(col, (255, 255, 255), 0.5)
            ghost = bool(np.asarray(w.get("ghost", np.zeros(self.run.n, bool)))[j])
            if ghost:  # il fantasma: solo il contorno, tratteggiato
                for k in range(0, 360, 40):
                    a0 = np.radians(k)
                    pygame.draw.arc(s, col, (c[0] - rad, c[1] - rad, 2 * rad, 2 * rad), a0, a0 + np.radians(22), 2)
            else:
                pygame.draw.circle(s, col, c, rad, 0 if j == i else 2)
            if teacher:  # la corona
                cx, cy = c[0], c[1] - rad - 4
                pygame.draw.polygon(s, TEACHER_C, [(cx - 7, cy + 3), (cx - 7, cy - 4), (cx - 3, cy), (cx, cy - 6),
                                                   (cx + 3, cy), (cx + 7, cy - 4), (cx + 7, cy + 3)])
            tip = w["pos"][j] + cfg.radius * 1.4 * np.array([np.cos(w["heading"][j]), np.sin(w["heading"][j])])
            pygame.draw.circle(s, BLUE_C, P(tip), 3)
        if bool(self.view[1].get("bite", False)):  # la bocca aperta: morde
            tip = me + cfg.radius * 1.5 * np.array([np.cos(hd), np.sin(hd)])
            pygame.draw.circle(s, (240, 120, 100), P(tip), 4)
        # L'oggetto su cui l'essere seguito è concentrato: un anello e una linea.
        has, _, _, _, t = self._attended(w, self.view[1])
        if has:
            K = cfg.batteries
            obj = w["bpos"][t] if t < K else w["pos"][t - K]
            pygame.draw.line(s, ME_C, P(me), P(obj), 1)
            pygame.draw.circle(s, ME_C, P(obj), max(8, int((cfg.battery_radius + 6) * sc)), 2)
        s.set_clip(None)
        for w_, (a0, a1) in enumerate((((box.left, box.top), (box.right, box.top)),
                                       ((box.right, box.top), (box.right, box.bottom)),
                                       ((box.left, box.bottom), (box.right, box.bottom)),
                                       ((box.left, box.top), (box.left, box.bottom)))):
            pygame.draw.line(s, WALL_C[w_], a0, a1, 4)  # i muri colorati: nord, est, sud, ovest
        self.targets.append((box, ("arena",)))

    def _stats(self, s):
        app, r, w = self.app, self.run, self.view[0]
        i = self.follow
        x, y = 16, 552
        if self.cfg.good_by == "grandezza":
            good = "grande" if int(w["good"]) == 1 else "piccola"
        else:
            good = "rossa" if int(w["good"]) == 0 else "gialla"
        line = lambda txt, col=TEXT: (s.blit(app.small.render(txt, True, col), (x, line.y)), setattr(line, "y", line.y + 16))
        line.y = y
        line(f"ora la pila buona è la {good} (cerchio bianco); zona attiva evidenziata", TEXT)
        line(f"inversione tra {w['next_inv'] - w['t']:4.0f} s   cambio di zona tra {w['next_zone'] - w['t']:4.0f} s",
             DIM)
        e = float(w["energy"][i]) / self.cfg.energy_max
        pygame.draw.rect(s, BORDER, (x, line.y + 2, 200, 10), 1)
        pygame.draw.rect(s, lerp(NEG, POS, e), (x + 1, line.y + 3, int(198 * e), 8))
        s.blit(app.small.render(f"energia {e:4.0%}   età {w['age'][i]:4.0f} s", True, TEXT), (x + 210, line.y))
        line.y += 18
        good_s, bad_s = r.flat("good_taken"), r.flat("bad_taken")
        last = int(60.0 / self.cfg.dt)
        g, b = good_s[-last:].sum(), bad_s[-last:].sum()
        tot = good_s.sum() + bad_s.sum()
        line(f"ultimo minuto, tutti i recinti: {g + b:.0f} pile prese, {g / max(g + b, 1):.0%} buone", TEXT)
        line(f"dall'inizio: {good_s.sum() / max(tot, 1):.0%} buone; morti di fame {r.flat('starved').sum():.0f}"
             + (f"; accoppiamenti {r.flat('matings').sum():.0f}" if r.phase == "B" else ""), DIM)
        if self.testing:
            from ..evo.pile_saves import per_network

            nets = per_network(r)
            me = nets[self.encl]
            done = sorted([x["scelta"] for x in nets if x["scelta"] is not None], reverse=True)
            rank = (done.index(me["scelta"]) + 1) if me["scelta"] is not None else None
            line(f"questa rete: {me['buone']} buone, {me['cattive']} cattive, morta {me['morti']} volte"
                 + (f", scelta {me['scelta']:.0%} ({rank}ª su {len(done)})" if rank else ""), PLAYER)
            line("PagSu/Giù: rete precedente / successiva   +/-: velocità   Esc: torna alla corsa", DIM)
        else:
            line("clic/Tab: segui  PagSu/Giù: recinto  P: reti una a una  Ctrl+S: salva", DIM)
        if self.teach[0]:
            kind = getattr(self.run, "teacher_kind", "fantasma")
            line(f"maestro {kind}: con la corona" + (", tratteggiato" if kind == "fantasma" else "")
                 + "; M lo toglie", TEACHER_C)
            line(f"quanto imparano ancora da lui: {self.teach[1]:.3f}", TEACHER_C)
            self._teach_curve(s, x + 420, line.y - 32)
        elif not self.testing and r.agent == "sezioni":
            line("M: metti un maestro in ogni recinto", DIM)
        if self.message:
            line(self.message, TARGET)

    def _teach_curve(self, s, x, y):
        """La curva di quanto si impara ancora dal maestro: quando si appiattisce,
        da lui non c'è più molto da imparare."""
        h = self.teach_hist
        r = pygame.Rect(x, y, 90, 30)
        pygame.draw.rect(s, BORDER, r, 1)
        if len(h) < 2:
            return
        v = np.array([e for _, e in h])
        top = max(float(v.max()), 1e-6)
        pts = [(r.x + 1 + int(88 * k / (len(v) - 1)), r.bottom - 2 - int(26 * e / top)) for k, e in enumerate(v)]
        pygame.draw.lines(s, TEACHER_C, False, pts, 1)

    # --- il cervello ----------------------------------------------------------

    def _brain(self, s):
        app = self.app
        area = pygame.Rect(540, 44, s.get_width() - 556, s.get_height() - 60)
        pygame.draw.rect(s, (18, 20, 28), area)
        pygame.draw.rect(s, BORDER, area, 1)
        world, br, raw = self.view
        i = self.follow
        senses = raw[i]
        fast = np.clip(senses[list(FAST)] - br["a_slow"], 0.0, 1.0) if "a_slow" in br else np.zeros(len(FAST))
        a = np.concatenate([senses, fast, self._fovea(world, br)])
        agent = self.run.agent
        title = {"sezioni": "Il cervello a sezioni dell'essere seguito",
                 "attore-critico": "Attore-critico: A -> E e V (niente sezioni in mezzo)",
                 "guidato": "Guidato a mano: una regola scritta, non un cervello"}[agent]
        s.blit(app.font.render(title, True, TEXT), (area.x + 10, area.y + 6))
        s.blit(app.small.render("passa il mouse su una sezione o un neurone per sapere cos'è", True, DIM),
               (area.x + 10, area.y + 26))
        top = area.y + 58
        # Le colonne, da sinistra a destra come scorre il segnale.
        xs = {"A": area.x + 18, "B": area.x + 250, "D": area.x + 345, "C": area.x + 450, "E": area.x + 640,
              "F": area.x + 775}
        if agent != "sezioni":
            xs["E"], xs["F"] = area.x + 420, area.x + 600
        br = dict(br)
        if "genome" in br:
            br["motor_w"] = np.asarray(br["genome"])[MOTOR].reshape(len(F_NEURONS), len(F_INPUTS))
        for k in ("f_in", "f_act"):
            if k in world and np.ndim(world[k]) == 2:
                br[k] = np.asarray(world[k])[i]
        A_pos = self._section_A(s, xs["A"], top, a)
        self._reflex_pos = A_pos[RAY.start + 1]
        E_pos = self._section_E(s, xs["E"], top + 40, br, agent)
        self._section_F(s, xs["F"], E_pos, br, agent)
        self._section_S(s, xs["E"], E_pos[-1][1] + 50, br)
        if agent == "sezioni":
            self._section_O(s, xs["C"] + 10, area.bottom - 150, br, world)
        if agent == "guidato":
            fov = N_SENSES + len(FAST)
            self._links(s, [(A_pos[fov + k], E_pos[0], 0.8, FIXED) for k in range(3)])
            s.blit(self.app.small.render("regola scritta a mano: guarda la pila buona (sa qual è), ci va; "
                                         "le altre le lascia; senza, va verso la zona (la sa)", True, DIM),
                   (self._arena_rect().right + 40, s.get_height() - 40))
            return
        if agent == "attore-critico":
            f = np.concatenate([a, [1.0]])
            self._weights_to_E(s, br["w_e"], [A_pos[k] for k in range(N_A)], f, E_pos)
            self._section_V(s, area.x + 420, area.bottom - 120, br, [E_pos[p] for p in range(N_PROGRAMS)])
            return
        self._fovea_pos = A_pos[N_A - len(FOVEA):]
        B_pos = self._section_B(s, xs["B"], top, br, A_pos, a)
        D_pos = self._section_D(s, xs["D"], top, br, self._fovea_pos)
        C_pos = self._section_C(s, xs["C"], top, br, B_pos, D_pos, E_pos)
        f = np.concatenate([br["state_b"], br["mem"], [1.0]])
        self._weights_to_E(s, br["w_e"], B_pos + D_pos, f, E_pos, limit=10)
        self._section_V(s, xs["D"] - 50, area.bottom - 150, br,
                        [C_pos[0], E_pos[0], D_pos[len(D_pos) // 2]])
        self._legend(s, area)

    LABELS = {"A": "A sensi", "B": "B cosa c'è", "D": "D appena", "C": "C/L ricordi", "E": "E scelta",
              "F": "F motore", "V": "V critico (δ)", "U": "U", "S": "S attenzione", "O": "O valore"}

    def _objects(self, world):
        """Gli oggetti in vista nel recinto guardato (1, n, M)."""
        cfg = self.cfg
        bcolor, bbig = np.asarray(world["bcolor"])[None], np.asarray(world["bbig"])[None]
        good = battery_good(cfg, bcolor, bbig, np.atleast_1d(np.asarray(world["good"])))
        ghost = world.get("ghost")
        extra = {} if ghost is None else {"ghost": np.asarray(ghost, bool)[None],
                                          "ghost_hide": np.asarray(world["ghost_hide"])[None]}
        return objects_of(world["pos"][None], world["heading"][None], np.asarray(world["speed"])[None],
                          world["bpos"][None], bcolor, bbig, np.asarray(world["bon"], bool)[None], good, cfg, **extra)

    def _attended(self, world, br):
        """(c'è, caratteristiche, vicinanza, angolo, indice) dell'oggetto attento dall'essere seguito."""
        t = int(br.get("att_target", -1))
        if t < 0:
            return False, np.zeros(len(FEATURES)), 0.0, 0.0, -1
        o = self._objects(world)
        i = self.follow
        return True, o["feat"][0, i, t], float(o["near"][0, i, t]), float(o["ang"][0, i, t]), t

    def _fovea(self, world, br):
        has, feat, near, ang, _ = self._attended(world, br)
        return fovea_of(feat[None], np.array([near]), np.array([ang]), np.array([has]),
                        np.array([float(br.get("att_new", 0.0))]))[0]

    def _section_S(self, s, x, y, br):
        """S: la mappa di priorità (9 posizioni, da sinistra a destra) e, sotto
        ogni posizione, le quattro parti che somma (innata, valore, importanza,
        incertezza)."""
        app = self.app
        self._label(s, "S", x - 4, y - 22)
        r = pygame.Rect(x, y, 230, 92)
        pygame.draw.rect(s, (30, 34, 46), r)
        pygame.draw.rect(s, BORDER, r, 1)
        prob = np.asarray(br.get("att_prob", np.zeros(MAP)))
        has = np.asarray(br.get("att_has", np.zeros(MAP, bool)))
        parts = np.asarray(br.get("att_parts", np.zeros((MAP, 4))))
        place = int(np.argmax(prob)) if prob.sum() > 0 else -1
        colours = ((150, 150, 165), None, (240, 160, 70), DELTA_C)
        for k in range(MAP):
            c = (r.x + 14 + k * 24, r.y + 13)
            v = float(prob[k])
            pygame.draw.circle(s, lerp((40, 44, 58), (250, 230, 140), min(1.0, 2 * v)) if has[k] else (26, 28, 36), c, 8)
            pygame.draw.circle(s, PLAYER if k == place and v > 0 else BORDER, c, 8, 2 if k == place else 1)
            if not has[k]:
                continue
            base = r.y + 50  # le quattro parti: barrette verso l'alto (attira) o il basso (respinge)
            for j in range(4):
                val = float(np.clip(parts[k, j] / 2.0, -1.0, 1.0))
                col = (POS if val > 0 else NEG) if j == 1 else colours[j]
                h = int(20 * abs(val))
                pygame.draw.rect(s, col, (c[0] - 8 + 4 * j, base - h if val > 0 else base, 3, max(h, 1)))
        pygame.draw.line(s, BORDER, (r.x + 4, r.y + 50), (r.right - 4, r.y + 50), 1)
        s.blit(app.small.render("barre: innata, valore,", True, DIM), (r.x + 6, r.y + 56))
        s.blit(app.small.render("importanza, incertezza", True, DIM), (r.x + 6, r.y + 72))
        self.targets.append((r, ("section", "S")))

    def _section_O(self, s, x, y, br, world):
        """O: per ogni caratteristica, piacere (verde), dolore (rosso),
        incertezza (viola); in alto i due neuroni di uscita."""
        app = self.app
        self._label(s, "O", x - 4, y - 22)
        r = pygame.Rect(x, y, 170, 112)
        pygame.draw.rect(s, (30, 34, 46), r)
        pygame.draw.rect(s, BORDER, r, 1)
        n_f = len(FEATURES)
        pl = np.asarray(br.get("o_pl", np.zeros(n_f + 1)))
        pa = np.asarray(br.get("o_pa", np.zeros(n_f + 1)))
        un = np.asarray(br.get("o_assoc", np.zeros(n_f + 1)))
        out = np.asarray(br.get("o_out", np.zeros(2)))
        s.blit(app.small.render(f"guardo {out[0]:+.2f} preso {out[1]:+.2f}", True,
                                POS if out[0] > 0 else (NEG if out[0] < 0 else TEXT)), (r.x + 5, r.y + 2))
        for k, name in enumerate(FEATURES + ("sfondo",)):
            yy = r.y + 19 + k * 13
            s.blit(app.small.render(name[:8], True, TEXT if k < n_f else DIM), (r.x + 5, yy - 3))
            for j, (v, col) in enumerate(((pl[k], POS), (pa[k], NEG))):
                w = int(30 * float(np.clip(abs(v), 0.0, 1.0)))
                xx = r.x + 72 + 34 * j
                pygame.draw.rect(s, BORDER, (xx, yy + 1, 30, 7), 1)
                pygame.draw.rect(s, col if v >= 0 else lerp(col, (40, 40, 40), 0.6), (xx, yy + 1, w, 7))
            pygame.draw.rect(s, DELTA_C, (r.x + 142, yy + 1, int(24 * float(np.clip(un[k], 0.0, 1.0))), 7))
        self.targets.append((r, ("section", "O")))

    def _links(self, s, links):
        """Collegamenti semplici: (da, a, forza 0-1, colore)."""
        for p0, p1, v, col in links:
            pygame.draw.line(s, lerp((50, 54, 70), col, v), p0, p1, 2)

    def _label(self, s, key, x, y, w=0):
        name = self.LABELS[key]
        r = pygame.Rect(x - 4, y - 2, self.app.small.size(name)[0] + 8, 18)
        pygame.draw.rect(s, (30, 34, 46), r)
        s.blit(self.app.small.render(name, True, PLAYER), (x, y))
        self.targets.append((r, ("section", key)))

    def _section_A(self, s, x, top, a):
        """A compatta: gli occhi come griglia (colore × sinistra, centro, destra),
        poi raggi, urti, corpo e la fovea. Restituisce il punto di ogni canale."""
        app = self.app
        self._label(s, "A", x, top - 20)
        pos = [(0, 0)] * N_SENSES
        fast_of = {c: k for k, c in enumerate(FAST)}
        cell, row = 16, 18
        gx = x + 92
        for k, h in enumerate(("sx", "c", "dx")):
            s.blit(app.small.render(h, True, DIM), (gx + k * (cell + 6) + 2, top - 4))
        y = top + 12

        def grid_row(name, col, first):
            nonlocal y
            s.blit(app.small.render(name, True, lerp(DIM, col, 0.5)), (x, y))
            for j in range(3):
                k = first + j
                v = float(a[k])
                r = pygame.Rect(gx + j * (cell + 6), y, cell, cell - 2)
                pygame.draw.rect(s, lerp((34, 38, 50), col, v), r)
                pygame.draw.rect(s, (60, 64, 80), r, 1)
                if k in fast_of:
                    fv = float(a[N_SENSES + fast_of[k]])
                    pygame.draw.rect(s, (235, 235, 245), (r.x, r.bottom + 1, int(r.w * fv), 2))
                pos[k] = (gx + 3 * (cell + 6) + 6, y + 7)
                self.targets.append((r, ("A", k)))
            y += row

        for e, (name, col) in enumerate(EYE_ROWS):
            grid_row(name, col, 3 * e)
        y += 4
        grid_row("distanza", (255, 150, 60), RAY.start)
        grid_row("urto", (255, 110, 90), BUMP.start)
        y += 4
        for k, name, col in ((ENERGY, "energia", TEXT), (WATCHED, "mi guardano", BLUE_C), (TASTE, "gusto", POS),
                             (PAIN, "dolore", NEG)):
            v = float(a[k])
            s.blit(app.small.render(name, True, lerp(DIM, col, 0.4 + 0.6 * v)), (x, y))
            bar = pygame.Rect(gx, y + 2, 3 * (cell + 6) - 6, 9)
            pygame.draw.rect(s, (40, 44, 58), bar)
            pygame.draw.rect(s, lerp((60, 64, 80), col, 0.3 + 0.7 * v), (bar.x, bar.y, int(bar.w * v), bar.h))
            if k in fast_of:
                fv = float(a[N_SENSES + fast_of[k]])
                pygame.draw.rect(s, (235, 235, 245), (bar.x, bar.bottom + 1, int(bar.w * fv), 2))
            pos[k] = (gx + 3 * (cell + 6) + 6, y + 7)
            self.targets.append((pygame.Rect(x, y - 2, 200, row - 2), ("A", k)))
            y += row
        y += 6
        fov = N_SENSES + len(FAST)
        fovea_pos = []
        colours = [RED_C, YELLOW_C, BLUE_C, TEXT, TEXT, BEING_C] + [ME_C] * 4
        for f in range(len(FOVEA)):
            v = float(a[fov + f])
            s.blit(app.small.render(FOVEA_SHORT[f], True, lerp(DIM, colours[f], 0.4 + 0.6 * v)), (x, y))
            bar = pygame.Rect(gx, y + 2, 3 * (cell + 6) - 6, 9)
            pygame.draw.rect(s, (40, 44, 58), bar)
            pygame.draw.rect(s, lerp((60, 64, 80), colours[f], 0.3 + 0.7 * v), (bar.x, bar.y, int(bar.w * v), bar.h))
            fovea_pos.append((gx + 3 * (cell + 6) + 6, y + 7))
            self.targets.append((pygame.Rect(x, y - 2, 200, row - 2), ("A", fov + f)))
            y += row
        for p in pos + fovea_pos:
            pygame.draw.circle(s, (70, 76, 96), p, 2)
        # I canali "cosa è cambiato" hanno il punto del loro senso (tacca chiara sotto).
        return pos + [pos[c] for c in FAST] + fovea_pos

    def _section_B(self, s, x, top, br, A_pos, a):
        self._label(s, "B", x - 10, top - 20, 80)
        st = br["state_b"]
        nb = len(st)
        pos = []
        if "proto" in br and nb == br["proto"].shape[0]:
            proto = br["proto"]
            for j in range(nb):
                y = top + 10 + j * (480 / nb)
                pos.append((x + 20, int(y)))
            # A -> B: a cosa risponde ogni neurone (i due canali più alti del suo prototipo).
            for j in range(nb):
                strong = np.argsort(-proto[j][:N_SENSES])[:2]
                hot = st[j] > 0.3 or self.hover == ("B", j)
                for k in strong:
                    if proto[j][k] < 0.25:
                        continue
                    col = lerp((40, 44, 58), PLAYER if hot else (90, 96, 120), float(proto[j][k]))
                    pygame.draw.line(s, col, A_pos[k], pos[j], 2 if hot else 1)
        else:  # senza B (lesione): E e gli altri leggono direttamente A
            s.blit(self.app.small.render("tolto: si legge A", True, DIM), (x - 10, top + 10))
            return [A_pos[k] for k in range(N_A)]
        win = int(st.argmax())
        for j, p in enumerate(pos):
            v = float(st[j])
            pygame.draw.circle(s, lerp((40, 44, 58), (250, 230, 140), v), p, 9)
            pygame.draw.circle(s, PLAYER if j == win and v > 0.2 else BORDER, p, 9, 2 if j == win else 1)
            self.targets.append((pygame.Rect(p[0] - 10, p[1] - 10, 20, 20), ("B", j)))
        return pos

    def _section_D(self, s, x, top, br, fovea_pos):
        """D: l'ultimo oggetto guardato, una copia della fovea che resta."""
        self._label(s, "D", x - 10, top - 20, 80)
        mem = br["mem"]
        pos = []
        for j, fp in enumerate(fovea_pos):
            p = (x + 20, fp[1])
            pygame.draw.line(s, (60, 66, 84), fp, p, 1)  # fovea -> D: copia (fissa, quando il cancello si apre)
            v = float(mem[j]) if j < len(mem) else 0.0
            pygame.draw.circle(s, lerp((40, 44, 58), (140, 210, 250), v), p, 7)
            pygame.draw.circle(s, BORDER, p, 7, 1)
            pos.append(p)
            self.targets.append((pygame.Rect(p[0] - 8, p[1] - 8, 16, 16), ("D", j)))
        return pos

    def _section_C(self, s, x, top, br, B_pos, D_pos, E_pos):
        self._label(s, "C", x - 10, top - 20, 130)
        code, u_c, u_l, c_in = br["code"], br["u_c"], br["u_l"], br["c_in"]
        n = len(code)
        cols = 8
        pos = []
        for j in range(n):
            p = (x + 14 + (j % cols) * 18, top + 60 + (j // cols) * 18)
            pos.append(p)
        src = self._fovea_pos + D_pos + B_pos  # l'ordine degli ingressi di C: fovea, D, B (contesto)
        lit = np.flatnonzero(code > 0)
        for j in lit:  # ingressi fissi e casuali dei neuroni accesi
            for k in np.flatnonzero(c_in[j] > 0):
                if k < len(src):
                    pygame.draw.line(s, FIXED, src[k], pos[j], 1)
        # Uscite verso E dei neuroni accesi: solo le più forti, C (spesse) e L (sottili).
        votes = [(abs(w), w, j, p_, width) for j in lit for p_ in range(N_HEADS)
                 for w, width in ((u_c[p_, j], 2), (u_l[p_, j], 1)) if abs(w) > 0.02]
        for _, w, j, p_, width in sorted(votes, reverse=True)[:16]:
            col = lerp((50, 54, 70), POS if w > 0 else NEG, min(abs(w) * 3, 1.0))
            pygame.draw.line(s, col, pos[j], E_pos[p_], width)
        for j, p in enumerate(pos):
            on = code[j] > 0
            pygame.draw.circle(s, (250, 230, 140) if on else (44, 48, 62), p, 6 if on else 4)
            self.targets.append((pygame.Rect(p[0] - 8, p[1] - 8, 16, 16), ("C", j)))
        return pos

    def _weights_to_E(self, s, w_e, src, f, E_pos, limit=14):
        """I pesi imparati verso E più forti (contano di più quelli con l'ingresso
        acceso adesso: linea spessa)."""
        w = w_e[:, : len(src)]
        score = np.abs(w) * (0.2 + f[None, : len(src)])
        top = float(np.abs(w).max()) if w.size else 0.0
        if top < 1e-6:
            x, y = E_pos[0][0] - 190, E_pos[-1][1] + 30
            s.blit(self.app.small.render("verso E: ancora niente di imparato", True, DIM), (x, y))
            s.blit(self.app.small.render("(nessuna sorpresa finora)", True, DIM), (x, y + 16))
            return
        for idx in np.argsort(-score.ravel())[:limit]:
            p_, k = divmod(int(idx), len(src))
            v = w[p_, k]
            if abs(v) < 0.1 * top:
                continue
            col = lerp((50, 54, 70), POS if v > 0 else NEG, abs(v) / top)
            pygame.draw.line(s, col, src[k], E_pos[p_], 2 if f[k] > 0.2 else 1)

    def _section_E(self, s, x, top, br, agent):
        app = self.app
        self._label(s, "E", x - 4, top - 30, 110)
        pi = np.append(br.get("pi", np.full(N_PROGRAMS, 1.0 / N_PROGRAMS)), float(br.get("p_bite", 0.0)))
        prog = int(br.get("program", -1)) if "program" in br else -1
        biting = bool(br.get("bite", False))
        pos = []
        for p_ in range(N_HEADS):
            y = top + 20 + p_ * 64 + (14 if p_ == BITE else 0)
            r = pygame.Rect(x, y - 16, 110, 34)
            on = p_ == prog or (p_ == BITE and biting)
            pygame.draw.rect(s, (60, 54, 30) if on else (30, 34, 46), r)
            pygame.draw.rect(s, PLAYER if on else BORDER, r, 2 if on else 1)
            s.blit(app.small.render(PROG_SHORT[p_], True, TEXT if on else DIM), (r.x + 6, r.y + 3))
            v = float(pi[p_])
            pygame.draw.rect(s, (44, 48, 62), (r.x + 6, r.y + 20, 98, 7))
            pygame.draw.rect(s, PLAYER if on else (150, 150, 170), (r.x + 6, r.y + 20, int(98 * v), 7))
            pos.append((r.x, y))
            self.targets.append((r, ("E", p_)))
        return pos

    # Da quale programma (indice in E) viene ogni ingresso di F; None = riflesso o costante.
    F_SOURCE = (0, 0, 0, 0, 1, 1, 1, 3, 3, 3, None, None, None, None)

    def _section_F(self, s, x, E_pos, br, agent):
        """F: 4 neuroni motori; le linee dai programmi di E (e dal riflesso dei
        muri) pesano quanto il loro contributo adesso; il morso va alla bocca."""
        app = self.app
        self._label(s, "F", x - 4, E_pos[0][1] - 46)
        W = br.get("motor_w")
        x_in, act = br.get("f_in"), br.get("f_act")
        if W is None or x_in is None:
            return
        pos = []
        for o in range(len(F_NEURONS)):
            p = (x + 14, E_pos[0][1] + o * 56)
            pos.append(p)
        contrib = W * x_in[None, :]
        for o, p in enumerate(pos):
            for i, src in enumerate(self.F_SOURCE):
                v = contrib[o, i]
                if abs(v) < 0.05:
                    continue
                start = (E_pos[src][0] + 110, E_pos[src][1]) if src is not None else self._reflex_pos
                if src == 3 and int(br.get("program", 3)) == 2:
                    start = (E_pos[2][0] + 110, E_pos[2][1])  # "lascialo" vaga come "esplora"
                col = lerp((50, 54, 70), POS if v > 0 else NEG, min(abs(v), 1.0))
                pygame.draw.line(s, col, start, p, 2)
        for o, p in enumerate(pos):
            v = float(act[o])
            pygame.draw.circle(s, lerp((40, 44, 58), (250, 230, 140), v), p, 9)
            pygame.draw.circle(s, BORDER, p, 9, 1)
            s.blit(app.small.render(F_NEURONS[o].replace("gira a sinistra", "gira sx").replace("gira a destra", "gira dx"),
                                    True, TEXT if v > 0.1 else DIM),
                   (p[0] + 14, p[1] - 7))
            self.targets.append((pygame.Rect(p[0] - 10, p[1] - 10, 80, 20), ("F", o)))
        bx, by = E_pos[BITE]
        mouth = (x + 14, by)
        on = bool(br.get("bite", False))
        pygame.draw.line(s, PLAYER if on else FIXED, (bx + 110, by), mouth, 2 if on else 1)
        pygame.draw.circle(s, (240, 120, 100) if on else (40, 44, 58), mouth, 9)
        pygame.draw.circle(s, BORDER, mouth, 9, 1)
        s.blit(app.small.render("bocca", True, TEXT if on else DIM), (mouth[0] + 14, mouth[1] - 7))

    def _section_V(self, s, x, y, br, targets):
        app = self.app
        self._label(s, "V", x - 4, y - 22, 130)
        d = float(br.get("delta", 0.0))
        r = pygame.Rect(x, y, 130, 40)
        pygame.draw.rect(s, (30, 34, 46), r)
        pygame.draw.rect(s, BORDER, r, 1)
        mid = r.x + 65
        span = int(np.clip(d * 200.0, -60, 60))
        col = POS if d > 0 else NEG
        pygame.draw.rect(s, col, (min(mid, mid + span), r.y + 8, abs(span), 10))
        pygame.draw.line(s, TEXT, (mid, r.y + 4), (mid, r.y + 22), 1)
        s.blit(app.small.render(f"δ {d:+.3f}", True, TEXT), (r.x + 6, r.y + 24))
        self.targets.append((r, ("section", "V")))
        if abs(d) > 0.01:  # la sorpresa arriva a chi impara
            for t in targets:
                pygame.draw.line(s, lerp((60, 50, 80), DELTA_C, min(abs(d) * 20, 1.0)), (mid, r.y), t, 1)
        if "unc" in br:
            u = float(br["unc"]) / 3.0
            ru = pygame.Rect(r.x, r.bottom + 6, 130, 34)
            pygame.draw.rect(s, (30, 34, 46), ru)
            pygame.draw.rect(s, BORDER, ru, 1)
            pygame.draw.rect(s, lerp((80, 80, 120), DELTA_C, u), (ru.x + 6, ru.y + 6, int(118 * u), 8))
            s.blit(app.small.render(f"U {float(br['unc']):.2f}", True, TEXT), (ru.x + 6, ru.y + 17))
            self.targets.append((ru, ("section", "U")))

    def _legend(self, s, area):
        app = self.app
        y = area.bottom - 22
        x = area.x + 10
        for col, txt in ((POS, "rafforza"), (NEG, "indebolisce"), (FIXED, "fisso, non impara"),
                         (DELTA_C, "sorpresa δ"), ((250, 230, 140), "acceso")):
            pygame.draw.line(s, col, (x, y + 7), (x + 22, y + 7), 3)
            s.blit(app.small.render(txt, True, DIM), (x + 28, y))
            x += 40 + app.small.size(txt)[0]
        s.blit(app.small.render("da C verso E: linee spesse = C (veloce), sottili = L (lenta)", True, DIM),
               (area.x + 10, y - 18))

    # --- le spiegazioni sotto il mouse ----------------------------------------

    def _explain(self) -> list[str]:
        key = self.hover
        world, br, raw = self.view
        i = self.follow
        kind = key[0]
        if kind == "section":
            return list(SECTION_HELP[key[1]])
        if kind == "arena":
            return ["Il recinto", "Pile rosse e gialle: quella col cerchio bianco è la buona, adesso. Le pile "
                    "compaiono nella zona evidenziata. Giallo pieno: l'essere seguito, con la sua vista (tre "
                    "settori) e i tre raggi per i muri; il puntino blu è la luce sul muso. Clic: segui un altro."]
        if kind == "A":
            k = key[1]
            if k < N_SENSES:
                return [f"A · {SHORT[k]}", SECTION_HELP["A"][1], f"valore adesso: {raw[i, k]:.2f}"]
            f = k - (N_A - len(FOVEA))
            return [f"A · fovea: {FOVEA[f]}", "La fovea descrive solo l'oggetto su cui l'essere è concentrato "
                    "(anello azzurro nel recinto).", f"valore adesso: {self._fovea(world, br)[f]:.2f}"]
        if kind == "B":
            j = key[1]
            proto = br.get("proto")
            out = [f"B · neurone {j + 1}", "Si è specializzato da solo: risponde quando l'ingresso somiglia al suo "
                   "prototipo. I canali più alti del prototipo:"]
            if proto is not None:
                top = np.argsort(-proto[j][:N_SENSES])[:4]
                out.append(", ".join(f"{SHORT[k]} {proto[j][k]:.2f}" for k in top))
            out.append(f"attività adesso: {br['state_b'][j]:.2f}")
            counts = br.get("counts")
            if counts is not None and counts[j].sum() > 0:
                c = counts[j] / counts[j].sum()
                out.append("quando vince, la vista è dominata da: "
                           + ", ".join(f"{n} {v:.0%}" for n, v in zip(("rosso", "giallo", "blu", "niente"), c)))
            return out
        if kind == "D":
            j = key[1]
            return [f"D · ricordo: {FOVEA[j]}", SECTION_HELP["D"][1], f"tenuto adesso: {br['mem'][j]:.2f}"]
        if kind == "C":
            j = key[1]
            votes = ", ".join(f"{PROG_SHORT[p]} {br['u_c'][p, j]:+.2f}/{br['u_l'][p, j]:+.2f}"
                              for p in range(N_HEADS))
            return [f"C · neurone {j + 1}", "Ingressi fissi da 4 canali a caso tra fovea, D e B. Voti verso E (C veloce / "
                    "L lenta):", votes, "acceso adesso" if br["code"][j] > 0 else "spento adesso"]
        if kind == "E":
            p_ = key[1]
            if p_ == BITE:
                return ["E · mordi", "Se mordere: si mangia solo mordendo una pila davanti al muso, e mordere costa "
                        "un po' di energia. È una scelta che si impara: mordere questo oggetto?",
                        f"probabilità adesso: {float(br.get('p_bite', 0.0)):.0%}"]
            pi = br.get("pi", np.full(N_PROGRAMS, 0.25))
            return [f"E · {PROGRAMS[p_]}", SECTION_HELP["E"][1], f"probabilità adesso: {float(pi[p_]):.0%}"]
        if kind == "F":
            o = key[1]
            W = np.asarray(br["genome"])[MOTOR].reshape(len(F_NEURONS), len(F_INPUTS)) if "genome" in br else None
            x_in = np.asarray(world["f_in"])[i] if "f_in" in world and np.ndim(world["f_in"]) == 2 else br.get("f_in")
            out = [f"F · {F_NEURONS[o]}", SECTION_HELP["F"][1]]
            if W is not None and x_in is not None:
                c = W[o] * x_in
                top = [k for k in np.argsort(-np.abs(c)) if abs(c[k]) > 0.02][:4]
                out.append("adesso lo spingono: " + (", ".join(f"{F_INPUTS[k]} {c[k]:+.2f}" for k in top) or "niente"))
            return out
        return []

    def _tooltip(self, s):
        lines = self._explain()
        if not lines:
            return
        app = self.app
        mx, my = pygame.mouse.get_pos()
        width = 380
        body = [lines[0]]
        for t in lines[1:]:
            body += wrap_text(app.small, t, width - 16)
        h = 10 + 16 * len(body)
        x = min(mx + 16, s.get_width() - width - 4)
        y = min(my + 16, s.get_height() - h - 4)
        r = pygame.Rect(x, y, width, h)
        pygame.draw.rect(s, (12, 14, 20), r)
        pygame.draw.rect(s, PLAYER, r, 1)
        for k, t in enumerate(body):
            s.blit(app.small.render(t, True, PLAYER if k == 0 else TEXT), (x + 8, y + 5 + 16 * k))

    def _help(self, s):
        app = self.app
        r = pygame.Rect(200, 120, s.get_width() - 400, s.get_height() - 240)
        pygame.draw.rect(s, (12, 14, 20), r)
        pygame.draw.rect(s, PLAYER, r, 1)
        lines = ["Arena delle pile e cervello a sezioni", "",
                 "Ogni recinto ha pile rosse e gialle: una ricarica, l'altra fa male; ogni 5 minuti si scambiano.",
                 "Le pile compaiono in un angolo, che cambia ogni 200 s. Gli esseri devono imparare quale colore",
                 "inseguire, dentro la loro vita: nessuno lo sa alla nascita.", "",
                 "Il cervello si legge da sinistra a destra: A sensi, B cosa c'è, D cosa è appena successo,",
                 "C/L cosa ho imparato (in fretta / piano), E cosa faccio, F come lo faccio; V il critico, U",
                 "l'incertezza. Passa il mouse su qualsiasi cosa per la spiegazione.", "",
                 "Comandi: clic o Tab segue un essere, PagSu/PagGiù cambia recinto, +/- velocità, T turbo,",
                 "Spazio pausa, Ctrl+S salva, P prova le reti una a una (ognuna da sola nel suo recinto),",
                 "M mette o toglie il maestro (oro, con la corona: sa quali pile sono buone; gli altri imparano",
                 "il valore delle cose anche dalla sua vita),",
                 "H questo aiuto, Esc torna al menu (salvando, se ci sono novità)."]
        for k, t in enumerate(lines):
            s.blit((app.font if k == 0 else app.small).render(t, True, PLAYER if k == 0 else TEXT),
                   (r.x + 20, r.y + 20 + 18 * k))
