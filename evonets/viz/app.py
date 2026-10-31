"""L'interfaccia grafica: una finestra sola con più schermate.

- `Launcher` (launcher.py): il menu da cui si sceglie che corsa lanciare.
- `RunScreen` (generational/run.py): la corsa a generazioni.
- `Inspector` (inspector.py): reti e genomi a tutto schermo.
- `TestScreen` (testbench.py): una sola rete messa alla prova nell'arena.

Una corsa aperta continua in sottofondo anche mentre si guardano le altre
schermate (B la mette in pausa).
"""


import numpy as np
import pygame

from ..config import Config
from .launcher import Launcher
from .render import Renderer

BACKGROUND_BUDGET = 0.025  # s per fotogramma alla corsa mentre si guarda un'altra schermata


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
        self.active_run = None  # la corsa aperta, se c'è
        self.running = True

    def show(self, screen) -> None:
        self.current = screen

    def test(self, selection, back) -> None:
        """Apre il banco di prova per una rete: (genoma, nome, descrizione, etichetta)."""
        from .generational.testbench import TestScreen

        if selection is not None:
            genome, _, title, label = selection
            cfg = self.active_run.cfg if self.active_run else back.evo.cfg
            self.show(TestScreen(self, cfg, genome, [label], title, back))

    def test_survivors(self, evo, back) -> str:
        """Apre il banco di prova con tutti i sopravvissuti insieme. Restituisce
        un messaggio se non ce ne sono."""
        from .generational.testbench import TestScreen

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
