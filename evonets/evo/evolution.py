import numpy as np

from ..config import Config
from ..net.brain import Genome
from ..net.controllers import NeuralController
from ..sim.world import World
from .genetics import reproduce

NORMAL = "normale"
ONE_PARENT = "un solo genitore"
RESTART = "nessun genitore: ripartenza dagli ultimi"
RANDOM = "nessun genitore: reti casuali"


class Evolution:
    """Il ciclo delle generazioni.

    Una generazione vive nell'arena finché tutti sono morti o hanno completato
    il compito. Poi:
      - chi ha completato è genitore e, se non ha raggiunto `max_age`, passa
        identico alla generazione successiva;
      - i posti restanti vanno ai figli, con genitori sorteggiati alla pari;
      - con un solo genitore, una quota dei figli nasce con mutazioni più forti;
      - senza genitori si riparte dagli ultimi genitori noti (tutti figli, una
        quota con mutazioni più forti) o, se non ce ne sono mai stati, da reti
        casuali.
    Ogni generazione affronta un'arena nuova, uguale per tutti.
    """

    def __init__(self, cfg: Config, seed: int = 0, task=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.world = World(cfg, cfg.population, seed, task)
        self.ctrl = NeuralController(cfg, cfg.population, seed)
        self.age = np.zeros(cfg.population, dtype=int)  # generazioni già vissute
        self.generation = 0
        self.last_parents: Genome | None = None
        self.history: list[dict] = []

    @property
    def finished(self) -> bool:
        w = self.world
        return not (w.alive & ~w.done).any() or w.t >= self.cfg.life_max

    def step(self) -> None:
        self.world.step(self.ctrl.act(self.world))

    def run_generation(self) -> dict:
        while not self.finished:
            self.step()
        return self.next_generation()

    def next_generation(self) -> dict:
        """Selezione e ricambio. Restituisce le statistiche della generazione
        appena conclusa (aggiunte anche a `history`)."""
        cfg, world, brains = self.cfg, self.world, self.ctrl.brains
        n = cfg.population
        winners = np.flatnonzero(world.done)
        stats = {
            "generation": self.generation,
            "completed": len(winners),
            "dead": int((~world.alive).sum()),
            "duration": world.t,
            "neurons": float(brains.active.sum(axis=1).mean()),
            "links": float(brains.links.sum(axis=(1, 2)).mean()),
            "oldest": int(self.age.max()),
        }

        genome = brains.genome
        if len(winners):
            parents = genome.take(winners)
            self.last_parents = parents
            keep = winners[self.age[winners] + 1 < cfg.max_age]
            survivors, survivor_age = genome.take(keep), self.age[keep] + 1
            stats["event"] = NORMAL if len(winners) > 1 else ONE_PARENT
        else:
            parents = self.last_parents
            survivors, survivor_age = genome.take([]), np.zeros(0, dtype=int)
            stats["event"] = RESTART if parents is not None else RANDOM

        if parents is None:
            fresh = NeuralController(cfg, n, int(self.rng.integers(2**31)))
            new = fresh.brains.genome
        else:
            n_children = n - len(survivors)
            n_strong = int(n_children * cfg.rescue_share) if len(winners) <= 1 else 0
            children = [reproduce(parents, n_children - n_strong, cfg, self.rng)]
            if n_strong:
                children.append(reproduce(parents, n_strong, cfg, self.rng, cfg.rescue_strength))
            new = Genome.concat([survivors, *children])
        stats["survivors"] = len(survivors)

        brains.set_genome(new)
        self.age = np.concatenate([survivor_age, np.zeros(n - len(survivors), dtype=int)])
        world.seed += 1
        world.reset()
        self.generation += 1
        self.history.append(stats)
        return stats
