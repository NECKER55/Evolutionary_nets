import json
from pathlib import Path

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
      - chi ha completato è genitore e passa identico alla generazione
        successiva, dove dovrà completare di nuovo per restare (`max_age`, se
        impostato, aggiunge un limite di età);
      - i posti restanti vanno ai figli, con genitori sorteggiati alla pari;
      - con un solo genitore, una quota dei figli nasce con mutazioni più forti;
      - senza genitori si riparte dagli ultimi genitori noti (tutti figli, una
        quota con mutazioni più forti) o, se non ce ne sono mai stati, da reti
        casuali.
    Ogni generazione affronta un'arena nuova, uguale per tutti.

    `champions` conserva, per ogni generazione con almeno un vincitore, il
    genoma del vincitore più anziano (a parità di età, il nato prima): è la
    traccia di come cambia la rete. Non è scelto a caso.

    Ogni individuo ha un numero (`ids`) che non cambia finché vive: un
    sopravvissuto lo conserva di generazione in generazione, ogni figlio ne
    riceve uno nuovo.

    Selezione finale (`start_selection`): a evoluzione finita, i sopravvissuti
    rifanno la prova in arene nuove per un numero di volte stabilito, senza
    figli; chi fallisce è eliminato. Restano gli individui che hanno vinto ogni
    volta.
    """

    def __init__(self, cfg: Config, seed: int = 0, task=None):
        self.cfg = cfg
        self.seed = seed  # il seme con cui è nata la corsa
        self.rng = np.random.default_rng(seed)
        self.world = World(cfg, cfg.population, seed, task)
        self.ctrl = NeuralController(cfg, cfg.population, seed)
        self.age = np.zeros(cfg.population, dtype=int)  # generazioni già vissute
        self.generation = 0
        self.last_parents: Genome | None = None
        self.history: list[dict] = []
        self.champions: list[tuple[int, int, Genome, int]] = []  # (generazione, età, genoma, numero)
        self.ids = np.arange(cfg.population)
        self.next_id = cfg.population
        self.born = np.zeros(cfg.population, dtype=int)  # generazione di nascita
        self.selection: dict | None = None  # fase di selezione finale, se c'è stata

    @property
    def finished(self) -> bool:
        w = self.world
        return not (w.alive & ~w.done).any() or w.t >= self.cfg.life_max

    def step(self, manual: tuple[int, np.ndarray] | None = None) -> None:
        """Un passo di simulazione. `manual` = (agente, comandi) sostituisce
        l'uscita della rete di quell'agente (guida a mano dal viewer)."""
        actions = self.ctrl.act(self.world)
        if manual is not None:
            actions = actions.copy()
            actions[manual[0]] = manual[1]
        self.world.step(actions)

    def oldest(self) -> dict | None:
        """L'individuo più anziano in gara: posto nella popolazione, numero,
        generazioni vinte di fila e generazione di nascita. None se sono tutti
        appena nati."""
        if not len(self.age) or self.age.max() == 0:
            return None
        i = int(np.argmax(self.age))
        age = int(self.age[i])
        return {"index": i, "id": int(self.ids[i]), "age": age, "born": int(self.born[i])}

    def _new_ids(self, k: int) -> np.ndarray:
        ids = np.arange(self.next_id, self.next_id + k)
        self.next_id += k
        return ids

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
            "oldest_id": int(self.ids[np.argmax(self.age)]),
        }
        if hasattr(brains, "inhibitory"):  # quota di inibitori tra i neuroni interni esistenti
            stats["inhibitory"] = float(brains.inhibitory[brains.active].mean()) if brains.active.any() else 0.0

        genome = brains.genome
        if len(winners):
            # Il vincitore più anziano; a parità di età il nato prima (numero più basso).
            champion = winners[np.lexsort((self.ids[winners], -self.age[winners]))[0]]
            self.champions.append((self.generation, int(self.age[champion]), genome.take([champion]),
                                   int(self.ids[champion])))
            parents = genome.take(winners)
            self.last_parents = parents
            keep = winners if cfg.max_age <= 0 else winners[self.age[winners] + 1 < cfg.max_age]
            survivors, survivor_age, survivor_ids = genome.take(keep), self.age[keep] + 1, self.ids[keep]
            survivor_born = self.born[keep]
            stats["event"] = NORMAL if len(winners) > 1 else ONE_PARENT
        else:
            parents = self.last_parents
            survivors, survivor_age, survivor_ids = genome.take([]), np.zeros(0, dtype=int), self.ids[:0]
            survivor_born = self.born[:0]
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
        self.ids = np.concatenate([survivor_ids, self._new_ids(n - len(survivors))])
        self.born = np.concatenate([survivor_born, np.full(n - len(survivors), self.generation + 1)])
        world.n = n  # dopo una selezione finale la popolazione torna piena
        world.seed += 1
        world.reset()
        self.generation += 1
        self.history.append(stats)
        return stats

    # --- selezione finale ----------------------------------------------------

    @property
    def selecting(self) -> bool:
        return self.selection is not None and self.selection["active"]

    def _shrink(self, keep) -> None:
        """Restano solo gli individui `keep`; arena nuova."""
        brains = self.ctrl.brains
        brains.set_genome(brains.genome.take(keep))
        self.age, self.ids, self.born = self.age[keep], self.ids[keep], self.born[keep]
        self.world.n = len(keep)
        self.world.seed += 1
        self.world.reset()

    def start_selection(self, rounds: int) -> int:
        """Comincia la selezione finale: i sopravvissuti (chi ha vinto l'ultima
        generazione) rifaranno la prova `rounds` volte. Restituisce quanti
        partono; 0 se non c'è nessun sopravvissuto (e allora non comincia)."""
        keep = np.flatnonzero(self.age > 0)
        if not len(keep) or rounds <= 0:
            return 0
        self.selection = {"rounds": rounds, "done": 0, "entered": len(keep), "log": [],
                          "active": True, "note": ""}
        self._shrink(keep)
        return len(keep)

    def next_selection_round(self) -> dict:
        """Chiude una prova della selezione finale: chi ha fallito è eliminato.
        Se fallissero tutti, restano quelli della prova precedente (sono pari)."""
        sel, world = self.selection, self.world
        winners = np.flatnonzero(world.done)
        sel["done"] += 1
        entry = {"round": sel["done"], "entered": world.n, "passed": len(winners)}
        sel["log"].append(entry)
        if len(winners):
            self.age[winners] += 1
            keep = winners
        else:
            keep = np.arange(world.n)
            sel["note"] = (f"alla prova {sel['done']} nessuno ha completato: "
                           "restano tutti quelli della prova precedente")
        if not len(winners) or sel["done"] >= sel["rounds"]:
            sel["active"] = False
        self._shrink(keep)
        return entry

    def save(self, path) -> None:
        """Salva la popolazione all'inizio della generazione corrente, con ciò
        che serve per riprendere l'evoluzione da lì."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        g = self.ctrl.brains.genome
        data = dict(weights=g.weights, traits=g.traits, conn=g.conn, active=g.active, age=self.age,
                    ids=self.ids, next_id=self.next_id, born=self.born, population=self.cfg.population,
                    net_type=self.cfg.net_type, seed=self.seed,
                    generation=self.generation, world_seed=self.world.seed,
                    history=json.dumps(self.history))
        if self.last_parents is not None:
            lp = self.last_parents
            data.update(lp_weights=lp.weights, lp_traits=lp.traits, lp_conn=lp.conn, lp_active=lp.active)
        if self.champions:
            gens, ages, genomes, ids = zip(*self.champions)
            ch = Genome.concat(list(genomes))
            data.update(ch_generation=np.array(gens), ch_age=np.array(ages), ch_id=np.array(ids),
                        ch_weights=ch.weights, ch_traits=ch.traits, ch_conn=ch.conn, ch_active=ch.active)
        if self.selection is not None:
            data["selection"] = json.dumps(self.selection)
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, cfg: Config, path) -> "Evolution":
        """Riprende da un salvataggio. Dimensione della popolazione e tipo di
        rete sono quelli salvati; la forma delle reti deve coincidere con la
        configurazione attuale."""
        d = np.load(path)

        def traits(prefix=""):  # i salvataggi più vecchi avevano solo la soglia
            if prefix + "traits" in d:
                return d[prefix + "traits"]
            return d[prefix + "bias"][:, :, None]

        n = len(d["weights"])
        # Dopo una selezione finale gli individui sono meno della popolazione:
        # la generazione in corso la fanno solo loro, la successiva torna piena.
        cfg.population = int(d["population"]) if "population" in d else n
        cfg.net_type = str(d["net_type"]) if "net_type" in d else "rate"
        evo = cls(cfg, int(d["world_seed"]))
        expected = evo.ctrl.brains.weights.shape[1:]
        if d["weights"].shape[1:] != expected:
            raise ValueError(
                f"{path}: reti di forma {d['weights'].shape[1:]}, ma la configurazione attuale "
                f"prevede {expected} (sono cambiati i raggi, i sensi o net_hidden_max?)"
            )
        evo.ctrl.brains.set_genome(Genome(d["weights"], traits(), d["conn"], d["active"]))
        evo.age = d["age"]
        if "ids" in d:
            evo.ids, evo.next_id, evo.born = d["ids"], int(d["next_id"]), d["born"]
        else:
            evo.ids, evo.born = np.arange(n), evo.generation - evo.age
        if n != cfg.population:
            evo.world.n = n
            evo.world.reset()
        evo.generation = int(d["generation"])
        evo.seed = int(d["seed"]) if "seed" in d else None
        evo.history = json.loads(str(d["history"]))
        if "lp_weights" in d:
            evo.last_parents = Genome(d["lp_weights"], traits("lp_"), d["lp_conn"], d["lp_active"])
        if "ch_weights" in d:
            ch = Genome(d["ch_weights"], d["ch_traits"], d["ch_conn"], d["ch_active"])
            ids = d["ch_id"] if "ch_id" in d else np.full(len(d["ch_age"]), -1)
            evo.champions = [(int(g), int(a), ch.take([k]), int(i))
                             for k, (g, a, i) in enumerate(zip(d["ch_generation"], d["ch_age"], ids))]
        if "selection" in d:
            evo.selection = json.loads(str(d["selection"]))
        return evo
