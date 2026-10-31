import dataclasses

import numpy as np

from ...brains.freeform.controllers import NeuralController
from ...brains.freeform.development import population_stats
from ...brains.freeform.genetics import grow, reproduce
from ...brains.freeform.genome import META, Genome, pad
from ...brains.freeform.womb import prenatal
from ...config import Config
from . import ladder, saves
from .exam import Exam, candidates
from .saves import code_version
from .world import World

NORMAL = "normale"
ONE_PARENT = "un solo genitore"
RESTART = "nessun genitore: ripartenza dagli ultimi"
RANDOM = "nessun genitore: reti casuali"

def run_together(evos: list["Evolution"]) -> list[dict]:
    """Una generazione di più evoluzioni (per esempio più semi dello stesso
    esperimento) simulate insieme sulla GPU, nello stesso lotto. Servono la
    stessa popolazione, lo stesso tipo di rete e lo stesso numero di zone di
    veleno, lo stesso compito (le manopole del mondo del cibo possono essere
    diverse: la scala e i suoi controlli girano insieme); i posti di
    recettori e neuroni vengono pareggiati al più grande
    (posti spenti: non cambia nulla per nessuno). Restituisce le statistiche
    di ciascuna."""
    from ...gpu import available
    from .gpu_worlds import GpuWorlds, supported
    ok = (all(e.cfg.engine == "gpu" for e in evos) and available()
          and all(supported(e.world, e.ctrl) for e in evos)
          and len({(e.world.n, e.cfg.net_type, len(e.world.hazards.pos), e.cfg.task, e.cfg.food_count_easy,
                    e.cfg.predator_count_max) for e in evos}) == 1)
    if ok:
        r = max(e.ctrl.brains.n_in for e in evos)
        h = max(e.ctrl.brains.n_hidden for e in evos)
        for e in evos:
            b = e.ctrl.brains
            if (b.n_in, b.n_hidden) != (r, h):
                b.set_genome(pad(b.genome, r, h))
                e.ctrl.reset(e.world.seed)
                e._sim = None
        GpuWorlds(evos[0].cfg, [(e.world, e.ctrl) for e in evos], [e.world.seed for e in evos]).run()
    return [e.run_generation() for e in evos]


def receptor_stats(g: Genome, links: np.ndarray) -> dict:
    """Medie per individuo sui recettori: quanti sono accesi, quanti sono
    usati (hanno almeno un collegamento effettivo in uscita) e, tra gli usati,
    quanti sono specializzati su ciascun fenomeno (un fenomeno pesa almeno
    l'80% della loro sensibilità) o generalisti. Più la sensibilità media per
    fenomeno dei recettori accesi."""
    on = g.r_active
    used = on & links[:, :, : g.n_rec].any(axis=1)
    gain = np.exp(g.r_sens) * on[:, :, None]  # (N, R, P)
    total = np.maximum(gain.sum(axis=2), 1e-300)
    dom, share = gain.argmax(axis=2), gain.max(axis=2) / total
    kind = np.where(share >= 0.8, dom, gain.shape[2])  # l'ultimo = generalista
    n_on, n = on.sum(), len(on)
    return {
        "receptors": float(on.sum(axis=1).mean()),
        "used": float(used.sum(axis=1).mean()),
        "used_by": [float(((kind == p) & used).sum() / n) for p in range(gain.shape[2] + 1)],
        "specialized": float(((share >= 0.8) & used).sum() / max(used.sum(), 1)),
        "sensitivity": [float(x) for x in gain.sum(axis=(0, 1)) / max(n_on, 1)],
    }


def describe_change(changes: dict) -> str:
    return ", ".join(f"{k} {old} → {new}" for k, (old, new) in changes.items())


class Evolution:
    """Il ciclo delle generazioni.

    Una generazione vive nell'arena finché tutti sono morti o hanno completato
    il compito. Poi:
      - chi ha completato è genitore e passa identico alla generazione
        successiva, dove dovrà completare di nuovo per restare (`max_age`, se
        impostato, aggiunge un limite di età);
      - i posti restanti vanno ai figli, con genitori sorteggiati alla pari;
      - senza genitori si riparte dagli ultimi genitori noti (tutti figli) o, se
        non ce ne sono mai stati, da una popolazione primordiale nuova.
    Non c'è nessun "soccorso" con mutazioni rinforzate: in natura non esiste.
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

    `log` è il registro dell'ambiente: quando la corsa è nata o è stata
    ripresa (con quale versione del codice) e ogni cambio della
    configurazione, con la generazione in cui è avvenuto. Viene salvato con la
    configurazione completa, così una corsa si può sempre ricostruire.
    """

    def __init__(self, cfg: Config, seed: int = 0, task=None):
        self.cfg = cfg
        ladder.prepare(cfg)  # i controlli hanno le manopole ferme
        self.seed = seed  # il seme con cui è nata la corsa
        self.rng = np.random.default_rng(seed)
        self.world = World(cfg, cfg.population, seed, task)
        self.ctrl = NeuralController(cfg, cfg.population, seed)
        self._birth(np.arange(cfg.population))
        self.age = np.zeros(cfg.population, dtype=int)  # generazioni già vissute
        self.generation = 0
        self.last_parents: Genome | None = None
        self.history: list[dict] = []
        self.champions: list[tuple[int, int, Genome, int]] = []  # (generazione, età, genoma, numero)
        self.ids = np.arange(cfg.population)
        self.next_id = cfg.population
        self.born = np.zeros(cfg.population, dtype=int)  # generazione di nascita
        self.selection: dict | None = None  # fase di selezione finale, se c'è stata
        self.log: list[dict] = [{"generation": 0, "event": "inizio", "code": code_version()}]
        self.exams: list[dict] = []  # risultati degli esami fatti (vedi arenas/generational/exam.py)
        self._sim = None  # la generazione in corso sulla GPU, se il motore è "gpu"
        self.exam: Exam | None = None  # esame in corso
        self._seen = dataclasses.asdict(cfg)  # configurazione già registrata

    def _birth(self, rows: np.ndarray) -> None:
        """I neonati `rows` passano dal grembo (solo con lo sviluppo: il genoma
        diretto non ha un fenotipo da rifinire). Vedi brains/freeform/womb.py."""
        if self.cfg.genome == "sviluppo":
            prenatal(self.ctrl.brains, rows, self.rng, self.cfg.engine)
            self.ctrl.reset(self.world.seed)

    def _lived(self) -> None:
        """Quanto ha già vissuto ogni individuo (una vita intera per ogni
        generazione superata): l'età conta per il periodo critico, e un
        sopravvissuto non torna bambino in un'arena nuova."""
        self.ctrl.brains.lived = self.age * float(self.cfg.life_max)

    def note_changes(self) -> dict:
        """Registra i cambi della configurazione dall'ultima volta (per esempio
        le costanti ritoccate a mano durante la corsa). Restituisce i cambi."""
        now = dataclasses.asdict(self.cfg)
        changes = {k: [self._seen.get(k), v] for k, v in now.items() if self._seen.get(k) != v}
        if changes:
            self.log.append({"generation": self.generation, "event": "cambio", "changes": changes})
            self._seen = now
        return changes

    @property
    def finished(self) -> bool:
        w = self.world
        return w.over

    def gpu(self):
        """La simulazione della generazione in corso sul motore compilato, o
        None se si usa il motore NumPy (scelto, o JAX non disponibile, o un
        compito che il motore GPU non conosce)."""
        if self.cfg.engine != "gpu":
            return None
        if self._sim is None:
            from ...gpu import available
            if not available():
                return None
            from .gpu_worlds import GpuWorlds, supported
            if not supported(self.world, self.ctrl):
                return None
            self._sim = GpuWorlds(self.cfg, [(self.world, self.ctrl)], [self.world.seed])
        return self._sim

    def step(self, manual: tuple[int, np.ndarray] | None = None) -> None:
        """Un passo di simulazione. `manual` = (agente, comandi) sostituisce
        l'uscita della rete di quell'agente (guida a mano dal viewer)."""
        self.advance(1, manual)

    def advance(self, k: int, manual: tuple[int, np.ndarray] | None = None) -> None:
        """k passi (meno se la generazione finisce prima)."""
        sim = self.gpu()
        if sim is not None:
            sim.advance(k, manual)
            return
        for _ in range(k):
            if self.finished:
                return
            actions = self.ctrl.act(self.world)
            if manual is not None:
                actions = actions.copy()
                actions[manual[0]] = manual[1]
            self.world.step(actions, self.ctrl.upkeep)

    def start_exam(self, **kw) -> Exam:
        """Comincia l'esame dei più anziani della popolazione attuale. Si
        porta avanti con `exam.run(budget)` e si chiude con `collect_exam`.
        `kw` va a `Exam` (arene e copie: cambiarle solo per le prove del codice)."""
        idx = candidates(self.age, self.ids)
        self.exam = Exam(self.cfg, self.ctrl.brains.genome.take(idx), self.generation, self.ids[idx], **kw)
        return self.exam

    def collect_exam(self) -> dict | None:
        """Se l'esame in corso è finito, ne registra il risultato e lo
        restituisce."""
        if self.exam is None or not self.exam.finished:
            return None
        result = self.exam.result()
        self.exams.append(result)
        self.exam = None
        return result

    def exam_due(self) -> bool:
        """È ora dell'esame automatico (ogni `exam_every` generazioni)?"""
        every = self.cfg.exam_every
        return (every > 0 and self.generation > 0 and self.generation % every == 0 and self.exam is None
                and not any(e["generation"] == self.generation and not e.get("lesion") for e in self.exams))

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
        sim = self.gpu()
        if sim is not None:
            sim.run()
        while not self.finished:
            self.step()
        return self.next_generation()

    def next_generation(self) -> dict:
        """Selezione e ricambio. Restituisce le statistiche della generazione
        appena conclusa (aggiunte anche a `history`)."""
        self.note_changes()
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
            "wiring": float(brains.wire_length.mean()),
            **receptor_stats(brains.genome, brains.links),
            "upkeep": float(brains.upkeep.mean()),
            # Plasticità: modulatori per individuo, quota di chi può imparare.
            "modulators": float(brains.modulatory.sum(axis=1).mean()),
            "learners": float(brains.learns.mean()),
            # Sistema di valore innato: forza media dei nuclei di gusto e dolore.
            "value": [float(v) for v in brains.value.mean(axis=0)],
            # Sviluppo: marcatori accesi e tipi di neuroni interni.
            **population_stats(brains.genome),
            "mutation": {k: float(np.exp(np.median(brains.genome.meta[:, i]))) for i, k in enumerate(META)},
            "oldest": int(self.age.max()),
            "oldest_id": int(self.ids[np.argmax(self.age)]),
        }
        if hasattr(brains, "inhibitory"):  # quota di inibitori tra i neuroni interni esistenti
            stats["inhibitory"] = float(brains.inhibitory[brains.active].mean()) if brains.active.any() else 0.0
        if cfg.task == "cibo":
            stats["dials"] = ladder.dials(cfg)  # le manopole con cui ha vissuto
            stats["position"] = ladder.position(cfg)
            stats["eaten"] = float(world.task.eaten.mean())
            stats["bad_eaten"] = float(world.task.bad_eaten.mean())  # frutti cattivi mangiati (errori)
            if world.n == len(getattr(world.task, "start_dist", ())):
                stats["luck"] = ladder.luck(world.task.start_dist, world.done)
        moved = ladder.update(cfg, self.history + [stats])
        if moved:  # vale dalla prossima generazione
            self.log.append({"generation": self.generation + 1, "event": "scala", **moved})
            self._seen.update(ladder.dials(cfg))

        genome = brains.genome
        if len(winners):
            # Il vincitore più anziano; a parità di età il nato prima (numero più basso).
            champion = winners[np.lexsort((self.ids[winners], -self.age[winners]))[0]]
            self.champions.append((self.generation, int(self.age[champion]), genome.take([champion]),
                                   int(self.ids[champion])))
            parents = genome.take(winners)
            self.last_parents = parents
            # Fecondità: come in natura, più cibo mangiato in tutta la vita, più figli.
            fecundity = None
            if cfg.fecundity_by_food and hasattr(world.task, "eaten"):
                fecundity = world.task.eaten[winners].astype(float)
            # Sopravvivono i maturi ancora vivi (con la vita che continua si può
            # morire anche dopo essere diventati maturi).
            keep = winners[world.alive[winners]]
            keep = keep if cfg.max_age <= 0 else keep[self.age[keep] + 1 < cfg.max_age]
            survivors, survivor_age, survivor_ids = genome.take(keep), self.age[keep] + 1, self.ids[keep]
            survivor_born = self.born[keep]
            stats["event"] = NORMAL if len(winners) > 1 else ONE_PARENT
        else:
            parents, fecundity = self.last_parents, None
            survivors, survivor_age, survivor_ids = genome.take([]), np.zeros(0, dtype=int), self.ids[:0]
            survivor_born = self.born[:0]
            stats["event"] = RESTART if parents is not None else RANDOM

        if parents is None:
            fresh = NeuralController(cfg, n, int(self.rng.integers(2**31)))
            new = fresh.brains.genome
        else:
            children = reproduce(parents, n - len(survivors), cfg, self.rng, fecundity)
            new = Genome.concat([survivors, children])
        stats["survivors"] = len(survivors)
        new = grow(new, cfg)  # posti nuovi se qualcuno li ha quasi finiti
        if self.last_parents is not None:
            self.last_parents = pad(self.last_parents, new.n_rec, new.n_hidden)

        brains.set_genome(new)
        self._birth(np.arange(len(survivors), n))
        self.age = np.concatenate([survivor_age, np.zeros(n - len(survivors), dtype=int)])
        self.ids = np.concatenate([survivor_ids, self._new_ids(n - len(survivors))])
        self.born = np.concatenate([survivor_born, np.full(n - len(survivors), self.generation + 1)])
        world.n = n  # dopo una selezione finale la popolazione torna piena
        world.seed += 1
        world.reset()
        self._lived()
        self.ctrl.reset(world.seed)
        self._sim = None
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
        self._lived()
        self.ctrl.reset(self.world.seed)
        self._sim = None

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
        """Salva la popolazione all'inizio della generazione corrente (vedi `saves.save`)."""
        saves.save(self, path)

    @classmethod
    def load(cls, cfg: Config, path) -> "Evolution":
        """Riprende da un salvataggio con le regole più recenti (vedi `saves.load`)."""
        return saves.load(cls, cfg, path)
