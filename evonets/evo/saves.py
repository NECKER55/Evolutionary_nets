"""Salvataggi delle corse: formato, scrittura sicura, caricamento con le
regole più recenti e conversione dei formati vecchi.

Un salvataggio contiene la popolazione all'inizio della generazione corrente,
la storia, i campioni, gli esami, la configurazione completa e il registro
dell'ambiente. Riprendendolo si gioca sempre con le regole del codice attuale:
dal file vengono solo i genomi e lo stato della corsa (`RUN_STATE`).
"""

import dataclasses
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from ..config import Config
from ..net.brain import META, Genome, meta_start
from ..net.receptors import scales
from . import ladder

# Versione del formato dei salvataggi. 1 = con geometria del cervello ma senza
# configurazione; 2 = con configurazione completa e registro dell'ambiente;
# 3 = recettori universali, posti che crescono, mutazioni come geni;
# 4 = adattamento sensoriale (i file del formato 3 si aprono con l'adattamento spento);
# 5 = plasticità (i file precedenti si aprono senza plasticità né modulatori);
# 6 = sensi calibrati: la sensibilità è un moltiplicatore della scala fisica di
#     ogni fenomeno (i file precedenti si convertono a parità di quanti);
# 7 = il cervello cresce con le sue cellule (le distanze si ricalcolano);
# 8 = discendenza dei geni (identità e famiglia, per l'incrocio tra omologhi)
#     e gene del sesso (i file precedenti: identità = posto, sesso 0,5).
FORMAT = 8
# Geni della plasticità aggiunti nel formato 5, per chi non li aveva: η quasi
# nullo, regola a zero, nessun modulatore.
NO_PLASTICITY = np.array([-30.0, 0.0, 0.0, 0.0, 0.0, -5.0])
# Geni aggiunti dopo il formato 3, con il valore che vuol dire "assente".
ADDED_IN_4 = {"r_tau": 0.0, "r_sub": 0.0, "r_div": 0.0}
# Lo stato di una corsa: le costanti che dicono quale corsa è e dove è
# arrivata (rete e forma dei genomi, popolazione, mondo scelto, modo e
# posizione della scala, frequenza dell'esame). Riprendendo un salvataggio si
# conservano; tutte le altre costanti sono regole e vengono sempre dal codice
# attuale, così ogni corsa si può continuare con le regole più recenti (i
# cambi finiscono nel registro).
RUN_STATE = ("sim_mode", "sensor_set", "net_type", "population", "net_hidden_slots", "receptor_slots", "task",
             "n_hazards", "ladder",
             *ladder.DIALS, "exam_every")
# Cosa cambia aprendo un file di un formato vecchio: (fino al formato, nota).
CONVERSIONS = (
    (3, "recettori senza adattamento sensoriale"),
    (4, "neuroni senza plasticità né modulatori"),
    (5, "sensibilità dei recettori convertite in moltiplicatori della scala fisica"),
    (6, "distanze nel cervello ricalcolate: ora il cervello cresce con le sue cellule"),
    (7, "discendenza dei geni dal loro posto, gene del sesso a 0,5"),
)


def code_version() -> str:
    """Il commit git del codice in uso ("+modifiche" se ce ne sono di non
    committate), o "sconosciuta" fuori da un repository."""
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=5, check=False).stdout.strip()
        return head + ("+modifiche" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "sconosciuta"


def save(evo, path) -> None:
    """Salva la popolazione all'inizio della generazione corrente, con ciò che
    serve per riprendere l'evoluzione da lì.

    La scrittura è sicura: si scrive un file temporaneo accanto e lo si mette
    al posto del vecchio in un colpo solo. Se il programma si interrompe a
    metà, il salvataggio precedente resta intatto, e chi lo legge nel
    frattempo non trova mai un file a metà."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    evo.note_changes()
    g = evo.ctrl.brains.genome
    data = dict(**{f: getattr(g, f) for f in Genome.FIELDS}, age=evo.age,
                ids=evo.ids, next_id=evo.next_id, born=evo.born, population=evo.cfg.population,
                net_type=evo.cfg.net_type, seed=evo.seed,
                generation=evo.generation, world_seed=evo.world.seed,
                history=json.dumps(evo.history), format=FORMAT,
                config=json.dumps(dataclasses.asdict(evo.cfg)), log=json.dumps(evo.log),
                exams=json.dumps(evo.exams))
    if evo.last_parents is not None:
        data.update({"lp_" + f: getattr(evo.last_parents, f) for f in Genome.FIELDS})
    if evo.champions:
        gens, ages, genomes, ids = zip(*evo.champions)
        ch = Genome.concat(list(genomes))  # i posti vengono pareggiati
        data.update(ch_generation=np.array(gens), ch_age=np.array(ages), ch_id=np.array(ids),
                    **{"ch_" + f: getattr(ch, f) for f in Genome.FIELDS})
    if evo.selection is not None:
        data["selection"] = json.dumps(evo.selection)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load(cls, cfg: Config, path):
    """Riprende da un salvataggio con le regole più recenti: dal file vengono
    la popolazione, i genomi e lo stato della corsa (`RUN_STATE`); tutte le
    altre costanti restano quelle di `cfg` (il codice attuale). Ogni regola
    diversa da quella con cui la corsa era stata salvata è annotata nel
    registro, alla generazione della ripresa. I salvataggi precedenti al
    formato 3 (sensi fissi) non si aprono. `cls` è la classe dell'evoluzione."""
    with np.load(path) as d:
        fmt = int(d["format"]) if "format" in d else 1
        if fmt > FORMAT:
            raise ValueError(f"{path}: salvataggio di formato {fmt}, più nuovo di questo codice ({FORMAT})")
        if fmt < 3:
            raise ValueError(f"{path}: salvataggio di formato {fmt}, con i sensi fissi di prima dei recettori "
                             "universali: non si può più caricare")
        fields = {f.name for f in dataclasses.fields(cfg)}
        saved = json.loads(str(d["config"]))
        unknown = {k: v for k, v in saved.items() if k not in fields}  # costanti che il codice non conosce più
        for k in RUN_STATE:
            if k in saved:
                setattr(cfg, k, saved[k])
        rules = {k: [saved.get(k), getattr(cfg, k)] for k in sorted(fields - set(RUN_STATE) - {"engine"})
                 if saved.get(k) != getattr(cfg, k)}

        base = 1 if str(d["net_type"]) == "rate" else 3  # geni propri prima della plasticità

        def genome(prefix=""):
            shape = d[prefix + "r_active"].shape
            g = Genome(*(d[prefix + f] if prefix + f in d
                         else None if f in Genome.LINEAGE else np.full(shape, ADDED_IN_4[f])
                         for f in Genome.FIELDS))
            if g.meta.shape[1] < len(META):  # geni nati dopo (il sesso): il valore di partenza
                start = np.log(np.maximum(meta_start(cfg), cfg.mutation_floor))[g.meta.shape[1]:]
                g.meta = np.concatenate([g.meta, np.broadcast_to(start, (len(g), len(start)))], axis=1)
            if fmt < 6:  # prima la sensibilità era in quanti: stessa risposta, nuova unità
                g.r_sens = g.r_sens - np.log(scales(cfg))
            if g.traits.shape[2] == base:
                extra = np.broadcast_to(NO_PLASTICITY, g.traits.shape[:2] + (len(NO_PLASTICITY),))
                g.traits = np.concatenate([g.traits, extra], axis=2)
            return g

        # Dopo una selezione finale gli individui sono meno della popolazione:
        # la generazione in corso la fanno solo loro, la successiva torna piena.
        n = len(d["weights"])
        cfg.population = int(d["population"])
        cfg.net_type = str(d["net_type"])
        evo = cls(cfg, int(d["world_seed"]))
        evo.ctrl.brains.set_genome(genome())
        evo.ctrl.reset(evo.world.seed)
        evo.age = d["age"]
        evo.ids, evo.next_id, evo.born = d["ids"], int(d["next_id"]), d["born"]
        if n != cfg.population:
            evo.world.n = n
            evo.world.reset()
        evo._sim = None
        evo.generation = int(d["generation"])
        evo.seed = int(d["seed"])
        evo.history = json.loads(str(d["history"]))
        if "lp_weights" in d:
            evo.last_parents = genome("lp_")
        if "ch_weights" in d:
            ch = genome("ch_")
            evo.champions = [(int(g), int(a), ch.take([k]), int(i))
                             for k, (g, a, i) in enumerate(zip(d["ch_generation"], d["ch_age"], d["ch_id"]))]
        if "selection" in d:
            evo.selection = json.loads(str(d["selection"]))
        evo.exams = json.loads(str(d["exams"]))
        evo.log = json.loads(str(d["log"]))
    resumed = {"generation": evo.generation, "event": "ripresa", "code": code_version()}
    notes = []
    if fmt < FORMAT:
        parts = [text for upto, text in CONVERSIONS if fmt <= upto]
        notes.append(f"salvataggio di formato {fmt}: convertito ({'; '.join(parts)})")
    if unknown:
        notes.append("costanti salvate che questo codice non conosce più: " + ", ".join(unknown))
    if notes:
        resumed["note"] = " | ".join(notes)
    if rules:  # [valore con cui era stata salvata (None = non esisteva), valore di adesso]
        resumed["rules"] = rules
    evo.log.append(resumed)
    evo._seen = dataclasses.asdict(cfg)
    return evo
