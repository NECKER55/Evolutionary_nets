"""L'esame: una batteria di prove fisse per misurare quanto sono capaci gli
agenti, sempre uguale per tutta la vita di una corsa e mai usata per la
selezione.

Il successo nel compito del momento non basta come misura: se l'ambiente
diventa più difficile, la percentuale scende anche quando gli agenti
migliorano. L'esame invece non cambia mai:
- le costanti dell'ambiente d'esame sono scritte qui (`ENVIRONMENT`), non
  lette dalla configurazione della corsa: se cambi energia, veleno o fisica
  della corsa, l'esame resta quello;
- le arene hanno semi fissi, diversi da prova a prova e lontani da quelli
  dell'evoluzione e delle prove dal banco.
Dalla corsa vengono solo le cose che riguardano la rete (tipo, dimensione,
sensi), perché senza quelle la rete non si potrebbe nemmeno costruire.

Se un giorno le prove cambiano, si alza `VERSION`: i punteggi di versioni
diverse non si confrontano.
"""

import dataclasses
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..net.brain import Genome
from ..sim.tasks import FarTarget, MovingTarget, ReachTarget
from .trial import Trial

VERSION = 1
EXAM_SEED = 2_000_000  # le arene della prova k partono da EXAM_SEED + 10_000 * k

# Ambiente d'esame, fissato al momento della versione 1. Sono i valori di
# default di allora: non seguono i cambi futuri della configurazione.
ENVIRONMENT = dict(
    arena_w=200.0, arena_h=140.0, dt=1.0 / 60.0,
    agent_radius=1.0, thrust_accel=60.0, reverse_factor=0.5, lin_drag=2.0, lat_drag=8.0,
    turn_accel=40.0, ang_drag=6.0, wall_restitution=0.3,
    energy_init=150.0, metabolism_basal=2.5, metabolism_move=2.5, metabolism_turn=0.5, wire_cost=0.004,
    vision_fov_deg=120.0, vision_range=30.0, sensor_noise=0.0,
    life_max=90.0, target_radius=3.0,
    hazard_drain=60.0, hazard_falloff=2.0, n_hazards=12, hazard_radius_min=5.0, hazard_radius_max=12.0,
    hazard_omega_min=0.2, hazard_omega_max=0.8, hazard_pulse=0.15, hazard_speed=1.0, spawn_margin=3.0,
    # Fisica dei segnali dei recettori universali (ancora non usata dalla versione 1).
    light_attenuation=12.0, light_samples=5, wall_color=0.5, wall_color_width=0.3, wall_albedo=0.4,
    target_color=0.85, target_color_width=0.06, target_albedo=1.0,
    hazard_color=0.25, hazard_color_width=0.06, hazard_glow=0.05,
    toxin_molecule=0.15, hazard_smell=0.0, target_molecule=0.6, target_smell=0.0, smell_range=10.0,
    vibration_range=20.0, target_vibration=0.0, target_vibration_freq=0.6,
    hazard_vibration=0.0, hazard_vibration_freq=0.3,
    ambient_temperature=0.5, heat_range=15.0, target_heat=0.0, hazard_heat=0.0,
    magnetic_field=0.0, magnetic_angle_deg=90.0, receptor_dark=1.0, receptor_saturation=100.0,
)

# Le costanti che invece vengono dalla corsa: riguardano la rete (forma,
# tipo, come nasce) o l'evoluzione, non il mondo in cui l'agente vive. Ogni
# costante della configurazione deve stare in uno dei due elenchi: un test lo
# controlla. Quando si aggiunge fisica nuova bisogna quindi decidere se
# l'esame la include (in ENVIRONMENT, con il suo valore d'esame: di solito
# "spenta" per la versione 1) o se segue la corsa (qui).
FROM_RUN = frozenset({
    "vision_rays",  # numero di ingressi della rete: senza, la rete non si costruisce
    "net_type", "net_hidden_max", "net_weight_std", "net_conn_prob", "net_conn_range", "net_self_prob",
    "spike_substeps", "spike_gain", "spike_inhibitory_share", "sensors_excitatory_only",
    "population", "max_age", "rescue_share", "rescue_strength", "crossover_rate",
    "mut_weight_prob", "mut_weight_std", "mut_conn_add", "mut_conn_del", "mut_neuron_toggle",
    "mut_pos_prob", "mut_pos_std", "exam_every",
})


@dataclass(frozen=True)
class Test:
    key: str  # nome breve, per tabelle e grafici
    name: str
    what: str  # cosa misura
    changes: dict = field(default_factory=dict)  # rispetto a ENVIRONMENT
    task: type = ReachTarget


TESTS = (
    Test("base", "base", "il compito di sempre, in arene mai viste"),
    Test("lontano", "bersaglio lontano", "esplorazione: il bersaglio nasce ad almeno 90 u da tutti",
         task=FarTarget),
    Test("veleno+", "molto veleno", "prudenza: il doppio delle zone di veleno", {"n_hazards": 24}),
    Test("veleno0", "nessun veleno", "se il veleno serviva solo da ostacolo o anche da indizio",
         {"n_hazards": 0}),
    Test("arena", "arena mai vista", "generalizzazione: arena quadrata 160 x 160",
         {"arena_w": 160.0, "arena_h": 160.0}),
    Test("rumore", "sensi disturbati", "robustezza: rumore casuale su ogni senso (dev. std 0.1)",
         {"sensor_noise": 0.1}),
    Test("mobile", "bersaglio in movimento", "inseguire: il bersaglio si muove a 5 u/s",
         task=MovingTarget),
)


def exam_config(run_cfg: Config, test: Test) -> Config:
    """La configurazione di una prova: la rete dalla corsa, tutto il resto
    dall'esame."""
    return dataclasses.replace(run_cfg, **{**ENVIRONMENT, **test.changes})


def candidates(age: np.ndarray, ids: np.ndarray, k: int = 20) -> np.ndarray:
    """Chi fa l'esame: i `k` individui più anziani della popolazione (a
    parità di età, il nato prima). Sono quelli che hanno superato più prove:
    rappresentano ciò che la popolazione ha consolidato, non i figli appena
    nati."""
    return np.lexsort((ids, -age))[:k]


def _arena(run_cfg: Config, genomes: Genome, i: int, k: int, clones: int) -> np.ndarray:
    """Arena `k` della prova `i`: quante copie di ogni rete completano. Ogni
    arena è indipendente dalle altre, così si possono fare in parallelo con lo
    stesso risultato."""
    t = TESTS[i]
    trial = Trial(exam_config(run_cfg, t), genomes, 1, clones, seed=EXAM_SEED + 10_000 * i + k, task=t.task())
    while not trial.finished:
        trial.step()
    return trial.wins


def default_workers() -> int:
    """Processi per l'esame: tutti i core tranne due, lasciati all'evoluzione
    e all'interfaccia."""
    return max(1, (os.cpu_count() or 1) - 2)


class Exam:
    """Mette `genomes` alla prova su tutte le `TESTS`: `arenas` arene per
    prova, `clones` copie di ogni rete per arena.

    Ogni arena è un lavoro a sé. Con `workers` > 1 i lavori girano in processi
    separati, in parallelo e senza togliere tempo a chi li ha lanciati; con
    `workers` = 1 si fanno qui, a pezzi (`run(budget)`), come `Trial`. Il
    risultato è lo stesso. Non tocca nulla dell'evoluzione: farlo o non farlo
    non cambia la corsa.
    """

    def __init__(self, run_cfg: Config, genomes: Genome, generation: int, ids=None,
                 arenas: int = 8, clones: int = 5, workers: int | None = None):
        self.run_cfg, self.genomes, self.generation = run_cfg, genomes, generation
        self.ids = None if ids is None else [int(i) for i in ids]
        self.arenas, self.clones = arenas, clones
        self.todo = [(i, k) for i in range(len(TESTS)) for k in range(arenas)]
        self.wins = np.zeros((len(TESTS), len(genomes)), dtype=int)  # per prova e rete: copie arrivate
        self.done = 0
        self.started = time.perf_counter()
        self.seconds = 0.0
        self.workers = default_workers() if workers is None else workers
        self._pool, self._futures, self._trial = None, {}, None
        if self.workers > 1:
            try:
                self._pool = ProcessPoolExecutor(min(self.workers, len(self.todo)),
                                                 mp_context=multiprocessing.get_context("spawn"))
                self._futures = {self._pool.submit(_arena, run_cfg, genomes, i, k, clones): (i, k)
                                 for i, k in self.todo}
                self.todo = []
            except OSError:  # niente processi: si fa tutto qui
                self._pool, self.workers = None, 1

    @property
    def finished(self) -> bool:
        return self.done >= len(TESTS) * self.arenas

    @property
    def progress(self) -> float:
        """Quota fatta, da 0 a 1."""
        return self.done / (len(TESTS) * self.arenas)

    def _record(self, i: int, wins: np.ndarray) -> None:
        self.wins[i] += wins
        self.done += 1
        if self.finished:
            self.seconds = time.perf_counter() - self.started
            self.close()

    def run(self, budget: float = float("inf")) -> None:
        """In parallelo: raccoglie le arene finite (con `budget` infinito
        aspetta tutte). Qui: avanza per circa `budget` secondi di calcolo."""
        if self._pool is not None:
            ready = wait(self._futures, timeout=None if budget == float("inf") else 0).done
            for f in ready:
                i, _ = self._futures.pop(f)
                self._record(i, f.result())
            return
        deadline = time.perf_counter() + budget
        while not self.finished and time.perf_counter() < deadline:
            if self._trial is None:
                i, k = self.todo.pop(0)
                t = TESTS[i]
                self._trial = (i, Trial(exam_config(self.run_cfg, t), self.genomes, 1, self.clones,
                                        seed=EXAM_SEED + 10_000 * i + k, task=t.task()))
            i, trial = self._trial
            trial.step()
            if trial.finished:
                self._trial = None
                self._record(i, trial.wins)

    def close(self) -> None:
        """Ferma i processi (anche a esame non finito: il risultato si perde)."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    @property
    def rates(self) -> dict[str, np.ndarray]:
        """Per prova: quota di prove completate da ogni rete."""
        return {t.key: self.wins[i] / (self.arenas * self.clones) for i, t in enumerate(TESTS)}

    def result(self) -> dict:
        """Il risultato, pronto per il registro della corsa (solo tipi JSON)."""
        rates = self.rates
        scores = {k: float(r.mean()) for k, r in rates.items()}
        return {
            "generation": self.generation,
            "version": VERSION,
            "who": len(self.genomes),
            "ids": self.ids,
            "arenas": self.arenas,
            "clones": self.clones,
            "scores": scores,
            "score": float(np.mean(list(scores.values()))),
            "best": float(np.mean(list(rates.values()), axis=0).max()),
            "seconds": round(self.seconds, 1),
            "workers": self.workers,
        }
