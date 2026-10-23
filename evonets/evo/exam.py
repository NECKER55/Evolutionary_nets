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

Le corse nel mondo del cibo fanno un'altra batteria, l'esame a livelli
(versione 4, docs/SPEC_MONDO.md): 8 livelli fissi lungo la scala di
difficoltà, dal mondo basico al complesso. Il risultato è un profilo:
diventare più intelligenti vuol dire migliorare sui livelli difficili senza
perdere quelli facili.

Prove di lesione (`lesion`): lo stesso esame con una parte del cervello tolta
(un senso, i neuroni interni, la plasticità). Dicono cosa gli agenti usano
davvero.
"""

import dataclasses
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..net.brain import Genome
from ..sim.signals import PHENOMENA
from ..sim.tasks import FarTarget, Foraging, MovingTarget, ReachTarget
from . import ladder
from .trial import Trial

VERSION = 3  # 2: recettori universali; 3: sensi calibrati e movimento con il costo naturale
_LEVELS_V6 = 6  # esame a livelli del mondo del cibo; 5: odori, nube che sfugge, 4 cibi al massimo;
# 6: fino a 10 cibi in 240 s, cibo da 100 di energia, un'arena per agente;
# 7: quarto stadio, le rocce (i livelli coprono la scala da 0 a 4);
# 8: quinto stadio, i frutti buoni e cattivi (scala da 0 a 5)
LEVELS_VERSION = 8
EXAM_SEED = 2_000_000  # le arene della prova k partono da EXAM_SEED + 10_000 * k

# Ambiente d'esame, fissato al momento della versione 3. Sono i valori di
# default di allora: non seguono i cambi futuri della configurazione.
ENVIRONMENT = dict(
    arena_w=200.0, arena_h=140.0, dt=1.0 / 60.0,
    agent_radius=1.0, thrust_accel=60.0, reverse_factor=0.5, lin_drag=2.0, lat_drag=8.0,
    turn_accel=40.0, ang_drag=6.0, wall_restitution=0.3,
    energy_init=150.0, energy_max=150.0, reserve_mass=0.0, metabolism_basal=2.5, metabolism_move=10.0, metabolism_turn=2.0, metabolism_exponent=2.0,
    wire_cost=0.004,
    sensor_noise=0.0, receptor_cell_cost=0.002, receptor_sens_cost=0.008, neuron_cost=0.01,
    life_max=90.0, target_radius=3.0,
    hazard_drain=60.0, hazard_falloff=2.0, n_hazards=12, hazard_radius_min=5.0, hazard_radius_max=12.0,
    hazard_omega_min=0.2, hazard_omega_max=0.8, hazard_pulse=0.15, hazard_speed=1.0, spawn_margin=3.0,
    # Fisica dei segnali dei recettori universali.
    light_attenuation=12.0, light_samples=5, wall_color=0.5, wall_color_width=0.3, wall_albedo=0.4,
    target_color=0.85, target_color_width=0.06, target_albedo=1.0,
    hazard_color=0.25, hazard_color_width=0.06, hazard_glow=0.05,
    toxin_molecule=0.15, hazard_smell=0.0, target_molecule=0.6, target_smell=0.0, smell_range=10.0,
    vibration_range=20.0, target_vibration=0.0, target_vibration_freq=0.6,
    hazard_vibration=0.0, hazard_vibration_freq=0.3,
    ambient_temperature=0.5, heat_range=15.0, target_heat=0.0, hazard_heat=0.0,
    magnetic_field=0.0, magnetic_angle_deg=90.0, receptor_dark=1.0, receptor_saturation=300.0,
    receptor_scale_light=300.0, receptor_scale_chemical=200.0, receptor_scale_mechanical=100.0,
    receptor_scale_thermal=100.0, receptor_scale_magnetic=100.0, receptor_scale_internal=70.0,
    # Il mondo del cibo (non usato dalla versione 3, che è nel mondo del bersaglio).
    task='bersaglio', forage_abundance=0.0, forage_need=0.0, forage_prey=0.0, forage_predators=0.0,
    forage_obstacles=0.0, rock_count_max=8, rock_radius_min=6.0, rock_radius_max=16.0, rock_gap=10.0, rock_color=0.45,
    rock_color_width=0.2, rock_albedo=0.4, forage_variety=0.0, fruit_color=0.65, fruit_molecule=0.5,
    bad_fruit_damage=30.0,
    food_count_easy=12, food_count_hard=3, food_radius_easy=4.0, food_radius_hard=2.0, food_need_easy=1,
    food_need_hard=5, prey_speed_max=12.0, prey_flee_range=15.0, predator_count_max=3, predator_speed_max=15.0,
    predator_range=40.0, predator_radius=1.5, predator_damage=60.0, food_energy=40.0, food_min_dist=20.0,
    predator_min_dist=40.0, food_color=0.85, food_color_width=0.06, food_albedo=1.0, predator_color=0.05,
    predator_color_width=0.06, predator_albedo=0.8, predator_vibration=0.5, predator_vibration_freq=0.4,
    forage_scent=0.0, food_molecule=0.6, food_smell=0.0, predator_molecule=0.35, predator_smell=0.0,
    odor_range=25.0, scent_time_max=3.0, forage_own_arena=False,
    mature_keep_living=False,  # l'esame misura chi diventa maturo: lì la prova finisce
)

# Il mondo del cibo dell'esame a livelli (versione 5): scritto qui, non letto
# dalla corsa.
FORAGE = dict(task="cibo", n_hazards=0, life_max=240.0, food_count_hard=4, food_need_hard=10, food_energy=100.0,
              food_smell=1.0,
              predator_smell=1.0, forage_own_arena=True)

# Le costanti che invece vengono dalla corsa: riguardano la rete (forma,
# tipo, come nasce) o l'evoluzione, non il mondo in cui l'agente vive. Ogni
# costante della configurazione deve stare in uno dei due elenchi: un test lo
# controlla. Quando si aggiunge fisica nuova bisogna quindi decidere se
# l'esame la include (in ENVIRONMENT, con il suo valore d'esame: di solito
# "spenta" per la versione 1) o se segue la corsa (qui).
FROM_RUN = frozenset({
    "sim_mode", "sensor_set", "genome", "marker_slots", "marker_initial", "marker_initial_std",
    "marker_gradient_std", "marker_table_std", "marker_map_gradient", "marker_map_affinity", "marker_map_std",
    "develop_threshold", "develop_precision", "prenatal_time", "wave_speed", "wave_width", "wave_interval",
    "prenatal_noise", "critical_initial_boost", "critical_initial_tau", "taste_tau", "value_initial", "net_type", "net_weight_std", "net_conn_prob", "net_conn_range", "net_self_prob",
    "net_hidden_slots", "receptor_slots", "pool_fill", "pool_step", "brain_density",
    "receptor_initial", "receptor_initial_spread_deg", "receptor_initial_aperture_deg",
    "receptor_initial_gain", "receptor_initial_width", "receptor_initial_tau", "receptor_initial_adaptation",
    "plasticity", "plasticity_initial_rate", "plasticity_initial_rule", "hebb_initial_rate", "homeo_initial_rate", "homeo_tau", "early_initial_tau", "capture_rate",
    "modulatory_initial_share",
    "modulation_range", "synapse_max",
    "spike_substeps", "spike_gain", "spike_inhibitory_share", "sensors_excitatory_only",
    "population", "max_age", "crossover_rate",
    "mut_weight_prob", "mut_weight_std", "mut_conn_add", "mut_conn_del", "mut_neuron_toggle",
    "mut_pos_prob", "mut_pos_std", "mut_dup", "receptor_dup_split", "fecundity_by_food", "mut_rec_drift", "mutation_floor", "exam_every", "engine",
    "ladder", "ladder_window", "ladder_up", "ladder_down", "ladder_step", "ladder_crash",
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


def level_world(d: dict) -> dict:
    """Cosa c'è in un livello: cibi in arena, da mangiare, velocità di fuga,
    tempo della nube, predatori."""
    e = {**ENVIRONMENT, **FORAGE}
    return {
        "food": int(round(e["food_count_easy"] + (e["food_count_hard"] - e["food_count_easy"]) * d["forage_abundance"])),
        "need": int(round(e["food_need_easy"] + (e["food_need_hard"] - e["food_need_easy"]) * d["forage_need"])),
        "flee": e["prey_speed_max"] * d["forage_prey"],
        "scent": e["scent_time_max"] * d["forage_scent"],
        "predators": int(np.ceil(e["predator_count_max"] * d["forage_predators"])),
        "rocks": int(np.ceil(e["rock_count_max"] * d.get("forage_obstacles", 0.0))),
        "bad": int(round(round(e["food_count_easy"] + (e["food_count_hard"] - e["food_count_easy"])
                               * d["forage_abundance"]) * 0.5 * d.get("forage_variety", 0.0))),
    }


def _level(k: int, n: int = 8) -> Test:
    x = ladder.TOP * k / (n - 1)
    d = ladder.at(x)
    w = level_world(d)
    what = (f"{w['food']} cibi, {w['need']} da mangiare, prede a {w['flee']:.0f} u/s, nube in {w['scent']:.1f} s, "
            f"{w['predators']} predatori, {w['rocks']} rocce, {w['bad']} frutti cattivi")
    return Test(f"L{k}", f"livello {k}", what, {**FORAGE, **d}, task=Foraging)


# L'esame a livelli: posizioni 0, 3/7, ..., 3 della scala (manopole scritte
# qui: non seguono la scala della corsa).
LEVELS = tuple(_level(k) for k in range(8))

# Le lesioni possibili: un senso (sensibilità a quel fenomeno azzerata in
# tutti i recettori), i neuroni interni, la plasticità.
LESIONS = ("neuroni", "plasticità") + PHENOMENA


def battery(run_cfg: Config) -> tuple[Test, ...]:
    """Le prove dell'esame di una corsa: i livelli per il mondo del cibo, le
    prove della versione 3 per il mondo del bersaglio."""
    return LEVELS if run_cfg.task == "cibo" else TESTS


def lesioned(run_cfg: Config, genomes: Genome, lesion: str | None) -> tuple[Config, Genome]:
    """La corsa e i genomi con la lesione applicata (copie: gli originali non
    cambiano)."""
    if lesion is None:
        return run_cfg, genomes
    if lesion not in LESIONS:
        raise ValueError(f"lesione sconosciuta: {lesion!r} (valide: {', '.join(LESIONS)})")
    if lesion == "plasticità":
        return dataclasses.replace(run_cfg, plasticity=False), genomes
    g = genomes.take(np.arange(len(genomes)))
    if lesion == "neuroni":
        g.active[:] = False
    else:
        g.r_sens[..., PHENOMENA.index(lesion)] = -30.0  # sensibilità e^-30: nulla
    return run_cfg, g


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
    t = battery(run_cfg)[i]
    trial = Trial(exam_config(run_cfg, t), genomes, 1, clones, seed=EXAM_SEED + 10_000 * i + k, task=t.task())
    while not trial.finished:
        trial.step()
    return trial.wins


def _gpu_ready() -> bool:
    from ..gpu import available
    return available()


def default_workers() -> int:
    """Processi per l'esame: tutti i core tranne due, lasciati all'evoluzione
    e all'interfaccia."""
    return max(1, (os.cpu_count() or 1) - 2)


class Exam:
    """Mette `genomes` alla prova su tutte le prove della corsa (`battery`): `arenas` arene per
    prova, `clones` copie di ogni rete per arena.

    Ogni arena è un lavoro a sé. Con `workers` > 1 i lavori girano in processi
    separati, in parallelo e senza togliere tempo a chi li ha lanciati; con
    `workers` = 1 si fanno qui, a pezzi (`run(budget)`), come `Trial`. Il
    risultato è lo stesso. Non tocca nulla dell'evoluzione: farlo o non farlo
    non cambia la corsa.
    """

    def __init__(self, run_cfg: Config, genomes: Genome, generation: int, ids=None,
                 arenas: int = 8, clones: int = 5, workers: int | None = None, lesion: str | None = None):
        self.lesion = lesion
        run_cfg, genomes = lesioned(run_cfg, genomes, lesion)
        self.run_cfg, self.genomes, self.generation = run_cfg, genomes, generation
        self.tests = battery(run_cfg)
        self.ids = None if ids is None else [int(i) for i in ids]
        self.arenas, self.clones = arenas, clones
        self.todo = [(i, k) for i in range(len(self.tests)) for k in range(arenas)]
        self.wins = np.zeros((len(self.tests), len(genomes)), dtype=int)  # per prova e rete: copie arrivate
        self.done = 0
        self.started = time.perf_counter()
        self.seconds = 0.0
        self.workers = default_workers() if workers is None else workers
        self._pool, self._futures, self._trial = None, {}, None
        self._thread, self._error = None, None
        if run_cfg.engine == "gpu" and _gpu_ready():
            # Sulla GPU tutte le arene di una prova girano nello stesso lotto,
            # in un filo a parte: chi ha lanciato l'esame (l'interfaccia, la
            # corsa) non si ferma ad aspettarlo. Niente processi separati: ognuno
            # occuperebbe la scheda video per conto suo.
            self.workers = 0
            self.todo = []
            self._thread = threading.Thread(target=self._run_on_gpu, name="esame", daemon=True)
            self._thread.start()
            return
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
        return self.done >= len(self.tests) * self.arenas

    @property
    def progress(self) -> float:
        """Quota fatta, da 0 a 1."""
        return self.done / (len(self.tests) * self.arenas)

    def _record(self, i: int, wins: np.ndarray) -> None:
        self.wins[i] += wins
        self.done += 1
        if self.finished:
            self.seconds = time.perf_counter() - self.started
            self.close()

    def _run_on_gpu(self) -> None:
        try:
            for i, t in enumerate(self.tests):
                trial = Trial(exam_config(self.run_cfg, t), self.genomes, self.arenas, self.clones,
                              seed=EXAM_SEED + 10_000 * i, task=t.task())
                trial.run(float("inf"))
                self.wins[i] = trial.wins
                self.seconds = time.perf_counter() - self.started
                self.done += self.arenas
        except Exception as err:  # riportato a chi raccoglie il risultato
            self._error = err

    def run(self, budget: float = float("inf")) -> None:
        """In parallelo (processi o, sulla GPU, un filo a parte): raccoglie le
        arene finite (con `budget` infinito aspetta tutte). Qui: avanza per
        circa `budget` secondi di calcolo."""
        if self._thread is not None:
            self._thread.join(None if budget == float("inf") else 0.0)
            if self._error is not None:
                raise self._error
            return
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
                t = self.tests[i]
                self._trial = (i, Trial(exam_config(self.run_cfg, t), self.genomes, 1, self.clones,
                                        seed=EXAM_SEED + 10_000 * i + k, task=t.task()))
            i, trial = self._trial
            trial.step()
            if trial.finished:
                self._trial = None
                self._record(i, trial.wins)

    def close(self) -> None:
        """Ferma i processi (anche a esame non finito: il risultato si perde).
        Un esame sulla GPU finisce da solo in sottofondo e viene scartato."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    @property
    def rates(self) -> dict[str, np.ndarray]:
        """Per prova: quota di prove completate da ogni rete."""
        return {t.key: self.wins[i] / (self.arenas * self.clones) for i, t in enumerate(self.tests)}

    def result(self) -> dict:
        """Il risultato, pronto per il registro della corsa (solo tipi JSON)."""
        rates = self.rates
        scores = {k: float(r.mean()) for k, r in rates.items()}
        return {
            "generation": self.generation,
            "version": LEVELS_VERSION if self.tests is LEVELS else VERSION,
            "lesion": self.lesion,
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
