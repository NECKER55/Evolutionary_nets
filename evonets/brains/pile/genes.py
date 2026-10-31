"""I canali dei sensi, la struttura e i geni comuni agli agenti dell'arena delle pile."""

from dataclasses import dataclass, field, fields

import numpy as np

from ...arenas.pile.world import (
    EYES,
    FEATURES,
    N_FEATURES,
    N_FIN,
    N_FOUT,
    N_PROGRAMS,
    N_SENSES,
    PAIN,
    SECTORS,
    TASTE,
    default_motor,
)

FAST = (*range(EYES.start, EYES.stop), TASTE, PAIN)  # canali con la loro versione "cosa è cambiato": occhi, gusto, dolore
N_HEADS = N_PROGRAMS + 1  # E sceglie un programma, e se mordere (l'ultima "testa")
BITE = N_PROGRAMS
FOVEA = tuple(f"oggetto {f}" for f in FEATURES) + ("oggetto vicino", "oggetto a sinistra", "oggetto a destra",
                                                   "oggetto nuovo")
N_A = N_SENSES + len(FAST) + len(FOVEA)
IDENTITY = N_FEATURES  # i primi canali della fovea dicono che cos'è l'oggetto; gli altri dove sta
MAP = 9  # posizioni della mappa di priorità (neuroni-direzione)
APPROACH, AWAY, LEAVE, EXPLORE = range(4)


def fovea_of(feat, near, ang, has, new, xp=np):
    """I canali della fovea: le caratteristiche dell'oggetto attento (N, F),
    poi vicinanza, lato e "nuovo". `has`: se c'è un oggetto attento."""
    side = SECTORS[-1]
    return xp.concatenate([feat * has[:, None], xp.stack([
        xp.where(has, near, 0.0), xp.where(has, xp.clip(-ang / side, 0.0, 1.0), 0.0),
        xp.where(has, xp.clip(ang / side, 0.0, 1.0), 0.0), new], 1)], 1)


class Interface:
    """A: fissa. Per ogni canale il valore ("com'è"); per occhi, gusto e
    dolore anche quanto è appena salito ("cosa è cambiato", τ dal gene
    `tau_a`); poi la fovea."""

    def __init__(self, n: int):
        self.slow = np.zeros((n, len(FAST)))

    def __call__(self, raw: np.ndarray, fovea: np.ndarray, keep: np.ndarray) -> np.ndarray:
        x = raw[:, FAST]
        fast = np.clip(x - self.slow, 0.0, 1.0)
        self.slow = keep[:, None] * self.slow + (1.0 - keep[:, None]) * x
        return np.concatenate([raw, fast, fovea], axis=1)

    def reset(self, who):
        self.slow[who] = 0.0


@dataclass
class Genes:
    """Le costanti del cervello, una per essere: nella fase A questi valori per
    tutti, nella fase B le tara l'evoluzione. Nel genoma: in scala
    logaritmica i tempi e le velocità (`LOG`), tra 0 e 1 le quote (`UNIT`),
    con il segno `gate_bias`, le altre non negative."""

    # Velocità di apprendimento.
    eta_b: float = 0.05  # B, per secondo
    eta_c: float = 0.1  # C, per passo (la sorpresa è in unità di quella solita): in fretta, poi satura (c_max)
    eta_e: float = 0.01  # E: lento (le abitudini)
    eta_v: float = 0.5
    eta_vc: float = 5.0  # V: la parte che legge C impara in fretta (il valore di un oggetto in un colpo)
    eta_l: float = 0.01  # L, per secondo di riposo
    eta_o: float = 1.5  # O, per episodio (× l'incertezza della caratteristica)
    eta_gate: float = 0.003  # il cancello di D
    eta_predict: float = 0.5  # B: la previsione del prossimo ingresso, per secondo
    # Tempi (s).
    tau_a: float = 0.3  # A: "cosa è cambiato"
    tau_b: float = 0.2  # B: reazione
    tau_d: float = 4.0  # D: quanto dura il ricordo
    horizon: float = 4.0  # V dell'attore-critico (un critico solo): quanto lontano guarda
    h_short: float = 2.0  # V del cervello a sezioni: i tre orizzonti
    h_mid: float = 8.0
    h_long: float = 30.0
    trace_actor: float = 2.0  # traccia di eleggibilità di E, C, del cancello e della salienza
    trace_critic: float = 1.0  # traccia del critico (λ)
    forget: float = 180.0  # C: per dimenticare
    win_window: float = 30.0  # B: su quanto tempo si conta chi vince (la "coscienza")
    u_fast: float = 20.0  # U: "adesso"
    u_slow: float = 200.0  # U e soglia: "il solito"
    decide_every: float = 0.5  # E e l'attenzione: ogni quanto si rimettono in gara le scelte
    inhibit_time: float = 3.0  # attenzione: quanto si ignora ciò che si è appena lasciato
    new_tau: float = 0.5  # attenzione: quanto dura il "nuovo oggetto"
    novelty_tau: float = 0.5  # attenzione: quanto resta nuovo un oggetto appena comparso
    # Scelte e soglie.
    temperature: float = 0.5  # E: esplorazione
    stick: float = 1.5  # E: il programma in corso parte avvantaggiato
    surprise_rate: float = 1.0  # U: quanto le sorprese accelerano C
    surprise_explore: float = 1.0  # U, solo senza arbitraggio ("R"): quanto alzano l'esplorazione
    surprise_boost: float = 2.0  # B impara (1 + questo · |δ|) volte di più
    predict: float = 0.5  # B: peso della previsione del prossimo ingresso
    conscience: float = 2.0  # B: quanto un prototipo che vince poco viene aiutato
    gate_event: float = 8.0  # D: quanto un oggetto nuovo apre il cancello
    gate_bias: float = -6.0  # D: chiuso a riposo (un cancello socchiuso cancellerebbe il ricordo)
    context: float = 0.1  # C: peso del contesto (B) rispetto all'oggetto guardato (1)
    memory: float = 0.5  # C: peso dell'oggetto ricordato in D
    c_threshold: float = 0.4  # C: sotto questo ingresso nessun neurone si accende (niente da vedere né ricordare)
    c_sparse: float = 0.05  # C: quota di neuroni che l'inibizione lascia accesi
    c_max: float = 0.3  # C: ogni voto satura qui (conta l'ultima esperienza)
    rest_energy: float = 0.6  # L: "riposo" = carichi oltre questa quota e fermi a ricaricarsi
    tag: float = 0.5  # tracce strette: si marca solo da neuroni accesi sopra questo
    floor: float = 1.0  # soglia: insegna solo la parte di |δ| sopra questo × la sorpresa solita
    arb_fast: float = 2.0  # arbitraggio: il voto di C vale fino a 1 + questo
    arb_slow: float = 0.5  # arbitraggio: quelli di E ed L fino a 1 - questo
    att_shift: float = 0.7  # arbitraggio: ciò che O dice all'attenzione fino a 1 - questo
    # Valore legato al corpo.
    need: float = 0.8  # il piacere di una pila va da (1 - need) da sazi a (1 + need) affamati
    effort: float = 1.0  # quanto pesa l'energia spesa (1: spendere una pila vale quanto mangiarne una)
    critic_b: float = 0.2  # V: peso di B tra gli ingressi del critico
    critic_mix: float = 0.0  # quanto pesa la sorpresa dei critici rispetto al valore grezzo (0: per ora peggiora)
    observe: float = 1.0  # V: quanto si impara dalla vita del maestro (× eta_v), rispetto alla propria
    # Salienza innata.
    sal_near: float = 2.0
    sal_novelty: float = 1.5
    sal_stick: float = 1.0
    sal_inhibit: float = 4.0
    sal_temperature: float = 0.4
    sal_unc: float = 0.5  # attenzione: quanto attirano le caratteristiche incerte (Pearce e Hall)
    sal_value: float = 2.0  # attenzione: quanto attira ciò che vale (O)
    sal_import: float = 0.5  # attenzione: quanto attira ciò che conta, buono o cattivo (O)
    trace_o: float = 0.5  # O: quanto resta "in mente" ciò che si è guardato o preso (s)
    assoc_rate: float = 0.3  # O: quanto in fretta l'incertezza segue le sorprese, per episodio
    assoc_slow: float = 0.03  # O: la sorpresa "solita", per episodio (più lenta)
    k_rate: float = 0.3  # K: quanto pesa l'ultimo episodio nella prova di ogni contesto
    k_stick: float = 0.15  # K: di quanto un altro contesto deve prevedere meglio per prendere il posto
    k_new: float = 1.5  # K: si recluta un contesto nuovo se tutti sbagliano oltre questo × l'errore solito
    k_wait: float = 4.0  # K: episodi minimi tra un cambio di contesto e un reclutamento
    pain_weight: float = 2.0  # O: quanto pesa il dolore rispetto al piacere nel valore (la perdita pesa di più)

    LOG = ("eta_b", "eta_c", "eta_e", "eta_v", "eta_vc", "eta_l", "h_short", "h_mid", "h_long", "eta_o", "eta_gate", "eta_predict", "tau_a", "tau_b",
           "tau_d", "horizon", "trace_actor", "trace_critic", "forget", "win_window", "u_fast", "u_slow",
           "decide_every", "inhibit_time", "new_tau", "novelty_tau", "temperature", "c_threshold", "c_max",
           "sal_temperature", "observe", "trace_o")
    UNIT = ("rest_energy", "tag", "c_sparse", "k_rate", "arb_slow", "att_shift", "memory", "need", "critic_mix", "assoc_rate", "assoc_slow")
    SIGNED = ("gate_bias",)


GENE_NAMES = tuple(f.name for f in fields(Genes))
INNATE_CAP = 0.2  # tetto dei valori innati legati ai sensi (per secondo di senso pieno)
DELTA_MAX = 3.0  # δ satura qui
WEIGHT_MAX = 5.0  # nessun peso imparato oltre questo
VOLATILITY_MAX = 4.0  # O: al massimo tante volte più veloce quando il mondo è appena cambiato
RATE_MAX_O = 1.9  # O: passo massimo per episodio (con il passo normalizzato, da 2 in su si diverge)
OBS_AVG = 0.1  # quanto pesa ogni episodio del maestro nella media di quanto si impara ancora da lui
N_MOTOR = N_FOUT * N_FIN
# Il genoma: prima la parte comune (le costanti, i valori innati per ogni
# senso, i pesi di F), poi i blocchi propri di ogni cervello (`Agent.blocks`):
# i pesi di partenza e le connessioni, che si ereditano (ciò che si impara
# durante la vita no).
N_BASE = len(GENE_NAMES) + N_SENSES + N_MOTOR
INNATE = slice(len(GENE_NAMES), len(GENE_NAMES) + N_SENSES)
MOTOR = slice(INNATE.stop, N_BASE)


def default_genome() -> np.ndarray:
    """La parte comune del genoma, con i valori di partenza."""
    g = Genes()
    v = [np.log(getattr(g, k)) if k in Genes.LOG else getattr(g, k) for k in GENE_NAMES]
    return np.concatenate([np.asarray(v, dtype=float), np.zeros(N_SENSES), default_motor().ravel()])


def decode(genome, xp=np) -> dict:
    """(N, geni) -> {nome: (N,)}, "innate" (N, N_SENSES) e "motor" (N, 4, 14), nei loro limiti."""
    out = {}
    for k, name in enumerate(GENE_NAMES):
        v = genome[:, k]
        if name in Genes.LOG:
            out[name] = xp.exp(v)
        elif name in Genes.UNIT:
            out[name] = xp.clip(v, 0.0, 1.0)
        elif name in Genes.SIGNED:
            out[name] = v
        else:
            out[name] = xp.maximum(v, 0.0)
    out["innate"] = xp.clip(genome[:, INNATE], -INNATE_CAP, INNATE_CAP)
    out["motor"] = genome[:, MOTOR].reshape(-1, N_FOUT, N_FIN)
    return out


def mutate(genome: np.ndarray, rng: np.random.Generator, sigma: float = 0.1) -> np.ndarray:
    g = genome + rng.normal(0.0, sigma, genome.shape)
    g[:, INNATE] = np.clip(g[:, INNATE], -INNATE_CAP, INNATE_CAP)
    return g


def inherit(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Il figlio ha il genoma di uno dei due genitori (a caso), intero: nessun
    mescolamento gene per gene, che spezzerebbe i blocchi (un prototipo di B
    metà del padre e metà della madre non rappresenta più niente)."""
    return np.where(rng.random((a.shape[0], 1)) < 0.5, a, b)


def value_of(felt, energy, p: dict, xp=np):
    """Quanto vale ciò che il corpo ha sentito in questo passo. `felt` (n, 3):
    piacere (energia presa da una pila buona, una pila = 1), dolore, energia
    spesa (stessa scala). Il piacere cresce con la fame; spendere energia
    dispiace."""
    hunger = 1.0 - energy
    return (felt[:, 0] * (1.0 + p["need"] * (2.0 * hunger - 1.0)) - felt[:, 1] - p["effort"] * felt[:, 2])


@dataclass
class Sizes:
    """La struttura (fissa): quanti neuroni e quante connessioni; e le lesioni."""

    b: int = 16  # prototipi di B
    b_winners: int = 3
    c: int = 256  # neuroni di C (molti più degli ingressi, come le cellule di Kenyon)
    c_fan_in: int = 6
    k: int = 4  # contesti (un neurone ciascuno, per ora)
    lesions: frozenset = field(default_factory=frozenset)
