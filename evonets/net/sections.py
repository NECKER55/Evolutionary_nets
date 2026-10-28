"""Il cervello a sezioni (docs/SPEC_CERVELLO_SEZIONI.md) e i due agenti di
confronto per l'arena delle pile (docs/SPEC_ARENA_PILE.md).

Si fissa la struttura, non il contenuto: le sezioni, chi parla con chi, a che
velocità cambia ciascuna, con che famiglia di regole. Nessun neurone nasce
con un significato: i pesi che imparano partono da zero o a caso, e ciò che
rappresentano emerge (si misura con `selectivity` e le lesioni).

- O valore degli oggetti (come la corteccia orbitofrontale e l'amigdala):
  per ogni caratteristica percepibile (`sim.pile.FEATURES`: colore,
  grandezza, se si muove), più uno "sfondo", quanto piacere e quanto dolore
  porta di solito, separati. Impara dagli esiti, non da ogni passo: un
  episodio comincia quando il corpo sente piacere o dolore e finisce quando
  smette; alla fine, Rescorla-Wagner con le caratteristiche in competizione
  sulla traccia di ciò che si aveva in mente prima dell'esito (fovea e D,
  `trace_o` s): il merito va alla caratteristica che lo prevede, le altre
  restano a zero ("blocking"). Il valore non dipende da quanto bene ci si
  comporta (conta solo com'è andata quando è successo qualcosa). Ogni
  caratteristica ha la sua incertezza (Pearce e Hall: quanto i suoi esiti
  sorprendono di recente), che accelera l'apprendimento e attira lo
  sguardo. Uscite: per ogni oggetto il valore (piacere × fame di adesso −
  dolore: da sazi il cibo vale meno senza imparare niente), l'importanza
  (piacere + dolore: conta, buono o cattivo che sia) e l'incertezza; e due
  neuroni, "quanto vale ciò che guardo" e "quanto vale ciò che ho preso"
  (D), letti dal critico e da E. I valori di partenza sono geni.
- Attenzione (S): una mappa di priorità fatta di posizioni, come il
  collicolo superiore: `MAP` neuroni-direzione che coprono il campo visivo,
  ognuno con l'oggetto più vicino nella sua direzione (un oggetto nuovo
  accende la sua posizione: non nasce nessun neurone). S non impara: ogni
  neurone somma la salienza innata (vicino, appena comparso, un po' di
  inerzia, inibizione del ritorno) e ciò che O dice dell'oggetto: il valore
  (gene `sal_value`: attira il buono), l'importanza (`sal_import`: attira ciò
  che conta, anche il pericolo) e l'incertezza (`sal_unc`). Vince una
  posizione (con un po' di caso); l'oggetto attento è il suo.
- A interfaccia (fissa): sensi grezzi -> canali "com'è" e "cosa è cambiato",
  più la fovea: colore, vicinanza e lato dell'oggetto attento.
- B nucleo stabile: prototipi in competizione (Hebb competitivo, come il
  k-means, con la "coscienza" di DeSieno perché nessuno resti muto), lenti;
  impara di più nei momenti di sorpresa (|δ|) e verso ciò che viene dopo
  (previsione del prossimo ingresso).
- D memoria di lavoro: tiene l'ultimo oggetto guardato (una copia della
  fovea, τ di secondi), anche quando non si vede più: "che cosa ho appena
  preso" quando arriva il gusto o il dolore. Un cancello decide quando
  scrivere (innato: un oggetto nuovo nella fovea), e impara con δ.
- C memoria veloce: ingresso fisso, casuale e sparso (come le cellule di
  Kenyon del corpo a fungo); si accendono tutti i neuroni sopra soglia (non
  un numero fisso: con "i 4 più attivi" tra tanti pari merito vinceva il
  caso, e ogni incontro con lo stesso oggetto accendeva neuroni diversi);
  impara solo
  l'uscita verso E, in fretta, e dimentica in minuti. Legge soprattutto
  che cos'è l'oggetto (il colore nella fovea e in D, non dove sta), più un
  po' di contesto (B, con peso `context`): così lo stesso oggetto accende
  gli stessi neuroni a ogni incontro, e ciò che si impara una volta vale la
  volta dopo (con tutto lo stato di B, o anche con la vicinanza e il lato
  dell'oggetto, ogni incontro aveva un codice diverso).
- L memoria lunga: la stessa uscita di C, cento volte più lenta, che si
  avvicina a C solo a riposo: ciò che si inverte di continuo si annulla.
- E selezione (legge B, D e i due neuroni di O): programmi in competizione (softmax con rumore = esplorazione);
  attore: Δw = η · δ · traccia di (azione − attesa) × ingresso.
- V critico: TD, tre critici con orizzonti diversi (breve, medio, lungo: in
  natura zone e neuroni a dopamina diversi scontano il futuro in modo
  diverso), δ = la media delle loro sorprese. Leggono ciò che prevede il
  valore in modo stabile: che cosa si ha davanti (fovea), che cosa si è
  appena preso (D), la situazione riconosciuta (C, che impara in fretta: il
  valore di un oggetto si impara in un colpo), l'energia, i due neuroni di
  O; B con poco peso.
- Quanto pesa il critico è un gene (`critic_mix`): la sorpresa che insegna è
  una via di mezzo tra il valore grezzo e quella dei critici.
- Il valore è legato al corpo: il piacere di una pila cresce con la fame
  (`need`), l'energia spesa (muoversi, mordere) dà un dispiacere (`effort`).
- U incertezza: le sorprese vere (sopra la soglia) di adesso rispetto al
  solito; guida l'arbitraggio (sotto) e accelera C.
- E sceglie anche se mordere (sì o no, a ogni decisione): si mangia solo
  mordendo, e mordere costa un po'. C ed L votano anche sul morso.
- F motore: un circuito innato di neuroni (`sim.pile.motor_net`) dai
  programmi e dall'oggetto attento a 4 neuroni motori (gira a sinistra, a
  destra, avanti, indietro); non impara nella vita, i suoi pesi sono geni.

Due regole sull'apprendimento con δ, solo nel cervello a sezioni:
- tracce strette: una sinapsi di E o del cancello di D si marca solo se il
  neurone da cui parte è acceso sopra `tag` (in natura si marca solo dove il
  neurone si attiva davvero);
- soglia adattiva: insegna solo la parte di δ che supera `floor` volte la
  sorpresa media degli ultimi minuti (la dopamina ha un livello di base e
  adatta la sua scala: Tobler, Fiorillo, Schultz 2005). Il critico impara
  invece da tutto δ: è il suo errore.

Arbitraggio (Daw, Niv, Dayan 2005; l'incertezza inattesa e la noradrenalina,
Yu e Dayan 2005): quando le sorprese sono tante (U alta) il mondo è cambiato,
le abitudini non sono affidabili e la decisione passa alla memoria veloce:
il voto di C conta di più, quelli di E ed L di meno, e la salienza imparata
meno di quella innata (ci si guarda attorno). Quando le sorprese calano,
tutto torna come prima. U accelera solo C.

Tutte le costanti del cervello sono geni (`Genes`, una per essere): tempi,
velocità, soglie, pesi della salienza innata, forza dell'arbitraggio. Nella
fase A valgono i valori di partenza per tutti; nella fase B li tara
l'evoluzione. Restano fissi solo la struttura (`Sizes`: quanti neuroni,
quante connessioni) e i limiti di sicurezza (`DELTA_MAX`, `WEIGHT_MAX`,
`INNATE_CAP`).

Il maestro (apprendimento per osservazione): in ogni recinto uno degli
esseri (il primo) può essere un maestro, guidato a mano come `Guided` (sa
quali pile sono buone). Gli altri gli "guardano dentro": quando un suo
episodio finisce, O di ogni allievo applica la stessa regola con la traccia
del maestro (ciò che lui aveva in mente) e il suo esito (velocità
`eta_o` × `observe`). Si impara che cosa fa bene o male a lui, non il
valore della sua situazione: è ciò che si trasferisce. Quanto c'è ancora da
imparare da lui: `obs_err`, la media di |errore| sui suoi episodi.

Lesioni (`lesions`): "B" (E, V, C, D leggono A invece di B), "D", "C", "L",
"U", "V" (niente critico: δ = ricompensa grezza), "S" (salienza solo innata),
"O" (niente valore degli oggetti: non impara e dice zero),
"R" (niente arbitraggio: U accelera E, aggiunge caso e non sposta i voti,
com'era prima).
"""

from dataclasses import dataclass, field, fields

import numpy as np

from ..sim.pile import (ENERGY, EYES, FEATURES, N_FEATURES, N_FIN, N_FOUT, N_PROGRAMS, N_SENSES, PAIN, SECTORS, TASTE,
                        default_motor)

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
    eta_c: float = 20.0  # C, per aggiornamento (δ è già per passo): in un colpo, poi satura (c_max)
    eta_e: float = 1.0
    eta_v: float = 0.5
    eta_vc: float = 5.0  # V: la parte che legge C impara in fretta (il valore di un oggetto in un colpo)
    eta_l: float = 0.01  # L, per secondo di riposo
    eta_o: float = 1.5  # O, per episodio (× l'incertezza della caratteristica)
    eta_gate: float = 0.5  # il cancello di D
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
    c_threshold: float = 0.9  # C: si accendono tutti i neuroni sopra questa soglia
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

    LOG = ("eta_b", "eta_c", "eta_e", "eta_v", "eta_vc", "eta_l", "h_short", "h_mid", "h_long", "eta_o", "eta_gate", "eta_predict", "tau_a", "tau_b",
           "tau_d", "horizon", "trace_actor", "trace_critic", "forget", "win_window", "u_fast", "u_slow",
           "decide_every", "inhibit_time", "new_tau", "novelty_tau", "temperature", "c_threshold", "c_max",
           "sal_temperature", "observe", "trace_o")
    UNIT = ("rest_energy", "tag", "arb_slow", "att_shift", "memory", "need", "critic_mix", "assoc_rate")
    SIGNED = ("gate_bias",)


GENE_NAMES = tuple(f.name for f in fields(Genes))
INNATE_CAP = 0.2  # tetto dei valori innati legati ai sensi (per secondo di senso pieno)
DELTA_MAX = 3.0  # δ satura qui
WEIGHT_MAX = 5.0  # nessun peso imparato oltre questo
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
    c: int = 64  # neuroni di C
    c_fan_in: int = 4
    lesions: frozenset = field(default_factory=frozenset)


def priority_map(objs, MAPN, xp=np):
    """Per ogni posizione della mappa: c'è un oggetto, quale (il più vicino in
    quella direzione), le sue caratteristiche e la sua vicinanza.
    -> (has (n, P), obj (n, P), feat (n, P, F), near (n, P))."""
    vis, near, ang = objs["vis"], objs["near"], objs["ang"]
    width = (SECTORS[-1] - SECTORS[0]) / MAPN
    where_ = xp.clip(xp.floor((ang - SECTORS[0]) / width), 0, MAPN - 1).astype(int)
    in_pos = vis[:, :, None] & (where_[:, :, None] == xp.arange(MAPN)[None, None, :])
    obj = xp.argmax(xp.where(in_pos, near[:, :, None], -1.0), axis=1)  # (n, P)
    has = in_pos.any(1)
    feat = xp.take_along_axis(objs["feat"], obj[:, :, None], axis=1)
    near_p = xp.take_along_axis(near, obj, axis=1) * has
    return has, obj, feat, near_p


# --- O: il valore degli oggetti (vedi la docstring del modulo) ---------------------


def o_read(o_pl, o_pa, o_assoc, feat, energy, p: dict, xp=np):
    """Ciò che O dice di oggetti con caratteristiche `feat` (n, ..., F):
    (valore, importanza, incertezza), ognuno (n, ...). Lo sfondo (l'ultimo
    peso) non entra: è ciò che capita senza un oggetto."""
    F = feat.shape[-1]
    need = 1.0 + p["need"] * (2.0 * (1.0 - energy) - 1.0)  # la fame di adesso
    value = o_pl[:, :F] * need[:, None] - o_pa[:, :F]
    weight = xp.maximum(o_pl[:, :F], 0.0) + xp.maximum(o_pa[:, :F], 0.0)
    extra = (1,) * (feat.ndim - 2)
    shape = lambda w: w.reshape(w.shape[:1] + extra + w.shape[1:])
    return ((feat * shape(value)).sum(-1), (feat * shape(weight)).sum(-1), (feat * shape(o_assoc[:, :F])).sum(-1))


def o_learn(o_pl, o_pa, o_assoc, trace, pleasure, pain, rate, p: dict, xp=np):
    """Rescorla-Wagner alla fine di un episodio: `trace` (n, F + 1) ciò che si
    aveva in mente (l'ultimo è lo sfondo, sempre 1), `pleasure` e `pain` (n,)
    l'esito, `rate` (n,) la velocità (zero: non si impara). Le
    caratteristiche competono sulla stessa previsione. -> (o_pl, o_pa,
    o_assoc, |errore|)."""
    rate = xp.minimum(rate, RATE_MAX_O)  # oltre, la regola diventa instabile
    err_pl = pleasure - (o_pl * trace).sum(1)
    err_pa = pain - (o_pa * trace).sum(1)
    # Diviso per quanta traccia c'è: con molte caratteristiche accese insieme
    # la correzione totale supererebbe l'errore, e i pesi rimbalzerebbero.
    step = rate[:, None] * o_assoc * trace / (1.0 + (trace**2).sum(1))[:, None]
    o_pl = xp.clip(o_pl + step * err_pl[:, None], -WEIGHT_MAX, WEIGHT_MAX)
    o_pa = xp.clip(o_pa + step * err_pa[:, None], -WEIGHT_MAX, WEIGHT_MAX)
    size = xp.minimum(xp.abs(err_pl) + xp.abs(err_pa), 1.0)
    on = (rate > 0)[:, None]
    o_assoc = xp.where(on, o_assoc + p["assoc_rate"][:, None] * trace * (size[:, None] - o_assoc), o_assoc)
    return o_pl, o_pa, o_assoc, xp.abs(err_pl) + xp.abs(err_pa)


def o_episodes(st: dict, felt, xp=np) -> tuple[dict, object]:
    """Gli episodi: comincia quando il corpo sente piacere o dolore (`felt`
    (n, 3)), finisce quando smette. All'inizio si fotografa la traccia (ciò che
    si aveva in mente prima dell'esito). -> (stato nuovo, finito (n,)); i
    totali dell'episodio finito restano in `ep_pl`, `ep_pa`, `ep_e` di `st`."""
    active = felt[:, 0] + felt[:, 1] > 1e-6
    done = st["ep_on"] & ~active
    start = active & ~st["ep_on"]
    n = felt.shape[0]
    snap = xp.concatenate([st["o_e"], xp.ones((n, 1))], 1)
    keep = st["ep_on"] & active
    new = {"ep_on": active,
           "ep_e": xp.where(start[:, None], snap, st["ep_e"]),
           "ep_pl": xp.where(keep, st["ep_pl"], 0.0) + felt[:, 0] * active,
           "ep_pa": xp.where(keep, st["ep_pa"], 0.0) + felt[:, 1] * active}
    return new, done


def o_step(st: dict, felt, energy, master, on, p: dict, les, xp=np) -> dict:
    """Un passo di O, prima dell'attenzione: chiude gli episodi finiti e
    impara dai propri e (chi osserva, `on` (n,)) da quelli del maestro del
    proprio recinto (`master` (n,): il suo indice). -> gli array cambiati."""
    if "O" in les:
        return {}
    new, done = o_episodes(st, felt, xp)
    pl, pa, e = st["ep_pl"], st["ep_pa"], st["ep_e"]
    o_pl, o_pa, o_assoc, err = o_learn(st["o_pl"], st["o_pa"], st["o_assoc"], e, pl, pa, p["eta_o"] * done, p, xp)
    out = dict(new, o_err=xp.where(done, err, st["o_err"]))
    # Dal maestro: la stessa regola, con ciò che lui aveva in mente e il suo esito.
    seen = on & done[master]
    o_pl, o_pa, o_assoc, err_m = o_learn(o_pl, o_pa, o_assoc, e[master], pl[master], pa[master],
                                         p["eta_o"] * p["observe"] * seen, p, xp)
    out.update(o_pl=o_pl, o_pa=o_pa, o_assoc=o_assoc,
               obs_err=xp.where(seen, st["obs_err"] + OBS_AVG * (err_m - st["obs_err"]), st["obs_err"]))
    return out


def o_trace(o_e, fov, m, p: dict, dt: float, xp=np):
    """La traccia di O, dopo l'attenzione e D: ciò che si guarda o si ricorda
    di aver preso si accende, poi si spegne in `trace_o` secondi."""
    now = xp.maximum(fov[:, :IDENTITY], m[:, :IDENTITY])
    return xp.maximum(o_e * xp.exp(-dt / p["trace_o"])[:, None], now)


class Attention:
    """S: la mappa di priorità (vedi la docstring del modulo). L'oggetto
    attento è `target` (indice negli oggetti in vista, -1 = nessuno)."""

    def __init__(self, n: int, m: int, dt: float, rng: np.random.Generator):
        self.n, self.m, self.dt, self.rng = n, m, dt, rng
        self.target = np.full(n, -1)
        self.inhib = np.full(n, -1)
        self.inhib_t = np.zeros(n)
        self.prev_vis = np.zeros((n, m), dtype=bool)
        self.novelty = np.zeros((n, m))
        self.new = np.zeros(n)
        self.prob = np.zeros((n, MAP))  # la mappa: probabilità di ogni posizione (per lo schermo)
        self.has_pos = np.zeros((n, MAP), dtype=bool)
        self.parts = np.zeros((n, MAP, 4))  # per lo schermo: innata, valore, importanza, incertezza

    def update(self, objs: dict, decide: np.ndarray, p: dict, extra=None, learned=None, o=None) -> None:
        """Sceglie l'oggetto (a ogni decisione, o subito se quello attento è
        sparito). `extra` (n, m): punteggio in più per oggetto (il guidato bara
        così). `o`: (o_pl, o_pa, o_assoc, energia), ciò che O sa; `learned` (n,):
        quanto conta valore e importanza (arbitraggio)."""
        dt, n = self.dt, self.n
        vis = objs["vis"]
        idx = np.arange(n)
        self.novelty = self.novelty * np.exp(-dt / p["novelty_tau"])[:, None] + (vis & ~self.prev_vis)
        self.prev_vis = vis.copy()
        self.inhib_t = np.maximum(self.inhib_t - dt, 0.0)
        lost = (self.target >= 0) & ~vis[idx, np.maximum(self.target, 0)]
        self.target = np.where(lost, -1, self.target)
        self.new *= np.exp(-dt / p["new_tau"])
        has, obj, feat, near = priority_map(objs, MAP)
        self.has_pos = has
        choose = (decide | (self.target < 0)) & has.any(1)
        nov = np.take_along_axis(self.novelty, obj, axis=1)
        innate = (p["sal_near"][:, None] * near + p["sal_novelty"][:, None] * nov
                  + p["sal_stick"][:, None] * (obj == self.target[:, None])
                  - p["sal_inhibit"][:, None] * ((obj == self.inhib[:, None]) & (self.inhib_t > 0)[:, None]))
        value = imp = unc = np.zeros_like(innate)
        if o is not None:
            value, imp, unc = o_read(o[0], o[1], o[2], feat, o[3], p)
            trust = np.ones(n) if learned is None else learned
            value = trust[:, None] * p["sal_value"][:, None] * value
            imp = trust[:, None] * p["sal_import"][:, None] * imp
            unc = p["sal_unc"][:, None] * unc
        self.parts = np.stack([innate, value, imp, unc], -1) * has[:, :, None]
        score = innate + value + imp + unc
        if extra is not None:
            score = score + np.take_along_axis(extra, obj, axis=1)
        z = np.where(has, score / p["sal_temperature"][:, None], -np.inf)
        z = z - np.where(has.any(1, keepdims=True), z.max(1, keepdims=True), 0.0)
        prob = np.where(has, np.exp(z), 0.0)
        prob /= np.maximum(prob.sum(1, keepdims=True), 1e-12)
        self.prob = prob
        if not choose.any():
            return
        place = (np.cumsum(prob, 1) < self.rng.random((n, 1))).sum(1).clip(0, MAP - 1)
        sel = np.where(choose, obj[idx, place], self.target)
        changed = choose & (sel != self.target)
        self.new = np.where(changed, 1.0, self.new)
        self.target = sel

    def leave(self, mask: np.ndarray, p: dict) -> None:
        """"Lascialo": l'oggetto attento viene ignorato per qualche secondo."""
        self.inhib = np.where(mask, self.target, self.inhib)
        self.inhib_t = np.where(mask, p["inhibit_time"], self.inhib_t)
        self.target = np.where(mask, -1, self.target)

    def seen(self, objs: dict):
        """(c'è, caratteristiche (n, F), vicinanza, angolo) dell'oggetto attento."""
        idx, t = np.arange(self.n), np.maximum(self.target, 0)
        has = self.target >= 0
        return has, objs["feat"][idx, t], objs["near"][idx, t], objs["ang"][idx, t]

    def fovea(self, objs: dict) -> np.ndarray:
        has, feat, near, ang = self.seen(objs)
        return fovea_of(feat, near, ang, has, self.new)

    def reset(self, who) -> None:
        for x, v in ((self.target, -1), (self.inhib, -1), (self.inhib_t, 0.0), (self.novelty, 0.0),
                     (self.prev_vis, False), (self.new, 0.0), (self.prob, 0.0), (self.has_pos, False),
                     (self.parts, 0.0)):
            x[who] = v


class Agent:
    """Ciò che serve all'arena: `step(raw, reward, objs)` -> programmi (N,),
    `att` l'attenzione (l'oggetto attento guida F), e `reborn(who, genomes)`
    quando un corpo rinasce."""

    name = "agente"

    def __init__(self, n: int, dt: float, seed: int = 0, genomes=None, objects: int = 1):
        self.n, self.dt = n, dt
        self.rng = np.random.default_rng(seed)
        self._given = None if genomes is None else np.array(genomes, dtype=float)
        self.genome = np.tile(default_genome(), (n, 1)) if genomes is None else self._given
        self.p = decode(self.genome)
        self.blocks = {}
        self.att = Attention(n, objects, dt, self.rng)
        self.clock = np.zeros(n)  # da quanto si è deciso l'ultima volta (ognuno ha il suo ritmo)
        self.program = np.full(n, EXPLORE)
        self.pi = np.full((n, N_PROGRAMS), 1.0 / N_PROGRAMS)
        self.bite = np.zeros(n, dtype=bool)  # si morde (deciso a ogni decisione)
        self.p_bite = np.full(n, 0.5)

    def _attach(self, specs) -> None:
        """I blocchi propri del cervello: (nome, forma, valori di partenza (n, ...))."""
        start, parts = N_BASE, []
        for name, shape, init in specs:
            size = int(np.prod(shape))
            self.blocks[name] = (slice(start, start + size), tuple(shape))
            parts.append(np.asarray(init, dtype=float).reshape(self.n, size))
            start += size
        if self._given is None:
            self.genome = np.concatenate([self.genome[:, :N_BASE]] + parts, 1)
        elif self._given.shape[1] != start:
            raise ValueError(f"genoma di {self._given.shape[1]} geni, questo cervello ne vuole {start}")
        self.p = decode(self.genome)

    def start(self, name: str, who=slice(None)) -> np.ndarray:
        """I valori di partenza del blocco `name` (dal genoma)."""
        sl, shape = self.blocks[name]
        g = self.genome[who, sl]
        return g.reshape((g.shape[0],) + shape)

    def _tick(self) -> np.ndarray:
        """(n,) chi decide in questo passo (ognuno ogni `decide_every`)."""
        self.clock += self.dt
        decide = self.clock >= self.p["decide_every"] - 1e-9
        self.clock[decide] = 0.0
        return decide

    def innate_reward(self, raw: np.ndarray) -> np.ndarray:
        """I valori innati legati ai sensi (per passo)."""
        return (self.p["innate"] * raw).sum(1) * self.dt

    def reborn(self, who: np.ndarray, genomes=None) -> None:
        if genomes is not None:
            self.genome[who] = genomes
            self.p = decode(self.genome)
        self.att.reset(who)
        self.bite[who] = False
        self.clock[who] = 0.0


class Guided(Agent):
    """Guidato a mano, per controllare il mondo: sa quali pile sono buone
    (`objs["good"]`). Si concentra sulla pila buona più
    vicina e ci va; se vede solo pile cattive le lascia ed esplora (chi lo usa
    lo fa andare verso la zona, che sa)."""

    name = "guidato"

    def __init__(self, n, dt, seed=0, genomes=None, objects: int = 1, batteries: int = 0):
        super().__init__(n, dt, seed, genomes, objects)
        self._attach(())
        self.batteries = batteries

    def step(self, raw, reward, objs):
        decide = self._tick()
        self.att.update(objs, decide, self.p, extra=np.where(objs["good"], 10.0, -10.0))
        has, _, near, _ = self.att.seen(objs)
        ok = has & objs["good"][np.arange(self.n), np.maximum(self.att.target, 0)]
        self.program = np.where(ok, APPROACH, EXPLORE)
        self.bite = ok & (near > 0.9)  # morde solo la buona, quando ce l'ha davanti
        self.p_bite = self.bite.astype(float)
        self.att.leave(has & ~ok, self.p)  # non è la buona: la lascia
        self.pi = np.eye(N_PROGRAMS)[self.program]
        return self.program


class _Learner(Agent):
    """L'attore (E) e il critico (V) comuni ai due agenti che imparano."""

    def _init_actor_critic(self, n_e: int, n_v: int, critics: int = 1):
        n = self.n
        self.w_e = np.zeros((n, N_HEADS, n_e))
        self.e_e = np.zeros((n, N_HEADS, n_e))
        self.w_v = np.zeros((n, critics, n_v))  # uno per orizzonte
        self.e_v = np.zeros((n, critics, n_v))
        self.v_prev = np.zeros((n, critics))
        self.value = np.zeros(n)  # la previsione (media dei critici)
        self.trust = np.zeros(n)  # quanto pesa il critico nella sorpresa che insegna (dal gene)
        self.fresh = np.ones(n, dtype=bool)  # appena nato: niente δ al primo passo
        self.program = self.rng.integers(0, N_PROGRAMS, n)
        self.delta = np.zeros(n)
        self.teach = np.zeros(n)  # la parte di δ che insegna (dopo la soglia)

    def _td(self, f_v, reward, horizons, eta, use_critic: bool = True) -> np.ndarray:
        """δ e l'aggiornamento dei critici (TD(λ), passo normalizzato).
        `horizons` (n, K): l'orizzonte di ogni critico; `eta` (n, ingressi): la
        velocità per ingresso. δ è la media delle sorprese dei K critici."""
        p, dt = self.p, self.dt
        gamma = np.exp(-dt / horizons)  # (n, K)
        value = (self.w_v * f_v[:, None, :]).sum(-1)
        if use_critic:
            deltas = np.where(self.fresh[:, None], 0.0, reward[:, None] + gamma * value - self.v_prev)
        else:
            deltas = np.repeat(reward[:, None], horizons.shape[1], 1)
        deltas = np.clip(deltas, -DELTA_MAX, DELTA_MAX)  # la dopamina satura
        norm = 1.0 + (f_v**2).sum(1)[:, None] + (self.e_v * f_v[:, None, :]).sum(-1)  # con ingressi costanti la traccia si accumula
        self.w_v += (deltas / norm)[:, :, None] * eta[:, None, :] * self.e_v
        self.e_v = (gamma * np.exp(-dt / p["trace_critic"])[:, None])[:, :, None] * self.e_v + f_v[:, None, :]
        self.v_prev = (self.w_v * f_v[:, None, :]).sum(-1)
        self.fresh[:] = False
        self.value = self.v_prev.mean(1)
        # Quanto pesa il critico (gene `critic_mix`): la sorpresa che insegna è
        # una via di mezzo tra il valore grezzo e la sorpresa dei critici. Con un
        # comportamento ancora casuale il critico non ha niente da prevedere e
        # aggiunge rumore; quando il comportamento diventa prevedibile, conviene.
        full = deltas.mean(1)
        self.trust = p["critic_mix"] if use_critic else 0.0 * full
        self.delta = reward + self.trust * (full - reward)
        return self.delta

    def _act(self, logits, f_e, teach, eta, temp, decide, f_tag=None):
        """Attore: impara con la parte di δ che insegna, sulla traccia; poi chi
        è al momento di decidere (`decide`, (n,)) sceglie. `f_tag`: gli
        ingressi che possono marcare."""
        p, dt = self.p, self.dt
        norm = 1.0 + (f_e**2).sum(1)
        self.w_e = np.clip(self.w_e + (eta * teach / norm)[:, None, None] * self.e_e, -WEIGHT_MAX, WEIGHT_MAX)
        self.e_e *= np.exp(-dt / p["trace_actor"])[:, None, None]
        if not decide.any():
            return self.program
        z = (logits[:, :N_PROGRAMS] + p["stick"][:, None] * np.eye(N_PROGRAMS)[self.program]) / temp[:, None]
        z -= z.max(1, keepdims=True)
        pi = np.exp(z)
        pi /= pi.sum(1, keepdims=True)
        u = self.rng.random((self.n, 1))
        program = (np.cumsum(pi, 1) < u).sum(1).clip(0, N_PROGRAMS - 1)
        # Il morso: sì o no (probabilità dalla sua testa).
        p_bite = 1.0 / (1.0 + np.exp(-np.clip(logits[:, BITE] / temp, -30, 30)))
        bite = self.rng.random(self.n) < p_bite
        self.program = np.where(decide, program, self.program)
        self.pi = np.where(decide[:, None], pi, self.pi)
        self.p_bite = np.where(decide, p_bite, self.p_bite)
        self.bite = np.where(decide, bite, self.bite)
        dev = np.concatenate([np.eye(N_PROGRAMS)[program] - pi, (bite - p_bite)[:, None]], 1) * decide[:, None]
        tag = f_e if f_tag is None else f_tag
        self.e_e += dev[:, :, None] * tag[:, None, :]  # la scelta rispetto all'attesa
        self._chose(dev)
        self.att.leave(decide & (self.program == LEAVE), p)
        return self.program

    def _chose(self, deviation):
        pass

    def _reset_actor_critic(self, who):
        for x in (self.w_e, self.e_e, self.w_v, self.e_v, self.v_prev, self.value, self.delta, self.teach, self.trust):
            x[who] = 0.0
        self.fresh[who] = True


class ActorCritic(_Learner):
    """Il riferimento: attore-critico classico (TD, Doya 2000) che legge
    direttamente l'interfaccia A (con la fovea). Stessi programmi, stessa
    regola di E e V del cervello a sezioni, salienza solo innata; senza B, C,
    D, L e U, senza tracce strette e senza soglia."""

    name = "attore-critico"

    def __init__(self, n, dt, seed=0, genomes=None, objects: int = 1):
        super().__init__(n, dt, seed, genomes, objects)
        self.a = Interface(n)
        self._init_actor_critic(N_A + 1, N_A + 1)
        self._attach((("e0", (N_HEADS, N_A + 1), np.zeros((n, N_HEADS, N_A + 1))),
                      ("v0", (1, N_A + 1), np.zeros((n, 1, N_A + 1)))))
        self._born(slice(None))

    def _born(self, who) -> None:
        """I pesi di partenza, dal genoma."""
        self.w_e[who], self.w_v[who] = self.start("e0", who), self.start("v0", who)

    def step(self, raw, reward, objs):
        p = self.p
        decide = self._tick()
        self.att.update(objs, decide, p)
        a = self.a(raw, self.att.fovea(objs), np.exp(-self.dt / p["tau_a"]))
        f = np.concatenate([a, np.ones((self.n, 1))], 1)
        r = value_of(reward, raw[:, ENERGY], p) + self.innate_reward(raw)
        delta = self._td(f, r, p["horizon"][:, None], np.repeat(p["eta_v"][:, None], f.shape[1], 1))
        self.teach = delta
        logits = (self.w_e * f[:, None, :]).sum(-1)
        return self._act(logits, f, delta, p["eta_e"], p["temperature"], decide)

    def reborn(self, who, genomes=None):
        super().reborn(who, genomes)
        self.a.reset(who)
        self._reset_actor_critic(who)
        self._born(who)


def c_threshold(fov, p: dict, xp=np):
    """La soglia di C: se si guarda qualcosa, ciò che si vede (sopra la soglia:
    solo i neuroni che lo ricevono); altrimenti ciò che si ricorda in D (con
    peso `memory`, la soglia scende in proporzione)."""
    seeing = fov[:, :IDENTITY].max(1) > 0
    return xp.where(seeing, p["c_threshold"], p["c_threshold"] * p["memory"])


def teacher_mask(N: int, group: int, xp=np):
    """(N,) il primo essere di ogni recinto (quello che può fare da maestro)."""
    return xp.arange(N) % group == 0


def teach_override(program, bite, has, ok, near, teacher, xp=np):
    """Il maestro decide come `Guided`: va verso la pila buona che guarda e la
    morde quando ce l'ha davanti; altrimenti esplora. -> (programma, morso, da lasciare)."""
    program = xp.where(teacher, xp.where(ok, APPROACH, EXPLORE), program)
    bite = xp.where(teacher, ok & (near > 0.9), bite)
    return program, bite, teacher & has & ~ok


class Sections(_Learner):
    """Il cervello a sezioni (vedi la docstring del modulo)."""

    name = "sezioni"

    def __init__(self, n, dt, seed=0, genomes=None, sizes: Sizes | None = None, objects: int = 1):
        super().__init__(n, dt, seed, genomes, objects)
        self.s = s = sizes or Sizes()
        self.les = set(s.lesions)
        self.a = Interface(n)
        nb = N_A if "B" in self.les else s.b
        self.nb = nb
        # B: prototipi di partenza (dal genoma: a caso, senza significato, nella
        # fase A; l'evoluzione può cambiarli nella fase B) e previsione.
        self.proto = np.zeros((n, s.b, N_A))
        self.pred = np.zeros((n, N_A, s.b))
        self.win_rate = np.full((n, s.b), 1.0 / s.b)
        self.state_b = np.zeros((n, nb))
        self.prev_b = np.zeros((n, nb))
        nf = len(FOVEA)
        # D: l'ultimo oggetto guardato (copia della fovea), e il cancello (pesi che imparano).
        self.mem = np.zeros((n, nf))
        self.w_gate = np.zeros((n, nf))
        self.e_gate = np.zeros((n, nf))
        # C: ingresso fisso, casuale e sparso (che cos'è l'oggetto guardato e quello
        # ricordato in D, più il contesto da B); uscite verso E (C veloce, L lenta).
        wiring = np.zeros((n, s.c, 2 * IDENTITY + nb))  # le connessioni di partenza: a caso, 4 per neurone
        for j in range(s.c):
            pick = np.argsort(self.rng.random((n, 2 * IDENTITY + nb)), axis=1)[:, : s.c_fan_in]
            np.put_along_axis(wiring[:, j], pick, 1.0, axis=1)
        self.c_in = np.zeros_like(wiring)
        self.u_c = np.zeros((n, N_HEADS, s.c))
        self.e_c = np.zeros((n, N_HEADS, s.c))
        self.u_l = np.zeros((n, N_HEADS, s.c))
        self.code = np.zeros((n, s.c))
        # U e soglia: medie della sorpresa.
        self.d_fast = np.zeros(n)
        self.d_slow = np.full(n, 1e-3)
        self.t_slow = np.full(n, 1e-4)
        self.unc = np.zeros(n)
        # O: piacere e dolore per caratteristica (più lo sfondo), incertezza,
        # traccia, l'episodio in corso; i due neuroni di uscita.
        F1 = N_FEATURES + 1
        self.o_pl, self.o_pa = np.zeros((n, F1)), np.zeros((n, F1))
        self.o_assoc = np.ones((n, F1))
        self.o_e = np.zeros((n, N_FEATURES))
        self.ep_on = np.zeros(n, dtype=bool)
        self.ep_pl, self.ep_pa = np.zeros(n), np.zeros(n)
        self.ep_e = np.zeros((n, F1))
        self.o_err = np.zeros(n)
        self.o_out = np.zeros((n, 2))  # quanto vale ciò che guardo, ciò che ho preso
        # E legge B, D, O e la costante. I tre critici leggono la fovea, D, C,
        # l'energia, B (con poco peso), O e la costante.
        self.n_v = nf + nf + s.c + 1 + nb + 2 + 1
        self._init_actor_critic(nb + nf + 2 + 1, self.n_v, critics=3)
        # Il maestro (vedi la docstring del modulo).
        self.teacher = np.zeros(n, dtype=bool)
        self.group = 1  # esseri per recinto (il maestro di i è il primo del suo recinto)
        self.obs_err = np.zeros(n)
        # Nel genoma: i pesi di partenza (prototipi di B, E, V, salienza) e le
        # connessioni di C (un gene per connessione possibile: c'è se supera 0,5).
        self._attach((("proto", (s.b, N_A), self.rng.random((n, s.b, N_A)) * 0.3),
                      ("c_in", wiring.shape[1:], wiring),
                      ("e0", (N_HEADS, nb + nf + 3), np.zeros((n, N_HEADS, nb + nf + 3))),
                      ("v0", (3, self.n_v), np.zeros((n, 3, self.n_v))),
                      ("o0", (2, F1), np.zeros((n, 2, F1)))))
        self._born(slice(None))

    def _born(self, who) -> None:
        """Pesi e connessioni di partenza, dal genoma."""
        self.proto[who] = self.start("proto", who)
        self.c_in[who] = (self.start("c_in", who) > 0.5).astype(float)
        self.w_e[who], self.w_v[who] = self.start("e0", who), self.start("v0", who)
        o0 = self.start("o0", who)
        self.o_pl[who], self.o_pa[who] = o0[:, 0], o0[:, 1]  # i valori innati delle cose (geni)

    # --- B ---------------------------------------------------------------------------

    def _b(self, a: np.ndarray, delta: np.ndarray) -> np.ndarray:
        s, p, dt = self.s, self.p, self.dt
        if "B" in self.les:
            self.state_b = a
            return a
        d = ((self.proto - a[:, None, :]) ** 2).sum(-1)  # (n, b): distanza da ogni prototipo
        d = d - p["conscience"][:, None] * (1.0 / s.b - self.win_rate)  # chi vince poco è aiutato
        order = np.argsort(d, axis=1)
        kth = np.take_along_axis(d, order[:, s.b_winners: s.b_winners + 1], 1)
        y = np.clip(1.0 - d / np.maximum(kth, 1e-9), 0.0, 1.0)  # i più vicini, gli altri a zero
        y = y / np.maximum(y.max(1, keepdims=True), 1e-9)
        # Imparare: i vincitori si avvicinano all'ingresso (e, con la previsione, a ciò che segue).
        err = a - np.einsum("nab,nb->na", self.pred, self.prev_b)  # quanto il passo prima aveva previsto male
        self.pred += (p["eta_predict"] * dt)[:, None, None] * err[:, :, None] * self.prev_b[:, None, :]
        rate = (p["eta_b"] * dt * (1.0 + p["surprise_boost"] * np.minimum(np.abs(delta), 1.0)))[:, None, None]
        self.proto += rate * y[:, :, None] * ((a[:, None, :] - self.proto) + p["predict"][:, None, None] * err[:, None, :])
        self.win_rate += (dt / p["win_window"])[:, None] * ((y > 0) / s.b_winners - self.win_rate)
        self.prev_b = self.state_b.copy()
        self.state_b += (dt / p["tau_b"])[:, None] * (y - self.state_b)
        return self.state_b

    # --- D ---------------------------------------------------------------------------

    def _d(self, fov: np.ndarray, teach: np.ndarray) -> np.ndarray:
        """Si scrive quando nella fovea entra un oggetto nuovo (innato) o quando
        il cancello lo ha imparato; altrimenti si tiene, anche a fovea vuota."""
        p, dt = self.p, self.dt
        if "D" in self.les:
            return np.zeros_like(fov)
        self.w_gate = np.clip(self.w_gate + (p["eta_gate"] * teach)[:, None] * self.e_gate, -2 * WEIGHT_MAX,
                              2 * WEIGHT_MAX)
        z = p["gate_event"] * fov[:, -1] + (self.w_gate * fov).sum(1) + p["gate_bias"]
        g = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        tagged = np.where(fov > p["tag"][:, None], fov, 0.0)
        self.e_gate = np.exp(-dt / p["trace_actor"])[:, None] * self.e_gate + (g * (1.0 - g))[:, None] * tagged
        keep = np.exp(-dt / p["tau_d"])[:, None]
        self.mem = (1.0 - g)[:, None] * keep * self.mem + g[:, None] * fov
        return self.mem

    # --- C ---------------------------------------------------------------------------

    def _c(self, b: np.ndarray, m: np.ndarray, fov: np.ndarray) -> np.ndarray:
        s, p = self.s, self.p
        if "C" in self.les:
            return np.zeros((self.n, s.c))
        h = np.concatenate([fov[:, :IDENTITY], p["memory"][:, None] * m[:, :IDENTITY], p["context"][:, None] * b], 1)
        z = np.einsum("ncj,nj->nc", self.c_in, h)
        self.code = (z >= c_threshold(fov, p)[:, None]).astype(float)
        return self.code

    def _chose(self, deviation):
        self.e_c += deviation[:, :, None] * self.code[:, None, :]  # solo i neuroni accesi al momento della scelta

    # --- Un passo --------------------------------------------------------------------

    def step(self, raw, reward, objs):
        p, dt = self.p, self.dt
        decide = self._tick()
        arbitrate = "R" not in self.les and "U" not in self.les
        k = np.clip(self.unc / 3.0, 0.0, 1.0) if arbitrate else np.zeros(self.n)
        teaching = self.teacher.any()
        master = np.arange(self.n) // self.group * self.group
        on = ~self.teacher & self.teacher[master]
        # O: gli episodi finiti (propri e del maestro) insegnano, prima di guardare.
        for k_, v in o_step(self.__dict__, reward, raw[:, ENERGY], master, on, p, self.les).items():
            setattr(self, k_, v)
        extra = np.where(self.teacher[:, None], np.where(objs["good"], 10.0, -10.0), 0.0) if teaching else None
        o = None if "S" in self.les else (self.o_pl, self.o_pa, self.o_assoc, raw[:, ENERGY])
        self.att.update(objs, decide, p, extra=extra, learned=1.0 - p["att_shift"] * k, o=o)
        fov = self.att.fovea(objs)
        a = self.a(raw, fov, np.exp(-dt / p["tau_a"]))
        r = value_of(reward, raw[:, ENERGY], p) + self.innate_reward(raw)
        teach_prev = self.teach
        b = self._b(a, self.delta)
        m = self._d(fov, teach_prev)
        c = self._c(b, m, fov)
        if "O" not in self.les:
            self.o_e = o_trace(self.o_e, fov, m, p, dt)
            ids = np.stack([fov[:, :IDENTITY], m[:, :IDENTITY]], 1)
            self.o_out = o_read(self.o_pl, self.o_pa, self.o_assoc, ids, raw[:, ENERGY], p)[0]
        vo = self.o_out
        one = np.ones((self.n, 1))
        f_v = np.concatenate([fov, m, c, raw[:, ENERGY:ENERGY + 1], p["critic_b"][:, None] * b, vo, one], 1)
        nf = fov.shape[1]
        eta_v = np.repeat(p["eta_v"][:, None], f_v.shape[1], 1)
        eta_v[:, 2 * nf: 2 * nf + c.shape[1]] = p["eta_vc"][:, None]  # la parte su C: in fretta
        horizons = np.stack([p["h_short"], p["h_mid"], p["h_long"]], 1)
        delta = self._td(f_v, r, horizons, eta_v, use_critic="V" not in self.les)
        # Soglia adattiva: insegna solo la parte di δ sopra la sorpresa solita.
        self.d_slow += (dt / p["u_slow"]) * (np.abs(delta) - self.d_slow)
        teach = np.sign(delta) * np.maximum(np.abs(delta) - p["floor"] * self.d_slow, 0.0)
        self.teach = teach
        # U: le sorprese vere (sopra la soglia) di adesso rispetto al solito (con
        # la media di tutto δ i pochi dolori di un'inversione si perdevano nel
        # rumore del critico, e U non si accorgeva di niente).
        self.d_fast += (dt / p["u_fast"]) * (np.abs(teach) - self.d_fast)
        self.t_slow += (dt / p["u_slow"]) * (np.abs(teach) - self.t_slow)
        if "U" not in self.les:
            self.unc = np.clip(self.d_fast / np.maximum(self.t_slow, 1e-4) - 1.0, 0.0, 3.0)
        boost = 1.0 + p["surprise_rate"] * self.unc
        # C impara in fretta e dimentica; L si avvicina a C solo a riposo.
        step = p["eta_c"] * boost * teach / (1.0 + self.code.sum(1))  # diviso per i neuroni accesi
        cap = p["c_max"][:, None, None]
        self.u_c = np.clip(self.u_c + step[:, None, None] * self.e_c - self.u_c * (dt / p["forget"])[:, None, None],
                           -cap, cap)
        self.e_c *= np.exp(-dt / p["trace_actor"])[:, None, None]
        if "L" not in self.les:
            # Riposo: carichi e fermi a ricaricarsi (il gusto della pila che si consuma).
            rest = (raw[:, ENERGY] > p["rest_energy"]) & (raw[:, TASTE] > 0.1)
            self.u_l += (p["eta_l"] * dt * rest)[:, None, None] * (self.u_c - self.u_l)
        # Con l'arbitraggio U non accelera più E e non aggiunge caso (era rumore):
        # sposta il peso della decisione verso C. Senza ("R"), com'era prima.
        slow_boost = np.ones(self.n) if arbitrate else boost
        f_e = np.concatenate([b, m, vo, one], 1)
        tag = p["tag"][:, None]
        f_tag = np.concatenate([np.where(b > tag, b, 0.0), np.where(m > tag, m, 0.0), vo, one], 1)
        habit = (self.w_e * f_e[:, None, :]).sum(-1)
        if "L" not in self.les:
            habit += np.einsum("npc,nc->np", self.u_l, c)
        self.votes = {"E+L": habit, "C": np.einsum("npc,nc->np", self.u_c, c), "k": k}
        logits = (1.0 - p["arb_slow"] * k)[:, None] * habit + (1.0 + p["arb_fast"] * k)[:, None] * self.votes["C"]
        temp = p["temperature"] * (1.0 + (0.0 if arbitrate else p["surprise_explore"]) * self.unc)
        program = self._act(logits, f_e, teach, p["eta_e"] * slow_boost, temp, decide, f_tag)
        if teaching:
            # Il maestro decide da sé (come il guidato).
            has, _, near, _ = self.att.seen(objs)
            ok = has & objs["good"][np.arange(self.n), np.maximum(self.att.target, 0)]
            program, self.bite, leave = teach_override(program, self.bite, has, ok, near, self.teacher)
            self.p_bite = np.where(self.teacher, self.bite.astype(float), self.p_bite)
            self.att.leave(leave, p)
            self.program = program
        return program

    def set_teacher(self, mask: np.ndarray, group: int) -> None:
        """Chi fa da maestro (`mask` (n,)); `group`: esseri per recinto. Chi
        smette di esserlo rinasce: al suo posto, un giovane."""
        gone = self.teacher & ~mask
        self.teacher, self.group = mask.copy(), group
        if gone.any():
            self.reborn(gone)

    def reborn(self, who, genomes=None):
        super().reborn(who, genomes)
        self.a.reset(who)
        self._reset_actor_critic(who)
        self.pred[who] = 0.0
        self.win_rate[who] = 1.0 / self.s.b
        for x in (self.state_b, self.prev_b, self.mem, self.w_gate, self.e_gate, self.u_c, self.e_c, self.u_l,
                  self.code, self.d_fast, self.unc, self.o_assoc, self.o_e, self.ep_pl, self.ep_pa, self.ep_e,
                  self.o_err, self.o_out, self.obs_err):
            x[who] = 0.0
        self.o_assoc[who] = 1.0
        self.ep_on[who] = False
        self.d_slow[who] = 1e-3
        self.t_slow[who] = 1e-4
        self._born(who)

    def winner(self) -> np.ndarray:
        """(N,) il neurone di B più acceso (per le misure di emersione)."""
        return self.state_b.argmax(1)


AGENTS = {"guidato": Guided, "attore-critico": ActorCritic, "sezioni": Sections}
