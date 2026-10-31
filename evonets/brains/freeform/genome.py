"""Il genoma delle reti a forma libera: i tratti di ogni individuo, i tassi di mutazione
come geni e la partenza primordiale."""

from dataclasses import dataclass

import numpy as np

from ...config import Config
from .geometry import brain_side, connect_prob, distances
from .receptors import P, ReceptorGenes

# Geni del meccanismo di mutazione, uno per individuo (vedi genetics.mutate e
# docs/SPEC_FASE_C.md, sezione 5). Nel genoma stanno come logaritmi.
META = ("weight_prob", "weight_std", "conn_add", "conn_del", "toggle", "dup", "pos_prob", "pos_std",
        "rec_drift", "conn_range", "sex", "canal")
# "sex": probabilità che un figlio di questo individuo (come madre) nasca con
# un partner invece che come sua copia. Come i tassi, evolve.
# "canal": precisione dello sviluppo (development.py), la pendenza della soglia
# con cui l'attrazione tra due cellule diventa un collegamento. Alta: fratelli
# identici hanno lo stesso cervello; bassa: lo sviluppo tira a sorte.
META_INDEX = {name: k for k, name in enumerate(META)}


def meta_start(cfg: Config) -> np.ndarray:
    """I valori di partenza dei geni delle mutazioni (dalla configurazione)."""
    return np.array([cfg.mut_weight_prob, cfg.mut_weight_std, cfg.mut_conn_add, cfg.mut_conn_del,
                     cfg.mut_neuron_toggle, cfg.mut_dup, cfg.mut_pos_prob, cfg.mut_pos_std,
                     cfg.mut_rec_drift, cfg.net_conn_range, cfg.crossover_rate, cfg.develop_precision])


PROBABILITIES = [META_INDEX[k] for k in ("weight_prob", "conn_add", "conn_del", "toggle", "dup", "pos_prob", "sex")]
# Come in natura gli effetti delle mutazioni sono per lo più piccoli (la
# varianza che le mutazioni aggiungono in una generazione è circa un
# millesimo di quella dovuta all'ambiente; Lynch e Walsh) e i mutatori
# cambiano i tassi di un fattore, non di miliardi di volte. Le ampiezze hanno
# quindi un tetto: un ritocco sposta un gene al più di circa quanto
# differiscono due individui alla partenza (1), una posizione di 0,3 lati del
# cervello, un gene di recettore di 0,2 volte la sua scala. Senza tetto
# l'ampiezza, che muta con se stessa, si autoalimentava: in 60 generazioni
# senza selezione arrivava a 100 e i geni a ±170. La portata dei collegamenti
# al più 3 lati del cervello (oltre è già uniforme).
CEILING = {"weight_std": 1.0, "pos_std": 0.3, "rec_drift": 0.2, "conn_range": 3.0}


# Geni della plasticità, in coda ai geni propri di ogni neurone interno o
# motore (vedi docs/SPEC_FASE_E.md). Riguardano le sinapsi che il neurone
# RICEVE: Δw = η · M · (A·pre·post + B·pre + C·post + D), con
# η = plasticity_initial_rate · e^gene (gene 0 = plasticità debole di partenza).
# "mod" > 0 rende un neurone interno modulatore: non trasmette, rilascia M.
# "hebb": apprendimento hebbiano di base, sempre attivo, senza modulatori
# (docs/SPEC_SVILUPPO.md, sezione 10): Δw = η_h · pre · post, con
# η_h = hebb_initial_rate · e^gene.
# "target" e "homeo": scaling omeostatico (docs/SPEC_SVILUPPO.md, sezione 4).
# Il neurone tiene la media recente ā della propria attività e moltiplica le
# sinapsi che riceve per exp(h · dt · (a* − ā)), con a* = sigmoide(target)
# l'attività desiderata e h = homeo_initial_rate · e^homeo la forza.
# "persist": quanto dura la traccia provvisoria (early-LTP) che l'hebbiano
# lascia sulle sinapsi che il neurone riceve, τ = early_initial_tau · e^gene.
# "intrinsic": forza dell'omeostasi dell'eccitabilità (la corrente di fondo),
# un meccanismo distinto dallo scaling delle sinapsi (Desai 1999):
# homeo_initial_rate · e^gene.
PLASTIC = ("eta", "A", "B", "C", "D", "mod", "hebb", "target", "homeo", "persist", "intrinsic")
# Dopo la plasticità, i geni propri dello sviluppo:
# "gain", quanto pesano gli ingressi che il neurone riceve (come il numero di
# recettori postsinaptici), in scala logaritmica: peso = affinità · e^gain
# (development.py);
# "boost" e "closure", il periodo critico: l'hebbiano del neurone è
# B = critical_initial_boost · e^boost volte più forte all'inizio dello
# sviluppo e torna alla velocità dell'adulto con il tempo
# τ_c = critical_initial_tau · e^closure (s di età, dal grembo):
# η_h(età) = η_h · (1 + (B − 1) · e^(−età/τ_c)).
# "elig": quanto dura la traccia di eleggibilità (docs/SPEC_FASE_E.md,
# sezione "Traccia di eleggibilità"), τ_e = eligibility_initial_tau · e^gene:
# la regola modulata non cambia subito la sinapsi, propone un cambiamento che
# resta nella traccia; il modulatore che arriva dopo (piacere o dolore) lo
# applica con il suo segno. È in coda: i salvataggi di prima lo ricevono a 0.
GENES = PLASTIC + ("gain", "boost", "closure", "elig")


# Un gene alza una velocità o una forza al più di e^12 volte (circa 160.000)
# rispetto al suo valore di partenza: oltre non ha più significato fisico, e i
# numeri a 32 bit della GPU traboccherebbero (l'ampiezza delle mutazioni è un
# gene e può arrivare a 100). Verso il basso nessun limite: un gene può
# spegnersi del tutto, come un gene rotto.
GENE_SPAN = 12.0


def scale_gene(gene):
    """e^gene, con il gene limitato in alto a GENE_SPAN."""
    return np.exp(np.minimum(gene, GENE_SPAN))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clamp_meta(meta: np.ndarray, cfg: Config) -> None:
    """Sul posto: nessun tasso sotto l'errore minimo di copiatura, nessuna
    probabilità sopra 1 e nessuna ampiezza sopra il suo tetto (`CEILING`)."""
    np.maximum(meta, np.log(cfg.mutation_floor), out=meta)
    meta[:, PROBABILITIES] = np.minimum(meta[:, PROBABILITIES], 0.0)
    for name, top in CEILING.items():
        k = META_INDEX[name]
        meta[:, k] = np.minimum(meta[:, k], np.log(top))


def fit_traits(g: "Genome", n_traits: int) -> "Genome":
    """I geni propri dei neuroni nati dopo un salvataggio (in coda) ricevono
    0: il valore di partenza. Restituisce lo stesso genoma."""
    missing = n_traits - g.traits.shape[2]
    if missing > 0:
        g.traits = np.concatenate([g.traits, np.zeros(g.traits.shape[:2] + (missing,))], axis=2)
    return g


def live_links(conn: np.ndarray, active: np.ndarray, r_active: np.ndarray) -> np.ndarray:
    """Collegamenti che contano davvero: presenti nel genoma e tra neuroni
    esistenti (recettori e interni accesi, motori sempre).
    conn (N, T, S), active (N, H), r_active (N, R) -> (N, T, S)."""
    n = len(conn)
    n_out = conn.shape[1] - active.shape[1]
    target_on = np.concatenate([active, np.ones((n, n_out), dtype=bool)], axis=1)
    source_on = np.concatenate([r_active, target_on], axis=1)
    return conn & target_on[:, :, None] & source_on[:, None, :]


def retention(gene: np.ndarray) -> np.ndarray:
    """Reti a impulsi: quota di carica che un neurone trattiene a ogni passo,
    dal gene "perdita". Da ~0.5 (neurone rapido) a ~0.97 (lento), tipicamente ~0.8."""
    return 1.0 / (1.0 + np.exp(-(1.5 + gene)))


@dataclass
class Genome:
    """I genomi di un gruppo di individui, uno per riga.

    Cervello (vedi `Brains`):
      weights (N, T, S), conn (N, T, S), traits (N, T, K), active (N, H), pos (N, H, 2)
      con S = R + H + n_out sorgenti (recettori, interni, motori) e
      T = H + n_out destinatari (interni, motori).
    Recettori (vedi `receptors.ReceptorGenes`): i campi `r_*`, R posti.
    Meccanismo di mutazione: `meta` (N, len(META)), logaritmi.

    R e H sono "posti": un recettore o un neurone esiste solo se acceso. Sono
    uguali per tutta la popolazione e crescono quando servono (`pad`).

    Discendenza dei geni (per l'incrocio tra omologhi, genetics.crossover):
    `r_id` (N, R), `h_id` (N, H) identità del gene di ogni recettore e neurone
    interno, `r_fam`, `h_fam` la famiglia (il gene antenato comune a tutte le
    sue copie). Un gene che non viene da una duplicazione ha l'identità del suo
    posto: tutti discendono dagli stessi antenati. Una copia riceve
    un'identità nuova e la famiglia dell'originale.

    Sviluppo (`Config.genome = "sviluppo"`, vedi development.py), M posti per
    i geni dei marcatori: `r_mark` (N, R, M) e `n_mark` (N, T, M) quanto ogni
    recettore e ogni neurone interno o motore esprime ogni marcatore (in scala
    logistica), `m_grad` (N, M, 2) quanto l'espressione di ogni marcatore
    segue la posizione nel cervello (gradienti, come Eph ed efrine),
    `m_table` (N, M, M) la tabella di affinità (marcatore sull'assone ->
    marcatore sul bersaglio, con segno), `m_active`, `m_id`, `m_fam` (N, M)
    esistenza e discendenza dei marcatori. Con il genoma diretto M = 0. In
    questo caso `weights` e `conn` sono il cervello sviluppato (il fenotipo),
    non geni: nascono dallo sviluppo e non mutano.

    Fenotipo alla nascita, dopo il grembo (brains/freeform/womb.py, solo con lo
    sviluppo): `born_avg` (N, T) la frequenza media di ogni neurone e
    `born_bias` (N, T) la sua corrente di fondo. NaN = non sviluppato: valgono
    i geni (a* e la soglia del genoma).

    Sistema di valore innato: `value` (N, 2) la forza dei due nuclei (gusto ->
    M positivo, dolore -> M negativo), in scala logaritmica: forza =
    value_initial · e^gene.
    """

    weights: np.ndarray
    traits: np.ndarray
    conn: np.ndarray
    active: np.ndarray
    pos: np.ndarray
    r_active: np.ndarray
    r_sens: np.ndarray
    r_mu: np.ndarray
    r_width: np.ndarray
    r_body: np.ndarray
    r_look: np.ndarray
    r_aperture: np.ndarray
    r_y: np.ndarray
    r_tau: np.ndarray
    r_sub: np.ndarray
    r_div: np.ndarray
    meta: np.ndarray
    r_id: np.ndarray | None = None
    h_id: np.ndarray | None = None
    r_fam: np.ndarray | None = None
    h_fam: np.ndarray | None = None
    r_mark: np.ndarray | None = None
    n_mark: np.ndarray | None = None
    m_grad: np.ndarray | None = None
    m_table: np.ndarray | None = None
    m_active: np.ndarray | None = None
    m_id: np.ndarray | None = None
    m_fam: np.ndarray | None = None
    born_avg: np.ndarray | None = None
    born_bias: np.ndarray | None = None
    value: np.ndarray | None = None

    FIELDS = ("weights", "traits", "conn", "active", "pos", "r_active", "r_sens", "r_mu", "r_width",
              "r_body", "r_look", "r_aperture", "r_y", "r_tau", "r_sub", "r_div", "meta",
              "r_id", "h_id", "r_fam", "h_fam",
              "r_mark", "n_mark", "m_grad", "m_table", "m_active", "m_id", "m_fam",
              "born_avg", "born_bias", "value")
    RECEPTOR_FIELDS = FIELDS[5:16]
    LINEAGE = FIELDS[17:21]
    MARKERS = FIELDS[21:28]
    BORN = FIELDS[28:30]

    def __post_init__(self):
        n = len(self.weights)
        for name, slots in (("r_id", self.r_active.shape[1]), ("r_fam", self.r_active.shape[1]),
                            ("h_id", self.active.shape[1]), ("h_fam", self.active.shape[1])):
            if getattr(self, name) is None:  # i geni del posto: discendenza comune
                setattr(self, name, np.broadcast_to(np.arange(slots, dtype=np.int64), (n, slots)).copy())
        if self.m_active is None:  # senza marcatori (genoma diretto)
            self.m_active = np.zeros((n, 0), dtype=bool)
        m = self.m_active.shape[1]
        empty = {"r_mark": (n, self.r_active.shape[1], m), "n_mark": (n, self.weights.shape[1], m),
                 "m_grad": (n, m, 2), "m_table": (n, m, m)}
        for name, shape in empty.items():
            if getattr(self, name) is None:
                setattr(self, name, np.zeros(shape))
        for name in ("m_id", "m_fam"):
            if getattr(self, name) is None:
                setattr(self, name, np.broadcast_to(np.arange(m, dtype=np.int64), (n, m)).copy())
        for name in self.BORN:
            if getattr(self, name) is None:
                setattr(self, name, np.full(self.weights.shape[:2], np.nan))
        if self.value is None:  # i nuclei di valore alla loro forza di partenza
            self.value = np.zeros((n, 2))

    def unborn(self) -> None:
        """Sul posto: nessun fenotipo alla nascita (un figlio prima del grembo)."""
        for name in self.BORN:
            getattr(self, name)[:] = np.nan
    def __len__(self) -> int:
        return len(self.weights)

    @property
    def n_rec(self) -> int:
        return self.r_active.shape[1]

    @property
    def n_hidden(self) -> int:
        return self.active.shape[1]

    @property
    def n_out(self) -> int:
        return self.weights.shape[1] - self.n_hidden

    @property
    def n_markers(self) -> int:
        """Posti per i marcatori (0 con il genoma diretto)."""
        return self.m_active.shape[1]

    @property
    def receptors(self) -> ReceptorGenes:
        """I geni dei recettori (gli stessi array, non una copia)."""
        return ReceptorGenes(*(getattr(self, f) for f in self.RECEPTOR_FIELDS))

    def rate(self, name: str) -> np.ndarray:
        """(N,) un gene del meccanismo di mutazione, nella sua scala naturale."""
        return np.exp(self.meta[:, META_INDEX[name]])

    def take(self, idx) -> "Genome":
        """Copia indipendente degli individui indicati."""
        return Genome(*(getattr(self, f)[idx].copy() for f in self.FIELDS))

    @staticmethod
    def concat(parts: list["Genome"]) -> "Genome":
        """Uno sotto l'altro; i posti vengono pareggiati al più grande."""
        r = max(p.n_rec for p in parts)
        h = max(p.n_hidden for p in parts)
        m = max(p.n_markers for p in parts)
        parts = [pad(p, r, h, m) for p in parts]
        return Genome(*(np.concatenate([getattr(p, f) for p in parts]) for f in Genome.FIELDS))


def pad(g: Genome, n_rec: int, n_hidden: int, n_markers: int | None = None) -> Genome:
    """Lo stesso genoma con più posti (spenti, geni a zero) per recettori,
    neuroni interni e marcatori. Non cambia nulla per l'individuo: i posti
    nuovi non esistono finché una mutazione non li accende o una duplicazione
    non li riempie."""
    m = g.n_markers if n_markers is None else n_markers
    dr, dh, dm = n_rec - g.n_rec, n_hidden - g.n_hidden, m - g.n_markers
    if dr == 0 and dh == 0 and dm == 0:
        return g
    if dr < 0 or dh < 0 or dm < 0:
        raise ValueError("pad: i posti possono solo crescere")
    n, r, h = len(g), g.n_rec, g.n_hidden

    def grow(a, axis, at, k):
        shape = list(a.shape)
        shape[axis] = k
        return np.insert(a, [at] * k, np.zeros(shape, dtype=a.dtype), axis=axis) if k else a

    def links(a):  # (N, T, S): righe nuove dopo H, colonne nuove dopo R e dopo R + H
        a = grow(a, 1, h, dh)
        a = grow(a, 2, r + h, dh)
        return grow(a, 2, r, dr)

    rec = {f: grow(getattr(g, f), 1, r, dr) for f in Genome.RECEPTOR_FIELDS}
    rec["r_body"][:, r:] = rec["r_look"][:, r:] = [1.0, 0.0]  # vettori unitari validi
    return Genome(
        links(g.weights), grow(g.traits, 1, h, dh), links(g.conn), grow(g.active, 1, h, dh),
        np.concatenate([g.pos, np.full((n, dh, 2), 0.5)], axis=1), **rec, meta=g.meta,
        # I posti nuovi hanno l'identità del loro posto (nessun gene ancora).
        r_id=np.concatenate([g.r_id, np.broadcast_to(np.arange(r, r + dr), (n, dr))], axis=1),
        r_fam=np.concatenate([g.r_fam, np.broadcast_to(np.arange(r, r + dr), (n, dr))], axis=1),
        h_id=np.concatenate([g.h_id, np.broadcast_to(np.arange(h, h + dh), (n, dh))], axis=1),
        h_fam=np.concatenate([g.h_fam, np.broadcast_to(np.arange(h, h + dh), (n, dh))], axis=1),
        **_pad_markers(g, grow, r, dr, h, dh, dm),
        **{name: np.insert(getattr(g, name), [h] * dh, np.nan, axis=1) for name in Genome.BORN},
        value=g.value,
    )


def _pad_markers(g: Genome, grow, r: int, dr: int, h: int, dh: int, dm: int) -> dict:
    """I campi dei marcatori con i posti nuovi: recettori dopo R, interni dopo
    H, marcatori in fondo (spenti, con l'identità del loro posto)."""
    n, m = len(g), g.n_markers
    r_mark = grow(grow(g.r_mark, 1, r, dr), 2, m, dm)
    n_mark = grow(grow(g.n_mark, 1, h, dh), 2, m, dm)
    table = grow(grow(g.m_table, 1, m, dm), 2, m, dm)
    new_ids = np.broadcast_to(np.arange(m, m + dm), (n, dm))
    return dict(r_mark=r_mark, n_mark=n_mark, m_grad=grow(g.m_grad, 1, m, dm), m_table=table,
                m_active=grow(g.m_active, 1, m, dm), m_id=np.concatenate([g.m_id, new_ids], axis=1),
                m_fam=np.concatenate([g.m_fam, new_ids], axis=1))


def primordial(cfg: Config, n: int, n_out: int, n_traits: int, rng: np.random.Generator) -> Genome:
    """La partenza primordiale (docs/SPEC_FASE_C.md, sezione 7): pochi
    recettori generalisti, nessun neurone interno acceso, collegamenti
    secondo la geometria. Gli individui differiscono un poco nei geni."""
    fixed = None
    if cfg.sim_mode == "cervello":  # sensi fissi, decisi da noi
        from .senses import genes

        fixed = genes(cfg, n)
    R = len(fixed["r_y"][0]) if fixed else cfg.receptor_slots
    H, std = cfg.net_hidden_slots, cfg.net_weight_std
    on = R if fixed else min(cfg.receptor_initial, R)
    # Recettori: i primi `on` accesi, distribuiti a ventaglio davanti; gli
    # altri posti hanno geni a caso ma spenti (pseudogeni).
    angle = rng.uniform(-np.pi, np.pi, (n, R))
    if on:
        spread = np.radians(cfg.receptor_initial_spread_deg)
        angle[:, :on] = np.linspace(-spread / 2, spread / 2, on) if on > 1 else 0.0
    unit = np.stack([np.cos(angle), np.sin(angle)], axis=2)
    r_active = np.zeros((n, R), dtype=bool)
    r_active[:, :on] = True
    r_sens = np.log(cfg.receptor_initial_gain) + rng.normal(0.0, 0.2, (n, R, P))
    # Sintonia: nessun colore, odore o grandezza è privilegiato. Ogni individuo
    # ha la sua, a caso su tutto l'asse; i suoi recettori sono copie dello
    # stesso gene (come gli organi pari di un corpo bilaterale) e differiscono
    # appena.
    r_mu = np.clip(rng.random((n, 1, P)) + rng.normal(0.0, 0.05, (n, R, P)), 0.0, 1.0)
    r_width = np.log(cfg.receptor_initial_width) + rng.normal(0.0, 0.1, (n, R, P))
    r_aperture = np.full((n, R), np.log(np.radians(cfg.receptor_initial_aperture_deg)))
    # Mappa topografica: il nervo entra nel cervello a un'altezza che segue la
    # direzione in cui il recettore guarda.
    r_y = np.clip(0.5 + angle / (2.0 * np.pi), 0.0, 1.0)
    # Adattamento rudimentale fin da subito, diverso da individuo a individuo.
    r_tau = np.log(cfg.receptor_initial_tau) + rng.normal(0.0, 0.3, (n, R))
    r_sub = rng.uniform(0.0, cfg.receptor_initial_adaptation, (n, R))
    r_div = rng.uniform(0.0, cfg.receptor_initial_adaptation, (n, R))
    if fixed:
        r_active, r_sens, r_mu, r_width = fixed["r_active"], fixed["r_sens"], fixed["r_mu"], fixed["r_width"]
        r_aperture, r_y, r_tau, r_sub, r_div = (fixed[k] for k in ("r_aperture", "r_y", "r_tau", "r_sub", "r_div"))
        body_unit, unit = fixed["r_body"], fixed["r_look"]
    else:
        body_unit = unit.copy()

    pos = rng.random((n, H, 2))
    meta = np.log(np.maximum(meta_start(cfg), cfg.mutation_floor))[None, :] + rng.normal(0.0, 0.1, (n, len(META)))
    clamp_meta(meta, cfg)
    dist = distances(cfg, pos, r_y, n_out, brain_side(cfg, r_active, np.zeros((n, H), dtype=bool), n_out))
    conn = rng.random(dist.shape) < connect_prob(cfg, dist, R, np.exp(meta[:, META_INDEX["conn_range"]]))
    T, S = H + n_out, R + H + n_out
    return Genome(
        weights=rng.normal(0.0, std, (n, T, S)), traits=rng.normal(0.0, std, (n, T, n_traits)),
        conn=conn, active=np.zeros((n, H), dtype=bool), pos=pos,
        r_active=r_active, r_sens=r_sens, r_mu=r_mu, r_width=r_width, r_body=body_unit, r_look=unit,
        r_aperture=r_aperture, r_y=r_y, r_tau=r_tau, r_sub=r_sub, r_div=r_div, meta=meta,
    )
