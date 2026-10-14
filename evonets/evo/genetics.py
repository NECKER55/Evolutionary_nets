"""Come nascono i figli: incrocio, mutazione, duplicazione (vedi
docs/SPEC_FASE_C.md, sezioni 4 e 5).

Quanto muta un figlio non è fissato da noi: ogni individuo porta i geni del
proprio meccanismo di mutazione (`Genome.meta`), che vengono ereditati e
mutano a loro volta. Come in natura, dove i tassi dipendono dalle proteine che
copiano e riparano il DNA, scritte in geni come tutti gli altri, questi geni
mutano con lo stesso meccanismo e la stessa frequenza di ogni altro gene:
con la probabilità e l'ampiezza di ritocco dell'individuo stesso. Non c'è una
velocità di cambiamento dei tassi scelta da noi. La configurazione dà solo i
valori di partenza e un errore minimo di copiatura (`mutation_floor`).
"""

import numpy as np

from ..config import Config
from ..net.brain import META, META_INDEX, Genome, clamp_meta, pad
from ..net.geometry import connect_prob, distances
from ..net.receptors import MAX_APERTURE, MIN_APERTURE

# Deriva dei geni continui dei recettori, in multipli del gene `rec_drift`:
# ogni gene ha la sua scala naturale.
DRIFT_SCALE = {"r_sens": 5.0, "r_mu": 1.0, "r_width": 5.0, "r_aperture": 5.0, "turn": 3.0,
               "r_tau": 5.0, "r_sub": 1.0, "r_div": 2.0}


def crossover(parents: Genome, mothers: np.ndarray, fathers: np.ndarray,
              rng: np.random.Generator) -> Genome:
    """Un figlio per coppia (mothers[i], fathers[i]).

    Il genoma si mescola a blocchi interi, presi ciascuno da un solo genitore:
    - un blocco per neurone interno o motore: l'esistenza, i geni propri, la
      posizione e tutti i collegamenti in ingresso (così i pesi che lavorano
      insieme restano insieme);
    - un blocco per recettore: tutti i suoi geni;
    - il meccanismo di mutazione, intero.
    I posti si abbinano per posizione: il posto k di un genitore con il posto k
    dell'altro.
    """
    n = len(mothers)
    t, h, r = parents.traits.shape[1], parents.n_hidden, parents.n_rec

    def pick(field, from_father):
        a = getattr(parents, field)
        mask = from_father.reshape(from_father.shape + (1,) * (a.ndim - from_father.ndim))
        return np.where(mask, a[fathers], a[mothers])

    blocks = rng.random((n, t)) < 0.5
    receptors = rng.random((n, r)) < 0.5
    meta = rng.random((n, 1)) < 0.5
    return Genome(
        pick("weights", blocks), pick("traits", blocks), pick("conn", blocks),
        pick("active", blocks[:, :h]), pick("pos", blocks[:, :h]),
        *(pick(f, receptors) for f in Genome.RECEPTOR_FIELDS),
        meta=pick("meta", meta),
    )


def _turn(v: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c], axis=-1)


def duplicate_neuron(g: Genome, i: int, h: int, f: int, rng: np.random.Generator) -> None:
    """Il neurone interno h dell'individuo i si duplica nel posto libero f.

    La copia riceve gli stessi ingressi e manda le stesse uscite; le uscite di
    originale e copia vengono dimezzate. Così insieme fanno esattamente quello
    che faceva l'originale da solo: la duplicazione non cambia il
    comportamento, ma la copia è libera di divergere."""
    r = g.n_rec
    src_h, src_f = r + h, r + f
    for a in (g.weights, g.conn):
        a[i, :, src_f] = a[i, :, src_h]  # uscite (anche verso se stesso)
        a[i, f, :] = a[i, h, :]  # ingressi, compreso quello appena copiato
    g.weights[i, :, src_h] *= 0.5
    g.weights[i, :, src_f] *= 0.5
    g.traits[i, f] = g.traits[i, h]
    g.active[i, f] = True
    std = np.exp(g.meta[i, META_INDEX["pos_std"]])
    g.pos[i, f] = np.clip(g.pos[i, h] + rng.normal(0.0, std, 2), 0.0, 1.0)


def duplicate_receptor(g: Genome, i: int, k: int, f: int, rng: np.random.Generator) -> None:
    """Il recettore k dell'individuo i si duplica nel posto libero f: stessi
    geni e stesse uscite, uscite dimezzate. Il comportamento medio non cambia;
    le due copie hanno rumore indipendente, quindi la loro somma è già un poco
    più precisa."""
    for field in Genome.RECEPTOR_FIELDS:
        getattr(g, field)[i, f] = getattr(g, field)[i, k]
    for a in (g.weights, g.conn):
        a[i, :, f] = a[i, :, k]
    g.weights[i, :, k] *= 0.5
    g.weights[i, :, f] *= 0.5
    std = np.exp(g.meta[i, META_INDEX["pos_std"]])
    g.r_y[i, f] = np.clip(g.r_y[i, k] + rng.normal(0.0, std), 0.0, 1.0)


def _duplicate(g: Genome, rng: np.random.Generator, p: np.ndarray) -> None:
    """Ogni neurone interno e ogni recettore acceso si duplica con la
    probabilità `p` (N,) del suo individuo, nel primo posto libero. Senza posti
    liberi la duplicazione non avviene (i posti crescono alla generazione
    dopo, vedi `grow`)."""
    for active, copy in ((g.active, duplicate_neuron), (g.r_active, duplicate_receptor)):
        events = np.argwhere(active & (rng.random(active.shape) < p[:, None]))
        for i, k in events:
            free = np.flatnonzero(~active[i])
            if len(free):
                copy(g, int(i), int(k), int(free[0]), rng)
                active[i, free[0]] = True


def mutate(genome: Genome, cfg: Config, rng: np.random.Generator, strength: float = 1.0) -> None:
    """Muta sul posto, con i tassi del figlio (ereditati dai genitori).
    `strength` scala frequenza e ampiezza (>1 = variazioni più consistenti,
    per il soccorso)."""
    g = genome
    n = len(g)
    inherited = np.exp(g.meta)  # i tassi di questa nascita: quelli ereditati

    def base(name):
        return inherited[:, META_INDEX[name]]

    def rate(name):
        return base(name) * strength

    def chance(shape, p):
        return rng.random(shape) < np.minimum(1.0, p).reshape((n,) + (1,) * (len(shape) - 1))

    def noise(shape, std):
        return rng.standard_normal(shape) * std.reshape((n,) + (1,) * (len(shape) - 1))

    # 1. Pesi, geni propri dei neuroni e geni del meccanismo di mutazione:
    # questi ultimi sono geni come gli altri (in scala logaritmica: un ritocco
    # cambia un tasso di un fattore, non di una quantità fissa). I tassi usati
    # in questa nascita sono quelli ereditati, letti prima del ritocco.
    p_w, std_w = rate("weight_prob"), base("weight_std") * strength
    for values in (g.weights, g.traits, g.meta):
        values += chance(values.shape, p_w) * noise(values.shape, std_w)
    clamp_meta(g.meta, cfg)

    # 2. Posizioni nel cervello: interni e ingressi dei recettori.
    p_pos, std_pos = rate("pos_prob"), base("pos_std") * strength
    g.pos += chance(g.pos.shape[:2], p_pos)[:, :, None] * noise(g.pos.shape, std_pos)
    np.clip(g.pos, 0.0, 1.0, out=g.pos)
    g.r_y += chance(g.r_y.shape, p_pos) * noise(g.r_y.shape, std_pos)
    np.clip(g.r_y, 0.0, 1.0, out=g.r_y)

    # 3. Geni dei recettori: ognuno deriva nella sua scala.
    drift = base("rec_drift") * strength
    for field in ("r_sens", "r_mu", "r_width", "r_aperture", "r_tau", "r_sub", "r_div"):
        a = getattr(g, field)
        a += chance(a.shape, p_w) * noise(a.shape, drift * DRIFT_SCALE[field])
    np.clip(g.r_mu, 0.0, 1.0, out=g.r_mu)
    np.clip(g.r_sub, 0.0, 1.0, out=g.r_sub)
    np.maximum(g.r_div, 0.0, out=g.r_div)
    np.clip(g.r_tau, np.log(cfg.dt), np.log(1000.0), out=g.r_tau)
    np.clip(g.r_aperture, np.log(MIN_APERTURE), np.log(MAX_APERTURE), out=g.r_aperture)
    for field in ("r_body", "r_look"):
        a = getattr(g, field)
        a[:] = _turn(a, chance(a.shape[:2], p_w) * noise(a.shape[:2], drift * DRIFT_SCALE["turn"]))

    # 4. Collegamenti: uno nuovo compare con una probabilità che cala con la
    # distanza, come alla nascita, con la portata λ dell'individuo.
    n_rec, n_out = g.n_rec, g.n_out
    dist = distances(cfg, g.pos, g.r_y, n_out)
    shape = connect_prob(cfg, dist, n_rec, base("conn_range")) / cfg.net_conn_prob
    p_add = np.minimum(1.0, rate("conn_add")[:, None, None] * shape)
    appear = ~g.conn & (rng.random(g.conn.shape) < p_add)
    vanish = g.conn & chance(g.conn.shape, rate("conn_del"))
    g.conn ^= appear | vanish

    # 5. Neuroni e recettori si accendono o si spengono; poi le duplicazioni.
    g.active ^= chance(g.active.shape, rate("toggle"))
    g.r_active ^= chance(g.r_active.shape, rate("toggle"))
    _duplicate(g, rng, np.minimum(1.0, rate("dup")))


def reproduce(parents: Genome, n_children: int, cfg: Config, rng: np.random.Generator,
              strength: float = 1.0) -> Genome:
    """`n_children` figli da genitori scelti tutti con la stessa probabilità.

    Una quota `crossover_rate` nasce da due genitori diversi, gli altri sono
    copie di uno solo; poi tutti vengono mutati.
    """
    n_parents = len(parents)
    mothers = rng.integers(0, n_parents, n_children)
    fathers = mothers.copy()
    if n_parents > 1:
        two = rng.random(n_children) < cfg.crossover_rate
        # Uno scarto tra 1 e n-1 garantisce un padre diverso dalla madre.
        fathers[two] = (mothers[two] + rng.integers(1, n_parents, int(two.sum()))) % n_parents
    children = crossover(parents, mothers, fathers, rng)
    mutate(children, cfg, rng, strength)
    return children


def grow(genome: Genome, cfg: Config) -> Genome:
    """Se qualcuno usa più di `pool_fill` dei posti per i recettori o per i
    neuroni interni, tutta la popolazione riceve `pool_step` posti nuovi,
    spenti. È solo memoria: per l'evoluzione non esiste un numero massimo."""
    r, h = genome.n_rec, genome.n_hidden
    if genome.r_active.sum(axis=1).max(initial=0) > cfg.pool_fill * r:
        r += cfg.pool_step
    if genome.active.sum(axis=1).max(initial=0) > cfg.pool_fill * h:
        h += cfg.pool_step
    return pad(genome, r, h)


__all__ = ["META", "crossover", "grow", "mutate", "reproduce"]
