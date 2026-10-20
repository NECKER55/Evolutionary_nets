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
from ..net.geometry import brain_side, connect_prob, distances
from ..net.receptors import MAX_APERTURE, MIN_APERTURE

# Deriva dei geni continui dei recettori, in multipli del gene `rec_drift`:
# ogni gene ha la sua scala naturale.
DRIFT_SCALE = {"r_sens": 5.0, "r_mu": 1.0, "r_width": 5.0, "r_aperture": 5.0, "turn": 3.0,
               "r_tau": 5.0, "r_sub": 1.0, "r_div": 2.0}


def crossover(parents: Genome, mothers: np.ndarray, fathers: np.ndarray,
              rng: np.random.Generator, split: bool = True) -> Genome:
    """Un figlio per coppia (mothers[i], fathers[i]); con madre = padre il
    figlio è una sua copia.

    Come nella meiosi, si appaiano i geni **omologhi**, quelli con la stessa
    identità di discendenza (`Genome.r_id`, `h_id`), ovunque stiano nei due
    genomi. Il genoma si mescola a blocchi interi, presi ciascuno da un solo
    genitore:
    - un blocco per neurone interno o motore: l'esistenza, i geni propri, la
      posizione e tutti i collegamenti in ingresso, che seguono l'identità
      delle sorgenti (un ingresso dal naso resta collegato al naso);
    - un blocco per recettore: tutti i suoi geni;
    - il meccanismo di mutazione, intero.
    Un gene presente in un solo genitore passa al figlio con probabilità 1/2,
    come una variante che solo uno dei due porta. Poi, tra copie della stessa
    famiglia, lo scambio ineguale (`unequal_crossing_over`).
    """
    child = parents.take(mothers)  # la madre fa da cornice
    for c in np.flatnonzero(mothers != fathers):
        _recombine(child, int(c), parents, int(fathers[c]), rng)
        unequal_crossing_over(child, int(c), rng, split)
    return child


def _compose(on_m, id_m, on_f, id_f, rng):
    """Quali geni (recettori o neuroni) ha il figlio e da chi: restituisce il
    nuovo `on` del figlio, {posto del figlio: posto del padre} per i geni presi
    dal padre e l'elenco dei posti riempiti con geni che la madre non aveva."""
    on = on_m.copy()
    id_m = id_m.copy()  # identità della madre prima dei geni del padre
    mother = {int(id_m[k]): k for k in np.flatnonzero(on_m)}
    father = {int(id_f[k]): k for k in np.flatnonzero(on_f)}
    from_father, inserted = {}, []
    for gid, k in mother.items():
        if gid in father:
            if rng.random() < 0.5:
                from_father[k] = father[gid]
        elif rng.random() < 0.5:  # solo nella madre: passa con 1/2
            on[k] = False
    for gid, kf in father.items():
        if gid not in mother and rng.random() < 0.5:  # solo nel padre: passa con 1/2
            # Nel suo locus, se nella madre è libero (lì lei porta la versione
            # spenta dello stesso gene), altrimenti nel primo posto libero.
            free = np.flatnonzero(~on & (id_m == gid))
            if not len(free):
                free = np.flatnonzero(~on)
            if len(free):
                j = int(free[0])
                on[j] = True
                from_father[j] = kf
                inserted.append(j)
    return on, from_father, inserted


def _recombine(g: Genome, c: int, p: Genome, f: int, rng: np.random.Generator) -> None:
    """Il figlio c (copia della madre) riceve i blocchi del padre f."""
    R, H, n_out = g.n_rec, g.n_hidden, g.n_out
    old_r, old_h = g.r_id[c].copy(), g.h_id[c].copy()
    r_on, rec_f, rec_new = _compose(g.r_active[c], g.r_id[c], p.r_active[f], p.r_id[f], rng)
    h_on, hid_f, hid_new = _compose(g.active[c], g.h_id[c], p.active[f], p.h_id[f], rng)
    for j, kf in rec_f.items():
        for name in Genome.RECEPTOR_FIELDS + ("r_id", "r_fam"):
            getattr(g, name)[c, j] = getattr(p, name)[f, kf]
    g.r_active[c] = r_on
    g.active[c] = h_on
    # Le sorgenti del padre nel figlio, per identità (recettori e interni
    # esistenti); i motori sono sempre gli stessi.
    for j, hf in hid_f.items():
        g.pos[c, j], g.h_id[c, j], g.h_fam[c, j] = p.pos[f, hf], p.h_id[f, hf], p.h_fam[f, hf]
    def where(ids, on):  # identità -> posto nel figlio (i geni accesi prevalgono su quelli spenti)
        out = {int(ids[k]): k for k in np.flatnonzero(~on)}
        out.update({int(ids[k]): k for k in np.flatnonzero(on)})
        return out

    where_r, where_h = where(g.r_id[c], r_on), where(g.h_id[c], h_on)
    src = np.full(R + H + n_out, -1)
    for kf in range(R):
        src[kf] = where_r.get(int(p.r_id[f, kf]), -1)
    for hf in range(H):
        src[R + hf] = R + where_h[int(p.h_id[f, hf])] if int(p.h_id[f, hf]) in where_h else -1
    src[R + H:] = np.arange(R + H, R + H + n_out)
    ok = np.flatnonzero(src >= 0)
    # I blocchi dei destinatari presi dal padre: interni (secondo l'identità) e motori (a caso).
    rows = {j: hf for j, hf in hid_f.items()}
    rows.update({H + o: H + o for o in range(n_out) if rng.random() < 0.5})
    for t, tf in rows.items():
        w, conn = np.zeros(R + H + n_out), np.zeros(R + H + n_out, dtype=bool)
        w[src[ok]], conn[src[ok]] = p.weights[f, tf, ok], p.conn[f, tf, ok]
        g.weights[c, t], g.conn[c, t], g.traits[c, t] = w, conn, p.traits[f, tf]
    # I blocchi rimasti alla madre non hanno ingressi dai geni che lei non
    # aveva (salvo che fossero nel loro locus, dove lei ne portava la versione spenta).
    fresh = [j for j in rec_new if old_r[j] != g.r_id[c, j]] + [R + j for j in hid_new if old_h[j] != g.h_id[c, j]]
    if fresh:
        mother_rows = [t for t in range(H + n_out) if t not in rows]
        g.conn[c][np.ix_(mother_rows, fresh)] = False
    if rng.random() < 0.5:
        g.meta[c] = p.meta[f]


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Quanto si somigliano due copie (1 = identiche): e^(-differenza media dei geni)."""
    return float(np.exp(-np.abs(a - b).mean()))


def _receptor_genes(g: Genome, c: int, k: int) -> np.ndarray:
    return np.concatenate([np.ravel(getattr(g, name)[c, k]) for name in Genome.RECEPTOR_FIELDS[1:]])


def _neuron_genes(g: Genome, c: int, h: int) -> np.ndarray:
    return np.concatenate([g.traits[c, h], g.weights[c, h] * g.conn[c, h], g.pos[c, h]])


def unequal_crossing_over(g: Genome, c: int, rng: np.random.Generator, split: bool = True) -> None:
    """Scambio ineguale (crossing-over ineguale): nella meiosi due copie della
    stessa famiglia possono appaiarsi per sbaglio, e il figlio riceve una
    copia in più o una in meno, con la stessa probabilità. In natura è il modo
    principale in cui le famiglie di geni crescono e si accorciano. In una
    famiglia di n copie avviene al più uno scambio per meiosi, con
    probabilità tasso di duplicazione × (n − 1) (più copie, più occasioni di
    appaiarsi male), e solo se le due copie scelte si somigliano: con
    probabilità pari alla loro somiglianza (1 se identiche, sempre meno man
    mano che divergono)."""
    p = float(np.exp(g.meta[c, META_INDEX["dup"]]))
    for on, fam, genes, copy in ((g.r_active, g.r_fam, _receptor_genes, None),
                                 (g.active, g.h_fam, _neuron_genes, duplicate_neuron)):
        slots = np.flatnonzero(on[c])
        for family in np.unique(fam[c, slots]):
            members = [int(k) for k in slots if fam[c, k] == family]
            if len(members) < 2 or rng.random() >= p * (len(members) - 1):
                continue
            a, b = rng.choice(members, 2, replace=False)
            if rng.random() >= _similarity(genes(g, c, a), genes(g, c, b)):
                continue
            if rng.random() < 0.5:  # una copia in meno
                on[c, b] = False
                continue
            free = np.flatnonzero(~on[c])  # una copia in più
            if len(free):
                if copy is None:
                    duplicate_receptor(g, c, int(a), int(free[0]), rng, split)
                else:
                    copy(g, c, int(a), int(free[0]), rng)
                on[c, free[0]] = True


def new_id(rng: np.random.Generator) -> int:
    """Identità di un gene nuovo (una copia): un numero a caso tra 2^40 e 2^62,
    diverso da tutte le identità dei posti e, in pratica, da ogni altra copia."""
    return int(rng.integers(2**40, 2**62))


def _turn(v: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c], axis=-1)


def duplicate_neuron(g: Genome, i: int, h: int, f: int, rng: np.random.Generator) -> None:
    """Il neurone interno h dell'individuo i si duplica nel posto libero f.

    La copia riceve gli stessi ingressi e manda le stesse uscite; le uscite di
    originale e copia vengono dimezzate. Così insieme fanno esattamente quello
    che faceva l'originale da solo: alla nascita la duplicazione non cambia il
    comportamento, ma la copia è libera di divergere. (Con la plasticità le
    due copie imparano ciascuna per conto suo: durante la vita la neutralità
    non è più esatta.)"""
    r = g.n_rec
    src_h, src_f = r + h, r + f
    for a in (g.weights, g.conn):
        a[i, :, src_f] = a[i, :, src_h]  # uscite (anche verso se stesso)
        a[i, f, :] = a[i, h, :]  # ingressi, compreso quello appena copiato
    g.weights[i, :, src_h] *= 0.5
    g.weights[i, :, src_f] *= 0.5
    g.traits[i, f] = g.traits[i, h]
    g.active[i, f] = True
    g.h_id[i, f], g.h_fam[i, f] = new_id(rng), g.h_fam[i, h]
    std = np.exp(g.meta[i, META_INDEX["pos_std"]])
    g.pos[i, f] = np.clip(g.pos[i, h] + rng.normal(0.0, std, 2), 0.0, 1.0)


def duplicate_receptor(g: Genome, i: int, k: int, f: int, rng: np.random.Generator, split: bool = False) -> None:
    """Il recettore k dell'individuo i si duplica nel posto libero f: stessi
    geni e stessi collegamenti.

    `split` (come nell'olfatto, dove ogni cellula esprime un solo gene
    recettore): si duplica il gene, non l'organo. Le cellule che ora
    esprimono il gene nuovo, l'equivalente di una cellula normale
    (sensibilità 1 sul fenomeno più forte, stesso profilo), passano alla
    copia con le loro sinapsi: la copia prende quella quota della
    sensibilità e dei pesi in uscita, l'originale la perde. Cellule, costo e
    segnale restano quelli di prima e la copia nasce quasi neutrale, libera di
    derivare. (Con i pesi interi la copia non era neutrale: un recettore con
    adattamento divisivo risponde al contrasto, quindi una cellula sola manda
    un segnale grande quanto tutto l'organo, solo più rumoroso.)

    Senza `split`, e per i recettori che valgono meno di due cellule (costano
    pochissimo e non lavorano a contrasto): copia intera e uscite dimezzate,
    esattamente neutrale."""
    for field in Genome.RECEPTOR_FIELDS:
        getattr(g, field)[i, f] = getattr(g, field)[i, k]
    for a in (g.weights, g.conn):
        a[i, :, f] = a[i, :, k]
    g.r_id[i, f], g.r_fam[i, f] = new_id(rng), g.r_fam[i, k]
    top = float(np.exp(g.r_sens[i, k]).max())
    share = 1.0 / top if split and top >= 2.0 else 0.5
    if split and top >= 2.0:
        g.r_sens[i, f] = g.r_sens[i, k] + np.log(share)
        g.r_sens[i, k] = g.r_sens[i, k] + np.log1p(-share)
    g.weights[i, :, f] *= share
    g.weights[i, :, k] *= 1.0 - share
    std = np.exp(g.meta[i, META_INDEX["pos_std"]])
    g.r_y[i, f] = np.clip(g.r_y[i, k] + rng.normal(0.0, std), 0.0, 1.0)


def _duplicate(g: Genome, rng: np.random.Generator, p: np.ndarray, split: bool = False) -> None:
    """Ogni neurone interno e ogni recettore acceso si duplica con la
    probabilità `p` (N,) del suo individuo, nel primo posto libero. Senza posti
    liberi la duplicazione non avviene (i posti crescono alla generazione
    dopo, vedi `grow`). `split`: vedi `duplicate_receptor`."""
    receptor = lambda g_, i, k, f, r: duplicate_receptor(g_, i, k, f, r, split)
    for active, copy in ((g.active, duplicate_neuron), (g.r_active, receptor)):
        events = np.argwhere(active & (rng.random(active.shape) < p[:, None]))
        for i, k in events:
            free = np.flatnonzero(~active[i])
            if len(free):
                copy(g, int(i), int(k), int(free[0]), rng)
                active[i, free[0]] = True


MAX_SENSITIVITY = 100.0  # volte una cellula normale: limite fisico della cattura dei quanti
RECEPTOR_GENE_LENGTH = 28  # geni di un recettore: acceso, 6 x (sensibilità, sintonia, larghezza), corpo,
# direzione, apertura, nervo, adattamento (τ, sottrattivo, divisivo)


def knockout(g: Genome, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probabilità che un recettore (N, R) e un neurone interno (N, H) accesi
    si rompano in questa nascita.

    Come in natura, rompere un gene si può quasi ovunque: la probabilità è
    1 − (1 − μ)^L, con μ la probabilità di mutazione per elemento genetico
    (gene "toggle" dell'individuo) e L la lunghezza del gene. Ripararlo invece
    richiede la mutazione giusta in un punto solo: probabilità μ. Il rapporto
    tra rottura e riparazione è quindi circa la lunghezza del gene, e i geni
    lunghi si rompono più spesso. Per un neurone L sono i suoi geni propri più
    i collegamenti in ingresso che porta: un neurone più collegato è un gene
    più lungo e più fragile. Un recettore ha lunghezza fissa."""
    m = np.minimum(1.0, mu)[:, None]
    n_traits = g.traits.shape[2]
    length = n_traits + g.conn[:, : g.n_hidden, :].sum(axis=2)
    return 1.0 - (1.0 - m) ** RECEPTOR_GENE_LENGTH * np.ones(g.r_active.shape), 1.0 - (1.0 - m) ** length


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
    # Una cellula non cattura più quanti di quanti ne arrivano: la sensibilità ha
    # un massimo fisico (anche nei recettori spenti, che altrimenti derivano
    # senza limiti e, riaccesi, costerebbero un'energia assurda).
    np.clip(g.r_sens, None, np.log(MAX_SENSITIVITY), out=g.r_sens)
    np.clip(g.r_width, np.log(0.005), np.log(10.0), out=g.r_width)
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
    dist = distances(cfg, g.pos, g.r_y, n_out, brain_side(cfg, g.r_active, g.active, n_out))
    shape = connect_prob(cfg, dist, n_rec, base("conn_range")) / cfg.net_conn_prob
    p_add = np.minimum(1.0, rate("conn_add")[:, None, None] * shape)
    appear = ~g.conn & (rng.random(g.conn.shape) < p_add)
    vanish = g.conn & chance(g.conn.shape, rate("conn_del"))
    g.conn ^= appear | vanish

    # 5. Neuroni e recettori si rompono o (raramente) si riparano; poi le
    # duplicazioni. Vedi `knockout`.
    mu = rate("toggle")
    on, off = knockout(g, mu)
    g.active = np.where(g.active, rng.random(g.active.shape) >= off,
                        rng.random(g.active.shape) < np.minimum(1.0, mu)[:, None])
    g.r_active = np.where(g.r_active, rng.random(g.r_active.shape) >= on,
                          rng.random(g.r_active.shape) < np.minimum(1.0, mu)[:, None])
    _duplicate(g, rng, np.minimum(1.0, rate("dup")), cfg.receptor_dup_split)


def reproduce(parents: Genome, n_children: int, cfg: Config, rng: np.random.Generator,
              strength: float = 1.0) -> Genome:
    """`n_children` figli da genitori scelti tutti con la stessa probabilità.

    Ogni figlio ha una madre; con la probabilità scritta nel gene "sex" della
    madre nasce con un partner (un altro genitore a caso), altrimenti è una sua
    copia. Poi tutti vengono mutati.
    """
    n_parents = len(parents)
    mothers = rng.integers(0, n_parents, n_children)
    fathers = mothers.copy()
    if n_parents > 1:
        two = rng.random(n_children) < parents.rate("sex")[mothers]
        # Uno scarto tra 1 e n-1 garantisce un padre diverso dalla madre.
        fathers[two] = (mothers[two] + rng.integers(1, n_parents, int(two.sum()))) % n_parents
    children = crossover(parents, mothers, fathers, rng, cfg.receptor_dup_split)
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
