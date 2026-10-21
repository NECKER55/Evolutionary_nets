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
from ..net import development
from ..net.brain import GENES, META, META_INDEX, PLASTIC, Genome, clamp_meta, pad
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
        _recombine(child, int(c), parents, int(mothers[c]), int(fathers[c]), rng)
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


def _recombine(g: Genome, c: int, p: Genome, mo: int, f: int, rng: np.random.Generator) -> None:
    """Il figlio c (copia della madre mo) riceve i blocchi del padre f."""
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
    if g.n_markers:
        cells = ({j: ("f", kf) for j, kf in rec_f.items()}, {t: ("f", tf) for t, tf in rows.items()})
        _recombine_markers(g, c, p, mo, f, cells, rng)


def _recombine_markers(g: Genome, c: int, p: Genome, mo: int, f: int, cells, rng) -> None:
    """I marcatori nell'incrocio (docs/SPEC_SVILUPPO.md, sezione 10). Ogni
    gene di marcatore (esistenza, gradiente, riga della tabella: come si lega
    sull'assone) passa intero da un genitore, abbinato per identità; uno solo
    di un genitore passa con probabilità 1/2. Le colonne della tabella e le
    espressioni nelle cellule seguono le identità: una cellula presa da un
    genitore esprime ogni marcatore come in quel genitore, se lui lo ha;
    altrimenti come nell'altro genitore, se ha la stessa cellula; altrimenti
    poco (gene 0). `cells` = ({recettore: fonte}, {destinatario: fonte}) per
    le cellule prese dal padre."""
    on, from_f, _ = _compose(p.m_active[mo], p.m_id[mo], p.m_active[f], p.m_id[f], rng)
    M = g.n_markers
    ids = p.m_id[mo].copy()
    src = [("m", k) for k in range(M)]
    for k, kf in from_f.items():
        ids[k], src[k] = p.m_id[f, kf], ("f", kf)
    g.m_active[c], g.m_id[c] = on, ids
    who = {"m": mo, "f": f}
    slot = {}  # identità -> posto, in ogni genitore (i marcatori accesi prevalgono su quelli spenti)
    for key, i in who.items():
        slot[key] = {int(p.m_id[i, k]): k for k in np.flatnonzero(~p.m_active[i])}
        slot[key].update({int(p.m_id[i, k]): k for k in np.flatnonzero(p.m_active[i])})
    for k, (key, kk) in enumerate(src):
        g.m_fam[c, k], g.m_grad[c, k] = p.m_fam[who[key], kk], p.m_grad[who[key], kk]
    # Tabella: la riga a viene dal genitore del marcatore a; ogni colonna b
    # dal posto che ha in quel genitore il marcatore b.
    for a, (key, ka) in enumerate(src):
        i = who[key]
        for b in range(M):
            kb = slot[key].get(int(ids[b]))
            g.m_table[c, a, b] = p.m_table[i, ka, kb] if kb is not None else 0.0

    def express(marks_p, cell_p_of, child_marks):
        """child_marks (cellule, M) del figlio; cell_p_of(key) = posto della
        stessa cellula nel genitore key (o None)."""
        for b in range(M):
            gid = int(ids[b])
            for cell in range(child_marks.shape[0]):
                own, other = cell_p_of(cell)
                value = 0.0
                for key, pc in (own, other):
                    if pc is not None and gid in slot[key]:
                        value = marks_p[who[key], pc, slot[key][gid]]
                        break
                child_marks[cell, b] = value

    rec_cells, tgt_cells = cells
    r_where = {key: {int(p.r_id[who[key], k]): k for k in range(p.n_rec)} for key in who}
    h_where = {key: {int(p.h_id[who[key], k]): k for k in range(p.n_hidden)} for key in who}

    def receptor(cell):
        key, pc = rec_cells.get(cell, ("m", cell))
        other = "m" if key == "f" else "f"
        return (key, pc), (other, r_where[other].get(int(g.r_id[c, cell])))

    def target(cell):
        key, pc = tgt_cells.get(cell, ("m", cell))
        other = "m" if key == "f" else "f"
        if cell >= g.n_hidden:  # motori: sempre gli stessi
            return (key, pc), (other, cell)
        return (key, pc), (other, h_where[other].get(int(g.h_id[c, cell])))

    express(p.r_mark, receptor, g.r_mark[c])
    express(p.n_mark, target, g.n_mark[c])


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
    kinds = [(g.r_active, g.r_fam, _receptor_genes, None), (g.active, g.h_fam, _neuron_genes, duplicate_neuron)]
    if g.n_markers:
        kinds.append((g.m_active, g.m_fam, development.marker_genes,
                      lambda g_, i, k, f, r: development.duplicate_marker(g_, i, k, f, new_id(r))))
    for on, fam, genes, copy in kinds:
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
    g.n_mark[i, f] = g.n_mark[i, h]  # stessa identità (sviluppo)
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
    g.r_mark[i, f] = g.r_mark[i, k]  # stessa identità (sviluppo)
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


def _duplicate(g: Genome, rng: np.random.Generator, p: np.ndarray, split: bool = False,
               receptors: bool = True) -> None:
    """Ogni neurone interno e ogni recettore acceso si duplica con la
    probabilità `p` (N,) del suo individuo, nel primo posto libero. Senza posti
    liberi la duplicazione non avviene (i posti crescono alla generazione
    dopo, vedi `grow`). `split`: vedi `duplicate_receptor`."""
    receptor = lambda g_, i, k, f, r: duplicate_receptor(g_, i, k, f, r, split)
    marker = lambda g_, i, k, f, r: development.duplicate_marker(g_, i, k, f, new_id(r))
    kinds = ((g.active, duplicate_neuron), (g.r_active, receptor)) if receptors else ((g.active, duplicate_neuron),)
    if g.n_markers:
        kinds += ((g.m_active, marker),)
    for active, copy in kinds:
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
    if g.n_markers:  # sviluppo: i collegamenti non sono geni, i marcatori espressi sì
        length = n_traits + np.broadcast_to(g.m_active.sum(axis=1)[:, None], g.active.shape)
    else:
        length = n_traits + g.conn[:, : g.n_hidden, :].sum(axis=2)
    return 1.0 - (1.0 - m) ** RECEPTOR_GENE_LENGTH * np.ones(g.r_active.shape), 1.0 - (1.0 - m) ** length


def mutate(genome: Genome, cfg: Config, rng: np.random.Generator) -> None:
    """Muta sul posto, con i tassi del figlio (ereditati dai genitori)."""
    g = genome
    n = len(g)
    inherited = np.exp(g.meta)  # i tassi di questa nascita: quelli ereditati

    def base(name):
        return inherited[:, META_INDEX[name]]

    def rate(name):
        return base(name)

    def chance(shape, p):
        return rng.random(shape) < np.minimum(1.0, p).reshape((n,) + (1,) * (len(shape) - 1))

    def noise(shape, std):
        return rng.standard_normal(shape) * std.reshape((n,) + (1,) * (len(shape) - 1))

    # 1. Pesi, geni propri dei neuroni e geni del meccanismo di mutazione:
    # questi ultimi sono geni come gli altri (in scala logaritmica: un ritocco
    # cambia un tasso di un fattore, non di una quantità fissa). I tassi usati
    # in questa nascita sono quelli ereditati, letti prima del ritocco.
    p_w, std_w = rate("weight_prob"), base("weight_std")
    fixed = cfg.sim_mode == "cervello"  # sensi fissi: i geni dei recettori non mutano mai
    developed = g.n_markers > 0  # sviluppo: i pesi non sono geni, i marcatori sì
    genes = [g.traits, g.meta] + ([g.n_mark, g.m_grad, g.m_table] if developed else [g.weights])
    if developed and not fixed:
        genes.append(g.r_mark)
    for values in genes:
        values += chance(values.shape, p_w) * noise(values.shape, std_w)
    clamp_meta(g.meta, cfg)

    # 2. Posizioni nel cervello: interni e ingressi dei recettori.
    p_pos, std_pos = rate("pos_prob"), base("pos_std")
    g.pos += chance(g.pos.shape[:2], p_pos)[:, :, None] * noise(g.pos.shape, std_pos)
    np.clip(g.pos, 0.0, 1.0, out=g.pos)
    if not fixed:
        g.r_y += chance(g.r_y.shape, p_pos) * noise(g.r_y.shape, std_pos)
        np.clip(g.r_y, 0.0, 1.0, out=g.r_y)
        _mutate_receptors(g, cfg, chance, noise, p_w, base("rec_drift"))

    # 4. Collegamenti (genoma diretto): uno nuovo compare con una probabilità
    # che cala con la distanza, come alla nascita, con la portata λ
    # dell'individuo. Con lo sviluppo i collegamenti non sono geni: nascono
    # dai marcatori.
    if not developed:
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
    if not fixed:
        g.r_active = np.where(g.r_active, rng.random(g.r_active.shape) >= on,
                              rng.random(g.r_active.shape) < np.minimum(1.0, mu)[:, None])
    if developed:  # anche un marcatore si rompe (più facilmente se lega molti altri) o si ripara
        length = development.MARKER_BASE_LENGTH + 2 * g.m_active.sum(axis=1)
        broken = 1.0 - (1.0 - np.minimum(1.0, mu)) ** length
        g.m_active = np.where(g.m_active, rng.random(g.m_active.shape) >= broken[:, None],
                              rng.random(g.m_active.shape) < np.minimum(1.0, mu)[:, None])
    _duplicate(g, rng, np.minimum(1.0, rate("dup")), cfg.receptor_dup_split, receptors=not fixed)


def _mutate_receptors(g: Genome, cfg: Config, chance, noise, p_w, drift) -> None:
    """Geni dei recettori: ognuno deriva nella sua scala."""
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


def reproduce(parents: Genome, n_children: int, cfg: Config, rng: np.random.Generator,
              fecundity: np.ndarray | None = None) -> Genome:
    """`n_children` figli dai genitori.

    Ogni figlio ha una madre; con la probabilità scritta nel gene "sex" della
    madre nasce con un partner (un altro genitore), altrimenti è una sua
    copia. Poi tutti vengono mutati. Madri e partner si scelgono con
    probabilità proporzionale a `fecundity` (per esempio il cibo mangiato in
    tutta la vita: più risorse, più figli, come in natura); senza, tutti i
    genitori valgono uguale.
    """
    n_parents = len(parents)
    if fecundity is not None and not np.sum(fecundity) > 0:
        fecundity = None  # nessuno ha risorse da contare: tutti uguali
    if fecundity is None:
        mothers = rng.integers(0, n_parents, n_children)
    else:
        weight = np.asarray(fecundity, dtype=float) / np.sum(fecundity)
        mothers = rng.choice(n_parents, n_children, p=weight)
    fathers = mothers.copy()
    if n_parents > 1:
        two = rng.random(n_children) < parents.rate("sex")[mothers]
        if fecundity is None:
            # Uno scarto tra 1 e n-1 garantisce un padre diverso dalla madre.
            fathers[two] = (mothers[two] + rng.integers(1, n_parents, int(two.sum()))) % n_parents
        else:
            for c in np.flatnonzero(two):  # il partner: un altro genitore, secondo la sua fecondità
                w = weight.copy()
                w[mothers[c]] = 0.0
                if w.sum() > 0:
                    fathers[c] = rng.choice(n_parents, p=w / w.sum())
    children = crossover(parents, mothers, fathers, rng, cfg.receptor_dup_split)
    mutate(children, cfg, rng)
    if children.n_markers:  # ogni figlio sviluppa il suo cervello; i genitori tengono il loro
        development.develop(children, cfg, rng, children.traits.shape[2] - len(GENES) + len(PLASTIC))
    return children


def grow(genome: Genome, cfg: Config) -> Genome:
    """Se qualcuno usa più di `pool_fill` dei posti per i recettori o per i
    neuroni interni, tutta la popolazione riceve `pool_step` posti nuovi,
    spenti. È solo memoria: per l'evoluzione non esiste un numero massimo."""
    r, h = genome.n_rec, genome.n_hidden
    if cfg.sim_mode != "cervello" and genome.r_active.sum(axis=1).max(initial=0) > cfg.pool_fill * r:
        r += cfg.pool_step
    if genome.active.sum(axis=1).max(initial=0) > cfg.pool_fill * h:
        h += cfg.pool_step
    m = genome.n_markers
    if m and genome.m_active.sum(axis=1).max(initial=0) > cfg.pool_fill * m:
        m += max(1, cfg.pool_step // 2)
    return pad(genome, r, h, m)


__all__ = ["META", "crossover", "grow", "mutate", "reproduce"]
