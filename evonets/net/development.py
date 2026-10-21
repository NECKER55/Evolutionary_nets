"""Lo sviluppo del cervello (docs/SPEC_SVILUPPO.md): con `Config.genome =
"sviluppo"` il genoma non scrive le sinapsi una per una, ma le regole per
costruirle, come in natura (chemoaffinità di Sperry).

Ogni cellula (recettore, neurone interno, motore) esprime dei **marcatori**,
molecole di superficie: quanto ne esprime dipende da un gene della cellula e,
per i marcatori a gradiente, dalla sua posizione nel cervello (come Eph ed
efrine, i cui gradienti costruiscono le mappe topografiche). Una **tabella di
affinità**, gene dell'individuo, dice quanto un marcatore sull'assone si lega
a un marcatore sul bersaglio, con il segno. Alla nascita:

1. affinità(s → t) = Σ_a Σ_b espressione_s[a] · tabella[a, b] · espressione_t[b];
2. il collegamento nasce con probabilità
   p = p_geometria(d) · (1 − e^(−|affinità|)): senza affinità non nasce nulla,
   con affinità forte la probabilità satura, la distanza conta come sempre
   (`geometry.connect_prob`, con la portata λ dell'individuo);
3. il peso iniziale è affinità · e^guadagno, con il guadagno gene del neurone
   che riceve, limitato a ±`synapse_max`. Nelle reti a impulsi il segno viene
   dal tipo del neurone che trasmette (legge di Dale): l'affinità dà la forza.

Lo sviluppo è in parte casuale (quali collegamenti nascono), come in natura:
fratelli identici hanno cervelli simili ma non uguali sinapsi per sinapsi. Il
risultato, il cervello sviluppato, sta in `Genome.weights` e `Genome.conn` e
appartiene all'individuo.

Nella simulazione "solo cervello" anche i marcatori dei sensi sono fissi:
uguali per tutti, senza gradiente, mai mutati.
"""

import numpy as np

from .geometry import brain_side, connect_prob, distances, motor_anchors

GRADIENT_SPAN = 1.0  # le posizioni nel cervello vanno da -0,5 a +0,5 attorno al centro
# Lunghezza del gene di un marcatore per la probabilità di rompersi (vedi
# genetics.knockout): esistenza, gradiente (2), più una riga e una colonna
# della tabella per ogni marcatore esistente.
MARKER_BASE_LENGTH = 3


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def positions(g) -> np.ndarray:
    """(N, S, 2): dove sta ogni cellula nel cervello, in [0, 1]: recettori sul
    bordo sinistro all'altezza del loro nervo, interni dove dice il loro gene,
    motori sul bordo destro."""
    n = len(g)
    receptors = np.stack([np.zeros(g.r_y.shape), g.r_y], axis=-1)
    motors = np.broadcast_to(motor_anchors(g.n_out), (n, g.n_out, 2))
    return np.concatenate([receptors, g.pos, motors], axis=1)


def expression(g, fixed_senses: bool = False) -> np.ndarray:
    """(N, S, M): quanto ogni cellula esprime ogni marcatore, tra 0 e 1. I
    marcatori spenti non si esprimono. Con `fixed_senses` i recettori non
    seguono i gradienti (sensi dati, uguali per tutti)."""
    xy = positions(g) - 0.5  # (N, S, 2)
    drive = np.concatenate([g.r_mark, g.n_mark], axis=1)  # (N, S, M)
    gradient = np.einsum("nsd,nmd->nsm", xy, g.m_grad) * GRADIENT_SPAN
    if fixed_senses:
        gradient[:, : g.n_rec] = 0.0
    return sigmoid(drive + gradient) * g.m_active[:, None, :]


def affinity(g, fixed_senses: bool = False) -> np.ndarray:
    """(N, T, S): affinità di ogni sorgente s per ogni bersaglio t."""
    e = expression(g, fixed_senses)
    targets = e[:, g.n_rec :]
    return np.einsum("nsa,nab,ntb->nts", e, g.m_table, targets)


def develop(g, cfg, rng: np.random.Generator, gain_index: int) -> None:
    """Sul posto: il cervello sviluppato (`weights`, `conn`) di tutti gli
    individui di `g`, dai loro marcatori. `gain_index` è il posto del gene del
    guadagno nei geni propri dei neuroni."""
    fixed = cfg.sim_mode == "cervello"
    a = affinity(g, fixed)
    side = brain_side(cfg, g.r_active, g.active, g.n_out)
    dist = distances(cfg, g.pos, g.r_y, g.n_out, side)
    reach = np.exp(g.meta[:, _conn_range_index()])
    p = connect_prob(cfg, dist, g.n_rec, reach) * (1.0 - np.exp(-np.abs(a)))
    g.conn = rng.random(p.shape) < p
    gain = np.exp(g.traits[:, :, gain_index])  # (N, T)
    g.weights = np.clip(a * gain[:, :, None], -cfg.synapse_max, cfg.synapse_max) * g.conn


def _conn_range_index() -> int:
    from .brain import META_INDEX

    return META_INDEX["conn_range"]


def initial_markers(cfg, n: int, n_rec: int, n_targets: int, rng: np.random.Generator) -> dict:
    """I geni dei marcatori alla partenza (docs/SPEC_SVILUPPO.md, sezione 6):
    `marker_initial` marcatori accesi su `marker_slots` posti, espressioni,
    gradienti e tabella a caso, diversi da individuo a individuo. Nella
    simulazione "solo cervello" le espressioni dei sensi sono fisse, uguali
    per tutti (estratte con un seme del set di sensi)."""
    m, on = cfg.marker_slots, min(cfg.marker_initial, cfg.marker_slots)
    std = cfg.marker_initial_std
    r_mark = rng.normal(0.0, std, (n, n_rec, m))
    if cfg.sim_mode == "cervello":
        senses = np.random.default_rng(_seed(cfg.sensor_set)).normal(0.0, std, (n_rec, m))
        r_mark = np.broadcast_to(senses, (n, n_rec, m)).copy()
    active = np.zeros((n, m), dtype=bool)
    active[:, :on] = True
    return dict(r_mark=r_mark, n_mark=rng.normal(0.0, std, (n, n_targets, m)),
                m_grad=rng.normal(0.0, cfg.marker_gradient_std, (n, m, 2)),
                m_table=rng.normal(0.0, cfg.marker_table_std, (n, m, m)), m_active=active)


def _seed(name: str) -> int:
    """Un seme fisso per un set di sensi (lo stesso a ogni esecuzione)."""
    return sum((k + 1) * ord(ch) for k, ch in enumerate(name)) + 1009


def duplicate_marker(g, i: int, k: int, f: int, new_id: int) -> None:
    """Il marcatore k dell'individuo i si duplica nel posto libero f. La copia
    è espressa uguale in tutte le cellule, con lo stesso gradiente; riga e
    colonna della tabella si dividono a metà tra originale e copia, così
    ogni affinità resta identica: alla nascita la duplicazione non cambia il
    cervello, e la copia è libera di divergere."""
    for marks in (g.r_mark, g.n_mark):
        marks[i, :, f] = marks[i, :, k]
    g.m_grad[i, f] = g.m_grad[i, k]
    t = g.m_table[i]
    t[f, :] = t[k, :]
    t[:, f] = t[:, k]
    t[f, f] = t[k, k]
    t[[k, f], :] *= 0.5
    t[:, [k, f]] *= 0.5
    g.m_active[i, f] = True
    g.m_id[i, f], g.m_fam[i, f] = new_id, g.m_fam[i, k]


def marker_genes(g, c: int, k: int) -> np.ndarray:
    """I geni del marcatore k dell'individuo c (per la somiglianza tra copie)."""
    return np.concatenate([g.r_mark[c, :, k], g.n_mark[c, :, k], g.m_grad[c, k], g.m_table[c, k, :],
                           g.m_table[c, :, k]])
