import numpy as np

from ..config import Config
from ..net.brain import Genome
from ..net.geometry import connect_prob, distances


def crossover(parents: Genome, mothers: np.ndarray, fathers: np.ndarray,
              rng: np.random.Generator) -> Genome:
    """Un figlio per coppia (mothers[i], fathers[i]).

    Il genoma si mescola a blocchi interi, uno per neurone: per ogni neurone il
    figlio prende da un solo genitore l'esistenza, i geni propri e tutti i
    collegamenti in ingresso, e un neurone interno porta con sé la sua
    posizione. Così i pesi che lavorano insieme restano insieme.
    """
    n_blocks = parents.traits.shape[1]
    n_hidden = parents.active.shape[1]
    from_father = rng.random((len(mothers), n_blocks)) < 0.5
    return Genome(
        np.where(from_father[:, :, None], parents.weights[fathers], parents.weights[mothers]),
        np.where(from_father[:, :, None], parents.traits[fathers], parents.traits[mothers]),
        np.where(from_father[:, :, None], parents.conn[fathers], parents.conn[mothers]),
        np.where(from_father[:, :n_hidden], parents.active[fathers], parents.active[mothers]),
        np.where(from_father[:, :n_hidden, None], parents.pos[fathers], parents.pos[mothers]),
    )


def mutate(genome: Genome, cfg: Config, rng: np.random.Generator, strength: float = 1.0) -> None:
    """Muta sul posto. `strength` scala sia la frequenza sia l'ampiezza delle
    mutazioni (>1 = variazioni più consistenti)."""
    def chance(shape, p):
        return rng.random(shape) < min(1.0, p * strength)

    std = cfg.mut_weight_std * strength
    for values in (genome.weights, genome.traits):
        hit = chance(values.shape, cfg.mut_weight_prob)
        values[hit] += rng.normal(0.0, std, int(hit.sum()))

    # I neuroni interni si spostano di poco, restando nel cervello.
    moved = chance(genome.pos.shape[:2], cfg.mut_pos_prob)
    genome.pos[moved] += rng.normal(0.0, cfg.mut_pos_std * strength, (int(moved.sum()), 2))
    np.clip(genome.pos, 0.0, 1.0, out=genome.pos)

    # Un collegamento compare con una probabilità che cala con la distanza,
    # come alla nascita: mut_conn_add tra neuroni vicinissimi, molto meno tra
    # neuroni lontani. Usa il peso già scritto in quel posto del genoma.
    t, s = genome.conn.shape[1:]
    n_in, n_out = s - t, t - genome.active.shape[1]
    shape = connect_prob(cfg, distances(cfg, genome.pos, n_in, n_out), n_in) / cfg.net_conn_prob
    p_add = np.minimum(1.0, cfg.mut_conn_add * strength * shape)
    appear = ~genome.conn & (rng.random(genome.conn.shape) < p_add)
    vanish = genome.conn & chance(genome.conn.shape, cfg.mut_conn_del)
    genome.conn ^= appear | vanish

    genome.active ^= chance(genome.active.shape, cfg.mut_neuron_toggle)


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
