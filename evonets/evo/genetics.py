import numpy as np

from ..config import Config
from ..net.brain import Genome


def crossover(parents: Genome, mothers: np.ndarray, fathers: np.ndarray,
              rng: np.random.Generator) -> Genome:
    """Un figlio per coppia (mothers[i], fathers[i]).

    Il genoma si mescola a blocchi interi, uno per neurone: per ogni neurone il
    figlio prende da un solo genitore l'esistenza, la soglia e tutti i
    collegamenti in ingresso. Così i pesi che lavorano insieme restano insieme.
    """
    n_blocks = parents.bias.shape[1]
    n_hidden = parents.active.shape[1]
    from_father = rng.random((len(mothers), n_blocks)) < 0.5
    return Genome(
        np.where(from_father[:, :, None], parents.weights[fathers], parents.weights[mothers]),
        np.where(from_father, parents.bias[fathers], parents.bias[mothers]),
        np.where(from_father[:, :, None], parents.conn[fathers], parents.conn[mothers]),
        np.where(from_father[:, :n_hidden], parents.active[fathers], parents.active[mothers]),
    )


def mutate(genome: Genome, cfg: Config, rng: np.random.Generator, strength: float = 1.0) -> None:
    """Muta sul posto. `strength` scala sia la frequenza sia l'ampiezza delle
    mutazioni (>1 = variazioni più consistenti)."""
    def chance(shape, p):
        return rng.random(shape) < min(1.0, p * strength)

    std = cfg.mut_weight_std * strength
    for values in (genome.weights, genome.bias):
        hit = chance(values.shape, cfg.mut_weight_prob)
        values[hit] += rng.normal(0.0, std, int(hit.sum()))

    # Un collegamento che compare usa il peso già scritto in quel posto del genoma.
    appear = ~genome.conn & chance(genome.conn.shape, cfg.mut_conn_add)
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
