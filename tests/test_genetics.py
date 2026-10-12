import numpy as np

from evonets import Config
from evonets.evo import crossover, mutate, reproduce
from evonets.net import Brains

N_IN, N_OUT = 10, 2
NO_MUTATION = dict(mut_weight_prob=0.0, mut_conn_add=0.0, mut_conn_del=0.0, mut_neuron_toggle=0.0)
FIELDS = ("weights", "bias", "conn", "active")


def population(n=40, seed=0, **kw):
    cfg = Config(**kw)
    return cfg, Brains(cfg, n, N_IN, N_OUT, seed).genome


def same(a, b) -> bool:
    return all(np.array_equal(getattr(a, f), getattr(b, f)) for f in FIELDS)


def test_copy_without_mutation_is_identical_and_independent():
    cfg, parents = population(crossover_rate=0.0, **NO_MUTATION)
    kids = reproduce(parents.take([3]), 5, cfg, np.random.default_rng(0))
    assert len(kids) == 5
    assert same(kids, parents.take([3, 3, 3, 3, 3]))
    kids.weights += 1.0
    assert not np.array_equal(kids.weights[0], parents.weights[3])  # il genitore non cambia


def test_crossover_takes_each_neuron_block_whole_from_one_parent():
    cfg, parents = population()
    rng = np.random.default_rng(1)
    mothers, fathers = np.zeros(30, dtype=int), np.ones(30, dtype=int)
    kids = crossover(parents, mothers, fathers, rng)
    h = parents.active.shape[1]
    from_mother = from_father = 0
    for k in range(30):
        for block in range(parents.bias.shape[1]):
            def is_from(p):
                ok = (np.array_equal(kids.weights[k, block], parents.weights[p, block])
                      and np.array_equal(kids.conn[k, block], parents.conn[p, block])
                      and kids.bias[k, block] == parents.bias[p, block])
                return ok and (block >= h or kids.active[k, block] == parents.active[p, block])
            m, f = is_from(0), is_from(1)
            assert m or f  # mai un miscuglio dentro lo stesso neurone
            from_mother += m
            from_father += f
    assert from_mother > 300 and from_father > 300  # contribuiscono entrambi, circa metà


def test_crossover_of_identical_parents_changes_nothing():
    cfg, parents = population()
    kids = crossover(parents, np.full(8, 5), np.full(8, 5), np.random.default_rng(2))
    assert same(kids, parents.take(np.full(8, 5)))


def test_two_parent_children_never_pick_the_same_parent_twice():
    cfg, parents = population(n=2, crossover_rate=1.0, **NO_MUTATION)
    kids = reproduce(parents, 200, cfg, np.random.default_rng(3))
    clones = sum(same(kids.take([k]), parents.take([p])) for k in range(200) for p in (0, 1))
    assert clones < 5  # un clone esatto richiederebbe 34 blocchi tutti dallo stesso genitore


def test_mutation_is_small_and_touches_every_kind_of_gene():
    cfg, g = population(n=200)
    before = g.take(np.arange(200))
    mutate(g, cfg, np.random.default_rng(4))
    changed_w = (g.weights != before.weights).mean()
    assert 0.08 < changed_w < 0.12  # ~mut_weight_prob
    assert np.abs(g.weights - before.weights).max() < 6 * cfg.mut_weight_std
    assert (g.bias != before.bias).any()
    flipped = g.conn != before.conn
    assert (flipped & ~before.conn).any() and (flipped & before.conn).any()  # compaiono e spariscono
    assert flipped.mean() < 0.01
    toggled = (g.active != before.active).mean()
    assert 0.003 < toggled < 0.03


def test_strength_makes_mutations_more_frequent_and_larger():
    cfg, a = population(n=200)
    _, b = population(n=200)
    start = a.take(np.arange(200))
    mutate(a, cfg, np.random.default_rng(5), strength=1.0)
    mutate(b, cfg, np.random.default_rng(5), strength=4.0)
    assert (b.conn != start.conn).sum() > 2.5 * (a.conn != start.conn).sum()
    assert (b.active != start.active).sum() > 2 * (a.active != start.active).sum()
    assert np.abs(b.weights - start.weights).mean() > 6 * np.abs(a.weights - start.weights).mean()


def test_same_rng_seed_gives_the_same_children():
    cfg, parents = population()
    a = reproduce(parents, 50, cfg, np.random.default_rng(6))
    b = reproduce(parents, 50, cfg, np.random.default_rng(6))
    assert same(a, b)


def test_children_are_working_networks():
    cfg, parents = population()
    kids = reproduce(parents.take([0, 1, 2]), 60, cfg, np.random.default_rng(7), strength=3.0)
    brains = Brains(cfg, 40, N_IN, N_OUT, 0)
    brains.set_genome(kids)
    assert brains.n == 60 and brains.state.shape == (60, brains.n_hidden + N_OUT)
    out = brains.step(np.random.default_rng(8).uniform(-1, 1, (60, N_IN)))
    assert out.shape == (60, N_OUT) and np.isfinite(out).all()
    assert (brains.state[:, : brains.n_hidden][~brains.active] == 0.0).all()
