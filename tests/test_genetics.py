import numpy as np

from evonets import Config
from evonets.evo import crossover, mutate, reproduce
from evonets.evo.genetics import duplicate_neuron, duplicate_receptor, grow
from evonets.net import Brains
from evonets.net.brain import META_INDEX, Genome
from nets import rich

R, N_OUT = 10, 2
# Tutti i tassi a zero (al minimo fisico) e nessuna deriva dei tassi stessi.
FROZEN = dict(mut_weight_prob=1e-12, mut_conn_add=1e-12, mut_conn_del=1e-12, mut_neuron_toggle=1e-12,
              mut_dup=1e-12, mut_pos_prob=1e-12, mut_rec_drift=1e-12, mutation_floor=1e-12)
FIELDS = Genome.FIELDS


def population(n=40, seed=0, **kw):
    b = rich(n, seed, **kw)
    return b.cfg, b.genome


def same(a, b, fields=FIELDS) -> bool:
    return all(np.array_equal(getattr(a, f), getattr(b, f)) for f in fields)


def test_copy_without_mutation_is_identical_and_independent():
    cfg, parents = population(crossover_rate=0.0, **FROZEN)
    kids = reproduce(parents.take([3]), 5, cfg, np.random.default_rng(0))
    assert len(kids) == 5
    assert same(kids, parents.take([3, 3, 3, 3, 3]), [f for f in FIELDS if f != "meta"])
    kids.weights += 1.0
    assert not np.array_equal(kids.weights[0], parents.weights[3])  # il genitore non cambia


def same_genes(parents):
    """I due genitori portano gli stessi geni (stessi neuroni e recettori accesi)."""
    parents.active[1] = parents.active[0]
    parents.r_active[1] = parents.r_active[0]
    return parents


def test_crossover_takes_each_block_whole_from_one_parent():
    cfg, parents = population()
    same_genes(parents)
    rng = np.random.default_rng(1)
    mothers, fathers = np.zeros(30, dtype=int), np.ones(30, dtype=int)
    kids = crossover(parents, mothers, fathers, rng)
    h = parents.n_hidden
    counts = [0, 0]
    for k in range(30):
        for block in range(parents.traits.shape[1]):
            def is_from(p):
                ok = (np.array_equal(kids.weights[k, block], parents.weights[p, block])
                      and np.array_equal(kids.conn[k, block], parents.conn[p, block])
                      and np.array_equal(kids.traits[k, block], parents.traits[p, block]))
                return ok and (block >= h or (kids.active[k, block] == parents.active[p, block]
                                              and np.array_equal(kids.pos[k, block], parents.pos[p, block])))
            m, f = is_from(0), is_from(1)
            assert m or f  # mai un miscuglio dentro lo stesso neurone
            counts[0] += m
            counts[1] += f
        for r in range(R):  # recettori: tutti i geni dallo stesso genitore
            src = [p for p in (0, 1) if all(np.array_equal(getattr(kids, f)[k, r], getattr(parents, f)[p, r])
                                           for f in Genome.RECEPTOR_FIELDS)]
            assert src
        assert any(np.array_equal(kids.meta[k], parents.meta[p]) for p in (0, 1))
    assert counts[0] > 100 and counts[1] > 100  # contribuiscono entrambi


def test_two_parent_children_never_pick_the_same_parent_twice():
    cfg, parents = population(n=2, **FROZEN)
    same_genes(parents)
    parents.meta[:, META_INDEX["sex"]] = 0.0  # le madri si riproducono sempre con un partner
    kids = reproduce(parents, 200, cfg, np.random.default_rng(3))
    clones = sum(same(kids.take([k]), parents.take([p]), ("weights", "conn", "r_sens"))
                 for k in range(200) for p in (0, 1))
    assert clones < 5


def test_mutation_rates_are_genes_that_are_inherited_and_drift():
    cfg, g = population(n=300)
    g.meta[:, META_INDEX["weight_prob"]] = np.log(np.where(np.arange(300) < 150, 0.02, 0.4))
    before = g.take(np.arange(300))
    mutate(g, cfg, np.random.default_rng(4))
    changed = (g.weights != before.weights).mean(axis=(1, 2))
    assert changed[:150].mean() < 0.05 < 0.3 < changed[150:].mean()  # ognuno muta coi suoi tassi
    # I geni dei tassi sono geni come gli altri: mutano con la stessa frequenza.
    meta_changed = (g.meta != before.meta).mean(axis=1)
    assert meta_changed[:150].mean() < 0.06 and 0.3 < meta_changed[150:].mean() < 0.5


def test_mutation_rates_never_fall_below_the_copying_floor():
    cfg, g = population(n=50, mut_weight_prob=1.0, mut_weight_std=5.0)
    for k in range(5):
        mutate(g, cfg, np.random.default_rng(k))
    assert (g.meta >= np.log(cfg.mutation_floor) - 1e-12).all()
    probs = [META_INDEX[k] for k in ("weight_prob", "conn_add", "conn_del", "toggle", "dup", "pos_prob")]
    assert (g.meta[:, probs] <= 0.0).all()


def test_mutation_touches_every_kind_of_gene():
    cfg, g = population(n=200)
    before = g.take(np.arange(200))
    mutate(g, cfg, np.random.default_rng(5))
    flipped = g.conn != before.conn
    assert (flipped & ~before.conn).any() and (flipped & before.conn).any()  # compaiono e spariscono
    for f in ("traits", "pos", "r_sens", "r_mu", "r_width", "r_look", "r_body", "r_aperture", "r_y"):
        assert (getattr(g, f) != getattr(before, f)).any(), f
    assert (g.active != before.active).any() and (g.r_active != before.r_active).any()


def test_strength_makes_mutations_more_frequent_and_larger():
    cfg, a = population(n=200)
    _, b = population(n=200)
    start = a.take(np.arange(200))
    mutate(a, cfg, np.random.default_rng(5), strength=1.0)
    mutate(b, cfg, np.random.default_rng(5), strength=4.0)
    assert (b.conn != start.conn).sum() > 2.5 * (a.conn != start.conn).sum()
    assert np.abs(b.weights - start.weights).mean() > 6 * np.abs(a.weights - start.weights).mean()


def test_same_rng_seed_gives_the_same_children():
    cfg, parents = population()
    a = reproduce(parents, 50, cfg, np.random.default_rng(6))
    b = reproduce(parents, 50, cfg, np.random.default_rng(6))
    assert same(a, b)


def test_neuron_duplication_does_not_change_behaviour():
    """Senza apprendimento è esatta; con la plasticità le due copie imparano
    ciascuna per conto suo, quindi la neutralità vale solo alla nascita."""
    for net in ("rate", "spike-ei"):
        b = rich(20, seed=8, net=net, conn=0.4, plasticity=False)
        g = b.genome
        for i in range(20):  # ogni individuo duplica un suo neurone acceso in un posto libero
            on, off = np.flatnonzero(g.active[i]), np.flatnonzero(~g.active[i])
            if len(on) and len(off):
                g.conn[i, off[0], :] = True  # geni vecchi nel posto libero: devono sparire
                duplicate_neuron(g, i, int(on[0]), int(off[0]), np.random.default_rng(i))
        dup = type(b)(b.cfg, 20, b.n_out, 0)
        dup.set_genome(g.take(np.arange(20)))
        orig = rich(20, seed=8, net=net, conn=0.4, plasticity=False)
        obs = np.random.default_rng(9).uniform(0, 1, (30, 20, R))
        for o in obs:
            assert np.allclose(orig.step(o), dup.step(o), atol=1e-12), net


def test_receptor_duplication_copies_genes_and_halves_outputs():
    b = rich(5, seed=10, R=6)
    g = b.genome
    g.r_active[0, 5] = False
    w_before = g.weights[0, :, 2].copy()
    duplicate_receptor(g, 0, 2, 5, np.random.default_rng(0))
    assert g.r_active[0, 5] and np.array_equal(g.r_sens[0, 5], g.r_sens[0, 2])
    assert np.allclose(g.weights[0, :, 2], w_before / 2) and np.allclose(g.weights[0, :, 5], w_before / 2)


def test_a_receptor_gene_duplicates_like_in_smell_one_cell_moves_to_the_copy():
    b = rich(5, seed=10, R=6)
    g = b.genome
    g.r_active[0, 5] = False
    g.r_sens[0, 2] = np.log([60.0, 2.0, 1.0, 0.5, 0.5, 0.5])  # un naso maturo
    total = np.exp(g.r_sens[0, 2]).copy()
    w_before = g.weights[0, :, 2].copy()
    duplicate_receptor(g, 0, 2, 5, np.random.default_rng(0), split=True)
    copy, orig = np.exp(g.r_sens[0, 5]), np.exp(g.r_sens[0, 2])
    assert np.isclose(copy.max(), 1.0) and np.allclose(copy / copy.max(), total / total.max())  # stesso profilo
    assert np.allclose(copy + orig, total)  # cellule (e costo) invariate
    # Le sinapsi seguono le cellule: la copia ne prende la quota, l'originale la perde.
    assert np.allclose(g.weights[0, :, 5], w_before / 60) and np.allclose(g.weights[0, :, 2], w_before * 59 / 60)
    assert np.array_equal(g.r_mu[0, 5], g.r_mu[0, 2]) and np.array_equal(g.conn[0, :, 5], g.conn[0, :, 2])
    # Un recettore di meno di due cellule (economico) si copia intero, uscite dimezzate.
    g.r_active[0, 4] = False
    g.r_sens[0, 1] = np.log(np.full(6, 1.5))
    w_small = g.weights[0, :, 1].copy()
    duplicate_receptor(g, 0, 1, 4, np.random.default_rng(0), split=True)
    assert np.allclose(np.exp(g.r_sens[0, 1]), 1.5) and np.allclose(np.exp(g.r_sens[0, 4]), 1.5)
    assert np.allclose(g.weights[0, :, 4], w_small / 2) and np.allclose(g.weights[0, :, 1], w_small / 2)


def test_slots_grow_when_someone_fills_them():
    cfg = Config()
    g = Brains(cfg, 10, N_OUT, 0).genome
    assert grow(g, cfg).n_hidden == cfg.net_hidden_slots  # nessuno li usa: non crescono
    g.active[3] = True
    big = grow(g, cfg)
    assert big.n_hidden == cfg.net_hidden_slots + cfg.pool_step
    assert np.array_equal(big.active[:, : cfg.net_hidden_slots], g.active) and not big.active[:, -1].any()


def test_children_are_working_networks():
    cfg, parents = population()
    kids = reproduce(parents.take([0, 1, 2]), 60, cfg, np.random.default_rng(7), strength=3.0)
    brains = Brains(cfg, 4, N_OUT, 0)
    brains.set_genome(kids)
    assert brains.n == 60 and brains.state.shape == (60, brains.n_hidden + N_OUT)
    out = brains.step(np.random.default_rng(8).uniform(0, 1, (60, brains.n_in)))
    assert out.shape == (60, N_OUT) and np.isfinite(out).all()
    assert (brains.state[:, : brains.n_hidden][~brains.active] == 0.0).all()


def test_a_birth_uses_the_inherited_rates_not_the_ones_it_passes_on():
    cfg, g = population(n=200)
    g.meta[:, META_INDEX["conn_del"]] = np.log(1e-12)  # nessun collegamento sparisce...
    g.meta[:, META_INDEX["dup"]] = np.log(1e-12)  # (né viene riscritto da una duplicazione)
    before = g.take(np.arange(200))
    mutate(g, cfg, np.random.default_rng(11))
    assert not (before.conn & ~g.conn).any()  # ...anche se il gene del tasso muta in questa nascita


def test_breaking_is_easier_than_repairing_and_long_genes_break_more():
    from evonets.evo.genetics import RECEPTOR_GENE_LENGTH, knockout
    cfg, g = population(n=50, conn=0.0)
    g.conn[:, 0, :] = True  # il neurone 0 riceve da tutti: gene lungo
    mu = np.full(50, 0.001)
    rec, neu = knockout(g, mu)
    assert np.allclose(rec, 1 - 0.999 ** RECEPTOR_GENE_LENGTH)
    assert (neu[:, 0] > 3 * neu[:, 1]).all()  # più collegamenti, più fragile (31 geni contro 7)
    assert (neu[:, 1] > 2 * 0.001).all()  # rompere è più probabile che riparare (μ)


def test_without_selection_brains_do_not_grow_by_themselves():
    cfg = Config()
    rng = np.random.default_rng(0)
    g = Brains(cfg, 100, N_OUT, 0).genome
    for _ in range(80):
        g = grow(reproduce(g, 100, cfg, rng), cfg)
    assert g.active.sum(axis=1).mean() < 4 and g.r_active.sum(axis=1).mean() < 4


def test_receptor_sensitivity_has_a_physical_maximum():
    from evonets.evo.genetics import MAX_SENSITIVITY
    cfg, g = population(n=20)
    g.r_sens[:] = 50.0  # derivata all'infinito
    mutate(g, cfg, np.random.default_rng(0))
    assert np.exp(g.r_sens).max() <= MAX_SENSITIVITY * (1 + 1e-9)


def test_homologous_genes_pair_wherever_they_are_and_links_follow_them():
    cfg, parents = population(n=2)
    g = parents
    g.r_active[:] = False
    g.r_active[0, 1] = g.r_active[1, 3] = True  # lo stesso gene, in posti diversi
    g.r_id[0, 1] = g.r_id[1, 3] = g.r_fam[0, 1] = g.r_fam[1, 3] = 77
    g.active[:] = False  # niente neuroni interni: solo recettore -> motori
    H = g.n_hidden
    g.conn[:, :, 3] = False  # nessun collegamento latente dal posto 3
    g.conn[1, H:, 3], g.weights[1, H:, 3] = True, 2.5  # nel padre il recettore parla ai motori
    kids = crossover(g, np.zeros(200, dtype=int), np.ones(200, dtype=int), np.random.default_rng(0))
    assert (kids.r_active[:, 1]).all() and not kids.r_active[:, 3].any()  # resta nel posto della madre
    from_father = kids.conn[:, H:, 1] & np.isclose(kids.weights[:, H:, 1], 2.5)
    assert from_father.any() and not kids.conn[:, H:, 3].any()  # i collegamenti del padre seguono il gene


def test_a_gene_only_one_parent_has_passes_to_half_the_children():
    cfg, parents = population(n=2)
    g = parents
    g.r_active[:] = False
    g.r_active[0, 0] = g.r_active[1, 0] = True  # un gene comune
    g.r_active[1, 2] = True
    g.r_id[1, 2] = 999  # e uno che ha solo il padre
    kids = crossover(g, np.zeros(2000, dtype=int), np.ones(2000, dtype=int), np.random.default_rng(1))
    has = (kids.r_id == 999) & kids.r_active
    assert has.sum(axis=1).max() == 1 and 0.45 < has.any(axis=1).mean() < 0.55
    assert kids.r_active[:, 0].all()


def test_unequal_crossing_over_adds_or_removes_a_copy_of_similar_genes():
    from evonets.evo.genetics import unequal_crossing_over

    cfg, g = population(n=1)
    g.r_active[:] = False
    g.r_active[0, :2] = True
    for name in Genome.RECEPTOR_FIELDS[1:]:  # due copie identiche della stessa famiglia
        getattr(g, name)[0, 1] = getattr(g, name)[0, 0]
    g.r_fam[0, 1] = g.r_fam[0, 0]
    g.active[:] = False
    g.meta[0, META_INDEX["dup"]] = 0.0  # tasso 1: con copie identiche avviene sempre
    counts = []
    for s in range(400):
        k = g.take([0])
        unequal_crossing_over(k, 0, np.random.default_rng(s))
        counts.append(int(k.r_active[0].sum()))
    assert set(counts) == {1, 3} and 0.4 < np.mean(np.array(counts) == 3) < 0.6
    g.r_mu[0, 1] = 1.0 - g.r_mu[0, 0]  # copie molto diverse: lo scambio si fa raro
    g.r_sens[0, 1] += 3.0
    events = 0
    for s in range(400):
        k = g.take([0])
        unequal_crossing_over(k, 0, np.random.default_rng(s))
        events += int(k.r_active[0].sum() != 2)
    assert events < 200


def test_how_often_mothers_mate_is_their_own_gene():
    def clones(sex):
        cfg, parents = population(n=4, **FROZEN)
        parents.meta[:, META_INDEX["sex"]] = np.log(sex)
        kids = reproduce(parents, 200, cfg, np.random.default_rng(5))
        return sum(any(same(kids.take([k]), parents.take([p]), ("weights", "conn")) for p in range(4))
                   for k in range(200))

    assert clones(1e-9) == 200  # madri che non si accoppiano mai: solo copie
    assert clones(1.0) < 20  # madri che si accoppiano sempre
