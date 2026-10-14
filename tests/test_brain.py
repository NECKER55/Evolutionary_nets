import numpy as np

from evonets import Config, World
from evonets.net import Brains, NeuralController
from evonets.net.brain import pad
from nets import rich

R, N_OUT = 10, 2


def test_shapes_and_output_range():
    b = rich()
    h = b.n_hidden
    assert b.weights.shape == (50, h + N_OUT, R + h + N_OUT)
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = b.step(rng.uniform(0, 1, (50, R)))
        assert out.shape == (50, N_OUT) and (np.abs(out) <= 1.0).all()


def test_primordial_start_is_minimal():
    cfg = Config()
    b = Brains(cfg, 100, N_OUT, 0)
    assert b.n_in == cfg.receptor_slots and b.n_hidden == cfg.net_hidden_slots
    assert (b.genome.r_active.sum(axis=1) == cfg.receptor_initial).all()
    assert not b.active.any()  # nessun neurone interno acceso
    gain = b.receptors.gain[:, : cfg.receptor_initial]
    assert np.allclose(np.median(gain), cfg.receptor_initial_gain, rtol=0.2)  # generalisti


def test_missing_neurons_and_receptors_stay_silent_and_have_no_links():
    b = rich()
    b.genome.r_active[:, 3] = False
    b.set_genome(b.genome)
    rng = np.random.default_rng(1)
    for _ in range(10):
        b.step(rng.uniform(0, 1, (50, R)))
    off = ~b.active
    assert (b.state[:, : b.n_hidden][off] == 0.0).all()
    assert not b.links[:, : b.n_hidden, :][off].any()  # nessun ingresso verso di loro
    assert not b.links[:, :, R : R + b.n_hidden].transpose(0, 2, 1)[off].any()  # né uscita
    assert not b.links[:, :, 3].any()  # il recettore spento non manda nulla


def test_changing_the_genes_of_a_missing_neuron_changes_nothing():
    a, b = rich(seed=3), rich(seed=3)
    off = np.argwhere(~b.active)[0]
    b.weights[off[0], off[1], :] += 5.0
    b.weights[off[0], :, R + off[1]] += 5.0
    b.compile()
    obs = np.random.default_rng(2).uniform(0, 1, (4, 50, R))
    for o in obs:
        assert np.array_equal(a.step(o), b.step(o))


def test_loops_give_memory():
    """Stesso ingresso, uscita diversa a seconda di ciò che è successo prima."""
    b = rich(conn=0.5)
    rng = np.random.default_rng(4)
    probe = rng.uniform(0, 1, (50, R))
    for _ in range(5):
        b.step(rng.uniform(0, 1, (50, R)))
    after_history = b.step(probe).copy()
    b.reset_state()
    fresh = b.step(probe)
    assert not np.allclose(after_history, fresh)


def test_same_seed_same_population_and_reset_restores_behaviour():
    a, b = rich(seed=5), rich(seed=5)
    obs = np.random.default_rng(6).uniform(0, 1, (50, R))
    first = a.step(obs).copy()
    assert np.array_equal(first, b.step(obs))
    a.step(obs)
    a.reset_state()
    assert np.array_equal(a.step(obs), first)


def test_padding_adds_switched_off_slots_and_changes_nothing():
    b = rich(seed=7)
    obs = np.random.default_rng(8).uniform(0, 1, (3, 50, R))
    before = [b.step(o).copy() for o in obs]
    g = pad(b.genome, R + 4, b.n_hidden + 8)
    assert g.n_rec == R + 4 and g.n_hidden == b.n_hidden + 8
    assert not g.r_active[:, R:].any() and not g.active[:, -8:].any()
    big = Brains(b.cfg, 50, N_OUT, 0)
    big.set_genome(g)
    padded_obs = [np.concatenate([o, np.zeros((50, 4))], axis=1) for o in obs]
    assert all(np.array_equal(big.step(o), x) for o, x in zip(padded_obs, before))


def test_upkeep_counts_wiring_neurons_and_receptors():
    b = rich(seed=9)
    cfg, g = b.cfg, b.genome
    expected = (cfg.wire_cost * b.wire_length + cfg.neuron_cost * b.active.sum(axis=1)
                + cfg.receptor_cell_cost * g.r_active.sum(axis=1)
                + cfg.receptor_sens_cost * (np.exp(g.r_sens) * g.r_active[:, :, None]).sum(axis=(1, 2)))
    assert np.allclose(b.upkeep, expected)


def test_neural_controller_drives_the_world():
    cfg = Config()
    w = World(cfg, 30, seed=1)
    ctrl = NeuralController(cfg, 30, seed=1)
    start = w.pos.copy()
    for _ in range(200):
        w.step(ctrl.act(w), ctrl.upkeep)
    assert (np.linalg.norm(w.pos - start, axis=1) > 1.0).any()


def test_networks_are_computed_only_for_who_is_racing_with_identical_results():
    """Morti e arrivati non vengono più calcolati: per chi è in gara i comandi
    devono restare identici, bit per bit, al calcolo di tutta la popolazione.
    (Che anche i sensi letti solo per alcuni siano identici lo controlla
    test_receptors.)"""
    for net in ("rate", "spike-ei"):
        fast, full = rich(40, seed=2, net=net), rich(40, seed=2, net=net)
        rng = np.random.default_rng(3)
        racing = np.ones(40, dtype=bool)
        for t in range(120):
            if t == 30:
                racing[::3] = False
            if t == 60:
                racing[1::5] = False
            obs = rng.uniform(0, 1, (40, R))
            b = full.step(obs)
            rows = np.flatnonzero(racing)
            a = fast.step(obs) if racing.all() else fast.step_some(obs[rows], rows)
            assert np.array_equal(a, b if racing.all() else b[rows])
