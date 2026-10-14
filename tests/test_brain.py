import numpy as np

from evonets import Config, World
from evonets.net import Brains
from evonets.net import NeuralController

N_IN, N_OUT = 10, 2


def brains(n=50, seed=0, **kw):
    return Brains(Config(**kw), n, N_IN, N_OUT, seed)


def test_shapes_and_output_range():
    b = brains()
    h = b.n_hidden
    assert b.weights.shape == (50, h + N_OUT, N_IN + h + N_OUT)
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = b.step(rng.uniform(-1, 1, (50, N_IN)))
        assert out.shape == (50, N_OUT) and (np.abs(out) <= 1.0).all()


def test_population_has_varied_hidden_sizes():
    counts = brains(n=200).active.sum(axis=1)
    assert counts.min() < 5 and counts.max() > 25


def test_missing_hidden_neurons_stay_silent_and_have_no_links():
    b = brains()
    rng = np.random.default_rng(1)
    for _ in range(10):
        b.step(rng.uniform(-1, 1, (50, N_IN)))
    off = ~b.active
    assert (b.state[:, : b.n_hidden][off] == 0.0).all()
    assert not b.links[:, : b.n_hidden, :][off].any()  # nessun ingresso verso di loro
    assert not b.links[:, :, N_IN : N_IN + b.n_hidden].transpose(0, 2, 1)[off].any()  # né uscita


def test_changing_the_genes_of_a_missing_neuron_changes_nothing():
    a, b = brains(seed=3), brains(seed=3)
    off = np.argwhere(~b.active)[0]
    b.weights[off[0], off[1], :] += 5.0
    b.weights[off[0], :, N_IN + off[1]] += 5.0
    b.compile()
    obs = np.random.default_rng(2).uniform(-1, 1, (4, 50, N_IN))
    for o in obs:
        assert np.array_equal(a.step(o), b.step(o))


def test_loops_give_memory():
    """Stesso ingresso, uscita diversa a seconda di ciò che è successo prima."""
    b = brains(net_conn_prob=0.5)
    rng = np.random.default_rng(4)
    probe = rng.uniform(-1, 1, (50, N_IN))
    for _ in range(5):
        b.step(rng.uniform(-1, 1, (50, N_IN)))
    after_history = b.step(probe).copy()
    b.reset_state()
    fresh = b.step(probe)
    assert not np.allclose(after_history, fresh)


def test_same_seed_same_population_and_reset_restores_behaviour():
    a, b = brains(seed=5), brains(seed=5)
    obs = np.random.default_rng(6).uniform(-1, 1, (50, N_IN))
    first = a.step(obs).copy()
    assert np.array_equal(first, b.step(obs))
    a.step(obs)
    a.reset_state()
    assert np.array_equal(a.step(obs), first)


def test_neural_controller_drives_the_world():
    cfg = Config()
    w = World(cfg, 30, seed=1)
    ctrl = NeuralController(cfg, 30, seed=1)
    assert ctrl.brains.n_in == cfg.vision_rays * 4 + 6
    start = w.pos.copy()
    for _ in range(200):
        w.step(ctrl.act(w))
    assert (np.linalg.norm(w.pos - start, axis=1) > 1.0).any()


def test_networks_are_computed_only_for_who_is_racing_with_identical_results():
    """Morti e arrivati non vengono più calcolati: per chi è in gara i comandi
    devono restare identici, bit per bit, al calcolo di tutta la popolazione."""
    for net in ("rate", "spike-ei"):
        cfg = Config(net_type=net)
        world = World(cfg, 40, seed=1)
        fast, full = NeuralController(cfg, 40, seed=2), NeuralController(cfg, 40, seed=2)
        for t in range(120):
            if t == 30:
                world.alive[::3] = False  # alcuni muoiono, altri arrivano
            if t == 60:
                world.done[1::5] &= world.alive[1::5]
            racing = world.alive & ~world.done
            a = fast.act(world)
            b = full.brains.step(full.senses.observe(world))
            assert np.array_equal(a[racing], b[racing])
            assert (a[~racing] == 0.0).all()
            world.step(b)
