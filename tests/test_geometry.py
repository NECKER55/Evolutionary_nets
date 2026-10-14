import numpy as np
import pytest

from evonets import Config, World
from evonets.evo import Evolution
from evonets.evo.genetics import crossover, mutate
from evonets.net import NeuralController
from evonets.net.geometry import anchors, connect_prob, distances

N_IN = 42  # 9 raggi x 4 canali + 6 sensi del corpo


def test_senses_on_the_left_like_a_retina_motors_on_the_right():
    cfg = Config()
    senses, motors = anchors(cfg, N_IN, 2)
    assert (senses[:, 0] == 0.0).all() and (motors[:, 0] == 1.0).all()
    assert (np.diff(senses[:, 1]) > 0).all()  # raggi dall'alto in basso, poi il corpo
    assert senses[35, 1] < senses[36, 1]  # la vista sopra, il corpo sotto


def test_link_probability_falls_with_distance_and_self_links_are_rare():
    cfg = Config()
    pos = np.array([[[0.5, 0.5], [0.6, 0.5]]])
    cfg.net_hidden_max = 2
    d = distances(cfg, pos, N_IN, 2)
    p = connect_prob(cfg, d, N_IN)
    # n0 riceve da n1 (vicino) molto più facilmente che da un senso (lontano).
    assert p[0, 0, N_IN + 1] > 3 * p[0, 0, 0]
    assert np.isclose(p[0, 0, N_IN + 1], cfg.net_conn_prob * np.exp(-0.1 / cfg.net_conn_range))
    assert p[0, 0, N_IN] == cfg.net_self_prob and p[0, 1, N_IN + 1] == cfg.net_self_prob


def test_random_networks_are_mostly_local():
    cfg = Config()
    b = NeuralController(cfg, 300, seed=1).brains
    t = np.arange(b.dist.shape[1])
    off_self = np.ones(b.dist.shape[1:], dtype=bool)
    off_self[t, N_IN + t] = False
    near = (b.dist < 0.15) & off_self
    far = (b.dist > 0.9) & off_self
    assert b.conn[near].mean() > 5 * b.conn[far].mean()


def test_new_links_from_mutation_also_prefer_near_neurons():
    cfg = Config(mut_conn_add=0.5, mut_conn_del=0.0, mut_pos_prob=0.0)
    b = NeuralController(cfg, 300, seed=2).brains
    g = b.genome.take(np.arange(300))
    g.conn[:] = False
    mutate(g, cfg, np.random.default_rng(0))
    d = distances(cfg, g.pos, N_IN, 2)
    assert g.conn[d < 0.15].mean() > 5 * g.conn[d > 0.9].mean()


def test_position_is_inherited_with_its_neuron_and_drifts_inside_the_brain():
    cfg = Config()
    g = NeuralController(cfg, 2, seed=3).brains.genome.take([0, 1])
    child = crossover(g, np.array([0] * 50), np.array([1] * 50), np.random.default_rng(1))
    for k in range(50):
        for h in range(cfg.net_hidden_max):
            # posizione e collegamenti in ingresso arrivano dallo stesso genitore
            src = 0 if np.array_equal(child.weights[k, h], g.weights[0, h]) else 1
            assert np.array_equal(child.pos[k, h], g.pos[src, h])
    big = Config(mut_pos_prob=1.0, mut_pos_std=0.5)
    mutate(child, big, np.random.default_rng(2))
    assert ((child.pos >= 0.0) & (child.pos <= 1.0)).all()


def test_wiring_costs_energy_in_proportion_to_its_length():
    cfg = Config(n_hazards=0)
    ctrl = NeuralController(cfg, 2, seed=4)
    b = ctrl.brains
    g = b.genome.take([0, 0])
    g.active[:] = True
    g.conn[:] = False
    g.conn[:, 0, 0] = True  # un solo collegamento: senso 0 -> n0
    g.pos[0, 0] = [0.1, 0.0]  # vicino al senso
    g.pos[1, 0] = [0.9, 0.0]  # lontano
    b.set_genome(g)
    assert b.wire_length[1] > 5 * b.wire_length[0]
    world = World(cfg, 2, seed=0)
    for _ in range(60):
        world.step(np.zeros((2, 2)), ctrl.upkeep)
    spent = cfg.energy_init - world.energy
    extra = spent - cfg.metabolism_basal * 60 * cfg.dt
    assert np.allclose(extra, cfg.wire_cost * b.wire_length * 60 * cfg.dt)


def test_saves_without_positions_are_refused(tmp_path):
    e = Evolution(Config(population=10, net_hidden_max=4), 0)
    e.save(tmp_path / "pop.npz")
    data = dict(np.load(tmp_path / "pop.npz"))
    del data["pos"]
    np.savez(tmp_path / "old.npz", **data)
    with pytest.raises(ValueError, match="posizione"):
        Evolution.load(Config(net_hidden_max=4), tmp_path / "old.npz")
