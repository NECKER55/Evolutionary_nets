import numpy as np
import pytest

from evonets import Config, World
from evonets.evo import Evolution
from evonets.evo.genetics import crossover, mutate
from evonets.net import NeuralController
from evonets.net.geometry import brain_side, connect_prob, distances, motor_anchors, receptor_anchors
from nets import rich

R = 10


def test_receptors_enter_on_the_left_where_their_gene_says_motors_on_the_right():
    r_y = np.array([[0.1, 0.7]])
    xy = receptor_anchors(r_y)
    assert (xy[..., 0] == 0.0).all() and np.array_equal(xy[..., 1], r_y)
    assert (motor_anchors(2)[:, 0] == 1.0).all()


def test_primordial_receptors_form_a_topographic_map():
    cfg = Config(receptor_initial=4, receptor_slots=4)
    g = NeuralController(cfg, 3).brains.genome
    look = np.degrees(np.arctan2(g.r_look[..., 1], g.r_look[..., 0]))
    order = np.argsort(look[0])
    assert (np.diff(g.r_y[0, order]) > 0).all()  # chi guarda più a destra entra più in basso


def test_link_probability_falls_with_distance_and_self_links_are_rare():
    cfg = Config()
    pos = np.array([[[0.5, 0.5], [0.6, 0.5]]])
    d = distances(cfg, pos, np.array([[0.5]]), 2)
    reach = np.array([0.3])
    p = connect_prob(cfg, d, 1, reach)
    # n0 riceve da n1 (vicino) molto più facilmente che dal recettore (lontano).
    assert p[0, 0, 2] > 3 * p[0, 0, 0]
    assert np.isclose(p[0, 0, 2], cfg.net_conn_prob * np.exp(-0.1 / 0.3))
    assert p[0, 0, 1] == cfg.net_self_prob and p[0, 1, 2] == cfg.net_self_prob
    assert connect_prob(cfg, d, 1, np.array([0.6]))[0, 0, 0] > p[0, 0, 0]  # λ più lunga: più collegamenti lontani


def test_new_links_from_mutation_prefer_near_neurons():
    cfg = Config(mut_conn_add=0.5, mut_conn_del=1e-9, mut_pos_prob=1e-9)
    g = rich(300, seed=2).genome
    g.conn[:] = False
    side = brain_side(cfg, g.r_active, g.active, 2)
    mutate(g, cfg, np.random.default_rng(0))
    d = distances(cfg, g.pos, g.r_y, 2, side)  # nel cervello della sua grandezza vera
    near, far = d < np.quantile(d, 0.2), d > np.quantile(d, 0.8)
    assert g.conn[near].mean() > 2 * g.conn[far].mean()


def test_position_is_inherited_with_its_neuron_and_drifts_inside_the_brain():
    g = rich(2, seed=3).genome
    child = crossover(g, np.array([0] * 50), np.array([1] * 50), np.random.default_rng(1))
    for k in range(50):
        for h in range(g.n_hidden):
            src = 0 if np.array_equal(child.weights[k, h], g.weights[0, h]) else 1
            assert np.array_equal(child.pos[k, h], g.pos[src, h])
    big = Config(mut_pos_prob=1.0, mut_pos_std=0.5)
    child.meta[:, :] = np.log(np.array([0.1, 0.3, 0.01, 0.01, 0.01, 0.005, 1.0, 0.5, 0.05, 0.3]))
    mutate(child, big, np.random.default_rng(2))
    assert ((child.pos >= 0.0) & (child.pos <= 1.0)).all() and ((child.r_y >= 0) & (child.r_y <= 1)).all()


def test_wiring_costs_energy_in_proportion_to_its_length():
    cfg = Config(n_hazards=0, receptor_cell_cost=0.0, receptor_sens_cost=0.0, neuron_cost=0.0)
    ctrl = NeuralController(cfg, 2, seed=4)
    b = ctrl.brains
    g = b.genome.take([0, 0])
    g.active[:] = True
    g.conn[:] = False
    g.conn[:, 0, 0] = True  # un solo collegamento: recettore 0 -> n0
    g.r_y[:, 0] = 0.0
    g.pos[0, 0] = [0.1, 0.0]  # vicino al recettore
    g.pos[1, 0] = [0.9, 0.0]  # lontano
    b.set_genome(g)
    assert b.wire_length[1] > 5 * b.wire_length[0]
    world = World(cfg, 2, seed=0)
    for _ in range(60):
        world.step(np.zeros((2, 2)), ctrl.upkeep)
    spent = cfg.energy_init - world.energy
    extra = spent - cfg.metabolism_basal * 60 * cfg.dt
    assert np.allclose(extra, cfg.wire_cost * b.wire_length * 60 * cfg.dt)


def test_saves_of_the_old_format_are_refused(tmp_path):
    e = Evolution(Config(population=10), 0)
    e.save(tmp_path / "pop.npz")
    data = dict(np.load(tmp_path / "pop.npz"))
    data["format"] = np.array(2)
    np.savez(tmp_path / "old.npz", **data)
    with pytest.raises(ValueError, match="sensi fissi"):
        Evolution.load(Config(), tmp_path / "old.npz")


def test_the_brain_grows_with_its_cells():
    cfg = Config()
    small = brain_side(cfg, np.ones((1, 2), bool), np.zeros((1, 0), bool), 2)  # 4 cellule
    big = brain_side(cfg, np.ones((1, 30), bool), np.ones((1, 68), bool), 2)  # 100 cellule
    assert np.isclose(small[0], 0.2) and np.isclose(big[0], 1.0)
    # un organismo primordiale ha sensi e motori vicini: collegarli è probabile
    b = NeuralController(cfg, 500, seed=3).brains
    reach = np.exp(b.genome.meta[:, 9])
    p = connect_prob(cfg, b.dist, b.n_in, reach)[:, b.n_hidden:, :2].mean()
    assert p > 0.25
