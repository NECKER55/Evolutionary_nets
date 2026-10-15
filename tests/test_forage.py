"""Il mondo del cibo (docs/SPEC_MONDO.md) nel motore NumPy."""

import numpy as np

from evonets import Config, World
from evonets.sim import signals
from evonets.sim.tasks import Foraging


def forage(n=4, seed=0, **kw):
    cfg = Config(task="cibo", n_hazards=0, **kw)
    w = World(cfg, n, seed=seed)
    w.vel[:] = 0.0
    return cfg, w


def test_the_world_starts_the_same_for_everyone_and_far_from_food():
    cfg, w = forage(n=50)
    t = w.task
    assert isinstance(t, Foraging) and t.n_food == cfg.food_count_easy and t.need == 1
    assert (t.food == t.food[0]).all()  # stessa copia per tutti all'inizio
    d = np.linalg.norm(w.pos[:, None, :] - t.food[0][None], axis=2)
    assert (d >= cfg.food_min_dist).all()


def test_dials_set_how_hard_the_world_is():
    _, w = forage(forage_abundance=1.0, forage_need=1.0, forage_prey=1.0, forage_predators=1.0)
    t = w.task
    assert t.n_food == 3 and np.isclose(t.food_r, 2.0) and t.need == 5
    assert t.prey_speed == 12.0 and t.n_pred == 3 and t.pred_speed == 15.0


def test_eating_gives_energy_counts_and_food_reappears_only_in_your_copy():
    cfg, w = forage()
    t = w.task
    w.pos[0] = t.food[0, 0]  # l'agente 0 è sul primo cibo
    w.energy[0] = 50.0
    before = t.food.copy()
    w.step(np.zeros((w.n, 2)))
    assert t.eaten[0] == 1 and t.eaten[1:].sum() == 0
    assert np.isclose(w.energy[0], 50.0 + cfg.food_energy - cfg.metabolism_basal * cfg.dt)
    assert not np.allclose(t.food[0, 0], before[0, 0])  # ricomparso altrove
    assert np.allclose(t.food[1:], before[1:])  # le altre copie non cambiano
    assert w.done[0]  # con un cibo solo richiesto ha completato


def test_energy_does_not_exceed_the_full_stomach():
    cfg, w = forage()
    w.pos[0] = w.task.food[0, 0]
    w.step(np.zeros((w.n, 2)))
    assert w.energy[0] <= cfg.energy_init


def test_more_food_is_needed_when_the_dial_says_so():
    cfg, w = forage(forage_need=1.0)
    t = w.task
    for k in range(4):
        w.pos[0] = t.food[0, k % t.n_food]
        w.step(np.zeros((w.n, 2)))
    assert t.eaten[0] == 4 and not w.done[0]
    w.pos[0] = t.food[0, 0]
    w.step(np.zeros((w.n, 2)))
    assert w.done[0]


def test_prey_flee_from_a_near_agent_and_stay_when_it_is_far():
    cfg, w = forage(forage_prey=1.0)
    t = w.task
    w.pos[0] = t.food[0, 0] + np.array([5.0, 0.0])  # vicino al primo cibo
    near_before, far_before = t.food[0, 0].copy(), t.food[0, 1].copy()
    w.step(np.zeros((w.n, 2)))
    moved = t.food[0, 0] - near_before
    assert moved[0] < 0 and np.isclose(np.linalg.norm(moved), cfg.prey_speed_max * cfg.dt, rtol=0.05)
    if np.linalg.norm(t.food[0, 1] - w.pos[0]) > cfg.prey_flee_range:
        assert np.allclose(t.food[0, 1], far_before)


def test_predators_chase_and_bite():
    cfg, w = forage(forage_predators=1.0)
    t = w.task
    w.pos[0] = t.pred[0, 0] + np.array([10.0, 0.0])
    w.pos[1] = t.pred[1, 0] + np.array([0.5, 0.0])  # già a contatto
    e1 = w.energy[1]
    before = t.pred[0, 0].copy()
    w.step(np.zeros((w.n, 2)))
    assert t.pred[0, 0][0] > before[0]  # si avvicina
    assert w.energy[1] < e1 - cfg.predator_damage * cfg.dt * 0.9


def test_a_dead_agent_cannot_eat_itself_back_to_life():
    cfg, w = forage()
    w.energy[0] = 1e-6
    w.pos[0] = w.task.food[0, 0]
    w.step(np.zeros((w.n, 2)))
    assert not w.alive[0] and w.task.eaten[0] == 0


def test_receptors_see_food_and_predators_and_hear_predators():
    cfg, w = forage(forage_predators=1.0)
    t = w.task
    t.food[0, 0] = w.pos[0] + np.array([8.0, 0.0])
    w.heading[0] = 0.0
    seen = signals.light(cfg, w, w.pos[:1] + [cfg.agent_radius, 0.0], np.zeros((1, 1)), np.array([0]))
    assert seen["cibo"][0] > 0.4 and seen["parete"][0] == 0.0
    t.pred[0, 0] = w.pos[0] + np.array([-8.0, 0.0])
    seen = signals.light(cfg, w, w.pos[:1] - [cfg.agent_radius, 0.0], np.full((1, 1), np.pi), np.array([0]))
    assert seen["predatore"][0] > 0.4
    vib = signals.vibrations(cfg, w, w.pos[:1], np.array([0]))
    assert len(vib) == cfg.predator_count_max and vib[0][1][0] > 0.0


def test_an_agent_killed_by_a_bite_cannot_eat_in_the_same_step():
    cfg, w = forage(forage_predators=1.0)
    t = w.task
    w.pos[0] = t.pred[0, 0] + np.array([0.5, 0.0])
    t.food[0, 0] = w.pos[0]
    w.energy[0] = 1.0
    w.step(np.zeros((w.n, 2)))
    assert not w.alive[0] and t.eaten[0] == 0
