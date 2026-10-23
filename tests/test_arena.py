"""L'arena continua (evonets/sim/arena.py, evonets/evo/ecology.py,
docs/SPEC_ARENA.md): mondo senza bordi, sensi attraverso il bordo, pasti,
vecchiaia, nascite, colonizzatori, salvataggi."""

import numpy as np

from evonets import Config
from evonets.evo.ecology import Ecology
from evonets.net import Brains
from evonets.sim.arena import ArenaWorld


def small(**kw) -> Config:
    """Un'arena piccola per i test: niente esame, pochi posti."""
    base = dict(arena_size_w=400.0, arena_size_h=320.0, arena_cap=60, arena_food_max=80, arena_food_rate=5.0,
                arena_min_alive=0, exam_every=0, engine="numpy")
    return Config(**{**base, **kw})


def empty_world(cfg: Config, n: int = 4) -> ArenaWorld:
    w = ArenaWorld(cfg, n, n, seed=0)
    w.task.food_on[:] = False
    w.grids_dirty = True
    return w


def test_the_world_has_no_edges():
    w = empty_world(small())
    w.pos[0] = [399.5, 10.0]
    w.heading[0] = 0.0
    w.vel[0] = [30.0, 0.0]
    for _ in range(10):
        w.step(np.zeros((w.n, 2)))
    assert 0.0 <= w.pos[0, 0] < 10.0  # uscito a destra, rientrato a sinistra


def test_light_sees_food_and_others_also_across_the_edge_but_not_itself():
    cfg = small()
    w = empty_world(cfg)
    w.alive[:] = [True, True, False, False]
    w.pos[0] = [398.0, 100.0]  # guarda a destra, verso il bordo
    w.pos[1] = [6.0, 100.0]  # dall'altra parte del bordo, a 8 u
    w.grids_dirty = True
    seen = w.light_of(np.array([[399.0, 100.0]]), np.zeros((1, 3)), np.array([0]))
    assert seen["essere"][0] > 0.4 and seen["cibo"][0] == 0.0 and seen["parete"][0] == 0.0
    w.task.food[0], w.task.food_on[0] = [390.0, 100.0], True  # cibo alle spalle: non si vede guardando avanti
    w.grids_dirty = True
    back = w.light_of(np.array([[399.0, 100.0]]), np.full((1, 3), np.pi), np.array([0]))
    assert back["cibo"][0] > 0.4
    alone = w.light_of(np.array([[399.0, 100.0]]), np.zeros((1, 3)), np.array([1]))
    assert alone["essere"][0] > 0.4  # il recettore di 1 vede 0, non se stesso


def test_the_nearest_eats_and_food_regrows_from_the_patches():
    cfg = small(arena_food_rate=0.0)
    w = empty_world(cfg)
    w.alive[:] = True
    w.task.food[0], w.task.food_on[0] = [100.0, 100.0], True
    w.pos[0], w.pos[1], w.pos[2], w.pos[3] = [101.0, 100.0], [103.5, 100.0], [300.0, 300.0], [200.0, 50.0]
    w.energy[:] = 100.0
    w.grids_dirty = True
    w.step(np.zeros((w.n, 2)))
    assert not w.task.food_on[0] and w.task.eaten[0] == 1 and w.task.eaten[1] == 0
    assert w.energy[0] > w.energy[1] + cfg.food_energy - 1.0 and w.taste[0] > 0.9
    w.task.grow(w, 10)
    assert w.task.food_on.sum() == 10


def test_old_age_follows_gompertz_and_hunger_kills():
    cfg = small()
    h = lambda t: 1.0 - np.exp(-cfg.arena_aging_h0 * cfg.arena_aging_tau * (np.exp(t / cfg.arena_aging_tau) - 1.0))
    assert abs(h(600.0) - 0.5) < 0.01 and h(300.0) < 0.1 and h(900.0) > 0.99
    w = empty_world(cfg)
    w.energy[0] = 0.01
    w.step(np.zeros((w.n, 2)))
    assert not w.alive[0] and w.deaths["fame"] == 1


def test_brains_replace_keeps_what_the_others_learned():
    cfg = Config()
    b = Brains(cfg, 6, 2, 0)
    for _ in range(30):
        b.step(np.random.default_rng(1).random((6, b.n_in)), np.ones((6, 2)))
    learned, state = b.w_eff.copy(), b.state.copy()
    child = Brains(cfg, 2, 2, 9).genome
    b.replace(np.array([1, 4]), child)
    others = [0, 2, 3, 5]
    assert np.array_equal(b.w_eff[others], learned[others]) and np.array_equal(b.state[others], state[others])
    assert np.array_equal(b.genome.weights[[1, 4]], child.weights) and not b.state[[1, 4]].any()


def test_a_well_fed_adult_has_a_child_beside_it_and_pays_for_it():
    cfg = small()
    e = Ecology(cfg, seed=1, initial=3)
    w = e.world
    w.age[:3] = 100.0
    w.energy[0] = 400.0
    w.energy[1:3] = 100.0
    w.pos[0] = [200.0, 160.0]
    w.pos[1:3] = [[50.0, 50.0], [350.0, 250.0]]
    e._births()
    child = np.flatnonzero(w.alive & (w.age == 0.0))
    assert len(child) == 1 and child[0] >= 3
    assert np.hypot(*(w.pos[child[0]] - w.pos[0])) <= cfg.arena_birth_distance + 1e-6
    assert w.energy[child[0]] == cfg.arena_child_energy and w.energy[0] == 400.0 - cfg.arena_birth_cost
    e._births()  # appena partorito: deve aspettare
    assert (w.alive & (w.age == 0.0)).sum() == 1


def test_a_compatible_neighbour_becomes_the_partner():
    cfg = small()
    e = Ecology(cfg, seed=2, initial=3)
    w = e.world
    w.age[:3] = 100.0
    w.pos[:3] = [[200.0, 160.0], [205.0, 160.0], [300.0, 300.0]]
    assert e.compatible(0, 1)  # all'inizio tutti hanno i geni degli stessi posti
    assert e._partner(0, np.array([0, 1, 2])) == 1  # il vicino, non quello lontano
    e.ctrl.brains.genome.r_id[1] += 10**9  # geni di un'altra discendenza
    assert not e.compatible(0, 1) and e._partner(0, np.array([0, 1, 2])) == 0


def test_colonists_arrive_only_below_the_minimum():
    cfg = small(arena_min_alive=10)
    e = Ecology(cfg, seed=3, initial=4)
    e._colonists()
    assert e.world.alive.sum() == 10 and e.totals["colonizzatori"] == 6
    e._colonists()
    assert e.totals["colonizzatori"] == 6


def test_an_arena_run_saves_and_resumes_exactly(tmp_path):
    cfg = small(arena_stats_every=5.0)
    e = Ecology(cfg, seed=4, initial=30)
    e.run(6.0)
    assert e.history and e.history[0]["alive"] > 0
    e.save(tmp_path / "arena.npz")
    back = Ecology.load(small(arena_stats_every=5.0), tmp_path / "arena.npz")
    e.run(2.0)
    back.run(2.0)
    assert np.allclose(e.world.pos, back.world.pos) and np.array_equal(e.world.alive, back.world.alive)
    assert back.log[-1]["event"] == "ripresa"
