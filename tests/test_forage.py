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
    cfg, w = forage(n=50, forage_own_arena=False)
    t = w.task
    assert isinstance(t, Foraging) and t.n_food == cfg.food_count_easy and t.need == 1
    assert (t.food == t.food[0]).all()  # stessa copia per tutti all'inizio
    d = np.linalg.norm(w.pos[:, None, :] - t.food[0][None], axis=2)
    assert (d >= cfg.food_min_dist).all()


def test_with_an_own_arena_everyone_starts_somewhere_else_and_far_from_its_food():
    cfg, w = forage(n=50, forage_predators=1.0)
    t = w.task
    assert cfg.forage_own_arena and not (t.food == t.food[0]).all() and not (t.pred == t.pred[0]).all()
    assert len(set(t.ptr)) > 1 and (t.ptr % 2 == 0).all()  # ognuno scorre la coda delle ricomparse da un punto suo
    on = t.food_on[:, :, None]
    d = np.linalg.norm(w.pos[:, None, :] - t.food, axis=2)
    assert (np.where(t.food_on, d, np.inf).min(axis=1) >= cfg.food_min_dist).all()
    d = np.linalg.norm(w.pos[:, None, :] - t.pred, axis=2)
    assert (np.where(t.pred_on, d, np.inf).min(axis=1) >= cfg.predator_min_dist).all()
    assert on.shape == (50, cfg.food_count_easy, 1)


def test_dials_set_how_hard_the_world_is():
    cfg, w = forage(forage_abundance=1.0, forage_need=1.0, forage_prey=1.0, forage_predators=1.0)
    t = w.task
    assert t.n_food == cfg.food_count_hard and np.isclose(t.food_r, 2.0) and t.need == cfg.food_need_hard
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
    assert w.energy[0] <= cfg.energy_max


def test_more_food_is_needed_when_the_dial_says_so():
    cfg, w = forage(forage_need=1.0)
    t = w.task
    for k in range(t.need - 1):
        w.pos[0] = t.food[0, k % t.n_food]
        w.step(np.zeros((w.n, 2)))
    assert t.eaten[0] == t.need - 1 and not w.done[0]
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


def test_food_and_predators_smell_with_their_own_molecule():
    cfg, w = forage(forage_predators=1.0)
    t = w.task
    points = t.food[:2, 0] + np.array([[cfg.odor_range + t.food_r, 0.0]] * 2)  # una portata dal bordo del primo cibo
    odors = dict(t.odors(w, points, np.arange(2)))
    assert set(odors) == {cfg.food_molecule, cfg.predator_molecule}
    alone = np.exp(-1.0)  # il primo cibo da solo; gli altri aggiungono la loro nube
    assert (odors[cfg.food_molecule] >= alone - 1e-9).all()
    cfg.food_smell = cfg.predator_smell = 0.0
    assert t.odors(w, points, np.arange(2)) == []


def test_a_new_food_cloud_forms_and_an_eaten_one_fades():
    cfg, w = forage(forage_scent=1.0)
    t = w.task
    tau = cfg.scent_time_max
    assert t.scent_tau == tau
    old = t.food[0, 0].copy()
    w.pos[0] = old  # l'agente 0 è sopra il primo cibo: lo mangia
    w.step(np.zeros((w.n, 2)))
    assert t.eaten[0] == 1 and t.age[0, 0] == 0.0 and t.ghost_t[0, 0] == 0.0
    assert np.allclose(t.ghost[0, 0], old)
    for _ in range(int(tau / cfg.dt)):  # dopo τ secondi
        w.step(np.zeros((w.n, 2)))
    on, trace = t.scent()
    assert np.isclose(on[0, 0], 1 - np.exp(-1), atol=0.01) and np.isclose(trace[0, 0], np.exp(-1), atol=0.01)
    assert on[1, 0] == 1.0 and trace[1, 0] < 1e-6  # chi non ha mangiato: nube formata da sempre, nessuna traccia


def test_without_the_scent_dial_the_cloud_is_always_formed():
    cfg, w = forage()
    w.task.age[:] = 0.0
    on, trace = w.task.scent()
    assert (on == 1.0).all() and (trace == 0.0).all()


def test_a_mature_agent_keeps_living_and_eating():
    cfg, w = forage(forage_need=0.0)  # ne basta uno
    t = w.task
    w.pos[0] = t.food[0, 0]
    w.step(np.array([[1.0, 0.0]] * w.n))
    assert w.done[0] and w.keep_living and w.racing[0]
    before = w.pos[0].copy()
    w.step(np.array([[1.0, 0.0]] * w.n))
    assert not np.allclose(w.pos[0], before)  # continua a muoversi
    w.pos[0] = t.food[0, 1]
    w.step(np.zeros((w.n, 2)))
    assert t.eaten[0] == 2  # e a mangiare


def test_without_keep_living_a_mature_agent_stops_as_before():
    cfg, w = forage(forage_need=0.0, mature_keep_living=False)
    w.pos[0] = w.task.food[0, 0]
    w.step(np.zeros((w.n, 2)))
    assert w.done[0] and not w.racing[0]


# --- Le rocce (quarto stadio) --------------------------------------------------

def rocky(n=4, **kw):
    cfg, w = forage(n, forage_obstacles=1.0, **kw)
    return cfg, w, w.task


def test_rocks_appear_with_the_fourth_dial_and_nothing_starts_inside_them():
    cfg, w, t = rocky(n=20)
    assert t.n_rock == cfg.rock_count_max and (t.rock_r > 0).all()
    edge = lambda p, body: (np.linalg.norm(p[:, :, None, :] - t.rock[:, None], axis=3) - t.rock_r[:, None]
                            - body).min(axis=2)
    assert (edge(t.food, t.food_r) >= -1e-6).all() and (edge(t.pred, cfg.predator_radius) >= -1e-6).all()
    assert (edge(w.pos[:, None, :], cfg.agent_radius) > 0).all()
    _, plain = forage(n=4)
    assert plain.task.n_rock == 0 and not (plain.task.rock_r > 0).any()


def test_a_rock_stops_the_agent_and_presses_on_the_side_that_touches_it():
    cfg, w, t = rocky(n=1)
    w.pos[0] = [100.0, 100.0]
    t.rock[0, 0], t.rock_r[0, 0] = [100.0 + 10.0 + cfg.agent_radius + 0.2, 100.0], 10.0
    w.heading[0] = 0.0
    t.rock_r[0, 1:] = 0.0  # una roccia sola
    for _ in range(20):
        w.step(np.array([[1.0, 0.0]]))
    gap = np.linalg.norm(w.pos[0] - t.rock[0, 0]) - t.rock_r[0, 0] - cfg.agent_radius
    assert abs(gap) < 1e-6 and w.bump[0]
    toward = (t.rock[0, 0] - w.pos[0]) / np.linalg.norm(t.rock[0, 0] - w.pos[0])
    normals = np.array([toward, -toward])  # verso la roccia e dalla parte opposta
    press = signals.pressure(cfg, w, np.array([0, 0]), normals)
    assert press[0] > 0.99 and press[1] == 0.0


def test_a_rock_hides_what_is_behind_it_and_is_seen_grey():
    cfg, w, t = rocky(n=1)
    w.pos[0] = [100.0, 100.0]
    t.rock_r[0] = 0.0
    t.food[0, 0] = [130.0, 100.0]
    look = lambda: signals.light(cfg, w, w.pos[:1], np.zeros((1, 1)), np.array([0]))
    assert look()["cibo"][0] > 0
    t.rock[0, 0], t.rock_r[0, 0] = [115.0, 100.0], 5.0
    seen = look()
    assert seen["cibo"][0] == 0.0 and seen["roccia"][0] > 0.4


def test_food_never_comes_back_inside_a_rock():
    cfg, w, t = rocky(n=6, forage_abundance=0.0, mature_keep_living=True)
    inside = 0
    for _ in range(300):
        w.pos[:] = t.food[:, 0]  # ognuno sopra il suo primo cibo: lo mangia, e il cibo ricompare
        w.energy[:] = 100.0
        w.step(np.zeros((w.n, 2)))
        d = np.linalg.norm(t.food[:, :, None, :] - t.rock[:, None], axis=3) - t.rock_r[:, None]
        inside += int((d < t.food_r - 1e-6).sum())
    assert t.eaten.sum() > 200 and inside == 0


def test_the_ladder_has_a_fourth_stage_and_old_runs_open_without_rocks():
    from evonets.evo import ladder

    assert ladder.TOP == 4.0 and ladder.STAGE_NAMES[-1] == "rocce"
    assert ladder.at(3.5)["forage_obstacles"] == 0.5 and ladder.at(3.5)["forage_predators"] == 1.0
    assert ladder.at(2.5)["forage_obstacles"] == 0.0
