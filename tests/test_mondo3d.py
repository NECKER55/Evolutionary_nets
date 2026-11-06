"""Il mondo 3D (evonets/arenas/mondo3d/world.py) e i suoi agenti di controllo
(evonets/brains/mondo3d/): stesse regole dell'arena delle pile, corpi veri."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from evonets.arenas.mondo3d.maps import PLANS, WALL, Layout
from evonets.arenas.mondo3d.run import Run3D, measures
from evonets.arenas.mondo3d.world import Mondo3D, Mondo3DConfig
from evonets.arenas.pile.world import BUMP

ONE, ZERO = np.ones((1, 1)), np.zeros((1, 1))


def world(n=1, **kw):
    """Un recinto senza pile, se non quelle messe dal test."""
    w = Mondo3D(Mondo3DConfig(beings=n, **kw), 1, seed=0)
    w.bon[:] = False
    w.bwait[:] = 1e9
    return w


def put(w, pos, heading=0.0):
    w.pos[0, 0], w.heading[0, 0] = pos, heading
    w.sync()


def pile(w, pos, color=0, big=1, k=0):
    w.bpos[0, k], w.bcolor[0, k], w.bbig[0, k], w.bon[0, k], w.bage[0, k] = pos, color, big, True, 0.0
    w.sync()


def steps(w, k, thrust=ZERO, steer=ZERO, bite=None):
    for _ in range(k):
        w.step(thrust, steer, bite)


def test_the_two_wheels_drive_and_turn_like_the_flat_arena():
    w = world()
    put(w, [100.0, 250.0])
    steps(w, 40, thrust=ONE)
    assert w.pos[0, 0, 0] == pytest.approx(100.0 + 120.0 * 2.0, abs=30.0)  # 2 s a tutta velocità
    assert w.speed[0, 0] == pytest.approx(120.0, abs=5.0)
    assert abs(w.pos[0, 0, 1] - 250.0) < 3.0
    steps(w, 20)
    before = w.heading[0, 0]
    steps(w, 10, steer=ONE)
    steps(w, 10)
    assert w.heading[0, 0] - before == pytest.approx(3.0 * 0.5, abs=0.3)  # 0,5 s a tutto sterzo


def test_walls_stop_the_body():
    w = world()
    put(w, [450.0, 250.0])
    steps(w, 60, thrust=ONE)
    assert w.pos[0, 0, 0] <= 500.0 - w.cfg.radius + 1.0
    assert w.pos[0, 0, 0] > 480.0


def test_the_same_rules_a_good_battery_charges_and_a_bad_one_hurts():
    w = world()
    w.good[:] = 0
    put(w, [100.0, 250.0])
    pile(w, [122.0, 250.0], color=0)
    steps(w, 5, bite=ONE.astype(bool))
    assert not w.bon[0, 0] and w.charging[0, 0] > 0.0
    energy = w.energy[0, 0]
    steps(w, 40)
    assert w.energy[0, 0] > energy + 20.0
    pile(w, [w.pos[0, 0, 0] + 22.0, w.pos[0, 0, 1]], color=1)
    steps(w, 5, bite=ONE.astype(bool))
    assert w.pain[0, 0] > 0.5 and not w.taken[-1][4]


def test_a_body_pushes_a_battery_and_the_rules_see_where_it_went():
    w = world()
    put(w, [100.0, 250.0])
    pile(w, [140.0, 250.0])
    steps(w, 40, thrust=ONE)
    assert w.bpos[0, 0, 0] > 170.0
    assert w.pos[0, 0, 0] < w.bpos[0, 0, 0]


def test_a_missing_battery_cannot_be_seen_or_touched():
    w = world()
    put(w, [100.0, 250.0])
    pile(w, [160.0, 250.0])
    seen = w.view(0, 0)
    w.bon[:] = False
    w.sync()
    gone = w.view(0, 0)
    c = w.cfg.image // 2
    assert (seen["rgb"] != gone["rgb"]).any()
    assert gone["depth"].min() > 0.0 and gone["depth"][c, c] > seen["depth"][c, c] + 100.0
    steps(w, 40, thrust=ONE)
    assert w.pos[0, 0, 0] > 170.0


def test_the_camera_sees_the_colour_and_the_depth_is_the_true_distance():
    w = world()
    put(w, [100.0, 250.0])
    pile(w, [200.0, 250.0], color=0)
    v, n = w.view(0, 0), w.cfg.image
    c = n // 2
    assert v["rgb"].shape == (n, n, 3) and v["depth"].shape == (n, n)
    r, g, b = v["rgb"][c, c - 2:c + 2].mean(0)
    assert r > 1.5 * g and r > 1.5 * b  # rossa, davanti al muso
    # Dalla telecamera (8 cm davanti al centro) alla faccia della pila.
    assert v["depth"][c, c] == pytest.approx(100.0 - 8.0 - 8.0, abs=1.5)


def test_a_pile_is_really_in_its_own_enclosure():
    w = Mondo3D(Mondo3DConfig(beings=1), 3, seed=0)
    w.bon[:], w.bwait[:] = False, 1e9
    w.pos[:], w.heading[:] = [[100.0, 250.0]], 0.0
    w.bon[1, 0], w.bpos[1, 0], w.bcolor[1, 0], w.bbig[1, 0] = True, [200.0, 250.0], 0, 1
    w.sync()
    assert (w.labels(1, 0) > 0).sum() > 100
    assert (w.labels(0, 0) == 0).all() and (w.labels(2, 0) == 0).all()


def test_every_camera_shows_its_own_being_and_enclosures_are_separate():
    w = Mondo3D(Mondo3DConfig(beings=2), 2, seed=0)
    w.pos[:] = [[100.0, 250.0], [400.0, 250.0]]
    w.heading[:] = [0.0, np.pi]
    w.sync()
    assert (w.view(0, 0)["rgb"] != w.view(0, 1)["rgb"]).any()
    low = w.view(1, 0)["depth"][w.cfg.image // 2 + 2:]
    assert not (low > 2000.0 - 1.0).any()  # il muro chiude: niente vista sull'altro recinto
    steps_ = np.ones((2, 2))
    w.step(steps_, 0.0 * steps_)
    assert w.pos.shape == (2, 2, 2) and (w.pos > 0.0).all() and (w.pos < 500.0).all()


def test_the_starved_are_reborn_where_the_rules_say():
    w = world()
    put(w, [100.0, 250.0])
    w.energy[0, 0] = 0.001
    w.step(ZERO, ZERO)
    assert w.energy[0, 0] == w.cfg.energy_start
    assert np.allclose(w.data.qpos[w.qadr[0, 0]:w.qadr[0, 0] + 2], (w.pos[0, 0] + w.origin[0]) * 0.01)


def test_the_same_seed_gives_the_same_life():
    lives = []
    for seed in (3, 3, 4):
        r = Run3D(Mondo3DConfig(beings=2), "perfetto", 1, seed)
        r.advance(300)
        lives.append((r.w.pos.copy(), r.w.energy.copy(), r.w.taken[:]))
    assert (lives[0][0] == lives[1][0]).all() and (lives[0][1] == lives[1][1]).all() and lives[0][2] == lives[1][2]
    assert (lives[0][0] != lives[2][0]).any()


def test_the_perfect_agent_eats_only_good_ones_through_an_inversion_and_beats_the_base():
    cfg = lambda: Mondo3DConfig(inversion_every=30.0, zone_every=45.0, image=64)
    perfect, base = Run3D(cfg(), "perfetto", 1, 0), Run3D(cfg(), "base", 1, 0)
    perfect.advance(1600)
    base.advance(1600)
    p, b = measures(perfect), measures(base)
    assert p["buone"] >= 8 and p["cattive"] == 0 and p["morti di fame"] == 0
    assert p["pile/min"] > b["pile/min"]


def test_the_base_agent_leaves_a_wall_by_looking_at_its_depth():
    r = Run3D(Mondo3DConfig(image=64), "base", 1, 0)
    w = r.w
    w.bon[:] = False
    w.bwait[:] = 1e9
    w.pos[0, 0], w.heading[0, 0] = [470.0, 250.0], 0.0
    w.sync()
    r.advance(100)
    assert w.pos[0, 0, 0] < 480.0 or abs(w.heading[0, 0]) > 1.0


def test_a_crash_into_a_wall_does_not_tip_the_body_over():
    w = world()
    put(w, [440.0, 250.0])
    steps(w, 100, thrust=ONE)
    steps(w, 40, thrust=-ONE)
    assert w.pos[0, 0, 0] < 470.0  # è ripartito: le ruote toccano ancora terra


def test_the_perfect_agent_never_stands_still_with_a_battery_to_reach():
    r = Run3D(Mondo3DConfig(beings=3), "perfetto", 1, 1)
    w, still, worst = r.w, np.zeros(3), 0.0
    for _ in range(2400):
        yaw = w.heading.copy()
        r.step()
        idle = (r.brain.target[0] >= 0) & (w.charging[0] <= 0.0) & (np.abs(w.speed[0]) < 2.0) \
            & (np.abs(w.heading - yaw)[0] < 0.01)
        still = np.where(idle, still + 1.0, 0.0)
        worst = max(worst, still.max())
    assert worst * w.cfg.dt < 5.0 and w.deaths["fame"] == 0
    assert w.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0  # la fisica non è mai esplosa


def test_the_marks_on_the_floor_are_for_us_and_never_reach_the_camera():
    w = world()
    w.good[:] = 0
    put(w, [100.0, 250.0])
    pile(w, [200.0, 250.0], color=0)
    before = w.view(0, 0)
    w.overview(0, 200, 200, follow=0)
    after = w.view(0, 0)
    assert (before["rgb"] == after["rgb"]).all() and (before["depth"] == after["depth"]).all()


def test_the_depth_colours_tell_near_from_far_where_most_of_the_image_is():
    from evonets.viz.mondo3d.view import SKY_C, depth_image, depth_scale

    assert depth_scale(np.array([10.0]))[0] < depth_scale(np.array([25.0]))[0] < depth_scale(np.array([50.0]))[0]
    near, far, sky = depth_image(np.array([[10.0, 400.0, 2000.0]]))[0]
    assert abs(near.astype(int) - far.astype(int)).sum() > 100 and tuple(sky) == SKY_C


def test_a_new_battery_is_never_born_on_top_of_a_being_and_fades_in():
    r = Run3D(Mondo3DConfig(beings=3, image=64), "perfetto", 1, 0)
    w, was = r.w, r.w.bon.copy()
    seen = 0
    for _ in range(1500):
        r.step()
        for k in np.flatnonzero(w.bon[0] & ~was[0]):
            seen += 1
            assert np.hypot(*(w.pos[0] - w.bpos[0, k]).T).min() > 50.0
            assert w.model.geom_rgba[w.pgeom[0, k], 3] < 0.2  # appena nata: quasi trasparente
        was = w.bon.copy()
    assert seen > 10


def test_a_battery_fades_out_before_it_expires():
    w = world()
    pile(w, [300.0, 300.0])
    w.bage[0, 0] = w.cfg.battery_life - 0.3
    w.born[0, 0] = 0.0
    w.step(ZERO, ZERO)
    assert w.bon[0, 0] and 0.2 < w.model.geom_rgba[w.pgeom[0, 0], 3] < 0.8


# --- Le mappe a stanze ---------------------------------------------------------------


def rooms(name="stanze", **kw):
    return world(map=name, **kw)


def test_the_open_map_has_no_inner_walls_and_the_size_it_always_had():
    w = world()
    assert w.cfg.size == 500.0 and len(w.layouts[0].rects) == 0 and not w.layouts[0].doors


@pytest.mark.parametrize("name, side", (("stanze", 1200.0), ("labirinto", 1300.0)))
def test_every_room_can_be_reached_and_the_map_fills_the_enclosure(name, side):
    w = rooms(name)
    assert w.cfg.size == side
    assert (w.layouts[0]._next >= 0).all()


def test_the_same_seed_gives_the_same_house_and_another_seed_another():
    plan = PLANS["stanze"]
    first, again, other = Layout(plan, 0, 4), Layout(plan, 0, 4), Layout(plan, 1, 4)
    assert (first.rects == again.rects).all() and (first.tints == again.tints).all()
    assert first.rects.shape != other.rects.shape or (first.rects != other.rects).any()


def test_a_being_is_never_born_inside_a_wall():
    w = rooms()
    for _ in range(50):
        w.reborn(np.ones((1, 1), dtype=bool))
        assert not len(w.layouts[0].near_wall(w.pos[0, 0], w.cfg.radius + WALL))


def test_inner_walls_stop_the_body_and_are_felt_as_a_bump():
    w = rooms()
    put(w, (w.layouts[0].cell - 60.0, 50.0))
    steps(w, 80, thrust=ONE)
    assert w.pos[0, 0, 0] < w.layouts[0].cell
    assert w.senses()[0, 0, BUMP].any()


def test_a_wall_hides_what_is_behind_it_from_the_truth_too():
    w = rooms()
    l = w.layouts[0]
    wall = next(r for r in l.rects if r[3] - r[1] > 100.0)  # un muro lungo in y
    x, y = wall[0] - 60.0, (wall[1] + wall[3]) / 2.0
    put(w, (x, y))
    pile(w, (x + 130.0, y))
    assert not w.objects()["vis"][0, 0, 0]
    pile(w, (x + 40.0, y))
    assert w.objects()["vis"][0, 0, 0]


@pytest.mark.parametrize("name", ("stanze", "labirinto"))
def test_the_way_leads_from_any_room_to_any_other_through_the_doors(name):
    l = rooms(name).layouts[0]
    centre = lambda r: (np.array((r % l.cells, r // l.cells)) + 0.5) * l.cell
    for start in range(l.cells ** 2):
        for goal in range(l.cells ** 2):
            at, passed = centre(start), 0
            while l.room_of(at) != goal:
                stop, done = l.way(at, centre(goal))
                assert not l.hides(at, stop) and not done
                at, passed = stop, passed + 1
                assert passed < 4 * l.cells ** 2


def test_the_perfect_agent_crosses_the_house_when_the_zone_changes():
    r = Run3D(Mondo3DConfig(beings=1, map="stanze", image=64, zone_every=45.0), "perfetto", 1, 0)
    zones = set()
    for _ in range(4):
        r.advance(int(50.0 / r.w.cfg.dt))
        zones.add(int(r.w.zone[0]))
    good = sum(1 for x in r.w.taken if x[4])
    r.close()
    assert len(zones) >= 2 and good >= 12 and r.w.deaths["fame"] == 0


# --- La pagina del cervello ----------------------------------------------------------


@pytest.mark.parametrize("agent", ("ragionante", "vedente finto"))
def test_the_brain_page_draws_while_the_run_goes_on(agent):
    import pygame

    from evonets.brains.ragionante.watch import Watch
    from evonets.viz.mondo3d.brain_page import BrainPage

    pygame.font.init()

    class App:
        font = pygame.font.SysFont("monospace", 15)
        small = pygame.font.SysFont("monospace", 12)

    r = Run3D(Mondo3DConfig(beings=1, image=64), agent, 1, 0)
    r.watch = Watch(r.w, r.brain)
    r.advance(40)
    s, page = pygame.Surface((1440, 860)), BrainPage()
    page.draw(s, App, r, 0, 0, (700, 200), False)
    assert len(page.spots) >= 9
    r.close()


def test_the_watcher_reads_a_bite_and_only_then_its_memory_moves():
    from evonets.brains.ragionante.watch import Watch

    r = Run3D(Mondo3DConfig(beings=1, image=64), "vedente finto", 1, 0)
    r.brain.motor.left[:] = 0.0
    r.watch = watch = Watch(r.w, r.brain)
    watch.before()
    assert not watch.h.any()
    pile_seen = watch.seen
    r.w.taken.append((0.0, 0, 0, 0, True))
    pile_seen.on[0, 0, 0], pile_seen.kind[0, 0, 0], pile_seen.dist[0, 0, 0] = True, 0, 20.0
    watch.taken = 0
    watch.after()
    r.close()
    assert watch.h.any() and watch.reads[0, 0] == 1 and watch.rows[0, 0][-1][1] == "piacere"
