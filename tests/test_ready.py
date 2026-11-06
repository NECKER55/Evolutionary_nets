"""La sezione P con reti già pronte: la memoria degli oggetti, il tracker e le
domande a SlimSAM. Non scarica niente: servono solo i pacchetti del gruppo `percezione`."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from evonets.arenas.mondo3d.world import Mondo3DConfig
from evonets.brains.ragionante.ready import EMBED, GREY, VIEWS, WINDOW, ObjectMemory, Tracker, fixed, hue, overlap, proposals


def unit(seed):
    v = np.random.default_rng(seed).normal(size=EMBED).astype(np.float32)
    return v / np.linalg.norm(v)


def test_a_new_thing_goes_in_memory_and_is_known_the_second_time():
    m = ObjectMemory()
    a, b = unit(1), unit(2)
    assert m.know(a) == (0, True)
    assert m.know(b) == (1, True)
    near = a + 0.1 * unit(3)
    assert m.know(near / np.linalg.norm(near)) == (0, False)
    assert m.kinds == 2


def test_memory_survives_on_disk(tmp_path):
    m = ObjectMemory()
    a = unit(1)
    who, _ = m.know(a)
    m.save(tmp_path / "m.npz")
    back = ObjectMemory.load(tmp_path / "m.npz")
    assert back.know(a) == (who, False)
    assert back.know(unit(2)) == (1, True)  # i numeri nuovi non si sovrappongono


def test_an_object_keeps_a_limited_number_of_views_and_learns_new_sides():
    m = ObjectMemory()
    who, _ = m.know(unit(1))
    other = unit(2)
    m.add(other, who)
    assert m.closest(other) == (who, pytest.approx(1.0))
    for k in range(3, VIEWS + 40):
        m.add(unit(k), who)
    assert (m.owner == who).sum() == VIEWS


def test_an_empty_memory_knows_nothing():
    assert ObjectMemory().closest(unit(1)) == (-1, -1.0)


def test_the_tracker_keeps_an_object_that_moves_a_little_and_never_gives_it_twice():
    t = Tracker()
    t.last = [((10, 10, 30, 30), 7)]
    assert t.follow([(12, 11, 32, 31), (11, 10, 31, 30), (100, 100, 110, 110)]).count(7) == 1
    assert t.follow([(100, 100, 110, 110)]) == [-1]
    assert overlap((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_colour_ignores_light_but_not_hue():
    pixels = np.array([[200.0, 50.0, 30.0]])
    assert hue(pixels) == pytest.approx(hue(pixels * 0.5))
    assert hue(pixels) @ hue(np.array([[30.0, 50.0, 200.0]])) < 0.0
    assert (hue(np.array([[120.0, 120.0, 120.0]])) == 0.0).all()


def test_only_what_stands_on_the_floor_is_worth_a_question():
    cfg = Mondo3DConfig()
    side, spread = 264, np.tan(np.radians(60.0))
    down = (np.arange(side)[:, None] + 0.5 - side / 2.0) / (side / 2.0) * spread
    floor = np.where(down > 0.0, cfg.camera_height / np.maximum(down, 1e-6), 2000.0) * np.ones((side, side))
    assert len(proposals(floor, cfg)) == 0
    thing = floor.copy()
    thing[131:138, 200:215] = 150.0  # alta 12 cm, a 1,5 m
    points = proposals(thing, cfg)
    assert len(points) == 1 and points[0] == pytest.approx((134.0 / side, 207.0 / side), abs=0.02)


def test_the_fixed_crop_keeps_the_true_size_at_any_distance():
    """Lo stesso oggetto a 1 m e a 2 m occupa la stessa parte del ritaglio; uno largo il
    doppio, alla stessa distanza, ne occupa il doppio in larghezza."""
    rgb, spread = np.full((720, 1280, 3), 200, dtype=np.uint8), np.tan(np.radians(60.0))
    px = lambda cm, z: round(cm * 640.0 / spread / z)  # quanti pixel fanno `cm` a `z` cm

    def share(cm, z):
        mask = np.zeros((720, 1280), dtype=bool)
        mask[360:360 + px(10.0, z), 600:600 + px(cm, z)] = True
        crop = fixed(rgb, mask, np.full((720, 1280), float(z)))
        assert crop.shape[0] == crop.shape[1] == px(WINDOW, z)
        return (crop != GREY).any(-1).any(0).mean()

    assert share(10.0, 100.0) == pytest.approx(10.0 / WINDOW, abs=0.02)
    assert share(10.0, 200.0) == pytest.approx(share(10.0, 100.0), abs=0.03)
    assert share(20.0, 100.0) == pytest.approx(2.0 * share(10.0, 100.0), abs=0.03)


def test_points_for_sam_fall_on_the_object_also_in_a_wide_image():
    cfg = Mondo3DConfig()
    high, wide, spread = 720, 1280, np.tan(np.radians(60.0))
    down = (np.arange(high)[:, None] + 0.5 - high / 2.0) / (wide / 2.0) * spread
    floor = np.where(down > 0.0, cfg.camera_height / np.maximum(down, 1e-6), 2000.0) * np.ones((high, wide))
    assert len(proposals(floor, cfg)) == 0
    thing = floor.copy()
    thing[355:380, 900:960] = 150.0
    points = proposals(thing, cfg)
    assert len(points) == 1 and points[0] == pytest.approx((367.0 / high, 930.0 / wide), abs=0.02)
