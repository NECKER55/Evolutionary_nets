"""La sezione P, la percezione: la geometria comune (vision.py), il detector finto e il
comandante che ha solo gli occhi. Le reti pronte stanno in test_ready.py."""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import evonets.brains.ragionante as ragionante
from evonets.arenas.mondo3d.run import Run3D, measures
from evonets.arenas.mondo3d.world import Mondo3D, Mondo3DConfig
from evonets.brains.ragionante.vision import EMBED, FEATURES, KINDS, MAX_OBJECTS, FakeDetector, Percept, box_of, locate

folder = Path(ragionante.__file__).parent


def world(pile=(180.0, 250.0), colour=1, big=1):
    """Un essere a guardare verso +x e, se serve, una pila sola davanti a lui."""
    w = Mondo3D(Mondo3DConfig(beings=1), 1, 3)
    w.bon[:], w.bwait[:] = False, 1e9
    w.pos[0, 0], w.heading[0, 0] = [100.0, 250.0], 0.0
    if pile is not None:
        w.bon[0, 0], w.bpos[0, 0], w.bcolor[0, 0], w.bbig[0, 0] = True, pile, colour, big
    w.sync()
    return w


def test_the_labels_show_what_is_there_and_nothing_else():
    w = world()
    assert (w.labels(0, 0) == KINDS.index("pila gialla") + 1).sum() > 100
    w.bcolor[0, 0] = 0
    assert (w.labels(0, 0) == KINDS.index("pila rossa") + 1).sum() > 100
    empty = world(None)
    assert (empty.labels(0, 0) == 0).all()


def test_locate_puts_an_object_where_the_camera_sees_it():
    cfg = Mondo3DConfig()
    mask = np.zeros((132, 132), dtype=bool)
    mask[60:80, 100:120] = True  # a destra del centro
    ang, dist, wide = locate(mask, np.full((132, 132), 100.0), cfg)
    expected = np.arctan2((110.0 - 66.0) / 66.0 * 100.0 * np.tan(np.radians(60.0)), 100.0 + cfg.camera_ahead)
    assert ang == pytest.approx(-expected, abs=1e-6) and dist == pytest.approx(184.4, abs=0.3)
    assert wide == pytest.approx(20.0 / 66.0 * 100.0 * np.tan(np.radians(60.0)))
    assert box_of(mask) == (100, 60, 120, 80)


def test_the_fake_detector_gives_the_interface_of_p():
    w = world()
    fake = FakeDetector(w).see()
    assert fake.on.shape == (1, 1, MAX_OBJECTS) and fake.vector.shape[:3] == (1, 1, MAX_OBJECTS)
    assert fake.embedding(350.0).shape == (1, 1, MAX_OBJECTS, EMBED)
    assert fake.features(350.0).shape == (1, 1, MAX_OBJECTS, FEATURES)
    k = np.flatnonzero(fake.on[0, 0])[0]
    assert fake.kind[0, 0, k] == 1


def test_seeing_knows_colours_but_not_what_is_good():
    source = (folder / "vision.py").read_text()
    assert "good" not in source.replace("# ", "") and "battery_good" not in source


def test_the_seeing_commander_reads_no_positions_from_the_simulator():
    source = (folder / "seeing.py").read_text()
    for truth in ("bpos", "battery_good", "bcolor", "objects()"):
        assert truth not in source


def test_the_seeing_commander_reaches_a_pile_it_sees():
    r = Run3D(Mondo3DConfig(beings=1), "vedente finto", 1, 2)
    r.advance(int(180.0 / r.cfg.dt))
    assert len(r.w.taken) >= 4
    assert measures(r)["tarature"] <= 3  # il corpo obbedisce: niente ritarature a vuoto


def test_with_mistakes_it_also_bites_what_is_bad():
    w = Mondo3D(Mondo3DConfig(beings=1), 3, 2)
    from evonets.brains.ragionante.brain import Reasoning
    from evonets.brains.ragionante.seeing import Seeing

    brain = Reasoning(w, 2, Seeing(w, FakeDetector(w), 2, mistakes=0.6))
    for _ in range(int(150.0 / w.cfg.dt)):
        w.step(*brain.act())
    assert sum(not ok for *_, ok in w.taken) >= 2


def test_changing_who_commands_does_not_touch_the_motor():
    assert "vision" not in (folder / "motor.py").read_text()
    assert "wiring" not in (folder / "seeing.py").read_text()


@pytest.mark.parametrize("frame", [(264, 0), (640, 480), (1280, 720)])
def test_angle_distance_and_width_come_out_right_at_every_resolution(frame):
    """Una pila a 30° a sinistra e a 150 cm: da maschera e profondità si ritrovano
    dov'è e quanto è larga, con l'immagine quadrata e con quelle larghe."""
    w = Mondo3D(Mondo3DConfig(beings=1, image=frame[0], image_height=frame[1]), 1, 3)
    w.bon[:], w.bwait[:] = False, 1e9
    w.pos[0, 0], w.heading[0, 0] = [100.0, 250.0], 0.0
    turn = np.radians(30.0)
    w.bon[0, 0], w.bpos[0, 0], w.bcolor[0, 0], w.bbig[0, 0] = True, [100.0 + 150.0 * np.cos(turn), 250.0 + 150.0 * np.sin(turn)], 1, 1
    w.sync()
    v = w.view(0, 0)
    assert v["rgb"].shape[:2] == w.cfg.frame[::-1] and v["depth"].shape == w.cfg.frame[::-1]
    ang, dist, wide = locate(w.labels(0, 0) > 0, v["depth"], w.cfg)
    left, top, right, down = box_of(w.labels(0, 0) > 0)
    box = (left / w.cfg.frame[0], top / w.cfg.frame[1], right / w.cfg.frame[0], down / w.cfg.frame[1])
    assert np.degrees(ang) == pytest.approx(30.0, abs=1.5)
    assert dist == pytest.approx(150.0, abs=6.0)
    assert wide == pytest.approx(2.0 * w.cfg.battery_radius, rel=0.15)
    assert 0.0 <= box[0] < box[2] <= 0.5 and 0.0 <= box[1] < box[3] <= 1.0  # a sinistra dell'immagine
    w.close()
