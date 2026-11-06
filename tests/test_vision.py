"""La sezione P, la percezione: la rete nostra (vision.py), il detector finto e il
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
from evonets.brains.ragionante.vision import EMBED, KINDS, MAX_OBJECTS, Detector, FakeDetector, Percept, blobs

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


def test_blobs_put_an_object_where_the_camera_sees_it():
    cfg = Mondo3DConfig()
    classes = np.zeros((132, 132), dtype=int)
    classes[60:80, 100:120] = 1  # a destra del centro
    found = blobs(classes, np.full((132, 132), 100.0), cfg)
    assert len(found) == 1
    kind, ang, dist, wide, box = found[0]
    expected = np.arctan2((110.0 - 66.0) / 66.0 * 100.0 * np.tan(np.radians(60.0)), 100.0 + cfg.camera_ahead)
    assert kind == 0 and ang == pytest.approx(-expected, abs=1e-6) and dist == pytest.approx(184.4, abs=0.3)


def test_a_thin_line_along_an_edge_is_not_an_object():
    classes = np.zeros((132, 132), dtype=int)
    classes[np.arange(100), np.arange(100)] = 2  # l'angolo tra due muri
    classes[10:120, 50] = 2
    assert blobs(classes, np.full((132, 132), 100.0), Mondo3DConfig()) == []


@pytest.mark.parametrize("colour,angle", [(0, 0.0), (1, 0.0), (1, 0.6), (0, -0.6)])
def test_the_real_detector_finds_a_pile_in_the_right_place(colour, angle):
    w = world((100.0 + 90.0 * np.cos(angle), 250.0 + 90.0 * np.sin(angle)), colour)
    seen = Detector(w).see()
    on = np.flatnonzero(seen.on[0, 0])
    assert len(on) == 1 and seen.kind[0, 0, on[0]] == colour
    assert seen.ang[0, 0, on[0]] == pytest.approx(angle, abs=np.radians(3.0))
    assert seen.dist[0, 0, on[0]] == pytest.approx(90.0, rel=0.1)


def test_the_real_detector_sees_nothing_in_an_empty_enclosure_even_by_a_wall():
    w = world(None)
    seen = Detector(w)
    assert not seen.see().on.any()
    w.pos[0, 0], w.heading[0, 0] = [470.0, 250.0], 0.0
    w.sync()
    found = seen.see()
    assert not (found.on[0, 0] & (found.kind[0, 0] < 2)).any()


def test_fake_and_real_detector_give_the_same_interface():
    w = world()
    fake, real = FakeDetector(w).see(), Detector(w).see()
    for name in Percept.__dataclass_fields__:
        assert getattr(fake, name).shape == getattr(real, name).shape
    assert fake.on.shape == (1, 1, MAX_OBJECTS)
    assert fake.embedding(350.0).shape == real.embedding(350.0).shape == (1, 1, MAX_OBJECTS, EMBED)
    k = np.flatnonzero(fake.on[0, 0])[0]
    assert fake.kind[0, 0, k] == real.kind[0, 0, np.flatnonzero(real.on[0, 0])[0]] == 1


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
