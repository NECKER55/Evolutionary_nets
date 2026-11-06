"""La sezione P con reti già pronte: la memoria degli oggetti, il tracker, il ritaglio a
scala fissa e i nomi dalla banca. Non scarica niente: servono solo i pacchetti del gruppo `percezione`."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from evonets.arenas.mondo3d.world import Mondo3DConfig
from evonets.brains.ragionante.bank import Bank
from evonets.brains.ragionante.ready import EMBED, GREY, VIEWS, WINDOW, ObjectMemory, Tracker, fixed, hue, overlap, things
from evonets.brains.ragionante.vision import FEATURES, KINDS, VECTOR, Percept


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


def test_the_bank_names_only_things_of_the_world_and_only_if_they_look_alike():
    bank = Bank()
    rows = [bank.names.index(n) for n in ("rossa cilindro grande", "gialla cilindro piccola", "essere", "blu scatola grande")]
    vectors = np.stack([bank.vectors[np.flatnonzero(bank.kind == k)[0]] for k in rows])
    kinds, score = bank.name(vectors)
    assert list(kinds) == [KINDS.index("pila rossa"), KINDS.index("pila gialla"), KINDS.index("essere"), -1]
    assert np.allclose(score, 1.0, atol=1e-5)
    assert (bank.name(vectors, named=1.1)[0] == -1).all()  # nessuna somiglia abbastanza: nessun nome


def test_what_leaves_p_has_no_names_and_an_empty_place_is_all_zeros():
    p = Percept.empty(1, 1)
    p.on[0, 0, 0], p.kind[0, 0, 0], p.dist[0, 0, 0], p.size[0, 0, 0] = True, -1, 100.0, 16.0
    p.vector[0, 0, 0] = 1.0 / np.sqrt(VECTOR)
    out = p.features(350.0)
    assert out.shape == (1, 1, p.on.shape[-1], FEATURES) and FEATURES == VECTOR + 5
    assert out[0, 0, 0, VECTOR] == pytest.approx(0.16) and out[0, 0, 0, -1] == 1.0
    assert not out[0, 0, 1:].any()
    assert not p.embedding(350.0)[0, 0, 0, :len(KINDS)].any()  # senza nome, nessun tipo acceso


def test_the_true_objects_in_view_come_from_the_simulator():
    pytest.importorskip("mujoco")
    from evonets.arenas.mondo3d.world import Mondo3D

    w = Mondo3D(Mondo3DConfig(beings=2), 1, 3)
    w.bon[:], w.bwait[:] = False, 1e9
    w.pos[0], w.heading[0] = [[100.0, 250.0], [220.0, 230.0]], 0.0
    w.bon[0, 0], w.bpos[0, 0], w.bcolor[0, 0], w.bbig[0, 0] = True, [180.0, 270.0], 1, 1
    w.sync()
    names = sorted(n for _, n in things(w, 0, 0))
    assert names == ["essere", "pila gialla"]
    w.close()



def test_two_things_in_the_same_image_are_never_the_same_object():
    m = ObjectMemory()
    a = unit(1)
    assert m.know(a) == (0, True)
    assert m.know(a, taken=[0]) == (1, True)  # identica, ma la prima è già in questa immagine
    assert m.know(a, taken=[1]) == (0, False)
