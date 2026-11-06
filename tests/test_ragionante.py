"""Il cervello ragionante (evonets/brains/ragionante/): per ora la sezione C, il
motore che si tara da solo su un corpo che non conosce."""

import ast
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import evonets.brains.ragionante as ragionante
from evonets.arenas.mondo3d.bodies import BODIES
from evonets.arenas.mondo3d.run import Run3D, measures
from evonets.arenas.mondo3d.world import Mondo3D, Mondo3DConfig
from evonets.brains.ragionante import BIRTH, RETUNE, Motor


def run(body=None, enclosures=1, seed=0, **kw):
    return Run3D(Mondo3DConfig(body=body or tuple(BODIES)[0], image=64, **kw), "ragionante", enclosures, seed)


def seconds(r, s):
    r.advance(int(round(s / r.cfg.dt)))


def test_this_brain_does_not_use_the_sections_brain():
    for file in Path(ragionante.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert "brains.pile" not in (node.module or "") and not (node.module or "").startswith("..pile")


def test_a_body_is_a_wiring_and_feels_what_it_really_did():
    w = Mondo3D(Mondo3DConfig(body="due ruote e morso", image=64), 1, 0)
    w.bon[:] = False
    w.bwait[:] = 1e9
    w.pos[0, 0], w.heading[0, 0] = [100.0, 250.0], 0.0
    w.sync()
    forward = np.array([[[1.0, 1.0, 1.0]]])  # le due ruote avanti, bocca chiusa
    for _ in range(30):
        w.step(*w.wires(forward))
    speed, spin, jaw = w.felt()[0, 0]
    assert speed > 0.9 and abs(spin) < 0.1 and jaw == 1.0
    assert w.pos[0, 0, 0] > 150.0
    left = np.array([[[-1.0, 1.0, 0.0]]])  # ruote opposte: gira sul posto
    for _ in range(20):
        w.step(*w.wires(left))
    speed, spin, jaw = w.felt()[0, 0]
    assert abs(speed) < 0.1 and spin > 0.7 and jaw == 0.0


def test_the_hidden_neurons_learn_the_inverse_of_a_hidden_table():
    hidden = np.array(BODIES["tutto mescolato"][0], dtype=float)
    m = Motor((1, 1), 3, 0.05)
    for _ in range(4000):
        u = m.step(np.zeros((1, 1, 3)), (m.u @ hidden.T), np.zeros((1, 1), dtype=bool))
        m.left[:] = 1.0  # resta in taratura
        assert u.shape == (1, 1, 3)
    aims = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.3, -0.3, 0.0], [0.0, 0.0, 0.8]])
    assert np.abs(m.forward(aims)[0, 0] @ hidden.T - aims).max() < 0.2


def test_the_hidden_neurons_learn_a_body_that_a_table_cannot():
    """Un motore che sotto 0,5 non si muove: nessuna tabella lo traduce bene."""
    def body(u):
        return np.sign(u) * np.clip(np.abs(u) - 0.5, 0.0, 1.0) * 2.0

    m = Motor((1, 1), 3, 0.05)
    for _ in range(6000):
        m.step(np.zeros((1, 1, 3)), body(m.u), np.zeros((1, 1), dtype=bool))
        m.left[:] = 1.0
    aims = m.rng.uniform(-0.8, 0.8, (300, 3))
    table = np.linalg.lstsq(m.seen_x[0, 0], m.seen_u[0, 0], rcond=None)[0]  # la tabella migliore possibile
    net = np.abs(body(m.forward(aims)[0, 0].clip(-1.0, 1.0)) - aims).mean()
    best = np.abs(body((aims @ table).clip(-1.0, 1.0)) - aims).mean()
    assert net < 0.7 * best


def test_tuning_can_be_asked_at_any_moment_and_keeps_what_it_knows():
    m = Motor((1, 2), 3, 0.05)
    m.left[:] = 0.0
    before = [p.copy() for p in m.params]
    m.count[:] = 50
    m.tune(10.0, np.array([[True, False]]))
    assert m.tuning.tolist() == [[True, False]] and m.count.tolist() == [[0, 50]]
    assert all((a == b).all() for a, b in zip(before, m.params))


def test_changing_body_and_tuning_do_not_touch_the_commander():
    r = run("spinta, sterzo, morso")
    seconds(r, 5.0)
    boss, state = r.brain.boss, {k: id(v) for k, v in vars(r.brain.boss).items()}
    r.brain.motor.tune(RETUNE)
    r.w.set_body("quattro motori e morso")
    seconds(r, 1.0)
    assert r.brain.boss is boss and set(vars(boss)) == set(state)
    for file in ("base.py", "perfect.py"):
        source = (Path(ragionante.__file__).parents[1] / "mondo3d" / file).read_text()
        assert "wiring" not in source and "motor" not in source


def test_nothing_is_learned_during_a_crash():
    m = Motor((1, 1), 3, 0.05)
    felt = np.ones((1, 1, 3))
    born = [p.copy() for p in m.params]
    for _ in range(100):
        m.step(np.zeros((1, 1, 3)), felt, np.ones((1, 1), dtype=bool))
    assert all((a == b).all() for a, b in zip(born, m.params))


@pytest.mark.parametrize("body", tuple(BODIES))
def test_every_body_is_learned_from_scratch_among_the_walls(body):
    r = run(body)
    assert r.brain.motor.tuning.all()
    seconds(r, BIRTH)
    eaten = len(r.w.taken)
    seconds(r, 120.0)
    m = measures(r)
    assert not r.brain.motor.tuning.all() or m["tarature"] > 1
    assert len(r.w.taken) - eaten >= 8 and m["morti di fame"] == 0


def test_a_new_body_makes_it_retune_by_itself_and_recover():
    r = run("spinta, sterzo, morso", seed=1)
    seconds(r, BIRTH + 60.0)
    before = int(r.brain.motor.tunings[0, 0])
    r.w.set_body("uscite in disordine")
    seconds(r, 2.0 * RETUNE)
    assert r.brain.motor.tunings[0, 0] > before  # nessuno l'ha avvertito
    seconds(r, RETUNE)
    eaten = len(r.w.taken)
    seconds(r, 90.0)
    assert len(r.w.taken) - eaten >= 5


def test_without_intention_there_is_no_command():
    m = Motor((2, 1), 5, 0.05)
    m.left[:] = 0.0
    assert (m.step(np.zeros((2, 1, 3)), np.zeros((2, 1, 3)), np.zeros((2, 1), dtype=bool)) == 0.0).all()


def test_a_body_that_does_the_opposite_is_noticed_without_a_goal():
    m = Motor((2, 1), 3, 0.05)
    m.left[:] = 0.0
    wanted = np.array([[[0.0, 0.8, 0.0]], [[0.0, 0.8, 0.0]]])
    for _ in range(120):
        m.step(wanted, wanted * [[[1.0]], [[-1.0]]], np.zeros((2, 1), dtype=bool))
    assert m.lost.tolist() == [[False], [True]]


def test_a_mouth_that_does_not_open_makes_it_retune():
    r = run("spinta, sterzo, morso", seed=1)
    seconds(r, BIRTH)
    m = r.brain.motor
    before = int(m.tunings[0, 0])
    m.params[2][..., 2, :] = 0.0  # la bocca resta chiusa, le ruote obbediscono
    for _ in range(int(120.0 / r.cfg.dt)):
        r.advance(1)
        if m.tunings[0, 0] > before:
            break
    assert m.tunings[0, 0] > before


def test_more_outputs_start_from_a_new_network():
    r = run()
    seconds(r, 2.0)
    r.w.set_body("quattro motori e morso")
    seconds(r, 1.0)
    assert r.brain.motor.response.shape == (1, 1, 5, 3) and r.brain.motor.tuning.all()


def test_the_screen_shows_the_section_and_the_buttons_tune_and_change_body():
    import pygame

    from evonets.viz.mondo3d.view import Mondo3DScreen

    pygame.font.init()
    pygame.display.init()

    class App:
        font = pygame.font.SysFont("monospace", 15)
        small = pygame.font.SysFont("monospace", 12)

    surface = pygame.Surface((1440, 700))
    r = run()
    screen = Mondo3DScreen(App(), r)
    seconds(r, BIRTH + 1.0)
    screen.draw(surface)
    tune, change = screen.buttons
    assert not r.brain.motor.tuning.any()
    screen.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=tune.rect.center))
    assert r.brain.motor.tuning.all() and r.brain.motor.left[0, 0] == RETUNE
    for _ in BODIES:  # ogni corpo si disegna, anche con 5 uscite
        screen.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=change.rect.center))
        r.advance(2)
        screen.draw(surface)
    assert r.w.body == tuple(BODIES)[0]


def test_keys_change_being_and_enclosure_and_explain_themselves_when_there_is_only_one():
    import pygame

    from evonets.viz.mondo3d.view import Mondo3DScreen

    pygame.font.init()
    pygame.display.init()
    press = lambda sc, key: sc.handle(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
    many = Mondo3DScreen(None, run(enclosures=2, beings=3))
    press(many, pygame.K_TAB)
    press(many, pygame.K_RIGHT)
    press(many, pygame.K_DOWN)
    assert (many.follow, many.encl) == (2, 1)
    press(many, pygame.K_LEFT)
    press(many, pygame.K_UP)
    assert (many.follow, many.encl) == (1, 0) and not many.note
    one = Mondo3DScreen(None, run())
    press(one, pygame.K_TAB)
    assert "solo essere" in one.note
    press(one, pygame.K_DOWN)
    assert "solo recinto" in one.note
