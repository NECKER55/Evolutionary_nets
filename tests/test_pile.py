"""L'arena delle pile (evonets/sim/pile.py, docs/SPEC_ARENA_PILE.md) e il
cervello a sezioni (evonets/net/sections.py, docs/SPEC_CERVELLO_SEZIONI.md)."""

import numpy as np

from evonets.evo.pile_test import run
from evonets.net.sections import (
    INNATE_CAP,
    N_GENOME,
    ActorCritic,
    Sections,
    Sizes,
    cross,
    decode,
    default_genome,
    mutate,
)
from evonets.sim.pile import N_SENSES, SENSES, PileConfig, PileWorld


def world(n=2, **kw):
    w = PileWorld(PileConfig(beings=n, **kw), 1, seed=0)
    w.bon[:] = False
    w.bwait[:] = 1e9  # niente pile nuove, se non quelle messe dal test
    return w


def put_battery(w, pos, color):
    w.bpos[0, 0], w.bcolor[0, 0], w.bon[0, 0], w.bage[0, 0] = pos, color, True, 0.0


def test_the_eyes_see_the_battery_in_front_and_not_oneself():
    w = world()
    w.pos[0] = [[100.0, 250.0], [400.0, 400.0]]
    w.heading[0] = [0.0, 0.0]
    put_battery(w, [200.0, 250.0], 0)
    s = w.senses()[0, 0]
    assert s[SENSES.index("occhio rosso centro")] > 0.6
    assert s[SENSES.index("occhio rosso sinistra")] == s[SENSES.index("occhio giallo centro")] == 0.0
    assert s[SENSES.index("occhio blu centro")] == 0.0  # l'altro è lontano dal centro della vista e di spalle


def test_a_good_battery_charges_while_standing_still_and_a_bad_one_hurts():
    w = world(n=1)
    w.pos[0, 0], w.heading[0, 0] = [100.0, 100.0], 0.0
    w.good[0] = 0
    put_battery(w, [115.0, 100.0], 0)
    e0 = w.energy[0, 0]
    pleasure = w.step(np.full((1, 1), 0.1), np.zeros((1, 1)))["pleasure"].sum()  # piano: la prende
    x = w.pos[0, 0, 0]
    for _ in range(int(3.0 / w.cfg.dt)):
        pleasure += w.step(np.full((1, 1), 1.0), np.zeros((1, 1)))["pleasure"].sum()  # spinge, ma mangia fermo
        if w.charging[0, 0] > 0:
            assert w.pos[0, 0, 0] == x
    assert abs(pleasure - 1.0) < 1e-6 and w.energy[0, 0] > e0 + 20.0
    put_battery(w, [w.pos[0, 0, 0] + 15.0, w.pos[0, 0, 1]], 1)
    w.heading[0, 0] = 0.0
    e1 = w.energy[0, 0]
    out = w.step(np.zeros((1, 1)), np.zeros((1, 1)))
    assert out["pain"][0, 0] > 0 and w.pain[0, 0] == 1.0 and w.energy[0, 0] < e1 - w.cfg.bad_energy + 1e-9
    assert w.taken[-1][4] is False


def test_only_a_battery_in_front_of_the_mouth_is_taken():
    w = world(n=1)
    w.pos[0, 0], w.heading[0, 0] = [100.0, 100.0], 0.0
    put_battery(w, [100.0, 115.0], 0)  # di lato
    w.step(np.zeros((1, 1)), np.zeros((1, 1)))
    assert w.bon[0, 0] and not w.taken


def test_inversions_and_zone_changes():
    w = PileWorld(PileConfig(inversion_every=1.0, zone_every=2.0), 3, seed=1)
    good, zone = w.good.copy(), w.zone.copy()
    for _ in range(int(2.05 / w.cfg.dt)):
        w.step(np.zeros((3, w.n)), np.zeros((3, w.n)))
    assert len(w.inversions) == 2 and (w.good == good).all()  # scambiati due volte
    assert len(w.zone_changes) == 1 and (w.zone != zone).all()
    w.bon[:] = False
    w.bwait[:] = 0.0
    w.step(np.zeros((3, w.n)), np.zeros((3, w.n)))
    centre = w.zone_centres()[w.zone][:, None, :]
    assert (np.abs(w.bpos - centre) <= w.cfg.zone_half + 1e-9).all()  # le pile nuove nella zona nuova


def test_walls_hold_and_the_rays_feel_them():
    w = world(n=1)
    w.pos[0, 0], w.heading[0, 0] = [480.0, 250.0], 0.0
    for _ in range(40):
        w.step(np.ones((1, 1)), np.zeros((1, 1)))
    assert w.pos[0, 0, 0] <= w.cfg.size - w.cfg.radius
    assert w.senses()[0, 0, SENSES.index("distanza centro")] == 1.0


def test_two_beings_face_to_face_mate_and_pay_for_it():
    w = world(mating=True)
    w.pos[0] = [[200.0, 200.0], [220.0, 200.0]]
    w.heading[0] = [0.0, np.pi]
    w.energy[:] = 80.0
    mates = []
    for _ in range(int(2.5 / w.cfg.dt)):
        mates += w.step(np.zeros((1, 2)), np.zeros((1, 2)))["mates"]
    assert mates == [(0, 0, 1)]
    assert (w.energy[0] < 80.0 - w.cfg.mate_cost + 1e-9).all()


def test_the_starved_are_reborn():
    w = world(n=1)
    w.energy[:] = 0.01
    out = w.step(np.ones((1, 1)), np.zeros((1, 1)))
    assert out["dead"][0, 0] and w.energy[0, 0] == w.cfg.energy_start and w.deaths["fame"] == 1


def test_the_guided_being_chooses_and_survives():
    """Chi sa il colore buono e la zona (il controllo del mondo) sceglie bene e
    non muore di fame; chi parte senza sapere niente sceglie quasi a caso."""
    guided = run("guidato", "A", 10, 4.0, seed=0, engine="numpy")
    blind = run("attore-critico", "A", 10, 4.0, seed=0, engine="numpy")
    assert guided["buone"] > 0.7 and guided["morti di fame al minuto"] < 0.02
    assert blind["buone"] < 0.65


# --- Il cervello a sezioni, pezzo per pezzo --------------------------------------------


def test_working_memory_holds_after_an_event_and_fades():
    b = Sections(1, 0.05, seed=0)
    pattern = np.zeros((1, b.nb))
    pattern[0, 3] = 1.0
    b._d(pattern, np.ones(1), np.zeros(1))  # un evento: il cancello si apre
    for _ in range(int(2.0 / 0.05)):
        held = b._d(np.zeros((1, b.nb)), np.zeros(1), np.zeros(1)).copy()  # senza ingresso
    assert held[0, 3] > 0.4  # dopo 2 s c'è ancora (τ = 4 s)
    for _ in range(int(20.0 / 0.05)):
        held = b._d(np.zeros((1, b.nb)), np.zeros(1), np.zeros(1))
    assert held[0, 3] < 0.05


def test_fast_memory_gives_sparse_codes_that_tell_situations_apart():
    b = Sections(1, 0.05, seed=0)
    rng = np.random.default_rng(0)
    codes = []
    for _ in range(2):
        x = (rng.random((1, b.nb)) > 0.7).astype(float)
        codes.append(b._c(x, np.zeros_like(x)).copy())
    assert all(c.sum() == b.s.c_winners for c in codes)
    assert (codes[0] * codes[1]).sum() < b.s.c_winners  # due situazioni diverse: codici diversi


def test_the_stable_core_gives_two_situations_two_different_winners():
    b = Sections(1, 0.05, seed=0, sizes=Sizes(conscience=2.0))
    p1 = np.zeros((1, b.proto.shape[2]))
    p2 = p1.copy()
    p1[0, :5], p2[0, 5:10] = 1.0, 1.0
    b.p["eta_b"][:] = 2.0  # qui si prova la regola, non la sua lentezza
    for k in range(400):
        x = p1 if (k // 20) % 2 == 0 else p2
        b._b(x, np.zeros(1))
    wins = []
    for x in (p1, p2):
        for _ in range(30):
            b._b(x, np.zeros(1))
        wins.append(int(b.state_b.argmax()))
    assert wins[0] != wins[1]


def test_the_actor_critic_learns_which_program_pays():
    n = 50
    a = ActorCritic(n, 0.05, seed=0)
    raw = np.zeros((n, N_SENSES))
    reward = np.zeros(n)
    for _ in range(4000):
        prog = a.step(raw, reward)
        reward = np.where(prog == 3, 0.05, 0.0)  # solo "esplora" rende
    assert a.pi[:, 3].mean() > 0.5  # da 0,2 (a caso)


def test_every_lesion_runs():
    for les in ("B", "D", "C", "L", "U", "V"):
        r = run("sezioni", "A", 2, 0.5, seed=0, lesions=(les,), engine="numpy")
        assert r["lesioni"] == [les]


def test_genomes_cross_mutate_and_stay_in_their_limits():
    rng = np.random.default_rng(0)
    a, b = default_genome()[None], default_genome()[None] + 1.0
    child = cross(a, b, rng)
    assert ((child == a) | (child == b)).all()
    g = mutate(np.tile(default_genome(), (5, 1)), rng, sigma=5.0)
    assert g.shape == (5, N_GENOME)
    assert (np.abs(decode(g)["innate"]) <= INNATE_CAP).all()


def test_phase_b_mates_and_children_carry_the_parents_genes():
    r = run("sezioni", "B", 4, 3.0, seed=0, engine="numpy")
    assert r["fase"] == "B"  # gira con accoppiamento e rinascite da incrocio


# --- La versione sulla GPU -------------------------------------------------------------


def test_gpu_senses_are_the_numpy_senses():
    from evonets.gpu.pile import init_world, senses

    cfg = PileConfig()
    w = PileWorld(cfg, 5, seed=3)
    for _ in range(40):
        w.step(np.full((5, w.n), 0.7), np.full((5, w.n), 0.3))
    state = init_world(cfg, 5, 3)
    for k in ("pos", "heading", "bpos", "bcolor", "bon", "charging", "energy", "taste", "pain"):
        state[k] = getattr(w, k)
    assert np.allclose(np.asarray(senses(cfg, state)), w.senses(), atol=1e-4)


def test_gpu_and_numpy_agree_on_average():
    """Stesse regole, numeri casuali diversi: le medie coincidono."""
    for agent in ("guidato", "sezioni"):
        a = run(agent, "A", 20, 3.0, seed=0, engine="gpu")
        b = run(agent, "A", 20, 3.0, seed=0, engine="numpy")
        assert a["motore"] == "PileGpu" and b["motore"] == "NumpyRun"
        assert abs(a["pile al minuto"] - b["pile al minuto"]) < 0.25 * b["pile al minuto"] + 0.1
        assert abs(a["buone"] - b["buone"]) < 0.08


def test_gpu_phase_b_and_lesions_run():
    for les in ((), ("B",), ("D", "C")):
        r = run("sezioni", "B", 4, 1.0, seed=0, lesions=les, engine="gpu")
        assert r["motore"] == "PileGpu"


# --- Il menu e la schermata ------------------------------------------------------------


def test_the_menu_starts_the_battery_arena_and_the_screen_draws_the_brain(monkeypatch):
    import os

    monkeypatch.setitem(os.environ, "SDL_VIDEODRIVER", "dummy")
    import pygame

    from evonets.viz.app import App
    from evonets.viz.pile_view import PileScreen

    app = App({})
    menu = app.launcher
    menu.task.select("pile")
    menu.p_encl.value, menu.chance.index = 3, 1
    menu.draw(app.screen)  # le righe dell'arena delle pile
    assert menu._visible() is menu.pile_fields
    for agent in ("sezioni", "attore-critico", "guidato"):
        menu.p_agent.select(agent)
        menu.start()
        screen = app.current
        assert isinstance(screen, PileScreen) and screen.run.agent == agent
        for _ in range(3):
            screen.update()
            screen.draw(app.screen)
        screen.hover = ("section", "V")
        screen.draw(app.screen)  # anche la spiegazione sotto il mouse
        screen.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
        assert app.current is menu
    pygame.quit()
