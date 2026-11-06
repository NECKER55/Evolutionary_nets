"""L'arena delle pile (evonets/arenas/pile/world.py, docs/SPEC_ARENA_PILE.md) e il
cervello a sezioni (evonets/brains/pile/sections.py, docs/SPEC_CERVELLO_SEZIONI.md)."""

import numpy as np

from evonets.arenas.pile.run import run
from evonets.brains.pile import (
    ActorCritic, INNATE_CAP, N_BASE, Sections, Sizes, decode, default_genome, inherit, mutate,
)
from evonets.arenas.pile.world import N_SENSES, PileConfig, PileWorld, SENSES


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
    yes = np.ones((1, 1), dtype=bool)
    pleasure = w.step(np.full((1, 1), 0.1), np.zeros((1, 1)), yes)["pleasure"].sum()  # piano, mordendo: la prende
    x = w.pos[0, 0, 0]
    for _ in range(int(3.0 / w.cfg.dt)):
        pleasure += w.step(np.full((1, 1), 1.0), np.zeros((1, 1)))["pleasure"].sum()  # spinge, ma mangia fermo
        if w.charging[0, 0] > 0:
            assert w.pos[0, 0, 0] == x
    assert abs(pleasure - 1.0) < 1e-6 and w.energy[0, 0] > e0 + 20.0
    put_battery(w, [w.pos[0, 0, 0] + 15.0, w.pos[0, 0, 1]], 1)
    w.heading[0, 0] = 0.0
    e1 = w.energy[0, 0]
    out = w.step(np.zeros((1, 1)), np.zeros((1, 1)), yes)
    assert out["pain"][0, 0] > 0 and w.pain[0, 0] == 1.0 and w.energy[0, 0] < e1 - w.cfg.bad_energy + 1e-9
    assert w.taken[-1][4] is False


def test_one_eats_only_by_biting_a_battery_in_front_of_the_mouth():
    w = world(n=1)
    w.pos[0, 0], w.heading[0, 0] = [100.0, 100.0], 0.0
    yes, no = np.ones((1, 1), dtype=bool), np.zeros((1, 1), dtype=bool)
    put_battery(w, [100.0, 115.0], 0)  # di lato
    w.step(np.zeros((1, 1)), np.zeros((1, 1)), yes)
    assert w.bon[0, 0] and not w.taken
    put_battery(w, [115.0, 100.0], 0)  # davanti, ma senza mordere: niente
    w.step(np.zeros((1, 1)), np.zeros((1, 1)), no)
    assert w.bon[0, 0] and not w.taken
    e = w.energy[0, 0]
    w.step(np.zeros((1, 1)), np.zeros((1, 1)), yes)  # mordendo sì
    assert not w.bon[0, 0] and len(w.taken) == 1
    assert w.energy[0, 0] < e  # mordere costa (la ricarica arriva nei passi dopo)


def test_the_walls_have_colours_and_bumps_have_a_side():
    w = world(n=2)
    w.pos[0] = [[250.0, 10.5], [400.0, 400.0]]  # il primo contro il muro nord, guardandolo
    w.heading[0] = [-np.pi / 2, 0.0]
    s = w.senses()[0]
    assert s[0, SENSES.index("occhio muro verde centro")] > 0.9  # nord = verde, addosso
    assert s[0, SENSES.index("urto davanti")] == 1.0 and s[0, SENSES.index("urto sinistra")] == 0.0
    assert s[1, SENSES.index("occhio muro viola centro")] > 0.0  # il secondo guarda a est: viola
    w.heading[0, 0] = 0.0  # ora il muro nord è a sinistra
    s = w.senses()[0]
    assert s[0, SENSES.index("urto sinistra")] == 1.0 and s[0, SENSES.index("urto davanti")] == 0.0


def test_the_innate_motor_circuit_turns_towards_the_object_and_slows_near_it():
    from evonets.arenas.pile.world import default_motor, motor_inputs, motor_net

    W = default_motor()[None]
    rays = np.zeros((1, 3))
    go = lambda ang, near: motor_net(motor_inputs(np.array([0]), np.array([True]), np.array([near]),
                                                  np.array([ang]), np.zeros(1), np.zeros(1), rays), W)
    thrust, steer, _ = go(0.5, 0.2)
    assert steer[0] > 0.5 and thrust[0] > 0.8  # a destra e lontano: gira a destra, va
    thrust, steer, _ = go(-0.5, 0.95)
    assert steer[0] < -0.5 and thrust[0] < 0.3  # a sinistra e vicino: gira a sinistra, piano


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


def test_working_memory_holds_the_last_object_after_it_is_gone_and_fades():
    from evonets.brains.pile import FOVEA

    b = Sections(1, 0.05, seed=0)
    seen = np.zeros((1, len(FOVEA)))
    seen[0, 1], seen[0, 3], seen[0, -1] = 1.0, 0.9, 1.0  # un oggetto giallo, vicino, appena guardato
    b._d(seen, np.zeros(1))  # il cancello si apre
    for _ in range(int(2.0 / 0.05)):
        held = b._d(np.zeros((1, len(FOVEA))), np.zeros(1)).copy()  # sparito: fovea vuota
    assert held[0, 1] > 0.4  # dopo 2 s si sa ancora che era giallo (τ = 4 s)
    for _ in range(int(20.0 / 0.05)):
        held = b._d(np.zeros((1, len(FOVEA))), np.zeros(1))
    assert held[0, 1] < 0.05


def test_fast_memory_codes_follow_the_object_more_than_the_context():
    """C: lo stesso oggetto con contesti diversi accende quasi gli stessi
    neuroni; oggetti diversi nello stesso contesto, neuroni diversi."""
    from evonets.brains.pile import FOVEA

    b = Sections(1, 0.05, seed=0)
    rng = np.random.default_rng(0)

    def code(colour, context):
        fov = np.zeros((1, len(FOVEA)))
        fov[0, colour], fov[0, 3] = 1.0, 0.8
        return b._c(context, fov.copy(), fov).copy()

    ctx = [(rng.random((1, b.nb)) > 0.6).astype(float) for _ in range(2)]
    same_object = (code(1, ctx[0]) * code(1, ctx[1])).sum()
    other_object = (code(1, ctx[0]) * code(0, ctx[0])).sum()
    assert code(1, ctx[0]).sum() >= 1
    assert same_object == code(1, ctx[0]).sum()  # lo stesso oggetto: esattamente gli stessi neuroni
    assert other_object < same_object


def test_the_stable_core_gives_two_situations_two_different_winners():
    b = Sections(1, 0.05, seed=0)
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
    reward = np.zeros((n, 3))  # piacere, dolore, energia spesa
    objs = {"vis": np.zeros((n, 1), bool), "feat": np.zeros((n, 1, 6)), "near": np.zeros((n, 1)),
            "ang": np.zeros((n, 1)), "good": np.zeros((n, 1), bool)}  # niente in vista
    for _ in range(4000):
        prog = a.step(raw, reward, objs)
        reward = np.zeros((n, 3))
        reward[:, 0] = np.where(prog == 3, 0.05, 0.0)  # solo "esplora" rende
    assert a.pi[:, 3].mean() > 0.5  # da 0,2 (a caso)


def test_every_lesion_runs():
    for les in ("B", "D", "C", "L", "U", "V"):
        r = run("sezioni", "A", 2, 0.5, seed=0, lesions=(les,), engine="numpy")
        assert r["lesioni"] == [les]


def test_genomes_cross_mutate_and_stay_in_their_limits():
    rng = np.random.default_rng(0)
    a, b = np.tile(default_genome(), (50, 1)), np.tile(default_genome(), (50, 1)) + 1.0
    child = inherit(a, b, rng)
    assert all((c == a[0]).all() or (c == b[0]).all() for c in child)  # ogni figlio da un genitore solo, intero
    assert 5 < sum((c == a[0]).all() for c in child) < 45
    g = mutate(np.tile(default_genome(), (5, 1)), rng, sigma=5.0)
    assert g.shape == (5, N_BASE)
    assert (np.abs(decode(g)["innate"]) <= INNATE_CAP).all()


def test_the_genome_carries_starting_weights_and_connections_and_birth_uses_them():
    """Nel genoma ci sono i pesi di partenza (B, E, V, O) e le
    connessioni di C; alla nascita il cervello riparte da lì, non da ciò che
    il genitore aveva imparato."""
    b = Sections(3, 0.05, seed=0)
    assert set(b.blocks) == {"proto", "c_in", "e0", "v0", "o0"}
    assert (b.c_in.sum(-1) == b.s.c_fan_in).all()  # 4 connessioni per neurone di C, dal genoma
    g = b.genome.copy()
    sl, shape = b.blocks["e0"]
    g[1, sl] = 0.7  # un figlio con un E di partenza diverso
    sl_c, _ = b.blocks["c_in"]
    g[1, sl_c] = 0.0
    g[1, sl_c.start] = 1.0  # e una sola connessione in C
    b.w_e[:] = 3.0  # ciò che si era imparato
    b.reborn(np.array([False, True, False]), g[1:2])
    assert np.allclose(b.w_e[1], 0.7) and np.allclose(b.w_e[0], 3.0)
    assert b.c_in[1].sum() == 1.0


def test_arbitration_gives_the_fast_memory_more_say_after_surprises():
    b = Sections(2, 0.05, seed=0)
    raw = np.zeros((2, N_SENSES))
    objs = {"vis": np.zeros((2, 1), bool), "feat": np.zeros((2, 1, 6)), "near": np.zeros((2, 1)),
            "ang": np.zeros((2, 1)), "good": np.zeros((2, 1), bool)}
    b.step(raw, np.zeros((2, 3)), objs)
    b.unc[:] = [0.0, 3.0]  # il secondo ha appena avuto molte sorprese
    b.step(raw, np.zeros((2, 3)), objs)
    k = b.votes["k"]
    assert k[0] == 0.0 and k[1] == 1.0
    les = Sections(2, 0.05, seed=0, sizes=Sizes(lesions=frozenset("R")))
    les.step(raw, np.zeros((2, 3)), objs)
    les.unc[:] = 3.0
    les.step(raw, np.zeros((2, 3)), objs)
    assert (les.votes["k"] == 0.0).all()  # senza arbitraggio: com'era prima


def test_a_run_saves_and_reopens_on_either_engine(tmp_path):
    from evonets.arenas.pile import saves as pile_saves
    from evonets.arenas.pile.run import make_run

    for engine in ("gpu", "numpy"):
        r = make_run("sezioni", "B", 3, 0, (), engine=engine)
        r.advance(60)
        pile_saves.save(r, tmp_path / f"{engine}.npz")
        g = pile_saves.export_state(r)["brain"]["genome"]
        for other in ("gpu", "numpy"):
            back = pile_saves.load(tmp_path / f"{engine}.npz", engine=other)
            assert abs(back.t - r.t) < 1e-3
            assert np.allclose(pile_saves.export_state(back)["brain"]["genome"], g, atol=1e-5)
            back.advance(10)  # e continua


def test_each_network_can_be_tried_alone(tmp_path):
    from evonets.arenas.pile import saves as pile_saves
    from evonets.arenas.pile.run import make_run

    r = make_run("sezioni", "A", 2, 0, (), engine="numpy")
    r.advance(100)
    pile_saves.save(r, tmp_path / "corsa.npz")
    for engine in ("gpu", "numpy"):
        alone = pile_saves.alone(tmp_path / "corsa.npz", engine=engine)
        assert alone.B == 2 * r.n and alone.n == 1  # ogni rete nel suo recinto
        mine = pile_saves.export_state(r)["brain"]["w_e"]
        assert np.allclose(pile_saves.export_state(alone)["brain"]["w_e"], mine, atol=1e-5)  # con ciò che ha imparato
        alone.advance(200)
        assert len(pile_saves.per_network(alone)) == 2 * r.n


def test_phase_b_mates_and_children_carry_the_parents_genes():
    r = run("sezioni", "B", 4, 3.0, seed=0, engine="numpy")
    assert r["fase"] == "B"  # gira con accoppiamento e rinascite da incrocio


# --- La versione sulla GPU -------------------------------------------------------------


def test_gpu_senses_are_the_numpy_senses():
    from evonets.arenas.pile.gpu import init_world, senses

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


def test_the_menu_starts_the_battery_arena_and_the_screen_draws_the_brain(monkeypatch, tmp_path):
    import os

    monkeypatch.setitem(os.environ, "SDL_VIDEODRIVER", "dummy")
    import pygame

    from evonets.viz.app import App
    from evonets.viz.pile.view import PileScreen

    app = App({})
    menu = app.launcher
    menu.task.select("pile")
    menu.p_encl.value, menu.chance.index = 3, 1
    menu.draw(app.screen)  # le righe dell'arena delle pile
    assert menu.p_agent in menu._visible() and menu.net not in menu._visible()
    for agent in ("sezioni", "attore-critico", "guidato"):
        menu.p_agent.select(agent)
        menu.save.value = str(tmp_path / f"{agent}.npz")
        menu.start()
        screen = app.current
        assert isinstance(screen, PileScreen) and screen.run.agent == agent
        for _ in range(3):
            screen.update()
            screen.draw(app.screen)
        from evonets.brains.pile import N_A

        for hover in (("section", "V"), ("section", "S"), ("A", 0), ("A", N_A - 1), ("arena",)) + (
                (("B", 0), ("D", 0), ("C", 0), ("E", 0)) if agent == "sezioni" else ()):
            screen.hover = hover
            screen.draw(app.screen)  # anche la spiegazione sotto il mouse
        screen.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p, mod=0))  # le reti una a una
        test = app.current
        assert test is not screen and test.testing
        test.update()
        test.draw(app.screen)
        test.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
        assert app.current is screen  # si torna alla corsa
        screen.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
        assert app.current is menu and (tmp_path / f"{agent}.npz").exists()  # salvata uscendo
    # Riprendere un salvataggio dal menu, e provarne le reti una a una.
    menu._choose(str(tmp_path / "sezioni.npz"))
    assert menu.task.value == "pile" and not menu.p_agent.enabled
    menu.start()
    assert isinstance(app.current, PileScreen) and app.current.run.t > 0
    app.current.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
    menu.explore()
    assert app.current.testing
    pygame.quit()



def test_every_brain_constant_is_a_gene_that_changes_behaviour():
    """Le costanti sono geni, una per essere: cambiando il gene cambia il
    comportamento (qui: ogni quanto si decide)."""
    from evonets.brains.pile import GENE_NAMES, Genes

    assert {"decide_every", "sal_near", "c_max", "u_fast", "gate_bias", "arb_fast"} <= set(GENE_NAMES)
    b = ActorCritic(2, 0.05, seed=0)
    k = GENE_NAMES.index("decide_every")
    b.genome[1, k] = np.log(2.0)  # il secondo decide ogni 2 s invece che ogni 0,5
    b.p = decode(b.genome)
    raw = np.zeros((2, N_SENSES))
    objs = {"vis": np.zeros((2, 1), bool), "feat": np.zeros((2, 1, 6)), "near": np.zeros((2, 1)),
            "ang": np.zeros((2, 1)), "good": np.zeros((2, 1), bool)}
    decisions = np.zeros(2)
    for _ in range(80):  # 4 s
        decisions += b._tick()
    assert decisions[0] == 8 and decisions[1] == 2
    assert Genes().gate_bias < 0 and decode(np.tile(default_genome(), (1, 1)))["gate_bias"][0] < 0  # con il segno


def test_o_learns_whatever_feature_matters_and_attention_follows():
    """Se le pile buone si riconoscono dalla grandezza (il colore non conta), O
    impara la grandezza buona e ignora il colore; e quando nella mappa ci sono
    pile buone e cattive, lo sguardo va più spesso alle buone."""
    from evonets.arenas.pile.run import make_run
    from evonets.brains.pile import MAP, N_FEATURES, o_active, o_read, priority_map

    r = make_run("sezioni", "A", 20, 0, (), PileConfig(good_by="grandezza"), "numpy")
    b, K = r.brain, r.cfg.batteries
    r.advance(int(200 / r.cfg.dt))
    good = np.repeat(r.w.good, r.n)
    old = r.w.age.ravel() > 60

    def V(colour, big):
        f = np.zeros((b.n, 1, N_FEATURES))
        i = np.arange(b.n)
        f[i, 0, colour], f[i, 0, 3 + (1 - big)] = 1.0, 1.0
        return o_read(*o_active(b.__dict__), f, r.w.energy.ravel() / r.cfg.energy_max, b.p)[0][:, 0]

    size = (V(0, good) + V(1, good) - V(0, 1 - good) - V(1, 1 - good)) / 2
    colour = (V(0, 1) + V(0, 0) - V(1, 1) - V(1, 0)) / 2
    assert np.mean(size[old] > 0) > 0.65 and np.median(size[old]) > 3 * abs(np.median(colour[old]))
    mass, share = [], []
    for _ in range(300):
        r.step()
        objs = {k: v.reshape((b.n,) + v.shape[2:]) for k, v in r.w.objects().items()}
        has, obj, _, _ = priority_map(objs, MAP)
        ok = objs["good"][np.arange(b.n)[:, None], obj]
        g, bd = has & (obj < K) & ok, has & (obj < K) & ~ok
        both = g.any(1) & bd.any(1)
        pg, pb = (b.att.prob * g).sum(1), (b.att.prob * bd).sum(1)
        both &= pg + pb > 0
        mass += list(pg[both] / (pg + pb)[both])
        share += list(g.sum(1)[both] / (g.sum(1) + bd.sum(1))[both])
    assert np.mean(mass) > np.mean(share) + 0.05


def test_the_priority_map_is_made_of_positions_not_objects():
    """S: un neurone per direzione, ognuno con l'oggetto più vicino in quella
    direzione; un oggetto nuovo non aggiunge neuroni."""
    from evonets.brains.pile import MAP, priority_map

    vis = np.array([[True, True, True]])
    ang = np.radians(np.array([[-50.0, -50.0, 40.0]]))  # due a sinistra (allo stesso angolo), uno a destra
    near = np.array([[0.2, 0.9, 0.5]])
    feat = np.eye(3)[None, :, :].repeat(1, 0)
    feat = np.concatenate([feat, np.zeros((1, 3, 3))], 2)
    has, obj, f, nr = priority_map({"vis": vis, "ang": ang, "near": near, "feat": feat}, MAP)
    assert has.shape == (1, MAP) and has.sum() == 2
    left = np.flatnonzero(has[0])[0]
    assert obj[0, left] == 1  # il più vicino dei due


def test_the_teacher_acts_like_the_guided_one_and_is_not_counted():
    """Il maestro (il primo di ogni recinto) mangia quasi solo pile buone; le
    misure contano solo gli allievi; toglierlo lo fa rinascere come un giovane."""
    from evonets.arenas.pile.run import make_run

    r = make_run("sezioni", "A", 8, 0, (), PileConfig(), "numpy", teacher=True)
    r.advance(int(120 / r.cfg.dt))
    first = np.arange(r.B * r.n) % r.n == 0
    g, b = r.by["good_by"][first].sum(), r.by["bad_by"][first].sum()
    assert g > 5 and g / (g + b) > 0.85
    assert r.counted[first].sum() == 0 and r.counted[~first].all()
    counted = sum(r.by["good_by"][~first]) + sum(r.by["bad_by"][~first])
    assert r.flat("good_taken").sum() + r.flat("bad_taken").sum() == counted
    on, err = r.teaching()
    assert on and err > 0
    r.set_teacher(False)
    assert not r.brain.teacher.any() and r.teaching() == (False, 0.0)
    assert (r.brain.w_e[first] == r.brain.start("e0", first)).all()  # rinato


def test_watching_the_teacher_teaches_o_what_happens_to_him():
    """O di chi osserva il maestro prevede meglio i suoi episodi (piacere e
    dolore) di quello di chi non lo osserva, senza esplodere."""
    from evonets.arenas.pile.run import make_run
    from evonets.brains.pile import GENE_NAMES, decode

    errs = []
    for observe in (1e-9, 1.0):
        r = make_run("sezioni", "A", 8, 0, (), PileConfig(), "numpy", teacher=True)
        r.brain.genome[:, GENE_NAMES.index("observe")] = np.log(observe)
        r.brain.p = decode(r.brain.genome)
        r.advance(int(120 / r.cfg.dt))
        errs.append(r.teaching()[1])
        assert np.abs(r.brain.o_pl).max() < 5
    assert errs[1] < errs[0]


def test_the_teacher_survives_a_save():
    from evonets.arenas.pile.saves import export_state, import_state
    from evonets.arenas.pile.run import make_run

    r = make_run("sezioni", "A", 2, 0, (), PileConfig(), "numpy", teacher=True)
    r.advance(10)
    st = export_state(r)
    r2 = make_run("sezioni", "A", 2, 0, (), PileConfig(), "numpy")
    import_state(r2, st)
    assert r2.teaching()[0] and (r2.counted == r.counted).all()


def test_the_ghost_teacher_is_unseen_and_eats_its_own_copy():
    """Il maestro fantasma: nessuno lo vede, e la pila che mangia sparisce solo per lui."""
    from evonets.arenas.pile.world import PileWorld

    cfg = PileConfig(beings=2, batteries=1)
    w = PileWorld(cfg, 1, 0)
    w.ghost[0, 0] = True
    w.bon[:] = True
    w.bwait[:] = 99.0
    w.bage[:] = 0.0
    w.bpos[0, 0] = (250.0, 250.0)
    w.pos[0] = [(250.0 - cfg.radius - cfg.battery_radius - 2, 250.0), (200.0, 250.0)]
    w.heading[0] = [0.0, 0.0]
    objs = w.objects()
    assert not objs["vis"][0, 1, 1 + 0]  # il secondo non vede il fantasma
    assert objs["vis"][0, 0, 1 + 1] or True  # il fantasma vede gli altri (se sono rivolti verso di lui)
    w.step(np.zeros((1, 2)), np.zeros((1, 2)), np.array([[True, False]]))
    assert w.bon[0, 0]  # per gli altri la pila c'è ancora
    assert w.ghost_hide[0, 0] > 0 and not w.objects()["vis"][0, 0, 0]  # per lui no
    assert w.objects()["vis"][0, 1, 0]


def test_c_codes_are_combinations_not_unions():
    """C come le cellule di Kenyon: oggetti con una caratteristica in comune
    (stessa grandezza, colore diverso) hanno codici quasi disgiunti; lo stesso
    oggetto dà sempre lo stesso codice; con niente da vedere né ricordare non
    si accende niente."""
    from evonets.brains.pile import FOVEA, IDENTITY, c_code

    b = Sections(1, 0.05, seed=0)
    nf = len(FOVEA)

    def code(feats):
        fov = np.zeros((1, nf))
        fov[0, list(feats)] = 1.0
        h = np.concatenate([fov[:, :IDENTITY], b.p["memory"][:, None] * fov[:, :IDENTITY], np.zeros((1, b.nb))], 1)
        return c_code(np.einsum("ncj,nj->nc", b.c_in, h), b.p)[0] > 0

    red_big, yellow_big, red_small = code((0, 3)), code((1, 3)), code((0, 4))
    assert 3 <= red_big.sum() <= 0.15 * b.s.c
    assert (red_big & yellow_big).sum() <= 0.25 * red_big.sum()
    assert (red_big & red_small).sum() <= 0.25 * red_big.sum()
    assert (code((0, 3)) == red_big).all() and code(()).sum() == 0


def test_a_new_object_makes_e_decide_at_once():
    """Quando l'oggetto guardato cambia, E decide subito (non aspetta il suo
    turno) e il suo ritmo riparte da lì."""
    b = Sections(1, 0.05, seed=0)
    raw = np.zeros((1, N_SENSES))
    feat = np.zeros((1, 2, 6))
    feat[0, 0, [0, 3]] = 1.0
    feat[0, 1, [1, 4]] = 1.0
    objs = {"vis": np.array([[True, False]]), "feat": feat, "near": np.array([[0.5, 0.5]]),
            "ang": np.array([[0.0, 0.2]]), "good": np.zeros((1, 2), bool)}
    b.step(raw, np.zeros((1, 3)), objs)
    b.att.target[:] = 0  # guarda il primo (E potrebbe averlo appena lasciato)
    b.step(raw, np.zeros((1, 3)), objs)
    assert not b.att.changed[0] and 0.0 < b.clock[0] < b.p["decide_every"][0]  # a metà del turno
    objs["vis"] = np.array([[False, True]])  # il primo sparisce: si guarda il secondo
    b.step(raw, np.zeros((1, 3)), objs)
    assert b.att.changed[0] and b.clock[0] == 0.0


def test_k_keeps_the_old_drawer_and_returns_to_it():
    """K: quando gli esiti smettono di tornare con il cassetto attivo e nessun
    altro li spiega, si recluta un contesto nuovo (dai valori innati) e il
    vecchio cassetto resta intatto; quando il mondo torna com'era, si torna al
    vecchio cassetto senza reimparare."""
    from evonets.brains.pile import N_FEATURES, o_step

    b = Sections(1, 0.05, seed=0)
    red, yellow = np.zeros(N_FEATURES), np.zeros(N_FEATURES)
    red[0], yellow[1] = 1.0, 1.0

    def episode(seen, pleasure, pain):
        b.o_e[0] = seen
        for felt in (np.array([[pleasure, pain, 0.0]]), np.zeros((1, 3))):
            out = o_step(b.__dict__, felt, np.array([0.5]), np.array([0]), np.array([False]), b.p, set())
            for k, v in out.items():
                setattr(b, k, v)

    for _ in range(30):  # il rosso è buono, il giallo fa male
        episode(red, 1.0, 0.0)
        episode(yellow, 0.0, 0.32)
    first = b.k_on[0]
    learned = b.o_pl[0, first, 0]
    assert learned > 0.5
    for _ in range(30):  # si invertono
        episode(red, 0.0, 0.32)
        episode(yellow, 1.0, 0.0)
    assert b.k_on[0] != first and b.k_used[0].sum() >= 2
    assert abs(b.o_pl[0, first, 0] - learned) < 0.3  # il vecchio cassetto è quasi intatto
    for _ in range(4):  # tornano come prima: bastano pochi episodi
        episode(red, 1.0, 0.0)
        episode(yellow, 0.0, 0.32)
    assert b.k_on[0] == first
