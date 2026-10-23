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
    assert w.energy[0] > w.energy[1] + cfg.arena_food_energy - 1.0 and w.taste[0] > 0.9
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


# --- La GPU fa le stesse cose del processore ------------------------------------

def _gpu_pair(net_type: str):
    import jax.numpy as jnp

    from evonets.gpu import arena as ga
    from evonets.net.controllers import NeuralController

    cfg = small(net_type=net_type, arena_food_rate=0.0, arena_aging_h0=0.0)
    w = ArenaWorld(cfg, 20, 20, seed=5)
    rng = np.random.default_rng(0)
    w.pos[:] = rng.uniform([0, 0], [400, 320], (20, 2))
    w.pos[0] = [399.0, 50.0]  # vicino al bordo: si vede dall'altra parte
    w.pos[1] = [3.0, 50.0]
    w.task.food_on[:] = False
    w.task.food[:30] = w.pos[:15].repeat(2, axis=0)[:30] + rng.uniform(-12, 12, (30, 2))
    w.task.food[:30] %= [400, 320]
    w.task.food_on[:30] = True
    w.grids_dirty = True
    ctrl = NeuralController(cfg, 20, 3)
    gpu = ga.ArenaGpu(w, ctrl, 1, plain_bodies(20))
    return cfg, w, ctrl, gpu, ga, jnp


def plain_bodies(n):
    """Corpi del mondo base: taglia 1, nessun segnale, nessun genitore."""
    return {"size": np.ones(n), "sig_color": np.zeros(n), "sig_mol": np.zeros(n), "uid": np.arange(n),
            "parent": np.full(n, -1), "young": np.zeros(n)}


def test_the_gpu_senses_the_same_as_the_processor():
    for net in ("rate", "spike-ei"):
        cfg, w, ctrl, gpu, ga, jnp = _gpu_pair(net)
        cpu = ctrl.brains.prepared().quanta(w, np.arange(w.n))
        s = gpu.s
        food = ga.neighbours(ga.table(s["food"], s["food_on"] > 0, gpu.geo), s["pos"], gpu.geo)
        beings = ga.neighbours(ga.table(s["pos"], s["alive"] > 0, gpu.geo), s["pos"], gpu.geo)
        on_gpu = np.asarray(ga.quanta(gpu.ph, gpu.P, s, food, beings))
        # La GPU cerca i vicini attorno al centro dell'essere, il processore
        # attorno al recettore (1 u più in là): oggetti lontani oltre 80 u
        # possono contare in uno e non nell'altro (al più qualche per cento).
        assert np.allclose(on_gpu, cpu, rtol=2e-2, atol=1e-2), net


def test_a_gpu_step_moves_and_feeds_like_the_processor():
    cfg, w, ctrl, gpu, ga, jnp = _gpu_pair("rate")
    actions = np.random.default_rng(2).uniform(-1, 1, (w.n, 2))
    s = gpu.s
    food = ga.neighbours(ga.table(s["food"], s["food_on"] > 0, gpu.geo), s["pos"], gpu.geo)
    beings = ga.neighbours(ga.table(s["pos"], s["alive"] > 0, gpu.geo), s["pos"], gpu.geo)
    out = ga.physics(gpu.ph, gpu.P, s, jnp.asarray(actions, dtype=jnp.float32), jnp.zeros((w.n, 4)), food, beings)
    w.step(actions, ctrl.upkeep)
    assert np.allclose(np.asarray(out["pos"]), w.pos, atol=1e-3)
    assert np.allclose(np.asarray(out["energy"]), w.energy, atol=1e-2)
    assert np.array_equal(np.asarray(out["food_on"]) > 0, w.task.food_on)
    assert np.array_equal(np.asarray(out["eaten"]).astype(int), w.task.eaten)


def test_gpu_arena_runs_births_and_resumes(tmp_path):
    cfg = small(engine="gpu", arena_stats_every=5.0, arena_min_alive=20)
    e = Ecology(cfg, seed=6, initial=30)
    assert e.gpu is not None
    e.world.energy[:5] = 400.0
    e.world.age[:5] = 100.0
    e.gpu.push_world()
    e.run(6.0)
    assert e.totals["nati"] >= 1 and e.history
    e.save(tmp_path / "gpu.npz")
    back = Ecology.load(small(engine="gpu", arena_stats_every=5.0, arena_min_alive=20), tmp_path / "gpu.npz")
    assert back.gpu is not None and np.array_equal(back.world.alive, e.world.alive)
    back.run(1.0)


# --- Il mondo che si complica (solo con il motore JAX) ----------------------------

def _scene(**dials):
    """Due esseri e un cibo, in un'arena piccola, con le manopole date: lo
    stato sulla GPU da cui provare un passo di fisica."""
    import jax.numpy as jnp

    from evonets.gpu import arena as ga
    from evonets.net.controllers import NeuralController

    cfg = small(net_type="rate", arena_food_rate=0.0, arena_aging_h0=0.0, **dials)
    w = ArenaWorld(cfg, 3, 3, seed=5)
    w.task.food_on[:] = False
    w.pos[:] = [[100.0, 100.0], [102.0, 100.0], [300.0, 300.0]]
    w.heading[:] = [0.0, np.pi, 0.0]
    w.energy[:] = 100.0
    w.grids_dirty = True
    gpu = ga.ArenaGpu(w, NeuralController(cfg, 3, 3), 1, plain_bodies(3))
    return cfg, w, gpu, ga, jnp


def _physics(gpu, ga, jnp, extra):
    s = gpu.s
    food = ga.neighbours(ga.table(s["food"], s["food_on"] > 0, gpu.geo), s["pos"], gpu.geo)
    beings = ga.neighbours(ga.table(s["pos"], s["alive"] > 0, gpu.geo), s["pos"], gpu.geo)
    return ga.physics(gpu.ph, gpu.P, s, jnp.zeros((3, 2)), jnp.asarray(extra, dtype=jnp.float32), food, beings)


def test_the_bigger_one_bites_the_smaller_one_in_front():
    cfg, w, gpu, ga, jnp = _scene(arena_predation=1.0)
    gpu.s["size"] = jnp.array([2.0, 1.0, 1.0])
    out = _physics(gpu, ga, jnp, [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    e = np.asarray(out["energy"])
    bite = cfg.arena_bite_rate * 2.0 * cfg.dt
    assert e[1] < 100.0 - 0.9 * bite and np.asarray(out["pain"])[1] == 1.0  # il piccolo è morso
    assert e[0] > e[2] - 0.5  # il grande guadagna (o quasi pareggia il costo del morso)
    gpu.s["size"] = jnp.array([1.0, 1.0, 1.0])  # alla pari: nessuno morde
    out = _physics(gpu, ga, jnp, [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    assert np.asarray(out["pain"])[:2].max() == 0.0


def test_bad_fruit_hurts_and_does_not_feed():
    cfg, w, gpu, ga, jnp = _scene()
    gpu.s["food"] = gpu.s["food"].at[0].set(jnp.array([100.5, 103.0]))  # più vicino al primo
    gpu.s["food_on"] = gpu.s["food_on"].at[0].set(1.0)
    gpu.s["food_bad"] = gpu.s["food_bad"].at[0].set(1.0)
    gpu.s["food_color"] = gpu.s["food_color"].at[0].set(1.0)
    out = _physics(gpu, ga, jnp, np.zeros((3, 4)))
    e = np.asarray(out["energy"])
    assert e.min() < 100.0 - cfg.bad_fruit_damage + 1.0 and np.asarray(out["pain"]).max() == 1.0
    assert float(out["meals"]) == 0.0 and float(out["bad_meals"]) == 1.0


def test_hard_food_needs_biting_and_big_food_needs_two():
    cfg, w, gpu, ga, jnp = _scene()
    gpu.s["food"] = gpu.s["food"].at[0].set(jnp.array([101.0, 103.0]))
    gpu.s["food_on"] = gpu.s["food_on"].at[0].set(1.0)
    gpu.s["food_kind"] = gpu.s["food_kind"].at[0].set(ga.HARD)
    gpu.s["food_r"] = gpu.s["food_r"].at[0].set(4.0)
    out = _physics(gpu, ga, jnp, np.zeros((3, 4)))
    assert float(out["food_on"][0]) == 1.0  # senza morsi resta lì
    gpu.s["work"] = gpu.s["work"].at[0].set(cfg.arena_hard_work)
    out = _physics(gpu, ga, jnp, [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert float(out["food_on"][0]) == 0.0 and float(out["energy"][0]) > 100.0 + cfg.arena_food_energy
    gpu.s["food_kind"] = gpu.s["food_kind"].at[0].set(ga.BIG)
    gpu.s["food_r"] = gpu.s["food_r"].at[0].set(6.0)
    out = _physics(gpu, ga, jnp, [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert float(out["food_on"][0]) == 1.0  # uno solo non basta
    out = _physics(gpu, ga, jnp, [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    share = cfg.arena_food_energy * cfg.arena_big_energy / 2
    assert float(out["food_on"][0]) == 0.0 and abs(float(out["energy"][1]) - (100.0 + share)) < 2.0


def test_a_parent_feeds_only_its_own_young_child():
    cfg, w, gpu, ga, jnp = _scene()
    gpu.s["parent"] = jnp.array([-1, 0, -1], dtype=jnp.int32)
    gpu.s["young"] = jnp.array([0.0, 1.0, 0.0])
    out = _physics(gpu, ga, jnp, [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    e = np.asarray(out["energy"])
    given = cfg.arena_feed_rate * cfg.dt
    assert e[1] > e[2] + 0.7 * given and e[0] < e[2] - 0.9 * given
    gpu.s["parent"] = jnp.array([-1, 2, -1], dtype=jnp.int32)  # non è suo figlio
    out = _physics(gpu, ga, jnp, [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert abs(float(out["energy"][1]) - float(out["energy"][2])) < 1e-3


def test_night_dims_the_world_but_not_the_signals():
    from evonets.gpu import arena as ga
    import jax.numpy as jnp

    cfg = small(arena_night=1.0)
    ph = {k: jnp.asarray(v) for k, v in ga.arena_phys(cfg).items()}
    assert abs(float(ga.daylight(ph, 0.0)) - 1.0) < 1e-6
    assert abs(float(ga.daylight(ph, cfg.arena_day / 2)) - 0.15) < 1e-4


def test_the_world_ladder_raises_one_dial_when_calm_and_lowers_it_when_troubled():
    cfg = small(engine="gpu", arena_min_alive=5, arena_dial_calm=2)
    e = Ecology(cfg, seed=7, initial=20)
    assert e.gpu is not None
    e.history = [{"colonists": 0, "alive": 20}] * 2
    e._world_ladder()
    assert cfg.arena_depletion == cfg.arena_dial_step and cfg.arena_fruits == 0.0
    assert e.log[-1]["event"] == "mondo"
    e._dial_at = -10
    e.history = [{"colonists": 3, "alive": 5}] * 2
    e._world_ladder()
    assert cfg.arena_depletion == 0.0


def test_eating_wears_out_the_patch():
    cfg = small(engine="gpu", arena_depletion=1.0, arena_world_ladder=False)
    e = Ecology(cfg, seed=8, initial=5)
    task = e.world.task
    patches = task.patches
    task.food[:10] = patches.centre[0]
    task.food_on[:10] = True
    e._food_seen = task.food_on.copy()
    task.food_on[:10] = False  # mangiati
    before = patches.health[0]
    e._wear(0.0)
    assert patches.health[0] <= before - 10 * cfg.arena_patch_wear + 1e-9
