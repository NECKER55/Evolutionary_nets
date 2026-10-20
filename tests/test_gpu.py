"""Il motore GPU (JAX) deve fare ciò che fa il motore NumPy: si confrontano le
parti una per una partendo dallo stesso stato, con la tolleranza dei 32 bit."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evonets import Config, World
from evonets.gpu import engine
from evonets.gpu.sim import GpuWorlds, pack_params, pack_state
from evonets.net import NeuralController
from evonets.sim import signals
from evonets.sim.tasks import Foraging, MovingTarget
from nets import rich

ALL_ON = dict(target_vibration=0.7, hazard_vibration=0.4, hazard_smell=0.3, target_smell=0.5, target_heat=0.4,
              hazard_heat=0.2, magnetic_field=0.8)


def setup(net="rate", n=12, seed=0, task=None, **kw):
    cfg = Config(net_type=net, n_hazards=6, **kw)
    w = World(cfg, n, seed=seed, task=task)
    ctrl = NeuralController(cfg, n, seed=seed)
    b = rich(n, seed=seed, net=net, R=8, H=10, conn=0.35, **kw)
    ctrl.brains.set_genome(b.genome)
    ctrl.reset(seed)
    w.vel[:] = np.random.default_rng(seed).normal(0, 8, (n, 2))
    w.pos[0] = (cfg.agent_radius, 40.0)  # uno appoggiato alla parete
    w.bump[0] = True
    return cfg, w, ctrl


def device(cfg, w, ctrl):
    ph = {k: jnp.asarray(v) for k, v in engine.phys_of(cfg).items()}
    P = {k: jnp.asarray(v) for k, v in pack_params(w, ctrl).items()}
    s = pack_state(w, ctrl, jax.random.PRNGKey(0))
    s = {k: jnp.asarray(v) if k != "key" else v for k, v in s.items()}
    return ph, P, s


def test_quanta_match_the_numpy_receptors():
    cfg, w, ctrl = setup(**ALL_ON)
    ph, P, s = device(cfg, w, ctrl)
    ref = ctrl.brains.prepared().quanta(w, np.arange(w.n))
    got = np.asarray(engine.quanta(ph, P, s))
    assert np.allclose(got, ref, rtol=2e-4, atol=1e-3)


@pytest.mark.parametrize("net", ["rate", "spike-ei"])
def test_brain_and_plasticity_match(net):
    cfg, w, ctrl = setup(net)
    b = ctrl.brains
    assert b.learns.any()
    ph, P, s = device(cfg, w, ctrl)
    rng = np.random.default_rng(1)
    racing = jnp.ones(w.n, jnp.float32)
    for _ in range(40):
        obs = rng.uniform(0, 1, (w.n, b.n_in))
        ref = b.step(obs)
        if net == "rate":
            new, out = engine.rate_brain(ph, P, s, jnp.asarray(obs, jnp.float32), racing, b.n_hidden)
        else:
            new, out = engine.spike_brain(ph, P, s, jnp.asarray(obs, jnp.float32), racing, b.n_hidden,
                                          cfg.spike_substeps)
        s = {**s, **new}
        assert np.allclose(np.asarray(out), ref, atol=2e-4)
    assert np.allclose(np.asarray(s["w"]), b.w_eff, atol=2e-4)


def test_physics_matches_the_world():
    cfg, w, ctrl = setup(task=MovingTarget(5.0))
    ph, P, s = device(cfg, w, ctrl)
    rng = np.random.default_rng(2)
    upkeep = ctrl.upkeep
    for _ in range(200):
        a = rng.uniform(-1, 1, (w.n, 2))
        w.step(a, upkeep)
        s = {**s, **engine.physics(ph, P, s, jnp.asarray(a, jnp.float32), P["upkeep"])}
    for k, ref in (("pos", w.pos), ("energy", w.energy), ("heading", w.heading), ("target", w.task.target)):
        assert np.allclose(np.asarray(s[k]), ref, atol=2e-3), k
    assert np.array_equal(np.asarray(s["alive"]) > 0, w.alive) and np.array_equal(np.asarray(s["done"]) > 0, w.done)


@pytest.mark.parametrize("net", ["rate", "spike-ei"])
def test_whole_steps_match_without_noise(net, monkeypatch):
    """Senza rumore (lo stesso numero casuale non si può riprodurre tra i due
    motori) passi interi identici entro la tolleranza."""
    cfg, w, ctrl = setup(net, n=16, receptor_dark=0.0)
    monkeypatch.setattr(engine.jax.random, "normal", lambda key, shape, dtype=None: jnp.zeros(shape, jnp.float32))

    class Zeros:
        def standard_normal(self, shape):
            return np.zeros(shape)

    gw = GpuWorlds(cfg, [(w, ctrl)], [0])
    GpuWorlds._compiled.clear()
    w2, c2 = World(cfg, w.n, seed=0), NeuralController(cfg, w.n)
    w2.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w.__dict__.items()})
    c2.brains.set_genome(ctrl.brains.genome)
    c2.reset(0)
    c2.noise_rng = Zeros()
    for _ in range(60):
        w2.step(c2.act(w2), c2.upkeep)
    gw.advance(60)
    GpuWorlds._compiled.clear()
    assert np.allclose(w.pos, w2.pos, atol=5e-3) and np.allclose(w.energy, w2.energy, atol=5e-3)
    assert np.allclose(ctrl.obs, c2.obs, atol=5e-3)


def test_finished_worlds_stand_still_while_others_go_on():
    cfg = Config(metabolism_basal=40.0, n_hazards=0)  # si muore in circa 4 s
    pairs = []
    for k in range(3):
        w, c = World(cfg, 10, seed=k), NeuralController(cfg, 10, seed=k)
        pairs.append((w, c))
    pairs[1][0].energy[:] = 10.0  # il secondo mondo finisce prima
    gw = GpuWorlds(cfg, pairs, [0, 1, 2])
    gw.run()
    t = [w.t for w, _ in pairs]
    assert t[1] < t[0] and all(not (w.alive & ~w.done).any() for w, _ in pairs)


# --- Il mondo del cibo (docs/SPEC_MONDO.md) ---------------------------------

FORAGE_ALL = dict(task="cibo", forage_abundance=0.5, forage_need=1.0, forage_prey=1.0, forage_predators=1.0,
                  forage_scent=1.0)


def forage_setup(net="rate", n=12, **kw):
    cfg, w, ctrl = setup(net, n=n, task=Foraging(), **kw)
    cfg = w.cfg = Config(net_type=net, n_hazards=6, **{**FORAGE_ALL, **kw})
    w.reset()  # il compito prende le manopole dalla configurazione
    w.vel[:] = np.random.default_rng(0).normal(0, 8, (n, 2))
    t = w.task
    rng = np.random.default_rng(5)
    # Cibo e predatori sparsi vicino agli agenti, diversi in ogni copia.
    t.food = np.clip(w.pos[:, None, :] + rng.normal(0, 12, t.food.shape), 3.0, 137.0)
    t.pred = np.clip(w.pos[:, None, :] + rng.normal(0, 15, t.pred.shape), 3.0, 137.0)
    # Nubi a metà: cibi appena comparsi e tracce di cibi appena mangiati.
    t.age = rng.uniform(0.0, 4.0, t.age.shape)
    t.ghost = np.clip(w.pos[:, None, :] + rng.normal(0, 12, t.ghost.shape), 3.0, 137.0)
    t.ghost_t = rng.uniform(0.0, 4.0, t.ghost_t.shape)
    return cfg, w, ctrl


def test_foraging_quanta_match_the_numpy_receptors():
    cfg, w, ctrl = forage_setup(**ALL_ON)
    ph, P, s = device(cfg, w, ctrl)
    ref = ctrl.brains.prepared().quanta(w, np.arange(w.n))
    got = np.asarray(engine.quanta(ph, P, s))
    assert np.allclose(got, ref, rtol=2e-4, atol=1e-3)
    seen = signals.light(cfg, w, w.pos, np.zeros((w.n, 1)), np.arange(w.n))
    assert seen["cibo"].any() or seen["predatore"].any()  # la prova vede davvero cibo e predatori


def test_foraging_physics_matches_the_world():
    cfg, w, ctrl = forage_setup(n=24, energy_max=450.0, reserve_mass=0.5)  # anche le riserve e il loro peso
    w.energy[:] = np.linspace(20.0, 440.0, w.n)
    ph, P, s = device(cfg, w, ctrl)
    rng = np.random.default_rng(2)
    for _ in range(300):
        a = rng.uniform(-1, 1, (w.n, 2))
        w.step(a, ctrl.upkeep)
        s = {**s, **engine.physics(ph, P, s, jnp.asarray(a, jnp.float32), P["upkeep"])}
    t = w.task
    assert t.eaten.sum() > 0 and (w.energy < cfg.energy_init).any() and (w.energy > cfg.energy_init).any()
    for k, ref in (("pos", w.pos), ("energy", w.energy), ("food", t.food), ("pred", t.pred), ("age", t.age),
                   ("ghost", t.ghost), ("ghost_t", t.ghost_t)):
        assert np.allclose(np.asarray(s[k]), ref, atol=2e-3), k
    assert np.array_equal(np.asarray(s["eaten"]), t.eaten) and np.array_equal(np.asarray(s["ptr"]), t.ptr)
    assert np.array_equal(np.asarray(s["alive"]) > 0, w.alive) and np.array_equal(np.asarray(s["done"]) > 0, w.done)


@pytest.mark.parametrize("net", ["rate", "spike-ei"])
def test_foraging_whole_steps_match_without_noise(net, monkeypatch):
    cfg, w, ctrl = forage_setup(net, n=16, receptor_dark=0.0)
    monkeypatch.setattr(engine.jax.random, "normal", lambda key, shape, dtype=None: jnp.zeros(shape, jnp.float32))

    class Zeros:
        def standard_normal(self, shape):
            return np.zeros(shape)

    w2, c2 = World(cfg, w.n, seed=0), NeuralController(cfg, w.n)
    w2.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w.__dict__.items()})
    w2.task = copy.deepcopy(w.task)
    gw = GpuWorlds(cfg, [(w, ctrl)], [0])
    GpuWorlds._compiled.clear()
    c2.brains.set_genome(ctrl.brains.genome)
    c2.reset(0)
    c2.noise_rng = Zeros()
    for _ in range(60):
        w2.step(c2.act(w2), c2.upkeep)
    gw.advance(60)
    GpuWorlds._compiled.clear()
    assert np.allclose(w.pos, w2.pos, atol=5e-3) and np.allclose(w.energy, w2.energy, atol=5e-3)
    assert np.allclose(ctrl.obs, c2.obs, atol=5e-3)
    assert np.allclose(w.task.food, w2.task.food, atol=5e-3) and np.array_equal(w.task.eaten, w2.task.eaten)


def test_a_foraging_generation_runs_to_the_end_on_the_gpu():
    cfg = Config(task="cibo", n_hazards=0, forage_predators=1.0)
    pairs = [(World(cfg, 20, seed=k), NeuralController(cfg, 20, seed=k)) for k in range(2)]
    GpuWorlds(cfg, pairs, [0, 1]).run()
    for w, _ in pairs:
        assert not (w.alive & ~w.done).any() or w.t >= cfg.life_max
        assert np.array_equal(w.done, w.task.eaten >= w.task.need)
