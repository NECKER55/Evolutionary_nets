"""Più mondi simulati insieme sulla GPU (o, senza scheda NVIDIA, sulla CPU
con JAX).

`GpuWorlds` prende coppie (World, NeuralController) del motore NumPy, ne
copia lo stato sulla GPU, fa girare passi o generazioni intere e riporta il
risultato negli stessi oggetti, così il resto del programma (evoluzione,
statistiche, interfaccia, salvataggi) non cambia.

Tutti i mondi di un lotto devono avere la stessa forma: stesso numero di
agenti, di recettori, di neuroni, di zone di veleno e lo stesso tipo di rete.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..sim.tasks import FarTarget, Foraging, MovingTarget, ReachTarget
from . import engine
from .engine import F

SUPPORTED_TASKS = (ReachTarget, FarTarget, MovingTarget, Foraging)


def supported(world, ctrl) -> bool:
    """Il motore GPU sa simulare questo mondo?"""
    return type(world.task) in SUPPORTED_TASKS and type(ctrl).__name__ == "NeuralController"


def _f(a):
    return np.asarray(a, dtype=np.float32)


def pack_params(world, ctrl) -> dict:
    """Le costanti di una generazione per un mondo: geni compilati e arena."""
    b = ctrl.brains
    p = b.prepared()
    hz = world.hazards
    task = world.task
    has_target = hasattr(task, "target")
    params = dict(
        angles=p.angles, body=p.body, active=p.active, centre=p.centre, half=p.half,
        c_wall=p.c_wall, c_target=p.c_target, c_glow=p.c_glow, mu_c=p.mu_c, sg_c=p.sg_c, g_c=p.g_c,
        c_press=p.c_press, mu_m=p.mu_m, sg_m=p.sg_m, g_m=p.g_m, mu_t=p.mu_t, sg_t=p.sg_t, g_t=p.g_t,
        g_mag=p.g_mag, c_body=p.c_body, keep=p.keep, sub=p.sub, div=p.div,
        links=b.links, target_on=b.target_on, eta=b.eta, rule=b.rule, kmod=b.kmod,
        mod_sign=b._mod_sign, learns=b.learns, upkeep=b.upkeep, hebb=b.hebb, sign=b.sign, intrinsic=b.intrinsic, boost=b.boost, closure=b.closure, early_keep=b.early_keep,
        homeo=b.homeo, target_act=b.target_act,
        anchor=hz.anchor.reshape(-1, 2), amp=hz.amp.reshape(-1, 2), omega=hz.omega.reshape(-1, 2),
        pulse_omega=hz.pulse_omega, base_radius=hz.base_radius,
        target_r=task.radius if has_target else 0.0, has_target=float(has_target),
        keep_living=float(world.keep_living),
    )
    if b.KIND == "spike-ei":
        params.update(keep_n=b.keep, active_n=b.active)
    if isinstance(task, Foraging):  # le manopole di questo mondo
        cfg = world.cfg
        params.update(c_food=p.c_food, c_pred=p.c_pred, food_rad=np.where(task.food_on, task.food_r, 0.0),
                      pred_rad=np.where(task.pred_on, cfg.predator_radius, 0.0), food_r=task.food_r,
                      need=task.need, prey_speed=task.prey_speed, pred_speed=task.pred_speed, queue=task.queue,
                      scent_tau=task.scent_tau)
    return {k: _f(v) for k, v in params.items()}


def pack_state(world, ctrl, key) -> dict:
    b = ctrl.brains
    hz = world.hazards
    task = world.task
    target = task.target if hasattr(task, "target") else np.zeros(2)
    tvel = getattr(task, "vel", np.zeros(2))
    state = dict(
        pos=world.pos, vel=world.vel, heading=world.heading, ang_vel=world.ang_vel, energy=world.energy,
        alive=world.alive, done=world.done, bump=world.bump, pain=world.pain,
        phase=hz.phase.reshape(-1, 2), pulse_phase=hz.pulse_phase, target=target, target_vel=tvel,
        t=world.t, state=b.state, w=b.w_eff, avg=b.avg, early=b.early, bias_now=b.bias_now, clock=b.clock, done_t=np.where(world.done, world.t, np.inf),
        obs=ctrl.obs if ctrl.obs.shape == (b.n, b.n_in) else np.zeros((b.n, b.n_in)),
        m=ctrl.adapt if ctrl.adapt.shape == (b.n, b.n_in) else np.full((b.n, b.n_in), np.nan),
    )
    if b.KIND == "spike-ei":
        state.update(charge=b.charge, spikes=b.spikes)
    if isinstance(task, Foraging):
        state.update(food=task.food, pred=task.pred, ptr=task.ptr, eaten=task.eaten, age=task.age,
                     ghost=task.ghost, ghost_t=task.ghost_t)
    state = {k: _f(v) for k, v in state.items()}
    state["key"] = key
    return state


def unpack(world, ctrl, s: dict) -> None:
    """Riporta lo stato di un mondo dalla GPU agli oggetti NumPy."""
    b = ctrl.brains
    g = {k: np.asarray(v, dtype=float) for k, v in s.items() if k != "key"}
    world.pos, world.vel, world.heading, world.ang_vel = g["pos"], g["vel"], g["heading"], g["ang_vel"]
    world.energy, world.pain, world.t = g["energy"], g["pain"], float(g["t"])
    world.alive, world.done, world.bump = g["alive"] > 0, g["done"] > 0, g["bump"] > 0
    world.done_t = g["done_t"]
    hz = world.hazards
    hz.phase, hz.pulse_phase = g["phase"].reshape(hz.phase.shape), g["pulse_phase"]
    hz._update()
    if hasattr(world.task, "target"):
        world.task.target = g["target"]
        if hasattr(world.task, "vel"):
            world.task.vel = g["target_vel"]
    if "food" in g:
        t = world.task
        t.food, t.pred = g["food"], g["pred"]
        t.ptr, t.eaten = g["ptr"].astype(int), g["eaten"].astype(int)
        t.age, t.ghost, t.ghost_t = g["age"], g["ghost"], g["ghost_t"]
    b.state, b.avg, ctrl.obs, ctrl.adapt = g["state"], g["avg"], g["obs"], g["m"]
    b.early, b.bias_now, b.clock = g["early"], g["bias_now"], g["clock"]
    if b.KIND == "spike-ei":
        b.charge, b.spikes = g["charge"], g["spikes"]
    if not np.array_equal(g["w"], b.w_eff):
        b.w_eff, b.learned, b._subset = g["w"], True, None


class GpuWorlds:
    """B mondi (World + NeuralController) simulati insieme sulla GPU."""

    _compiled: dict = {}  # funzioni compilate, per forma e tipo di rete

    def __init__(self, cfg, pairs: list, seeds: list[int]):
        self.cfg, self.pairs = cfg, pairs
        b = pairs[0][1].brains
        self.net, self.h = b.KIND, b.n_hidden
        keys = jnp.stack([jax.random.PRNGKey(int(s) % (2**31)) for s in seeds])
        self.params = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *[pack_params(w, c) for w, c in pairs])
        self.state = jax.tree_util.tree_map(lambda *a: jnp.stack(a),
                                            *[pack_state(w, c, k) for (w, c), k in zip(pairs, keys)])
        self.n = pairs[0][0].n
        self.forage = "food" in self.state

    def _fns(self):
        key = (self.net, self.h, self.cfg.spike_substeps, self.forage)
        if key not in GpuWorlds._compiled:
            step = engine.make_step(self.net, self.h, self.cfg.spike_substeps)
            vstep = jax.vmap(step, in_axes=(None, 0, 0, None))
            vrun = jax.vmap(engine.running, in_axes=(None, 0, 0))

            def advance(ph, P, s, manual, k):
                """k passi (o meno, se tutti i mondi finiscono): i mondi finiti
                restano fermi."""
                def cond(c):
                    i, s = c
                    return (i < k) & vrun(ph, P, s).any()

                def body(c):
                    i, s = c
                    live = vrun(ph, P, s)
                    new = vstep(ph, P, s, manual)
                    s = jax.tree_util.tree_map(
                        lambda a, b: jnp.where(live.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), new, s)
                    return i + 1, s

                return jax.lax.while_loop(cond, body, (0, s))[1]

            GpuWorlds._compiled[key] = jax.jit(advance)
        return GpuWorlds._compiled[key]

    def advance(self, k: int, manual=None) -> None:
        """Fa avanzare tutti i mondi di k passi (si fermano da soli a fine
        generazione) e riporta il risultato negli oggetti NumPy."""
        ph = {kk: jnp.asarray(v) for kk, v in engine.phys_of(self.cfg).items()}
        if manual is None:
            manual = (jnp.int32(0), jnp.zeros(2, F), jnp.float32(0.0))
        else:
            idx, cmd = manual
            manual = (jnp.int32(idx), jnp.asarray(cmd, F), jnp.float32(1.0))
        self.state = self._fns()(ph, self.params, self.state, manual, jnp.int32(k))
        self.sync()

    def run(self) -> None:
        """Fino alla fine della generazione in tutti i mondi."""
        self.advance(2**30)

    def sync(self) -> None:
        host = jax.device_get(self.state)
        for b, (world, ctrl) in enumerate(self.pairs):
            unpack(world, ctrl, {k: v[b] for k, v in host.items()})
