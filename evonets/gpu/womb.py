"""Il grembo sulla GPU (vedi net/womb.py): la stessa rete e la stessa
plasticità del passo di vita (`engine.rate_brain`, `engine.spike_brain`), su
tutta la popolazione, con i soli neonati in gara (le forme restano fisse e la
funzione non si ricompila a ogni generazione)."""

import jax
import jax.numpy as jnp
import numpy as np

from . import engine

_compiled: dict = {}

KEYS = ("links", "target_on", "rule", "kmod", "mod_sign", "learns", "hebb", "sign", "intrinsic", "early_keep",
        "homeo", "target_act")


def _fn(kind: str, n_hidden: int, substeps: int):
    key = (kind, n_hidden, substeps)
    if key not in _compiled:
        def womb(ph, P, s, obs, racing):
            def body(s, xs):
                k, o = xs
                tau = jnp.minimum(ph["homeo_tau"], (k + 1.0) * ph["dt"])  # ā = media di tutto il grembo finora
                p = {**ph, "homeo_tau": tau}
                if kind == "rate":
                    new, _ = engine.rate_brain(p, P, s, o, racing, n_hidden)
                else:
                    new, _ = engine.spike_brain(p, P, s, o, racing, n_hidden, substeps)
                return {**s, **new}, None

            steps = jnp.arange(obs.shape[0], dtype=jnp.float32)
            return jax.lax.scan(body, s, (steps, obs))[0]

        _compiled[key] = jax.jit(womb)
    return _compiled[key]


def run(b, rows: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, ...]:
    f32 = lambda a: jnp.asarray(np.asarray(a, dtype=np.float32))
    n, T = b.n, b.n_hidden + b.n_out
    P = {k: f32(getattr(b, k)) for k in KEYS if k != "mod_sign"}
    P["mod_sign"] = f32(b._mod_sign)
    P["eta"] = jnp.zeros((n, T), jnp.float32)  # niente ricompensa: la plasticità modulata tace
    P["active"] = jnp.ones((n, b.n_in), jnp.float32)  # solo la forma: quanti sono i sensi
    if b.KIND == "spike-ei":
        P.update(keep_n=f32(b.keep), active_n=f32(b.active))
    racing = np.zeros(n, dtype=np.float32)
    racing[rows] = 1.0
    s = dict(state=jnp.zeros((n, T)), w=f32(b.w_eff), early=jnp.zeros(b.w_eff.shape), bias_now=f32(b.bias_now),
             avg=jnp.zeros((n, T)))
    if b.KIND == "spike-ei":
        s.update(charge=jnp.zeros((n, T)), spikes=jnp.zeros((n, b.n_hidden)))
    ph = {k: jnp.float32(v) for k, v in engine.phys_of(b.cfg).items()}
    out = _fn(b.KIND, b.n_hidden, b.cfg.spike_substeps)(ph, P, s, f32(obs), jnp.asarray(racing))
    take = lambda k: np.asarray(out[k], dtype=float)[rows]
    return take("w"), take("avg"), take("bias_now")
