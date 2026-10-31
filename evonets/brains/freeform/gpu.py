"""Le reti a forma libera sulla GPU (JAX): fanno ciò che fanno `Brains.step` e
`SpikingBrains.step`. Sono funzioni pure, usate dai motori delle arene e dal grembo; qui stanno anche
le costanti della configurazione che passano al passo compilato."""

import jax.numpy as jnp
import numpy as np

from .plasticity import plasticity

F = jnp.float32

# Costanti della configurazione usate dal passo: passano come numeri (non
# compilati dentro), così cambiarle dal vivo non richiede di ricompilare.
PHYS = (
    "arena_w", "arena_h", "dt", "agent_radius", "thrust_accel", "reverse_factor", "lin_drag", "lat_drag",
    "turn_accel", "ang_drag", "wall_restitution", "energy_init", "energy_max", "reserve_mass",
    "metabolism_basal", "metabolism_move",
    "metabolism_turn", "metabolism_exponent", "light_attenuation", "hazard_drain", "hazard_falloff", "hazard_pulse", "hazard_speed",
    "toxin_molecule", "hazard_smell", "target_molecule", "target_smell", "smell_range", "vibration_range",
    "target_vibration", "target_vibration_freq", "hazard_vibration", "hazard_vibration_freq",
    "ambient_temperature", "heat_range", "target_heat", "hazard_heat", "receptor_dark", "receptor_saturation",
    "sensor_noise", "synapse_max", "homeo_tau", "capture_rate", "surprise", "surprise_horizon", "surprise_smooth",
    "critic_rate", "taste_tau", "spike_gain", "life_max",
    # Il mondo del cibo.
    "prey_flee_range", "predator_range", "predator_radius", "predator_damage", "food_energy", "food_min_dist",
    "predator_vibration", "predator_vibration_freq",
    "food_molecule", "food_smell", "predator_molecule", "predator_smell", "odor_range", "fruit_molecule",
    "bad_fruit_damage",
)


def phys_of(cfg) -> dict:
    p = {k: np.float32(getattr(cfg, k)) for k in PHYS}
    a = np.radians(cfg.magnetic_angle_deg)
    p["magnetic"] = (cfg.magnetic_field * np.array([np.cos(a), np.sin(a)])).astype(np.float32)
    return p


def _tuning(q, mu, sigma):
    return jnp.exp(-0.5 * ((q - mu) / sigma) ** 2)


def _rotate(vx, vy, c, s):
    return vx * c - vy * s, vx * s + vy * c


def body(ph, s):
    """Come `signals.body`: (N, 9) grandezze del corpo in [0, 1]."""
    c, sn = jnp.cos(s["heading"]), jnp.sin(s["heading"])
    vx, vy = s["vel"][:, 0], s["vel"][:, 1]
    v_max = ph["thrust_accel"] / ph["lin_drag"]
    w_max = ph["turn_accel"] / ph["ang_drag"]
    fwd, lat, rot = (vx * c + vy * sn) / v_max, (vy * c - vx * sn) / v_max, s["ang_vel"] / w_max

    def pos(x):
        return jnp.clip(x, 0.0, 1.0)

    return jnp.stack([jnp.clip(s["energy"] / ph["energy_max"], 0.0, 1.0), s["taste"], s["pain"],
                      pos(fwd), pos(-fwd), pos(lat), pos(-lat), pos(rot), pos(-rot)], 1)


# --- Reti ------------------------------------------------------------------


def learn(ph, P, s, pre, post, after_hidden, racing, centred):
    """Come `Brains._learn` (la stessa `plasticity.plasticity`), solo per chi
    impara ed è in gara. Restituisce lo stato della plasticità aggiornato."""
    M = jnp.einsum("nth,nh->nt", P["kmod"], after_hidden * P["mod_sign"])
    p = {**P, "w_eff": s["w"], "early": s["early"], "elig": s["elig"], "critic": s["critic"], "bias_now": s["bias_now"], "avg": s["avg"],
         "clock": s["clock"], "signal": jnp.stack([s["taste"], s["pain"]], 1)}
    w, early, bias, avg, elig, critic = plasticity(ph, jnp, p, pre, post, M, P["active"].shape[1], centred)
    on = (P["learns"] * racing)[:, None] > 0
    return dict(w=jnp.where(on[:, :, None], w, s["w"]), early=jnp.where(on[:, :, None], early, s["early"]),
                elig=jnp.where(on[:, :, None], elig, s["elig"]), critic=jnp.where(on, critic, s["critic"]),
                bias_now=jnp.where(on, bias, s["bias_now"]), avg=jnp.where(on, avg, s["avg"]),
                clock=s["clock"] + ph["dt"] * racing)  # l'età avanza per chi è in gara


def rate_brain(ph, P, s, obs, racing, n_hidden):
    """Come `Brains.step`: rete continua più plasticità."""
    before = s["state"]
    x = jnp.concatenate([obs, before], 1)
    after = jnp.tanh(jnp.einsum("nts,ns->nt", s["w"], x) + s["bias_now"]) * P["target_on"]
    on = racing[:, None] > 0
    after = jnp.where(on, after, before)
    return dict(state=after, **learn(ph, P, s, x, after, after[:, :n_hidden], racing, True)), after[:, n_hidden:]


def spike_brain(ph, P, s, obs, racing, n_hidden, substeps):
    """Come `SpikingBrains.step`: neuroni a impulsi più plasticità."""
    h = n_hidden
    keep, drive = P["keep_n"], 1.0 - P["keep_n"]
    act = P["active_n"]
    charge, spikes, state = s["charge"], s["spikes"], s["state"]
    for _ in range(substeps):
        x = jnp.concatenate([obs, spikes, state[:, h:]], 1)
        current = jnp.einsum("nts,ns->nt", s["w"], x) + s["bias_now"]
        current = current.at[:, :h].multiply(ph["spike_gain"])
        charge = jnp.maximum(keep * charge + drive * current, -1.0)
        inner = charge[:, :h]
        fired = (inner >= 1.0) & (act > 0)
        inner = jnp.where(fired | (act <= 0), 0.0, inner)
        charge = charge.at[:, :h].set(inner)
        spikes = fired.astype(F)
        state = state.at[:, :h].set(0.8 * state[:, :h] + 0.2 * spikes)
        state = state.at[:, h:].set(jnp.tanh(jnp.maximum(charge[:, h:], 0.0)))
    on = racing[:, None] > 0
    charge = jnp.where(on, charge, s["charge"])
    spikes = jnp.where(on, spikes, s["spikes"])
    state = jnp.where(on, state, s["state"])
    pre = jnp.concatenate([obs, state], 1)
    plastic = learn(ph, P, s, pre, state, state[:, :h], racing, False)
    m = state[:, h:]
    out = jnp.stack([m[:, 0] - m[:, 1], m[:, 2] - m[:, 3]], 1)
    return dict(charge=charge, spikes=spikes, state=state, **plastic), out
