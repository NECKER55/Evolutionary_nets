"""I cervelli dell'arena delle pile sulla GPU (JAX): lo stesso algoritmo di `sections.py`,
`actor_critic.py` e `guided.py`, scritto come funzioni pure. Lo stato è un dizionario di array con
gli esseri in fila; la corsa compilata che li usa sta in `arenas/pile/gpu.py`."""


import jax
import jax.numpy as jnp
import numpy as np

from ...arenas.pile import world as P
from .. import pile as S

F = jnp.float32


ATT_KEYS = ("att_target", "att_inhib", "att_inhib_t", "att_prev_vis", "att_novelty", "att_new", "att_prob", "att_has",
            "att_parts")


def init_brain(agent: str, N: int, dt: float, seed: int, cfg: P.PileConfig, genomes=None,
               sizes: S.Sizes | None = None) -> tuple[dict, dict]:
    """Lo stato di partenza, dalla versione NumPy (stessa partenza), e dove
    stanno nel genoma i pesi e le connessioni di partenza (`Agent.blocks`)."""
    kw = S.brain_kwargs(agent, cfg, sizes.lesions if sizes else ())
    if agent == "sezioni" and sizes is not None:
        kw["sizes"] = sizes
    b = S.AGENTS[agent](N, dt, seed, genomes=genomes, **kw)
    at = b.att
    out = {"genome": b.genome, "clock": b.clock, "program": b.program.astype(np.int32), "pi": b.pi,
           "bite": b.bite, "p_bite": b.p_bite,
           "att_target": at.target.astype(np.int32), "att_inhib": at.inhib.astype(np.int32),
           "att_prob": at.prob, "att_has": at.has_pos, "att_parts": at.parts, "att_changed": at.changed,
           "att_inhib_t": at.inhib_t, "att_prev_vis": at.prev_vis, "att_novelty": at.novelty, "att_new": at.new}
    if agent == "guidato":
        return out, b.blocks
    out.update(a_slow=b.a.slow, w_e=b.w_e, e_e=b.e_e, w_v=b.w_v, e_v=b.e_v, v_prev=b.v_prev, value=b.value,
               trust=b.trust,
               fresh=b.fresh,
               delta=b.delta, teach=b.teach)
    if agent == "sezioni":
        out.update(proto=b.proto, pred=b.pred, win_rate=b.win_rate, state_b=b.state_b, prev_b=b.prev_b, mem=b.mem,
                   w_gate=b.w_gate, e_gate=b.e_gate, c_in=b.c_in, u_c=b.u_c, e_c=b.e_c, u_l=b.u_l, code=b.code,
                   d_fast=b.d_fast, d_slow=b.d_slow, t_slow=b.t_slow, unc=b.unc, counts=np.zeros((N, b.nb, 4)),
                   teacher=b.teacher, obs_err=b.obs_err, o_pl=b.o_pl, o_pa=b.o_pa, o_assoc=b.o_assoc, o_e=b.o_e,
                   ep_on=b.ep_on, ep_pl=b.ep_pl, ep_pa=b.ep_pa, ep_e=b.ep_e, o_err=b.o_err, o_out=b.o_out,
                   k_on=b.k_on.astype(np.int32), k_ev=b.k_ev, k_used=b.k_used, k_usual=b.k_usual, k_since=b.k_since,
                   k_stale=b.k_stale, o_prior=b.o_prior)
    return out, b.blocks


def decode(genome):
    return S.decode(genome, xp=jnp)


def _attend(br, objs, decide, p, dt, key, extra=None, learned=None, o=None):
    """L'attenzione: come `sections.Attention.update` (la mappa di priorità)."""
    vis = objs["vis"]
    n, m = vis.shape
    idx = jnp.arange(n)
    before = br["att_target"]
    br["att_novelty"] = br["att_novelty"] * jnp.exp(-dt / p["novelty_tau"])[:, None] + (vis & ~br["att_prev_vis"])
    br["att_prev_vis"] = vis
    br["att_inhib_t"] = jnp.maximum(br["att_inhib_t"] - dt, 0.0)
    target = br["att_target"]
    lost = (target >= 0) & ~vis[idx, jnp.maximum(target, 0)]
    target = jnp.where(lost, -1, target)
    br["att_new"] = br["att_new"] * jnp.exp(-dt / p["new_tau"])
    has, obj, feat, near = S.priority_map(objs, S.MAP, xp=jnp)
    br["att_has"] = has
    choose = (decide | (target < 0)) & has.any(1)
    nov = jnp.take_along_axis(br["att_novelty"], obj, axis=1)
    innate = (p["sal_near"][:, None] * near + p["sal_novelty"][:, None] * nov
              + p["sal_stick"][:, None] * (obj == target[:, None])
              - p["sal_inhibit"][:, None] * ((obj == br["att_inhib"][:, None]) & (br["att_inhib_t"] > 0)[:, None]))
    value = imp = unc = jnp.zeros_like(innate)
    if o is not None:
        value, imp, unc = S.o_read(o[0], o[1], o[2], feat, o[3], p, xp=jnp)
        trust = jnp.ones(n) if learned is None else learned
        value = trust[:, None] * p["sal_value"][:, None] * value
        imp = trust[:, None] * p["sal_import"][:, None] * imp
        unc = p["sal_unc"][:, None] * unc
    br["att_parts"] = jnp.stack([innate, value, imp, unc], -1) * has[:, :, None]
    score = innate + value + imp + unc
    if extra is not None:
        score = score + jnp.take_along_axis(extra, obj, axis=1)
    z = jnp.where(has, score / p["sal_temperature"][:, None], -1e9)
    prob = jax.nn.softmax(z, axis=1) * has
    prob = prob / jnp.maximum(prob.sum(1, keepdims=True), 1e-12)
    br["att_prob"] = prob
    place = jax.random.categorical(key, z, axis=1)
    sel = jnp.where(choose, obj[idx, place], target).astype(jnp.int32)
    changed = choose & (sel != target)
    br["att_new"] = jnp.where(changed, 1.0, br["att_new"])
    br["att_target"] = sel
    br["att_changed"] = sel != before


def seen_of(br, objs):
    idx, t = jnp.arange(br["att_target"].shape[0]), jnp.maximum(br["att_target"], 0)
    return br["att_target"] >= 0, objs["feat"][idx, t], objs["near"][idx, t], objs["ang"][idx, t]


def _leave(br, mask, p):
    br["att_inhib"] = jnp.where(mask, br["att_target"], br["att_inhib"])
    br["att_inhib_t"] = jnp.where(mask, p["inhibit_time"], br["att_inhib_t"])
    br["att_target"] = jnp.where(mask, -1, br["att_target"])


def _interface(br, raw, objs, p, dt):
    keep = jnp.exp(-dt / p["tau_a"])[:, None]
    x = raw[:, list(S.FAST)]
    fast = jnp.clip(x - br["a_slow"], 0.0, 1.0)
    br["a_slow"] = keep * br["a_slow"] + (1.0 - keep) * x
    has, feat, near, ang = seen_of(br, objs)
    fov = S.fovea_of(feat, near, ang, has, br["att_new"], xp=jnp)
    return jnp.concatenate([raw, fast, fov], 1)


def _td(br, p, f_v, reward, horizons, eta, dt, use_critic=True):
    """Come `Learner._td`: K critici (orizzonti `horizons` (N, K)), δ = media."""
    gamma = jnp.exp(-dt / horizons)
    value = (br["w_v"] * f_v[:, None, :]).sum(-1)
    if use_critic:
        deltas = jnp.where(br["fresh"][:, None], 0.0, reward[:, None] + gamma * value - br["v_prev"])
    else:
        deltas = jnp.repeat(reward[:, None], horizons.shape[1], 1)
    deltas = jnp.clip(deltas, -S.DELTA_MAX, S.DELTA_MAX)
    norm = 1.0 + (f_v**2).sum(1)[:, None] + (br["e_v"] * f_v[:, None, :]).sum(-1)
    br["w_v"] = br["w_v"] + (deltas / norm)[:, :, None] * eta[:, None, :] * br["e_v"]
    br["e_v"] = (gamma * jnp.exp(-dt / p["trace_critic"])[:, None])[:, :, None] * br["e_v"] + f_v[:, None, :]
    br["v_prev"] = (br["w_v"] * f_v[:, None, :]).sum(-1)
    br["fresh"] = jnp.zeros_like(br["fresh"])
    br["value"] = br["v_prev"].mean(1)
    full = deltas.mean(1)
    br["trust"] = p["critic_mix"] if use_critic else jnp.zeros_like(full)
    br["delta"] = reward + br["trust"] * (full - reward)
    return br["delta"]


def _act(br, logits, f_e, teach, eta, temp, p, dt, decide, key, f_tag=None, code=None, stick_on=None):
    """`decide` (N,): chi decide in questo passo."""
    np_ = S.N_PROGRAMS
    k_prog, k_bite = jax.random.split(key)
    norm = 1.0 + (f_e**2).sum(1)
    br["w_e"] = jnp.clip(br["w_e"] + (eta * teach / norm)[:, None, None] * br["e_e"], -S.WEIGHT_MAX, S.WEIGHT_MAX)
    br["e_e"] = br["e_e"] * jnp.exp(-dt / p["trace_actor"])[:, None, None]
    on = jnp.ones_like(temp) if stick_on is None else stick_on
    z = (logits[:, :np_] + (p["stick"] * on)[:, None] * jax.nn.one_hot(br["program"], np_)) / temp[:, None]
    pi = jax.nn.softmax(z, axis=1)
    choice = jax.random.categorical(k_prog, z, axis=1).astype(jnp.int32)
    chosen = jax.nn.one_hot(choice, np_)
    p_bite = jax.nn.sigmoid(jnp.clip(logits[:, S.BITE] / temp, -30, 30))
    bite = jax.random.uniform(k_bite, p_bite.shape) < p_bite
    br["program"] = jnp.where(decide, choice, br["program"])
    br["pi"] = jnp.where(decide[:, None], pi, br["pi"])
    br["bite"] = jnp.where(decide, bite, br["bite"])
    br["p_bite"] = jnp.where(decide, p_bite, br["p_bite"])
    dev = jnp.concatenate([chosen - pi, (bite.astype(F) - p_bite)[:, None]], 1) * decide[:, None]
    tag = f_e if f_tag is None else f_tag
    br["e_e"] = br["e_e"] + dev[:, :, None] * tag[:, None, :]
    if code is not None:
        br["e_c"] = br["e_c"] + dev[:, :, None] * code[:, None, :]
    _leave(br, decide & (br["program"] == S.LEAVE), p)
    return br["program"]


def sections(br, p, raw, objs, reward, dt, sz: S.Sizes, les: frozenset, decide, key, group: int):
    k_att, k_act = jax.random.split(key)
    arbitrate = "R" not in les and "U" not in les
    k = jnp.clip(br["unc"] / 3.0, 0.0, 1.0) if arbitrate else jnp.zeros_like(br["unc"])
    teacher = br["teacher"]
    n = raw.shape[0]
    master = jnp.arange(n) // group * group
    on = ~teacher & teacher[master]
    br.update(S.o_step(br, reward, raw[:, P.ENERGY], master, on, p, les, xp=jnp))  # O e K: gli episodi finiti insegnano
    br["k_on"] = br["k_on"].astype(jnp.int32)
    extra = jnp.where(teacher[:, None], jnp.where(objs["good"], 10.0, -10.0), 0.0)  # il maestro sa quali sono buone
    o = None if "S" in les else (*S.o_active(br, xp=jnp), raw[:, P.ENERGY])
    _attend(br, objs, decide, p, dt, k_att, extra=extra, learned=1.0 - p["att_shift"] * k, o=o)
    decide = decide | br["att_changed"]  # un oggetto diverso: si decide subito
    br["clock"] = jnp.where(br["att_changed"], 0.0, br["clock"])
    a = _interface(br, raw, objs, p, dt)
    fov = a[:, -len(S.FOVEA):]
    r = S.value_of(reward, raw[:, P.ENERGY], p, xp=jnp) + (p["innate"] * raw).sum(1) * dt
    teach_prev = br["teach"]
    # B
    if "B" in les:
        b = a
        br["state_b"] = a
    else:
        d = ((br["proto"] - a[:, None, :]) ** 2).sum(-1) - p["conscience"][:, None] * (1.0 / sz.b - br["win_rate"])
        kth = jnp.sort(d, axis=1)[:, sz.b_winners: sz.b_winners + 1]
        y = jnp.clip(1.0 - d / jnp.maximum(kth, 1e-9), 0.0, 1.0)
        y = y / jnp.maximum(y.max(1, keepdims=True), 1e-9)
        err = a - jnp.einsum("nab,nb->na", br["pred"], br["prev_b"])
        br["pred"] = br["pred"] + (p["eta_predict"] * dt)[:, None, None] * err[:, :, None] * br["prev_b"][:, None, :]
        rate = (p["eta_b"] * dt * (1.0 + p["surprise_boost"] * jnp.minimum(jnp.abs(br["delta"]), 1.0)))[:, None, None]
        br["proto"] = br["proto"] + rate * y[:, :, None] * ((a[:, None, :] - br["proto"])
                                                            + p["predict"][:, None, None] * err[:, None, :])
        br["win_rate"] = br["win_rate"] + (dt / p["win_window"])[:, None] * ((y > 0) / sz.b_winners - br["win_rate"])
        br["prev_b"] = br["state_b"]
        br["state_b"] = br["state_b"] + (dt / p["tau_b"])[:, None] * (y - br["state_b"])
        b = br["state_b"]
    # D
    if "D" in les:
        m = jnp.zeros_like(fov)
    else:
        br["w_gate"] = jnp.clip(br["w_gate"] + (p["eta_gate"] * teach_prev)[:, None] * br["e_gate"],
                                -2 * S.WEIGHT_MAX, 2 * S.WEIGHT_MAX)
        z = p["gate_event"] * fov[:, -1] + (br["w_gate"] * fov).sum(1) + p["gate_bias"]
        g = jax.nn.sigmoid(jnp.clip(z, -30, 30))
        tagged = jnp.where(fov > p["tag"][:, None], fov, 0.0)
        br["e_gate"] = jnp.exp(-dt / p["trace_actor"])[:, None] * br["e_gate"] + (g * (1.0 - g))[:, None] * tagged
        keep = jnp.exp(-dt / p["tau_d"])[:, None]
        br["mem"] = (1.0 - g)[:, None] * keep * br["mem"] + g[:, None] * fov
        m = br["mem"]
    # C
    if "C" in les:
        c = jnp.zeros((n, sz.c), F)
    else:
        h = jnp.concatenate([fov[:, :S.IDENTITY], p["memory"][:, None] * m[:, :S.IDENTITY],
                             p["context"][:, None] * b], 1)
        z = jnp.einsum("ncj,nj->nc", br["c_in"], h)
        c = S.c_code(z, p, xp=jnp)
    br["code"] = c
    if "O" not in les:
        br["o_e"] = S.o_trace(br["o_e"], fov, m, p, dt, xp=jnp)
        ids = jnp.stack([fov[:, :S.IDENTITY], m[:, :S.IDENTITY]], 1)
        br["o_out"] = S.o_read(*S.o_active(br, xp=jnp), ids, raw[:, P.ENERGY], p, xp=jnp)[0]
    vo = br["o_out"]
    one = jnp.ones((n, 1), F)
    f_v = jnp.concatenate([fov, m, c, raw[:, P.ENERGY:P.ENERGY + 1], p["critic_b"][:, None] * b, vo, one], 1)
    nf = fov.shape[1]
    is_c = (jnp.arange(f_v.shape[1]) >= 2 * nf) & (jnp.arange(f_v.shape[1]) < 2 * nf + sz.c)
    eta_v = jnp.where(is_c[None], p["eta_vc"][:, None], p["eta_v"][:, None])  # la parte su C: in fretta
    horizons = jnp.stack([p["h_short"], p["h_mid"], p["h_long"]], 1)
    delta = _td(br, p, f_v, r, horizons, eta_v, dt, use_critic="V" not in les)
    br["d_slow"] = br["d_slow"] + (dt / p["u_slow"]) * (jnp.abs(delta) - br["d_slow"])
    teach = S.teach_of(delta, br["d_slow"], p, xp=jnp)
    br["teach"] = teach
    br["d_fast"] = br["d_fast"] + (dt / p["u_fast"]) * (jnp.abs(teach) - br["d_fast"])
    br["t_slow"] = br["t_slow"] + (dt / p["u_slow"]) * (jnp.abs(teach) - br["t_slow"])
    if "U" not in les:
        br["unc"] = jnp.clip(br["d_fast"] / jnp.maximum(br["t_slow"], 1e-4) - 1.0, 0.0, 3.0)
    boost = 1.0 + p["surprise_rate"] * br["unc"]
    step = p["eta_c"] * boost * teach / (1.0 + c.sum(1))
    cap = p["c_max"][:, None, None]
    br["u_c"] = jnp.clip(br["u_c"] + step[:, None, None] * br["e_c"] - br["u_c"] * (dt / p["forget"])[:, None, None],
                         -cap, cap)
    br["e_c"] = br["e_c"] * jnp.exp(-dt / p["trace_actor"])[:, None, None]
    if "L" not in les:
        rest = (raw[:, P.ENERGY] > p["rest_energy"]) & (raw[:, P.TASTE] > 0.1)
        br["u_l"] = br["u_l"] + (p["eta_l"] * dt * rest)[:, None, None] * (br["u_c"] - br["u_l"])
    slow_boost = jnp.ones_like(boost) if arbitrate else boost
    f_e = jnp.concatenate([b, m, vo, one], 1)
    tag = p["tag"][:, None]
    f_tag = jnp.concatenate([jnp.where(b > tag, b, 0.0), jnp.where(m > tag, m, 0.0), vo, one], 1)
    habit = (br["w_e"] * f_e[:, None, :]).sum(-1)
    if "L" not in les:
        habit = habit + jnp.einsum("npc,nc->np", br["u_l"], c)
    fast = jnp.einsum("npc,nc->np", br["u_c"], c)
    bound = p["stick"][:, None] * jnp.einsum("npc,nc->np", br["e_c"], c) / jnp.maximum(c.sum(1), 1.0)[:, None]
    logits = (1.0 - p["arb_slow"] * k)[:, None] * habit + (1.0 + p["arb_fast"] * k)[:, None] * fast + bound
    temp = p["temperature"] * (1.0 + (0.0 if arbitrate else p["surprise_explore"]) * br["unc"])
    seeing = br["att_target"] >= 0
    prog = _act(br, logits, f_e, teach, p["eta_e"] * slow_boost, temp, p, dt, decide, k_act, f_tag=f_tag, code=c,
                stick_on=(~seeing).astype(F))
    # Il maestro decide da sé.
    has, _, near, _ = seen_of(br, objs)
    ok = has & objs["good"][jnp.arange(n), jnp.maximum(br["att_target"], 0)]
    prog, br["bite"], leave = S.teach_override(prog, br["bite"], has, ok, near, teacher, xp=jnp)
    prog = prog.astype(jnp.int32)
    br["p_bite"] = jnp.where(teacher, br["bite"].astype(F), br["p_bite"])
    _leave(br, leave, p)
    br["program"] = prog
    eye = raw[:, P.EYES].reshape(n, 3, 3).max(-1)
    seen = jnp.where(eye.max(1) > 0.2, eye.argmax(1), 3)
    br["counts"] = br["counts"].at[jnp.arange(n), br["state_b"].argmax(1), seen].add(1.0)
    return prog


def actor_critic(br, p, raw, objs, reward, dt, decide, key):
    k_att, k_act = jax.random.split(key)
    _attend(br, objs, decide, p, dt, k_att)
    a = _interface(br, raw, objs, p, dt)
    f = jnp.concatenate([a, jnp.ones((raw.shape[0], 1), F)], 1)
    r = S.value_of(reward, raw[:, P.ENERGY], p, xp=jnp) + (p["innate"] * raw).sum(1) * dt
    delta = _td(br, p, f, r, p["horizon"][:, None], jnp.repeat(p["eta_v"][:, None], f.shape[1], 1), dt)
    br["teach"] = delta
    logits = (br["w_e"] * f[:, None, :]).sum(-1)
    return _act(br, logits, f, delta, p["eta_e"], p["temperature"], p, dt, decide, k_act)


def guided(br, objs, decide, p, dt, key):
    _attend(br, objs, decide, p, dt, key, extra=jnp.where(objs["good"], 10.0, -10.0))
    has, _, near, _ = seen_of(br, objs)
    ok = has & objs["good"][jnp.arange(has.shape[0]), jnp.maximum(br["att_target"], 0)]
    prog = jnp.where(ok, S.APPROACH, S.EXPLORE).astype(jnp.int32)
    br["bite"] = ok & (near > 0.9)
    br["p_bite"] = br["bite"].astype(F)
    _leave(br, has & ~ok, p)
    br["program"], br["pi"] = prog, jax.nn.one_hot(prog, S.N_PROGRAMS)
    return prog


def reborn(br, dead, child, agent, sz, key, blocks):
    """I cervelli dei corpi rinati: ciò che si era imparato sparisce, e si
    riparte dai pesi e dalle connessioni di partenza del (nuovo) genoma."""
    br = dict(br)
    if child is not None:
        br["genome"] = jnp.where(dead[:, None], child, br["genome"])

    def zero(x):
        return jnp.where(dead.reshape((-1,) + (1,) * (x.ndim - 1)), jnp.zeros_like(x), x)

    for k in ("att_inhib_t", "att_prev_vis", "att_novelty", "att_new", "att_prob", "att_has", "att_parts",
              "att_changed"):
        br[k] = zero(br[k])
    br["att_target"] = jnp.where(dead, -1, br["att_target"])
    br["att_inhib"] = jnp.where(dead, -1, br["att_inhib"])
    br["clock"] = jnp.where(dead, 0.0, br["clock"])
    br["bite"] = br["bite"] & ~dead
    if agent == "guidato":
        return br
    for k in ("a_slow", "w_e", "e_e", "w_v", "e_v", "v_prev", "value", "delta", "teach", "trust"):
        br[k] = zero(br[k])

    br["fresh"] = br["fresh"] | dead
    if agent == "sezioni":
        for k in ("pred", "state_b", "prev_b", "mem", "w_gate", "e_gate", "u_c", "e_c", "u_l", "code", "d_fast",
                  "unc", "counts", "o_e", "ep_pl", "ep_pa", "ep_e", "o_err", "o_out", "obs_err"):
            br[k] = zero(br[k])
        br["o_assoc"] = jnp.where(dead[:, None, None, None], 0.5, br["o_assoc"])
        for k_ in ("k_ev", "k_since", "k_stale"):
            br[k_] = zero(br[k_])
        br["k_on"] = jnp.where(dead, 0, br["k_on"]).astype(jnp.int32)
        br["k_usual"] = jnp.where(dead, 0.5, br["k_usual"])
        first = jnp.arange(br["k_used"].shape[1])[None] == 0
        br["k_used"] = jnp.where(dead[:, None], first, br["k_used"])
        br["ep_on"] = br["ep_on"] & ~dead
        br["win_rate"] = jnp.where(dead[:, None], 1.0 / sz.b, br["win_rate"])
        br["d_slow"] = jnp.where(dead, 1e-3, br["d_slow"])
        br["t_slow"] = jnp.where(dead, 1e-4, br["t_slow"])
    n = dead.shape[0]
    start = lambda name: br["genome"][:, blocks[name][0]].reshape((n,) + blocks[name][1])
    born = {"e0": "w_e", "v0": "w_v", "proto": "proto"}
    for name, key_ in born.items():
        if name in blocks:
            br[key_] = jnp.where(dead.reshape((-1,) + (1,) * (br[key_].ndim - 1)), start(name), br[key_])
    if "o0" in blocks:  # i valori innati delle cose
        o0 = start("o0")
        br["o_pl"] = jnp.where(dead[:, None, None], o0[:, None, 0], br["o_pl"])
        br["o_pa"] = jnp.where(dead[:, None, None], o0[:, None, 1], br["o_pa"])
        br["o_prior"] = jnp.where(dead[:, None, None], o0, br["o_prior"])
    if "c_in" in blocks:
        br["c_in"] = jnp.where(dead[:, None, None], (start("c_in") > 0.5).astype(F), br["c_in"])
    return br
