"""L'arena continua sulla GPU (JAX): fa ciò che fanno `ArenaWorld.step`,
`NeuralController.act`, `Prepared.respond` e `Brains.step` nel motore NumPy
(sim/arena.py), con lo stesso cervello e la stessa plasticità del motore GPU
a generazioni (engine.py: `rate_brain`, `spike_brain`), più il mondo che si
complica (docs/SPEC_ARENA.md, sezione 15), che c'è solo qui:

- il corpo: la taglia (raggio, metabolismo, riserva, spinta) e quattro azioni
  comandate da motori in più: mordere, luce, odore, nutrire;
- i frutti rossi e gialli, buoni o cattivi; il cibo duro (va morso a lungo) e
  grande (va morso in due);
- la predazione (si morde chi è abbastanza più piccolo), le cure (si nutre un
  proprio figlio), i segnali (luce del proprio colore, odore della propria
  molecola), la notte (la luce del mondo cala, quella dei segnali no).

A manopole spente, con taglia 1 e azioni ferme, è il mondo base.

Sulla GPU gira un secondo di mondo intero (`arena_tick` / dt passi) senza
tornare su Python. Sul processore, una volta al secondo, restano le cose
rare: nascite, morti per vecchiaia, ricrescita del cibo, salute e colori
delle macchie, manopole, statistiche. I neonati arrivano sulla GPU riga per
riga (`set_rows`), senza ricompilare e senza toccare gli altri.

I vicini si trovano con una griglia di celle rifatta a ogni passo: ogni
cella tiene al più `CELL_SLOTS` oggetti per tipo (cibo, esseri).

Differenze volute rispetto al motore NumPy: numeri a 32 bit, numeri casuali
di JAX, urti risolti tutti insieme, due esseri alla stessa distanza dallo
stesso cibo lo mangiano entrambi (caso quasi impossibile).
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..net.receptors import LIGHT, scales
from .engine import F, _rotate, _tuning, body, phys_of, rate_brain, spike_brain

CELL_SLOTS = 16  # oggetti per cella e per tipo che i sensi vedono
ROWS = 64  # righe aggiornate insieme alla nascita (si ripete se sono di più)
BLOCK = 256  # si calcolano i posti fino all'ultimo vivo, arrotondati a multipli di BLOCK
FOOD_KEYS = ("food", "food_on", "food_r", "food_color", "food_bad", "food_kind", "work")
EXTRA = 4  # azioni: mordere, luce, odore, nutrire
NORMAL, HARD, BIG = 0, 1, 2  # tipi di cibo


# --- La griglia dei vicini -----------------------------------------------------

def _wrap(d, size):
    return d - size * jnp.floor(d / size + 0.5)


def _cells(pos, cw, ch, ncx, ncy):
    cx = jnp.floor(pos[..., 0] / cw).astype(jnp.int32) % ncx
    cy = jnp.floor(pos[..., 1] / ch).astype(jnp.int32) % ncy
    return cx, cy


def table(pos, use, geo):
    """(celle, CELL_SLOTS): gli indici degli oggetti `use` di ogni cella, -1
    dove non ce ne sono."""
    cw, ch, ncx, ncy = geo
    cx, cy = _cells(pos, cw, ch, ncx, ncy)
    n_cells = ncx * ncy
    cell = jnp.where(use, cy * ncx + cx, n_cells)
    order = jnp.argsort(cell)
    ranked = cell[order]
    ids = jnp.arange(n_cells)
    start = jnp.searchsorted(ranked, ids)
    end = jnp.searchsorted(ranked, ids, side="right")
    j = start[:, None] + jnp.arange(CELL_SLOTS)[None, :]
    return jnp.where(j < end[:, None], order[jnp.clip(j, 0, len(order) - 1)], -1)


def neighbours(tab, pos, geo):
    """(N, 9 · CELL_SLOTS): gli oggetti delle 9 celle attorno a ogni posizione."""
    cw, ch, ncx, ncy = geo
    cx, cy = _cells(pos, cw, ch, ncx, ncy)
    dx = jnp.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    dy = jnp.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    cells = ((cy[:, None] + dy) % ncy) * ncx + (cx[:, None] + dx) % ncx  # (N, 9)
    return tab[cells].reshape(pos.shape[0], -1)


def _spectral(mu, sigma, color, width):
    """Come `receptors.spectral` con albedo 1."""
    s2 = sigma ** 2 + width ** 2
    return sigma / jnp.sqrt(s2) * jnp.exp(-0.5 * (mu - color) ** 2 / s2)


def daylight(ph, t):
    """La luce del mondo: 1 di giorno, fino a 1 - 0,85 · manopola a mezzanotte."""
    night = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * t / ph["arena_day"])
    return 1.0 - 0.85 * ph["arena_night"] * night


def radius(ph, s):
    """Il raggio di ogni essere: √taglia volte quello di base."""
    return ph["agent_radius"] * jnp.sqrt(s["size"])


# --- Sensi -------------------------------------------------------------------

def quanta(ph, P, s, near_food, near_being):
    """Come `Prepared.quanta` nell'arena: luce di cibo ed esseri vicini (con la
    luce dei segnali), odori (cibo per colore, esseri, segnali), tatto degli
    altri esseri, temperatura dell'ambiente, bussola, corpo. (N, R)."""
    N, R = P["active"].shape
    W, H = ph["arena_size_w"], ph["arena_size_h"]
    rad = radius(ph, s)
    c, sn = jnp.cos(s["heading"])[:, None], jnp.sin(s["heading"])[:, None]
    nx, ny = _rotate(P["body"][..., 0], P["body"][..., 1], c, sn)  # (N, R)
    px = (s["pos"][:, 0, None] + rad[:, None] * nx) % W
    py = (s["pos"][:, 1, None] + rad[:, None] * ny) % H
    me = jnp.arange(N)[:, None]
    fi = jnp.maximum(near_food, 0)
    bi = jnp.maximum(near_being, 0)
    f_ok = (near_food >= 0) & (s["food_on"][fi] > 0)  # (N, Cf)
    b_ok = (near_being >= 0) & (near_being != me) & (s["alive"][bi] > 0)  # (N, Cb)
    food, beings = s["food"][fi], s["pos"][bi]  # (N, C, 2)
    f_r, b_r = s["food_r"][fi], rad[bi]  # (N, C)
    red = s["food_color"][fi] == 0
    fx = _wrap(food[:, None, :, 0] - px[..., None], W)
    fy = _wrap(food[:, None, :, 1] - py[..., None], H)
    gx = _wrap(beings[:, None, :, 0] - px[..., None], W)
    gy = _wrap(beings[:, None, :, 1] - py[..., None], H)
    # Luce: per ogni direzione campione il primo oggetto colpito.
    ang = s["heading"][:, None, None] + P["angles"]  # (N, R, M)
    dx, dy = jnp.cos(ang)[..., None], jnp.sin(ang)[..., None]  # (N, R, M, 1)

    def hits(ox, oy, rr, ok):
        rr = rr[:, None, None, :]
        pj = ox[:, :, None, :] * dx + oy[:, :, None, :] * dy
        dsc = rr * rr - (ox * ox + oy * oy)[:, :, None, :] + pj * pj
        hf = jnp.sqrt(jnp.maximum(dsc, 0.0))
        d = jnp.where(ok[:, None, None, :] & (dsc >= 0) & (pj + hf >= 0), jnp.maximum(pj - hf, 0.0), jnp.inf)
        return d.min(-1), jnp.argmin(d, -1)  # (N, R, M)

    d_food, k_food = hits(fx, fy, f_r, f_ok)
    d_being, k_being = hits(gx, gy, b_r, b_ok)
    L = ph["light_attenuation"]
    day = daylight(ph, s["t"])
    food_first = d_food <= d_being
    fade_food = jnp.where(food_first, jnp.exp(-d_food / L), 0.0)
    fade_being = jnp.where(~food_first, jnp.exp(-d_being / L), 0.0)
    hit_red = jnp.take_along_axis(red, k_food.reshape(N, -1), 1).reshape(k_food.shape)
    seen_red = (fade_food * hit_red).mean(-1)
    seen_yellow = (fade_food * ~hit_red).mean(-1)
    out = day * (P["c_food"] * seen_red + P["c_fruit"] * seen_yellow + P["c_being"] * fade_being.mean(-1))
    # La luce dei segnali dell'essere colpito: del suo colore, non cala di notte.
    who = jnp.take_along_axis(bi, k_being.reshape(N, -1), 1).reshape(k_being.shape)  # (N, R, M)
    glow = s["glow"][who] * _spectral(P["mu_l"][..., None], P["sg_l"][..., None], s["sig_color"][who],
                                     ph["being_color_width"])
    out += P["g_l"] * (fade_being * glow).mean(-1)
    # Odori: cibo (molecola del suo colore), esseri, segnali (molecola propria).
    lam = ph["odor_range"]
    near_f = jnp.exp(-jnp.maximum(jnp.sqrt(fx * fx + fy * fy) - f_r[:, None, :], 0.0) / lam) * f_ok[:, None, :]
    near_b = jnp.exp(-jnp.maximum(jnp.sqrt(gx * gx + gy * gy) - b_r[:, None, :], 0.0) / lam) * b_ok[:, None, :]
    smell = ph["food_smell"]
    out += P["g_c"] * _tuning(ph["food_molecule"], P["mu_c"], P["sg_c"]) * smell * (near_f * red[:, None, :]).sum(-1)
    out += P["g_c"] * _tuning(ph["fruit_molecule"], P["mu_c"], P["sg_c"]) * smell * (near_f * ~red[:, None, :]).sum(-1)
    out += P["g_c"] * _tuning(ph["being_molecule"], P["mu_c"], P["sg_c"]) * ph["being_smell"] * near_b.sum(-1)
    sig = _tuning(s["sig_mol"][bi][:, None, :], P["mu_c"][..., None], P["sg_c"][..., None])  # (N, R, Cb)
    out += P["g_c"] * ph["arena_signal_smell"] * (near_b * sig * s["smell"][bi][:, None, :]).sum(-1)
    # Tatto: gli altri esseri toccati, sul lato che li guarda.
    bx = _wrap(beings[..., 0] - s["pos"][:, 0, None], W)  # (N, Cb), dal centro del corpo
    by = _wrap(beings[..., 1] - s["pos"][:, 1, None], H)
    dist = jnp.sqrt(bx * bx + by * by)
    touch = b_ok & (dist > 1e-9) & (dist <= rad[:, None] + b_r + 1e-6)
    facing = (bx[:, None, :] * nx[..., None] + by[:, None, :] * ny[..., None]) / jnp.maximum(dist, 1e-9)[:, None, :]
    out += P["c_press"] * (touch[:, None, :] * jnp.maximum(0.0, facing)).sum(-1)
    # Temperatura dell'ambiente, bussola, corpo (l'energia rispetto alla propria riserva).
    out += P["g_t"] * _tuning(jnp.clip(ph["ambient_temperature"], 0.0, 1.0), P["mu_t"], P["sg_t"])
    centre = s["heading"][:, None] + P["centre"]
    out += P["g_mag"] * jnp.maximum(0.0, jnp.cos(centre) * ph["magnetic"][0] + jnp.sin(centre) * ph["magnetic"][1])
    out += jnp.einsum("nrk,nk->nr", P["c_body"], body(ph, {**s, "energy": s["energy"] / s["size"]}))
    return out


def respond(ph, P, s, key, alive, near_food, near_being):
    """Come `Prepared.respond`: rumore, adattamento, saturazione. I posti
    liberi tengono risposta e media di prima."""
    n = quanta(ph, P, s, near_food, near_being)
    noisy = jnp.maximum(n + jnp.sqrt(n + ph["receptor_dark"]) * jax.random.normal(key, n.shape, F), 0.0)
    m = jnp.where(jnp.isnan(s["m"]), noisy, s["m"])
    r = jnp.clip((noisy - P["sub"] * m) / (ph["receptor_saturation"] + P["div"] * m), 0.0, 1.0) * P["active"]
    m = P["keep"] * m + (1.0 - P["keep"]) * noisy
    on = alive[:, None] > 0
    return jnp.where(on, r, s["obs"]), jnp.where(on, m, s["m"])


# --- Mondo ---------------------------------------------------------------------

def physics(ph, P, s, actions, extra, near_food, near_being):
    """Come `ArenaWorld.step` (senza la vecchiaia, contata una volta al secondo
    sul processore), più il corpo e le azioni: movimento sul toro, urti,
    energia, morsi, cure, pasti (normali, duri, grandi, frutti cattivi),
    morti. `extra` (N, 4): mordere, luce, odore, nutrire, tra 0 e 1."""
    dt = ph["dt"]
    W, H = ph["arena_size_w"], ph["arena_size_h"]
    size = s["size"]
    rad = radius(ph, s)
    alive = s["alive"] > 0
    extra = extra * alive[:, None]
    bite, light, odor, feed = extra[:, 0], extra[:, 1], extra[:, 2], extra[:, 3]
    act = jnp.clip(actions, -1.0, 1.0) * alive[:, None]
    thrust = jnp.where(act[:, 0] < 0.0, act[:, 0] * ph["reverse_factor"], act[:, 0])
    turn = act[:, 1]
    ang_vel = (s["ang_vel"] + turn * ph["turn_accel"] * dt) * jnp.exp(-ph["ang_drag"] * dt)
    heading = s["heading"] + ang_vel * dt
    heading = (heading + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    fx, fy = jnp.cos(heading), jnp.sin(heading)
    vel = s["vel"]
    v_fwd = vel[:, 0] * fx + vel[:, 1] * fy
    lx, ly = vel[:, 0] - v_fwd * fx, vel[:, 1] - v_fwd * fy
    v_fwd = (v_fwd + thrust * ph["thrust_accel"] * size ** -0.25 * dt) * jnp.exp(-ph["lin_drag"] * dt)
    k = jnp.exp(-ph["lat_drag"] * dt)
    vel = jnp.stack([v_fwd * fx + lx * k, v_fwd * fy + ly * k], 1)
    vel = jnp.where(alive[:, None], vel, 0.0)
    ang_vel = jnp.where(alive, ang_vel, 0.0)
    bounds = jnp.stack([W, H])
    pos = (s["pos"] + vel * dt) % bounds
    # Urti: ognuno si sposta di metà della compenetrazione con ogni vicino e
    # perde la velocità con cui gli si avvicinava.
    n = pos.shape[0]
    me = jnp.arange(n)[:, None]
    bi = jnp.maximum(near_being, 0)
    ok = (near_being >= 0) & (near_being != me) & (s["alive"][bi] > 0) & alive[:, None]
    ox = _wrap(pos[bi][..., 0] - pos[:, 0, None], W)
    oy = _wrap(pos[bi][..., 1] - pos[:, 1, None], H)
    d = jnp.sqrt(ox * ox + oy * oy)
    reach = rad[:, None] + rad[bi]
    hit = ok & (d < reach)
    safe = jnp.maximum(d, 1e-9)
    ux, uy = jnp.where(d > 1e-9, ox / safe, 1.0), jnp.where(d > 1e-9, oy / safe, 0.0)
    overlap = jnp.where(hit, reach - d, 0.0)
    push = -0.5 * jnp.stack([(overlap * ux).sum(1), (overlap * uy).sum(1)], 1)
    closing = jnp.where(hit, jnp.maximum(vel[:, 0, None] * ux + vel[:, 1, None] * uy, 0.0), 0.0)
    vel = vel - jnp.stack([(closing * ux).sum(1), (closing * uy).sum(1)], 1)
    pos = (pos + push) % bounds
    bump = hit.any(1)
    # Energia: metabolismo (Kleiber), azioni, segnali.
    e_max = ph["energy_max"] * size
    mass = 1.0 + ph["reserve_mass"] * jnp.maximum(s["energy"] / size - ph["energy_init"], 0.0) / ph["energy_init"]
    kleiber = size ** 0.75
    effort = (kleiber * (ph["metabolism_basal"]
                         + mass * ph["metabolism_move"] * jnp.abs(act[:, 0]) ** ph["metabolism_exponent"]
                         + mass * ph["metabolism_turn"] * jnp.abs(turn) ** ph["metabolism_exponent"])
              + P["upkeep"] + ph["arena_bite_cost"] * bite + ph["arena_signal_cost"] * (light + odor)) * alive
    energy = jnp.maximum(s["energy"] - effort * dt, 0.0)
    # Morsi: a un essere toccato davanti (entro 60°) e abbastanza più piccolo.
    touching = ok & (d <= reach + 0.5)
    ahead = (ox * fx[:, None] + oy * fy[:, None]) / safe > 0.5
    smaller = size[:, None] >= ph["arena_bite_ratio"] * size[bi]
    bites = touching & ahead & smaller
    dmg = jnp.where(bites, ph["arena_bite_rate"] * ph["arena_predation"] * bite[:, None] * size[:, None] * dt, 0.0)
    flat_b = bi.reshape(-1)
    taken = jax.ops.segment_sum(dmg.reshape(-1), flat_b, n)  # quanto si vorrebbe togliere a ciascuno
    scale = jnp.where(taken > 0, jnp.minimum(1.0, energy / jnp.maximum(taken, 1e-9)), 0.0)
    dmg = dmg * scale[bi]
    lost = jax.ops.segment_sum(dmg.reshape(-1), flat_b, n)
    energy = energy - lost + ph["arena_bite_gain"] * dmg.sum(1)
    bitten = lost > 0
    killed = alive & bitten & (energy <= 1e-6)
    # Cure: chi nutre passa energia a un proprio figlio toccato non ancora adulto.
    child = touching & (s["parent"][bi] == s["uid"][:, None]) & (s["young"][bi] > 0)
    give = jnp.where(child, ph["arena_feed_rate"] * feed[:, None] * dt, 0.0)
    give = give * jnp.minimum(1.0, energy / jnp.maximum(give.sum(1), 1e-9))[:, None]
    energy = energy - give.sum(1) + ph["arena_feed_gain"] * jax.ops.segment_sum(give.reshape(-1), flat_b, n)
    # Pasti.
    fi = jnp.maximum(near_food, 0)
    f_ok = (near_food >= 0) & (s["food_on"][fi] > 0) & alive[:, None]
    qx = _wrap(s["food"][fi][..., 0] - pos[:, 0, None], W)
    qy = _wrap(s["food"][fi][..., 1] - pos[:, 1, None], H)
    dist = jnp.where(f_ok, jnp.sqrt(qx * qx + qy * qy), jnp.inf)
    at = dist <= s["food_r"][fi] + rad[:, None]  # (N, Cf): chi tocca quale cibo
    kind = s["food_kind"][fi]
    n_food = s["food"].shape[0]
    flat_f = fi.reshape(-1)
    # Normale (e frutti): va al vivo più vicino tra chi lo tocca.
    plain = at & (kind == NORMAL)
    best = jax.ops.segment_min(jnp.where(plain, dist, jnp.inf).reshape(-1), flat_f, n_food)
    win_plain = plain & (dist <= best[fi])
    # Duro: si accumula il morso di chi lo tocca; quando basta lo prende chi morde di più.
    hard = at & (kind == HARD)
    work = s["work"] + jax.ops.segment_sum(jnp.where(hard, bite[:, None] * dt, 0.0).reshape(-1), flat_f, n_food)
    open_hard = (s["food_kind"] == HARD) & (work >= ph["arena_hard_work"]) & (s["food_on"] > 0)
    top = jax.ops.segment_max(jnp.where(hard, bite[:, None], -1.0).reshape(-1), flat_f, n_food)
    win_hard = hard & open_hard[fi] & (bite[:, None] >= top[fi]) & (bite[:, None] > 0)
    # Grande: si apre se almeno 2 lo mordono insieme; l'energia si divide tra loro.
    big = at & (kind == BIG) & (bite[:, None] > 0.5)
    n_big = jax.ops.segment_sum(big.reshape(-1).astype(F), flat_f, n_food)
    open_big = (n_big >= 2.0) & (s["food_on"] > 0)
    win_big = big & open_big[fi]
    bad = s["food_bad"][fi] > 0
    unit = ph["food_energy"]
    gain = (unit * (win_plain & ~bad) + unit * ph["arena_hard_energy"] * win_hard
            + jnp.where(win_big, unit * ph["arena_big_energy"] / jnp.maximum(n_big[fi], 1.0), 0.0))
    good_meals = ((win_plain & ~bad).sum(1) + win_hard.sum(1) + win_big.sum(1)).astype(F)
    bad_meals = (win_plain & bad).sum(1).astype(F)
    energy = energy + gain.sum(1) - ph["bad_fruit_damage"] * bad_meals
    eaten = jax.ops.segment_max((win_plain | win_hard | win_big).reshape(-1).astype(jnp.int32), flat_f, n_food) > 0
    food_on = jnp.where(eaten, 0.0, s["food_on"])
    work = jnp.where(eaten | (s["food_on"] <= 0), 0.0, work)
    energy = jnp.clip(energy, 0.0, e_max)
    taste = jnp.minimum(s["taste"] * jnp.exp(-dt / ph["taste_tau"]) + good_meals, 1.0)
    pain = jnp.minimum(bitten.astype(F) + bad_meals, 1.0)
    dead = alive & (energy <= 0.0)
    starved = dead & ~killed
    return dict(pos=pos, vel=vel, heading=heading, ang_vel=ang_vel, energy=jnp.where(dead, 0.0, energy),
                alive=(alive & ~dead).astype(F), bump=bump.astype(F), taste=taste, pain=pain,
                food_on=food_on, work=work, eaten=s["eaten"] + good_meals, glow=light, smell=odor, bite=bite,
                starved=s["starved"] + starved.sum(), killed=s["killed"] + killed.sum(),
                meals=s["meals"] + good_meals.sum(), bad_meals=s["bad_meals"] + bad_meals.sum(),
                fed=s["fed"] + give.sum(), t=s["t"] + dt)


def make_run(net_type: str, n_hidden: int, substeps: int, geo: tuple):
    """`run(ph, P, s, k)`: k passi di mondo sulla GPU."""
    base = 2 if net_type == "rate" else 4  # motori di movimento

    def step(ph, P, s):
        alive = s["alive"]
        near_food = neighbours(table(s["food"], s["food_on"] > 0, geo), s["pos"], geo)
        near_being = neighbours(table(s["pos"], alive > 0, geo), s["pos"], geo)
        key, sub = jax.random.split(s["key"])
        obs, m = respond(ph, P, s, sub, alive, near_food, near_being)
        if net_type == "rate":
            brain, out = rate_brain(ph, P, s, obs, alive, n_hidden)
        else:
            brain, out = spike_brain(ph, P, s, obs, alive, n_hidden, substeps)
        extra = jnp.clip(brain["state"][:, n_hidden + base: n_hidden + base + EXTRA], 0.0, 1.0)
        world = physics(ph, P, s, out[:, :2] * alive[:, None], extra, near_food, near_being)
        return {**s, **world, **brain, "obs": obs, "m": m, "key": key}

    def run(ph, P, s, k, m):
        """k passi sui soli primi m posti (gli altri sono liberi): si taglia,
        si calcola e si rimette a posto dentro lo stesso programma compilato."""
        n = P["active"].shape[0]
        if m >= n:
            return jax.lax.fori_loop(0, k, lambda i, s: step(ph, P, s), s)
        rows = [key for key, v in s.items() if key not in FOOD_KEYS and v.ndim and v.shape[0] == n]
        cut = {key: (v[:m] if key in rows else v) for key, v in s.items()}
        cut = jax.lax.fori_loop(0, k, lambda i, c: step(ph, {key: v[:m] for key, v in P.items()}, c), cut)
        return {key: (s[key].at[:m].set(v) if key in rows else v) for key, v in cut.items()}

    return jax.jit(run, static_argnames="m")


# --- Dal processore alla GPU e ritorno ---------------------------------------------

BRAIN_PARAMS = ("links", "target_on", "eta", "rule", "kmod", "learns", "upkeep", "hebb", "sign", "intrinsic",
                "boost", "closure", "value", "early_keep", "homeo", "target_act")
SENSE_PARAMS = ("angles", "body", "active", "centre", "half", "mu_c", "sg_c", "g_c", "c_press", "mu_m", "sg_m", "g_m",
                "mu_t", "sg_t", "g_t", "g_mag", "c_body", "keep", "sub", "div", "c_food", "c_fruit", "c_being")


def pack_params(brains) -> dict:
    """Le costanti di ogni posto: geni compilati del cervello e dei recettori."""
    p = brains.prepared()
    out = {k: getattr(p, k) for k in SENSE_PARAMS}
    out.update({k: getattr(brains, k) for k in BRAIN_PARAMS})
    out["mod_sign"] = brains._mod_sign
    genes = brains.receptors
    out.update(mu_l=genes.mu[..., LIGHT], sg_l=genes.sigma[..., LIGHT],
               g_l=(genes.gain * scales(brains.cfg))[..., LIGHT])  # per la luce dei segnali, di colore variabile
    if brains.KIND == "spike-ei":
        out.update(keep_n=brains.keep, active_n=brains.active)
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


BRAIN_STATE = ("state", "w_eff", "avg", "early", "bias_now", "clock", "charge", "spikes")
STATE_NAME = {"w_eff": "w"}  # come si chiamano sulla GPU


def pack_brain_state(brains, obs, adapt) -> dict:
    out = {STATE_NAME.get(k, k): getattr(brains, k) for k in BRAIN_STATE if hasattr(brains, k)}
    out.update(obs=obs, m=adapt)
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


WORLD = ("pos", "vel", "heading", "ang_vel", "energy", "taste", "bump")
ACTIONS = ("bite", "glow", "smell")  # livelli dell'ultimo passo (per l'interfaccia)
COUNTERS = ("starved", "killed", "meals", "bad_meals", "fed")


def arena_phys(cfg) -> dict:
    ph = phys_of(cfg)
    for k in ("arena_size_w", "arena_size_h", "being_molecule", "being_smell", "being_color_width", "arena_day",
              "arena_night", "arena_predation", "arena_bite_rate", "arena_bite_ratio", "arena_bite_gain",
              "arena_bite_cost", "arena_hard_work", "arena_hard_energy", "arena_big_energy", "arena_signal_cost",
              "arena_signal_smell", "arena_feed_rate", "arena_feed_gain"):
        ph[k] = np.float32(getattr(cfg, k))
    ph["food_energy"] = np.float32(cfg.arena_food_energy)  # nell'arena il cibo vale arena_food_energy
    return ph


class ArenaGpu:
    """L'arena di un'ecologia sulla GPU. Lo stato del mondo e dei cervelli vive
    qui; `pull` ne riporta sul processore ciò che serve, `push_world` ci porta
    ciò che il processore ha cambiato (nascite, morti, cibo, corpi)."""

    _compiled: dict = {}

    def __init__(self, world, ctrl, seed: int, bodies: dict):
        cfg = world.cfg
        self.cfg, self.world, self.ctrl = cfg, world, ctrl
        b = ctrl.brains
        self.geo = (float(world.cells[0]), float(world.cells[1]), int(world.cells[2]), int(world.cells[3]))
        self.P = {k: jnp.asarray(v) for k, v in pack_params(b).items()}
        s = pack_brain_state(b, ctrl.obs, ctrl.adapt)
        s.update({k: np.asarray(getattr(world, k), dtype=np.float32) for k in WORLD})
        s.update({k: np.zeros(world.n, np.float32) for k in ACTIONS})
        s.update({k: np.float32(0.0) for k in COUNTERS})
        s.update(alive=world.alive, pain=world.pain, t=world.t, eaten=world.task.eaten,
                 work=np.zeros(len(world.task.food), np.float32))
        self.s = {k: jnp.asarray(np.asarray(v, dtype=np.float32)) for k, v in s.items()}
        self.s["key"] = jax.random.PRNGKey(int(seed) % (2**31))
        self.push_world(bodies)
        self.set_phys()
        key = (b.KIND, b.n_hidden, cfg.spike_substeps, self.geo)
        if key not in ArenaGpu._compiled:
            ArenaGpu._compiled[key] = make_run(b.KIND, b.n_hidden, cfg.spike_substeps, self.geo)
        self._run = ArenaGpu._compiled[key]
        self.counts = {k: 0.0 for k in COUNTERS}

    def set_phys(self) -> None:
        """Le costanti della fisica (anche le manopole del mondo, che cambiano)."""
        self.ph = {k: jnp.asarray(v) for k, v in arena_phys(self.cfg).items()}

    def advance(self, k: int) -> None:
        """k passi sulla GPU, poi sul processore lo stato del mondo. Si
        calcolano solo i posti fino all'ultimo vivo (le nascite occupano i
        posti liberi più bassi): con pochi vivi si va molto più veloci."""
        n = self.world.n
        alive = np.flatnonzero(self.world.alive)
        m = n if not len(alive) else min(n, -(-(int(alive[-1]) + 1) // BLOCK) * BLOCK)
        self.s = self._run(self.ph, self.P, self.s, jnp.int32(k), m=m)
        self.world.t += k * self.cfg.dt  # il tempo resta sul processore, a 64 bit (a 32 bit si sbaglia)
        self.pull()

    def pull(self) -> None:
        """Mondo e cibo dalla GPU agli oggetti NumPy (i cervelli restano lì)."""
        w, task = self.world, self.world.task
        g = jax.device_get({k: self.s[k] for k in WORLD + ACTIONS + COUNTERS + ("alive", "food_on", "eaten")})
        for k in WORLD:
            setattr(w, k, np.asarray(g[k], dtype=float))
        w.bump = g["bump"] > 0
        w.alive = g["alive"] > 0
        for k in ACTIONS:
            setattr(w, k, np.asarray(g[k], dtype=float))
        task.food_on = g["food_on"] > 0
        task.eaten = np.asarray(g["eaten"]).astype(int)
        now = {k: float(g[k]) for k in COUNTERS}
        new = {k: now[k] - self.counts[k] for k in COUNTERS}
        self.counts = now
        w.deaths["fame"] += int(round(new["starved"]))
        w.deaths["predati"] = w.deaths.get("predati", 0) + int(round(new["killed"]))
        task.eaten_total += int(round(new["meals"]))
        task.bad_total += int(round(new["bad_meals"]))
        task.fed_total += new["fed"]
        w.grids_dirty = True

    def push_world(self, bodies: dict | None = None) -> None:
        """Il mondo dal processore alla GPU, dopo nascite, morti per vecchiaia,
        cibo nuovo; con `bodies` anche i corpi (taglia, segnali, parentela)."""
        w, task = self.world, self.world.task
        upd = {k: getattr(w, k) for k in WORLD}
        upd.update(alive=w.alive, food=task.food, food_on=task.food_on, eaten=task.eaten,
                   food_color=task.color, food_bad=task.bad, food_kind=task.kind, food_r=task.radius)
        if bodies is not None:
            upd.update(bodies)
        for k, v in upd.items():  # identità e genitori come interi: a 32 bit decimali si confonderebbero
            self.s[k] = jnp.asarray(np.asarray(v, dtype=np.int32 if k in ("uid", "parent") else np.float32))
        self.s["energy"] = jnp.asarray(np.asarray(w.energy, dtype=np.float32))

    def set_rows(self, rows: np.ndarray, params: dict, state: dict) -> None:
        """I posti `rows` ricevono costanti e stato nuovi (i neonati); gli altri
        non cambiano. Si aggiornano ROWS righe alla volta (meno compilazioni)."""
        for at in range(0, len(rows), ROWS):
            part = np.arange(at, min(at + ROWS, len(rows)))
            idx = np.resize(part, ROWS)  # si ripete l'ultima scrittura: stesso valore
            r = jnp.asarray(rows[idx], dtype=jnp.int32)
            self.P = _scatter(self.P, r, {k: jnp.asarray(v[idx]) for k, v in params.items()})
            self.s = _scatter(self.s, r, {k: jnp.asarray(v[idx]) for k, v in state.items()})

    def brain_state(self) -> dict:
        """Lo stato dei cervelli dalla GPU (per salvare)."""
        names = [STATE_NAME.get(k, k) for k in BRAIN_STATE] + ["obs", "m"]
        return {k: np.asarray(v, dtype=float) for k, v in jax.device_get(
            {k: self.s[k] for k in names if k in self.s}).items()}


@jax.jit
def _scatter(tree: dict, rows, values: dict) -> dict:
    out = dict(tree)
    for k, v in values.items():
        out[k] = tree[k].at[rows].set(v.astype(tree[k].dtype))
    return out
