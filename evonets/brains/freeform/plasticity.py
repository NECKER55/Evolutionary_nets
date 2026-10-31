"""La regola di plasticità delle reti a forma libera, uguale per NumPy e JAX."""




# Le costanti della configurazione che usa `plasticity`.
PLASTICITY_CONSTANTS = ("dt", "synapse_max", "homeo_tau", "capture_rate", "surprise", "surprise_horizon",
                        "surprise_smooth", "critic_rate")


def frequencies(xp, x, centred: bool):
    """Le frequenze (≥ 0) che corrispondono alle attività `x` dei neuroni.
    Nelle reti continue (`centred`) x = tanh(...) è lo scarto dalla frequenza
    spontanea, a metà tra zitto (-1) e massimo (+1): la frequenza è (1 + x)/2.
    Nelle reti a impulsi le attività sono già frequenze."""
    return (1.0 + x) / 2.0 if centred else xp.maximum(x, 0.0)


def plasticity(cfg, xp, p, pre, post, m, n_in: int, centred: bool):
    """Un passo di plasticità (vedi `Brains._learn`), uguale per NumPy e JAX
    (`xp` è il modulo). `p`: i geni compilati e lo stato di chi impara; pre
    (n, S) con i primi `n_in` i sensi, post (n, T), m (n, T) il modulatore che
    arriva a ogni neurone; `centred` per le reti continue (vedi
    `frequencies`). Le regole di Hebb e l'omeostasi usano le frequenze; la
    plasticità modulata, che è una regola scritta dai geni, i valori come
    sono. Restituisce (pesi, traccia provvisoria, corrente di fondo, ā,
    traccia di eleggibilità, critico)."""
    dt, top = cfg["dt"], cfg["synapse_max"]
    senses = xp.maximum(pre[:, :n_in], 0.0)  # risposte dei recettori: già frequenze
    # I nuclei di valore proiettano su tutto il cervello: il gusto aggiunge M
    # positivo, il dolore M negativo (p["signal"]: gusto e dolore, (n, 2)).
    reward = p["value"][:, 0] * p["signal"][:, 0] - p["value"][:, 1] * p["signal"][:, 1]
    # Sorpresa (TD continuo, Doya 2000): il critico prevede dai sensi il piacere
    # in arrivo, V = v·x; δ = piacere − V/orizzonte + dV/dt (dV/dt da una media
    # lenta di V). Impara con δ sulla traccia dei sensi; si riparte a ogni vita.
    c = p["critic"]
    v, seen, slow = c[:, :n_in], c[:, n_in:2 * n_in], c[:, 2 * n_in]
    value = (v * senses).sum(1)
    delta = reward - value / cfg["surprise_horizon"] + (value - slow) / cfg["surprise_smooth"]
    slow = slow + (dt / cfg["surprise_smooth"]) * (value - slow)
    fade = xp.exp(-dt / cfg["surprise_horizon"])
    seen = seen * fade + (1.0 - fade) * senses
    v = xp.clip(v + cfg["critic_rate"] * dt * delta[:, None] * seen, -10.0, 10.0)
    critic = xp.concatenate([v, seen, slow[:, None]], axis=1)
    m = m + ((1.0 - cfg["surprise"]) * reward + cfg["surprise"] * delta)[:, None]
    rate_pre = xp.concatenate([senses, frequencies(xp, pre[:, n_in:], centred)], axis=1)
    rate_post = frequencies(xp, post, centred)
    avg = p["avg"] + (dt / cfg["homeo_tau"]) * (rate_post - p["avg"])
    want, sign, links = p["target_act"], p["sign"], p["links"]
    excit = sign > 0.0
    gap = p["homeo"] * dt * (want - avg)  # (n, T): > 0 se il neurone è troppo zitto
    # Omeostasi: scala tutta la sinapsi (parte stabile e traccia).
    early = p["early"] * xp.exp(xp.where(excit, gap[:, :, None], -gap[:, :, None]))
    stable = xp.abs(p["w_eff"]) * xp.exp(xp.where(excit, gap[:, :, None], -gap[:, :, None])) - early
    bias = p["bias_now"] + p["intrinsic"] * dt * (want - avg)  # più corrente, più frequenza
    # Consolidamento della traccia dove arriva un modulatore, poi decadimento.
    caught = early * (1.0 - xp.exp(-cfg["capture_rate"] * xp.abs(m) * dt))[:, :, None]
    stable = stable + caught
    early = (early - caught) * p["early_keep"][:, :, None]
    strength = stable + early

    def saturate(d):  # i rafforzamenti rallentano vicino al massimo
        return xp.where(d > 0.0, d * xp.maximum(1.0 - strength / top, 0.0), d)

    theta = avg**2 / xp.maximum(want, 1e-3)
    bcm = (rate_post * (rate_post - theta))[:, :, None] * rate_pre[:, None, :]
    vogels = (rate_post - want)[:, :, None] * rate_pre[:, None, :]
    critical = 1.0 + (p["boost"] - 1.0) * xp.exp(-p["clock"][:, None] / p["closure"])  # periodo critico
    early = early + saturate((p["hebb"] * critical)[:, :, None] * xp.where(excit, bcm, vogels) * links)
    rule = p["rule"]
    drive = ((rule[..., 0] * post)[:, :, None] * pre[:, None, :] + rule[..., 1, None] * pre[:, None, :]
             + (rule[..., 2] * post + rule[..., 3])[:, :, None])
    # Traccia di eleggibilità (Izhikevich 2007): la regola propone, la traccia
    # ricorda la proposta per τ_e (media mobile: a regime vale la proposta), e
    # il modulatore la applica con il suo segno quando arriva, anche dopo.
    keep = p["elig_keep"][:, :, None]
    elig = p["elig"] * keep + (1.0 - keep) * drive * links
    stable = xp.clip(stable + saturate(sign * (p["eta"] * m)[:, :, None] * elig), 0.0, top)
    strength = xp.clip(stable + early, 0.0, top)
    return sign * strength, strength - stable, bias, avg, elig, critic
