"""O e K: il valore degli oggetti e i contesti del cervello a sezioni."""

import numpy as np

from .genes import IDENTITY, OBS_AVG, RATE_MAX_O, VOLATILITY_MAX, WEIGHT_MAX

# --- O: il valore degli oggetti (vedi la docstring di `sections.py`) ---------------------


def o_read(o_pl, o_pa, o_assoc, feat, energy, p: dict, xp=np):
    """Ciò che O dice di oggetti con caratteristiche `feat` (n, ..., F):
    (valore, importanza, incertezza), ognuno (n, ...). Lo sfondo (l'ultimo
    peso) non entra: è ciò che capita senza un oggetto. Il dolore pesa
    `pain_weight` volte il piacere. `o_assoc` (n, 2, F + 1): la sorpresa di
    adesso (l'incertezza) e quella solita."""
    F = feat.shape[-1]
    need = 1.0 + p["need"] * (2.0 * (1.0 - energy) - 1.0)  # la fame di adesso
    pain = p["pain_weight"][:, None] * o_pa[:, :F]
    value = o_pl[:, :F] * need[:, None] - pain
    weight = xp.maximum(o_pl[:, :F], 0.0) + xp.maximum(pain, 0.0)
    o_assoc = o_assoc[:, 0]
    extra = (1,) * (feat.ndim - 2)
    shape = lambda w: w.reshape(w.shape[:1] + extra + w.shape[1:])
    return ((feat * shape(value)).sum(-1), (feat * shape(weight)).sum(-1), (feat * shape(o_assoc[:, :F])).sum(-1))


def o_learn(o_pl, o_pa, o_assoc, trace, pleasure, pain, rate, p: dict, xp=np):
    """Rescorla-Wagner alla fine di un episodio: `trace` (n, F + 1) ciò che si
    aveva in mente (l'ultimo è lo sfondo, sempre 1), `pleasure` e `pain` (n,)
    l'esito, `rate` (n,) la velocità (zero: non si impara). Le
    caratteristiche competono sulla stessa previsione. -> (o_pl, o_pa,
    o_assoc, |errore|)."""
    err_pl = pleasure - (o_pl * trace).sum(1)
    err_pa = pain - (o_pa * trace).sum(1)
    # Volatilità (Behrens e colleghi, 2007): si impara più in fretta quando
    # le sorprese di adesso superano quelle solite (il mondo è cambiato), più
    # piano quando il mondo è stabile.
    now, usual = o_assoc[:, 0], o_assoc[:, 1]
    boost = xp.clip(now / xp.maximum(usual, 0.05), 0.5, VOLATILITY_MAX)
    # Diviso per quanta traccia c'è: con molte caratteristiche accese insieme
    # la correzione totale supererebbe l'errore, e i pesi rimbalzerebbero.
    step = xp.minimum(rate[:, None] * boost, RATE_MAX_O) * trace / (1.0 + (trace**2).sum(1))[:, None]
    o_pl = xp.clip(o_pl + step * err_pl[:, None], -WEIGHT_MAX, WEIGHT_MAX)
    o_pa = xp.clip(o_pa + step * err_pa[:, None], -WEIGHT_MAX, WEIGHT_MAX)
    size = xp.minimum(xp.abs(err_pl) + xp.abs(err_pa), 2.0)[:, None]
    on = (rate > 0)[:, None]
    now = xp.where(on, now + p["assoc_rate"][:, None] * trace * (size - now), now)
    usual = xp.where(on, usual + p["assoc_slow"][:, None] * trace * (size - usual), usual)
    return o_pl, o_pa, xp.stack([now, usual], 1), xp.abs(err_pl) + xp.abs(err_pa)


def o_episodes(st: dict, felt, xp=np) -> tuple[dict, object]:
    """Gli episodi: comincia quando il corpo sente piacere o dolore (`felt`
    (n, 3)), finisce quando smette. All'inizio si fotografa la traccia (ciò che
    si aveva in mente prima dell'esito). -> (stato nuovo, finito (n,)); i
    totali dell'episodio finito restano in `ep_pl`, `ep_pa`, `ep_e` di `st`."""
    active = felt[:, 0] + felt[:, 1] > 1e-6
    done = st["ep_on"] & ~active
    start = active & ~st["ep_on"]
    n = felt.shape[0]
    snap = xp.concatenate([st["o_e"], xp.ones((n, 1))], 1)
    keep = st["ep_on"] & active
    new = {"ep_on": active,
           "ep_e": xp.where(start[:, None], snap, st["ep_e"]),
           "ep_pl": xp.where(keep, st["ep_pl"], 0.0) + felt[:, 0] * active,
           "ep_pa": xp.where(keep, st["ep_pa"], 0.0) + felt[:, 1] * active}
    return new, done


def o_active(st: dict, xp=np):
    """Il cassetto di O del contesto attivo: (o_pl, o_pa, o_assoc)."""
    idx = xp.arange(st["k_on"].shape[0])
    return st["o_pl"][idx, st["k_on"]], st["o_pa"][idx, st["k_on"]], st["o_assoc"][idx, st["k_on"]]


def k_learn(st: dict, trace, pleasure, pain, rate, mask, p: dict, les, xp=np) -> dict:
    """Un episodio finito (`mask` (n,)) insegna a K e a O. Prima si decide a
    quale contesto appartiene (la prova di ognuno; si cambia o si recluta un
    contesto nuovo, che parte dai valori innati), poi impara solo quel
    cassetto: così i primi esiti di un regime nuovo non rovinano il cassetto
    vecchio (vedi la docstring di `sections.py`). -> stato nuovo e |errore| del
    contesto scelto."""
    n, NK = st["k_ev"].shape
    idx = xp.arange(n)
    on, used = st["k_on"], st["k_used"]
    o_pl, o_pa, o_assoc = st["o_pl"], st["o_pa"], st["o_assoc"]
    # La prova: quanto bene ogni contesto avrebbe previsto questo esito.
    err_all = (xp.abs(pleasure[:, None] - (o_pl * trace[:, None, :]).sum(-1))
               + xp.abs(pain[:, None] - (o_pa * trace[:, None, :]).sum(-1)))
    ev = xp.where(mask[:, None] & used, st["k_ev"] + p["k_rate"][:, None] * (xp.minimum(err_all, 2.0) - st["k_ev"]),
                  st["k_ev"])
    since = st["k_since"] + mask
    k_on, stale = on, st["k_stale"]
    if "K" not in les:
        hot = xp.arange(NK)[None] == on[:, None]
        stale = xp.where(mask[:, None] & ~hot, stale + 1.0, xp.where(hot, 0.0, stale))
        ev_used = xp.where(used, ev, 1e9)
        best = xp.argmin(ev_used, axis=1)
        switch = mask & (ev_used[idx, best] < ev[idx, on] - p["k_stick"]) & (best != on)
        # Nessuno spiega: una causa nuova (il primo contesto libero, o il meno usato di recente),
        # che parte dai valori innati. Solo se nessun contesto esistente spiega
        # nemmeno questo esito: se uno lo spiega, è la sua prova che deve vincere.
        limit = p["k_new"] * xp.maximum(st["k_usual"], 0.05)
        none_now = xp.where(used, err_all, 1e9).min(1) > limit
        recruit = mask & (ev[idx, on] > limit) & none_now & (since >= p["k_wait"])
        slot = xp.argmax(xp.where(used, stale, 1e9) * ~hot, axis=1)
        new_hot = (xp.arange(NK)[None] == slot[:, None]) & recruit[:, None]
        o_pl = xp.where(new_hot[:, :, None], st["o_prior"][:, None, 0], o_pl)
        o_pa = xp.where(new_hot[:, :, None], st["o_prior"][:, None, 1], o_pa)
        o_assoc = xp.where(new_hot[:, :, None, None], 0.5, o_assoc)
        ev = xp.where(new_hot, st["k_usual"][:, None], ev)
        used = used | new_hot
        k_on = xp.where(recruit, slot, xp.where(switch, best, on))
        changed = k_on != on
        since = xp.where(changed, 0.0, since)
        stale = xp.where(changed[:, None] & (xp.arange(NK)[None] == k_on[:, None]), 0.0, stale)
    # Impara solo il cassetto scelto.
    a_pl, a_pa, a_as, err = o_learn(o_pl[idx, k_on], o_pa[idx, k_on], o_assoc[idx, k_on], trace, pleasure, pain, rate,
                                    p, xp)
    hot = xp.arange(NK)[None] == k_on[:, None]
    o_pl = xp.where(hot[:, :, None], a_pl[:, None], o_pl)
    o_pa = xp.where(hot[:, :, None], a_pa[:, None], o_pa)
    o_assoc = xp.where(hot[:, :, None, None], a_as[:, None], o_assoc)
    usual = xp.where(mask, st["k_usual"] + p["assoc_slow"] * (xp.minimum(err, 2.0) - st["k_usual"]), st["k_usual"])
    return {"o_pl": o_pl, "o_pa": o_pa, "o_assoc": o_assoc, "k_ev": ev, "k_used": used, "k_on": k_on,
            "k_usual": usual, "k_since": since, "k_stale": stale}, err


def o_step(st: dict, felt, energy, master, on, p: dict, les, xp=np) -> dict:
    """Un passo di O e K, prima dell'attenzione: chiude gli episodi finiti e
    impara dai propri e (chi osserva, `on` (n,)) da quelli del maestro del
    proprio recinto (`master` (n,): il suo indice). -> gli array cambiati."""
    if "O" in les:
        return {}
    new, done = o_episodes(st, felt, xp)
    pl, pa, e = st["ep_pl"], st["ep_pa"], st["ep_e"]
    k_out, err = k_learn(st, e, pl, pa, p["eta_o"] * done, done, p, les, xp)
    out = dict(new, o_err=xp.where(done, err, st["o_err"]), **k_out)
    # Dal maestro: la stessa regola, con ciò che lui aveva in mente e il suo esito.
    seen = on & done[master]
    k_out, err_m = k_learn(dict(st, **k_out), e[master], pl[master], pa[master], p["eta_o"] * p["observe"] * seen, seen,
                           p, les, xp)
    out.update(k_out, obs_err=xp.where(seen, st["obs_err"] + OBS_AVG * (err_m - st["obs_err"]), st["obs_err"]))
    return out


def o_trace(o_e, fov, m, p: dict, dt: float, xp=np):
    """La traccia di O, dopo l'attenzione e D: ciò che si guarda o si ricorda
    di aver preso si accende, poi si spegne in `trace_o` secondi."""
    now = xp.maximum(fov[:, :IDENTITY], m[:, :IDENTITY])
    return xp.maximum(o_e * xp.exp(-dt / p["trace_o"])[:, None], now)
