"""Il grembo: l'attività spontanea prima della nascita (docs/SPEC_SVILUPPO.md,
sezione 3 e Fase 2), solo con lo sviluppo.

In natura il cervello lavora prima di ricevere esperienza: la retina, la
coclea e il midollo producono **attività spontanea** (le onde della retina
scorrono sulle cellule vicine) e su questa attività Hebb e l'omeostasi
rifiniscono il cablaggio appena sviluppato. Qui, per `prenatal_time` secondi:

- i sensi ricevono **onde** che scorrono lungo l'ordine dei nervi (l'altezza a
  cui entrano nel cervello, `r_y`): due sensi vicini sono attivi insieme, due
  lontani no; più un rumore di fondo debole;
- la rete gira e agiscono Hebb, l'omeostasi delle sinapsi e della soglia; la
  plasticità modulata no (nel grembo non c'è ricompensa);
- la media dell'attività ā parte vuota e si costruisce con l'attività vera
  (media di tutto il grembo finché è più breve di `homeo_tau`);
- il corpo non si muove.

Alla nascita ciò che si è formato diventa stabile: il cervello rifinito
(`weights`), ā e la corrente di fondo vanno nel fenotipo dell'individuo
(`Genome.born_avg`, `born_bias`) e ogni vita riparte da lì.
"""

import numpy as np


def waves(cfg, r_y: np.ndarray, r_active: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(passi, N, R): le risposte dei sensi nel grembo. Per ogni individuo le
    onde partono a intervalli casuali (in media `wave_interval`) da un capo o
    dall'altro e scorrono a `wave_speed`; ogni senso risponde quando l'onda
    passa all'altezza del suo nervo."""
    steps = int(round(cfg.prenatal_time / cfg.dt))
    n = len(r_y)
    t = np.arange(steps) * cfg.dt
    k = int(np.ceil(cfg.prenatal_time / cfg.wave_interval)) * 3 + 2
    starts = np.cumsum(rng.exponential(cfg.wave_interval, (n, k)), axis=1) - cfg.wave_interval
    up = rng.random((n, k)) < 0.5
    current = np.maximum((t[:, None, None] >= starts[None]).sum(axis=2) - 1, 0)  # (passi, N): l'ultima onda partita
    rows = np.arange(n)[None, :]
    since = t[:, None] - starts[rows, current]
    travel = cfg.wave_speed * np.maximum(since, 0.0)
    span = 1.0 + 6.0 * cfg.wave_width
    front = np.where(up[rows, current], -3.0 * cfg.wave_width + travel, 1.0 + 3.0 * cfg.wave_width - travel)
    front = np.where((since >= 0) & (travel <= span), front, np.inf)
    obs = np.exp(-0.5 * ((r_y[None] - front[:, :, None]) / cfg.wave_width) ** 2)
    obs += cfg.prenatal_noise * rng.standard_normal(obs.shape)
    return np.clip(obs, 0.0, 1.0) * r_active[None]


def run(brains, rows: np.ndarray, obs: np.ndarray, engine: str = "gpu") -> tuple[np.ndarray, ...]:
    """Il grembo per gli individui `rows` di `brains` (appena compilati),
    sulla GPU se c'è. Restituisce (pesi, ā, corrente di fondo) alla nascita,
    (len(rows), ...)."""
    if engine == "gpu":
        try:
            from ..gpu import womb as gpu_womb
        except ImportError:  # senza JAX: il motore NumPy
            gpu_womb = None
        if gpu_womb is not None:
            return gpu_womb.run(brains, rows, obs)
    return run_numpy(brains, rows, obs)


def run_numpy(brains, rows: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, ...]:
    b, cfg = brains, brains.cfg
    tau, eta = cfg.homeo_tau, b.eta
    b.eta = np.zeros_like(eta)  # niente ricompensa: la plasticità modulata tace
    b.avg[rows] = 0.0
    try:
        for k in range(len(obs)):
            cfg.homeo_tau = min(tau, (k + 1) * cfg.dt)  # ā = media di tutto ciò che è successo finora
            b.step_some(obs[k][rows], rows)
    finally:
        cfg.homeo_tau, b.eta = tau, eta
    return b.w_eff[rows].copy(), b.avg[rows].copy(), b.bias_now[rows].copy()


def prenatal(brains, rows: np.ndarray, rng: np.random.Generator, engine: str = "gpu") -> None:
    """I neonati `rows` passano dal grembo; il risultato va nel loro fenotipo
    e il cervello si ricompila (ogni vita parte da lì)."""
    if not len(rows):
        return
    g = brains.genome
    obs = waves(brains.cfg, g.r_y, g.r_active, rng)
    w, avg, bias = run(brains, rows, obs, engine)
    g.weights[rows], g.born_avg[rows], g.born_bias[rows] = w, avg, bias
    brains.set_genome(g)
