"""La scala di difficoltà del mondo del cibo (docs/SPEC_MONDO.md, punto 3).

Le manopole (`forage_*` nella configurazione) salgono a stadi, uno alla volta:
prima abbondanza e cibi necessari insieme, poi il cibo che sfugge (prede che
scappano e nube d'odore che si forma e svanisce), poi i predatori, poi le
rocce (ostacoli e rifugi), poi i frutti (buoni o cattivi a seconda della
vita: si impara quale mangiare). Tutta la scala si riassume quindi in una
sola **posizione** x tra 0 (tutto basico) e 5 (tutto complesso): lo stadio i ha le sue manopole a
clip(x - i, 0, 1). Salire muove il primo stadio non ancora al massimo,
scendere l'ultimo non ancora a zero.

La progressione è adattiva e serve solo a non far estinguere la popolazione e
a non lasciarla in un mondo troppo facile (le soglie sono strumenti di
addestramento, non natura). A ogni generazione si guarda la quota di chi ha
completato il compito, mediata sulle ultime `ladder_window`:
  - sopra `ladder_up` la posizione sale di `ladder_step`;
  - sotto `ladder_down` scende di `ladder_step`;
  - se nell'ultima generazione nessuno ha completato scende di `ladder_crash`
    (e l'evoluzione riparte dagli ultimi genitori, come sempre).

Modi (`cfg.ladder`): "scala" = progressione adattiva; "basico" e
"complesso" = le due popolazioni di controllo, con le manopole ferme a 0 o a 1;
"" = nessuna scala (manopole come le ha messe chi ha fatto la corsa).
"""

import numpy as np

STAGES = (("forage_abundance", "forage_need"), ("forage_prey", "forage_scent"), ("forage_predators",),
          ("forage_obstacles",), ("forage_variety",))
DIALS = tuple(d for stage in STAGES for d in stage)
TOP = float(len(STAGES))
MODES = ("", "scala", "basico", "complesso")


def at(x: float) -> dict[str, float]:
    """Le manopole nella posizione x della scala."""
    return {d: round(float(np.clip(x - i, 0.0, 1.0)), 6) for i, stage in enumerate(STAGES) for d in stage}


def position(cfg) -> float:
    """La posizione della scala in cui stanno le manopole della
    configurazione (la media di ogni stadio, sommata)."""
    return float(sum(np.mean([getattr(cfg, d) for d in stage]) for stage in STAGES))


STAGE_NAMES = ("abbondanza e cibi necessari", "prede e odore che sfugge", "predatori", "rocce", "frutti")


def stage(cfg) -> tuple[int, float]:
    """Lo stadio in corso (0, 1, 2) e quanto è avanti, da 0 a 1."""
    x = position(cfg)
    i = min(int(x), len(STAGES) - 1)
    return i, x - i


def dials(cfg) -> dict[str, float]:
    return {d: float(getattr(cfg, d)) for d in DIALS}


def prepare(cfg) -> None:
    """Mette le manopole al loro posto per i controlli (ferme per sempre)."""
    if cfg.ladder not in MODES:
        raise ValueError(f"scala sconosciuta: {cfg.ladder!r} (valide: {', '.join(map(repr, MODES))})")
    if cfg.ladder == "basico":
        for d, v in at(0.0).items():
            setattr(cfg, d, v)
    elif cfg.ladder == "complesso":
        for d, v in at(TOP).items():
            setattr(cfg, d, v)


def recent_rate(cfg, history: list[dict]) -> float:
    """Quota di chi ha completato, mediata sulle ultime `ladder_window` generazioni."""
    window = history[-max(cfg.ladder_window, 1):]
    return float(np.mean([h["completed"] for h in window])) / cfg.population


def update(cfg, history: list[dict]) -> dict:
    """Muove le manopole dopo una generazione (`history` comprende già le
    sue statistiche). Restituisce i cambi {manopola: [prima, dopo]} e, se c'è
    stato un movimento, il motivo e la quota che l'ha deciso."""
    if cfg.ladder != "scala" or not history:
        return {}
    rate = recent_rate(cfg, history)
    if history[-1]["completed"] == 0:
        delta, why = -cfg.ladder_crash, "nessuno ha completato"
    elif rate > cfg.ladder_up:
        delta, why = cfg.ladder_step, f"riuscita media {rate:.0%} sopra {cfg.ladder_up:.0%}"
    elif rate < cfg.ladder_down:
        delta, why = -cfg.ladder_step, f"riuscita media {rate:.0%} sotto {cfg.ladder_down:.0%}"
    else:
        return {}
    x = position(cfg)
    new = at(float(np.clip(x + delta, 0.0, TOP)))
    changes = {d: [getattr(cfg, d), v] for d, v in new.items() if getattr(cfg, d) != v}
    for d, (_, v) in changes.items():
        setattr(cfg, d, v)
    if not changes:
        return {}
    return {"changes": changes, "why": why, "rate": rate, "position": position(cfg)}


def luck(start_dist: np.ndarray, done: np.ndarray) -> dict[str, float]:
    """Quanto pesa ancora la fortuna: quota di chi ha completato tra il quarto
    nato più vicino al cibo e il quarto nato più lontano. Se le due quote sono
    molto diverse, completare dipende soprattutto da dove si nasce."""
    lo, hi = np.quantile(start_dist, [0.25, 0.75])
    return {"near": float(done[start_dist <= lo].mean()), "far": float(done[start_dist >= hi].mean())}
