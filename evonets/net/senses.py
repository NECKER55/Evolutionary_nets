"""Sensi fissi, decisi da noi, per la simulazione "solo cervello"
(`Config.sim_mode = "cervello"`): tutti gli individui hanno gli stessi
recettori, che non mutano, non si duplicano e non si rompono. Evolve solo il
cervello, che deve imparare a usarli per sopravvivere.

Ogni senso è un recettore universale (net/receptors.py) specializzato su un
solo fenomeno: sensibilità piena su quello, nulla sugli altri. Gli angoli sono
in gradi, positivi verso destra; la sensibilità è un moltiplicatore della
cellula normale (1); l'adattamento è (τ in secondi, sottrattivo, divisivo).
Il costo in energia è quello di sempre (`receptor_cell_cost`,
`receptor_sens_cost`): con il set naturale circa 0,6 energia/s.
"""

from dataclasses import dataclass

import numpy as np

from ..sim.signals import (
    BODY_QUANTITIES,
    CHEMICAL,
    INTERNAL,
    LIGHT,
    MECHANICAL,
    PHENOMENA,
)

QUALITY = {name: q for q, name in BODY_QUANTITIES}


@dataclass(frozen=True)
class Sense:
    name: str
    phenomenon: int
    quality: float  # centro della sintonia (colore, molecola, frequenza, grandezza del corpo)
    width: float  # larghezza della sintonia
    gain: float  # sensibilità, in cellule normali
    body: float = 0.0  # dove sta sul corpo (gradi dal muso)
    look: float = 0.0  # dove guarda (gradi dal muso)
    aperture: float = 360.0  # gradi
    adapt: tuple = (1.0, 0.0, 0.0)  # τ (s), sottrattivo, divisivo
    nerve: float = 0.5  # dove entra il nervo nel cervello (0-1)


NOSE = (0.3, 0.8, 0.75)  # come il naso evoluto in corsa-20261020-200723: risponde ai cambiamenti
EYE = (1.0, 0.2, 0.5)  # adattamento leggero: risponde soprattutto al contrasto

SETS = {
    "naturale": (
        Sense("naso sinistro", CHEMICAL, 0.6, 0.10, 15.0, body=-30, look=-30, aperture=330, adapt=NOSE, nerve=0.35),
        Sense("naso destro", CHEMICAL, 0.6, 0.10, 15.0, body=30, look=30, aperture=330, adapt=NOSE, nerve=0.65),
        Sense("occhio rosso sinistro", LIGHT, 0.85, 0.08, 10.0, body=-15, look=-25, aperture=60, adapt=EYE,
              nerve=0.4),
        Sense("occhio rosso destro", LIGHT, 0.85, 0.08, 10.0, body=15, look=25, aperture=60, adapt=EYE, nerve=0.6),
        Sense("occhio blu sinistro", LIGHT, 0.05, 0.08, 6.0, body=-90, look=-90, aperture=120, adapt=EYE,
              nerve=0.2),
        Sense("occhio blu destro", LIGHT, 0.05, 0.08, 6.0, body=90, look=90, aperture=120, adapt=EYE, nerve=0.8),
        Sense("vibrazioni", MECHANICAL, 0.4, 0.1, 6.0, adapt=(1.0, 0.5, 0.0), nerve=0.1),
        Sense("contatto", MECHANICAL, 0.0, 0.1, 2.0, aperture=180, nerve=0.5),
        Sense("energia", INTERNAL, QUALITY["energia"], 0.04, 4.0, nerve=0.95),
        Sense("velocità", INTERNAL, QUALITY["avanti"], 0.04, 2.0, nerve=0.05),
        Sense("gusto", INTERNAL, QUALITY["gusto"], 0.03, 4.0, nerve=0.85),
        Sense("dolore", INTERNAL, QUALITY["dolore"], 0.03, 4.0, nerve=0.15),
    ),
}
OFF = -30.0  # sensibilità e^-30 sugli altri fenomeni: nulla


def senses(cfg) -> tuple[Sense, ...]:
    if cfg.sensor_set not in SETS:
        raise ValueError(f"set di sensi sconosciuto: {cfg.sensor_set!r} (validi: {', '.join(SETS)})")
    return SETS[cfg.sensor_set]


def genes(cfg, n: int) -> dict[str, np.ndarray]:
    """I geni dei recettori del set, uguali per `n` individui (un posto per senso)."""
    s = senses(cfg)
    R, P = len(s), len(PHENOMENA)

    def unit(deg):
        a = np.radians(np.array(deg, dtype=float))
        return np.stack([np.cos(a), np.sin(a)], axis=1)

    sens = np.full((R, P), OFF)
    mu = np.full((R, P), 0.5)
    width = np.full((R, P), np.log(cfg.receptor_initial_width))
    for k, x in enumerate(s):
        sens[k, x.phenomenon] = np.log(x.gain)
        mu[k, x.phenomenon] = x.quality
        width[k, x.phenomenon] = np.log(x.width)
    one = {
        "r_active": np.ones(R, dtype=bool), "r_sens": sens, "r_mu": mu, "r_width": width,
        "r_body": unit([x.body for x in s]), "r_look": unit([x.look for x in s]),
        "r_aperture": np.log(np.radians([x.aperture for x in s])),
        "r_y": np.array([x.nerve for x in s]),
        "r_tau": np.log([x.adapt[0] for x in s]), "r_sub": np.array([x.adapt[1] for x in s]),
        "r_div": np.array([x.adapt[2] for x in s]),
    }
    return {k: np.broadcast_to(v, (n,) + v.shape).copy() for k, v in one.items()}
