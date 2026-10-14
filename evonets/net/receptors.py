"""Il recettore universale: un solo tipo di cellula sensoriale, i cui geni
decidono a cosa è sensibile (vedi docs/SPEC_FASE_C.md, sezione 3).

Ogni recettore ha, per ciascuno dei 6 fenomeni fisici, una sensibilità e una
sintonia (una campana sull'asse della qualità: colore, molecola, frequenza,
temperatura, grandezza del corpo). Nasce generalista e si specializza se
conviene: un recettore sensibile a tutto somma segnali diversi in un numero
solo, e ogni sensibilità costa.

La risposta conta "quanti" con rumore, come i fotorecettori:
    n = Σ_p g_p · s_p              quanti attesi
    ñ = n + sqrt(n + n_buio) · ε   ε ~ N(0, 1)
    r = clip(ñ / n_max, 0, 1)
Più sensibilità = meno rumore relativo, ma saturazione prima e più costo.
"""

from dataclasses import dataclass

import numpy as np

from ..sim import signals
from ..sim.signals import CHEMICAL, INTERNAL, LIGHT, MAGNETIC, MECHANICAL, PHENOMENA, THERMAL

P = len(PHENOMENA)
MIN_APERTURE = np.radians(1.0)
MAX_APERTURE = 2.0 * np.pi


@dataclass
class ReceptorGenes:
    """I recettori di un gruppo di individui: N individui, R posti ciascuno.

    active   (N, R)     se il recettore esiste
    sens     (N, R, P)  logaritmo della sensibilità per fenomeno: g = exp(sens)
    mu       (N, R, P)  centro della sintonia sull'asse della qualità
    width    (N, R, P)  logaritmo della larghezza della sintonia
    body     (N, R, 2)  punto del corpo, vettore unitario nel riferimento del muso
    look     (N, R, 2)  direzione in cui guarda, vettore unitario nel riferimento del muso
    aperture (N, R)     logaritmo dell'apertura (radianti)
    brain_y  (N, R)     dove entra il suo nervo nel cervello (x = 0)
    """

    active: np.ndarray
    sens: np.ndarray
    mu: np.ndarray
    width: np.ndarray
    body: np.ndarray
    look: np.ndarray
    aperture: np.ndarray
    brain_y: np.ndarray

    FIELDS = ("active", "sens", "mu", "width", "body", "look", "aperture", "brain_y")

    def __len__(self) -> int:
        return len(self.active)

    def take(self, idx) -> "ReceptorGenes":
        return ReceptorGenes(*(getattr(self, f)[idx].copy() for f in self.FIELDS))

    @property
    def gain(self) -> np.ndarray:
        """(N, R, P) sensibilità g, zero per i recettori spenti."""
        return np.exp(self.sens) * self.active[:, :, None]

    @property
    def sigma(self) -> np.ndarray:
        return np.exp(self.width)

    @property
    def aperture_rad(self) -> np.ndarray:
        return np.clip(np.exp(self.aperture), MIN_APERTURE, MAX_APERTURE)


def tuning(q, mu, sigma):
    """La campana della sintonia, con picco 1."""
    return np.exp(-0.5 * ((q - mu) / sigma) ** 2)


def spectral(mu, sigma, color, color_width, albedo):
    """Quota della luce di un oggetto (spettro: campana di centro `color` e
    larghezza `color_width`) che un recettore di sintonia (mu, sigma)
    raccoglie. È l'integrale esatto del prodotto delle due campane, diviso per
    l'area dello spettro: 1 se la sintonia copre tutto lo spettro."""
    s2 = sigma ** 2 + color_width ** 2
    return albedo * sigma / np.sqrt(s2) * np.exp(-0.5 * (mu - color) ** 2 / s2)


def _rotate(v, cos, sin):
    """Ruota i vettori v (..., 2) dal riferimento del muso a quello del mondo."""
    return np.stack([v[..., 0] * cos - v[..., 1] * sin, v[..., 0] * sin + v[..., 1] * cos], axis=-1)


def stimuli(cfg, world, genes: ReceptorGenes, rows=None) -> np.ndarray:
    """(n, R, P): il segnale s_p che ogni recettore riceve da ogni fenomeno,
    già pesato dalla sua sintonia, senza sensibilità né rumore. Con `rows`
    solo per quegli agenti (genes ha già una riga per ciascuno di essi)."""
    rows = np.arange(world.n) if rows is None else rows
    n, R = genes.active.shape
    mu, sigma = genes.mu, genes.sigma
    heading = world.heading[rows][:, None]
    cos, sin = np.cos(heading), np.sin(heading)
    normal = _rotate(genes.body, cos, sin)  # (n, R, 2) nel mondo
    points = world.pos[rows][:, None, :] + cfg.agent_radius * normal
    flat = points.reshape(-1, 2)
    agent = np.repeat(rows, R)
    out = np.zeros((n, R, P))

    # Luce: direzioni campione distribuite nell'apertura.
    look = _rotate(genes.look, cos, sin)
    centre = np.arctan2(look[..., 1], look[..., 0])
    m = cfg.light_samples
    offsets = (np.arange(m) + 0.5) / m - 0.5
    angles = centre[..., None] + genes.aperture_rad[..., None] * offsets  # (n, R, M)
    seen = signals.light(cfg, world, flat, angles.reshape(-1, m))
    mu_l, sg_l = mu[..., LIGHT], sigma[..., LIGHT]
    out[..., LIGHT] = (
        spectral(mu_l, sg_l, cfg.wall_color, cfg.wall_color_width, cfg.wall_albedo) * seen["parete"].reshape(n, R)
        + spectral(mu_l, sg_l, cfg.target_color, cfg.target_color_width, cfg.target_albedo)
        * seen["bersaglio"].reshape(n, R)
        + spectral(mu_l, sg_l, cfg.hazard_color, cfg.hazard_color_width, cfg.hazard_glow)
        * seen["veleno"].reshape(n, R)
    )

    # Sostanze, nel punto del corpo dove sta il recettore.
    for molecule, conc in signals.chemicals(cfg, world, flat):
        out[..., CHEMICAL] += tuning(molecule, mu[..., CHEMICAL], sigma[..., CHEMICAL]) * conc.reshape(n, R)

    # Meccanica: pressione (frequenza 0) e vibrazioni dentro l'apertura.
    mu_m, sg_m = mu[..., MECHANICAL], sigma[..., MECHANICAL]
    press = signals.pressure(cfg, world, agent, normal.reshape(-1, 2)).reshape(n, R)
    out[..., MECHANICAL] = tuning(0.0, mu_m, sg_m) * press
    half = genes.aperture_rad / 2.0
    for freq, amp, came_from in signals.vibrations(cfg, world, flat):
        off = np.abs((came_from.reshape(n, R) - centre + np.pi) % (2.0 * np.pi) - np.pi)
        out[..., MECHANICAL] += tuning(freq, mu_m, sg_m) * amp.reshape(n, R) * (off <= half)

    # Temperatura: la sintonia è sulla temperatura stessa.
    temp = signals.temperature(cfg, world, flat).reshape(n, R)
    out[..., THERMAL] = tuning(temp, mu[..., THERMAL], sigma[..., THERMAL])

    # Magnetismo: una bussola, senza asse della qualità.
    field = signals.magnetic(cfg)
    out[..., MAGNETIC] = np.maximum(0.0, look[..., 0] * field[0] + look[..., 1] * field[1])

    # Interno: le grandezze del corpo, pesate dalla sintonia.
    values = signals.body(cfg, world, rows)  # (n, 7)
    for k, (q, _) in enumerate(signals.BODY_QUANTITIES):
        out[..., INTERNAL] += tuning(q, mu[..., INTERNAL], sigma[..., INTERNAL]) * values[:, k : k + 1]
    return out


def respond(cfg, world, genes: ReceptorGenes, rng: np.random.Generator, rows=None) -> np.ndarray:
    """(n, R) risposta di ogni recettore in [0, 1], con rumore e saturazione.
    I recettori spenti danno 0."""
    s = stimuli(cfg, world, genes, rows)
    n = (genes.gain * s).sum(axis=2)
    noisy = n + np.sqrt(n + cfg.receptor_dark) * rng.standard_normal(n.shape)
    return np.clip(noisy / cfg.receptor_saturation, 0.0, 1.0) * genes.active
