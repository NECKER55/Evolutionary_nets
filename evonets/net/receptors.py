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

Adattamento (solo nel calcolo di `Prepared`, con lo stato `m`): il recettore
tiene una media recente m dei suoi quanti, con tempo caratteristico τ, e
    r = clip((ñ − sub · m) / (n_max + div · m), 0, 1)
Con sub = 1 a segnale costante la risposta torna a zero (risponde ai
cambiamenti, come i batteri); con div grande risponde al rapporto tra segnale
e media (al contrasto, come la retina). Con sub = div = 0 è come senza.
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
    tau      (N, R)     logaritmo del tempo di adattamento (secondi)
    sub      (N, R)     adattamento sottrattivo, da 0 a 1 (batteri: risponde ai cambiamenti)
    div      (N, R)     adattamento divisivo, da 0 in su (retina: risponde al contrasto)
    """

    active: np.ndarray
    sens: np.ndarray
    mu: np.ndarray
    width: np.ndarray
    body: np.ndarray
    look: np.ndarray
    aperture: np.ndarray
    brain_y: np.ndarray
    tau: np.ndarray
    sub: np.ndarray
    div: np.ndarray

    FIELDS = ("active", "sens", "mu", "width", "body", "look", "aperture", "brain_y", "tau", "sub", "div")

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


SHORT = ("luce", "sost.", "mecc.", "temp.", "magn.", "int.")


def dominant(genes: ReceptorGenes, i: int, k: int) -> tuple[int, float]:
    """Il fenomeno a cui il recettore k dell'individuo i è più sensibile, e
    quale quota della sua sensibilità totale gli dedica."""
    g = np.exp(genes.sens[i, k])
    p = int(np.argmax(g))
    return p, float(g[p] / g.sum())


def label(genes: ReceptorGenes, i: int, k: int) -> str:
    """Nome breve: il fenomeno dominante e la sintonia, oppure "generalista"."""
    p, share = dominant(genes, i, k)
    if share < 0.8:
        return f"r{k} generalista"
    if p == MAGNETIC:
        return f"r{k} {SHORT[p]}"
    return f"r{k} {SHORT[p]} {genes.mu[i, k, p]:.2f}"


def _deg(v) -> float:
    """Angolo in gradi di un vettore nel riferimento del muso; positivo = destra."""
    return float(np.degrees(np.arctan2(v[1], v[0])))


def describe(genes: ReceptorGenes, i: int, k: int) -> list[str]:
    """Tutti i geni del recettore k dell'individuo i, in righe brevi."""
    g = np.exp(genes.sens[i, k])
    total = g.sum()
    lines = [
        f"sul corpo a {_deg(genes.body[i, k]):+.0f}° (+ = destra)",
        f"guarda a {_deg(genes.look[i, k]):+.0f}°, apertura {np.degrees(genes.aperture_rad[i, k]):.0f}°",
        f"sensibilità totale {total:.1f}",
        f"adattamento: sottr. {genes.sub[i, k]:.2f}, div. {genes.div[i, k]:.2f}, "
        f"in {np.exp(genes.tau[i, k]):.2g}s",
        "fenomeno  g  quota  sintonia",
    ]
    for p in np.argsort(-g):
        tune = "" if p == MAGNETIC else f" {genes.mu[i, k, p]:.2f}±{genes.sigma[i, k, p]:.2f}"
        lines.append(f"  {SHORT[p]:6s}{g[p]:6.1f} {g[p] / total:4.0%}{tune}")
    return lines


class Prepared:
    """I recettori di un gruppo di individui pronti per il calcolo veloce.

    Tutto ciò che dipende solo dai geni (sintonie, quote di luce degli
    oggetti, sensibilità) si calcola una volta; a ogni passo resta solo ciò che
    dipende dal mondo. Il risultato è lo stesso di `respond` (lo controlla un
    test). Va rifatto se cambiano i geni o i colori e le emissioni degli
    oggetti nella configurazione.
    """

    def __init__(self, cfg, genes: ReceptorGenes):
        self.cfg, self.active = cfg, genes.active
        g, mu, sg = genes.gain, genes.mu, genes.sigma
        self.keep = np.exp(-cfg.dt / np.exp(genes.tau))  # quanto della media resta a ogni passo
        self.sub, self.div = genes.sub, genes.div
        self.n, self.R = genes.active.shape
        self.body = genes.body
        m = cfg.light_samples
        offsets = (np.arange(m) + 0.5) / m - 0.5
        centre = np.arctan2(genes.look[..., 1], genes.look[..., 0])  # rispetto al muso
        self.centre, self.half = centre, genes.aperture_rad / 2.0
        self.angles = centre[..., None] + genes.aperture_rad[..., None] * offsets  # (n, R, M)
        L, C, M = g[..., LIGHT], g[..., CHEMICAL], g[..., MECHANICAL]
        self.c_wall = L * spectral(mu[..., LIGHT], sg[..., LIGHT], cfg.wall_color, cfg.wall_color_width,
                                   cfg.wall_albedo)
        self.c_target = L * spectral(mu[..., LIGHT], sg[..., LIGHT], cfg.target_color, cfg.target_color_width,
                                     cfg.target_albedo)
        self.c_glow = L * spectral(mu[..., LIGHT], sg[..., LIGHT], cfg.hazard_color, cfg.hazard_color_width,
                                   cfg.hazard_glow)
        self.mu_c, self.sg_c, self.g_c = mu[..., CHEMICAL], sg[..., CHEMICAL], C
        self.c_press = M * tuning(0.0, mu[..., MECHANICAL], sg[..., MECHANICAL])
        self.mu_m, self.sg_m, self.g_m = mu[..., MECHANICAL], sg[..., MECHANICAL], M
        self.mu_t, self.sg_t, self.g_t = mu[..., THERMAL], sg[..., THERMAL], g[..., THERMAL]
        self.g_mag = g[..., MAGNETIC]
        q = np.array([qk for qk, _ in signals.BODY_QUANTITIES])
        self.c_body = g[..., INTERNAL, None] * tuning(q, mu[..., INTERNAL, None], sg[..., INTERNAL, None])

    def take(self, rows) -> "Prepared":
        new = object.__new__(Prepared)
        for k, v in self.__dict__.items():
            new.__dict__[k] = v[rows] if isinstance(v, np.ndarray) else v
        new.n = len(rows)
        return new

    def quanta(self, world, rows) -> np.ndarray:
        """(n, R) quanti attesi n = Σ_p g_p · s_p, senza rumore."""
        cfg, n, R = self.cfg, self.n, self.R
        heading = world.heading[rows][:, None]
        cos, sin = np.cos(heading), np.sin(heading)
        normal = _rotate(self.body, cos, sin)
        points = (world.pos[rows][:, None, :] + cfg.agent_radius * normal).reshape(-1, 2)
        seen = signals.light(cfg, world, points, (heading[..., None] + self.angles).reshape(n * R, -1))
        out = (self.c_wall * seen["parete"].reshape(n, R) + self.c_target * seen["bersaglio"].reshape(n, R)
               + self.c_glow * seen["veleno"].reshape(n, R))
        for molecule, conc in signals.chemicals(cfg, world, points):
            out += self.g_c * tuning(molecule, self.mu_c, self.sg_c) * conc.reshape(n, R)
        if world.bump[rows].any():
            agent = np.repeat(rows, R)
            out += self.c_press * signals.pressure(cfg, world, agent, normal.reshape(-1, 2)).reshape(n, R)
        vib = signals.vibrations(cfg, world, points)
        if vib:
            centre = np.arctan2(sin, cos) + self.centre
            for freq, amp, came_from in vib:
                off = np.abs((came_from.reshape(n, R) - centre + np.pi) % (2.0 * np.pi) - np.pi)
                out += self.g_m * tuning(freq, self.mu_m, self.sg_m) * amp.reshape(n, R) * (off <= self.half)
        temp = signals.temperature(cfg, world, points).reshape(n, R)
        out += self.g_t * tuning(temp, self.mu_t, self.sg_t)
        field = signals.magnetic(cfg)
        if field.any():
            look = np.arctan2(sin, cos) + self.centre
            out += self.g_mag * np.maximum(0.0, np.cos(look) * field[0] + np.sin(look) * field[1])
        out += np.einsum("nrk,nk->nr", self.c_body, signals.body(cfg, world, rows))
        return out

    def respond(self, world, rows, rng: np.random.Generator, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(n, R) risposte in [0, 1] con rumore, saturazione e adattamento, e
        la nuova media `m` (nan = recettore appena nato: la media parte dalla
        prima lettura, così nessuno nasce "abbagliato")."""
        cfg = self.cfg
        n = self.quanta(world, rows)
        noisy = np.maximum(n + np.sqrt(n + cfg.receptor_dark) * rng.standard_normal(n.shape), 0.0)
        m = np.where(np.isnan(m), noisy, m)
        r = (noisy - self.sub * m) / (cfg.receptor_saturation + self.div * m)
        m = self.keep * m + (1.0 - self.keep) * noisy
        return np.clip(r, 0.0, 1.0) * self.active, m
