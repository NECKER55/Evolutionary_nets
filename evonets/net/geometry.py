"""Geometria del cervello: dove stanno i neuroni e quanto è probabile che due
neuroni si colleghino.

Il cervello è un foglio quadrato, come un pezzo di corteccia, e **cresce con le
sue cellule**: la superficie è proporzionale al numero di cellule accese
(recettori, neuroni interni, motori), a densità costante `brain_density` (100
cellule per unità di superficie: con 100 cellule il lato è 1). Un organismo
primitivo con poche cellule ha sensi e motori vicini, come la rete nervosa di
un'idra; un cervello grande li ha lontani, e i collegamenti lunghi costano di
più. Le posizioni nei geni sono in [0, 1] e vengono scalate al lato vero.
- I recettori entrano dal bordo sinistro (x = 0), all'altezza data dal loro
  gene `r_y`: all'inizio segue la direzione in cui guardano (una mappa
  topografica), poi deriva con le mutazioni.
- I motori stanno in posizioni fisse sul bordo destro (x = 1).
- I neuroni interni stanno dove dice il loro gene di posizione.

Come nella corteccia, la probabilità che esista un collegamento cala in modo
esponenziale con la distanza d tra i due neuroni:

    p(d) = net_conn_prob * exp(-d / λ)

dove λ, la portata tipica dei collegamenti, è un gene di ogni individuo. I
collegamenti lunghi sono rari ma possibili. Un neurone che si collega a se
stesso (d = 0) fa eccezione: in natura è raro, e ha la sua probabilità
`net_self_prob`.
"""

import numpy as np


def motor_anchors(n_out: int) -> np.ndarray:
    """Posizioni fisse (n_out, 2) dei motori, sul bordo destro."""
    return np.stack([np.ones(n_out), (np.arange(n_out) + 0.5) / n_out], axis=1)


def receptor_anchors(r_y: np.ndarray) -> np.ndarray:
    """(N, R, 2): dove entrano i nervi dei recettori, sul bordo sinistro."""
    return np.stack([np.zeros(r_y.shape), r_y], axis=-1)


def brain_side(cfg, r_active: np.ndarray, active: np.ndarray, n_out: int) -> np.ndarray:
    """(N,) lato del cervello di ogni individuo: √(cellule accese / densità)."""
    cells = r_active.sum(axis=1) + active.sum(axis=1) + n_out
    return np.sqrt(cells / cfg.brain_density)


def distances(cfg, pos: np.ndarray, r_y: np.ndarray, n_out: int, side: np.ndarray | None = None) -> np.ndarray:
    """pos (N, H, 2) posizioni degli interni, r_y (N, R) dei recettori ->
    (N, T, S) distanza tra ogni destinatario (interni, motori) e ogni sorgente
    (recettori, interni, motori), nel cervello di lato `side` (N,) (1 se non
    dato)."""
    n = len(pos)
    motors = np.broadcast_to(motor_anchors(n_out), (n, n_out, 2))
    targets = np.concatenate([pos, motors], axis=1)
    sources = np.concatenate([receptor_anchors(r_y), targets], axis=1)
    d = np.linalg.norm(targets[:, :, None, :] - sources[:, None, :, :], axis=3)
    return d if side is None else d * side[:, None, None]


def connect_prob(cfg, dist: np.ndarray, n_rec: int, reach: np.ndarray) -> np.ndarray:
    """(N, T, S): probabilità che un collegamento esista alla nascita, con la
    portata `reach` (N,) di ogni individuo. Le mutazioni che aggiungono
    collegamenti seguono la stessa forma (vedi `genetics.mutate`)."""
    p = cfg.net_conn_prob * np.exp(-dist / reach[:, None, None])
    t = np.arange(dist.shape[1])
    p[:, t, n_rec + t] = cfg.net_self_prob  # il neurone con se stesso
    return p
