"""Geometria del cervello: dove stanno i neuroni e quanto è probabile che due
neuroni si colleghino.

Il cervello è un foglio quadrato di lato 1, come un pezzo di corteccia.
- I sensi stanno in posizioni fisse sul bordo sinistro (x = 0), disposti come
  la retina: i raggi dall'alto in basso nell'ordine del campo visivo, i canali
  di uno stesso raggio vicini tra loro; sotto, i sensi del corpo.
- I motori stanno in posizioni fisse sul bordo destro (x = 1).
- I neuroni interni stanno dove dice il loro gene di posizione.

Come nella corteccia, la probabilità che esista un collegamento cala in modo
esponenziale con la distanza d tra i due neuroni:

    p(d) = net_conn_prob * exp(-d / net_conn_range)

I collegamenti lunghi sono rari ma possibili. Un neurone che si collega a se
stesso (d = 0) fa eccezione: in natura è raro, e ha la sua probabilità
`net_self_prob`.
"""

import numpy as np

from .sensors import BODY, CHANNELS


def anchors(cfg, n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Posizioni fisse (n_in, 2) dei sensi e (n_out, 2) dei motori."""
    n_vis = cfg.vision_rays * len(CHANNELS)
    if n_in == n_vis + len(BODY):
        # La vista occupa i primi 4/5 del bordo, i sensi del corpo il resto.
        y = np.r_[(np.arange(n_vis) + 0.5) / n_vis * 0.8, 0.84 + (np.arange(len(BODY)) + 0.5) / len(BODY) * 0.16]
    else:  # ingressi di forma diversa (per esempio nei test): in fila
        y = (np.arange(n_in) + 0.5) / n_in
    senses = np.stack([np.zeros(n_in), y], axis=1)
    motors = np.stack([np.ones(n_out), (np.arange(n_out) + 0.5) / n_out], axis=1)
    return senses, motors


def distances(cfg, pos: np.ndarray, n_in: int, n_out: int) -> np.ndarray:
    """pos (N, H, 2) posizioni degli interni -> (N, T, S) distanza tra ogni
    destinatario (interni, motori) e ogni sorgente (sensi, interni, motori)."""
    n = len(pos)
    senses, motors = anchors(cfg, n_in, n_out)
    motors_n = np.broadcast_to(motors, (n, n_out, 2))
    targets = np.concatenate([pos, motors_n], axis=1)
    sources = np.concatenate([np.broadcast_to(senses, (n, n_in, 2)), targets], axis=1)
    return np.linalg.norm(targets[:, :, None, :] - sources[:, None, :, :], axis=3)


def connect_prob(cfg, dist: np.ndarray, n_in: int) -> np.ndarray:
    """(N, T, S): probabilità che un collegamento esista alla nascita di una
    rete casuale. Le mutazioni che aggiungono collegamenti seguono la stessa
    forma (vedi `genetics.mutate`)."""
    p = cfg.net_conn_prob * np.exp(-dist / cfg.net_conn_range)
    t = np.arange(dist.shape[1])
    p[:, t, n_in + t] = cfg.net_self_prob  # il neurone con se stesso
    return p
