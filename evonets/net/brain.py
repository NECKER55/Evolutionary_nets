from dataclasses import dataclass

import numpy as np

from ..config import Config


@dataclass
class Genome:
    """I genomi di un gruppo di individui, uno per riga (vedi `Brains`)."""

    weights: np.ndarray
    bias: np.ndarray
    conn: np.ndarray
    active: np.ndarray

    def __len__(self) -> int:
        return len(self.weights)

    def take(self, idx) -> "Genome":
        """Copia indipendente degli individui indicati."""
        return Genome(self.weights[idx].copy(), self.bias[idx].copy(),
                      self.conn[idx].copy(), self.active[idx].copy())

    @staticmethod
    def concat(parts: list["Genome"]) -> "Genome":
        return Genome(*(np.concatenate([getattr(p, f) for p in parts])
                        for f in ("weights", "bias", "conn", "active")))


class Brains:
    """Le reti di tutta la popolazione, una per riga.

    Ogni rete ha `n_in` ingressi (i sensi), fino a `net_hidden_max` neuroni
    interni e `n_out` uscite (i motori). I collegamenti sono liberi: qualsiasi
    neurone (ingresso, interno, uscita) può parlare a qualsiasi neurone interno
    o di uscita, se stesso compreso. Non esistono strati.

    A ogni passo tutti i neuroni si aggiornano insieme a partire dai valori del
    passo precedente, che restano in `state`: gli anelli sono la memoria.

    Genoma (ciò che l'evoluzione potrà toccare):
      weights (N, T, S)  peso con segno: >0 eccita, <0 inibisce
      conn    (N, T, S)  quali collegamenti esistono
      active  (N, H)     quali neuroni interni esistono
      bias    (N, T)
    dove S = n_in + H + n_out sono le sorgenti e T = H + n_out i destinatari.
    """

    def __init__(self, cfg: Config, n: int, n_in: int, n_out: int, seed: int = 0):
        self.n, self.n_in, self.n_out = n, n_in, n_out
        self.n_hidden = h = cfg.net_hidden_max
        rng = np.random.default_rng(seed)
        targets, sources = h + n_out, n_in + h + n_out
        std = cfg.net_weight_std
        self.weights = rng.normal(0.0, std, (n, targets, sources))
        self.bias = rng.normal(0.0, std, (n, targets))
        self.conn = rng.random((n, targets, sources)) < cfg.net_conn_prob
        # Ogni individuo nasce con una frazione diversa di neuroni interni attivi.
        self.active = rng.random((n, h)) < rng.random((n, 1))
        self.compile()
        self.reset_state()

    @property
    def genome(self) -> Genome:
        return Genome(self.weights, self.bias, self.conn, self.active)

    def set_genome(self, genome: Genome) -> None:
        """Sostituisce l'intera popolazione (stesso numero di individui o no)."""
        self.weights, self.bias = genome.weights, genome.bias
        self.conn, self.active = genome.conn, genome.active
        self.n = len(genome)
        self.compile()
        self.reset_state()

    def compile(self) -> None:
        """Da chiamare dopo ogni modifica al genoma: prepara i pesi effettivi,
        azzerando ciò che tocca un neurone interno inesistente."""
        ones_in = np.ones((self.n, self.n_in), dtype=bool)
        ones_out = np.ones((self.n, self.n_out), dtype=bool)
        self.target_on = np.concatenate([self.active, ones_out], axis=1)
        source_on = np.concatenate([ones_in, self.active, ones_out], axis=1)
        self.links = self.conn & self.target_on[:, :, None] & source_on[:, None, :]
        self.w_eff = self.weights * self.links

    def reset_state(self) -> None:
        self.state = np.zeros((self.n, self.n_hidden + self.n_out))

    def step(self, obs: np.ndarray) -> np.ndarray:
        """obs (N, n_in) -> uscite (N, n_out) in [-1, 1]."""
        x = np.concatenate([obs, self.state], axis=1)
        z = (self.w_eff @ x[:, :, None])[:, :, 0] + self.bias
        self.state = np.tanh(z) * self.target_on
        return self.state[:, self.n_hidden :]
