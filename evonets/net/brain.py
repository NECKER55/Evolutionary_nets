from dataclasses import dataclass

import numpy as np

from ..config import Config


def live_links(conn: np.ndarray, active: np.ndarray, n_in: int) -> np.ndarray:
    """Collegamenti che contano davvero: presenti nel genoma e tra neuroni
    esistenti. conn (N, T, S), active (N, H) -> (N, T, S)."""
    n = len(conn)
    n_out = conn.shape[1] - active.shape[1]
    target_on = np.concatenate([active, np.ones((n, n_out), dtype=bool)], axis=1)
    source_on = np.concatenate([np.ones((n, n_in), dtype=bool), target_on], axis=1)
    return conn & target_on[:, :, None] & source_on[:, None, :]


def retention(gene: np.ndarray) -> np.ndarray:
    """Reti a impulsi: quota di carica che un neurone trattiene a ogni passo,
    dal gene "perdita". Da ~0.5 (neurone rapido) a ~0.97 (lento), tipicamente ~0.8."""
    return 1.0 / (1.0 + np.exp(-(1.5 + gene)))


@dataclass
class Genome:
    """I genomi di un gruppo di individui, uno per riga (vedi `Brains`)."""

    weights: np.ndarray
    traits: np.ndarray
    conn: np.ndarray
    active: np.ndarray

    def __len__(self) -> int:
        return len(self.weights)

    def take(self, idx) -> "Genome":
        """Copia indipendente degli individui indicati."""
        return Genome(self.weights[idx].copy(), self.traits[idx].copy(),
                      self.conn[idx].copy(), self.active[idx].copy())

    @staticmethod
    def concat(parts: list["Genome"]) -> "Genome":
        return Genome(*(np.concatenate([getattr(p, f) for p in parts])
                        for f in ("weights", "traits", "conn", "active")))


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
      traits  (N, T, K)  geni propri di ogni neurone; qui uno solo, la soglia (bias)
    dove S = n_in + H + n_out sono le sorgenti e T = H + n_out i destinatari.
    """

    KIND = "rate"
    N_TRAITS = 1
    OUT_LABELS = ("spinta", "sterzo")

    @staticmethod
    def trait_text(traits: np.ndarray, hidden: bool) -> list[str]:
        """I geni propri di un neurone, in parole."""
        return [f"soglia {traits[0]:+.3f}"]

    def __init__(self, cfg: Config, n: int, n_in: int, n_out: int, seed: int = 0):
        self.n, self.n_in, self.n_out = n, n_in, n_out
        self.n_hidden = h = cfg.net_hidden_max
        rng = np.random.default_rng(seed)
        targets, sources = h + n_out, n_in + h + n_out
        std = cfg.net_weight_std
        self.weights = rng.normal(0.0, std, (n, targets, sources))
        self.traits = rng.normal(0.0, std, (n, targets, self.N_TRAITS))
        self.conn = rng.random((n, targets, sources)) < cfg.net_conn_prob
        # Ogni individuo nasce con una frazione diversa di neuroni interni attivi.
        self.active = rng.random((n, h)) < rng.random((n, 1))
        self._init_genes(cfg, rng)
        self.compile()
        self.reset_state()

    def _init_genes(self, cfg: Config, rng: np.random.Generator) -> None:
        """Per i tipi di rete che hanno geni con una distribuzione iniziale propria."""

    @property
    def bias(self) -> np.ndarray:
        return self.traits[:, :, 0]

    @property
    def genome(self) -> Genome:
        return Genome(self.weights, self.traits, self.conn, self.active)

    def set_genome(self, genome: Genome) -> None:
        """Sostituisce l'intera popolazione (stesso numero di individui o no)."""
        self.weights, self.traits = genome.weights, genome.traits
        self.conn, self.active = genome.conn, genome.active
        self.n = len(genome)
        self.compile()
        self.reset_state()

    def compile(self) -> None:
        """Da chiamare dopo ogni modifica al genoma: prepara i pesi effettivi,
        azzerando ciò che tocca un neurone interno inesistente."""
        ones_out = np.ones((self.n, self.n_out), dtype=bool)
        self.target_on = np.concatenate([self.active, ones_out], axis=1)
        self.links = live_links(self.conn, self.active, self.n_in)
        self._subset = None  # copie per `step_some`, da rifare
        self.w_eff = self.weights * self.links

    def reset_state(self) -> None:
        self.state = np.zeros((self.n, self.n_hidden + self.n_out))

    # Geni compilati di cui `step_some` usa una copia ridotta.
    SUBSET = ("w_eff", "bias", "target_on")

    def _rows(self, rows: np.ndarray) -> dict:
        """Copie dei geni compilati per i soli individui `rows`. Si rifanno
        solo quando l'insieme cambia (in una generazione chi esce non rientra)."""
        if self._subset is None or not np.array_equal(self._subset["rows"], rows):
            self._subset = {"rows": rows, **{k: getattr(self, k)[rows] for k in self.SUBSET}}
        return self._subset

    def step(self, obs: np.ndarray) -> np.ndarray:
        """obs (N, n_in) -> uscite (N, n_out) in [-1, 1]."""
        self.state, out = self._advance(obs, self.state, self)
        return out

    def step_some(self, obs: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Come `step`, ma solo per gli individui `rows` (obs ha una riga per
        ciascuno): gli altri restano fermi. Ogni rete calcola per conto suo,
        quindi il risultato per `rows` è identico a quello di `step`."""
        sub = self._rows(rows)
        state, out = self._advance(obs, self.state[rows], sub)
        self.state[rows] = state
        return out

    def _advance(self, obs, state, g):
        """Un passo a partire da `state`, con i geni compilati `g` (self o una
        copia ridotta). Restituisce (nuovo stato, uscite)."""
        get = g.get if isinstance(g, dict) else lambda k: getattr(g, k)
        x = np.concatenate([obs, state], axis=1)
        z = (get("w_eff") @ x[:, :, None])[:, :, 0] + get("bias")
        state = np.tanh(z) * get("target_on")
        return state, state[:, self.n_hidden :]


class SpikingBrains(Brains):
    """Reti a impulsi (neuroni "leaky integrate-and-fire") con neuroni
    eccitatori o inibitori, stessa struttura libera di `Brains`.

    Neuroni interni: accumulano carica dagli ingressi, la perdono col tempo e,
    superata la soglia 1, emettono un impulso e si azzerano. Agli altri neuroni
    arriva solo l'impulso (0 o 1). Ognuno ha un tipo fisso: eccitatorio o
    inibitorio su *tutti* i suoi collegamenti in uscita, come in natura.
    Sensi: non emettono impulsi, entrano come corrente continua.
    Motori: quattro, a coppie antagoniste. Non emettono impulsi: sommano nel
    tempo ciò che ricevono, come un muscolo, e non possono andare sotto zero.
    Il comando è la differenza nella coppia.

    Ogni neurone ha tre geni propri: la corrente di fondo (bias), la velocità
    con cui perde carica e il tipo (negativo = inibitorio).

    Il peso di un collegamento che parte da un neurone indica solo la forza: se
    ne usa il valore assoluto, e il segno viene dal tipo di chi trasmette. I
    motori sono sempre eccitatori. I sensi restano liberi (il segno è del
    collegamento) a meno di `sensors_excitatory_only`. La corrente di fondo non
    è un collegamento e resta con il suo segno.

    La rete fa `spike_substeps` passi per passo di fisica.
    """

    KIND = "spike-ei"
    N_TRAITS = 3
    OUT_LABELS = ("avanti", "indietro", "destra", "sinistra")

    @staticmethod
    def trait_text(traits: np.ndarray, hidden: bool) -> list[str]:
        kind = ("inibitorio" if traits[2] < 0.0 else "eccitatorio") if hidden else "- (solo per gli interni)"
        return [f"corrente di fondo {traits[0]:+.3f}",
                f"carica trattenuta {float(retention(traits[1])):.0%} (gene {traits[1]:+.3f})",
                f"tipo {kind} (gene {traits[2]:+.3f})"]

    def __init__(self, cfg: Config, n: int, n_in: int, n_out: int = 4, seed: int = 0):
        self.substeps, self.gain = cfg.spike_substeps, cfg.spike_gain
        self.sensors_excitatory_only = cfg.sensors_excitatory_only
        super().__init__(cfg, n, n_in, n_out, seed)

    def _init_genes(self, cfg: Config, rng: np.random.Generator) -> None:
        strength = np.abs(self.traits[:, :, 2])
        inhibitory = rng.random(strength.shape) < cfg.spike_inhibitory_share
        self.traits[:, :, 2] = np.where(inhibitory, -strength, strength)

    @property
    def inhibitory(self) -> np.ndarray:
        """(N, H): True dove il neurone interno è inibitorio."""
        return self.traits[:, : self.n_hidden, 2] < 0.0

    def compile(self) -> None:
        super().compile()
        self.keep = retention(self.traits[:, :, 1])
        sign = np.ones((self.n, self.n_in + self.n_hidden + self.n_out))
        sign[:, self.n_in : self.n_in + self.n_hidden] = np.where(self.inhibitory, -1.0, 1.0)
        w = np.abs(self.weights) * sign[:, None, :]
        if not self.sensors_excitatory_only:
            w[:, :, : self.n_in] = self.weights[:, :, : self.n_in]
        self.w_eff = w * self.links

    SUBSET = ("w_eff", "bias", "keep", "active")

    def reset_state(self) -> None:
        h = self.n_hidden
        self.charge = np.zeros((self.n, h + self.n_out))
        self.spikes = np.zeros((self.n, h))
        # Per il viewer e per i collegamenti in uscita dai motori: attività
        # recente dei neuroni interni (media mobile degli impulsi) e livello dei motori.
        self.state = np.zeros((self.n, h + self.n_out))

    def step(self, obs: np.ndarray) -> np.ndarray:
        """obs (N, n_in) -> comandi (N, 2) in (-1, 1): spinta e sterzata."""
        (self.charge, self.spikes, self.state), out = self._advance(
            obs, (self.charge, self.spikes, self.state), self)
        return out

    def step_some(self, obs: np.ndarray, rows: np.ndarray) -> np.ndarray:
        sub = self._rows(rows)
        (charge, spikes, state), out = self._advance(
            obs, (self.charge[rows], self.spikes[rows], self.state[rows]), sub)
        self.charge[rows], self.spikes[rows], self.state[rows] = charge, spikes, state
        return out

    def _advance(self, obs, states, g):
        get = g.get if isinstance(g, dict) else lambda k: getattr(g, k)
        w_eff, bias, keep, active = get("w_eff"), get("bias"), get("keep"), get("active")
        charge, spikes, state = states
        h = self.n_hidden
        drive = 1.0 - keep
        for _ in range(self.substeps):
            x = np.concatenate([obs, spikes, state[:, h:]], axis=1)
            current = (w_eff @ x[:, :, None])[:, :, 0] + bias
            current[:, :h] *= self.gain
            charge = np.maximum(keep * charge + drive * current, -1.0)
            inner = charge[:, :h]
            fired = (inner >= 1.0) & active
            inner[fired | ~active] = 0.0
            spikes = fired.astype(float)
            state[:, :h] = 0.8 * state[:, :h] + 0.2 * spikes
            state[:, h:] = np.tanh(np.maximum(charge[:, h:], 0.0))
        m = state[:, h:]
        return (charge, spikes, state), np.stack([m[:, 0] - m[:, 1], m[:, 2] - m[:, 3]], axis=1)
