"""Le reti a impulsi (`--net spike-ei`)."""

import numpy as np

from ...config import Config
from .genome import GENES, PLASTIC, Genome, retention
from .rate import Brains


class SpikingBrains(Brains):
    """Reti a impulsi (neuroni "leaky integrate-and-fire") con neuroni
    eccitatori o inibitori, stessa struttura libera di `Brains`.

    Neuroni interni: accumulano carica dagli ingressi, la perdono col tempo e,
    superata la soglia 1, emettono un impulso e si azzerano. Agli altri neuroni
    arriva solo l'impulso (0 o 1). Ognuno ha un tipo fisso: eccitatorio o
    inibitorio su *tutti* i suoi collegamenti in uscita, come in natura.
    Recettori: non emettono impulsi, entrano come corrente continua.
    Motori: quattro, a coppie antagoniste. Non emettono impulsi: sommano nel
    tempo ciò che ricevono, come un muscolo, e non possono andare sotto zero.
    Il comando è la differenza nella coppia.

    Ogni neurone ha tre geni propri: la corrente di fondo (bias), la velocità
    con cui perde carica e il tipo (negativo = inibitorio).

    Il peso di un collegamento che parte da un neurone indica solo la forza: se
    ne usa il valore assoluto, e il segno viene dal tipo di chi trasmette. I
    motori sono sempre eccitatori. I recettori restano liberi (il segno è del
    collegamento) a meno di `sensors_excitatory_only`. La corrente di fondo non
    è un collegamento e resta con il suo segno.

    La rete fa `spike_substeps` passi per passo di fisica.
    """

    KIND = "spike-ei"
    CENTRED = False
    BASE_TRAITS = 3  # corrente di fondo, perdita, tipo; poi la plasticità
    N_TRAITS = BASE_TRAITS + len(GENES)
    GAIN = BASE_TRAITS + len(PLASTIC)  # il gene del guadagno
    OUT_LABELS = ("avanti", "indietro", "destra", "sinistra")

    @classmethod
    def trait_text(cls, traits: np.ndarray, hidden: bool) -> list[str]:
        kind = ("inibitorio" if traits[2] < 0.0 else "eccitatorio") if hidden else "- (solo per gli interni)"
        return [f"corrente di fondo {traits[0]:+.3f}",
                f"carica trattenuta {float(retention(traits[1])):.0%} (gene {traits[1]:+.3f})",
                f"tipo {kind} (gene {traits[2]:+.3f})"] + cls.plastic_text(traits, hidden)

    def __init__(self, cfg: Config, n: int, n_out: int = 4, seed: int = 0):
        self.substeps, self.gain = cfg.spike_substeps, cfg.spike_gain
        self.sensors_excitatory_only = cfg.sensors_excitatory_only
        super().__init__(cfg, n, n_out, seed)

    def _init_genes(self, cfg: Config, rng: np.random.Generator, genome: Genome) -> None:
        strength = np.abs(genome.traits[:, :, 2])
        inhibitory = rng.random(strength.shape) < cfg.spike_inhibitory_share
        genome.traits[:, :, 2] = np.where(inhibitory, -strength, strength)

    @property
    def inhibitory(self) -> np.ndarray:
        """(N, H): True dove il neurone interno è inibitorio."""
        return self.traits[:, : self.n_hidden, 2] < 0.0

    def _compile_extra(self) -> None:
        self.keep = retention(self.traits[:, :, 1])
        # Segno di ogni sorgente: +1 eccitatoria, -1 inibitoria, 0 libera (i
        # recettori, salvo `sensors_excitatory_only`). Vale anche per ciò che
        # si impara (legge di Dale) e per il segno della modulazione.
        sign = np.ones((self.n, self.n_in + self.n_hidden + self.n_out))
        sign[:, self.n_in : self.n_in + self.n_hidden] = np.where(self.inhibitory, -1.0, 1.0)
        if not self.sensors_excitatory_only:
            sign[:, : self.n_in] = 0.0
        self.dale = sign
        # Un modulatore eccitatorio rilascia M positivo ("bene"), uno
        # inibitorio M negativo ("male").
        self._mod_sign = np.where(self.inhibitory, -1.0, 1.0) * self.modulatory

    def _born_weights(self) -> np.ndarray:
        top = self.cfg.synapse_max
        strength = np.clip(self.weights, -top, top)
        w = np.where(self.dale[:, None, :] == 0.0, strength, np.abs(strength) * self.dale[:, None, :])
        return w * self.links

    def _pre(self, obs, before, after) -> np.ndarray:
        """Tracce recenti degli impulsi (e livelli dei motori) dopo il passo."""
        return np.concatenate([obs, after], axis=1)

    SUBSET = ("w_eff", "bias_now", "keep", "active")

    def reset_state(self) -> None:
        h = self.n_hidden
        self._forget()
        self.charge = np.zeros((self.n, h + self.n_out))
        self.spikes = np.zeros((self.n, h))
        # Per il viewer e per i collegamenti in uscita dai motori: attività
        # recente dei neuroni interni (media mobile degli impulsi) e livello dei motori.
        self.state = np.zeros((self.n, h + self.n_out))
        self._reset_activity()

    def step(self, obs: np.ndarray, signal: np.ndarray | None = None) -> np.ndarray:
        """obs (N, R) -> comandi (N, 2) in (-1, 1): spinta e sterzata."""
        before = self.state.copy()
        (self.charge, self.spikes, self.state), out = self._advance(
            obs, (self.charge, self.spikes, self.state), self)
        self._learn(np.arange(self.n), obs, before, self.state, signal=signal)
        self.clock += self.cfg.dt  # l'età avanza
        return out

    def step_some(self, obs: np.ndarray, rows: np.ndarray, signal: np.ndarray | None = None) -> np.ndarray:
        sub = self._rows(rows)
        before = self.state[rows]
        (charge, spikes, state), out = self._advance(
            obs, (self.charge[rows], self.spikes[rows], before.copy()), sub)
        self.charge[rows], self.spikes[rows], self.state[rows] = charge, spikes, state
        self._learn(rows, obs, before, state, sub, signal)
        self.clock[rows] += self.cfg.dt  # l'età avanza
        return out

    def _advance(self, obs, states, g):
        get = g.get if isinstance(g, dict) else lambda k: getattr(g, k)
        w_eff, bias, keep, active = get("w_eff"), get("bias_now"), get("keep"), get("active")
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
