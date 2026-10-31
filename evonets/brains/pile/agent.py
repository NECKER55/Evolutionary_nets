"""La base degli agenti (`Agent`) e l'attore-critico comune a chi impara (`Learner`)."""

import numpy as np

from ...arenas.pile.world import N_PROGRAMS
from .attention import Attention
from .genes import BITE, DELTA_MAX, EXPLORE, LEAVE, N_BASE, N_HEADS, WEIGHT_MAX, decode, default_genome


class Agent:
    """Ciò che serve all'arena: `step(raw, reward, objs)` -> programmi (N,),
    `att` l'attenzione (l'oggetto attento guida F), e `reborn(who, genomes)`
    quando un corpo rinasce."""

    name = "agente"

    def __init__(self, n: int, dt: float, seed: int = 0, genomes=None, objects: int = 1):
        self.n, self.dt = n, dt
        self.rng = np.random.default_rng(seed)
        self._given = None if genomes is None else np.array(genomes, dtype=float)
        self.genome = np.tile(default_genome(), (n, 1)) if genomes is None else self._given
        self.p = decode(self.genome)
        self.blocks = {}
        self.att = Attention(n, objects, dt, self.rng)
        self.clock = np.zeros(n)  # da quanto si è deciso l'ultima volta (ognuno ha il suo ritmo)
        self.program = np.full(n, EXPLORE)
        self.pi = np.full((n, N_PROGRAMS), 1.0 / N_PROGRAMS)
        self.bite = np.zeros(n, dtype=bool)  # si morde (deciso a ogni decisione)
        self.p_bite = np.full(n, 0.5)

    def _attach(self, specs) -> None:
        """I blocchi propri del cervello: (nome, forma, valori di partenza (n, ...))."""
        start, parts = N_BASE, []
        for name, shape, init in specs:
            size = int(np.prod(shape))
            self.blocks[name] = (slice(start, start + size), tuple(shape))
            parts.append(np.asarray(init, dtype=float).reshape(self.n, size))
            start += size
        if self._given is None:
            self.genome = np.concatenate([self.genome[:, :N_BASE]] + parts, 1)
        elif self._given.shape[1] != start:
            raise ValueError(f"genoma di {self._given.shape[1]} geni, questo cervello ne vuole {start}")
        self.p = decode(self.genome)

    def start(self, name: str, who=slice(None)) -> np.ndarray:
        """I valori di partenza del blocco `name` (dal genoma)."""
        sl, shape = self.blocks[name]
        g = self.genome[who, sl]
        return g.reshape((g.shape[0],) + shape)

    def _tick(self) -> np.ndarray:
        """(n,) chi decide in questo passo (ognuno ogni `decide_every`)."""
        self.clock += self.dt
        decide = self.clock >= self.p["decide_every"] - 1e-9
        self.clock[decide] = 0.0
        return decide

    def innate_reward(self, raw: np.ndarray) -> np.ndarray:
        """I valori innati legati ai sensi (per passo)."""
        return (self.p["innate"] * raw).sum(1) * self.dt

    def reborn(self, who: np.ndarray, genomes=None) -> None:
        if genomes is not None:
            self.genome[who] = genomes
            self.p = decode(self.genome)
        self.att.reset(who)
        self.bite[who] = False
        self.clock[who] = 0.0


class Learner(Agent):
    """L'attore (E) e il critico (V) comuni ai due agenti che imparano."""

    def _init_actor_critic(self, n_e: int, n_v: int, critics: int = 1):
        n = self.n
        self.w_e = np.zeros((n, N_HEADS, n_e))
        self.e_e = np.zeros((n, N_HEADS, n_e))
        self.w_v = np.zeros((n, critics, n_v))  # uno per orizzonte
        self.e_v = np.zeros((n, critics, n_v))
        self.v_prev = np.zeros((n, critics))
        self.value = np.zeros(n)  # la previsione (media dei critici)
        self.trust = np.zeros(n)  # quanto pesa il critico nella sorpresa che insegna (dal gene)
        self.fresh = np.ones(n, dtype=bool)  # appena nato: niente δ al primo passo
        self.program = self.rng.integers(0, N_PROGRAMS, n)
        self.delta = np.zeros(n)
        self.teach = np.zeros(n)  # la parte di δ che insegna (dopo la soglia)

    def _td(self, f_v, reward, horizons, eta, use_critic: bool = True) -> np.ndarray:
        """δ e l'aggiornamento dei critici (TD(λ), passo normalizzato).
        `horizons` (n, K): l'orizzonte di ogni critico; `eta` (n, ingressi): la
        velocità per ingresso. δ è la media delle sorprese dei K critici."""
        p, dt = self.p, self.dt
        gamma = np.exp(-dt / horizons)  # (n, K)
        value = (self.w_v * f_v[:, None, :]).sum(-1)
        if use_critic:
            deltas = np.where(self.fresh[:, None], 0.0, reward[:, None] + gamma * value - self.v_prev)
        else:
            deltas = np.repeat(reward[:, None], horizons.shape[1], 1)
        deltas = np.clip(deltas, -DELTA_MAX, DELTA_MAX)  # la dopamina satura
        norm = 1.0 + (f_v**2).sum(1)[:, None] + (self.e_v * f_v[:, None, :]).sum(-1)  # con ingressi costanti la traccia si accumula
        self.w_v += (deltas / norm)[:, :, None] * eta[:, None, :] * self.e_v
        self.e_v = (gamma * np.exp(-dt / p["trace_critic"])[:, None])[:, :, None] * self.e_v + f_v[:, None, :]
        self.v_prev = (self.w_v * f_v[:, None, :]).sum(-1)
        self.fresh[:] = False
        self.value = self.v_prev.mean(1)
        # Quanto pesa il critico (gene `critic_mix`): la sorpresa che insegna è
        # una via di mezzo tra il valore grezzo e la sorpresa dei critici. Con un
        # comportamento ancora casuale il critico non ha niente da prevedere e
        # aggiunge rumore; quando il comportamento diventa prevedibile, conviene.
        full = deltas.mean(1)
        self.trust = p["critic_mix"] if use_critic else 0.0 * full
        self.delta = reward + self.trust * (full - reward)
        return self.delta

    def _act(self, logits, f_e, teach, eta, temp, decide, f_tag=None, stick_on=None):
        """Attore: impara con la parte di δ che insegna, sulla traccia; poi chi
        è al momento di decidere (`decide`, (n,)) sceglie. `f_tag`: gli
        ingressi che possono marcare."""
        p, dt = self.p, self.dt
        norm = 1.0 + (f_e**2).sum(1)
        self.w_e = np.clip(self.w_e + (eta * teach / norm)[:, None, None] * self.e_e, -WEIGHT_MAX, WEIGHT_MAX)
        self.e_e *= np.exp(-dt / p["trace_actor"])[:, None, None]
        if not decide.any():
            return self.program
        on = np.ones(self.n) if stick_on is None else stick_on
        z = (logits[:, :N_PROGRAMS] + (p["stick"] * on)[:, None] * np.eye(N_PROGRAMS)[self.program]) / temp[:, None]
        z -= z.max(1, keepdims=True)
        pi = np.exp(z)
        pi /= pi.sum(1, keepdims=True)
        u = self.rng.random((self.n, 1))
        program = (np.cumsum(pi, 1) < u).sum(1).clip(0, N_PROGRAMS - 1)
        # Il morso: sì o no (probabilità dalla sua testa).
        p_bite = 1.0 / (1.0 + np.exp(-np.clip(logits[:, BITE] / temp, -30, 30)))
        bite = self.rng.random(self.n) < p_bite
        self.program = np.where(decide, program, self.program)
        self.pi = np.where(decide[:, None], pi, self.pi)
        self.p_bite = np.where(decide, p_bite, self.p_bite)
        self.bite = np.where(decide, bite, self.bite)
        dev = np.concatenate([np.eye(N_PROGRAMS)[program] - pi, (bite - p_bite)[:, None]], 1) * decide[:, None]
        tag = f_e if f_tag is None else f_tag
        self.e_e += dev[:, :, None] * tag[:, None, :]  # la scelta rispetto all'attesa
        self._chose(dev)
        self.att.leave(decide & (self.program == LEAVE), p)
        return self.program

    def _chose(self, deviation):
        pass

    def _reset_actor_critic(self, who):
        for x in (self.w_e, self.e_e, self.w_v, self.e_v, self.v_prev, self.value, self.delta, self.teach, self.trust):
            x[who] = 0.0
        self.fresh[who] = True
