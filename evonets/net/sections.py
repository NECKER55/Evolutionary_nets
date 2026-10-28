"""Il cervello a sezioni (docs/SPEC_CERVELLO_SEZIONI.md) e i due agenti di
confronto per l'arena delle pile (docs/SPEC_ARENA_PILE.md).

Si fissa la struttura, non il contenuto: le sezioni, chi parla con chi, a che
velocità cambia ciascuna, con che famiglia di regole. Nessun neurone nasce
con un significato: i pesi che imparano partono da zero o a caso, e ciò che
rappresentano emerge (si misura con `selectivity` e le lesioni).

- A interfaccia (fissa): sensi grezzi -> canali "com'è" e "cosa è cambiato".
- B nucleo stabile: prototipi in competizione (Hebb competitivo, come il
  k-means, con la "coscienza" di DeSieno perché nessuno resti muto), lenti;
  impara di più nei momenti di sorpresa (|δ|) e verso ciò che viene dopo
  (previsione del prossimo ingresso).
- D memoria di lavoro: tiene una copia dello stato di B (τ di secondi); un
  cancello decide quando scrivere, ed è l'unica cosa che impara (con δ).
- C memoria veloce: ingresso fisso, casuale e sparso (come le cellule di
  Kenyon del corpo a fungo), pochi neuroni accesi per situazione; impara solo
  l'uscita verso E, in fretta, e dimentica in minuti.
- L memoria lunga: la stessa uscita di C, cento volte più lenta, che si
  avvicina a C solo a riposo: ciò che si inverte di continuo si annulla.
- E selezione: programmi in competizione (softmax con rumore = esplorazione);
  attore: Δw = η · δ · traccia di (azione − attesa) × ingresso.
- V critico: TD, previsione da B, D, C; δ = r + γ·V(t) − V(t−1).
- U incertezza: media recente di |δ| sulla media lunga; alza velocità di C ed
  E e l'esplorazione.
- F motore: i programmi fissi di `sim.pile.Motor`.

Lesioni (`lesions`): "B" (E, V, C, D leggono A invece di B), "D", "C", "L",
"U", "V" (niente critico: δ = ricompensa grezza).
"""

from dataclasses import dataclass, field, fields

import numpy as np

from ..sim.pile import EYES, N_PROGRAMS, N_SENSES

FAST = (*range(EYES.start, EYES.stop), 15, 16)  # canali con la loro versione "cosa è cambiato": occhi, gusto, dolore
N_A = N_SENSES + len(FAST)


class Interface:
    """A: fissa. Per ogni canale il valore ("com'è"); per occhi, gusto e
    dolore anche quanto è appena salito ("cosa è cambiato", τ = `tau`)."""

    def __init__(self, n: int, dt: float, tau: float = 0.3):
        self.keep = np.exp(-dt / tau)
        self.slow = np.zeros((n, len(FAST)))

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        x = raw[:, FAST]
        fast = np.clip(x - self.slow, 0.0, 1.0)
        self.slow = self.keep * self.slow + (1.0 - self.keep) * x
        return np.concatenate([raw, fast], axis=1)

    def reset(self, who):
        self.slow[who] = 0.0


def event_of(a: np.ndarray) -> np.ndarray:
    """Quanto è successo di nuovo adesso (0-1): il massimo dei canali veloci."""
    return a[:, N_SENSES:].max(axis=1)


@dataclass
class Genes:
    """I parametri che l'evoluzione tara (fase B); nella fase A sono questi
    valori per tutti. Velocità e tempi in scala logaritmica nel genoma."""

    eta_b: float = 0.05  # B, per secondo
    eta_c: float = 0.5  # C, per aggiornamento (δ è già per passo)
    eta_e: float = 1.0
    eta_v: float = 0.1
    eta_l: float = 0.01  # L, per secondo di riposo
    tau_d: float = 4.0  # D: quanto dura il ricordo (s)
    temperature: float = 0.5  # E: esplorazione
    surprise_rate: float = 1.0  # U: quanto le sorprese alzano le velocità
    surprise_explore: float = 1.0  # U: quanto alzano l'esplorazione
    horizon: float = 4.0  # V: quanto lontano guarda (s)

    LOG = ("eta_b", "eta_c", "eta_e", "eta_v", "eta_l", "tau_d", "temperature", "horizon")


GENE_NAMES = tuple(f.name for f in fields(Genes))
INNATE_CAP = 0.2  # tetto dei valori innati legati ai sensi (per secondo di senso pieno)
N_GENOME = len(GENE_NAMES) + N_SENSES  # i parametri, poi i valori innati per ogni senso


def default_genome() -> np.ndarray:
    g = Genes()
    v = [np.log(getattr(g, k)) if k in Genes.LOG else getattr(g, k) for k in GENE_NAMES]
    return np.concatenate([np.asarray(v, dtype=float), np.zeros(N_SENSES)])


def decode(genome: np.ndarray) -> dict:
    """(N, N_GENOME) -> {nome: (N,)} più "innate" (N, N_SENSES), nei loro limiti."""
    out = {}
    for k, name in enumerate(GENE_NAMES):
        v = genome[:, k]
        out[name] = np.exp(v) if name in Genes.LOG else np.maximum(v, 0.0)
    out["innate"] = np.clip(genome[:, len(GENE_NAMES):], -INNATE_CAP, INNATE_CAP)
    return out


def mutate(genome: np.ndarray, rng: np.random.Generator, sigma: float = 0.1) -> np.ndarray:
    g = genome + rng.normal(0.0, sigma, genome.shape)
    g[:, len(GENE_NAMES):] = np.clip(g[:, len(GENE_NAMES):], -INNATE_CAP, INNATE_CAP)
    return g


def cross(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Incrocio uniforme: ogni gene da uno dei due genitori."""
    return np.where(rng.random(a.shape) < 0.5, a, b)


class Agent:
    """Ciò che serve all'arena: `step(raw, reward)` -> programmi (N,), e
    `reborn(who, genomes)` quando un corpo rinasce."""

    name = "agente"

    def __init__(self, n: int, dt: float, seed: int = 0, genomes=None):
        self.n, self.dt = n, dt
        self.rng = np.random.default_rng(seed)
        self.genome = np.tile(default_genome(), (n, 1)) if genomes is None else np.array(genomes, dtype=float)
        self.p = decode(self.genome)

    def innate_reward(self, raw: np.ndarray) -> np.ndarray:
        """I valori innati legati ai sensi (per passo)."""
        return (self.p["innate"] * raw).sum(1) * self.dt

    def reborn(self, who: np.ndarray, genomes=None) -> None:
        if genomes is not None:
            self.genome[who] = genomes
            self.p = decode(self.genome)


class Guided(Agent):
    """Guidato a mano, per controllare il mondo: sa qual è il colore buono
    (`good`, (N,), aggiornato da fuori) e lo insegue; altrimenti esplora;
    il cattivo davanti e vicino lo evita."""

    name = "guidato"

    def __init__(self, n, dt, seed=0, genomes=None):
        super().__init__(n, dt, seed, genomes)
        self.good = np.zeros(n, dtype=int)

    def step(self, raw, reward):
        eye = raw[:, EYES].reshape(-1, 3, 3)
        idx = np.arange(self.n)
        good, bad = eye[idx, self.good], eye[idx, 1 - self.good]
        bad_ahead = (bad[:, 1] > 0.6) & (bad[:, 1] > good[:, 1])  # una cattiva davanti, più vicina della buona
        self.program = np.where(bad_ahead, 4, np.where(good.max(-1) > 0.0, self.good, 3))
        self.pi = np.eye(N_PROGRAMS)[self.program]
        return self.program


class _Learner(Agent):
    """L'attore (E) e il critico (V) comuni ai due agenti che imparano."""

    decide_every = 0.5  # s: ogni quanto E rimette in gara i programmi (tra una scelta e l'altra il programma resta)
    stick = 1.5  # il programma in corso parte avvantaggiato (la selezione dei gangli della base è "appiccicosa")

    def _init_actor_critic(self, n_e: int, n_v: int):
        n = self.n
        self.w_e = np.zeros((n, N_PROGRAMS, n_e))
        self.e_e = np.zeros((n, N_PROGRAMS, n_e))
        self.w_v = np.zeros((n, n_v))
        self.e_v = np.zeros((n, n_v))
        self.v_prev = np.zeros(n)
        self.fresh = np.ones(n, dtype=bool)  # appena nato: niente δ al primo passo
        self.program = self.rng.integers(0, N_PROGRAMS, n)
        self.pi = np.full((n, N_PROGRAMS), 1.0 / N_PROGRAMS)
        self.clock = 0.0
        self.delta = np.zeros(n)

    def _td(self, f_v: np.ndarray, reward: np.ndarray, use_critic: bool = True) -> np.ndarray:
        """δ e l'aggiornamento del critico (TD(λ), passo normalizzato)."""
        p, dt = self.p, self.dt
        gamma = np.exp(-dt / p["horizon"])
        value = (self.w_v * f_v).sum(1)
        if use_critic:
            delta = np.where(self.fresh, 0.0, reward + gamma * value - self.v_prev)
        else:
            delta = reward
        norm = 1.0 + (f_v**2).sum(1)
        self.w_v += (p["eta_v"] * delta / norm)[:, None] * self.e_v
        self.e_v = (gamma * np.exp(-dt / 1.0))[:, None] * self.e_v + f_v
        self.v_prev = (self.w_v * f_v).sum(1)
        self.fresh[:] = False
        self.delta = delta
        return delta

    def _act(self, logits: np.ndarray, f_e: np.ndarray, delta: np.ndarray, eta: np.ndarray, temp: np.ndarray):
        """Attore: impara con δ sulla traccia, poi (ogni `decide_every`) sceglie."""
        dt = self.dt
        norm = 1.0 + (f_e**2).sum(1)
        self.w_e += (eta * delta / norm)[:, None, None] * self.e_e
        self.e_e *= np.exp(-dt / 2.0)
        self.clock += dt
        if self.clock >= self.decide_every - 1e-9:
            self.clock = 0.0
            z = (logits + self.stick * np.eye(N_PROGRAMS)[self.program]) / temp[:, None]
            z -= z.max(1, keepdims=True)
            pi = np.exp(z)
            pi /= pi.sum(1, keepdims=True)
            u = self.rng.random((self.n, 1))
            self.program = (np.cumsum(pi, 1) < u).sum(1).clip(0, N_PROGRAMS - 1)
            self.pi = pi
            chosen = np.eye(N_PROGRAMS)[self.program]
            self.e_e += (chosen - pi)[:, :, None] * f_e[:, None, :]  # la scelta rispetto all'attesa
            self._chose(chosen - pi)
        return self.program

    def _chose(self, deviation):
        pass

    def _reset_actor_critic(self, who):
        self.w_e[who] = 0.0
        self.e_e[who] = 0.0
        self.w_v[who] = 0.0
        self.e_v[who] = 0.0
        self.v_prev[who] = 0.0
        self.fresh[who] = True


class ActorCritic(_Learner):
    """Il riferimento: attore-critico classico (TD, Doya 2000) che legge
    direttamente l'interfaccia A. Stessi programmi, stessa regola di E e V del
    cervello a sezioni, ma senza B, C, D, L e U."""

    name = "attore-critico"

    def __init__(self, n, dt, seed=0, genomes=None):
        super().__init__(n, dt, seed, genomes)
        self.a = Interface(n, dt)
        self._init_actor_critic(N_A + 1, N_A + 1)

    def step(self, raw, reward):
        a = self.a(raw)
        f = np.concatenate([a, np.ones((self.n, 1))], 1)
        r = reward + self.innate_reward(raw)
        delta = self._td(f, r)
        logits = (self.w_e * f[:, None, :]).sum(-1)
        return self._act(logits, f, delta, self.p["eta_e"], self.p["temperature"])

    def reborn(self, who, genomes=None):
        super().reborn(who, genomes)
        self.a.reset(who)
        self._reset_actor_critic(who)


@dataclass
class Sizes:
    b: int = 16  # prototipi di B
    b_winners: int = 3
    c: int = 64  # neuroni di C
    c_fan_in: int = 4
    c_winners: int = 4
    tau_b: float = 0.2  # s: reazione di B
    surprise_boost: float = 2.0  # B impara (1 + questo · |δ|) volte di più
    predict: float = 0.5  # B: peso della previsione del prossimo ingresso
    eta_predict: float = 0.5  # per secondo
    conscience: float = 2.0  # quanto un prototipo che vince poco viene aiutato
    gate_event: float = 8.0  # D: quanto un evento nuovo apre il cancello (innato)
    gate_bias: float = -6.0  # chiuso a riposo (0,25% a passo): un cancello socchiuso cancellerebbe il ricordo
    eta_gate: float = 0.5
    forget: float = 180.0  # C: s per dimenticare
    rest_energy: float = 0.6  # L: "riposo" = carichi (energia sopra questa quota) e fermi a ricaricarsi
    lesions: frozenset = field(default_factory=frozenset)


class Sections(_Learner):
    """Il cervello a sezioni (vedi la docstring del modulo)."""

    name = "sezioni"

    def __init__(self, n, dt, seed=0, genomes=None, sizes: Sizes | None = None):
        super().__init__(n, dt, seed, genomes)
        self.s = s = sizes or Sizes()
        self.les = set(s.lesions)
        self.a = Interface(n, dt)
        nb = N_A if "B" in self.les else s.b
        self.nb = nb
        # B: prototipi a caso (nessun significato alla nascita) e previsione.
        self.proto = self.rng.random((n, s.b, N_A)) * 0.3
        self.pred = np.zeros((n, N_A, s.b))
        self.win_rate = np.full((n, s.b), 1.0 / s.b)
        self.state_b = np.zeros((n, nb))
        self.prev_b = np.zeros((n, nb))
        # D: la copia di B, e il cancello (pesi che imparano).
        self.mem = np.zeros((n, nb))
        self.w_gate = np.zeros((n, nb))
        self.e_gate = np.zeros((n, nb))
        # C: ingresso fisso, casuale e sparso (da B e D); uscite verso E (C veloce, L lenta).
        self.c_in = np.zeros((n, s.c, 2 * nb))
        for j in range(s.c):
            pick = np.argsort(self.rng.random((n, 2 * nb)), axis=1)[:, : s.c_fan_in]
            np.put_along_axis(self.c_in[:, j], pick, 1.0, axis=1)
        self.u_c = np.zeros((n, N_PROGRAMS, s.c))
        self.e_c = np.zeros((n, N_PROGRAMS, s.c))
        self.u_l = np.zeros((n, N_PROGRAMS, s.c))
        self.code = np.zeros((n, s.c))
        # U: medie di |δ|.
        self.d_fast = np.zeros(n)
        self.d_slow = np.full(n, 1e-3)
        self.unc = np.zeros(n)
        # E legge B, D e la costante; V legge B, D, C e la costante.
        self._init_actor_critic(2 * nb + 1, 2 * nb + s.c + 1)

    # --- B ---------------------------------------------------------------------------

    def _b(self, a: np.ndarray, delta: np.ndarray) -> np.ndarray:
        s, dt = self.s, self.dt
        if "B" in self.les:
            self.state_b = a
            return a
        d = ((self.proto - a[:, None, :]) ** 2).sum(-1)  # (n, b): distanza da ogni prototipo
        d = d - s.conscience * (1.0 / s.b - self.win_rate)  # chi vince poco è aiutato
        order = np.argsort(d, axis=1)
        kth = np.take_along_axis(d, order[:, s.b_winners: s.b_winners + 1], 1)
        y = np.clip(1.0 - d / np.maximum(kth, 1e-9), 0.0, 1.0)  # i più vicini, gli altri a zero
        y = y / np.maximum(y.max(1, keepdims=True), 1e-9)
        # Imparare: i vincitori si avvicinano all'ingresso (e, con la previsione, a ciò che segue).
        err = a - np.einsum("nab,nb->na", self.pred, self.prev_b)  # quanto il passo prima aveva previsto male
        self.pred += s.eta_predict * dt * err[:, :, None] * self.prev_b[:, None, :]
        rate = (self.p["eta_b"] * dt * (1.0 + s.surprise_boost * np.abs(delta)))[:, None, None]
        self.proto += rate * y[:, :, None] * ((a[:, None, :] - self.proto) + s.predict * err[:, None, :])
        self.win_rate += (dt / 30.0) * ((y > 0) / s.b_winners - self.win_rate)
        self.prev_b = self.state_b.copy()
        self.state_b += (dt / s.tau_b) * (y - self.state_b)
        return self.state_b

    # --- D ---------------------------------------------------------------------------

    def _d(self, b: np.ndarray, event: np.ndarray, delta: np.ndarray) -> np.ndarray:
        s, dt = self.s, self.dt
        if "D" in self.les:
            return np.zeros_like(b)
        self.w_gate += (s.eta_gate * delta)[:, None] * self.e_gate
        z = s.gate_event * event + (self.w_gate * b).sum(1) + s.gate_bias
        g = 1.0 / (1.0 + np.exp(-z))
        self.e_gate = np.exp(-dt / 2.0) * self.e_gate + (g * (1.0 - g))[:, None] * b
        keep = np.exp(-dt / self.p["tau_d"])[:, None]
        self.mem = (1.0 - g)[:, None] * keep * self.mem + g[:, None] * b
        return self.mem

    # --- C ---------------------------------------------------------------------------

    def _c(self, b: np.ndarray, m: np.ndarray) -> np.ndarray:
        s = self.s
        if "C" in self.les:
            return np.zeros((self.n, s.c))
        h = np.concatenate([b, m], 1)
        z = np.einsum("ncj,nj->nc", self.c_in, h)
        top = np.argpartition(-z, s.c_winners - 1, axis=1)[:, : s.c_winners]
        code = np.zeros_like(z)
        np.put_along_axis(code, top, 1.0, axis=1)
        self.code = code * (z > 1e-6)
        return self.code

    def _chose(self, deviation):
        self.e_c += deviation[:, :, None] * self.code[:, None, :]

    # --- Un passo --------------------------------------------------------------------

    def step(self, raw, reward):
        s, p, dt = self.s, self.p, self.dt
        a = self.a(raw)
        r = reward + self.innate_reward(raw)
        delta_prev = self.delta
        b = self._b(a, delta_prev)
        m = self._d(b, event_of(a), delta_prev)
        c = self._c(b, m)
        one = np.ones((self.n, 1))
        delta = self._td(np.concatenate([b, m, c, one], 1), r, use_critic="V" not in self.les)
        # U: molte sorprese di recente rispetto al solito = il mondo è cambiato.
        if "U" not in self.les:
            self.d_fast += (dt / 20.0) * (np.abs(delta) - self.d_fast)
            self.d_slow += (dt / 200.0) * (np.abs(delta) - self.d_slow)
            self.unc = np.clip(self.d_fast / np.maximum(self.d_slow, 1e-6) - 1.0, 0.0, 3.0)
        boost = 1.0 + p["surprise_rate"] * self.unc
        # C impara in fretta e dimentica; L si avvicina a C solo a riposo.
        self.u_c += (p["eta_c"] * boost * delta)[:, None, None] * self.e_c - self.u_c * (dt / s.forget)
        self.e_c *= np.exp(-dt / 2.0)
        if "L" not in self.les:
            # Riposo: carichi e fermi a ricaricarsi (il gusto della pila che si consuma).
            rest = (raw[:, 13] > s.rest_energy) & (raw[:, 15] > 0.1)
            self.u_l += (p["eta_l"] * dt * rest)[:, None, None] * (self.u_c - self.u_l)
        f_e = np.concatenate([b, m, one], 1)
        logits = (self.w_e * f_e[:, None, :]).sum(-1) + np.einsum("npc,nc->np", self.u_c, c)
        if "L" not in self.les:
            logits += np.einsum("npc,nc->np", self.u_l, c)
        temp = p["temperature"] * (1.0 + p["surprise_explore"] * self.unc)
        return self._act(logits, f_e, delta, p["eta_e"] * boost, temp)

    def reborn(self, who, genomes=None):
        super().reborn(who, genomes)
        k = int(np.asarray(who).sum()) if np.asarray(who).dtype == bool else len(who)
        self.a.reset(who)
        self._reset_actor_critic(who)
        self.proto[who] = self.rng.random((k, self.s.b, N_A)) * 0.3
        self.pred[who] = 0.0
        self.win_rate[who] = 1.0 / self.s.b
        for x in (self.state_b, self.prev_b, self.mem, self.w_gate, self.e_gate, self.u_c, self.e_c, self.u_l,
                  self.code, self.d_fast, self.unc):
            x[who] = 0.0
        self.d_slow[who] = 1e-3

    def winner(self) -> np.ndarray:
        """(N,) il neurone di B più acceso (per le misure di emersione)."""
        return self.state_b.argmax(1)


AGENTS = {"guidato": Guided, "attore-critico": ActorCritic, "sezioni": Sections}
