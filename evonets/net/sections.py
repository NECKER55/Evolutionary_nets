"""Il cervello a sezioni (docs/SPEC_CERVELLO_SEZIONI.md) e i due agenti di
confronto per l'arena delle pile (docs/SPEC_ARENA_PILE.md).

Si fissa la struttura, non il contenuto: le sezioni, chi parla con chi, a che
velocità cambia ciascuna, con che famiglia di regole. Nessun neurone nasce
con un significato: i pesi che imparano partono da zero o a caso, e ciò che
rappresentano emerge (si misura con `selectivity` e le lesioni).

- Attenzione: in ogni momento l'essere è concentrato su un oggetto (una pila
  o un compagno). Lo sceglie per salienza: una parte innata (vicino, appena
  comparso, un po' di inerzia, e chi è stato appena lasciato viene ignorato
  per qualche secondo: l'inibizione del ritorno) e una imparata (un peso per
  colore, che δ modifica: ciò che ha portato sorprese positive cattura lo
  sguardo). La scelta ha un po' di caso.
- A interfaccia (fissa): sensi grezzi -> canali "com'è" e "cosa è cambiato",
  più la fovea: colore, vicinanza e lato dell'oggetto attento.
- B nucleo stabile: prototipi in competizione (Hebb competitivo, come il
  k-means, con la "coscienza" di DeSieno perché nessuno resti muto), lenti;
  impara di più nei momenti di sorpresa (|δ|) e verso ciò che viene dopo
  (previsione del prossimo ingresso).
- D memoria di lavoro: tiene l'ultimo oggetto guardato (una copia della
  fovea, τ di secondi), anche quando non si vede più: "che cosa ho appena
  preso" quando arriva il gusto o il dolore. Un cancello decide quando
  scrivere (innato: un oggetto nuovo nella fovea), e impara con δ.
- C memoria veloce: ingresso fisso, casuale e sparso (come le cellule di
  Kenyon del corpo a fungo); si accendono tutti i neuroni sopra soglia (non
  un numero fisso: con "i 4 più attivi" tra tanti pari merito vinceva il
  caso, e ogni incontro con lo stesso oggetto accendeva neuroni diversi);
  impara solo
  l'uscita verso E, in fretta, e dimentica in minuti. Legge soprattutto
  che cos'è l'oggetto (il colore nella fovea e in D, non dove sta), più un
  po' di contesto (B, con peso `context`): così lo stesso oggetto accende
  gli stessi neuroni a ogni incontro, e ciò che si impara una volta vale la
  volta dopo (con tutto lo stato di B, o anche con la vicinanza e il lato
  dell'oggetto, ogni incontro aveva un codice diverso).
- L memoria lunga: la stessa uscita di C, cento volte più lenta, che si
  avvicina a C solo a riposo: ciò che si inverte di continuo si annulla.
- E selezione: programmi in competizione (softmax con rumore = esplorazione);
  attore: Δw = η · δ · traccia di (azione − attesa) × ingresso.
- V critico: TD, previsione da B, D, C; δ = r + γ·V(t) − V(t−1).
- U incertezza: media recente di |δ| sulla media lunga; alza velocità di C ed
  E e l'esplorazione.
- E sceglie anche se mordere (sì o no, a ogni decisione): si mangia solo
  mordendo, e mordere costa un po'. C ed L votano anche sul morso.
- F motore: un circuito innato di neuroni (`sim.pile.motor_net`) dai
  programmi e dall'oggetto attento a 4 neuroni motori (gira a sinistra, a
  destra, avanti, indietro); non impara nella vita, i suoi pesi sono geni.

Due regole sull'apprendimento con δ, solo nel cervello a sezioni:
- tracce strette: una sinapsi di E o del cancello di D si marca solo se il
  neurone da cui parte è acceso sopra `tag` (in natura si marca solo dove il
  neurone si attiva davvero);
- soglia adattiva: insegna solo la parte di δ che supera `floor` volte la
  sorpresa media degli ultimi minuti (la dopamina ha un livello di base e
  adatta la sua scala: Tobler, Fiorillo, Schultz 2005). Il critico impara
  invece da tutto δ: è il suo errore.

Lesioni (`lesions`): "B" (E, V, C, D leggono A invece di B), "D", "C", "L",
"U", "V" (niente critico: δ = ricompensa grezza), "S" (salienza solo innata).
"""

from dataclasses import dataclass, field, fields

import numpy as np

from ..sim.pile import ENERGY, EYES, N_FIN, N_FOUT, N_PROGRAMS, N_SENSES, PAIN, SECTORS, TASTE, default_motor

FAST = (*range(EYES.start, EYES.stop), TASTE, PAIN)  # canali con la loro versione "cosa è cambiato": occhi, gusto, dolore
N_HEADS = N_PROGRAMS + 1  # E sceglie un programma, e se mordere (l'ultima "testa")
BITE = N_PROGRAMS
FOVEA = ("oggetto rosso", "oggetto giallo", "oggetto blu", "oggetto vicino", "oggetto a sinistra",
         "oggetto a destra", "oggetto nuovo")
N_A = N_SENSES + len(FAST) + len(FOVEA)
IDENTITY = 3  # i primi canali della fovea dicono che cos'è l'oggetto (il colore); gli altri dove sta
APPROACH, AWAY, LEAVE, EXPLORE = range(4)


def fovea_of(col, near, ang, has, new, xp=np):
    """I canali della fovea: (N, 7). `col`, `near`, `ang` dell'oggetto attento,
    `has` se c'è, `new` quanto è appena diventato l'oggetto attento."""
    one = (col[:, None] == xp.arange(3)[None]) & has[:, None]
    side = SECTORS[-1]
    return xp.concatenate([one.astype(float), xp.stack([
        xp.where(has, near, 0.0), xp.where(has, xp.clip(-ang / side, 0.0, 1.0), 0.0),
        xp.where(has, xp.clip(ang / side, 0.0, 1.0), 0.0), new], 1)], 1)


class Interface:
    """A: fissa. Per ogni canale il valore ("com'è"); per occhi, gusto e
    dolore anche quanto è appena salito ("cosa è cambiato", τ = `tau`); poi la
    fovea."""

    def __init__(self, n: int, dt: float, tau: float = 0.3):
        self.keep = np.exp(-dt / tau)
        self.slow = np.zeros((n, len(FAST)))

    def __call__(self, raw: np.ndarray, fovea: np.ndarray) -> np.ndarray:
        x = raw[:, FAST]
        fast = np.clip(x - self.slow, 0.0, 1.0)
        self.slow = self.keep * self.slow + (1.0 - self.keep) * x
        return np.concatenate([raw, fast, fovea], axis=1)

    def reset(self, who):
        self.slow[who] = 0.0


@dataclass
class Genes:
    """I parametri che l'evoluzione tara (fase B); nella fase A sono questi
    valori per tutti. Velocità e tempi in scala logaritmica nel genoma."""

    eta_b: float = 0.05  # B, per secondo
    eta_c: float = 20.0  # C, per aggiornamento (δ è già per passo): in un colpo, poi satura (c_max)
    eta_e: float = 1.0
    eta_v: float = 0.1
    eta_l: float = 0.01  # L, per secondo di riposo
    eta_att: float = 0.5  # salienza imparata
    tau_d: float = 4.0  # D: quanto dura il ricordo (s)
    temperature: float = 0.5  # E: esplorazione
    surprise_rate: float = 1.0  # U: quanto le sorprese alzano le velocità
    surprise_explore: float = 1.0  # U: quanto alzano l'esplorazione
    horizon: float = 4.0  # V: quanto lontano guarda (s)

    LOG = ("eta_b", "eta_c", "eta_e", "eta_v", "eta_l", "eta_att", "tau_d", "temperature", "horizon")


GENE_NAMES = tuple(f.name for f in fields(Genes))
INNATE_CAP = 0.2  # tetto dei valori innati legati ai sensi (per secondo di senso pieno)
DELTA_MAX = 3.0  # δ satura qui
WEIGHT_MAX = 5.0  # nessun peso imparato oltre questo
N_MOTOR = N_FOUT * N_FIN
N_GENOME = len(GENE_NAMES) + N_SENSES + N_MOTOR  # i parametri, i valori innati per ogni senso, i pesi di F
INNATE = slice(len(GENE_NAMES), len(GENE_NAMES) + N_SENSES)
MOTOR = slice(INNATE.stop, N_GENOME)


def default_genome() -> np.ndarray:
    g = Genes()
    v = [np.log(getattr(g, k)) if k in Genes.LOG else getattr(g, k) for k in GENE_NAMES]
    return np.concatenate([np.asarray(v, dtype=float), np.zeros(N_SENSES), default_motor().ravel()])


def decode(genome: np.ndarray) -> dict:
    """(N, N_GENOME) -> {nome: (N,)}, "innate" (N, N_SENSES) e "motor" (N, 4, 14), nei loro limiti."""
    out = {}
    for k, name in enumerate(GENE_NAMES):
        v = genome[:, k]
        out[name] = np.exp(v) if name in Genes.LOG else np.maximum(v, 0.0)
    out["innate"] = np.clip(genome[:, INNATE], -INNATE_CAP, INNATE_CAP)
    out["motor"] = genome[:, MOTOR].reshape(-1, N_FOUT, N_FIN)
    return out


def mutate(genome: np.ndarray, rng: np.random.Generator, sigma: float = 0.1) -> np.ndarray:
    g = genome + rng.normal(0.0, sigma, genome.shape)
    g[:, INNATE] = np.clip(g[:, INNATE], -INNATE_CAP, INNATE_CAP)
    return g


def cross(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Incrocio uniforme: ogni gene da uno dei due genitori."""
    return np.where(rng.random(a.shape) < 0.5, a, b)


@dataclass
class Sizes:
    b: int = 16  # prototipi di B
    b_winners: int = 3
    c: int = 64  # neuroni di C
    c_fan_in: int = 4
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
    context: float = 0.1  # C: peso del contesto (B) rispetto all'identità dell'oggetto guardato (1)
    memory: float = 0.5  # C: peso dell'oggetto ricordato in D
    c_threshold: float = 0.9  # C: si accendono tutti i neuroni sopra questa soglia (non un numero fisso)
    c_max: float = 0.3  # C: ogni voto satura qui, così conta l'ultima esperienza e un dolore ribalta tutto
    tag: float = 0.5  # tracce strette: si marca solo da neuroni accesi sopra questo
    floor: float = 1.0  # soglia: insegna solo la parte di |δ| sopra questo × la sorpresa media
    lesions: frozenset = field(default_factory=frozenset)


# Salienza innata (uguale per tutti): vicinanza, novità, inerzia, inibizione del ritorno.
SALIENCE = dict(near=2.0, novelty=1.5, stick=1.0, inhibit=4.0, inhibit_time=3.0, temperature=0.4, new_tau=0.5)


class Attention:
    """L'oggetto su cui ogni essere è concentrato (vedi la docstring del modulo)."""

    def __init__(self, n: int, m: int, dt: float, rng: np.random.Generator):
        self.n, self.m, self.dt, self.rng = n, m, dt, rng
        self.target = np.full(n, -1)
        self.inhib = np.full(n, -1)
        self.inhib_t = np.zeros(n)
        self.prev_vis = np.zeros((n, m), dtype=bool)
        self.novelty = np.zeros((n, m))
        self.w = np.zeros((n, 3))  # salienza imparata, per colore
        self.e = np.zeros((n, 3))  # la sua traccia
        self.new = np.zeros(n)

    def update(self, objs: dict, decide: bool, extra=None) -> None:
        """Sceglie l'oggetto (a ogni decisione, o subito se quello attento è
        sparito). `extra` (n, m): punteggio in più (il guidato bara così)."""
        S, dt, n = SALIENCE, self.dt, self.n
        vis, col, near = objs["vis"], objs["col"], objs["near"]
        idx = np.arange(n)
        self.novelty = self.novelty * np.exp(-dt / 0.5) + (vis & ~self.prev_vis)
        self.prev_vis = vis.copy()
        self.inhib_t = np.maximum(self.inhib_t - dt, 0.0)
        lost = (self.target >= 0) & ~vis[idx, np.maximum(self.target, 0)]
        self.target = np.where(lost, -1, self.target)
        self.new *= np.exp(-dt / S["new_tau"])
        self.e *= np.exp(-dt / 2.0)
        choose = (decide | (self.target < 0)) & vis.any(1)
        if not choose.any():
            return
        colour = np.eye(3)[col]  # (n, m, 3)
        slots = np.arange(self.m)[None]
        score = (S["near"] * near + S["novelty"] * self.novelty + (colour * self.w[:, None, :]).sum(-1)
                 + S["stick"] * (slots == self.target[:, None])
                 - S["inhibit"] * ((slots == self.inhib[:, None]) & (self.inhib_t > 0)[:, None]))
        if extra is not None:
            score = score + extra
        z = np.where(vis, score / S["temperature"], -np.inf)
        z = z - np.where(vis.any(1, keepdims=True), z.max(1, keepdims=True), 0.0)
        p = np.where(vis, np.exp(z), 0.0)
        p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
        sel = (np.cumsum(p, 1) < self.rng.random((n, 1))).sum(1).clip(0, self.m - 1)
        sel = np.where(choose, sel, self.target)
        changed = choose & (sel != self.target)
        self.new = np.where(changed, 1.0, self.new)
        self.target = sel
        # Traccia della salienza: il colore scelto rispetto a quello atteso.
        dev = colour[idx, np.maximum(sel, 0)] - (p[:, :, None] * colour).sum(1)
        self.e += np.where(choose[:, None], dev, 0.0)

    def learn(self, delta: np.ndarray, eta: np.ndarray) -> None:
        self.w = np.clip(self.w + (eta * delta)[:, None] * self.e, -5.0, 5.0)

    def leave(self, mask: np.ndarray) -> None:
        """"Lascialo": l'oggetto attento viene ignorato per qualche secondo."""
        self.inhib = np.where(mask, self.target, self.inhib)
        self.inhib_t = np.where(mask, SALIENCE["inhibit_time"], self.inhib_t)
        self.target = np.where(mask, -1, self.target)

    def seen(self, objs: dict):
        """(c'è, colore, vicinanza, angolo) dell'oggetto attento."""
        idx, t = np.arange(self.n), np.maximum(self.target, 0)
        has = self.target >= 0
        return has, objs["col"][idx, t], objs["near"][idx, t], objs["ang"][idx, t]

    def fovea(self, objs: dict) -> np.ndarray:
        has, col, near, ang = self.seen(objs)
        return fovea_of(col, near, ang, has, self.new)

    def reset(self, who) -> None:
        for x, v in ((self.target, -1), (self.inhib, -1), (self.inhib_t, 0.0), (self.novelty, 0.0),
                     (self.prev_vis, False), (self.w, 0.0), (self.e, 0.0), (self.new, 0.0)):
            x[who] = v


class Agent:
    """Ciò che serve all'arena: `step(raw, reward, objs)` -> programmi (N,),
    `att` l'attenzione (l'oggetto attento guida F), e `reborn(who, genomes)`
    quando un corpo rinasce."""

    name = "agente"
    decide_every = 0.5  # s: ogni quanto E (e l'attenzione) rimettono in gara le scelte

    def __init__(self, n: int, dt: float, seed: int = 0, genomes=None, objects: int = 1):
        self.n, self.dt = n, dt
        self.rng = np.random.default_rng(seed)
        self.genome = np.tile(default_genome(), (n, 1)) if genomes is None else np.array(genomes, dtype=float)
        self.p = decode(self.genome)
        self.att = Attention(n, objects, dt, self.rng)
        self.clock = 0.0
        self.program = np.full(n, EXPLORE)
        self.pi = np.full((n, N_PROGRAMS), 1.0 / N_PROGRAMS)
        self.bite = np.zeros(n, dtype=bool)  # si morde (deciso a ogni decisione)
        self.p_bite = np.full(n, 0.5)

    def _tick(self) -> bool:
        self.clock += self.dt
        if self.clock >= self.decide_every - 1e-9:
            self.clock = 0.0
            return True
        return False

    def innate_reward(self, raw: np.ndarray) -> np.ndarray:
        """I valori innati legati ai sensi (per passo)."""
        return (self.p["innate"] * raw).sum(1) * self.dt

    def reborn(self, who: np.ndarray, genomes=None) -> None:
        if genomes is not None:
            self.genome[who] = genomes
            self.p = decode(self.genome)
        self.att.reset(who)
        self.bite[who] = False


class Guided(Agent):
    """Guidato a mano, per controllare il mondo: sa qual è il colore buono
    (`good`, (N,), aggiornato da fuori). Si concentra sulla pila buona più
    vicina e ci va; se vede solo pile cattive le lascia ed esplora (chi lo usa
    lo fa andare verso la zona, che sa)."""

    name = "guidato"

    def __init__(self, n, dt, seed=0, genomes=None, objects: int = 1, batteries: int = 0):
        super().__init__(n, dt, seed, genomes, objects)
        self.good = np.zeros(n, dtype=int)
        self.batteries = batteries

    def step(self, raw, reward, objs):
        decide = self._tick()
        good = objs["col"] == self.good[:, None]
        battery = np.arange(objs["col"].shape[1])[None] < self.batteries
        self.att.update(objs, decide, extra=np.where(good & battery, 10.0, -10.0))
        has, col, near, _ = self.att.seen(objs)
        ok = has & (col == self.good)
        self.program = np.where(ok, APPROACH, EXPLORE)
        self.bite = ok & (near > 0.9)  # morde solo la buona, quando ce l'ha davanti
        self.p_bite = self.bite.astype(float)
        self.att.leave(has & ~ok)  # non è la buona: la lascia
        self.pi = np.eye(N_PROGRAMS)[self.program]
        return self.program


class _Learner(Agent):
    """L'attore (E) e il critico (V) comuni ai due agenti che imparano."""

    stick = 1.5  # il programma in corso parte avvantaggiato (la selezione dei gangli della base è "appiccicosa")

    def _init_actor_critic(self, n_e: int, n_v: int):
        n = self.n
        self.w_e = np.zeros((n, N_HEADS, n_e))
        self.e_e = np.zeros((n, N_HEADS, n_e))
        self.w_v = np.zeros((n, n_v))
        self.e_v = np.zeros((n, n_v))
        self.v_prev = np.zeros(n)
        self.fresh = np.ones(n, dtype=bool)  # appena nato: niente δ al primo passo
        self.program = self.rng.integers(0, N_PROGRAMS, n)
        self.delta = np.zeros(n)
        self.teach = np.zeros(n)  # la parte di δ che insegna (dopo la soglia)

    def _td(self, f_v: np.ndarray, reward: np.ndarray, use_critic: bool = True) -> np.ndarray:
        """δ e l'aggiornamento del critico (TD(λ), passo normalizzato)."""
        p, dt = self.p, self.dt
        gamma = np.exp(-dt / p["horizon"])
        value = (self.w_v * f_v).sum(1)
        if use_critic:
            delta = np.where(self.fresh, 0.0, reward + gamma * value - self.v_prev)
        else:
            delta = reward
        delta = np.clip(delta, -DELTA_MAX, DELTA_MAX)  # la dopamina satura
        norm = 1.0 + (f_v**2).sum(1) + (self.e_v * f_v).sum(1)  # sulla traccia: con ingressi costanti si accumula
        self.w_v += (p["eta_v"] * delta / norm)[:, None] * self.e_v
        self.e_v = (gamma * np.exp(-dt / 1.0))[:, None] * self.e_v + f_v
        self.v_prev = (self.w_v * f_v).sum(1)
        self.fresh[:] = False
        self.delta = delta
        return delta

    def _act(self, logits, f_e, teach, eta, temp, decide, f_tag=None):
        """Attore: impara con la parte di δ che insegna, sulla traccia; poi, se
        è il momento, sceglie. `f_tag`: gli ingressi che possono marcare."""
        dt = self.dt
        norm = 1.0 + (f_e**2).sum(1)
        self.w_e = np.clip(self.w_e + (eta * teach / norm)[:, None, None] * self.e_e, -WEIGHT_MAX, WEIGHT_MAX)
        self.e_e *= np.exp(-dt / 2.0)
        if decide:
            z = (logits[:, :N_PROGRAMS] + self.stick * np.eye(N_PROGRAMS)[self.program]) / temp[:, None]
            z -= z.max(1, keepdims=True)
            pi = np.exp(z)
            pi /= pi.sum(1, keepdims=True)
            u = self.rng.random((self.n, 1))
            self.program = (np.cumsum(pi, 1) < u).sum(1).clip(0, N_PROGRAMS - 1)
            self.pi = pi
            # Il morso: sì o no (probabilità dalla sua testa).
            self.p_bite = 1.0 / (1.0 + np.exp(-np.clip(logits[:, BITE] / temp, -30, 30)))
            self.bite = self.rng.random(self.n) < self.p_bite
            dev = np.concatenate([np.eye(N_PROGRAMS)[self.program] - pi, (self.bite - self.p_bite)[:, None]], 1)
            tag = f_e if f_tag is None else f_tag
            self.e_e += dev[:, :, None] * tag[:, None, :]  # la scelta rispetto all'attesa
            self._chose(dev)
            self.att.leave(self.program == LEAVE)
        return self.program

    def _chose(self, deviation):
        pass

    def _reset_actor_critic(self, who):
        for x in (self.w_e, self.e_e, self.w_v, self.e_v, self.v_prev, self.delta, self.teach):
            x[who] = 0.0
        self.fresh[who] = True


class ActorCritic(_Learner):
    """Il riferimento: attore-critico classico (TD, Doya 2000) che legge
    direttamente l'interfaccia A (con la fovea). Stessi programmi, stessa
    regola di E e V del cervello a sezioni, salienza solo innata; senza B, C,
    D, L e U, senza tracce strette e senza soglia."""

    name = "attore-critico"

    def __init__(self, n, dt, seed=0, genomes=None, objects: int = 1):
        super().__init__(n, dt, seed, genomes, objects)
        self.a = Interface(n, dt)
        self._init_actor_critic(N_A + 1, N_A + 1)

    def step(self, raw, reward, objs):
        decide = self._tick()
        self.att.update(objs, decide)
        a = self.a(raw, self.att.fovea(objs))
        f = np.concatenate([a, np.ones((self.n, 1))], 1)
        delta = self._td(f, reward + self.innate_reward(raw))
        self.teach = delta
        logits = (self.w_e * f[:, None, :]).sum(-1)
        return self._act(logits, f, delta, self.p["eta_e"], self.p["temperature"], decide)

    def reborn(self, who, genomes=None):
        super().reborn(who, genomes)
        self.a.reset(who)
        self._reset_actor_critic(who)


def c_threshold(fov, s: Sizes, xp=np):
    """La soglia di C: se si guarda qualcosa, ciò che si vede (sopra 0,9: solo
    i neuroni che lo ricevono); altrimenti ciò che si ricorda in D (con peso
    `memory`, la soglia scende in proporzione)."""
    seeing = fov[:, :IDENTITY].max(1) > 0
    return xp.where(seeing, s.c_threshold, s.c_threshold * s.memory)


class Sections(_Learner):
    """Il cervello a sezioni (vedi la docstring del modulo)."""

    name = "sezioni"

    def __init__(self, n, dt, seed=0, genomes=None, sizes: Sizes | None = None, objects: int = 1):
        super().__init__(n, dt, seed, genomes, objects)
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
        nf = len(FOVEA)
        # D: l'ultimo oggetto guardato (copia della fovea), e il cancello (pesi che imparano).
        self.mem = np.zeros((n, nf))
        self.w_gate = np.zeros((n, nf))
        self.e_gate = np.zeros((n, nf))
        # C: ingresso fisso, casuale e sparso (che cos'è l'oggetto guardato e quello
        # ricordato in D, più il contesto da B); uscite verso E (C veloce, L lenta).
        self.c_in = np.zeros((n, s.c, 2 * IDENTITY + nb))
        for j in range(s.c):
            pick = np.argsort(self.rng.random((n, 2 * IDENTITY + nb)), axis=1)[:, : s.c_fan_in]
            np.put_along_axis(self.c_in[:, j], pick, 1.0, axis=1)
        self.u_c = np.zeros((n, N_HEADS, s.c))
        self.e_c = np.zeros((n, N_HEADS, s.c))
        self.u_l = np.zeros((n, N_HEADS, s.c))
        self.code = np.zeros((n, s.c))
        # U e soglia: medie di |δ|.
        self.d_fast = np.zeros(n)
        self.d_slow = np.full(n, 1e-3)
        self.unc = np.zeros(n)
        # E legge B, D e la costante; V legge B, D, C e la costante.
        self._init_actor_critic(nb + nf + 1, nb + nf + s.c + 1)

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
        rate = (self.p["eta_b"] * dt * (1.0 + s.surprise_boost * np.minimum(np.abs(delta), 1.0)))[:, None, None]
        self.proto += rate * y[:, :, None] * ((a[:, None, :] - self.proto) + s.predict * err[:, None, :])
        self.win_rate += (dt / 30.0) * ((y > 0) / s.b_winners - self.win_rate)
        self.prev_b = self.state_b.copy()
        self.state_b += (dt / s.tau_b) * (y - self.state_b)
        return self.state_b

    # --- D ---------------------------------------------------------------------------

    def _d(self, fov: np.ndarray, teach: np.ndarray) -> np.ndarray:
        """Si scrive quando nella fovea entra un oggetto nuovo (innato) o quando
        il cancello lo ha imparato; altrimenti si tiene, anche a fovea vuota."""
        s, dt = self.s, self.dt
        if "D" in self.les:
            return np.zeros_like(fov)
        self.w_gate = np.clip(self.w_gate + (s.eta_gate * teach)[:, None] * self.e_gate, -2 * WEIGHT_MAX, 2 * WEIGHT_MAX)
        z = s.gate_event * fov[:, -1] + (self.w_gate * fov).sum(1) + s.gate_bias
        g = 1.0 / (1.0 + np.exp(-z))
        self.e_gate = np.exp(-dt / 2.0) * self.e_gate + (g * (1.0 - g))[:, None] * np.where(fov > s.tag, fov, 0.0)
        keep = np.exp(-dt / self.p["tau_d"])[:, None]
        self.mem = (1.0 - g)[:, None] * keep * self.mem + g[:, None] * fov
        return self.mem

    # --- C ---------------------------------------------------------------------------

    def _c(self, b: np.ndarray, m: np.ndarray, fov: np.ndarray) -> np.ndarray:
        s = self.s
        if "C" in self.les:
            return np.zeros((self.n, s.c))
        h = np.concatenate([fov[:, :IDENTITY], s.memory * m[:, :IDENTITY], s.context * b], 1)
        z = np.einsum("ncj,nj->nc", self.c_in, h)
        self.code = (z >= c_threshold(fov, s)[:, None]).astype(float)
        return self.code

    def _chose(self, deviation):
        self.e_c += deviation[:, :, None] * self.code[:, None, :]  # solo i neuroni accesi al momento della scelta

    # --- Un passo --------------------------------------------------------------------

    def step(self, raw, reward, objs):
        s, p, dt = self.s, self.p, self.dt
        decide = self._tick()
        self.att.update(objs, decide)
        fov = self.att.fovea(objs)
        a = self.a(raw, fov)
        r = reward + self.innate_reward(raw)
        teach_prev = self.teach
        b = self._b(a, self.delta)
        m = self._d(fov, teach_prev)
        c = self._c(b, m, fov)
        one = np.ones((self.n, 1))
        delta = self._td(np.concatenate([b, m, c, one], 1), r, use_critic="V" not in self.les)
        # Soglia adattiva e U: le medie della sorpresa.
        self.d_fast += (dt / 20.0) * (np.abs(delta) - self.d_fast)
        self.d_slow += (dt / 200.0) * (np.abs(delta) - self.d_slow)
        if "U" not in self.les:
            self.unc = np.clip(self.d_fast / np.maximum(self.d_slow, 1e-6) - 1.0, 0.0, 3.0)
        teach = np.sign(delta) * np.maximum(np.abs(delta) - s.floor * self.d_slow, 0.0)
        self.teach = teach
        boost = 1.0 + p["surprise_rate"] * self.unc
        # C impara in fretta e dimentica; L si avvicina a C solo a riposo.
        step = p["eta_c"] * boost * teach / (1.0 + self.code.sum(1))  # diviso per i neuroni accesi
        self.u_c = np.clip(self.u_c + step[:, None, None] * self.e_c - self.u_c * (dt / s.forget), -s.c_max, s.c_max)
        self.e_c *= np.exp(-dt / 2.0)
        if "L" not in self.les:
            # Riposo: carichi e fermi a ricaricarsi (il gusto della pila che si consuma).
            rest = (raw[:, ENERGY] > s.rest_energy) & (raw[:, TASTE] > 0.1)
            self.u_l += (p["eta_l"] * dt * rest)[:, None, None] * (self.u_c - self.u_l)
        if "S" not in self.les:
            self.att.learn(teach, p["eta_att"] * boost)
        f_e = np.concatenate([b, m, one], 1)
        f_tag = np.concatenate([np.where(b > s.tag, b, 0.0), np.where(m > s.tag, m, 0.0), one], 1)
        logits = (self.w_e * f_e[:, None, :]).sum(-1) + np.einsum("npc,nc->np", self.u_c, c)
        if "L" not in self.les:
            logits += np.einsum("npc,nc->np", self.u_l, c)
        temp = p["temperature"] * (1.0 + p["surprise_explore"] * self.unc)
        return self._act(logits, f_e, teach, p["eta_e"] * boost, temp, decide, f_tag)

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
