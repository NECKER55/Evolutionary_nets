from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from ..config import Config
from .geometry import brain_side, connect_prob, distances
from .receptors import P, Prepared, ReceptorGenes

# Geni del meccanismo di mutazione, uno per individuo (vedi genetics.mutate e
# docs/SPEC_FASE_C.md, sezione 5). Nel genoma stanno come logaritmi.
META = ("weight_prob", "weight_std", "conn_add", "conn_del", "toggle", "dup", "pos_prob", "pos_std",
        "rec_drift", "conn_range")
META_INDEX = {name: k for k, name in enumerate(META)}


def meta_start(cfg: Config) -> np.ndarray:
    """I valori di partenza dei geni delle mutazioni (dalla configurazione)."""
    return np.array([cfg.mut_weight_prob, cfg.mut_weight_std, cfg.mut_conn_add, cfg.mut_conn_del,
                     cfg.mut_neuron_toggle, cfg.mut_dup, cfg.mut_pos_prob, cfg.mut_pos_std,
                     cfg.mut_rec_drift, cfg.net_conn_range])


PROBABILITIES = [META_INDEX[k] for k in ("weight_prob", "conn_add", "conn_del", "toggle", "dup", "pos_prob")]


# Geni della plasticità, in coda ai geni propri di ogni neurone interno o
# motore (vedi docs/SPEC_FASE_E.md). Riguardano le sinapsi che il neurone
# RICEVE: Δw = η · M · (A·pre·post + B·pre + C·post + D), con
# η = plasticity_initial_rate · e^gene (gene 0 = plasticità debole di partenza).
# "mod" > 0 rende un neurone interno modulatore: non trasmette, rilascia M.
PLASTIC = ("eta", "A", "B", "C", "D", "mod")


def clamp_meta(meta: np.ndarray, cfg: Config) -> None:
    """Sul posto: nessun tasso sotto l'errore minimo di copiatura, nessuna
    probabilità sopra 1 e nessuna ampiezza sopra 100 (oltre non ha più
    significato: il cervello ha lato 1, i pesi stanno per lo più entro ±3)."""
    np.clip(meta, np.log(cfg.mutation_floor), np.log(100.0), out=meta)
    meta[:, PROBABILITIES] = np.minimum(meta[:, PROBABILITIES], 0.0)


def live_links(conn: np.ndarray, active: np.ndarray, r_active: np.ndarray) -> np.ndarray:
    """Collegamenti che contano davvero: presenti nel genoma e tra neuroni
    esistenti (recettori e interni accesi, motori sempre).
    conn (N, T, S), active (N, H), r_active (N, R) -> (N, T, S)."""
    n = len(conn)
    n_out = conn.shape[1] - active.shape[1]
    target_on = np.concatenate([active, np.ones((n, n_out), dtype=bool)], axis=1)
    source_on = np.concatenate([r_active, target_on], axis=1)
    return conn & target_on[:, :, None] & source_on[:, None, :]


def retention(gene: np.ndarray) -> np.ndarray:
    """Reti a impulsi: quota di carica che un neurone trattiene a ogni passo,
    dal gene "perdita". Da ~0.5 (neurone rapido) a ~0.97 (lento), tipicamente ~0.8."""
    return 1.0 / (1.0 + np.exp(-(1.5 + gene)))


@dataclass
class Genome:
    """I genomi di un gruppo di individui, uno per riga.

    Cervello (vedi `Brains`):
      weights (N, T, S), conn (N, T, S), traits (N, T, K), active (N, H), pos (N, H, 2)
      con S = R + H + n_out sorgenti (recettori, interni, motori) e
      T = H + n_out destinatari (interni, motori).
    Recettori (vedi `receptors.ReceptorGenes`): i campi `r_*`, R posti.
    Meccanismo di mutazione: `meta` (N, len(META)), logaritmi.

    R e H sono "posti": un recettore o un neurone esiste solo se acceso. Sono
    uguali per tutta la popolazione e crescono quando servono (`pad`).
    """

    weights: np.ndarray
    traits: np.ndarray
    conn: np.ndarray
    active: np.ndarray
    pos: np.ndarray
    r_active: np.ndarray
    r_sens: np.ndarray
    r_mu: np.ndarray
    r_width: np.ndarray
    r_body: np.ndarray
    r_look: np.ndarray
    r_aperture: np.ndarray
    r_y: np.ndarray
    r_tau: np.ndarray
    r_sub: np.ndarray
    r_div: np.ndarray
    meta: np.ndarray

    FIELDS = ("weights", "traits", "conn", "active", "pos", "r_active", "r_sens", "r_mu", "r_width",
              "r_body", "r_look", "r_aperture", "r_y", "r_tau", "r_sub", "r_div", "meta")
    RECEPTOR_FIELDS = FIELDS[5:16]

    def __len__(self) -> int:
        return len(self.weights)

    @property
    def n_rec(self) -> int:
        return self.r_active.shape[1]

    @property
    def n_hidden(self) -> int:
        return self.active.shape[1]

    @property
    def n_out(self) -> int:
        return self.weights.shape[1] - self.n_hidden

    @property
    def receptors(self) -> ReceptorGenes:
        """I geni dei recettori (gli stessi array, non una copia)."""
        return ReceptorGenes(*(getattr(self, f) for f in self.RECEPTOR_FIELDS))

    def rate(self, name: str) -> np.ndarray:
        """(N,) un gene del meccanismo di mutazione, nella sua scala naturale."""
        return np.exp(self.meta[:, META_INDEX[name]])

    def take(self, idx) -> "Genome":
        """Copia indipendente degli individui indicati."""
        return Genome(*(getattr(self, f)[idx].copy() for f in self.FIELDS))

    @staticmethod
    def concat(parts: list["Genome"]) -> "Genome":
        """Uno sotto l'altro; i posti vengono pareggiati al più grande."""
        r = max(p.n_rec for p in parts)
        h = max(p.n_hidden for p in parts)
        parts = [pad(p, r, h) for p in parts]
        return Genome(*(np.concatenate([getattr(p, f) for p in parts]) for f in Genome.FIELDS))


def pad(g: Genome, n_rec: int, n_hidden: int) -> Genome:
    """Lo stesso genoma con più posti (spenti, geni a zero) per recettori e
    neuroni interni. Non cambia nulla per l'individuo: i posti nuovi non
    esistono finché una mutazione non li accende o una duplicazione non li
    riempie."""
    dr, dh = n_rec - g.n_rec, n_hidden - g.n_hidden
    if dr == 0 and dh == 0:
        return g
    if dr < 0 or dh < 0:
        raise ValueError("pad: i posti possono solo crescere")
    n, r, h = len(g), g.n_rec, g.n_hidden

    def grow(a, axis, at, k):
        shape = list(a.shape)
        shape[axis] = k
        return np.insert(a, [at] * k, np.zeros(shape, dtype=a.dtype), axis=axis) if k else a

    def links(a):  # (N, T, S): righe nuove dopo H, colonne nuove dopo R e dopo R + H
        a = grow(a, 1, h, dh)
        a = grow(a, 2, r + h, dh)
        return grow(a, 2, r, dr)

    rec = {f: grow(getattr(g, f), 1, r, dr) for f in Genome.RECEPTOR_FIELDS}
    rec["r_body"][:, r:] = rec["r_look"][:, r:] = [1.0, 0.0]  # vettori unitari validi
    return Genome(
        links(g.weights), grow(g.traits, 1, h, dh), links(g.conn), grow(g.active, 1, h, dh),
        np.concatenate([g.pos, np.full((n, dh, 2), 0.5)], axis=1), **rec, meta=g.meta,
    )


def primordial(cfg: Config, n: int, n_out: int, n_traits: int, rng: np.random.Generator) -> Genome:
    """La partenza primordiale (docs/SPEC_FASE_C.md, sezione 7): pochi
    recettori generalisti, nessun neurone interno acceso, collegamenti
    secondo la geometria. Gli individui differiscono un poco nei geni."""
    R, H, std = cfg.receptor_slots, cfg.net_hidden_slots, cfg.net_weight_std
    on = min(cfg.receptor_initial, R)
    # Recettori: i primi `on` accesi, distribuiti a ventaglio davanti; gli
    # altri posti hanno geni a caso ma spenti (pseudogeni).
    angle = rng.uniform(-np.pi, np.pi, (n, R))
    if on:
        spread = np.radians(cfg.receptor_initial_spread_deg)
        angle[:, :on] = np.linspace(-spread / 2, spread / 2, on) if on > 1 else 0.0
    unit = np.stack([np.cos(angle), np.sin(angle)], axis=2)
    r_active = np.zeros((n, R), dtype=bool)
    r_active[:, :on] = True
    r_sens = np.log(cfg.receptor_initial_gain) + rng.normal(0.0, 0.2, (n, R, P))
    r_mu = np.clip(0.5 + rng.normal(0.0, 0.05, (n, R, P)), 0.0, 1.0)
    r_width = np.log(cfg.receptor_initial_width) + rng.normal(0.0, 0.1, (n, R, P))
    r_aperture = np.full((n, R), np.log(np.radians(cfg.receptor_initial_aperture_deg)))
    # Mappa topografica: il nervo entra nel cervello a un'altezza che segue la
    # direzione in cui il recettore guarda.
    r_y = np.clip(0.5 + angle / (2.0 * np.pi), 0.0, 1.0)
    # Adattamento rudimentale fin da subito, diverso da individuo a individuo.
    r_tau = np.log(cfg.receptor_initial_tau) + rng.normal(0.0, 0.3, (n, R))
    r_sub = rng.uniform(0.0, cfg.receptor_initial_adaptation, (n, R))
    r_div = rng.uniform(0.0, cfg.receptor_initial_adaptation, (n, R))

    pos = rng.random((n, H, 2))
    meta = np.log(meta_start(cfg))[None, :] + rng.normal(0.0, 0.1, (n, len(META)))
    clamp_meta(meta, cfg)
    dist = distances(cfg, pos, r_y, n_out, brain_side(cfg, r_active, np.zeros((n, H), dtype=bool), n_out))
    conn = rng.random(dist.shape) < connect_prob(cfg, dist, R, np.exp(meta[:, META_INDEX["conn_range"]]))
    T, S = H + n_out, R + H + n_out
    return Genome(
        weights=rng.normal(0.0, std, (n, T, S)), traits=rng.normal(0.0, std, (n, T, n_traits)),
        conn=conn, active=np.zeros((n, H), dtype=bool), pos=pos,
        r_active=r_active, r_sens=r_sens, r_mu=r_mu, r_width=r_width, r_body=unit.copy(), r_look=unit,
        r_aperture=r_aperture, r_y=r_y, r_tau=r_tau, r_sub=r_sub, r_div=r_div, meta=meta,
    )


class Brains:
    """Le reti di tutta la popolazione, una per riga.

    Ogni rete ha R recettori (i sensi: vedi receptors.py), H neuroni interni e
    `n_out` motori. R e H sono posti: contano solo quelli accesi. I
    collegamenti sono liberi: qualsiasi neurone (recettore, interno, motore)
    può parlare a qualsiasi neurone interno o motore, se stesso compreso. Non
    esistono strati.

    A ogni passo tutti i neuroni si aggiornano insieme a partire dai valori del
    passo precedente, che restano in `state`: gli anelli sono la memoria.

    I collegamenti nascono soprattutto tra neuroni vicini, e ogni collegamento
    effettivo costa energia in proporzione alla sua lunghezza (`wire_length`).
    Costano anche ogni neurone interno acceso e ogni recettore (`upkeep`).
    """

    KIND = "rate"
    BASE_TRAITS = 1  # soglia; poi i geni della plasticità
    N_TRAITS = BASE_TRAITS + len(PLASTIC)
    OUT_LABELS = ("spinta", "sterzo")

    @classmethod
    def trait_text(cls, traits: np.ndarray, hidden: bool) -> list[str]:
        """I geni propri di un neurone, in parole."""
        return [f"soglia {traits[0]:+.3f}"] + cls.plastic_text(traits, hidden)

    @classmethod
    def plastic_text(cls, traits: np.ndarray, hidden: bool) -> list[str]:
        eta, a, b, c, d, mod = traits[cls.BASE_TRAITS:]
        lines = [f"plasticità in entrata: η0 x {np.exp(eta):.2g}",
                 f"  A {a:+.2f} pre·post   B {b:+.2f} pre",
                 f"  C {c:+.2f} post       D {d:+.2f}"]
        if hidden:
            lines.append("modulatore: sì (non trasmette, rilascia M)" if mod > 0
                         else f"modulatore: no (gene {mod:+.2f})")
        return lines

    def __init__(self, cfg: Config, n: int, n_out: int = 2, seed: int = 0):
        self.cfg, self.n_out = cfg, n_out
        rng = np.random.default_rng(seed)
        genome = primordial(cfg, n, n_out, self.N_TRAITS, rng)
        self._init_genes(cfg, rng, genome)
        self._init_plasticity(cfg, rng, genome)
        self.set_genome(genome)

    def _init_plasticity(self, cfg: Config, rng: np.random.Generator, genome: Genome) -> None:
        """Plasticità debole e diversa da individuo a individuo; una quota
        `modulatory_initial_share` dei neuroni nasce modulatrice."""
        t = genome.traits[:, :, self.BASE_TRAITS:]
        t[..., 0] = rng.normal(0.0, 0.5, t.shape[:2])
        t[..., 1:5] = rng.normal(0.0, cfg.plasticity_initial_rule, t.shape[:2] + (4,))
        offset = NormalDist().inv_cdf(1.0 - cfg.modulatory_initial_share)
        t[..., 5] = rng.normal(0.0, 1.0, t.shape[:2]) - offset

    def _init_genes(self, cfg: Config, rng: np.random.Generator, genome: Genome) -> None:
        """Per i tipi di rete che hanno geni con una distribuzione iniziale propria."""

    @property
    def bias(self) -> np.ndarray:
        return self.traits[:, :, 0]

    @property
    def modulatory(self) -> np.ndarray:
        """(N, H): True dove il neurone interno è un modulatore acceso."""
        return (self.traits[:, : self.n_hidden, self.BASE_TRAITS + 5] > 0.0) & self.active

    @property
    def genome(self) -> Genome:
        return self._genome

    def set_genome(self, genome: Genome) -> None:
        """Sostituisce l'intera popolazione (stesso numero di individui o no,
        stessi posti o no)."""
        self._genome = genome
        self.weights, self.traits, self.conn = genome.weights, genome.traits, genome.conn
        self.active, self.pos = genome.active, genome.pos
        self.n, self.n_in, self.n_hidden = len(genome), genome.n_rec, genome.n_hidden
        self.compile()
        self.reset_state()

    @property
    def receptors(self) -> ReceptorGenes:
        return self._genome.receptors

    def compile(self) -> None:
        """Da chiamare dopo ogni modifica al genoma: prepara i pesi effettivi,
        azzerando ciò che tocca un neurone o un recettore inesistente."""
        g, cfg = self._genome, self.cfg
        R, H = self.n_in, self.n_hidden
        ones_out = np.ones((self.n, self.n_out), dtype=bool)
        self.target_on = np.concatenate([self.active, ones_out], axis=1)
        self.links = live_links(self.conn, self.active, g.r_active)
        # I modulatori non trasmettono: le loro uscite non esistono (né costano).
        mod = self.modulatory
        self.links[:, :, R : R + H] &= ~mod[:, None, :]
        self.side = brain_side(cfg, g.r_active, self.active, self.n_out)  # il cervello cresce con le cellule
        self.dist = distances(cfg, self.pos, g.r_y, self.n_out, self.side)
        # Plasticità: M arriva a ogni neurone dai modulatori, per diffusione,
        # calando con la distanza (trasmissione di volume).
        p = self.traits[:, :, self.BASE_TRAITS :]
        self.eta, self.rule = cfg.plasticity_initial_rate * np.exp(p[..., 0]), p[..., 1:5]
        self.kmod = np.exp(-self.dist[:, :, R : R + H] / cfg.modulation_range) * mod[:, None, :]
        self.learns = mod.any(axis=1) & cfg.plasticity  # chi ha almeno un modulatore
        self._mod_sign = mod.astype(float)
        # Lunghezza totale dei collegamenti effettivi di ogni rete: il suo cablaggio.
        self.wire_length = (self.dist * self.links).sum(axis=(1, 2))
        # Quanto costa mantenere questo corpo e questo cervello, energia/s.
        gain = self.receptors.gain
        self.upkeep = (cfg.wire_cost * self.wire_length + cfg.neuron_cost * self.active.sum(axis=1)
                       + cfg.receptor_cell_cost * g.r_active.sum(axis=1)
                       + cfg.receptor_sens_cost * gain.sum(axis=(1, 2)))
        self._subset = None  # copie per `step_some`, da rifare
        self._prepared = None  # recettori pronti per il calcolo, da rifare
        self._compile_extra()
        self.w_eff = self._born_weights()
        self.learned = False

    def _compile_extra(self) -> None:
        """Per i tipi di rete con geni compilati propri (prima dei pesi)."""

    def _born_weights(self) -> np.ndarray:
        """I pesi effettivi alla nascita, dal genoma."""
        return self.weights * self.links

    # Segno obbligato dei pesi (legge di Dale): None = libero.
    dale = None

    def reset_state(self) -> None:
        """Vita nuova: stato a zero e pesi di nuovo quelli del genoma (ciò che
        si è imparato non si eredita e non passa da un'arena all'altra)."""
        self.state = np.zeros((self.n, self.n_hidden + self.n_out))
        self._forget()

    def _forget(self) -> None:
        if self.learned:
            self.w_eff = self._born_weights()
            self._subset = None
            self.learned = False

    # Geni compilati di cui `step_some` usa una copia ridotta.
    SUBSET = ("w_eff", "bias", "target_on")

    def _rows(self, rows: np.ndarray) -> dict:
        """Copie dei geni compilati per i soli individui `rows`. Si rifanno
        solo quando l'insieme cambia (in una generazione chi esce non rientra)."""
        if self._subset is None or not np.array_equal(self._subset["rows"], rows):
            self._subset = {"rows": rows, **{k: getattr(self, k)[rows] for k in self.SUBSET},
                            "receptors": self.prepared().take(rows)}
        return self._subset

    def prepared(self, rows: np.ndarray | None = None) -> Prepared:
        """I recettori degli individui `rows` (tutti se None), pronti per il
        calcolo dei sensi."""
        if rows is not None:
            return self._rows(rows)["receptors"]
        if self._prepared is None:
            self._prepared = Prepared(self.cfg, self._genome.receptors)
        return self._prepared

    def step(self, obs: np.ndarray) -> np.ndarray:
        """obs (N, R) -> uscite (N, n_out) in [-1, 1]."""
        before = self.state
        self.state, out = self._advance(obs, self.state, self)
        self._learn(np.arange(self.n), obs, before, self.state)
        return out

    def step_some(self, obs: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Come `step`, ma solo per gli individui `rows` (obs ha una riga per
        ciascuno): gli altri restano fermi. Ogni rete calcola per conto suo,
        quindi il risultato per `rows` è identico a quello di `step`."""
        sub = self._rows(rows)
        before = self.state[rows]
        state, out = self._advance(obs, before, sub)
        self.state[rows] = state
        self._learn(rows, obs, before, state, sub)
        return out

    def _pre(self, obs, before, after) -> np.ndarray:
        """Attività di ogni sorgente che ha appena agito sui destinatari."""
        return np.concatenate([obs, before], axis=1)

    def _learn(self, rows, obs, before, after, sub=None) -> None:
        """Plasticità modulata (docs/SPEC_FASE_E.md): per ogni collegamento
        effettivo Δw = η · M · (A·pre·post + B·pre + C·post + D), con η, A, B,
        C, D del neurone che riceve e M che arriva per diffusione dai
        modulatori. Solo chi ha almeno un modulatore impara."""
        if not self.cfg.plasticity:
            return
        local = np.flatnonzero(self.learns[rows])
        if not len(local):
            return
        who = rows[local]
        H = self.n_hidden
        pre = self._pre(obs[local], before[local], after[local])  # (n, S)
        post = after[local]  # (n, T)
        m = np.einsum("nth,nh->nt", self.kmod[who], after[local, :H] * self._mod_sign[who])
        rule = self.rule[who]
        drive = ((rule[..., 0] * post)[:, :, None] * pre[:, None, :] + rule[..., 1, None] * pre[:, None, :]
                 + (rule[..., 2] * post + rule[..., 3])[:, :, None])
        w = self.w_eff[who] + (self.eta[who] * m)[:, :, None] * drive * self.links[who]
        limit = self.cfg.synapse_max
        np.clip(w, -limit, limit, out=w)
        if self.dale is not None:  # un neurone eccitatorio non diventa inibitorio, e viceversa
            sign = self.dale[who][:, None, :]
            w = np.where(sign > 0, np.maximum(w, 0.0), np.where(sign < 0, np.minimum(w, 0.0), w))
        self.w_eff[who] = w
        if sub is not None:
            sub["w_eff"][local] = w
        elif self._subset is not None:
            self._subset = None
        self.learned = True

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
    BASE_TRAITS = 3  # corrente di fondo, perdita, tipo; poi la plasticità
    N_TRAITS = BASE_TRAITS + len(PLASTIC)
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
        w = np.where(self.dale[:, None, :] == 0.0, self.weights, np.abs(self.weights) * self.dale[:, None, :])
        return w * self.links

    def _pre(self, obs, before, after) -> np.ndarray:
        """Tracce recenti degli impulsi (e livelli dei motori) dopo il passo."""
        return np.concatenate([obs, after], axis=1)

    SUBSET = ("w_eff", "bias", "keep", "active")

    def reset_state(self) -> None:
        h = self.n_hidden
        self._forget()
        self.charge = np.zeros((self.n, h + self.n_out))
        self.spikes = np.zeros((self.n, h))
        # Per il viewer e per i collegamenti in uscita dai motori: attività
        # recente dei neuroni interni (media mobile degli impulsi) e livello dei motori.
        self.state = np.zeros((self.n, h + self.n_out))

    def step(self, obs: np.ndarray) -> np.ndarray:
        """obs (N, R) -> comandi (N, 2) in (-1, 1): spinta e sterzata."""
        before = self.state.copy()
        (self.charge, self.spikes, self.state), out = self._advance(
            obs, (self.charge, self.spikes, self.state), self)
        self._learn(np.arange(self.n), obs, before, self.state)
        return out

    def step_some(self, obs: np.ndarray, rows: np.ndarray) -> np.ndarray:
        sub = self._rows(rows)
        before = self.state[rows]
        (charge, spikes, state), out = self._advance(
            obs, (self.charge[rows], self.spikes[rows], before.copy()), sub)
        self.charge[rows], self.spikes[rows], self.state[rows] = charge, spikes, state
        self._learn(rows, obs, before, state, sub)
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
