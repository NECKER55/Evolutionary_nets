"""Le reti continue (`--net rate`): la classe `Brains`, base anche delle reti a impulsi."""

from statistics import NormalDist

import numpy as np

from ...config import Config
from .genome import GENES, PLASTIC, Genome, live_links, primordial, scale_gene, sigmoid
from .geometry import brain_side, distances
from .plasticity import PLASTICITY_CONSTANTS, frequencies, plasticity
from .receptors import Prepared, ReceptorGenes


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
    CENTRED = True  # attività = scarto dalla frequenza spontanea (vedi `frequencies`)
    BASE_TRAITS = 1  # soglia; poi i geni della plasticità
    N_TRAITS = BASE_TRAITS + len(GENES)
    GAIN = BASE_TRAITS + len(PLASTIC)  # il gene del guadagno
    OUT_LABELS = ("spinta", "sterzo")

    @classmethod
    def trait_text(cls, traits: np.ndarray, hidden: bool) -> list[str]:
        """I geni propri di un neurone, in parole."""
        return [f"soglia {traits[0]:+.3f}"] + cls.plastic_text(traits, hidden)

    @classmethod
    def plastic_text(cls, traits: np.ndarray, hidden: bool) -> list[str]:
        eta, a, b, c, d, mod, hebb, target, homeo, persist, intrinsic = traits[cls.BASE_TRAITS:][:len(PLASTIC)]
        lines = [f"plasticità in entrata: η0 x {np.exp(eta):.2g}",
                 f"  A {a:+.2f} pre·post   B {b:+.2f} pre",
                 f"  C {c:+.2f} post       D {d:+.2f}",
                 f"hebbiano di base: η_h0 x {np.exp(hebb):.2g}",
                 f"omeostasi: attività desiderata {float(sigmoid(target)):.2f}, forza h0 x {np.exp(homeo):.2g}",
                 f"traccia provvisoria: dura τ0 x {np.exp(persist):.2g}",
                 f"omeostasi della soglia: forza h0 x {np.exp(intrinsic):.2g}",
                 f"guadagno degli ingressi (sviluppo): x {np.exp(traits[cls.BASE_TRAITS + len(PLASTIC)]):.2g}",
                 f"periodo critico: hebbiano x B0 x {np.exp(traits[cls.BASE_TRAITS + len(PLASTIC) + 1]):.2g}"
                 f" all'inizio, si chiude in τ0 x {np.exp(traits[cls.BASE_TRAITS + len(PLASTIC) + 2]):.2g}",
                 f"traccia di eleggibilità: dura τ0 x {np.exp(traits[cls.BASE_TRAITS + len(PLASTIC) + 3]):.2g}"]
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
        genome.traits[:, :, self.GAIN] = 0.0  # guadagno 1: conta solo con lo sviluppo
        # Periodo critico: forte e diverso da individuo a individuo.
        genome.traits[:, :, self.GAIN + 1 : self.GAIN + 3] = rng.normal(0.0, 0.5, genome.traits.shape[:2] + (2,))
        if cfg.genome == "sviluppo":  # il primo cervello nasce dai marcatori
            from . import development

            genome.traits[:, :, self.GAIN] = rng.normal(0.0, 0.2, genome.traits.shape[:2])

            for name, value in development.initial_markers(cfg, n, genome.n_rec, genome.weights.shape[1],
                                                           rng).items():
                setattr(genome, name, value)
            genome.m_id = np.broadcast_to(np.arange(cfg.marker_slots, dtype=np.int64), (n, cfg.marker_slots)).copy()
            genome.m_fam = genome.m_id.copy()
            development.develop(genome, cfg, rng, self.GAIN)
        self.set_genome(genome)

    def _init_plasticity(self, cfg: Config, rng: np.random.Generator, genome: Genome) -> None:
        """Plasticità debole e diversa da individuo a individuo; una quota
        `modulatory_initial_share` dei neuroni nasce modulatrice."""
        t = genome.traits[:, :, self.BASE_TRAITS:]
        t[..., 0] = rng.normal(0.0, 0.5, t.shape[:2])
        t[..., 1:5] = rng.normal(0.0, cfg.plasticity_initial_rule, t.shape[:2] + (4,))
        t[..., 1] += cfg.plasticity_innate_rule  # regola innata: A·pre·post (vedi Config)
        offset = NormalDist().inv_cdf(1.0 - cfg.modulatory_initial_share)
        t[..., 5] = rng.normal(0.0, 1.0, t.shape[:2]) - offset
        t[..., 6] = rng.normal(0.0, 0.5, t.shape[:2])  # hebbiano di base: debole e diverso da individuo a individuo
        t[..., 7] = rng.normal(0.0, 1.0, t.shape[:2])  # omeostasi: attività desiderata a caso,
        t[..., 8] = rng.normal(0.0, 0.5, t.shape[:2])  # forza debole e casuale
        t[..., 9] = rng.normal(0.0, 0.5, t.shape[:2])  # durata della traccia provvisoria
        t[..., 10] = rng.normal(0.0, 0.5, t.shape[:2])  # omeostasi della soglia: debole e casuale
        t[..., len(PLASTIC) + 3] = 0.0  # traccia di eleggibilità: uguale per tutti alla partenza, poi muta

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

    # Ciò che cambia durante una vita: lo stato della rete e ciò che si è imparato.
    LIVE = ("w_eff", "early", "elig", "critic", "bias_now", "avg", "clock", "state", "charge", "spikes")

    def replace(self, rows: np.ndarray, genome: Genome) -> None:
        """Gli individui `rows` diventano `genome` (una riga per ciascuno, con
        gli stessi posti): nascono, con lo stato e i pesi della nascita. Tutti
        gli altri continuano la loro vita da dove erano, con ciò che hanno
        imparato (serve all'arena continua, dove si nasce uno alla volta)."""
        rows = np.asarray(rows)
        kept = {k: getattr(self, k).copy() for k in self.LIVE if hasattr(self, k)}
        g = self._genome
        for f in Genome.FIELDS:
            getattr(g, f)[rows] = getattr(genome, f)
        self.compile()
        others = np.ones(self.n, dtype=bool)
        others[rows] = False
        for k, v in kept.items():
            now = getattr(self, k)
            now[others] = v[others]
            if k in ("state", "charge", "spikes"):
                now[rows] = 0.0
        self.learned = True
        self._subset = None

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
        self.eta, self.rule = cfg.plasticity_initial_rate * scale_gene(p[..., 0]), p[..., 1:5]
        self.hebb = cfg.hebb_initial_rate * scale_gene(p[..., 6])  # (N, T), del neurone che riceve
        d = self.traits[:, :, self.GAIN + 1 :]
        self.boost = cfg.critical_initial_boost * scale_gene(d[..., 0])  # (N, T): B
        self.closure = cfg.critical_initial_tau * scale_gene(d[..., 1])  # (N, T): τ_c, s
        self.lived = np.zeros(self.n)  # secondi già vissuti in vite precedenti (li dà l'evoluzione)
        self.value = cfg.value_initial * scale_gene(g.value)  # (N, 2): forza dei nuclei di gusto e dolore
        self.target_act = sigmoid(p[..., 7])  # (N, T): a*
        self.homeo = cfg.homeo_initial_rate * scale_gene(p[..., 8])  # (N, T): h, per secondo
        self.intrinsic = cfg.homeo_initial_rate * scale_gene(p[..., 10])  # (N, T), per secondo
        self.early_keep = np.exp(-cfg.dt / (cfg.early_initial_tau * scale_gene(p[..., 9])))  # (N, T), per passo
        self.elig_keep = np.exp(-cfg.dt / (cfg.eligibility_initial_tau * scale_gene(d[..., 2])))  # (N, T), per passo
        self.kmod = np.exp(-self.dist[:, :, R : R + H] / cfg.modulation_range) * mod[:, None, :]
        # Chi impara: con l'hebbiano di base e l'omeostasi, tutti (salvo plasticità spenta).
        self.learns = np.full(self.n, cfg.plasticity) & (
            (self.hebb > 0).any(axis=1) | (self.homeo > 0).any(axis=1) | mod.any(axis=1) | (self.value > 0).any(axis=1))
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
        # Il tipo di ogni sinapsi, fissato alla nascita: +1 eccitatoria, -1
        # inibitoria (legge di Dale per i neuroni, segno del gene per i sensi e
        # per le reti continue). La plasticità cambia la forza, mai il tipo.
        self.sign = np.where(self.w_eff < 0.0, -1.0, 1.0)
        self.learned = False
        self._reset_activity()

    def frequencies(self) -> np.ndarray:
        """(N, H + n_out): la frequenza attuale di ogni neurone interno e motore."""
        return frequencies(np, self.state, self.CENTRED)

    def _compile_extra(self) -> None:
        """Per i tipi di rete con geni compilati propri (prima dei pesi)."""

    def _born_weights(self) -> np.ndarray:
        """I pesi effettivi alla nascita, dal genoma: nessuna sinapsi oltre il
        suo massimo fisico (`synapse_max`)."""
        top = self.cfg.synapse_max
        return np.clip(self.weights, -top, top) * self.links

    # Segno obbligato dei pesi (legge di Dale): None = libero.
    dale = None

    def reset_state(self) -> None:
        """Vita nuova: stato a zero e pesi di nuovo quelli del genoma (ciò che
        si è imparato non si eredita e non passa da un'arena all'altra)."""
        self.state = np.zeros((self.n, self.n_hidden + self.n_out))
        self._forget()
        self._reset_activity()

    def _reset_activity(self) -> None:
        """Ogni vita comincia dallo stato alla nascita: ā e corrente di fondo
        come li ha lasciati il grembo (brains/freeform/womb.py); per chi non ci è passato
        (genoma diretto) ā = a* e la corrente del genoma. Senza tracce
        provvisorie."""
        g = self._genome
        # Età all'inizio di questa vita (s, dal grembo): per il periodo critico.
        born = np.isfinite(g.born_avg[:, -1]) if g.born_avg.shape[1] else np.zeros(self.n, dtype=bool)
        self.clock = np.where(born, self.cfg.prenatal_time, 0.0) + self.lived
        self.avg = np.where(np.isfinite(g.born_avg), g.born_avg, self.target_act)
        self.early = np.zeros_like(self.w_eff)
        self.elig = np.zeros_like(self.w_eff)
        self.critic = np.zeros((self.n, 2 * self.n_in + 1))  # sorpresa: il critico riparte a ogni vita
        self.bias_now = np.where(np.isfinite(g.born_bias), g.born_bias, self.bias)
        self._subset = None

    def _forget(self) -> None:
        if self.learned:
            self.w_eff = self._born_weights()
            self._subset = None
            self.learned = False

    # Geni compilati di cui `step_some` usa una copia ridotta.
    SUBSET = ("w_eff", "bias_now", "target_on")

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

    def step(self, obs: np.ndarray, signal: np.ndarray | None = None) -> np.ndarray:
        """obs (N, R) -> uscite (N, n_out) in [-1, 1]. `signal` (N, 2): gusto e
        dolore, per il sistema di valore (None = zero)."""
        before = self.state
        self.state, out = self._advance(obs, self.state, self)
        self._learn(np.arange(self.n), obs, before, self.state, signal=signal)
        self.clock += self.cfg.dt  # l'età avanza
        return out

    def step_some(self, obs: np.ndarray, rows: np.ndarray, signal: np.ndarray | None = None) -> np.ndarray:
        """Come `step`, ma solo per gli individui `rows` (obs e `signal` hanno
        una riga per ciascuno): gli altri restano fermi. Ogni rete calcola per
        conto suo, quindi il risultato per `rows` è identico a quello di `step`."""
        sub = self._rows(rows)
        before = self.state[rows]
        state, out = self._advance(obs, before, sub)
        self.state[rows] = state
        self._learn(rows, obs, before, state, sub, signal)
        self.clock[rows] += self.cfg.dt  # l'età avanza
        return out

    def _pre(self, obs, before, after) -> np.ndarray:
        """Attività di ogni sorgente che ha appena agito sui destinatari."""
        return np.concatenate([obs, before], axis=1)

    def _learn(self, rows, obs, before, after, sub=None, signal=None) -> None:
        """Plasticità durante la vita (docs/REGOLE.md, "Imparare durante la
        vita"). Ogni sinapsi ha un tipo fisso (eccitatoria o inibitoria) e una
        forza |w| in [0, synapse_max], somma di una parte stabile e di una
        traccia provvisoria. Le attività contano come frequenze (|attività|).
        A ogni passo, per ogni neurone che riceve:
        - scaling omeostatico (Turrigiano): le sinapsi eccitatorie per
          exp(h · dt · (a* − ā)), le inibitorie per l'inverso; ā è la media
          recente dell'attività (tempo caratteristico `homeo_tau`);
        - omeostasi dell'eccitabilità (Desai 1999): la corrente di fondo si
          sposta di h · dt · (a* − ā) nel verso che avvicina l'attività ad a*;
        - hebbiano di base, sempre attivo, con η_h del neurone: sulle
          eccitatorie la regola BCM, Δ|w| = η_h · pre · post · (post − θ) con
          la soglia mobile θ = ā² / a*; sulle inibitorie la regola di Vogels,
          Δ|w| = η_h · pre · (post − a*). Scrive la traccia provvisoria
          (early-LTP), che svanisce con il tempo τ del neurone (gene
          "persist");
        - consolidamento (tagging and capture, Frey e Morris 1997): dove c'è
          un modulatore la traccia diventa stabile, una quota
          1 − exp(−capture_rate · |M| · dt) a passo;
        - plasticità modulata (docs/SPEC_FASE_E.md): Δ|w| = segno · η · M ·
          (A·pre·post + B·pre + C·post + D), sulla parte stabile;
        - saturazione graduale (van Rossum 2000): ogni rafforzamento è
          moltiplicato per 1 − |w| / synapse_max;
        - periodo critico: l'hebbiano vale η_h · (1 + (B − 1) · e^(−età/τ_c)),
          con l'età in secondi dal grembo (`clock`);
        - valore innato: a M si aggiunge forza_gusto · gusto − forza_dolore ·
          dolore, uguale per tutti i neuroni (`signal`: (len(rows), 2), gusto
          e dolore di ogni individuo; None = zero)."""
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
        p = {k: getattr(self, k)[who] for k in ("w_eff", "early", "elig", "critic", "bias_now", "avg", "sign", "links",
                                                "target_act", "homeo", "intrinsic", "hebb", "eta", "rule", "early_keep",
                                                "elig_keep", "boost",
                                                "closure", "clock", "value")}
        p["signal"] = np.zeros((len(who), 2)) if signal is None else signal[local]
        consts = {k: getattr(self.cfg, k) for k in PLASTICITY_CONSTANTS}
        w, early, bias, avg, elig, critic = plasticity(consts, np, p, pre, post, m, self.n_in, self.CENTRED)
        self.w_eff[who], self.early[who], self.bias_now[who], self.avg[who] = w, early, bias, avg
        self.elig[who], self.critic[who] = elig, critic
        if sub is not None:
            sub["w_eff"][local], sub["bias_now"][local] = w, bias
        elif self._subset is not None:
            self._subset = None
        self.learned = True

    def _advance(self, obs, state, g):
        """Un passo a partire da `state`, con i geni compilati `g` (self o una
        copia ridotta). Restituisce (nuovo stato, uscite)."""
        get = g.get if isinstance(g, dict) else lambda k: getattr(g, k)
        x = np.concatenate([obs, state], axis=1)
        z = (get("w_eff") @ x[:, :, None])[:, :, 0] + get("bias_now")
        state = np.tanh(z) * get("target_on")
        return state, state[:, self.n_hidden :]
