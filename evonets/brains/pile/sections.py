"""Il cervello a sezioni (docs/SPEC_CERVELLO_SEZIONI.md) e i due agenti di
confronto per l'arena delle pile (docs/SPEC_ARENA_PILE.md).

Si fissa la struttura, non il contenuto: le sezioni, chi parla con chi, a che
velocità cambia ciascuna, con che famiglia di regole. Nessun neurone nasce
con un significato: i pesi che imparano partono da zero o a caso, e ciò che
rappresentano emerge (si misura con `selectivity` e le lesioni).

- O valore degli oggetti (come la corteccia orbitofrontale e l'amigdala):
  per ogni caratteristica percepibile (`sim.pile.FEATURES`: colore,
  grandezza, se si muove), più uno "sfondo", quanto piacere e quanto dolore
  porta di solito, separati. Impara dagli esiti, non da ogni passo: un
  episodio comincia quando il corpo sente piacere o dolore e finisce quando
  smette; alla fine, Rescorla-Wagner con le caratteristiche in competizione
  sulla traccia di ciò che si aveva in mente prima dell'esito (fovea e D,
  `trace_o` s): il merito va alla caratteristica che lo prevede, le altre
  restano a zero ("blocking"). Il valore non dipende da quanto bene ci si
  comporta (conta solo com'è andata quando è successo qualcosa). Ogni
  caratteristica ha la sua incertezza (Pearce e Hall: quanto i suoi esiti
  sorprendono di recente), che accelera l'apprendimento e attira lo
  sguardo. Uscite: per ogni oggetto il valore (piacere × fame di adesso −
  dolore: da sazi il cibo vale meno senza imparare niente), l'importanza
  (piacere + dolore: conta, buono o cattivo che sia) e l'incertezza; e due
  neuroni, "quanto vale ciò che guardo" e "quanto vale ciò che ho preso"
  (D), letti dal critico e da E. I valori di partenza sono geni.
- K contesti (lo stato nascosto del mondo): ciò che non si vede e si deduce
  dagli esiti (per esempio quale colore è buono adesso). Ogni contesto ha il
  suo cassetto di O (piacere e dolore per caratteristica): la memoria a lungo
  termine è legata al contesto, come nell'indicizzazione dell'ippocampo e
  negli stati della corteccia orbitofrontale. Dopo ogni episodio ogni
  contesto confronta la sua previsione con l'esito, e accumula una "prova"
  (l'errore recente, `k_rate`); vince il contesto che prevede meglio, con un
  vantaggio per quello in corso (`k_stick`); impara solo quello attivo. Se
  nessuno spiega gli esiti (errore oltre `k_new` volte il solito) si recluta
  un contesto nuovo, che parte dai valori innati (una causa nuova). Prima si
  decide il contesto, poi impara solo il suo cassetto: così, quando
  il mondo torna com'era, il vecchio cassetto è intatto e ci si torna in
  pochi episodi invece di reimparare. Prima versione: un neurone per
  contesto (`Sizes.k`); il passo successivo è un contesto sparso, un pattern
  su più neuroni, in cui contesti simili condividano ciò che sanno.
- Attenzione (S): una mappa di priorità fatta di posizioni, come il
  collicolo superiore: `MAP` neuroni-direzione che coprono il campo visivo,
  ognuno con l'oggetto più vicino nella sua direzione (un oggetto nuovo
  accende la sua posizione: non nasce nessun neurone). S non impara: ogni
  neurone somma la salienza innata (vicino, appena comparso, un po' di
  inerzia, inibizione del ritorno) e ciò che O dice dell'oggetto: il valore
  (gene `sal_value`: attira il buono), l'importanza (`sal_import`: attira ciò
  che conta, anche il pericolo) e l'incertezza (`sal_unc`). Vince una
  posizione (con un po' di caso); l'oggetto attento è il suo.
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
  Kenyon del corpo a fungo: molti più neuroni che ingressi, 6 connessioni
  ciascuno). Un'inibizione che cresce con l'attività (come il neurone APL)
  lascia accesi circa `c_sparse` dei neuroni, i più eccitati: quelli che
  ricevono più caratteristiche insieme. Così il codice di un oggetto è la
  sua combinazione ("rossa e grande"), non l'unione delle sue
  caratteristiche (con un solo ingresso per accendersi, "rossa grande" e
  "gialla grande" condividevano più neuroni di "rossa grande" e "rossa
  piccola"). Se al confine ci sono troppi pari merito, l'inibizione sale e
  li esclude: lo stesso oggetto dà sempre lo stesso codice. Impara solo
  l'uscita verso E, in fretta, e dimentica in minuti. Legge soprattutto
  che cos'è l'oggetto (il colore nella fovea e in D, non dove sta), più un
  po' di contesto (B, con peso `context`): così lo stesso oggetto accende
  gli stessi neuroni a ogni incontro, e ciò che si impara una volta vale la
  volta dopo (con tutto lo stato di B, o anche con la vicinanza e il lato
  dell'oggetto, ogni incontro aveva un codice diverso).
- L memoria lunga: la stessa uscita di C, cento volte più lenta, che si
  avvicina a C solo a riposo: ciò che si inverte di continuo si annulla.
- E selezione (legge B, D e i due neuroni di O): programmi in competizione (softmax con rumore = esplorazione);
  attore: Δw = η · δ · traccia di (azione − attesa) × ingresso.
- V critico: TD, tre critici con orizzonti diversi (breve, medio, lungo: in
  natura zone e neuroni a dopamina diversi scontano il futuro in modo
  diverso), δ = la media delle loro sorprese. Leggono ciò che prevede il
  valore in modo stabile: che cosa si ha davanti (fovea), che cosa si è
  appena preso (D), la situazione riconosciuta (C, che impara in fretta: il
  valore di un oggetto si impara in un colpo), l'energia, i due neuroni di
  O; B con poco peso.
- Quanto pesa il critico è un gene (`critic_mix`): la sorpresa che insegna è
  una via di mezzo tra il valore grezzo e quella dei critici.
- Il valore è legato al corpo: il piacere di una pila cresce con la fame
  (`need`), l'energia spesa (muoversi, mordere) dà un dispiacere (`effort`).
- U incertezza: le sorprese vere (sopra la soglia) di adesso rispetto al
  solito; guida l'arbitraggio (sotto) e accelera C.
- E sceglie anche se mordere (sì o no, a ogni decisione): si mangia solo
  mordendo, e mordere costa un po'. C ed L votano anche sul morso.
- F motore: un circuito innato di neuroni (`sim.pile.motor_net`) dai
  programmi e dall'oggetto attento a 4 neuroni motori (gira a sinistra, a
  destra, avanti, indietro); non impara nella vita, i suoi pesi sono geni.

Due regole sull'apprendimento con δ, solo nel cervello a sezioni:
- tracce strette: una sinapsi di E o del cancello di D si marca solo se il
  neurone da cui parte è acceso sopra `tag` (in natura si marca solo dove il
  neurone si attiva davvero);
- soglia e scala adattive: insegna solo la parte di δ che supera `floor`
  volte la sorpresa media degli ultimi minuti, misurata in unità di quella
  sorpresa media (la dopamina ha un livello di base e adatta la sua scala
  alla distribuzione dei premi: Tobler, Fiorillo, Schultz 2005), con il
  tetto della dopamina. Così E, C e il cancello di D imparano allo stesso
  modo con premi grandi o piccoli, e con qualunque passo di tempo. Il tetto
  schiaccia però le sorprese grandi e rare (il dolore di un morso cattivo):
  si sceglie un po' peggio (59% di pile buone invece di 66%) ma si mangia e
  si sopravvive meglio (docs/TRAGUARDI_PILE.md). Il critico impara invece da
  tutto δ: è il suo errore.
- inerzia legata agli oggetti: quando E sceglie, le sinapsi da C (il codice
  dell'oggetto guardato) al programma scelto si marcano per qualche secondo
  (la stessa traccia che serve a imparare), e finché la marcatura dura
  votano per quel programma (`stick`): tornando su un oggetto guardato poco
  fa, riaffiora ciò che si era deciso per lui; un oggetto nuovo non ha
  inerzia. È la memoria di lavoro nelle sinapsi (Mongillo, Barak, Tsodyks
  2008). Quando l'oggetto guardato cambia, E decide subito. L'inerzia sul
  programma resta solo quando non si guarda niente.

Arbitraggio (Daw, Niv, Dayan 2005; l'incertezza inattesa e la noradrenalina,
Yu e Dayan 2005): quando le sorprese sono tante (U alta) il mondo è cambiato,
le abitudini non sono affidabili e la decisione passa alla memoria veloce:
il voto di C conta di più, quelli di E ed L di meno, e la salienza imparata
meno di quella innata (ci si guarda attorno). Quando le sorprese calano,
tutto torna come prima. U accelera solo C.

Tutte le costanti del cervello sono geni (`Genes`, una per essere): tempi,
velocità, soglie, pesi della salienza innata, forza dell'arbitraggio. Nella
fase A valgono i valori di partenza per tutti; nella fase B li tara
l'evoluzione. Restano fissi solo la struttura (`Sizes`: quanti neuroni,
quante connessioni) e i limiti di sicurezza (`DELTA_MAX`, `WEIGHT_MAX`,
`INNATE_CAP`).

Il maestro (apprendimento per osservazione): in ogni recinto uno degli
esseri (il primo) può essere un maestro, guidato a mano come `Guided` (sa
quali pile sono buone). Gli altri gli "guardano dentro": quando un suo
episodio finisce, O di ogni allievo applica la stessa regola con la traccia
del maestro (ciò che lui aveva in mente) e il suo esito (velocità
`eta_o` × `observe`). Si impara che cosa fa bene o male a lui, non il
valore della sua situazione: è ciò che si trasferisce. Quanto c'è ancora da
imparare da lui: `obs_err`, la media di |errore| sui suoi episodi.

Lesioni (`lesions`): "B" (E, V, C, D leggono A invece di B), "D", "C", "L",
"U", "V" (niente critico: δ = ricompensa grezza), "S" (salienza solo innata),
"O" (niente valore degli oggetti: non impara e dice zero), "K" (un contesto solo),
"R" (niente arbitraggio: U accelera E, aggiunge caso e non sposta i voti,
com'era prima).
"""

import numpy as np

from ...arenas.pile.world import ENERGY, N_FEATURES, TASTE
from .agent import Learner
from .genes import APPROACH, DELTA_MAX, EXPLORE, FOVEA, IDENTITY, N_A, N_HEADS, WEIGHT_MAX, Interface, Sizes, value_of
from .value import o_active, o_read, o_step, o_trace


def c_code(z, p: dict, xp=np):
    """I neuroni di C accesi (vedi la docstring del modulo): l'inibizione lascia
    accesi circa `c_sparse` dei neuroni, i più eccitati; i pari merito al
    confine solo se non sono troppi. `z` (n, C): l'ingresso di ogni neurone."""
    n, C = z.shape
    k = xp.clip(xp.round(p["c_sparse"] * C), 1, C).astype(int)
    thr = xp.take_along_axis(-xp.sort(-z, axis=1), (k - 1)[:, None], axis=1)
    eps = 1e-6
    hi = z > thr + eps
    tie = xp.abs(z - thr) <= eps
    n_hi, n_tie = hi.sum(1, keepdims=True), tie.sum(1, keepdims=True)
    allow = (n_hi + n_tie <= 2 * k[:, None]) | (n_hi == 0)
    return ((hi | (tie & allow)) & (z >= p["c_threshold"][:, None])).astype(z.dtype)


def teach_of(delta, d_slow, p: dict, xp=np):
    """La sorpresa che insegna: la parte di δ sopra `floor` volte la sorpresa
    solita (`d_slow`), in unità di quella sorpresa solita (la scala si
    adatta), con il tetto della dopamina."""
    usual = xp.maximum(d_slow, 1e-4)
    teach = xp.sign(delta) * xp.maximum(xp.abs(delta) - p["floor"] * usual, 0.0) / usual
    return xp.clip(teach, -DELTA_MAX, DELTA_MAX)


def teacher_mask(N: int, group: int, xp=np):
    """(N,) il primo essere di ogni recinto (quello che può fare da maestro)."""
    return xp.arange(N) % group == 0


def teach_override(program, bite, has, ok, near, teacher, xp=np):
    """Il maestro decide come `Guided`: va verso la pila buona che guarda e la
    morde quando ce l'ha davanti; altrimenti esplora. -> (programma, morso, da lasciare)."""
    program = xp.where(teacher, xp.where(ok, APPROACH, EXPLORE), program)
    bite = xp.where(teacher, ok & (near > 0.9), bite)
    return program, bite, teacher & has & ~ok


class Sections(Learner):
    """Il cervello a sezioni (vedi la docstring del modulo)."""

    name = "sezioni"

    def __init__(self, n, dt, seed=0, genomes=None, sizes: Sizes | None = None, objects: int = 1):
        super().__init__(n, dt, seed, genomes, objects)
        self.s = s = sizes or Sizes()
        self.les = set(s.lesions)
        self.a = Interface(n)
        nb = N_A if "B" in self.les else s.b
        self.nb = nb
        # B: prototipi di partenza (dal genoma: a caso, senza significato, nella
        # fase A; l'evoluzione può cambiarli nella fase B) e previsione.
        self.proto = np.zeros((n, s.b, N_A))
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
        wiring = np.zeros((n, s.c, 2 * IDENTITY + nb))  # le connessioni di partenza: a caso, 4 per neurone
        for j in range(s.c):
            pick = np.argsort(self.rng.random((n, 2 * IDENTITY + nb)), axis=1)[:, : s.c_fan_in]
            np.put_along_axis(wiring[:, j], pick, 1.0, axis=1)
        self.c_in = np.zeros_like(wiring)
        self.u_c = np.zeros((n, N_HEADS, s.c))
        self.e_c = np.zeros((n, N_HEADS, s.c))
        self.u_l = np.zeros((n, N_HEADS, s.c))
        self.code = np.zeros((n, s.c))
        # U e soglia: medie della sorpresa.
        self.d_fast = np.zeros(n)
        self.d_slow = np.full(n, 1e-3)
        self.t_slow = np.full(n, 1e-4)
        self.unc = np.zeros(n)
        # O: piacere e dolore per caratteristica (più lo sfondo), incertezza,
        # traccia, l'episodio in corso; i due neuroni di uscita.
        F1 = N_FEATURES + 1
        NK = s.k
        self.o_pl, self.o_pa = np.zeros((n, NK, F1)), np.zeros((n, NK, F1))  # un cassetto per contesto
        self.o_assoc = np.full((n, NK, 2, F1), 0.5)  # sorpresa di adesso (l'incertezza) e quella solita
        # K: il contesto attivo, la prova di ognuno, quali sono in uso, l'errore solito.
        self.k_on = np.zeros(n, dtype=int)
        self.k_ev = np.zeros((n, NK))
        self.k_used = np.zeros((n, NK), dtype=bool)
        self.k_used[:, 0] = True
        self.k_usual = np.full(n, 0.5)
        self.k_since = np.zeros(n)
        self.k_stale = np.zeros((n, NK))
        self.o_prior = np.zeros((n, 2, F1))  # i valori innati delle cose (dal genoma): da qui parte un contesto nuovo
        self.o_e = np.zeros((n, N_FEATURES))
        self.ep_on = np.zeros(n, dtype=bool)
        self.ep_pl, self.ep_pa = np.zeros(n), np.zeros(n)
        self.ep_e = np.zeros((n, F1))
        self.o_err = np.zeros(n)
        self.o_out = np.zeros((n, 2))  # quanto vale ciò che guardo, ciò che ho preso
        # E legge B, D, O e la costante. I tre critici leggono la fovea, D, C,
        # l'energia, B (con poco peso), O e la costante.
        self.n_v = nf + nf + s.c + 1 + nb + 2 + 1
        self._init_actor_critic(nb + nf + 2 + 1, self.n_v, critics=3)
        # Il maestro (vedi la docstring del modulo).
        self.teacher = np.zeros(n, dtype=bool)
        self.group = 1  # esseri per recinto (il maestro di i è il primo del suo recinto)
        self.obs_err = np.zeros(n)
        # Nel genoma: i pesi di partenza (prototipi di B, E, V, salienza) e le
        # connessioni di C (un gene per connessione possibile: c'è se supera 0,5).
        self._attach((("proto", (s.b, N_A), self.rng.random((n, s.b, N_A)) * 0.3),
                      ("c_in", wiring.shape[1:], wiring),
                      ("e0", (N_HEADS, nb + nf + 3), np.zeros((n, N_HEADS, nb + nf + 3))),
                      ("v0", (3, self.n_v), np.zeros((n, 3, self.n_v))),
                      ("o0", (2, F1), np.zeros((n, 2, F1)))))
        self._born(slice(None))

    def _born(self, who) -> None:
        """Pesi e connessioni di partenza, dal genoma."""
        self.proto[who] = self.start("proto", who)
        self.c_in[who] = (self.start("c_in", who) > 0.5).astype(float)
        self.w_e[who], self.w_v[who] = self.start("e0", who), self.start("v0", who)
        o0 = self.start("o0", who)
        self.o_pl[who], self.o_pa[who] = o0[:, None, 0], o0[:, None, 1]  # i valori innati delle cose (geni)
        self.o_prior[who] = o0

    # --- B ---------------------------------------------------------------------------

    def _b(self, a: np.ndarray, delta: np.ndarray) -> np.ndarray:
        s, p, dt = self.s, self.p, self.dt
        if "B" in self.les:
            self.state_b = a
            return a
        d = ((self.proto - a[:, None, :]) ** 2).sum(-1)  # (n, b): distanza da ogni prototipo
        d = d - p["conscience"][:, None] * (1.0 / s.b - self.win_rate)  # chi vince poco è aiutato
        order = np.argsort(d, axis=1)
        kth = np.take_along_axis(d, order[:, s.b_winners: s.b_winners + 1], 1)
        y = np.clip(1.0 - d / np.maximum(kth, 1e-9), 0.0, 1.0)  # i più vicini, gli altri a zero
        y = y / np.maximum(y.max(1, keepdims=True), 1e-9)
        # Imparare: i vincitori si avvicinano all'ingresso (e, con la previsione, a ciò che segue).
        err = a - np.einsum("nab,nb->na", self.pred, self.prev_b)  # quanto il passo prima aveva previsto male
        self.pred += (p["eta_predict"] * dt)[:, None, None] * err[:, :, None] * self.prev_b[:, None, :]
        rate = (p["eta_b"] * dt * (1.0 + p["surprise_boost"] * np.minimum(np.abs(delta), 1.0)))[:, None, None]
        self.proto += rate * y[:, :, None] * ((a[:, None, :] - self.proto) + p["predict"][:, None, None] * err[:, None, :])
        self.win_rate += (dt / p["win_window"])[:, None] * ((y > 0) / s.b_winners - self.win_rate)
        self.prev_b = self.state_b.copy()
        self.state_b += (dt / p["tau_b"])[:, None] * (y - self.state_b)
        return self.state_b

    # --- D ---------------------------------------------------------------------------

    def _d(self, fov: np.ndarray, teach: np.ndarray) -> np.ndarray:
        """Si scrive quando nella fovea entra un oggetto nuovo (innato) o quando
        il cancello lo ha imparato; altrimenti si tiene, anche a fovea vuota."""
        p, dt = self.p, self.dt
        if "D" in self.les:
            return np.zeros_like(fov)
        self.w_gate = np.clip(self.w_gate + (p["eta_gate"] * teach)[:, None] * self.e_gate, -2 * WEIGHT_MAX,
                              2 * WEIGHT_MAX)
        z = p["gate_event"] * fov[:, -1] + (self.w_gate * fov).sum(1) + p["gate_bias"]
        g = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        tagged = np.where(fov > p["tag"][:, None], fov, 0.0)
        self.e_gate = np.exp(-dt / p["trace_actor"])[:, None] * self.e_gate + (g * (1.0 - g))[:, None] * tagged
        keep = np.exp(-dt / p["tau_d"])[:, None]
        self.mem = (1.0 - g)[:, None] * keep * self.mem + g[:, None] * fov
        return self.mem

    # --- C ---------------------------------------------------------------------------

    def _c(self, b: np.ndarray, m: np.ndarray, fov: np.ndarray) -> np.ndarray:
        s, p = self.s, self.p
        if "C" in self.les:
            return np.zeros((self.n, s.c))
        h = np.concatenate([fov[:, :IDENTITY], p["memory"][:, None] * m[:, :IDENTITY], p["context"][:, None] * b], 1)
        z = np.einsum("ncj,nj->nc", self.c_in, h)
        self.code = c_code(z, p)
        return self.code

    def _chose(self, deviation):
        self.e_c += deviation[:, :, None] * self.code[:, None, :]  # solo i neuroni accesi al momento della scelta

    # --- Un passo --------------------------------------------------------------------

    def step(self, raw, reward, objs):
        p, dt = self.p, self.dt
        decide = self._tick()
        arbitrate = "R" not in self.les and "U" not in self.les
        k = np.clip(self.unc / 3.0, 0.0, 1.0) if arbitrate else np.zeros(self.n)
        teaching = self.teacher.any()
        master = np.arange(self.n) // self.group * self.group
        on = ~self.teacher & self.teacher[master]
        # O: gli episodi finiti (propri e del maestro) insegnano, prima di guardare.
        for k_, v in o_step(self.__dict__, reward, raw[:, ENERGY], master, on, p, self.les).items():
            setattr(self, k_, v)
        extra = np.where(self.teacher[:, None], np.where(objs["good"], 10.0, -10.0), 0.0) if teaching else None
        o = None if "S" in self.les else (*o_active(self.__dict__), raw[:, ENERGY])
        self.att.update(objs, decide, p, extra=extra, learned=1.0 - p["att_shift"] * k, o=o)
        decide = decide | self.att.changed  # un oggetto diverso: si decide subito
        self.clock[self.att.changed] = 0.0
        fov = self.att.fovea(objs)
        a = self.a(raw, fov, np.exp(-dt / p["tau_a"]))
        r = value_of(reward, raw[:, ENERGY], p) + self.innate_reward(raw)
        teach_prev = self.teach
        b = self._b(a, self.delta)
        m = self._d(fov, teach_prev)
        c = self._c(b, m, fov)
        if "O" not in self.les:
            self.o_e = o_trace(self.o_e, fov, m, p, dt)
            ids = np.stack([fov[:, :IDENTITY], m[:, :IDENTITY]], 1)
            self.o_out = o_read(*o_active(self.__dict__), ids, raw[:, ENERGY], p)[0]
        vo = self.o_out
        one = np.ones((self.n, 1))
        f_v = np.concatenate([fov, m, c, raw[:, ENERGY:ENERGY + 1], p["critic_b"][:, None] * b, vo, one], 1)
        nf = fov.shape[1]
        eta_v = np.repeat(p["eta_v"][:, None], f_v.shape[1], 1)
        eta_v[:, 2 * nf: 2 * nf + c.shape[1]] = p["eta_vc"][:, None]  # la parte su C: in fretta
        horizons = np.stack([p["h_short"], p["h_mid"], p["h_long"]], 1)
        delta = self._td(f_v, r, horizons, eta_v, use_critic="V" not in self.les)
        # Soglia adattiva: insegna solo la parte di δ sopra la sorpresa solita.
        self.d_slow += (dt / p["u_slow"]) * (np.abs(delta) - self.d_slow)
        teach = teach_of(delta, self.d_slow, p)
        self.teach = teach
        # U: le sorprese vere (sopra la soglia) di adesso rispetto al solito (con
        # la media di tutto δ i pochi dolori di un'inversione si perdevano nel
        # rumore del critico, e U non si accorgeva di niente).
        self.d_fast += (dt / p["u_fast"]) * (np.abs(teach) - self.d_fast)
        self.t_slow += (dt / p["u_slow"]) * (np.abs(teach) - self.t_slow)
        if "U" not in self.les:
            self.unc = np.clip(self.d_fast / np.maximum(self.t_slow, 1e-4) - 1.0, 0.0, 3.0)
        boost = 1.0 + p["surprise_rate"] * self.unc
        # C impara in fretta e dimentica; L si avvicina a C solo a riposo.
        step = p["eta_c"] * boost * teach / (1.0 + self.code.sum(1))  # diviso per i neuroni accesi
        cap = p["c_max"][:, None, None]
        self.u_c = np.clip(self.u_c + step[:, None, None] * self.e_c - self.u_c * (dt / p["forget"])[:, None, None],
                           -cap, cap)
        self.e_c *= np.exp(-dt / p["trace_actor"])[:, None, None]
        if "L" not in self.les:
            # Riposo: carichi e fermi a ricaricarsi (il gusto della pila che si consuma).
            rest = (raw[:, ENERGY] > p["rest_energy"]) & (raw[:, TASTE] > 0.1)
            self.u_l += (p["eta_l"] * dt * rest)[:, None, None] * (self.u_c - self.u_l)
        # Con l'arbitraggio U non accelera più E e non aggiunge caso (era rumore):
        # sposta il peso della decisione verso C. Senza ("R"), com'era prima.
        slow_boost = np.ones(self.n) if arbitrate else boost
        f_e = np.concatenate([b, m, vo, one], 1)
        tag = p["tag"][:, None]
        f_tag = np.concatenate([np.where(b > tag, b, 0.0), np.where(m > tag, m, 0.0), vo, one], 1)
        habit = (self.w_e * f_e[:, None, :]).sum(-1)
        if "L" not in self.les:
            habit += np.einsum("npc,nc->np", self.u_l, c)
        # L'inerzia legata all'oggetto: le sinapsi C -> E marcate dalle ultime scelte.
        bound = p["stick"][:, None] * np.einsum("npc,nc->np", self.e_c, c) / np.maximum(c.sum(1), 1.0)[:, None]
        self.votes = {"E+L": habit, "C": np.einsum("npc,nc->np", self.u_c, c), "k": k, "inerzia": bound}
        logits = ((1.0 - p["arb_slow"] * k)[:, None] * habit + (1.0 + p["arb_fast"] * k)[:, None] * self.votes["C"]
                  + bound)
        temp = p["temperature"] * (1.0 + (0.0 if arbitrate else p["surprise_explore"]) * self.unc)
        seeing = self.att.target >= 0
        program = self._act(logits, f_e, teach, p["eta_e"] * slow_boost, temp, decide, f_tag, stick_on=~seeing)
        if teaching:
            # Il maestro decide da sé (come il guidato).
            has, _, near, _ = self.att.seen(objs)
            ok = has & objs["good"][np.arange(self.n), np.maximum(self.att.target, 0)]
            program, self.bite, leave = teach_override(program, self.bite, has, ok, near, self.teacher)
            self.p_bite = np.where(self.teacher, self.bite.astype(float), self.p_bite)
            self.att.leave(leave, p)
            self.program = program
        return program

    def set_teacher(self, mask: np.ndarray, group: int) -> None:
        """Chi fa da maestro (`mask` (n,)); `group`: esseri per recinto. Chi
        smette di esserlo rinasce: al suo posto, un giovane."""
        gone = self.teacher & ~mask
        self.teacher, self.group = mask.copy(), group
        if gone.any():
            self.reborn(gone)

    def reborn(self, who, genomes=None):
        super().reborn(who, genomes)
        self.a.reset(who)
        self._reset_actor_critic(who)
        self.pred[who] = 0.0
        self.win_rate[who] = 1.0 / self.s.b
        for x in (self.state_b, self.prev_b, self.mem, self.w_gate, self.e_gate, self.u_c, self.e_c, self.u_l,
                  self.code, self.d_fast, self.unc, self.o_assoc, self.o_e, self.ep_pl, self.ep_pa, self.ep_e,
                  self.o_err, self.o_out, self.obs_err, self.k_ev, self.k_since, self.k_stale):
            x[who] = 0.0
        self.o_assoc[who] = 0.5
        self.k_on[who], self.k_usual[who] = 0, 0.5
        self.k_used[who] = False
        self.k_used[who, 0] = True
        self.ep_on[who] = False
        self.d_slow[who] = 1e-3
        self.t_slow[who] = 1e-4
        self._born(who)

    def winner(self) -> np.ndarray:
        """(N,) il neurone di B più acceso (per le misure di emersione)."""
        return self.state_b.argmax(1)
