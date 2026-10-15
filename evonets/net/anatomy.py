"""Anatomia di una singola rete: chi esiste, chi parla con chi e quali parti
contano davvero per i motori. Serve a guardare e descrivere una rete, non a
calcolarla."""

from collections import deque

import numpy as np

from . import receptors
from .brain import META, Genome, live_links
from .controllers import NeuralController
from .geometry import motor_anchors, receptor_anchors

KIND_NAMES = {"rate": "continua", "spike-ei": "a impulsi E/I"}
MOD_VISIBLE = 0.05  # sotto questa quota di M un neurone si considera non modulato


def sizes(genome: Genome) -> tuple[np.ndarray, np.ndarray]:
    """Per ogni individuo: neuroni interni esistenti e collegamenti effettivi."""
    links = live_links(genome.conn, genome.active, genome.r_active)
    return genome.active.sum(axis=1), links.sum(axis=(1, 2))


def node_labels(genome: Genome, out_labels) -> list[str]:
    """Nome di ogni neurone dell'individuo (genoma di una riga), nell'ordine
    delle sorgenti: recettori, interni, motori."""
    rec = genome.receptors
    return ([receptors.label(rec, 0, k) for k in range(genome.n_rec)]
            + [f"n{h}" for h in range(genome.n_hidden)] + list(out_labels))


class Anatomy:
    """La rete di un individuo (un genoma di una riga), pronta da descrivere.

    I neuroni sono numerati come le sorgenti del genoma: prima i recettori,
    poi gli interni, poi i motori. `w` sono i pesi effettivi (per le reti a impulsi il
    segno è già quello dato dal tipo del neurone che trasmette).
    """

    def __init__(self, cfg, genome: Genome):
        brains = NeuralController(cfg, 1).brains
        brains.set_genome(genome)
        self.brains = brains
        self.kind = brains.KIND
        self.n_in, self.n_hidden, self.n_out = brains.n_in, brains.n_hidden, brains.n_out
        self.n_nodes = self.n_in + self.n_hidden + self.n_out
        self.conn = genome.conn[0]
        self.links = brains.links[0]  # (T, S)
        self.w = brains.w_eff[0]
        self.traits = genome.traits[0]
        self.genome = genome
        self.exists = np.ones(self.n_nodes, dtype=bool)
        self.exists[: self.n_in] = genome.r_active[0]
        self.exists[self.n_in : self.n_in + self.n_hidden] = genome.active[0]
        self.inhibitory = brains.inhibitory[0] if hasattr(brains, "inhibitory") else None
        self.modulatory = brains.modulatory[0]  # (H,)
        self.kmod = brains.kmod[0]  # (T, H): quanto M di ogni modulatore arriva a ogni neurone
        self.labels = node_labels(genome, brains.OUT_LABELS)
        # Fenomeno dominante di ogni recettore (per il colore nei disegni).
        dom = [receptors.dominant(genome.receptors, 0, k) for k in range(self.n_in)]
        self.phenomenon = [p for p, _ in dom]
        self.specialized = [share >= 0.8 for _, share in dom]
        self.xy = np.concatenate([receptor_anchors(genome.r_y[0]), genome.pos[0],
                                  motor_anchors(self.n_out)])  # posizione di ogni neurone nel cervello
        self.dist = brains.dist[0]  # (T, S)
        self.wire_length = float(brains.wire_length[0])
        self.wire_cost = cfg.wire_cost * self.wire_length
        self.upkeep = float(brains.upkeep[0])
        self.useful = self._reaches_motors()
        # Un modulatore non ha uscite: è utile se modula qualcuno che conta.
        for h in np.flatnonzero(self.modulatory):
            reached = self.n_in + np.flatnonzero(self.kmod[:, h] > MOD_VISIBLE)
            if self.useful[reached].any():
                self.useful[self.n_in + h] = True
        self.depth = self._depth_from_senses()

    # --- struttura ---------------------------------------------------------

    def is_hidden(self, v: int) -> bool:
        return self.n_in <= v < self.n_in + self.n_hidden

    def is_modulator(self, v: int) -> bool:
        return self.is_hidden(v) and bool(self.modulatory[v - self.n_in])

    def modulated_by(self, v: int) -> list[tuple[int, float]]:
        """Neuroni (numero, quota di M) che il modulatore v raggiunge."""
        h = v - self.n_in
        out = [(self.n_in + t, float(k)) for t, k in enumerate(self.kmod[:, h])
               if k > MOD_VISIBLE and self.exists[self.n_in + t]]
        return sorted(out, key=lambda x: -x[1])

    def is_motor(self, v: int) -> bool:
        return v >= self.n_in + self.n_hidden

    def inputs_of(self, v: int) -> np.ndarray:
        """Sorgenti dei collegamenti effettivi verso il neurone v."""
        if v < self.n_in:
            return np.zeros(0, dtype=int)
        return np.flatnonzero(self.links[v - self.n_in])

    def outputs_of(self, v: int) -> np.ndarray:
        """Destinatari (come numeri di neurone) dei collegamenti in uscita da v."""
        return self.n_in + np.flatnonzero(self.links[:, v])

    def weight(self, src: int, dst: int) -> float:
        return float(self.w[dst - self.n_in, src])

    def length(self, src: int, dst: int) -> float:
        """Lunghezza del collegamento da src a dst, in lati del cervello."""
        return float(self.dist[dst - self.n_in, src])

    def edges(self) -> np.ndarray:
        """(E, 2): coppie (sorgente, destinatario) dei collegamenti effettivi."""
        tgt, src = np.nonzero(self.links)
        return np.stack([src, tgt + self.n_in], axis=1)

    def _reaches_motors(self) -> np.ndarray:
        """Vero per i neuroni da cui parte almeno un cammino verso un motore:
        gli altri sono rami morti, non cambiano il comportamento."""
        seen = np.zeros(self.n_nodes, dtype=bool)
        queue = deque(range(self.n_in + self.n_hidden, self.n_nodes))
        seen[list(queue)] = True
        while queue:
            for s in self.inputs_of(queue.popleft()):
                if not seen[s]:
                    seen[s] = True
                    queue.append(s)
        return seen

    def _depth_from_senses(self) -> np.ndarray:
        """Passi minimi dai sensi a ogni neurone; -1 se nessun senso lo raggiunge
        (si muove solo per la propria soglia o per gli anelli)."""
        depth = np.full(self.n_nodes, -1)
        senses = np.flatnonzero(self.exists[: self.n_in])
        depth[senses] = 0
        queue = deque(senses)
        while queue:
            s = queue.popleft()
            for v in self.outputs_of(s):
                if depth[v] < 0:
                    depth[v] = depth[s] + 1
                    queue.append(v)
        return depth

    def summary(self) -> dict:
        hidden = slice(self.n_in, self.n_in + self.n_hidden)
        e = self.edges()
        useful_edges = self.useful[e[:, 0]] & self.useful[e[:, 1]] if len(e) else np.zeros(0, dtype=bool)
        # Anelli: collegamenti che tornano indietro (o restano allo stesso passo)
        # rispetto alla distanza dai sensi, autoanelli compresi.
        back = [(s, d) for s, d in e if s >= self.n_in and (s == d or self.depth[s] >= self.depth[d])]
        out = {
            "receptors": int(self.exists[: self.n_in].sum()),
            "modulators": int((self.modulatory & self.exists[hidden]).sum()),
            "neurons": int(self.exists[hidden].sum()),
            "useful_neurons": int((self.exists & self.useful)[hidden].sum()),
            "links": len(e),
            "useful_links": int(useful_edges.sum()),
            "senses_used": int(self.useful[: self.n_in].sum()),
            "loops": len(back),
            "wire_length": self.wire_length,
            "wire_cost": self.wire_cost,
            "upkeep": self.upkeep,
        }
        if self.inhibitory is not None:
            out["inhibitory"] = int((self.inhibitory & self.exists[hidden]).sum())
        return out

    # --- descrizione -------------------------------------------------------

    def describe(self, v: int) -> list[str]:
        """Righe di testo sul neurone v: geni propri e collegamenti più forti."""
        name = self.labels[v]
        if v < self.n_in:
            role = "recettore"
        elif self.is_motor(v):
            role = "motore"
        else:
            kind = ""
            if self.inhibitory is not None:
                kind = " inibitorio" if self.inhibitory[v - self.n_in] else " eccitatorio"
            role = f"neurone interno{kind}"
            if self.is_modulator(v):
                role = "MODULATORE" + (f" ({'M negativo' if self.inhibitory[v - self.n_in] else 'M positivo'})"
                                       if self.inhibitory is not None else "")
        x, y = self.xy[v]
        lines = [f"{name}: {role}", f"posizione nel cervello ({x:.2f}, {y:.2f})"]
        if not self.exists[v]:
            lines.append("spento: i suoi geni restano nel genoma")
        elif not self.useful[v]:
            lines.append("ramo morto: non arriva ai motori")
        if v < self.n_in:
            lines += receptors.describe(self.genome.receptors, 0, v)
        else:
            lines += self.brains.trait_text(self.traits[v - self.n_in], self.is_hidden(v))
            d = int(self.depth[v])
            lines.append(f"distanza dai recettori: {d} pass{'o' if d == 1 else 'i'}" if d > 0
                         else "nessun recettore lo raggiunge")
        if self.is_modulator(v):
            reached = self.modulated_by(v)
            lines.append(f"non trasmette: rilascia M, che arriva a {len(reached)} neuroni")
            lines += [f"  {self.labels[t]:12s} M x{k:.2f}" for t, k in reached[:8]]
        ins, outs = self.inputs_of(v), self.outputs_of(v)
        lines.append(f"collegamenti: {len(ins)} entrano, {len(outs)} escono")
        for title, pairs in (("riceve da", [(s, self.weight(s, v), self.length(s, v)) for s in ins]),
                             ("manda a", [(d, self.weight(v, d), self.length(v, d)) for d in outs])):
            if pairs:
                pairs.sort(key=lambda p: -abs(p[1]))
                lines.append("")
                lines.append(f"{title}: peso, effetto, lunghezza")
                lines += [f"  {self.labels[n]:12s}{w:+6.2f} {'eccita' if w > 0 else 'inibisce':8s} {d:.2f}"
                          + ("" if self.useful[n] else " morto") for n, w, d in pairs]
        return lines

    def genome_text(self, title: str = "") -> str:
        """Il genoma completo in forma leggibile, blocco per blocco (un blocco
        per neurone interno o motore), compresi i geni latenti."""
        s = self.summary()
        head = [title] if title else []
        head += [
            f"rete {KIND_NAMES[self.kind]}: {s['receptors']} recettori ({s['senses_used']} usati), "
            f"{s['neurons']} neuroni interni ({s['useful_neurons']} utili), {s['links']} collegamenti "
            f"({s['useful_links']} utili), anelli {s['loops']}, cablaggio {s['wire_length']:.1f}; "
            f"corpo e cervello costano {s['upkeep']:.2f} energia/s",
            "pesi efficaci: >0 eccita, <0 inibisce"
            + (" (dai neuroni interni il segno lo dà il tipo)" if self.inhibitory is not None else ""),
            "latente = gene presente ma inattivo perché un capo del collegamento è spento",
            "",
        ]
        body = []
        rec = self.genome.receptors
        for k in range(self.n_in):
            state = "esiste" if self.exists[k] else "SPENTO"
            body.append(f"[{self.labels[k]}] {state} | nervo a ({self.xy[k, 0]:.3f}, {self.xy[k, 1]:.3f})")
            body += ["    " + line for line in receptors.describe(rec, 0, k)]
        meta = "; ".join(f"{name} {np.exp(v):.3g}" for name, v in zip(META, self.genome.meta[0]))
        body.append(f"[mutazioni] {meta}")
        for t in range(self.n_hidden + self.n_out):
            v = self.n_in + t
            state = "esiste" if self.exists[v] else "SPENTO"
            if self.exists[v] and not self.useful[v]:
                state += ", ramo morto"
            traits = "; ".join(self.brains.trait_text(self.traits[t], self.is_hidden(v)))
            x, y = self.xy[v]
            body.append(f"[{self.labels[v]}] {state} | posizione ({x:.3f}, {y:.3f}) | {traits}")
            for src in np.flatnonzero(self.conn[t]):
                live = self.links[t, src]
                w = self.w[t, src] if live else self.brains.weights[0, t, src]
                body.append(f"    <- {self.labels[src]:14s} {w:+.3f}  lunghezza {self.dist[t, src]:.3f}"
                            f"{'' if live else '  (latente)'}")
        return "\n".join(head + body) + "\n"
