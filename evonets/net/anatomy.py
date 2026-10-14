"""Anatomia di una singola rete: chi esiste, chi parla con chi e quali parti
contano davvero per i motori. Serve a guardare e descrivere una rete, non a
calcolarla."""

from collections import deque

import numpy as np

from .brain import Genome, live_links
from .controllers import NeuralController
from .sensors import BODY, CHANNELS

KIND_NAMES = {"rate": "continua", "spike-ei": "a impulsi E/I"}


def sizes(genome: Genome, n_in: int) -> tuple[np.ndarray, np.ndarray]:
    """Per ogni individuo: neuroni interni esistenti e collegamenti effettivi."""
    links = live_links(genome.conn, genome.active, n_in)
    return genome.active.sum(axis=1), links.sum(axis=(1, 2))


def node_labels(n_in: int, n_hidden: int, out_labels) -> list[str]:
    """Nome di ogni neurone, nell'ordine delle sorgenti: sensi, interni, motori."""
    n_vis = n_in - len(BODY)
    n_ch = len(CHANNELS)
    senses = [f"r{k // n_ch + 1} {CHANNELS[k % n_ch]}" for k in range(n_vis)] + list(BODY)
    return senses + [f"n{h}" for h in range(n_hidden)] + list(out_labels)


class Anatomy:
    """La rete di un individuo (un genoma di una riga), pronta da descrivere.

    I neuroni sono numerati come le sorgenti del genoma: prima i sensi, poi gli
    interni, poi i motori. `w` sono i pesi effettivi (per le reti a impulsi il
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
        self.exists = np.ones(self.n_nodes, dtype=bool)
        self.exists[self.n_in : self.n_in + self.n_hidden] = genome.active[0]
        self.inhibitory = brains.inhibitory[0] if hasattr(brains, "inhibitory") else None
        self.labels = node_labels(self.n_in, self.n_hidden, brains.OUT_LABELS)
        self.n_vision = self.n_in - len(BODY)
        self.useful = self._reaches_motors()
        self.depth = self._depth_from_senses()

    # --- struttura ---------------------------------------------------------

    def is_hidden(self, v: int) -> bool:
        return self.n_in <= v < self.n_in + self.n_hidden

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
        depth[: self.n_in] = 0
        queue = deque(range(self.n_in))
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
            "neurons": int(self.exists[hidden].sum()),
            "useful_neurons": int((self.exists & self.useful)[hidden].sum()),
            "links": len(e),
            "useful_links": int(useful_edges.sum()),
            "senses_used": int(self.useful[: self.n_in].sum()),
            "loops": len(back),
        }
        if self.inhibitory is not None:
            out["inhibitory"] = int((self.inhibitory & self.exists[hidden]).sum())
        return out

    # --- descrizione -------------------------------------------------------

    def describe(self, v: int) -> list[str]:
        """Righe di testo sul neurone v: geni propri e collegamenti più forti."""
        name = self.labels[v]
        if v < self.n_in:
            role = "senso"
        elif self.is_motor(v):
            role = "motore"
        else:
            kind = ""
            if self.inhibitory is not None:
                kind = " inibitorio" if self.inhibitory[v - self.n_in] else " eccitatorio"
            role = f"neurone interno{kind}"
        lines = [f"{name}: {role}"]
        if not self.exists[v]:
            lines.append("spento: i suoi geni restano nel genoma")
        elif not self.useful[v]:
            lines.append("ramo morto: non arriva ai motori")
        if v >= self.n_in:
            lines += self.brains.trait_text(self.traits[v - self.n_in], self.is_hidden(v))
        ins, outs = self.inputs_of(v), self.outputs_of(v)
        lines.append(f"collegamenti: {len(ins)} entrano, {len(outs)} escono")
        for title, pairs in (("riceve da", [(s, self.weight(s, v)) for s in ins]),
                             ("manda a", [(d, self.weight(v, d)) for d in outs])):
            if pairs:
                pairs.sort(key=lambda p: -abs(p[1]))
                lines.append("")
                lines.append(f"{title} (peso, effetto):")
                lines += [f"  {self.labels[n]:13s}{w:+7.3f} {'eccita' if w > 0 else 'inibisce'}"
                          + ("" if self.useful[n] else " (morto)") for n, w in pairs]
        return lines

    def genome_text(self, title: str = "") -> str:
        """Il genoma completo in forma leggibile, blocco per blocco (un blocco
        per neurone interno o motore), compresi i geni latenti."""
        s = self.summary()
        head = [title] if title else []
        head += [
            f"rete {KIND_NAMES[self.kind]}: {s['neurons']}/{self.n_hidden} neuroni interni "
            f"({s['useful_neurons']} utili), {s['links']} collegamenti ({s['useful_links']} utili), "
            f"sensi usati {s['senses_used']}/{self.n_in}, anelli {s['loops']}",
            "pesi efficaci: >0 eccita, <0 inibisce"
            + (" (dai neuroni interni il segno lo dà il tipo)" if self.inhibitory is not None else ""),
            "latente = gene presente ma inattivo perché un capo del collegamento è spento",
            "",
        ]
        body = []
        for t in range(self.n_hidden + self.n_out):
            v = self.n_in + t
            state = "esiste" if self.exists[v] else "SPENTO"
            if self.exists[v] and not self.useful[v]:
                state += ", ramo morto"
            traits = "; ".join(self.brains.trait_text(self.traits[t], self.is_hidden(v)))
            body.append(f"[{self.labels[v]}] {state} | {traits}")
            for src in np.flatnonzero(self.conn[t]):
                live = self.links[t, src]
                w = self.w[t, src] if live else self.brains.weights[0, t, src]
                body.append(f"    <- {self.labels[src]:14s} {w:+.3f}{'' if live else '  (latente)'}")
        return "\n".join(head + body) + "\n"
