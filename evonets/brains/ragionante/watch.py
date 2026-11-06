"""L e W accanto a un agente, senza guidarlo: leggono i morsi veri, tengono la memoria e
dicono che cosa si aspettano. Non comandano niente: servono a vedere come lavorano
nel mondo vero, finché non esiste chi le usa. Ogni rinascita azzera la memoria."""

import jax
import jax.numpy as jnp
import numpy as np

from . import memory
from .events import ACTIONS, BITE, EMBED, OUTCOMES, PAIN, PLEASURE
from .record_events import bitten
from .train_memory import path_of
from .vision import FakeDetector

SHOWN = 6  # eventi ricordati per lo schermo
PILES = 2  # i tipi di `KINDS` che sono pile


class Watch:
    def __init__(self, world, brain, path=None):
        self.w, self.brain = world, brain
        self.p = memory.load(path or path_of("a", "_largo"))
        self.eyes = getattr(brain.boss, "eyes", None) or FakeDetector(world)
        self.h = np.zeros((world.B, world.n, memory.MEMORY))
        self.seen, self.need, self.taken = None, None, 0
        self.rows = {key: [] for key in np.ndindex(world.B, world.n)}  # (oggetto, esito, previsione, dubbio)
        self.reads = np.zeros((world.B, world.n), dtype=int)  # eventi scritti
        p = self.p
        self._ask = jax.jit(lambda h, q: self._forward(h, q))
        self._write = jax.jit(lambda h, x: memory.write(p, {"h": h}, x)["h"])

    def _forward(self, h, q) -> tuple:
        """Come `memory.predict`, ma tiene anche l'attività dei due strati di W."""
        p, x = self.p, jnp.concatenate([h, q])
        a = jnp.tanh(p["m1"] @ x + p["b1"])
        c = jnp.tanh(p["m2"] @ a + p["b2"])
        y = p["out"] @ c + p["b3"]
        return memory.pleasure(jax.nn.log_softmax(y[:-1])), y[-1], y[:-1], a, c

    def _query(self, obj, need: float):
        return memory.query_of(jnp.asarray(obj, jnp.float32), jnp.array(BITE), jnp.array(need, jnp.float32))

    def before(self) -> None:
        """Prima di un passo: che cosa si vede, e quanto serve energia."""
        self.seen = getattr(self.brain.boss, "seen", None) or self.eyes.see()
        self.need = 1.0 - self.w.energy / self.w.cfg.energy_max
        self.taken = len(self.w.taken)

    def after(self) -> None:
        """Dopo un passo: ogni morso diventa un evento."""
        w = self.w
        for _, b, i, colour, ok in w.taken[self.taken:]:
            obj = bitten(self.seen, w.cfg, b, i, colour)
            if obj is not None:
                self._event(b, i, obj, ok, ("rossa", "gialla")[colour])
        born = w.age <= w.cfg.dt
        if born.any() and w.t > w.cfg.dt:
            self.h[born] = 0.0

    def _event(self, b, i, obj, ok: bool, label: str) -> None:
        need = float(self.need[b, i])
        q = self._query(obj, need)
        good, doubt, logits, *_ = self._ask(jnp.asarray(self.h[b, i], jnp.float32), q)
        out = PLEASURE if ok else PAIN
        wrong = -jax.nn.log_softmax(logits)[out]
        x = jnp.concatenate([q[:EMBED + len(ACTIONS)], jax.nn.one_hot(out, len(OUTCOMES)), jnp.array([need, wrong])])
        self.h[b, i] = np.asarray(self._write(jnp.asarray(self.h[b, i], jnp.float32), x))
        self.reads[b, i] += 1
        self.rows[b, i] = (self.rows[b, i] + [(label, OUTCOMES[out], float(good), float(doubt))])[-SHOWN:]

    def looks(self, b: int, i: int) -> tuple:
        """Che cosa si aspetta dalle pile in vista, adesso. -> ([(colore, distanza, piacere, dubbio)],
        attività del primo strato di W, del secondo, logit degli esiti: tutto per la pila più vicina)."""
        seen = self.seen
        out, layers = [], (np.zeros(memory.MODEL), np.zeros(memory.MODEL), np.zeros(len(OUTCOMES)))
        if seen is None:
            return out, *layers
        keys = self.w.cfg.eye_range
        on = np.flatnonzero(seen.on[b, i] & (seen.kind[b, i] < PILES))
        on = on[np.argsort(seen.dist[b, i, on])]
        need = float(1.0 - self.w.energy[b, i] / self.w.cfg.energy_max)
        h = jnp.asarray(self.h[b, i], jnp.float32)
        for k, m in enumerate(on):
            good, doubt, logits, a, c = self._ask(h, self._query(seen.embedding(keys)[b, i, m], need))
            out.append((int(seen.kind[b, i, m]), float(seen.dist[b, i, m]), float(good), float(doubt)))
            if k == 0:
                layers = np.asarray(a), np.asarray(c), np.asarray(jax.nn.softmax(logits))
        return out, *layers
