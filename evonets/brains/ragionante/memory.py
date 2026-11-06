"""Le sezioni L e W insieme, un world model a eventi.

La sezione L legge gli eventi di una vita e li riassume; la sezione W, data quella
memoria, dice che cosa succederebbe con un oggetto e un'azione, e quanto se ne fida.
Mentre si vive i pesi fissi non cambiano: cambia la memoria. Ciò che si addestra è
il modo di imparare in una vita, su tante vite con regole diverse (`train_memory.py`).

Due varianti della sezione L, sullo stesso esame:

- "a": una GRU, il ricordo sta nell'attività;
- "b": una rete ricorrente con sinapsi veloci in stile Hopfield, che si rafforzano con
  l'attività e decadono: il ricordo sta anche nei pesi, e la regola è appresa.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .events import ACTIONS, EMBED, OUTCOMES, PAIN, PLEASURE

MEMORY = 64  # neuroni della sezione L
MODEL = 128  # neuroni per strato della sezione W
INPUT = EMBED + len(ACTIONS) + len(OUTCOMES) + 2  # oggetto, azione, esito, bisogno, sorpresa
QUERY = EMBED + len(ACTIONS) + 1  # oggetto, azione, bisogno
# Ogni caratteristica alla stessa scala. Senza, il colore varia di 1 e la grandezza di 0,32: il
# gradiente trascura la grandezza e la rete impara solo i colori.
SCALE = jnp.array([1.0, 1.0, 1.0, 3.0, 1.7, 1.0, 1.0])


def init(key, variant: str) -> dict:
    keys = iter(jax.random.split(key, 16))
    w = lambda a, b, s=None: jax.random.normal(next(keys), (a, b)) * (s if s is not None else 1.0 / np.sqrt(b))
    params = {"m1": w(MODEL, MEMORY + QUERY), "b1": jnp.zeros(MODEL), "m2": w(MODEL, MODEL), "b2": jnp.zeros(MODEL),
              "out": w(len(OUTCOMES) + 1, MODEL, 0.01), "b3": jnp.zeros(len(OUTCOMES) + 1)}
    n, x = MEMORY, INPUT
    if variant == "a":
        for gate in ("z", "r", "h"):
            params[f"{gate}x"], params[f"{gate}h"], params[f"{gate}b"] = w(n, x), w(n, n), jnp.zeros(n)
    else:
        params |= {"rec": w(n, n), "inp": w(n, x), "bias": jnp.zeros(n), "keep": jnp.array(2.0), "gain": jnp.array(-2.0),
                   "learn": w(1, n, 0.01), "lb": jnp.zeros(1)}
    return params


def blank(params: dict) -> dict:
    """La memoria alla nascita."""
    state = {"h": jnp.zeros(MEMORY)}
    return state | ({"fast": jnp.zeros((MEMORY, MEMORY))} if "rec" in params else {})


def write(p: dict, s: dict, x) -> dict:
    """La sezione L legge un evento."""
    h = s["h"]
    if "rec" not in p:
        z = jax.nn.sigmoid(p["zx"] @ x + p["zh"] @ h + p["zb"])
        r = jax.nn.sigmoid(p["rx"] @ x + p["rh"] @ h + p["rb"])
        new = jnp.tanh(p["hx"] @ x + p["hh"] @ (r * h) + p["hb"])
        return {"h": (1.0 - z) * h + z * new}
    new = jnp.tanh(p["inp"] @ x + (p["rec"] + s["fast"]) @ h + p["bias"])
    how = jax.nn.sigmoid(p["learn"] @ new + p["lb"])[0]  # la rete decide quanto vale la pena ricordare
    fast = jax.nn.sigmoid(p["keep"]) * s["fast"] + how * jax.nn.sigmoid(p["gain"]) * jnp.outer(new, new) / MEMORY * 8.0
    return {"h": new, "fast": fast}


def predict(p: dict, h, query) -> tuple:
    """La sezione W: -> (logit degli esiti, quanto si aspetta di sbagliare)."""
    x = jnp.concatenate([h, query])
    a = jnp.tanh(p["m1"] @ x + p["b1"])
    a = jnp.tanh(p["m2"] @ a + p["b2"])
    y = p["out"] @ a + p["b3"]
    return y[:-1], y[-1]


def query_of(obj, act, need):
    return jnp.concatenate([obj * SCALE, jax.nn.one_hot(act, len(ACTIONS)), need[None]])


def live(p: dict, data: dict, blind: bool = False) -> dict:
    """Una vita, evento per evento: prima prevede, poi vede l'esito e lo scrive.
    `blind`: la sezione W non riceve la memoria (la lesione)."""

    def event(s, e):
        q = query_of(e["obj"], e["act"], e["need"])
        logits, doubt = predict(p, s["h"] * (0.0 if blind else 1.0), q)
        log = jax.nn.log_softmax(logits)
        wrong = -log[e["out"]]
        x = jnp.concatenate([q[:EMBED + len(ACTIONS)], jax.nn.one_hot(e["out"], len(OUTCOMES)), e["need"][None],
                             jax.lax.stop_gradient(wrong)[None]])
        return write(p, s, x), {"wrong": wrong, "doubt": doubt, "log": log}

    keep = {k: data[k] for k in ("obj", "act", "need", "out")}
    return jax.lax.scan(event, blank(p), keep)[1]


def loss(p: dict, data: dict):
    r = jax.vmap(lambda d: live(p, d))(data)
    return r["wrong"].mean() + 0.1 * ((r["doubt"] - jax.lax.stop_gradient(r["wrong"])) ** 2).mean()


def pleasure(log) -> jnp.ndarray:
    """Dato un morso, quanto è probabile il piacere invece del dolore."""
    good, bad = jnp.exp(log[..., PLEASURE]), jnp.exp(log[..., PAIN])
    return good / (good + bad + 1e-9)


def forecast(p: dict, data: dict, blind: bool = False) -> np.ndarray:
    """(N, T): quanto è probabile il piacere, per gli eventi in cui si morde."""
    r = jax.vmap(lambda d: live(p, d, blind))(data)
    return np.asarray(pleasure(r["log"]))


def save(params: dict, path) -> None:
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in params.items()})


def load(path) -> dict:
    data = np.load(path)
    return {k: jnp.asarray(data[k]) for k in data.files}


