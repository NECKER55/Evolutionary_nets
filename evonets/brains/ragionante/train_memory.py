"""Addestra le sezioni L e W sul mondo a eventi: ogni passo sono vite nuove, ognuna con
la sua regola nascosta, e una vita non si rivede mai.

Uso: `uv run python -m evonets.brains.ragionante.train_memory --variante a --passi 4000`.
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import memory
from .events import LEVELS, bite_accuracy, lives

BATCH = 256
RATE = 0.002
CLIP = 1.0  # il gradiente lungo una vita può esplodere


def path_of(variant: str, tag: str = "") -> Path:
    return Path(__file__).parent / f"memory_{variant}{tag}.npz"


def train(variant: str, names: tuple, steps: int, seed: int, params=None, every: int = 500, path=None) -> dict:
    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    params = params or memory.init(k, variant)
    slow = jax.tree_util.tree_map(jnp.zeros_like, params)
    fast = jax.tree_util.tree_map(jnp.zeros_like, params)

    @jax.jit
    def step(params, slow, fast, t, k):
        data = lives(k, BATCH, names)
        value, grads = jax.value_and_grad(memory.loss)(params, data)
        size = jnp.sqrt(sum((g * g).sum() for g in jax.tree_util.tree_leaves(grads)))
        grads = jax.tree_util.tree_map(lambda g: g * jnp.minimum(1.0, CLIP / (size + 1e-6)), grads)
        slow = jax.tree_util.tree_map(lambda s, g: 0.9 * s + 0.1 * g, slow, grads)
        fast = jax.tree_util.tree_map(lambda s, g: 0.999 * s + 0.001 * g * g, fast, grads)
        rate = RATE * jnp.minimum(1.0, 2.0 * (1.0 - t / steps))
        params = jax.tree_util.tree_map(
            lambda p, s, f: p - rate * (s / (1.0 - 0.9 ** (t + 1.0))) / (jnp.sqrt(f / (1.0 - 0.999 ** (t + 1.0))) + 1e-8),
            params, slow, fast)
        return params, slow, fast, value

    start = time.time()
    for t in range(steps):
        key, k = jax.random.split(key)
        params, slow, fast, value = step(params, slow, fast, float(t), k)
        if t % every == 0 or t == steps - 1:
            key, k = jax.random.split(key)
            test = lives(k, 512, names)
            accuracy = bite_accuracy(memory.forecast(params, test), test, 5)
            print(f"passo {t}: errore {float(value):.3f}, morsi giusti {accuracy:.3f} ({time.time() - start:.0f} s, "
                  f"{(t + 1) * BATCH} vite)", flush=True)
            if path:
                memory.save(params, path)
    return params


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variante", choices=("a", "b"), default="a")
    p.add_argument("--livello", choices=tuple(LEVELS), default="1")
    p.add_argument("--passi", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    train(a.variante, LEVELS[a.livello], a.passi, a.seed, path=path_of(a.variante, f"_{a.livello}"))


if __name__ == "__main__":
    main()
