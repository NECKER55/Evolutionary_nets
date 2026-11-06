"""Addestra la sezione P sulle etichette del simulatore: raccoglie immagini vere
in mondi a caso e impara che cosa c'è in ogni pixel.

Uso: `uv run python -m evonets.brains.ragionante.train_vision --immagini 3000`.
I pesi finiscono in `vision.npz`. Le prove d'esame stanno in `tests/test_ragionante.py`.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from ...arenas.mondo3d.maps import PLANS
from ...arenas.mondo3d.run import Run3D
from ...arenas.mondo3d.world import Mondo3D, Mondo3DConfig
from .vision import WEIGHTS, logits, network, sampled, shrink

EVERY = 7  # passi del mondo tra un'immagine e l'altra
STAGED = 0.6  # quota di scene costruite (`stage`); le altre vengono da esseri che guidano
WALLS = 0.5  # quanti esseri di una scena stanno contro un muro
CLOSE = 0.5  # quanti hanno una pila a un passo
RATE = 0.003
BATCH = 24
WHO = ("perfetto", "ragionante", "vedente finto")  # chi guida: il perfetto non va mai vicino ai muri, gli altri sì
OBJECT = 3.0  # quanto pesa un pixel di oggetto rispetto allo sfondo: sono pochi


def stage(w, rng) -> None:
    """Una scena costruita: esseri ovunque, anche contro i muri e rivolti verso di
    loro, e pile a un passo dal muso. Chi guida da solo in queste viste ci arriva
    poco, e sono proprio quelle in cui una pila fantasma blocca l'essere."""
    cfg = w.cfg
    near = cfg.radius + 3.0
    w.pos[0] = rng.uniform(near, cfg.size - near, (w.n, 2))
    for i in range(w.n):
        if rng.random() < WALLS:
            side = rng.integers(0, 2)
            w.pos[0, i, side] = rng.choice((near, cfg.size - near)) + rng.uniform(-2.0, 2.0)
            w.pos[0, i, 1 - side] = rng.uniform(near, cfg.size - near)
    w._clear(w.pos, np.ones((w.B, w.n), dtype=bool))
    w.heading[0] = rng.uniform(-np.pi, np.pi, w.n)
    w.bcolor[0] = rng.integers(0, 2, cfg.batteries)
    w.bbig[0] = rng.integers(0, 2, cfg.batteries)
    w.bon[0] = rng.random(cfg.batteries) < 0.7
    w.bpos[0] = rng.uniform(cfg.battery_radius, cfg.size - cfg.battery_radius, (cfg.batteries, 2))
    for i in range(w.n):
        if rng.random() < CLOSE:
            k, far, turn = rng.integers(cfg.batteries), rng.uniform(near + 8.0, 70.0), w.heading[0, i] + rng.uniform(-1.0, 1.0)
            w.bon[0, k], w.bpos[0, k] = True, w.pos[0, i] + far * np.array((np.cos(turn), np.sin(turn)))
    for k in range(cfg.batteries):
        w.bon[0, k] &= not len(w.layouts[0].near_wall(w.bpos[0, k], cfg.battery_radius))  # non dentro un muro
    w.sync()


def sample(w, x, y) -> None:
    for b, i in np.ndindex(w.B, w.n):
        v = w.view(b, i)
        x.append(shrink(v["rgb"], v["depth"]))
        y.append(sampled(w.labels(b, i), x[-1]))


def collect(images: int, seed: int) -> tuple:
    """Immagini e verità da mondi diversi, con esseri che girano davvero: `images` in tutto."""
    x, y = [], []
    run = None
    worlds = [Mondo3D(Mondo3DConfig(beings=3, map=name), 1, seed) for name in PLANS]  # ogni mappa ha muri suoi
    rng = np.random.default_rng(seed)
    while len(x) < images * STAGED:
        world = worlds[len(x) // 3 % len(worlds)]  # 3 immagini a scena
        stage(world, rng)
        sample(world, x, y)
    for world in worlds:
        world.close()
    while len(x) < images:
        if run is None or run.w.t > 120.0:
            if run:
                run.close()
            seed += 1
            run = Run3D(Mondo3DConfig(beings=3, map=tuple(PLANS)[seed // len(WHO) % len(PLANS)]), WHO[seed % len(WHO)], 1, seed)
            run.advance(100)
        run.advance(EVERY)
        sample(run.w, x, y)
    run.close()
    return np.stack(x), np.stack(y)


def loss(params, x, y):
    log = jax.nn.log_softmax(logits(params, x))
    each = -jnp.take_along_axis(log, y[..., None], -1)[..., 0]
    weight = jnp.where(y > 0, OBJECT, 1.0)
    return (weight * each).sum() / weight.sum()


def load(path=WEIGHTS) -> list:
    data = np.load(path)
    return [[jnp.asarray(data[f"k{i}"]), jnp.asarray(data[f"b{i}"])] for i in range(len(data.files) // 2)]


def train(x, y, steps: int, seed: int, params=None) -> list:
    params = params or network(jax.random.PRNGKey(seed))
    leaves, tree = jax.tree_util.tree_flatten(params)
    slow = [jnp.zeros_like(p) for p in leaves]
    fast = [jnp.zeros_like(p) for p in leaves]

    @jax.jit
    def step(leaves, slow, fast, t, xb, yb):
        value, grads = jax.value_and_grad(loss)(jax.tree_util.tree_unflatten(tree, leaves), xb, yb)
        grads = jax.tree_util.tree_leaves(grads)
        slow = [0.9 * s + 0.1 * g for s, g in zip(slow, grads)]
        fast = [0.999 * f + 0.001 * g * g for f, g in zip(fast, grads)]
        rate = RATE * jnp.minimum(1.0, 2.0 * (1.0 - t / steps))  # l'ultima metà scende a zero
        return ([p - rate * (s / (1.0 - 0.9 ** (t + 1))) / (jnp.sqrt(f / (1.0 - 0.999 ** (t + 1))) + 1e-8)
                 for p, s, f in zip(leaves, slow, fast)], slow, fast, value)

    rng = np.random.default_rng(seed)
    start = time.time()
    for t in range(steps):
        pick = rng.integers(0, len(x), BATCH)
        leaves, slow, fast, value = step(leaves, slow, fast, float(t), x[pick], y[pick].astype(int))
        if t % 200 == 0:
            print(f"passo {t}: errore {float(value):.4f} ({time.time() - start:.0f} s)", flush=True)
    return jax.tree_util.tree_unflatten(tree, leaves)


def save(params, path=WEIGHTS) -> None:
    np.savez_compressed(path, **{f"k{i}": np.asarray(k) for i, (k, _) in enumerate(params)},
                        **{f"b{i}": np.asarray(b) for i, (_, b) in enumerate(params)})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--immagini", type=int, default=3000)
    p.add_argument("--passi", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--riprendi", action="store_true", help="parte dai pesi che ci sono")
    a = p.parse_args()
    x, y = collect(a.immagini, a.seed)
    print(f"{len(x)} immagini, pixel di oggetto: {(y > 0).mean():.2%}", flush=True)
    save(train(x, y, a.passi, a.seed, load() if a.riprendi else None))


if __name__ == "__main__":
    main()
