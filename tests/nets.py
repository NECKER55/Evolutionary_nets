"""Reti di prova più ricche di quelle primordiali: tanti recettori accesi,
neuroni interni in numero variabile, collegamenti fitti."""

import numpy as np

from evonets import Config
from evonets.net import Brains, SpikingBrains


def rich(n=50, seed=0, R=10, H=12, net="rate", conn=0.3, **kw) -> Brains:
    cfg = Config(net_type=net, receptor_slots=R, receptor_initial=R, net_hidden_slots=H, **kw)
    kind, n_out = (Brains, 2) if net == "rate" else (SpikingBrains, 4)
    b = kind(cfg, n, n_out, seed)
    rng = np.random.default_rng(seed + 1000)
    g = b.genome
    g.active[:] = rng.random((n, H)) < rng.random((n, 1))
    g.conn[:] = rng.random(g.conn.shape) < conn
    b.set_genome(g)
    return b
