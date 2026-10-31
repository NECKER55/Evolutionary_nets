"""S: l'attenzione e la mappa di priorità, comune a tutti gli agenti."""

import numpy as np

from ...arenas.pile.world import SECTORS
from .genes import MAP, fovea_of
from .value import o_read


def priority_map(objs, MAPN, xp=np):
    """Per ogni posizione della mappa: c'è un oggetto, quale (il più vicino in
    quella direzione), le sue caratteristiche e la sua vicinanza.
    -> (has (n, P), obj (n, P), feat (n, P, F), near (n, P))."""
    vis, near, ang = objs["vis"], objs["near"], objs["ang"]
    width = (SECTORS[-1] - SECTORS[0]) / MAPN
    where_ = xp.clip(xp.floor((ang - SECTORS[0]) / width), 0, MAPN - 1).astype(int)
    in_pos = vis[:, :, None] & (where_[:, :, None] == xp.arange(MAPN)[None, None, :])
    obj = xp.argmax(xp.where(in_pos, near[:, :, None], -1.0), axis=1)  # (n, P)
    has = in_pos.any(1)
    feat = xp.take_along_axis(objs["feat"], obj[:, :, None], axis=1)
    near_p = xp.take_along_axis(near, obj, axis=1) * has
    return has, obj, feat, near_p


class Attention:
    """S: la mappa di priorità (vedi la docstring di `sections.py`). L'oggetto
    attento è `target` (indice negli oggetti in vista, -1 = nessuno)."""

    def __init__(self, n: int, m: int, dt: float, rng: np.random.Generator):
        self.n, self.m, self.dt, self.rng = n, m, dt, rng
        self.target = np.full(n, -1)
        self.inhib = np.full(n, -1)
        self.inhib_t = np.zeros(n)
        self.prev_vis = np.zeros((n, m), dtype=bool)
        self.novelty = np.zeros((n, m))
        self.new = np.zeros(n)
        self.prob = np.zeros((n, MAP))  # la mappa: probabilità di ogni posizione (per lo schermo)
        self.has_pos = np.zeros((n, MAP), dtype=bool)
        self.parts = np.zeros((n, MAP, 4))  # per lo schermo: innata, valore, importanza, incertezza
        self.changed = np.zeros(n, dtype=bool)

    def update(self, objs: dict, decide: np.ndarray, p: dict, extra=None, learned=None, o=None) -> None:
        """Sceglie l'oggetto (a ogni decisione, o subito se quello attento è
        sparito). `extra` (n, m): punteggio in più per oggetto (il guidato bara
        così). `o`: (o_pl, o_pa, o_assoc, energia), ciò che O sa; `learned` (n,):
        quanto conta valore e importanza (arbitraggio)."""
        dt, n = self.dt, self.n
        vis = objs["vis"]
        idx = np.arange(n)
        before = self.target.copy()
        self.novelty = self.novelty * np.exp(-dt / p["novelty_tau"])[:, None] + (vis & ~self.prev_vis)
        self.prev_vis = vis.copy()
        self.inhib_t = np.maximum(self.inhib_t - dt, 0.0)
        lost = (self.target >= 0) & ~vis[idx, np.maximum(self.target, 0)]
        self.target = np.where(lost, -1, self.target)
        self.new *= np.exp(-dt / p["new_tau"])
        has, obj, feat, near = priority_map(objs, MAP)
        self.has_pos = has
        choose = (decide | (self.target < 0)) & has.any(1)
        nov = np.take_along_axis(self.novelty, obj, axis=1)
        innate = (p["sal_near"][:, None] * near + p["sal_novelty"][:, None] * nov
                  + p["sal_stick"][:, None] * (obj == self.target[:, None])
                  - p["sal_inhibit"][:, None] * ((obj == self.inhib[:, None]) & (self.inhib_t > 0)[:, None]))
        value = imp = unc = np.zeros_like(innate)
        if o is not None:
            value, imp, unc = o_read(o[0], o[1], o[2], feat, o[3], p)
            trust = np.ones(n) if learned is None else learned
            value = trust[:, None] * p["sal_value"][:, None] * value
            imp = trust[:, None] * p["sal_import"][:, None] * imp
            unc = p["sal_unc"][:, None] * unc
        self.parts = np.stack([innate, value, imp, unc], -1) * has[:, :, None]
        score = innate + value + imp + unc
        if extra is not None:
            score = score + np.take_along_axis(extra, obj, axis=1)
        z = np.where(has, score / p["sal_temperature"][:, None], -np.inf)
        z = z - np.where(has.any(1, keepdims=True), z.max(1, keepdims=True), 0.0)
        prob = np.where(has, np.exp(z), 0.0)
        prob /= np.maximum(prob.sum(1, keepdims=True), 1e-12)
        self.prob = prob
        if choose.any():
            place = (np.cumsum(prob, 1) < self.rng.random((n, 1))).sum(1).clip(0, MAP - 1)
            sel = np.where(choose, obj[idx, place], self.target)
            changed = choose & (sel != self.target)
            self.new = np.where(changed, 1.0, self.new)
            self.target = sel
        self.changed = self.target != before  # l'oggetto guardato è cambiato (anche: sparito)

    def leave(self, mask: np.ndarray, p: dict) -> None:
        """"Lascialo": l'oggetto attento viene ignorato per qualche secondo."""
        self.inhib = np.where(mask, self.target, self.inhib)
        self.inhib_t = np.where(mask, p["inhibit_time"], self.inhib_t)
        self.target = np.where(mask, -1, self.target)

    def seen(self, objs: dict):
        """(c'è, caratteristiche (n, F), vicinanza, angolo) dell'oggetto attento."""
        idx, t = np.arange(self.n), np.maximum(self.target, 0)
        has = self.target >= 0
        return has, objs["feat"][idx, t], objs["near"][idx, t], objs["ang"][idx, t]

    def fovea(self, objs: dict) -> np.ndarray:
        has, feat, near, ang = self.seen(objs)
        return fovea_of(feat, near, ang, has, self.new)

    def reset(self, who) -> None:
        for x, v in ((self.target, -1), (self.inhib, -1), (self.inhib_t, 0.0), (self.novelty, 0.0),
                     (self.prev_vis, False), (self.new, 0.0), (self.prob, 0.0), (self.has_pos, False),
                     (self.parts, 0.0), (self.changed, False)):
            x[who] = v
