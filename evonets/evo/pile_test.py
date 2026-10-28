"""Le prove nell'arena delle pile (docs/SPEC_ARENA_PILE.md).

Fase A: popolazione fissa, chi muore rinasce con lo stesso cervello di
partenza del suo tipo (nessuna evoluzione): si misura l'apprendimento dentro
la vita. Fase B: accoppiamento muso contro muso; chi muore rinasce con un
incrocio dei geni dell'ultima coppia del suo recinto (o di due del recinto a
caso, se non ce n'è ancora una): l'evoluzione tara i parametri e i valori
innati.

Due motori con la stessa interfaccia (`NumpyRun`, `gpu.pile.PileGpu`):
`advance(k)`, `world_of(b)`, `brain_of(i)`, `senses_of(b)`, le serie passo per
passo (`flat`), da cui `measures` calcola le misure.

Uso: `uv run python -m evonets.evo.pile_test --agente sezioni --minuti 20`;
`--confronto` fa i tre agenti e le lesioni; `--help` per le opzioni.
"""

import argparse
import json
import time

import numpy as np

from ..net.sections import AGENTS, GENE_NAMES, N_GENOME, Sections, Sizes, cross, decode, mutate
from ..sim.pile import EYES, N_PROGRAMS, PROGRAMS, Motor, PileConfig, PileWorld

WINDOWS = ((-30, 0), (0, 15), (15, 30), (30, 60), (60, 90))  # s rispetto all'inversione
SERIES = ("good_taken", "bad_taken", "starved", "aged", "matings", "inversion", "zone_change")


def start_genomes(agent: str, phase: str, N: int, seed: int, dt: float):
    """Fase B: genomi di partenza un po' diversi tra loro; fase A: quelli di base per tutti."""
    if phase != "B":
        return None
    return mutate(np.tile(AGENTS[agent](1, dt).genome, (N, 1)), np.random.default_rng([seed, 1]), 0.2)


class NumpyRun:
    """Una corsa nell'arena delle pile sul processore (riferimento)."""

    def __init__(self, cfg: PileConfig, agent: str, phase: str, enclosures: int, seed: int = 0, lesions=()):
        cfg.mating = phase == "B"
        self.cfg, self.agent, self.phase = cfg, agent, phase
        self.w = PileWorld(cfg, enclosures, seed)
        self.B, self.n = self.w.B, self.w.n
        N = self.B * self.n
        kw = {"sizes": Sizes(lesions=frozenset(lesions))} if agent == "sezioni" else {}
        self.brain = AGENTS[agent](N, cfg.dt, seed, genomes=start_genomes(agent, phase, N, seed, cfg.dt), **kw)
        self.motor = Motor((self.B, self.n), np.random.default_rng([seed, 2]))
        self.rng = np.random.default_rng([seed, 3])
        self.pairs = {}  # recinto -> (genoma a, genoma b) dell'ultima coppia
        self.reward = np.zeros(N)
        self.usage = np.zeros(N_PROGRAMS)
        self.counts = np.zeros((N, self.brain.nb, 4)) if isinstance(self.brain, Sections) else None
        self.series = {k: [] for k in SERIES}

    @property
    def t(self) -> float:
        return self.w.t

    def step(self) -> None:
        w, cfg, brain, B, n = self.w, self.cfg, self.brain, self.B, self.n
        N = B * n
        raw = w.senses()
        flat = raw.reshape(N, -1)
        if self.agent == "guidato":
            brain.good = np.repeat(w.good, n)
        program = brain.step(flat, self.reward)
        self.usage += np.bincount(program, minlength=N_PROGRAMS)
        if self.counts is not None:  # chi vince in B, rispetto a quale colore domina la vista
            eye = flat[:, EYES].reshape(N, 3, 3).max(-1)
            seen = np.where(eye.max(1) > 0.2, eye.argmax(1), 3)
            self.counts[np.arange(N), brain.winner(), seen] += 1.0
        thrust, steer = self.motor(program.reshape(B, n), raw, cfg.dt)
        if self.agent == "guidato":  # sa anche dov'è la zona: quando esplora ci va
            to = w.zone_centres()[w.zone][:, None, :] - w.pos
            ang = (np.arctan2(to[..., 1], to[..., 0]) - w.heading + np.pi) % (2 * np.pi) - np.pi
            go = (program.reshape(B, n) == 3) & (np.hypot(to[..., 0], to[..., 1]) > cfg.zone_half)
            steer = np.where(go, np.clip(2.0 * ang, -1.0, 1.0), steer)
        before = (len(w.taken), dict(w.deaths), len(w.matings), len(w.inversions), len(w.zone_changes))
        out = w.step(thrust, steer)
        new = w.taken[before[0]:]
        self.series["good_taken"].append(sum(1 for x in new if x[4]))
        self.series["bad_taken"].append(sum(1 for x in new if not x[4]))
        self.series["starved"].append(w.deaths["fame"] - before[1]["fame"])
        self.series["aged"].append(w.deaths["vecchiaia"] - before[1]["vecchiaia"])
        self.series["matings"].append(len(w.matings) - before[2])
        self.series["inversion"].append(len(w.inversions) > before[3])
        self.series["zone_change"].append(len(w.zone_changes) > before[4])
        self.reward = (out["pleasure"] - out["pain"]).reshape(N)
        for b, i, j in out["mates"]:
            self.pairs[b] = (brain.genome[b * n + i].copy(), brain.genome[b * n + j].copy())
        dead = out["dead"].reshape(N)
        if dead.any():
            child = None
            if self.phase == "B":
                child = np.zeros((int(dead.sum()), N_GENOME))
                for k, s in enumerate(np.flatnonzero(dead)):
                    b = s // n
                    if b in self.pairs:
                        ga, gb = self.pairs[b]
                    else:
                        ga, gb = (brain.genome[b * n + x] for x in self.rng.integers(0, n, 2))
                    child[k] = mutate(cross(ga[None], gb[None], self.rng), self.rng)[0]
            brain.reborn(dead, child)
            if self.counts is not None:
                self.counts[dead] = 0.0
            self.reward[dead] = 0.0

    def advance(self, k: int) -> None:
        for _ in range(int(k)):
            self.step()

    def world_of(self, b: int) -> dict:
        w = self.w
        out = {k: getattr(w, k)[b] for k in ("pos", "heading", "speed", "energy", "age", "charging", "taste", "pain",
                                             "good", "zone", "bpos", "bcolor", "bon", "rest")}
        out.update(t=w.t, next_inv=w.next_inversion, next_zone=w.next_zone)
        return out

    def brain_of(self, i: int) -> dict:
        b = self.brain
        keys = ("genome", "w_e", "w_v", "program", "pi", "delta", "proto", "pred", "win_rate", "state_b", "mem",
                "w_gate", "c_in", "u_c", "u_l", "code", "unc")
        out = {k: np.asarray(getattr(b, k))[i] for k in keys if hasattr(b, k)}
        if hasattr(b, "a"):
            out["a_slow"] = b.a.slow[i]
        if self.counts is not None:
            out["counts"] = self.counts[i]
        return out

    def senses_of(self, b: int) -> np.ndarray:
        return self.w.senses()[b]

    def view(self, b: int, i: int) -> tuple[dict, dict, np.ndarray]:
        return self.world_of(b), self.brain_of(i), self.senses_of(b)

    def flat(self, name):
        return np.asarray(self.series[name], dtype=float)


def make_run(agent="sezioni", phase="A", enclosures=200, seed=0, lesions=(), cfg: PileConfig | None = None,
             engine="gpu"):
    """La corsa sul motore richiesto (la GPU se c'è, altrimenti il processore)."""
    cfg = cfg or PileConfig()
    if engine == "gpu":
        from ..gpu import available

        if available():
            from ..gpu.pile import PileGpu

            return PileGpu(cfg, agent, phase, enclosures, seed, lesions,
                           genomes=start_genomes(agent, phase, enclosures * cfg.beings, seed, cfg.dt))
    return NumpyRun(cfg, agent, phase, enclosures, seed, lesions)


def run(agent: str = "sezioni", phase: str = "A", enclosures: int = 200, minutes: float = 20.0, seed: int = 0,
        lesions=(), cfg: PileConfig | None = None, log=None, every: float = 300.0, engine: str = "gpu") -> dict:
    r = make_run(agent, phase, enclosures, seed, lesions, cfg, engine)
    dt = r.cfg.dt
    steps = int(round(minutes * 60.0 / dt))
    block = int(round(every / dt))
    t0 = time.time()
    history = []
    done = 0
    while done < steps:
        k = min(block, steps - done)
        r.advance(k)
        done += k
        if log:
            row = {"minuto": round(r.t / 60.0, 1), **measures(r), "s": round(time.time() - t0)}
            if phase == "B":
                p = decode(np.asarray(r.state["brain"]["genome"]) if hasattr(r, "state") else r.brain.genome)
                row["geni"] = {k: round(float(np.median(p[k])), 3) for k in GENE_NAMES}
                row["valori innati"] = np.round(np.median(p["innate"], 0), 3).tolist()
            history.append(row)
            log(json.dumps(row, ensure_ascii=False))
    usage = usage_of(r)
    res = {"agente": agent, "fase": phase, "lesioni": sorted(lesions), "motore": type(r).__name__, **measures(r),
           "programmi": dict(zip(PROGRAMS, np.round(usage / max(usage.sum(), 1), 3).tolist())),
           "s": round(time.time() - t0)}
    counts = counts_of(r)
    if counts is not None:
        res["emersione B"] = selectivity(counts)
    res["storia"] = history
    return res


def usage_of(r) -> np.ndarray:
    return np.asarray(r.state["misc"]["usage"]) if hasattr(r, "state") else r.usage


def counts_of(r):
    if hasattr(r, "state"):
        c = r.state["brain"].get("counts")
        return None if c is None else np.asarray(c)
    return r.counts


def measures(r) -> dict:
    """Le misure (docs/SPEC_ARENA_PILE.md, "Le misure"), dalle serie passo per passo."""
    dt = r.cfg.dt
    good, bad = r.flat("good_taken"), r.flat("bad_taken")
    steps = len(good)
    if not steps:
        return {}
    t = (np.arange(steps) + 1) * dt
    minutes = steps * dt / 60.0
    beings = r.B * r.n
    total = good + bad
    out = {"pile al minuto": round(float(total.sum()) / minutes / beings, 3),
           "buone": round(float(good.sum() / max(total.sum(), 1)), 3),
           "morti di fame al minuto": round(float(r.flat("starved").sum()) / minutes / beings, 4),
           "accoppiamenti al minuto": round(float(r.flat("matings").sum()) / minutes / r.B, 3)}
    inversions = t[r.flat("inversion") > 0]
    win = {f"{a}..{b} s": [] for a, b in WINDOWS}
    curves = []
    for T in inversions:
        if T + WINDOWS[-1][1] > t[-1] or T + WINDOWS[0][0] < 0:
            continue
        for a, b in WINDOWS:
            m = (t >= T + a) & (t < T + b)
            if total[m].sum():
                win[f"{a}..{b} s"].append(good[m].sum() / total[m].sum())
        bins = []
        for k in range(0, 90, 5):
            m = (t >= T + k) & (t < T + k + 5)
            bins.append(good[m].sum() / total[m].sum() if total[m].sum() else np.nan)
        curves.append(bins)
    out["scelta intorno alle inversioni"] = {k: round(float(np.mean(v)), 3) if v else None for k, v in win.items()}
    if curves:
        c = np.nanmean(np.array(curves), 0)
        ok = np.flatnonzero(c >= 0.75)
        out["secondi per adattarsi"] = int(ok[0] * 5) if len(ok) else None
    changes = t[r.flat("zone_change") > 0]
    after = [total[(t >= T) & (t < T + 30)].sum() / 0.5 for T in changes if T + 30 <= t[-1]]
    if after and total.sum():
        out["pile dopo un cambio di zona (rispetto alla media)"] = round(
            float(np.mean(after)) / (total.sum() / minutes), 3)
    return out


def selectivity(counts: np.ndarray) -> dict:
    """Quanto il neurone vincente di B dice quale colore domina la vista:
    informazione mutua normalizzata (0 = niente, 1 = lo dice sempre), per chi
    ha vissuto abbastanza; e quanti neuroni di B vincono davvero."""
    nmi, used = [], []
    for c in counts:
        tot = c.sum()
        if tot < 600:
            continue
        p = c / tot
        pw, ps = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.nansum(p * np.log(p / (pw * ps)))
            hs = -np.nansum(ps * np.log(ps))
        nmi.append(mi / hs if hs > 0 else 0.0)
        used.append(int((c.sum(1) / tot > 0.02).sum()))
    if not nmi:
        return {}
    return {"informazione sul colore": round(float(np.mean(nmi)), 3), "neuroni usati": round(float(np.mean(used)), 1),
            "vite misurate": len(nmi)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agente", choices=tuple(AGENTS), default="sezioni")
    ap.add_argument("--fase", choices=("A", "B"), default="A")
    ap.add_argument("--recinti", type=int, default=200)
    ap.add_argument("--minuti", type=float, default=20.0)
    ap.add_argument("--lesioni", default="", help="sezioni da togliere, per esempio D o B,C (solo per sezioni)")
    ap.add_argument("--confronto", action="store_true", help="i tre agenti e le lesioni del cervello a sezioni")
    ap.add_argument("--motore", choices=("gpu", "numpy"), default="gpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    show = lambda r: print(json.dumps({k: v for k, v in r.items() if k != "storia"}, ensure_ascii=False), flush=True)
    if args.confronto:
        for agent, les in (("guidato", ()), ("attore-critico", ()), ("sezioni", ()), ("sezioni", ("B",)),
                           ("sezioni", ("D",)), ("sezioni", ("C",)), ("sezioni", ("L",)), ("sezioni", ("U",)),
                           ("sezioni", ("V",))):
            show(run(agent, "A", args.recinti, args.minuti, args.seed, les, engine=args.motore))
        return
    lesions = tuple(x for x in args.lesioni.split(",") if x)
    log = (lambda s: print(s, flush=True))
    show(run(args.agente, args.fase, args.recinti, args.minuti, args.seed, lesions, log=log, engine=args.motore))


if __name__ == "__main__":
    main()
