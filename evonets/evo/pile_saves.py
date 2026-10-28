"""Salvataggi delle corse nell'arena delle pile (docs/SPEC_ARENA_PILE.md) e
la prova delle reti una a una.

Un salvataggio contiene il mondo di tutti i recinti, il cervello di ogni
essere (il genoma, cioè i pesi e le connessioni di partenza, e tutto ciò che
ha imparato), le serie delle misure e i contatori per essere. La forma è la
stessa per i due motori (i nomi sono quelli della versione GPU): un
salvataggio fatto sulla GPU si riapre sul processore e viceversa. Alla
ripresa valgono le regole attuali del codice; dal file vengono solo lo stato
e la corsa (cervello, fase, lesioni, recinti, esseri per recinto, seme).

`alone` mette ogni rete da sola in un recinto suo, tutte insieme: così si
provano una a una, ognuna con i suoi contatori.
"""

import json
import os
from pathlib import Path

import numpy as np

from ..net.sections import decode
from ..sim.pile import PileConfig
from .pile_test import SERIES, NumpyRun, make_run

FORMAT = 1
WORLD_KEYS = ("pos", "heading", "speed", "energy", "age", "charging", "taste", "pain", "good", "zone", "bpos",
              "bcolor", "bbig", "bon", "bwait", "bage", "pair_time", "rest", "ghost", "ghost_hide")
MOTOR_KEYS = {"wander": "wander", "escape": "escape", "f_in": "inputs", "f_act": "activity"}
BRAIN_ATTR = {"att_target": ("att", "target"), "att_inhib": ("att", "inhib"), "att_inhib_t": ("att", "inhib_t"),
              "att_prev_vis": ("att", "prev_vis"), "att_novelty": ("att", "novelty"),
              "att_new": ("att", "new"), "a_slow": ("a", "slow")}
BRAIN_KEYS = ("genome", "program", "pi", "bite", "p_bite", "w_e", "e_e", "w_v", "e_v", "v_prev", "value", "trust", "fresh", "delta",
              "teach", "proto", "pred", "win_rate", "state_b", "prev_b", "mem", "w_gate", "e_gate", "c_in", "u_c",
              "e_c", "u_l", "code", "d_fast", "d_slow", "t_slow", "unc", "teacher", "obs_err", "o_pl", "o_pa",
              "o_assoc", "o_e", "ep_on", "ep_pl", "ep_pa", "ep_e", "o_err", "o_out")
OBSERVING = ("teacher", "obs_err")
BY = ("good_by", "bad_by", "deaths_by")


def _get(obj, path):
    for name in path:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


# --- Lo stato, uguale per i due motori --------------------------------------------


def export_state(run) -> dict:
    """{"world", "brain", "misc", "series"}: array NumPy, con i nomi della GPU."""
    if not isinstance(run, NumpyRun):
        import jax

        st = jax.device_get(run.state)
        out = {part: {k: np.asarray(v) for k, v in st[part].items()} for part in ("world", "brain", "misc")}
        out["series"] = {k: run.flat(k) for k in SERIES}
        return out
    w, b = run.w, run.brain
    world = {k: np.asarray(getattr(w, k)) for k in WORLD_KEYS}
    world.update(t=np.float64(w.t), next_inv=np.float64(w.next_inversion), next_zone=np.float64(w.next_zone))
    for key, attr in MOTOR_KEYS.items():
        v = getattr(run.motor, attr, None)
        if v is not None:
            world[key] = np.asarray(v)
    brain = {"clock": np.asarray(b.clock)}
    for k in BRAIN_KEYS:
        if hasattr(b, k):
            brain[k] = np.asarray(getattr(b, k))
    for k, path in BRAIN_ATTR.items():
        v = _get(b, path)
        if v is not None:
            brain[k] = np.asarray(v)
    if run.counts is not None:
        brain["counts"] = run.counts
    G = b.genome.shape[1]
    pair, has = np.zeros((run.B, 2, G)), np.zeros(run.B, dtype=bool)
    for k, (ga, gb) in run.pairs.items():
        pair[k], has[k] = (ga, gb), True
    misc = {"reward": run.reward, "usage": run.usage, "pair": pair, "has_pair": has, "counted": run.counted, **run.by}
    return {"world": world, "brain": brain, "misc": misc, "series": {k: run.flat(k) for k in SERIES}}


def import_state(run, state: dict, parts=("world", "brain", "misc", "series")) -> None:
    """Rimette `state` (di `export_state`) nella corsa. Gli array devono avere la
    forma di quelli della corsa (stesso cervello, recinti, esseri)."""
    if not isinstance(run, NumpyRun):
        import jax.numpy as jnp

        from ..gpu.pile import F

        st = {part: dict(run.state[part]) for part in ("world", "brain", "misc")}
        for part in ("world", "brain", "misc"):
            if part not in parts:
                continue
            for k, v in state[part].items():
                if k in st[part]:
                    _check(part, k, np.shape(st[part][k]), np.shape(v))
                    st[part][k] = jnp.asarray(v, F) if np.asarray(v).dtype.kind == "f" else jnp.asarray(v)
        run.state = st
        run.t = float(np.asarray(st["world"]["t"]))
        if "series" in parts:
            run.series = {k: [np.asarray(state["series"][k])] for k in SERIES}
        return
    w, b = run.w, run.brain
    if "world" in parts:
        ws = state["world"]
        for k in WORLD_KEYS:
            _check("world", k, np.shape(getattr(w, k)), np.shape(ws[k]))
            setattr(w, k, np.array(ws[k]))
        w.t, w.next_inversion, w.next_zone = float(ws["t"]), float(ws["next_inv"]), float(ws["next_zone"])
        for key, attr in MOTOR_KEYS.items():
            if key in ws:
                setattr(run.motor, attr, np.array(ws[key]))
    if "brain" in parts:
        bs = state["brain"]
        if "clock" in bs:
            b.clock = np.broadcast_to(np.asarray(bs["clock"], dtype=float), b.clock.shape).copy()
        for k in BRAIN_KEYS:
            if k in bs and hasattr(b, k):
                _check("brain", k, np.shape(getattr(b, k)), np.shape(bs[k]))
                setattr(b, k, np.array(bs[k]))
        for k, path in BRAIN_ATTR.items():
            owner = _get(b, path[:-1])
            if k in bs and owner is not None:
                setattr(owner, path[-1], np.array(bs[k]))
        b.p = decode(b.genome)
        if "counts" in bs and run.counts is not None:
            run.counts = np.array(bs["counts"])
    if "misc" in parts:
        ms = state["misc"]
        run.reward, run.usage = np.array(ms["reward"]), np.array(ms["usage"])
        run.pairs = {k: (ms["pair"][k, 0].copy(), ms["pair"][k, 1].copy()) for k in np.flatnonzero(ms["has_pair"])}
        run.by = {k: np.array(ms[k]) for k in BY if k in ms}
        if "counted" in ms:
            run.counted = np.array(ms["counted"])
    if "series" in parts:
        run.series = {k: list(np.asarray(state["series"][k])) for k in SERIES}


def _check(part, key, want, got):
    if tuple(want) != tuple(got):
        raise ValueError(f"{part}/{key}: nel salvataggio {tuple(got)}, questa corsa vuole {tuple(want)} "
                         "(un salvataggio di un'altra forma o di una versione vecchia del cervello)")


# --- Salvare e riaprire --------------------------------------------------------------


def meta_of(run) -> dict:
    return {"agent": run.agent, "phase": run.phase, "lesions": sorted(run.lesions), "enclosures": run.B,
            "beings": run.n, "seed": run.seed, "minutes": round(run.t / 60.0, 2), "engine": type(run).__name__,
            "good_by": run.cfg.good_by, "teacher": getattr(run, "teacher_kind", "fantasma")}


def save(run, path) -> None:
    """Scrive la corsa in `path` (.npz). Scrittura sicura: prima un file
    accanto, poi lo scambio."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    st = export_state(run)
    data = {"kind": np.array("pile"), "format": np.array(FORMAT), "meta": np.array(json.dumps(meta_of(run)))}
    for part, prefix in (("world", "w_"), ("brain", "b_"), ("misc", "m_"), ("series", "s_")):
        data.update({prefix + k: np.asarray(v) for k, v in st[part].items()})
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)


def read(path) -> tuple[dict, dict]:
    """(meta, stato) da un salvataggio."""
    with np.load(path) as d:
        if "kind" not in d or str(d["kind"]) != "pile":
            raise ValueError(f"{path}: non è un salvataggio dell'arena delle pile")
        if int(d["format"]) > FORMAT:
            raise ValueError(f"{path}: formato {int(d['format'])}, più nuovo di questo codice ({FORMAT})")
        meta = json.loads(str(d["meta"]))
        state = {part: {} for part in ("world", "brain", "misc", "series")}
        for name in d.files:
            for part, prefix in (("world", "w_"), ("brain", "b_"), ("misc", "m_"), ("series", "s_")):
                if name.startswith(prefix):
                    state[part][name[len(prefix):]] = d[name]
    return meta, state


def load(path, engine: str = "gpu", cfg: PileConfig | None = None):
    """La corsa salvata in `path`, pronta a continuare (regole attuali)."""
    meta, state = read(path)
    cfg = cfg or PileConfig(beings=meta["beings"], good_by=meta.get("good_by", "colore"))
    run = make_run(meta["agent"], meta["phase"], meta["enclosures"], meta["seed"], tuple(meta["lesions"]), cfg,
                   engine)
    import_state(run, state)
    run.teacher_kind = meta.get("teacher", "fantasma")
    return run


def alone(source, engine: str = "gpu", seed: int = 1):
    """Ogni rete di `source` (una corsa o un salvataggio) da sola in un recinto
    suo, tutte insieme (fase A: chi muore rinasce con la stessa rete di
    partenza). Il cervello è quello di adesso, con ciò che ha imparato."""
    if isinstance(source, (str, Path)):
        meta, state = read(source)
    else:
        meta, state = meta_of(source), export_state(source)
    N = meta["enclosures"] * meta["beings"]
    cfg = PileConfig(beings=1, good_by=meta.get("good_by", "colore"))
    run = make_run(meta["agent"], "A", N, seed, tuple(meta["lesions"]), cfg, engine)
    # Ciò che riguarda il recinto di prima (chi si guardava, chi era appena
    # comparso) ricomincia; resta ciò che si è imparato (anche O). Da soli
    # non c'è nessun maestro.
    brain = {k: v for k, v in state["brain"].items()
             if not (k.startswith("att_") or k == "counts" or k in OBSERVING)}
    import_state(run, {"brain": brain}, parts=("brain",))
    run.origin = meta
    return run


def per_network(run) -> list[dict]:
    """Per ogni rete (in `alone`: un recinto ciascuna): pile buone e cattive,
    morti, quota di buone."""
    if isinstance(run, NumpyRun):
        by = run.by
    else:
        by = {k: np.asarray(run.state["misc"][k]) for k in BY}
    out = []
    for i in range(len(by["good_by"])):
        g, b = float(by["good_by"][i]), float(by["bad_by"][i])
        out.append({"rete": i, "buone": int(g), "cattive": int(b), "morti": int(by["deaths_by"][i]),
                    "scelta": round(g / (g + b), 3) if g + b else None})
    return out
