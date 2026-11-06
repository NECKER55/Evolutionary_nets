"""L'esame dell'identità delle reti pronte, su oggetti che nessuno ha mai visto.

Ogni oggetto si guarda da alcuni punti di vista, la memoria si salva su disco e si
ricarica: è "qualche giorno dopo". Poi si rivede ogni oggetto da altri punti di
vista, insieme a oggetti mai incontrati, e si conta quanti ne riassocia. Infine
oggetti sovrapposti e cambi di colore, visti e non visti.

Uso: `uv run --group percezione python -m evonets.brains.ragionante.exam_identity`.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from .ready import Eyes, ObjectMemory, Tracker, box_of

SIDE = 264
CAMERA = SimpleNamespace(camera_height=11.0, eye_range=350.0)
# Forme e colori mai visti, nessuno nel mondo delle pile: (tipo, misure in m, colore).
KNOWN = {"cubo blu": ("box", ".07 .07 .07", ".1 .2 .9"), "palla viola": ("sphere", ".08", ".6 .1 .8"),
         "capsula verde": ("capsule", ".04 .05", ".1 .7 .2"), "lastra rosa": ("box", ".1 .03 .08", ".95 .5 .7"),
         "uovo azzurro": ("ellipsoid", ".05 .07 .09", ".3 .8 .9"), "barile bianco": ("cylinder", ".06 .08", ".92 .92 .92")}
NEVER = {"cubo marrone": ("box", ".08 .05 .06", ".45 .25 .1"), "palla nera": ("sphere", ".06", ".08 .08 .08"),
         "capsula arancio": ("capsule", ".05 .03", "1 .5 .05")}
FIRST = (0.0, 40.0, -40.0)  # gradi: da dove si impara un oggetto
LATER = (90.0, 180.0, -130.0)  # da dove lo si rivede, giorni dopo


def scene(objects: list) -> mujoco.MjModel:
    """`objects`: [(tipo, misure, colore, x, y)], in m."""
    geoms = "".join(f'<geom name="o{k}" type="{t}" size="{s}" rgba="{c} 1" pos="{x} {y} .1"/>'
                    for k, (t, s, c, x, y) in enumerate(objects))
    return mujoco.MjModel.from_xml_string(f"""<mujoco><visual><global offwidth="264" offheight="264"/>
<headlight ambient=".45 .45 .45" diffuse=".25 .25 .25" specular="0 0 0"/><map znear=".0005" zfar="2"/></visual>
<statistic extent="10" center="0 0 0"/>
<asset><texture name="s" type="skybox" builtin="gradient" rgb1=".75 .85 .95" rgb2=".4 .5 .65" width="64" height="64"/>
<texture name="t" type="2d" builtin="checker" rgb1=".36 .4 .36" rgb2=".31 .35 .31" width="64" height="64"/>
<material name="m" texture="t" texrepeat="200 200"/></asset>
<worldbody><geom type="plane" size="80 80 .1" material="m"/>
<light pos="0 0 6" dir="0 0 -1" directional="true" diffuse=".5 .5 .5"/>
<camera name="eye" pos="0 0 .11" xyaxes="0 -1 0 0 0 1" fovy="120"/>{geoms}</worldbody></mujoco>""")


class Stage:
    """Un set: si guardano gli oggetti da dove si vuole."""

    def __init__(self, objects: list):
        self.model = scene(objects)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.r = mujoco.Renderer(self.model, SIDE, SIDE)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 0

    def look(self, target, angle: float, away: float = 0.7) -> tuple:
        """Dal punto a `away` m dal bersaglio, nella direzione `angle`, verso il bersaglio. -> (rgb, depth in cm)"""
        a = np.radians(angle)
        spot = np.array([*target, 0.11]) - away * np.array([np.cos(a), np.sin(a), 0.0])
        self.model.cam_pos[0] = spot
        right, up, back = (np.sin(a), -np.cos(a), 0.0), (0.0, 0.0, 1.0), (-np.cos(a), -np.sin(a), 0.0)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.array([right, up, back]).T.ravel())
        self.model.cam_quat[0] = q
        mujoco.mj_forward(self.model, self.data)
        self.r.update_scene(self.data, camera=self.cam)
        rgb = self.r.render().copy()
        self.r.enable_depth_rendering()
        depth = self.r.render().copy() * 100.0
        self.r.disable_depth_rendering()
        return rgb, depth

    def close(self) -> None:
        self.r.close()


def biggest(eyes, rgb, depth):
    """Il vettore dell'oggetto più centrale. -> (vettore, maschera) o (None, None)."""
    masks = eyes.masks(rgb, depth, CAMERA)
    if not masks:
        return None, None
    centre = [abs((box_of(m)[0] + box_of(m)[2]) / 2.0 - SIDE / 2.0) for m in masks]
    best = masks[int(np.argmin(centre))]
    return eyes.describe(rgb, [best])[0], best


def views(eyes, stage, position, angles) -> list:
    out = [biggest(eyes, *stage.look(position, a))[0] for a in angles]
    return [v for v in out if v is not None]


def reid(eyes, same: float) -> dict:
    """Riassocia dopo un salvataggio su disco? -> misure per soglia `same`."""
    stuff = list(KNOWN.items()) + list(NEVER.items())
    positions = [(float(k) * 8.0, 0.0) for k in range(len(stuff))]
    stage = Stage([(*v, *p) for (_, v), p in zip(stuff, positions)])
    memory, ids = ObjectMemory(same), {}
    for k, name in enumerate(KNOWN):
        first = views(eyes, stage, positions[k], FIRST)
        ids[name], _ = memory.know(first[0])
        for v in first[1:]:
            memory.add(v, ids[name])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "memoria.npz"
        memory.save(path)
        memory = ObjectMemory.load(path, same)
    right = wrong = forgot = 0
    for k, name in enumerate(KNOWN):
        for v in views(eyes, stage, positions[k], LATER):
            who, score = memory.closest(v)
            if score < same:
                forgot += 1
            elif who == ids[name]:
                right += 1
            else:
                wrong += 1
    new = old = 0
    for k, name in enumerate(NEVER, len(KNOWN)):
        for v in views(eyes, stage, positions[k], FIRST + LATER):
            new += memory.closest(v)[1] < same
            old += memory.closest(v)[1] >= same
    stage.close()
    return {"soglia": same, "riconosciute": right, "scambiate": wrong, "dimenticate": forgot,
            "nuove giuste": new, "nuove credute note": old}


def overlapping(eyes) -> dict:
    """Quanti oggetti trova, con due oggetti che si coprono in parte."""
    cases = {"colori diversi, si toccano": (("box", ".06 .06 .08", ".1 .2 .9", .7, 0), ("box", ".06 .06 .08", ".9 .1 .1", .7, -.13)),
             "uguali, si toccano": (("box", ".06 .06 .08", ".1 .2 .9", .7, 0), ("box", ".06 .06 .08", ".1 .2 .9", .7, -.125)),
             "uguali, uno dietro l'altro": (("box", ".06 .06 .08", ".1 .2 .9", .6, 0), ("box", ".06 .06 .08", ".1 .2 .9", .95, -.08)),
             "diversi, uno dietro l'altro": (("sphere", ".08", ".6 .1 .8", .6, 0), ("box", ".06 .06 .08", ".1 .7 .2", .95, -.08))}
    out = {}
    for name, objects in cases.items():
        stage = Stage(list(objects))
        rgb, depth = stage.look((0.7, 0.0), 0.0, 0.7)
        out[name] = len(eyes.masks(rgb, depth, CAMERA))
        stage.close()
    return out


def recolour(eyes, same: float) -> dict:
    """Un cubo blu che diventa arancione: sotto gli occhi, e fuori dalla vista."""
    stage = Stage([("box", ".07 .07 .07", ".1 .2 .9", 1.0, 0.0)])
    memory, tracker = ObjectMemory(same), Tracker()
    rgb, depth = stage.look((1.0, 0.0), 0.0)
    v, mask = biggest(eyes, rgb, depth)
    who, _ = memory.know(v)
    tracker.last = [(box_of(mask), who)]
    colours = np.linspace((.1, .2, .9), (1., .5, .05), 6)
    steps = []
    for c in colours[1:]:
        stage.model.geom_rgba[1, :3] = c
        rgb, depth = stage.look((1.0, 0.0), 0.0)
        v, mask = biggest(eyes, rgb, depth)
        box = box_of(mask)
        got = tracker.follow([box])[0]
        tracker.last = [(box, got)]
        steps.append(got == who)
    followed = all(steps)
    fresh = ObjectMemory(same)
    stage.model.geom_rgba[1, :3] = colours[0]
    start = biggest(eyes, *stage.look((1.0, 0.0), 0.0))[0]
    fresh.know(start)
    stage.model.geom_rgba[1, :3] = colours[-1]
    after = biggest(eyes, *stage.look((1.0, 0.0), 0.0))[0]
    score = float(fresh.closest(after)[1])
    stage.close()
    return {"visto cambiare: resta lo stesso": bool(followed), "non visto: somiglianza": round(score, 3),
            "non visto: riconosciuto": bool(score >= same)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--soglie", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    a = p.parse_args()
    eyes = Eyes()
    for same in a.soglie:
        print(json.dumps(reid(eyes, same), ensure_ascii=False), flush=True)
    print(json.dumps(overlapping(eyes), ensure_ascii=False), flush=True)
    print(json.dumps(recolour(eyes, a.soglie[-1]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
