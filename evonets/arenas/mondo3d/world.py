"""L'arena delle pile in 3D: le stesse regole di `PileWorld` (energia, morso,
zone, inversioni), che qui si ereditano, ma i corpi e le immagini li fa MuJoCo
(docs/SPEC_CERVELLO_RAGIONANTE.md, "Il mondo in 3D").

Ogni essere è un corpo su due ruote con una telecamera sul muso: immagine a
colori e profondità. Il telaio si inclina poco e non si ribalta: spinto contro
un muro, si incastrerebbe per sempre con le ruote a vuoto. Le pile sono cilindri che scivolano sul pavimento: un urto
le sposta, e nessun oggetto attraversa l'altro. Una pila assente non si vede e
non si tocca.

Un corpo (`BODIES`) è un cablaggio che il cervello non conosce: dalle sue uscite
a spinta, sterzo e morso. Il corpo sente che cosa ha fatto davvero (`felt`).

Le misure del mondo restano quelle di `PileConfig` (1 unità = 1 cm), la
fisica lavora in metri (`SCALE`). `pos`, `heading`, `speed` e `bpos` sono letti
dalla fisica a ogni passo. Con asse z verso l'alto, una rotta che cresce gira a
sinistra (nell'arena 2D, disegnata con la y verso il basso, a destra).
"""

import atexit
import os
import queue
import sys
import threading
import weakref
from concurrent.futures import Future
from dataclasses import dataclass

os.environ.setdefault("MUJOCO_GL", "egl")  # disegna senza finestre; va deciso prima di importare mujoco

import mujoco
import numpy as np

from ..pile.world import SECTORS, PileConfig, PileWorld, _wrap
from .bodies import BODIES

SCALE = 0.01  # metri per unità del mondo
RED_RGB, YELLOW_RGB = (0.92, 0.31, 0.27), (0.94, 0.82, 0.24)
PILE_RGB = (RED_RGB, YELLOW_RGB)
WALL_RGB = ((0.35, 0.78, 0.43), (0.67, 0.43, 0.90), (0.94, 0.59, 0.24), (0.90, 0.90, 0.92))  # nord, est, sud, ovest
BODY_RGB = (0.59, 0.78, 0.63)
MUZZLE_RGB = (0.35, 0.51, 1.0)


@dataclass
class Mondo3DConfig(PileConfig):
    """`PileConfig` più ciò che esiste solo in 3D."""

    beings: int = 1
    body: str = "spinta, sterzo, morso"  # un nome di `BODIES`
    image: int = 264  # lato dell'immagine della telecamera, in pixel
    physics_dt: float = 0.005  # s: passo della fisica (in un passo del mondo ce ne stanno più d'uno)
    wheel_radius: float = 3.0
    track: float = 14.0  # distanza tra le ruote
    wall_height: float = 30.0
    pile_height: float = 12.0
    pile_small: float = 0.6  # raggio di una pila piccola rispetto a una grande
    camera_height: float = 11.0
    pile_clear: float = 60.0  # una pila nuova nasce almeno così lontana dagli esseri: altrimenti è morsa appena nata
    pile_fade: float = 0.6  # s per comparire e per sparire: una pila non scatta dal nulla
    gap: float = 300.0  # spazio tra un recinto e l'altro


def _rgba(rgb, a=1.0) -> str:
    return f"{rgb[0]} {rgb[1]} {rgb[2]} {a}"


def _xml(cfg: Mondo3DConfig, B: int) -> str:
    """Il modello: tutti i recinti in fila nello stesso mondo. I muri li tengono
    separati, per i corpi e per le telecamere."""
    m = lambda x: x * SCALE
    size, r, wheel = m(cfg.size), m(cfg.radius), m(cfg.wheel_radius)
    wall_h, thick = m(cfg.wall_height), 0.05
    bodies = []
    for b in range(B):
        ox = b * m(cfg.size + cfg.gap)
        h = size / 2.0
        for w, (cx, cy, sx, sy) in enumerate(((h, -thick, h + 2 * thick, thick), (size + thick, h, thick, h + 2 * thick),
                                              (h, size + thick, h + 2 * thick, thick), (-thick, h, thick, h + 2 * thick))):
            bodies.append(f'<geom class="solid" type="box" pos="{ox + cx} {cy} {wall_h / 2}" '
                          f'size="{sx} {sy} {wall_h / 2}" rgba="{_rgba(WALL_RGB[w])}"/>')
        bodies.append(f'<geom name="floor{b}" type="box" class="ground" pos="{ox + h} {h} -0.05" '
                      f'size="{h} {h} 0.05" material="floor"/>')
        for k in range(cfg.batteries):
            bodies.append(f'''<body name="pile{b}_{k}" pos="{ox} 0 {m(cfg.pile_height) / 2}">
  <joint name="pile{b}_{k}x" type="slide" axis="1 0 0" damping="4"/><joint name="pile{b}_{k}y" type="slide" axis="0 1 0" damping="4"/>
  <geom name="pile{b}_{k}" class="solid" type="cylinder" size="{m(cfg.battery_radius)} {m(cfg.pile_height) / 2}" mass="0.3"/>
</body>''')
        for i in range(cfg.beings):
            n = f"{b}_{i}"
            track = m(cfg.track) / 2.0
            bodies.append(f'''<body name="being{n}" pos="0 0 {wheel + 0.002}">
  <joint name="being{n}x" type="slide" axis="1 0 0"/><joint name="being{n}y" type="slide" axis="0 1 0"/><joint name="being{n}z" type="slide" axis="0 0 1"/>
  <joint name="being{n}yaw" type="hinge" axis="0 0 1"/>
  <joint name="being{n}pitch" type="hinge" axis="0 1 0" range="-6 6" limited="true"/><joint name="being{n}roll" type="hinge" axis="1 0 0" range="-6 6" limited="true"/>
  <geom class="solid" type="cylinder" pos="0 0 {r * 0.45}" size="{r} {r * 0.5}" mass="1.5" rgba="{_rgba(BODY_RGB)}"/>
  <geom class="solid" type="box" pos="{r * 0.85} 0 {r}" size="{r * 0.2} {r * 0.5} {r * 0.15}" mass="0.01" rgba="{_rgba(MUZZLE_RGB)}"/>
  <geom class="slip" type="sphere" pos="{r * 0.75} 0 {-wheel + 0.01}" size="0.01"/>
  <geom class="slip" type="sphere" pos="{-r * 0.75} 0 {-wheel + 0.01}" size="0.01"/>
  <camera name="eye{n}" pos="{r + 0.012} 0 {m(cfg.camera_height) - wheel}" xyaxes="0 -1 0 0 0 1" fovy="{np.degrees(SECTORS[-1]) * 2.0}"/>
  <body name="left{n}" pos="0 {track} 0"><joint name="left{n}" type="hinge" axis="0 1 0" damping="0.001" armature="0.002"/><geom class="tyre" type="sphere" size="{wheel}" mass="0.1"/></body>
  <body name="right{n}" pos="0 {-track} 0"><joint name="right{n}" type="hinge" axis="0 1 0" damping="0.001" armature="0.002"/><geom class="tyre" type="sphere" size="{wheel}" mass="0.1"/></body>
</body>''')
    actuators = "".join(f'<velocity name="{side}{b}_{i}" joint="{side}{b}_{i}" kv="0.15" ctrlrange="-100 100" forcerange="-1.0 1.0"/>'
                        for b in range(B) for i in range(cfg.beings) for side in ("left", "right"))
    return f'''<mujoco>
<compiler angle="degree"/>
<option timestep="{cfg.physics_dt}" integrator="implicitfast"/>
<size memory="-1"/>
<statistic extent="{size * 2}" center="{size / 2} {size / 2} 0"/>
<visual><global offwidth="1024" offheight="768"/><quality shadowsize="0" offsamples="0"/>
  <map znear="0.0005" zfar="2"/>
  <headlight ambient="0.45 0.45 0.45" diffuse="0.25 0.25 0.25" specular="0 0 0"/></visual>
<asset>
  <texture name="sky" type="skybox" builtin="gradient" rgb1="0.75 0.85 0.95" rgb2="0.4 0.5 0.65" width="64" height="64"/>
  <texture name="tiles" type="2d" builtin="checker" rgb1="0.36 0.40 0.36" rgb2="0.31 0.35 0.31" width="64" height="64"/>
  <material name="floor" texture="tiles" texrepeat="{cfg.size / 50.0} {cfg.size / 50.0}" texuniform="false"/>
</asset>
<default>
  <default class="solid"><geom contype="1" conaffinity="1" friction="0.3"/></default>
  <default class="ground"><geom contype="0" conaffinity="2" friction="1"/></default>
  <default class="tyre"><geom contype="2" conaffinity="0" friction="1.5" rgba="0.1 0.1 0.1 1"/></default>
  <default class="slip"><geom contype="2" conaffinity="0" condim="1" priority="1" mass="0.01" rgba="0.1 0.1 0.1 1"/></default>
</default>
<worldbody>
<light pos="{size / 2} {size / 2} 6" dir="0 0 -1" directional="true" castshadow="false" diffuse="0.5 0.5 0.5"/>
{"".join(bodies)}
</worldbody>
<actuator>{actuators}</actuator>
</mujoco>'''


class Painter:
    """Un thread solo per disegnare: quello principale ha già il contesto
    grafico di SDL, e un thread non può averne due."""

    def __init__(self):
        self.jobs = queue.SimpleQueue()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            done, fn, args = self.jobs.get()
            try:
                done.set_result(fn(*args))
            except BaseException as err:
                done.set_exception(err)

    def call(self, fn, *args):
        done = Future()
        self.jobs.put((done, fn, args))
        return done.result()


class Mondo3D(PileWorld):
    """`enclosures` recinti con `cfg.beings` esseri ciascuno: stessa interfaccia
    di `PileWorld`, più `view` (ciò che vede un essere) e `overview`."""

    def __init__(self, cfg: Mondo3DConfig, enclosures: int = 1, seed: int = 0):
        super().__init__(cfg, enclosures, seed)
        self.substeps = max(1, round(cfg.dt / cfg.physics_dt))
        self.model = mujoco.MjModel.from_xml_string(_xml(cfg, enclosures))
        self.data = mujoco.MjData(self.model)
        B, n, K = self.B, self.n, cfg.batteries
        m, name = self.model, lambda kind, x: mujoco.mj_name2id(self.model, kind, x)
        self.origin = np.stack([np.arange(B) * (cfg.size + cfg.gap), np.zeros(B)], -1)
        adr = lambda j: m.jnt_qposadr[name(mujoco.mjtObj.mjOBJ_JOINT, j)]
        vel = lambda j: m.jnt_dofadr[name(mujoco.mjtObj.mjOBJ_JOINT, j)]
        idx = lambda f: np.array([[f(b, k) for k in range(K)] for b in range(B)])
        self.px, self.py = idx(lambda b, k: adr(f"pile{b}_{k}x")), idx(lambda b, k: adr(f"pile{b}_{k}y"))
        self.pvx, self.pvy = idx(lambda b, k: vel(f"pile{b}_{k}x")), idx(lambda b, k: vel(f"pile{b}_{k}y"))
        self.pgeom = idx(lambda b, k: name(mujoco.mjtObj.mjOBJ_GEOM, f"pile{b}_{k}"))
        each = lambda f: np.array([[f(f"{b}_{i}") for i in range(n)] for b in range(B)])
        self.qadr = each(lambda s: adr(f"being{s}x"))  # x, y, z, rotta, beccheggio, rollio: consecutivi
        self.vadr = each(lambda s: vel(f"being{s}x"))
        self.left = each(lambda s: name(mujoco.mjtObj.mjOBJ_ACTUATOR, f"left{s}"))
        self.right = each(lambda s: name(mujoco.mjtObj.mjOBJ_ACTUATOR, f"right{s}"))
        self.cam = each(lambda s: name(mujoco.mjtObj.mjOBJ_CAMERA, f"eye{s}"))
        self.wheel_dofs = each(lambda s: np.array([vel(f"left{s}"), vel(f"right{s}")]))
        self.wheel_q = each(lambda s: np.array([adr(f"left{s}"), adr(f"right{s}")]))
        self.shown = np.zeros((B, K), dtype=bool)  # le pile che la fisica ha adesso
        self.shown_pos = np.zeros((B, K, 2))
        self.bon_was = np.zeros((B, K), dtype=bool)
        self._hide(np.ones((B, K), dtype=bool))
        self._place(np.ones((B, n), dtype=bool))
        mujoco.mj_forward(self.model, self.data)
        self._renderers = {}
        self._gl = Painter()
        self.spin = np.zeros((B, n))  # rotazione dell'ultimo passo, in quote di quella massima
        self.jaw = np.zeros((B, n))
        self.born = np.zeros((B, K))  # l'età di ogni pila quando è comparsa
        self.set_body(cfg.body)

    # --- Dalle regole alla fisica e ritorno ----------------------------------------

    def _place(self, mask) -> None:
        """Porta i corpi `mask` (B, n) dove dicono `pos` e `heading`, fermi."""
        d = self.data
        for b, i in zip(*np.nonzero(mask)):
            q = self.qadr[b, i]
            xy = (self.pos[b, i] + self.origin[b]) * SCALE
            d.qpos[q:q + 6] = [xy[0], xy[1], 0.0, self.heading[b, i], 0.0, 0.0]
            d.qvel[self.vadr[b, i]:self.vadr[b, i] + 6] = 0.0
            d.qvel[self.wheel_dofs[b, i]] = 0.0
            d.qpos[self.wheel_q[b, i]] = 0.0
            d.ctrl[[self.left[b, i], self.right[b, i]]] = 0.0

    def _spawn(self, mask):
        """Come nell'arena delle pile, ma in 3D i corpi occupano spazio: una
        pila nuova cerca un posto libero (dopo qualche tentativo si accontenta)."""
        super()._spawn(mask)
        cfg = self.cfg
        lo, hi = cfg.battery_radius, cfg.size - cfg.battery_radius
        for b, k in zip(*np.nonzero(mask)):
            for _ in range(12):
                others = self.bon[b].copy()
                others[k] = False
                far = np.hypot(*(self.pos[b] - self.bpos[b, k]).T).min() >= cfg.pile_clear
                if far and np.hypot(*(self.bpos[b][others] - self.bpos[b, k]).T).min(initial=np.inf) >= 3.0 * cfg.battery_radius:
                    break
                centre = self.zone_centres()[self.zone[b]]
                self.bpos[b, k] = np.clip(centre + self.rng.uniform(-cfg.zone_half, cfg.zone_half, 2), lo, hi)
        self.born[mask] = self.bage[mask]

    def _hide(self, mask) -> None:
        g = self.pgeom[mask]
        self.model.geom_rgba[g, 3] = 0.0
        self.model.geom_contype[g] = 0
        self.model.geom_conaffinity[g] = 0
        self.shown[mask] = False

    def _show(self, mask) -> None:
        """Le pile `mask` (B, K) compaiono dove dice `bpos`, col loro aspetto."""
        cfg, m, d = self.cfg, self.model, self.data
        for b, k in zip(*np.nonzero(mask)):
            g = self.pgeom[b, k]
            radius = cfg.battery_radius * SCALE * (1.0 if self.bbig[b, k] else cfg.pile_small)
            m.geom_size[g, 0] = radius
            m.geom_rbound[g] = float(np.hypot(radius, m.geom_size[g, 1]))
            m.geom_rgba[g] = (*PILE_RGB[self.bcolor[b, k]], 1.0)
            m.geom_contype[g] = 1
            m.geom_conaffinity[g] = 1
            d.qpos[[self.px[b, k], self.py[b, k]]] = (self.bpos[b, k] + self.origin[b]) * SCALE
            d.qvel[[self.pvx[b, k], self.pvy[b, k]]] = 0.0
        self.shown[mask] = True
        self.shown_pos[mask] = self.bpos[mask]

    def sync(self) -> None:
        """Dopo aver cambiato a mano `pos`, `heading` o le pile (solo per le prove)."""
        self._place(np.ones((self.B, self.n), dtype=bool))
        self._hide(~self.bon)
        self._show(self.bon)
        mujoco.mj_forward(self.model, self.data)

    def _move(self, thrust, steer):
        cfg, m, d = self.cfg, self.model, self.data
        # Spinta e sterzo diventano la velocità delle due ruote.
        ahead = thrust * cfg.speed * np.where(thrust < 0, cfg.reverse, 1.0)
        spin = np.clip(steer, -1.0, 1.0) * cfg.turn * 0.5 * cfg.track
        d.ctrl[self.left] = (ahead - spin) / cfg.wheel_radius
        d.ctrl[self.right] = (ahead + spin) / cfg.wheel_radius
        mujoco.mj_step(m, d, self.substeps)
        self._read()

    def _read(self) -> None:
        """Ciò che le regole devono sapere, letto dalla fisica."""
        d = self.data
        q, v = self.qadr, self.vadr
        self.pos = np.stack([d.qpos[q], d.qpos[q + 1]], -1) / SCALE - self.origin[:, None]
        before, self.heading = self.heading, _wrap(d.qpos[q + 3])
        self.spin = _wrap(self.heading - before) / (self.cfg.turn * self.cfg.dt)
        self.speed = (d.qvel[v] * np.cos(self.heading) + d.qvel[v + 1] * np.sin(self.heading)) / SCALE
        here = self.shown
        self.bpos[here] = (np.stack([d.qpos[self.px[here]], d.qpos[self.py[here]]], -1)) / SCALE - \
            self.origin[np.nonzero(here)[0]]
        self.shown_pos[here] = self.bpos[here]

    def step(self, thrust, steer, bite=None) -> dict:
        out = super().step(thrust, steer, bite)
        self._hide(self.shown & ~self.bon)
        moved = self.bon & (~self.shown | (self.bpos != self.shown_pos).any(-1))
        self._show(moved)
        fade = self.cfg.pile_fade
        alpha = np.minimum(self.bage - self.born, self.cfg.battery_life - self.bage) / fade
        self.model.geom_rgba[self.pgeom[self.bon], 3] = np.clip(alpha[self.bon], 0.05, 1.0)
        return out

    def reborn(self, mask):
        super().reborn(mask)
        self._place(mask)

    # --- Il corpo --------------------------------------------------------------------

    def set_body(self, name: str) -> None:
        """Cambia il cablaggio, anche a metà vita: il cervello non viene avvisato."""
        self.body = name
        wiring, self.outputs = BODIES[name]
        self.wiring = np.array(wiring, dtype=float)

    def wires(self, u):
        """Dalle uscite `u` (B, n, K) ai comandi di `step`: (spinta, sterzo, morso)."""
        thrust, steer, jaw = np.moveaxis(u @ self.wiring.T, -1, 0)
        self.jaw = np.clip(jaw, 0.0, 1.0)
        return np.clip(thrust, -1.0, 1.0), np.clip(steer, -1.0, 1.0), jaw > 0.5

    def felt(self) -> np.ndarray:
        """Che cosa ha fatto il corpo nell'ultimo passo (B, n, 3): vedi `FELT`."""
        return np.stack([self.speed / self.cfg.speed, self.spin, self.jaw], -1)

    # --- Le immagini -----------------------------------------------------------------

    def _renderer(self, h: int, w: int) -> mujoco.Renderer:
        if (h, w) not in self._renderers:
            self._renderers[h, w] = mujoco.Renderer(self.model, h, w)
            me = weakref.ref(self)  # registrato dopo MuJoCo, quindi all'uscita chiude prima di lui
            atexit.register(lambda: me() and me().close())
        return self._renderers[h, w]

    def _view(self, b, i, rgb, depth) -> dict:
        r = self._renderer(self.cfg.image, self.cfg.image)
        out = {}
        r.update_scene(self.data, camera=int(self.cam[b, i]))
        if rgb:
            out["rgb"] = r.render().copy()
        if depth:
            r.enable_depth_rendering()
            out["depth"] = r.render().copy() / SCALE
            r.disable_depth_rendering()
        return out

    def view(self, b: int, i: int, rgb: bool = True, depth: bool = True) -> dict:
        """Ciò che vede l'essere `i` del recinto `b`: "rgb" (H, W, 3) in byte e
        "depth" (H, W), la distanza in unità del mondo."""
        return self._gl.call(self._view, b, i, rgb, depth)

    def _marks(self, scene, b, follow) -> None:
        """Sul pavimento della vista da fuori: dischi verdi sotto le pile buone,
        rossi sotto le cattive, la zona attiva e l'essere seguito."""
        cfg = self.cfg

        def add(kind, xy, size, rgba):
            g = scene.geoms[scene.ngeom]
            pos = [*((xy + self.origin[b]) * SCALE), 0.003]
            mujoco.mjv_initGeom(g, kind, np.array(size) * SCALE, np.array(pos), np.eye(3).ravel(), np.array(rgba))
            scene.ngeom += 1

        good = self.battery_good()[b]
        for k in np.flatnonzero(self.bon[b]):
            colour = (0.2, 0.9, 0.3, 0.9) if good[k] else (0.9, 0.1, 0.1, 0.9)
            add(mujoco.mjtGeom.mjGEOM_CYLINDER, self.bpos[b, k], (1.7 * cfg.battery_radius, 0.0, 0.2), colour)
        line = (0.95, 0.95, 0.95, 0.6)
        c, h = self.zone_centres()[self.zone[b]], cfg.zone_half
        for dx, dy, sx, sy in ((0, -h, h, 1.5), (0, h, h, 1.5), (-h, 0, 1.5, h), (h, 0, 1.5, h)):
            add(mujoco.mjtGeom.mjGEOM_BOX, c + (dx, dy), (sx, sy, 0.2), line)
        if follow is not None:
            add(mujoco.mjtGeom.mjGEOM_CYLINDER, self.pos[b, follow], (1.6 * cfg.radius, 0.0, 0.2), (0.3, 0.9, 1.0, 0.9))

    def _overview(self, b, h, w, follow, chase) -> np.ndarray:
        cfg, r = self.cfg, self._renderer(h, w)
        cam = mujoco.MjvCamera()
        if not chase:
            c = (self.origin[b] + cfg.size / 2.0) * SCALE
            cam.lookat[:] = c[0], c[1], 0.0
            cam.distance, cam.elevation, cam.azimuth = 1.25 * cfg.size * SCALE, -90.0, 90.0
        else:
            c = (self.pos[b, follow] + self.origin[b]) * SCALE
            cam.lookat[:] = c[0], c[1], 0.1
            cam.distance, cam.elevation = 1.1, -28.0
            cam.azimuth = float(np.degrees(self.heading[b, follow]))
        r.update_scene(self.data, camera=cam)
        self._marks(r.scene, b, follow)
        return r.render().copy()

    def overview(self, b: int, h: int = 500, w: int = 500, follow: int | None = None, chase: bool = False) -> np.ndarray:
        """Il recinto `b` dall'alto, oppure (`chase`) l'essere `follow` visto da
        dietro, con i segni di `_marks`. Solo per noi: nessun essere riceve questa
        immagine, e nessun essere sa quali pile sono buone."""
        return self._gl.call(self._overview, b, h, w, follow, chase)

    def _free(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers = {}

    def close(self) -> None:
        if self._gl is not None:
            if self._renderers:
                self._gl.call(self._free)
            self._gl = None

    def __del__(self):
        # Se l'interprete sta chiudendo il thread è già fermo: ci pensa `atexit`.
        if getattr(self, "_gl", None) is not None and not sys.is_finalizing():
            self.close()
