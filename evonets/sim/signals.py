"""La fisica dei segnali: cosa arriva in un punto del mondo, per ogni fenomeno.

Qui sta solo la fisica, uguale per tutti: quanta luce di ogni oggetto arriva
lungo una direzione, quanta sostanza c'è in un punto, quale pressione, quali
vibrazioni, che temperatura, che campo magnetico. Cosa un recettore ne fa
(a cosa è sensibile, con quale sintonia) sta in `net/receptors.py`.

Ogni funzione lavora su K punti insieme (una riga per recettore di ogni
agente) e riceve `agent` (K,): a quale agente appartiene ogni punto.

Fenomeni e assi della qualità (vedi docs/SPEC_FASE_C.md, sezione 2):
- luce: colore. Mezzo torbido, attenuazione exp(-d / light_attenuation).
  Pareti e bersaglio sono opachi, le nubi di veleno brillano e non nascondono.
- sostanze: molecola. Tossina del veleno e odore del bersaglio.
- meccanica: frequenza. Pressione da contatto (frequenza 0) e vibrazioni.
- temperatura: la temperatura stessa.
- magnetismo: nessun asse, solo direzione e intensità.
- interno: quale grandezza del corpo (vedi BODY_QUANTITIES).
"""

import numpy as np
from numba import njit, prange

LIGHT, CHEMICAL, MECHANICAL, THERMAL, MAGNETIC, INTERNAL = range(6)
PHENOMENA = ("luce", "sostanze", "meccanica", "temperatura", "magnetismo", "interno")

# Grandezze del corpo sull'asse dell'interno: (posizione sull'asse, nome). Le
# grandezze con segno sono divise in due metà, perché un recettore dà solo
# valori positivi. Il movimento sta vicino, l'energia lontano.
BODY_QUANTITIES = (
    (0.05, "energia"),
    (0.35, "avanti"),
    (0.45, "indietro"),
    (0.60, "derapata destra"),
    (0.70, "derapata sinistra"),
    (0.85, "rotazione destra"),
    (0.95, "rotazione sinistra"),
)


@njit(cache=True, parallel=True)
def _light_kernel(origins, angles, arena_w, arena_h, L, target, target_r, hz_pos, hz_r):
    """Il cuore di `light`, compilato: un ciclo per recettore, direzione
    campione e oggetto. Ogni recettore è indipendente (risultato identico in
    parallelo)."""
    k_all, m = angles.shape
    wall = np.zeros(k_all)
    targ = np.zeros(k_all)
    glow = np.zeros(k_all)
    for k in prange(k_all):
        ox, oy = origins[k, 0], origins[k, 1]
        for j in range(m):
            dx, dy = np.cos(angles[k, j]), np.sin(angles[k, j])
            # Pareti: c'è sempre una parete davanti.
            tx = ((arena_w if dx > 0.0 else 0.0) - ox) / dx if dx != 0.0 else np.inf
            ty = ((arena_h if dy > 0.0 else 0.0) - oy) / dy if dy != 0.0 else np.inf
            d_hit = max(min(tx, ty), 0.0)
            hit_target = False
            if target_r > 0.0:
                fx, fy = target[0] - ox, target[1] - oy
                proj = fx * dx + fy * dy
                disc = target_r * target_r - (fx * fx + fy * fy) + proj * proj
                if disc >= 0.0:
                    half = np.sqrt(disc)
                    if proj + half >= 0.0:
                        d_t = max(proj - half, 0.0)
                        if d_t < d_hit:
                            d_hit, hit_target = d_t, True
            fade = np.exp(-d_hit / L)
            if hit_target:
                targ[k] += fade
            else:
                wall[k] += fade
            # Nubi: luce lungo il tratto attraversato, fino al primo opaco.
            for h in range(hz_pos.shape[0]):
                fx, fy = hz_pos[h, 0] - ox, hz_pos[h, 1] - oy
                proj = fx * dx + fy * dy
                disc = hz_r[h] * hz_r[h] - (fx * fx + fy * fy) + proj * proj
                if disc < 0.0:
                    continue
                half = np.sqrt(disc)
                s_in = min(max(proj - half, 0.0), d_hit)
                s_out = min(proj + half, d_hit)
                if s_out > s_in:
                    glow[k] += L * (np.exp(-s_in / L) - np.exp(-s_out / L))
        wall[k] /= m
        targ[k] /= m
        glow[k] /= m
    return wall, targ, glow


def light(cfg, world, origins: np.ndarray, angles: np.ndarray) -> dict[str, np.ndarray]:
    """Quanta luce di ogni tipo di oggetto arriva a K recettori, mediata sulle
    loro direzioni campione. origins (K, 2), angles (K, M) assoluti.

    Restituisce {"parete", "bersaglio", "veleno"} -> (K,). Il valore è già
    attenuato dal mezzo ma non ancora moltiplicato per lo spettro dell'oggetto
    (lo fa il recettore con la sua sintonia): per gli opachi è exp(-d/L) se
    il raggio li colpisce per primi, per il veleno l'integrale della sua
    luce lungo il tratto attraversato, per unità di luminosità."""
    task = world.task
    has_target = hasattr(task, "target")
    target = np.asarray(task.target, dtype=float) if has_target else np.zeros(2)
    hz = world.hazards
    wall, targ, glow = _light_kernel(
        np.ascontiguousarray(origins, dtype=float), np.ascontiguousarray(angles, dtype=float),
        float(cfg.arena_w), float(cfg.arena_h), float(cfg.light_attenuation),
        target, float(task.radius) if has_target else 0.0,
        np.ascontiguousarray(hz.pos, dtype=float).reshape(-1, 2), np.ascontiguousarray(hz.radius, dtype=float))
    return {"parete": wall, "bersaglio": targ, "veleno": glow}


@njit(cache=True, parallel=True)
def _toxin_kernel(points, hz_pos, hz_r, falloff, leak, smell_range):
    """Tossina nei punti: dentro una zona (profondità ** falloff), come
    `HazardField.intensity`; fuori, `leak · exp(-distanza dal bordo / smell_range)`."""
    out = np.zeros(points.shape[0])
    for k in prange(points.shape[0]):
        for h in range(hz_pos.shape[0]):
            dx, dy = points[k, 0] - hz_pos[h, 0], points[k, 1] - hz_pos[h, 1]
            edge = np.sqrt(dx * dx + dy * dy) - hz_r[h]
            if edge < 0.0:
                out[k] += min(-edge / hz_r[h], 1.0) ** falloff
            elif leak > 0.0:
                out[k] += leak * np.exp(-edge / smell_range)
    return out


def chemicals(cfg, world, points: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Concentrazioni nei punti (K, 2): [(molecola, concentrazione (K,))].
    Dentro una zona di veleno la tossina è l'intensità del danno; fuori, se
    `hazard_smell` > 0, un odore che cala con la distanza dal bordo."""
    hz = world.hazards
    tox = _toxin_kernel(np.ascontiguousarray(points, dtype=float), np.ascontiguousarray(hz.pos, dtype=float).reshape(-1, 2),
                        np.ascontiguousarray(hz.radius, dtype=float), float(cfg.hazard_falloff),
                        float(cfg.hazard_smell), float(cfg.smell_range))
    out = [(cfg.toxin_molecule, tox)]
    task = world.task
    if cfg.target_smell > 0.0 and hasattr(task, "target"):
        d = np.linalg.norm(points - task.target, axis=1) - task.radius
        out.append((cfg.target_molecule, cfg.target_smell * np.exp(-np.maximum(d, 0.0) / cfg.smell_range)))
    return out


def pressure(cfg, world, agent: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Pressione (K,) sui recettori la cui normale (K, 2), nel riferimento del
    mondo, guarda una parete che l'agente sta toccando."""
    r = cfg.agent_radius
    pos = world.pos[agent]
    lo, hi = pos <= r, pos >= np.array([cfg.arena_w - r, cfg.arena_h - r])
    # Pareti toccate: sinistra, destra (asse x), in basso, in alto (asse y).
    return (lo[:, 0] * np.maximum(0.0, -normals[:, 0]) + hi[:, 0] * np.maximum(0.0, normals[:, 0])
            + lo[:, 1] * np.maximum(0.0, -normals[:, 1]) + hi[:, 1] * np.maximum(0.0, normals[:, 1]))


def vibrations(cfg, world, points: np.ndarray) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Vibrazioni che arrivano nei punti (K, 2): [(frequenza, ampiezza (K,),
    direzione di provenienza (K,) in radianti)]. Solo le sorgenti accese."""
    out = []
    sources = []
    task = world.task
    if cfg.target_vibration > 0.0 and hasattr(task, "target"):
        sources.append((cfg.target_vibration_freq, cfg.target_vibration, task.target[None, :]))
    if cfg.hazard_vibration > 0.0 and len(world.hazards.pos):
        sources.append((cfg.hazard_vibration_freq, cfg.hazard_vibration, world.hazards.pos))
    for freq, amp, centers in sources:
        for c in centers:
            v = c - points
            d = np.linalg.norm(v, axis=1)
            out.append((freq, amp * np.exp(-d / cfg.vibration_range), np.arctan2(v[:, 1], v[:, 0])))
    return out


def temperature(cfg, world, points: np.ndarray) -> np.ndarray:
    """Temperatura (K,) in [0, 1]: ambiente più le sorgenti di calore."""
    t = np.full(len(points), cfg.ambient_temperature)
    task = world.task
    if cfg.target_heat > 0.0 and hasattr(task, "target"):
        d = np.linalg.norm(points - task.target, axis=1) - task.radius
        t += cfg.target_heat * np.exp(-np.maximum(d, 0.0) / cfg.heat_range)
    hz = world.hazards
    if cfg.hazard_heat > 0.0 and len(hz.pos):
        d = np.maximum(hz._edge_distance(points), 0.0)
        t += cfg.hazard_heat * np.exp(-d / cfg.heat_range).sum(axis=1)
    return np.clip(t, 0.0, 1.0)


def magnetic(cfg) -> np.ndarray:
    """Il campo magnetico (2,), uniforme."""
    a = np.radians(cfg.magnetic_angle_deg)
    return cfg.magnetic_field * np.array([np.cos(a), np.sin(a)])


def body(cfg, world, rows: np.ndarray) -> np.ndarray:
    """Le grandezze del corpo (n, 7) in [0, 1], nell'ordine di BODY_QUANTITIES."""
    heading = world.heading[rows]
    cos, sin = np.cos(heading), np.sin(heading)
    vel = world.vel[rows]
    v_max = cfg.thrust_accel / cfg.lin_drag
    w_max = cfg.turn_accel / cfg.ang_drag
    fwd = (vel[:, 0] * cos + vel[:, 1] * sin) / v_max
    lat = (vel[:, 1] * cos - vel[:, 0] * sin) / v_max
    rot = world.ang_vel[rows] / w_max

    def pos(x):
        return np.clip(x, 0.0, 1.0)

    return np.stack([np.clip(world.energy[rows] / cfg.energy_init, 0.0, 1.0),
                     pos(fwd), pos(-fwd), pos(lat), pos(-lat), pos(rot), pos(-rot)], axis=1)
