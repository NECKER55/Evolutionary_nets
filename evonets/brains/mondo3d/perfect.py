"""L'agente perfetto: un controllo con tutte le risposte, non un cervello."""

import numpy as np

from ...arenas.pile.world import _wrap

AVOID = 50.0  # cm: da qui devia per non urtare una pila cattiva
PATIENCE = 2.0  # s accanto a una pila senza riuscire a morderla (una cattiva è in mezzo)
LEAVE = 4.0  # s poi per quanto la si lascia stare
ROUTE = 200.0  # cm da aggiungere a una tappa: non è la meta, non ci si ferma


class Perfect:
    """Legge la verità dal simulatore: dove sono tutte le pile, quali sono
    buone adesso, dov'è la zona. Non usa la telecamera. Ognuno va alla pila buona
    più vicina che nessun compagno ha già preso di mira; se non ce ne sono
    aspetta nella zona. Con le stanze conosce anche la strada: passa dalle porte. Se una cattiva gli impedisce di mordere senza rischi,
    cambia pila. È un limite per i confronti: la sua scelta è sempre
    buona, ma la strada è avida, quindi non è il massimo teorico di pile al minuto."""

    def __init__(self, world, seed: int = 0):
        self.w = world
        self.target = np.full((world.B, world.n), -1)
        self.blocked = np.zeros((world.B, world.n))  # s passati accanto alla meta senza morderla
        self.left = np.zeros((world.B, world.cfg.batteries))  # s prima di riprovare una pila lasciata
        self.far = np.zeros((world.B, world.n))  # quanto dista la meta
        self.stop = np.zeros((world.B, world.n))  # a che distanza dalla meta ci si ferma

    def _assign(self, good, dist) -> None:
        """A ogni essere libero la sua pila buona, dalle coppie più vicine."""
        self.target[:] = -1
        for b in range(self.w.B):
            taken, busy = set(), self.w.charging[b] > 0.0
            pairs = np.argsort(np.where(good[b][None, :], dist[b], np.inf), axis=None)
            for i, k in zip(*np.unravel_index(pairs, dist[b].shape)):
                if not good[b, k]:
                    break
                if not busy[i] and self.target[b, i] < 0 and k not in taken:
                    self.target[b, i] = k
                    taken.add(k)

    def act(self):
        w, cfg = self.w, self.w.cfg
        B, n = w.B, w.n
        self.left = np.maximum(self.left - cfg.dt, 0.0)
        good, bad = w.bon & w.battery_good(), w.bon & ~w.battery_good()
        rel = w.bpos[:, None] - w.pos[:, :, None]  # (B, n, K, 2)
        dist = np.hypot(rel[..., 0], rel[..., 1])
        self._assign(good & (self.left <= 0.0), dist)
        has = self.target >= 0
        b, i = np.indices((B, n))
        k = np.maximum(self.target, 0)
        goal = np.where(has[..., None], w.bpos[b, k], w.zone_centres()[w.zone][:, None]).copy()
        routed = np.zeros((B, n), dtype=bool)
        for x, y in np.ndindex(B, n):
            goal[x, y], arrived = w.layouts[x].way(w.pos[x, y], goal[x, y])
            routed[x, y] = not arrived
        to = goal - w.pos
        far = np.hypot(to[..., 0], to[..., 1]) + routed * ROUTE
        way = to / np.maximum(far, 1e-6)[..., None]
        reach = cfg.radius + cfg.battery_radius + cfg.bite_reach
        # Le cattive vicine e davanti spostano di lato, mai indietro: la meta resta
        # raggiungibile. Accanto alla meta niente deviazioni: bisogna guardarla.
        ahead = (rel * way[:, :, None]).sum(-1)
        side = rel - ahead[..., None] * way[:, :, None]
        push = np.clip(1.0 - dist / AVOID, 0.0, 1.0) * bad[:, None] * ((ahead > 0.0) & (ahead < far[..., None])) \
            * (far > reach + 8.0)[..., None]
        way = way - 2.0 * (side / np.maximum(dist, 1e-6)[..., None] * push[..., None]).sum(2)
        ang = _wrap(np.arctan2(way[..., 1], way[..., 0]) - w.heading)
        stop = np.where(has, reach - 4.0, cfg.zone_half / 2.0)
        self.far, self.stop = far, stop
        thrust = np.clip((far - stop) / 50.0, 0.0, 1.0) * np.clip(1.0 - np.abs(ang) / 0.8, 0.0, 1.0)
        steer = np.clip(3.0 * ang, -1.0, 1.0)
        # Il morso prende la prima pila a portata: mai se ce n'è una cattiva.
        front = np.abs(_wrap(np.arctan2(rel[..., 1], rel[..., 0]) - w.heading[..., None])) < np.radians(cfg.mouth)
        near_bad = ((dist < reach) & front & bad[:, None]).any(-1)
        aim = _wrap(np.arctan2(to[..., 1], to[..., 0]) - w.heading)
        ready = has & (far < reach) & (np.abs(aim) < np.radians(cfg.mouth)) & ~near_bad
        bite = ready & (np.abs(w.speed) < cfg.take_speed * cfg.speed)
        self.blocked = np.where(has & (far < reach + 8.0) & ~bite, self.blocked + cfg.dt, 0.0)
        give_up = self.blocked > PATIENCE
        self.left[b[give_up], k[give_up]] = LEAVE
        return thrust, steer, bite
