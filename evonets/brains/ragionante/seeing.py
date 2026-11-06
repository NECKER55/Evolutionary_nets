"""Un comandante finto che ha solo gli occhi: guida con ciò che dà la percezione,
non con la verità. Dal simulatore legge una cosa sola, quale colore è buono adesso:
quello un giorno lo dovrà imparare il cervello. Le regole sono scritte a mano."""

import numpy as np

from ...arenas.pile.world import BUMP

AVOID = 50.0  # cm: da qui devia per non urtare una pila cattiva
PATIENCE = 2.0  # s accanto a una pila senza riuscire a morderla
OPINION = 6.0  # s per cui, con `mistakes`, resta convinto di una stessa cosa
LEAVE = 3.0  # s in cui poi se ne va a cercare altro
MEMORY = 0.5  # s per cui ricorda una pila che ha smesso di vedere
ESCAPE = 1.5  # s di marcia indietro girando, dopo un urto mentre cerca


class Seeing:
    def __init__(self, world, eyes, seed: int = 0, mistakes: float = 0.0):
        """`mistakes`: quota di tempo in cui crede buono il colore sbagliato. Serve a chi
        deve registrare anche il dolore: senza errori non si morde mai una pila cattiva."""
        if world.cfg.good_by != "colore":
            raise ValueError("per ora il vedente sa solo quale colore è buono")
        self.w, self.eyes, self.mistakes = world, eyes, mistakes
        self.rng = np.random.default_rng([seed, 11])
        s = (world.B, world.n)
        self.seen = None  # l'ultimo `Percept`, anche per lo schermo
        self.blocked, self.away = np.zeros(s), np.zeros(s)
        self.ang, self.far, self.ago = np.zeros(s), np.zeros(s), np.full(s, np.inf)  # la pila di mira
        self.turn = self.rng.choice((-1.0, 1.0), s)  # da che parte cerca
        self.escape = np.zeros(s)  # s di fuga da un urto ancora da fare
        self.wrong = np.zeros(s, dtype=bool)  # chi adesso crede buono il colore sbagliato

    def _goal(self) -> np.ndarray:
        """Il colore che ognuno crede buono."""
        return np.where(self.wrong, 1 - self.w.good[:, None], self.w.good[:, None])

    def _aim(self, seen) -> tuple:
        """La pila buona più vicina: se non se ne vede una, quella di un attimo fa."""
        w = self.w
        wanted = seen.on & (seen.kind == self._goal()[..., None]) & (self.away[..., None] <= 0.0)
        dist = np.where(wanted, seen.dist, np.inf)
        k = dist.argmin(-1)[..., None]
        has = wanted.any(-1)
        self.ago = np.where(has, 0.0, self.ago + w.cfg.dt)
        self.ang = np.where(has, np.take_along_axis(seen.ang, k, -1)[..., 0], self.ang)
        self.far = np.where(has, np.take_along_axis(seen.dist, k, -1)[..., 0], self.far)
        return has | (self.ago < MEMORY), self.ang, self.far

    def act(self):
        w, cfg = self.w, self.w.cfg
        seen = self.seen = self.eyes.see()
        change = self.rng.random(self.wrong.shape) < cfg.dt / OPINION
        self.wrong = np.where(change, self.rng.random(self.wrong.shape) < self.mistakes, self.wrong)
        self.away = np.maximum(self.away - cfg.dt, 0.0)
        has, ang, far = self._aim(seen)
        reach = cfg.radius + cfg.battery_radius + cfg.bite_reach
        bad = seen.on & (seen.kind < 2) & (seen.kind != self._goal()[..., None])
        # Come il perfetto: le cattive vicine e davanti spostano di lato, accanto alla meta niente deviazioni.
        rel = seen.dist[..., None] * np.stack([np.cos(seen.ang), np.sin(seen.ang)], -1)
        way = np.stack([np.cos(ang), np.sin(ang)], -1)
        ahead = (rel * way[..., None, :]).sum(-1)
        side = rel - ahead[..., None] * way[..., None, :]
        push = np.clip(1.0 - seen.dist / AVOID, 0.0, 1.0) * bad * ((ahead > 0.0) & (ahead < far[..., None])) \
            * (far > reach + 8.0)[..., None]
        way = way - 2.0 * (side / np.maximum(seen.dist, 1e-6)[..., None] * push[..., None]).sum(-2)
        course = np.arctan2(way[..., 1], way[..., 0])
        stop = reach - 4.0
        thrust = np.clip((far - stop) / 50.0, 0.0, 1.0) * np.clip(1.0 - np.abs(course) / 0.8, 0.0, 1.0)
        steer = np.clip(3.0 * course, -1.0, 1.0)
        # Chi non vede niente cerca in cerchio; se urta, si stacca indietreggiando e girando.
        self.escape = np.maximum(self.escape - cfg.dt, 0.0)
        bump = w.senses()[..., BUMP].any(-1) & (self.escape <= 0.0) & ~has
        self.escape[bump] = ESCAPE
        flip = self.rng.random(has.shape) < 0.1 * cfg.dt
        self.turn = np.where(flip | bump, self.rng.choice((-1.0, 1.0), has.shape), self.turn)
        backing = self.escape > 0.0
        thrust = np.where(has, thrust, np.where(backing, -1.0, 0.6))
        steer = np.where(has, steer, np.where(backing, 1.0, 0.5) * self.turn)
        front = np.abs(seen.ang) < np.radians(cfg.mouth)
        near_bad = ((seen.dist < reach) & front & bad).any(-1)
        ready = has & (far < reach) & (np.abs(ang) < np.radians(cfg.mouth)) & ~near_bad
        bite = ready & (np.abs(w.speed) < cfg.take_speed * cfg.speed)
        self.blocked = np.where(has & (far < reach + 8.0) & ~bite, self.blocked + cfg.dt, 0.0)
        leave = self.blocked > PATIENCE
        self.away[leave], self.blocked[leave], self.ago[leave] = LEAVE, 0.0, np.inf
        return thrust, steer, bite
