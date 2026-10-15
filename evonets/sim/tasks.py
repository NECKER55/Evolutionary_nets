import numpy as np


class ReachTarget:
    """Primo compito: toccare il bersaglio. Uno solo, condiviso e non consumabile:
    ogni agente affronta la stessa prova senza competere con gli altri.

    Un compito è un oggetto con `name`, `reset(world)` e
    `completed(world) -> (N,) bool`; per renderlo più arduo se ne scrive un altro
    con la stessa interfaccia e lo si passa a `World`. Facoltativi:
    `spawn_ok(world, pos) -> (N,) bool` (dove gli agenti possono nascere) e
    `step(world)` (il compito cambia durante la vita).
    """

    name = "raggiungi il bersaglio"

    def reset(self, world) -> None:
        cfg = world.cfg
        r = cfg.target_radius
        self.radius = r
        self.target = world.rng.uniform([r, r], [cfg.arena_w - r, cfg.arena_h - r])

    def completed(self, world) -> np.ndarray:
        d = np.linalg.norm(world.pos - self.target, axis=1)
        return d <= self.radius + world.cfg.agent_radius


class FarTarget(ReachTarget):
    """Il bersaglio nasce lontano da tutti: almeno `min_dist` da ogni agente.
    Misura l'esplorazione."""

    name = "bersaglio lontano"

    def __init__(self, min_dist: float = 90.0):
        self.min_dist = min_dist

    def spawn_ok(self, world, pos: np.ndarray) -> np.ndarray:
        return np.linalg.norm(pos - self.target, axis=1) >= self.min_dist


class MovingTarget(ReachTarget):
    """Il bersaglio si muove a velocità costante e rimbalza sulle pareti:
    bisogna inseguirlo, non solo andare dove lo si è visto."""

    name = "bersaglio in movimento"

    def __init__(self, speed: float = 5.0):
        self.speed = speed

    def reset(self, world) -> None:
        super().reset(world)
        a = world.rng.uniform(-np.pi, np.pi)
        self.vel = self.speed * np.array([np.cos(a), np.sin(a)])

    def step(self, world) -> None:
        cfg, r = world.cfg, self.radius
        self.target = self.target + self.vel * cfg.dt
        hi = np.array([cfg.arena_w - r, cfg.arena_h - r])
        out = (self.target < r) | (self.target > hi)
        self.vel = np.where(out, -self.vel, self.vel)
        self.target = np.clip(self.target, r, hi)
