import numpy as np


class ReachTarget:
    """Primo compito: toccare il bersaglio. Uno solo, condiviso e non consumabile:
    ogni agente affronta la stessa prova senza competere con gli altri.

    Un compito è un oggetto con `name`, `reset(world)` e
    `completed(world) -> (N,) bool`; per renderlo più arduo se ne scrive un altro
    con la stessa interfaccia e lo si passa a `World`.
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
