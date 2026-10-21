import numpy as np

from ..config import Config
from .hazards import HazardField
from .tasks import TASKS


class World:
    """Simulazione batch di N agenti, senza dipendenze grafiche.

    Lo stato è in array NumPy (una riga per agente) e `step` riceve le azioni di
    tutti gli agenti insieme: chi le produce (tastiera, mock, rete) è esterno.
    Gli agenti non collidono tra loro: ognuno vive la propria prova nell'arena.

    `done` segna chi ha completato il compito (nel mondo del cibo: chi è
    maturo). Con `keep_living` chi ha completato continua a vivere, mangiare e
    rischiare (mondo del cibo con `mature_keep_living`), altrimenti si ferma ed
    è in salvo. `racing` dice chi è ancora in gara.

    `done` è l'unico lasciapassare per la
    riproduzione. Chi completa è in salvo (esce dalla simulazione), chi muore
    prima non può più completare.
    """

    def __init__(self, cfg: Config, n_agents: int, seed: int = 0, task=None):
        self.cfg = cfg
        self.task = task or TASKS[cfg.task]()
        self.n = n_agents
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        cfg = self.cfg
        self.rng = np.random.default_rng(self.seed)
        self.t = 0.0
        self.hazards = HazardField(cfg, self.rng)
        self.task.reset(self)
        self.keep_living = bool(cfg.mature_keep_living and getattr(self.task, "keeps_living", False))
        self.pos = self._spawn_positions()
        if hasattr(self.task, "spawned"):
            self.task.spawned(self)
        self.heading = self.rng.uniform(-np.pi, np.pi, self.n)
        self.vel = np.zeros((self.n, 2))
        self.ang_vel = np.zeros(self.n)
        self.energy = np.full(self.n, cfg.energy_init)
        self.alive = np.ones(self.n, dtype=bool)
        self.done = np.zeros(self.n, dtype=bool)
        self.pain = np.zeros(self.n)  # dolore: veleno e morsi dell'ultimo passo, in [0, 1]
        self.taste = np.zeros(self.n)  # gusto: impulso quando si mangia, che svanisce
        self.bump = np.zeros(self.n, dtype=bool)  # a contatto con una parete

    def _spawn_positions(self) -> np.ndarray:
        cfg = self.cfg
        r = cfg.agent_radius
        lo = np.array([r, r])
        hi = np.array([cfg.arena_w - r, cfg.arena_h - r])
        pos = self.rng.uniform(lo, hi, (self.n, 2))
        for _ in range(100):
            bad = self.hazards.clearance(pos) < r + cfg.spawn_margin
            if hasattr(self.task, "spawn_ok"):
                bad |= ~self.task.spawn_ok(self, pos)
            if not bad.any():
                break
            pos[bad] = self.rng.uniform(lo, hi, (int(bad.sum()), 2))
        return pos

    def step(self, actions: np.ndarray, upkeep: np.ndarray | None = None) -> None:
        """actions: (N, 2) in [-1, 1] -> colonne (spinta, sterzata).
        upkeep: (N,) energia/s che ogni agente spende in più per mantenere il
        proprio corpo (oggi: il cablaggio del cervello). Il mondo non sa da dove
        viene: lo decide chi controlla gli agenti."""
        cfg = self.cfg
        dt = cfg.dt
        act = np.clip(actions, -1.0, 1.0)
        thrust = np.where(act[:, 0] < 0.0, act[:, 0] * cfg.reverse_factor, act[:, 0])
        turn = act[:, 1]

        # Rotazione
        self.ang_vel += turn * cfg.turn_accel * dt
        self.ang_vel *= np.exp(-cfg.ang_drag * dt)
        self.heading += self.ang_vel * dt
        self.heading = (self.heading + np.pi) % (2.0 * np.pi) - np.pi

        # Traslazione: la velocità è scomposta lungo l'asse dell'agente e
        # trasversalmente, con attriti diversi (lat_drag alto = "ruote").
        fwd = np.stack([np.cos(self.heading), np.sin(self.heading)], axis=1)
        v_fwd = np.einsum("ij,ij->i", self.vel, fwd)
        v_lat = self.vel - v_fwd[:, None] * fwd
        v_fwd = (v_fwd + thrust * cfg.thrust_accel * dt) * np.exp(-cfg.lin_drag * dt)
        v_lat *= np.exp(-cfg.lat_drag * dt)
        self.vel = v_fwd[:, None] * fwd + v_lat

        # Morti (e arrivati, se si fermano) restano fermi dove sono.
        active = self.racing
        self.vel[~active] = 0.0
        self.ang_vel[~active] = 0.0
        self.pos += self.vel * dt

        # Pareti: rimbalzo anelastico
        r = cfg.agent_radius
        lo = np.array([r, r])
        hi = np.array([cfg.arena_w - r, cfg.arena_h - r])
        hit = ((self.pos < lo) & (self.vel < 0.0)) | ((self.pos > hi) & (self.vel > 0.0))
        self.vel[hit] *= -cfg.wall_restitution
        np.clip(self.pos, lo, hi, out=self.pos)
        self.bump = ((self.pos <= lo) | (self.pos >= hi)).any(axis=1)

        # Energia: la consumano il metabolismo e i pericoli, tanto più quanto si
        # è vicini al loro centro
        self.hazards.step(dt)
        toxicity = self.hazards.intensity(self.pos) * active
        self.pain = np.minimum(toxicity, 1.0)
        self.taste *= np.exp(-dt / cfg.taste_tau)
        # Metabolismo: si paga lo sforzo comandato, non la velocità raggiunta;
        # le riserve oltre l'energia di partenza pesano e rendono il movimento più caro.
        mass = 1.0 + cfg.reserve_mass * np.maximum(self.energy - cfg.energy_init, 0.0) / cfg.energy_init
        effort = (
            cfg.metabolism_basal
            + mass * cfg.metabolism_move * np.abs(act[:, 0]) ** cfg.metabolism_exponent
            + mass * cfg.metabolism_turn * np.abs(turn) ** cfg.metabolism_exponent
            + (0.0 if upkeep is None else upkeep)
        ) * active
        spent = (effort + cfg.hazard_drain * toxicity) * dt
        self.energy = np.maximum(self.energy - spent, 0.0)
        # Il compito può cambiare il mondo e l'energia (mangiare, predatori)
        # prima che si contino i vivi.
        if hasattr(self.task, "step"):
            self.task.step(self)
        self.alive &= self.energy > 0.0
        self.done |= self.task.completed(self) & self.alive
        self.t += dt

    @property
    def racing(self) -> np.ndarray:
        """Chi è ancora in gara: vivo, e non arrivato (o arrivato ma la vita continua)."""
        return self.alive & (~self.done | self.keep_living)

    @property
    def over(self) -> bool:
        """La prova è finita: nessuno in gara, o tempo scaduto."""
        return not self.racing.any() or self.t >= self.cfg.life_max

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.vel, axis=1)
