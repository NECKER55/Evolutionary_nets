import numpy as np

from evonets import Config, World
from evonets.net import RandomWalkController


def no_hazards(**kw) -> Config:
    return Config(n_hazards=0, **kw)


def test_terminal_speed_matches_thrust_over_drag():
    cfg = no_hazards(arena_w=10_000.0, arena_h=10_000.0)
    w = World(cfg, 1)
    w.pos[:] = 5_000.0
    for _ in range(600):
        w.step(np.array([[1.0, 0.0]]))
    assert np.isclose(w.speed[0], cfg.thrust_accel / cfg.lin_drag, rtol=0.03)


def test_reverse_is_weaker():
    cfg = no_hazards(arena_w=10_000.0, arena_h=10_000.0)
    w = World(cfg, 2)
    w.pos[:] = 5_000.0
    for _ in range(600):
        w.step(np.array([[1.0, 0.0], [-1.0, 0.0]]))
    assert np.isclose(w.speed[1], w.speed[0] * cfg.reverse_factor, rtol=1e-6)


def test_lateral_velocity_decays_faster_than_forward():
    cfg = no_hazards(arena_w=10_000.0, arena_h=10_000.0)
    w = World(cfg, 2)
    w.pos[:] = 5_000.0
    w.heading[:] = 0.0
    w.vel[0] = (10.0, 0.0)  # lungo il muso
    w.vel[1] = (0.0, 10.0)  # di traverso
    for _ in range(30):
        w.step(np.zeros((2, 2)))
    assert w.speed[1] < w.speed[0] * 0.2


def test_agents_never_leave_arena():
    cfg = no_hazards()
    w = World(cfg, 200, seed=3)
    ctrl = RandomWalkController(200, seed=3)
    r = cfg.agent_radius
    for _ in range(3000):
        w.step(ctrl.act(w))
        assert w.pos.min() >= r - 1e-9
        assert (w.pos[:, 0] <= cfg.arena_w - r + 1e-9).all()
        assert (w.pos[:, 1] <= cfg.arena_h - r + 1e-9).all()


def test_hazards_stay_inside_arena_and_spawn_is_safe():
    cfg = Config(n_hazards=12)
    w = World(cfg, 300, seed=1)
    assert w.alive.all()
    assert (w.hazards.clearance(w.pos) >= cfg.agent_radius).all()
    size = np.array([cfg.arena_w, cfg.arena_h])
    for _ in range(5000):
        w.hazards.step(cfg.dt)
        rad = w.hazards.radius[:, None]
        assert (w.hazards.pos - rad >= -1e-9).all()
        assert (w.hazards.pos + rad <= size + 1e-9).all()


NO_METABOLISM = dict(metabolism_basal=0.0, metabolism_move=0.0, metabolism_turn=0.0)


def one_static_hazard(**kw) -> Config:
    """Un solo pericolo fermo e metabolismo spento, per isolare l'effetto del veleno."""
    return Config(n_hazards=1, hazard_speed=0.0, hazard_pulse=0.0, **NO_METABOLISM, **kw)


def test_resting_agent_lives_energy_over_basal_seconds():
    cfg = no_hazards()
    w = World(cfg, 1)
    w.task.target = np.array([-500.0, -500.0])
    lifetime = round(cfg.energy_init / cfg.metabolism_basal / cfg.dt)
    for _ in range(lifetime - 2):
        w.step(np.zeros((1, 2)))
    assert w.alive[0]
    for _ in range(4):
        w.step(np.zeros((1, 2)))
    assert not w.alive[0]


def test_effort_costs_energy_in_both_directions():
    cfg = no_hazards(arena_w=10_000.0, arena_h=10_000.0)
    w = World(cfg, 4)
    w.pos[:] = 5_000.0
    w.task.target = np.array([-500.0, -500.0])
    w.step(np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]))
    lost = (cfg.energy_init - w.energy) / cfg.dt
    assert np.allclose(lost, [
        cfg.metabolism_basal,
        cfg.metabolism_basal + cfg.metabolism_move,
        cfg.metabolism_basal + cfg.metabolism_move,
        cfg.metabolism_basal + cfg.metabolism_turn,
    ])


def test_agent_that_completed_the_task_stops_paying():
    cfg = no_hazards()
    w = World(cfg, 1)
    w.pos[0] = w.task.target
    w.step(np.zeros((1, 2)))
    assert w.done[0]
    e = w.energy[0]
    for _ in range(600):
        w.step(np.ones((1, 2)))
    assert w.energy[0] == e and w.alive[0]


def test_hazard_drain_grows_from_edge_to_center():
    cfg = one_static_hazard()
    w = World(cfg, 4)
    c, rad = w.hazards.pos[0], w.hazards.radius[0]
    w.pos[:] = c + np.outer([0.0, 0.5, 0.95, 1.2], [rad, 0.0])
    w.step(np.zeros((4, 2)))
    lost = cfg.energy_init - w.energy
    assert np.isclose(lost[0], cfg.hazard_drain * cfg.dt)  # centro: danno pieno
    assert lost[0] > lost[1] > lost[2] > 0.0
    assert lost[2] < 0.01 * lost[0]  # bordo: quasi nulla
    assert lost[3] == 0.0  # fuori: niente
    assert w.alive.all()  # nessuno muore al primo contatto


def test_agent_dies_when_energy_runs_out_and_then_freezes():
    cfg = one_static_hazard()
    w = World(cfg, 2)
    w.pos[0] = w.hazards.pos[0]
    w.pos[1] = w.hazards.pos[0] + (w.hazards.radius[0] + 5.0, 0.0)
    steps_to_die = int(np.ceil(cfg.energy_init / (cfg.hazard_drain * cfg.dt)))
    for _ in range(steps_to_die - 2):
        w.step(np.zeros((2, 2)))
    assert w.alive[0]
    for _ in range(3):
        w.step(np.zeros((2, 2)))
    assert not w.alive[0] and w.energy[0] == 0.0
    frozen = w.pos[0].copy()
    for _ in range(60):
        w.step(np.array([[1.0, 1.0], [0.0, 0.0]]))
    assert np.array_equal(w.pos[0], frozen)
    assert w.alive[1] and w.energy[1] == cfg.energy_init


def test_touching_target_completes_task_and_agent_is_safe():
    cfg = one_static_hazard()
    w = World(cfg, 3)
    w.task.target = w.hazards.pos[0].copy()  # bersaglio nel punto più tossico
    w.pos[0] = w.task.target
    w.pos[1] = w.task.target + (w.task.radius + cfg.agent_radius + 0.5, 0.0)
    w.pos[2] = (cfg.agent_radius, cfg.agent_radius)
    w.step(np.zeros((3, 2)))
    assert list(w.done) == [True, False, False]
    e = w.energy[0]
    for _ in range(300):
        w.step(np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
    assert w.alive[0] and w.energy[0] == e  # in salvo: fermo e non più drenato
    assert np.allclose(w.pos[0], w.task.target)


def test_dead_agent_cannot_complete_task():
    cfg = one_static_hazard()
    w = World(cfg, 1)
    w.alive[0] = False
    w.pos[0] = w.task.target
    w.step(np.zeros((1, 2)))
    assert not w.done[0]


def test_same_seed_is_deterministic():
    runs = []
    for _ in range(2):
        w = World(Config(), 50, seed=7)
        ctrl = RandomWalkController(50, seed=7)
        for _ in range(500):
            w.step(ctrl.act(w))
        runs.append((w.pos.copy(), w.alive.copy()))
    assert np.array_equal(runs[0][0], runs[1][0])
    assert np.array_equal(runs[0][1], runs[1][1])


def test_hazard_speed_change_does_not_teleport():
    cfg = Config()
    w = World(cfg, 1)
    for _ in range(300):
        w.hazards.step(cfg.dt)
    before = w.hazards.pos.copy()
    cfg.hazard_speed = 3.0
    w.hazards.step(cfg.dt)
    # Si muovono al più alla nuova velocità, senza salti.
    fastest = (w.hazards.amp * w.hazards.omega).max() * cfg.hazard_speed * cfg.dt
    assert 0.0 < np.abs(w.hazards.pos - before).max() <= fastest * 1.001
