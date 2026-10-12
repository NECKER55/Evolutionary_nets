import numpy as np

from evonets import Config, World
from evonets.net.sensors import BODY, Senses, Vision

WALL, HAZARD, TARGET = 0, 1, 2


def setup(**kw):
    """Un agente al centro che guarda verso +x, con il raggio 1 dritto davanti."""
    cfg = Config(n_hazards=1, hazard_speed=0.0, hazard_pulse=0.0, vision_rays=3,
                 vision_fov_deg=90.0, vision_range=20.0, **kw)
    w = World(cfg, 1)
    w.pos[0] = (50.0, 35.0)
    w.heading[0] = 0.0
    park_hazard(w, (-500.0, -500.0))
    w.task.target = np.array([-500.0, -500.0])
    return cfg, w, Vision(cfg)


def park_hazard(w, where):
    """Sposta il pericolo in un punto fisso, in modo che ci resti anche dopo uno step."""
    w.hazards.amp[:] = 0.0
    w.hazards.anchor[0] = where
    w.hazards.pos[0] = where


def test_shape_and_empty_view():
    cfg, w, v = setup()
    obs = v.sense(w)
    assert obs.shape == (1, 3, 4) and v.size == 12
    assert (obs == 0.0).all()  # pareti oltre il raggio visivo, nient'altro in vista


def test_wall_closeness_straight_ahead():
    cfg, w, v = setup()
    w.pos[0] = (cfg.arena_w - 5.0, 35.0)
    obs = v.sense(w)[0]
    assert np.isclose(obs[1, WALL], 1.0 - 5.0 / 20.0)
    # I raggi laterali a 45° incontrano la stessa parete più lontano.
    assert np.allclose(obs[[0, 2], WALL], 1.0 - 5.0 * np.sqrt(2.0) / 20.0)


def test_hazard_seen_at_its_edge_distance():
    cfg, w, v = setup()
    rad = w.hazards.radius[0]
    w.hazards.pos[0] = (50.0 + 10.0 + rad, 35.0)  # bordo a 10 u dritto davanti
    obs = v.sense(w)[0]
    assert np.isclose(obs[1, HAZARD], 0.5)
    assert obs[1, TARGET] == 0.0


def test_inside_hazard_reads_full_closeness_on_every_ray():
    cfg, w, v = setup()
    w.hazards.pos[0] = w.pos[0]
    assert (v.sense(w)[0, :, HAZARD] == 1.0).all()


def test_target_visible_ahead_but_not_behind_or_out_of_range():
    cfg, w, v = setup()
    tr = w.task.radius
    w.task.target = np.array([50.0 + 8.0 + tr, 35.0])
    assert np.isclose(v.sense(w)[0, 1, TARGET], 1.0 - 8.0 / 20.0)
    w.task.target = np.array([50.0 - 8.0 - tr, 35.0])  # alle spalle
    assert (v.sense(w)[0, :, TARGET] == 0.0).all()
    w.task.target = np.array([50.0 + 25.0 + tr, 35.0])  # troppo lontano
    assert (v.sense(w)[0, :, TARGET] == 0.0).all()


def test_view_rotates_with_heading():
    cfg, w, v = setup()
    tr = w.task.radius
    w.task.target = np.array([50.0, 35.0 + 8.0 + tr])  # a +90° (y verso il basso)
    assert (v.sense(w)[0, :, TARGET] == 0.0).all()
    w.heading[0] = np.pi / 2
    assert np.isclose(v.sense(w)[0, 1, TARGET], 0.6)


def test_reserved_delivery_channel_is_silent_until_a_task_defines_a_goal():
    cfg, w, v = setup()
    w.pos[0] = (50.0, 35.0)
    assert (v.sense(w)[..., 3] == 0.0).all()
    w.task.goal, w.task.goal_radius = np.array([62.0, 35.0]), 2.0
    assert np.isclose(v.sense(w)[0, 1, 3], 0.5)


def test_observation_layout_and_body_senses():
    cfg, w, v = setup()
    s = Senses(cfg)
    assert s.size == 3 * 4 + len(BODY) == len(s.labels)
    w.vel[0] = (cfg.thrust_accel / cfg.lin_drag / 2.0, 0.0)  # metà velocità massima, in avanti
    w.energy[0] = cfg.energy_init / 4.0
    body = dict(zip(BODY, s.observe(w)[0, -len(BODY):]))
    assert np.isclose(body["vel. avanti"], 0.5) and np.isclose(body["vel. traverso"], 0.0)
    assert np.isclose(body["energia"], 0.25)
    assert body["dolore"] == 0.0 and body["urto"] == 0.0
    w.heading[0] = -np.pi / 2  # ora la stessa velocità è tutta di traverso
    body = dict(zip(BODY, s.observe(w)[0, -len(BODY):]))
    assert np.isclose(body["vel. avanti"], 0.0) and np.isclose(abs(body["vel. traverso"]), 0.5)


def test_pain_and_bump_follow_the_world():
    cfg, w, v = setup()
    s = Senses(cfg)
    park_hazard(w, w.pos[0].copy())
    w.step(np.zeros((1, 2)))
    assert np.isclose(s.observe(w)[0, -2], 1.0)  # al centro del veleno: dolore pieno
    park_hazard(w, (-500.0, -500.0))
    w.pos[0] = (0.0, 35.0)
    w.step(np.zeros((1, 2)))
    obs = s.observe(w)[0]
    assert obs[-2] == 0.0 and obs[-1] == 1.0
