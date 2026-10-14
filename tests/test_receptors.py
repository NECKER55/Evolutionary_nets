import numpy as np
import pytest

from evonets import Config, World
from evonets.net.receptors import Prepared, ReceptorGenes, respond, spectral, stimuli
from evonets.sim import signals
from evonets.sim.signals import CHEMICAL, INTERNAL, LIGHT, MAGNETIC, MECHANICAL, THERMAL


def genes(n=1, R=1, **kw):
    """Recettori di prova: accesi, sensibilità 1, sintonia larga, puntati avanti."""
    P = 6
    g = ReceptorGenes(
        active=np.ones((n, R), dtype=bool),
        sens=np.zeros((n, R, P)),
        mu=np.full((n, R, P), 0.5),
        width=np.full((n, R, P), np.log(10.0)),  # larghissima: accetta tutto
        body=np.tile([1.0, 0.0], (n, R, 1)),
        look=np.tile([1.0, 0.0], (n, R, 1)),
        aperture=np.full((n, R), np.log(np.radians(1.0))),
        brain_y=np.full((n, R), 0.5),
        tau=np.zeros((n, R)),
        sub=np.zeros((n, R)),
        div=np.zeros((n, R)),
    )
    for k, v in kw.items():
        getattr(g, k)[...] = v
    return g


def world(cfg=None, n=1, pos=(100.0, 70.0), heading=0.0, target=(150.0, 70.0), hazards=()):
    cfg = cfg or Config()
    w = World(cfg, n, seed=0)
    w.pos[:] = pos
    w.heading[:] = heading
    w.vel[:] = 0.0
    w.ang_vel[:] = 0.0
    w.task.target = np.array(target, dtype=float)
    hz = np.array(hazards, dtype=float).reshape(-1, 3)
    w.hazards.pos, w.hazards.radius = hz[:, :2], hz[:, 2]
    return w


def test_spectral_overlap_is_the_exact_integral():
    mu, sigma, c, w, a = 0.7, 0.1, 0.8, 0.05, 0.9
    q = np.linspace(-2, 3, 200_001)
    t = np.exp(-0.5 * ((q - mu) / sigma) ** 2)
    spectrum = a * np.exp(-0.5 * ((q - c) / w) ** 2) / (w * np.sqrt(2 * np.pi))
    assert np.isclose(spectral(mu, sigma, c, w, a), np.trapezoid(t * spectrum, q), rtol=1e-6)
    assert np.isclose(spectral(0.5, 100.0, c, w, a), a, rtol=1e-3)  # sintonia larga: tutta la luce


def test_light_of_the_target_fades_with_distance():
    cfg = Config()
    w = world(cfg, target=(130.0, 70.0))
    seen = signals.light(cfg, w, np.array([[101.0, 70.0]]), np.zeros((1, 1)))
    d = 130.0 - cfg.target_radius - 101.0
    assert np.isclose(seen["bersaglio"][0], np.exp(-d / cfg.light_attenuation))
    assert seen["parete"][0] == 0.0  # il bersaglio nasconde la parete dietro


def test_walls_are_seen_when_nothing_is_in_front():
    cfg = Config()
    w = world(cfg)
    seen = signals.light(cfg, w, np.array([[101.0, 70.0]]), np.full((1, 1), np.pi))  # verso sinistra
    assert np.isclose(seen["parete"][0], np.exp(-101.0 / cfg.light_attenuation))
    assert seen["bersaglio"][0] == 0.0


def test_a_poison_cloud_glows_along_the_crossed_segment_and_hides_nothing():
    cfg = Config()
    L = cfg.light_attenuation
    w = world(cfg, target=(160.0, 70.0), hazards=[(120.0, 70.0, 5.0)])
    seen = signals.light(cfg, w, np.array([[101.0, 70.0]]), np.zeros((1, 1)))
    s_in, s_out = 120.0 - 5.0 - 101.0, 120.0 + 5.0 - 101.0
    assert np.isclose(seen["veleno"][0], L * (np.exp(-s_in / L) - np.exp(-s_out / L)))
    assert np.isclose(seen["bersaglio"][0], np.exp(-(160.0 - 3.0 - 101.0) / L))


def test_a_cloud_behind_the_target_is_not_seen():
    cfg = Config()
    w = world(cfg, target=(110.0, 70.0), hazards=[(130.0, 70.0, 5.0)])
    seen = signals.light(cfg, w, np.array([[101.0, 70.0]]), np.zeros((1, 1)))
    assert seen["veleno"][0] == 0.0


def test_a_receptor_tuned_to_the_target_colour_sees_the_target_more_than_one_tuned_to_walls():
    cfg = Config()
    w = world(cfg, target=(115.0, 70.0))
    red = genes(mu=cfg.target_color, width=np.log(0.05))
    grey = genes(mu=cfg.wall_color, width=np.log(0.05))
    assert stimuli(cfg, w, red)[0, 0, LIGHT] > 10 * stimuli(cfg, w, grey)[0, 0, LIGHT]


def test_pressure_only_on_the_side_touching_the_wall():
    cfg = Config()
    w = world(cfg, pos=(cfg.agent_radius, 70.0), heading=0.0)  # appoggiato alla parete sinistra
    g = genes(R=2)
    g.body[0, 0] = [-1.0, 0.0]  # dietro: verso la parete
    g.body[0, 1] = [1.0, 0.0]  # davanti
    s = stimuli(cfg, w, g)[0, :, MECHANICAL]
    assert s[0] > 0.99 and s[1] == 0.0


def test_toxin_inside_poison_equals_the_damage_and_is_odourless_outside_by_default():
    cfg = Config()
    w = world(cfg, pos=(100.0, 70.0), hazards=[(100.0, 70.0, 8.0)])
    g = genes(body=[0.0, 0.0], mu=cfg.toxin_molecule)  # al centro del corpo, per semplicità
    inside = stimuli(cfg, w, g)[0, 0, CHEMICAL]
    assert np.isclose(inside, w.hazards.intensity(w.pos)[0])
    w.pos[:] = (100.0, 90.0)
    assert stimuli(cfg, w, g)[0, 0, CHEMICAL] == 0.0
    smelly = Config(hazard_smell=0.5)
    w2 = world(smelly, pos=(100.0, 90.0), hazards=[(100.0, 70.0, 8.0)])
    expected = 0.5 * np.exp(-(20.0 - 8.0) / smelly.smell_range)
    assert np.isclose(stimuli(smelly, w2, g)[0, 0, CHEMICAL], expected)


def test_vibrations_are_heard_only_inside_the_aperture():
    cfg = Config(target_vibration=1.0)
    w = world(cfg, target=(120.0, 70.0))
    front = genes(aperture=np.log(np.radians(60.0)))
    back = genes(aperture=np.log(np.radians(60.0)), look=[-1.0, 0.0])
    assert stimuli(cfg, w, front)[0, 0, MECHANICAL] > 0.0
    assert stimuli(cfg, w, back)[0, 0, MECHANICAL] == 0.0


def test_temperature_and_compass():
    cfg = Config(target_heat=0.5, magnetic_field=1.0, magnetic_angle_deg=0.0)
    w = world(cfg, target=(103.0, 70.0))
    warm = genes(mu=1.0, width=np.log(0.1))
    assert stimuli(cfg, w, warm)[0, 0, THERMAL] > stimuli(Config(), world(), warm)[0, 0, THERMAL]
    east, west = genes(), genes(look=[-1.0, 0.0])
    assert np.isclose(stimuli(cfg, w, east)[0, 0, MAGNETIC], 1.0)
    assert stimuli(cfg, w, west)[0, 0, MAGNETIC] == 0.0


def test_internal_receptor_reads_the_body():
    cfg = Config()
    w = world(cfg)
    w.vel[:] = (15.0, 0.0)  # metà della velocità massima, in avanti
    forward = genes(mu=0.35, width=np.log(0.01))
    backward = genes(mu=0.45, width=np.log(0.01))
    assert np.isclose(stimuli(cfg, w, forward)[0, 0, INTERNAL], 0.5)
    assert np.isclose(stimuli(cfg, w, backward)[0, 0, INTERNAL], 0.0)


def test_noise_follows_counting_statistics_and_saturates():
    cfg = Config()
    w = world(cfg, target=(110.0, 70.0))
    g = genes(n=4000, sens=-50.0)  # insensibile a tutto...
    g.sens[:, :, INTERNAL] = np.log(40.0)  # ...tranne l'energia, che vale 1
    g.mu[:, :, INTERNAL], g.width[:, :, INTERNAL] = 0.05, np.log(0.01)
    w = world(cfg, n=4000, target=(110.0, 70.0))
    r = respond(cfg, w, g, np.random.default_rng(0))[:, 0] * cfg.receptor_saturation
    assert np.isclose(r.mean(), 40.0, rtol=0.01)
    assert np.isclose(r.var(), 40.0 + cfg.receptor_dark, rtol=0.1)
    g.sens[:, :, INTERNAL] = np.log(1000.0)  # dieci volte la saturazione
    assert (respond(cfg, w, g, np.random.default_rng(0)) == 1.0).all()


def test_switched_off_receptors_are_silent_and_same_seed_same_noise():
    cfg = Config()
    w = world(cfg, n=3)
    g = genes(n=3, R=4)
    g.active[:, 1] = False
    a = respond(cfg, w, g, np.random.default_rng(5))
    b = respond(cfg, w, g, np.random.default_rng(5))
    assert (a[:, 1] == 0.0).all() and np.array_equal(a, b)


def test_some_rows_give_the_same_stimuli_as_all_rows():
    cfg = Config()
    w = World(cfg, 6, seed=3)
    g = genes(n=6, R=3, aperture=np.log(np.radians(90.0)))
    g.look[:, 1] = [0.0, 1.0]
    rows = np.array([1, 4])
    assert np.allclose(stimuli(cfg, w, g)[rows], stimuli(cfg, w, g.take(rows), rows))


@pytest.mark.parametrize("aperture", [1.0, 90.0, 360.0])
def test_wider_aperture_averages_more_directions(aperture):
    cfg = Config()
    w = world(cfg, target=(120.0, 70.0))
    s = stimuli(cfg, w, genes(aperture=np.log(np.radians(aperture))))[0, 0, LIGHT]
    assert 0.0 < s <= 1.0


def test_prepared_fast_path_gives_the_same_quanta_as_the_reference():
    cfg = Config(target_vibration=0.7, hazard_smell=0.3, target_heat=0.4, magnetic_field=0.8)
    rng = np.random.default_rng(1)
    n, R = 7, 5
    g = genes(n=n, R=R)
    g.sens[:] = rng.normal(1.0, 1.0, g.sens.shape)
    g.mu[:] = rng.random(g.mu.shape)
    g.width[:] = np.log(rng.uniform(0.05, 0.5, g.width.shape))
    angle = rng.uniform(-np.pi, np.pi, (n, R, 2))
    g.body[:] = np.stack([np.cos(angle[..., 0]), np.sin(angle[..., 0])], axis=-1)
    g.look[:] = np.stack([np.cos(angle[..., 1]), np.sin(angle[..., 1])], axis=-1)
    g.aperture[:] = np.log(rng.uniform(0.05, 6.0, (n, R)))
    g.active[0, 1] = False
    w = World(cfg, n, seed=2)
    w.pos[0] = (cfg.agent_radius, 50.0)  # uno appoggiato alla parete
    w.bump[0] = True
    w.vel[:] = rng.normal(0, 10, (n, 2))
    reference = (g.gain * stimuli(cfg, w, g)).sum(axis=2)
    assert np.allclose(Prepared(cfg, g).quanta(w, np.arange(n)), reference)
    rows = np.array([2, 5])
    assert np.allclose(Prepared(cfg, g).take(rows).quanta(w, rows), reference[rows])


def _energy_receptor(n=1, quanta=40.0, **kw):
    """Un recettore sensibile solo all'energia (che vale 1): n quanti costanti."""
    g = genes(n=n, sens=-50.0, **kw)
    g.sens[:, :, INTERNAL] = np.log(quanta)
    g.mu[:, :, INTERNAL], g.width[:, :, INTERNAL] = 0.05, np.log(0.01)
    return g


def test_without_adaptation_the_fast_path_matches_the_reference_response():
    cfg = Config()
    w = world(cfg, n=500)
    g = _energy_receptor(500)
    ref = respond(cfg, w, g, np.random.default_rng(3))
    fast, _ = Prepared(cfg, g).respond(w, np.arange(500), np.random.default_rng(3), np.full((500, 1), np.nan))
    assert np.allclose(ref, fast)


def test_subtractive_adaptation_answers_changes_and_forgets_a_constant_signal():
    cfg = Config(receptor_dark=1e-9)
    w = world(cfg, n=200)
    g = _energy_receptor(200, quanta=60.0, sub=1.0, tau=np.log(0.5))
    w.energy[:] = cfg.energy_init / 3  # il segnale vale 1/3: 20 quanti
    prep, m, rng = Prepared(cfg, g), np.full((200, 1), np.nan), np.random.default_rng(0)
    first, m = prep.respond(w, np.arange(200), rng, m)
    for _ in range(300):  # 5 s a segnale costante: 10 volte τ
        r, m = prep.respond(w, np.arange(200), rng, m)
    assert r.mean() < 0.05  # adattato: quasi zero
    w.energy[:] = cfg.energy_init  # il segnale triplica di colpo: 60 quanti
    jump, m = prep.respond(w, np.arange(200), rng, m)
    assert jump.mean() > 0.3  # risponde al cambiamento (40 quanti in più su 100)


def test_divisive_adaptation_answers_contrast_not_level():
    cfg = Config(receptor_dark=1e-9)
    out = []
    for level in (1.0, 4.0):  # stesso contrasto (+50%) su due livelli di fondo diversi
        w = world(cfg, n=400)
        w.energy[:] = cfg.energy_init / 2
        g = _energy_receptor(400, quanta=40.0 * level, div=50.0, tau=np.log(0.2))
        prep, m, rng = Prepared(cfg, g), np.full((400, 1), np.nan), np.random.default_rng(1)
        for _ in range(200):
            _, m = prep.respond(w, np.arange(400), rng, m)
        w.energy[:] = cfg.energy_init * 0.75
        r, _ = prep.respond(w, np.arange(400), rng, m)
        out.append(r.mean())
    assert np.isclose(out[0], out[1], rtol=0.15)  # senza adattamento sarebbero 4 volte diverse
