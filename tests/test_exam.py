import dataclasses

import numpy as np

from evonets import Config, World
from evonets.evo import Evolution
from evonets.evo.exam import ENVIRONMENT, FROM_RUN, TESTS, Exam, candidates, exam_config
from evonets.net import NeuralController
from evonets.sim.tasks import FarTarget, MovingTarget


def small(**kw):
    return Config(population=12, net_hidden_max=4, **kw)


def test_exam_environment_does_not_follow_the_run():
    run = small(energy_init=40.0, n_hazards=2, net_type="spike-ei")
    for t in TESTS:
        cfg = exam_config(run, t)
        assert cfg.net_type == "spike-ei" and cfg.net_hidden_max == 4  # la rete viene dalla corsa
        for k, v in ENVIRONMENT.items():
            assert getattr(cfg, k) == t.changes.get(k, v)  # l'ambiente dall'esame
    assert run.energy_init == 40.0  # la corsa non viene toccata


def test_candidates_are_the_oldest_then_the_first_born():
    age = np.array([0, 3, 1, 3, 0])
    ids = np.array([10, 7, 5, 2, 1])
    assert list(candidates(age, ids, 3)) == [3, 1, 2]


def test_far_target_is_far_from_everyone():
    w = World(Config(), 50, seed=3, task=FarTarget(90.0))
    assert (np.linalg.norm(w.pos - w.task.target, axis=1) >= 90.0).all()


def test_moving_target_moves_and_stays_in_the_arena():
    cfg = Config()
    w = World(cfg, 2, seed=1, task=MovingTarget(5.0))
    start = w.task.target.copy()
    for _ in range(3000):
        w.step(np.zeros((2, 2)))
        assert (w.task.target >= 0).all() and w.task.target[0] <= cfg.arena_w and w.task.target[1] <= cfg.arena_h
    assert not np.allclose(start, w.task.target)


def test_sensor_noise_disturbs_only_when_asked():
    clean, noisy = small(), small(sensor_noise=0.1)
    w = World(clean, 12, seed=0)
    a = NeuralController(clean, 12, seed=0)
    b = NeuralController(noisy, 12, seed=0)
    obs = a.senses.observe(w)
    assert np.array_equal(a._noisy(obs), obs)
    assert not np.array_equal(b._noisy(obs), obs)


def test_exam_does_not_change_the_evolution(tmp_path):
    a, b = Evolution(small(), 4), Evolution(small(), 4)
    for e in (a, b):
        e.run_generation()
    b.start_exam(arenas=1, workers=1).run()  # un'arena per prova basta per il test
    b.collect_exam()
    for e in (a, b):
        e.run_generation()
    assert np.array_equal(a.ctrl.brains.weights, b.ctrl.brains.weights)
    assert np.array_equal(a.world.pos, b.world.pos)
    r = b.exams[0]
    assert set(r["scores"]) == {t.key for t in TESTS} and 0.0 <= r["score"] <= 1.0
    b.save(tmp_path / "pop.npz")
    assert Evolution.load(small(), tmp_path / "pop.npz").exams == b.exams


def test_exam_is_due_every_k_generations():
    e = Evolution(small(exam_every=2), 0)
    due = []
    for _ in range(4):
        e.run_generation()
        due.append(e.exam_due())
    assert due == [False, True, False, True]
    e.cfg.exam_every = 0
    assert not e.exam_due()


def test_same_score_in_parallel_and_here():
    cfg = small()
    g = NeuralController(cfg, 3, seed=9).brains.genome
    here = Exam(cfg, g, 0, arenas=2, clones=2, workers=1)
    while not here.finished:
        here.run(0.05)  # a pezzi, come nell'interfaccia
    par = Exam(cfg, g, 0, arenas=2, clones=2, workers=4)
    par.run()
    assert par.workers == 4 and np.array_equal(here.wins, par.wins)
    assert here.result()["scores"] == par.result()["scores"]


def test_every_constant_is_either_exam_environment_or_from_the_run():
    """Se fallisce, è stata aggiunta (o tolta) una costante alla configurazione:
    decidi se l'esame la fissa (ENVIRONMENT in evo/exam.py, con il valore
    d'esame) o se viene dalla corsa (FROM_RUN). Se l'esame deve cambiare
    davvero, alza anche VERSION."""
    fields = {f.name for f in dataclasses.fields(Config)}
    env, run = set(ENVIRONMENT), set(FROM_RUN)
    assert not env & run, f"in tutti e due gli elenchi: {sorted(env & run)}"
    assert not fields - env - run, f"costanti nuove da classificare: {sorted(fields - env - run)}"
    assert not (env | run) - fields, f"costanti che non esistono più: {sorted((env | run) - fields)}"
