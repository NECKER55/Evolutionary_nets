"""La scala di difficoltà, i controlli, l'esame a livelli e le lesioni
(docs/SPEC_MONDO.md, punti 3 e 4)."""

import numpy as np
import pytest

from evonets import Config
from evonets.evo import Evolution, ladder
from evonets.evo.evolution import run_together
from evonets.evo.exam import LESIONS, LEVELS, LEVELS_VERSION, Exam, battery, lesioned


def forage(**kw):
    return Config(**{"task": "cibo", "n_hazards": 0, "population": 20, "net_hidden_slots": 4, **kw})


def test_one_position_sums_up_the_whole_ladder():
    assert ladder.at(0.0) == dict.fromkeys(ladder.DIALS, 0.0)
    assert ladder.at(ladder.TOP) == dict.fromkeys(ladder.DIALS, 1.0)
    d = ladder.at(1.5)  # primo stadio finito, prede a metà, predatori assenti
    assert d == {"forage_abundance": 1.0, "forage_need": 1.0, "forage_prey": 0.5, "forage_scent": 0.5,
                 "forage_predators": 0.0}
    cfg = forage(**d)
    assert ladder.position(cfg) == 1.5


def history(cfg, rates):
    return [{"completed": round(r * cfg.population)} for r in rates]


def test_the_ladder_climbs_when_easy_drops_when_hard_and_stays_in_between():
    cfg = forage(ladder="scala")
    high, mid, low = cfg.ladder_up + 0.1, (cfg.ladder_up + cfg.ladder_down) / 2, cfg.ladder_down / 2
    assert ladder.update(cfg, history(cfg, [high] * 10))["position"] == pytest.approx(0.02)
    assert cfg.forage_abundance == cfg.forage_need == 0.02 and cfg.forage_prey == 0.0
    assert ladder.update(cfg, history(cfg, [mid] * 10)) == {}  # tra le soglie: ferma
    ladder.update(cfg, history(cfg, [low] * 10))
    assert ladder.position(cfg) == pytest.approx(0.0)
    assert ladder.update(cfg, history(cfg, [0.05] * 10)) == {}  # non scende sotto zero


def test_nobody_completing_drops_the_ladder_more():
    cfg = forage(ladder="scala", **ladder.at(1.2))
    moved = ladder.update(cfg, history(cfg, [0.5] * 9 + [0.0]))
    assert moved["position"] == pytest.approx(1.2 - cfg.ladder_crash) and "nessuno" in moved["why"]
    assert cfg.forage_prey == pytest.approx(0.15) and cfg.forage_abundance == 1.0


def test_the_top_is_the_top():
    cfg = forage(ladder="scala", **ladder.at(ladder.TOP))
    assert ladder.update(cfg, history(cfg, [0.9] * 10)) == {}


def test_without_a_ladder_or_in_the_controls_the_dials_never_move():
    for mode, x in (("", 0.7), ("basico", 0.0), ("complesso", ladder.TOP)):
        cfg = forage(ladder=mode, **ladder.at(0.7))
        ladder.prepare(cfg)
        assert ladder.position(cfg) == pytest.approx(x)
        assert ladder.update(cfg, history(cfg, [0.9] * 10)) == {}
    with pytest.raises(ValueError):
        ladder.prepare(forage(ladder="facile"))


def test_evolution_moves_the_ladder_logs_it_and_the_next_world_follows(tmp_path):
    e = Evolution(forage(ladder="scala", engine="numpy"), seed=3)
    e.world.done[:] = True  # tutti completano: la scala sale
    s = e.next_generation()
    assert s["position"] == 0.0 and s["dials"]["forage_prey"] == 0.0 and "luck" in s and "eaten" in s
    assert e.log[-1]["event"] == "scala" and e.log[-1]["generation"] == 1
    assert ladder.position(e.cfg) == pytest.approx(0.02)
    assert e.world.cfg is e.cfg and e.world.task.food_r < e.cfg.food_radius_easy  # il mondo nuovo segue
    e.save(tmp_path / "pop.npz")
    back = Evolution.load(forage(), tmp_path / "pop.npz")
    assert back.cfg.ladder == "scala" and ladder.position(back.cfg) == pytest.approx(0.02)
    assert not any(x["event"] == "cambio" for x in back.log)  # la scala non si registra due volte


def test_luck_compares_the_nearest_and_the_farthest_quarter():
    dist = np.arange(8.0)
    done = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    assert ladder.luck(dist, done) == {"near": 1.0, "far": 0.0}


def test_the_ladder_and_its_controls_run_together_in_one_batch():
    evos = [Evolution(forage(ladder=m, population=30), seed=1) for m in ("scala", "basico", "complesso")]
    stats = run_together(evos)
    assert [s["position"] for s in stats] == [0.0, 0.0, ladder.TOP]
    assert evos[2].world.task.n_pred == evos[2].cfg.predator_count_max


def test_the_level_exam_climbs_the_ladder_and_does_not_follow_the_run():
    assert battery(forage()) is LEVELS and battery(Config()) is not LEVELS
    pos = [ladder.position(Config(**{k: v for k, v in t.changes.items() if k in ladder.DIALS})) for t in LEVELS]
    assert len(LEVELS) == 8 and pos[0] == 0.0 and pos[-1] == ladder.TOP and np.all(np.diff(pos) > 0)
    assert all(t.changes["task"] == "cibo" and t.changes["n_hazards"] == 0 for t in LEVELS)


def test_a_level_exam_gives_a_profile():
    e = Evolution(forage(engine="numpy", population=6), seed=0)
    e.start_exam(arenas=1, clones=1, workers=1).run()
    r = e.collect_exam()
    assert r["version"] == LEVELS_VERSION and list(r["scores"]) == [t.key for t in LEVELS]
    assert r["lesion"] is None


def test_lesions_remove_a_part_and_leave_the_originals_alone():
    e = Evolution(forage(), seed=0)
    g = e.ctrl.brains.genome
    g.active[:, :2] = True
    cfg, lg = lesioned(e.cfg, g, "neuroni")
    assert cfg is e.cfg and not lg.active.any() and g is not lg
    cfg, lg = lesioned(e.cfg, g, "luce")
    assert (lg.r_sens[..., 0] == -30.0).all() and (lg.r_sens[..., 1:] == g.r_sens[..., 1:]).all()
    assert not (g.r_sens[..., 0] == -30.0).all()
    cfg, lg = lesioned(e.cfg, g, "plasticità")
    assert not cfg.plasticity and e.cfg.plasticity and lg is g
    assert "luce" in LESIONS
    with pytest.raises(ValueError):
        lesioned(e.cfg, g, "cuore")


def test_a_lesioned_exam_is_recorded_as_such():
    e = Evolution(forage(engine="numpy", population=4), seed=0)
    g = e.ctrl.brains.genome
    g.active[:, :2] = True  # i primordiali nascono senza neuroni interni
    e.ctrl.brains.set_genome(g)
    ex = e.start_exam(arenas=1, clones=1, workers=1, lesion="neuroni")
    assert not ex.genomes.active.any() and e.ctrl.brains.genome.active.any()
    ex.run()
    assert e.collect_exam()["lesion"] == "neuroni"


def test_a_lesion_exam_does_not_replace_the_scheduled_exam():
    evo = Evolution(forage(exam_every=5, engine="numpy"), 1)
    evo.generation = 10
    evo.exams = [{"generation": 10, "lesion": "luce"}]
    assert evo.exam_due()  # la lesione è un'altra misura: l'esame della corsa va fatto lo stesso
    evo.exams.append({"generation": 10, "lesion": None})
    assert not evo.exam_due()
