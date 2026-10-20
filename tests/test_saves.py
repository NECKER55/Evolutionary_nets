import json

import numpy as np

from evonets import Config
from evonets.evo import Evolution


def small(**kw):
    return Config(population=10, net_hidden_slots=4, **kw)


def test_resume_uses_the_saved_configuration(tmp_path):
    e = Evolution(small(energy_init=123.0, n_hazards=3), 0)
    e.save(tmp_path / "pop.npz")
    cfg = small()  # valori diversi da quelli della corsa
    back = Evolution.load(cfg, tmp_path / "pop.npz")
    assert cfg.energy_init == 123.0 and cfg.n_hazards == 3
    assert back.cfg is cfg
    assert [x["event"] for x in back.log] == ["inizio", "ripresa"]


def test_changes_during_a_run_are_logged_with_their_generation(tmp_path):
    e = Evolution(small(), 0)
    e.run_generation()
    e.cfg.wire_cost = 0.01  # come un ritocco con C durante la corsa
    e.run_generation()
    e.run_generation()
    changes = [x for x in e.log if x["event"] == "cambio"]
    assert len(changes) == 1
    assert changes[0]["generation"] == 1 and changes[0]["changes"] == {"wire_cost": [0.004, 0.01]}
    e.save(tmp_path / "pop.npz")
    back = Evolution.load(small(), tmp_path / "pop.npz")
    assert back.cfg.wire_cost == 0.01
    assert [x["event"] for x in back.log] == ["inizio", "cambio", "ripresa"]


def test_saved_config_is_complete_json(tmp_path):
    e = Evolution(small(), 0)
    e.save(tmp_path / "pop.npz")
    d = np.load(tmp_path / "pop.npz")
    assert int(d["format"]) == 7
    assert json.loads(str(d["config"]))["energy_init"] == e.cfg.energy_init
    assert json.loads(str(d["log"]))[0]["code"]


def test_a_run_saved_before_the_odors_resumes_without_them(tmp_path):
    import json

    cfg = Config(task="cibo", n_hazards=0, population=10, net_hidden_slots=4, engine="numpy")
    path = tmp_path / "vecchia.npz"
    Evolution(cfg, 3).save(path)
    with np.load(path) as d:
        data = dict(d)
    old = json.loads(str(data["config"]))
    for k in ("food_smell", "predator_smell", "forage_scent"):
        del old[k]  # come un salvataggio fatto prima degli odori
    data["config"] = json.dumps(old)
    np.savez_compressed(path, **data)
    evo = Evolution.load(Config(), path)
    assert evo.cfg.food_smell == 0.0 and evo.cfg.predator_smell == 0.0
    assert "food_smell" in evo.log[-1]["absent"]
