import json

import numpy as np

from evonets import Config
from evonets.evo import Evolution


def small(**kw):
    return Config(population=10, net_hidden_slots=4, **kw)


def test_resume_keeps_the_run_state_and_plays_with_the_current_rules(tmp_path):
    e = Evolution(small(energy_init=123.0, n_hazards=3, net_type="spike-ei", task="cibo", ladder="scala",
                        forage_abundance=0.4, forage_need=0.4, exam_every=7), 0)
    e.save(tmp_path / "pop.npz")
    cfg = Config()  # il codice di adesso
    back = Evolution.load(cfg, tmp_path / "pop.npz")
    assert back.cfg is cfg
    # Lo stato della corsa viene dal salvataggio...
    assert (cfg.net_type, cfg.population, cfg.task, cfg.ladder, cfg.n_hazards, cfg.exam_every) == \
        ("spike-ei", 10, "cibo", "scala", 3, 7)
    assert cfg.forage_abundance == cfg.forage_need == 0.4 and cfg.net_hidden_slots == 4
    # ...le regole dal codice attuale, e il cambio è nel registro.
    assert cfg.energy_init == Config().energy_init
    assert [x["event"] for x in back.log] == ["inizio", "ripresa"]
    assert back.log[-1]["rules"] == {"energy_init": [123.0, Config().energy_init]}


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
    assert back.cfg.wire_cost == 0.004  # alla ripresa valgono di nuovo le regole del codice
    assert [x["event"] for x in back.log] == ["inizio", "cambio", "ripresa"]
    assert back.log[-1]["rules"] == {"wire_cost": [0.01, 0.004]}


def test_saved_config_is_complete_json(tmp_path):
    e = Evolution(small(), 0)
    e.save(tmp_path / "pop.npz")
    d = np.load(tmp_path / "pop.npz")
    assert int(d["format"]) == 12
    assert json.loads(str(d["config"]))["energy_init"] == e.cfg.energy_init
    assert json.loads(str(d["log"]))[0]["code"]


def test_a_run_saved_before_new_rules_existed_continues_with_them(tmp_path):
    cfg = Config(task="cibo", n_hazards=0, population=10, net_hidden_slots=4, engine="numpy")
    path = tmp_path / "vecchia.npz"
    Evolution(cfg, 3).save(path)
    with np.load(path) as d:
        data = dict(d)
    old = json.loads(str(data["config"]))
    for k in ("food_smell", "receptor_dup_split"):
        del old[k]  # come un salvataggio fatto prima di queste regole
    old["costante_sparita"] = 1.0  # e con una che il codice non conosce più
    data["config"] = json.dumps(old)
    np.savez_compressed(path, **data)
    evo = Evolution.load(Config(), path)
    assert evo.cfg.food_smell == Config().food_smell and evo.cfg.receptor_dup_split
    rules = evo.log[-1]["rules"]
    assert rules["food_smell"] == [None, Config().food_smell] and rules["receptor_dup_split"] == [None, True]
    assert "costante_sparita" in evo.log[-1]["note"]
    evo.run_generation()  # e la corsa continua
