"""L'elenco dei salvataggi nel menu (evonets/viz/launcher.py): riepilogo,
raggruppamento per giorno e filtro, senza aprire finestre."""

from datetime import date, datetime

from evonets import Config
from evonets.evo import Evolution
from evonets.viz.launcher import SaveSummary, day_label, grouped, summarize


def test_days_are_named_like_people_say_them():
    today = date(2026, 10, 23)
    assert day_label(today, today) == "Oggi, 23 ottobre"
    assert day_label(date(2026, 10, 22), today) == "Ieri, 22 ottobre"
    assert day_label(date(2026, 10, 20), today) == "martedì 20 ottobre"
    assert day_label(date(2025, 12, 31), today) == "mercoledì 31 dicembre 2025"


def test_saves_are_grouped_by_day_newest_first_and_filtered_by_words():
    x = lambda name, when, facts: SaveSummary(name, when, name, facts)
    saves = [x("a", datetime(2026, 10, 22, 9, 0), "genoma diretto · a impulsi"),
             x("b", datetime(2026, 10, 23, 8, 0), "genoma sviluppo · a impulsi"),
             x("c", datetime(2026, 10, 23, 18, 0), "genoma sviluppo · rete continua")]
    groups = grouped(saves, date(2026, 10, 23))
    assert [label for label, _ in groups] == ["Oggi, 23 ottobre", "Ieri, 22 ottobre"]
    assert [s.name for s in groups[0][1]] == ["c", "b"]
    assert [s.name for _, items in grouped(saves, date(2026, 10, 23), "sviluppo impulsi") for s in items] == ["b"]


def test_a_real_save_is_summarized_without_loading_the_brains(tmp_path):
    e = Evolution(Config(population=12, genome="sviluppo", task="cibo", n_hazards=0, net_type="spike-ei",
                         ladder="scala", engine="numpy"), 0)
    e.run_generation()
    e.save(tmp_path / "pop.npz")
    s = summarize(tmp_path / "pop.npz")
    assert not s.error and "gen. 1" in s.facts and "12 ind." in s.facts and "genoma sviluppo" in s.facts
    assert "a impulsi" in s.facts and "cibo, scala 0,00" in s.facts
    (tmp_path / "rotto.npz").write_bytes(b"non un salvataggio")
    assert summarize(tmp_path / "rotto.npz").error
