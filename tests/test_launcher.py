"""L'elenco dei salvataggi nel menu (evonets/viz/launcher.py): riepilogo,
raggruppamento per giorno e filtro, senza aprire finestre."""

from datetime import date, datetime

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from evonets import Config
from evonets.arenas.generational.evolution import Evolution
from evonets.viz.launcher import Launcher, SaveSummary, day_label, grouped, summarize
from evonets.viz.widgets import Choice


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


def menu(**defaults):
    import pygame

    pygame.font.init()
    pygame.display.init()

    class App:
        font = pygame.font.SysFont("monospace", 15)
        small = pygame.font.SysFont("monospace", 12)
        big = pygame.font.SysFont("monospace", 24, bold=True)

    return Launcher(App(), defaults), pygame.Surface((1440, 700)), pygame


def test_every_option_explains_itself_and_every_value_too():
    m, _, _ = menu()
    for task in ("bersaglio", "pile", "mondo3d"):
        m.task.select(task)
        for field in m._visible():
            assert len(field.help) > 20, field.label
            if isinstance(field, Choice):
                for value, _ in field.options:
                    assert field.details.get(value), (field.label, value)


def test_the_explanation_changes_with_the_value():
    m, _, _ = menu()
    m.net.select("rate")
    continuous = m.net.explain()
    m.net.change(1)
    assert m.net.value == "spike-ei" and m.net.explain() != continuous and "impulso" in m.net.explain()


@pytest.mark.parametrize("task", ["bersaglio", "cibo", "arena", "pile", "mondo3d"])
def test_every_page_fits_the_window_and_the_text_stays_beside_the_row(task):
    m, s, pygame = menu()
    m.task.select(task)
    for page in range(len(m._sections())):
        m._go(page)
        for _ in range(8):  # il valore più lungo di ogni riga
            m.draw(s)
            for field in m._sections()[page][1]:
                assert field.rect.bottom < 700 - 90, field.label
                assert field.rect.right < 700
            for field in m._sections()[page][1]:
                if isinstance(field, Choice) and field is not m.task:
                    field.index = (field.index + 1) % len(field.options)


def test_pages_change_by_click_and_keys_and_rows_flow_between_them():
    m, s, pygame = menu()
    key = lambda k: m.handle(pygame.event.Event(pygame.KEYDOWN, key=k, unicode="", mod=0))
    m.draw(s)
    m.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=m.tabs[1][0].center))
    assert m._page() == 1
    key(pygame.K_PAGEDOWN)
    assert m._page() == 2
    key(pygame.K_PAGEUP)
    key(pygame.K_UP)
    assert m._page() == 0
    key(pygame.K_F1)
    assert m.help_open
    m.draw(s)
    key(pygame.K_a)
    assert not m.help_open


def test_a_click_changes_only_the_row_on_the_page_in_view():
    """Le righe delle altre pagine tengono il riquadro di quando erano disegnate:
    un clic nello stesso punto non deve cambiare anche loro."""
    m, s, pygame = menu()
    click = lambda pos: m.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
    m.task.select("mondo3d")
    m.draw(s)
    m.p_good.rect, m.m_map.rect = m.task.rect.move(0, 40), m.task.rect.move(0, 80)
    click(m.tabs[1][0].center)
    m.draw(s)
    before = (m.task.value, m.p_good.value, m.m_map.value)
    agent, body = m.m_agent.index, m.m_body.index
    click((m.m_agent.rect.right - 5, m.m_agent.rect.centery))
    click((m.m_body.rect.right - 5, m.m_body.rect.centery))
    assert (m.m_agent.index, m.m_body.index) != (agent, body)
    assert (m.task.value, m.p_good.value, m.m_map.value) == before


def test_the_start_button_launches_the_3d_world_with_the_choices_made():
    pytest.importorskip("mujoco")
    from evonets.viz.mondo3d.view import Mondo3DScreen

    m, s, pygame = menu()
    shown = []
    m.app.show = shown.append
    m.task.select("mondo3d")
    m.m_agent.select("perfetto")
    m.m_body.select("due ruote e morso")
    m.m_beings.value = 2
    m.draw(s)
    m.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=m.start_btn.rect.center))
    assert isinstance(shown[0], Mondo3DScreen)
    assert shown[0].run.agent == "perfetto" and shown[0].run.w.body == "due ruote e morso" and shown[0].run.n == 2
    shown[0].close()


def test_every_agent_with_a_motor_can_choose_a_body_and_who_sees_only_knows_colours():
    m, s, _ = menu()
    m.task.select("mondo3d")
    m.p_good.select("grandezza")
    m.m_agent.select("perfetto")
    m.draw(s)
    assert not m.m_body.enabled and m.p_good.enabled and m.p_good.value == "grandezza"
    m.m_agent.select("vedente pronto")
    m.draw(s)
    assert m.m_body.enabled and not m.p_good.enabled and m.p_good.value == "colore"


def test_the_brain_page_opens_with_n_and_the_run_keeps_going():
    pytest.importorskip("mujoco")
    from evonets.viz.mondo3d.view import Mondo3DScreen

    m, s, pygame = menu()
    shown = []
    m.app.show = shown.append
    m.task.select("mondo3d")
    m.m_agent.select("ragionante")
    m.m_map.select("stanze")
    m.m_beings.value = 1
    m.draw(s)
    m.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=m.start_btn.rect.center))
    screen = shown[0]
    assert isinstance(screen, Mondo3DScreen) and screen.run.w.cfg.map == "stanze"
    screen.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n, unicode="n", mod=0))
    t = screen.run.t
    screen.update()
    screen.update()
    screen.draw(s)
    assert screen.page and screen.run.t > t
    screen.close()
