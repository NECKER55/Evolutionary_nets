import numpy as np

from evonets import Config
from evonets.evo import Evolution
from evonets.evo.evolution import NORMAL, ONE_PARENT, RANDOM, RESTART

FIELDS = ("weights", "traits", "conn", "active")
NO_MUTATION = dict(mut_weight_prob=0.0, mut_conn_add=0.0, mut_conn_del=0.0, mut_neuron_toggle=0.0)


def evo(seed=0, **kw):
    return Evolution(Config(population=30, net_hidden_max=6, **kw), seed)


def individual(genome, i):
    return [getattr(genome, f)[i].copy() for f in FIELDS]


def equal(a, b) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def finish(e, winners):
    """Chiude la generazione dichiarando chi ha completato il compito."""
    e.world.done[:] = False
    e.world.done[list(winners)] = True
    return e.next_generation()


def test_winners_survive_unchanged_and_the_rest_are_their_children():
    e = evo(crossover_rate=0.0, **NO_MUTATION)
    before = e.ctrl.brains.genome
    winners = [4, 11, 20]
    originals = [individual(before, i) for i in winners]
    s = finish(e, winners)
    after = e.ctrl.brains.genome
    assert s["event"] == NORMAL and s["completed"] == 3 and s["survivors"] == 3
    assert len(after) == 30 and e.generation == 1
    for k, orig in enumerate(originals):
        assert equal(individual(after, k), orig)  # i sopravvissuti stanno in testa
    assert list(e.age[:3]) == [1, 1, 1] and (e.age[3:] == 0).all()
    # Senza mutazione né incrocio ogni figlio è la copia di un vincitore: nessun
    # perdente ha lasciato traccia.
    for k in range(3, 30):
        assert any(equal(individual(after, k), o) for o in originals)


def test_new_generation_gets_a_new_arena_and_a_clean_start():
    e = evo()
    old_hazards, old_target = e.world.hazards.pos.copy(), e.world.task.target.copy()
    for _ in range(50):
        e.step()
    finish(e, [0, 1])
    w = e.world
    assert w.t == 0.0 and w.alive.all() and not w.done.any()
    assert (w.energy == e.cfg.energy_init).all()
    assert not np.allclose(w.hazards.pos, old_hazards) and not np.allclose(w.task.target, old_target)
    assert (e.ctrl.brains.state == 0.0).all()


def test_extinction_before_any_success_restarts_from_random_networks():
    e = evo()
    old = e.ctrl.brains.genome.weights.copy()
    s = finish(e, [])
    assert s["event"] == RANDOM and s["survivors"] == 0
    assert not np.array_equal(e.ctrl.brains.genome.weights, old)
    assert (e.age == 0).all()


def test_extinction_after_a_success_restarts_from_the_last_parents():
    e = evo(crossover_rate=0.0, **NO_MUTATION)
    winners = [2, 7]
    originals = [individual(e.ctrl.brains.genome, i) for i in winners]
    finish(e, winners)
    s = finish(e, [])
    assert s["event"] == RESTART and s["survivors"] == 0
    after = e.ctrl.brains.genome
    for k in range(30):  # tutti discendono dagli ultimi genitori, nessuna rete casuale
        assert any(equal(individual(after, k), o) for o in originals)
    assert (e.age == 0).all()


def distance_from(genome, parent_weights):
    return np.abs(genome.weights - parent_weights).mean(axis=(1, 2))


def test_single_parent_adds_children_with_stronger_variations():
    e = evo()
    parent = e.ctrl.brains.genome.weights[9].copy()
    s = finish(e, [9])
    assert s["event"] == ONE_PARENT and s["survivors"] == 1
    d = distance_from(e.ctrl.brains.genome, parent)
    n_strong = int(29 * e.cfg.rescue_share)
    normal, strong = d[1 : 30 - n_strong], d[30 - n_strong :]
    assert d[0] == 0.0  # il genitore sopravvive identico
    assert strong.min() > 3 * normal.max()


def test_two_parents_get_no_stronger_variations():
    e = evo(crossover_rate=0.0)
    g = e.ctrl.brains.genome
    parents = [g.weights[3].copy(), g.weights[9].copy()]
    finish(e, [3, 9])
    new = e.ctrl.brains.genome
    d = np.minimum(distance_from(new, parents[0]), distance_from(new, parents[1]))
    assert d[2:].max() < 3 * np.median(d[2:])  # un solo livello di mutazione


def test_by_default_a_parent_lives_as_long_as_it_keeps_completing_the_task():
    e = evo(crossover_rate=0.0)
    champion = individual(e.ctrl.brains.genome, 5)
    where = 5
    for generation in range(1, 9):
        assert finish(e, [where])["survivors"] == 1
        where = 0  # i sopravvissuti stanno in testa
        assert equal(individual(e.ctrl.brains.genome, 0), champion)
        assert e.age[0] == generation
    finish(e, [3])  # stavolta fallisce: sparisce, resta solo chi ha completato
    assert not equal(individual(e.ctrl.brains.genome, 0), champion)
    assert e.age[0] == 1 and (e.age[1:] == 0).all()


def test_with_an_age_limit_an_individual_dies_of_old_age_but_reproduces_that_last_time():
    e = evo(crossover_rate=0.0, max_age=3, **NO_MUTATION)
    champion = individual(e.ctrl.brains.genome, 5)
    assert finish(e, [5])["survivors"] == 1  # seconda vita
    assert finish(e, [0])["survivors"] == 1  # terza e ultima
    s = finish(e, [0])
    assert s["survivors"] == 0 and s["oldest"] == 2
    after = e.ctrl.brains.genome
    assert all(equal(individual(after, k), champion) for k in range(30))  # restano i figli
    assert (e.age == 0).all()


def test_generation_runs_to_the_end_and_is_reproducible():
    runs = []
    for _ in range(2):
        e = evo(seed=3)
        stats = [e.run_generation() for _ in range(3)]
        runs.append(stats)
        for s in stats:
            assert s["completed"] + s["dead"] == 30  # nessuno resta a metà
            assert s["duration"] <= e.cfg.energy_init / e.cfg.metabolism_basal + 0.1
    assert runs[0] == runs[1]


def test_life_max_caps_a_generation_when_metabolism_is_off():
    e = evo(metabolism_basal=0.0, metabolism_move=0.0, metabolism_turn=0.0,
            n_hazards=0, life_max=2.0)
    s = e.run_generation()
    assert 2.0 <= s["duration"] < 2.1


def test_save_and_load_resume_exactly_where_evolution_stopped(tmp_path):
    a = evo(seed=4)
    for _ in range(3):
        a.run_generation()
    path = tmp_path / "sub" / "pop.npz"
    a.save(path)
    b = Evolution.load(Config(net_hidden_max=6), path)
    assert b.cfg.population == 30 and b.generation == 3 and b.history == a.history
    # Dal salvataggio in poi le due evoluzioni vedono le stesse arene e si
    # comportano allo stesso modo (il sorteggio dei figli riparte, quindi si
    # confronta la generazione in corso, non le successive).
    while not a.finished:
        a.step()
        b.step()
    assert np.array_equal(a.world.done, b.world.done) and np.array_equal(a.world.pos, b.world.pos)


def test_load_refuses_networks_of_a_different_shape(tmp_path):
    a = evo()
    a.save(tmp_path / "pop.npz")
    try:
        Evolution.load(Config(net_hidden_max=9), tmp_path / "pop.npz")
    except ValueError as err:
        assert "forma" in str(err)
    else:
        raise AssertionError("doveva rifiutare il salvataggio")


def test_manual_override_replaces_only_that_agents_commands():
    a, b = evo(seed=5, n_hazards=0), evo(seed=5, n_hazards=0)
    for _ in range(30):
        a.step()
        b.step((3, np.array([1.0, 1.0])))
    same = np.isclose(a.world.pos, b.world.pos).all(axis=1)
    assert not same[3] and same[np.arange(30) != 3].all()
