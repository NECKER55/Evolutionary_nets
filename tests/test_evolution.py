import numpy as np

from evonets import Config
from evonets.evo import Evolution
from evonets.evo.evolution import NORMAL, ONE_PARENT, RANDOM, RESTART
from evonets.net.brain import Genome, pad

FIELDS = Genome.FIELDS
# Tutti i tassi al minimo e nessuna deriva dei tassi stessi: copie esatte.
NO_MUTATION = dict(mut_weight_prob=1e-12, mut_conn_add=1e-12, mut_conn_del=1e-12, mut_neuron_toggle=1e-12,
                   mut_dup=1e-12, mut_pos_prob=1e-12, mut_rec_drift=1e-12, mutation_floor=1e-12)


def evo(seed=0, **kw):
    return Evolution(Config(population=30, net_hidden_slots=6, **kw), seed)


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


def test_a_single_parent_has_ordinary_children_no_rescue():
    """Con un solo genitore nessun "soccorso": i figli mutano come sempre."""
    e = evo()
    parent = e.ctrl.brains.genome.weights[9].copy()
    s = finish(e, [9])
    assert s["event"] == ONE_PARENT and s["survivors"] == 1
    d = distance_from(e.ctrl.brains.genome, parent)
    assert d[0] == 0.0  # il genitore sopravvive identico
    first, second = d[1:15], d[15:]
    assert 0.5 < np.median(second) / np.median(first) < 2.0  # nessuna metà con mutazioni rinforzate


def same_as(genome, i, ref) -> bool:
    """L'individuo i è `ref` (genoma di una riga), a meno dei posti spenti
    aggiunti nel frattempo dalla crescita dei posti."""
    return equal(individual(genome, i), individual(pad(ref, genome.n_rec, genome.n_hidden), 0))


def test_by_default_a_parent_lives_as_long_as_it_keeps_completing_the_task():
    e = evo(crossover_rate=0.0)
    champion = e.ctrl.brains.genome.take([5])
    where = 5
    for generation in range(1, 9):
        assert finish(e, [where])["survivors"] == 1
        where = 0  # i sopravvissuti stanno in testa
        assert same_as(e.ctrl.brains.genome, 0, champion)
        assert e.age[0] == generation
    finish(e, [3])  # stavolta fallisce: sparisce, resta solo chi ha completato
    assert not same_as(e.ctrl.brains.genome, 0, champion)
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
    b = Evolution.load(Config(), path)
    assert b.cfg.population == 30 and b.generation == 3 and b.history == a.history
    # Dal salvataggio in poi le due evoluzioni vedono le stesse arene e si
    # comportano allo stesso modo (il sorteggio dei figli riparte, quindi si
    # confronta la generazione in corso, non le successive).
    while not a.finished:
        a.step()
        b.step()
    assert np.array_equal(a.world.done, b.world.done) and np.array_equal(a.world.pos, b.world.pos)


def test_load_restores_the_saved_slots_and_genomes(tmp_path):
    a = evo()
    a.ctrl.brains.genome.active[0, :] = True  # uno usa tutti i posti: alla prossima generazione crescono
    a.ctrl.brains.set_genome(a.ctrl.brains.genome)
    finish(a, [0, 1])
    assert a.ctrl.brains.n_hidden > 6
    a.save(tmp_path / "pop.npz")
    cfg = Config(net_hidden_slots=9)
    b = Evolution.load(cfg, tmp_path / "pop.npz")
    assert cfg.net_hidden_slots == 6 and b.ctrl.brains.n_hidden == a.ctrl.brains.n_hidden
    assert all(np.array_equal(getattr(a.ctrl.brains.genome, f), getattr(b.ctrl.brains.genome, f)) for f in FIELDS)


def test_manual_override_replaces_only_that_agents_commands():
    a, b = evo(seed=5, n_hazards=0), evo(seed=5, n_hazards=0)
    for _ in range(30):
        a.step()
        b.step((3, np.array([1.0, 1.0])))
    same = np.isclose(a.world.pos, b.world.pos).all(axis=1)
    assert not same[3] and same[np.arange(30) != 3].all()


def test_each_generation_with_winners_keeps_its_oldest_winner_as_champion(tmp_path):
    e = evo()
    finish(e, [4])  # gen. 0: un vincitore, età 0
    finish(e, [])  # gen. 1: nessuno, nessun campione
    oldest = individual(e.ctrl.brains.genome, 2)
    e.age[2] = 5
    finish(e, [1, 2, 3])  # gen. 2: vince il più anziano
    assert [(g, a) for g, a, _, _ in e.champions] == [(0, 0), (2, 5)]
    assert equal(individual(e.champions[1][2], 0), oldest)
    e.save(tmp_path / "pop.npz")
    b = Evolution.load(Config(), tmp_path / "pop.npz")
    assert [(g, a) for g, a, _, _ in b.champions] == [(0, 0), (2, 5)]
    assert equal(individual(b.champions[1][2], 0), oldest)


def test_a_trial_tests_one_network_on_fixed_arenas_and_is_reproducible():
    from evonets.evo.trial import Trial

    e = evo()
    g = e.ctrl.brains.genome.take([3])
    results = []
    for _ in range(2):
        t = Trial(e.cfg, g, arenas=3, clones=10)
        while not t.finished:
            t.run(1.0)
        assert len(t.completed) == 3 and all(0 <= c <= 10 for c in t.completed)
        assert len(t.times) == sum(t.completed)
        results.append((t.completed, t.times))
    assert results[0] == results[1]
    assert (e.ctrl.brains.genome.weights[3] == g.weights[0]).all()  # la rete provata non cambia


def test_individuals_keep_their_number_while_they_survive():
    e = evo()
    first = e.ids[[4, 9]].copy()
    finish(e, [4, 9])
    assert list(e.ids[:2]) == list(first)  # i sopravvissuti stanno in testa
    assert len(set(e.ids.tolist())) == 30 and e.ids[2:].min() >= 30  # i figli hanno numeri nuovi
    o = e.oldest()
    assert o["age"] == 1 and o["id"] in first and o["born"] == 0
    assert (e.born[2:] == 1).all()  # i figli nascono nella generazione che stanno per correre
    assert e.history[-1]["oldest_id"] in range(30)


def test_final_selection_keeps_only_who_wins_every_round_and_makes_no_children(tmp_path):
    e = evo()
    finish(e, [1, 2, 3, 4])
    finalists = e.ids[:4].copy()
    assert e.start_selection(3) == 4 and e.selecting
    assert e.world.n == 4 and len(e.ctrl.brains.genome) == 4
    finish_round = lambda winners: (e.world.done.__setitem__(slice(None), False),
                                    e.world.done.__setitem__(list(winners), True), e.next_selection_round())[-1]
    assert finish_round([0, 2, 3]) == {"round": 1, "entered": 4, "passed": 3}
    assert list(e.ids) == list(finalists[[0, 2, 3]])
    finish_round([1, 2])
    assert list(e.ids) == list(finalists[[2, 3]]) and e.selecting
    finish_round([])  # fallisce tutti: restano i due della prova prima, e finisce
    assert list(e.ids) == list(finalists[[2, 3]]) and not e.selecting
    assert e.selection["note"] and e.generation == 1  # nessuna generazione nuova
    assert e.oldest()["born"] == 0 and e.oldest()["age"] == 3  # la nascita non cambia, le prove si contano
    # Salvati e ricaricati, ripartono in due e la generazione dopo torna piena.
    e.save(tmp_path / "sel.npz")
    b = Evolution.load(Config(), tmp_path / "sel.npz")
    assert b.world.n == 2 and b.cfg.population == 30 and list(b.ids) == list(e.ids)
    b.run_generation()
    assert b.world.n == 30 and len(b.ctrl.brains.genome) == 30


def test_watching_does_not_change_the_result():
    """Il viewer disegna i sensi leggendo l'ultima lettura (`ctrl.obs`), senza
    rileggerli: rileggerli consumerebbe il rumore. La corsa resta identica."""
    plain, watched = evo(seed=7), evo(seed=7)
    for _ in range(3):
        plain.run_generation()
    for _ in range(3):
        while not watched.finished:
            _ = watched.ctrl.obs.copy(), watched.ctrl.brains.state.copy()  # come fa il disegno
            watched.step()
        watched.next_generation()
    assert plain.history == watched.history
    assert np.array_equal(plain.ctrl.brains.genome.weights, watched.ctrl.brains.genome.weights)


def test_the_champion_is_the_oldest_winner_and_on_a_tie_the_first_born():
    e = evo()
    e.age[[3, 8, 12]] = [2, 5, 5]
    e.ids[[8, 12]] = [40, 7]  # #7 è nato prima di #40
    finish(e, [3, 8, 12])
    assert e.champions[-1][3] == 7 and e.champions[-1][1] == 5


def test_a_trial_can_test_a_group_and_tells_each_network_apart():
    from evonets.evo.trial import Trial

    e = evo()
    g = e.ctrl.brains.genome.take([1, 2, 3])
    t = Trial(e.cfg, g, arenas=2, clones=5)
    while not t.finished:
        t.run(1.0)
    assert t.wins.shape == (3,) and t.wins.sum() == sum(t.completed)
    assert ((t.rates() >= 0) & (t.rates() <= 1)).all()
