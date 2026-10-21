import numpy as np

from evonets import Config
from evonets.evo import Evolution
from evonets.net import Brains, SpikingBrains
from evonets.net.brain import PLASTIC

R = 1
MOD = PLASTIC.index("mod")


def circuit(kind=Brains, mod_sign=+1.0, **kw):
    """Recettore r0 -> n0, e il modulatore n1 sempre attivo vicino a n0.
    Regola di n0: solo A (Hebb puro modulato)."""
    kw.setdefault("hebb_initial_rate", 0.0)  # qui si prova la plasticità modulata: hebbiano di base spento
    kw.setdefault("homeo_initial_rate", 0.0)  # e omeostasi spenta
    kw.setdefault("critical_initial_boost", 1.0)  # e nessun periodo critico
    cfg = Config(net_hidden_slots=2, receptor_slots=R, receptor_initial=R, plasticity_initial_rate=0.05, **kw)
    b = kind(cfg, 1, 2 if kind is Brains else 4, 0)
    g = b.genome
    base = kind.BASE_TRAITS
    g.weights[:] = 0.0
    g.conn[:] = False
    g.traits[:] = 0.0
    g.active[:] = True
    g.traits[0, :, base + MOD] = -1.0  # nessun modulatore...
    g.traits[0, 1, base + MOD] = 1.0  # ...tranne n1
    g.traits[0, 0, base + 1] = 1.0  # n0 impara con A = 1
    g.pos[0] = [[0.5, 0.5], [0.55, 0.5]]
    g.conn[0, 0, 0] = True  # r0 -> n0
    g.weights[0, 0, 0] = 0.5
    g.conn[0, 1, 0] = True  # r0 -> n1: il modulatore si accende con lo stesso stimolo
    g.weights[0, 1, 0] = 3.0 * mod_sign if kind is Brains else 3.0
    g.traits[0, 0, 0] = 0.0
    if kind is SpikingBrains:
        g.traits[0, :, 2] = 1.0  # tutti eccitatori...
        g.traits[0, 1, 2] = mod_sign  # ...il modulatore secondo il segno voluto
        g.traits[0, 1, 0] = 1.0  # corrente di fondo: scatta di continuo
        g.traits[0, :, 1] = -3.0  # neuroni rapidi
    b.set_genome(g)
    return b


def run(b, steps=200, value=1.0):
    for _ in range(steps):
        b.step(np.full((1, R), value))
    return b.w_eff[0, 0, 0]


def test_modulated_hebb_strengthens_a_synapse_used_while_m_is_positive():
    b = circuit()
    assert run(b) > 0.5 + 0.1


def test_negative_modulation_weakens_it():
    b = circuit(mod_sign=-1.0)
    assert run(b) < 0.5 - 0.1


def test_without_a_modulator_nothing_is_learned():
    b = circuit()
    g = b.genome
    g.traits[0, 1, Brains.BASE_TRAITS + MOD] = -1.0  # n1 non è più modulatore
    b.set_genome(g)
    assert run(b) == 0.5


def test_modulators_do_not_transmit():
    b = circuit()
    g = b.genome
    g.conn[0, 0, R + 1] = True  # n1 -> n0
    g.weights[0, 0, R + 1] = 2.0
    b.set_genome(g)
    assert not b.links[0, 0, R + 1]


def test_learning_is_forgotten_at_a_new_life_and_never_written_in_the_genome():
    b = circuit()
    run(b)
    assert b.w_eff[0, 0, 0] != 0.5 and b.weights[0, 0, 0] == 0.5  # il genoma non cambia
    b.reset_state()
    assert b.w_eff[0, 0, 0] == 0.5


def test_synapses_saturate_and_spiking_ones_keep_their_sign():
    b = circuit(synapse_max=0.8)
    assert np.isclose(run(b, 2000), 0.8)
    s = circuit(SpikingBrains)
    g = s.genome
    g.conn[0, 0, R + 0] = False
    g.traits[0, 0, SpikingBrains.BASE_TRAITS + 1] = -5.0  # regola che spinge verso il basso
    s.set_genome(g)
    run(s, 500)
    excit = s.dale[0] > 0
    assert (s.w_eff[0][:, excit] >= 0.0).all()  # un eccitatorio non diventa inibitorio


def test_spiking_modulators_learn_too():
    s = circuit(SpikingBrains)
    assert run(s) > 0.5


def test_learning_gives_the_same_result_on_the_racing_rows():
    from nets import rich
    fast, full = rich(30, seed=4, conn=0.4), rich(30, seed=4, conn=0.4)
    assert full.learns.any()
    rng = np.random.default_rng(1)
    rows = np.arange(0, 30, 2)
    for _ in range(50):
        obs = rng.uniform(0, 1, (30, full.n_in))
        a = fast.step_some(obs[rows], rows)
        b = full.step(obs)[rows]
        assert np.allclose(a, b)
    assert np.allclose(fast.w_eff[rows], full.w_eff[rows])


def test_old_saves_open_without_plasticity(tmp_path):
    e = Evolution(Config(population=10), 0)
    e.save(tmp_path / "pop.npz")
    data = dict(np.load(tmp_path / "pop.npz"))
    for k in list(data):
        if k.endswith("traits"):
            data[k] = data[k][:, :, :1]
    data["format"] = np.array(4)
    np.savez(tmp_path / "old.npz", **data)
    back = Evolution.load(Config(), tmp_path / "old.npz")
    assert back.ctrl.brains.traits.shape[2] == Brains.N_TRAITS
    # Niente modulatori (la plasticità modulata è spenta); l'hebbiano di base parte dal suo valore iniziale.
    assert not back.ctrl.brains.modulatory.any() and "plasticità" in back.log[-1]["note"]
    assert np.allclose(back.ctrl.brains.hebb, Config().hebb_initial_rate)


TARGET, HOMEO = Brains.BASE_TRAITS + PLASTIC.index("target"), Brains.BASE_TRAITS + PLASTIC.index("homeo")
HEBB = Brains.BASE_TRAITS + PLASTIC.index("hebb")


def test_saves_without_the_desired_activity_are_tuned_on_a_trial_life(tmp_path):
    """Un salvataggio del formato 9 non ha il gene dell'attività desiderata:
    aprendolo, a* di ogni neurone diventa la sua frequenza media in una vita
    di prova (la popolazione era in equilibrio), non 0,5 per tutti."""
    e = Evolution(Config(population=10, net_type="spike-ei"), 0)
    e.save(tmp_path / "pop.npz")
    data = dict(np.load(tmp_path / "pop.npz"))
    old = SpikingBrains.BASE_TRAITS + PLASTIC.index("hebb") + 1  # geni fino all'hebbiano
    for k in list(data):
        if k.endswith("traits"):
            data[k] = data[k][:, :, :old]
    data["format"] = np.array(9)
    np.savez(tmp_path / "old.npz", **data)
    back = Evolution.load(Config(), tmp_path / "old.npz")
    b = back.ctrl.brains
    motors = b.target_act[:, b.n_hidden:]  # i motori lavorano sempre: hanno un'attività vera
    assert not np.allclose(motors, 0.5) and ((motors >= 0.02) & (motors <= 0.98)).all()
    assert "tarata" in back.log[-1]["note"]
    assert back.world.n == 10 and back.world.t == 0.0
    again = Evolution.load(Config(), tmp_path / "old.npz")
    assert np.allclose(again.ctrl.brains.target_act, b.target_act)  # ripetibile


def freq(w):
    """Frequenza di n0 quando r0 = 1 gli arriva con il peso w (rete continua)."""
    return (1.0 + np.tanh(w)) / 2.0


def plain(weight=0.5, **kw):
    """r0 -> n0 senza modulatori e senza plasticità modulata."""
    b = circuit(**kw)
    g = b.genome
    g.traits[0, :, Brains.BASE_TRAITS + MOD] = -1.0
    g.traits[0, :, Brains.BASE_TRAITS] = -30.0
    g.weights[0, 0, 0] = weight
    b.set_genome(g)
    return b, g


def test_basic_hebbian_bcm_brings_activity_towards_the_desired_level_without_modulators():
    """Regola BCM: sopra la soglia mobile θ = ā²/a* la sinapsi si rafforza,
    sotto si indebolisce; a regime l'attività va verso a*. n0 parte da
    tanh(0,5) ≈ 0,46, cioè frequenza (1 + 0,46)/2 ≈ 0,73."""
    b, g = plain(hebb_initial_rate=0.05, homeo_tau=0.1)
    for gene, sign in ((-2.0, -1), (2.0, +1)):  # a* ≈ 0,12 e 0,88
        g.traits[0, :, TARGET] = gene
        b.set_genome(g)
        assert b.learns[0]
        w = run(b, steps=3000)
        assert np.sign(w - 0.5) == sign
        assert abs(freq(w) - b.target_act[0, 0]) < abs(freq(0.5) - b.target_act[0, 0])
    g.traits[0, :, HEBB] = -40.0  # il gene a zero la spegne
    b.set_genome(g)
    assert run(b) == 0.5


def test_basic_hebbian_on_inhibitory_synapses_follows_vogels():
    """Una sinapsi inibitoria si rafforza dove il neurone è più attivo del
    desiderato e si indebolisce dove lo è meno; resta sempre inibitoria."""
    for gene, stronger in ((-2.0, True), (4.0, False)):  # a* ≈ 0,12 e 0,98
        b, g = plain(weight=-0.5, hebb_initial_rate=0.05)
        g.traits[0, 0, 0] = 1.5  # n0 acceso dalla corrente di fondo: frequenza (1 + tanh(1)) / 2 ≈ 0,88
        g.traits[0, :, TARGET] = gene
        b.set_genome(g)
        w = run(b, steps=600)
        assert w <= 0 and (w < -0.5) == stronger  # al più si spegne: non diventa eccitatoria


def test_plasticity_changes_strength_never_the_kind_of_synapse():
    """Spinta forte verso zero (a* bassissimo, omeostasi e BCM forti): la
    sinapsi eccitatoria si indebolisce ma non diventa inibitoria; intanto il
    neurone abbassa anche la propria eccitabilità."""
    b, g = plain(hebb_initial_rate=1.0, homeo_initial_rate=50.0, homeo_tau=0.05)
    g.traits[0, :, TARGET] = -8.0
    b.set_genome(g)
    w = run(b, steps=2000)
    assert 0.0 <= w < 0.25 and b.bias_now[0, 0] < b.bias[0, 0]


def test_homeostatic_scaling_of_inhibitory_synapses_goes_the_other_way():
    """Neurone troppo zitto: le eccitatorie crescono, le inibitorie calano."""
    b, g = plain(weight=-0.5, homeo_initial_rate=1.0, homeo_tau=0.1)
    g.traits[0, 0, 0] = 0.8  # corrente di fondo: frequenza (1 + tanh(0,3)) / 2 ≈ 0,65
    g.traits[0, :, TARGET] = 2.0  # desidera 0,88
    b.set_genome(g)
    assert -0.5 < run(b, steps=600) <= 0.0


def test_homeostatic_scaling_brings_activity_towards_the_desired_level():
    """n0 riceve solo da r0, sempre a 1: con il peso 0,5 la sua attività è
    tanh(0,5) ≈ 0,46, frequenza ≈ 0,73. Se desidera meno, le sinapsi in entrata si indeboliscono;
    se desidera di più, si rafforzano; con la forza a zero non cambia nulla."""
    b, g = plain(homeo_initial_rate=1.0, homeo_tau=0.1)
    for gene, sign in ((-2.0, -1), (2.0, +1)):  # a* ≈ 0,12 e 0,88
        g.traits[0, :, TARGET] = gene
        b.set_genome(g)
        w = run(b, steps=600)
        assert np.sign(w - 0.5) == sign
        assert abs(freq(w) - b.target_act[0, 0]) < abs(freq(0.5) - b.target_act[0, 0])
    g.traits[0, :, HOMEO] = -40.0
    b.set_genome(g)
    assert run(b) == 0.5


def test_hebbian_trace_fades_unless_a_modulator_consolidates_it():
    """La traccia provvisoria (early-LTP) svanisce: senza modulatori, finito lo
    stimolo, la sinapsi torna verso il peso di nascita. Con un modulatore
    acceso durante l'apprendimento la traccia diventa stabile e resta."""
    def lesson(modulated):
        b = circuit(hebb_initial_rate=0.5, early_initial_tau=1.0, capture_rate=5.0, homeo_tau=0.1)
        g = b.genome
        g.traits[0, :, Brains.BASE_TRAITS] = -30.0  # niente plasticità modulata: il modulatore solo consolida
        g.traits[0, :, TARGET] = 2.0  # n0 desidera più attività di quella che ha: BCM rafforza
        if not modulated:
            g.traits[0, :, Brains.BASE_TRAITS + MOD] = -1.0
        b.set_genome(g)
        learned = run(b, steps=300)
        rest = run(b, steps=900, value=0.0)  # stimolo spento: la traccia non si rinforza più
        return learned, rest, float(np.abs(b.early[0, 0, 0]))
    learned, rest, trace = lesson(False)
    assert learned > 0.5 and rest - 0.5 < 0.2 * (learned - 0.5) and trace < 0.01
    learned_m, rest_m, _ = lesson(True)
    assert learned_m > 0.5 and rest_m - 0.5 > 0.8 * (learned_m - 0.5)


def test_strengthening_slows_down_near_the_maximum():
    b, g = plain(hebb_initial_rate=0.05, early_initial_tau=1e6, homeo_tau=0.1)
    g.traits[0, :, TARGET] = 6.0
    b.set_genome(g)
    w = run(b, steps=3000)
    assert 0.5 < w < b.cfg.synapse_max


def test_critical_period_makes_hebb_strong_early_and_closes_with_age():
    """Con B = 100 e τ_c = 1 s l'hebbiano dei primi secondi è molto più forte;
    un individuo che ha già vissuto (sopravvissuto) non torna bambino."""
    def learned(boost, lived=0.0):
        b, g = plain(hebb_initial_rate=0.002, critical_initial_boost=boost, critical_initial_tau=1.0,
                     homeo_tau=0.1)
        g.traits[0, :, TARGET] = 2.0  # BCM rafforza
        b.set_genome(g)
        b.lived[:] = lived
        b.reset_state()
        return run(b, steps=300) - 0.5

    adult, young, old = learned(1.0), learned(100.0), learned(100.0, lived=240.0)
    assert young > 5 * adult > 0
    assert np.isclose(old, adult, rtol=1e-3)
    b, _ = plain(critical_initial_boost=100.0, critical_initial_tau=1.0)
    for _ in range(60):
        b.step(np.ones((1, R)))
    assert np.isclose(b.clock[0], 1.0)  # l'età avanza di dt a ogni passo
