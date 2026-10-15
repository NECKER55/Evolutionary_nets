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
    assert not back.ctrl.brains.learns.any() and "plasticità" in back.log[-1]["note"]
