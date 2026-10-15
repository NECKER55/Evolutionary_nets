import numpy as np

from evonets import Config
from evonets.evo import Evolution, crossover, reproduce
from evonets.net import Brains, NeuralController, SpikingBrains
from nets import rich

N_IN = 10


def brains(n=60, seed=0, **kw):
    return rich(n, seed, R=N_IN, net="spike-ei", **kw)


def single(weights_in=0.0, bias=0.0, leak=0.0, substeps=1, gain=1.0):
    """Un individuo con un solo neurone interno (il n. 0) collegato all'ingresso 0."""
    cfg = Config(net_hidden_slots=2, receptor_slots=1, receptor_initial=1, spike_substeps=substeps,
                 spike_gain=gain)
    b = SpikingBrains(cfg, 1, 4, 0)
    b.weights[:] = 0.0
    b.traits[:] = 0.0  # tipo 0 = eccitatorio
    b.conn[:] = False
    b.active[:] = [[True, False]]
    b.conn[0, 0, 0] = True
    b.weights[0, 0, 0] = weights_in
    b.traits[0, 0, 0] = bias
    b.traits[0, :, 1] = leak
    b.compile()
    b.reset_state()
    return b


def test_commands_have_the_same_shape_and_range_as_the_other_network():
    b = brains()
    rng = np.random.default_rng(0)
    for _ in range(30):
        out = b.step(rng.uniform(-1, 1, (60, N_IN)))
        assert out.shape == (60, 2) and (np.abs(out) < 1.0).all()
    assert set(np.unique(b.spikes)) <= {0.0, 1.0}


def test_neuron_charges_up_fires_and_resets():
    b = single(weights_in=2.0)
    obs = np.ones((1, 1))
    charges, fired = [], []
    for _ in range(12):
        b.step(obs)
        charges.append(b.charge[0, 0])
        fired.append(b.spikes[0, 0])
    first = fired.index(1.0)
    assert first >= 2  # serve accumulare: non scatta al primo colpo
    assert all(a < c for a, c in zip(charges[: first - 1], charges[1:first]))  # sale
    assert charges[first] == 0.0  # dopo l'impulso si azzera
    assert sum(fired) >= 2  # e ricomincia


def test_weak_input_never_reaches_threshold_and_charge_leaks_away():
    b = single(weights_in=0.9)
    for _ in range(200):
        b.step(np.ones((1, 1)))
    assert b.spikes.sum() == 0.0 and 0.8 < b.charge[0, 0] < 1.0
    for _ in range(200):
        b.step(np.zeros((1, 1)))
    assert abs(b.charge[0, 0]) < 1e-6  # senza stimolo torna a riposo


def test_stronger_input_means_more_frequent_spikes():
    def rate(w):
        b = single(weights_in=w)
        return sum(b.step(np.ones((1, 1))) is not None and b.spikes[0, 0] for _ in range(300))
    assert rate(1.5) < rate(3.0) < rate(8.0)


def test_leak_gene_sets_how_fast_a_neuron_reacts():
    def steps_to_fire(leak):
        b = single(weights_in=2.0, leak=leak)
        for k in range(1, 500):
            b.step(np.ones((1, 1)))
            if b.spikes[0, 0]:
                return k
    assert steps_to_fire(-2.0) < steps_to_fire(0.0) < steps_to_fire(2.0)


def test_motors_are_antagonist_pairs_that_cannot_go_negative():
    b = single()
    h = b.n_hidden
    for motor in range(4):
        b.conn[0, h + motor, 0] = True
    b.weights[0, h:, 0] = [2.0, 0.0, -3.0, 1.0]  # avanti forte, destra inibito, sinistra attivo
    b.compile()
    for _ in range(100):
        out = b.step(np.ones((1, 1)))
    m = b.state[0, h:]
    assert (m >= 0.0).all() and m[2] == 0.0  # l'inibizione non spinge "al contrario"
    assert out[0, 0] > 0.9  # avanti - indietro
    assert out[0, 1] < -0.7  # destra - sinistra: gira a sinistra


def test_missing_neurons_never_fire():
    b = brains()
    rng = np.random.default_rng(1)
    total = np.zeros_like(b.spikes)
    for _ in range(50):
        b.step(rng.uniform(-1, 1, (60, N_IN)))
        total += b.spikes
    assert (total[~b.active] == 0.0).all() and total[b.active].sum() > 0.0


def test_reset_restores_behaviour_and_population_is_reproducible():
    a, b = brains(seed=3), brains(seed=3)
    obs = np.random.default_rng(2).uniform(-1, 1, (60, N_IN))
    first = a.step(obs).copy()
    assert np.array_equal(first, b.step(obs))
    for _ in range(5):
        a.step(obs)
    a.reset_state()
    assert np.array_equal(a.step(obs), first)


def test_leak_gene_is_inherited_and_mutated_with_the_rest_of_the_neuron():
    b = brains(20)
    cfg, g = b.cfg, b.genome
    assert g.traits.shape[2] == SpikingBrains.N_TRAITS == 9
    kids = reproduce(g.take([0]), 30, cfg, np.random.default_rng(0))
    assert (kids.traits[:, :, 1] != g.traits[0, :, 1]).any()
    b.set_genome(kids)
    assert np.isfinite(b.step(np.zeros((30, N_IN)))).all()


def test_net_type_selects_the_network_and_is_kept_by_save_and_load(tmp_path):
    assert type(NeuralController(Config(), 5).brains) is Brains
    assert type(NeuralController(Config(net_type="spike-ei"), 5).brains) is SpikingBrains
    e = Evolution(Config(net_type="spike-ei", population=20, net_hidden_slots=6), 1)
    e.run_generation()
    e.save(tmp_path / "pop.npz")
    cfg = Config(net_hidden_slots=6)  # il tipo viene dal file, non dalla configurazione
    loaded = Evolution.load(cfg, tmp_path / "pop.npz")
    assert cfg.net_type == "spike-ei" and type(loaded.ctrl.brains) is SpikingBrains
    assert np.array_equal(loaded.ctrl.brains.traits, e.ctrl.brains.traits)


# --- Tipi fissi: neuroni eccitatori o inibitori ---


def outgoing(b):
    """Pesi effettivi in uscita dai neuroni interni: (N, destinatari, H)."""
    return b.w_eff[:, :, N_IN : N_IN + b.n_hidden]


def test_about_a_fifth_of_neurons_are_born_inhibitory():
    assert 0.17 < brains(n=300).inhibitory.mean() < 0.23
    assert brains(spike_inhibitory_share=0.5).inhibitory.mean() > 0.4


def test_every_neuron_is_all_excitatory_or_all_inhibitory():
    b = brains(n=80)
    out = outgoing(b)
    inh = b.inhibitory[:, None, :]
    assert (out[np.broadcast_to(inh, out.shape)] <= 0.0).all()
    assert (out[np.broadcast_to(~inh, out.shape)] >= 0.0).all()
    assert (out < 0.0).any() and (out > 0.0).any()
    # La forza è il valore assoluto del peso scritto nel genoma.
    raw = b.weights[:, :, N_IN : N_IN + b.n_hidden]
    assert np.array_equal(np.abs(out), np.abs(raw) * b.links[:, :, N_IN : N_IN + b.n_hidden])


def test_motors_only_excite_and_senses_are_free_unless_restricted():
    b = brains()
    assert (b.w_eff[:, :, N_IN + b.n_hidden :] >= 0.0).all()
    senses = b.w_eff[:, :, :N_IN]
    assert (senses < 0.0).any() and (senses > 0.0).any()
    strict = brains(sensors_excitatory_only=True)
    assert (strict.w_eff[:, :, :N_IN] >= 0.0).all()


def test_flipping_the_type_gene_flips_all_outgoing_links_and_nothing_else():
    b = brains()
    who, neuron = np.argwhere(b.active & outgoing(b).any(axis=1))[0]
    before = b.w_eff[who].copy()
    b.traits[who, neuron, 2] *= -1.0
    b.compile()
    col = N_IN + neuron
    assert np.array_equal(b.w_eff[who][:, col], -before[:, col])
    others = np.arange(before.shape[1]) != col
    assert np.array_equal(b.w_eff[who][:, others], before[:, others])


def downstream_spikes(sender_inhibitory: bool) -> float:
    """Il neurone 0 scatta di continuo ed è collegato al neurone 1, che riceve
    anche uno stimolo proprio. Conta gli impulsi del neurone 1."""
    b = SpikingBrains(Config(net_hidden_slots=2, receptor_slots=1, receptor_initial=1, spike_gain=1.0), 1, 4, 0)
    b.weights[:] = 0.0
    b.traits[:] = 0.0
    b.conn[:] = False
    b.active[:] = True
    b.conn[0, 0, 0] = b.conn[0, 1, 0] = b.conn[0, 1, 1] = True  # ingresso->0, ingresso->1, 0->1
    b.weights[0, 0, 0] = 8.0
    b.weights[0, 1, 0] = 2.0
    b.weights[0, 1, 1] = 3.0  # sorgente 1 = neurone interno 0 (dopo l'unico ingresso)
    b.traits[0, 0, 2] = -1.0 if sender_inhibitory else 1.0
    b.traits[0, 1, 2] = 1.0
    b.compile()
    b.reset_state()
    total = 0.0
    for _ in range(100):
        b.step(np.ones((1, 1)))
        total += b.spikes[0, 1]
    return total


def test_inhibitory_neuron_quiets_its_target_and_excitatory_one_drives_it():
    assert downstream_spikes(sender_inhibitory=True) < downstream_spikes(sender_inhibitory=False)
    assert downstream_spikes(sender_inhibitory=True) == 0.0


def test_type_travels_with_the_neuron_in_crossover():
    g = brains().genome
    kids = crossover(g, np.zeros(40, dtype=int), np.ones(40, dtype=int), np.random.default_rng(0))
    kid_type = kids.traits[:, :, 2]
    for block in range(g.traits.shape[1]):
        from_mother = (kids.weights[:, block] == g.weights[0, block]).all(axis=1)
        assert (kid_type[from_mother, block] == g.traits[0, block, 2]).all()
        assert (kid_type[~from_mother, block] == g.traits[1, block, 2]).all()


def test_selectable_evolves_and_survives_save_and_load(tmp_path):
    cfg = Config(net_type="spike-ei", population=20, net_hidden_slots=6)
    assert type(NeuralController(cfg, 5).brains) is SpikingBrains
    e = Evolution(cfg, 1)
    s = e.run_generation()
    assert 0.0 <= s["inhibitory"] <= 1.0
    e.save(tmp_path / "pop.npz")
    cfg2 = Config(net_hidden_slots=6)
    loaded = Evolution.load(cfg2, tmp_path / "pop.npz")
    assert cfg2.net_type == "spike-ei"
    assert np.array_equal(loaded.ctrl.brains.w_eff, e.ctrl.brains.w_eff)
