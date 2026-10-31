"""Lo sviluppo del cervello dai marcatori (evonets/brains/freeform/development.py)."""

import numpy as np

from evonets import Config
from evonets.arenas.generational.evolution import Evolution
from evonets.brains.freeform.genetics import crossover, mutate, reproduce
from evonets.brains.freeform import Brains, SpikingBrains
from evonets.brains.freeform import development as dev
from evonets.brains.freeform.genome import Genome, META_INDEX
from evonets.brains.freeform.geometry import brain_side, distances


def grown(n=60, seed=0, kind=Brains, **kw):
    cfg = Config(genome="sviluppo", **kw)
    b = kind(cfg, n, 2 if kind is Brains else 4, seed)
    return cfg, b


def test_without_affinity_no_synapse_grows():
    cfg, b = grown()
    g = b.genome
    g.m_table[:] = 0.0
    dev.develop(g, cfg, np.random.default_rng(0), Brains.GAIN)
    assert not g.conn.any() and not g.weights.any()


def test_connection_probability_is_canalized_and_weight_is_affinity_times_gain():
    cfg, b = grown(n=1)
    g = b.genome
    a = dev.affinity(g)
    side = brain_side(cfg, g.r_active, g.active, g.n_out)
    dist = distances(cfg, g.pos, g.r_y, g.n_out, side)
    k = g.rate("canal")
    floor = dev.sigmoid(-k * cfg.develop_threshold)[:, None, None]
    arriving = np.abs(a) * np.exp(-dist / g.rate("conn_range")[:, None, None])
    expected = np.maximum(dev.sigmoid(k[:, None, None] * (arriving - cfg.develop_threshold)) - floor, 0) / (1 - floor)
    t = np.arange(dist.shape[1])
    expected[:, t, g.n_rec + t] = cfg.net_self_prob * (1.0 - np.exp(-np.abs(a[:, t, g.n_rec + t])))
    many = Genome.concat([g] * 4000)
    dev.develop(many, cfg, np.random.default_rng(1), Brains.GAIN)
    assert np.allclose(many.conn.mean(axis=0), expected[0], atol=0.03)
    gain = np.exp(g.traits[0, :, Brains.GAIN])
    young = dev.young_strength(cfg, a, dist, g.n_rec, g.rate("conn_range"))[0]
    w = np.clip(young * gain[:, None], -cfg.synapse_max, cfg.synapse_max)
    assert np.allclose(many.weights[many.conn], np.broadcast_to(w, many.conn.shape)[many.conn])


def test_young_synapses_are_born_weak_and_the_wiring_does_not_change():
    """Una sinapsi nasce con forza che cresce da zero oltre la soglia, con il
    segno dell'affinità; quali collegamenti nascono non dipende da quanto sono
    forti (la canalizzazione resta la stessa)."""
    cfg = Config(genome="sviluppo")
    a = np.array([[[0.5, 1.1, 2.0, -3.0, 0.8]]])  # l'ultima sorgente è il bersaglio stesso
    near = np.zeros((1, 1, 5))
    w = dev.young_strength(cfg, a, near, 4, np.array([1e9]))[0, 0]
    assert np.allclose(w[:4], [0.0, 0.1, 1.0, -2.0]) and np.isclose(w[4], 0.8 * cfg.synapse_young)
    brains = [Brains(Config(genome="sviluppo", synapse_young=y), 50, 2, 4) for y in (0.5, 2.0)]
    assert np.array_equal(brains[0].genome.conn, brains[1].genome.conn)


def test_strong_attraction_always_grows_a_connection_and_weak_almost_never():
    """La canalizzazione: lontano dalla soglia lo sviluppo non tira a sorte."""
    cfg = Config(genome="sviluppo")
    attraction = np.array([[[0.0, 0.5, 2.0, 4.0, 0.0]]])  # l'ultima sorgente è il bersaglio stesso
    near, far_reach = np.full((1, 1, 5), 0.01), np.array([1e9])
    p = dev.connect_prob(cfg, attraction, near, 4, far_reach, np.array([cfg.develop_precision]))[0, 0]
    assert p[0] == 0.0 and p[1] < 0.03 and p[2] > 0.99 and p[3] > 0.999
    noisy = dev.connect_prob(cfg, attraction, near, 4, far_reach, np.array([0.5]))[0, 0]
    assert 0.2 < noisy[2] < 0.8  # con poca precisione anche le vie forti sono un terno al lotto


def test_identical_twins_share_most_of_their_brain():
    """Con la precisione di partenza un clone ritrova quasi tutto il cervello
    del genitore (senza canalizzazione era circa un terzo)."""
    cfg, b = grown(n=300, kind=SpikingBrains)
    g = b.genome
    twin = g.take(np.arange(len(g)))
    dev.develop(twin, cfg, np.random.default_rng(7), Brains.GAIN)
    assert (g.conn & twin.conn).sum() / g.conn.sum() > 0.8


def test_markers_follow_position_gradients():
    """Con un gradiente lungo x un marcatore è espresso di più (o di meno)
    andando dai sensi (bordo sinistro) ai motori (bordo destro), come Eph ed
    efrine."""
    cfg, b = grown(n=1)
    g = b.genome
    g.r_mark[:], g.n_mark[:], g.m_grad[:] = 0.0, 0.0, 0.0
    g.m_grad[0, 0] = [6.0, 0.0]
    e = dev.expression(g)
    senses, motors = e[0, : g.n_rec, 0], e[0, -g.n_out :, 0]
    assert (senses < 0.1).all() and (motors > 0.9).all()
    assert np.allclose(e[0, :, 1], 0.5)  # senza gradiente: uguale ovunque


def test_spiking_networks_take_the_sign_from_the_transmitting_neuron():
    cfg, b = grown(kind=SpikingBrains, n=40)
    links = b.links
    R, H = b.n_in, b.n_hidden
    inhib = b.inhibitory[:, None, :] & links[:, :, R : R + H]
    excit = ~b.inhibitory[:, None, :] & links[:, :, R : R + H]
    assert (b.w_eff[:, :, R : R + H][inhib] <= 0).all() and (b.w_eff[:, :, R : R + H][excit] >= 0).all()


def test_marker_duplication_does_not_change_any_affinity():
    cfg, b = grown(n=10)
    g = b.genome
    before = dev.affinity(g)
    for i in range(len(g)):
        k = int(np.flatnonzero(g.m_active[i])[0])
        f = int(np.flatnonzero(~g.m_active[i])[0])
        dev.duplicate_marker(g, i, k, f, 10**12 + i)
    assert np.allclose(dev.affinity(g), before)
    f = cfg.marker_initial
    assert (g.m_active.sum(axis=1) == f + 1).all() and (g.m_fam[:, f] == g.m_fam[:, 0]).all()


def permuted(g: Genome, order) -> Genome:
    """Lo stesso individuo con i posti dei marcatori in un altro ordine."""
    p = g.take(np.arange(len(g)))
    for name in ("r_mark", "n_mark"):
        setattr(p, name, getattr(g, name)[:, :, order].copy())
    for name in ("m_grad", "m_active", "m_id", "m_fam"):
        setattr(p, name, getattr(g, name)[:, order].copy())
    p.m_table = g.m_table[:, order][:, :, order].copy()
    return p


def test_markers_pair_by_identity_in_crossover_wherever_they_sit():
    """Padre = madre con i marcatori in posti diversi: comunque si incrocino,
    il figlio ha le stesse affinità della madre."""
    cfg, b = grown(n=1)
    g = b.genome
    g.m_active[:] = True  # tutti accesi: nessuno è "solo di un genitore"
    order = np.array([2, 0, 3, 1, 5, 4])
    pair = Genome.concat([g, permuted(g, order)])
    rng = np.random.default_rng(0)
    for _ in range(20):
        child = crossover(pair, np.array([0]), np.array([1]), rng, split=False)
        assert np.allclose(dev.affinity(child), dev.affinity(g))


def test_with_development_weights_and_connections_are_not_genes():
    cfg, b = grown(n=50)
    g = b.genome.take(np.arange(50))
    g.meta[:, META_INDEX["dup"]] = np.log(cfg.mutation_floor)  # niente duplicazioni (copiano anche il cervello)
    w, c = g.weights.copy(), g.conn.copy()
    marks = g.n_mark.copy()
    mutate(g, cfg, np.random.default_rng(0))
    assert np.array_equal(g.conn, c) and np.array_equal(g.weights, w)  # muta il genoma, non il cervello
    assert not np.array_equal(g.n_mark, marks)
    kids = reproduce(b.genome, 50, cfg, np.random.default_rng(1))
    a = dev.affinity(kids)
    assert not kids.conn[a == 0.0].any()  # i figli sono sviluppati dai loro marcatori


def test_brain_mode_fixes_the_markers_of_the_senses():
    cfg, b = grown(n=40, sim_mode="cervello")
    g = b.genome
    assert np.allclose(g.r_mark, g.r_mark[:1])  # uguali per tutti
    first = g.r_mark[0].copy()
    rng = np.random.default_rng(0)
    for _ in range(5):
        g = reproduce(g, 40, cfg, rng)
    for i in range(len(g)):  # ogni marcatore, e ogni sua copia, come nel set di sensi
        for k in range(g.n_markers):
            if g.m_fam[i, k] < first.shape[1]:
                assert np.allclose(g.r_mark[i, :, k], first[:, g.m_fam[i, k]])
    # come la retina, i sensi seguono i gradienti secondo l'altezza del loro nervo
    g.m_grad[:] = 5.0
    e = dev.expression(g)
    xy = dev.positions(g)[:, : g.n_rec] - 0.5
    drive = g.r_mark + np.einsum("nsd,nmd->nsm", xy, g.m_grad)
    assert np.allclose(e[:, : g.n_rec], dev.sigmoid(drive) * g.m_active[:, None, :])


def test_in_brain_mode_the_eyes_know_their_side():
    """Sensi fissi: l'occhio rosso destro (nervo a 0,6) esprime di più il
    marcatore della destra, il sinistro quello della sinistra, e chi nasce con
    un riflesso lo ha molto più spesso verso il lato dello stimolo che al
    contrario (geometria.BILATERAL: motori 2 = destra, 3 = sinistra)."""
    from evonets.brains.freeform.senses import SETS

    cfg, b = grown(n=1000, kind=SpikingBrains, sim_mode="cervello")
    g = b.genome
    names = [x.name for x in SETS[cfg.sensor_set]]
    left, right = names.index("occhio rosso sinistro"), names.index("occhio rosso destro")
    e = dev.expression(g)
    assert (e[:, right, 0] > e[:, left, 0]).mean() > 0.9 and (e[:, left, 1] > e[:, right, 1]).mean() > 0.9
    H = g.n_hidden
    w = np.abs(g.weights)
    toward = w[:, H + 2, right] + w[:, H + 3, left]
    away = w[:, H + 2, left] + w[:, H + 3, right]
    assert (toward > away + 0.1).mean() > 2.0 * (away > toward + 0.1).mean()  # alla partenza circa 22% contro 8%


def test_paired_senses_are_the_same_organ_and_neither_side_is_favoured():
    """Sensi fissi: sinistro e destro di una coppia hanno gli stessi marcatori
    propri (lo stesso organo) e alla nascita sono usati altrettanto spesso."""
    from evonets.brains.freeform.genome import live_links
    from evonets.brains.freeform.senses import SETS, organs

    cfg, b = grown(n=3000, kind=SpikingBrains, sim_mode="cervello")
    g = b.genome
    organ = organs(cfg)
    names = [x.name for x in SETS[cfg.sensor_set]]
    left, right = names.index("occhio rosso sinistro"), names.index("occhio rosso destro")
    assert organ[left] == organ[right] and organ[left] != organ[names.index("occhio giallo sinistro")]
    assert np.allclose(g.r_mark[:, left], g.r_mark[:, right])
    used = live_links(g.conn, g.active, g.r_active)[:, :, : g.n_rec].any(axis=1)
    for k in range(0, 8, 2):  # le quattro coppie: nasi, occhi rossi, gialli, blu
        assert abs(used[:, k].mean() - used[:, k + 1].mean()) < 0.04


def test_a_marker_is_not_more_fragile_because_others_exist():
    cfg = Config(genome="sviluppo", marker_slots=8, marker_initial=8, mut_dup=1e-5)
    rng = np.random.default_rng(0)
    lost = []
    for on in (2, 8):
        b = Brains(cfg, 4000, 2, 1)
        g = b.genome
        g.m_active[:, on:] = False
        g.meta[:, META_INDEX["toggle"]] = np.log(0.01)
        before = g.m_active.sum()
        mutate(g, cfg, rng)
        lost.append(1.0 - (g.m_active[:, :on]).sum() / before)
    assert abs(lost[0] - lost[1]) < 0.01 and abs(lost[0] - (1 - 0.99 ** dev.MARKER_GENE_LENGTH)) < 0.01


def test_development_runs_save_and_resume(tmp_path):
    cfg = Config(population=12, genome="sviluppo", task="cibo", n_hazards=0, net_type="spike-ei")
    e = Evolution(cfg, 0)
    for _ in range(2):
        e.run_generation()
    e.save(tmp_path / "pop.npz")
    back = Evolution.load(Config(), tmp_path / "pop.npz")
    assert back.cfg.genome == "sviluppo"
    g, h = e.ctrl.brains.genome, back.ctrl.brains.genome
    for f in Genome.FIELDS:
        assert np.array_equal(getattr(g, f), getattr(h, f), equal_nan=getattr(g, f).dtype.kind == "f"), f
    back.run_generation()


def test_saves_from_before_development_open_with_the_direct_genome(tmp_path):
    import json

    e = Evolution(Config(population=10), 0)
    e.save(tmp_path / "pop.npz")
    data = dict(np.load(tmp_path / "pop.npz"))
    saved = json.loads(str(data["config"]))
    del saved["genome"]
    data["config"] = np.array(json.dumps(saved))
    for k in list(data):
        if any(k.endswith(f) for f in Genome.MARKERS):
            del data[k]
    data["format"] = np.array(12)
    np.savez(tmp_path / "old.npz", **data)
    back = Evolution.load(Config(genome="sviluppo"), tmp_path / "old.npz")
    assert back.cfg.genome == "diretto" and back.ctrl.brains.genome.n_markers == 0


# --- Il grembo (brains/freeform/womb.py) ---------------------------------------------------

from evonets.brains.freeform import womb  # noqa: E402


def test_womb_waves_make_neighbouring_senses_fire_together():
    cfg = Config()
    r_y = np.array([[0.1, 0.15, 0.8, 0.5]])
    on = np.array([[True, True, True, False]])
    obs = womb.waves(cfg, np.repeat(r_y, 50, 0), np.repeat(on, 50, 0), np.random.default_rng(0))
    assert obs.shape == (int(round(cfg.prenatal_time / cfg.dt)), 50, 4)
    assert ((obs >= 0) & (obs <= 1)).all() and not obs[:, :, 3].any()  # i sensi spenti tacciono
    x = obs.reshape(-1, 4)
    near, far = np.corrcoef(x[:, 0], x[:, 1])[0, 1], np.corrcoef(x[:, 0], x[:, 2])[0, 1]
    assert near > 0.5 and far < 0.2


def test_newborns_come_out_of_the_womb_with_their_own_activity():
    for net in ("rate", "spike-ei"):
        e = Evolution(Config(population=20, genome="sviluppo", net_type=net, engine="numpy"), 0)
        b, g = e.ctrl.brains, e.ctrl.brains.genome
        on = b.target_on
        assert np.isfinite(g.born_avg[on]).all() and np.isfinite(g.born_bias).all()
        assert not np.allclose(g.born_avg[on], b.target_act[on])  # non più "si nasce con ā = a*"
        assert np.allclose(b.avg, np.where(np.isfinite(g.born_avg), g.born_avg, b.target_act))  # ogni vita parte da lì


def test_the_womb_gives_the_same_result_on_the_gpu():
    for net in ("rate", "spike-ei"):
        cfg = Config(population=16, genome="sviluppo", net_type=net, hebb_initial_rate=0.05, homeo_initial_rate=0.5)
        b = (Brains if net == "rate" else SpikingBrains)(cfg, 16, 2 if net == "rate" else 4, 0)
        obs = womb.waves(cfg, b.genome.r_y, b.genome.r_active, np.random.default_rng(1))
        rows = np.arange(4, 16)
        gpu = womb.run(b, rows, obs, "gpu")
        b.set_genome(b.genome)
        cpu = womb.run_numpy(b, rows, obs)
        for x, y in zip(gpu, cpu):
            assert np.allclose(x, y, atol=1e-3)
        assert not np.allclose(gpu[0], b._born_weights()[rows])  # con Hebb forte il grembo rifinisce i pesi


def test_only_newborns_go_through_the_womb_and_the_direct_genome_never_does():
    e = Evolution(Config(population=30, genome="sviluppo", engine="numpy"), 0)
    before = dict(zip(e.ids.tolist(), e.ctrl.brains.genome.born_avg.copy()))
    e.world.done[:10] = True  # dieci vincitori: sopravvivono
    e.next_generation()
    g = e.ctrl.brains.genome
    assert np.isfinite(g.born_avg[:, -1]).all()  # tutti sono nati: i motori hanno ā
    kept = [k for k, i in enumerate(e.ids.tolist()) if i in before]
    assert kept  # qualcuno è sopravvissuto
    for k in kept:  # i sopravvissuti tengono il loro stato alla nascita: non rinascono
        assert np.array_equal(g.born_avg[k], before[int(e.ids[k])], equal_nan=True)
    d = Evolution(Config(population=12, engine="numpy"), 0)
    assert np.isnan(d.ctrl.brains.genome.born_avg).all()


# --- Vedere l'identità ---------------------------------------------------------

def test_cells_with_the_same_markers_have_the_same_colour_and_type():
    cfg, b = grown(n=1, marker_slots=4)
    g = b.genome
    g.active[0, :4] = True
    g.n_mark[0, 0] = g.n_mark[0, 1] = [3.0, -3.0, 0.0, 0.0]  # due neuroni gemelli
    g.n_mark[0, 2] = [-3.0, 3.0, 0.0, 0.0]
    g.n_mark[0, 3] = [3.0, 3.0, 0.0, 0.0]
    g.m_grad[:] = 0.0
    colours = dev.identity(g, 0)
    R = g.n_rec
    assert np.allclose(colours[R], colours[R + 1]) and not np.allclose(colours[R], colours[R + 2])
    assert dev.cell_types(g, 0, R + np.arange(4)) == 3


def test_a_marker_and_its_copies_share_a_colour():
    cfg, b = grown(n=1)
    g = b.genome
    dev.duplicate_marker(g, 0, 0, 2, 10**12)
    c = dev.marker_colors(g, 0)
    assert np.allclose(c[0], c[2]) and not np.allclose(c[0], c[1])


def test_generation_statistics_count_markers_types_and_value():
    e = Evolution(Config(population=12, genome="sviluppo", engine="numpy"), 0)
    s = e.run_generation()
    assert s["markers"] == e.cfg.marker_initial and "types" in s and s["value"] == [1.0, 1.0]
    from evonets.brains.freeform.anatomy import Anatomy

    a = Anatomy(e.cfg, e.ctrl.brains.genome.take([0]))
    assert a.summary()["markers"] == e.cfg.marker_initial and a.color.shape == (a.n_nodes, 3)
    assert any("marcatori espressi" in line for line in a.describe(a.n_nodes - 1))


def test_founder_left_right_axis_wires_each_receptor_mostly_to_its_own_side():
    cfg, b = grown(n=2000, kind=SpikingBrains, net_type="spike-ei")
    g, H = b.genome, b.genome.n_hidden
    a = dev.affinity(g)[:, H:, :2]  # motori <- i due recettori di partenza (-30°, +30°)
    right, left = 2, 3
    assert np.median(a[:, left, 0]) > 2 * np.median(a[:, right, 0])
    assert np.median(a[:, right, 1]) > 2 * np.median(a[:, left, 1])
    w = g.weights[:, H:, :2]
    steer = (w[:, right, 1] - w[:, left, 1]) - (w[:, right, 0] - w[:, left, 0])
    assert (steer > 0).mean() > 2 * (steer < 0).mean() and (steer < 0).any()  # c'è variazione


def test_without_the_map_the_founder_markers_are_random():
    cfg, b = grown(n=2000, kind=SpikingBrains, net_type="spike-ei", marker_map_gradient=0.0)
    g, H = b.genome, b.genome.n_hidden
    a = dev.affinity(g)[:, H:, :2]
    assert abs(np.median(a[:, 3, 0]) - np.median(a[:, 2, 0])) < 0.2


def test_founder_tuning_is_neutral_across_individuals_and_shared_by_paired_receptors():
    cfg, b = grown(n=2000)
    mu = b.genome.r_mu  # (N, R, P)
    assert 0.2 < mu[:, 0, 0].std() and abs(mu[:, 0, 0].mean() - 0.5) < 0.05  # nessun colore privilegiato
    assert np.abs(mu[:, 0] - mu[:, 1]).mean() < 0.1  # i due recettori: copie dello stesso gene
