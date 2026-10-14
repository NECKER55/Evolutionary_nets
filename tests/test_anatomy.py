import numpy as np

from evonets import Config
from evonets.net import NeuralController
from evonets.net.anatomy import Anatomy, sizes


def blank(net_type="rate"):
    """Un individuo senza collegamenti, con tutti i neuroni interni accesi."""
    cfg = Config(net_type=net_type, net_hidden_max=4)
    g = NeuralController(cfg, 1).brains.genome.take([0])
    g.conn[:] = False
    g.active[:] = True
    return cfg, g


def link(g, n_in, src, dst, w=1.0):
    g.conn[0, dst - n_in, src] = True
    g.weights[0, dst - n_in, src] = w


def test_dead_branches_and_distance_from_the_senses():
    cfg, g = blank()
    n_in = 42
    h0, h1, h2, h3, thrust = n_in, n_in + 1, n_in + 2, n_in + 3, n_in + 4
    link(g, n_in, 0, h0)  # senso -> h0 -> h1 -> spinta
    link(g, n_in, h0, h1)
    link(g, n_in, h1, thrust)
    link(g, n_in, 5, h2)  # senso -> h2: non arriva ai motori
    link(g, n_in, h3, h3)  # h3 parla solo a se stesso
    a = Anatomy(cfg, g)
    assert a.useful[[0, h0, h1, thrust]].all()
    assert not a.useful[[5, h2, h3]].any()
    assert list(a.depth[[h0, h1, thrust, h2, h3]]) == [1, 2, 3, 1, -1]
    s = a.summary()
    assert (s["neurons"], s["useful_neurons"], s["links"], s["useful_links"], s["senses_used"], s["loops"]) \
        == (4, 2, 5, 3, 1, 1)
    neurons, links = sizes(g, n_in)
    assert neurons[0] == 4 and links[0] == 5


def test_switched_off_neurons_keep_their_genes_as_latent():
    cfg, g = blank()
    n_in = 42
    link(g, n_in, 0, n_in + 1, 0.5)
    link(g, n_in, n_in + 1, n_in + 4)
    g.active[0, 1] = False
    a = Anatomy(cfg, g)
    assert a.summary()["links"] == 0
    text = a.genome_text()
    assert "[n1] SPENTO" in text and "(latente)" in text and "+0.500" in text


def test_spiking_weights_take_their_sign_from_the_neuron_type():
    cfg, g = blank("spike-ei")
    n_in = 42
    g.traits[0, :, 2] = 1.0  # tutti eccitatori...
    g.traits[0, 0, 2] = -1.0  # ...tranne n0
    link(g, n_in, 3, n_in, 0.8)
    link(g, n_in, n_in, n_in + 4, 0.7)  # dal neurone inibitorio: segno negativo
    a = Anatomy(cfg, g)
    assert a.weight(n_in, n_in + 4) == -0.7
    assert a.summary()["inhibitory"] == 1
    assert any("inibitorio" in line for line in a.describe(n_in))
    assert np.isclose(a.weight(3, n_in), 0.8)
