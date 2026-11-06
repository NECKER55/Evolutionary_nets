"""Le sezioni L e W e il mondo a eventi."""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from evonets.brains.ragionante import events, memory
from evonets.brains.ragionante.events import BITE, HELD_OUT, LEVELS, NOTHING, PLEASURE, ideal, lives, remembered, table
from evonets.brains.ragionante.train_memory import train
from evonets.brains.ragionante.vision import EMBED, KINDS, Percept


def test_events_use_the_same_object_format_as_perception():
    assert events.EMBED == EMBED
    p = Percept.empty(1, 1)
    p.on[0, 0, 0], p.kind[0, 0, 0], p.size[0, 0, 0], p.dist[0, 0, 0] = True, KINDS.index("pila gialla"), 16.0, 70.0
    real = p.embedding(350.0)[0, 0, 0]
    fake = events.embed(jnp.array([[1, 1, 1]]), jax.random.PRNGKey(0))[0]
    assert real[:4] == pytest.approx(np.asarray(fake[:4]), abs=1e-6)
    assert real[4] == pytest.approx(float(fake[4]), abs=1e-6)


def test_only_a_bite_has_an_outcome_and_it_follows_the_rule():
    d = lives(jax.random.PRNGKey(0), 32, ("colore",))
    out, act, bits, flipped = (np.asarray(d[k]) for k in ("out", "act", "bits", "flipped"))
    assert ((out != NOTHING) == (act == BITE)).all()
    a = np.asarray(d["a"])[:, None]
    expected = (bits[..., 0] == a) ^ flipped
    assert ((out == PLEASURE) == (expected & (act == BITE))).all()
    assert flipped.any()  # gli scambi ci sono


def test_every_life_is_different_even_with_the_same_rule():
    d = lives(jax.random.PRNGKey(0), 64, ("colore",))
    assert len({np.asarray(b).tobytes() for b in d["bits"]}) == 64


def test_the_input_does_not_contain_the_answer():
    """Il bisogno è di prima dell'evento: con l'esito dentro, la rete batteva il limite teorico."""
    d = lives(jax.random.PRNGKey(1), 256, ("colore",))
    need, out = np.asarray(d["need"]), np.asarray(d["out"])
    assert (need[:, 0] == 0.5).all()
    change = np.diff(need, axis=1)
    pleasure = out[:, :-1] == PLEASURE  # l'esito di un evento si vede nel bisogno dell'evento dopo
    assert change[pleasure].mean() < change[~pleasure].mean()


def test_held_out_rules_are_really_out_of_training():
    assert not any(set(HELD_OUT) & set(names) for names in LEVELS.values())


def test_the_wide_family_contains_neither_held_out_rule_nor_its_opposite():
    from evonets.brains.ragionante.events import OBJECTS, TABLES

    rules = {tuple(int(v) for v in x) for x in np.asarray(TABLES)}
    near = tuple(OBJECTS[:, 2])
    either = tuple((OBJECTS[:, 0] | OBJECTS[:, 1]).tolist())
    assert len(rules) > 50
    for banned in (near, either):
        assert banned not in rules and tuple(1 - v for v in banned) not in rules


def test_the_limit_beats_the_simple_memory():
    d = lives(jax.random.PRNGKey(2), 256, LEVELS["2"])
    limit = events.bite_accuracy(ideal(d, LEVELS["2"]), d, 5)
    assert limit > 0.88 and limit > events.bite_accuracy(remembered(d), d, 5) + 0.1


def test_the_table_reads_like_a_life():
    one = jax.tree_util.tree_map(lambda x: x[0], lives(jax.random.PRNGKey(0), 1, ("colore",)))
    row = table(one)[0]
    assert row[0] == 1 and row[1] in ("rossa", "gialla") and row[4] in events.ACTIONS and row[5] in events.OUTCOMES


@pytest.mark.parametrize("variant", ["a", "b"])
def test_memory_is_empty_at_birth_and_weights_do_not_move_in_a_life(variant):
    p = memory.init(jax.random.PRNGKey(0), variant)
    assert not np.asarray(memory.blank(p)["h"]).any()
    one = jax.tree_util.tree_map(lambda x: x[0], lives(jax.random.PRNGKey(0), 1, ("colore",)))
    before = jax.tree_util.tree_map(np.asarray, p)
    memory.live(p, one)
    assert all((a == np.asarray(b)).all() for a, b in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(p)))


def test_fast_synapses_change_in_a_life_and_decay():
    p = memory.init(jax.random.PRNGKey(0), "b")
    s = memory.write(p, memory.blank(p), jnp.ones(memory.INPUT))
    assert np.abs(np.asarray(s["fast"])).max() > 0.0
    quiet = memory.write(p | {"gain": jnp.array(-50.0)}, s, jnp.zeros(memory.INPUT))
    assert np.abs(np.asarray(quiet["fast"])).sum() < np.abs(np.asarray(s["fast"])).sum()


def test_a_short_training_learns_the_first_level_and_needs_its_memory():
    p = train("a", LEVELS["1"], 260, 0, every=1000)
    d = lives(jax.random.PRNGKey(5), 512, LEVELS["1"])
    assert events.bite_accuracy(memory.forecast(p, d), d, 5) > 0.9
    assert events.bite_accuracy(memory.forecast(p, d, blind=True), d, 5) < 0.6


def test_this_section_knows_nothing_of_the_body():
    for name in ("memory.py", "events.py"):
        source = (Path(memory.__file__).parent / name).read_text()
        assert "wiring" not in source and "motor" not in source
