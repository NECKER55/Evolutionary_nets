"""La banca di embedding (evonets/brains/ragionante/bank.py)."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from evonets.brains.ragionante.bank import PATH, RAW, SIZE, Bank, fit, kinds, name, recognised, shrink


def clusters(rng, n=12, each=20):
    centres = rng.normal(size=(n, RAW))
    raw = np.repeat(centres, each, 0) + 0.2 * rng.normal(size=(n * each, RAW))
    return raw / np.linalg.norm(raw, axis=-1, keepdims=True), np.repeat(np.arange(n), each)


def test_the_reduction_keeps_what_tells_objects_apart():
    raw, kind = clusters(np.random.default_rng(0))
    mean, basis, kept = fit(raw)
    v = shrink(raw, mean, basis)
    assert v.shape == (len(raw), SIZE) and kept > 0.5
    assert np.isclose(np.linalg.norm(v, axis=-1), 1.0).all()
    assert recognised(v, kind) > 0.99


def test_a_being_is_the_last_kind_and_names_are_all_different():
    all_kinds = kinds()
    assert all_kinds[-1][0] == "essere" and len({name(k) for k in all_kinds}) == len(all_kinds)


@pytest.mark.skipif(not PATH.exists(), reason="serve la banca registrata")
def test_the_bank_finds_a_view_close_to_the_one_asked():
    bank = Bank()
    rng = np.random.default_rng(0)
    v = bank.pick(rng, 3, 0.2, 90.0)
    assert v.shape == (SIZE,) and np.isclose(np.linalg.norm(v), 1.0)
    assert ((bank.vectors == v).all(1) & (bank.kind == 3)).any()


@pytest.mark.skipif(not PATH.exists(), reason="serve la banca registrata")
def test_the_bank_tells_colours_and_forms_but_hardly_sizes():
    bank = Bank()
    table = kinds()
    colour = np.array([table[k][0] for k in bank.kind])
    form = np.array([table[k][1] for k in bank.kind])
    size = np.array([table[k][2] for k in bank.kind])
    assert recognised(bank.vectors, bank.kind, colour) > 0.98
    assert recognised(bank.vectors, bank.kind, form) > 0.95
    assert recognised(bank.vectors, bank.kind, size) < 0.8  # il ritaglio si ridimensiona: la grandezza si perde
