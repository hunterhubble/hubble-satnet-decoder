"""Tests for LFSR-7 whitening / de-scrambling."""

import numpy as np

from hubble_satnet_decoder.whitening import data_de_scrambling, lfsr7_symbols


def test_lfsr7_output_length():
    syms = lfsr7_symbols(seed=0, nsym=10)
    assert len(syms) == 10


def test_lfsr7_values_are_6bit():
    syms = lfsr7_symbols(seed=5, nsym=50)
    for s in syms:
        assert 0 <= s < 64


def test_lfsr7_different_seeds_differ():
    a = lfsr7_symbols(seed=0, nsym=20)
    b = lfsr7_symbols(seed=1, nsym=20)
    assert a != b


def test_lfsr7_deterministic():
    a = lfsr7_symbols(seed=7, nsym=30)
    b = lfsr7_symbols(seed=7, nsym=30)
    assert a == b


def test_data_de_scrambling_roundtrip():
    """Scrambling twice with the same channel restores the original."""
    original = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
    scrambled = data_de_scrambling(original, channel_num=3)
    recovered = data_de_scrambling(scrambled, channel_num=3)
    np.testing.assert_array_equal(recovered, original)


def test_data_de_scrambling_changes_data():
    original = np.arange(20, dtype=int)
    scrambled = data_de_scrambling(original, channel_num=5)
    assert not np.array_equal(scrambled, original)
