"""Tests for protocol constants and derived values."""

import numpy as np

from hubble_satnet_decoder import constants


def test_default_sample_rate():
    assert constants.SAMPLE_RATE == 781_250


def test_samples_per_symbol():
    expected = int(8e-3 * 781_250)  # 6250
    assert constants.samples_per_symbol == expected


def test_fft_freqs_length():
    assert len(constants.fft_freqs) == constants.samples_per_symbol


def test_slot_samples_contains_both_versions():
    assert -1 in constants.slot_samples
    assert 1 in constants.slot_samples


def test_slot_gap_vneg1():
    gap = int(1.6e-3 * constants.SAMPLE_RATE)
    assert constants.slot_samples[-1]["gap"] == gap
    assert constants.slot_samples[-1]["slot"] == constants.samples_per_symbol + gap


def test_slot_gap_v1():
    gap = int(0.8e-3 * constants.SAMPLE_RATE)
    assert constants.slot_samples[1]["gap"] == gap
    assert constants.slot_samples[1]["slot"] == constants.samples_per_symbol + gap


def test_templates_built():
    assert -1 in constants.templates
    assert 1 in constants.templates
    for ver in (-1, 1):
        tmpl = constants.templates[ver]
        assert "uint8" in tmpl
        assert tmpl["uint8"].dtype == np.uint8
        assert tmpl["uint8"].shape[0] == constants.TEMPLATE_FREQ_BINS
        assert tmpl["duration_s"] > 0


def test_device_channel_spacing_silabs():
    sr = constants.SYNTH_RES["silabs"]
    expected = round(constants.CHANNEL_SPACING / sr) * sr
    assert constants.DEVICE_CHANNEL_SPACING["silabs"] == expected


def test_device_channel_spacing_nordic():
    sr = constants.SYNTH_RES["nordic"]
    expected = round(constants.CHANNEL_SPACING / sr) * sr
    assert constants.DEVICE_CHANNEL_SPACING["nordic"] == expected


def test_preamble_code_v1():
    assert constants.PREAMBLE_CODE_V1 == [63, 0, 63, 0, 63, 0, 63, 63]
    assert len(constants.on_indices_v1) == 5
    assert len(constants.off_indices_v1) == 3


def test_hopping_seqs():
    assert len(constants.HOPPING_SEQS) == 4
    for seq in constants.HOPPING_SEQS:
        assert len(seq) == constants.NUM_CHANNELS
        assert sorted(seq) == list(range(constants.NUM_CHANNELS))


def test_rs_arrays_same_length():
    assert len(constants.RS_K_V1) == len(constants.RS_N_V1)
    assert len(constants.RS_K_VNEG1) == len(constants.RS_N_VNEG1)


def test_configure_changes_sample_rate():
    constants.configure(1_000_000)
    assert constants.SAMPLE_RATE == 1_000_000
    assert constants.samples_per_symbol == int(8e-3 * 1_000_000)
    assert len(constants.fft_freqs) == constants.samples_per_symbol
    # Restore
    constants.configure(constants.DEFAULT_SAMPLE_RATE)
    assert constants.SAMPLE_RATE == 781_250
