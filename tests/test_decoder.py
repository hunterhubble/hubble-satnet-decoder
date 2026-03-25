"""Tests for decode_signal with synthetic IQ data."""

import numpy as np

from fast_decoder import constants
from fast_decoder.chipset import get_chipset_stats, reset_chipset_stats
from fast_decoder.decoder import decode_signal
from fast_decoder.demod import build_chan_mask, demod_one_symbol, interp_peak


def test_decode_signal_returns_triple():
    """decode_signal on pure noise should return three empty lists."""
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(constants.SAMPLE_RATE) +
             1j * rng.standard_normal(constants.SAMPLE_RATE)).astype(np.complex64) * 0.001
    packets, detections, attempts = decode_signal(noise)
    assert isinstance(packets, list)
    assert isinstance(detections, list)
    assert isinstance(attempts, list)


def test_decode_signal_noise_no_detections():
    """Pure white noise should produce no detections."""
    rng = np.random.default_rng(123)
    noise = (rng.standard_normal(constants.SAMPLE_RATE) +
             1j * rng.standard_normal(constants.SAMPLE_RATE)).astype(np.complex64) * 0.001
    packets, _, _ = decode_signal(noise)
    assert len(packets) == 0


def test_build_chan_mask_shape():
    mask = build_chan_mask(F0=1000.0, synth_res_val=488.28125)
    assert mask.shape == constants.fft_freqs.shape
    assert mask.dtype == np.bool_


def test_interp_peak_exact():
    """When the peak is perfectly centred the interpolated freq should match."""
    freqs = np.arange(10, dtype=float) * 100
    psd = np.zeros(10)
    psd[5] = 1.0
    result = interp_peak(psd, 5, freqs)
    assert abs(result - 500.0) < 1e-6


def test_chipset_stats_reset():
    reset_chipset_stats()
    stats = get_chipset_stats()
    assert len(stats) == 0
