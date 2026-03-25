"""FSK demodulation helpers: channel masking, timing search, peak interpolation."""

import numpy as np

from . import constants


def build_chan_mask(F0: float, synth_res_val: float) -> np.ndarray:
    """Boolean mask for 64-FSK bandwidth around *F0*."""
    a = F0 - 1 * synth_res_val
    b = F0 + (constants.NUM_FSK_BINS + 1) * synth_res_val
    return (constants.fft_freqs >= min(a, b)) & (constants.fft_freqs <= max(a, b))


def interp_peak(psd: np.ndarray, bin_idx: int, freqs: np.ndarray) -> float:
    """Parabolic interpolation around an FFT peak for sub-bin accuracy."""
    n = len(psd)
    b = bin_idx
    if b <= 0 or b >= n - 1:
        return float(freqs[b])
    alpha = psd[b - 1]
    beta = psd[b]
    gamma = psd[b + 1]
    denom = alpha - 2 * beta + gamma
    if abs(denom) < 1e-30:
        return float(freqs[b])
    delta = 0.5 * (alpha - gamma) / denom
    bin_spacing = freqs[1] - freqs[0] if n > 1 else 1.0
    return float(freqs[b] + delta * bin_spacing)


def demod_one_symbol(
    sig_segment: np.ndarray,
    F0: float,
    synth_res_val: float,
    chan_mask: np.ndarray,
) -> tuple[int, float, float]:
    """Return ``(fsk_bin, peak_freq, peak_power)`` for one symbol slot."""
    spectrum = np.fft.fft(sig_segment)
    psd = np.abs(spectrum) ** 2
    psd_masked = psd.copy()
    psd_masked[~chan_mask] = 0.0
    peak_bin = int(np.argmax(psd_masked))
    peak_freq = interp_peak(psd_masked, peak_bin, constants.fft_freqs)
    fsk_bin = int(round((peak_freq - F0) / synth_res_val))
    fsk_bin = max(0, min(constants.NUM_FSK_BINS - 1, fsk_bin))
    return fsk_bin, peak_freq, float(psd_masked[peak_bin])


# Pre-computed timing search offsets (recomputed when configure() is called)
_TIMING_SEARCH: int = int(0.1e-3 * constants.SAMPLE_RATE)  # ±78 samples
_TIMING_STEP: int = max(1, (2 * _TIMING_SEARCH) // 9)       # 10 positions
_TIMING_STEPS: np.ndarray = np.arange(
    -_TIMING_SEARCH, _TIMING_SEARCH + 1, _TIMING_STEP
)


def _recompute_timing() -> None:
    """Refresh timing search offsets after :func:`constants.configure`."""
    global _TIMING_SEARCH, _TIMING_STEP, _TIMING_STEPS
    _TIMING_SEARCH = int(0.1e-3 * constants.SAMPLE_RATE)
    _TIMING_STEP = max(1, (2 * _TIMING_SEARCH) // 9)
    _TIMING_STEPS = np.arange(-_TIMING_SEARCH, _TIMING_SEARCH + 1, _TIMING_STEP)


def demod_best(
    signal: np.ndarray,
    s0: int,
    F0: float,
    synth_res_val: float,
    chan_mask: np.ndarray,
) -> tuple[int, float, int]:
    """Demod with ±0.1 ms timing search.

    Returns ``(fsk_bin, peak_freq, best_offset)``.
    """
    sym_len = constants.samples_per_symbol
    best_bin, best_freq, best_power, best_off = 0, 0.0, -1.0, 0
    for off in _TIMING_STEPS:
        s = s0 + int(off)
        if s < 0 or s + sym_len > len(signal):
            continue
        fsk_bin, peak_freq, peak_power = demod_one_symbol(
            signal[s: s + sym_len], F0, synth_res_val, chan_mask,
        )
        if peak_power > best_power:
            best_bin, best_freq, best_power, best_off = fsk_bin, peak_freq, peak_power, int(off)
    return best_bin, best_freq, best_off
