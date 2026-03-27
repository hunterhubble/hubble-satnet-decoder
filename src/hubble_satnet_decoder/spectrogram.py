"""Spectrogram computation for IQ data chunks.

This module provides only the *computation* (scipy-based).  Rendering
(PIL / matplotlib) lives in the consuming application (e.g. sdr-docker).
"""

import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram

from . import constants


def compute_spec_chunk(iq_chunk: np.ndarray) -> np.ndarray:
    """Compute the visualisation spectrogram for an IQ chunk.

    Parameters
    ----------
    iq_chunk : complex64 1-D array
        Raw IQ samples (typically 0.5 s worth).

    Returns
    -------
    Sxx_dB : float32 2-D array  (freq × time)
        Power spectral density in dB.
    """
    chunk = iq_chunk - iq_chunk.mean()
    f, t, Sxx = scipy_spectrogram(
        chunk, fs=constants.SAMPLE_RATE, nperseg=constants.NFFT_VIS,
        noverlap=constants.NOVERLAP_VIS, return_onesided=False,
    )
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    dc_idx = len(f) // 2
    Sxx[dc_idx, :] = 0.0
    return (10.0 * np.log10(Sxx + 1e-12)).astype(np.float32)
