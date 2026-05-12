"""Protocol constants and derived parameters for the Hubble PHY decoder.

All sample-rate-dependent derived values are computed at module load using
``DEFAULT_SAMPLE_RATE``.  Call :func:`configure` with a different sample rate
to recompute them (e.g. for offline analysis at a different Fs).
"""

import numpy as np
import reedsolo as rs

# ---------------------------------------------------------------------------
# Sample rate
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE: int = 781_250  # 6.25 MHz / 8

SAMPLE_RATE: int = DEFAULT_SAMPLE_RATE

ADC_FULL_SCALE: float = 1.0

# ---------------------------------------------------------------------------
# Protocol constants (sample-rate independent)
# ---------------------------------------------------------------------------

SYMBOL_DURATION_S: float = 8e-3
PREAMBLE_LEN: int = 8
PREAMBLE_BITS: list[int] = [1, 0, 1, 0, 1, 0, 1, 1]

GAP_DURATIONS: dict[int, float] = {
    -1: 1.6e-3,  # v-1: 1.6 ms inter-symbol gap (OOK)
     1: 0.8e-3,  # v1:  0.8 ms inter-symbol gap (FSK)
}

NUM_FSK_BINS: int = 64

# v-1 specific
SYMBOLS_PER_PACKET_VNEG1: int = 32
DATA_LEN_VNEG1: int = SYMBOLS_PER_PACKET_VNEG1 - PREAMBLE_LEN  # 24
FREQ_STEP_VNEG1: float = 488.28125

# v1 Reed-Solomon block sizes: index 0 = header, 1-4 = PDU lengths
RS_K_V1: list[int] = [2, 13, 18, 25, 30]   # data symbols
RS_N_V1: list[int] = [6, 23, 30, 39, 46]   # total (data + parity)

# v-1 Reed-Solomon block sizes
RS_K_VNEG1: list[int] = [11, 13, 15, 17, 19, 21, 23, 25]
RS_N_VNEG1: list[int] = [21, 23, 27, 29, 33, 35, 39, 41]

NUM_HEADER_SYMS: int = RS_N_V1[0]  # 6
NUM_CHANNELS: int = 19
LO_CHANNEL: int = 9
CHANNEL_SPACING: float = 25_750.0  # Hz

SYNTH_RES: dict[str, float] = {
    "ti": 366.2119,
    "nordic": 488.28125,
    "silabs": 296.0,
    "esp": 400.0,
    "atmosic": 500.0,
}

# Actual per-device channel spacing: each synthesiser can only hop in
# integer multiples of its synth_res.  Silabs is a special case (protocol
# defines its spacing as exactly 25500 Hz).
DEVICE_CHANNEL_SPACING: dict[str, float] = {
    name: round(CHANNEL_SPACING / sr) * sr
    for name, sr in SYNTH_RES.items()
}

NUM_SYM_PER_HOP: int = 16

HOPPING_SEQS: list[list[int]] = [
    [3, 14, 5, 6, 9, 2, 12, 8, 15, 4, 11, 13, 17, 10, 1, 7, 0, 18, 16],
    [10, 3, 15, 5, 0, 17, 13, 6, 11, 4, 8, 18, 9, 14, 1, 12, 7, 16, 2],
    [14, 5, 11, 3, 8, 2, 18, 4, 10, 13, 9, 1, 16, 17, 0, 6, 15, 12, 7],
    [7, 0, 11, 18, 4, 2, 13, 5, 10, 17, 3, 9, 16, 14, 8, 12, 1, 6, 15],
]

PAYLOAD_LEN_BYTES_V1: list[int] = [0, 4, 9, 13]

# Preamble ON/OFF indices
preamble_on_idx: list[int] = [i for i, b in enumerate(PREAMBLE_BITS) if b == 1]
preamble_off_idx: list[int] = [i for i, b in enumerate(PREAMBLE_BITS) if b == 0]

# v1 preamble code indices (for F63/F0 estimation)
PREAMBLE_CODE_V1: list[int] = [63, 0, 63, 0, 63, 0, 63, 63]
on_indices_v1: list[int] = [i for i, b in enumerate(PREAMBLE_CODE_V1) if b == 63]
off_indices_v1: list[int] = [i for i, b in enumerate(PREAMBLE_CODE_V1) if b == 0]

# ---------------------------------------------------------------------------
# Detection spectrogram parameters
# ---------------------------------------------------------------------------

NFFT_DET: int = 625
NOVERLAP_DET: int = 0

# Template matching
DETECTION_THRESHOLD: float = 0.5
TEMPLATE_FREQ_BINS: int = 3
MAX_RAW: int = 10_000

# Filtering
PREAMBLE_F0_SNR_MIN: float = 5.0
MIN_ENERGY_DBFS: float = -80.0

# De-duplication tolerances
F0_TOL: float = 2 * max(SYNTH_RES.values())
TIME_TOL: float = 0.5

# Verbose flag (override for diagnostics)
VERBOSE: bool = False

# ---------------------------------------------------------------------------
# Visualisation spectrogram defaults (used by compute_spec_chunk)
# ---------------------------------------------------------------------------

NFFT_VIS: int = 2**12  # 4096
NOVERLAP_VIS: int = NFFT_VIS // 4  # 1024

# ---------------------------------------------------------------------------
# Derived timing values (recomputed by configure())
# ---------------------------------------------------------------------------

samples_per_symbol: int = 0
fft_freqs: np.ndarray = np.array([])
slot_samples: dict[int, dict[str, int]] = {}
time_step_s: float = 0.0
bins_on: int = 0
NMS_TIME_BINS: int = 0
NMS_FREQ_BINS: int = 15
templates: dict[int, dict] = {}

# ---------------------------------------------------------------------------
# Reed-Solomon init
# ---------------------------------------------------------------------------

rs_prim = rs.find_prime_polys(c_exp=6, fast_primes=True, single=False)[0]
rs.init_tables(c_exp=6, prim=rs_prim)


# ---------------------------------------------------------------------------
# configure() — compute all sample-rate-dependent values
# ---------------------------------------------------------------------------

def configure(sample_rate: int | None = None) -> None:
    """Recompute all sample-rate-dependent derived values.

    Call with no arguments to initialise at ``DEFAULT_SAMPLE_RATE``, or
    pass a custom rate for offline analysis.
    """
    global SAMPLE_RATE, samples_per_symbol, fft_freqs, slot_samples
    global time_step_s, bins_on, NMS_TIME_BINS, templates

    if sample_rate is not None:
        SAMPLE_RATE = sample_rate

    samples_per_symbol = int(SYMBOL_DURATION_S * SAMPLE_RATE)
    fft_freqs = np.fft.fftfreq(samples_per_symbol, d=1.0 / SAMPLE_RATE)

    slot_samples.clear()
    for ver, gap_s in GAP_DURATIONS.items():
        gap = int(gap_s * SAMPLE_RATE)
        slot_samples[ver] = {"gap": gap, "slot": samples_per_symbol + gap}

    time_step_s = NFFT_DET / SAMPLE_RATE
    bins_on = int(round(SYMBOL_DURATION_S / time_step_s))

    NMS_TIME_BINS = max(
        bins_on + int(round(g / time_step_s)) for g in GAP_DURATIONS.values()
    ) * 4

    # Build preamble templates (both protocols)
    templates.clear()
    for phy_ver, gap_s in GAP_DURATIONS.items():
        bins_gap = int(round(gap_s / time_step_s))
        bins_per_slot = bins_on + bins_gap
        pattern_1d = []
        for bit in PREAMBLE_BITS:
            if bit == 1:
                pattern_1d.extend([1.0] * bins_on + [0.0] * bins_gap)
            else:
                pattern_1d.extend([0.0] * bins_per_slot)
        pattern_1d_arr = np.array(pattern_1d, dtype=np.float32)
        template_2d = np.tile(pattern_1d_arr, (TEMPLATE_FREQ_BINS, 1))
        template_uint8 = (template_2d * 255).astype(np.uint8)
        templates[phy_ver] = {
            "uint8": template_uint8,
            "gap_s": gap_s,
            "bins_gap": bins_gap,
            "bins_per_slot": bins_per_slot,
            "width": len(pattern_1d),
            "duration_s": len(pattern_1d) * time_step_s,
        }


# Auto-configure on import
configure(DEFAULT_SAMPLE_RATE)
