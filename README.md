# fast-decoder

[![CI](https://github.com/hubblenetwork/fast-decoder/actions/workflows/ci.yml/badge.svg)](https://github.com/hubblenetwork/fast-decoder/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Hubble PHY v-1 / v1 preamble detector, FSK decoder, and spectrogram
computation — extracted as a standalone, pip-installable library.

## Install

```bash
pip install git+https://github.com/hubblenetwork/fast-decoder.git
```

Or for development:

```bash
git clone https://github.com/hubblenetwork/fast-decoder.git
cd fast-decoder
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from fast_decoder import decode_signal, configure

# (optional) override the default sample rate
configure(sample_rate=781_250)

# Load 1 second of IQ data (complex64)
iq = np.load("capture.npy")

packets, detections, attempts = decode_signal(iq)
for pkt in packets:
    print(f"Device 0x{pkt['ntw_id']:08X}  seq={pkt['seq_num']}  "
          f"chipset={pkt.get('chipset', 'v-1')}")
```

## API reference

### `decode_signal(signal) -> (packets, detections, attempts)`

Full dual-protocol decode pipeline on a 1-second IQ chunk.

### `detect_preambles(spec_img, t_det, f_det) -> (time, freq, scores, phy_ver)`

Dual-template preamble detection via OpenCV + NMS.

### `compute_spec_chunk(iq_chunk) -> Sxx_dB`

Compute a visualisation spectrogram (freq × time) for an IQ chunk.

### `configure(sample_rate=None)`

Recompute all sample-rate-dependent derived values.

### Constants

```python
from fast_decoder import (
    SYNTH_RES,             # per-chipset synthesiser resolution (Hz)
    CHANNEL_SPACING,       # nominal channel spacing (Hz)
    DEVICE_CHANNEL_SPACING,# actual per-device channel spacing
    HOPPING_SEQS,          # frequency hopping sequences
    RS_N_V1, RS_K_V1,      # Reed-Solomon block sizes (v1)
    PREAMBLE_CODE_V1,      # v1 preamble code [63,0,63,0,63,0,63,63]
)
```

### Chipset statistics

```python
from fast_decoder import get_chipset_stats, reset_chipset_stats

stats = get_chipset_stats()   # {"nordic": {"detected": 5, "ok": 4, ...}, ...}
reset_chipset_stats()
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
