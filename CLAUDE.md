# hubble-satnet-decoder

Hubble PHY v-1 / v1 preamble detector, FSK decoder, and spectrogram computation.

## Build & test

```bash
pip install -e ".[dev]"
ruff check src/
pytest
```

## Project structure

- `src/hubble_satnet_decoder/constants.py` — protocol constants and sample-rate-dependent derived values
- `src/hubble_satnet_decoder/detector.py` — OpenCV template-matching preamble detector
- `src/hubble_satnet_decoder/decoder.py` — full decode pipeline (`decode_signal`, `_decode_v1`, `_decode_vneg1`)
- `src/hubble_satnet_decoder/demod.py` — FSK demodulation helpers with timing search
- `src/hubble_satnet_decoder/whitening.py` — LFSR-7 de-scrambling
- `src/hubble_satnet_decoder/chipset.py` — chipset identification and per-chipset statistics
- `src/hubble_satnet_decoder/spectrogram.py` — scipy-based spectrogram computation

## Conventions

- Hatchling build with `src` layout
- Ruff is the sole linter (no mypy, no black)
- All public API is re-exported from `__init__.py`
- `configure(sample_rate)` recomputes derived constants when using a non-default sample rate
