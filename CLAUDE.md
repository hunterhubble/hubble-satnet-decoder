# fast-decoder

Hubble PHY v-1 / v1 preamble detector, FSK decoder, and spectrogram computation.

## Build & test

```bash
pip install -e ".[dev]"
ruff check src/
pytest
```

## Project structure

- `src/fast_decoder/constants.py` — protocol constants and sample-rate-dependent derived values
- `src/fast_decoder/detector.py` — OpenCV template-matching preamble detector
- `src/fast_decoder/decoder.py` — full decode pipeline (`decode_signal`, `_decode_v1`, `_decode_vneg1`)
- `src/fast_decoder/demod.py` — FSK demodulation helpers with timing search
- `src/fast_decoder/whitening.py` — LFSR-7 de-scrambling
- `src/fast_decoder/chipset.py` — chipset identification and per-chipset statistics
- `src/fast_decoder/spectrogram.py` — scipy-based spectrogram computation

## Conventions

- Hatchling build with `src` layout
- Ruff is the sole linter (no mypy, no black)
- All public API is re-exported from `__init__.py`
- `configure(sample_rate)` recomputes derived constants when using a non-default sample rate
