## [1.1.1] - 2026-03-31

### Fixed
- Fix inflated detection stats: deduplicate preamble detections before decode
  loop so each physical packet is counted once
- Fix frequency hop calculation: use round() instead of floor() for channel
  spacing quantization, matching device firmware formula

### Changed
- Simplify FSK demodulation: remove timing search (demod_best) and drift
  accumulation; use fixed half-symbol offset via demod_one_symbol
- Remove dual hop-mode fallback; single absolute hop frequency calculation
