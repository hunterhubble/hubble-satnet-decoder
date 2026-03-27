## [1.0.1] - 2026-03-27

### Added
- Add fallback hop method to resolve rounding ambiguity in PDU decode

### Fixed
- Fix hop frequency: current-relative delta with round(), not round(A)-round(B)
- Per-chipset hop quantisation: int() for Nordic/TI/ESP/Atmosic, round() for Silabs
- Fix hop frequency calculation to match firmware quantization

### Maintenance
- Add /release skill for automated version bumps and PyPI releases
- Upgrade release workflow with tag verification, GitHub Releases, and OIDC publishing
- Add PyPI publish workflow, bump to v1.0.0
- Rename package from fast-decoder to hubble-satnet-decoder
- Initial release: extract decoder from pluto-sdr-docker
