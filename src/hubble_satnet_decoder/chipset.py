"""Chipset identification and per-chipset decode statistics."""

from . import constants

# -- Per-chipset decode statistics ------------------------------------------

_chipset_stats: dict[str, dict[str, int]] = {}
_last_attempt: dict = {}


def cs_inc(chipset: str, field: str) -> None:
    """Increment a per-chipset decode counter."""
    if chipset not in _chipset_stats:
        _chipset_stats[chipset] = {
            "detected": 0, "snr_fail": 0, "header_fail": 0,
            "pdu_fail": 0, "ok": 0,
        }
    _chipset_stats[chipset][field] += 1
    _last_attempt["chipset"] = chipset
    if field != "detected":
        _last_attempt["reason"] = field


def get_chipset_stats() -> dict:
    """Return a copy of the per-chipset decode statistics."""
    return dict(_chipset_stats)


def reset_chipset_stats() -> None:
    """Clear all per-chipset decode statistics."""
    _chipset_stats.clear()


def get_last_attempt() -> dict:
    """Return the mutable last-attempt dictionary (shared with decoder)."""
    return _last_attempt


def identify_chipset(measured_synth_res: float) -> tuple[str, float]:
    """Return ``(chipset_name, synth_res_value)`` for the closest matching chipset."""
    best_name, best_val, best_err = "", 0.0, float("inf")
    for name, val in constants.SYNTH_RES.items():
        err = abs(measured_synth_res - val)
        if err < best_err:
            best_name, best_val, best_err = name, val, err
    return best_name, best_val
