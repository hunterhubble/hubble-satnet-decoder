"""LFSR-7 whitening / de-scrambling for PHY v1 PDU symbols."""

import numpy as np


def lfsr7_symbols(seed: int, nsym: int) -> list[int]:
    """Generate LFSR-7 whitening symbols (6-bit) for de-scrambling.

    Parameters
    ----------
    seed : int
        Channel number used as the LFSR seed.
    nsym : int
        Number of 6-bit symbols to produce.
    """
    state = (3 << 5) | (0b1000000 | seed)
    bits: list[int] = []
    for _ in range(nsym * 6):
        bits.append((state & 0x40) >> 6)
        fb = ((state >> 6) ^ (state >> 3)) & 1
        state = ((state << 1) & 0x7F) | fb
    return [int("".join(map(str, bits[i * 6:(i + 1) * 6])), 2) for i in range(nsym)]


def data_de_scrambling(pdu_symbols: np.ndarray, channel_num: int) -> np.ndarray:
    """XOR PDU symbols with the LFSR-7 whitening sequence for *channel_num*."""
    whitening = np.array(lfsr7_symbols(channel_num, len(pdu_symbols)))
    return pdu_symbols ^ whitening
