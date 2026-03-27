"""Dual-protocol preamble detection and packet decoder (PHY v-1 and v1).

Pipeline:
  1. Dual-template preamble detection via OpenCV matchTemplate + NMS
  2. Protocol-specific demodulation (OOK for v-1, FSK with hopping for v1)
  3. Reed-Solomon decode and MAC parsing
"""

from collections import Counter

import numpy as np
import reedsolo as rs
from scipy.signal import spectrogram as scipy_spectrogram

from . import constants
from .chipset import cs_inc, get_last_attempt, identify_chipset
from .demod import build_chan_mask, demod_best, interp_peak
from .detector import detect_preambles
from .whitening import data_de_scrambling

# -- Diagnostic counter for v1 verbose output -------------------------------
_v1_diag_counter = 0


def _rs_decode(
    symbols, rs_n_array: list[int], rs_k_array: list[int]
) -> tuple[np.ndarray, int]:
    """Reed-Solomon decode over GF(64). Returns ``(decoded, n_corrections)``."""
    sym_list = symbols.tolist() if isinstance(symbols, np.ndarray) else list(symbols)
    try:
        idx = rs_n_array.index(len(sym_list))
    except ValueError:
        return np.full(0, -1, dtype=int), -1
    rs_k = rs_k_array[idx]
    num_ecc = rs_n_array[idx] - rs_k
    try:
        rx_syms, _, errata_pos = rs.rs_correct_msg(sym_list, num_ecc, fcr=1)
        return np.array(rx_syms), len(errata_pos)
    except Exception:
        return np.full(rs_k, -1, dtype=int), -1


# ===========================================================================
# PHY v-1 decode
# ===========================================================================

def _decode_vneg1(signal, start_sample, sps):
    """Decode a PHY v-1 packet: OOK preamble, no header, no hopping.

    Returns ``(pkt_info, result)`` or ``(None, None)`` on failure.
    """
    _last_attempt = get_last_attempt()
    end_sample = start_sample + constants.SYMBOLS_PER_PACKET_VNEG1 * sps["slot"]
    if end_sample > len(signal):
        return None, None

    # F0 estimation via ON-OFF spectral differencing
    psd_on = np.zeros(constants.samples_per_symbol)
    for sym in constants.preamble_on_idx:
        s0 = start_sample + sym * sps["slot"]
        psd_on += np.abs(np.fft.fft(signal[s0: s0 + constants.samples_per_symbol])) ** 2
    psd_on /= len(constants.preamble_on_idx)

    psd_off = np.zeros(constants.samples_per_symbol)
    for sym in constants.preamble_off_idx:
        s0 = start_sample + sym * sps["slot"]
        psd_off += np.abs(np.fft.fft(signal[s0: s0 + constants.samples_per_symbol])) ** 2
    psd_off /= len(constants.preamble_off_idx)

    psd_diff = psd_on - psd_off
    F0_bin = int(np.argmax(psd_diff))
    F0 = constants.fft_freqs[F0_bin]
    F0_snr = psd_diff[F0_bin] / (np.median(np.abs(psd_diff)) + 1e-30)

    if F0_snr < constants.PREAMBLE_F0_SNR_MIN:
        _last_attempt["reason"] = "snr_fail"
        return None, None

    total_energy_dBFS = 10.0 * np.log10(
        psd_on[F0_bin] / (constants.samples_per_symbol * constants.ADC_FULL_SCALE) ** 2 + 1e-30
    )
    _last_attempt.update(F0_hz=float(F0), energy_dB=float(total_energy_dBFS))

    # Channel mask and per-symbol peak frequency
    chan_mask = build_chan_mask(F0, constants.FREQ_STEP_VNEG1)
    sym_peak_freqs = np.zeros(constants.SYMBOLS_PER_PACKET_VNEG1)
    for sym in range(constants.SYMBOLS_PER_PACKET_VNEG1):
        s0 = start_sample + sym * sps["slot"]
        spectrum = np.fft.fft(signal[s0: s0 + constants.samples_per_symbol])
        psd = np.abs(spectrum) ** 2
        psd_m = psd.copy()
        psd_m[~chan_mask] = 0.0
        pk = int(np.argmax(psd_m))
        sym_peak_freqs[sym] = constants.fft_freqs[pk]

    # Decode data symbols (slots 8..31)
    data_bins = []
    for sym_idx in range(constants.PREAMBLE_LEN, constants.SYMBOLS_PER_PACKET_VNEG1):
        freq_offset = sym_peak_freqs[sym_idx] - F0
        fsk_bin = int(round(freq_offset / constants.FREQ_STEP_VNEG1))
        fsk_bin = max(0, min(constants.NUM_FSK_BINS - 1, fsk_bin))
        data_bins.append(fsk_bin)

    # RS decode v-1: length symbols at positions 0, 9, 18
    len_sym_0 = data_bins[0]
    len_sym_9 = data_bins[9]
    len_sym_18 = data_bins[18]
    len_idx = Counter([len_sym_0, len_sym_9, len_sym_18]).most_common(1)[0][0]

    if len_idx < 0 or len_idx >= len(constants.RS_K_VNEG1):
        return None, None

    rs_k = constants.RS_K_VNEG1[len_idx]
    rs_n = constants.RS_N_VNEG1[len_idx]
    num_ecc = rs_n - rs_k

    codeword = data_bins[1:9] + data_bins[10:18] + data_bins[19:]
    if len(codeword) < rs_n:
        return None, None
    codeword_rs = codeword[:rs_n]

    syndromes = rs.rs_calc_syndromes(codeword_rs, num_ecc, fcr=1)
    has_errors = not all(s == 0 for s in syndromes[1:])

    if has_errors:
        try:
            corrected = rs.rs_correct_msg(codeword_rs, num_ecc, fcr=1)
            mac_syms = list(corrected[0])
            n_corr = len(corrected[2])
        except rs.ReedSolomonError:
            return None, None
    else:
        mac_syms = list(codeword_rs[:rs_k])
        n_corr = 0

    # Parse v-1 MAC
    bits = "".join(f"{s:06b}" for s in mac_syms)
    if len(bits) < 44:
        return None, None

    ntw_id = int(bits[0:34], 2)
    seq_num = int(bits[34:44], 2)
    auth_tag = int(bits[44:60], 2) if len(bits) >= 60 else 0

    return (
        {"F0_hz": float(F0), "total_energy_dB": float(total_energy_dBFS)},
        {
            "phy_ver": -1, "ntw_id": ntw_id, "seq_num": seq_num,
            "auth_tag": auth_tag, "rs_errors": n_corr,
        },
    )


# ===========================================================================
# PHY v1 decode
# ===========================================================================

def _decode_v1(signal, start_sample, sps):
    """Decode a PHY v1 packet: FSK preamble, header, frequency hopping.

    Returns ``(pkt_info, result)`` or ``(None, None)`` on failure.
    """
    global _v1_diag_counter
    _v1_diag_counter += 1
    _diag = constants.VERBOSE or (_v1_diag_counter % 50 == 1)
    _last_attempt = get_last_attempt()

    # F63 / F0 estimation
    psd_63 = np.zeros(constants.samples_per_symbol)
    for sym in constants.on_indices_v1:
        s0 = start_sample + sym * sps["slot"]
        if s0 + constants.samples_per_symbol > len(signal):
            return None, None
        psd_63 += np.abs(np.fft.fft(signal[s0: s0 + constants.samples_per_symbol])) ** 2
    psd_63 /= len(constants.on_indices_v1)

    psd_0 = np.zeros(constants.samples_per_symbol)
    for sym in constants.off_indices_v1:
        s0 = start_sample + sym * sps["slot"]
        if s0 + constants.samples_per_symbol > len(signal):
            return None, None
        psd_0 += np.abs(np.fft.fft(signal[s0: s0 + constants.samples_per_symbol])) ** 2
    psd_0 /= len(constants.off_indices_v1)

    psd_diff_63 = psd_63 - psd_0
    F63_bin = int(np.argmax(psd_diff_63))
    F63_snr = float(psd_diff_63[F63_bin] / (np.median(np.abs(psd_diff_63)) + 1e-30))

    psd_diff_0 = psd_0 - psd_63
    F0_bin = int(np.argmax(psd_diff_0))

    if F63_snr < constants.PREAMBLE_F0_SNR_MIN:
        _last_attempt["reason"] = "snr_fail"
        if _diag:
            print(f"[v1-DIAG] FAIL snr: F63_snr={F63_snr:.1f} < {constants.PREAMBLE_F0_SNR_MIN}")
        return None, None

    F0 = interp_peak(psd_diff_0, F0_bin, constants.fft_freqs)
    F63 = interp_peak(psd_diff_63, F63_bin, constants.fft_freqs)

    synth_res_signed = (F63 - F0) / 63.0
    measured_synth_res = abs(synth_res_signed)
    chipset_name, table_synth_res = identify_chipset(measured_synth_res)
    synth_res_val = table_synth_res if synth_res_signed >= 0 else -table_synth_res
    cs_inc(chipset_name, "detected")

    if _diag:
        sign_char = "+" if synth_res_signed >= 0 else "-"
        print(
            f"[v1-DIAG] F0={F0:.1f} F63={F63:.1f} Hz, "
            f"meas_sr={sign_char}{measured_synth_res:.2f} -> "
            f"{chipset_name}(val={synth_res_val:.1f}), snr={F63_snr:.1f}"
        )

    total_energy_dBFS = 10.0 * np.log10(
        max(psd_0[F0_bin], psd_63[F63_bin])
        / (constants.samples_per_symbol * constants.ADC_FULL_SCALE) ** 2
        + 1e-30
    )
    _last_attempt.update(
        F0_hz=float(F0), energy_dB=float(total_energy_dBFS),
        measured_synth_res=round(measured_synth_res, 2),
        synth_res_val=round(synth_res_val, 1),
        F63_snr=round(F63_snr, 1),
    )

    # Demodulate header (6 symbols, same channel) with timing tolerance
    chan_mask = build_chan_mask(F0, synth_res_val)
    header_syms = []
    drift = 0
    for h in range(constants.NUM_HEADER_SYMS):
        sym_abs_idx = constants.PREAMBLE_LEN + h
        s0 = start_sample + sym_abs_idx * sps["slot"] + drift
        if s0 + constants.samples_per_symbol > len(signal):
            return None, None
        fsk_bin, _, off = demod_best(signal, s0, F0, synth_res_val, chan_mask)
        drift += off
        header_syms.append(fsk_bin)

    _last_attempt["header_syms"] = list(header_syms)

    # RS decode header
    header_decoded, header_n_corr = _rs_decode(
        np.array(header_syms, dtype=int), constants.RS_N_V1, constants.RS_K_V1
    )
    if header_n_corr < 0:
        cs_inc(chipset_name, "header_fail")
        if _diag:
            print(f"[v1-DIAG] FAIL header RS: syms={header_syms}, chipset={chipset_name}")
        return None, None

    # Parse header
    hdr_bits = f"{int(header_decoded[0]):06b}{int(header_decoded[1]):06b}"
    _phy_ver = int(hdr_bits[0:4], 2)  # noqa: F841 — parsed but not used downstream
    pkt_len_idx = int(hdr_bits[4:6], 2)
    hop_seq_idx = int(hdr_bits[6:8], 2)
    channel_num = int(hdr_bits[8:12], 2)

    # 4 bits encode 0-15; channels 16-18 alias to 0-2.
    candidate_b = channel_num + 16
    if candidate_b < constants.NUM_CHANNELS:
        measured_center = F0 + 31.5 * synth_res_val
        nominal_a = (channel_num - constants.LO_CHANNEL) * constants.CHANNEL_SPACING
        nominal_b = (candidate_b - constants.LO_CHANNEL) * constants.CHANNEL_SPACING
        if abs(measured_center - nominal_b) < abs(measured_center - nominal_a):
            channel_num = candidate_b

    num_pdu_symbols = constants.RS_N_V1[pkt_len_idx + 1]
    hopping_seq = constants.HOPPING_SEQS[hop_seq_idx]
    _last_attempt.update(
        header_n_corr=int(header_n_corr),
        channel_num=channel_num, hop_seq_idx=hop_seq_idx,
        pkt_len_idx=pkt_len_idx, num_pdu_symbols=num_pdu_symbols,
    )

    try:
        hop_index = hopping_seq.index(channel_num)
    except ValueError:
        _last_attempt["reason"] = "hop_fail"
        return None, None

    # Demodulate PDU with frequency hopping and timing tolerance.
    # Hop offsets use per-chipset quantisation of absolute channel indices:
    #   quantize(ch * CHANNEL_SPACING / synth_res) * synth_res
    # Silabs firmware uses round(); Nordic/TI/ESP/Atmosic use int (truncation).
    # Using absolute indices (not LO-relative) is critical because
    # q(N*x) - q(M*x) != q((N-M)*x) in general.
    current_channel = channel_num
    F0_current = F0
    pdu_syms = []
    sr_abs = table_synth_res
    _quantize = constants.SYNTH_QUANTIZE.get(chipset_name, int)
    preamble_ch_freq = _quantize(channel_num * constants.CHANNEL_SPACING / sr_abs) * sr_abs

    for p_idx in range(num_pdu_symbols):
        sym_abs_idx = constants.PREAMBLE_LEN + constants.NUM_HEADER_SYMS + p_idx
        s0 = start_sample + sym_abs_idx * sps["slot"] + drift
        if s0 + constants.samples_per_symbol > len(signal):
            break

        next_channel = hopping_seq[
            (hop_index + sym_abs_idx // constants.NUM_SYM_PER_HOP) % constants.NUM_CHANNELS
        ]
        if next_channel != current_channel:
            next_ch_freq = _quantize(next_channel * constants.CHANNEL_SPACING / sr_abs) * sr_abs
            F0_current = F0 + (next_ch_freq - preamble_ch_freq)
            chan_mask = build_chan_mask(F0_current, synth_res_val)
            current_channel = next_channel

        fsk_bin, _, off = demod_best(signal, s0, F0_current, synth_res_val, chan_mask)
        drift += off
        pdu_syms.append(fsk_bin)

    if len(pdu_syms) != num_pdu_symbols:
        _last_attempt["reason"] = "pdu_incomplete"
        return None, None

    # De-scramble + RS decode PDU
    pdu_raw = np.array(pdu_syms, dtype=int)
    de_scrambled = data_de_scrambling(pdu_raw, channel_num)
    pdu_decoded, pdu_n_corr = _rs_decode(de_scrambled, constants.RS_N_V1, constants.RS_K_V1)
    if pdu_n_corr < 0:
        _last_attempt["pdu_syms_head"] = pdu_syms[:10]
        cs_inc(chipset_name, "pdu_fail")
        if _diag:
            print(
                f"[v1-DIAG] FAIL pdu RS: chipset={chipset_name}, ch={channel_num}, "
                f"hop={hop_seq_idx}, hdr_corr={header_n_corr}, "
                f"pdu_raw={pdu_syms[:8]}..., sr_val={synth_res_val:.1f}"
            )
        return None, None

    # Parse v1 MAC
    mac_syms = pdu_decoded.tolist()
    bits = "".join(f"{int(s):06b}" for s in mac_syms)

    payload_proto_ver = int(bits[0:2], 2)
    seq_num = int(bits[2:12], 2)
    ntw_id = int(bits[12:44], 2)
    auth_tag = int(bits[44:76], 2)
    remaining_bits = bits[76:]

    payload_bytes = constants.PAYLOAD_LEN_BYTES_V1[pkt_len_idx]
    payload_bits_len = payload_bytes * 8
    if payload_bits_len > 0 and len(remaining_bits) >= payload_bits_len:
        payload_val = int(remaining_bits[:payload_bits_len], 2)
    elif payload_bits_len == 0:
        payload_val = 0
    else:
        payload_val = int(remaining_bits, 2) if remaining_bits else 0

    nominal_center_hz = (channel_num - constants.LO_CHANNEL) * constants.CHANNEL_SPACING
    measured_center_hz = F0 + 31.5 * table_synth_res
    freq_delta_hz = measured_center_hz - nominal_center_hz

    cs_inc(chipset_name, "ok")
    if _diag:
        print(
            f"[v1-DIAG] OK: chipset={chipset_name}, meas_sr={measured_synth_res:.2f}, "
            f"ntw=0x{ntw_id:08X}, seq={seq_num}, ch={channel_num}, "
            f"hdr_corr={header_n_corr}, pdu_corr={pdu_n_corr}, "
            f"freq_delta={freq_delta_hz:.0f}"
        )

    return (
        {"F0_hz": float(F0), "total_energy_dB": float(total_energy_dBFS)},
        {
            "phy_ver": 1, "ntw_id": ntw_id, "seq_num": seq_num,
            "auth_tag": auth_tag, "payload_proto_ver": payload_proto_ver,
            "payload_val": payload_val, "payload_bytes": payload_bytes,
            "chipset": chipset_name,
            "channel_num": channel_num, "hop_seq_idx": hop_seq_idx,
            "header_n_corr": int(header_n_corr), "pdu_n_corr": int(pdu_n_corr),
            "measured_synth_res": round(measured_synth_res, 2),
            "num_pdu_symbols": num_pdu_symbols,
            "freq_delta_hz": round(freq_delta_hz, 1),
        },
    )


# ===========================================================================
# Full decode pipeline
# ===========================================================================

def decode_signal(signal):
    """Full dual-protocol decode pipeline on a 1-second IQ chunk.

    Returns ``(decoded_packets, detection_list, all_attempts)``.

    - *decoded_packets*: successfully decoded packets with MAC fields.
    - *detection_list*:  all preamble detections (for box overlay on spectrogram).
    - *all_attempts*:    per-detection decode outcomes (chipset, decoded, reason).
    """
    _last_attempt = get_last_attempt()
    sig = signal.copy()
    sig -= sig.mean()

    # Detection spectrogram
    f_det, t_det, Sxx_det = scipy_spectrogram(
        sig, fs=constants.SAMPLE_RATE, nperseg=constants.NFFT_DET,
        noverlap=constants.NOVERLAP_DET, return_onesided=False,
    )
    f_det = np.fft.fftshift(f_det)
    Sxx_det = np.fft.fftshift(Sxx_det, axes=0)
    dc_idx = len(f_det) // 2
    Sxx_det[dc_idx, :] = 0.0

    Sxx_dB = (10.0 * np.log10(Sxx_det + 1e-12)).astype(np.float32)
    plow, phigh = np.percentile(Sxx_dB, [2, 99.5])
    if phigh <= plow:
        return [], [], []
    spec_img = np.clip((Sxx_dB - plow) / (phigh - plow) * 255, 0, 255).astype(np.uint8)

    # Dual-template detection
    det_time_s, det_freq_hz, det_scores, det_phy_ver = detect_preambles(
        spec_img, t_det, f_det
    )
    if len(det_time_s) == 0:
        return [], [], []

    # Build detection info list (for box overlay)
    detection_list = []
    for i in range(len(det_time_s)):
        ver = int(det_phy_ver[i])
        detection_list.append({
            "time_s": float(det_time_s[i]),
            "freq_hz": float(det_freq_hz[i]),
            "phy_ver": ver,
            "score": float(det_scores[i]),
            "preamble_duration_s": constants.templates[ver]["duration_s"],
        })

    # Decode each detection (dispatch by protocol version)
    decoded_packets = []
    all_attempts = []
    for det in detection_list:
        start_sample = int(round(det["time_s"] * constants.SAMPLE_RATE))
        ver = det["phy_ver"]
        sps = constants.slot_samples[ver]

        _last_attempt.clear()
        if ver == -1:
            pkt_info, result = _decode_vneg1(sig, start_sample, sps)
        else:
            pkt_info, result = _decode_v1(sig, start_sample, sps)

        attempt = {
            "time_s": det["time_s"],
            "freq_hz": det["freq_hz"],
            "phy_ver": ver,
            "score": det["score"],
            "start_sample": start_sample,
            "decoded": False,
            "reason": "unknown",
        }
        attempt.update(_last_attempt)

        if pkt_info is not None and pkt_info["total_energy_dB"] >= constants.MIN_ENERGY_DBFS:
            result["time_s"] = det["time_s"]
            result["freq_hz"] = det["freq_hz"]
            result["F0_hz"] = pkt_info["F0_hz"]
            result["total_energy_dB"] = pkt_info["total_energy_dB"]
            result["score"] = det["score"]
            result["preamble_duration_s"] = det["preamble_duration_s"]
            ver = result["phy_ver"]
            if ver == -1:
                total_syms = constants.SYMBOLS_PER_PACKET_VNEG1
            else:
                total_syms = (
                    constants.PREAMBLE_LEN
                    + constants.NUM_HEADER_SYMS
                    + result.get("num_pdu_symbols", 0)
                )
            result["signal_duration_s"] = (
                total_syms * constants.slot_samples[ver]["slot"] / constants.SAMPLE_RATE
            )
            decoded_packets.append(result)
            attempt["decoded"] = True
            attempt["reason"] = "ok"
            attempt["ntw_id"] = result.get("ntw_id")
            attempt["seq_num"] = result.get("seq_num")
        elif pkt_info is not None:
            attempt["reason"] = "energy_too_low"

        all_attempts.append(attempt)

    # De-duplicate
    unique: list[dict] = []
    for pkt in decoded_packets:
        is_dup = False
        for upkt in unique:
            if (abs(pkt["F0_hz"] - upkt["F0_hz"]) < constants.F0_TOL
                    and abs(pkt["time_s"] - upkt["time_s"]) < constants.TIME_TOL):
                if pkt["total_energy_dB"] > upkt["total_energy_dB"]:
                    upkt.update(pkt)
                is_dup = True
                break
        if not is_dup:
            unique.append(pkt)

    return unique, unique, all_attempts
