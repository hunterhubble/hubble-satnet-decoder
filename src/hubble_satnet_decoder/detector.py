"""Dual-template preamble detection via OpenCV matchTemplate + NMS."""

import cv2
import numpy as np

from . import constants


def detect_preambles(
    spec_img: np.ndarray,
    t_det: np.ndarray,
    f_det: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dual-template preamble detection via OpenCV + NMS.

    Parameters
    ----------
    spec_img : uint8 2-D array
        Normalised detection spectrogram image.
    t_det, f_det : 1-D arrays
        Time and frequency axes of the detection spectrogram.

    Returns
    -------
    det_time_s, det_freq_hz, det_scores, det_phy_ver
    """
    all_raw_y, all_raw_x, all_raw_scores, all_raw_ver = [], [], [], []

    for phy_ver, tmpl in sorted(constants.templates.items()):
        match_result = cv2.matchTemplate(
            spec_img, tmpl["uint8"], cv2.TM_CCOEFF_NORMED
        )
        ry, rx = np.where(match_result > constants.DETECTION_THRESHOLD)
        if len(ry) > 0:
            r_scores = match_result[ry, rx]
            all_raw_y.append(ry)
            all_raw_x.append(rx)
            all_raw_scores.append(r_scores)
            all_raw_ver.append(np.full(len(ry), phy_ver, dtype=int))
        del match_result

    if not all_raw_y:
        return np.array([]), np.array([]), np.array([]), np.array([], dtype=int)

    raw_y = np.concatenate(all_raw_y)
    raw_x = np.concatenate(all_raw_x)
    raw_scores = np.concatenate(all_raw_scores)
    raw_phy_ver = np.concatenate(all_raw_ver)

    if len(raw_scores) > constants.MAX_RAW:
        top_k = np.argpartition(-raw_scores, constants.MAX_RAW)[:constants.MAX_RAW]
        raw_y, raw_x, raw_scores = raw_y[top_k], raw_x[top_k], raw_scores[top_k]
        raw_phy_ver = raw_phy_ver[top_k]

    order = np.argsort(-raw_scores)
    raw_y, raw_x, raw_scores = raw_y[order], raw_x[order], raw_scores[order]
    raw_phy_ver = raw_phy_ver[order]

    keep: list[int] = []
    for i in range(len(raw_scores)):
        suppress = False
        for j in keep:
            if (abs(int(raw_x[i]) - int(raw_x[j])) < constants.NMS_TIME_BINS
                    and abs(int(raw_y[i]) - int(raw_y[j])) < constants.NMS_FREQ_BINS):
                suppress = True
                break
        if not suppress:
            keep.append(i)

    half_h = constants.TEMPLATE_FREQ_BINS // 2
    det_x = raw_x[keep]
    det_y = raw_y[keep]
    det_scores = raw_scores[keep]
    det_phy_ver = raw_phy_ver[keep]
    det_time_s = t_det[np.clip(det_x, 0, len(t_det) - 1)]
    det_freq_hz = f_det[np.clip(det_y + half_h, 0, len(f_det) - 1)]
    return det_time_s, det_freq_hz, det_scores, det_phy_ver
