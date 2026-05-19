from __future__ import annotations

import numpy as np

CLASS_PLEURAL_LINE = 2
CLASS_B_LINE = 4
CLASS_CONFLUENCE = 5


def roi_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pleural = np.where(mask == CLASS_PLEURAL_LINE)
    if len(pleural[0]) == 0:
        return None

    top = int(np.max(pleural[0]))
    left = int(np.min(pleural[1]))
    right = int(np.max(pleural[1]))

    bline_depth = np.where(mask == CLASS_B_LINE)[0]
    confluence_depth = np.where(mask == CLASS_CONFLUENCE)[0]
    if len(bline_depth) == 0 and len(confluence_depth) == 0:
        return None

    bottom = int(max(
        np.max(bline_depth) if len(bline_depth) else 0,
        np.max(confluence_depth) if len(confluence_depth) else 0,
    ) + 1)

    if top >= bottom or left >= right:
        return None
    return top, bottom, left, right


def get_roi(mask: np.ndarray) -> np.ndarray | None:
    bbox = roi_bbox(mask)
    if bbox is None:
        return None
    top, bottom, left, right = bbox
    return mask[top:bottom, left:right]


def bline_fraction(roi: np.ndarray | None) -> np.ndarray | None:
    if roi is None:
        return None
    return np.logical_or(roi == CLASS_B_LINE, roi == CLASS_CONFLUENCE).mean(axis=1)


def calc_blas(mask: np.ndarray) -> float:
    fractions = bline_fraction(get_roi(mask))
    if fractions is None or len(fractions) == 0:
        return 0.0
    try:
        from scipy.integrate import simpson

        area = simpson(fractions)
    except Exception:
        area = np.trapz(fractions)
    return float(area / len(fractions))

