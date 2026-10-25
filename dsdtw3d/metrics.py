import numpy as np
from numba import njit


@njit
def dtw_l1(a, b):
    """Exact DTW, L1 local distance, normalized by the sum of sequence lengths."""
    previous = np.full(len(b) + 1, np.inf)
    previous[0] = 0.0
    for i in range(len(a)):
        current = np.full(len(b) + 1, np.inf)
        for j in range(len(b)):
            cost = np.abs(a[i] - b[j]).sum()
            current[j + 1] = cost + min(previous[j], previous[j + 1], current[j])
        previous = current
    return previous[-1] / (len(a) + len(b))


def eer(genuine, forged):
    positive, negative = np.sort(genuine), np.sort(forged)
    if not len(positive) or not len(negative) or not np.isfinite(np.r_[positive, negative]).all():
        raise ValueError("EER needs finite genuine and forgery scores")
    thresholds = np.r_[-np.inf, np.unique(np.r_[positive, negative]), np.inf]
    far = np.searchsorted(negative, thresholds, side="right") / len(negative)
    frr = 1 - np.searchsorted(positive, thresholds, side="right") / len(positive)
    index = int(np.flatnonzero(far >= frr)[0])
    if index == 0:
        value, threshold = far[0], thresholds[0]
    else:
        before, after = far[index - 1] - frr[index - 1], far[index] - frr[index]
        weight = -before / (after - before)
        value = far[index - 1] + weight * (far[index] - far[index - 1])
        low, high = thresholds[index - 1:index + 1]
        threshold = low + weight * (high - low) if np.isfinite(low) and np.isfinite(high) else high
    return dict(eer_percent=float(100 * value), eer_threshold=float(threshold),
                genuine_trials=len(positive), forgery_trials=len(negative))
