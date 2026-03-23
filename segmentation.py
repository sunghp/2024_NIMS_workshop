"""
EMG Segment Detection Module

Uses Hilbert transform envelope and adaptive thresholding
to automatically detect muscle contraction segments.
"""

import numpy as np
from itertools import groupby
from scipy.signal import hilbert


def compute_envelope(signal: np.ndarray) -> np.ndarray:
    """
    Compute the amplitude envelope of a signal using the Hilbert transform.

    The analytic signal is obtained via the Hilbert transform, and its
    absolute value gives the instantaneous amplitude (envelope).

    Args:
        signal: Input signal (typically rectified EMG).

    Returns:
        Envelope of the input signal.
    """
    analytic_signal = hilbert(np.abs(signal))
    return np.abs(analytic_signal)


def compute_adaptive_threshold(
    envelope: np.ndarray, std_multiplier: float = 2.0
) -> float:
    """
    Compute an adaptive threshold based on envelope statistics.

    threshold = mean(envelope) + std_multiplier * std(envelope)

    Args:
        envelope: Amplitude envelope of the signal.
        std_multiplier: Number of standard deviations above the mean.

    Returns:
        Threshold value.
    """
    return envelope.mean() + std_multiplier * envelope.std()


def detect_segments(
    envelope: np.ndarray, threshold: float, merge_gap: int = 4000
) -> list[tuple[int, int]]:
    """
    Detect contiguous above-threshold regions and merge nearby segments.

    Steps:
        1. Find all indices where envelope exceeds the threshold.
        2. Group contiguous indices into segments.
        3. Merge segments that are closer than `merge_gap` samples.

    Args:
        envelope: Amplitude envelope of the signal.
        threshold: Amplitude threshold for detection.
        merge_gap: Maximum gap (in samples) between segments to merge.

    Returns:
        List of (start_index, end_index) tuples for each detected segment.
    """
    above_threshold = np.where(envelope > threshold)[0]

    if len(above_threshold) == 0:
        return []

    # Group contiguous indices
    groups = [
        list(group)
        for _, group in groupby(
            enumerate(above_threshold), lambda x: x[0] - x[1]
        )
    ]
    raw_segments = [(g[0][1], g[-1][1]) for g in groups]

    # Merge nearby segments
    merged = []
    current_start, current_end = raw_segments[0]

    for start, end in raw_segments[1:]:
        if start - current_end <= merge_gap:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged
