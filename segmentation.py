"""
EMG Segmentation Module

Extracts amplitude envelope via Hilbert transform and detects
muscle contraction segments using adaptive thresholding.
"""

import numpy as np
from scipy.signal import hilbert
from itertools import groupby


def compute_envelope(signal: np.ndarray) -> np.ndarray:
    """
    Compute the amplitude envelope using the Hilbert transform.

    The analytic signal's magnitude gives the instantaneous amplitude,
    providing a smooth envelope that captures contraction intensity.

    Args:
        signal: Input (smoothed) EMG signal.

    Returns:
        Amplitude envelope array.
    """
    analytic = hilbert(signal)
    return np.abs(analytic)


def compute_adaptive_threshold(envelope: np.ndarray,
                                std_multiplier: float = 2.0) -> float:
    """
    Compute an adaptive detection threshold.

    threshold = mean(envelope) + std_multiplier * std(envelope)

    This adapts to each subject's signal characteristics automatically.

    Args:
        envelope: Amplitude envelope.
        std_multiplier: Number of standard deviations above mean.

    Returns:
        Threshold value.
    """
    return envelope.mean() + std_multiplier * envelope.std()


def detect_segments(envelope: np.ndarray, threshold: float,
                    merge_gap: int = 4000) -> list[tuple[int, int]]:
    """
    Detect contiguous above-threshold regions and merge nearby segments.

    Steps:
        1. Find all indices where envelope exceeds the threshold.
        2. Group contiguous indices into segments.
        3. Merge segments closer than merge_gap samples.

    Args:
        envelope: Amplitude envelope of the signal.
        threshold: Amplitude threshold for detection.
        merge_gap: Maximum gap (samples) between segments to merge.

    Returns:
        List of (start_index, end_index) tuples for each segment.
    """
    above_threshold = np.where(envelope > threshold)[0]

    if len(above_threshold) == 0:
        return []

    groups = [
        list(group)
        for _, group in groupby(
            enumerate(above_threshold), lambda x: x[0] - x[1]
        )
    ]
    raw_segments = [(g[0][1], g[-1][1]) for g in groups]

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
