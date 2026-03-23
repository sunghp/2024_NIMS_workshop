"""
EMG Analysis Module

Computes segment-level RMS values for muscle fatigue analysis
and provides visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_segment_rms(signal: np.ndarray, segments: list[tuple[int, int]]) -> dict:
    """
    Compute RMS and RMS-weighted duration for each detected segment.

    For each segment:
        - RMS = sqrt(mean(signal^2))  within the segment
        - RMS_wid = RMS * segment_duration  (energy-like metric)

    Args:
        signal: Full EMG signal array.
        segments: List of (start_index, end_index) tuples.

    Returns:
        Dictionary containing:
            - 'rms': List of RMS values per segment.
            - 'rms_wid': List of RMS * duration values per segment.
            - 'rms_total': Sum of all RMS values.
            - 'rms_wid_total': Sum of all RMS_wid values.
            - 'rms_cumulative_ratio': Cumulative RMS ratio (%).
            - 'rms_wid_cumulative_ratio': Cumulative RMS_wid ratio (%).
    """
    rms_values = []
    rms_wid_values = []

    for start, end in segments:
        segment = signal[start : end + 1]
        rms = np.sqrt(np.mean(segment**2))
        duration = end - start
        rms_values.append(rms)
        rms_wid_values.append(rms * duration)

    rms_total = sum(rms_values)
    rms_wid_total = sum(rms_wid_values)

    # Cumulative ratios (%)
    rms_cumulative = np.cumsum(rms_values) / rms_total * 100 if rms_total > 0 else []
    rms_wid_cumulative = (
        np.cumsum(rms_wid_values) / rms_wid_total * 100 if rms_wid_total > 0 else []
    )

    return {
        "rms": rms_values,
        "rms_wid": rms_wid_values,
        "rms_total": rms_total,
        "rms_wid_total": rms_wid_total,
        "rms_cumulative_ratio": rms_cumulative,
        "rms_wid_cumulative_ratio": rms_wid_cumulative,
    }


def plot_envelope_with_segments(
    time: np.ndarray,
    envelope: np.ndarray,
    threshold: float,
    segments: list[tuple[int, int]],
    rms_values: list[float],
    save_path: str | None = None,
) -> None:
    """
    Plot the signal envelope with detected segments highlighted.

    Args:
        time: Time array.
        envelope: Amplitude envelope.
        threshold: Detection threshold.
        segments: List of (start, end) index tuples.
        rms_values: RMS value for each segment.
        save_path: If provided, save the figure to this path.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(time, envelope, color="orange", label="Envelope")
    plt.axhline(y=threshold, color="red", linestyle="--", label="Threshold")

    for i, (start, end) in enumerate(segments):
        plt.axvspan(
            time[start],
            time[end],
            color="green",
            alpha=0.3,
            label=f"Segment {i+1}: RMS={rms_values[i]:.3f}" if i < 5 else None,
        )

    plt.title("EMG Envelope with Detected Contraction Segments")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_rms_bars(
    rms_values: list[float],
    title: str = "RMS Values per Segment",
    ylabel: str = "RMS Value",
    save_path: str | None = None,
) -> None:
    """
    Bar chart of RMS values across segments.

    Args:
        rms_values: List of values to plot.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: If provided, save the figure to this path.
    """
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(rms_values)), rms_values, color="skyblue")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.title(title)
    plt.xlabel("Segment Index")
    plt.ylabel(ylabel)
    plt.grid(axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
