"""
EMG Analysis Module

Computes segment-level RMS values for muscle fatigue analysis
and provides visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_segment_rms(signal: np.ndarray,
                        segments: list[tuple[int, int]],
                        fs: float = 250.0) -> dict:
    """
    Compute RMS and RMS-weighted duration for each detected segment.

    Args:
        signal: Full EMG signal array.
        segments: List of (start_index, end_index) tuples.
        fs: Sampling rate (Hz), used to convert samples to seconds.

    Returns:
        Dictionary with rms, rms_wid, durations, totals, and cumulative ratios.
    """
    rms_list = []
    rms_wid_list = []
    durations = []

    for start, end in segments:
        seg = signal[start:end + 1]
        rms = np.sqrt(np.mean(seg ** 2))
        duration = (end - start) / fs
        rms_list.append(rms)
        rms_wid_list.append(rms * duration)
        durations.append(duration)

    rms_total = sum(rms_list)
    rms_wid_total = sum(rms_wid_list)

    rms_cum = np.cumsum(rms_list) / rms_total * 100 if rms_total > 0 else []
    rms_wid_cum = np.cumsum(rms_wid_list) / rms_wid_total * 100 if rms_wid_total > 0 else []

    return {
        'rms': rms_list,
        'rms_wid': rms_wid_list,
        'durations': durations,
        'rms_total': rms_total,
        'rms_wid_total': rms_wid_total,
        'rms_cumulative_ratio': list(rms_cum),
        'rms_wid_cumulative_ratio': list(rms_wid_cum),
    }


def plot_envelope_with_segments(time: np.ndarray, envelope: np.ndarray,
                                 threshold: float,
                                 segments: list[tuple[int, int]],
                                 rms_values: list[float],
                                 save_path: str | None = None):
    """
    Plot the envelope with detected segments highlighted.

    Args:
        time: Time array.
        envelope: Amplitude envelope.
        threshold: Detection threshold.
        segments: List of (start, end) tuples.
        rms_values: RMS value for each segment.
        save_path: If provided, save figure to this path.
    """
    plt.figure(figsize=(14, 5))
    plt.plot(time, envelope, 'b-', linewidth=0.5, label='Envelope')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold:.2f}')

    for i, (start, end) in enumerate(segments):
        plt.axvspan(time[start], time[end], alpha=0.3, color='green')
        mid = (time[start] + time[end]) / 2
        plt.text(mid, envelope[start:end + 1].max(),
                 f'S{i + 1}\nRMS={rms_values[i]:.2f}',
                 ha='center', va='bottom', fontsize=7)

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('EMG Envelope with Detected Contraction Segments')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_rms_bars(values: list[float], title: str, ylabel: str,
                  save_path: str | None = None):
    """
    Plot bar chart of per-segment values.

    Args:
        values: List of values to plot.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: If provided, save figure to this path.
    """
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(values)), values, color='skyblue')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.title(title)
    plt.xlabel('Segment Index')
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
