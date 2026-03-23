"""
EMG Muscle Fatigue Analysis Pipeline

End-to-end pipeline for analyzing muscle fatigue from EMG signals:
    1. Load raw EMG data
    2. Denoise (Notch + Bandpass filtering)
    3. Smooth via FFT low-pass filtering
    4. Extract amplitude envelope (Hilbert transform)
    5. Detect muscle contraction segments (adaptive thresholding)
    6. Compute per-segment RMS for fatigue quantification

Usage:
    python main.py --file <path_to_csv> [options]
"""

import argparse
import numpy as np
from preprocessing import load_emg_data, denoise_emg, smooth_signal_fft
from segmentation import compute_envelope, compute_adaptive_threshold, detect_segments
from analysis import compute_segment_rms, plot_envelope_with_segments, plot_rms_bars


def run_pipeline(
    file_path: str,
    sampling_rate: float = 250.0,
    cutoff_freq: float = 5.0,
    std_multiplier: float = 2.0,
    merge_gap: int = 4000,
    save_dir: str | None = None,
) -> dict:
    """
    Execute the full EMG analysis pipeline.

    Args:
        file_path: Path to the EMG data file.
        sampling_rate: Sampling rate in Hz.
        cutoff_freq: FFT smoothing cutoff frequency (Hz).
        std_multiplier: Threshold = mean + std_multiplier * std.
        merge_gap: Max gap (samples) for merging nearby segments.
        save_dir: Directory to save output figures.

    Returns:
        Dictionary with segments, RMS results, and threshold info.
    """
    # 1. Load data
    print(f"Loading data from {file_path}...")
    time, emg_raw = load_emg_data(file_path)
    print(f"  Loaded {len(emg_raw)} samples ({time[-1]:.1f}s)")

    # 2. Denoise
    print("Applying noise removal filters...")
    emg_denoised = denoise_emg(emg_raw, sampling_rate)

    # 3. Smooth
    print(f"FFT smoothing (cutoff={cutoff_freq}Hz)...")
    emg_smooth = smooth_signal_fft(emg_denoised, sampling_rate, cutoff_freq)

    # 4. Envelope extraction
    print("Computing Hilbert envelope...")
    envelope = compute_envelope(emg_smooth)

    # 5. Segment detection
    threshold = compute_adaptive_threshold(envelope, std_multiplier)
    segments = detect_segments(envelope, threshold, merge_gap)
    print(f"  Detected {len(segments)} contraction segments (threshold={threshold:.2f})")

    # 6. RMS analysis
    results = compute_segment_rms(emg_raw, segments)

    for i, (start, end) in enumerate(segments):
        print(
            f"  Segment {i+1}: {time[start]:.2f}s - {time[end]:.2f}s, "
            f"RMS={results['rms'][i]:.3f}"
        )

    # 7. Visualization
    save_envelope = f"{save_dir}/envelope_segments.png" if save_dir else None
    save_rms = f"{save_dir}/rms_per_segment.png" if save_dir else None
    save_rms_wid = f"{save_dir}/rms_wid_per_segment.png" if save_dir else None

    plot_envelope_with_segments(
        time, envelope, threshold, segments, results["rms"], save_envelope
    )
    plot_rms_bars(results["rms"], "RMS per Segment", "RMS Value", save_rms)
    plot_rms_bars(
        results["rms_wid"],
        "RMS × Duration per Segment",
        "RMS × Duration",
        save_rms_wid,
    )

    return {
        "segments": segments,
        "results": results,
        "threshold": threshold,
        "time": time,
        "envelope": envelope,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMG Muscle Fatigue Analysis")
    parser.add_argument("--file", type=str, required=True, help="Path to EMG CSV file")
    parser.add_argument("--fs", type=float, default=250.0, help="Sampling rate (Hz)")
    parser.add_argument("--cutoff", type=float, default=5.0, help="FFT cutoff freq (Hz)")
    parser.add_argument("--std", type=float, default=2.0, help="Threshold std multiplier")
    parser.add_argument("--gap", type=int, default=4000, help="Segment merge gap (samples)")
    parser.add_argument("--save", type=str, default=None, help="Output directory for figures")

    args = parser.parse_args()

    run_pipeline(
        file_path=args.file,
        sampling_rate=args.fs,
        cutoff_freq=args.cutoff,
        std_multiplier=args.std,
        merge_gap=args.gap,
        save_dir=args.save,
    )
