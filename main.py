"""
EMG Muscle Fatigue Analysis Pipeline

End-to-end pipeline for analyzing muscle fatigue from EMG signals.

Single-session mode:
    python main.py --file <path_to_csv>

Multi-session mode (fatigue tracking with spline interpolation):
    python main.py --multi <file1.csv> <file2.csv> ... --timestamps <t1> <t2> ...
"""

import argparse
import numpy as np
from preprocessing import load_emg_data, denoise_emg, smooth_signal_fft
from segmentation import compute_envelope, compute_adaptive_threshold, detect_segments
from analysis import compute_segment_rms, plot_envelope_with_segments, plot_rms_bars
from fatigue_model import FatigueModel


def run_single_session(
    file_path: str,
    sampling_rate: float = 250.0,
    cutoff_freq: float = 5.0,
    std_multiplier: float = 2.0,
    merge_gap: int = 4000,
    save_dir: str | None = None,
) -> dict:
    """
    Execute the full EMG analysis pipeline for a single session.

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
    print(f"Loading data from {file_path}...")
    time, emg_raw = load_emg_data(file_path)
    print(f"  Loaded {len(emg_raw)} samples ({time[-1]:.1f}s)")

    print("Applying noise removal filters...")
    emg_denoised = denoise_emg(emg_raw, sampling_rate)

    print(f"FFT smoothing (cutoff={cutoff_freq}Hz)...")
    emg_smooth = smooth_signal_fft(emg_denoised, sampling_rate, cutoff_freq)

    print("Computing Hilbert envelope...")
    envelope = compute_envelope(emg_smooth)

    threshold = compute_adaptive_threshold(envelope, std_multiplier)
    segments = detect_segments(envelope, threshold, merge_gap)
    print(f"  Detected {len(segments)} contraction segments (threshold={threshold:.2f})")

    results = compute_segment_rms(emg_raw, segments, sampling_rate)

    for i, (start, end) in enumerate(segments):
        print(
            f"  Segment {i+1}: {time[start]:.2f}s - {time[end]:.2f}s, "
            f"RMS={results['rms'][i]:.3f}, Duration={results['durations'][i]:.2f}s"
        )

    # Visualization
    save_envelope = f"{save_dir}/envelope_segments.png" if save_dir else None
    save_rms = f"{save_dir}/rms_per_segment.png" if save_dir else None
    save_rms_wid = f"{save_dir}/rms_wid_per_segment.png" if save_dir else None

    plot_envelope_with_segments(
        time, envelope, threshold, segments, results["rms"], save_envelope
    )
    plot_rms_bars(results["rms"], "RMS per Segment", "RMS Value", save_rms)
    plot_rms_bars(
        results["rms_wid"], "RMS x Duration per Segment",
        "RMS x Duration", save_rms_wid
    )

    return {
        "segments": segments,
        "results": results,
        "threshold": threshold,
        "time": time,
        "envelope": envelope,
    }


def run_multi_session(
    file_paths: list[str],
    timestamps: list[float],
    sampling_rate: float = 250.0,
    cutoff_freq: float = 5.0,
    std_multiplier: float = 2.0,
    merge_gap: int = 4000,
    save_dir: str | None = None,
) -> FatigueModel:
    """
    Run fatigue analysis across multiple sessions with spline-based capacity tracking.

    The first session establishes a baseline muscle capacity. Subsequent sessions
    compute fatigue ratio (usage / capacity) where capacity is refined via cubic
    spline interpolation as more data points accumulate.

    Args:
        file_paths: List of EMG data file paths (one per session).
        timestamps: List of session times (e.g., day numbers).
        sampling_rate: Sampling rate in Hz.
        cutoff_freq: FFT smoothing cutoff frequency (Hz).
        std_multiplier: Threshold = mean + std_multiplier * std.
        merge_gap: Max gap (samples) for merging nearby segments.
        save_dir: Directory to save output figures.

    Returns:
        FatigueModel with all sessions recorded.
    """
    assert len(file_paths) == len(timestamps), \
        "Number of files must match number of timestamps"

    model = FatigueModel()

    for i, (fpath, ts) in enumerate(zip(file_paths, timestamps)):
        print(f"\n{'='*60}")
        print(f"Session {i+1} (t={ts}): {fpath}")
        print(f"{'='*60}")

        time, emg_raw = load_emg_data(fpath)
        emg_denoised = denoise_emg(emg_raw, sampling_rate)
        emg_smooth = smooth_signal_fft(emg_denoised, sampling_rate, cutoff_freq)
        envelope = compute_envelope(emg_smooth)

        threshold = compute_adaptive_threshold(envelope, std_multiplier)
        segments = detect_segments(envelope, threshold, merge_gap)
        results = compute_segment_rms(emg_raw, segments, sampling_rate)

        print(f"  Detected {len(segments)} segments")
        print(f"  Total RMS energy: {results['rms_wid_total']:.4f}")

        record = model.add_session(
            timestamp=ts,
            segment_rms_values=results['rms'],
            segment_durations=results['durations'],
        )

        print(f"  Capacity: {record.total_capacity:.4f}")
        print(f"  Fatigue ratio: {record.fatigue_ratio:.4f}")

        if model._spline is not None:
            predicted = model.predict_capacity(ts)
            print(f"  Spline-predicted capacity: {predicted:.4f}")

    # Summary and visualization
    model.summary()

    save_path = f"{save_dir}/capacity_trend.png" if save_dir else None
    model.plot_capacity_trend(save_path)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMG Muscle Fatigue Analysis")
    subparsers = parser.add_subparsers(dest="mode", help="Analysis mode")

    # Single session
    single = subparsers.add_parser("single", help="Analyze a single EMG session")
    single.add_argument("--file", type=str, required=True, help="Path to EMG CSV")
    single.add_argument("--fs", type=float, default=250.0, help="Sampling rate (Hz)")
    single.add_argument("--cutoff", type=float, default=5.0, help="FFT cutoff (Hz)")
    single.add_argument("--std", type=float, default=2.0, help="Threshold multiplier")
    single.add_argument("--gap", type=int, default=4000, help="Segment merge gap")
    single.add_argument("--save", type=str, default=None, help="Output directory")

    # Multi session
    multi = subparsers.add_parser("multi", help="Track fatigue across sessions")
    multi.add_argument("--files", nargs="+", required=True, help="EMG CSV files")
    multi.add_argument("--timestamps", nargs="+", type=float, required=True,
                        help="Session timestamps (e.g., day numbers)")
    multi.add_argument("--fs", type=float, default=250.0, help="Sampling rate (Hz)")
    multi.add_argument("--cutoff", type=float, default=5.0, help="FFT cutoff (Hz)")
    multi.add_argument("--std", type=float, default=2.0, help="Threshold multiplier")
    multi.add_argument("--gap", type=int, default=4000, help="Segment merge gap")
    multi.add_argument("--save", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    if args.mode == "single":
        run_single_session(
            file_path=args.file, sampling_rate=args.fs,
            cutoff_freq=args.cutoff, std_multiplier=args.std,
            merge_gap=args.gap, save_dir=args.save,
        )
    elif args.mode == "multi":
        run_multi_session(
            file_paths=args.files, timestamps=args.timestamps,
            sampling_rate=args.fs, cutoff_freq=args.cutoff,
            std_multiplier=args.std, merge_gap=args.gap,
            save_dir=args.save,
        )
    else:
        parser.print_help()
