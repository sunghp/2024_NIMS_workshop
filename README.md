# EMG Muscle Fatigue Analysis

Automated pipeline for analyzing muscle fatigue patterns from EMG (Electromyography) signals. Developed during the **NIMS Industrial Mathematics Problem-Solving Workshop**, where a partnering company provided real-world EMG data for muscle fatigue prediction.

> **Note**: EMG data files are not included as they are proprietary data provided by the partnering company.

## Overview

This project goes beyond simple signal processing to build a **multi-session fatigue model**. The core idea:

1. **First measurement**: Estimate the muscle's total usable capacity (baseline) from cumulative RMS energy across all contraction segments.
2. **Subsequent measurements**: Compute fatigue as usage/capacity ratio.
3. **Capacity evolves**: Muscles grow with training, so total capacity is not fixed. We track capacity changes across sessions using **cubic spline interpolation**, making fatigue estimates more precise as more sessions are recorded.

## Pipeline

```
Raw EMG Signal
    │
    ├── 1. Noise Removal
    │       ├── Notch filter (60Hz, 120Hz powerline interference)
    │       └── Bandpass filter (20-124Hz EMG band)
    │
    ├── 2. FFT Smoothing
    │       └── Low-pass filtering (retain < 5Hz components)
    │
    ├── 3. Envelope Extraction
    │       └── Hilbert transform → instantaneous amplitude
    │
    ├── 4. Segment Detection
    │       ├── Adaptive thresholding (mean + k*std)
    │       └── Contiguous region grouping + gap-based merging
    │
    ├── 5. Single-Session Fatigue Quantification
    │       ├── Per-segment RMS
    │       ├── RMS x Duration (energy metric)
    │       └── Cumulative ratio analysis
    │
    └── 6. Multi-Session Fatigue Modeling
            ├── Baseline capacity estimation (Session 1)
            ├── Usage / Capacity ratio per session
            ├── Cubic spline interpolation of capacity trend
            └── Progressively refined fatigue estimation
```

## Multi-Session Fatigue Model

The key mathematical contribution is the spline-based capacity tracking:

```
Session 1 (Day 0):   Capacity_0 = Σ(RMS_i × Duration_i)    ← baseline
Session 2 (Day 3):   Capacity_1 = measured
                      Fatigue = Usage / Capacity_1

Session 3 (Day 7):   Capacity_spline = CubicSpline(t0, t1, t2)(t=7)
                      Fatigue = Usage / Capacity_spline       ← refined estimate
        ...
Session N:            Spline fit over all N points → better precision
```

As sessions accumulate, the spline captures trends like muscle growth from training or decline from overuse, and the fatigue ratio becomes increasingly precise.

## Tech Stack

- **Python** (NumPy, SciPy, Pandas, Matplotlib)
- **Signal Processing**: Butterworth filters, Notch filters, FFT/IFFT, Hilbert transform
- **Mathematical Modeling**: Cubic spline interpolation (SciPy), adaptive thresholding
- **Analysis**: RMS computation, segment merging, cumulative ratio analysis

## Project Structure

```
├── main.py             # End-to-end pipeline (single + multi-session CLI)
├── preprocessing.py    # Data loading, filtering, FFT smoothing
├── segmentation.py     # Envelope extraction, segment detection
├── analysis.py         # RMS computation, visualization
├── fatigue_model.py    # Multi-session fatigue model with spline interpolation
└── README.md
```

## Usage

### Single Session Analysis

```bash
python main.py single --file <path_to_emg_csv> [--fs 250] [--cutoff 5.0] [--std 2.0] [--gap 4000] [--save ./results]
```

### Multi-Session Fatigue Tracking

```bash
python main.py multi --files session1.csv session2.csv session3.csv --timestamps 0 3 7 [--save ./results]
```

| Argument       | Description                              | Default  |
|----------------|------------------------------------------|----------|
| `--file/files` | Path(s) to EMG data (CSV)                | required |
| `--timestamps` | Session time points (multi-session mode) | required |
| `--fs`         | Sampling rate (Hz)                       | 250      |
| `--cutoff`     | FFT smoothing cutoff (Hz)                | 5.0      |
| `--std`        | Threshold = mean + std * this            | 2.0      |
| `--gap`        | Max gap for segment merging (samples)    | 4000     |
| `--save`       | Directory to save output figures         | None     |

## Key Approaches

- **Hilbert Transform Envelope**: Provides a smooth, physically meaningful amplitude envelope for reliable contraction detection.
- **Adaptive Thresholding**: Detection threshold adapts to each subject's signal characteristics (mean + k*std), making the pipeline robust across individuals.
- **Segment Merging**: Prevents over-segmentation from brief signal dropouts during sustained contractions.
- **Spline-Interpolated Capacity**: Tracks how muscle capacity evolves across sessions, enabling progressively more accurate fatigue estimation as measurement data accumulates.
