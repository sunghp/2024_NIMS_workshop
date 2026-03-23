# EMG Muscle Fatigue Analysis

Automated pipeline for analyzing muscle fatigue patterns from EMG (Electromyography) signals. Developed during the **NIMS Industrial Mathematics Problem-Solving Workshop**, where a partnering company provided real-world EMG data for muscle contraction prediction.

## Overview

This project processes raw EMG signals to quantify muscle fatigue over repeated contractions. The pipeline automatically detects individual contraction segments and computes per-segment RMS values to track how muscle activation changes over time.

## Pipeline

```
Raw EMG Signal
    │
    ├── 1. Noise Removal
    │       ├── Notch filter (60Hz, 120Hz powerline interference)
    │       └── Bandpass filter (20–124Hz EMG band)
    │
    ├── 2. FFT Smoothing
    │       └── Low-pass filtering (retain < 5Hz components)
    │
    ├── 3. Envelope Extraction
    │       └── Hilbert transform → instantaneous amplitude
    │
    ├── 4. Segment Detection
    │       ├── Adaptive thresholding (mean + k·std)
    │       └── Contiguous region grouping + gap-based merging
    │
    └── 5. Fatigue Quantification
            ├── Per-segment RMS
            ├── RMS × Duration (energy metric)
            └── Cumulative ratio analysis
```

## Tech Stack

- **Python** (NumPy, SciPy, Pandas, Matplotlib)
- **Signal Processing**: Butterworth filters, Notch filters, FFT/IFFT, Hilbert transform
- **Analysis**: RMS computation, adaptive thresholding, segment merging

## Project Structure

```
├── main.py             # End-to-end pipeline with CLI
├── preprocessing.py    # Data loading, filtering, smoothing
├── segmentation.py     # Envelope extraction, segment detection
├── analysis.py         # RMS computation, visualization
└── README.md
```

## Usage

```bash
python main.py --file <path_to_emg_csv> [--fs 250] [--cutoff 5.0] [--std 2.0] [--gap 4000] [--save ./results]
```

| Argument    | Description                        | Default |
|-------------|------------------------------------|---------|
| `--file`    | Path to EMG data (CSV)             | required|
| `--fs`      | Sampling rate (Hz)                 | 250     |
| `--cutoff`  | FFT smoothing cutoff (Hz)          | 5.0     |
| `--std`     | Threshold = mean + std × this      | 2.0     |
| `--gap`     | Max gap for segment merging (samples) | 4000 |
| `--save`    | Directory to save output figures   | None    |

## Note

The EMG dataset used in this project was provided by a partnering company through the NIMS workshop and is not included in this repository due to confidentiality.

## Key Approach

- **Hilbert Transform Envelope**: Instead of simple rectification + moving average, the Hilbert transform provides a smooth, physically meaningful amplitude envelope that better captures the instantaneous energy of muscle contractions.
- **Adaptive Thresholding**: The detection threshold adapts to each subject's signal characteristics (mean + k·std), making the pipeline robust across different individuals and recording conditions.
- **Segment Merging**: Nearby above-threshold regions are merged using a configurable gap parameter, preventing over-segmentation from brief signal dropouts during sustained contractions.
