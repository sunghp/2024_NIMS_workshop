"""
EMG Signal Preprocessing Module

Handles data loading, noise removal (Notch + Bandpass filtering),
and FFT-based signal smoothing.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from scipy.fft import fft, ifft


def load_emg_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load EMG data from a CSV file.

    Args:
        file_path: Path to the data file.

    Returns:
        time: Time array in seconds.
        emg_data: Raw EMG signal array.
    """
    df = pd.read_csv(file_path)
    time_col = [c for c in df.columns if 'time' in c.lower()][0]
    emg_col = [c for c in df.columns if 'emg' in c.lower()][0]
    return df[time_col].values, df[emg_col].values


def apply_notch_filter(signal: np.ndarray, freq: float,
                       fs: float, Q: float = 30.0) -> np.ndarray:
    """
    Apply a notch filter to remove powerline interference.

    Args:
        signal: Input signal.
        freq: Frequency to remove (e.g., 60Hz).
        fs: Sampling rate (Hz).
        Q: Quality factor.

    Returns:
        Filtered signal.
    """
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal)


def apply_bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float,
                          fs: float, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter.

    Args:
        signal: Input signal.
        lowcut: Lower cutoff frequency (Hz).
        highcut: Upper cutoff frequency (Hz).
        fs: Sampling rate (Hz).
        order: Filter order.

    Returns:
        Bandpass-filtered signal.
    """
    nyquist = fs / 2.0
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return filtfilt(b, a, signal)


def denoise_emg(signal: np.ndarray, fs: float,
                notch_freqs: list[float] = [60.0, 120.0],
                bandpass: tuple[float, float] = (20.0, 124.0)) -> np.ndarray:
    """
    Full denoising pipeline: notch filters + bandpass filter.

    Args:
        signal: Raw EMG signal.
        fs: Sampling rate (Hz).
        notch_freqs: Powerline frequencies to remove.
        bandpass: (low, high) cutoff frequencies for bandpass.

    Returns:
        Denoised EMG signal.
    """
    result = signal.copy()
    for freq in notch_freqs:
        result = apply_notch_filter(result, freq, fs)
    result = apply_bandpass_filter(result, bandpass[0], bandpass[1], fs)
    return result


def smooth_signal_fft(signal: np.ndarray, fs: float,
                      cutoff: float = 5.0) -> np.ndarray:
    """
    Smooth signal by retaining only low-frequency FFT components.

    Args:
        signal: Input signal.
        fs: Sampling rate (Hz).
        cutoff: Maximum frequency to retain (Hz).

    Returns:
        Smoothed signal (real part of inverse FFT).
    """
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    spectrum = fft(signal)
    spectrum[np.abs(freqs) > cutoff] = 0
    return np.real(ifft(spectrum))
