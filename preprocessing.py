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
    Load EMG data from a CSV/TXT file.

    Args:
        file_path: Path to the data file (CSV format with 'Time' and ' EMG(CH1)' columns).

    Returns:
        time: Time array in seconds.
        emg_data: Raw EMG signal array.
    """
    data = pd.read_csv(file_path, sep=",")
    time = data["Time"].values
    emg_data = data[" EMG(CH1)"].values
    return time, emg_data


def apply_notch_filter(
    signal: np.ndarray, freq: float, quality_factor: float, sampling_rate: float
) -> np.ndarray:
    """
    Apply a notch filter to remove a specific frequency component.

    Args:
        signal: Input signal array.
        freq: Frequency to remove (Hz).
        quality_factor: Q factor of the notch filter.
        sampling_rate: Sampling rate of the signal (Hz).

    Returns:
        Filtered signal with the target frequency removed.
    """
    b, a = iirnotch(freq, quality_factor, sampling_rate)
    return filtfilt(b, a, signal)


def apply_bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    sampling_rate: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter.

    Args:
        signal: Input signal array.
        lowcut: Lower cutoff frequency (Hz).
        highcut: Upper cutoff frequency (Hz).
        sampling_rate: Sampling rate of the signal (Hz).
        order: Filter order (default: 4).

    Returns:
        Bandpass-filtered signal.
    """
    nyquist = sampling_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def denoise_emg(
    emg_data: np.ndarray,
    sampling_rate: float = 250.0,
    notch_freqs: list[float] = [60.0, 120.0],
    bandpass_range: tuple[float, float] = (20.0, 124.0),
) -> np.ndarray:
    """
    Full denoising pipeline: Notch filtering + Bandpass filtering.

    Removes powerline interference (60Hz and harmonics) and retains
    only the EMG-relevant frequency band.

    Args:
        emg_data: Raw EMG signal.
        sampling_rate: Sampling rate (Hz).
        notch_freqs: List of frequencies to notch out.
        bandpass_range: (low, high) cutoff frequencies for bandpass filter.

    Returns:
        Denoised EMG signal.
    """
    quality_factor = 30.0
    signal = emg_data.copy()

    for freq in notch_freqs:
        signal = apply_notch_filter(signal, freq, quality_factor, sampling_rate)

    signal = apply_bandpass_filter(
        signal, bandpass_range[0], bandpass_range[1], sampling_rate
    )
    return signal


def smooth_signal_fft(
    signal: np.ndarray, sampling_rate: float, cutoff_freq: float = 5.0
) -> np.ndarray:
    """
    Smooth a signal by retaining only low-frequency components via FFT.

    Args:
        signal: Input signal array.
        sampling_rate: Sampling rate (Hz).
        cutoff_freq: Maximum frequency to retain (Hz).

    Returns:
        Smoothed signal with only low-frequency components.
    """
    fft_result = fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1 / sampling_rate)

    mask = np.abs(frequencies) <= cutoff_freq
    filtered_fft = np.zeros_like(fft_result, dtype=complex)
    filtered_fft[mask] = fft_result[mask]

    return np.real(ifft(filtered_fft))
