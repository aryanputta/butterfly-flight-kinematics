"""
analysis.py — Signal processing and kinematic analysis

Implements Savitzky-Golay smoothing, central difference numerical differentiation,
FFT-based frequency estimation, and comprehensive kinematics computation.
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple, Optional


def smooth_signal(data: np.ndarray, window_length: int = 11,
                  polyorder: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to reduce noise while preserving signal shape.
    
    The Savitzky-Golay filter is preferred over simple moving average because
    it preserves higher moments (width, height, asymmetry) of the signal,
    which is critical for biomechanical data where waveform shape matters.
    
    Parameters
    ----------
    data : np.ndarray
        Input signal (1D).
    window_length : int
        Length of the filter window (must be odd and > polyorder).
    polyorder : int
        Order of the polynomial used to fit the samples.
    
    Returns
    -------
    smoothed : np.ndarray
        Smoothed signal, same length as input.
    """
    if len(data) < window_length:
        # Not enough data for the window — return as-is
        return data.copy()
    
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    return savgol_filter(data, window_length, polyorder)


def differentiate(data: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute numerical derivative using central differences.
    
    Central difference formula: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    
    At boundaries, forward/backward differences are used:
    - f'(x₀) ≈ (f(x₁) - f(x₀)) / h         (forward)
    - f'(xₙ) ≈ (f(xₙ) - f(xₙ₋₁)) / h       (backward)
    
    Parameters
    ----------
    data : np.ndarray
        Input signal (1D).
    dt : float
        Time step between samples (1/fps for video data).
    
    Returns
    -------
    derivative : np.ndarray
        Numerical derivative, same length as input.
    """
    n = len(data)
    if n < 2:
        return np.zeros_like(data)
    
    deriv = np.zeros(n)
    
    # Central difference for interior points
    deriv[1:-1] = (data[2:] - data[:-2]) / (2.0 * dt)
    
    # Forward difference at left boundary
    deriv[0] = (data[1] - data[0]) / dt
    
    # Backward difference at right boundary
    deriv[-1] = (data[-1] - data[-2]) / dt
    
    return deriv


def compute_frequency(angle_data: np.ndarray, fps: float,
                      min_freq: float = 1.0,
                      max_freq: float = 100.0) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate wingbeat frequency using FFT and peak detection.
    
    Parameters
    ----------
    angle_data : np.ndarray
        Angular displacement time series.
    fps : float
        Video frame rate (samples per second).
    min_freq : float
        Minimum expected wingbeat frequency (Hz).
    max_freq : float
        Maximum expected wingbeat frequency (Hz).
    
    Returns
    -------
    dominant_freq : float
        Estimated wingbeat frequency in Hz.
    freqs : np.ndarray
        Frequency axis for the spectrum.
    magnitudes : np.ndarray
        FFT magnitude spectrum (positive frequencies only).
    """
    n = len(angle_data)
    if n < 4:
        return 0.0, np.array([]), np.array([])
    
    # Remove DC component (mean)
    centered = angle_data - np.mean(angle_data)
    
    # Apply Hanning window to reduce spectral leakage
    windowed = centered * np.hanning(n)
    
    # Compute FFT
    yf = fft(windowed)
    xf = fftfreq(n, 1.0 / fps)
    
    # Take only positive frequencies
    pos_mask = xf > 0
    freqs = xf[pos_mask]
    magnitudes = 2.0 / n * np.abs(yf[pos_mask])
    
    # Filter to biologically plausible range
    range_mask = (freqs >= min_freq) & (freqs <= max_freq)
    
    if not np.any(range_mask):
        return 0.0, freqs, magnitudes
    
    # Find peaks in the spectrum
    filtered_mags = magnitudes.copy()
    filtered_mags[~range_mask] = 0.0
    
    peaks, properties = find_peaks(filtered_mags, height=0.01 * np.max(filtered_mags))
    
    if len(peaks) == 0:
        # Fall back to global maximum in range
        idx = np.argmax(filtered_mags)
        dominant_freq = freqs[idx]
    else:
        # Use the highest peak
        best_peak = peaks[np.argmax(filtered_mags[peaks])]
        dominant_freq = freqs[best_peak]
    
    return dominant_freq, freqs, magnitudes


def estimate_frequency_peaks(angle_data: np.ndarray, fps: float,
                             min_distance_frames: int = 5) -> float:
    """
    Estimate wingbeat frequency using time-domain peak detection.
    
    This is a complementary method to FFT — useful for validation.
    
    Parameters
    ----------
    angle_data : np.ndarray
        Angular displacement time series.
    fps : float
        Frame rate.
    min_distance_frames : int
        Minimum frames between peaks.
    
    Returns
    -------
    freq : float
        Estimated frequency in Hz.
    """
    smoothed = smooth_signal(angle_data, window_length=7, polyorder=2)
    peaks, _ = find_peaks(smoothed, distance=min_distance_frames)
    
    if len(peaks) < 2:
        return 0.0
    
    # Average period between consecutive peaks
    periods = np.diff(peaks) / fps
    mean_period = np.mean(periods)
    
    if mean_period <= 0:
        return 0.0
    
    return 1.0 / mean_period


def compute_kinematics(angles: np.ndarray, fps: float,
                       smooth_window: int = 11,
                       smooth_poly: int = 3) -> Dict:
    """
    Compute full kinematic analysis from angular displacement data.
    
    Pipeline:
    1. Smooth raw angles with Savitzky-Golay filter
    2. Compute angular velocity via central difference
    3. Compute angular acceleration via central difference of velocity
    4. Estimate wingbeat frequency via FFT
    5. Compute stroke amplitude (peak-to-peak)
    
    Parameters
    ----------
    angles : np.ndarray
        Raw angular displacement in radians.
    fps : float
        Frame rate.
    smooth_window : int
        Savitzky-Golay window length.
    smooth_poly : int
        Savitzky-Golay polynomial order.
    
    Returns
    -------
    kinematics : dict
        Keys: angles_raw, angles_smoothed, angular_velocity, angular_acceleration,
              wingbeat_freq_fft, wingbeat_freq_peaks, stroke_amplitude,
              fft_freqs, fft_magnitudes, time
    """
    dt = 1.0 / fps
    n = len(angles)
    time = np.arange(n) * dt
    
    # Step 1: Smooth
    angles_smoothed = smooth_signal(angles, smooth_window, smooth_poly)
    
    # Step 2: Angular velocity (rad/s) via central difference
    angular_velocity = differentiate(angles_smoothed, dt)
    
    # Step 3: Angular acceleration (rad/s²) via central difference of velocity
    angular_acceleration = differentiate(angular_velocity, dt)
    
    # Step 4: Frequency estimation
    wingbeat_freq_fft, fft_freqs, fft_mags = compute_frequency(angles_smoothed, fps)
    wingbeat_freq_peaks = estimate_frequency_peaks(angles_smoothed, fps)
    
    # Step 5: Stroke amplitude (peak-to-peak in degrees)
    stroke_amplitude_rad = np.max(angles_smoothed) - np.min(angles_smoothed)
    stroke_amplitude_deg = np.degrees(stroke_amplitude_rad)
    
    return {
        "time": time,
        "angles_raw": angles,
        "angles_smoothed": angles_smoothed,
        "angular_velocity": angular_velocity,
        "angular_acceleration": angular_acceleration,
        "wingbeat_freq_fft": wingbeat_freq_fft,
        "wingbeat_freq_peaks": wingbeat_freq_peaks,
        "stroke_amplitude_rad": stroke_amplitude_rad,
        "stroke_amplitude_deg": stroke_amplitude_deg,
        "fft_freqs": fft_freqs,
        "fft_magnitudes": fft_mags,
    }
