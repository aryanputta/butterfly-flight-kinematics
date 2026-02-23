"""
test_analysis.py — Tests for signal processing and kinematic analysis

Uses synthetic sinusoidal signals with known ground truth to verify:
- Frequency detection accuracy (FFT and peak-based)
- Central difference derivative accuracy
- Savitzky-Golay filter phase preservation
"""

import numpy as np
import pytest
from src.analysis import (
    smooth_signal,
    differentiate,
    compute_frequency,
    estimate_frequency_peaks,
    compute_kinematics,
)


class TestSmoothSignal:
    """Tests for Savitzky-Golay smoothing."""
    
    def test_preserves_constant_signal(self):
        """Smoothing a constant signal should return the same signal."""
        data = np.ones(100) * 5.0
        smoothed = smooth_signal(data, window_length=11, polyorder=3)
        np.testing.assert_allclose(smoothed, data, atol=1e-10)
    
    def test_preserves_linear_signal(self):
        """Smoothing a linear signal should preserve it (polyorder >= 1)."""
        data = np.linspace(0, 10, 100)
        smoothed = smooth_signal(data, window_length=11, polyorder=3)
        np.testing.assert_allclose(smoothed, data, atol=1e-8)
    
    def test_reduces_noise(self):
        """Smoothed signal should have lower variance than noisy input."""
        np.random.seed(42)
        clean = np.sin(np.linspace(0, 4 * np.pi, 200))
        noisy = clean + np.random.normal(0, 0.3, 200)
        smoothed = smooth_signal(noisy, window_length=15, polyorder=3)
        
        # Smoothed should be closer to clean than noisy is
        error_noisy = np.mean((noisy - clean) ** 2)
        error_smoothed = np.mean((smoothed - clean) ** 2)
        assert error_smoothed < error_noisy
    
    def test_short_signal_returns_copy(self):
        """Signal shorter than window should be returned as-is."""
        data = np.array([1.0, 2.0, 3.0])
        smoothed = smooth_signal(data, window_length=11, polyorder=3)
        np.testing.assert_array_equal(smoothed, data)
    
    def test_phase_preservation(self):
        """Peak locations should not shift significantly after smoothing."""
        t = np.linspace(0, 2 * np.pi, 200)
        data = np.sin(t)
        smoothed = smooth_signal(data, window_length=11, polyorder=3)
        
        # Peak of sin should be near index 50 (π/2)
        raw_peak = np.argmax(data)
        smoothed_peak = np.argmax(smoothed)
        assert abs(raw_peak - smoothed_peak) <= 1


class TestDifferentiate:
    """Tests for central difference numerical differentiation."""
    
    def test_linear_function(self):
        """Derivative of f(x) = 3x should be 3 everywhere."""
        dt = 0.01
        x = np.arange(0, 1, dt)
        data = 3.0 * x
        deriv = differentiate(data, dt)
        
        # Interior points should be very close to 3.0
        np.testing.assert_allclose(deriv[1:-1], 3.0, atol=1e-10)
    
    def test_quadratic_function(self):
        """Derivative of f(x) = x² should be 2x."""
        dt = 0.001
        x = np.arange(0, 1, dt)
        data = x ** 2
        deriv = differentiate(data, dt)
        expected = 2.0 * x
        
        # Central difference is exact for quadratics
        np.testing.assert_allclose(deriv[1:-1], expected[1:-1], atol=1e-6)
    
    def test_sinusoidal_function(self):
        """Derivative of sin(t) should approximate cos(t)."""
        dt = 0.001
        t = np.arange(0, 2 * np.pi, dt)
        data = np.sin(t)
        deriv = differentiate(data, dt)
        expected = np.cos(t)
        
        # Interior points should be close (central diff error ~ O(dt²))
        np.testing.assert_allclose(deriv[10:-10], expected[10:-10], atol=1e-4)
    
    def test_single_point(self):
        """Single-point array should return zero."""
        data = np.array([5.0])
        deriv = differentiate(data, 1.0)
        assert deriv[0] == 0.0
    
    def test_output_length_matches_input(self):
        """Output length should equal input length."""
        data = np.random.randn(50)
        deriv = differentiate(data, 0.01)
        assert len(deriv) == len(data)


class TestComputeFrequency:
    """Tests for FFT-based frequency estimation."""
    
    def test_known_frequency(self):
        """Detect a known 10 Hz signal."""
        fps = 100.0
        duration = 5.0
        t = np.arange(0, duration, 1.0 / fps)
        freq_true = 10.0
        data = np.sin(2 * np.pi * freq_true * t)
        
        freq_detected, freqs, mags = compute_frequency(data, fps)
        
        # Should be within 0.5 Hz of true frequency
        assert abs(freq_detected - freq_true) < 0.5, \
            f"Detected {freq_detected} Hz, expected {freq_true} Hz"
    
    def test_known_frequency_with_noise(self):
        """Detect a 15 Hz signal buried in noise."""
        np.random.seed(42)
        fps = 100.0
        duration = 5.0
        t = np.arange(0, duration, 1.0 / fps)
        freq_true = 15.0
        data = np.sin(2 * np.pi * freq_true * t) + 0.5 * np.random.randn(len(t))
        
        freq_detected, _, _ = compute_frequency(data, fps)
        assert abs(freq_detected - freq_true) < 1.0
    
    def test_multiple_frequencies_picks_dominant(self):
        """When two frequencies are present, should pick the larger amplitude one."""
        fps = 100.0
        duration = 5.0
        t = np.arange(0, duration, 1.0 / fps)
        data = 2.0 * np.sin(2 * np.pi * 8.0 * t) + 0.5 * np.sin(2 * np.pi * 20.0 * t)
        
        freq_detected, _, _ = compute_frequency(data, fps)
        # Should detect 8 Hz (larger amplitude)
        assert abs(freq_detected - 8.0) < 1.0
    
    def test_empty_signal(self):
        """Short signal should return 0 frequency."""
        data = np.array([1.0, 2.0])
        freq, _, _ = compute_frequency(data, 100.0)
        assert freq == 0.0


class TestEstimateFrequencyPeaks:
    """Tests for time-domain peak-based frequency estimation."""
    
    def test_known_frequency(self):
        """Detect a 10 Hz signal using peak detection."""
        fps = 100.0
        duration = 5.0
        t = np.arange(0, duration, 1.0 / fps)
        freq_true = 10.0
        data = np.sin(2 * np.pi * freq_true * t)
        
        freq_detected = estimate_frequency_peaks(data, fps)
        assert abs(freq_detected - freq_true) < 1.0


class TestComputeKinematics:
    """Tests for the full kinematics pipeline."""
    
    def test_returns_all_keys(self):
        """Output should contain all expected keys."""
        fps = 100.0
        t = np.arange(0, 2, 1.0 / fps)
        angles = np.sin(2 * np.pi * 5 * t)
        
        result = compute_kinematics(angles, fps)
        
        expected_keys = [
            "time", "angles_raw", "angles_smoothed",
            "angular_velocity", "angular_acceleration",
            "wingbeat_freq_fft", "wingbeat_freq_peaks",
            "stroke_amplitude_rad", "stroke_amplitude_deg",
            "fft_freqs", "fft_magnitudes",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_time_axis_correct(self):
        """Time axis should match expected duration."""
        fps = 100.0
        n = 200
        angles = np.sin(np.linspace(0, 4 * np.pi, n))
        
        result = compute_kinematics(angles, fps)
        
        assert len(result["time"]) == n
        np.testing.assert_allclose(result["time"][-1], (n - 1) / fps)
    
    def test_stroke_amplitude(self):
        """Stroke amplitude of sin wave should be ~2 (peak-to-peak)."""
        fps = 100.0
        t = np.arange(0, 5, 1.0 / fps)
        angles = np.sin(2 * np.pi * 5 * t)
        
        result = compute_kinematics(angles, fps)
        
        # Peak-to-peak of sin is 2.0
        assert abs(result["stroke_amplitude_rad"] - 2.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
