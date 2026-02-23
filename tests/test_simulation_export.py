"""
test_simulation_export.py — Tests for simulation export and new analysis functions.
"""

import numpy as np
import pytest
import os
import tempfile

from src.analysis import (
    butterworth_lowpass,
    ransac_fit_rigid,
    compute_stroke_angle,
)
from src.error_propagation import (
    estimate_tracking_noise,
    angular_error_from_pixel_noise,
    velocity_noise_amplification,
    acceleration_noise_amplification,
)
from src.simulation_export import (
    theta_2harm,
    export_csv,
    export_julia,
    export_matlab,
    sensitivity_jacobian,
    uncertainty_propagation,
)


class TestButterworthLowpass:

    def test_preserves_low_frequency(self):
        """A 5 Hz signal should pass through a 20 Hz cutoff unchanged."""
        fps = 100.0
        t = np.arange(0, 2, 1.0 / fps)
        signal = np.sin(2 * np.pi * 5.0 * t)
        filtered = butterworth_lowpass(signal, fps, cutoff_hz=20.0)
        # correlation should be very high
        corr = np.corrcoef(signal[20:-20], filtered[20:-20])[0, 1]
        assert corr > 0.99

    def test_attenuates_high_frequency(self):
        """A 40 Hz signal should be strongly attenuated by a 10 Hz cutoff."""
        fps = 100.0
        t = np.arange(0, 2, 1.0 / fps)
        signal = np.sin(2 * np.pi * 40.0 * t)
        filtered = butterworth_lowpass(signal, fps, cutoff_hz=10.0)
        # should be much smaller amplitude
        assert np.std(filtered) < 0.1 * np.std(signal)

    def test_short_signal_returns_copy(self):
        """Signal shorter than filter order should return as-is."""
        data = np.array([1.0, 2.0, 3.0])
        result = butterworth_lowpass(data, 100.0, 10.0, order=4)
        np.testing.assert_array_equal(result, data)


class TestRANSACRigid:

    def test_detects_inliers(self):
        """With a clean rigid motion, all points should be inliers."""
        np.random.seed(42)
        pts_prev = np.random.randn(20, 2).astype(np.float32) * 50 + 100
        # Pure translation
        pts_curr = pts_prev + np.array([5.0, -3.0])
        mask, residuals = ransac_fit_rigid(pts_prev, pts_curr)
        assert np.all(mask)

    def test_rejects_outliers(self):
        """Should reject points that don't follow rigid motion."""
        np.random.seed(42)
        pts_prev = np.random.randn(20, 2).astype(np.float32) * 50 + 100
        pts_curr = pts_prev + np.array([5.0, -3.0])
        # Add 3 outliers with large displacement
        pts_curr[0] += 50.0
        pts_curr[1] += 40.0
        pts_curr[2] -= 60.0
        mask, residuals = ransac_fit_rigid(pts_prev, pts_curr, inlier_threshold=5.0)
        assert not mask[0]
        assert not mask[1]
        assert not mask[2]
        assert np.sum(mask) >= 15  # most of the rest should be inliers

    def test_few_points_returns_all_valid(self):
        """With < 3 points, should return all as inliers."""
        pts = np.array([[1, 2], [3, 4]], dtype=np.float32)
        mask, _ = ransac_fit_rigid(pts, pts + 1)
        assert np.all(mask)


class TestStrokeAngle:

    def test_horizontal_tip(self):
        """Tip directly to the right of thorax → θ ≈ 0."""
        tip = np.array([[200, 100]], dtype=np.float64)
        thorax = np.array([100, 100], dtype=np.float64)
        theta = compute_stroke_angle(tip, thorax)
        assert abs(theta[0]) < 0.01

    def test_vertical_tip(self):
        """Tip directly above thorax (y < thorax_y in image) → θ ≈ π/2."""
        tip = np.array([[100, 50]], dtype=np.float64)
        thorax = np.array([100, 100], dtype=np.float64)
        theta = compute_stroke_angle(tip, thorax)
        assert abs(theta[0] - np.pi / 2) < 0.01

    def test_known_angle(self):
        """45 degree angle."""
        tip = np.array([[200, 0]], dtype=np.float64)
        thorax = np.array([100, 100], dtype=np.float64)
        theta = compute_stroke_angle(tip, thorax)
        assert abs(theta[0] - np.pi / 4) < 0.01


class TestErrorPropagation:

    def test_angular_error_scaling(self):
        """Angular error should decrease with wing length."""
        sigma_p = 0.3
        r_short = np.array([50.0])
        r_long = np.array([200.0])
        err_short = angular_error_from_pixel_noise(sigma_p, r_short)
        err_long = angular_error_from_pixel_noise(sigma_p, r_long)
        assert err_short[0] > err_long[0]
        # should scale as 1/r
        ratio = err_short[0] / err_long[0]
        assert abs(ratio - 4.0) < 0.1

    def test_velocity_noise_scales_with_fps(self):
        """Velocity noise should be proportional to fps."""
        sigma_p = 0.3
        v60 = velocity_noise_amplification(sigma_p, 60.0)
        v120 = velocity_noise_amplification(sigma_p, 120.0)
        assert abs(v120 / v60 - 2.0) < 0.01

    def test_acceleration_noise_scales_with_fps_squared(self):
        """Acceleration noise should scale as fps²."""
        sigma_p = 0.3
        a60 = acceleration_noise_amplification(sigma_p, 60.0)
        a120 = acceleration_noise_amplification(sigma_p, 120.0)
        assert abs(a120 / a60 - 4.0) < 0.01

    def test_fb_noise_estimate(self):
        """Known FB errors should give expected noise estimate."""
        fb = np.array([0.6, 0.8, 0.4, 0.6, 0.5])
        sigma = estimate_tracking_noise(fb)
        assert sigma == np.median(fb) / 2.0


class TestSimulationExport:

    def test_csv_export(self):
        """CSV export should produce a readable file."""
        params = {"A": 1.0, "f": 10.0, "h": 0.2, "phi": 0.5, "offset": 0.0}
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            export_csv(params, 0.0, 1.0, path, n_points=100)
            import pandas as pd
            data = pd.read_csv(path, comment='#')
            assert len(data) == 100
            assert "time_s" in data.columns
        finally:
            os.unlink(path)

    def test_julia_export(self):
        """Julia export should produce valid syntax."""
        params = {"A": 1.0, "f": 10.0, "h": 0.2, "phi": 0.5, "offset": 0.0}
        with tempfile.NamedTemporaryFile(suffix=".jl", delete=False, mode='w') as f:
            path = f.name
        try:
            export_julia(params, path)
            with open(path) as f:
                content = f.read()
            assert "function theta(t)" in content
            assert "sin" in content
        finally:
            os.unlink(path)

    def test_sensitivity_jacobian(self):
        """Jacobian entries should have correct signs/shapes."""
        t = np.linspace(0, 1, 100)
        params = {"A": 1.0, "f": 10.0, "h": 0.2, "phi": 0.5, "offset": 0.0}
        jac = sensitivity_jacobian(t, params)
        assert jac["dtheta_dA"].shape == (100,)
        assert jac["dtheta_doffset"].shape == (100,)
        # dtheta/doffset should always be 1
        np.testing.assert_allclose(jac["dtheta_doffset"], 1.0)

    def test_jacobian_matches_numerical(self):
        """Analytical Jacobian should match finite differences."""
        t = np.linspace(0, 0.5, 200)
        params = {"A": 1.0, "f": 10.0, "h": 0.2, "phi": 0.5, "offset": 0.1}
        jac = sensitivity_jacobian(t, params)

        eps = 1e-6
        theta_base = theta_2harm(t, **{k: v for k, v in params.items()})

        # check dtheta/dA numerically
        params_pert = dict(params)
        params_pert["A"] += eps
        theta_pert = theta_2harm(t, **{k: v for k, v in params_pert.items()})
        numerical = (theta_pert - theta_base) / eps
        np.testing.assert_allclose(jac["dtheta_dA"], numerical, atol=1e-4)

    def test_uncertainty_propagation(self):
        """Monte Carlo uncertainty should produce reasonable bounds."""
        t = np.linspace(0, 0.5, 100)
        params = {"A": 1.0, "f": 10.0, "h": 0.2, "phi": 0.5, "offset": 0.0}
        uncertainties = {"A": 0.05, "f": 0.1, "h": 0.02, "phi": 0.1}
        result = uncertainty_propagation(t, params, uncertainties, n_samples=500)
        assert result["theta_std"].shape == (100,)
        assert np.all(result["theta_ci_upper"] >= result["theta_ci_lower"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
