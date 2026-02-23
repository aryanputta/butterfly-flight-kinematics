"""
error_propagation.py — Noise propagation analysis for kinematic extraction.

Pixel tracking noise σ_p propagates through the pipeline:

  Angular error:      σ_θ = σ_p√2 / r                (r = wing length in px)
  Velocity error:     σ_v = σ_p · fps · √2 / 2       (central difference)
  Acceleration error: σ_a = σ_p · fps² · √6           (second difference)

Acceleration amplifies noise by fps² — raw second derivatives are
extremely noisy without proper filtering.
"""

import numpy as np


def estimate_tracking_noise(fb_errors: np.ndarray) -> float:
    """Estimate pixel noise σ_p from forward-backward residuals.

    The FB error is a round-trip displacement. For tracked points:
        σ_fb = 2σ_p  →  σ_p = median(fb_error) / 2
    Uses median for robustness against uncaught tracking failures.
    """
    flat = fb_errors.flatten()
    flat = flat[np.isfinite(flat) & (flat > 0)]
    if len(flat) == 0:
        return 0.3  # conservative default
    return float(np.median(flat)) / 2.0


def angular_error_from_pixel_noise(sigma_p: float,
                                   wing_lengths: np.ndarray) -> np.ndarray:
    """Angular error σ_θ(t) from pixel noise and wing length.

    Derivation via error propagation on θ = atan2(Δy, Δx):
        σ_θ = σ_p · √2 / r
    Both tip and thorax contribute noise, hence the √2 factor.
    """
    r = np.maximum(wing_lengths, 1.0)
    return sigma_p * np.sqrt(2.0) / r


def velocity_noise_amplification(sigma_p: float, fps: float) -> float:
    """Expected velocity noise σ_v (px/s) from central difference.

    σ_v = σ_p · fps · √2 / 2
    """
    return sigma_p * fps * np.sqrt(2.0) / 2.0


def acceleration_noise_amplification(sigma_p: float, fps: float) -> float:
    """Expected acceleration noise σ_a (px/s²) from second difference.

    σ_a = σ_p · fps² · √6

    At 120 fps, σ_p = 0.3 px → σ_a ≈ 10,584 px/s². This is why
    raw acceleration is useless without filtering.
    """
    return sigma_p * fps * fps * np.sqrt(6.0)


def angular_velocity_noise(sigma_p: float, fps: float,
                           wing_length: float) -> float:
    """Expected angular velocity noise σ_ω (rad/s).

    σ_ω = σ_p · fps / r
    """
    r = max(wing_length, 1.0)
    return sigma_p * fps / r


def angular_acceleration_noise(sigma_p: float, fps: float,
                               wing_length: float) -> float:
    """Expected angular acceleration noise σ_α (rad/s²).

    σ_α = σ_p · fps² · √3 / r
    """
    r = max(wing_length, 1.0)
    return sigma_p * fps * fps * np.sqrt(3.0) / r


def noise_reduction_summary(sigma_p_raw: float = 0.5,
                            sigma_p_subpix: float = 0.05,
                            wing_length_px: float = 100.0,
                            fps: float = 120.0,
                            filter_window: int = 11) -> dict:
    """Compare noise levels before/after pipeline improvements."""
    r = wing_length_px
    filter_reduction = 1.0 / np.sqrt(filter_window)
    improvement = sigma_p_raw / sigma_p_subpix

    return {
        "raw_pixel_noise_px": sigma_p_raw,
        "subpixel_noise_px": sigma_p_subpix,
        "improvement_factor": improvement,
        "angular_error_raw_deg": np.degrees(sigma_p_raw * np.sqrt(2) / r),
        "angular_error_subpix_deg": np.degrees(sigma_p_subpix * np.sqrt(2) / r),
        "velocity_noise_raw_px_s": velocity_noise_amplification(sigma_p_raw, fps),
        "velocity_noise_subpix_px_s": velocity_noise_amplification(sigma_p_subpix, fps),
        "accel_noise_raw_px_s2": acceleration_noise_amplification(sigma_p_raw, fps),
        "accel_noise_subpix_px_s2": acceleration_noise_amplification(sigma_p_subpix, fps),
        "accel_noise_filtered_px_s2": (
            acceleration_noise_amplification(sigma_p_subpix, fps) * filter_reduction
        ),
    }
