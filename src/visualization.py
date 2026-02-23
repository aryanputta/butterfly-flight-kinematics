"""
visualization.py — Publication-quality plotting

Generates trajectory plots, angle time-series, velocity/acceleration profiles,
FFT frequency spectra, and validation error charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from typing import Dict, Optional


# Use a clean style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')


def _setup_figure(figsize=(10, 6)):
    """Create a figure with standard formatting."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _save_or_show(fig, save_path: Optional[str] = None, dpi: int = 150):
    """Save figure to file or display interactively."""
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"[INFO] Saved plot: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_trajectory(centroids: np.ndarray, wing_tips: np.ndarray,
                    save_path: Optional[str] = None, species: str = ""):
    """
    Plot 2D trajectory of body centroid and wing tip.
    
    Parameters
    ----------
    centroids : np.ndarray, shape (N, 2)
        Body centroid positions.
    wing_tips : np.ndarray, shape (N, 2)
        Wing tip positions.
    save_path : str, optional
        File path to save figure.
    species : str
        Species name for title.
    """
    fig, ax = _setup_figure(figsize=(10, 8))
    
    # Plot with color gradient showing progression over time
    n = len(centroids)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    
    ax.scatter(centroids[:, 0], centroids[:, 1], c=colors, s=8, alpha=0.7,
               label="Body centroid", zorder=3)
    ax.scatter(wing_tips[:, 0], wing_tips[:, 1], c=colors, s=4, alpha=0.5,
               marker="^", label="Wing tip", zorder=2)
    
    # Connect centroid to wing tip for a few sample frames
    step = max(1, n // 20)
    for i in range(0, n, step):
        ax.plot([centroids[i, 0], wing_tips[i, 0]],
                [centroids[i, 1], wing_tips[i, 1]],
                color=colors[i], alpha=0.3, linewidth=0.8)
    
    ax.set_xlabel("x (pixels)", fontsize=12)
    ax.set_ylabel("y (pixels)", fontsize=12)
    ax.set_title(f"Flight Trajectory — {species}" if species else "Flight Trajectory", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Image coordinates have y increasing downward
    
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, n))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Frame", fontsize=10)
    
    _save_or_show(fig, save_path)


def plot_angle_timeseries(time: np.ndarray, angles_raw: np.ndarray,
                          angles_smoothed: np.ndarray,
                          save_path: Optional[str] = None, species: str = ""):
    """
    Plot angular displacement over time (raw and smoothed).
    
    Parameters
    ----------
    time : np.ndarray
        Time axis in seconds.
    angles_raw : np.ndarray
        Raw angular displacement in degrees.
    angles_smoothed : np.ndarray
        Smoothed angular displacement in degrees.
    """
    fig, ax = _setup_figure()
    
    ax.plot(time, np.degrees(angles_raw), color="#ADB5BD", alpha=0.5,
            linewidth=0.8, label="Raw", zorder=1)
    ax.plot(time, np.degrees(angles_smoothed), color="#228BE6", linewidth=1.5,
            label="Savitzky-Golay smoothed", zorder=2)
    
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Angular Displacement (°)", fontsize=12)
    ax.set_title(f"Wing Angle vs Time — {species}" if species else "Wing Angle vs Time", fontsize=14)
    ax.legend(fontsize=10)
    
    _save_or_show(fig, save_path)


def plot_velocity_acceleration(time: np.ndarray, velocity: np.ndarray,
                               acceleration: np.ndarray,
                               save_path: Optional[str] = None, species: str = ""):
    """
    Plot angular velocity and acceleration on dual-axis plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_vel = "#228BE6"
    color_acc = "#FA5252"
    
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Angular Velocity (rad/s)", color=color_vel, fontsize=12)
    line1 = ax1.plot(time, velocity, color=color_vel, linewidth=1.2, alpha=0.9, label="Velocity")
    ax1.tick_params(axis="y", labelcolor=color_vel)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Angular Acceleration (rad/s²)", color=color_acc, fontsize=12)
    line2 = ax2.plot(time, acceleration, color=color_acc, linewidth=1.0, alpha=0.7, label="Acceleration")
    ax2.tick_params(axis="y", labelcolor=color_acc)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc="upper right")
    
    title = f"Angular Velocity & Acceleration — {species}" if species else "Angular Velocity & Acceleration"
    ax1.set_title(title, fontsize=14)
    fig.tight_layout()
    
    _save_or_show(fig, save_path)


def plot_frequency_spectrum(freqs: np.ndarray, magnitudes: np.ndarray,
                            dominant_freq: float,
                            save_path: Optional[str] = None, species: str = ""):
    """
    Plot FFT magnitude spectrum with dominant frequency marked.
    """
    fig, ax = _setup_figure()
    
    # Only plot up to a reasonable frequency
    max_plot_freq = min(50.0, np.max(freqs) if len(freqs) > 0 else 50.0)
    mask = freqs <= max_plot_freq
    
    ax.plot(freqs[mask], magnitudes[mask], color="#228BE6", linewidth=1.2)
    ax.fill_between(freqs[mask], magnitudes[mask], alpha=0.15, color="#228BE6")
    
    if dominant_freq > 0:
        ax.axvline(dominant_freq, color="#FA5252", linestyle="--", linewidth=1.5,
                   label=f"Dominant: {dominant_freq:.1f} Hz")
    
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    title = f"Wingbeat Frequency Spectrum — {species}" if species else "Wingbeat Frequency Spectrum"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    
    _save_or_show(fig, save_path)


def plot_validation(per_frame_errors: np.ndarray, mean_error: float,
                    save_path: Optional[str] = None):
    """
    Plot validation errors: per-frame MAE as bar chart.
    """
    fig, ax = _setup_figure(figsize=(10, 5))
    
    n = len(per_frame_errors)
    x = np.arange(n)
    
    colors = ["#51CF66" if e < mean_error else "#FA5252" for e in per_frame_errors]
    ax.bar(x, per_frame_errors, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axhline(mean_error, color="#343A40", linestyle="--", linewidth=1.5,
               label=f"Mean MAE: {mean_error:.2f} px")
    
    ax.set_xlabel("Annotated Frame", fontsize=12)
    ax.set_ylabel("Pixel Error", fontsize=12)
    ax.set_title("Tracking Validation: Per-Frame Pixel Error", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks(x)
    
    _save_or_show(fig, save_path)


def plot_dashboard(kinematics: Dict, tracking_results: Dict,
                   save_path: Optional[str] = None, species: str = ""):
    """
    Generate a comprehensive dashboard with all plots in a single figure.
    
    This is useful for quick overview and README screenshots.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    time = kinematics["time"]
    
    # Top-left: Trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    centroids = tracking_results["centroids"]
    wing_tips = tracking_results["wing_tips"]
    n = len(centroids)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    ax1.scatter(centroids[:, 0], centroids[:, 1], c=colors, s=4, alpha=0.7)
    ax1.scatter(wing_tips[:, 0], wing_tips[:, 1], c=colors, s=2, alpha=0.5, marker="^")
    ax1.set_xlabel("x (px)")
    ax1.set_ylabel("y (px)")
    ax1.set_title("Flight Trajectory")
    ax1.set_aspect("equal")
    ax1.invert_yaxis()
    
    # Top-right: Angle vs Time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, np.degrees(kinematics["angles_raw"]), color="#ADB5BD", alpha=0.5, linewidth=0.8)
    ax2.plot(time, np.degrees(kinematics["angles_smoothed"]), color="#228BE6", linewidth=1.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (°)")
    ax2.set_title("Wing Angle vs Time")
    
    # Bottom-left: Velocity & Acceleration
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time, kinematics["angular_velocity"], color="#228BE6", linewidth=1.0, label="Velocity")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Angular Velocity (rad/s)", color="#228BE6")
    ax3b = ax3.twinx()
    ax3b.plot(time, kinematics["angular_acceleration"], color="#FA5252", linewidth=0.8,
              alpha=0.7, label="Acceleration")
    ax3b.set_ylabel("Accel. (rad/s²)", color="#FA5252")
    ax3.set_title("Velocity & Acceleration")
    
    # Bottom-right: FFT
    ax4 = fig.add_subplot(gs[1, 1])
    freqs = kinematics["fft_freqs"]
    mags = kinematics["fft_magnitudes"]
    if len(freqs) > 0:
        max_f = min(50.0, np.max(freqs))
        mask = freqs <= max_f
        ax4.plot(freqs[mask], mags[mask], color="#228BE6", linewidth=1.2)
        ax4.fill_between(freqs[mask], mags[mask], alpha=0.15, color="#228BE6")
        if kinematics["wingbeat_freq_fft"] > 0:
            ax4.axvline(kinematics["wingbeat_freq_fft"], color="#FA5252",
                        linestyle="--", label=f'{kinematics["wingbeat_freq_fft"]:.1f} Hz')
            ax4.legend()
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Magnitude")
    ax4.set_title("Frequency Spectrum")
    
    fig.suptitle(f"Wing Kinematics Dashboard — {species}" if species else "Wing Kinematics Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    
    _save_or_show(fig, save_path)
