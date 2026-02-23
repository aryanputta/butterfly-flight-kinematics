"""
export.py — Data export module

Exports tracking results and kinematic analysis to CSV with metadata headers.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Optional
from datetime import datetime


def to_csv(kinematics: Dict, tracking_results: Dict,
           output_path: str, species: str = "Unknown") -> str:
    """
    Export kinematic analysis results to CSV with metadata header.
    
    Output CSV contains:
    - Header comment lines with metadata (species, fps, frequency, etc.)
    - Columns: frame, time_s, angle_rad, angle_deg, angle_smoothed_rad,
               angle_smoothed_deg, angular_velocity_rad_s, angular_acceleration_rad_s2,
               centroid_x, centroid_y, wing_tip_x, wing_tip_y
    
    Parameters
    ----------
    kinematics : dict
        Output from analysis.compute_kinematics().
    tracking_results : dict
        Output from tracker.run_tracking().
    output_path : str
        Path to save CSV file.
    species : str
        Species name.
    
    Returns
    -------
    output_path : str
        Path where CSV was saved.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    metadata = tracking_results.get("metadata", {})
    fps = metadata.get("fps", 30.0)
    frames = tracking_results["frames"]
    centroids = tracking_results["centroids"]
    wing_tips = tracking_results["wing_tips"]
    
    # Build DataFrame
    n = len(kinematics["time"])
    
    df = pd.DataFrame({
        "frame": frames[:n],
        "time_s": kinematics["time"],
        "angle_rad": kinematics["angles_raw"],
        "angle_deg": np.degrees(kinematics["angles_raw"]),
        "angle_smoothed_rad": kinematics["angles_smoothed"],
        "angle_smoothed_deg": np.degrees(kinematics["angles_smoothed"]),
        "angular_velocity_rad_s": kinematics["angular_velocity"],
        "angular_acceleration_rad_s2": kinematics["angular_acceleration"],
        "centroid_x": centroids[:n, 0],
        "centroid_y": centroids[:n, 1],
        "wing_tip_x": wing_tips[:n, 0],
        "wing_tip_y": wing_tips[:n, 1],
    })
    
    # Write metadata header as comments, then write DataFrame
    with open(output_path, "w") as f:
        f.write(f"# Species: {species}\n")
        f.write(f"# Source: Mitsunori Denda NJVID Insect Flight Collection\n")
        f.write(f"# URL: https://www.njvid.net/showcollection.php?pid=njcore:16509\n")
        f.write(f"# Frame Rate (fps): {fps}\n")
        f.write(f"# Total Frames Tracked: {n}\n")
        f.write(f"# Duration (s): {kinematics['time'][-1]:.4f}\n")
        f.write(f"# Wingbeat Frequency FFT (Hz): {kinematics['wingbeat_freq_fft']:.2f}\n")
        f.write(f"# Wingbeat Frequency Peaks (Hz): {kinematics['wingbeat_freq_peaks']:.2f}\n")
        f.write(f"# Stroke Amplitude (deg): {kinematics['stroke_amplitude_deg']:.2f}\n")
        f.write(f"# Export Date: {datetime.now().isoformat()}\n")
        f.write(f"#\n")
    
    df.to_csv(output_path, mode="a", index=False)
    
    print(f"[INFO] Exported {n} frames to {output_path}")
    return output_path


def generate_summary(kinematics: Dict, species: str = "Unknown") -> str:
    """
    Generate a human-readable summary of the kinematic analysis.
    
    Parameters
    ----------
    kinematics : dict
        Output from analysis.compute_kinematics().
    species : str
        Species name.
    
    Returns
    -------
    summary : str
        Formatted summary string.
    """
    lines = [
        f"═══ Kinematic Summary: {species} ═══",
        f"  Duration:                {kinematics['time'][-1]:.3f} s",
        f"  Frames analyzed:         {len(kinematics['time'])}",
        f"  Wingbeat frequency (FFT): {kinematics['wingbeat_freq_fft']:.2f} Hz",
        f"  Wingbeat frequency (peaks): {kinematics['wingbeat_freq_peaks']:.2f} Hz",
        f"  Stroke amplitude:        {kinematics['stroke_amplitude_deg']:.1f}°",
        f"  Max angular velocity:    {np.max(np.abs(kinematics['angular_velocity'])):.1f} rad/s",
        f"  Max angular acceleration: {np.max(np.abs(kinematics['angular_acceleration'])):.1f} rad/s²",
        "═" * 45,
    ]
    return "\n".join(lines)
