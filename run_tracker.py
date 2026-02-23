"""
run_tracker.py — Main entry point for the wing tracking pipeline

Orchestrates: video loading → ROI selection → tracking → analysis → export → plots

Usage:
    # With real video (interactive GUI)
    python run_tracker.py --video data/raw/morpho_peleides.mp4 --species "Morpho peleides"
    
    # With synthetic demo data (no video required)
    python run_tracker.py --demo
    
    # Generate 3D wing model
    python run_tracker.py --demo --model
"""

import argparse
import numpy as np
import os
import sys

from src.analysis import compute_kinematics
from src.export import to_csv, generate_summary
from src.visualization import (
    plot_trajectory,
    plot_angle_timeseries,
    plot_velocity_acceleration,
    plot_frequency_spectrum,
    plot_dashboard,
)
from src.wing_model import generate_full_model, generate_flap_sequence, SPECIES_PARAMS


def generate_synthetic_data(fps: float = 120.0, duration: float = 2.0,
                            freq: float = 12.0, species: str = "Synthetic") -> dict:
    """
    Generate synthetic tracking data for demonstration and testing.
    
    Simulates a butterfly with:
    - Oscillating wing angle at specified frequency
    - Slight forward body translation
    - Realistic noise
    
    Parameters
    ----------
    fps : float
        Simulated frame rate.
    duration : float
        Duration in seconds.
    freq : float
        Wingbeat frequency in Hz.
    species : str
        Species label for metadata.
    
    Returns
    -------
    results : dict
        Synthetic tracking results matching run_tracking() output format.
    """
    np.random.seed(42)
    
    n_frames = int(fps * duration)
    t = np.arange(n_frames) / fps
    
    # Simulate body centroid — slight forward drift
    cx = 320 + 30 * t + 2 * np.random.randn(n_frames)
    cy = 240 + 5 * np.sin(0.5 * t) + 1.5 * np.random.randn(n_frames)
    
    # Simulate wing angle — nonlinear oscillation (not pure sine)
    # Adding harmonics to make it more realistic
    angle = (0.8 * np.sin(2 * np.pi * freq * t)
             + 0.15 * np.sin(4 * np.pi * freq * t + 0.3)
             + 0.05 * np.sin(6 * np.pi * freq * t + 0.7))
    angle += 0.05 * np.random.randn(n_frames)  # Measurement noise
    
    # Simulate wing tip position from angle
    wing_length = 45  # pixels
    wx = cx + wing_length * np.cos(angle)
    wy = cy + wing_length * np.sin(angle)
    
    results = {
        "frames": np.arange(n_frames),
        "centroids": np.column_stack([cx, cy]),
        "wing_tips": np.column_stack([wx, wy]),
        "angles": angle,
        "metadata": {
            "fps": fps,
            "frame_count": n_frames,
            "width": 640,
            "height": 480,
            "duration_s": duration,
            "species": species,
        },
    }
    
    return results


def run_demo(args):
    """Run the full pipeline with synthetic data."""
    species = args.species if args.species else "Morpho peleides"
    
    print(f"{'═' * 60}")
    print(f"  denda-njvid-flight-tracker — DEMO MODE")
    print(f"  Species: {species} (synthetic data)")
    print(f"{'═' * 60}")
    print()
    
    # Generate synthetic data
    print("[1/5] Generating synthetic tracking data...")
    results = generate_synthetic_data(
        fps=120.0, duration=2.0, freq=12.0, species=species
    )
    
    # Compute kinematics
    print("[2/5] Computing kinematics (smoothing, differentiation, FFT)...")
    kinematics = compute_kinematics(
        results["angles"],
        results["metadata"]["fps"],
    )
    
    # Print summary
    summary = generate_summary(kinematics, species)
    print()
    print(summary)
    print()
    
    # Export CSV
    print("[3/5] Exporting to CSV...")
    os.makedirs("output/csv", exist_ok=True)
    csv_path = to_csv(
        kinematics, results,
        f"output/csv/{species.lower().replace(' ', '_')}_kinematics.csv",
        species=species,
    )
    
    # Generate plots
    print("[4/5] Generating visualization plots...")
    os.makedirs("output/plots", exist_ok=True)
    prefix = f"output/plots/{species.lower().replace(' ', '_')}"
    
    plot_trajectory(results["centroids"], results["wing_tips"],
                    save_path=f"{prefix}_trajectory.png", species=species)
    
    plot_angle_timeseries(kinematics["time"], kinematics["angles_raw"],
                         kinematics["angles_smoothed"],
                         save_path=f"{prefix}_angle.png", species=species)
    
    plot_velocity_acceleration(kinematics["time"], kinematics["angular_velocity"],
                               kinematics["angular_acceleration"],
                               save_path=f"{prefix}_velocity_accel.png", species=species)
    
    plot_frequency_spectrum(kinematics["fft_freqs"], kinematics["fft_magnitudes"],
                           kinematics["wingbeat_freq_fft"],
                           save_path=f"{prefix}_fft.png", species=species)
    
    plot_dashboard(kinematics, results,
                   save_path=f"{prefix}_dashboard.png", species=species)
    
    # Generate 3D model if requested
    if args.model:
        print("[5/5] Generating 3D wing model...")
        if species in SPECIES_PARAMS:
            generate_full_model(species, flap_angle_deg=0.0)
            generate_full_model(species, flap_angle_deg=30.0)
            generate_full_model(species, flap_angle_deg=-20.0)
            if args.sequence:
                generate_flap_sequence(species)
        else:
            print(f"  [SKIP] No morphological parameters for '{species}'. "
                  f"Available: {list(SPECIES_PARAMS.keys())}")
    else:
        print("[5/5] Skipping 3D model (use --model to enable)")
    
    print()
    print(f"{'═' * 60}")
    print(f"  Done! Results saved to output/")
    print(f"  CSV:   {csv_path}")
    print(f"  Plots: {prefix}_*.png")
    print(f"{'═' * 60}")


def run_video(args):
    """Run the full pipeline with a real video."""
    from src.tracker import run_tracking
    
    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        print(f"  Download from NJVID and place in data/raw/")
        print(f"  https://www.njvid.net/showcollection.php?pid=njcore:16509")
        sys.exit(1)
    
    species = args.species if args.species else "Unknown"
    
    print(f"{'═' * 60}")
    print(f"  denda-njvid-flight-tracker")
    print(f"  Video: {args.video}")
    print(f"  Species: {species}")
    print(f"{'═' * 60}")
    print()
    
    # Run tracking
    print("[1/5] Running interactive tracking...")
    print("  → Select ROI on first frame, then click on wing tip")
    results = run_tracking(
        args.video,
        species=species,
        threshold=args.threshold,
    )
    
    n_tracked = len(results["frames"])
    print(f"  Tracked {n_tracked} frames successfully")
    
    if n_tracked < 10:
        print("[ERROR] Too few frames tracked. Try adjusting threshold.")
        sys.exit(1)
    
    # Compute kinematics
    print("[2/5] Computing kinematics...")
    kinematics = compute_kinematics(results["angles"], results["metadata"]["fps"])
    
    summary = generate_summary(kinematics, species)
    print()
    print(summary)
    print()
    
    # Export CSV
    print("[3/5] Exporting to CSV...")
    os.makedirs("output/csv", exist_ok=True)
    csv_path = to_csv(
        kinematics, results,
        f"output/csv/{species.lower().replace(' ', '_')}_kinematics.csv",
        species=species,
    )
    
    # Generate plots
    print("[4/5] Generating plots...")
    os.makedirs("output/plots", exist_ok=True)
    prefix = f"output/plots/{species.lower().replace(' ', '_')}"
    
    plot_trajectory(results["centroids"], results["wing_tips"],
                    save_path=f"{prefix}_trajectory.png", species=species)
    plot_angle_timeseries(kinematics["time"], kinematics["angles_raw"],
                         kinematics["angles_smoothed"],
                         save_path=f"{prefix}_angle.png", species=species)
    plot_velocity_acceleration(kinematics["time"], kinematics["angular_velocity"],
                               kinematics["angular_acceleration"],
                               save_path=f"{prefix}_velocity_accel.png", species=species)
    plot_frequency_spectrum(kinematics["fft_freqs"], kinematics["fft_magnitudes"],
                           kinematics["wingbeat_freq_fft"],
                           save_path=f"{prefix}_fft.png", species=species)
    plot_dashboard(kinematics, results,
                   save_path=f"{prefix}_dashboard.png", species=species)
    
    # Validation
    if args.validate:
        print("[5/5] Running validation (annotate 20 frames)...")
        from src.validate import create_annotation_tool, validation_report
        os.makedirs("output/validation", exist_ok=True)
        val_results = create_annotation_tool(
            args.video, results["wing_tips"], results["frames"],
            n_frames=20,
            output_path="output/validation/validation_results.json",
        )
        report = validation_report(val_results)
        print()
        print(report)
        
        from src.visualization import plot_validation
        plot_validation(
            np.array(val_results["errors"]),
            val_results["mean_mae"],
            save_path="output/validation/validation_errors.png",
        )
    else:
        print("[5/5] Skipping validation (use --validate to enable)")
    
    # 3D model
    if args.model and species in SPECIES_PARAMS:
        print("[EXTRA] Generating 3D wing model...")
        generate_full_model(species)
    
    print()
    print(f"{'═' * 60}")
    print(f"  Done! Results saved to output/")
    print(f"{'═' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract wing kinematics from insect flight video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tracker.py --demo                            # Run with synthetic data
  python run_tracker.py --demo --model                    # Demo + 3D wing model
  python run_tracker.py --demo --model --sequence         # Demo + flap animation 
  python run_tracker.py --video data/raw/video.mp4        # Real video
  python run_tracker.py --video data/raw/video.mp4 --validate  # With validation
        """,
    )
    
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic demo data (no video required)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file")
    parser.add_argument("--species", type=str, default=None,
                        help="Species name")
    parser.add_argument("--threshold", type=int, default=60,
                        help="Binary threshold for centroid detection (default: 60)")
    parser.add_argument("--validate", action="store_true",
                        help="Run manual validation on 20 frames")
    parser.add_argument("--model", action="store_true",
                        help="Generate 3D parametric wing model")
    parser.add_argument("--sequence", action="store_true",
                        help="Generate full flap animation sequence (with --model)")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo(args)
    elif args.video:
        run_video(args)
    else:
        parser.print_help()
        print("\n[TIP] Try: python run_tracker.py --demo")


if __name__ == "__main__":
    main()
