#!/usr/bin/env python3
"""
run_pipeline.py — Unified Butterfly Tracking & 3D Simulation

This script provides a single entry point to:
1. Run the Multi-Keypoint Tracker (with optional live display)
2. Process the resulting keypoints and generate a 3D simulation
3. Export articulated STL meshes and motion plots

Usage:
    python3 run_pipeline.py data/raw/morpho_peleides.mp4 --live
"""

import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Unified Butterfly Tracking & 3D Simulation")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--live", action="store_true", help="Show live tracking visualization")
    parser.add_argument("--limit-frames", type=int, default=0, help="Limit number of frames to process")
    parser.add_argument("--output-dir", default="output/combined", help="Root output directory")
    args = parser.parse_args()

    # Paths
    track_dir = os.path.join(args.output_dir, "tracking")
    model_dir = os.path.join(args.output_dir, "3d_simulation")
    csv_path = os.path.join(track_dir, "keypoints_all_frames.csv")
    
    os.makedirs(args.output_dir, exist_ok=True)

    print("═" * 60)
    print("  BUTTERFLY BIOMECHANICS PIPELINE")
    print("  Live Tracking → Dense Keypoints → 3D Simulation")
    print("═" * 60)
    print()

    # 1. Run Tracker
    print("[STEP 1/2] Running Dense Keypoint Tracker...")
    track_cmd = [
        sys.executable, "multipoint_tracker.py",
        args.video,
        "--output-dir", track_dir
    ]
    if args.live:
        track_cmd.append("--live")
    if args.limit_frames > 0:
        track_cmd.extend(["--limit-frames", str(args.limit_frames)])
    
    try:
        subprocess.run(track_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Tracker failed with exit code {e.returncode}")
        sys.exit(1)

    print("\n" + "─" * 40)
    
    # 2. Run Kinematics
    if os.path.exists(csv_path):
        print("[STEP 1b/2] Running Kinematic Analysis (Movement & Area)...")
        kin_cmd = [
            sys.executable, "extract_kinematics.py",
            csv_path,
            "30.0" # FPS
        ]
        try:
            subprocess.run(kin_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[WARNING] Kinematics failed with exit code {e.returncode}")
    else:
        print(f"\n[ERROR] Tracking CSV not found at {csv_path}")
        sys.exit(1)

    print("\n" + "─" * 40)

    # 3. Run 3D Model
    if os.path.exists(csv_path):
        print("[STEP 2/2] Running Parametric 3D Model Simulation...")
        model_cmd = [
            sys.executable, "parametric_3d_model.py",
            csv_path,
            "--output-dir", model_dir,
            "--animate"
        ]
        try:
            subprocess.run(model_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] 3D Model failed with exit code {e.returncode}")
            sys.exit(1)
    else:
        print(f"\n[ERROR] Tracking CSV not found at {csv_path}")
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  ✓ PIPELINE COMPLETE")
    print(f"  Tracking Data:  {track_dir}")
    print(f"  3D Simulation:  {model_dir}")
    print("═" * 60)

if __name__ == "__main__":
    main()
