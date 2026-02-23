#!/usr/bin/env python3
"""
extract_kinematics.py — Kinematic Data Extraction for Butterfly Flight

Computes:
1. Displacement (dx, dy) and total distance per frame for all anatomical landmarks.
2. Instantaneous Velocity (px/s) using video FPS.
3. Wing Area Dynamics: Total blue area and change in area per frame.

Usage:
    python3 extract_kinematics.py output/combined/tracking/keypoints_all_frames.csv 30.0
"""

import sys
import pandas as pd
import numpy as np
import os

def extract_kinematics(csv_path, fps=30.0, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    # Load tracking data
    # Skip metadata lines starting with #
    df = pd.read_csv(csv_path, comment='#')
    
    # Identify anatomical columns
    # Format: head_x, head_y, head_conf, ...
    anat_names = [col.replace('_x', '') for col in df.columns if col.endswith('_x') and not col.startswith('feat')]
    
    kinematics = df[['frame', 'time_s']].copy()
    
    # 1. Displacement & Velocity
    for name in anat_names:
        x_col, y_col = f"{name}_x", f"{name}_y"
        
        # Calculate deltas
        dx = df[x_col].diff().fillna(0)
        dy = df[y_col].diff().fillna(0)
        dist = np.sqrt(dx**2 + dy**2)
        
        # Velocity (pixels per second)
        velocity = dist * fps
        
        kinematics[f"{name}_dx"] = dx
        kinematics[f"{name}_dy"] = dy
        kinematics[f"{name}_dist"] = dist
        kinematics[f"{name}_velocity"] = velocity

    # Approximate wing area via convex hull of 4 wing tips
    wings = ['left_fw_tip', 'right_fw_tip', 'left_hw_tip', 'right_hw_tip']
    valid_wings = [w for w in wings if w in anat_names]
    
    if len(valid_wings) == 4:
        def get_poly_area(row):
            pts = []
            for w in valid_wings:
                if row[f"{w}_x"] > 0:
                    pts.append([row[f"{w}_x"], row[f"{w}_y"]])
            if len(pts) < 3: return 0
            # Shoelace formula for area
            pts = np.array(pts)
            x = pts[:, 0]
            y = pts[:, 1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        kinematics['wing_proxy_area'] = df.apply(get_poly_area, axis=1)
        kinematics['area_change'] = kinematics['wing_proxy_area'].diff().fillna(0)

    # Save to CSV
    out_path = os.path.join(output_dir, "kinematics_per_frame.csv")
    kinematics.to_csv(out_path, index=False)
    print(f"  [KINEMATICS] Saved {len(kinematics)} frames of movement data to {out_path}")
    return out_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_kinematics.py <csv_path> [fps]")
        sys.exit(1)
    
    csv_p = sys.argv[1]
    fps_val = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    extract_kinematics(csv_p, fps_val)
