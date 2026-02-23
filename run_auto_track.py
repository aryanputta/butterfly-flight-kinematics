#!/usr/bin/env python3
"""
run_auto_track.py — Automated wing tracking for Morpho peleides NJVID video

Uses OpenCV computer vision to:
1. Detect butterfly wings via HSV color segmentation (iridescent blue)
2. Track wing extent (bounding area of blue pixels) frame-by-frame
3. Measure wing angular displacement from wing area morphology
4. Compute kinematics, export CSV, generate plots

Usage:
    python3 run_auto_track.py data/raw/morpho_peleides.mp4
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from src.analysis import compute_kinematics
from src.export import to_csv, generate_summary
from src.visualization import (
    plot_trajectory, plot_angle_timeseries,
    plot_velocity_acceleration, plot_frequency_spectrum,
    plot_dashboard
)


def detect_blue_wings(frame):
    """
    Detect Morpho peleides wings using HSV color segmentation.
    The iridescent blue wings have a very distinctive color signature
    in the hue channel (range ~90-130 in OpenCV HSV).
    
    Returns: (wing_mask, wing_area, centroid, bbox, wing_tips)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Blue/cyan range for Morpho iridescent wings
    # OpenCV H range: 0-179, so blue is roughly 90-130
    lower_blue = np.array([85, 30, 30])
    upper_blue = np.array([135, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Also include teal/dark blue regions
    lower_teal = np.array([75, 20, 20])
    upper_teal = np.array([85, 255, 255])
    mask_teal = cv2.inRange(hsv, lower_teal, upper_teal)
    mask = cv2.bitwise_or(mask, mask_teal)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    wing_area = cv2.countNonZero(mask)
    
    if wing_area < 100:
        return mask, 0, None, None, None
    
    # Find contours of wing region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, wing_area, None, None, None
    
    # Combine all wing contours
    all_pts = np.vstack(contours)
    
    # Centroid of blue region
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return mask, wing_area, None, None, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    bbox = cv2.boundingRect(all_pts)
    
    # Wing tips: leftmost and rightmost points of the blue region
    leftmost = tuple(all_pts[all_pts[:, :, 0].argmin()][0])
    rightmost = tuple(all_pts[all_pts[:, :, 0].argmax()][0])
    topmost = tuple(all_pts[all_pts[:, :, 1].argmin()][0])
    bottommost = tuple(all_pts[all_pts[:, :, 1].argmax()][0])
    
    return mask, wing_area, (cx, cy), bbox, {
        'left': leftmost, 'right': rightmost,
        'top': topmost, 'bottom': bottommost
    }


def detect_dark_body(frame, mask_blue):
    """
    Detect the dark butterfly body near the center of the blue region.
    The body is a dark elongated structure between the wings.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Dark regions only
    _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Dilate blue mask to get body-adjacent region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blue_dilated = cv2.dilate(mask_blue, kernel, iterations=3)
    
    # Body = dark AND near blue wings AND not blue itself
    body_mask = cv2.bitwise_and(dark_mask, blue_dilated)
    body_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(mask_blue))
    
    kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel_sm)
    
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Largest dark contour near wings = body
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def compute_wing_angle(wing_tips, centroid):
    """
    Compute wing opening angle from the wing extent.
    Uses the angle subtended from body centroid to left and right wing tips.
    """
    if wing_tips is None or centroid is None:
        return None
    
    cx, cy = centroid
    lx, ly = wing_tips['left']
    rx, ry = wing_tips['right']
    
    # Angle from centroid to left tip
    angle_left = np.arctan2(ly - cy, lx - cx)
    # Angle from centroid to right tip
    angle_right = np.arctan2(ry - cy, rx - cx)
    
    # Wing spread angle (total opening)
    spread = abs(angle_right - angle_left)
    if spread > np.pi:
        spread = 2 * np.pi - spread
    
    return spread


def track_video(video_path, species="Morpho peleides", max_frames=900, start_frame=30):
    """
    Track butterfly wings using color-based computer vision.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Video: {video_path}")
    print(f"  Resolution: {w}x{h}, FPS: {fps:.1f}, Total: {total} frames")
    print(f"  Processing frames {start_frame} to {start_frame + max_frames}")
    print()
    
    # Skip to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames_list = []
    centroids = []
    wing_tips_list = []
    angles = []
    areas = []
    
    detected = 0
    missed = 0
    
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx = start_frame + i
        
        # Detect blue wings
        mask, area, wing_centroid, bbox, tips = detect_blue_wings(frame)
        
        if area < 500 or wing_centroid is None or tips is None:
            missed += 1
            continue
        
        # Try to find body centroid
        body_centroid = detect_dark_body(frame, mask)
        if body_centroid is None:
            body_centroid = wing_centroid  # fallback
        
        # Compute wing angle
        angle = compute_wing_angle(tips, body_centroid)
        if angle is None:
            missed += 1
            continue
        
        frames_list.append(frame_idx)
        centroids.append(list(body_centroid))
        wing_tips_list.append([tips['right'][0], tips['right'][1]])
        angles.append(angle)
        areas.append(area)
        detected += 1
        
        if detected % 100 == 0:
            print(f"    Frame {frame_idx}: area={area}px², angle={np.degrees(angle):.1f}°")
    
    cap.release()
    
    print(f"\n  → Tracked {detected} frames, missed {missed}")
    print(f"  → Blue wing area range: {min(areas):.0f} - {max(areas):.0f} px²")
    print(f"  → Angle range: {np.degrees(min(angles)):.1f}° - {np.degrees(max(angles)):.1f}°")
    
    return {
        "frames": np.array(frames_list),
        "centroids": np.array(centroids),
        "wing_tips": np.array(wing_tips_list),
        "angles": np.array(angles),
        "areas": np.array(areas),
        "metadata": {
            "fps": fps,
            "frame_count": detected,
            "width": w,
            "height": h,
            "duration_s": detected / fps if fps > 0 else 0,
            "species": species,
        }
    }


def save_annotated_frames(video_path, results, output_dir, n_frames=8):
    """Save sample frames with tracking annotations drawn on them."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    indices = np.linspace(0, len(results['frames']) - 1, n_frames, dtype=int)
    
    for i, idx in enumerate(indices):
        frame_num = results['frames'][idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Re-detect blue wings for visualization
        mask, area, wc, bbox, tips = detect_blue_wings(frame)
        
        # Draw blue wing mask as overlay
        blue_overlay = frame.copy()
        blue_overlay[mask > 0] = [255, 100, 0]  # Cyan overlay on detected wings
        frame = cv2.addWeighted(frame, 0.6, blue_overlay, 0.4, 0)
        
        # Draw centroid
        cx, cy = int(results['centroids'][idx][0]), int(results['centroids'][idx][1])
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        cv2.putText(frame, "BODY", (cx + 12, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw wing tip
        wx, wy = int(results['wing_tips'][idx][0]), int(results['wing_tips'][idx][1])
        cv2.circle(frame, (wx, wy), 8, (0, 0, 255), -1)
        cv2.putText(frame, "WING TIP", (wx + 12, wy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw wing tips if available
        if tips:
            for label, pt in tips.items():
                cv2.circle(frame, pt, 5, (255, 255, 0), -1)
            # Draw lines from centroid to tips
            cv2.line(frame, (cx, cy), tips['left'], (255, 255, 0), 2)
            cv2.line(frame, (cx, cy), tips['right'], (255, 255, 0), 2)
        
        # Frame info
        angle_deg = np.degrees(results['angles'][idx])
        cv2.putText(frame, f"Frame {frame_num} | Wing angle: {angle_deg:.1f} deg | Area: {area}px",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        path = os.path.join(output_dir, f"tracked_{i:02d}_frame{frame_num}.png")
        cv2.imwrite(path, frame)
    
    cap.release()
    print(f"  [INFO] Saved {n_frames} annotated frames to {output_dir}")


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/morpho_peleides.mp4"
    species = sys.argv[2] if len(sys.argv) > 2 else "Morpho peleides"
    
    print("═" * 60)
    print("  Automated Wing Kinematics Tracker")
    print("  HSV Color-Based Computer Vision Pipeline")
    print("═" * 60)
    print()
    
    # --- Step 1: Track ---
    print("[1/5] Running CV tracking on real video...")
    results = track_video(video_path, species=species, max_frames=900, start_frame=30)
    
    if results is None or len(results['angles']) < 20:
        print("[ERROR] Insufficient tracking data.")
        sys.exit(1)
    
    # --- Step 2: Kinematics ---
    print("\n[2/5] Computing kinematics from real tracking data...")
    kinematics = compute_kinematics(
        results['angles'],
        results['metadata']['fps'],
        smooth_window=15,
        smooth_poly=3
    )
    
    summary = generate_summary(kinematics, species)
    print(summary)
    
    # --- Step 3: Export CSV ---
    print("[3/5] Exporting CSV...")
    os.makedirs("output/csv", exist_ok=True)
    csv_path = to_csv(kinematics, results,
                      "output/csv/morpho_peleides_real.csv",
                      species=species)
    
    # --- Step 4: Plots ---
    print("[4/5] Generating plots...")
    os.makedirs("output/plots", exist_ok=True)
    
    plot_trajectory(results['centroids'], results['wing_tips'],
                    species=f"{species} (NJVID real data)",
                    save_path="output/plots/real_trajectory.png")
    
    plot_angle_timeseries(kinematics['time'],
                          kinematics['angles_raw'],
                          kinematics['angles_smoothed'],
                          species=f"{species} (real)",
                          save_path="output/plots/real_angle.png")
    
    plot_velocity_acceleration(kinematics['time'],
                                kinematics['angular_velocity'],
                                kinematics['angular_acceleration'],
                                species=f"{species} (real)",
                                save_path="output/plots/real_velocity_accel.png")
    
    plot_frequency_spectrum(kinematics['fft_freqs'],
                            kinematics['fft_magnitudes'],
                            kinematics['wingbeat_freq_fft'],
                            species=f"{species} (real)",
                            save_path="output/plots/real_fft.png")
    
    plot_dashboard(kinematics, results, species=f"{species} (NJVID real data)",
                   save_path="output/plots/real_dashboard.png")
    
    # --- Step 5: Annotated frames ---
    print("[5/5] Saving annotated frames with tracking overlay...")
    save_annotated_frames(video_path, results, "output/tracked_frames/")
    
    print()
    print("═" * 60)
    print("  ✓ Real video tracking complete!")
    print(f"  CSV:    {csv_path}")
    print(f"  Plots:  output/plots/real_*.png")
    print(f"  Frames: output/tracked_frames/")
    print("═" * 60)


if __name__ == "__main__":
    main()
