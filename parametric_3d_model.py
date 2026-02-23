#!/usr/bin/env python3
"""
parametric_3d_model.py

Reads keypoint CSV from multipoint_tracker.py, extracts wing angles per
frame, and generates articulated 3D butterfly meshes (STL).

Usage:
    python3 parametric_3d_model.py output/multipoint/keypoints_all_frames.csv

    # With animation (every 5th frame):
    python3 parametric_3d_model.py output/multipoint/keypoints_all_frames.csv --animate --step 5

    # From synthetic data:
    python3 parametric_3d_model.py --synthetic --animate
"""

import argparse
import os
import sys
import numpy as np
import csv

try:
    from stl import mesh as stl_mesh
except ImportError:
    print("[ERROR] numpy-stl required: pip install numpy-stl")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')


# ──────────────────────────────────────────────────────────
#  Keypoint CSV reader
# ──────────────────────────────────────────────────────────

KP_NAMES = ['head', 'thorax', 'right_fw_tip', 'left_fw_tip',
            'left_hw_tip', 'right_hw_tip']


def load_keypoints_csv(path):
    meta = {}
    data_rows = []

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    key, val = line[1:].strip().split(':', 1)
                    meta[key.strip()] = val.strip()
                continue
            data_rows.append(line.strip())

    if not data_rows:
        raise ValueError(f"No data in {path}")

    reader = csv.reader(data_rows)
    header = next(reader)
    rows = list(reader)
    n = len(rows)

    frames = np.zeros(n, dtype=int)
    time = np.zeros(n)
    kps = {name: np.zeros((n, 2)) for name in KP_NAMES}
    confs = {name: np.zeros(n) for name in KP_NAMES}

    if 'head_x' in header:
        col_idx = {h: i for i, h in enumerate(header)}
        for i, row in enumerate(rows):
            if i >= n:
                break
            try:
                frames[i] = int(row[col_idx['frame']])
                time[i] = float(row[col_idx['time_s']])
            except (IndexError, ValueError):
                pass
        for name in KP_NAMES:
            xcol, ycol, ccol = f"{name}_x", f"{name}_y", f"{name}_conf"
            if xcol not in col_idx or ycol not in col_idx:
                continue
            for i, row in enumerate(rows):
                if i >= n:
                    break
                try:
                    xs, ys = row[col_idx[xcol]].strip(), row[col_idx[ycol]].strip()
                    kps[name][i, 0] = float(xs) if xs else np.nan
                    kps[name][i, 1] = float(ys) if ys else np.nan
                    confs[name][i] = float(row[col_idx[ccol]]) if ccol in col_idx and row[col_idx[ccol]].strip() else 0.5
                except (IndexError, ValueError):
                    kps[name][i, :] = np.nan
                    confs[name][i] = 0.0
        for name in KP_NAMES:
            bad = np.isnan(kps[name][:, 0]) | np.isnan(kps[name][:, 1])
            if np.any(bad) and name != 'thorax':
                tx, ty = kps['thorax'][:, 0], kps['thorax'][:, 1]
                kps[name][bad, 0] = tx[bad]
                kps[name][bad, 1] = ty[bad]
    else:
        col = 2
        for i, row in enumerate(rows):
            frames[i] = int(row[0])
            time[i] = float(row[1])
            c = col
            for name in KP_NAMES:
                kps[name][i, 0] = float(row[c])
                kps[name][i, 1] = float(row[c + 1])
                confs[name][i] = float(row[c + 2])
                c += 3

    print(f"  Loaded {n} frames from {path} (format: {'multipoint' if 'head_x' in header else 'legacy 6-KP'})")
    return frames, time, kps, confs, meta


def generate_synthetic_keypoints(n_frames=300, fps=30.0):
    """
    Generate synthetic 6-keypoint data for testing.
    Simulates a butterfly flapping at ~10 Hz with asymmetric wings.
    """
    print(f"  Generating synthetic keypoints: {n_frames} frames at {fps} fps")

    t = np.arange(n_frames) / fps
    freq = 10.0  # Hz

    # Body center at (320, 240), slowly moving
    body_x = 320 + 20 * np.sin(0.5 * t)
    body_y = 240 + 10 * np.cos(0.3 * t)

    # Wing flap angles (radians from horizontal)
    fw_amplitude = np.radians(55)
    hw_amplitude = np.radians(40)

    # Phase offset between fore and hind wings
    fw_phase = 2 * np.pi * freq * t
    hw_phase = fw_phase + np.radians(15)  # hindwing lags

    fw_angle = fw_amplitude * np.sin(fw_phase)
    hw_angle = hw_amplitude * np.sin(hw_phase)

    # Add slight asymmetry
    fw_angle_L = fw_angle * 1.05
    fw_angle_R = fw_angle * 0.95

    # Generate keypoint positions
    wing_len = 80  # wing span in pixels (approx)
    hw_len = 60

    kps = {name: np.zeros((n_frames, 2)) for name in KP_NAMES}
    confs = {name: np.ones(n_frames) * 0.9 for name in KP_NAMES}

    for i in range(n_frames):
        bx, by = body_x[i], body_y[i]

        kps['thorax'][i] = [bx, by]
        kps['head'][i] = [bx, by - 20]

        # Forewing tips
        kps['right_fw_tip'][i] = [
            bx + wing_len * np.cos(fw_angle_R[i]),
            by - wing_len * np.sin(fw_angle_R[i]) * 0.5
        ]
        kps['left_fw_tip'][i] = [
            bx - wing_len * np.cos(fw_angle_L[i]),
            by - wing_len * np.sin(fw_angle_L[i]) * 0.5
        ]

        # Hindwing tips
        kps['right_hw_tip'][i] = [
            bx + hw_len * np.cos(hw_angle[i]) * 0.8,
            by + hw_len * np.sin(hw_angle[i]) * 0.3 + 30
        ]
        kps['left_hw_tip'][i] = [
            bx - hw_len * np.cos(hw_angle[i]) * 0.8,
            by + hw_len * np.sin(hw_angle[i]) * 0.3 + 30
        ]

    frames = np.arange(n_frames)
    time = t
    meta = {'Species': 'Synthetic', 'FPS': str(fps)}

    return frames, time, kps, confs, meta


# ──────────────────────────────────────────────────────────
#  Compute wing angles from keypoints
# ──────────────────────────────────────────────────────────

def compute_wing_angles(kps):
    """
    Compute 4 wing angles from keypoints (relative to thorax).
    Returns dict with angle time-series for each wing.
    """
    tx, ty = kps['thorax'][:, 0], kps['thorax'][:, 1]
    n = len(tx)

    angles = {}
    for wing_name in ['right_fw_tip', 'left_fw_tip', 'left_hw_tip', 'right_hw_tip']:
        wx, wy = kps[wing_name][:, 0], kps[wing_name][:, 1]
        dx = wx - tx
        dy = wy - ty
        angles[wing_name] = np.arctan2(-dy, dx)  # -dy because image y is inverted

    # Forewing spread
    angles['fw_spread'] = angles['right_fw_tip'] - angles['left_fw_tip']
    angles['hw_spread'] = angles['right_hw_tip'] - angles['left_hw_tip']

    return angles


# ──────────────────────────────────────────────────────────
#  PCA compression of wing motion
# ──────────────────────────────────────────────────────────

def compress_motion_pca(angles, n_components=3):
    """
    Compress wing motion into principal components.
    This captures the dominant motion patterns (e.g., flap, twist, phase).
    """
    keys = ['right_fw_tip', 'left_fw_tip', 'left_hw_tip', 'right_hw_tip']
    X = np.column_stack([angles[k] for k in keys])
    mean = X.mean(axis=0)
    X_centered = X - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    n_components = min(n_components, len(S))
    explained = np.cumsum(S**2) / np.sum(S**2)

    print(f"  PCA compression ({n_components} components):")
    for i in range(n_components):
        print(f"    PC{i+1}: {explained[i]*100:.1f}% variance explained")

    weights = U[:, :n_components] * S[:n_components]
    components = Vt[:n_components]  # n_components × 4

    return {
        'mean': mean,
        'weights': weights,
        'components': components,
        'explained_variance': explained[:n_components],
        'keys': keys,
        'n_components': n_components,
    }


def reconstruct_from_pca(pca_model, t_normalized):
    """
    Reconstruct wing angles at any time from PCA model.
    t_normalized: array of time values in [0, 1] range.
    """
    N_orig = len(pca_model['weights'])
    # Interpolate weights
    t_orig = np.linspace(0, 1, N_orig)
    weights_interp = np.zeros((len(t_normalized), pca_model['n_components']))
    for i in range(pca_model['n_components']):
        weights_interp[:, i] = np.interp(t_normalized, t_orig, pca_model['weights'][:, i])

    # Reconstruct
    X_recon = weights_interp @ pca_model['components'] + pca_model['mean']

    angles = {}
    for i, key in enumerate(pca_model['keys']):
        angles[key] = X_recon[:, i]

    return angles


# ──────────────────────────────────────────────────────────
#  3D mesh generation
# ──────────────────────────────────────────────────────────

def create_wing_surface(span, chord, n_span=10, n_chord=6,
                        camber=0.08, taper=0.4, sweep_deg=15):
    """
    Generate a wing surface mesh as triangulated vertices.
    Returns (vertices, faces) for a single wing.
    """
    verts = []
    for i in range(n_span + 1):
        s = i / n_span  # 0 = root, 1 = tip
        local_chord = chord * (1 - s * (1 - taper))
        local_sweep = s * span * np.tan(np.radians(sweep_deg))

        for j in range(n_chord + 1):
            c = j / n_chord
            x = s * span
            y = c * local_chord + local_sweep
            # Camber
            z = camber * local_chord * np.sin(np.pi * c) * (1 - s * 0.5)
            verts.append([x, y, z])

    verts = np.array(verts, dtype=np.float64)

    # Triangulate
    faces = []
    for i in range(n_span):
        for j in range(n_chord):
            v0 = i * (n_chord + 1) + j
            v1 = v0 + 1
            v2 = v0 + (n_chord + 1)
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    # Add bottom surface (slight thickness)
    n_top = len(verts)
    thickness = 0.3
    bottom = verts.copy()
    bottom[:, 2] -= thickness
    verts = np.vstack([verts, bottom])

    bottom_faces = []
    for f in faces:
        bottom_faces.append([f[2] + n_top, f[1] + n_top, f[0] + n_top])
    faces.extend(bottom_faces)

    return verts, np.array(faces, dtype=int)


def create_body(length=15, radius=3, n_seg=12, n_circ=8):
    """Create a simplified cylindrical body mesh."""
    verts = []
    for i in range(n_seg + 1):
        s = i / n_seg
        y = -length / 2 + s * length
        r = radius * (1 - 0.5 * abs(s - 0.4))  # tapered
        for j in range(n_circ):
            theta = 2 * np.pi * j / n_circ
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            verts.append([x, y, z])

    verts = np.array(verts, dtype=np.float64)

    faces = []
    for i in range(n_seg):
        for j in range(n_circ):
            v0 = i * n_circ + j
            v1 = i * n_circ + (j + 1) % n_circ
            v2 = (i + 1) * n_circ + j
            v3 = (i + 1) * n_circ + (j + 1) % n_circ
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    return verts, np.array(faces, dtype=int)


def rotate_wing(verts, angle_rad, axis='y'):
    """Rotate wing vertices about an axis at the root (x=0)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotated = verts.copy()
    if axis == 'y':
        rotated[:, 0] = verts[:, 0] * c - verts[:, 2] * s
        rotated[:, 2] = verts[:, 0] * s + verts[:, 2] * c
    elif axis == 'x':
        rotated[:, 1] = verts[:, 1] * c - verts[:, 2] * s
        rotated[:, 2] = verts[:, 1] * s + verts[:, 2] * c
    return rotated


def mirror_wing(verts):
    """Mirror wing across x=0 plane."""
    mirrored = verts.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    return mirrored


def build_butterfly_stl(fw_angle_right, fw_angle_left,
                        hw_angle_right, hw_angle_left,
                        wing_span=40, wing_chord=25,
                        hw_span=30, hw_chord=20,
                        body_length=15):
    """
    Build a complete butterfly STL at a given wing pose.
    Wing angles in radians (0 = flat, positive = upward).
    """
    meshes = []

    # Body
    body_v, body_f = create_body(length=body_length, radius=2.5)
    body_mesh = stl_mesh.Mesh(np.zeros(len(body_f), dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(body_f):
        body_mesh.vectors[i] = body_v[f]
    meshes.append(body_mesh)

    # Right forewing
    rfw_v, rfw_f = create_wing_surface(wing_span, wing_chord,
                                        camber=0.06, taper=0.35, sweep_deg=20)
    rfw_v = rotate_wing(rfw_v, fw_angle_right, axis='y')
    rfw_m = stl_mesh.Mesh(np.zeros(len(rfw_f), dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(rfw_f):
        rfw_m.vectors[i] = rfw_v[f]
    meshes.append(rfw_m)

    # Left forewing (mirrored)
    lfw_v, lfw_f = create_wing_surface(wing_span, wing_chord,
                                        camber=0.06, taper=0.35, sweep_deg=20)
    lfw_v = mirror_wing(lfw_v)
    lfw_v = rotate_wing(lfw_v, -fw_angle_left, axis='y')
    lfw_m = stl_mesh.Mesh(np.zeros(len(lfw_f), dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(lfw_f):
        lfw_m.vectors[i] = lfw_v[f]
    meshes.append(lfw_m)

    # Right hindwing
    rhw_v, rhw_f = create_wing_surface(hw_span, hw_chord,
                                        camber=0.04, taper=0.5, sweep_deg=10)
    rhw_v[:, 1] += body_length * 0.2  # offset behind forewing
    rhw_v = rotate_wing(rhw_v, hw_angle_right, axis='y')
    rhw_m = stl_mesh.Mesh(np.zeros(len(rhw_f), dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(rhw_f):
        rhw_m.vectors[i] = rhw_v[f]
    meshes.append(rhw_m)

    # Left hindwing
    lhw_v, lhw_f = create_wing_surface(hw_span, hw_chord,
                                        camber=0.04, taper=0.5, sweep_deg=10)
    lhw_v[:, 1] += body_length * 0.2
    lhw_v = mirror_wing(lhw_v)
    lhw_v = rotate_wing(lhw_v, -hw_angle_left, axis='y')
    lhw_m = stl_mesh.Mesh(np.zeros(len(lhw_f), dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(lhw_f):
        lhw_m.vectors[i] = lhw_v[f]
    meshes.append(lhw_m)

    # Combine all meshes
    total_faces = sum(m.vectors.shape[0] for m in meshes)
    combined = stl_mesh.Mesh(np.zeros(total_faces, dtype=stl_mesh.Mesh.dtype))
    offset = 0
    for m in meshes:
        n = m.vectors.shape[0]
        combined.vectors[offset:offset + n] = m.vectors
        offset += n

    return combined


# ──────────────────────────────────────────────────────────
#  Animation generation
# ──────────────────────────────────────────────────────────

def generate_animation(angles, output_dir, step=5, total_frames=None):
    """
    Generate STL animation sequence from wing angles.
    One STL file per sampled frame.
    """
    os.makedirs(output_dir, exist_ok=True)

    keys = ['right_fw_tip', 'left_fw_tip', 'left_hw_tip', 'right_hw_tip']
    n = len(angles[keys[0]])

    if total_frames is None:
        total_frames = n

    indices = range(0, min(n, total_frames), step)
    count = 0

    for i in indices:
        # Normalize angles to wing flap range
        # The tracked angles are in image space, convert to 3D flap angle
        # Map angle ∈ [min, max] → flap ∈ [-45°, +45°]
        fw_r = _normalize_flap(angles['right_fw_tip'], i)
        fw_l = _normalize_flap(angles['left_fw_tip'], i)
        hw_r = _normalize_flap(angles['right_hw_tip'], i)
        hw_l = _normalize_flap(angles['left_hw_tip'], i)

        butterfly = build_butterfly_stl(fw_r, fw_l, hw_r, hw_l)
        path = os.path.join(output_dir, f"frame_{i:05d}.stl")
        butterfly.save(path)
        count += 1

    print(f"  [3D] Exported {count} animation frames to {output_dir}")
    return count


def _normalize_flap(angle_series, idx, max_flap=np.radians(45)):
    """Normalize an angle series value at index to a flap range."""
    a = angle_series[idx]
    a_min, a_max = np.min(angle_series), np.max(angle_series)
    if a_max - a_min < 1e-6:
        return 0.0
    normalized = (a - a_min) / (a_max - a_min)  # 0..1
    return max_flap * (2 * normalized - 1)  # -max_flap..+max_flap


# ──────────────────────────────────────────────────────────
#  Static pose generation
# ──────────────────────────────────────────────────────────

def generate_key_poses(angles, output_dir, n_poses=6):
    """Export STLs at key moments (e.g., max up, max down, neutral)."""
    os.makedirs(output_dir, exist_ok=True)

    keys = ['right_fw_tip', 'left_fw_tip', 'left_hw_tip', 'right_hw_tip']
    n = len(angles[keys[0]])

    # Find key moments
    fw_spread = angles.get('fw_spread',
                           angles['right_fw_tip'] - angles['left_fw_tip'])
    key_indices = []
    key_indices.append(('neutral_start', 0))
    key_indices.append(('max_spread', int(np.argmax(fw_spread))))
    key_indices.append(('min_spread', int(np.argmin(fw_spread))))
    key_indices.append(('mid_point', n // 2))
    key_indices.append(('three_quarter', 3 * n // 4))
    key_indices.append(('neutral_end', n - 1))

    for label, idx in key_indices[:n_poses]:
        fw_r = _normalize_flap(angles['right_fw_tip'], idx)
        fw_l = _normalize_flap(angles['left_fw_tip'], idx)
        hw_r = _normalize_flap(angles['right_hw_tip'], idx)
        hw_l = _normalize_flap(angles['left_hw_tip'], idx)

        butterfly = build_butterfly_stl(fw_r, fw_l, hw_r, hw_l)
        path = os.path.join(output_dir, f"{label}_{idx:05d}.stl")
        butterfly.save(path)

    print(f"  [3D] Exported {len(key_indices[:n_poses])} key poses to {output_dir}")


# ──────────────────────────────────────────────────────────
#  PCA visualization
# ──────────────────────────────────────────────────────────

def plot_pca_summary(pca_model, time, output_dir):
    """Plot PCA decomposition of wing motion."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Wing Motion PCA Decomposition', fontsize=14, fontweight='bold')

    # PC weights over time
    for i in range(min(3, pca_model['n_components'])):
        axes[0, 0].plot(time, pca_model['weights'][:, i],
                        label=f"PC{i+1} ({pca_model['explained_variance'][i]*100:.1f}%)",
                        linewidth=0.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Weight')
    axes[0, 0].set_title('Principal Component Weights')
    axes[0, 0].legend()

    # Explained variance
    axes[0, 1].bar(range(1, pca_model['n_components'] + 1),
                   pca_model['explained_variance'] * 100)
    axes[0, 1].set_xlabel('Component')
    axes[0, 1].set_ylabel('Variance (%)')
    axes[0, 1].set_title('Explained Variance')

    # Component loadings
    labels = ['R-FW', 'L-FW', 'L-HW', 'R-HW']
    x = np.arange(4)
    width = 0.25
    for i in range(min(3, pca_model['n_components'])):
        axes[1, 0].bar(x + i * width, pca_model['components'][i],
                        width, label=f'PC{i+1}')
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_title('Component Loadings')
    axes[1, 0].legend()

    # Reconstruction error
    keys = pca_model['keys']
    X_orig = np.column_stack([
        pca_model['weights'] @ pca_model['components'] + pca_model['mean']
    ])
    axes[1, 1].text(0.1, 0.5,
                     f"Motion captured in {pca_model['n_components']} components\n"
                     f"Total variance: {pca_model['explained_variance'][-1]*100:.1f}%\n\n"
                     f"This means the 4-wing motion can be\n"
                     f"described by {pca_model['n_components']} independent motions\n"
                     f"(e.g., flap, twist, phase lag)",
                     transform=axes[1, 1].transAxes, fontsize=11,
                     bbox=dict(facecolor='lightyellow', alpha=0.5))
    axes[1, 1].set_title('Summary')
    axes[1, 1].axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pca_decomposition.png'), dpi=150)
    print(f"  [PLOT] pca_decomposition.png")


# ──────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parametric 3D Butterfly Model from Keypoint Data")
    parser.add_argument("csv", nargs='?', default=None,
                        help="Path to keypoints CSV from multipoint_tracker.py")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate and use synthetic keypoint data")
    parser.add_argument("--animate", action="store_true",
                        help="Generate full animation sequence")
    parser.add_argument("--step", type=int, default=5,
                        help="Frame step for animation (default: 5)")
    parser.add_argument("--output-dir", default="output/3d_model",
                        help="Output directory")
    args = parser.parse_args()

    print("═" * 60)
    print("  Parametric 3D Butterfly Model")
    print("  Data-driven • PCA-compressed • Replicable")
    print("═" * 60)
    print()

    # ── Load data ──
    if args.csv is not None:
        print("[1/5] Loading keypoint data from CSV...")
        frames, time, kps, confs, meta = load_keypoints_csv(args.csv)
        species = meta.get('Species', 'Unknown')
    elif args.synthetic:
        print("[1/5] Generating synthetic keypoint data...")
        frames, time, kps, confs, meta = generate_synthetic_keypoints(
            n_frames=300, fps=30.0)
        species = "Synthetic"
    else:
        print("[ERROR] Provide a CSV path or use --synthetic")
        sys.exit(1)

    # ── Compute wing angles ──
    print("[2/5] Computing wing angles from keypoints...")
    angles = compute_wing_angles(kps)

    # ── PCA compression ──
    print("[3/5] Compressing motion via PCA...")
    pca_model = compress_motion_pca(angles, n_components=3)

    # Save PCA summary
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    plot_pca_summary(pca_model, time,
                     os.path.join(args.output_dir, "plots"))

    # ── Key poses ──
    print("[4/5] Generating key pose STLs...")
    generate_key_poses(angles,
                       os.path.join(args.output_dir, "poses"))

    # ── Animation ──
    if args.animate:
        print("[5/5] Generating animation sequence...")
        n_anim = generate_animation(
            angles,
            os.path.join(args.output_dir, "animation"),
            step=args.step)
    else:
        print("[5/5] Skipping animation (use --animate to enable)")
        n_anim = 0

    # ── Demo with synthetic data (prove it works with different input) ──
    if args.csv is not None:
        print("\n[BONUS] Generating from synthetic data (proving replicability)...")
        _, syn_time, syn_kps, _, _ = generate_synthetic_keypoints(
            n_frames=100, fps=30.0)
        syn_angles = compute_wing_angles(syn_kps)
        generate_key_poses(syn_angles,
                           os.path.join(args.output_dir, "synthetic_poses"))
        print("  → Synthetic poses exported (different motion from same model)")

    print()
    print("═" * 60)
    print("  ✓ 3D model generation complete!")
    print(f"  Species: {species}")
    print(f"  Key poses: {args.output_dir}/poses/")
    if n_anim > 0:
        print(f"  Animation: {args.output_dir}/animation/ ({n_anim} frames)")
    print(f"  PCA plot:  {args.output_dir}/plots/")
    print("═" * 60)


if __name__ == "__main__":
    main()
