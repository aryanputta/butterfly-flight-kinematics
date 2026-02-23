#!/usr/bin/env python3
"""
build_3d_model.py

Usage:
    python3 build_3d_model.py [--angle DEGREES] [--animation] [--from-csv CSV_PATH]
"""

import sys
import os
import argparse
import numpy as np

try:
    from stl import mesh as stl_mesh
except ImportError:
    print("Installing numpy-stl...")
    os.system(f"{sys.executable} -m pip install numpy-stl")
    from stl import mesh as stl_mesh


# ══════════════════════════════════════════════════════════
#  Morpho peleides morphological parameters (mm scale)
# ══════════════════════════════════════════════════════════

MORPHO_PARAMS = {
    # Body
    "head_radius": 2.5,          # mm
    "thorax_length": 6.0,        # mm
    "thorax_width": 4.0,         # mm
    "abdomen_length": 12.0,      # mm
    "abdomen_width_base": 3.5,   # mm
    "abdomen_width_tip": 1.0,    # mm
    
    # Wings (forewing)
    "fw_span": 55.0,             # mm (one wing)
    "fw_chord_root": 25.0,       # mm
    "fw_chord_tip": 8.0,         # mm
    "fw_sweep_angle": 15.0,      # degrees
    "fw_camber": 0.04,           # fraction of chord
    "fw_thickness": 0.5,         # mm
    
    # Wings (hindwing)
    "hw_span": 40.0,             # mm
    "hw_chord_root": 22.0,       # mm
    "hw_chord_tip": 12.0,        # mm
    "hw_sweep_angle": 25.0,      # degrees
    "hw_camber": 0.03,
    "hw_thickness": 0.5,
    
    # Antenna
    "antenna_length": 20.0,      # mm
    "antenna_thickness": 0.5,    # mm
    "antenna_club_radius": 1.0,  # mm (club-tipped antennae)
}


def create_sphere(center, radius, n_segments=16):
    """Create a sphere mesh (for head, joints)."""
    vertices = []
    faces = []
    
    for i in range(n_segments + 1):
        lat = np.pi * i / n_segments - np.pi / 2
        for j in range(n_segments):
            lon = 2 * np.pi * j / n_segments
            x = center[0] + radius * np.cos(lat) * np.cos(lon)
            y = center[1] + radius * np.cos(lat) * np.sin(lon)
            z = center[2] + radius * np.sin(lat)
            vertices.append([x, y, z])
    
    for i in range(n_segments):
        for j in range(n_segments):
            p1 = i * n_segments + j
            p2 = i * n_segments + (j + 1) % n_segments
            p3 = (i + 1) * n_segments + j
            p4 = (i + 1) * n_segments + (j + 1) % n_segments
            faces.append([p1, p2, p4])
            faces.append([p1, p4, p3])
    
    return np.array(vertices), np.array(faces)


def create_ellipsoid(center, radii, n_segments=16):
    """Create an ellipsoid mesh."""
    vertices = []
    faces = []
    rx, ry, rz = radii
    
    for i in range(n_segments + 1):
        lat = np.pi * i / n_segments - np.pi / 2
        for j in range(n_segments):
            lon = 2 * np.pi * j / n_segments
            x = center[0] + rx * np.cos(lat) * np.cos(lon)
            y = center[1] + ry * np.cos(lat) * np.sin(lon)
            z = center[2] + rz * np.sin(lat)
            vertices.append([x, y, z])
    
    for i in range(n_segments):
        for j in range(n_segments):
            p1 = i * n_segments + j
            p2 = i * n_segments + (j + 1) % n_segments
            p3 = (i + 1) * n_segments + j
            p4 = (i + 1) * n_segments + (j + 1) % n_segments
            faces.append([p1, p2, p4])
            faces.append([p1, p4, p3])
    
    return np.array(vertices), np.array(faces)


def create_tapered_cylinder(start, end, r_start, r_end, n_segments=12):
    """Create a tapered cylinder (for abdomen, antennae)."""
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    if length == 0:
        return np.array([[0, 0, 0]]), np.array([[0, 0, 0]])
    
    direction = direction / length
    
    # Find perpendicular vectors
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, [0, 0, 1])
    else:
        perp1 = np.cross(direction, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    
    n_rings = 10
    vertices = []
    faces = []
    
    for i in range(n_rings + 1):
        t = i / n_rings
        center = np.array(start) + t * length * direction
        radius = r_start + t * (r_end - r_start)
        
        for j in range(n_segments):
            angle = 2 * np.pi * j / n_segments
            pt = center + radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(pt)
    
    for i in range(n_rings):
        for j in range(n_segments):
            p1 = i * n_segments + j
            p2 = i * n_segments + (j + 1) % n_segments
            p3 = (i + 1) * n_segments + j
            p4 = (i + 1) * n_segments + (j + 1) % n_segments
            faces.append([p1, p2, p4])
            faces.append([p1, p4, p3])
    
    return np.array(vertices), np.array(faces)


def create_wing_surface(span, chord_root, chord_tip, sweep_deg, camber, thickness,
                        n_span=20, n_chord=12, side='right'):
    """
    Create a wing surface with camber, taper, and sweep.
    Returns vertices and faces.
    """
    sweep_rad = np.radians(sweep_deg)
    vertices_top = []
    vertices_bottom = []
    
    for i in range(n_span + 1):
        t = i / n_span  # 0 = root, 1 = tip
        
        # Chord at this spanwise position (linear taper)
        chord = chord_root + t * (chord_tip - chord_root)
        
        # Sweep offset
        x_sweep = t * span * np.tan(sweep_rad)
        
        # Spanwise position
        y_pos = t * span * (1 if side == 'right' else -1)
        
        for j in range(n_chord + 1):
            s = j / n_chord  # 0 = leading edge, 1 = trailing edge
            
            # Chordwise position
            x = x_sweep + s * chord
            
            # Camber: parabolic profile, max at 30% chord
            z_camber = 4 * camber * chord * s * (1 - s)
            
            # Add slight dihedral (wing curves up)
            z_dihedral = 0.02 * span * t * t
            
            # Scalloped trailing edge (Morpho-like)
            edge_scallop = 0
            if s > 0.85:
                edge_scallop = 0.5 * np.sin(6 * np.pi * t) * (s - 0.85) / 0.15
            
            # Top surface
            z_top = z_camber + z_dihedral + thickness / 2 + edge_scallop
            vertices_top.append([x, y_pos, z_top])
            
            # Bottom surface
            z_bot = z_camber + z_dihedral - thickness / 2 + edge_scallop
            vertices_bottom.append([x, y_pos, z_bot])
    
    # Build faces
    faces = []
    n_c = n_chord + 1
    
    # Top surface faces
    n_top = len(vertices_top)
    for i in range(n_span):
        for j in range(n_chord):
            p1 = i * n_c + j
            p2 = i * n_c + j + 1
            p3 = (i + 1) * n_c + j
            p4 = (i + 1) * n_c + j + 1
            faces.append([p1, p2, p4])
            faces.append([p1, p4, p3])
    
    # Bottom surface faces (offset by n_top)
    for i in range(n_span):
        for j in range(n_chord):
            p1 = n_top + i * n_c + j
            p2 = n_top + i * n_c + j + 1
            p3 = n_top + (i + 1) * n_c + j
            p4 = n_top + (i + 1) * n_c + j + 1
            faces.append([p1, p4, p2])  # reversed winding
            faces.append([p1, p3, p4])
    
    # Leading edge (connect top and bottom at j=0)
    for i in range(n_span):
        p_top1 = i * n_c
        p_top2 = (i + 1) * n_c
        p_bot1 = n_top + i * n_c
        p_bot2 = n_top + (i + 1) * n_c
        faces.append([p_top1, p_bot1, p_bot2])
        faces.append([p_top1, p_bot2, p_top2])
    
    # Trailing edge (connect top and bottom at j=n_chord)
    for i in range(n_span):
        p_top1 = i * n_c + n_chord
        p_top2 = (i + 1) * n_c + n_chord
        p_bot1 = n_top + i * n_c + n_chord
        p_bot2 = n_top + (i + 1) * n_c + n_chord
        faces.append([p_top1, p_bot2, p_bot1])
        faces.append([p_top1, p_top2, p_bot2])
    
    # Root cap (connect top and bottom at i=0)
    for j in range(n_chord):
        p_top1 = j
        p_top2 = j + 1
        p_bot1 = n_top + j
        p_bot2 = n_top + j + 1
        faces.append([p_top1, p_top2, p_bot2])
        faces.append([p_top1, p_bot2, p_bot1])
    
    # Tip cap
    for j in range(n_chord):
        p_top1 = n_span * n_c + j
        p_top2 = n_span * n_c + j + 1
        p_bot1 = n_top + n_span * n_c + j
        p_bot2 = n_top + n_span * n_c + j + 1
        faces.append([p_top1, p_bot2, p_top2])
        faces.append([p_top1, p_bot1, p_bot2])
    
    vertices = np.array(vertices_top + vertices_bottom)
    faces = np.array(faces)
    
    return vertices, faces


def add_venation(vertices, faces, span, n_veins=5, vein_height=0.3, side='right'):
    """
    Add vein-like ridges on top of the wing surface for realism.
    These appear as raised ridges on the upper surface.
    """
    vein_verts = []
    vein_faces = []
    offset = len(vertices)
    
    for v in range(n_veins):
        t_start = 0.05
        t_end = 0.7 + 0.3 * (v / n_veins)
        angle = (v / (n_veins - 1)) * 60 - 30  # spread from -30 to +30 degrees
        angle_rad = np.radians(angle)
        
        n_pts = 15
        for i in range(n_pts):
            t = t_start + (t_end - t_start) * (i / (n_pts - 1))
            y = t * span * (1 if side == 'right' else -1)
            x = t * span * np.tan(np.radians(15)) + t * 25 * 0.4  # follow chord
            x += t * 5 * np.sin(angle_rad)
            z = vein_height * (1 - t * 0.5)
            
            # Ridge cross-section
            for dx, dz in [(-0.2, 0), (0, z), (0.2, 0)]:
                vein_verts.append([x + dx, y, dz + 0.3])
        
        # Connect vein segments
        for i in range(n_pts - 1):
            base = offset + v * n_pts * 3 + i * 3
            for j in range(2):
                vein_faces.append([base + j, base + j + 1, base + j + 4])
                vein_faces.append([base + j, base + j + 4, base + j + 3])
    
    if vein_verts:
        all_verts = np.vstack([vertices, np.array(vein_verts)])
        all_faces = np.vstack([faces, np.array(vein_faces)])
        return all_verts, all_faces
    return vertices, faces


def rotate_wing(vertices, angle_deg, axis_origin, axis='y'):
    """Rotate wing vertices around an axis (for flap animation)."""
    angle_rad = np.radians(angle_deg)
    rotated = vertices.copy()
    
    # Translate to origin
    rotated -= axis_origin
    
    if axis == 'y':
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        x = rotated[:, 0] * cos_a - rotated[:, 2] * sin_a
        z = rotated[:, 0] * sin_a + rotated[:, 2] * cos_a
        rotated[:, 0] = x
        rotated[:, 2] = z
    elif axis == 'x':
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        y = rotated[:, 1] * cos_a - rotated[:, 2] * sin_a
        z = rotated[:, 1] * sin_a + rotated[:, 2] * cos_a
        rotated[:, 1] = y
        rotated[:, 2] = z
    
    # Translate back
    rotated += axis_origin
    return rotated


def combine_meshes(mesh_parts):
    """Combine multiple (vertices, faces) tuples into one STL mesh."""
    all_verts = []
    all_faces = []
    offset = 0
    
    for verts, faces in mesh_parts:
        all_verts.append(verts)
        all_faces.append(faces + offset)
        offset += len(verts)
    
    all_verts = np.vstack(all_verts)
    all_faces = np.vstack(all_faces)
    
    # Create STL mesh
    m = stl_mesh.Mesh(np.zeros(len(all_faces), dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(all_faces):
        for j in range(3):
            idx = int(f[j])
            if idx < len(all_verts):
                m.vectors[i][j] = all_verts[idx]
    
    return m


def build_butterfly(flap_angle_deg=0, params=None):
    """
    Build complete butterfly model at a given wing flap angle.
    
    flap_angle_deg: 0 = wings flat, positive = wings up
    
    Returns: STL mesh object
    """
    if params is None:
        params = MORPHO_PARAMS
    
    parts = []
    
    # === BODY ===
    
    # Head (sphere)
    head_center = [0, 0, 3]
    head_v, head_f = create_sphere(head_center, params["head_radius"], n_segments=12)
    parts.append((head_v, head_f))
    
    # Eyes (two small spheres)
    for side in [-1, 1]:
        eye_center = [0, side * 2.0, 3.5]
        eye_v, eye_f = create_sphere(eye_center, 1.0, n_segments=8)
        parts.append((eye_v, eye_f))
    
    # Thorax (ellipsoid)
    thorax_center = [params["thorax_length"] / 2, 0, 1]
    thorax_v, thorax_f = create_ellipsoid(
        thorax_center,
        [params["thorax_length"] / 2, params["thorax_width"] / 2, 2.5],
        n_segments=12
    )
    parts.append((thorax_v, thorax_f))
    
    # Abdomen (tapered cylinder)
    abd_start = [params["thorax_length"], 0, 0.5]
    abd_end = [params["thorax_length"] + params["abdomen_length"], 0, -1]
    abd_v, abd_f = create_tapered_cylinder(
        abd_start, abd_end,
        params["abdomen_width_base"] / 2, params["abdomen_width_tip"] / 2,
        n_segments=10
    )
    parts.append((abd_v, abd_f))
    
    # Antennae (two curved cylinders)
    for side in [-1, 1]:
        ant_start = [0, side * 1.5, 4]
        ant_mid = [-8, side * 6, 12]
        ant_end = [-12, side * 8, 15]
        
        # Two segments per antenna
        ant1_v, ant1_f = create_tapered_cylinder(
            ant_start, ant_mid,
            params["antenna_thickness"], params["antenna_thickness"] * 0.8,
            n_segments=6
        )
        parts.append((ant1_v, ant1_f))
        
        ant2_v, ant2_f = create_tapered_cylinder(
            ant_mid, ant_end,
            params["antenna_thickness"] * 0.8, params["antenna_thickness"] * 0.5,
            n_segments=6
        )
        parts.append((ant2_v, ant2_f))
        
        # Club tip
        club_v, club_f = create_sphere(ant_end, params["antenna_club_radius"], n_segments=6)
        parts.append((club_v, club_f))
    
    # Legs (6 legs, simplified)
    for i in range(3):
        x_pos = 1 + i * 2
        for side in [-1, 1]:
            leg_start = [x_pos, side * 2.0, 0]
            leg_end = [x_pos + 1, side * 6.0, -4]
            leg_v, leg_f = create_tapered_cylinder(
                leg_start, leg_end, 0.3, 0.15, n_segments=6
            )
            parts.append((leg_v, leg_f))
    
    # === WINGS ===
    wing_root = np.array([params["thorax_length"] / 2, 0, 2])  # top of thorax
    
    for side_name in ['right', 'left']:
        # Forewing
        fw_v, fw_f = create_wing_surface(
            params["fw_span"], params["fw_chord_root"], params["fw_chord_tip"],
            params["fw_sweep_angle"], params["fw_camber"], params["fw_thickness"],
            n_span=20, n_chord=10, side=side_name
        )
        
        # Offset to wing root
        fw_v[:, 0] += wing_root[0]
        fw_v[:, 2] += wing_root[2]
        
        # Add venation ridges
        fw_v, fw_f = add_venation(fw_v, fw_f, params["fw_span"], n_veins=4, side=side_name)
        
        # Apply flap angle
        flap = flap_angle_deg if side_name == 'right' else -flap_angle_deg
        fw_v = rotate_wing(fw_v, flap, wing_root, axis='x')
        
        parts.append((fw_v, fw_f))
        
        # Hindwing (offset slightly back and down)
        hw_v, hw_f = create_wing_surface(
            params["hw_span"], params["hw_chord_root"], params["hw_chord_tip"],
            params["hw_sweep_angle"], params["hw_camber"], params["hw_thickness"],
            n_span=16, n_chord=8, side=side_name
        )
        
        hw_root = wing_root + np.array([3, 0, -0.5])
        hw_v[:, 0] += hw_root[0]
        hw_v[:, 2] += hw_root[2]
        
        hw_v = rotate_wing(hw_v, flap * 0.8, hw_root, axis='x')
        
        parts.append((hw_v, hw_f))
    
    return combine_meshes(parts)


def generate_flap_sequence(angles_deg, output_dir):
    """
    Generate STL files for each frame in a wing flap sequence.
    Useful for animation or multi-material 3D printing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, angle in enumerate(angles_deg):
        m = build_butterfly(flap_angle_deg=angle)
        path = os.path.join(output_dir, f"morpho_flap_{i:03d}_{angle:.0f}deg.stl")
        m.save(path)
    
    print(f"  → Generated {len(angles_deg)} flap sequence STLs in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate 3D-printable Morpho peleides model")
    parser.add_argument("--angle", type=float, default=0,
                        help="Wing flap angle in degrees (0=flat, 30=up)")
    parser.add_argument("--animation", action="store_true",
                        help="Generate full flap animation sequence")
    parser.add_argument("--from-csv", type=str, default=None,
                        help="Use angles from CSV tracking data")
    parser.add_argument("--output-dir", type=str, default="output/3d_models")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("═" * 60)
    print("  3D Printable Butterfly Model Generator")
    print("  Species: Morpho peleides")
    print("═" * 60)
    print()
    
    # --- Static model at specified angle ---
    print(f"[1] Building model at {args.angle}° flap angle...")
    m = build_butterfly(flap_angle_deg=args.angle)
    flat_path = os.path.join(args.output_dir, f"morpho_peleides_{int(args.angle)}deg.stl")
    m.save(flat_path)
    print(f"  → Saved: {flat_path}")
    
    # --- Key poses for 3D printing ---
    print("\n[2] Generating key pose models...")
    poses = {
        "wings_flat": 0,
        "wings_slight_up": 15,
        "wings_30deg": 30,
        "wings_45deg": 45,
        "wings_60deg": 60,
        "wings_90deg": 90,
    }
    
    for name, angle in poses.items():
        m = build_butterfly(flap_angle_deg=angle)
        path = os.path.join(args.output_dir, f"morpho_{name}.stl")
        m.save(path)
        print(f"  → {name}: {path}")
    
    # --- Animation sequence ---
    if args.animation:
        print("\n[3] Generating flap animation sequence...")
        
        if args.from_csv and os.path.exists(args.from_csv):
            import pandas as pd
            print(f"  → Loading real tracking data from {args.from_csv}")
            df = pd.read_csv(args.from_csv, comment='#')
            if 'thorax_x' in df.columns and 'right_fw_tip_x' in df.columns:
                tx = df['thorax_x'].values.astype(float)
                ty = df['thorax_y'].values.astype(float)
                angles_rad = np.zeros(len(df))
                n = 0
                for xcol, ycol in [('right_fw_tip_x', 'right_fw_tip_y'), ('left_fw_tip_x', 'left_fw_tip_y'),
                                   ('right_hw_tip_x', 'right_hw_tip_y'), ('left_hw_tip_x', 'left_hw_tip_y')]:
                    if xcol in df.columns and ycol in df.columns:
                        dx = df[xcol].values.astype(float) - tx
                        dy = df[ycol].values.astype(float) - ty
                        angles_rad += np.arctan2(-dy, dx)
                        n += 1
                if n > 0:
                    angles_rad /= n
                angles_deg = np.degrees(angles_rad)
            elif 'angle_raw_rad' in df.columns:
                angles_deg = np.degrees(df['angle_raw_rad'].values)
            elif 'angle_smoothed_rad' in df.columns:
                angles_deg = np.degrees(df['angle_smoothed_rad'].values)
            else:
                angles_deg = np.degrees(df.iloc[:, 1].values.astype(float))
            angles_deg = angles_deg - np.mean(angles_deg)
            step = max(1, len(angles_deg) // 30)
            angles_subset = angles_deg[::step]
            generate_flap_sequence(angles_subset, os.path.join(args.output_dir, "animation"))
        else:
            # Synthetic flap sequence (sinusoidal)
            t = np.linspace(0, 2, 30)  # 2 wingbeat cycles
            angles_deg = 40 * np.sin(2 * np.pi * 1.5 * t)  # ~1.5 Hz
            generate_flap_sequence(angles_deg, os.path.join(args.output_dir, "animation"))
    
    # --- Print info ---
    print()
    print("═" * 60)
    print("  ✓ 3D models generated!")
    print(f"  Output directory: {args.output_dir}")
    print()
    print("  Printing tips:")
    print("  • Use 0.1mm layer height for wing detail")
    print("  • Wings are 0.5mm thick — print with supports")
    print("  • Scale up 2x if printing with FDM (resin recommended)")
    print("  • Total wingspan: ~110mm at 1:1 scale")
    print("═" * 60)


if __name__ == "__main__":
    main()
