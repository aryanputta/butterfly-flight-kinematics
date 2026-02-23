"""
wing_model.py — Parametric 3D wing geometry generator

Creates simplified parametric 3D models of insect wings based on morphological
parameters (span, chord, camber, taper ratio, hinge angle). Exports to STL
for visualization and potential simulation import.

This module uses numpy-stl for mesh generation, avoiding the heavy cadquery
dependency. It creates a biologically-informed wing surface using parametric
curves.

Usage:
    python -m src.wing_model --species "Morpho peleides" --output output/models/
"""

import numpy as np
from stl import mesh as stl_mesh
import os
import argparse
from typing import Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────────────
# Species-specific morphological parameters
# Values are approximate, based on published literature
# ──────────────────────────────────────────────────────────────────

SPECIES_PARAMS = {
    "Morpho peleides": {
        "wing_span_mm": 60.0,       # Half-span (one wing)
        "root_chord_mm": 35.0,      # Chord at wing root
        "tip_chord_mm": 10.0,       # Chord at wing tip
        "camber_fraction": 0.08,    # Max camber as fraction of chord
        "camber_position": 0.3,     # Chordwise position of max camber
        "sweep_angle_deg": 15.0,    # Leading edge sweep
        "thickness_mm": 0.5,        # Wing membrane thickness
        "hinge_x_mm": 5.0,          # Hinge offset from body center
        "description": "Large Neotropical butterfly with broad, iridescent blue wings",
    },
    "Parantica sita": {
        "wing_span_mm": 50.0,
        "root_chord_mm": 25.0,
        "tip_chord_mm": 8.0,
        "camber_fraction": 0.05,
        "camber_position": 0.25,
        "sweep_angle_deg": 10.0,
        "thickness_mm": 0.3,
        "hinge_x_mm": 4.0,
        "description": "Danaidae butterfly, migratory, elongated forewing",
    },
}


def generate_wing_profile(chord: float, camber_frac: float,
                          camber_pos: float, n_points: int = 50) -> np.ndarray:
    """
    Generate a 2D airfoil-like profile using a parabolic camber line.
    
    The wing cross-section is modeled as a thin cambered plate,
    appropriate for insect wings which are essentially flexible membranes.
    
    Parameters
    ----------
    chord : float
        Chord length.
    camber_frac : float
        Maximum camber as fraction of chord.
    camber_pos : float
        Chordwise position of maximum camber (0 to 1).
    n_points : int
        Number of points along the chord.
    
    Returns
    -------
    profile : np.ndarray, shape (n_points, 2)
        (x, z) coordinates of the camber line.
    """
    x = np.linspace(0, chord, n_points)
    max_camber = camber_frac * chord
    
    # Parabolic camber line (NACA-style)
    z = np.zeros_like(x)
    x_norm = x / chord
    
    # Before max camber position
    mask_front = x_norm <= camber_pos
    if camber_pos > 0:
        z[mask_front] = max_camber * (2 * camber_pos * x_norm[mask_front] - x_norm[mask_front] ** 2) / (camber_pos ** 2)
    
    # After max camber position
    mask_back = x_norm > camber_pos
    if camber_pos < 1:
        z[mask_back] = max_camber * ((1 - 2 * camber_pos) + 2 * camber_pos * x_norm[mask_back] - x_norm[mask_back] ** 2) / ((1 - camber_pos) ** 2)
    
    return np.column_stack([x, z])


def generate_wing_surface(params: Dict, n_span: int = 30,
                          n_chord: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 3D wing surface mesh.
    
    The wing is constructed by interpolating profiles along the span,
    with linear taper and sweep.
    
    Parameters
    ----------
    params : dict
        Species morphological parameters.
    n_span : int
        Number of sections along the span.
    n_chord : int
        Number of points along each chord profile.
    
    Returns
    -------
    X, Y, Z : np.ndarray
        3D surface coordinates, each shape (n_span, n_chord).
    """
    span = params["wing_span_mm"]
    root_chord = params["root_chord_mm"]
    tip_chord = params["tip_chord_mm"]
    camber_frac = params["camber_fraction"]
    camber_pos = params["camber_position"]
    sweep = np.radians(params["sweep_angle_deg"])
    hinge_x = params["hinge_x_mm"]
    
    X = np.zeros((n_span, n_chord))
    Y = np.zeros((n_span, n_chord))
    Z = np.zeros((n_span, n_chord))
    
    for i, eta in enumerate(np.linspace(0, 1, n_span)):
        # Spanwise position
        y_pos = eta * span
        
        # Linear taper
        chord = root_chord + eta * (tip_chord - root_chord)
        
        # Generate profile at this span station
        profile = generate_wing_profile(chord, camber_frac * (1 - 0.5 * eta),
                                        camber_pos, n_chord)
        
        # Apply sweep offset
        x_offset = hinge_x + eta * span * np.tan(sweep)
        
        X[i, :] = profile[:, 0] + x_offset
        Y[i, :] = y_pos
        Z[i, :] = profile[:, 1]
    
    return X, Y, Z


def surface_to_stl(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                    thickness: float = 0.5) -> stl_mesh.Mesh:
    """
    Convert a parametric surface to an STL mesh with finite thickness.
    
    Creates both upper and lower surfaces offset by thickness,
    and stitches edges to form a watertight solid.
    
    Parameters
    ----------
    X, Y, Z : np.ndarray
        Surface coordinates, shape (n_span, n_chord).
    thickness : float
        Wing membrane thickness in mm.
    
    Returns
    -------
    wing_mesh : stl.mesh.Mesh
        STL mesh object.
    """
    n_span, n_chord = X.shape
    
    # Create upper and lower surfaces
    half_t = thickness / 2.0
    
    # Collect all triangles
    triangles = []
    
    # Upper surface
    for i in range(n_span - 1):
        for j in range(n_chord - 1):
            # Triangle 1
            v0 = [X[i, j], Y[i, j], Z[i, j] + half_t]
            v1 = [X[i+1, j], Y[i+1, j], Z[i+1, j] + half_t]
            v2 = [X[i, j+1], Y[i, j+1], Z[i, j+1] + half_t]
            triangles.append([v0, v1, v2])
            
            # Triangle 2
            v3 = [X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1] + half_t]
            triangles.append([v1, v3, v2])
    
    # Lower surface (reversed normals)
    for i in range(n_span - 1):
        for j in range(n_chord - 1):
            v0 = [X[i, j], Y[i, j], Z[i, j] - half_t]
            v1 = [X[i+1, j], Y[i+1, j], Z[i+1, j] - half_t]
            v2 = [X[i, j+1], Y[i, j+1], Z[i, j+1] - half_t]
            triangles.append([v0, v2, v1])  # Reversed winding
            
            v3 = [X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1] - half_t]
            triangles.append([v1, v2, v3])  # Reversed winding
    
    # Leading edge (j=0)
    for i in range(n_span - 1):
        v0_top = [X[i, 0], Y[i, 0], Z[i, 0] + half_t]
        v1_top = [X[i+1, 0], Y[i+1, 0], Z[i+1, 0] + half_t]
        v0_bot = [X[i, 0], Y[i, 0], Z[i, 0] - half_t]
        v1_bot = [X[i+1, 0], Y[i+1, 0], Z[i+1, 0] - half_t]
        triangles.append([v0_top, v0_bot, v1_top])
        triangles.append([v1_top, v0_bot, v1_bot])
    
    # Trailing edge (j=n_chord-1)
    j = n_chord - 1
    for i in range(n_span - 1):
        v0_top = [X[i, j], Y[i, j], Z[i, j] + half_t]
        v1_top = [X[i+1, j], Y[i+1, j], Z[i+1, j] + half_t]
        v0_bot = [X[i, j], Y[i, j], Z[i, j] - half_t]
        v1_bot = [X[i+1, j], Y[i+1, j], Z[i+1, j] - half_t]
        triangles.append([v0_top, v1_top, v0_bot])
        triangles.append([v1_top, v1_bot, v0_bot])
    
    # Root edge (i=0)
    for j_idx in range(n_chord - 1):
        v0_top = [X[0, j_idx], Y[0, j_idx], Z[0, j_idx] + half_t]
        v1_top = [X[0, j_idx+1], Y[0, j_idx+1], Z[0, j_idx+1] + half_t]
        v0_bot = [X[0, j_idx], Y[0, j_idx], Z[0, j_idx] - half_t]
        v1_bot = [X[0, j_idx+1], Y[0, j_idx+1], Z[0, j_idx+1] - half_t]
        triangles.append([v0_top, v1_top, v0_bot])
        triangles.append([v1_top, v1_bot, v0_bot])
    
    # Tip edge (i=n_span-1)
    i = n_span - 1
    for j_idx in range(n_chord - 1):
        v0_top = [X[i, j_idx], Y[i, j_idx], Z[i, j_idx] + half_t]
        v1_top = [X[i, j_idx+1], Y[i, j_idx+1], Z[i, j_idx+1] + half_t]
        v0_bot = [X[i, j_idx], Y[i, j_idx], Z[i, j_idx] - half_t]
        v1_bot = [X[i, j_idx+1], Y[i, j_idx+1], Z[i, j_idx+1] - half_t]
        triangles.append([v0_top, v0_bot, v1_top])
        triangles.append([v1_top, v0_bot, v1_bot])
    
    # Build mesh
    triangles = np.array(triangles)
    wing = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    
    for i, tri in enumerate(triangles):
        wing.vectors[i] = tri
    
    return wing


def generate_wing_pair(params: Dict, flap_angle_deg: float = 0.0,
                       n_span: int = 30, n_chord: int = 20) -> stl_mesh.Mesh:
    """
    Generate a mirrored wing pair (left + right) at a given flap angle.
    
    Parameters
    ----------
    params : dict
        Species morphological parameters.
    flap_angle_deg : float
        Wing flap angle in degrees (0 = horizontal, positive = up).
    n_span : int
        Spanwise resolution.
    n_chord : int
        Chordwise resolution.
    
    Returns
    -------
    combined_mesh : stl.mesh.Mesh
        Combined left + right wing mesh.
    """
    X, Y, Z = generate_wing_surface(params, n_span, n_chord)
    
    # Apply flap rotation around the root (x-axis rotation at y=0)
    flap_rad = np.radians(flap_angle_deg)
    Y_rot = Y * np.cos(flap_rad) - Z * np.sin(flap_rad)
    Z_rot = Y * np.sin(flap_rad) + Z * np.cos(flap_rad)
    
    # Right wing mesh
    right_wing = surface_to_stl(X, Y_rot, Z_rot, params["thickness_mm"])
    
    # Left wing = mirror about Y=0 plane
    left_wing = surface_to_stl(X, -Y_rot, Z_rot, params["thickness_mm"])
    
    # Combine meshes
    combined = stl_mesh.Mesh(np.concatenate([right_wing.data, left_wing.data]))
    
    return combined


def generate_body(length_mm: float = 30.0, radius_mm: float = 5.0,
                  n_segments: int = 20, n_radial: int = 12) -> stl_mesh.Mesh:
    """
    Generate a simplified insect body (elongated ellipsoid).
    
    Parameters
    ----------
    length_mm : float
        Body length.
    radius_mm : float
        Body radius at widest point.
    n_segments : int
        Axial segments.
    n_radial : int
        Radial segments.
    
    Returns
    -------
    body_mesh : stl.mesh.Mesh
        Body mesh.
    """
    triangles = []
    
    for i in range(n_segments):
        t0 = i / n_segments
        t1 = (i + 1) / n_segments
        
        # Ellipsoidal radius profile
        x0 = (t0 - 0.5) * length_mm
        x1 = (t1 - 0.5) * length_mm
        r0 = radius_mm * np.sqrt(1 - (2 * t0 - 1) ** 2)
        r1 = radius_mm * np.sqrt(1 - (2 * t1 - 1) ** 2)
        
        for j in range(n_radial):
            theta0 = 2 * np.pi * j / n_radial
            theta1 = 2 * np.pi * (j + 1) / n_radial
            
            v00 = [x0, r0 * np.cos(theta0), r0 * np.sin(theta0)]
            v01 = [x0, r0 * np.cos(theta1), r0 * np.sin(theta1)]
            v10 = [x1, r1 * np.cos(theta0), r1 * np.sin(theta0)]
            v11 = [x1, r1 * np.cos(theta1), r1 * np.sin(theta1)]
            
            triangles.append([v00, v10, v01])
            triangles.append([v10, v11, v01])
    
    triangles = np.array(triangles)
    body = stl_mesh.Mesh(np.zeros(len(triangles), dtype=stl_mesh.Mesh.dtype))
    for i, tri in enumerate(triangles):
        body.vectors[i] = tri
    
    return body


def generate_full_model(species: str, flap_angle_deg: float = 0.0,
                        output_dir: str = "output/models/") -> str:
    """
    Generate a complete insect model (body + wing pair) and save to STL.
    
    Parameters
    ----------
    species : str
        Species name (must be in SPECIES_PARAMS).
    flap_angle_deg : float
        Wing flap angle.
    output_dir : str
        Directory to save STL files.
    
    Returns
    -------
    output_path : str
        Path to saved STL file.
    """
    if species not in SPECIES_PARAMS:
        available = ", ".join(SPECIES_PARAMS.keys())
        raise ValueError(f"Unknown species: {species}. Available: {available}")
    
    params = SPECIES_PARAMS[species]
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate components
    wings = generate_wing_pair(params, flap_angle_deg)
    body = generate_body(
        length_mm=params["wing_span_mm"] * 0.5,
        radius_mm=params["root_chord_mm"] * 0.15,
    )
    
    # Combine all meshes
    combined = stl_mesh.Mesh(np.concatenate([wings.data, body.data]))
    
    # Save
    safe_name = species.lower().replace(" ", "_")
    angle_str = f"_flap{int(flap_angle_deg)}deg" if flap_angle_deg != 0 else ""
    filename = f"{safe_name}{angle_str}.stl"
    output_path = os.path.join(output_dir, filename)
    
    combined.save(output_path)
    print(f"[INFO] Saved 3D model: {output_path}")
    print(f"       Species: {species}")
    print(f"       {params['description']}")
    print(f"       Wing span: {params['wing_span_mm']} mm (half)")
    print(f"       Flap angle: {flap_angle_deg}°")
    
    return output_path


def generate_flap_sequence(species: str, output_dir: str = "output/models/",
                           n_frames: int = 12) -> list:
    """
    Generate a sequence of STL files showing wing flapping motion.
    
    Creates frames from max-up to max-down position for animation.
    
    Parameters
    ----------
    species : str
        Species name.
    output_dir : str
        Output directory.
    n_frames : int
        Number of frames in the flap cycle.
    
    Returns
    -------
    paths : list of str
        Paths to generated STL files.
    """
    # Full flap cycle: up → down → up
    angles = 45.0 * np.sin(np.linspace(0, 2 * np.pi, n_frames, endpoint=False))
    
    seq_dir = os.path.join(output_dir, "flap_sequence")
    os.makedirs(seq_dir, exist_ok=True)
    
    paths = []
    for i, angle in enumerate(angles):
        params = SPECIES_PARAMS[species]
        safe_name = species.lower().replace(" ", "_")
        
        wings = generate_wing_pair(params, angle)
        body = generate_body(
            length_mm=params["wing_span_mm"] * 0.5,
            radius_mm=params["root_chord_mm"] * 0.15,
        )
        combined = stl_mesh.Mesh(np.concatenate([wings.data, body.data]))
        
        path = os.path.join(seq_dir, f"{safe_name}_frame_{i:03d}.stl")
        combined.save(path)
        paths.append(path)
    
    print(f"[INFO] Generated {n_frames} flap sequence frames in {seq_dir}/")
    return paths


# ──────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate parametric 3D insect wing model")
    parser.add_argument("--species", type=str, default="Morpho peleides",
                        choices=list(SPECIES_PARAMS.keys()),
                        help="Species to model")
    parser.add_argument("--output", type=str, default="output/models/",
                        help="Output directory")
    parser.add_argument("--flap-angle", type=float, default=0.0,
                        help="Wing flap angle in degrees")
    parser.add_argument("--sequence", action="store_true",
                        help="Generate full flap sequence")
    parser.add_argument("--n-frames", type=int, default=12,
                        help="Number of frames in flap sequence")
    
    args = parser.parse_args()
    
    if args.sequence:
        generate_flap_sequence(args.species, args.output, args.n_frames)
    else:
        generate_full_model(args.species, args.flap_angle, args.output)


if __name__ == "__main__":
    main()
