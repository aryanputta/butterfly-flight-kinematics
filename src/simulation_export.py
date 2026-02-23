"""
simulation_export.py — Export fitted θ(t) for external solvers.

Supports:
  - MATLAB .mat files
  - Julia .jl self-contained functions
  - Generic CSV at configurable resolution
  - Parameter sensitivity Jacobian
  - Monte Carlo uncertainty propagation
"""

import numpy as np
import os
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────
#  θ(t) model evaluation
# ──────────────────────────────────────────────────────────

def theta_2harm(t: np.ndarray, A: float, f: float,
                h: float, phi: float, offset: float = 0.0) -> np.ndarray:
    """Two-harmonic wing stroke model.

    θ(t) = A·sin(2π·f·t) + h·sin(4π·f·t + φ) + offset
    """
    return (A * np.sin(2 * np.pi * f * t)
            + h * np.sin(4 * np.pi * f * t + phi)
            + offset)


def theta_3harm(t: np.ndarray, A: float, f: float,
                h: float, phi: float, offset: float,
                A3: float, phi3: float) -> np.ndarray:
    """Three-harmonic wing stroke model."""
    return (A * np.sin(2 * np.pi * f * t)
            + h * np.sin(4 * np.pi * f * t + phi)
            + A3 * np.sin(6 * np.pi * f * t + phi3)
            + offset)


# ──────────────────────────────────────────────────────────
#  Export formats
# ──────────────────────────────────────────────────────────

def export_csv(params: Dict, t_start: float, t_end: float,
               output_path: str, n_points: int = 10000,
               metadata: Optional[Dict] = None):
    """Export dense θ(t) time series as CSV.

    Includes header with metadata (species, wing length, assumptions).
    """
    t = np.linspace(t_start, t_end, n_points)
    theta = _eval_params(t, params)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("# Fitted wing stroke angle\n")
        f.write(f"# Model: {_model_string(params)}\n")
        if metadata:
            for k, v in metadata.items():
                f.write(f"# {k}: {v}\n")
        f.write("# Assumptions: 2D projection, rigid wing, symmetric stroke\n")
        f.write("#\n")
        f.write("time_s,theta_rad,theta_deg\n")
        for i in range(n_points):
            f.write(f"{t[i]:.8f},{theta[i]:.8f},{np.degrees(theta[i]):.4f}\n")

    print(f"  [EXPORT] {n_points} points → {output_path}")


def export_matlab(params: Dict, output_path: str,
                  metadata: Optional[Dict] = None):
    """Export as MATLAB .mat file + .m function.

    Creates both:
      - {name}.mat with parameter struct
      - {name}_func.m with callable theta(t) function
    """
    # .mat file
    try:
        from scipy.io import savemat
        mat_data = {
            "A": params["A"],
            "f": params["f"],
            "h": params["h"],
            "phi": params["phi"],
            "offset": params.get("offset", 0.0),
        }
        if "A3" in params:
            mat_data["A3"] = params["A3"]
            mat_data["phi3"] = params["phi3"]
        if metadata:
            mat_data["metadata"] = str(metadata)

        mat_path = output_path.replace(".m", ".mat")
        savemat(mat_path, mat_data)
        print(f"  [EXPORT] MATLAB .mat → {mat_path}")
    except ImportError:
        print("  [EXPORT] scipy.io not available, skipping .mat export")

    # .m function file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"% Fitted wing stroke angle model\n")
        f.write(f"% {_model_string(params)}\n")
        if metadata:
            for k, v in metadata.items():
                f.write(f"% {k}: {v}\n")
        f.write("% Assumptions: 2D projection, rigid wing\n\n")
        f.write("function theta = wing_theta(t)\n")
        f.write(f"    A      = {params['A']};\n")
        f.write(f"    f      = {params['f']};\n")
        f.write(f"    h      = {params['h']};\n")
        f.write(f"    phi    = {params['phi']};\n")
        f.write(f"    offset = {params.get('offset', 0.0)};\n")
        base_expr = "    theta = A.*sin(2*pi*f.*t) + h.*sin(4*pi*f.*t + phi)"
        if "A3" in params:
            f.write(f"    A3     = {params['A3']};\n")
            f.write(f"    phi3   = {params['phi3']};\n")
            base_expr += " + A3.*sin(6*pi*f.*t + phi3)"
        f.write(f"{base_expr} + offset;\n")
        f.write("end\n")

    print(f"  [EXPORT] MATLAB .m → {output_path}")


def export_julia(params: Dict, output_path: str):
    """Export as a self-contained Julia function file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"# Fitted wing stroke angle model\n")
        f.write(f"# {_model_string(params)}\n\n")
        f.write("function theta(t)\n")
        f.write(f"    A      = {params['A']}\n")
        f.write(f"    f      = {params['f']}\n")
        f.write(f"    h      = {params['h']}\n")
        f.write(f"    phi    = {params['phi']}\n")
        f.write(f"    offset = {params.get('offset', 0.0)}\n")
        base_expr = "    return A * sin(2π * f * t) + h * sin(4π * f * t + phi)"
        if "A3" in params:
            f.write(f"    A3     = {params['A3']}\n")
            f.write(f"    phi3   = {params['phi3']}\n")
            base_expr += " + A3 * sin(6π * f * t + phi3)"
        f.write(f"{base_expr} + offset\n")
        f.write("end\n")

    print(f"  [EXPORT] Julia .jl → {output_path}")


def export_cfd_header(params: Dict, output_path: str,
                      metadata: Optional[Dict] = None):
    """Export a CFD-compatible parameter header file.

    Generic key=value format readable by most CFD solvers.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("# Wing kinematics parameters for CFD\n")
        f.write("# Model: two-harmonic Fourier series\n")
        if metadata:
            for k, v in metadata.items():
                f.write(f"# {k} = {v}\n")
        f.write("#\n")
        f.write(f"amplitude_rad = {params['A']}\n")
        f.write(f"frequency_hz = {params['f']}\n")
        f.write(f"second_harmonic_amp_rad = {params['h']}\n")
        f.write(f"second_harmonic_phase_rad = {params['phi']}\n")
        f.write(f"offset_rad = {params.get('offset', 0.0)}\n")
        if "A3" in params:
            f.write(f"third_harmonic_amp_rad = {params['A3']}\n")
            f.write(f"third_harmonic_phase_rad = {params['phi3']}\n")
        f.write(f"frequency_rad_s = {params['f'] * 2 * np.pi}\n")
        f.write(f"period_s = {1.0 / params['f']}\n")

    print(f"  [EXPORT] CFD header → {output_path}")


# ──────────────────────────────────────────────────────────
#  Sensitivity analysis
# ──────────────────────────────────────────────────────────

def sensitivity_jacobian(t: np.ndarray, params: Dict) -> Dict[str, np.ndarray]:
    """Analytical Jacobian ∂θ/∂p for each parameter.

    Returns a dict mapping parameter name → sensitivity time series.
    These give the change in θ(t) per unit change in each parameter.
    """
    A = params["A"]
    f = params["f"]
    h = params["h"]
    phi = params["phi"]

    omega = 2 * np.pi * f

    jac = {
        "dtheta_dA": np.sin(omega * t),
        "dtheta_df": (A * 2 * np.pi * t * np.cos(omega * t)
                      + h * 4 * np.pi * t * np.cos(2 * omega * t + phi)),
        "dtheta_dh": np.sin(2 * omega * t + phi),
        "dtheta_dphi": h * np.cos(2 * omega * t + phi),
        "dtheta_doffset": np.ones_like(t),
    }

    if "A3" in params:
        A3 = params["A3"]
        phi3 = params["phi3"]
        jac["dtheta_dA3"] = np.sin(3 * omega * t + phi3)
        jac["dtheta_dphi3"] = A3 * np.cos(3 * omega * t + phi3)
        jac["dtheta_df"] += A3 * 6 * np.pi * t * np.cos(3 * omega * t + phi3)

    return jac


def uncertainty_propagation(t: np.ndarray, params: Dict,
                            param_uncertainties: Dict,
                            n_samples: int = 1000) -> Dict:
    """Monte Carlo uncertainty propagation.

    Samples parameters from their confidence intervals (assumed Gaussian)
    and evaluates θ(t) to produce a confidence band.

    Parameters
    ----------
    t : np.ndarray
        Evaluation times.
    params : dict
        Fitted parameters {A, f, h, phi, offset, ...}.
    param_uncertainties : dict
        Standard deviations for each parameter {A: σ_A, f: σ_f, ...}.
    n_samples : int
        Number of Monte Carlo samples.

    Returns
    -------
    result : dict
        theta_mean, theta_std, theta_ci_lower, theta_ci_upper (95%)
    """
    samples = np.zeros((n_samples, len(t)))

    for i in range(n_samples):
        p = {}
        for key in params:
            if key in param_uncertainties:
                p[key] = params[key] + np.random.randn() * param_uncertainties[key]
            else:
                p[key] = params[key]
        samples[i] = _eval_params(t, p)

    theta_mean = np.mean(samples, axis=0)
    theta_std = np.std(samples, axis=0)

    return {
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "theta_ci_lower": np.percentile(samples, 2.5, axis=0),
        "theta_ci_upper": np.percentile(samples, 97.5, axis=0),
    }


# ──────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────

def _eval_params(t: np.ndarray, params: Dict) -> np.ndarray:
    """Evaluate θ(t) from a params dict."""
    if "A3" in params:
        return theta_3harm(t, params["A"], params["f"], params["h"],
                           params["phi"], params.get("offset", 0.0),
                           params["A3"], params["phi3"])
    return theta_2harm(t, params["A"], params["f"], params["h"],
                       params["phi"], params.get("offset", 0.0))


def _model_string(params: Dict) -> str:
    """Readable model string."""
    s = (f"θ(t) = {params['A']:.4f}·sin(2π·{params['f']:.4f}·t) "
         f"+ {params['h']:.4f}·sin(4π·{params['f']:.4f}·t + {params['phi']:.4f})")
    if "A3" in params:
        s += f" + {params['A3']:.4f}·sin(6π·{params['f']:.4f}·t + {params['phi3']:.4f})"
    s += f" + {params.get('offset', 0.0):.4f}"
    return s


ASSUMPTIONS_DOC = """
Modeling Assumptions
--------------------
1. 2D Projection: Wing motion is extracted from a single monocular camera.
   Out-of-plane motion appears as foreshortening, which can bias stroke
   amplitude by up to ~15% depending on viewing angle.

2. Rigid Wing: The wing is modeled as a rigid plate rotating about the
   thorax. Real insect wings exhibit torsion and camber deformation,
   especially during stroke reversal.

3. Symmetric Stroke: The harmonic model assumes symmetric up/downstroke
   timing. Real wingbeats often have asymmetric half-strokes (faster
   downstroke). A third harmonic helps capture this.

4. Constant Frequency: The model assumes steady-state wingbeat frequency.
   Transient maneuvers (turning, acceleration) involve frequency modulation
   that this model does not capture.

5. No Wind/Body Motion: The stroke angle is measured relative to the thorax
   frame. Body pitch/roll oscillations are not separated from wing motion.
"""
