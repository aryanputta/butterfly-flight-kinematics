"""
camera_calibration.py — Camera intrinsics and lens distortion correction.

Brown-Conrady model:
    x' = x(1 + k1*r^2 + k2*r^4 + k3*r^6)
    y' = y(1 + k1*r^2 + k2*r^4 + k3*r^6)
    x' += 2*p1*x*y + p2*(r^2 + 2*x^2)
    y' += p1*(r^2 + 2*y^2) + 2*p2*x*y

Correcting distortion removes systematic nonlinear error (1-5 px at edges),
which directly biases θ(t) via σ_θ ≈ σ_p / r.
"""

import cv2
import numpy as np
import json
import os
from typing import Optional, Tuple, Dict


def load_calibration(json_path: str) -> Dict:
    """Load camera calibration from JSON.

    Expected format:
        {"camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]],
         "dist_coeffs": [k1,k2,p1,p2,k3],
         "image_size": [w,h]}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64).reshape(1, -1)
    w, h = data.get("image_size", [640, 480])

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1.0)

    return {
        "camera_matrix": K,
        "dist_coeffs": dist,
        "image_size": (w, h),
        "optimal_matrix": new_K,
        "roi": roi,
    }


def save_calibration(json_path: str, camera_matrix: np.ndarray,
                     dist_coeffs: np.ndarray, image_size: Tuple[int, int]):
    """Save camera calibration to JSON."""
    data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "image_size": list(image_size),
    }
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def undistort_frame(frame: np.ndarray, calib: Dict) -> np.ndarray:
    """Apply lens distortion correction to a full frame."""
    return cv2.undistort(
        frame, calib["camera_matrix"], calib["dist_coeffs"],
        None, calib["optimal_matrix"],
    )


def undistort_points(points: np.ndarray, calib: Dict) -> np.ndarray:
    """Correct tracked point positions for lens distortion.

    More efficient than undistorting the entire frame when only
    correcting coordinates for a handful of tracked points.
    """
    if len(points) == 0:
        return points.copy()

    pts = points.reshape(-1, 1, 2).astype(np.float64)
    corrected = cv2.undistortPoints(
        pts, calib["camera_matrix"], calib["dist_coeffs"],
        P=calib["optimal_matrix"],
    )
    return corrected.reshape(-1, 2).astype(np.float32)


def identity_calibration(width: int = 640, height: int = 480) -> Dict:
    """No-op calibration (identity K, zero distortion). Use as fallback."""
    K = np.array([
        [width, 0, width / 2],
        [0, height, height / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist = np.zeros((1, 5), dtype=np.float64)
    return {
        "camera_matrix": K,
        "dist_coeffs": dist,
        "image_size": (width, height),
        "optimal_matrix": K.copy(),
        "roi": (0, 0, width, height),
    }


def calibrate_from_checkerboard(image_paths: list,
                                board_size: Tuple[int, int] = (9, 6),
                                square_size_mm: float = 25.0,
                                output_path: Optional[str] = None) -> Dict:
    """Run camera calibration from checkerboard images.

    Requires >= 3 images with visible checkerboards. Corners are refined
    to subpixel accuracy before solving for intrinsics.
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points = []
    img_points = []
    img_size = None

    subpix_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
    )

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       subpix_criteria)
            obj_points.append(objp)
            img_points.append(corners)

    if len(obj_points) < 3:
        return identity_calibration(
            img_size[0] if img_size else 640,
            img_size[1] if img_size else 480,
        )

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    print(f"  [CALIB] RMS reprojection error: {ret:.4f} px")

    if output_path:
        save_calibration(output_path, K, dist, img_size)

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, img_size, alpha=1.0)

    return {
        "camera_matrix": K,
        "dist_coeffs": dist,
        "image_size": img_size,
        "optimal_matrix": new_K,
        "roi": roi,
        "rms_error": ret,
    }
