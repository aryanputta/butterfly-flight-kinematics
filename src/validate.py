"""
validate.py — Tracking validation module

Provides tools for manually annotating wing tip positions on random frames
and computing mean absolute pixel error against automated tracking.
"""

import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional


# Global state for annotation
_annotation_point = None


def _annotation_callback(event, x, y, flags, param):
    """Mouse callback for frame annotation."""
    global _annotation_point
    if event == cv2.EVENT_LBUTTONDOWN:
        _annotation_point = (x, y)


def create_annotation_tool(video_path: str, tracked_wing_tips: np.ndarray,
                           tracked_frames: np.ndarray,
                           n_frames: int = 20,
                           output_path: Optional[str] = None) -> Dict:
    """
    Interactive annotation tool: display N random frames and let the user
    click on the wing tip position. Compare against automated tracking.
    
    Parameters
    ----------
    video_path : str
        Path to original video.
    tracked_wing_tips : np.ndarray, shape (M, 2)
        Automated wing tip positions from tracker.
    tracked_frames : np.ndarray, shape (M,)
        Frame indices corresponding to tracked positions.
    n_frames : int
        Number of frames to annotate (default 20).
    output_path : str, optional
        Path to save annotation results as JSON.
    
    Returns
    -------
    results : dict
        Keys: manual_positions, tracked_positions, errors, frame_indices, mean_mae
    """
    global _annotation_point
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Select random frames from the tracked set
    available_indices = np.arange(len(tracked_frames))
    n_sample = min(n_frames, len(available_indices))
    sample_indices = np.sort(np.random.choice(available_indices, size=n_sample, replace=False))
    
    manual_positions = []
    tracked_positions = []
    frame_indices = []
    
    for idx in sample_indices:
        frame_num = tracked_frames[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Show automated tracking position
        auto_pos = tracked_wing_tips[idx]
        display = frame.copy()
        cv2.circle(display, (int(auto_pos[0]), int(auto_pos[1])), 5, (0, 0, 255), 2)
        cv2.putText(display, f"Frame {frame_num} — Click TRUE wing tip, then press any key",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, "Red circle = automated tracking", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        window_name = "Annotation"
        _annotation_point = None
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, _annotation_callback)
        cv2.imshow(window_name, display)
        
        while _annotation_point is None:
            key = cv2.waitKey(50) & 0xFF
            if key == 27:  # ESC to skip
                _annotation_point = tuple(auto_pos.astype(int))
                break
        
        manual_positions.append(_annotation_point)
        tracked_positions.append(tuple(auto_pos))
        frame_indices.append(int(frame_num))
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Compute errors
    manual_arr = np.array(manual_positions, dtype=float)
    tracked_arr = np.array(tracked_positions, dtype=float)
    errors = np.sqrt(np.sum((manual_arr - tracked_arr) ** 2, axis=1))
    
    results = {
        "manual_positions": manual_arr.tolist(),
        "tracked_positions": tracked_arr.tolist(),
        "errors": errors.tolist(),
        "frame_indices": frame_indices,
        "mean_mae": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "max_error": float(np.max(errors)),
        "n_frames_validated": len(errors),
    }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Validation results saved to {output_path}")
    
    return results


def compute_mae(predicted: np.ndarray, manual: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute mean absolute pixel error between predicted and manual positions.
    
    Parameters
    ----------
    predicted : np.ndarray, shape (N, 2)
        Predicted (automated) positions.
    manual : np.ndarray, shape (N, 2)
        Manually annotated positions.
    
    Returns
    -------
    mean_error : float
        Mean absolute pixel error (Euclidean distance).
    per_frame_errors : np.ndarray
        Per-frame Euclidean distance errors.
    """
    predicted = np.array(predicted, dtype=float)
    manual = np.array(manual, dtype=float)
    
    per_frame_errors = np.sqrt(np.sum((predicted - manual) ** 2, axis=1))
    mean_error = np.mean(per_frame_errors)
    
    return mean_error, per_frame_errors


def validation_report(results: Dict) -> str:
    """
    Generate a formatted validation report.
    
    Parameters
    ----------
    results : dict
        Output from create_annotation_tool() or similar.
    
    Returns
    -------
    report : str
        Formatted report string.
    """
    lines = [
        "═══ Tracking Validation Report ═══",
        f"  Frames validated:  {results['n_frames_validated']}",
        f"  Mean pixel error:  {results['mean_mae']:.2f} px",
        f"  Std pixel error:   {results['std_error']:.2f} px",
        f"  Max pixel error:   {results['max_error']:.2f} px",
        "",
    ]
    
    if results["mean_mae"] < 5.0:
        lines.append("  Assessment: EXCELLENT — Sub-5px tracking accuracy")
    elif results["mean_mae"] < 10.0:
        lines.append("  Assessment: GOOD — Acceptable tracking accuracy")
    elif results["mean_mae"] < 20.0:
        lines.append("  Assessment: FAIR — Consider parameter tuning")
    else:
        lines.append("  Assessment: POOR — Re-calibration recommended")
    
    lines.append("═" * 36)
    return "\n".join(lines)
