"""
tracker.py — Core video tracking module

Handles video loading, region-of-interest selection, body centroid detection
via contour analysis, and wing tip tracking via Lucas-Kanade optical flow.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List


def load_video(path: str) -> Tuple[cv2.VideoCapture, Dict]:
    """
    Open a video file and extract metadata.
    
    Parameters
    ----------
    path : str
        Path to video file.
    
    Returns
    -------
    cap : cv2.VideoCapture
        Opened video capture object.
    metadata : dict
        Video metadata: fps, frame_count, width, height, duration_s.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    
    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    metadata["duration_s"] = metadata["frame_count"] / metadata["fps"] if metadata["fps"] > 0 else 0.0
    
    return cap, metadata


def select_roi(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Allow user to manually select a region of interest on the first frame.
    
    Parameters
    ----------
    frame : np.ndarray
        First frame of video (BGR).
    
    Returns
    -------
    roi : tuple of (x, y, w, h)
        Bounding box of selected region.
    """
    roi = cv2.selectROI("Select Region of Interest", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Region of Interest")
    return roi


def crop_to_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop a frame to the given ROI (x, y, w, h)."""
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]


def detect_centroid(frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None,
                    threshold_value: int = 60) -> Optional[Tuple[float, float]]:
    """
    Detect the insect body centroid using adaptive thresholding and contour analysis.
    
    The largest contour in the ROI is assumed to be the insect body.
    Centroid is computed via image moments.
    
    Parameters
    ----------
    frame : np.ndarray
        Video frame (BGR).
    roi : tuple, optional
        Region of interest (x, y, w, h). If None, uses full frame.
    threshold_value : int
        Binary threshold for separating insect from background.
    
    Returns
    -------
    centroid : tuple of (cx, cy) or None
        Centroid coordinates in the FULL frame coordinate system.
    """
    if roi is not None:
        x_off, y_off, w, h = roi
        cropped = frame[y_off:y_off+h, x_off:x_off+w]
    else:
        x_off, y_off = 0, 0
        cropped = frame

    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Adaptive threshold — insect is typically darker than background
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Use the largest contour (assumed to be the insect)
    largest = max(contours, key=cv2.contourArea)
    
    # Filter out noise — minimum area threshold
    if cv2.contourArea(largest) < 50:
        return None
    
    # Compute centroid via moments
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    
    cx = M["m10"] / M["m00"] + x_off
    cy = M["m01"] / M["m00"] + y_off
    
    return (cx, cy)


# Global state for mouse callback
_clicked_point = None

def _mouse_callback(event, x, y, flags, param):
    """Mouse callback for manual point selection."""
    global _clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicked_point = (x, y)


def init_wing_tip(frame: np.ndarray) -> np.ndarray:
    """
    Allow user to manually click on the initial wing tip position.
    
    Parameters
    ----------
    frame : np.ndarray
        First frame (BGR).
    
    Returns
    -------
    point : np.ndarray
        Wing tip position as shape (1, 1, 2) float32 array for optical flow.
    """
    global _clicked_point
    _clicked_point = None
    
    display = frame.copy()
    cv2.putText(display, "Click on wing tip, then press any key", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    window_name = "Select Wing Tip"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _mouse_callback)
    cv2.imshow(window_name, display)
    
    while _clicked_point is None:
        if cv2.waitKey(50) & 0xFF == 27:  # ESC to cancel
            break
    
    cv2.destroyWindow(window_name)
    
    if _clicked_point is None:
        raise ValueError("No wing tip selected")
    
    point = np.array([[list(_clicked_point)]], dtype=np.float32)
    return point


def track_wing_tip(prev_gray: np.ndarray, curr_gray: np.ndarray,
                   prev_points: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Track wing tip position using Lucas-Kanade optical flow.
    
    Parameters
    ----------
    prev_gray : np.ndarray
        Previous frame (grayscale).
    curr_gray : np.ndarray
        Current frame (grayscale).
    prev_points : np.ndarray
        Previous tracked points, shape (n, 1, 2).
    
    Returns
    -------
    new_points : np.ndarray
        Updated tracked points.
    success : bool
        Whether tracking was successful.
    """
    # Lucas-Kanade parameters
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    new_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None, **lk_params
    )
    
    if new_points is None or status is None:
        return prev_points, False
    
    # Check if tracking was successful (status == 1)
    success = bool(status.flatten()[0])
    
    return new_points, success


def compute_angle(centroid: Tuple[float, float],
                  wing_tip: Tuple[float, float]) -> float:
    """
    Compute angular displacement of wing tip relative to body centroid.
    
    Angle is measured from the horizontal axis (positive x) using atan2,
    giving a value in radians in the range [-π, π].
    
    Parameters
    ----------
    centroid : tuple of (cx, cy)
        Body centroid position.
    wing_tip : tuple of (wx, wy)
        Wing tip position.
    
    Returns
    -------
    angle : float
        Angular displacement in radians.
    """
    dx = wing_tip[0] - centroid[0]
    dy = wing_tip[1] - centroid[1]
    return np.arctan2(dy, dx)


def run_tracking(video_path: str, species: str = "Unknown",
                 use_gui: bool = True,
                 roi: Optional[Tuple[int, int, int, int]] = None,
                 initial_wing_tip: Optional[Tuple[int, int]] = None,
                 threshold: int = 60) -> Dict:
    """
    Run the full tracking pipeline on a video.
    
    Parameters
    ----------
    video_path : str
        Path to video file.
    species : str
        Species name for metadata.
    use_gui : bool
        If True, use interactive GUI for ROI and wing tip selection.
    roi : tuple, optional
        Pre-defined ROI (x, y, w, h). Used when use_gui=False.
    initial_wing_tip : tuple, optional
        Pre-defined initial wing tip (x, y). Used when use_gui=False.
    threshold : int
        Binary threshold for centroid detection.
    
    Returns
    -------
    results : dict
        Tracking results with keys: frames, centroids, wing_tips, angles, metadata.
    """
    cap, metadata = load_video(video_path)
    metadata["species"] = species
    
    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")
    
    # Select ROI
    if use_gui and roi is None:
        roi = select_roi(first_frame)
    
    # Initialize wing tip
    if use_gui and initial_wing_tip is None:
        wing_points = init_wing_tip(first_frame)
    else:
        if initial_wing_tip is None:
            raise ValueError("Must provide initial_wing_tip when use_gui=False")
        wing_points = np.array([[list(initial_wing_tip)]], dtype=np.float32)
    
    # Prepare tracking storage
    frame_indices = []
    centroids = []
    wing_tips = []
    angles = []
    
    # Convert first frame to grayscale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    # Process first frame
    centroid = detect_centroid(first_frame, roi, threshold)
    if centroid is not None:
        wt = tuple(wing_points[0, 0].astype(float))
        angle = compute_angle(centroid, wt)
        
        frame_indices.append(0)
        centroids.append(centroid)
        wing_tips.append(wt)
        angles.append(angle)
    
    # Process remaining frames
    frame_idx = 1
    lost_count = 0
    max_lost = 10  # Maximum consecutive lost frames before stopping
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Track wing tip
        new_points, success = track_wing_tip(prev_gray, curr_gray, wing_points)
        
        # Detect centroid
        centroid = detect_centroid(frame, roi, threshold)
        
        if success and centroid is not None:
            wing_points = new_points
            wt = tuple(wing_points[0, 0].astype(float))
            angle = compute_angle(centroid, wt)
            
            frame_indices.append(frame_idx)
            centroids.append(centroid)
            wing_tips.append(wt)
            angles.append(angle)
            
            lost_count = 0
        else:
            lost_count += 1
            if lost_count >= max_lost:
                print(f"[WARN] Lost tracking for {max_lost} consecutive frames at frame {frame_idx}. Stopping.")
                break
        
        prev_gray = curr_gray
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    results = {
        "frames": np.array(frame_indices),
        "centroids": np.array(centroids),
        "wing_tips": np.array(wing_tips),
        "angles": np.array(angles),
        "metadata": metadata,
    }
    
    return results
