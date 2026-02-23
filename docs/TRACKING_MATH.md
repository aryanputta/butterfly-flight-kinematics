# Tracking Pipeline: Algorithms & Parameters

Reference for the math and algorithms behind each stage of the tracker.

---

## Pipeline Stages

| Stage | What it does | Code |
|-------|-------------|------|
| Isolation | HSV color filter + connected components → butterfly mask | `multipoint_tracker.py` |
| Keypoints | 11 anatomical landmarks + up to 25 texture features | `multipoint_tracker.py` |
| Tracking | Lucas–Kanade optical flow with forward–backward check | `multipoint_tracker.py` |
| Kinematics | Displacement, velocity, wing angles from tracked points | `extract_kinematics.py`, `src/analysis.py` |
| Calibration | Lens distortion correction | `src/camera_calibration.py` |
| Harmonic fit | Two/three-harmonic model fitting | `notebooks/harmonic_fitting.jl` |
| Export | MATLAB, Julia, CSV, CFD | `src/simulation_export.py` |
| 3D Model | Wing angles drive parametric STL meshes; PCA compresses motion | `parametric_3d_model.py` |

---

## 1. Lucas–Kanade Optical Flow

Tracks a set of points from frame to frame by finding where each point moved.

For each point, look at a small window of pixels (25×25). Find the displacement (u, v) that best matches the pixel intensities from one frame to the next, using a least-squares fit over the window.

Multi-scale: uses an image pyramid (3 levels) to handle large motions.

**Parameters (`multipoint_tracker.py`):**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `winSize` | (25, 25) | Pixel window for the fit |
| `maxLevel` | 3 | Pyramid levels |
| `criteria` | 30 iters, eps 0.01 | Convergence criteria |

---

## 2. Forward–Backward Validation

Catches tracking errors by checking consistency:

1. Track point forward: frame t → frame t+1
2. Track result backward: frame t+1 → frame t
3. Compare: if the round-trip error > 2 pixels, reject the track

**Parameter:** `FB_ERROR_THRESHOLD = 2.0` pixels (adaptive scaling available)

---

## 3. Shi–Tomasi Corner Detector

Finds points worth tracking — pixels with strong gradient in two directions.

| Parameter | Value | Effect |
|-----------|-------|--------|
| `maxCorners` | 25 | Max feature points per frame |
| `qualityLevel` | 0.02 | Minimum corner quality (relative) |
| `minDistance` | 10 px | Spacing between points |
| `blockSize` | 7 | Gradient computation block |

Subpixel refinement via `cv2.cornerSubPix` improves accuracy from ~0.5 px to ~0.05 px.

---

## 4. Wing Stroke Angle

From the thorax position and each wing tip, the stroke angle is:

```
θ(t) = atan2(-(tip_y - thorax_y), tip_x - thorax_x)
```

The minus sign on y flips image coordinates (y-down) to standard math (y-up). The result is unwrapped to remove ±π discontinuities.

---

## 5. Signal Conditioning

### Savitzky–Golay Filter
Polynomial least-squares fit over a sliding window. Preserves signal shape (peak height, width, asymmetry) better than a moving average. Used for velocity/acceleration computation.

### Butterworth Low-Pass Filter
Zero-phase (`filtfilt`) with configurable cutoff. Provides a strict frequency-domain cutoff. Used before harmonic fitting to prevent noise aliasing into the fit.

### RANSAC Outlier Rejection
Fits a 2D affine transform between consecutive-frame point clouds. Points whose residual exceeds a threshold are flagged as outliers (tracking drift, occlusion).

---

## 6. Noise Propagation

Pixel tracking noise σ_p propagates through derivatives:

| Quantity | Formula | 120 fps, σ_p = 0.3 px, r = 100 px |
|----------|---------|------------------------------------|
| Angular error | σ_θ = σ_p√2 / r | 0.24° |
| Velocity error | σ_ω = σ_p · fps / r | 0.36 rad/s |
| Acceleration error | σ_α = σ_p · fps² · √3 / r | 74.8 rad/s² |

Each derivative amplifies noise. Subpixel refinement (10×) + SavGol filtering (3×) → ~30× combined reduction.

See `src/error_propagation.py` for the full analysis.

---

## 7. Camera Calibration

Brown-Conrady model corrects radial and tangential lens distortion. Typical improvement: 1–5 px at frame edges.

See `src/camera_calibration.py`. Supports loading/saving calibration JSON, per-frame or per-point correction, and checkerboard calibration capture.

---

## 8. Harmonic Fitting

Two-harmonic model:
```
θ(t) = A·sin(2πft) + h·sin(4πft + φ) + offset
```

Fitted via nonlinear least squares (LsqFit.jl) with time normalization. Produces confidence intervals, R², and Durbin–Watson autocorrelation test.

See `notebooks/harmonic_fitting.jl`.

---

## 9. PCA Motion Compression

Stack 4 wing angle time series into a matrix (N frames × 4 angles), center, run SVD. First 3 components capture 99%+ of variance.

---

## 10. Tracking → 3D Model

1. Keypoint CSV → thorax + 4 wing tip positions per frame
2. Wing angles computed as above
3. Parametric wing meshes rotate by the tracked angle
4. Export as STL per frame

---

## Parameter Summary

| Component | Key parameter | Default |
|-----------|--------------|---------| 
| LK optical flow | winSize, maxLevel | (25,25), 3 |
| FB validation | error threshold | 2.0 px |
| Shi–Tomasi | maxCorners, qualityLevel | 25, 0.02 |
| Butterworth LPF | cutoff, order | configurable, 4 |
| RANSAC | inlier threshold | 3.0 px |
| Wing angle | atan2 from thorax→tip | — |
| PCA | n_components | 3 |
