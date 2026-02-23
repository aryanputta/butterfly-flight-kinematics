# Tracking Pipeline: Algorithms & Parameters

Reference for the math and algorithms behind each stage of the tracker.

---

## Pipeline Stages

| Stage | What it does | Code |
|-------|-------------|------|
| Isolation | HSV color filter + connected components → butterfly mask | `multipoint_tracker.py` |
| Keypoints | 11 anatomical landmarks + up to 25 texture features | `multipoint_tracker.py` |
| Tracking | Lucas–Kanade optical flow with forward–backward check | `multipoint_tracker.py` |
| Kinematics | Displacement, velocity, wing angles from tracked points | `extract_kinematics.py` |
| 3D Model | Wing angles drive parametric STL meshes; PCA compresses motion | `parametric_3d_model.py` |

---

## 1. Lucas–Kanade Optical Flow

Tracks a set of points from frame to frame by finding where each point moved.

**How it works:** For each point, look at a small window of pixels (21×21). Find the displacement `(u, v)` that best matches the pixel intensities from one frame to the next, using a least-squares fit over the window.

**Multi-scale:** Uses an image pyramid (3 levels of downsampling) to handle large motions that would otherwise break the small-displacement assumption.

**Parameters (`multipoint_tracker.py`):**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `winSize` | (21, 21) | Pixel window for the fit. Larger = more stable. |
| `maxLevel` | 3 | Pyramid levels. More = handles bigger motion. |
| `criteria` | 30 iters, eps 0.01 | When to stop refining. |

---

## 2. Forward–Backward Validation

Catches tracking errors by checking consistency:

1. Track point forward: frame t → frame t+1
2. Track result backward: frame t+1 → frame t
3. Compare: if the round-trip error > 2 pixels, reject the track

**Parameter:** `FB_ERROR_THRESHOLD = 2.0` pixels

---

## 3. Shi–Tomasi Corner Detector

Finds points worth tracking — pixels with strong gradient in two directions (corners, texture spots).

**Parameters (`multipoint_tracker.py`):**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `maxCorners` | 25 | Max feature points per frame |
| `qualityLevel` | 0.01 | Minimum corner quality (relative) |
| `minDistance` | 8 px | Spacing between points |
| `blockSize` | 7 | Gradient computation block |

---

## 4. Wing Angles

From the thorax position and each wing tip, compute the angle:

```
angle = atan2(-(tip_y - thorax_y), tip_x - thorax_x)
```

The minus sign on `y` flips image coordinates (y-down) to standard math (y-up). Computed for all four wing tips per frame.

---

## 5. Kinematics

- **Displacement:** Euclidean distance between consecutive frame positions (pixels)
- **Velocity:** displacement × fps (pixels/second)
- **Wing area:** Convex hull of the 4 wing tip positions (proxy for spread)

---

## 6. PCA Motion Compression

Stack the 4 wing angle time series into a matrix (N frames × 4 angles), center it, run SVD. The first 3 components capture 99%+ of the motion variance, compressing the full sequence into a few parameters per frame.

---

## 7. Tracking → 3D Model

1. Keypoint CSV (from tracker) provides thorax + 4 wing tip positions per frame
2. Wing angles computed as above
3. Parametric wing meshes (tapered, cambered surfaces) rotate by the tracked angle
4. Export as STL per frame → animation sequence

`build_3d_model.py --from-csv` reads a keypoints CSV and generates the flap sequence directly from tracked data.

---

## Parameter Summary

| Component | Key parameter | Default |
|-----------|--------------|---------|
| LK optical flow | winSize, maxLevel | (21,21), 3 |
| FB validation | error threshold | 2.0 px |
| Shi–Tomasi | maxCorners, qualityLevel | 25, 0.01 |
| Wing angle | atan2 from thorax→tip | — |
| PCA | n_components | 3 |
