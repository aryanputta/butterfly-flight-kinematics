#!/usr/bin/env python3
# Dense keypoint butterfly tracker. Usage: multipoint_tracker.py <video> [--output-dir output/multipoint]

import sys
import os
import argparse
import numpy as np
import cv2
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')


# ──────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────

ANAT_KP_NAMES = [
    'head', 'thorax', 'abdomen',
    'right_fw_tip', 'left_fw_tip',
    'right_hw_tip', 'left_hw_tip',
    'right_fw_mid', 'left_fw_mid',
    'right_hw_mid', 'left_hw_mid',
]
N_ANAT = len(ANAT_KP_NAMES)

MAX_FEATURE_PTS = 25
REDETECT_INTERVAL = 15
MIN_FEATURES_BEFORE_REDETECT = 8
FB_ERROR_THRESHOLD = 2.0
FB_ERROR_ADAPTIVE = True
SMOOTH_WINDOW = 5

LK_PARAMS = dict(
    winSize=(25, 25),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

FEATURE_PARAMS = dict(
    maxCorners=MAX_FEATURE_PTS,
    qualityLevel=0.02,
    minDistance=10,
    blockSize=7,
)
SUBPIX_WIN = (5, 5)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)


# ──────────────────────────────────────────────────────────
#  Stage 1: Butterfly isolation (rejects hand/wristband)
# ──────────────────────────────────────────────────────────

def isolate_butterfly(frame, prev_center=None, prev_area=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_img, w_img = frame.shape[:2]

    mask = cv2.inRange(hsv, np.array([88, 100, 50]), np.array([130, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n_labels <= 1:
        return None, None, 0

    skin_mask_img = create_skin_mask(frame)
    kernel_skin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    skin_dilated = cv2.dilate(skin_mask_img, kernel_skin)

    best_idx, best_score = -1, -1
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / (min(bw, bh) + 1)

        if area < 400:
            continue
        if aspect > 7.0:
            continue
        obj_mask = (labels == i).astype(np.uint8) * 255
        overlap_area = cv2.countNonZero(cv2.bitwise_and(obj_mask, skin_dilated))
        
        if area < 5000:
            if overlap_area > (area * 0.15):
                continue
        else:
            if overlap_area > (area * 0.5):
                continue

        score = float(area)
        if cx < 30 or cx > w_img - 30 or cy < 30 or cy > h_img - 30:
            score *= 0.3

        if prev_center is not None:
            dist = np.hypot(cx - prev_center[0], cy - prev_center[1])
            score *= (1.0 + 2.0 * max(0, 1 - dist / 200))

        if prev_area and prev_area > 0:
            ratio = area / prev_area
            if ratio < 0.1 or ratio > 10:
                score *= 0.2

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        return None, None, 0

    bfly_mask = (labels == best_idx).astype(np.uint8) * 255
    bcx, bcy = centroids[best_idx]
    merge_r = max(stats[best_idx, cv2.CC_STAT_WIDTH],
                  stats[best_idx, cv2.CC_STAT_HEIGHT])

    for i in range(1, n_labels):
        if i == best_idx:
            continue
        ci = centroids[i]
        ai = stats[i, cv2.CC_STAT_AREA]
        if np.hypot(ci[0] - bcx, ci[1] - bcy) < merge_r * 1.5 and ai > 200:
            ar = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) / \
                 (min(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) + 1)
            if ar < 5.0:
                bfly_mask = cv2.bitwise_or(
                    bfly_mask, (labels == i).astype(np.uint8) * 255)

    wing_area = cv2.countNonZero(bfly_mask)
    M = cv2.moments(bfly_mask)
    if M["m00"] == 0:
        return None, None, 0
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    return bfly_mask, center, wing_area


# ──────────────────────────────────────────────────────────
#  Stage 2: Anatomical landmarks + dense feature detection
# ──────────────────────────────────────────────────────────

def create_skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    skin1 = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([20, 180, 255]))
    skin2 = cv2.inRange(hsv, np.array([160, 30, 60]), np.array([180, 180, 255]))

    skin3 = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    skin_mask = cv2.bitwise_or(skin1, skin2)
    skin_mask = cv2.bitwise_or(skin_mask, skin3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    return skin_mask


def detect_body_axis(frame, bfly_mask):
    h_img, w_img = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    skin_mask = create_skin_mask(frame)
    M_b = cv2.moments(bfly_mask)
    if M_b["m00"] == 0:
        return None, None, None
    bcx = int(M_b["m10"] / M_b["m00"])
    bcy = int(M_b["m01"] / M_b["m00"])

    wing_contours, _ = cv2.findContours(bfly_mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    if not wing_contours:
        return None, None, None
    all_wing_pts = np.vstack(wing_contours)
    bx, by, bw, bh = cv2.boundingRect(all_wing_pts)

    central_margin = int(bw * 0.25)
    search_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    x_left = max(0, bx + central_margin)
    x_right = min(w_img, bx + bw - central_margin)
    search_mask[:, x_left:x_right] = 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    near_wings = cv2.dilate(bfly_mask, k, iterations=2)

    body_mask = cv2.bitwise_and(dark, near_wings)
    body_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(bfly_mask))
    body_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(skin_mask))
    body_mask = cv2.bitwise_and(body_mask, search_mask)

    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                  iterations=2)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    best = None
    best_score = -1
    for c in contours:
        area = cv2.contourArea(c)
        if area < 30:
            continue
        mc = cv2.moments(c)
        if mc["m00"] == 0:
            continue
        cx = int(mc["m10"] / mc["m00"])
        cy = int(mc["m01"] / mc["m00"])
        dist = np.hypot(cx - bcx, cy - bcy)

        if dist > max(bw, bh) * 0.5:
            continue
        x_center_dist = abs(cx - bcx)
        score = area / (1 + dist) / (1 + x_center_dist)

        if score > best_score:
            best_score = score
            best = c

    if best is None:
        return None, None, None

    bm = cv2.moments(best)
    thorax = (int(bm["m10"] / bm["m00"]), int(bm["m01"] / bm["m00"]))

    head = tuple(best[best[:, :, 1].argmin()][0])
    abdomen = tuple(best[best[:, :, 1].argmax()][0])
    for pt_name, pt in [('head', head), ('thorax', thorax)]:
        px, py = int(pt[0]), int(pt[1])
        if 0 <= px < w_img and 0 <= py < h_img:
            if skin_mask[py, px] > 0:
                if pt_name == 'head':
                    head = None
                else:
                    thorax = None

    return head, thorax, abdomen


def detect_anatomical_keypoints(frame, bfly_mask, center):
    kps = {n: None for n in ANAT_KP_NAMES}
    cfs = {n: 0.0 for n in ANAT_KP_NAMES}

    head, thorax, abdomen = detect_body_axis(frame, bfly_mask)

    if thorax is None:
        thorax = center
        cfs['thorax'] = 0.3
    else:
        cfs['thorax'] = 0.9
    kps['thorax'] = thorax

    if head is None:
        head = (thorax[0], max(0, thorax[1] - 18))
        cfs['head'] = 0.2
    else:
        cfs['head'] = 0.85
    kps['head'] = head

    if abdomen is None:
        abdomen = (thorax[0], min(frame.shape[0] - 1, thorax[1] + 18))
        cfs['abdomen'] = 0.2
    else:
        cfs['abdomen'] = 0.8
    kps['abdomen'] = abdomen

    contours, _ = cv2.findContours(bfly_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return kps, cfs

    all_pts = np.vstack(contours).reshape(-1, 2)
    tx, ty = thorax

    for quad, tip_name, mid_name in [
        ('UL', 'left_fw_tip',  'left_fw_mid'),
        ('UR', 'right_fw_tip', 'right_fw_mid'),
        ('LL', 'left_hw_tip',  'left_hw_mid'),
        ('LR', 'right_hw_tip', 'right_hw_mid'),
    ]:
        y_mask = (all_pts[:, 1] <= ty) if quad[0] == 'U' else (all_pts[:, 1] > ty)
        x_mask = (all_pts[:, 0] < tx) if quad[1] == 'L' else (all_pts[:, 0] >= tx)
        qpts = all_pts[y_mask & x_mask]

        if len(qpts) < 3:
            x_mask2 = (all_pts[:, 0] < tx) if quad[1] == 'L' else (all_pts[:, 0] >= tx)
            qpts = all_pts[x_mask2]

        if len(qpts) < 1:
            continue

        dists = np.sqrt((qpts[:, 0] - tx)**2 + (qpts[:, 1] - ty)**2)
        tip_idx = dists.argmax()
        kps[tip_name] = tuple(qpts[tip_idx])
        cfs[tip_name] = min(1.0, dists[tip_idx] / 50.0)

        mid_d = dists[tip_idx] / 2
        mid_candidates = np.abs(dists - mid_d)
        mid_idx = mid_candidates.argmin()
        kps[mid_name] = tuple(qpts[mid_idx])
        cfs[mid_name] = cfs[tip_name] * 0.8

    return kps, cfs


def detect_dense_features(gray, bfly_mask, anat_pts):
    features = cv2.goodFeaturesToTrack(gray, mask=bfly_mask, **FEATURE_PARAMS)

    if features is None:
        return np.empty((0, 2), dtype=np.float32)

    pts = features.reshape(-1, 2).astype(np.float32)

    if len(pts) > 0:
        pts = cv2.cornerSubPix(
            gray, pts, SUBPIX_WIN, (-1, -1), SUBPIX_CRITERIA
        ).reshape(-1, 2)

    if anat_pts is not None and len(anat_pts) > 0:
        filtered = []
        for p in pts:
            dists = np.sqrt(np.sum((anat_pts - p)**2, axis=1))
            if np.min(dists) > 5:
                filtered.append(p)
        pts = np.array(filtered, dtype=np.float32) if filtered else np.empty((0, 2), dtype=np.float32)

    return pts


# ──────────────────────────────────────────────────────────
#  Stage 3: Lucas-Kanade Optical Flow tracking
# ──────────────────────────────────────────────────────────

def track_with_optical_flow(prev_gray, curr_gray, prev_pts, fb_scale=1.0):
    if prev_pts is None or len(prev_pts) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=bool)

    pts = prev_pts.reshape(-1, 1, 2).astype(np.float32)
    thr = FB_ERROR_THRESHOLD * max(1.0, fb_scale) if FB_ERROR_ADAPTIVE else FB_ERROR_THRESHOLD

    next_pts, status_f, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None, **LK_PARAMS)

    if next_pts is None:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=bool)

    back_pts, status_b, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, next_pts, None, **LK_PARAMS)

    fb_error = np.sqrt(np.sum((pts - back_pts)**2, axis=2)).flatten()
    valid = (status_f.flatten() == 1) & \
            (status_b.flatten() == 1) & \
            (fb_error < thr)

    tracked = next_pts.reshape(-1, 2)
    return tracked, valid


def _median1d(arr, w):
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        segment = arr[lo:hi]
        valid = segment[~np.isnan(segment)]
        if len(valid) > 0:
            out[i] = np.median(valid)
    return out


def smooth_trajectories(results, window=None):
    w = (window if window is not None else SMOOTH_WINDOW)
    if w <= 0:
        return
    half = w // 2
    all_kps = results['anat_kps']
    N = len(all_kps)
    for name in ANAT_KP_NAMES:
        xs = np.array([all_kps[i].get(name, (np.nan, np.nan))[0] if all_kps[i].get(name) else np.nan for i in range(N)], dtype=np.float64)
        ys = np.array([all_kps[i].get(name, (np.nan, np.nan))[1] if all_kps[i].get(name) else np.nan for i in range(N)], dtype=np.float64)
        x_smooth = _median1d(xs, half)
        y_smooth = _median1d(ys, half)
        for i in range(N):
            if not np.isnan(x_smooth[i]) and not np.isnan(y_smooth[i]):
                all_kps[i][name] = (float(x_smooth[i]), float(y_smooth[i]))


# ──────────────────────────────────────────────────────────
#  Main tracking pipeline
# ──────────────────────────────────────────────────────────

def track_all_frames(video_path, output_dir="output/multipoint", options=None):
    if options is None: options = {}
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video: {video_path}")
    print(f"  Resolution: {w}×{h}, FPS: {fps:.1f}, Frames: {total}")
    print()

    prev_gray = None
    prev_center = None
    prev_area = None

    all_anat_kps = []
    all_anat_confs = []
    all_feature_pts = []
    all_feature_ids = []
    all_n_total = []

    next_feature_id = 0
    active_feature_pts = np.empty((0, 2), dtype=np.float32)
    active_feature_ids = np.empty(0, dtype=int)
    prev_anat_arr = None
    prev_anat_valid = None
    last_motion_scale = 2.0

    detected_count = 0
    # Limit handling from options
    limit = options.get('limit_frames', total) if options.get('limit_frames', 0) > 0 else total
    live = options.get('live', False)

    for fi in range(total):
        if fi >= limit: break
        ret, frame = cap.read()
        if not ret:
            all_anat_kps.append({n: (0.0, 0.0) for n in ANAT_KP_NAMES})
            all_anat_confs.append({n: 0.0 for n in ANAT_KP_NAMES})
            all_feature_pts.append(np.empty((0, 2)))
            all_feature_ids.append(np.empty(0, dtype=int))
            all_n_total.append(0)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bfly_mask, center, area = isolate_butterfly(frame, prev_center, prev_area)

        if bfly_mask is None or area < 500:
            all_anat_kps.append({n: (0.0, 0.0) for n in ANAT_KP_NAMES})
            all_anat_confs.append({n: 0.0 for n in ANAT_KP_NAMES})
            all_feature_pts.append(np.empty((0, 2)))
            all_feature_ids.append(np.empty(0, dtype=int))
            all_n_total.append(0)
            prev_gray = gray
            continue

        prev_center = center
        prev_area = area

        # ── Stage 2a: Anatomical landmarks ──
        anat_kps, anat_cfs = detect_anatomical_keypoints(frame, bfly_mask, center)

        if prev_gray is not None and prev_anat_arr is not None and np.any(prev_anat_valid):
            prev_valid_pts = prev_anat_arr[prev_anat_valid].astype(np.float32).reshape(-1, 1, 2)
            tracked_anat, anat_valid = track_with_optical_flow(
                prev_gray, gray, prev_valid_pts, fb_scale=max(1.0, last_motion_scale * 0.5))
            valid_indices = np.where(prev_anat_valid)[0]
            for j, i in enumerate(valid_indices):
                if j < len(anat_valid) and anat_valid[j]:
                    px, py = int(tracked_anat[j, 0]), int(tracked_anat[j, 1])
                    if 0 <= px < w and 0 <= py < h and bfly_mask[py, px] > 0:
                        anat_kps[ANAT_KP_NAMES[i]] = (float(tracked_anat[j, 0]), float(tracked_anat[j, 1]))

        anat_pts_arr = []
        for n in ANAT_KP_NAMES:
            if anat_kps[n] is not None:
                anat_pts_arr.append(list(anat_kps[n]))
        anat_pts_arr = np.array(anat_pts_arr, dtype=np.float32) if anat_pts_arr else np.empty((0, 2), dtype=np.float32)

        if prev_gray is not None and len(active_feature_pts) > 0:
            tracked, valid = track_with_optical_flow(
                prev_gray, gray, active_feature_pts,
                fb_scale=max(1.0, last_motion_scale * 0.5))

            kept_pts = []
            kept_ids = []
            displacements = []
            for j in range(len(tracked)):
                if not valid[j]:
                    continue
                px, py = int(tracked[j][0]), int(tracked[j][1])
                if 0 <= px < w and 0 <= py < h and bfly_mask[py, px] > 0:
                    kept_pts.append(tracked[j])
                    kept_ids.append(active_feature_ids[j])
                    d = np.sqrt((tracked[j, 0] - active_feature_pts[j, 0])**2 +
                                (tracked[j, 1] - active_feature_pts[j, 1])**2)
                    displacements.append(d)

            active_feature_pts = np.array(kept_pts, dtype=np.float32) if kept_pts else np.empty((0, 2), dtype=np.float32)
            active_feature_ids = np.array(kept_ids, dtype=int) if kept_ids else np.empty(0, dtype=int)
            if displacements:
                last_motion_scale = float(np.median(displacements))

        need_redetect = (fi % REDETECT_INTERVAL == 0) or \
                        (len(active_feature_pts) < MAX_FEATURE_PTS // 2) or \
                        (len(active_feature_pts) < MIN_FEATURES_BEFORE_REDETECT)

        if need_redetect:
            new_features = detect_dense_features(gray, bfly_mask, anat_pts_arr)

            if len(active_feature_pts) > 0 and len(new_features) > 0:
                merged = list(active_feature_pts)
                merged_ids = list(active_feature_ids)
                for nf in new_features:
                    dists = np.sqrt(np.sum((active_feature_pts - nf)**2, axis=1))
                    if np.min(dists) > 8:
                        merged.append(nf)
                        merged_ids.append(next_feature_id)
                        next_feature_id += 1
                active_feature_pts = np.array(merged, dtype=np.float32)
                active_feature_ids = np.array(merged_ids, dtype=int)
            elif len(new_features) > 0:
                active_feature_pts = new_features.astype(np.float32)
                active_feature_ids = np.arange(next_feature_id,
                                                next_feature_id + len(new_features))
                next_feature_id += len(new_features)

            if len(active_feature_pts) > MAX_FEATURE_PTS:
                active_feature_pts = active_feature_pts[:MAX_FEATURE_PTS]
                active_feature_ids = active_feature_ids[:MAX_FEATURE_PTS]

        all_anat_kps.append(anat_kps)
        all_anat_confs.append(anat_cfs)
        all_feature_pts.append(active_feature_pts.copy())
        all_feature_ids.append(active_feature_ids.copy())

        n_total = sum(1 for v in anat_kps.values() if v is not None) + len(active_feature_pts)
        all_n_total.append(n_total)
        detected_count += 1

        prev_anat_arr = np.zeros((N_ANAT, 2), dtype=np.float32)
        prev_anat_valid = np.zeros(N_ANAT, dtype=bool)
        for i, n in enumerate(ANAT_KP_NAMES):
            if anat_kps[n] is not None and anat_cfs.get(n, 0) > 0.2:
                prev_anat_arr[i] = anat_kps[n]
                prev_anat_valid[i] = True

        prev_gray = gray

        if (fi + 1) % 200 == 0 or fi == total - 1:
            pct = (fi + 1) / total * 100
            print(f"    [{fi+1}/{total}] {pct:.0f}%  |  "
                  f"anat: {sum(1 for v in anat_kps.values() if v is not None)}/{N_ANAT}  "
                  f"features: {len(active_feature_pts)}  "
                  f"total: {n_total} pts")

        # ── Live Visualization ──
        if live:
            viz = frame.copy()
            # Draw blue overlay
            overlay = viz.copy()
            overlay[bfly_mask > 0] = [255, 100, 0]
            viz = cv2.addWeighted(viz, 0.7, overlay, 0.3, 0)
            
            # Draw trackers
            for p in active_feature_pts:
                cv2.circle(viz, (int(p[0]), int(p[1])), 3, (255, 255, 0), -1)
            for j, name in enumerate(ANAT_KP_NAMES):
                pt = anat_kps.get(name)
                if pt:
                    cv2.circle(viz, (int(pt[0]), int(pt[1])), 5, ANAT_COLORS[j], -1)
            
            cv2.putText(viz, f"LIVE TRACKING | Frame {fi}/{total}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Butterfly Dense Tracker (Live)", viz)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("  [INFO] User interrupted live tracking.")
                break

    if live:
        cv2.destroyAllWindows()

    cap.release()

    avg_pts = np.mean(all_n_total) if all_n_total else 0
    print(f"\n  → Processed ALL {total} frames")
    print(f"  → Butterfly detected: {detected_count} ({detected_count/total*100:.1f}%)")
    print(f"  → Average tracked points/frame: {avg_pts:.1f}")

    results = {
        'anat_kps': all_anat_kps,
        'anat_confs': all_anat_confs,
        'feature_pts': all_feature_pts,
        'feature_ids': all_feature_ids,
        'n_total': all_n_total,
        'fps': fps,
        'total_frames': total,
        'width': w,
        'height': h,
    }
    if SMOOTH_WINDOW > 0:
        smooth_trajectories(results, SMOOTH_WINDOW)
    return results


# ──────────────────────────────────────────────────────────
#  CSV export
# ──────────────────────────────────────────────────────────

def export_csv(results, path, species="Morpho peleides"):
    fps = results['fps']

    with open(path, 'w', newline='') as f:
        f.write(f"# Species: {species}\n")
        f.write(f"# FPS: {fps:.2f}\n")
        f.write(f"# Total frames: {results['total_frames']}\n")
        f.write(f"# Resolution: {results['width']}x{results['height']}\n")
        f.write(f"# Anatomical keypoints: {', '.join(ANAT_KP_NAMES)}\n")
        f.write(f"# Dense features: Shi-Tomasi + Lucas-Kanade optical flow\n")
        f.write("#\n")

        writer = csv.writer(f)

        # Header
        header = ['frame', 'time_s']
        for n in ANAT_KP_NAMES:
            header += [f'{n}_x', f'{n}_y', f'{n}_conf']
        header += ['n_features']
        max_feat = max((len(pts) for pts in results['feature_pts']), default=0)
        for i in range(max_feat):
            header += [f'feat{i}_x', f'feat{i}_y']
        writer.writerow(header)

        for fi in range(len(results['anat_kps'])):
            kps = results['anat_kps'][fi]
            cfs = results['anat_confs'][fi]
            fpts = results['feature_pts'][fi]

            row = [fi, f"{fi / fps:.4f}"]
            for n in ANAT_KP_NAMES:
                pt = kps.get(n)
                c = cfs.get(n, 0.0)
                if pt is not None:
                    row += [f"{pt[0]:.1f}", f"{pt[1]:.1f}", f"{c:.3f}"]
                else:
                    row += ["0.0", "0.0", "0.000"]
            row += [len(fpts)]
            for p in fpts:
                row += [f"{p[0]:.1f}", f"{p[1]:.1f}"]
            for _ in range(max_feat - len(fpts)):
                row += ["", ""]
            writer.writerow(row)

    print(f"  [CSV] {results['total_frames']} frames → {path}")


# ──────────────────────────────────────────────────────────
#  Annotated frames
# ──────────────────────────────────────────────────────────

ANAT_COLORS = [
    (0, 255, 255),   # head
    (0, 255, 0),     # thorax
    (0, 200, 200),   # abdomen
    (0, 0, 255),     # right_fw_tip
    (255, 0, 0),     # left_fw_tip
    (0, 165, 255),   # right_hw_tip
    (255, 0, 255),   # left_hw_tip
    (100, 100, 255), # right_fw_mid
    (255, 100, 100), # left_fw_mid
    (100, 200, 255), # right_hw_mid
    (255, 100, 200), # left_hw_mid
]


def save_annotated_frames(video_path, results, out_dir, n=12):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    n_proc = len(results['anat_kps'])
    indices = np.linspace(0, n_proc - 1, min(n, n_proc), dtype=int)

    for ii, fi in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        kps = results['anat_kps'][fi]
        cfs = results['anat_confs'][fi]
        fpts = results['feature_pts'][fi]

        bfly_mask, _, _ = isolate_butterfly(frame)
        if bfly_mask is not None:
            overlay = frame.copy()
            overlay[bfly_mask > 0] = [255, 100, 0]
            frame = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)

        if kps.get('thorax') and kps.get('head'):
            cv2.line(frame, _int_pt(kps['head']), _int_pt(kps['thorax']),
                     (255, 255, 0), 2)
        if kps.get('thorax') and kps.get('abdomen'):
            cv2.line(frame, _int_pt(kps['thorax']), _int_pt(kps['abdomen']),
                     (255, 255, 0), 2)

        for wing in ['left_fw', 'right_fw', 'left_hw', 'right_hw']:
            tip_name = f'{wing}_tip'
            mid_name = f'{wing}_mid'
            if kps.get('thorax') and kps.get(tip_name):
                cv2.line(frame, _int_pt(kps['thorax']),
                         _int_pt(kps[tip_name]), (255, 255, 0), 1)
            if kps.get(mid_name) and kps.get(tip_name):
                cv2.line(frame, _int_pt(kps[mid_name]),
                         _int_pt(kps[tip_name]), (200, 200, 0), 1)

        for p in fpts:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (255, 255, 0), -1)
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 0, 0), 1)

        for j, name in enumerate(ANAT_KP_NAMES):
            pt = kps.get(name)
            if pt is None:
                continue
            x, y = int(pt[0]), int(pt[1])
            c = cfs.get(name, 0)
            r = max(5, int(7 * c))
            color = ANAT_COLORS[j]
            cv2.circle(frame, (x, y), r, color, -1)
            cv2.circle(frame, (x, y), r + 2, (255, 255, 255), 1)
            short = name.replace('right_', 'R ').replace('left_', 'L ') \
                         .replace('_tip', '').replace('_mid', ' mid') \
                         .replace('_fw', 'FW').replace('_hw', 'HW').upper()
            cv2.putText(frame, short, (x + 8, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        t = fi / results['fps']
        n_total = results['n_total'][fi]
        cv2.putText(frame,
                    f"Frame {fi} | t={t:.2f}s | {n_total} tracked points",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imwrite(os.path.join(out_dir, f"kp_{ii:02d}_f{fi:04d}.png"), frame)

    cap.release()
    print(f"  [FRAMES] {n} annotated → {out_dir}")


def _int_pt(pt):
    return (int(pt[0]), int(pt[1]))


# ──────────────────────────────────────────────────────────
#  Dashboard plots
# ──────────────────────────────────────────────────────────

def plot_dashboard(results, out_dir, species="Morpho peleides"):
    os.makedirs(out_dir, exist_ok=True)
    N = len(results['anat_kps'])
    fps = results['fps']
    t = np.arange(N) / fps

    # Extract anatomical keypoint arrays
    kp_xy = {}
    for n in ANAT_KP_NAMES:
        xs = np.array([results['anat_kps'][i].get(n, (0, 0))[0] for i in range(N)])
        ys = np.array([results['anat_kps'][i].get(n, (0, 0))[1] for i in range(N)])
        kp_xy[n] = (xs, ys)

    colors = ['#FFD700', '#00FF00', '#00CCCC', '#FF0000', '#0000FF',
              '#FF8C00', '#FF00FF', '#CC6666', '#6666CC', '#CCAA66', '#CC66AA']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Dense Tracking Dashboard — {species}', fontsize=14, fontweight='bold')

    axes[0, 0].plot(t, results['n_total'], 'b-', lw=0.5, alpha=0.7)
    axes[0, 0].axhline(np.mean(results['n_total']), color='r', ls='--', lw=1,
                        label=f"Mean: {np.mean(results['n_total']):.1f}")
    axes[0, 0].set_xlabel('Time (s)'); axes[0, 0].set_ylabel('# Tracked Points')
    axes[0, 0].set_title('Total Tracked Points per Frame')
    axes[0, 0].legend()

    tx, ty = kp_xy['thorax']
    for n, c in [('right_fw_tip', 'r'), ('left_fw_tip', 'b'),
                 ('right_hw_tip', 'orange'), ('left_hw_tip', 'm')]:
        ax_, ay_ = kp_xy[n]
        angle = np.degrees(np.arctan2(ay_ - ty, ax_ - tx))
        axes[0, 1].plot(t, angle, color=c, lw=0.4, alpha=0.7,
                        label=n.replace('_tip', '').replace('_', ' '))
    axes[0, 1].set_xlabel('Time (s)'); axes[0, 1].set_ylabel('Angle (°)')
    axes[0, 1].set_title('Wing Angles from Thorax')
    axes[0, 1].legend(fontsize=8)

    for j, n in enumerate(['thorax', 'head', 'right_fw_tip', 'left_fw_tip']):
        xs, ys = kp_xy[n]
        axes[1, 0].scatter(xs, ys, c=t, cmap='viridis', s=1, alpha=0.3)
    axes[1, 0].set_xlabel('X (px)'); axes[1, 0].set_ylabel('Y (px)')
    axes[1, 0].set_title('Key Trajectories (head, thorax, FW tips)')
    axes[1, 0].invert_yaxis()

    n_feat = [len(results['feature_pts'][i]) for i in range(N)]
    n_anat = [sum(1 for n in ANAT_KP_NAMES
                  if results['anat_kps'][i].get(n) is not None
                  and results['anat_confs'][i].get(n, 0) > 0.1)
              for i in range(N)]
    axes[1, 1].fill_between(t, 0, n_anat, alpha=0.5, label='Anatomical', color='green')
    axes[1, 1].fill_between(t, n_anat, [a + f for a, f in zip(n_anat, n_feat)],
                            alpha=0.5, label='Dense features', color='blue')
    axes[1, 1].set_xlabel('Time (s)'); axes[1, 1].set_ylabel('# Points')
    axes[1, 1].set_title('Point Source Breakdown')
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'tracking_dashboard.png'), dpi=150)
    print(f"  [PLOT] tracking_dashboard.png")


# ──────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dense Keypoint Butterfly Tracker")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--species", default="Morpho peleides")
    parser.add_argument("--output-dir", default="output/multipoint")
    parser.add_argument("--live", action="store_true", help="Show live tracking window")
    parser.add_argument("--limit-frames", type=int, default=0, help="Limit number of frames to process")
    args = parser.parse_args()

    # Create dummy results to pass config to track_all_frames
    config = {'live': args.live, 'limit_frames': args.limit_frames if args.limit_frames > 0 else 999999}
    
    print("═" * 60)
    print("  Dense Multi-Keypoint Butterfly Tracker")
    print("  Butterfly isolation → 11 anatomical + ~15 texture features")
    print("  Lucas-Kanade optical flow with forward-backward validation")
    print("═" * 60)
    print()

    print("[1/4] Tracking all frames...")
    options = {'live': args.live, 'limit_frames': args.limit_frames}
    results = track_all_frames(args.video, args.output_dir, options=options)

    print("\n[2/4] Exporting keypoint CSV...")
    csv_path = os.path.join(args.output_dir, "keypoints_all_frames.csv")
    export_csv(results, csv_path, args.species)

    print("\n[3/4] Saving annotated frames...")
    save_annotated_frames(args.video, results,
                          os.path.join(args.output_dir, "frames"), n=12)

    print("\n[4/4] Generating dashboard plots...")
    plot_dashboard(results, os.path.join(args.output_dir, "plots"), args.species)

    print()
    print("═" * 60)
    print("  ✓ Dense tracking complete!")
    print(f"  Avg points/frame: {np.mean(results['n_total']):.1f}")
    print(f"  CSV:    {csv_path}")
    print(f"  Frames: {args.output_dir}/frames/")
    print(f"  Plots:  {args.output_dir}/plots/")
    print("═" * 60)


if __name__ == "__main__":
    main()
