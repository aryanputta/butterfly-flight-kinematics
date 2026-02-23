#!/usr/bin/env python3
"""
Morpho peleides Wing Tracker
Anatomical keypoints on both wings with optical flow + gentle correction.
Matches the reference layout: L/R FW Tip, FW Mid, Center, HW Mid, HW Tip, Thorax.
Press Q to close. Video loops.
"""

import cv2
import numpy as np
import sys

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "data/raw/morpho_peleides.mp4"

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"Cannot open {VIDEO}"); sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = max(1, int(1000 / fps))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow("Morpho Wing Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Morpho Wing Tracker", 900, 650)
cv2.moveWindow("Morpho Wing Tracker", 50, 50)

LK = dict(winSize=(31, 31), maxLevel=4,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.003))


def butterfly_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin = (cv2.inRange(hsv, np.array([0, 20, 50]), np.array([30, 180, 255])) |
            cv2.inRange(hsv, np.array([155, 20, 50]), np.array([180, 180, 255])))
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    reject = cv2.dilate(skin | dark, np.ones((35, 35), np.uint8))
    blue = cv2.inRange(hsv, np.array([85, 35, 40]), np.array([135, 255, 255]))
    bfly = cv2.bitwise_and(blue, cv2.bitwise_not(reject))
    bfly = cv2.morphologyEx(bfly, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    bfly = cv2.morphologyEx(bfly, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=3)
    return bfly


def get_wings(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wings = [c for c in cnts if cv2.contourArea(c) > 800]
    if len(wings) >= 2:
        wings.sort(key=cv2.contourArea, reverse=True)
        return sorted(wings[:2], key=lambda c: cv2.boundingRect(c)[0])
    if len(wings) == 1 and cv2.contourArea(wings[0]) > 3000:
        c = wings[0]
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            pts = c.reshape(-1, 2)
            l, r = pts[pts[:, 0] < cx], pts[pts[:, 0] >= cx]
            if len(l) > 10 and len(r) > 10:
                return [l.reshape(-1, 1, 2), r.reshape(-1, 1, 2)]
    return None


def anatomical_keypoints(wing_contours):
    """Extract the 11 named anatomical keypoints from two wing contours."""
    roi = cv2.boundingRect(np.vstack(wing_contours))
    roi_cx = roi[0] + roi[2] // 2

    pts, names = [], []
    for contour in wing_contours:
        px = contour.reshape(-1, 2)
        M = cv2.moments(contour)
        if M["m00"] == 0: continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        side = "L" if cx < roi_cx else "R"

        fw_tip = px[px[:, 1].argmin()]
        below = px[px[:, 1] > cy]
        hw_tip = (below[below[:, 0].argmin()] if side == "L" else below[below[:, 0].argmax()]) \
            if len(below) > 0 else px[px[:, 1].argmax()]
        base = px[px[:, 0].argmax()] if side == "L" else px[px[:, 0].argmin()]
        fw_mid = np.array([(fw_tip[0] + cx) // 2, (fw_tip[1] + cy) // 2], np.float32)
        hw_mid = np.array([(hw_tip[0] + cx) // 2, (hw_tip[1] + cy) // 2], np.float32)

        pts.extend([fw_tip.astype(np.float32), fw_mid,
                     np.array([cx, cy], np.float32), hw_mid,
                     hw_tip.astype(np.float32)])
        names.extend([f"{side} FW Tip", f"{side} FW Mid", f"{side} Center",
                       f"{side} HW Mid", f"{side} HW Tip"])

    if len(pts) >= 10:
        thorax = (pts[2] + pts[7]) / 2
        pts.append(thorax.astype(np.float32))
        names.append("Thorax")
    return (pts, names) if len(pts) >= 10 else (None, None)


def local_snap(pt, mask, radius=50):
    x, y = int(round(pt[0])), int(round(pt[1]))
    h, w = mask.shape
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    if mask[y, x] > 0:
        return pt
    x1, y1 = max(0, x - radius), max(0, y - radius)
    x2, y2 = min(w, x + radius), min(h, y + radius)
    local = mask[y1:y2, x1:x2]
    ys, xs = np.where(local > 0)
    if len(xs) == 0:
        return pt
    nearest = np.argmin((xs - (x - x1)) ** 2 + (ys - (y - y1)) ** 2)
    return np.array([xs[nearest] + x1, ys[nearest] + y1], dtype=np.float32)


def nearest_match(old, new):
    """Match new detections to old tracked by proximity (prevents L/R swap)."""
    if len(old) != len(new): return new
    used = set()
    matched = [None] * len(old)
    old_a, new_a = np.array(old), np.array(new)
    for i in range(len(old)):
        dists = np.sum((new_a - old_a[i]) ** 2, axis=1)
        for j in np.argsort(dists):
            if j not in used:
                matched[i] = new[j]; used.add(j); break
    return matched


def color_for(nm):
    if "FW Tip" in nm: return (0, 255, 0)
    if "FW Mid" in nm: return (100, 255, 100)
    if "Center" in nm: return (255, 255, 0)
    if "HW Mid" in nm: return (80, 200, 220)
    if "HW Tip" in nm: return (0, 165, 255)
    if "Thorax" in nm: return (0, 0, 255)
    return (200, 200, 200)


# ════════════════════════════════════
tracked = None
names = []
prev_gray = None
redetect_cd = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        tracked = prev_gray = None; redetect_cd = 0
        continue

    fnum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_img, w_img = frame.shape[:2]
    mask = butterfly_mask(frame)

    # ═══ INIT: place anatomical keypoints ═══
    if tracked is None:
        wings = get_wings(mask)
        if wings:
            pts, nms = anatomical_keypoints(wings)
            if pts:
                tracked = pts; names = nms; redetect_cd = 8
        if tracked is None:
            cv2.rectangle(display, (0, 0), (w_img, 26), (15, 15, 15), -1)
            cv2.putText(display, "Morpho peleides Wing Tracker", (6, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(display, f"Scanning... Frame {fnum}/{total_frames}",
                        (w_img - 270, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 255), 1)
            prev_gray = gray.copy()
            cv2.imshow("Morpho Wing Tracker", display)
            if (cv2.waitKey(delay) & 0xFF) in (ord("q"), 27): break
            continue

    # ═══ OPTICAL FLOW + forward-backward check ═══
    if prev_gray is not None:
        old = np.array(tracked, np.float32).reshape(-1, 1, 2)
        fwd, st_f, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, old, None, **LK)
        if fwd is not None:
            bwd, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, fwd, None, **LK)
            for j in range(len(tracked)):
                if st_f[j][0] == 1 and st_b[j][0] == 1:
                    fb = np.linalg.norm(old[j][0] - bwd[j][0])
                    if fb < 2.5:
                        tracked[j] = fwd[j][0].copy()

    # ═══ LOCAL MASK SNAP (keeps on same wing) ═══
    for j in range(len(tracked)):
        tracked[j] = local_snap(tracked[j], mask)

    # ═══ GENTLE RE-DETECTION every 8 frames (95/5 blend) ═══
    redetect_cd -= 1
    if redetect_cd <= 0:
        wings = get_wings(mask)
        if wings:
            det_pts, _ = anatomical_keypoints(wings)
            if det_pts and len(det_pts) == len(tracked):
                matched = nearest_match(tracked, det_pts)
                for j in range(len(tracked)):
                    if matched[j] is not None:
                        tracked[j] = np.array([
                            0.95 * tracked[j][0] + 0.05 * matched[j][0],
                            0.95 * tracked[j][1] + 0.05 * matched[j][1],
                        ], np.float32)
        redetect_cd = 8

    # ═══ DRAW: skeleton + points ═══
    for side in ("L", "R"):
        sm = {}
        for j, nm in enumerate(names):
            if nm.startswith(side):
                sm[nm] = (int(tracked[j][0]), int(tracked[j][1]))
        lc = (255, 200, 50) if side == "L" else (50, 200, 255)
        ck = f"{side} Center"
        if ck in sm:
            for tip in (f"{side} FW Tip", f"{side} HW Tip"):
                if tip in sm:
                    cv2.line(display, sm[ck], sm[tip], lc, 2, cv2.LINE_AA)

    for j, (nm, pt) in enumerate(zip(names, tracked)):
        px, py = int(pt[0]), int(pt[1])
        c = color_for(nm)
        r = 7 if "Thorax" in nm else 5
        cv2.circle(display, (px, py), r, c, -1)
        cv2.circle(display, (px, py), r + 2, (255, 255, 255), 1)
        cv2.putText(display, nm, (px + 10, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, c, 1, cv2.LINE_AA)

    # HUD
    n = len(tracked)
    cv2.rectangle(display, (0, 0), (w_img, 26), (15, 15, 15), -1)
    cv2.putText(display, "Morpho peleides Wing Tracker", (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(display, f"Pts: {n} | Tracking | Frame {fnum}/{total_frames}",
                (w_img - 310, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.circle(display, (w_img - 320, 12), 5, (0, 255, 0), -1)
    cv2.rectangle(display, (0, h_img - 20), (w_img, h_img), (15, 15, 15), -1)
    cv2.putText(display, "Press Q to close | Loops continuously", (6, h_img - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

    prev_gray = gray.copy()
    cv2.imshow("Morpho Wing Tracker", display)
    if (cv2.waitKey(delay) & 0xFF) in (ord("q"), 27): break

cap.release()
cv2.destroyAllWindows()
