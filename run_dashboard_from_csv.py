#!/usr/bin/env python3
"""Generate tracking dashboard from existing keypoints CSV (no video needed)."""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multipoint_tracker import ANAT_KP_NAMES, plot_dashboard

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "output/combined/tracking/keypoints_all_frames.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output/combined/tracking/plots"
    df = pd.read_csv(csv_path, comment="#")
    n = len(df)
    fps = 29.97
    if "# FPS:" in open(csv_path).read()[:500]:
        for line in open(csv_path):
            if line.startswith("# FPS:"):
                fps = float(line.split(":", 1)[1].strip())
                break
    anat_kps = []
    anat_confs = []
    feature_pts = []
    n_total = []
    for i in range(n):
        kps = {}
        cfs = {}
        for name in ANAT_KP_NAMES:
            xc, yc, cc = f"{name}_x", f"{name}_y", f"{name}_conf"
            if xc in df.columns:
                x, y = df.loc[i, xc], df.loc[i, yc]
                if pd.notna(x) and pd.notna(y) and float(x) > 0 and float(y) > 0:
                    kps[name] = (float(x), float(y))
                    cfs[name] = float(df.loc[i, cc]) if cc in df.columns else 0.8
        anat_kps.append(kps)
        anat_confs.append(cfs)
        nf = int(df.loc[i, "n_features"]) if "n_features" in df.columns else 0
        pts = []
        for j in range(nf):
            fx, fy = f"feat{j}_x", f"feat{j}_y"
            if fx in df.columns and pd.notna(df.loc[i, fx]):
                pts.append([float(df.loc[i, fx]), float(df.loc[i, fy])])
        feature_pts.append(np.array(pts, dtype=np.float32) if pts else np.empty((0, 2)))
        n_total.append(sum(1 for v in kps.values() if v) + len(pts))
    results = {
        "anat_kps": anat_kps,
        "anat_confs": anat_confs,
        "feature_pts": feature_pts,
        "n_total": n_total,
        "fps": fps,
        "total_frames": n,
        "width": 640,
        "height": 464,
    }
    os.makedirs(out_dir, exist_ok=True)
    plot_dashboard(results, out_dir, "Morpho peleides")
    print(f"Dashboard saved to {out_dir}/tracking_dashboard.png")

if __name__ == "__main__":
    main()
