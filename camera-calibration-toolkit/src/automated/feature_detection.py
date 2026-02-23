"""
Feature Detection for Stereo Image Pairs
ECE 738 - Spring 2026, Project 3

Applies 4 feature detectors to rectified stereo pairs:
  - Harris Corner Detector
  - SIFT  (Scale-Invariant Feature Transform)
  - ORB   (Oriented FAST and Rotated BRIEF)
  - KAZE

Creative Component:
  - Cross-pair repeatability analysis (Mikolajczyk & Schmid, IJCV 2005)
"""

import cv2
import numpy as np
import os
import json
import csv
from itertools import product

# =============================================================================
# CONFIGURATION
# =============================================================================

LEFT_DIR  = "data/stereo_images/left/L_rectified"
RIGHT_DIR = "data/stereo_images/right/R_rectified"
PAIR_IDS  = [1, 7, 13]
RESULTS_DIR = "results/features"

NMS_DISTANCES = [10, 20]
NMS_D_DEFAULT = 10

HARRIS_BLOCK_SIZE = 2
HARRIS_KSIZE      = 3
HARRIS_K          = 0.04
HARRIS_THRESHOLD  = 0.10

SIFT_CONTRAST_THRESH = 0.02
SIFT_EDGE_THRESH     = 15
SIFT_N_FEATURES      = 0

ORB_N_FEATURES   = 8000
ORB_FAST_THRESH  = 7
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS     = 8

KAZE_THRESHOLD       = 0.0003
KAZE_N_OCTAVES       = 4
KAZE_N_OCTAVE_LAYERS = 4

REPEAT_RADII = [10, 15, 20]


# =============================================================================
# HELPER
# =============================================================================

def find_image(directory, stem):
    for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
        path = os.path.join(directory, stem + ext)
        if os.path.exists(path):
            return path
    return None


# =============================================================================
# FAST GRID-BASED NMS
# =============================================================================

def nms_keypoints(keypoints, D):
    """
    Grid-based Non-Maximum Suppression.
    Divides image into D x D cells, keeps strongest keypoint per cell.
    O(n log n) — works on millions of points instantly.
    """
    if len(keypoints) == 0:
        return []
    kps = sorted(keypoints, key=lambda k: k.response, reverse=True)
    grid = {}
    for kp in kps:
        col = int(kp.pt[0] / D)
        row = int(kp.pt[1] / D)
        cell = (row, col)
        if cell not in grid:
            grid[cell] = kp
    return list(grid.values())


# =============================================================================
# DETECTORS
# =============================================================================

def harris_to_keypoints(response_map, threshold_fraction):
    threshold = threshold_fraction * response_map.max()
    ys, xs = np.where(response_map > threshold)
    keypoints = []
    for x, y in zip(xs, ys):
        kp = cv2.KeyPoint(float(x), float(y), size=5,
                          response=float(response_map[y, x]))
        keypoints.append(kp)
    return keypoints


def detect_harris(gray):
    response = cv2.cornerHarris(gray, HARRIS_BLOCK_SIZE, HARRIS_KSIZE, HARRIS_K)
    response = cv2.normalize(response, None, 0, 1, cv2.NORM_MINMAX)
    return harris_to_keypoints(response, HARRIS_THRESHOLD)


def detect_sift(gray):
    sift = cv2.SIFT_create(
        nfeatures=SIFT_N_FEATURES,
        contrastThreshold=SIFT_CONTRAST_THRESH,
        edgeThreshold=SIFT_EDGE_THRESH
    )
    return list(sift.detect(gray, None))


def detect_orb(gray):
    orb = cv2.ORB_create(
        nfeatures=ORB_N_FEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_N_LEVELS,
        fastThreshold=ORB_FAST_THRESH
    )
    return list(orb.detect(gray, None))


def detect_kaze(gray):
    kaze = cv2.KAZE_create(
        threshold=KAZE_THRESHOLD,
        nOctaves=KAZE_N_OCTAVES,
        nOctaveLayers=KAZE_N_OCTAVE_LAYERS
    )
    return list(kaze.detect(gray, None))


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def count_corresponding(left_kps, right_kps, x_tol=20, y_tol=3):
    if not left_kps or not right_kps:
        return 0
    right_pts = np.array([kp.pt for kp in right_kps])
    count = 0
    for kp in left_kps:
        lx, ly = kp.pt
        dy = np.abs(right_pts[:, 1] - ly)
        dx = right_pts[:, 0] - lx
        if np.any((dy < y_tol) & (dx < x_tol) & (dx > -50)):
            count += 1
    return count


def detector_overlap(kps_a, kps_b, radius=10):
    if not kps_a or not kps_b:
        return 0.0
    pts_b = np.array([kp.pt for kp in kps_b])
    r2 = radius * radius
    count = 0
    for kp in kps_a:
        x, y = kp.pt
        dx = pts_b[:, 0] - x
        dy = pts_b[:, 1] - y
        if np.any(dx * dx + dy * dy < r2):
            count += 1
    return 100.0 * count / len(kps_a)


# =============================================================================
# CREATIVE COMPONENT — Cross-Pair Repeatability
# Reference: Mikolajczyk & Schmid, IJCV 2005
# =============================================================================

def repeatability_score(kps_ref, kps_other, radius=15):
    """
    Repeatability = fraction of reference keypoints that have a
    spatially close counterpart in another view.

    From Mikolajczyk & Schmid (2005):
        repeatability = |{p in ref : exists q in other s.t. dist(p,q) < r}|
                        / |ref|

    Purely geometric — no descriptors — measures how consistently
    the detector fires on the same physical points as the camera moves.
    """
    if not kps_ref or not kps_other:
        return 0.0
    pts_other = np.array([kp.pt for kp in kps_other])
    r2 = radius * radius
    count = 0
    for kp in kps_ref:
        x, y = kp.pt
        dx = pts_other[:, 0] - x
        dy = pts_other[:, 1] - y
        if np.any(dx * dx + dy * dy < r2):
            count += 1
    return 100.0 * count / len(kps_ref)


def cross_pair_repeatability(pair_keypoints, det_names, radii=REPEAT_RADII):
    print("\n" + "=" * 70)
    print("CREATIVE COMPONENT — Cross-Pair Repeatability")
    print("Reference: Mikolajczyk & Schmid, IJCV 2005")
    print("=" * 70)
    print("\nHow many features from Pair 1 reappear in Pairs 7 and 13?")
    print("(detector stability as camera position changes)\n")

    pair_ids = sorted(pair_keypoints.keys())
    if len(pair_ids) < 2:
        print("Need at least 2 pairs.")
        return {}

    ref_id = pair_ids[0]
    results = {}

    for det in det_names:
        print(f"  [{det}]")
        results[det] = {}
        kps_ref = pair_keypoints[ref_id][det]

        for other_id in pair_ids[1:]:
            kps_other = pair_keypoints[other_id][det]
            key = f"pair{ref_id}_vs_pair{other_id}"
            results[det][key] = {}

            for r in radii:
                score = repeatability_score(kps_ref, kps_other, radius=r)
                results[det][key][f"radius_{r}px"] = round(score, 1)
                print(f"    Pair {ref_id} -> Pair {other_id}  "
                      f"radius={r:2d}px:  {score:.1f}%")
        print()

    return results


# =============================================================================
# VISUALISATION
# =============================================================================

def draw_keypoints(image, keypoints, color=(0, 255, 0)):
    vis = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = max(2, int(kp.size / 2))
        cv2.circle(vis, (x, y), r, color, 1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 2, color, -1)
    return vis


def make_stereo_strip(left_vis, right_vis, label):
    for vis in [left_vis, right_vis]:
        for y in range(0, vis.shape[0], 40):
            col = (0, 80, 255) if y % 160 == 0 else (0, 40, 120)
            cv2.line(vis, (0, y), (vis.shape[1], y), col, 1)
    combined = np.hstack([left_vis, right_vis])
    bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(bar, "LEFT",  (combined.shape[1] // 4 - 30, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 1)
    cv2.putText(bar, "RIGHT", (3 * combined.shape[1] // 4 - 40, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 1)
    return np.vstack([bar, combined])


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("FEATURE DETECTION — ECE 738 Project 3")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "strips"), exist_ok=True)

    detectors = {
        "Harris": detect_harris,
        "SIFT":   detect_sift,
        "ORB":    detect_orb,
        "KAZE":   detect_kaze,
    }

    colors = {
        "Harris": (0, 255, 0),
        "SIFT":   (0, 128, 255),
        "ORB":    (255, 0, 128),
        "KAZE":   (255, 255, 0),
    }

    rows = []
    pair_keypoints = {}  # {pair_id: {det_name: [kps]}} — for repeatability

    for pair_id in PAIR_IDS:
        left_path  = find_image(LEFT_DIR,  f"L{pair_id}")
        right_path = find_image(RIGHT_DIR, f"R{pair_id}")

        if left_path is None or right_path is None:
            print(f"\nWARNING: Could not find pair {pair_id}, skipping.")
            continue

        print(f"\n{'─'*70}")
        print(f"Pair {pair_id}:  {os.path.basename(left_path)}  |  "
              f"{os.path.basename(right_path)}")
        print(f"{'─'*70}")

        img_left   = cv2.imread(left_path)
        img_right  = cv2.imread(right_path)
        gray_left  = cv2.cvtColor(img_left,  cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        all_left_kps  = {}
        all_right_kps = {}

        for det_name, det_fn in detectors.items():
            print(f"\n  [{det_name}]")

            raw_left  = det_fn(gray_left)
            raw_right = det_fn(gray_right)

            print(f"    Raw detections:  left={len(raw_left):5d}  "
                  f"right={len(raw_right):5d}")

            for D in NMS_DISTANCES:
                kps_l = nms_keypoints(raw_left,  D)
                kps_r = nms_keypoints(raw_right, D)
                print(f"    After NMS D={D:2d}:  left={len(kps_l):5d}  "
                      f"right={len(kps_r):5d}")

                if D == NMS_D_DEFAULT:
                    all_left_kps[det_name]  = kps_l
                    all_right_kps[det_name] = kps_r

                    corr     = count_corresponding(kps_l, kps_r)
                    corr_pct = 100.0 * corr / len(kps_l) if kps_l else 0
                    print(f"    Corresponding (approx): {corr} / {len(kps_l)} "
                          f"({corr_pct:.1f}%)")

                    rows.append({
                        "pair":          pair_id,
                        "detector":      det_name,
                        "nms_D":         D,
                        "raw_left":      len(raw_left),
                        "raw_right":     len(raw_right),
                        "nms_left":      len(kps_l),
                        "nms_right":     len(kps_r),
                        "corresponding": corr,
                        "corr_pct":      round(corr_pct, 1),
                    })

            vis_left  = draw_keypoints(img_left,  all_left_kps[det_name], colors[det_name])
            vis_right = draw_keypoints(img_right, all_right_kps[det_name], colors[det_name])

            label = (f"Pair {pair_id} | {det_name} | "
                     f"L:{len(all_left_kps[det_name])}  "
                     f"R:{len(all_right_kps[det_name])}  "
                     f"NMS D={NMS_D_DEFAULT}")
            strip = make_stereo_strip(vis_left, vis_right, label)

            cv2.imwrite(os.path.join(RESULTS_DIR, "strips",
                        f"pair{pair_id}_{det_name}_strip.jpg"), strip)

            for side, vis in [("L", vis_left), ("R", vis_right)]:
                cv2.imwrite(os.path.join(RESULTS_DIR, "visualizations",
                            f"pair{pair_id}_{side}_{det_name}.jpg"), vis)

        # Store for cross-pair repeatability
        pair_keypoints[pair_id] = {d: all_left_kps[d] for d in all_left_kps}

        # Detector overlap
        print(f"\n  Detector Overlap (% shared features, radius=10px):")
        det_names = list(detectors.keys())
        overlap_data = {}
        for a, b in product(det_names, det_names):
            if a == b:
                continue
            pct = detector_overlap(all_left_kps[a], all_left_kps[b])
            overlap_data[f"{a}_vs_{b}"] = round(pct, 1)
            print(f"    {a:8s} -> {b:8s}: {pct:.1f}%")

        with open(os.path.join(RESULTS_DIR,
                  f"pair{pair_id}_detector_overlap.json"), "w") as f:
            json.dump(overlap_data, f, indent=2)

        # Combined visualisation
        combined_left = img_left.copy()
        for det_name in det_names:
            for kp in all_left_kps[det_name]:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(combined_left, (x, y), 3, colors[det_name], -1)

        cv2.imwrite(os.path.join(RESULTS_DIR, "visualizations",
                    f"pair{pair_id}_ALL_detectors.jpg"), combined_left)

    # -------------------------------------------------------------------------
    # CREATIVE COMPONENT
    # -------------------------------------------------------------------------
    det_names = list(detectors.keys())
    repeat_results = cross_pair_repeatability(pair_keypoints, det_names)

    repeat_path = os.path.join(RESULTS_DIR, "cross_pair_repeatability.json")
    with open(repeat_path, "w") as f:
        json.dump(repeat_results, f, indent=2)
    print(f"Repeatability results saved to: {repeat_path}")

    # -------------------------------------------------------------------------
    # Summary CSV
    # -------------------------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, "feature_detection_summary.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary saved to: {csv_path}")

    # Final table
    print(f"\n{'='*70}")
    print(f"{'Pair':<6} {'Detector':<10} {'D':<4} "
          f"{'Raw L':<8} {'Raw R':<8} "
          f"{'NMS L':<8} {'NMS R':<8} {'Corr%':<8}")
    print("-" * 70)
    for r in rows:
        print(f"{r['pair']:<6} {r['detector']:<10} {r['nms_D']:<4} "
              f"{r['raw_left']:<8} {r['raw_right']:<8} "
              f"{r['nms_left']:<8} {r['nms_right']:<8} "
              f"{r['corr_pct']:<8}")

    print(f"\n{'='*70}")
    print(f"DONE — results in: {RESULTS_DIR}/")
    print(f"  strips/                       stereo pairs with epipolar lines")
    print(f"  visualizations/               individual + combined detector images")
    print(f"  cross_pair_repeatability.json creative component results")
    print(f"  feature_detection_summary.csv full summary table")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()