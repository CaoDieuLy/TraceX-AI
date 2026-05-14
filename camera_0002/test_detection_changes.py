"""Sanity tests for the Phase A detection changes (TraceX-AI).

Tests target tracking_pipeline.py helpers (no GPU required) plus ground-truth
driven sanity checks on the camera_0002 AICity scene.

Phase A changes ported to TraceX-AI:
  B1 - laplacian on crop                  (_crop_laplacian_variance)
  B2 - min_h=40 in _sanitize_dets_inplace (video_process.py)
  B3 - low-quality flag                   (_is_low_quality_crop + dataclass field)
  B4 - 8% padding when cropping           (_crop_from_bbox)
  B5 - NMS                                already present (_sanitize_dets_inplace, IoU 0.55)
  B7 - sort by conf * sqrt(area)          (video_process.py construction loop)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# TraceX-AI repo root: backend/services/metadata-service/app/api/routers/tracking_pipeline.py
# We import via the package path so relative imports inside the module work.
REPO_ROOT = Path("/teamspace/studios/this_studio/TraceX-AI")
sys.path.insert(0, str(REPO_ROOT / "backend/services/metadata-service"))

from app.api.routers.tracking_pipeline import (  # noqa: E402
    FrameDetection,
    TrackletObservation,
    TrackletQualityScorer,
    _crop_from_bbox,
    _crop_laplacian_variance,
    _is_low_quality_crop,
)

GT_PATH = REPO_ROOT / "camera_0002" / "ground_truth.txt"
VIDEO_PATH = REPO_ROOT / "camera_0002" / "video.mp4"
CAMERA_ID = 2


# ─────────────────────────────────────────────────────────────────────────
# UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────

def test_crop_padding_expands():
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    bbox = (100, 100, 200, 300)  # 100x200
    plain = _crop_from_bbox(img, bbox, padding_ratio=0.0)
    padded = _crop_from_bbox(img, bbox, padding_ratio=0.08)
    assert plain.shape == (200, 100, 3)
    exp_h = 200 + 2 * int(round(200 * 0.08))
    exp_w = 100 + 2 * int(round(100 * 0.08))
    assert padded.shape == (exp_h, exp_w, 3), padded.shape
    print(f"✓ B4 padding: {plain.shape[:2]} → {padded.shape[:2]}")


def test_crop_padding_clamps_at_edge():
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    bbox = (0, 0, 100, 200)
    padded = _crop_from_bbox(img, bbox, padding_ratio=0.08)
    assert padded.shape[0] >= 200 and padded.shape[1] >= 100
    # Top-left corner: no padding on top/left side
    assert padded.shape[0] <= 200 + int(round(200 * 0.08))
    print("✓ B4 padding clamps at frame edge")


def test_crop_laplacian_blurry_vs_sharp():
    sharp = np.zeros((120, 60, 3), dtype=np.uint8)
    sharp[::4] = 255
    blurry = cv2.GaussianBlur(sharp, (15, 15), 10)
    s_lap = _crop_laplacian_variance(sharp)
    b_lap = _crop_laplacian_variance(blurry)
    assert s_lap > b_lap * 5, f"sharp={s_lap}, blurry={b_lap}"
    print(f"✓ B1 laplacian: sharp={s_lap:.1f} >> blurry={b_lap:.1f}")


def test_crop_laplacian_handles_none():
    assert _crop_laplacian_variance(None) == 0.0
    assert _crop_laplacian_variance(np.zeros((2, 2, 3), dtype=np.uint8)) == 0.0
    print("✓ B1 laplacian returns 0.0 on missing / tiny crops")


def test_low_quality_flag_degenerate_wide():
    # Wide-and-short bbox (aspect ~0.5) → truly degenerate (clearly half body)
    assert _is_low_quality_crop((100, 100, 300, 200), frame_h=1080, frame_w=1920) is True
    print("✓ B3 flags clearly half-body wide boxes (aspect < 0.9)")


def test_low_quality_flag_bending_person_passes():
    # Slightly bent / squat person — aspect ~1.0, should NOT be flagged
    assert _is_low_quality_crop((500, 200, 700, 400), frame_h=1080, frame_w=1920) is False
    print("✓ B3 lets slightly bent persons through (aspect 1.0)")


def test_low_quality_flag_edge_touch():
    # Touches left edge
    assert _is_low_quality_crop((0, 100, 80, 300), frame_h=1080, frame_w=1920) is True
    # Touches right edge
    assert _is_low_quality_crop((1900, 100, 1920, 300), frame_h=1080, frame_w=1920) is True
    print("✓ B3 flags edge-clipped boxes")


def test_low_quality_flag_normal_pass():
    # Normal upright person box, well inside the frame
    assert _is_low_quality_crop((500, 200, 600, 500), frame_h=1080, frame_w=1920) is False
    print("✓ B3 lets normal upright bboxes through")


def test_framedetection_carries_flag():
    fd = FrameDetection(
        frame_index=0, timestamp_second=0.0, bbox=(0, 0, 10, 30),
        confidence=0.9, laplacian_score=100.0, is_low_quality_crop=True,
    )
    assert fd.is_low_quality_crop is True
    obs = TrackletObservation(
        frame_index=0, timestamp_second=0.0, bbox=(0, 0, 10, 30),
        confidence=0.9, laplacian_score=100.0, is_low_quality_crop=True,
    )
    assert obs.is_low_quality_crop is True
    print("✓ B3 flag wired through FrameDetection + TrackletObservation")


# ─────────────────────────────────────────────────────────────────────────
# GROUND-TRUTH DRIVEN TESTS ON CAMERA_0002
# ─────────────────────────────────────────────────────────────────────────

def load_gt_for_camera(camera_id: int) -> dict[int, list[tuple[int, int, int, int, int]]]:
    """Returns {frame_idx: [(person_id, x, y, w, h), ...]}.
    AICity schema: camera_id person_id frame_idx x y w h world_x world_y
    """
    frames: dict[int, list] = {}
    with GT_PATH.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            cam = int(parts[0])
            if cam != camera_id:
                continue
            pid = int(parts[1])
            fi = int(parts[2])
            x, y, w, h = (int(float(v)) for v in parts[3:7])
            frames.setdefault(fi, []).append((pid, x, y, w, h))
    return frames


def test_gt_min_h_filter_safety():
    gt = load_gt_for_camera(CAMERA_ID)
    heights = []
    widths = []
    for rows in gt.values():
        heights.extend(h for (_, _, _, _, h) in rows)
        widths.extend(w for (_, _, _, w, _) in rows)
    h_arr = np.array(heights)
    w_arr = np.array(widths)
    # Active thresholds (must match video_process.py _sanitize_dets_inplace defaults)
    MIN_H = 20
    MIN_W = 14
    pct_h = float((h_arr < MIN_H).mean() * 100)
    pct_w = float((w_arr < MIN_W).mean() * 100)
    pct_either = float(((h_arr < MIN_H) | (w_arr < MIN_W)).mean() * 100)
    print(f"  GT heights — p0.1={np.percentile(h_arr,0.1):.0f} "
          f"p1={np.percentile(h_arr,1):.0f} p50={np.percentile(h_arr,50):.0f}")
    print(f"  % GT < min_h={MIN_H}: {pct_h:.3f}%")
    print(f"  % GT < min_w={MIN_W}: {pct_w:.3f}%")
    print(f"  % GT dropped by either filter: {pct_either:.3f}%")
    assert pct_either <= 0.10, f"size filter drops {pct_either:.3f}% — over 0.1% budget"
    print(f"✓ B2 size filter drops ≤ 0.10% of GT (measured {pct_either:.3f}%)")


def test_gt_low_quality_flag_rate():
    gt = load_gt_for_camera(CAMERA_ID)
    FRAME_W, FRAME_H = 1920, 1080
    flagged = total = edge_only = aspect_only = 0
    for rows in gt.values():
        for (_, x, y, w, h) in rows:
            total += 1
            bbox = (x, y, x + w, y + h)
            bw = max(bbox[2] - bbox[0], 1)
            bh = max(bbox[3] - bbox[1], 1)
            aspect = bh / bw
            bad_aspect = aspect < 1.2 or aspect > 4.0
            touches = (bbox[0] < 5 or bbox[1] < 5
                       or bbox[2] > FRAME_W - 5 or bbox[3] > FRAME_H - 5)
            if touches:
                edge_only += 1
            if bad_aspect:
                aspect_only += 1
            if _is_low_quality_crop(bbox, FRAME_H, FRAME_W):
                flagged += 1
    pct = flagged / total * 100
    print(f"  total={total}  flagged={flagged} ({pct:.1f}%)  "
          f"edge={edge_only}  bad_aspect={aspect_only}")
    assert pct < 50.0
    print(f"✓ B3 low-quality flag rate {pct:.1f}% on camera_2 — embedding pool keeps majority")


def test_gt_crop_laplacian_distribution():
    gt = load_gt_for_camera(CAMERA_ID)
    sample_frames = sorted(gt.keys())
    # 500 frames → ~6k crops, gives stable 0.1th-percentile estimates
    if len(sample_frames) > 500:
        idx = np.linspace(0, len(sample_frames) - 1, 500).astype(int)
        sample_frames = [sample_frames[i] for i in idx]

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    laps = []
    for fi in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi - 1)  # GT is 1-indexed
        ok, frame = cap.read()
        if not ok:
            continue
        for (_, x, y, w, h) in gt[fi]:
            crop = _crop_from_bbox(frame, (x, y, x + w, y + h))
            lap = _crop_laplacian_variance(crop)
            if lap > 0:
                laps.append(lap)
    cap.release()
    if not laps:
        print("  (no crops sampled)")
        return
    arr = np.array(laps)
    p1, p10, p50, p90 = np.percentile(arr, [1, 10, 50, 90])
    print(f"  Crop laplacian on real GT (n={len(arr)}): "
          f"p1={p1:.1f} p10={p10:.1f} p50={p50:.1f} p90={p90:.1f}")
    thresh = 30.0
    pct_below = float((arr < thresh).mean() * 100)
    print(f"  % below min_laplacian={thresh}: {pct_below:.4f}%")
    assert pct_below <= 0.10, f"min_laplacian drops {pct_below:.3f}% — over budget"
    print(f"✓ B1 min_laplacian={thresh} (balanced for demo) drops "
          f"only {pct_below:.4f}% of GT crops")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("UNIT TESTS")
    print("=" * 70)
    test_crop_padding_expands()
    test_crop_padding_clamps_at_edge()
    test_crop_laplacian_blurry_vs_sharp()
    test_crop_laplacian_handles_none()
    test_low_quality_flag_degenerate_wide()
    test_low_quality_flag_bending_person_passes()
    test_low_quality_flag_edge_touch()
    test_low_quality_flag_normal_pass()
    test_framedetection_carries_flag()

    print()
    print("=" * 70)
    print("GROUND-TRUTH TESTS (camera_0002 — AICity)")
    print("=" * 70)
    test_gt_min_h_filter_safety()
    print()
    test_gt_low_quality_flag_rate()
    print()
    test_gt_crop_laplacian_distribution()

    print()
    print("All checks finished.")
