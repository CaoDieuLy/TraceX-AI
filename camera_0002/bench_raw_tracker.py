"""Benchmark raw tracker on camera_0002 GT.

Strategy: feed GT bboxes as "perfect detections" (confidence=0.99) into the
raw tracker, then measure how often the tracker keeps a single GT person ID
within one raw track (purity) and how often it switches (IDS).

We use GT — not the real detector — to isolate the tracker's matching
behavior from detection quality. RT-DETR errors would conflate the two.

Metrics:
  - purity_macro     : mean purity over raw tracks weighted by length
  - purity_micro     : (frames assigned to majority GT) / total frames
  - id_switches      : transitions to a different GT id within one raw track
  - frag_per_gt      : average raw tracks covering one GT person (>1 = over-segmented)
  - n_handoffs       : raw tracks whose purity < 0.95 (clear sign of cross-person merge)
  - time_ms_per_frame: wall-clock per frame

Run twice: once on HEAD (baseline) and once on the patched tracker (P1).
"""
from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

REPO = Path("/teamspace/studios/this_studio/TraceX-AI")
sys.path.insert(0, str(REPO / "backend/services/metadata-service"))

from app.api.routers.tracking_pipeline import (  # noqa: E402
    BodyPartAdaptiveTracker,
    FrameDetection,
    _crop_from_bbox,
)

GT_PATH = REPO / "camera_0002/ground_truth.txt"
VIDEO_PATH = REPO / "camera_0002/video.mp4"
CAMERA = 2
FRAME_FPS = 30.0          # source video fps
SAMPLE_FPS = 4            # what the pipeline samples at
WITH_CROPS = True
MAX_SAMPLED_FRAMES = 1500 # ~half of 2999 → ~6 minutes of video at 4 fps
# Down-scale frames before cropping to keep memory bounded
# (1920x1080 BGR = 6 MB × 1500 frames = 9 GB without this).
FRAME_DOWNSCALE = 1.0     # 1.0 keeps original; crops still come from full-res reads


def load_gt(max_sampled_frames: int = MAX_SAMPLED_FRAMES) -> dict[int, list[tuple[int, tuple[int, int, int, int]]]]:
    """Returns {sampled_frame_idx: [(gt_person_id, (x1, y1, x2, y2)), ...]}.

    Capped at max_sampled_frames so the benchmark fits in memory.
    """
    sample_stride = int(round(FRAME_FPS / SAMPLE_FPS))  # 30/4 → 7 or 8
    by_frame: dict[int, list] = defaultdict(list)
    with GT_PATH.open() as f:
        for line in f:
            p = line.split()
            if len(p) < 7 or int(p[0]) != CAMERA:
                continue
            src_frame = int(p[2])
            if (src_frame - 1) % sample_stride != 0:
                continue
            sampled_idx = (src_frame - 1) // sample_stride
            if sampled_idx >= max_sampled_frames:
                continue
            gt_id = int(p[1])
            x = int(float(p[3])); y = int(float(p[4]))
            w = int(float(p[5])); h = int(float(p[6]))
            by_frame[sampled_idx].append((gt_id, (x, y, x + w, y + h)))
    return by_frame


def build_detections(gt_by_frame, with_crops: bool = True):
    """Convert GT into FrameDetection. Streams video sequentially (no seek)
    and stores only the small crops, not full frames, to keep RAM bounded.
    """
    det_to_gt: dict[tuple[int, tuple], int] = {}
    dets_by_frame: dict[int, list[FrameDetection]] = {}
    sample_stride = int(round(FRAME_FPS / SAMPLE_FPS))
    target_to_sampled = {fi * sample_stride: fi for fi in gt_by_frame}
    max_src_frame = max(target_to_sampled) if target_to_sampled else -1

    crops_by_sampled: dict[int, dict[tuple, np.ndarray]] = {}
    if with_crops:
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        src_idx = 0
        n_read = 0
        while src_idx <= max_src_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if src_idx in target_to_sampled:
                sampled_idx = target_to_sampled[src_idx]
                crops_by_sampled[sampled_idx] = {}
                for gt_id, bbox in gt_by_frame[sampled_idx]:
                    crops_by_sampled[sampled_idx][bbox] = _crop_from_bbox(frame, bbox)
                n_read += 1
            src_idx += 1
        cap.release()
        print(f"  read {n_read}/{len(gt_by_frame)} sampled frames from video")

    for fi in sorted(gt_by_frame):
        items = gt_by_frame[fi]
        crops = crops_by_sampled.get(fi, {})
        frame_dets: list[FrameDetection] = []
        for gt_id, bbox in items:
            fd = FrameDetection(
                frame_index=fi,
                timestamp_second=fi / SAMPLE_FPS,
                bbox=bbox,
                confidence=0.99,
                laplacian_score=200.0,
                crop_bgr=crops.get(bbox) if with_crops else None,
            )
            frame_dets.append(fd)
            det_to_gt[(fi, bbox)] = gt_id
        dets_by_frame[fi] = frame_dets
    return dets_by_frame, det_to_gt


def evaluate(tracker_factory, label: str, with_crops: bool = False,
             cached_dets=None):
    print(f"\n──────── {label} ────────")
    gt_by_frame = load_gt()
    n_frames = len(gt_by_frame)
    n_total_dets = sum(len(v) for v in gt_by_frame.values())
    print(f"  frames sampled = {n_frames}  total GT bboxes = {n_total_dets}")

    if cached_dets is not None:
        dets_by_frame, det_to_gt = cached_dets
    else:
        dets_by_frame, det_to_gt = build_detections(gt_by_frame, with_crops=with_crops)

    tracker = tracker_factory()
    t0 = time.time()
    tracklets = tracker.track(
        video_id="cam2_bench", camera_id="2", detections_by_frame=dets_by_frame,
    )
    # Drain any active tracks at end-of-video
    # (the tracker only completes on expire, so we read state directly)
    for tid, obs in list(tracker.active.items()):
        if obs:
            from app.api.routers.tracking_pipeline import LocalTracklet
            tracklets = tracklets + (LocalTracklet("cam2_bench", "2", tid, tuple(obs)),)
    for tid, obs in list(tracker.buffer.items()):
        if obs:
            from app.api.routers.tracking_pipeline import LocalTracklet
            tracklets = tracklets + (LocalTracklet("cam2_bench", "2", tid, tuple(obs)),)
    elapsed = time.time() - t0
    print(f"  wall time = {elapsed:.2f}s   per-frame = {1000*elapsed/n_frames:.2f}ms")
    print(f"  raw tracklets = {len(tracklets)}")

    # ── Compute purity / IDS / fragments ──────────────────────────────────
    purities = []
    weights = []
    micro_correct = 0
    micro_total = 0
    id_switches = 0
    n_low_purity = 0
    track_to_gt_majority = []
    raw_per_gt = defaultdict(set)  # gt_id → set of raw track ids

    for t in tracklets:
        if not t.observations:
            continue
        gt_seq = []
        for o in t.observations:
            key = (o.frame_index, tuple(o.bbox))
            gt_seq.append(det_to_gt.get(key, -1))
        gt_seq = [g for g in gt_seq if g != -1]
        if not gt_seq:
            continue
        counter = Counter(gt_seq)
        majority_gt, majority_count = counter.most_common(1)[0]
        purity = majority_count / len(gt_seq)
        purities.append(purity)
        weights.append(len(gt_seq))
        micro_correct += majority_count
        micro_total += len(gt_seq)
        if purity < 0.95:
            n_low_purity += 1
        # Count ID switches: transitions where gt_seq[k] != gt_seq[k-1]
        for k in range(1, len(gt_seq)):
            if gt_seq[k] != gt_seq[k-1]:
                id_switches += 1
        track_to_gt_majority.append(majority_gt)
        raw_per_gt[majority_gt].add(t.track_id)

    if not purities:
        print("  (no tracks evaluated)")
        return None

    purity_macro = float(np.average(purities, weights=weights))
    purity_micro = micro_correct / max(micro_total, 1)
    frag_per_gt = float(np.mean([len(v) for v in raw_per_gt.values()]))
    gt_covered = len(raw_per_gt)

    print(f"  purity (micro) = {purity_micro:.4f}")
    print(f"  purity (macro, length-weighted) = {purity_macro:.4f}")
    print(f"  ID switches inside raw tracks   = {id_switches}")
    print(f"  raw tracks with purity < 0.95   = {n_low_purity}  ({100*n_low_purity/len(purities):.1f}%)")
    print(f"  fragments per GT person         = {frag_per_gt:.2f}")
    print(f"  GT persons covered              = {gt_covered}/25")

    return {
        "label": label,
        "n_tracklets": len(tracklets),
        "purity_micro": purity_micro,
        "purity_macro": purity_macro,
        "id_switches": id_switches,
        "low_purity": n_low_purity,
        "frag_per_gt": frag_per_gt,
        "gt_covered": gt_covered,
        "time_s": elapsed,
        "ms_per_frame": 1000 * elapsed / max(n_frames, 1),
    }


# ─────────────────────────────────────────────────────────────────────────
# Tracker factories
# ─────────────────────────────────────────────────────────────────────────

def baseline_tracker():
    """Pre-P1 settings — the values were the defaults before today's edit."""
    return BodyPartAdaptiveTracker(
        track_thresh=0.30,
        low_thresh=0.10,
        new_track_threshold=0.30,
        max_match_cost=0.80,          # legacy
        max_lowconf_match_cost=0.80,  # legacy = same as high
        min_active_iou_short_gap=0.0, # disabled
        max_center_jump_ratio=1e9,    # disabled
        min_track_frames=2,
        min_track_density=0.03,
        use_kalman=False,
    )


def p1_tracker():
    """P1 — tightened geometry, no Kalman."""
    return BodyPartAdaptiveTracker(
        track_thresh=0.30,
        low_thresh=0.10,
        new_track_threshold=0.30,
        min_track_frames=2,
        min_track_density=0.03,
        use_kalman=False,
    )


def p3_tracker():
    """P3 — P1 + dedicated SORT-style Kalman filter + Mahalanobis gate."""
    return BodyPartAdaptiveTracker(
        track_thresh=0.30,
        low_thresh=0.10,
        new_track_threshold=0.30,
        min_track_frames=2,
        min_track_density=0.03,
        use_kalman=True,
    )


def diff(b, p):
    def fmt(x):
        return f"{x:+.4f}" if isinstance(x, float) else f"{x:+d}"
    print("\n──────── COMPARISON ────────")
    print(f"  purity_micro    {b['purity_micro']:.4f} → {p['purity_micro']:.4f}  ({fmt(p['purity_micro']-b['purity_micro'])})")
    print(f"  purity_macro    {b['purity_macro']:.4f} → {p['purity_macro']:.4f}  ({fmt(p['purity_macro']-b['purity_macro'])})")
    print(f"  ID switches     {b['id_switches']} → {p['id_switches']}  ({fmt(p['id_switches']-b['id_switches'])})")
    print(f"  low-purity #    {b['low_purity']} → {p['low_purity']}  ({fmt(p['low_purity']-b['low_purity'])})")
    print(f"  frag_per_gt     {b['frag_per_gt']:.2f} → {p['frag_per_gt']:.2f}  ({fmt(p['frag_per_gt']-b['frag_per_gt'])})  ← higher = more over-seg (OK)")
    print(f"  GT covered      {b['gt_covered']} → {p['gt_covered']}")
    print(f"  ms / frame      {b['ms_per_frame']:.2f} → {p['ms_per_frame']:.2f}  ({fmt(p['ms_per_frame']-b['ms_per_frame'])})")


if __name__ == "__main__":
    # Build detections WITH crops once (slow ~30-60s) so all three runs share
    # the same input. Without crops, P2 appearance is a no-op.
    print("Building detections from GT + video crops (one-time, ~30-60s)...")
    gt_by_frame = load_gt()
    cached_dets = build_detections(gt_by_frame, with_crops=True)
    print(f"  done: {sum(len(v) for v in cached_dets[0].values())} dets with crops")

    base = evaluate(baseline_tracker, "BASELINE (pre-P1)", cached_dets=cached_dets)
    p1   = evaluate(p1_tracker,       "P1 (tightened geometry)", cached_dets=cached_dets)
    p3   = evaluate(p3_tracker,       "P3 (P1 + Kalman filter)", cached_dets=cached_dets)
    if base and p1:
        print("\n══════ BASELINE → P1 ══════")
        diff(base, p1)
    if p1 and p3:
        print("\n══════ P1 → P3 ══════")
        diff(p1, p3)
    if base and p3:
        print("\n══════ BASELINE → P3 (total gain) ══════")
        diff(base, p3)
