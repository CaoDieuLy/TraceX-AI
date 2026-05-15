"""Classify the 27% bbox-level errors of the P3 tracker on camera_0002.

For each raw track with purity < 0.95, find every "ID switch" (frame k where
gt_seq[k] != gt_seq[k-1]) and label it as one of:

  CROSSING       — at switch frame, BOTH GT persons are visible in the frame
                   (their trajectories actually cross — tracker swapped ID)
  OCCLUSION      — at switch frame, only the NEW GT is visible (the old one
                   disappeared or hadn't appeared yet — handoff to neighbor)
  SWAP_CROWD     — both GTs were present continuously for several frames
                   before AND after the switch (handoff inside a tight cluster)
  DETECTION_AMBIG— the swap-frame bbox overlaps BOTH GT boxes significantly
                   (IoU > 0.30 with both) → 1 detection covering 2 people
  UNKNOWN        — none of the above (rare, sanity bucket)

This decides which fix to prioritize next.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/teamspace/studios/this_studio/TraceX-AI")
sys.path.insert(0, str(REPO / "backend/services/metadata-service"))

from app.api.routers.tracking_pipeline import (  # noqa: E402
    BodyPartAdaptiveTracker,
    FrameDetection,
    LocalTracklet,
    _bbox_iou,
    _crop_from_bbox,
)

GT_PATH = REPO / "camera_0002/ground_truth.txt"
VIDEO_PATH = REPO / "camera_0002/video.mp4"
CAMERA = 2
FRAME_FPS = 30.0
SAMPLE_FPS = 4
MAX_SAMPLED_FRAMES = 1500


def load_gt(max_sampled_frames: int = MAX_SAMPLED_FRAMES):
    sample_stride = int(round(FRAME_FPS / SAMPLE_FPS))
    by_frame: dict[int, list] = defaultdict(list)
    with GT_PATH.open() as f:
        for line in f:
            p = line.split()
            if len(p) < 7 or int(p[0]) != CAMERA:
                continue
            src_frame = int(p[2])
            if (src_frame - 1) % sample_stride != 0:
                continue
            sampled = (src_frame - 1) // sample_stride
            if sampled >= max_sampled_frames:
                continue
            gt_id = int(p[1])
            x = int(float(p[3])); y = int(float(p[4]))
            w = int(float(p[5])); h = int(float(p[6]))
            by_frame[sampled].append((gt_id, (x, y, x + w, y + h)))
    return by_frame


def build_detections(gt_by_frame):
    """Re-uses the same flow as bench_raw_tracker but skips real crops (not
    needed for classification — Kalman tracker doesn't read crop_bgr)."""
    det_to_gt: dict[tuple[int, tuple], int] = {}
    dets_by_frame: dict[int, list[FrameDetection]] = {}
    for fi in sorted(gt_by_frame):
        items = gt_by_frame[fi]
        frame_dets: list[FrameDetection] = []
        for gt_id, bbox in items:
            fd = FrameDetection(
                frame_index=fi,
                timestamp_second=fi / SAMPLE_FPS,
                bbox=bbox,
                confidence=0.99,
                laplacian_score=200.0,
            )
            frame_dets.append(fd)
            det_to_gt[(fi, bbox)] = gt_id
        dets_by_frame[fi] = frame_dets
    return dets_by_frame, det_to_gt


def p3_tracker():
    return BodyPartAdaptiveTracker(
        track_thresh=0.30, low_thresh=0.10, new_track_threshold=0.30,
        min_track_frames=2, min_track_density=0.03,
        use_kalman=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────

# Build a GT presence index: for each GT person, set of frames they appear in
# and bbox per frame.
def build_gt_index(gt_by_frame):
    gt_presence: dict[int, dict[int, tuple]] = defaultdict(dict)
    for fi, rows in gt_by_frame.items():
        for gt_id, bbox in rows:
            gt_presence[gt_id][fi] = bbox
    return gt_presence


def classify_switch(
    *, switch_frame: int,
    old_gt: int, new_gt: int,
    swap_bbox: tuple,
    gt_presence: dict[int, dict[int, tuple]],
    look_back: int = 3,
    look_forward: int = 3,
) -> str:
    """Decide which handoff bucket the switch belongs to."""
    old_at = gt_presence.get(old_gt, {})
    new_at = gt_presence.get(new_gt, {})

    old_here = switch_frame in old_at
    new_here = switch_frame in new_at

    # DETECTION_AMBIG: one detection bbox overlaps BOTH GTs significantly.
    # We use the bbox the tracker actually assigned at switch_frame.
    if old_here and new_here:
        iou_old = _bbox_iou(swap_bbox, old_at[switch_frame])
        iou_new = _bbox_iou(swap_bbox, new_at[switch_frame])
        if iou_old > 0.30 and iou_new > 0.30:
            return "DETECTION_AMBIG"

    # CROSSING: both visible at switch frame (trajectories actually cross)
    if old_here and new_here:
        return "CROSSING"

    # OCCLUSION: new appears but old is gone (or vice versa)
    if new_here and not old_here:
        return "OCCLUSION"
    if old_here and not new_here:
        # Tracker handed off to someone who isn't even there yet — strange but
        # bucket as occlusion (the old person occluded → tracker grabbed
        # whatever bbox came next, mislabeled as 'new_gt' a frame later).
        return "OCCLUSION"

    # SWAP_CROWD: neither at switch_frame, but both present nearby
    near_old = any(switch_frame + d in old_at
                   for d in range(-look_back, look_forward + 1))
    near_new = any(switch_frame + d in new_at
                   for d in range(-look_back, look_forward + 1))
    if near_old and near_new:
        return "SWAP_CROWD"

    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────
# DRIVER
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("Loading GT and running P3 tracker...")
    gt_by_frame = load_gt()
    dets_by_frame, det_to_gt = build_detections(gt_by_frame)
    gt_presence = build_gt_index(gt_by_frame)

    tracker = p3_tracker()
    tracklets = tracker.track("cam2_bench", "2", dets_by_frame)
    # Drain final state
    for tid, obs in list(tracker.active.items()):
        if obs:
            tracklets = tracklets + (LocalTracklet("cam2_bench", "2", tid, tuple(obs)),)
    for tid, obs in list(tracker.buffer.items()):
        if obs:
            tracklets = tracklets + (LocalTracklet("cam2_bench", "2", tid, tuple(obs)),)
    print(f"  P3 produced {len(tracklets)} tracklets")

    bucket_counts: Counter = Counter()
    bucket_examples: dict[str, list] = defaultdict(list)
    total_switches = 0
    low_purity_tracks = 0

    for t in tracklets:
        if not t.observations:
            continue
        # Per-obs GT id
        seq = []
        for o in t.observations:
            seq.append((o.frame_index, o.bbox, det_to_gt.get((o.frame_index, o.bbox), -1)))
        seq = [s for s in seq if s[2] != -1]
        if len(seq) < 2:
            continue

        # Compute purity
        gt_ids = [s[2] for s in seq]
        counter = Counter(gt_ids)
        purity = counter.most_common(1)[0][1] / len(gt_ids)
        if purity >= 0.95:
            continue
        low_purity_tracks += 1

        # Find every ID switch
        for k in range(1, len(seq)):
            old_fi, _, old_id = seq[k - 1]
            cur_fi, cur_bbox, cur_id = seq[k]
            if cur_id == old_id:
                continue
            total_switches += 1
            label = classify_switch(
                switch_frame=cur_fi,
                old_gt=old_id, new_gt=cur_id,
                swap_bbox=cur_bbox,
                gt_presence=gt_presence,
            )
            bucket_counts[label] += 1
            if len(bucket_examples[label]) < 3:
                bucket_examples[label].append(
                    (t.track_id, old_fi, cur_fi, old_id, cur_id)
                )

    print(f"\nLow-purity tracks  : {low_purity_tracks}")
    print(f"Total ID switches  : {total_switches}")
    print("\n──── Switch-cause breakdown ────")
    for label in ("CROSSING", "OCCLUSION", "SWAP_CROWD", "DETECTION_AMBIG", "UNKNOWN"):
        n = bucket_counts.get(label, 0)
        pct = 100.0 * n / max(total_switches, 1)
        print(f"  {label:18s}  {n:4d}  ({pct:5.1f}%)")
    print("\n──── Examples (track_id, prev_fi, switch_fi, old_gt → new_gt) ────")
    for label, examples in bucket_examples.items():
        print(f"  [{label}]")
        for ex in examples:
            print(f"    track={ex[0]:>5s}  fi {ex[1]}→{ex[2]}  gt {ex[3]} → {ex[4]}")


if __name__ == "__main__":
    main()
