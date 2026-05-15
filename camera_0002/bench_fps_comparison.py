"""Compare tracker behavior at different sampling FPS on camera_0002 GT.

Why this is fair:
  - GT runs at 30 FPS internally. We subsample it to N FPS the same way the
    production pipeline subsamples video.
  - The tracker sees exactly the bboxes a perfect detector would have produced
    at that FPS — no RT-DETR error confounds the FPS effect.
  - Same time window of video (capped by ~250s real time) so we compare
    same scene at different temporal resolutions.

Metrics:
  - Purity: % of bboxes in a raw track that match the majority GT person.
    Higher = less mixing of identities.
  - ID switches: GT-id transitions inside one raw track.
  - Frag/GT: avg raw tracks per real person. Higher = more over-seg (OK if
    appearance merge can rejoin).
  - n_tracklets, ms/frame.
"""
from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/teamspace/studios/this_studio/TraceX-AI")
sys.path.insert(0, str(REPO / "backend/services/metadata-service"))

from app.api.routers.tracking_pipeline import (  # noqa: E402
    BodyPartAdaptiveTracker,
    FrameDetection,
    LocalTracklet,
)

GT_PATH = REPO / "camera_0002/ground_truth.txt"
CAMERA = 2
FRAME_FPS = 30.0
TIME_WINDOW_S = 250.0   # ~4 minutes of video, same wall-clock across FPS


def load_gt_at_fps(sample_fps: int):
    """Subsample GT at the given fps from a fixed time window."""
    stride = int(round(FRAME_FPS / sample_fps))  # 30/4=7, 30/5=6, 30/6=5, 30/8=4 (rounded)
    max_src_frame = int(TIME_WINDOW_S * FRAME_FPS)
    by_frame: dict[int, list] = defaultdict(list)
    with GT_PATH.open() as f:
        for line in f:
            p = line.split()
            if len(p) < 7 or int(p[0]) != CAMERA:
                continue
            src_frame = int(p[2])
            if src_frame > max_src_frame:
                continue
            if (src_frame - 1) % stride != 0:
                continue
            sampled = (src_frame - 1) // stride
            gt_id = int(p[1])
            x = int(float(p[3])); y = int(float(p[4]))
            w = int(float(p[5])); h = int(float(p[6]))
            by_frame[sampled].append((gt_id, (x, y, x + w, y + h)))
    return by_frame


def build_detections(gt_by_frame, sample_fps: int):
    det_to_gt: dict[tuple[int, tuple], int] = {}
    dets_by_frame: dict[int, list[FrameDetection]] = {}
    for fi in sorted(gt_by_frame):
        items = gt_by_frame[fi]
        frame_dets: list[FrameDetection] = []
        for gt_id, bbox in items:
            fd = FrameDetection(
                frame_index=fi,
                timestamp_second=fi / sample_fps,
                bbox=bbox,
                confidence=0.99,
                laplacian_score=200.0,
            )
            frame_dets.append(fd)
            det_to_gt[(fi, bbox)] = gt_id
        dets_by_frame[fi] = frame_dets
    return dets_by_frame, det_to_gt


def _fps_tuned_tracker(sample_fps: int, tune: bool = True) -> BodyPartAdaptiveTracker:
    """Tracker with thresholds scaled to the target FPS.

    Pixel-per-frame thresholds shrink with FPS (people move less per frame),
    buffer counts grow (keep wall-clock seconds constant), IoU expectations
    rise. The 4 FPS values mirror the production defaults; everything else
    derives from a ratio. When tune=False, returns the un-scaled tracker so
    we can isolate the FPS effect.
    """
    if not tune:
        return BodyPartAdaptiveTracker(
            track_thresh=0.30, low_thresh=0.10, new_track_threshold=0.30,
            min_track_frames=2, min_track_density=0.03,
        )

    ratio = 4.0 / sample_fps  # 4 fps → 1.0; 6 fps → 0.667; 8 fps → 0.5

    # Pixel-per-frame distances shrink with higher FPS
    head_dist = 120.0 * ratio
    foot_dist = 150.0 * ratio
    pred_dist = 180.0 * ratio
    jump_ratio = 2.0 * ratio

    # Buffer counts scale with FPS to preserve wall-clock seconds:
    # at 4 fps we want ~5s short buffer (20 frames). At 6 fps that's 30 frames.
    # At 2 fps we want 10 frames (still 5s). Scaling factor is sample_fps/4.
    fps_ratio = sample_fps / 4.0  # 2fps→0.5, 4fps→1.0, 6fps→1.5
    buf_frames = max(int(round(20 * fps_ratio)), 8)
    max_buf = max(int(round(300 * fps_ratio)), 100)

    # Higher FPS = adjacent frames have higher IoU naturally → tighten gate
    iou_gate = 0.05 / ratio   # 4fps=0.05, 6fps=0.075, 8fps=0.10
    iou_gate = min(iou_gate, 0.20)

    return BodyPartAdaptiveTracker(
        track_thresh=0.30, low_thresh=0.10, new_track_threshold=0.30,
        min_track_frames=2, min_track_density=0.03,
        max_head_center_distance=head_dist,
        max_foot_distance=foot_dist,
        max_predicted_distance=pred_dist,
        track_buffer=buf_frames,
        max_buffer_frames=max_buf,
        min_active_iou_short_gap=iou_gate,
        max_center_jump_ratio=jump_ratio,
    )


def evaluate(sample_fps: int, tune: bool = True, label: str | None = None):
    tag = label or (f"{sample_fps} FPS" + (" tuned" if tune else " untuned"))
    print(f"\n──────── {tag} ────────")
    gt = load_gt_at_fps(sample_fps)
    n_frames = len(gt)
    n_total = sum(len(v) for v in gt.values())
    print(f"  frames sampled = {n_frames}  total GT bboxes = {n_total}")

    dets_by_frame, det_to_gt = build_detections(gt, sample_fps)

    tracker = _fps_tuned_tracker(sample_fps, tune=tune)
    if tune:
        print(f"  tuned: head={tracker.max_head_center_distance:.0f} "
              f"foot={tracker.max_foot_distance:.0f} "
              f"buffer={tracker.track_buffer}f "
              f"iou_gate={tracker.min_active_iou_short_gap:.3f} "
              f"jump_ratio={tracker.max_center_jump_ratio:.2f}")
    t0 = time.time()
    tracklets = tracker.track("cam2_bench", "2", dets_by_frame)
    # Drain final state
    for tid, obs in list(tracker.active.items()):
        if obs:
            tracklets = tracklets + (LocalTracklet("cam2_bench", "2", tid, tuple(obs)),)
    for tid, obs in list(tracker.buffer.items()):
        if obs:
            tracklets = tracklets + (LocalTracklet("cam2_bench", "2", tid, tuple(obs)),)
    elapsed = time.time() - t0

    print(f"  wall time = {elapsed:.2f}s   per-frame = {1000*elapsed/n_frames:.2f}ms")
    print(f"  raw tracklets = {len(tracklets)}")

    purities = []
    weights = []
    micro_correct = 0
    micro_total = 0
    id_switches = 0
    n_low_purity = 0
    raw_per_gt = defaultdict(set)

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
        for k in range(1, len(gt_seq)):
            if gt_seq[k] != gt_seq[k-1]:
                id_switches += 1
        raw_per_gt[majority_gt].add(t.track_id)

    if not purities:
        return None
    purity_macro = float(np.average(purities, weights=weights))
    purity_micro = micro_correct / max(micro_total, 1)
    frag_per_gt = float(np.mean([len(v) for v in raw_per_gt.values()]))
    gt_covered = len(raw_per_gt)

    print(f"  purity (micro)            = {purity_micro:.4f}")
    print(f"  ID switches               = {id_switches}")
    print(f"  raw tracks purity < 0.95  = {n_low_purity}")
    print(f"  fragments per GT person   = {frag_per_gt:.2f}")
    print(f"  GT persons covered        = {gt_covered}/25")

    return dict(
        fps=sample_fps,
        n_frames=n_frames,
        n_tracklets=len(tracklets),
        purity_micro=purity_micro,
        purity_macro=purity_macro,
        id_switches=id_switches,
        low_purity=n_low_purity,
        frag_per_gt=frag_per_gt,
        gt_covered=gt_covered,
        ms_per_frame=1000 * elapsed / max(n_frames, 1),
    )


if __name__ == "__main__":
    configs = [
        (2, False, "2 FPS untuned"),
        (2, True,  "2 FPS TUNED"),
        (3, False, "3 FPS untuned"),
        (3, True,  "3 FPS TUNED"),
        (4, False, "4 FPS (production default)"),
        (5, False, "5 FPS untuned"),
        (6, False, "6 FPS untuned"),
        (8, False, "8 FPS untuned"),
    ]
    rows = []
    for fps, tune, label in configs:
        r = evaluate(fps, tune=tune, label=label)
        if r:
            r['label'] = label
            r['tune'] = tune
            rows.append(r)

    print("\n" + "═" * 88)
    print(f"{'config':<30}  {'n_tr':>5}  {'purity':>8}  {'IDsw':>5}  "
          f"{'lowp':>5}  {'frag/GT':>8}  {'cov':>4}  {'ms/fr':>6}")
    print("─" * 88)
    for r in rows:
        print(f"{r['label']:<30}  {r['n_tracklets']:>5}  "
              f"{r['purity_micro']:>8.4f}  {r['id_switches']:>5}  "
              f"{r['low_purity']:>5}  {r['frag_per_gt']:>8.2f}  "
              f"{r['gt_covered']:>3}/25  {r['ms_per_frame']:>6.2f}")

    baseline = next((r for r in rows if r['fps'] == 4 and not r['tune']), None)
    if baseline:
        print(f"\nΔ vs 4 FPS baseline (purity={baseline['purity_micro']:.4f}):")
        for r in rows:
            if r is baseline:
                continue
            print(
                f"  {r['label']:<30}  "
                f"purity {r['purity_micro'] - baseline['purity_micro']:+.4f}  "
                f"IDsw {r['id_switches'] - baseline['id_switches']:+d}  "
                f"frag/GT {r['frag_per_gt'] - baseline['frag_per_gt']:+.2f}"
            )
