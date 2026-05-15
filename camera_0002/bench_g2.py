"""Sweep G2 discriminative_margin on camera_0002 at 3 FPS (production config).

Tests if rejecting ambiguous Hungarian assignments (where 2 tracks compete
nearly-equally for one detection) reduces handoffs at the cost of more
fragmentation. The right margin depends on the cost matrix variance:
  - margin = 0:   no gate, current production behavior
  - margin = 0.05-0.15: reasonable range to test
  - margin > 0.20: probably too aggressive, breaks legit matches
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
    BodyPartAdaptiveTracker, FrameDetection, LocalTracklet,
)

GT_PATH = REPO / "camera_0002/ground_truth.txt"
CAMERA = 2
FRAME_FPS = 30.0
SAMPLE_FPS = 3
TIME_WINDOW_S = 250.0


def load_gt():
    stride = int(round(FRAME_FPS / SAMPLE_FPS))
    max_src = int(TIME_WINDOW_S * FRAME_FPS)
    by_frame: dict[int, list] = defaultdict(list)
    with GT_PATH.open() as f:
        for line in f:
            p = line.split()
            if len(p) < 7 or int(p[0]) != CAMERA:
                continue
            src = int(p[2])
            if src > max_src or (src - 1) % stride != 0:
                continue
            sampled = (src - 1) // stride
            gt_id = int(p[1])
            x, y = int(float(p[3])), int(float(p[4]))
            w, h = int(float(p[5])), int(float(p[6]))
            by_frame[sampled].append((gt_id, (x, y, x + w, y + h)))
    return by_frame


def build_dets(gt_by_frame):
    det_to_gt = {}
    by_frame: dict[int, list[FrameDetection]] = {}
    for fi in sorted(gt_by_frame):
        dets = []
        for gt_id, bbox in gt_by_frame[fi]:
            fd = FrameDetection(
                frame_index=fi, timestamp_second=fi / SAMPLE_FPS,
                bbox=bbox, confidence=0.99, laplacian_score=200.0,
            )
            dets.append(fd)
            det_to_gt[(fi, bbox)] = gt_id
        by_frame[fi] = dets
    return by_frame, det_to_gt


def evaluate(margin: float, label: str):
    print(f"\n──────── {label} ────────")
    gt = load_gt()
    dets_by_frame, det_to_gt = build_dets(gt)

    # Production tracker config at 3 FPS (auto-scaled defaults)
    _r4 = 4.0 / SAMPLE_FPS
    _rs = SAMPLE_FPS / 4.0
    tracker = BodyPartAdaptiveTracker(
        track_thresh=0.30, low_thresh=0.10, new_track_threshold=0.30,
        min_track_frames=2, min_track_density=0.03,
        max_head_center_distance=120.0 * _r4,
        max_foot_distance=150.0 * _r4,
        max_predicted_distance=180.0 * _r4,
        max_center_jump_ratio=2.0 * _r4,
        min_active_iou_short_gap=min(0.05 * _r4, 0.20),
        track_buffer=max(int(round(20 * _rs)), 8),
        max_buffer_frames=max(int(round(300 * _rs)), 100),
        discriminative_margin=margin,
    )

    t0 = time.time()
    tracklets = tracker.track("cam2", "2", dets_by_frame)
    for tid, obs in list(tracker.active.items()):
        if obs:
            tracklets = tracklets + (LocalTracklet("cam2", "2", tid, tuple(obs)),)
    for tid, obs in list(tracker.buffer.items()):
        if obs:
            tracklets = tracklets + (LocalTracklet("cam2", "2", tid, tuple(obs)),)
    elapsed = time.time() - t0

    purities = []
    weights = []
    micro_c = 0; micro_t = 0; idsw = 0; low_purity = 0
    raw_per_gt = defaultdict(set)
    for t in tracklets:
        seq = []
        for o in t.observations:
            g = det_to_gt.get((o.frame_index, tuple(o.bbox)), -1)
            if g != -1:
                seq.append(g)
        if not seq:
            continue
        c = Counter(seq)
        maj, cnt = c.most_common(1)[0]
        p = cnt / len(seq)
        purities.append(p); weights.append(len(seq))
        micro_c += cnt; micro_t += len(seq)
        if p < 0.95:
            low_purity += 1
        for k in range(1, len(seq)):
            if seq[k] != seq[k-1]:
                idsw += 1
        raw_per_gt[maj].add(t.track_id)

    purity = micro_c / max(micro_t, 1)
    frag = float(np.mean([len(v) for v in raw_per_gt.values()])) if raw_per_gt else 0
    cov = len(raw_per_gt)

    print(f"  margin={margin}  n_tr={len(tracklets)}  purity={purity:.4f}  "
          f"IDsw={idsw}  low-purity={low_purity}  frag/GT={frag:.2f}  "
          f"cov={cov}/25  ms/fr={1000*elapsed/len(gt):.2f}")
    return dict(margin=margin, n_tr=len(tracklets), purity=purity, idsw=idsw,
                low_purity=low_purity, frag=frag, cov=cov,
                ms_per_frame=1000 * elapsed / max(len(gt), 1))


if __name__ == "__main__":
    margins = [0.00, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    rows = []
    for m in margins:
        r = evaluate(m, f"discriminative_margin = {m}")
        rows.append(r)

    print("\n" + "═" * 80)
    print(f"{'margin':>7}  {'n_tr':>5}  {'purity':>8}  {'IDsw':>5}  "
          f"{'lowp':>5}  {'frag/GT':>8}  {'cov':>4}  {'ms/fr':>6}")
    print("─" * 80)
    for r in rows:
        print(f"{r['margin']:>7.2f}  {r['n_tr']:>5}  {r['purity']:>8.4f}  "
              f"{r['idsw']:>5}  {r['low_purity']:>5}  {r['frag']:>8.2f}  "
              f"{r['cov']:>3}/25  {r['ms_per_frame']:>6.2f}")

    baseline = next(r for r in rows if r["margin"] == 0.0)
    print(f"\nΔ vs margin=0 (purity={baseline['purity']:.4f}):")
    for r in rows:
        if r["margin"] == 0.0:
            continue
        # F-score: purity² / √(frag) — penalize fragmentation
        import math
        f_base = baseline["purity"] ** 2 / math.sqrt(max(baseline["frag"], 1))
        f_r = r["purity"] ** 2 / math.sqrt(max(r["frag"], 1))
        print(f"  margin={r['margin']:.2f}  purity {r['purity']-baseline['purity']:+.4f}  "
              f"IDsw {r['idsw']-baseline['idsw']:+d}  "
              f"frag/GT {r['frag']-baseline['frag']:+.2f}  "
              f"F-score {f_r-f_base:+.4f}")
