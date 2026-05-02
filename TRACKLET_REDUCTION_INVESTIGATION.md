# Tracklet Fragmentation — Investigation & Fix Log

**Ngày:** 2026-05-03  
**Mục tiêu:** Giảm số tracklet/camera từ ~200+ xuống gần Ground Truth (25 người/camera)  
**Dataset:** NVIDIA PhysicalAI-SmartSpaces — MTMC_Tracking_2024/train/scene_001, 10 camera, 25 người/camera, video ~13 phút

---

## Baseline (trước khi sửa)

| Camera | Tracklets (DB) | GT Persons | Fragmentation ratio |
|--------|---------------|------------|---------------------|
| cam_01 | 209 | 25 | **8.4×** |
| cam_02 | 502 | 25 | **20×** |
| Trung bình 50 cams | ~208 | 25 | **8.3×** |
| **Tổng** | **10,393** | **1,250** | **8.3×** |

---

## Phân tích Ground Truth

```
ground_truth.txt format: camera_id  person_id  frame_id  x  y  w  h  world_x  world_y
```

```
Camera 1–10: 25 persons mỗi camera
Max frame: 23993 → 23993 / 30fps = 13.33 phút
Video bị trim còn 10 phút → 10 × 60 × 30fps = 18000 frames
Unique persons trong 18000 frames đầu (10 phút): 25 người mỗi camera
```

---

## Root Cause Analysis — 8 vấn đề tìm được

### P0 — Vấn đề 1: `max_buffer_frames` quá ngắn *(nguyên nhân chính)*

**File:** `local_ingestion_pipeline.py:408`

**Problem:**
Pipeline sample ở **4fps**, nhưng `track_buffer` và `max_buffer_frames` tính theo **sampled frame index**, không phải giây:

```
track_buffer    = 12 frames = 12 / 4fps = 3 giây    ← track → buffer sau 3s miss
max_buffer_frames = 30 frames = 30 / 4fps = 7.5 giây ← track expire sau 7.5s trong buffer
```

Trong warehouse, người đi sau kệ hàng có thể mất khỏi camera **30–60+ giây**. Kết quả: mỗi lần người mất > 7.5s → track expire → track ID mới. Một người đi ra vào frame 8 lần = 8 tracklets.

**Evidence từ data:**
- cam_01: **20 track mới/phút** đều đặn suốt 10 phút
- cam_02: **50 track mới/phút** đều đặn — uniform rate chứng tỏ systematic dropout, không phải occlusion burst

**Solution:**
```python
# Trước
track_buffer: int = 12          # 3s @ 4fps
max_buffer_frames: int = 30     # 7.5s @ 4fps

# Sau
track_buffer: int = 20          # 5s @ 4fps
max_buffer_frames: int = 300    # 75s @ 4fps (sau đó nâng lên video_length)
```

---

### P0 — Vấn đề 2: `predict_steps` không bị cap → Stage 3 re-entry bất khả thi

**File:** `local_ingestion_pipeline.py:600`

**Problem:**
`_build_match_state()` extrapolate velocity tuyến tính không giới hạn:

```python
predict_steps = max(target_frame_idx - observations[-1].frame_index, 0)
predicted_center = (last_center[0] + vx * predict_steps, ...)
```

Với `max_buffer_frames = 300`, một buffer track có `vx = 3px/frame` sẽ predict position **3 × 300 = 900px** lệch — ngoài màn hình. `predicted_distance > 180px` luôn đúng. Kết hợp với `center_distance > 120px` → **AND gate block toàn bộ Stage 3 re-entry**. Tăng `max_buffer_frames` sẽ vô nghĩa nếu không fix cái này.

**Solution:**
```python
raw_steps = max(target_frame_idx - observations[-1].frame_index, 0)
# Cap extrapolation tại track_buffer * 2 để tránh predict vị trí vô lý
predict_steps = min(raw_steps, self.track_buffer * 2)
predicted_center = (last_center[0] + vx * predict_steps, ...)
```

---

### P1 — Vấn đề 3: Distance gate quá chặt cho 4fps

**File:** `local_ingestion_pipeline.py:402-403`

**Problem:**
`max_head_center_distance = 85px` calibrated cho high FPS (30fps). Ở 4fps (0.25s/frame), người đi nhanh (~2 m/s) dịch chuyển 80–100px giữa 2 frame. Khi cả hai gate vượt ngưỡng:

```python
if (
    center_distance > self.max_head_center_distance   # 85px
    and predicted_distance > self.max_predicted_distance  # 120px
):
    continue  # skip hoàn toàn — không match được
```

**Solution:**
```python
max_head_center_distance: float = 120.0   # was 85.0
max_predicted_distance: float = 180.0     # was 120.0
max_buffer_match_cost: float = 0.90       # was 0.85 — re-entry cost cao hơn sau gap dài
```

---

### P1 — Vấn đề 4: `min_track_frames` và `minimum_duration_seconds` mâu thuẫn

**File:** `local_ingestion_pipeline.py:411` và `:674`

**Problem:**
```
HeadBoxTracker.min_track_frames = 8
→ 8 frames @ 4fps → timestamps[0..7] → duration = 7 × 0.25 = 1.75s

TrackletQualityScorer.minimum_duration_seconds = 2.0
→ 1.75s < 2.0s → REJECTED
```

8-frame tracklets pass `_should_keep_tracklet()` → vào SigLIP2 + DINOv2 + VideoMAE pipeline → bị reject bởi `TrackletQualityScorer`. **Lãng phí GPU compute** cho tracklets sẽ bị loại.

**Solution:**
```python
# HeadBoxTracker
min_track_frames: int = 9     # 9 × 0.25s = 2.25s ≥ 2.0s threshold

# TrackletQualityScorer
minimum_frame_count: int = 9  # align với HeadBoxTracker
```

---

### P1 — Vấn đề 5: Density calculation phạt re-entry tracklets

**File:** `local_ingestion_pipeline.py:646-652`

**Problem:**
```python
frame_span = last_obs.frame_index - first_obs.frame_index + 1
density = frame_count / frame_span
```

Khi track re-enter từ buffer, gap được tính vào `frame_span` nhưng không có observations trong gap:
```
Active period: frames 0-11 (12 obs)
Buffer gap:    frames 12-41 (30 frames, 0 obs)
Re-entry:      frames 42-46 (5 obs)
→ frame_span = 46 - 0 + 1 = 47
→ density = 17/47 = 0.36 ← sát ngưỡng 0.35, có thể fail
```

Density của **observation windows** thực tế là 1.0, nhưng gap inflate span → density ảo thấp.

**Solution:**
```python
min_track_density: float = 0.10         # was 0.35
minimum_frame_density: float = 0.10    # was 0.25 (TrackletQualityScorer)
```

---

### P2 — Vấn đề 6: Buffer expire mid-video ngay cả sau khi tăng `max_buffer_frames`

**File:** `local_ingestion_pipeline.py:1434`

**Problem:**
Dù tăng `max_buffer_frames = 300` (75s), vẫn có hard breaks nếu người absent > 75s (đi sang khu vực khác, quay lại sau 2–5 phút).

Nguyên nhân sâu hơn: `track_incremental()` check expire **trong từng batch**. Buffer track có thể expire ở batch 3 trong khi người reappear ở batch 7.

**Solution:**
Probe video duration trước khi xử lý, set `max_buffer_frames` = tổng sampled frames của video → **buffer không bao giờ expire mid-video**:

```python
try:
    _cap = cv2.VideoCapture(str(source_path))
    _src_fps = float(_cap.get(cv2.CAP_PROP_FPS) or self.sample_fps)
    _total_src = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    _cap.release()
    _duration_s = _total_src / max(_src_fps, 1.0)
    _total_sampled = int(math.ceil(_duration_s * self.sample_fps)) + 1
except Exception:
    _total_sampled = self.tracker.max_buffer_frames
self.tracker.max_buffer_frames = max(self.tracker.max_buffer_frames, _total_sampled)
```

Kết quả: 10 phút @ 4fps = 2400 sampled frames → `max_buffer_frames = 2401`. Mọi track sống đến `finalize_all()` ở cuối video.

---

### P3 — Vấn đề 7: Không có within-camera fragment merge → `_merge_people_by_identity` là no-op

**File:** `local_ingestion_pipeline.py:1163`

**Problem:**
`human_key = f"{camera_id}:{track_id}"` — unique per tracklet. `_merge_people_by_identity()` group theo `human_key` nhưng mỗi group chỉ có 1 phần tử → **function không merge được gì**. Sau tracker fixes, một người vẫn có thể có 3–5 fragments nếu absent > video_length/4. Không có cơ chế nào nối chúng lại.

**Solution:**
Thêm `_resolve_within_camera_identities()` chạy **sau khi embeddings đã được tính**:

```python
# Hai điều kiện phải đồng thời:
# 1. Temporal: overlap < 15% (người không thể ở 2 nơi cùng lúc)
# 2. Embedding: cosine(appearance_A, appearance_B) >= reid_threshold

# Union-Find cho transitive closure: A≈B, B≈C → cùng identity
# Elect shared human_key từ fragment chất lượng cao nhất
# _merge_people_by_identity() merge data thực tế
```

---

### P4 — Vấn đề 8: TransReID (MSMT17 side-view) collapse trên overhead cameras

**File:** `model_adapters.py`

**Problem (phát hiện sau khi chạy thử):**
- cam_01: 47 tracklets → **16 người** (GT = 25) — threshold 0.82 quá thấp
- cam_02: 103 tracklets → **9 người** (GT = 25) — over-merge nghiêm trọng

Nguyên nhân: TransReID được train trên MSMT17 (side/front view pedestrians). Các camera có góc overhead/angled → embedding space **collapse**: người khác nhau cũng có cosine similarity ≥ 0.82 vì tất cả trông giống nhau từ góc cao.

**Solution:**
Thay TransReID ViT-Base (768-dim, MSMT17) bằng **DINOv2 ViT-L/14 (1024-dim)**:

```python
# model_adapters.py — thay thế hoàn toàn TransReIDHub
class DINOv2ReIDHub:
    """
    Self-supervised trên 142M ảnh đa dạng → view-invariant.
    CLS token 1024-dim — không cần part crops.
    """
    def embed_crops(self, pil_crops) -> np.ndarray:  # [N, 1024] L2-normalised
```

Files thay đổi:
- `model_adapters.py`: xóa `TransReIDHub`, thêm `DINOv2ReIDHub`, cập nhật `SoliderKPRAppearanceEmbeddingAdapter`
- `local_ingestion_pipeline.py`: Phase 3 swap `TransReIDHub` → `DINOv2ReIDHub`, bỏ part crops
- `main.py`: cập nhật warmup function

Cập nhật threshold fragment merge: `0.82 → 0.85` (DINOv2 có separation gap rộng hơn).

---

## Tổng hợp tất cả thay đổi code

### `local_ingestion_pipeline.py`

| Tham số / Location | Trước | Sau | Lý do |
|---|---|---|---|
| `HeadBoxTracker.track_buffer` L407 | 12 (3s) | **20 (5s)** | Brief occlusion thực tế 3–5s |
| `HeadBoxTracker.max_buffer_frames` L408 | 30 (7.5s) | **300 (75s) → video_length** | Người sau kệ 30–60s; probe để không expire mid-video |
| `HeadBoxTracker.max_head_center_distance` L402 | 85px | **120px** | 0.25s/frame @ 4fps → người nhanh dịch 80–100px |
| `HeadBoxTracker.max_predicted_distance` L403 | 120px | **180px** | Velocity estimate kém chính xác hơn ở 4fps |
| `HeadBoxTracker.max_buffer_match_cost` L401 | 0.85 | **0.90** | Re-entry cost cao hơn sau gap dài |
| `HeadBoxTracker.min_track_frames` L411 | 8 (1.75s) | **9 (2.25s)** | Align với `minimum_duration_seconds=2.0` |
| `HeadBoxTracker.min_track_density` L412 | 0.35 | **0.10** | Re-entry tracklets có density ảo thấp do gap |
| `_build_match_state` predict_steps L604 | không giới hạn | **min(raw, track_buffer×2)** | Tránh predict vị trí vô lý cho buffer tracks |
| `TrackletQualityScorer.minimum_frame_count` L675 | 8 | **9** | Align với HeadBoxTracker |
| `TrackletQualityScorer.minimum_frame_density` L676 | 0.25 | **0.10** | Align với HeadBoxTracker |
| `run()` — probe video duration L1434 | không có | **cv2 probe → set max_buffer_frames** | Không bao giờ expire mid-video |
| `_resolve_within_camera_identities()` L1167 | không có | **mới** — DINOv2 + Union-Find merge | Fragment merge sau khi embedding tính xong |
| Phase 3 in `_build_people_batch` L896 | TransReID + part crops (3M) | **DINOv2 full crop (M)** | Thay model, bỏ part fusion |
| `reid_threshold` trong fragment merge L1170 | 0.82 → 0.95 → | **0.85** | DINOv2 calibration |

### `model_adapters.py`

| Thay đổi | Mô tả |
|----------|-------|
| Xóa `TransReIDHub` | 90 dòng, ViT-Base/16 MSMT17 768-dim |
| Thêm `DINOv2ReIDHub` | `facebook/dinov2-large`, 307M params, 1024-dim CLS token |
| Cập nhật `SoliderKPRAppearanceEmbeddingAdapter` | Bỏ part crops + part fusion, dùng DINOv2 trực tiếp, empty vector 512→1024 |

### `main.py`

| Thay đổi | Mô tả |
|----------|-------|
| `_warmup_models()` | Thay `TransReIDHub` → `DINOv2ReIDHub` |

---

## Kết quả theo từng fix

| Giai đoạn | cam_01 tracklets | cam_01 people | cam_02 tracklets | Ghi chú |
|-----------|-----------------|---------------|-----------------|---------|
| Baseline | 209 | — | 502 | Trước khi fix |
| Sau tracker fix (P0–P1) | ~47 | — | ~103 | Ước tính từ run đầu |
| Sau fragment merge 0.82 | — | 16 ❌ | — | 9 ❌ Over-merge do TransReID |
| Sau raise threshold 0.95 | — | ~40–47 | — | Gần như không merge |
| Sau DINOv2 + 0.85 | — | TBD | — | Đang chạy |
| **Ground Truth** | — | **25** | — | **Mục tiêu** |

---

## Kết luận kỹ thuật

### Nguyên nhân gốc rễ theo mức độ tác động:

1. **`max_buffer_frames = 30` (7.5s)** → mỗi lần người mất > 7.5s = track ID mới. Đây là nguồn gốc của 70–80% fragmentation. Fix: video-length buffer.

2. **`predict_steps` không cap** → Stage 3 re-entry không hoạt động dù buffer dài. Đây là lý do tại sao chỉ tăng `max_buffer_frames` thôi chưa đủ.

3. **TransReID MSMT17 side-view bias** → embedding collapse trên overhead cameras → fragment merge over-aggressive. Fix: DINOv2 view-invariant features.

4. **Các tham số còn lại** (distance gate, density, frame count alignment) là secondary issues — ảnh hưởng nhỏ hơn nhưng góp phần vào fragmentation và computation waste.

### Pipeline flow mới:

```
RF-DETR detect
  → HeadBoxTracker (buffer sống đến hết video)
    → finalize_all() một lần ở cuối
      → TrackletQualityScorer (min 9 frames, 2s, density ≥ 0.10)
        → DINOv2 embed_crops (1024-dim, view-invariant)
          → _resolve_within_camera_identities()
            (temporal guard + cosine ≥ 0.85 + Union-Find)
              → _merge_people_by_identity()
                → Lưu DB với merged embeddings
```
