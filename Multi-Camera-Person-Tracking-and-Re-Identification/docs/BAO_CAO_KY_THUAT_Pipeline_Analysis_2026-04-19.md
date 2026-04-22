# BÁO CÁO KỸ THUẬT: Kiến trúc Pipeline Multi-Camera Person Tracking

**Ngày tạo:** 2026-04-19
**Dự án:** Multi-Camera Person Tracking and Re-Identification
**Phạm vi phân tích:** Detection → Tracking → VLM Input → Global Tracking

---

## Tóm tắt Điều hành

| Thành phần | Trạng thái | Chi tiết |
|-----------|-----------|---------|
| Detection Model | ✅ Đã implement | YOLO26-X, NMS-free |
| Tracking Model | ✅ Đã implement | ByteTrack-style Cascade Matching |
| VLM Input Source | ✅ Tracklet Crops | Ảnh crop người từ confirmed tracks |
| Global Tracking | ❌ Chưa implement | Chỉ xử lý per-video, không cross-camera |

---

## 1. MODEL DETECTION (YOLO26-X) ✅

### 1.1 Load Model YOLO26-X

**File:** `backend/legacy-engine/src_vlm/hospital_pipeline.py`

```67:74:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
@lru_cache(maxsize=1)
def _load_yolo26():
    """Load YOLO26-X - NMS-free, edge-optimized detector (56.9 AP)"""
    model = YOLO("yolo26x.pt")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.fuse()
    return model
```

### 1.2 Hàm Detection

```76:94:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
def _detect_people_yolo26(frame: np.ndarray, conf_threshold: float = 0.32) -> tuple[list[list[int]], list[float]]:
    """
    YOLO26-X detection - NMS-free end-to-end.
    Returns: (detections, confidences)
    """
    model = _load_yolo26()
    results = model(frame, verbose=False, conf=conf_threshold, classes=[0])  # class=0 là person

    detections: list[list[int]] = []
    confidences: list[float] = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x, y, w, h = box.xywh[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append([int(x), int(y), int(w), int(h)])
                confidences.append(conf)
    return detections, confidences
```

### 1.3 Tích hợp Detection vào Pipeline

```683:699:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
        # ── STEP 1: YOLO26-X Detection ───────────────────────────
        detections, det_confs = _detect_people_yolo26(frame, conf_threshold=0.32)

        # Filter by minimum person area
        filtered_det = []
        filtered_confs = []
        for bbox, conf in zip(detections, det_confs):
            if _bbox_area(bbox) >= DEFAULT_MIN_PERSON_AREA:  # 4,500 pixels
                filtered_det.append(bbox)
                filtered_confs.append(conf)

        detections = filtered_det
        det_confs = filtered_confs
```

---

## 2. MODEL TRACKING (ByteTrack) ✅

### 2.1 Định nghĩa Class ByteTrackStyleTracker

**File:** `backend/legacy-engine/src_vlm/hospital_pipeline.py`

```100:125:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
class ByteTrackStyleTracker:
    """
    ByteTrack-inspired cascade matching:
    1. Match high-confidence detections (≥0.35) first
    2. Match low-confidence detections (0.15-0.35) with unmatched tracks
    Reduces IDSw in crowded hospital scenes.
    """
    def __init__(
        self,
        high_conf_thresh: float = 0.35,
        low_conf_thresh: float = 0.15,
        iou_gate_high: float = 0.30,
        iou_gate_low: float = 0.25,
        max_age_seconds: float = 8.0,
        min_frames_to_confirm: int = 4,
    ):
        self.high_thresh = high_conf_thresh
        self.low_thresh = low_conf_thresh
        self.iou_gate_high = iou_gate_high
        self.iou_gate_low = iou_gate_low
        self.max_age = max_age_seconds
        self.min_confirm = min_frames_to_confirm

        self.next_track_id = 1
        self.active_tracks: dict[str, dict] = {}
        self.all_tracks: dict[str, dict] = {}
```

### 2.2 Logic Cascade Association

```142:186:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
    def associate(
        self,
        detections: list[list[int]],
        detections_conf: list[float],
        frame_idx: int,
        fps: float = 4.0,
    ) -> dict[int, str]:
        matched_track_ids: set[str] = set()
        assignments: dict[int, str] = {}

        # Split by confidence
        high_idx = [i for i, c in enumerate(detections_conf) if c >= self.high_thresh]
        low_idx = [i for i, c in enumerate(detections_conf) if self.low_thresh <= c < self.high_thresh]

        # ── Stage 1: High-confidence matching ─────────────────────
        for d_idx in high_idx:
            det_bbox = detections[d_idx]
            best_track_id = None
            best_iou = 0.0
            for track_id, track in self.active_tracks.items():
                if track_id in matched_track_ids:
                    continue
                iou = self._bbox_iou(track["last_bbox"], det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            if best_track_id and best_iou >= self.iou_gate_high:
                matched_track_ids.add(best_track_id)
                assignments[d_idx] = best_track_id

        # ── Stage 2: Low-confidence matching ──────────────────────
        for d_idx in low_idx:
            det_bbox = detections[d_idx]
            best_track_id = None
            best_iou = 0.0
            for track_id, track in self.active_tracks.items():
                if track_id in matched_track_ids:
                    continue
                iou = self._bbox_iou(track["last_bbox"], det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            if best_track_id and best_iou >= self.iou_gate_low:
                matched_track_ids.add(best_track_id)
                assignments[d_idx] = best_track_id
```

### 2.3 Khởi tạo Tracker trong Pipeline

```659:669:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
    # ── Initialize SOTA modules ─────────────────────────────────
    yolo_model = _load_yolo26()
    reid_extractor = _get_reid_extractor()
    tracker = ByteTrackStyleTracker(
        high_conf_thresh=0.35,
        low_conf_thresh=0.15,
        iou_gate_high=0.30,
        iou_gate_low=0.25,
        max_age_seconds=8.0,
        min_frames_to_confirm=4,
    )
    itself_engine = ITSELFSearchEngineLite()
```

### 2.4 Gọi Tracker Association

```715:720:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
        # ── STEP 3: ByteTrack Association ─────────────────────────
        assignments = tracker.associate(detections, det_confs, frame_idx, fps)
        for det_index, track_id in assignments.items():
            track = tracker.active_tracks.get(track_id)
            if track is not None:
                track.setdefault("embeddings", []).append(reid_embs[det_index])
```

---

## 3. NGUỒN INPUT CHO VLM ✅

### 3.1 Định nghĩa VLM Engine

**File:** `backend/legacy-engine/src_vlm/vlm_engine.py`

```12:35:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/vlm_engine.py
class VLM_Metadata_Engine:
    def __init__(self, use_mock=False):
        self.use_mock = use_mock
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.cpu_cores = multiprocessing.cpu_count()
        self.model_id = "Salesforce/blip-image-captioning-large"

        if use_mock:
            raise RuntimeError("Mock VLM mode is disabled for production ingestion.")
        if torch is None:
            raise RuntimeError("Torch is required for VLM metadata generation.")
        # ... model loading ...
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
```

### 3.2 Trích xuất Key Frames (KHÔNG phải toàn bộ video)

```52:70:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/vlm_engine.py
    def extract_key_frames(self, video_path, num_frames=8):
        """Trích xuất N khung hình — song song bằng ThreadPool (I/O bound)."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0:
            return []

        step = max(total_frames // num_frames, 1)
        frame_positions = [i * step for i in range(num_frames)]

        # Song song trích frame bằng ThreadPool
        with ThreadPoolExecutor(max_workers=min(num_frames, self.cpu_cores)) as pool:
            results = list(pool.map(
                self._extract_single_frame,
                [(video_path, pos) for pos in frame_positions]
            ))

        return [r for r in results if r is not None]
```

### 3.3 Batch Caption Generation

```72:89:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/vlm_engine.py
    def generate_captions_batch(self, images):
        """BATCH inference: gom tất cả ảnh, forward 1 lần duy nhất."""
        if not images:
            return []

        captions = []
        batch_size = self._vlm_batch_size()
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.model.generate(**inputs, max_new_tokens=70)
                else:
                    out = self.model.generate(**inputs, max_new_tokens=70)
            captions.extend(self.processor.batch_decode(out, skip_special_tokens=True))
        return captions
```

### 3.4 Xử lý VLM trong Pipeline - **TRACKLET CROPS** (Quan trọng!)

```785:793:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
    # ── STEP 5: VLM Captioning (BLIP) ───────────────────────────
    representative_frames = _read_frame_map(compressed_path, [int(candidate["frame_idx"]) for candidate in candidates])
    representative_crops: dict[str, Image.Image] = {}
    for candidate in candidates:
        crop = _crop_pil_from_frame(representative_frames.get(int(candidate["frame_idx"])), candidate["representative_bbox"])
        if crop is not None:
            representative_crops[candidate["candidate_id"]] = crop

    _apply_person_captions(candidates, compressed_path, vlm_engine, crop_map=representative_crops)
```

**🔑 ĐIỂM QUAN TRỌNG:** VLM nhận **TRACKLET CROPS** - ảnh crop từ bounding box của track, không phải video gốc hay detection frame thô.

### 3.5 Hàm Apply Captions

```858:920:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
def _apply_person_captions(
    candidates: list[dict],
    video_path: Path,
    vlm_engine,
    crop_map: dict[str, Image.Image] | None = None,
) -> None:
    # ... prepare jobs from crop_map ...
    try:
        captions = vlm_engine.generate_captions_batch([image for _, image in jobs])
    except Exception:
        captions = [_default_caption(candidate["camera_id"], candidate["track_id"]) for candidate, _ in jobs]

    caption_map = {candidate["candidate_id"]: _default_caption(candidate["camera_id"], candidate["track_id"]) for candidate in candidates}
    for (candidate, _), caption in zip(jobs, captions):
        normalized = " ".join(str(caption).split()).strip()
        caption_map[candidate["candidate_id"]] = normalized or _default_caption(candidate["camera_id"], candidate["track_id"])

    for candidate in candidates:
        candidate["person_caption"] = caption_map.get(candidate["candidate_id"], _default_caption(candidate["camera_id"], candidate["track_id"]))
        candidate["appearance_summary"] = candidate["person_caption"]
```

---

## 4. TẠO CANDIDATE/TRACKLET ✅

### 4.1 Xây dựng Candidates từ Tracks

```726:783:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
    # ── STEP 4: Build Candidates from Tracks ─────────────────────
    candidates: list[dict] = []
    for track_id, track in sorted(tracker.all_tracks.items(), key=lambda x: int(x[0])):
        frames = track["frames"]
        if len(frames) < DEFAULT_MIN_TRACK_FRAMES:  # 5 frames tối thiểu
            continue

        rep_frame = frames[-1]  # Frame cuối làm đại diện
        rep_bbox = track["bboxes"][-1]

        # Sample content frames
        content_frames = []
        sample_indices = np.linspace(0, len(frames) - 1, DEFAULT_CONTENT_FRAME_SAMPLES, dtype=int)  # 12 samples
        for idx in sample_indices:
            f_idx = frames[idx]
            bbox = track["bboxes"][idx]
            content_frames.append({
                "frame_idx": int(f_idx),
                "second": round(f_idx / fps, 3),
                "bbox": [int(v) for v in bbox],
            })

        # Calculate average embedding from track (temporal smoothing)
        track_embs = [np.asarray(embedding, dtype=np.float32) for embedding in track.get("embeddings", []) if embedding is not None]
        avg_embedding = np.mean(track_embs, axis=0) if track_embs else np.zeros(512, dtype=np.float32)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)

        candidate = {
            "candidate_id": f"{compressed_path.stem}_person_{track_id}",
            "video_id": compressed_path.name,
            "camera_id": camera_id,
            "track_id": str(track_id),
            "frame_idx": int(rep_frame),
            "start_frame": int(min(frames)),
            "end_frame": int(max(frames)),
            "bbox": [int(v) for v in rep_bbox],
            "representative_bbox": [int(v) for v in rep_bbox],
            "content_frames": content_frames,
            "embedding_vector": avg_embedding.tolist(),
            "person_caption": "",
            # ...
        }
        candidates.append(candidate)
```

---

## 5. GLOBAL TRACKING ❌ (CHƯA IMPLEMENT)

### 5.1 Track ID Reset cho mỗi Video

**File:** `backend/legacy-engine/src_vlm/hospital_pipeline.py`

```123:125:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
        self.next_track_id = 1  # ⚠️ RESET per-video - không có global ID
        self.active_tracks: dict[str, dict] = {}
        self.all_tracks: dict[str, dict] = {}
```

**⚠️ VẤN ĐỀ:** `track_id` bắt đầu từ 1 cho MỖI video, không có global tracking.

### 5.2 Metadata Aggregation - KHÔNG có Cross-Video Matching

```951:964:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
def load_all_people_metadata(metadata_dir: Path | None = None) -> list[dict]:
    """Load all people metadata from all videos - CHỈ AGGREGATE ĐƠN GIẢN"""
    metadata_root = Path(metadata_dir or METADATA_DIR)
    if not metadata_root.exists():
        return []
    people: list[dict] = []
    for metadata_path in sorted(metadata_root.glob("*.json")):
        if metadata_path.name == QUEUE_STATE_PATH.name:
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        for person in payload.get("people", []):
            if isinstance(person, dict):
                person.setdefault("metadata_path", str(metadata_path))
                people.append(person)
    return people
```

**⚠️ VẤN ĐỀ:** Hàm này chỉ **load tất cả metadata**, không thực hiện:
- So sánh embedding giữa các video
- Gán global ID
- Re-identification người qua các camera

### 5.3 Không có Cross-Camera ReID Matching

Tìm kiếm "global", "cross_camera", "reid_merge" trong codebase:

```kotlin
# Không tìm thấy kết quả - Global tracking CHƯA IMPLEMENT
```

---

## 6. SƠ ĐỒ PIPELINE

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VIDEO ĐẦU VÀO (.h265/.hevc)                     │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    BƯỚC 1: YOLO26-X DETECTION                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Dòng 683-699: detections, det_confs = _detect_people_yolo26()  │   │
│  │ • class=0 (chỉ person)                                          │   │
│  │ • conf_threshold=0.32                                           │   │
│  │ • min_area=4,500 pixels                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 BƯỚC 2: CLIP-ReID EMBEDDING                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Dòng 701-713: reid_extractor.extract_batch(crops)              │   │
│  │ • 512-d embedding mỗi detection                                  │   │
│  │ • L2 normalized                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 BƯỚC 3: BYTETRACK ASSOCIATION                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Dòng 715-720: tracker.associate(detections, det_confs, ...)     │   │
│  │ • High-conf (≥0.35) → IOU ≥ 0.30                                │   │
│  │ • Low-conf (0.15-0.35) → IOU ≥ 0.25                            │   │
│  │ • max_age=8 giây                                                 │   │
│  │ • min_frames_to_confirm=4                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              BƯỚC 4: BUILD CANDIDATES TỪ TRACKS                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Dòng 726-783: candidate = {...}                                 │   │
│  │ • track_id = local ID (RESET mỗi video) ⚠️                       │   │
│  │ • avg_embedding = trung bình temporal của track embeddings      │   │
│  │ • content_frames = 12 sampled frames                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│             BƯỚC 5: VLM CAPTIONING (BLIP)                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Dòng 785-793:                                                   │   │
│  │ representative_crops = {candidate_id: crop_image}               │   │
│  │ vlm_engine.generate_captions_batch(representative_crops)         │   │
│  │                                                                 │   │
│  │ ⚠️ ĐẦU VÀO: Ảnh CROP NGƯỜI từ tracklets,                        │   │
│  │    KHÔNG phải video/detection frames thô                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              BƯỚC 6: ITSELF-LITE FEATURES                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Dòng 795-808:                                                   │   │
│  │ itself_engine.extract_features_batch(representative_crops)        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      METADATA OUTPUT (JSON)                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ {                                                               │   │
│  │   "video_id": "...",                                           │   │
│  │   "people": [                                                    │   │
│  │     {                                                            │   │
│  │       "track_id": "1",  ← LOCAL per-video ⚠️                    │   │
│  │       "embedding_vector": [...],                                 │   │
│  │       "person_caption": "Mô tả từ BLIP",                        │   │
│  │       "camera_id": "cam_a"                                      │   │
│  │     }                                                            │   │
│  │   ]                                                              │   │
│  │ }                                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────────┐
                    │     ❌ KHÔNG CÓ GLOBAL TRACKING   │
                    │  (Chỉ per-video, không có      │
                    │   cross-camera re-identification)│
                    └─────────────────────────────────┘
```

---

## 7. BẢNG TÓM TẮT CODE REFERENCE

| Câu hỏi | File | Dòng | Câu trả lời |
|---------|------|-------|------------|
| YOLO26 model | `hospital_pipeline.py` | 67-74 | `YOLO("yolo26x.pt")` |
| Hàm detection | `hospital_pipeline.py` | 76-94 | `_detect_people_yolo26()` |
| Class ByteTrack | `hospital_pipeline.py` | 100-222 | `class ByteTrackStyleTracker` |
| Khởi tạo tracker | `hospital_pipeline.py` | 659-669 | `tracker = ByteTrackStyleTracker(...)` |
| Gọi tracker | `hospital_pipeline.py` | 715-720 | `tracker.associate(...)` |
| VLM model | `vlm_engine.py` | 17, 28-29 | `Salesforce/blip-image-captioning-large` |
| **Input VLM** | `hospital_pipeline.py` | 785-793 | **Tracklet crops, KHÔNG phải video** |
| Build candidate | `hospital_pipeline.py` | 726-783 | Track → Candidate |
| Track ID reset | `hospital_pipeline.py` | 123 | `next_track_id = 1` mỗi video |
| Global track | N/A | N/A | **CHƯA IMPLEMENT** |

---

## 8. CÁC THÀNH PHẦN CÒN THIẾU (Global Tracking)

### 8.1 Các Component cần thiết cho Global Tracking

```python
# CHƯA IMPLEMENT - Cần thêm:
class GlobalTracker:
    """Merge tracks qua các video/camera sử dụng embedding similarity"""
    
    def __init__(self):
        self.global_embeddings_db = {}  # person_id → embedding
        self.global_id_counter = 1
    
    def merge_tracks(self, candidates: list[dict]) -> list[dict]:
        """
        Dòng: TBD (chưa implement)
        So sánh candidates với global DB
        Gán global IDs
        """
        pass
    
    def cross_camera_reid(self, track_a, track_b) -> float:
        """
        Dòng: TBD (chưa implement)
        Embedding similarity + temporal overlap check
        """
        pass
```

### 8.2 Những gì cần thêm

| Thành phần | Hiện tại | Cần cho Global |
|-----------|---------|----------------|
| `track_id` | Local per-video | Global unique ID |
| `embedding_vector` | Per-track avg | Đã có ✓ |
| Cross-video matching | ❌ Không có | Embedding similarity search |
| Temporal alignment | ❌ Không có | Camera timestamp sync |
| Merged track storage | ❌ Không có | Global track database |

---

## Kết luận

| Thành phần | Trạng thái |
|-----------|-----------|
| **Detection** | ✅ YOLO26-X đã implement |
| **Tracking** | ✅ ByteTrack-style đã implement |
| **VLM Input** | ✅ **Tracklet crops** (KHÔNG phải video/detection frames) |
| **Global Tracking** | ❌ **CHƯA implement** - chỉ per-video |

---

## Câu trả lời cho câu hỏi của bạn

### 1. Có model track chưa?
**✅ CÓ** - ByteTrackStyleTracker đã implement đầy đủ

### 2. VLM model nhận gì đầu vào?
**✅ Tracklet crops** (ảnh crop người từ bounding box của track)
- Không phải video thô
- Không phải detection frame
- Mà là **ảnh crop từ person bbox** của mỗi tracklet

### 3. Global track có xử lý gì không?
**❌ KHÔNG** - Hiện tại chỉ xử lý per-video:
- `track_id` reset cho mỗi video
- Không có cross-camera re-identification
- Không có global ID assignment
