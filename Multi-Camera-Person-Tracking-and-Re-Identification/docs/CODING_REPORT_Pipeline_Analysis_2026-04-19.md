# Engineering Code Report: Multi-Camera Person Tracking Architecture

**Generated:** 2026-04-19
**Project:** Multi-Camera Person Tracking and Re-Identification
**Focus:** Detection → Tracking → VLM Input → Global Tracking Pipeline

---

## Executive Summary

| Component | Status | Details |
|-----------|--------|---------|
| Detection Model | ✅ Implemented | YOLO26-X, NMS-free |
| Tracking Model | ✅ Implemented | ByteTrack-style Cascade Matching |
| VLM Input Source | ✅ Tracklet Crops | Person crops from confirmed tracks |
| Global Tracking | ❌ Not Implemented | Per-video only, no cross-camera |

---

## 1. DETECTION MODEL ✅

### 1.1 YOLO26-X Model Loading

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

### 1.2 Detection Function

```76:94:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
def _detect_people_yolo26(frame: np.ndarray, conf_threshold: float = 0.32) -> tuple[list[list[int]], list[float]]:
    """
    YOLO26-X detection - NMS-free end-to-end.
    Returns: (detections, confidences)
    """
    model = _load_yolo26()
    results = model(frame, verbose=False, conf=conf_threshold, classes=[0])  # class=0 is person

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

### 1.3 Detection Integration in Pipeline

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

## 2. TRACKING MODEL ✅

### 2.1 ByteTrackStyleTracker Class Definition

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

### 2.2 Cascade Association Logic

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

### 2.3 Tracker Initialization in Pipeline

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

### 2.4 Tracker Association Call

```715:720:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
        # ── STEP 3: ByteTrack Association ─────────────────────────
        assignments = tracker.associate(detections, det_confs, frame_idx, fps)
        for det_index, track_id in assignments.items():
            track = tracker.active_tracks.get(track_id)
            if track is not None:
                track.setdefault("embeddings", []).append(reid_embs[det_index])
```

---

## 3. VLM INPUT SOURCE ✅

### 3.1 VLM Engine Definition

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

### 3.2 Key Frame Extraction (NOT Full Video)

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

### 3.4 VLM Processing in Pipeline - **TRACKLET CROPS** (Not Video)

```785:813:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
    # ── STEP 5: VLM Captioning (BLIP) ───────────────────────────
    representative_frames = _read_frame_map(compressed_path, [int(candidate["frame_idx"]) for candidate in candidates])
    representative_crops: dict[str, Image.Image] = {}
    for candidate in candidates:
        crop = _crop_pil_from_frame(representative_frames.get(int(candidate["frame_idx"])), candidate["representative_bbox"])
        if crop is not None:
            representative_crops[candidate["candidate_id"]] = crop

    _apply_person_captions(candidates, compressed_path, vlm_engine, crop_map=representative_crops)
```

**🔑 KEY POINT:** VLM nhận **TRACKLET CROPS** - không phải video gốc hay detection frames.

### 3.5 Caption Application Function

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

## 4. CANDIDATE/TRACKLET BUILDING ✅

### 4.1 Building Candidates from Tracks

```726:783:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
    # ── STEP 4: Build Candidates from Tracks ─────────────────────
    candidates: list[dict] = []
    for track_id, track in sorted(tracker.all_tracks.items(), key=lambda x: int(x[0])):
        frames = track["frames"]
        if len(frames) < DEFAULT_MIN_TRACK_FRAMES:  # 5 frames minimum
            continue

        rep_frame = frames[-1]  # Last frame as representative
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

## 5. GLOBAL TRACKING ❌ (NOT IMPLEMENTED)

### 5.1 Per-Video Track ID Reset

**File:** `backend/legacy-engine/src_vlm/hospital_pipeline.py`

```123:125:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
        self.next_track_id = 1  # ⚠️ RESET per-video - no global ID
        self.active_tracks: dict[str, dict] = {}
        self.all_tracks: dict[str, dict] = {}
```

**⚠️ PROBLEM:** `track_id` bắt đầu từ 1 cho MỖI video, không có global tracking.

### 5.2 Metadata Aggregation - NO Cross-Video Matching

```951:964:Multi-Camera-Person-Tracking-and-Re-Identification/backend/legacy-engine/src_vlm/hospital_pipeline.py
def load_all_people_metadata(metadata_dir: Path | None = None) -> list[dict]:
    """Load all people metadata from all videos - SIMPLE AGGREGATION ONLY"""
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

**⚠️ PROBLEM:** Hàm này chỉ **load tất cả metadata**, không thực hiện:
- Cross-video embedding comparison
- Global ID assignment
- Person re-identification across cameras

### 5.3 No Cross-Camera ReID Matching

Search cho "global", "cross_camera", "reid_merge" trong codebase:

```kotlin
# No results found - Global tracking NOT implemented
```

---

## 6. PIPELINE FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VIDEO INPUT (.h265/.hevc)                        │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 1: YOLO26-X DETECTION                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Line 683-699: detections, det_confs = _detect_people_yolo26()   │   │
│  │ • class=0 (person only)                                         │   │
│  │ • conf_threshold=0.32                                            │   │
│  │ • min_area=4,500 pixels                                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 STEP 2: CLIP-ReID EMBEDDING                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Line 701-713: reid_extractor.extract_batch(crops)                │   │
│  │ • 512-d embedding per detection                                  │   │
│  │ • L2 normalized                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 STEP 3: BYTETRACK ASSOCIATION                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Line 715-720: tracker.associate(detections, det_confs, ...)     │   │
│  │ • High-conf (≥0.35) → IOU ≥ 0.30                                 │   │
│  │ • Low-conf (0.15-0.35) → IOU ≥ 0.25                             │   │
│  │ • max_age=8 seconds                                              │   │
│  │ • min_frames_to_confirm=4                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              STEP 4: BUILD CANDIDATES FROM TRACKS                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Line 726-783: candidate = {...}                                   │   │
│  │ • track_id = local ID (RESET per video) ⚠️                        │   │
│  │ • avg_embedding = temporal mean of track embeddings              │   │
│  │ • content_frames = 12 sampled frames                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│             STEP 5: VLM CAPTIONING (BLIP)                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Line 785-793:                                                   │   │
│  │ representative_crops = {candidate_id: crop_image}              │   │
│  │ vlm_engine.generate_captions_batch(representative_crops)         │   │
│  │                                                                 │   │
│  │ ⚠️ INPUT: Person CROPS from tracklets, NOT video/detections     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              STEP 6: ITSELF-LITE FEATURES                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Line 795-808:                                                   │   │
│  │ itself_engine.extract_features_batch(representative_crops)      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      METADATA OUTPUT (JSON)                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ {                                                               │   │
│  │   "video_id": "...",                                            │   │
│  │   "people": [                                                    │   │
│  │     {                                                            │   │
│  │       "track_id": "1",  ← LOCAL per-video ⚠️                    │   │
│  │       "embedding_vector": [...],                                 │   │
│  │       "person_caption": "BLIP description",                      │   │
│  │       "camera_id": "cam_a"                                       │   │
│  │     }                                                            │   │
│  │   ]                                                              │   │
│  │ }                                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────────┐
                    │     ❌ NO GLOBAL TRACKING ❌     │
                    │  (Per-video only, no cross-     │
                    │   camera re-identification)     │
                    └─────────────────────────────────┘
```

---

## 7. KEY CODE REFERENCE SUMMARY

| Question | File | Lines | Answer |
|----------|------|-------|--------|
| YOLO26 model | `hospital_pipeline.py` | 67-74 | `YOLO("yolo26x.pt")` |
| Detection function | `hospital_pipeline.py` | 76-94 | `_detect_people_yolo26()` |
| ByteTrack class | `hospital_pipeline.py` | 100-222 | `class ByteTrackStyleTracker` |
| Tracker init | `hospital_pipeline.py` | 659-669 | `tracker = ByteTrackStyleTracker(...)` |
| Tracker call | `hospital_pipeline.py` | 715-720 | `tracker.associate(...)` |
| VLM model | `vlm_engine.py` | 17, 28-29 | `Salesforce/blip-image-captioning-large` |
| **VLM input** | `hospital_pipeline.py` | 785-793 | **Tracklet crops, NOT video** |
| Candidate build | `hospital_pipeline.py` | 726-783 | Track → Candidate |
| Track ID reset | `hospital_pipeline.py` | 123 | `next_track_id = 1` per video |
| Global track | N/A | N/A | **NOT IMPLEMENTED** |

---

## 8. MISSING FEATURES (Global Tracking)

### 8.1 Required Components for Global Tracking

```python
# NOT IMPLEMENTED - Would need:
class GlobalTracker:
    """Merge tracks across videos/cameras using embedding similarity"""
    
    def __init__(self):
        self.global_embeddings_db = {}  # person_id → embedding
        self.global_id_counter = 1
    
    def merge_tracks(self, candidates: list[dict]) -> list[dict]:
        """
        Line: TBD (not implemented)
        Compare candidates with existing global DB
        Assign global IDs
        """
        pass
    
    def cross_camera_reid(self, track_a, track_b) -> float:
        """
        Line: TBD (not implemented)
        Embedding similarity + temporal overlap check
        """
        pass
```

### 8.2 What Would Be Needed

| Component | Current | Needed for Global |
|-----------|---------|-------------------|
| `track_id` | Local per-video | Global unique ID |
| `embedding_vector` | Per-track avg | Already available ✓ |
| Cross-video matching | ❌ None | Embedding similarity search |
| Temporal alignment | ❌ None | Camera timestamp sync |
| Merged track storage | ❌ None | Global track database |

---

## Conclusion

1. **Detection**: ✅ YOLO26-X implemented
2. **Tracking**: ✅ ByteTrack-style implemented  
3. **VLM Input**: ✅ **Tracklet crops** (NOT video/detection frames)
4. **Global Tracking**: ❌ **NOT implemented** - per-video only

**VLM processes individual person crops from confirmed tracks, not raw video or detection outputs.**
