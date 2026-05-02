# MCPT — Multi-Camera Person Tracking & Re-Identification

> Hệ thống tìm kiếm và truy vết người trong mạng lưới camera giám sát, sử dụng ngôn ngữ tự nhiên, AI embedding, và xác nhận của người vận hành.

---

## Giới thiệu

Hệ thống vận hành hàng chục camera giám sát liên tục. Khi cần tìm một người cụ thể, người vận hành thường phải scrub thủ công qua nhiều luồng video — rất tốn thời gian và dễ bỏ sót.

**MCPT** giải quyết bài toán này theo hướng AI-first:

1. Video từ 50 camera được xử lý offline: phát hiện người, tracking thành tracklet, trích xuất embedding ngoại hình và hành động.
2. Khi cần tìm kiếm, người vận hành nhập mô tả tự nhiên. Hệ thống trả về danh sách ứng viên có embedding phù hợp nhất.
3. Người vận hành chọn đúng candidate → hệ thống **Trace** tự động dựng lại hành trình liên camera trong ±12h.
4. Người vận hành xác nhận/từ chối từng đoạn → hệ thống học thêm và re-trace.

**Người dùng cuối:** Nhân viên an ninh, bảo vệ, hoặc bất kỳ người vận hành nào cần tìm người trong hệ thống camera.

---

## Mục lục

1. [Key Features](#1-key-features)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
0. [**Secret Management — Đọc trước khi làm gì khác**](#0-secret-management--đọc-trước-khi-làm-gì-khác)
3. [Tech Stack](#3-tech-stack)
4. [AI Models](#4-ai-models)
5. [Pipeline Offline Indexing](#5-pipeline-offline-indexing)
6. [Pipeline Online Search](#6-pipeline-online-search)
7. [Pipeline Trace](#7-pipeline-trace)
8. [Luồng lưu trữ](#8-luồng-lưu-trữ)
9. [Cấu trúc thư mục](#9-cấu-trúc-thư-mục)
10. [Cài đặt và chạy từ đầu](#10-cài-đặt-và-chạy-từ-đầu)
11. [Biến môi trường](#11-biến-môi-trường)
12. [Database Schema](#12-database-schema)
13. [API Documentation](#13-api-documentation)
14. [Camera Topology](#14-camera-topology)
15. [Troubleshooting](#15-troubleshooting)
16. [Security Notes](#16-security-notes)
17. [Performance Notes](#17-performance-notes)
18. [Limitations & Future Work](#18-limitations--future-work)
19. [Thành viên nhóm](#19-thành-viên-nhóm)

---

## 0. Secret Management — Đọc trước khi làm gì khác

### Cấu trúc mới — Single Source of Truth

```
secrets/                          ← GITIGNORED toàn bộ (trừ .example)
├── master.env                    ← ĐÂY LÀ FILE DUY NHẤT CẦN CHỈNH
├── master.env.example            ← Template (committed, không có real value)
├── oauth/
│   ├── oauth2_credentials.json  ← Google OAuth client credentials
│   └── oauth2_token.pickle      ← Google OAuth refresh token (generated)
└── shared.env                   ← Auto-generated bởi sync_secrets.sh, không edit
```

Tất cả file khác (`infra/env/backend.env`, `infra/env/ai.env`, `secrets/shared.env`) được **tự động generate** từ `secrets/master.env` bởi `scripts/sync_secrets.sh`.

### Khi chuyển sang máy mới

```bash
# Máy CŨ — export secrets thành 1 file
bash scripts/export_secrets.sh
# → mcpt_secrets_20260430.tar.gz

# Máy MỚI — import + generate env files
git clone <REPO_URL> && cd A20-App-119
bash scripts/import_secrets.sh /path/to/mcpt_secrets_20260430.tar.gz
bash scripts/sync_secrets.sh --vps
```

### Cập nhật một secret

```bash
nano secrets/master.env
# Sửa dòng cần thay đổi (ví dụ: LIGHTNING_API_BASE_URL)
bash scripts/sync_secrets.sh --vps
```

### Danh sách secrets cần điền

| Biến | Lấy ở đâu | Thay đổi khi nào |
|------|-----------|-----------------|
| `VPS_HOST` | IP VPS | Khi đổi VPS |
| `VPS_PASSWORD` | Provider VPS | Khi đổi mật khẩu |
| `POSTGRES_PASSWORD` | Tự đặt | Setup lần đầu |
| `JWT_SECRET_KEY` | Tự generate random | Setup lần đầu |
| `LIGHTNING_API_BASE_URL` | LightningAI UI → API Builder → Settings → URL | **Mỗi khi restart L4** |
| `LIGHTNING_API_TOKEN` | Tự đặt | Khi muốn đổi |
| `TRACKING_SERVICE_URL` | Giống `LIGHTNING_API_BASE_URL` | **Mỗi khi restart L4** |
| `GOOGLE_DRIVE_ROOT_FOLDER_ID` | Google Drive URL của folder root | Setup lần đầu |
| `GOOGLE_DRIVE_SOURCE_STORAGE_FOLDER_ID` | Google Drive URL của folder `Storage/` | Setup lần đầu |
| `NEXT_PUBLIC_API_GATEWAY_URL` | IP VPS | Khi đổi VPS |

> **Quan trọng:** Sau mỗi lần restart LightningAI API Builder, `LIGHTNING_API_BASE_URL` và `TRACKING_SERVICE_URL` **bắt buộc phải cập nhật**.

---

## 1. Key Features

| Tính năng | Mô tả |
|-----------|-------|
| **Natural language search** | Tìm người bằng mô tả văn bản tự do, không cần ảnh mẫu |
| **Offline video indexing** | Xử lý video theo lô từ Google Drive, lưu embedding vào PostgreSQL |
| **Multi-model AI pipeline** | RF-DETR detection → HeadBoxTracker → DINOv2 embedding → VideoMAE action |
| **Within-camera fragment merge** | DINOv2 cosine similarity tự động gom các tracklet của cùng 1 người trong 1 camera |
| **Hybrid scoring search** | Kết hợp vector similarity, attribute matching, và semantic overlap |
| **Cross-camera deduplication** | Tự động gom nhóm cùng người xuất hiện ở nhiều camera |
| **Trace journey** | Truy vết hành trình liên camera trong ±12h từ một seed candidate |
| **Camera topology pruning** | BFS trên đồ thị camera để loại 90% camera không liên quan |
| **Human-in-the-loop** | Người vận hành xác nhận → gallery update → re-trace tự động |
| **Google Drive ingestion** | Video lưu trên Drive, queue worker tự động polling và gửi lên GPU |
| **LightningAI GPU inference** | Toàn bộ model inference chạy trên L4 GPU cloud |
| **Dockerized VPS deployment** | Stack đầy đủ chạy trên VPS qua Docker Compose |
| **PostgreSQL metadata storage** | Tracklet, embedding, timeline, attribute metadata lưu có cấu trúc |

---

## 2. Kiến trúc hệ thống

### Sơ đồ tổng thể

```
┌──────────────────────────────────────────────┐
│  MÁY CÁ NHÂN (bất kỳ OS, không cần GPU)     │
│                                              │
│  python move.py            ← move video      │
│  Browser → http://<VPS_IP> ← dùng UI        │
└──────────────────┬───────────────────────────┘
                   │ Google Drive API (OAuth2)
                   ▼
┌──────────────────────────────────────────────┐
│  GOOGLE DRIVE                                │
│    Temp/    ← upload video mới vào đây       │
│    Storage/ ← move.py tổ chức theo cấu trúc │
│      cam01/2026-04-28/                        │
│        cam01_2026-04-28_10-00.mp4            │
└──────────────────┬───────────────────────────┘
                   │ Queue worker poll (mỗi 30s)
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  VPS — Docker Compose                                            │
│                                                                  │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────────┐  │
│  │ api-gateway │   │ metadata-service │   │   ai-service     │  │
│  │ FastAPI     │   │ FastAPI          │   │   FastAPI        │  │
│  │ port 8000   │   │ port 8001        │   │   port 8002      │  │
│  └──────┬──────┘   └────────┬─────────┘   └────────┬─────────┘  │
│         └──────────────────┬┘                       │            │
│                    ┌───────┘───────────────────────┘            │
│                    │  PostgreSQL :5432                           │
│                    └───────────────────────────────────────────  │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  Frontend (Next.js) port 3000                               │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────┬───────────────────────────────────────────┘
                       │ HTTP POST (Bearer token)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  LIGHTNINGAI — API Builder (NVIDIA L4, 24GB VRAM)                │
│                                                                  │
│  Tracking Service (FastAPI port 8000)                            │
│  ├── RF-DETR 2XLarge        ← person detection                  │
│  ├── HeadBoxTracker          ← per-video tracking + fragment merge│
│  ├── DINOv2 ViT-L/14        ← view-invariant Re-ID (1024-dim)   │
│  ├── SigLIP2 ViT-L-16-512  ← text/image encoding + attributes   │
│  └── VideoMAE Large         ← video action recognition          │
└──────────────────────────────────────────────────────────────────┘
```

### Luồng dữ liệu tổng quan

```
Upload video vào Drive Temp/
  → move.py tổ chức vào Drive Storage/
    → Queue worker (VPS) phát hiện video mới
      → Gửi POST request tới LightningAI Tracking Service
        → Download video từ Drive, chạy AI pipeline
          → Trả JSON (tracklets, embedding, metadata)
            → VPS lưu vào PostgreSQL
              → Frontend hiển thị trong queue
                → Người dùng search → trace → xác nhận
```

---

## 3. Tech Stack

### Frontend

| Công nghệ | Vai trò |
|-----------|---------|
| Next.js 14+ | Framework React SSR/SSG |
| TypeScript | Type safety |
| Tailwind CSS | Styling utility-first |

### Backend

| Dịch vụ | Công nghệ | Vai trò |
|---------|-----------|---------|
| API Gateway | FastAPI + Uvicorn | REST public, auth JWT, proxy |
| Metadata Service | FastAPI + SQLAlchemy 2.0 | Orchestration, queue, trace, DB |
| AI Service | FastAPI | Search proxy, ranking, gọi tracking upstream |
| Tracking Service | FastAPI (trên LightningAI) | Full AI inference pipeline |

### AI / Computer Vision

| Thư viện | Vai trò |
|----------|---------|
| `rfdetr` | Load RF-DETR 2XLarge, batch inference |
| `transformers` | Load DINOv2 ViT-L/14 và VideoMAE Large |
| `open_clip` | Load SigLIP2, text/image encode |
| `opencv-python` | Decode video H.265, frame sampling |
| `torch` + `torchvision` | Tensor ops, transforms |
| `numpy` | Vector math, cosine similarity |

### Database

| Công nghệ | Vai trò |
|-----------|---------|
| PostgreSQL 16 | Primary DB — metadata, embedding, tracklet |
| SQLAlchemy 2.0 | ORM — model mapping, query builder |
| JSONB columns | Lưu raw_metadata phức tạp linh hoạt |

### Infrastructure

| Công nghệ | Vai trò |
|-----------|---------|
| Docker Compose | Orchestrate tất cả dịch vụ VPS |
| LightningAI API Builder | Managed GPU service — auto-start |
| Google Drive API v3 | Lưu trữ video gốc |

---

## 4. AI Models

### RF-DETR 2XLarge — Person Detection

| Thuộc tính | Chi tiết |
|------------|---------|
| **Mục đích** | Phát hiện người trong từng frame đã sample |
| **Input** | Batch PIL images |
| **Output** | `xyxy` bbox, `confidence`, `class_id` (person = 1, COCO 1-indexed) |
| **Checkpoint** | `rf-detr-xxlarge.pth` (484MB, COCO pretrained) |
| **Confidence threshold** | 0.32 (detection), 0.40 (tracker high-conf), 0.45 (new track) |

### HeadBoxTracker — Per-video Tracking

| Thuộc tính | Chi tiết |
|------------|---------|
| **Mục đích** | Gán track ID cho từng người, duy trì identity xuyên suốt video |
| **Cơ chế** | 3-stage cascade: high-conf → active, low-conf → active, unmatched → buffer re-entry |
| **Matching** | Head-box IoU + center distance + capped velocity prediction |
| **Buffer** | Track sống đến hết video (probe duration trước khi xử lý → không expire mid-video) |
| **Key params** | `track_buffer=20` (5s), `max_buffer_frames=video_length`, `max_head_center_distance=120px` |
| **Sample rate** | 4fps (default) — tất cả tham số calibrated cho 4fps |

### DINOv2 ViT-L/14 — Appearance Re-ID Embedding

| Thuộc tính | Chi tiết |
|------------|---------|
| **Mục đích** | Tạo vector đặc trưng ngoại hình 1024-dim để so sánh cross-camera và within-camera fragment merge |
| **Input** | PIL crop ảnh người (bất kỳ kích thước, AutoImageProcessor resize) |
| **Output** | `appearance_embedding_vector` — 1024-dim L2-normalized CLS token |
| **Model** | `facebook/dinov2-large` (307M params, tự supervised trên 142M ảnh đa dạng) |
| **Ưu điểm vs TransReID** | Generalise cho overhead/angled cameras — không bị bias view side/front của MSMT17 |
| **Override** | `MCPT_DINOV2_MODEL_ID` env var (HF hub ID hoặc local path) |
| **Lưu DB** | `person_candidates.raw_metadata['appearance_embedding_vector']` |

### SigLIP2 ViT-L-16-512/webli — Text-Image Encoding

| Thuộc tính | Chi tiết |
|------------|---------|
| **Mục đích** | Encode text query và image crops vào cùng embedding space 1024-dim |
| **Output** | 1024-dim L2-normalized vector |
| **Vai trò** | Zero-shot attribute classification (gender, age, shirt, pants, shoes, bag, hat, hair, skin) + action semantic embedding + query encoding |
| **Lưu DB** | `raw_metadata['attribute_embedding_vector']`, `raw_metadata['action_semantic_embedding']` |

### VideoMAE Large — Video Action Recognition

| Thuộc tính | Chi tiết |
|------------|---------|
| **Mục đích** | Phân loại hành động của người trong clip 2s |
| **Input** | 16 frames × 224×224 |
| **Output** | CLS token 1024-dim; kết hợp với SigLIP2 text để chọn action label |
| **Checkpoint** | `storage/model-weights/videomae-action/model.safetensors` (1.3GB, Kinetics-400) |
| **Action vocabulary** | `standing_or_slow_motion`, `walking_motion`, `running_or_fast_motion`, `bending_or_sit_like_motion`, `fall_like_motion`, `carrying_object_like_motion` |

---

## 5. Pipeline Offline Indexing

### Stage 1 — Input Video

Download video từ Google Drive URL về local filesystem LightningAI.

### Stage 2 — Frame Decode + Sample 4fps

OpenCV decode, lấy 1 frame mỗi 0.25s. Tính Laplacian variance để đánh giá blur.

Output: `SampledFrame[]` — mỗi frame có `frame_index` (sampled index), `timestamp_second`, `image`, `laplacian_score`.

### Stage 3 — Person Detection

RF-DETR 2XLarge predict. Filter `class_id == 1` (person). Output `FrameDetection[]` per frame.

### Stage 4 — Tracking per Video

HeadBoxTracker 3-stage cascade:
- **Stage 1:** high-conf detections → active tracks (IoU + center distance + velocity)
- **Stage 2:** low-conf detections → unmatched active tracks
- **Stage 3:** unmatched high-conf → buffer tracks (re-entry, capped velocity prediction)
- Buffer không expire mid-video (max_buffer_frames = ceil(video_duration × 4fps) + 1)
- `finalize_all()` ở cuối video finalize tất cả active + buffer tracks

Output: `LocalTracklet[]` — mỗi tracklet có `track_id`, `observations[]`.

### Stage 5 — Tracklet Quality Scoring

Filter: `frame_count ≥ 9`, `duration ≥ 2s`, `frame_density ≥ 0.10`, `avg_laplacian ≥ 12.0`, `avg_confidence ≥ 0.30`.

### Stage 6 — Best Frame Selection

Rank frames theo `laplacian × 0.4 + bbox_area × 0.3 + confidence × 0.3`. Lấy top frames cho feature extraction.

### Stage 7 — Attribute & Appearance Extraction

**7a. Static Attribute — SigLIP2 Zero-shot**

`gender` (male/female), `age_group` (child/adult/elderly). Output: `attribute_embedding_vector` (1024-dim SigLIP2 text).

**7b. Appearance Attribute — SigLIP2 Zero-shot**

8 fields: `head_accessory`, `hat`, `hair_color`, `skin_tone`, `shirt`, `pants`, `shoes`, `bag`.

**7c. Appearance Embedding — DINOv2 ViT-L/14**

DINOv2 encode full-body crop → 1024-dim CLS token. Quality-weighted pool across selected frames.

Output: `appearance_embedding_vector` — 1024-dim L2-normalized.

> Thay thế TransReID + KPR part fusion. DINOv2 CLS token đã capture full-body appearance mà không cần part crops.

### Stage 8 — Action Recognition

VideoMAE Large (sliding window 2s, stride 1.5s) + SigLIP2 text matching → `action_summary`, `action_semantic_embedding`.

### Stage 9 — Within-camera Identity Resolution

Sau khi tất cả tracklets đã có DINOv2 embedding, gom các fragment của cùng 1 người trong cùng camera:

1. Temporal guard: 2 fragment không được overlap > 15% → nếu overlap nhiều = 2 người khác nhau
2. Embedding similarity: cosine(appearance_embedding_A, appearance_embedding_B) ≥ 0.85
3. Union-Find transitive closure (A≈B, B≈C → A,B,C cùng identity)
4. Assign `human_key` chung → `_merge_people_by_identity()` merge data

### Stage 10 — Feature Aggregation & Lưu DB

Gom toàn bộ output thành 1 JSON per unique identity. Lưu vào `person_candidates`.

**Output structure:**
```json
{
  "track_id": "...",
  "camera_id": "cam_01",
  "human_key": "cam_01:track_id",
  "merged_tracklet_count": 3,
  "attribute_embedding_vector": [1024 floats],
  "appearance_embedding_vector": [1024 floats],
  "action_semantic_embedding": {"embedding_vector": [1024 floats], "labels": ["walking_motion"]},
  "tracklet_quality": {"frame_count": 147, "duration_seconds": 36.75, "frame_density": 0.98},
  "timeline": [{"start_second": 0.0, "end_second": 36.75, "action_summary": "walking_motion"}]
}
```

---

## 6. Pipeline Online Search

### Hybrid Scoring

```
Score = 0.42 × cosine(query_vector, appearance_embed)   ← DINOv2 / SigLIP2 1024-dim
      + 0.20 × cosine(query_vector, action_embed)       ← SigLIP2 1024-dim
      + 0.18 × attribute_keyword_match
      + 0.12 × jaccard_token_overlap
      + 0.05 × visibility_score
      + 0.03 × world_position_bonus
```

> DINOv2 và SigLIP2 đều là 1024-dim — không còn dimension mismatch như kiến trúc cũ (TransReID 768-dim vs SigLIP2 1024-dim).

### Luồng tổng quan

```
User nhập query text
  → Phase 1: Hard filter DB theo camera_ids
  → Phase 2: Local prefilter Jaccard shortlist
  → Phase 3: Multi-modal query parsing (SigLIP2 encode)
  → Phase 4: Hard filter time window
  → Phase 5: Hybrid scoring mỗi candidate
  → Phase 6: Sort by score
  → Phase 7: Cross-camera deduplication (cosine ≥ 0.85)
  ← Top-k candidates
```

---

## 7. Pipeline Trace

### Stage 1 — Build Seed + Time Window

Lấy candidate từ DB → `seed_time` = `recorded_start` + trung điểm timeline → window ±12h.

### Stage 2 — Dynamic Topology Pruning

BFS tối đa 3 hops từ `seed_camera` qua `camera_topology.json`. Giảm 80–90% candidate pool.

### Stage 3 — Spatiotemporal Retrieval

Query DB: `camera_id IN (camera_scope)` + `abs_time IN window`. Filter: `cosine(seed_appearance, candidate_appearance) ≥ 0.40`.

### Stage 4 — Trajectory Path Search

```
segment_score = 0.55 × appearance_similarity  ← DINOv2 cosine
              + 0.25 × topology_plausibility
              + 0.20 × velocity_score
```

### Stage 5 — Re-ranking + Evidence Clips

Loại bỏ temporal overlap và topology impossible (`topology_score < 0.05`).

### Stage 6 — Human-in-the-loop Feedback

Confirmed segments → average embedding → expand window +20% → re-trace với embedding mới.

---

## 8. Luồng lưu trữ

### Google Drive

```
VinUni/
├── Temp/        ← Upload video mới vào đây
└── Storage/     ← Cấu trúc chuẩn sau move.py
    ├── cam01/
    │   └── 2026-04-28/
    │       └── cam01_2026-04-28_10-00.mp4
    └── cam02/
        └── 2026-04-28/
```

**Naming convention bắt buộc:** `cam{XX}_{YYYY-MM-DD}_{HH-mm}.mp4`

### LightningAI Storage

```
storage/
├── tracking-ingestion/
│   ├── sources/   ← Video download từ Drive
│   └── metadata/  ← JSON output
└── model-weights/
    ├── videomae-action/
    │   ├── model.safetensors  (1.3GB)
    │   └── config.json
    └── rf-detr/
        └── rf-detr-xxlarge.pth  (484MB)
```

> DINOv2 và SigLIP2 download tự động từ HuggingFace Hub lần đầu. Override bằng `MCPT_DINOV2_MODEL_ID` nếu cần dùng local cache.

---

## 9. Cấu trúc thư mục

```
A20-App-119/
├── backend/
│   ├── services/
│   │   ├── api-gateway/
│   │   ├── metadata-service/
│   │   └── tracking-service/          ← Chạy trên LightningAI L4
│   │       └── app/
│   │           ├── main.py
│   │           ├── local_ingestion_pipeline.py  ← RF-DETR + HeadBoxTracker + merge
│   │           ├── model_adapters.py            ← DINOv2ReIDHub, VideoMAEHub, SigLIP2ModelHub
│   │           ├── tracklet_feature_pipeline.py
│   │           ├── tracklet_memory_bank.py      ← Cross-camera Re-ID
│   │           ├── service.py
│   │           └── config.py
│   └── config/
│       ├── camera_topology.json
│       └── camera_calibration.json
├── ai_service/
├── frontend/
├── infra/
│   ├── docker-compose.yml
│   ├── env/
│   └── docker/
├── scripts/
├── secrets/                           ← GITIGNORED
├── move.py
└── ingest_local.py                    ← CLI để chạy ingestion local/debug
```

---

## 10. Cài đặt và chạy từ đầu

### Prerequisites

| Thành phần | Yêu cầu |
|------------|---------|
| Máy cá nhân | Python 3.11+, git |
| VPS | Ubuntu 22.04+, Docker 24+, Docker Compose v2, RAM ≥ 12GB |
| LightningAI | Account, API Builder enabled, GPU L4 |
| Google Drive | Google Cloud project với Drive API enabled, OAuth2 credentials |

### Bước 1 — Clone repo

```bash
git clone <REPO_URL> && cd A20-App-119
```

### Bước 2 — Cấu hình môi trường

```bash
cp secrets/master.env.example secrets/master.env
nano secrets/master.env
# Điền: LIGHTNING_API_BASE_URL, TRACKING_SERVICE_URL, DB credentials
```

### Bước 3 — Start LightningAI API Builder

1. Vào LightningAI Studio → API Builder → `tracking-service`
2. Machine: **1 × L4**
3. On start command: `bash A20-App-119/scripts/start_tracking_service_api_builder.sh`
4. Click **Start** | Bật **Auto start**

Verify:
```bash
curl -H "Authorization: Bearer <LIGHTNING_API_TOKEN>" \
  https://8000-<HASH>.cloudspaces.litng.ai/health
# Expected: {"status":"ok","service":"tracking-service"}
```

### Bước 4 — Deploy VPS Stack

```bash
bash infra/vps/check-secrets.sh .
bash infra/vps/deploy.sh .
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env ps
```

### Bước 5 — Upload và xử lý video

```bash
# Dry run
python3 move.py --dry-run

# Move thực: Temp/ → Storage/
python3 move.py
```

Queue worker tự phát hiện trong 30s. Hoặc trigger thủ công:
```bash
docker exec mcpt-backend python -m metadata_app.queue_worker --once
```

### Bước 6 — Sử dụng hệ thống

Truy cập frontend: `http://<VPS_IP>:3000`

---

## 11. Biến môi trường

### `infra/env/backend.env`

| Biến | Ví dụ | Bắt buộc |
|------|-------|----------|
| `POSTGRES_HOST` | `postgres` | ✅ |
| `POSTGRES_PASSWORD` | `<secret>` | ✅ |
| `POSTGRES_DATABASE` | `video_tracking` | ✅ |
| `JWT_SECRET_KEY` | `<random>` | ✅ |
| `TRACKING_SERVICE_URL` | `https://8000-<HASH>.cloudspaces.litng.ai` | ✅ |
| `LIGHTNING_API_TOKEN` | `<token>` | ✅ |
| `QUEUE_PARALLEL_JOBS` | `3` | — |
| `STORAGE_INGEST_SAMPLE_FPS` | `4` | — |
| `GOOGLE_DRIVE_SOURCE_STORAGE_FOLDER_ID` | `1G6L...` | ✅ |

### Tracking Service env vars (LightningAI)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `MCPT_DINOV2_MODEL_ID` | `facebook/dinov2-large` | DINOv2 model ID hoặc local path |
| `MCPT_REID_BATCH_SIZE` | `64` | Batch size cho DINOv2 inference |
| `MCPT_REID_PRECISION` | `fp16` | Precision cho DINOv2 |
| `MCPT_DETECTOR_BATCH_SIZE` | `8` | Batch size cho RF-DETR |
| `MCPT_EMBEDDING_BATCH_SIZE` | `128` | Batch size cho SigLIP2 |

---

## 12. Database Schema

### `queue_video_assets`

| Column | Type | Mô tả |
|--------|------|-------|
| `video_id` | VARCHAR | Tên file video — primary key |
| `camera_id` | VARCHAR | Camera ID (e.g. `cam_01`) |
| `raw_video_metadata` | JSONB | sample_fps, tracklet_count, sampled_frame_count |
| `created_at` | TIMESTAMP | Thời gian insert |

### `person_candidates`

| Column | Type | Mô tả |
|--------|------|-------|
| `id` | INTEGER | Auto-increment PK |
| `candidate_id` | VARCHAR | UUID của tracklet/merged identity |
| `camera_id` | VARCHAR | Camera ID |
| `track_id` | VARCHAR | Local track ID (per video) |
| `human_key` | VARCHAR | Resolved identity key (sau within-camera merge) |
| `raw_metadata` | JSONB | Toàn bộ payload: `appearance_embedding_vector` (1024-dim DINOv2), `attribute_embedding_vector` (1024-dim SigLIP2), `action_semantic_embedding`, `timeline`, `tracklet_quality`, `merged_tracklet_count` |

---

## 13. API Documentation

### POST `/api/v1/auth/login`
```json
// Request
{"username": "admin", "password": "<PASSWORD>"}
// Response
{"access_token": "eyJ...", "token_type": "bearer"}
```

### POST `/search`
```json
// Request
{
  "query": "người đàn ông áo xanh đeo balo",
  "top_k": 10,
  "camera_ids": ["cam07", "cam08"],
  "time_from": "2026-04-28T08:00:00",
  "time_to": "2026-04-28T12:00:00"
}
```

### POST `/api/v1/trace/run`
```json
// Request
{"candidate_id": "uuid", "window_hours": 12.0}
```

### POST `/api/v1/trace/feedback`
```json
{
  "candidate_id": "uuid-seed",
  "confirmed_segment_ids": ["uuid-1", "uuid-2"],
  "rejected_segment_ids": ["uuid-3"]
}
```

### GET `/health` *(LightningAI)*
```json
{"status": "ok", "service": "tracking-service"}
```

---

## 14. Camera Topology

File: `backend/config/camera_topology.json` — **50 cameras**, **56 edges**.

| Khu vực | Cameras |
|---------|---------|
| Cổng chính, Drop-off, Bãi xe | cam01–cam06 |
| Sảnh chính, Lễ tân | cam07–cam12 |
| Hành lang tầng 1, Khám bệnh | cam13–cam22 |
| Xét nghiệm, Chẩn đoán | cam23–cam30 |
| Thang máy / Thang bộ | cam31–cam32 |
| Tầng 2–5 | cam33–cam48 |
| Mái, Rooftop | cam49–cam50 |

Mỗi edge có `min_seconds`, `max_seconds`, `typical_seconds`, `confidence`. Bidirectional.

---

## 15. Troubleshooting

### LightningAI URL thay đổi sau restart
```bash
nano secrets/master.env
# Cập nhật LIGHTNING_API_BASE_URL và TRACKING_SERVICE_URL
bash scripts/sync_secrets.sh --vps
```

### DINOv2 download chậm lần đầu

DINOv2 ViT-L (~1.2GB) download từ HuggingFace Hub lần đầu. Để dùng local cache:
```bash
# Trong LightningAI DevBox
huggingface-cli download facebook/dinov2-large --local-dir /teamspace/studios/storage/model-weights/dinov2-large/
# Sau đó set env var
export MCPT_DINOV2_MODEL_ID=/teamspace/studios/storage/model-weights/dinov2-large/
```

### Search trả kết quả rỗng

1. Có data trong DB? `SELECT COUNT(*) FROM person_candidates;`
2. LightningAI còn chạy? `curl .../health`
3. Thử bỏ `camera_ids` và `time_from`/`time_to`

### Trace không tìm được hành trình

1. Chưa đủ video trong time window
2. Seed tracklet chất lượng thấp (blur, crop nhỏ)
3. Threshold quá cao — thử giảm `min_similarity=0.35`

### PostgreSQL connection failed
```bash
docker exec mcpt-postgres pg_isready -U mcpt_user -d video_tracking
docker restart mcpt-postgres
```

### Within-camera merge quá aggressive (ít người quá)

Tăng `reid_threshold` trong `_resolve_within_camera_identities` ([local_ingestion_pipeline.py](backend/services/tracking-service/app/local_ingestion_pipeline.py)):
```python
reid_threshold: float = 0.85  # tăng lên 0.90–0.95 nếu cần
```

---

## 16. Security Notes

- **Không commit** `secret/`, `secrets/` lên Git.
- **OAuth token** (`oauth2_token.pickle`) có quyền đọc/ghi toàn bộ Drive — bảo vệ như password.
- **LightningAI Bearer token** xác thực mọi request tới GPU service. Rotate định kỳ.
- **PostgreSQL** không expose ra ngoài Docker network.
- Khi debug với `curl`, dùng placeholder như `<JWT_TOKEN>`, không paste token thật vào log/chat.

---

## 17. Performance Notes

| Chỉ số | Giá trị |
|--------|---------|
| Throughput tracking | ~5–7 phút / video 10 phút trên L4 (DINOv2 lớn hơn TransReID) |
| Parallel jobs | 3 video song song (`QUEUE_PARALLEL_JOBS=3`) |
| Detector batch | 8–40 frames/pass tùy GPU |
| 50 cameras × 10min | ~8–10 giờ xử lý (1 L4) |
| Search latency | 5–30s (cold start ~60–120s) |
| DINOv2 batch | 64 crops/pass @ fp16 |
| Expected tracklets/camera (GT=25) | ~25–50 sau fragment merge |

**Bottleneck:**
- **DINOv2 inference:** Lớn hơn TransReID (~307M vs 86M params). Tăng `MCPT_REID_BATCH_SIZE` trên A100/H100.
- **Fragment merge:** O(n²) per camera — với n=500 tracklets/camera không đáng kể (<1s).
- **Cold start L4:** Auto start mất 60–120s sau idle — bật Auto start để minimize.

---

## 18. Limitations & Future Work

| Giới hạn | Mô tả |
|----------|-------|
| **Fragment merge threshold cần calibration** | `reid_threshold=0.85` là initial value — cần validate trên từng camera layout |
| **Trace dựa trên heuristic** | Greedy path search không đảm bảo optimal trajectory |
| **Topology tĩnh** | `camera_topology.json` cần update thủ công nếu layout thay đổi |
| **Video ingestion theo lô** | Không hỗ trợ real-time streaming |
| **Chưa có CI/CD** | Deploy thủ công qua SSH |
| **Privacy** | Chưa có anonymization (blur face) cho video export |
| **Single L4 GPU** | Scale horizontal chưa hỗ trợ |
| **Không có automated tests** | `TODO: thêm pytest cho pipeline` |

---

## 19. Thành viên nhóm

| Tên | Vai trò |
|-----|---------|
| Cao Diệu Ly | Leader, PM, AI Research |
| Dương Văn Hiệp | AI Engineer, Backend, Data Pipeline |
| Bùi Văn Đạt | Backend, Frontend |
