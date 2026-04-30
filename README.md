# MCPT — Multi-Camera Person Tracking & Re-Identification

> Hệ thống tìm kiếm và truy vết người trong mạng lưới camera giám sát bệnh viện, sử dụng ngôn ngữ tự nhiên, AI embedding, và xác nhận của người vận hành.

---

## Giới thiệu

Bệnh viện vận hành hàng chục camera giám sát liên tục. Khi cần tìm một bệnh nhân, người nhà, hoặc đối tượng cụ thể, người vận hành thường phải scrub thủ công qua nhiều luồng video — rất tốn thời gian và dễ bỏ sót.

**MCPT** giải quyết bài toán này theo hướng AI-first:

1. Video từ 50 camera được xử lý offline: phát hiện người, tracking thành tracklet, trích xuất embedding ngoại hình và hành động.
2. Khi cần tìm kiếm, người vận hành nhập mô tả tự nhiên. Hệ thống trả về danh sách ứng viên có embedding phù hợp nhất.
3. Người vận hành chọn đúng candidate → hệ thống **Trace** tự động dựng lại hành trình liên camera trong ±12h.
4. Người vận hành xác nhận/từ chối từng đoạn → hệ thống học thêm và re-trace.

**Người dùng cuối:** Nhân viên an ninh, bảo vệ, điều dưỡng, hoặc bất kỳ người vận hành nào cần tìm người trong hệ thống camera bệnh viện.

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

### Vấn đề cũ (đã fix)

Trước đây secrets bị rải rác ở nhiều nơi và không kiểm soát được:

| File | Loại | Trạng thái |
|------|------|-----------|
| `secret/shared.env` | LightningAI + Google Drive + DB | Không gitignore → **commit nhầm** |
| `secret/deploy/*.env` | VPS IP, password | Không gitignore → **commit nhầm** |
| `infra/env/backend.env` | Docker Compose | Có real password → **commit nhầm** |
| `infra/env/ai.env` | Docker Compose | Có real token → **commit nhầm** |
| `secrets/shared.env` | Docker mount | Gitignore đúng, nhưng không đồng bộ |

Khi đổi máy phải cập nhật 5–6 file, dễ miss, dễ inconsistent.

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

# Chuyển file đó sang máy mới (download từ LightningAI UI, USB, SCP...)

# Máy MỚI — import + generate env files
git clone <REPO_URL> && cd A20-App-119
bash scripts/import_secrets.sh /path/to/mcpt_secrets_20260430.tar.gz
bash scripts/sync_secrets.sh --vps
```

**Lưu ý trên Windows / Git Bash:** nếu file `.tar.gz` nằm trong đường dẫn có dấu cách, hãy bọc path trong dấu `"` hoặc dùng path tương đối. Ví dụ:

```bash
# Đang đứng trong repo
bash scripts/import_secrets.sh "./mcpt_secrets_20260430.tar.gz"

# Hoặc dùng path tuyệt đối có dấu cách
bash scripts/import_secrets.sh "D:\python ky 9\A20-App-119\mcpt_secrets_20260430.tar.gz"

# Trên Git Bash/MSYS cũng có thể dùng kiểu path Unix
bash scripts/import_secrets.sh "/d/python ky 9/A20-App-119/mcpt_secrets_20260430.tar.gz"
```

> **Xem thêm:** [QUICKSTART.md](QUICKSTART.md) để biết flow đầy đủ bước setup lần đầu.

### Cập nhật một secret (ví dụ: LightningAI URL thay đổi)

```bash
# Chỉ chỉnh 1 file
nano secrets/master.env
# Sửa dòng: LIGHTNING_API_BASE_URL=https://8000-<NEW_HASH>.cloudspaces.litng.ai
# Sửa dòng: TRACKING_SERVICE_URL=https://8000-<NEW_HASH>.cloudspaces.litng.ai

# Sync ra tất cả nơi + restart VPS services
bash scripts/sync_secrets.sh --vps

# Hoặc chỉ sync local (không động VPS)
bash scripts/sync_secrets.sh
```

### `sync_secrets.sh` làm gì

Script `scripts/sync_secrets.sh` đọc `secrets/master.env` và cập nhật:

| Target | Dùng bởi |
|--------|---------|
| `infra/env/backend.env` | `docker compose --env-file` khi chạy local/VPS |
| `infra/env/ai.env` | `docker compose --env-file` cho ai-service |
| `infra/env/frontend.env` | `docker compose --env-file` cho frontend |
| `secrets/shared.env` | Docker volume mount + `shared_secret_runtime.py` |
| VPS (với `--vps` flag) | SCP files + restart backend/ai_service |

### Danh sách tất cả secrets cần điền

Mở `secrets/master.env` (copy từ `secrets/master.env.example`) và điền:

| Biến | Lấy ở đâu | Thay đổi khi nào |
|------|-----------|-----------------|
| `VPS_HOST` | IP VPS | Khi đổi VPS |
| `VPS_PASSWORD` | Provider VPS | Khi đổi mật khẩu VPS |
| `POSTGRES_PASSWORD` | Tự đặt | Khi setup lần đầu |
| `JWT_SECRET_KEY` | Tự generate random | Khi setup lần đầu |
| `LIGHTNING_API_BASE_URL` | LightningAI UI → API Builder → Settings → URL | **Mỗi khi restart L4** |
| `LIGHTNING_API_TOKEN` | Tự đặt (Bearer token) | Khi muốn đổi |
| `TRACKING_SERVICE_URL` | Giống `LIGHTNING_API_BASE_URL` | **Mỗi khi restart L4** |
| `GOOGLE_DRIVE_ROOT_FOLDER_ID` | Google Drive URL của folder `VinUni/` | Khi setup lần đầu |
| `GOOGLE_DRIVE_VINUNI_FOLDER_ID` | Google Drive URL của folder `VinUni/` | Khi setup lần đầu |
| `GOOGLE_DRIVE_SOURCE_STORAGE_FOLDER_ID` | Google Drive URL của folder `Storage/` | Khi setup lần đầu |
| `NEXT_PUBLIC_API_GATEWAY_URL` | IP VPS | Khi đổi VPS |

> **Quan trọng:** Sau mỗi lần restart LightningAI API Builder, `LIGHTNING_API_BASE_URL` và `TRACKING_SERVICE_URL` **bắt buộc phải cập nhật**. Đây là thao tác thường xuyên nhất.

### Google Drive OAuth Token

OAuth token không nằm trong `master.env` — nó là file binary:

```bash
# Lần đầu tạo token (chạy 1 lần, cần browser):
python3 -c "
from google_auth_oauthlib.flow import InstalledAppFlow
flow = InstalledAppFlow.from_client_secrets_file(
    'secrets/oauth/oauth2_credentials.json',
    scopes=['https://www.googleapis.com/auth/drive']
)
creds = flow.run_local_server(port=0)
import pickle
with open('secrets/oauth/oauth2_token.pickle', 'wb') as f:
    pickle.dump(creds, f)
print('Token saved to secrets/oauth/oauth2_token.pickle')
"

# Khi chuyển máy: file này đã nằm trong secrets/ được copy sang rồi
# Nếu token hết hạn (thường sau vài tháng), xóa và chạy lại lệnh trên
```

---

## 1. Key Features

| Tính năng | Mô tả |
|-----------|-------|
| **Natural language search** | Tìm người bằng mô tả văn bản tự do, không cần ảnh mẫu |
| **Offline video indexing** | Xử lý video theo lô từ Google Drive, lưu embedding vào PostgreSQL |
| **Multi-model AI pipeline** | RF-DETR detection → OCMCTrack → TransReID embedding → VideoMAE action |
| **Hybrid scoring search** | Kết hợp vector similarity, attribute matching, và semantic overlap |
| **Cross-camera deduplication** | Tự động gom nhóm cùng người xuất hiện ở nhiều camera |
| **Trace journey** | Truy vết hành trình liên camera trong ±12h từ một seed candidate |
| **Camera topology pruning** | BFS trên đồ thị camera để loại 90% camera không liên quan trước khi search |
| **Human-in-the-loop** | Người vận hành xác nhận → gallery update → re-trace tự động |
| **Google Drive ingestion** | Video lưu trên Drive, queue worker tự động polling và gửi lên GPU |
| **LightningAI GPU inference** | Toàn bộ model inference chạy trên L4 GPU cloud, không cần GPU local |
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
│  VinUni/                                     │
│    Temp/    ← upload video mới vào đây       │
│    Storage/ ← move.py tổ chức theo cấu trúc │
│      cam01/2026-04-28/                        │
│        cam01_2026-04-28_10-00.mp4            │
└──────────────────┬───────────────────────────┘
                   │ Queue worker poll (mỗi 30s)
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  VPS — <VPS_IP> (Docker Compose)                                 │
│                                                                  │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────────┐  │
│  │ api-gateway │   │ metadata-service │   │   ai-service     │  │
│  │ FastAPI     │   │ FastAPI          │   │   FastAPI        │  │
│  │ port 8000   │   │ port 8001        │   │   port 8002      │  │
│  │ (public)    │   │ (internal)       │   │   (internal)     │  │
│  └──────┬──────┘   └────────┬─────────┘   └────────┬─────────┘  │
│         │                   │                       │            │
│  ┌──────┴───────────────────┴───────────────────────┘            │
│  │  PostgreSQL :5432                                             │
│  │  queue_video_assets | person_candidates | users              │
│  └───────────────────────────────────────────────────────────── │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  Frontend (Next.js) port 3000                               │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────┬───────────────────────────────────────────┘
                       │ HTTP POST (Bearer token)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  LIGHTNINGAI — API Builder (NVIDIA L4, 24GB VRAM)                │
│  URL: https://8000-<HASH>.cloudspaces.litng.ai                   │
│                                                                  │
│  Tracking Service (FastAPI port 8000)                            │
│  ├── RF-DETR 2XLarge        ← person detection                  │
│  ├── OCMCTrack cascade      ← per-video tracking                 │
│  ├── TransReID ViT-Base     ← appearance Re-ID embedding         │
│  ├── SigLIP2 ViT-L-16-512  ← text/image encoding + attributes   │
│  └── VideoMAE Large         ← video action recognition          │
└──────────────────────────────────────────────────────────────────┘
```

### Vai trò từng thành phần

| Thành phần | Vai trò |
|------------|---------|
| **Máy cá nhân** | Chạy `move.py` để tổ chức video trên Drive. Truy cập frontend qua browser. Không cần GPU, không cần cài backend. |
| **Google Drive** | Lưu trữ video gốc. Thư mục `Temp/` là điểm upload. Thư mục `Storage/` là nơi queue worker scan để lấy video mới. |
| **move.py** | Script chạy local. Đọc `Temp/`, parse tên file, tạo cấu trúc thư mục trong `Storage/`, rồi move file qua Drive API. |
| **VPS** | Máy chủ chạy toàn bộ stack Docker: gateway, metadata service, AI service proxy, frontend, PostgreSQL. |
| **API Gateway** | Cổng REST public duy nhất. Xử lý auth JWT, validate request, proxy tới metadata service và AI service. Không chứa logic model. |
| **Metadata Service** | Orchestration chính: quản lý queue worker, call LightningAI, lưu kết quả vào DB, chạy trace pipeline. |
| **AI Service** | Proxy nội bộ cho search AI. Nhận query từ gateway, call metadata service + LightningAI tracking, trả kết quả có rank. |
| **PostgreSQL** | Database lưu toàn bộ metadata: video, tracklet, embedding vector, attribute, timeline. |
| **Frontend** | Next.js web app. Người vận hành dùng để search, xem candidate, chọn seed, xem trace kết quả. |
| **LightningAI Tracking Service** | Chạy trên GPU L4. Nhận video URL, download từ Drive, chạy toàn bộ AI pipeline, trả JSON kết quả. Đây là nơi duy nhất inference chạy. |

### Luồng dữ liệu tổng quan

```
Upload video vào Drive Temp/
  → move.py tổ chức vào Drive Storage/
    → Queue worker (VPS) phát hiện video mới
      → Gửi POST request tới LightningAI Tracking Service
        → Download video từ Drive, chạy AI pipeline
          → Trả JSON (tracklets, embedding, metadata)
            → VPS lưu vào PostgreSQL (queue_video_assets + person_candidates)
              → Frontend hiển thị trong queue
                → Người dùng search → trace → xác nhận
```

---

## 3. Tech Stack

### Frontend

| Công nghệ | Vai trò | Lý do chọn |
|-----------|---------|------------|
| Next.js 14+ | Framework React SSR/SSG | Routing, server components, SEO |
| TypeScript | Type safety | Giảm runtime error, IDE support tốt |
| Tailwind CSS | Styling utility-first | Nhanh prototype, dễ maintain |

Frontend **không chứa logic AI**. Toàn bộ call đi qua API Gateway.

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
| `open_clip` | Load SigLIP2, text/image encode |
| `transformers` | Load VideoMAE Large |
| `timm` | Load TransReID ViT-Base backbone |
| `opencv-python` | Decode video H.265, frame sampling |
| `torch` + `torchvision` | Tensor ops, transforms |
| `numpy` | Vector math, cosine similarity |

### Database

| Công nghệ | Vai trò |
|-----------|---------|
| PostgreSQL 16 | Primary DB — metadata, embedding, tracklet |
| SQLAlchemy 2.0 | ORM — model mapping, query builder |
| JSONB columns | Lưu raw_metadata phức tạp (embedding, timeline) linh hoạt |

### Infrastructure

| Công nghệ | Vai trò |
|-----------|---------|
| Docker Compose | Orchestrate tất cả dịch vụ VPS trong một stack |
| LightningAI API Builder | Managed GPU service — auto-start, scale |
| `sshpass` + bash | Deploy script lên VPS |

### External Services

| Dịch vụ | Vai trò |
|---------|---------|
| Google Drive API v3 | Lưu trữ video gốc, tổ chức folder, download URL cho tracking |
| LightningAI Cloud | GPU L4 inference — không cần hardware riêng |

---

## 4. AI Models

### RF-DETR 2XLarge — Person Detection

| Thuộc tính | Chi tiết |
|------------|---------|
| **Mục đích** | Phát hiện người trong từng frame đã sample |
| **Input** | Batch PIL images (40 frames/batch trên L4) |
| **Output** | `xyxy` bbox, `confidence`, `class_id` (person = 1, COCO 1-indexed) |
| **Chạy ở đâu** | LightningAI L4 GPU |
| **Checkpoint** | `rf-detr-xxlarge.pth` (484MB, COCO pretrained) |
| **Lưu DB** | Không lưu trực tiếp — kết quả đi vào tracker |
| **Ảnh hưởng search/trace** | Detection quality quyết định tracklet quality. Miss detection = miss person. |

### OCMCTrack-style Corrective Cascade — Tracker

| Thuộc tính | Chi tiết |
|------------|---------|
| **Mục đích** | Gán ID cho từng người xuyên suốt video 10 phút |
| **Input** | `FrameDetection[]` theo từng frame từ RF-DETR |
| **Output** | `LocalTracklet[]` — mỗi tracklet có `track_id`, danh sách observations (bbox, confidence, timestamp) |
| **Chạy ở đâu** | LightningAI (pure Python + NumPy, không cần GPU riêng) |
| **Cơ chế** | 3-stage cascade: (1) high-confidence IoU match → (2) low-confidence recovery → (3) corrective buffer re-association |
| **Scope** | Per-video, độc lập — `track_id` reset theo từng video 10 phút |
| **Lưu DB** | Không trực tiếp — tracklet đi qua quality scoring trước |

### TransReID ViT-Base — Appearance Re-ID Embedding

| Thuộc tính | Chi tiết |
|------------|---------|
| **Mục đích** | Tạo vector đặc trưng ngoại hình 768-dim để so sánh cross-camera |
| **Input** | PIL crop ảnh người (256×128), theo từng frame được chọn |
| **Output** | `appearance_embedding_vector` — 768-dim L2-normalized |
| **Chạy ở đâu** | LightningAI L4 GPU |
| **Checkpoint** | `storage/model-weights/transreid-reid/transformer_120.pth` (330MB, MSMT17, 4101 IDs) |
| **Part fusion** | KPR-style: `global × 0.55 + mean(upper_half, lower_half) × 0.45` |
| **Lưu DB** | `person_candidates.raw_metadata['appearance_embedding_vector']` |
| **Ảnh hưởng search** | Dùng để cosine similarity với query vector trong trace stage 3. Score weight: 0.42 trong hybrid scoring. |
| **Ảnh hưởng trace** | Seed embedding được so sánh với toàn bộ candidates trong window ±12h. Threshold cosine ≥ 0.40 để lọc. |

### SigLIP2 ViT-L-16-512/webli — Text-Image Encoding

| Thuộc tính | Chi tiết |
|------------|---------|
| **Mục đích** | Encode text query và image crops vào cùng embedding space 1024-dim |
| **Input** | Text string hoặc PIL image |
| **Output** | 1024-dim L2-normalized vector |
| **Chạy ở đâu** | LightningAI L4 GPU |
| **Pretrained** | WebLI, Feb 2025 (SOTA tại thời điểm release) |
| **Vai trò trong pipeline** | (1) Zero-shot attribute classification: gender, age_group, shirt, pants, shoes, bag, hat, hair_color, skin_tone (2) Encode action summary thành `action_semantic_embedding` (1024-dim) (3) Encode query text để hybrid scoring trong search |
| **Lưu DB** | `raw_metadata['attribute_embedding_vector']`, `raw_metadata['action_semantic_embedding']` |
| **Ảnh hưởng search** | Query encode bằng SigLIP2 → cosine similarity với action embedding (weight 0.20) |

### VideoMAE Large — Video Action Recognition

| Thuộc tính | Chi tiết |
|------------|---------|
| **Mục đích** | Phân loại hành động của người trong clip 2s |
| **Input** | 16 frames (resample từ clip bất kỳ độ dài) × 3 channels × 224×224 |
| **Output** | CLS token 1024-dim; kết hợp với SigLIP2 text để chọn action label |
| **Chạy ở đâu** | LightningAI L4 GPU |
| **Checkpoint** | `storage/model-weights/videomae-action/model.safetensors` (1.3GB, Kinetics-400) |
| **Action vocabulary** | `standing_or_slow_motion`, `walking_motion`, `running_or_fast_motion`, `bending_or_sit_like_motion`, `fall_like_motion`, `carrying_object_like_motion` |
| **Lưu DB** | `raw_metadata['action_semantic_embedding']` — gồm vector, labels, confidence |
| **Ảnh hưởng search** | Action embedding tham gia hybrid scoring khi query có mô tả hành động |

---

## 5. Pipeline Offline Indexing

Pipeline này chạy tự động trên LightningAI mỗi khi queue worker gửi video mới.

### Stage 1 — Input Video

| | |
|---|---|
| **Input** | URL video trên Google Drive |
| **Processing** | Tracking service download về local (`INGESTION_WORK_ROOT`) |
| **Output** | File `.mp4` H.265 (15–30fps, độ dài thường 10 phút) |
| **Module** | `ingestion_runtime.py` → `_download_public_url()` |

### Stage 2 — Frame Decode + Sample 5fps

| | |
|---|---|
| **Input** | File `.mp4` local |
| **Processing** | OpenCV decode, lấy 1 frame mỗi `1/5` giây. Tính Laplacian variance để đánh giá blur. |
| **Output** | `SampledFrame[]` — mỗi frame có `frame_index`, `timestamp_second`, `image`, `laplacian_score` |
| **Module** | `VideoFrameSampler.sample()` trong `local_ingestion_pipeline.py` |

> **Lý do sample 5fps:** Giảm computing cost. Với video 10 phút, 5fps = 3000 frames thay vì 18000 frames ở 30fps.

### Stage 3 — Person Detection

| | |
|---|---|
| **Input** | Batch frames (batch_size = 40 trên L4) |
| **Processing** | RF-DETR 2XLarge predict. Filter `class_id == 1` (person). |
| **Output** | `FrameDetection[]` per frame — `bbox`, `confidence`, `crop_bgr` |
| **Module** | `RFDETRPersonDetector.detect()` |

### Stage 4 — Tracking per Video

| | |
|---|---|
| **Input** | `FrameDetection[]` theo chronological order |
| **Processing** | OCMCTrack 3-stage cascade matching. Tạo track ID mới khi detection không match. |
| **Output** | `LocalTracklet[]` — mỗi tracklet: `video_id`, `track_id`, `observations[]` (bbox + confidence + timestamp) |
| **Module** | `OCMCTrackStyleTracker.track()` |

> **Lưu ý:** `track_id` chỉ có ý nghĩa trong phạm vi 1 video. Không phải global ID.

### Stage 5 — Tracklet Quality Scoring

| | |
|---|---|
| **Input** | `LocalTracklet[]` |
| **Processing** | Filter theo ngưỡng: confidence ≥ 0.3, duration ≥ 2s, laplacian đủ (không mờ quá) |
| **Output** | `accepted_tracklets[]` + `rejected_tracklets[]` |
| **Module** | `TrackletQualityScorer.score()` |

### Stage 6 — Best Frame Selection

| | |
|---|---|
| **Input** | Tracklet đã pass quality scoring |
| **Processing** | Rank tất cả frames theo: `laplacian × 0.4 + bbox_area × 0.3 + confidence × 0.3`. Lấy top 5–10 frames. |
| **Output** | `FrameSelectionOutput` — selected frames + pooling scores (Mean, Max, Quality-Weighted Mean) |
| **Module** | `HybridFrameSelector.select()` |

### Stage 7 — Attribute & Appearance Extraction (song song)

**7a. Static Attribute — SigLIP2 Zero-shot**

| | |
|---|---|
| **Input** | Top selected frames (PIL crops) |
| **Processing** | SigLIP2 zero-shot: image vs text prompts. Quality-weighted majority vote. |
| **Output** | `gender` (male/female), `age_group` (child/adult/elderly) |
| **Embedding** | `attribute_embedding_vector` — SigLIP2 text encode `"a photo of a {gender} {age_group}"` → 1024-dim |

**7b. Appearance Attribute — SigLIP2 Zero-shot**

| | |
|---|---|
| **Input** | Top selected frames (PIL crops) |
| **Processing** | SigLIP2 zero-shot cho từng attribute. Quality-weighted majority vote. |
| **Output** | `head_accessory`, `hat`, `hair_color`, `skin_tone`, `shirt`, `pants`, `shoes`, `bag` |

**7c. Appearance Embedding — TransReID KPR-part**

| | |
|---|---|
| **Input** | Top selected frames (PIL crops 256×128) |
| **Processing** | Tách full/upper/lower. TransReID encode từng phần. Fusion: `global × 0.55 + mean(upper, lower) × 0.45`. Quality-weighted pool. |
| **Output** | `appearance_embedding_vector` — 768-dim L2-normalized |

### Stage 8 — Action Recognition (tracklet duration ≥ 2s)

**8a. Clip Builder**

| | |
|---|---|
| **Input** | Tracklet observations (frames) |
| **Processing** | Sliding window 2s, stride 1.5s |
| **Output** | `ActionClip[]` — `clip_id`, `start_second`, `end_second`, `frames[]` |

**8b. VideoMAE + SigLIP2 Classification**

| | |
|---|---|
| **Input** | Frames của từng ActionClip |
| **Processing** | Resample 16 frames → VideoMAE Large → CLS token 1024-dim. Cosine similarity với SigLIP2 text của từng action label. |
| **Output** | `action_summary` (label), `confidence`, `action_semantic_embedding` (1024-dim vector) |

### Stage 9 — Feature Aggregation

| | |
|---|---|
| **Input** | Toàn bộ output từ stage 6–8 |
| **Processing** | Gom lại thành 1 dict per tracklet |
| **Output** | JSON payload per tracklet với đầy đủ fields |

**Output structure mỗi tracklet:**
```json
{
  "track_id": "...",
  "video_id": "cam01_2026-04-28_10-00.mp4",
  "attribute_summary": "male adult",
  "appearance_summary": "blue shirt, black pants, backpack",
  "semantic_attributes": ["male", "adult", "shirt:blue", "bag:present"],
  "attribute_embedding_vector": [1024 floats],
  "appearance_embedding_vector": [768 floats],
  "action_semantic_embedding": {
    "embedding_vector": [1024 floats],
    "labels": ["walking_motion"],
    "confidence": 0.82
  },
  "timeline": [
    {"start_second": 0.0, "end_second": 2.0, "action_summary": "walking_motion"}
  ]
}
```

### Stage 10 — Lưu vào PostgreSQL

| | |
|---|---|
| **Input** | JSON response từ LightningAI |
| **Processing** | `_append_processed_result()` — upsert video asset, insert person candidates |
| **Output** | `queue_video_assets` + `person_candidates` rows trong PostgreSQL |
| **Module** | `queue_runtime.py` (VPS Metadata Service) |
| **Timing** | Incremental — mỗi video được lưu ngay khi xong, không đợi cả batch |

---

## 6. Pipeline Online Search

### Luồng tổng quan

```
User nhập query text trên Frontend
  → POST /search (API Gateway)
    → POST /internal/search (AI Service)
      → rank_candidates() (Metadata Service)
        → Phase 1: Hard filter DB theo camera_ids (nếu có)
        → Phase 2: Local prefilter Jaccard similarity → shortlist
        → POST /api/v1/candidates/search (LightningAI)
          → Phase 3: Parse query multi-modal
          → Phase 4: Hard filter time window
          → Phase 5: Hybrid scoring mỗi candidate
          → Phase 6: Sort by score
          → Phase 7: Cross-camera deduplication
      ← Top-k candidates với score, preview_url, metadata
    ← Formatted response
  ← Hiển thị danh sách candidates trên Frontend
```

### Phase 1 — Hard Filter (DB Level)

Nếu request có `camera_ids`, query SQL giới hạn ngay: `WHERE camera_id IN (...)`. Giảm dataset cần search từ hàng chục nghìn xuống vài nghìn tracklet.

### Phase 2 — Local Prefilter (Metadata Service)

Jaccard token similarity giữa query text và `search_text` của candidate. Shortlist top-N để gửi lên LightningAI (tránh gửi hết toàn bộ DB).

### Phase 3 — Multi-modal Query Parsing (LightningAI)

| Sub-task | Cơ chế |
|----------|--------|
| Attribute parsing | Regex extract: gender (male/female/man/woman), màu sắc, quần áo, túi, mũ |
| Action parsing | Detect keywords: chạy, đứng, mang, walking, running... |
| Query encoding | SigLIP2 text encode → 1024-dim `query_vector` |

### Phase 4 — Hard Filter Time Window

Nếu `time_from`/`time_to` có trong request, lọc candidates theo `abs_time` của tracklet (= `recorded_start` + `timeline.start_second`).

### Phase 5 — Hybrid Scoring

```
Score = 0.42 × cosine(query_vector, appearance_embed)   ← TransReID space
      + 0.20 × cosine(query_vector, action_embed)       ← SigLIP2 space
      + 0.18 × attribute_keyword_match                  ← metadata match
      + 0.12 × jaccard_token_overlap                    ← lexical overlap
      + 0.05 × visibility_score                         ← quality bonus
      + 0.03 × world_position_bonus                     ← spatial context
```

> **Lưu ý:** `appearance_embed` (768-dim TransReID) và `query_vector` (1024-dim SigLIP2) khác dim → cosine similarity hiện tính qua `attribute_embedding_vector` (1024-dim SigLIP2), không phải trực tiếp TransReID vector. Xem `_rank_candidate_multimodal()`.

### Phase 6 — Sort by Score

Candidates được sort descending theo `score`.

### Phase 7 — Cross-camera Deduplication

- Cluster candidates có cosine similarity ≥ 0.85 (cùng embedding space)
- Mỗi cluster → 1 representative (người có score cao nhất)
- Giới hạn per-camera: `top_k / 5 + 1` candidates mỗi camera
- Mục đích: tránh 1 người chiếm hết Top-k

### Input mẫu

```json
{
  "query": "người đàn ông áo xanh đeo balo",
  "top_k": 10,
  "offset": 0,
  "camera_ids": ["cam07", "cam08"],
  "time_from": "2026-04-28T08:00:00",
  "time_to": "2026-04-28T12:00:00"
}
```

### Output mẫu

```json
{
  "results": [
    {
      "id": "uuid-candidate",
      "thumbnail_url": "/api/v1/candidates/uuid/preview",
      "description": "male adult · blue shirt · bag:present · cam08 · 10:23"
    }
  ]
}
```

---

## 7. Pipeline Trace

Trace được kích hoạt sau khi người vận hành **chọn đúng candidate** từ kết quả search.

### Stage 1 — Build Seed + Time Window

```python
POST /api/v1/trace/run
{
  "candidate_id": "uuid-of-selected-candidate",
  "window_hours": 12.0  # ±12h từ thời điểm xuất hiện của seed
}
```

- Lấy candidate từ DB → embedding, `camera_id`, `recorded_start`, `timeline`
- `seed_time` = `recorded_start` + trung điểm timeline
- `window_from` = `seed_time` − 12h
- `window_to` = `seed_time` + 12h

### Stage 2 — Dynamic Topology Pruning

- Dùng `camera_topology.json` — đồ thị 50 camera, 56 edges với `min_seconds`/`max_seconds` đi lại giữa các camera
- BFS tối đa 3 hops từ `seed_camera`
- Kết quả: tập `camera_scope` — chỉ các camera có thể người đó đi qua
- Camera không có trong topology (unknown layout) được giữ lại với độ ưu tiên thấp

> **Lý do:** 50 cameras × 24h = lượng dữ liệu rất lớn. Topology pruning giảm 80–90% candidate pool trước khi search.

### Stage 3 — Spatiotemporal Retrieval

- Query DB: `camera_id IN (camera_scope)` + `abs_time IN [window_from, window_to]`
- Filter: `cosine(seed_appearance, candidate_appearance) ≥ 0.40`
- Enrich kết quả với camera metadata từ topology (tên khu vực, tầng, loại zone)

> **Spatiotemporal = không gian + thời gian.** Không chỉ tìm người trông giống — còn phải xuất hiện đúng chỗ, đúng lúc.

### Stage 4 — Trajectory Path Search

- Sort candidates theo `abs_start` (chronological)
- Greedy chain: từ seed_camera → chọn candidate tiếp theo gần nhất theo thời gian và topology

Score mỗi segment:
```
segment_score = 0.55 × appearance_similarity  # TransReID cosine
              + 0.25 × topology_plausibility  # thời gian đi lại hợp lý không?
              + 0.20 × velocity_score         # elapsed_time ≥ 2s → valid
```

**topology_plausibility:**
- `elapsed < min_seconds` → 0.0 (physically impossible)
- `min_seconds ≤ elapsed ≤ max_seconds` → 1.0 (trong cửa sổ hợp lý)
- `elapsed > max_seconds` → decay theo exponential

### Stage 5 — Re-ranking + Build Evidence Clips

**Loại bỏ:**
- Temporal overlap: cùng người xuất hiện 2 camera cùng lúc
- Topology impossible: `topology_score < 0.05`

**Evidence Clip structure:**
```json
{
  "camera_id": "cam14",
  "camera_name": "Hành lang tầng 1 đoạn 1",
  "floor": 1,
  "start_time": "2026-04-28T08:31:00Z",
  "end_time": "2026-04-28T08:35:00Z",
  "duration_seconds": 240.0,
  "appearance_sim": 0.87,
  "segment_score": 0.79,
  "preview_url": "/api/v1/candidates/uuid/preview",
  "video_url": "/api/v1/queue/videos/cam14_.../file"
}
```

**Output mẫu:**
```
Đối tượng vào Cổng chính (cam01, tầng 1) lúc 08:12
  → Sảnh chính (cam07, tầng 1) lúc 08:25  [13 phút, sim=0.91]
  → Hành lang tầng 1 (cam14, tầng 1) lúc 08:31  [6 phút, sim=0.88]
  → Khu chờ khám (cam19, tầng 1) lúc 08:45  [14 phút, sim=0.85]
```

### Stage 6 — Human-in-the-loop Feedback

```python
POST /api/v1/trace/feedback
{
  "candidate_id": "uuid-seed",
  "confirmed_segment_ids": ["uuid-cam07", "uuid-cam14"],
  "rejected_segment_ids": ["uuid-cam19"]
}
```

**Gallery Update + Query Refinement:**
1. Average embedding của confirmed segments + seed → embedding mới giàu thông tin hơn (nhiều góc, ánh sáng khác nhau)
2. Expand window +20% (có thể có khoảng trống trước/sau)
3. Re-run trace với seed embedding đã cập nhật → lấp đầy khoảng trống

---

## 8. Luồng lưu trữ

### Google Drive

```
VinUni/
├── Temp/                           ← Upload video mới vào đây (bất kỳ thứ tự)
│   ├── cam01_2026-04-28_10-00.mp4
│   ├── cam02_2026-04-28_10-00.mp4
│   └── ...
│
└── Storage/                        ← Cấu trúc chuẩn sau move.py
    ├── cam01/
    │   ├── 2026-04-28/
    │   │   ├── cam01_2026-04-28_10-00.mp4
    │   │   └── cam01_2026-04-28_10-10.mp4
    │   └── 2026-04-29/
    │       └── cam01_2026-04-29_08-00.mp4
    ├── cam02/
    │   └── 2026-04-28/
    └── ...
```

**Naming convention bắt buộc:** `cam{XX}_{YYYY-MM-DD}_{HH-mm}.mp4`

Ví dụ: `cam01_2026-04-28_10-00.mp4` — Camera 01, ngày 28/4/2026, lúc 10:00.

### move.py

```bash
python3 move.py --dry-run   # Xem trước danh sách sẽ được move
python3 move.py             # Thực thi move Temp/ → Storage/
```

Yêu cầu: `secrets/oauth/oauth2_token.pickle` đã có.

Script sẽ:
1. List tất cả `.mp4` trong `Temp/`
2. Parse tên file theo pattern
3. `find_or_create_folder()` tạo `cam_XX/YYYY-MM-DD/` nếu chưa có
4. `move_file()` qua Drive API (`addParents/removeParents`) — không copy, không download

### Queue Worker (VPS)

```
Queue Worker poll Google Drive Storage/ mỗi 30 giây
  → Scan toàn bộ cam_XX/YYYY-MM-DD/*.mp4
  → Skip file đã có trong ProcessedStorage registry
  → Batch gửi lên LightningAI (QUEUE_PARALLEL_JOBS=3)
  → Lưu kết quả incremental vào PostgreSQL sau mỗi video
```

Trigger thủ công:
```bash
docker exec mcpt-backend python -m metadata_app.queue_worker --once
```

### LightningAI Storage (`/teamspace/studios/`)

Filesystem này được chia sẻ giữa LightningAI DevBox và API Builder (L4 machine).

```
storage/
├── tracking-ingestion/
│   ├── sources/      ← Video download từ Drive (reused khi INGESTION_REUSE_DOWNLOADED_MP4=true)
│   └── metadata/     ← JSON output kèm theo response
└── model-weights/
    ├── transreid-reid/
    │   └── transformer_120.pth        (330MB)
    ├── videomae-action/
    │   ├── model.safetensors          (1.3GB)
    │   └── config.json
    ├── osnet-reid/                    (16MB, backup reference)
    └── siglip2-person-reid/          (1.4GB, reference)
```

### VPS Storage

```
/workspace/storage/
├── queue/local/Storage/
│   ├── Metadata/           ← JSON metadata buffer
│   └── ProcessedStorage/   ← Registry fingerprint files (idempotency)
└── videos/
```

---

## 9. Cấu trúc thư mục

```
A20-App-119/
├── backend/
│   ├── services/
│   │   ├── api-gateway/               ← Public REST API (port 8000)
│   │   │   └── app/
│   │   │       ├── main.py            ← Endpoints: /search, /trace, /auth...
│   │   │       ├── config.py          ← METADATA_SERVICE_URL, AI_SERVICE_URL
│   │   │       └── services/
│   │   │           └── ai_client.py   ← HTTP client gọi ai-service
│   │   │
│   │   ├── metadata-service/          ← DB, queue, trace orchestration
│   │   │   └── app/
│   │   │       ├── main.py            ← FastAPI app + tất cả internal endpoints
│   │   │       ├── models.py          ← SQLAlchemy: QueueVideoAsset, PersonCandidate, User
│   │   │       ├── service.py         ← rank_candidates(), search logic
│   │   │       ├── queue_runtime.py   ← Queue worker: Drive scan, batch send to L4
│   │   │       ├── trace_service.py   ← Trace pipeline: topology + spatiotemporal
│   │   │       ├── drive_storage_ingest.py ← DriveStorageVideoScanner
│   │   │       ├── post_move_ingestion.py  ← Ingestion contract/policy
│   │   │       ├── database.py        ← SQLAlchemy engine, session
│   │   │       └── config.py          ← Env vars, settings
│   │   │
│   │   └── tracking-service/          ← Chạy trên LightningAI L4
│   │       ├── start_api_builder.sh   ← LightningAI start command
│   │       └── app/
│   │           ├── main.py            ← FastAPI: /ingestion/process, /candidates/search
│   │           ├── ingestion_runtime.py    ← Download + orchestrate pipeline
│   │           ├── local_ingestion_pipeline.py ← RF-DETR + OCMCTrack + Quality
│   │           ├── tracklet_feature_pipeline.py ← Feature extraction protocols
│   │           ├── model_adapters.py   ← TransReIDHub, VideoMAEHub, SigLIP2ModelHub
│   │           ├── service.py          ← search_candidates_remote() — hybrid scoring
│   │           ├── hardware_runtime.py ← GPU auto-detect, batch size tuning
│   │           ├── execution_plan.py   ← Resolve parallelism plan
│   │           └── schemas.py          ← Pydantic request/response schemas
│   │
│   ├── config/
│   │   └── camera_topology.json       ← Hospital camera graph (50 cams, 56 edges)
│   │
│
├── ai_service/
│   └── aiapp/
│       ├── main.py                    ← /internal/search + proxy tracking endpoints
│       ├── tracking_upstream.py       ← HTTP proxy helpers (proxy_post_json...)
│       ├── config.py                  ← TRACKING_SERVICE_URL, token
│       └── runtime_contract.py        ← Build tracking runtime contract response
│
├── frontend/                          ← Next.js web application
│   ├── app/                           ← App router pages
│   ├── components/                    ← UI components (search, trace, candidate...)
│   ├── features/                      ← Feature-level logic (SearchContext...)
│   └── lib/                           ← API clients, constants, types
│
├── infra/
│   ├── docker-compose.yml             ← VPS stack definition
│   ├── env/
│   │   ├── backend.env                ← Database, LightningAI URL, queue config
│   │   ├── ai.env                     ← AI service config
│   │   └── frontend.env               ← Next.js env
│   ├── docker/
│   │   ├── Dockerfile.backend
│   │   ├── Dockerfile.ai_service
│   │   └── backend_entrypoint.sh
│   └── vps/
│       ├── deploy.sh                  ← Build + up stack trên VPS
│       └── check-secrets.sh           ← Validate secrets trước deploy
│
├── scripts/
│   ├── start_tracking_service_api_builder.sh ← LightningAI start command wrapper
│   └── log_hook.py                    ← AI usage logging hook (Claude Code)
│
├── secret/                            ← GITIGNORED — không commit
│   ├── shared.env                     ← LIGHTNING_API_BASE_URL, tracking URL, DB
│   ├── deploy/
│   │   └── search-engine-119.shared.env ← VPS-specific env
│   └── oauth/
│       ├── oauth2_credentials.json    ← Google OAuth client credentials
│       └── oauth2_token.pickle        ← Refresh token đã authenticated
│
├── secrets/                           ← Copy của secret/ cho Docker container
│   └── oauth/
│
├── move.py                            ← Google Drive: Temp/ → Storage/ organizer
├── shared_secret_runtime.py           ← Load secrets/shared.env vào runtime env
└── rf-detr-xxlarge.pth               ← RF-DETR checkpoint (484MB, root của LightningAI)
```

---

## 10. Cài đặt và chạy từ đầu

### Prerequisites

| Thành phần | Yêu cầu |
|------------|---------|
| Máy cá nhân | Python 3.11+, pip, git |
| VPS | Ubuntu 22.04+, Docker 24+, Docker Compose v2, RAM ≥ 12GB |
| LightningAI | Account, API Builder enabled, GPU L4 |
| Google Drive | Google Cloud project với Drive API enabled, OAuth2 credentials |

### Bước 1 — Clone repo

```bash
git clone <REPO_URL>
cd A20-App-119
```

### Bước 2 — Chuẩn bị Google Drive

1. Tạo thư mục `VinUni/Temp/` và `VinUni/Storage/` trên Google Drive.
2. Lưu `Folder ID` của `Storage/` → dùng cho `GOOGLE_DRIVE_SOURCE_STORAGE_FOLDER_ID`.
3. Lưu `Folder ID` của `Temp/` → dùng trong `move.py` (`TEMP_FOLDER_ID`).

### Bước 3 — Tạo Google OAuth Credentials

1. Vào [Google Cloud Console](https://console.cloud.google.com) → APIs & Services → Credentials
2. Tạo OAuth 2.0 Client ID (Desktop App)
3. Download JSON → lưu vào `secrets/oauth/oauth2_credentials.json`

Chạy lần đầu để lấy token:
```bash
pip install google-auth-oauthlib google-api-python-client
python3 -c "
from google_auth_oauthlib.flow import InstalledAppFlow
flow = InstalledAppFlow.from_client_secrets_file(
    'secrets/oauth/oauth2_credentials.json',
    scopes=['https://www.googleapis.com/auth/drive']
)
creds = flow.run_local_server(port=0)
import pickle
with open('secrets/oauth/oauth2_token.pickle', 'wb') as f:
    pickle.load(creds, f)
"
# Copy sang secrets/
# token đã nằm trong secrets/oauth/ sẵn rồi
```

### Bước 4 — Cấu hình môi trường

```bash
# Tạo file secret chính
cp secrets/master.env.example secrets/master.env
nano secrets/master.env
# Điền: LIGHTNING_API_BASE_URL, TRACKING_SERVICE_URL, DB credentials
```

### Bước 5 — Start LightningAI API Builder

1. Vào LightningAI Studio → API Builder → `tracking-service`
2. Machine: **1 × L4**
3. On start command: `bash A20-App-119/scripts/start_tracking_service_api_builder.sh`
4. Click **Start** | Bật **Auto start**

Verify:
```bash
curl -H "Authorization: Bearer <LIGHTNING_API_TOKEN>" \
  https://8000-<HASH>.cloudspaces.litng.ai/health
# Expected: {"status":"ok","service":"tracking-service"}

curl -H "Authorization: Bearer <LIGHTNING_API_TOKEN>" \
  https://8000-<HASH>.cloudspaces.litng.ai/api/v1/runtime-config/hardware \
  | python3 -m json.tool | grep -E "gpu_model|detector_batch"
# Expected: "gpu_model": "NVIDIA L4", "detector_batch_size": 40
```

Sau khi confirm GPU OK, cập nhật URL trong secrets:
```bash
nano secrets/master.env
# LIGHTNING_API_BASE_URL=https://8000-<HASH>.cloudspaces.litng.ai
# TRACKING_SERVICE_URL=https://8000-<HASH>.cloudspaces.litng.ai
```

### Bước 6 — Deploy VPS Stack

```bash
ssh root@<VPS_IP>
cd /opt/mcpt/A20-App-119

# Lần đầu: clone repo
git clone <REPO_URL> .

# Update code
git pull

# Copy secrets (từ local hoặc secure transfer)
# Dùng: bash scripts/sync_secrets.sh --vps

# Kiểm tra secrets
bash infra/vps/check-secrets.sh .

# Deploy (build + up)
bash infra/vps/deploy.sh .

# Verify
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env ps
```

Kết quả mong đợi:
```
NAME              STATUS
mcpt-postgres     Up (healthy)
mcpt-backend      Up (healthy)
mcpt-ai-service   Up (healthy)
mcpt-frontend     Up
```

### Bước 7 — Upload và xử lý video

```bash
# Trên máy cá nhân:
# Đặt video vào Google Drive Temp/ theo naming convention:
# cam{XX}_{YYYY-MM-DD}_{HH-mm}.mp4

# Dry run
python3 move.py --dry-run
# Found 50 .mp4 files in Temp/
# [dry-run] move → Storage/cam01/2026-04-28/cam01_2026-04-28_10-00.mp4
# ...

# Move thực
python3 move.py
# Result: 50 moved, 0 skipped.
```

Queue worker sẽ tự động phát hiện trong 30s. Hoặc trigger thủ công:
```bash
ssh root@<VPS_IP>
docker exec mcpt-backend python -m metadata_app.queue_worker --once
```

### Bước 8 — Theo dõi tiến độ

```bash
# Xem log queue worker
ssh root@<VPS_IP> "cat /tmp/qw_run.log" | grep -E "completed|failed|people"
# 10:04:32 INFO Tracking completed cam01 people=225
# 10:09:15 INFO Tracking completed cam02 people=567

# Xem trong DB
docker exec mcpt-postgres psql -U mcpt_user -d video_tracking -c "
  SELECT camera_id, COUNT(*) as tracklets
  FROM person_candidates
  GROUP BY camera_id ORDER BY 2 DESC LIMIT 10;
"
```

### Bước 9 — Sử dụng hệ thống

Truy cập frontend: `http://<VPS_IP>:3000`

Test API:
```bash
# Đăng nhập
curl -X POST http://<VPS_IP>:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "<ADMIN_PASSWORD>"}'
# Response: {"access_token": "eyJ..."}

# Search
curl -X POST http://<VPS_IP>:8000/search \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"query": "người đàn ông áo xanh đeo balo", "top_k": 5}'

# Trace
curl -X POST http://<VPS_IP>:8000/api/v1/trace/run \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"candidate_id": "<UUID>", "window_hours": 12.0}'
```

---

## 11. Biến môi trường

### `infra/env/backend.env` — VPS Docker Stack

**Database**

| Biến | Ví dụ | Bắt buộc | Mô tả |
|------|-------|----------|-------|
| `POSTGRES_HOST` | `postgres` | ✅ | Hostname PostgreSQL trong Docker network |
| `POSTGRES_PORT` | `5432` | ✅ | Port PostgreSQL |
| `POSTGRES_USER` | `mcpt_user` | ✅ | DB user |
| `POSTGRES_PASSWORD` | `<POSTGRES_PASSWORD>` | ✅ | DB password — không commit |
| `POSTGRES_DATABASE` | `video_tracking` | ✅ | Database name |

**Authentication**

| Biến | Ví dụ | Bắt buộc | Mô tả |
|------|-------|----------|-------|
| `JWT_SECRET_KEY` | `<JWT_SECRET>` | ✅ | Secret key ký JWT token |

**LightningAI**

| Biến | Ví dụ | Bắt buộc | Mô tả |
|------|-------|----------|-------|
| `TRACKING_SERVICE_URL` | `https://8000-<HASH>.cloudspaces.litng.ai` | ✅ | URL LightningAI Tracking Service — cập nhật mỗi khi restart |
| `LIGHTNING_API_TOKEN` | `<LIGHTNING_API_TOKEN>` | ✅ | Bearer token để call LightningAI |
| `LIGHTNING_API_AUTH_HEADER` | `Authorization` | ✅ | Header name |
| `LIGHTNING_API_AUTH_PREFIX` | `Bearer ` | ✅ | Prefix trước token |

**Queue Worker**

| Biến | Ví dụ | Bắt buộc | Mô tả |
|------|-------|----------|-------|
| `QUEUE_PARALLEL_JOBS` | `3` | — | Số video gửi song song lên LightningAI |
| `QUEUE_POLL_INTERVAL_SECONDS` | `30` | — | Interval poll Google Drive |
| `QUEUE_MAX_SIZE` | `32` | — | Max số video trong queue DB |
| `STORAGE_INGEST_BATCH_SIZE` | `50` | — | Max video scan mỗi lần |
| `STORAGE_INGEST_SAMPLE_FPS` | `5` | — | FPS sample khi decode video |
| `STORAGE_INGEST_SOURCE_BACKEND` | `google_drive` | ✅ | Backend nguồn video |

**Google Drive**

| Biến | Ví dụ | Bắt buộc | Mô tả |
|------|-------|----------|-------|
| `GOOGLE_DRIVE_ENABLED` | `true` | ✅ | Bật/tắt Drive integration |
| `GOOGLE_DRIVE_SOURCE_STORAGE_FOLDER_ID` | `1G6L1d8l...` | ✅ | Folder ID của `Storage/` trên Drive |
| `GOOGLE_DRIVE_ROOT_FOLDER_ID` | `1gxKBTQ9...` | ✅ | Folder ID của `VinUni/` root |

### `secrets/shared.env` — Shared giữa Docker containers + LightningAI

```env
LIGHTNING_API_BASE_URL=https://8000-<HASH>.cloudspaces.litng.ai
TRACKING_SERVICE_URL=https://8000-<HASH>.cloudspaces.litng.ai
LIGHTNING_API_TOKEN=<LIGHTNING_API_TOKEN>
GOOGLE_DRIVE_ENABLED=true
GOOGLE_DRIVE_ROOT_FOLDER_ID=<ROOT_FOLDER_ID>
GOOGLE_DRIVE_VINUNI_FOLDER_ID=<VINUNI_FOLDER_ID>
MCPT_OAUTH2_CREDENTIALS_FILE=oauth/oauth2_credentials.json
MCPT_OAUTH2_TOKEN_FILE=oauth/oauth2_token.pickle
POSTGRES_DATABASE=video_tracking
POSTGRES_HOST=localhost
POSTGRES_USER=mcpt_user
POSTGRES_PASSWORD=<POSTGRES_PASSWORD>
POSTGRES_PORT=5432
```

> **Quan trọng:** File này được load bởi cả `shared_secret_runtime.py` (tracking service) và metadata service. Không commit lên Git.

---

## 12. Database Schema

### `queue_video_assets`

Lưu thông tin từng video đã được xử lý.

| Column | Type | Mô tả |
|--------|------|-------|
| `video_id` | VARCHAR | Tên file video (e.g. `cam01_2026-04-28_10-00.mp4`) — primary key |
| `camera_id` | VARCHAR | Camera ID (e.g. `cam01`) |
| `title` | VARCHAR | Display title |
| `source_filename` | VARCHAR | Tên file gốc |
| `source_mode` | VARCHAR | `storage_ingest` — nguồn từ Drive Storage |
| `storage_backend` | VARCHAR | `local_queue_storage` hoặc `google_drive_local_metadata` |
| `local_video_path` | VARCHAR | Path trên LightningAI filesystem |
| `local_metadata_path` | VARCHAR | Path JSON metadata |
| `raw_video_metadata` | JSONB | Toàn bộ metadata từ tracking service (sample_fps, tracklet_count...) |
| `available_link_video` | VARCHAR | URL xem video |
| `available_link_metadata` | VARCHAR | URL xem metadata JSON |
| `created_at` | TIMESTAMP | Thời gian insert |

### `person_candidates`

Lưu từng tracklet với embedding và metadata.

| Column | Type | Mô tả |
|--------|------|-------|
| `id` | INTEGER | Auto-increment primary key |
| `candidate_id` | VARCHAR | UUID của tracklet |
| `video_id` | VARCHAR | FK → `queue_video_assets.video_id` |
| `camera_id` | VARCHAR | Camera ID của tracklet |
| `track_id` | VARCHAR | Local track ID (unique per video) |
| `search_text` | TEXT | Text tổng hợp để full-text search |
| `raw_metadata` | JSONB | Toàn bộ payload từ tracking service bao gồm: `embedding_vector`, `appearance_embedding_vector`, `attribute_embedding_vector`, `action_semantic_embedding`, `timeline`, `semantic_attributes`, `appearance_summary` |
| `created_at` | TIMESTAMP | Thời gian insert |
| `updated_at` | TIMESTAMP | Thời gian update |

> **Note:** Các embedding vector lưu trong JSONB `raw_metadata`, không có column riêng. Truy cập qua `raw_metadata['appearance_embedding_vector']`.

### `users`

| Column | Mô tả |
|--------|-------|
| `id` | UUID primary key |
| `username` | Username đăng nhập |
| `hashed_password` | Bcrypt hash |
| `role` | `admin` / `operator` |
| `created_at` | Timestamp |

---

## 13. API Documentation

### POST `/api/v1/auth/login`

| | |
|---|---|
| **Service** | API Gateway |
| **Auth** | Không cần |
| **Mục đích** | Đăng nhập, lấy JWT token |

Request:
```json
{"username": "admin", "password": "<PASSWORD>"}
```
Response:
```json
{"access_token": "eyJ...", "token_type": "bearer"}
```

### POST `/search`

| | |
|---|---|
| **Service** | API Gateway → AI Service → LightningAI |
| **Auth** | Bearer JWT |
| **Mục đích** | Tìm kiếm người theo query text |

Request:
```json
{
  "query": "người đàn ông áo xanh đeo balo",
  "top_k": 10,
  "offset": 0,
  "camera_ids": ["cam07", "cam08"],
  "time_from": "2026-04-28T08:00:00",
  "time_to": "2026-04-28T12:00:00"
}
```
Response:
```json
{
  "results": [
    {
      "id": "uuid-candidate",
      "thumbnail_url": "/api/v1/candidates/uuid/preview",
      "description": "male adult · blue shirt · cam08 · walking_motion · score=0.84"
    }
  ]
}
```
Lỗi thường gặp: `502` khi LightningAI cold-start chưa xong.

### POST `/api/v1/trace/run`

| | |
|---|---|
| **Service** | API Gateway → Metadata Service |
| **Auth** | Bearer JWT |
| **Mục đích** | Trace hành trình từ seed candidate |

Request:
```json
{
  "candidate_id": "uuid-of-selected-candidate",
  "window_hours": 12.0
}
```
Response:
```json
{
  "seed_candidate_id": "uuid",
  "total_segments": 4,
  "cameras_visited": ["cam01", "cam07", "cam14", "cam19"],
  "overall_score": 0.83,
  "window_from": "2026-04-27T20:12:00Z",
  "window_to": "2026-04-28T20:12:00Z",
  "trajectory": [
    {
      "camera_id": "cam01",
      "camera_name": "Cổng chính ngoài trời",
      "start_time": "2026-04-28T08:12:00Z",
      "end_time": "2026-04-28T08:18:00Z",
      "appearance_sim": 0.91,
      "segment_score": 0.87,
      "preview_url": "...",
      "video_url": "..."
    }
  ]
}
```

### POST `/api/v1/trace/feedback`

| | |
|---|---|
| **Service** | API Gateway → Metadata Service |
| **Auth** | Bearer JWT |
| **Mục đích** | Human-in-the-loop: xác nhận/từ chối segments → re-trace |

Request:
```json
{
  "candidate_id": "uuid-seed",
  "confirmed_segment_ids": ["uuid-1", "uuid-2"],
  "rejected_segment_ids": ["uuid-3"],
  "window_hours": 12.0
}
```
Response: Giống `/trace/run` — trajectory mới sau re-trace.

### GET `/api/v1/queue/videos`

| | |
|---|---|
| **Service** | API Gateway → Metadata Service |
| **Auth** | Bearer JWT |
| **Mục đích** | Danh sách video đã/đang xử lý |

Response:
```json
{
  "count": 50,
  "items": [
    {
      "video_id": "cam01_2026-04-28_10-00.mp4",
      "camera_id": "cam01",
      "source_mode": "storage_ingest",
      "available_link_video": "/api/v1/queue/videos/cam01_.../file"
    }
  ]
}
```

### POST `/api/v1/queue/process-storage`

| | |
|---|---|
| **Service** | API Gateway → Metadata Service |
| **Auth** | Bearer JWT |
| **Mục đích** | Trigger queue worker thủ công (không đợi 30s) |

Request: `{}` (empty body)
Response:
```json
{
  "processed_videos": 3,
  "imported_source_files": ["cam01/2026-04-28/cam01_2026-04-28_10-00.mp4"],
  "evicted_video_ids": []
}
```

### GET `/health`

| | |
|---|---|
| **Service** | LightningAI Tracking Service |
| **Auth** | Bearer `<LIGHTNING_API_TOKEN>` |
| **Mục đích** | Kiểm tra tracking service còn sống |

Response: `{"status": "ok", "service": "tracking-service"}`

### POST `/api/v1/ingestion/process` *(LightningAI internal)*

| | |
|---|---|
| **Service** | LightningAI Tracking Service |
| **Auth** | Bearer `<LIGHTNING_API_TOKEN>` |
| **Mục đích** | Xử lý 1 video — endpoint chính của tracking service |

Request:
```json
{
  "source_url": "https://drive.usercontent.google.com/download?id=...",
  "source_filename": "cam01_2026-04-28_10-00.mp4",
  "camera_id": "cam01",
  "recorded_start": "2026-04-28T10:00:00",
  "output_video_dir": null,
  "output_metadata_dir": null,
  "metadata": {"source_mode": "storage_ingest", ...}
}
```
Response: JSON đầy đủ gồm `video` metadata và `people` array (tracklets).

---

## 14. Camera Topology

File: `backend/config/camera_topology.json`

**50 cameras** phân bổ theo khu vực trong bệnh viện:

| Khu vực | Cameras | Tầng |
|---------|---------|------|
| Cổng chính, Drop-off, Bãi xe | cam01–cam06 | 1 |
| Sảnh chính, Lễ tân, Đăng ký | cam07–cam12 | 1 |
| Hành lang tầng 1, Khám bệnh | cam13–cam22 | 1 |
| Xét nghiệm, Chẩn đoán hình ảnh | cam23–cam30 | 1 |
| Thang máy / Thang bộ (lên tầng 2) | cam31–cam32 | 1→2 |
| Tầng 2–5 (hành lang, phòng bệnh, ICU) | cam33–cam48 | 2–5 |
| Mái, Rooftop access | cam49–cam50 | 6 |

**Edge mẫu:**
```json
{
  "from": "cam07",
  "to": "cam08",
  "connectivity_type": "lobby_internal",
  "min_seconds": 5,
  "max_seconds": 45,
  "typical_seconds": 15,
  "confidence": 0.98
}
```

Bidirectional — mọi edge đều đi được 2 chiều.

---

## 15. Troubleshooting

### LightningAI URL thay đổi sau restart

**Triệu chứng:** `502 Bad Gateway` khi gọi search hoặc ingestion.

**Nguyên nhân:** Mỗi lần restart API Builder, LightningAI cấp URL mới.

**Fix:**
```bash
# Lấy URL mới từ LightningAI UI → API Builder → Settings → URL
nano secrets/master.env
# LIGHTNING_API_BASE_URL=https://8000-<NEW_HASH>.cloudspaces.litng.ai
# TRACKING_SERVICE_URL=https://8000-<NEW_HASH>.cloudspaces.litng.ai

# Cập nhật trên VPS
ssh root@<VPS_IP>
sed -i 's|TRACKING_SERVICE_URL=.*|TRACKING_SERVICE_URL=https://8000-<NEW_HASH>.cloudspaces.litng.ai|' \
  /opt/mcpt/A20-App-119/infra/env/backend.env
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env restart backend ai_service
```

### GPU không được detect trên LightningAI

**Triệu chứng:** `/api/v1/runtime-config/hardware` trả về `"gpu_model": "CPU-only"`.

**Nguyên nhân:** API Builder chưa khởi động đủ, hoặc đang chạy trên DevBox CPU.

**Fix:** Đợi thêm 2–3 phút sau khi Start. Nếu vẫn CPU, stop và start lại API Builder.

### OAuth token Google Drive lỗi

**Triệu chứng:** `queue_worker` log lỗi `google.auth.exceptions.RefreshError`.

**Nguyên nhân:** Token hết hạn hoặc bị revoke.

**Fix:**
```bash
# Xóa token cũ và chạy lại OAuth flow
rm secrets/oauth/oauth2_token.pickle
python3 -c "
from google_auth_oauthlib.flow import InstalledAppFlow
flow = InstalledAppFlow.from_client_secrets_file(
    'secrets/oauth/oauth2_credentials.json',
    scopes=['https://www.googleapis.com/auth/drive']
)
creds = flow.run_local_server(port=0)
import pickle
with open('secrets/oauth/oauth2_token.pickle', 'wb') as f:
    pickle.dump(creds, f)
"
# token đã nằm trong secrets/oauth/ sẵn rồi
# Restart backend container để load token mới
ssh root@<VPS_IP> "docker compose -f /opt/mcpt/A20-App-119/infra/docker-compose.yml \
  --env-file /opt/mcpt/A20-App-119/infra/env/backend.env restart backend"
```

### Queue worker không thấy video mới

**Triệu chứng:** Log không có dòng `Requesting tracking ingestion`.

**Kiểm tra:**
1. File trong `Storage/` có đúng naming convention không? (`cam{XX}_{YYYY-MM-DD}_{HH-mm}.mp4`)
2. Fingerprint đã có trong `ProcessedStorage` registry chưa?
3. Drive scan có thấy folder không?

```bash
# Clear registry để force re-scan
docker exec mcpt-backend find /workspace/storage/queue -path '*/ProcessedStorage/*.json' -delete
docker exec mcpt-backend python -m metadata_app.queue_worker --once
```

### Video sai naming convention

**Triệu chứng:** `move.py` báo `SKIP (bad name)`.

**Đúng:** `cam01_2026-04-28_10-00.mp4`

**Sai:** `Camera_01.mp4`, `cam1_20260428.mp4`, `cam01-2026-04-28.mp4`

Pattern bắt buộc: `cam_XX_YYYY-MM-DD_HH-mm.mp4` (cam + underscore + số).

### Docker service không chạy

```bash
# Xem log của service bị lỗi
docker logs mcpt-backend --tail=50
docker logs mcpt-postgres --tail=20

# Kiểm tra port conflict
ss -tlnp | grep -E '8000|3000|5432'

# Rebuild từ đầu
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env down
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env up -d --build
```

### PostgreSQL connection failed

**Triệu chứng:** Backend log `psycopg2.OperationalError: could not connect to server`.

**Kiểm tra:**
```bash
docker exec mcpt-postgres pg_isready -U mcpt_user -d video_tracking
# output: /var/run/postgresql:5432 - accepting connections
```

Nếu không healthy: `docker restart mcpt-postgres` và đợi 30s.

### Search trả kết quả rỗng

**Kiểm tra theo thứ tự:**
1. Có data trong DB chưa? `docker exec mcpt-postgres psql -U mcpt_user -d video_tracking -c "SELECT COUNT(*) FROM person_candidates;"`
2. LightningAI có đang chạy không? `curl -H "Authorization: Bearer <TOKEN>" https://.../health`
3. Query có quá strict không? Thử bỏ `camera_ids` và `time_from`/`time_to`
4. Kiểm tra log AI service: `docker logs mcpt-ai-service --tail=20`

### Trace không tìm được hành trình hợp lý

**Nguyên nhân phổ biến:**
1. Chưa có đủ video đã xử lý trong time window
2. Seed candidate chọn tracklet chất lượng thấp (blur, crop nhỏ)
3. Camera topology chưa cấu hình đúng cho layout thực tế

**Fix:** Giảm threshold similarity (`min_similarity=0.35`), tăng `window_hours`.

---

## 16. Security Notes

- **Không commit** `secret/`, `secrets/` lên Git. Kiểm tra `.gitignore`.
- **OAuth token** (`oauth2_token.pickle`) có quyền đọc/ghi toàn bộ Drive — bảo vệ như password.
- **JWT secret** phải random, đủ dài (≥ 32 chars). Không dùng giá trị mặc định trên production.
- **LightningAI Bearer token** (`LIGHTNING_API_TOKEN`) xác thực mọi request tới GPU service. Rotate định kỳ.
- **PostgreSQL password** không expose qua API. Chỉ accessible trong Docker network nội bộ.
- **API public** (`api-gateway :8000`) cần JWT cho tất cả endpoint trừ `/health` và `/api/v1/auth/login`.
- **VPS firewall** chỉ mở port 80/443 (frontend) và 8000 (API). Đóng 5432 với external.
- Khi debug với `curl`, dùng placeholder như `<JWT_TOKEN>`, không paste token thật vào log/chat.

---

## 17. Performance Notes

| Chỉ số | Giá trị thực tế |
|--------|----------------|
| Throughput tracking | ~4–5 phút / video 10 phút trên L4 |
| Parallel jobs | 3 video song song (`QUEUE_PARALLEL_JOBS=3`) |
| Detector batch | 40 frames/pass trên L4 |
| Video capacity | ~60 video/giờ với 1 L4 |
| 50 cameras × 10min/video | 500 phút video → ~8–9 giờ xử lý (1 L4) |
| Search latency | 5–30s (cold start L4 ~60–120s) |
| Trace latency | 5–30s tùy window size và số candidate |

**Bottleneck thường gặp:**
- **GPU inference:** Tăng `QUEUE_PARALLEL_JOBS` hoặc nâng machine type L4 → L40S
- **Drive download:** Video lớn (>500MB) mất thêm thời gian tải
- **PostgreSQL writes:** Lớn hơn 1000 candidates/video → tăng RAM DB
- **Cold start L4:** Auto start mất 60–120s cho lần gọi đầu sau idle

---

## 18. Limitations & Future Work

| Giới hạn | Mô tả |
|----------|-------|
| **Không có Global Person ID** | `track_id` chỉ có ý nghĩa trong 1 video. Không có cross-video identity liên tục. |
| **Trace dựa trên heuristic** | Greedy path search — không đảm bảo optimal trajectory. |
| **Topology tĩnh** | `camera_topology.json` cần update thủ công nếu layout bệnh viện thay đổi. |
| **Video ingestion theo lô** | Không hỗ trợ real-time streaming. Queue worker poll mỗi 30s. |
| **Không có monitoring dashboard** | Chưa có Grafana/Prometheus cho GPU utilization, queue depth. |
| **Chưa có CI/CD** | Deploy thủ công qua SSH. `TODO: thêm GitHub Actions pipeline.` |
| **Chưa benchmark trên data thật** | Độ chính xác Re-ID trên camera bệnh viện thực tế chưa được đánh giá chính thức. |
| **Privacy** | Chưa có anonymization (blur face) cho video export. `TODO: thêm trước khi production.` |
| **Single L4 GPU** | Scale horizontal (nhiều L4) chưa được hỗ trợ. |
| **Không có automated tests** | `TODO: thêm pytest cho pipeline, integration tests cho API.` |

---

## 19. Thành viên nhóm

| Tên | Vai trò |
|-----|---------|
| Cao Diệu Ly | Leader, PM, AI Research |
| Dương Văn Hiệp | AI Engineer, Backend, Data Pipeline |
| Bùi Văn Đạt | Backend, Frontend |
