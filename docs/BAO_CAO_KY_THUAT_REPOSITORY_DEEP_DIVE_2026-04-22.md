# Báo Cáo Kỹ Thuật Chuyên Sâu Repository `A20-App-119`

## 0. Phạm vi, cách đọc và kết luận nhanh

Tài liệu này là báo cáo reverse engineering + onboarding + runbook vận hành cho workspace hiện tại tại `D:\python ky 9\A20-App-119`.

Điểm quan trọng nhất cần nắm ngay:

- Repository gốc ở root `A20-App-119` **không phải** toàn bộ application chạy production.
- Ứng dụng thực tế đang được xây và deploy chủ yếu nằm trong thư mục:

```text
Multi-Camera-Person-Tracking-and-Re-Identification/
```

- Đây là một hệ thống `frontend + backend microservices + AI pipeline + Google Drive queue + PostgreSQL`.
- `tracking-service` có thể chạy local GPU, nhưng topology production trong code hiện tại nghiêng về:
  - VPS chạy `frontend`, `api-gateway`, `metadata-service`, `queue-worker`, `postgres`
  - `tracking-service`/GPU chạy remote qua `LightningAI`
- Google Drive được dùng như tầng ingest/queue video (`VinUni/Import_New`, `VinUni/Queue/.h265`), còn metadata trong branch/workspace hiện tại đã nghiêng về `local JSON + PostgreSQL`, không còn là thiết kế “Drive giữ metadata” thuần túy.

Nếu mục tiêu là chạy repo nhanh nhất:

1. chuẩn bị `shared.env` + OAuth Google Drive
2. dựng PostgreSQL
3. chạy Docker Compose cho `postgres`, `metadata-service`, `queue-worker`, `api-gateway`, `frontend`
4. trỏ `TRACKING_SERVICE_URL` tới LightningAI đang sống
5. kiểm tra `/api/v1/overview`, `/health`, đăng nhập, query candidate

Tài liệu này đi sâu tới mức file/module/function/config/script thực tế.

## 1. Tổng quan hệ thống

### 1.1. Repo này dùng để làm gì?

Hệ thống phục vụ bài toán:

- ingest video `.h265/.hevc` từ Google Drive hoặc source local
- chạy pipeline detect/tracking/metadata generation trên video
- index người (`person candidates`) vào PostgreSQL
- cho phép user đăng nhập, nhập text query, lấy top-k candidate
- user chọn một candidate
- hệ thống dựng artifact video tracking tổng hợp

Ở trạng thái code hiện tại, flow vận hành gần nhất là:

```text
Google Drive Import_New
  -> queue-worker
  -> tracking-service / LightningAI ingestion
  -> local metadata JSON + PostgreSQL + Queue/.h265
  -> frontend text query
  -> metadata-service shortlist
  -> tracking-service rerank
  -> top-k candidate
  -> build tracking artifact video
```

### 1.2. Kiến trúc tổng thể thuộc loại gì?

Kiến trúc thực tế là:

- `frontend + backend tách riêng`
- backend chia theo `microservice-ish services`
- có thành phần `AI pipeline app`
- có `queue worker`
- có `external inference service` qua LightningAI

Không phải monolith thuần. Cũng chưa phải microservice hoàn toàn “mature” vì:

- chưa có service discovery
- chưa có message broker đúng nghĩa
- chưa có migration framework chuẩn
- nhiều service đang dùng chung secret/runtime loader và cùng mount chung storage volume

### 1.3. Bức tranh lớn

```text
User Browser
  -> Frontend (Next.js)
  -> API Gateway (FastAPI facade)
      -> Metadata Service (FastAPI + PostgreSQL + queue metadata logic)
      -> Tracking Service (FastAPI + CV/AI orchestration)
          -> LightningAI GPU hoặc local legacy engine
  -> Google Drive (Import_New, Queue/.h265)
  -> Local storage volumes (metadata JSON, previews, tracking artifacts)
```

### 1.4. Phân vai thành phần

| Thành phần | Vai trò |
|---|---|
| `frontend/` | Next.js UI cho login, query, top-k, build tracking video |
| `backend/services/api-gateway/` | Gateway/public API cho frontend |
| `backend/services/metadata-service/` | Auth, DB CRUD, queue catalog, shortlist candidate, preview image, tracking artifact fallback |
| `backend/services/tracking-service/` | Orchestrate ingestion, query reranking, candidate track build, gọi LightningAI hoặc local engine |
| `backend/legacy-engine/` | Code CV/ReID/VLM cũ nhưng vẫn đang được tái sử dụng trong ingestion runtime |
| `infra/postgres/` | init SQL cho PostgreSQL |
| `infra/vps/` | provisioning/deploy/nginx/https cho VPS |
| `shared_secret_runtime.py` | loader hợp nhất env + OAuth secret + Google Drive client |
| `exchange.py` | script full-pipeline/bootstrap/integration test để ingest hàng loạt và seed DB |

## 2. Cấu trúc thư mục và vai trò từng phần

### 2.1. Root workspace `A20-App-119`

Các phần đáng chú ý:

```text
A20-App-119/
├─ Multi-Camera-Person-Tracking-and-Re-Identification/   # app chính
├─ shared_secret_runtime.py                              # secret/env runtime loader
├─ docs/                                                 # tài liệu kỹ thuật/audit
├─ secret/                                               # secret thực tế trong workspace hiện tại
├─ src/                                                  # scaffold khác, không phải app đang deploy chính
├─ ARCHITECTURE.md                                       # kiến trúc khái niệm, không hoàn toàn khớp runtime
└─ .env / .env.example                                   # env root
```

### 2.2. Lưu ý rất quan trọng về “repo thật”

Trong workspace này có hai lớp:

1. `root repo`
2. `nested app repo` trong `Multi-Camera-Person-Tracking-and-Re-Identification/`

Phần production flow, Docker, frontend/backend thực tế nằm ở nested repo.

Thư mục `src/` ở root chỉ có:

- `agent.py`
- `config.py`
- `tools.py`

Nó không phải entrypoint của website/queue/AI pipeline hiện tại.

### 2.3. Cấu trúc nested app repo

```text
Multi-Camera-Person-Tracking-and-Re-Identification/
├─ frontend/
├─ backend/
│  ├─ services/
│  │  ├─ api-gateway/
│  │  ├─ metadata-service/
│  │  └─ tracking-service/
│  ├─ legacy-engine/
│  └─ config/
├─ infra/
│  ├─ postgres/
│  ├─ nginx/
│  └─ vps/
├─ docs/
├─ docker-compose.yml
├─ docker-compose.edge.yml
├─ exchange.py
├─ setup_drive_folders.py
├─ start_postgres.py
└─ .env.example
```

### 2.4. Folder nào là entrypoint chạy app?

| Thành phần | Entrypoint |
|---|---|
| Frontend | `frontend/app/page.tsx`, chạy qua `next dev` hoặc `next start` |
| API Gateway | `backend/services/api-gateway/app/main.py` |
| Metadata Service | `backend/services/metadata-service/app/main.py` |
| Tracking Service | `backend/services/tracking-service/app/main.py` |
| Queue worker | `python -m app.queue_worker` trong image metadata-service |
| Full bootstrap pipeline | `exchange.py` |

### 2.5. Folder business logic

| Module | Business logic chính |
|---|---|
| `metadata-service/app/service.py` | auth, video/query CRUD, preview image, candidate formatting, shortlist, track request assembly, DB sync local queue |
| `metadata-service/app/queue_runtime.py` | Google Drive polling, import queue, queue eviction, upload/move/update DB |
| `tracking-service/app/service.py` | query embedding, rerank, remote worker call, tracking artifact build |
| `tracking-service/app/ingestion_runtime.py` | ingest video, convert/resolve/download, chạy legacy engine, upload output |
| `backend/legacy-engine/` | detector/tracker/VLM inference thực tế |

### 2.6. Folder config

| File | Mục đích |
|---|---|
| `shared_secret_runtime.py` | hợp nhất env root + shared env + OAuth path |
| `metadata-service/app/config.py` | config DB, queue, Drive, auth, tracking endpoint |
| `tracking-service/app/config.py` | config AI runtime, GPU, ffmpeg, Drive, Lightning |
| `api-gateway/app/config.py` | config URL downstream + CORS |
| `.env.example` | sample env cấp nested app |
| root `.env.example` | canonical env cấp workspace |

### 2.7. DB schema / migration / seed nằm ở đâu?

- ORM models: `backend/services/metadata-service/app/models.py`
- init SQL PostgreSQL: `infra/postgres/init/01-init.sql`
- seed thực tế: không có seed framework riêng; thay vào đó dùng:
  - `exchange.py`
  - `queue-worker`
  - `/api/v1/candidates/import-legacy`
  - startup sync `sync_local_queue_state()`

Điểm rất quan trọng:

- **Không có Alembic**
- **Không có migration scripts versioned**
- schema hiện được tạo bằng `Base.metadata.create_all(bind=engine)` trong startup của metadata-service

### 2.8. Folder tích hợp ngoài

| Tích hợp ngoài | File chính |
|---|---|
| Google Drive OAuth | `shared_secret_runtime.py`, `queue_runtime.py`, `ingestion_runtime.py` |
| LightningAI | `tracking-service/app/gpu_client.py`, `tracking-service/app/service.py`, `queue_runtime.py` |
| PostgreSQL | `metadata-service/app/database.py`, `exchange.py`, `start_postgres.py` |

### 2.9. Folder AI / model / inference

| File/Folder | Vai trò |
|---|---|
| `tracking-service/app/service.py` | text embedding, semantic rerank, anchor similarity |
| `tracking-service/app/ingestion_runtime.py` | ingestion pipeline orchestration |
| `backend/legacy-engine/src_vlm/` | VLM + hospital pipeline |
| `tracking-service/app/execution_plan.py` | execution planning theo hardware profile |
| `tracking-service/app/hardware_profiles.py` | profile GPU/host |
| `tracking-service/app/pipeline_profiles.py` | pipeline profile như `accuracy_first` |

### 2.10. Folder deploy/devops/infra

| File | Vai trò |
|---|---|
| `docker-compose.yml` | compose chuẩn cho local/dev/prod-like |
| `docker-compose.edge.yml` | profile tracking-service local GPU/NVIDIA |
| `infra/vps/provision.sh` | cài Docker/nginx/certbot trên VPS |
| `infra/vps/deploy.sh` | build/run stack Docker trên VPS |
| `infra/vps/configure_nginx.sh` | sinh nginx vhost |
| `infra/vps/enable_https.sh` | bật Let’s Encrypt |
| `infra/nginx/search-engine.conf.template` | reverse proxy template |
| `.github/workflows/mcpt-ci-cd.yml` | CI/CD build + copy + SSH deploy |

## 3. Hướng dẫn chạy repo cực kỳ chi tiết

### 3.1. Điều kiện tiên quyết

#### Hệ điều hành giả định

- Phát triển local: Windows 11 hoặc Linux
- Chạy production/VPS: Ubuntu 22.04+ là phù hợp nhất theo script hiện có

#### Runtime cần có

| Thành phần | Khuyến nghị |
|---|---|
| Python | `3.11.x` |
| Node.js | `20.x` |
| npm | đi kèm Node 20 |
| Docker Engine | 24+ |
| Docker Compose plugin | bản tương thích Docker hiện tại |
| PostgreSQL | `16` |
| ffmpeg | cần nếu chạy `tracking-service` local |
| NVIDIA runtime | chỉ cần nếu chạy local GPU |

#### Cách kiểm tra version

```bash
python --version
node --version
npm --version
docker --version
docker compose version
psql --version
ffmpeg -version
```

#### Cần GPU khi nào?

- Không cần GPU để:
  - chạy frontend
  - chạy api-gateway
  - chạy metadata-service
  - chạy queue-worker nếu tracking-service trỏ remote Lightning
- Cần GPU local khi:
  - bật `tracking-service` local bằng `COMPOSE_PROFILES=local-gpu`
  - hoặc chạy `docker-compose.edge.yml`

### 3.2. Cài dependencies

#### Cách 1: chạy toàn stack bằng Docker Compose

Đây là cách nhanh và đúng nhất với repo này.

Chạy từ:

```bash
cd Multi-Camera-Person-Tracking-and-Re-Identification
```

Build:

```bash
docker compose --env-file ../secrets/shared.env build
```

Hoặc nếu workspace hiện tại chỉ có thư mục `secret/` thay vì `secrets/`, cần điều chỉnh path env tương ứng:

```bash
docker compose --env-file ../secret/shared.env build
```

#### Cách 2: cài frontend riêng

```bash
cd Multi-Camera-Person-Tracking-and-Re-Identification/frontend
npm install
npm run dev
```

#### Cách 3: cài backend riêng bằng Python

Metadata service:

```bash
cd Multi-Camera-Person-Tracking-and-Re-Identification/backend/services/metadata-service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Tracking service:

```bash
cd Multi-Camera-Person-Tracking-and-Re-Identification/backend/services/tracking-service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

API gateway:

```bash
cd Multi-Camera-Person-Tracking-and-Re-Identification/backend/services/api-gateway
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Lưu ý:

- repo hiện không dùng `poetry`
- không có `pyproject.toml` làm package manager chuẩn cho app chính
- Python deps dựa trên `requirements.txt` từng service

### 3.3. Environment variables

Repo có hai lớp env:

1. env root `A20-App-119/.env(.example)`
2. env nested app `Multi-Camera-Person-Tracking-and-Re-Identification/.env.example`

Ngoài ra còn có canonical shared env do `shared_secret_runtime.py` tải từ:

- `MCPT_SHARED_ENV_FILE`
- hoặc mặc định `<MCPT_SECRETS_ROOT>/shared.env`

#### 3.3.1. Cơ chế nạp env thực tế

Trong `shared_secret_runtime.py`, loader sẽ:

- xác định `MCPT_SECRETS_ROOT`
- tìm `shared.env`
- resolve đường dẫn OAuth credentials/token
- export lại:
  - `MCPT_SECRETS_ROOT`
  - `MCPT_SHARED_ENV_FILE`
  - `MCPT_OAUTH2_CREDENTIALS_FILE`
  - `MCPT_OAUTH2_TOKEN_FILE`

Snippet thực tế:

```python
def load_runtime_env(*, include_tracking_service_env: bool = True, override: bool = True) -> list[Path]:
    ...
    os.environ["MCPT_SECRETS_ROOT"] = str(secrets_root)
    os.environ["MCPT_SHARED_ENV_FILE"] = str(shared_env)
    os.environ["MCPT_OAUTH2_CREDENTIALS_FILE"] = str(oauth2_credentials)
    os.environ["MCPT_OAUTH2_TOKEN_FILE"] = str(oauth2_token)
```

#### 3.3.2. Nhóm biến quan trọng

##### Nhóm DB

| Biến | Bắt buộc | Dùng ở đâu | Ý nghĩa |
|---|---|---|---|
| `POSTGRES_HOST` | Có | metadata-service, tracking-service, exchange.py | host DB |
| `POSTGRES_PORT` | Có | idem | cổng DB |
| `POSTGRES_DATABASE` hoặc `POSTGRES_DB` | Có | idem | tên DB |
| `POSTGRES_USER` | Có | idem | user |
| `POSTGRES_PASSWORD` | Có | idem | password |
| `DATABASE_URL` | Optional | metadata-service, exchange.py | override toàn bộ connection string |

`metadata-service/app/config.py` tự build URL:

```python
return f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
```

##### Nhóm auth

| Biến | Bắt buộc | Ý nghĩa |
|---|---|---|
| `JWT_SECRET_KEY` | Có | ký JWT cho login/register |
| `JWT_ALGORITHM` | Optional | mặc định `HS256` |

##### Nhóm service URLs

| Biến | Dùng ở đâu | Ý nghĩa |
|---|---|---|
| `METADATA_SERVICE_URL` | api-gateway | downstream metadata-service |
| `TRACKING_SERVICE_URL` | api-gateway, metadata-service, queue-worker | downstream tracking-service hoặc Lightning public endpoint |
| `NEXT_PUBLIC_API_GATEWAY_URL` | frontend | base URL cho mọi API call từ browser |

##### Nhóm Google Drive

| Biến | Ý nghĩa |
|---|---|
| `GOOGLE_DRIVE_ENABLED` | bật/tắt queue sync |
| `GOOGLE_DRIVE_ROOT_FOLDER_ID` | root để auto-create `VinUni` nếu chưa có `VINUNI_FOLDER_ID` |
| `GOOGLE_DRIVE_VINUNI_FOLDER_ID` | folder `VinUni` hiện hữu |
| `GOOGLE_DRIVE_MAKE_PUBLIC` | auto set permission public-read cho file upload |
| `MCPT_OAUTH2_CREDENTIALS_FILE` | đường dẫn OAuth client json |
| `MCPT_OAUTH2_TOKEN_FILE` | token pickle |
| `GOOGLE_DRIVE_*_FOLDER_NAME` | tên folder con như `Queue`, `Import_New`, `.h265`, `Metadata` |

##### Nhóm Lightning / AI

| Biến | Ý nghĩa |
|---|---|
| `LIGHTNING_API_BASE_URL` | base URL LightningAI |
| `LIGHTNING_API_ENDPOINT` | endpoint mặc định ở tracking-service |
| `LIGHTNING_API_TOKEN` | bearer token hoặc token tương đương |
| `LIGHTNING_API_AUTH_HEADER` | tên header auth |
| `LIGHTNING_API_AUTH_PREFIX` | prefix auth, thường `Bearer` |
| `LIGHTNING_TIMEOUT_SECONDS` | timeout call remote |
| `TRACKING_SERVICE_PREFER_LOCAL` | metadata-service ưu tiên local tracking-service nếu sẵn |

##### Nhóm queue/storage

| Biến | Ý nghĩa |
|---|---|
| `QUEUE_LOCAL_ROOT` | local artifact root cho queue |
| `QUEUE_MAX_SIZE` | số video tối đa giữ trong queue |
| `QUEUE_POLL_INTERVAL_SECONDS` | chu kỳ poll Import_New |
| `VIDEO_STORAGE_ROOT` | local upload root cho video managed |
| `INGESTION_WORK_ROOT` | workspace ingest/tracking |
| `VIDEO_CONVERSION_OUTPUT_DIR` | output convert video |
| `VIDEO_DOWNLOAD_OUTPUT_DIR` | output download/copy tracking result |

#### 3.3.3. Các file env cần hiểu rõ

##### `Multi-Camera-Person-Tracking-and-Re-Identification/.env.example`

Đây là sample env gần runtime app nhất. Nó mô tả:

- ports
- service URL
- JWT
- Google Drive
- Lightning endpoint
- queue/storage

##### Root `.env.example`

Đây là canonical env cấp workspace, thêm:

- AI prompt logging hooks
- canonical secret root
- ghi chú production

##### Điểm mơ hồ/cấu hình lệch cần nêu thẳng

Trong code/docs có drift:

- README và compose sample dùng `../secrets/shared.env`
- workspace hiện tại thực tế đang có `secret/` chứ không phải `secrets/`
- `shared_secret_runtime.py` mặc định trỏ `secrets/`

Nghĩa là nếu clone mới và chạy theo README mà không sửa cấu trúc secret, app rất dễ không đọc đúng env/OAuth file.

### 3.4. Cách chạy local

### 3.4.1. Chạy local nhanh nhất bằng Docker Compose

```bash
cd Multi-Camera-Person-Tracking-and-Re-Identification
docker compose --env-file ../secret/shared.env up --build
```

Nếu bạn dùng đúng thư mục `secrets/` như README:

```bash
docker compose --env-file ../secrets/shared.env up --build
```

Service mặc định trong `docker-compose.yml`:

- `postgres`
- `metadata-service`
- `queue-worker`
- `api-gateway`
- `frontend`

`tracking-service` **không tự bật** vì nằm dưới profile `local-gpu`.

### 3.4.2. Port mapping mặc định

| Service | Port host | Port container |
|---|---|---|
| frontend | `3000` | `3000` |
| api-gateway | `8000` | `8000` |
| metadata-service | `8001` | `8000` |
| tracking-service | `8002` | `8000` |
| postgres | `5432` | `5432` |

### 3.4.3. Healthcheck cần thử

```bash
curl http://localhost:8000/api/v1/overview
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:3000
```

### 3.4.4. Dấu hiệu chạy đúng

- `frontend` mở được UI
- `/api/v1/overview` trả metric JSON
- đăng ký/đăng nhập thành công
- `queue/videos` trả danh sách hoặc `count = 0`
- nếu Drive bật và có OAuth đúng, `queue-worker` không crash

### 3.4.5. Chạy riêng từng service không qua Docker

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Metadata service:

```bash
cd backend/services/metadata-service
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

API gateway:

```bash
cd backend/services/api-gateway
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Queue worker:

```bash
cd backend/services/metadata-service
python -m app.queue_worker
```

Tracking service local:

```bash
cd backend/services/tracking-service
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

### 3.5. Cách chạy bằng Docker / Docker Compose

#### 3.5.1. `docker-compose.yml`

Đây là compose chuẩn nhất của repo hiện tại.

Snippet quan trọng:

```yaml
services:
  postgres:
    image: postgres:16-alpine
  metadata-service:
    build:
      dockerfile: ./backend/services/metadata-service/Dockerfile
  queue-worker:
    command: ["python", "-m", "app.queue_worker"]
  tracking-service:
    profiles:
      - local-gpu
  api-gateway:
    build:
      context: ./backend/services/api-gateway
  frontend:
    build:
      context: ./frontend
```

#### 3.5.2. Volumes quan trọng

| Volume | Công dụng |
|---|---|
| `postgres_data` | data DB |
| `storage_data` | storage chung cho metadata-service / queue-worker / tracking-service |

Bind mounts:

- `./data:/workspace/data:ro`
- `../:/workspace/a20-root:ro`

Điểm rất quan trọng:

- container mount cả repo root vào `/workspace/a20-root`
- secret runtime trong container kỳ vọng `MCPT_SECRETS_ROOT=/workspace/a20-root/secrets`
- nếu trên host bạn chỉ có `secret/`, container có thể không tìm đúng canonical path nếu không override env

#### 3.5.3. Cách xem logs

```bash
docker compose logs -f frontend
docker compose logs -f api-gateway
docker compose logs -f metadata-service
docker compose logs -f queue-worker
docker compose logs -f postgres
```

#### 3.5.4. Exec vào container

```bash
docker compose exec metadata-service sh
docker compose exec api-gateway sh
docker compose exec postgres psql -U mcpt_user -d mcpt
```

### 3.6. Cách chạy production-like ở local

#### Frontend production mode

```bash
cd frontend
npm install
npm run build
npm run start
```

#### Backend production-like

Dùng Docker Compose là gần production nhất vì Dockerfiles đều chạy `uvicorn` worker:

```bash
docker compose --env-file ../secret/shared.env up --build -d
```

#### Test end-to-end local

1. mở `http://localhost:3000`
2. đăng ký user hoặc login user có sẵn
3. kiểm tra `overview`
4. nếu DB có candidate, query text
5. nếu Drive bật, upload `.h265` vào `Import_New` hoặc bấm `Move`

## 4. Frontend Deep Dive

### 4.1. Framework và mode render

Frontend dùng:

- `Next.js 15.4.6`
- `React 19.1.1`
- `TypeScript`

`frontend/package.json`:

```json
{
  "scripts": {
    "dev": "next dev -H 0.0.0.0 -p 3000",
    "build": "next build",
    "start": "next start -H 0.0.0.0 -p 3000"
  }
}
```

`frontend/next.config.mjs` đang để `output: "standalone"`, phù hợp Docker/prod.

### 4.2. Entry point frontend

Entrypoint UI chính:

- `frontend/app/page.tsx`

Đây là một `client component`:

```tsx
"use client";
import { FormEvent, useEffect, useMemo, useState } from "react";
```

Điều đó có nghĩa:

- logic auth/query/build video chạy hoàn toàn ở client
- không dùng server actions
- không có route segment phức tạp

### 4.3. Router

Hiện frontend gần như là `single-page workspace`.

Có hai state màn hình chính:

- màn hình auth
- màn hình workspace sau login

Chưa có `app/login/page.tsx`, `app/query/page.tsx` tách route thật. UI tách bằng state trong `page.tsx`.

### 4.4. State management

State management chỉ dùng `useState`.

Không dùng:

- Redux
- Zustand
- Jotai
- React Query / TanStack Query

Các state chính trong `page.tsx`:

```tsx
const [token, setToken] = useState("");
const [user, setUser] = useState<User | null>(null);
const [overview, setOverview] = useState<Overview | null>(null);
const [queueVideos, setQueueVideos] = useState<QueueVideo[]>([]);
const [candidateQuery, setCandidateQuery] = useState("doctor carrying a medical box");
const [candidateResults, setCandidateResults] = useState<Candidate[]>([]);
const [selectedCandidateId, setSelectedCandidateId] = useState("");
const [trackingResult, setTrackingResult] = useState<TrackingResult | null>(null);
```

### 4.5. Token/auth flow trên frontend

Token được giữ trong `localStorage`:

```tsx
const TOKEN_KEY = "mcpt_access_token";

function persistToken(nextToken: string) {
  localStorage.setItem(TOKEN_KEY, nextToken);
  setToken(nextToken);
}
```

`apiFetch()` tự gắn bearer token:

```tsx
async function apiFetch(path: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers ?? {});
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers, cache: "no-store" });
  if (response.status === 401) {
    clearSession();
    throw new Error("Session expired. Please log in again.");
  }
  return response;
}
```

Nhận xét kỹ thuật:

- đơn giản, dễ hiểu
- nhưng không có refresh token
- không có secure cookie
- `localStorage` nghĩa là token có thể bị lộ nếu XSS

### 4.6. Data fetching

Frontend dùng `fetch` thuần.

`refreshWorkspace()` load song song:

```tsx
const [overviewResponse, meResponse, queueResponse] = await Promise.all([
  fetch(`${API_BASE}/api/v1/overview`, { cache: "no-store" }),
  apiFetch("/api/v1/auth/me"),
  fetch(`${API_BASE}/api/v1/queue/videos`, { cache: "no-store" }),
]);
```

Flow UI chính:

- login/register -> `/api/v1/auth/login` hoặc `/api/v1/auth/register`
- move -> `/api/v1/queue/process-imports`
- candidate search -> `/api/v1/candidates/search`
- build tracking -> `/api/v1/candidates/track`

### 4.7. Màn hình và business mapping

#### Màn hình auth

- submit login/register
- nhận `access_token`
- lưu token vào browser

#### Màn hình workspace

Bao gồm các khối:

1. overview system metrics
2. Move button cho `Import_New -> Queue`
3. text query form
4. top-k candidate cards
5. chọn candidate
6. build tracking output preview
7. queue video list

### 4.8. Preview candidate

Candidate có `preview_image_url` trả từ metadata-service:

```tsx
function candidatePreviewUrl(candidate: Candidate | null | undefined): string {
  if (!candidate) {
    return "";
  }
  return withApiBase(candidate.preview_image_url ?? null);
}
```

Điểm này bám với backend route:

- `GET /api/v1/candidates/{candidate_id}/preview`

### 4.9. Error/loading handling

- một state `loading`
- một state `message`
- một state `error`

Đây là cơ chế tối giản, chưa có:

- retry policy
- per-request status
- optimistic update
- caching layer

### 4.10. Nhận định frontend

Ưu điểm:

- rất trực tiếp
- dễ debug theo request flow
- ít abstraction

Hạn chế:

- toàn bộ logic dồn vào một file `page.tsx`
- chưa có component split rõ ràng
- token lưu ở `localStorage`
- chưa có route-level separation thật

## 5. Backend Deep Dive

## 5.1. Tổng thể backend

Backend gồm 3 service chính:

1. `api-gateway`
2. `metadata-service`
3. `tracking-service`

### 5.2. API Gateway

#### Entrypoint

- `backend/services/api-gateway/app/main.py`

Vai trò:

- làm public API cho frontend
- proxy request xuống metadata-service và tracking-service
- giữ cho frontend chỉ cần biết một base URL

#### Config

- `backend/services/api-gateway/app/config.py`

Snippet:

```python
class Settings(BaseSettings):
    metadata_service_url: str = "http://metadata-service:8000"
    tracking_service_url: str = "http://tracking-service:8000"
    cors_allowed_origins: str = "http://localhost:3000"
```

#### Vai trò kỹ thuật

- proxy auth routes sang metadata-service
- proxy candidate search/track
- tổng hợp `/api/v1/overview` từ nhiều service
- proxy artifact video/manifest

### 5.3. Metadata Service

#### Entrypoint

- `backend/services/metadata-service/app/main.py`

Startup:

```python
@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        sync_local_queue_state(session, only_if_empty=True)
```

Ý nghĩa rất quan trọng:

- schema DB được tạo tự động lúc service start
- nếu DB rỗng, service cố gắng rehydrate từ local queue metadata

#### Kiến trúc phân lớp

| Lớp | File |
|---|---|
| main/routes | `app/main.py` |
| auth util | `app/auth.py` |
| auth dependency | `app/deps.py` |
| config | `app/config.py` |
| database/session | `app/database.py` |
| ORM models | `app/models.py` |
| DTO/schema | `app/schemas.py` |
| business logic | `app/service.py` |
| queue runtime | `app/queue_runtime.py` |
| queue worker CLI | `app/queue_worker.py` |

#### Auth

- JWT bằng `python-jose`
- password hash bằng `passlib[bcrypt]`

`deps.py` dùng `HTTPBearer(auto_error=False)`, decode JWT lấy `sub = email`, rồi query user.

#### Route quan trọng

| Route | Vai trò |
|---|---|
| `/api/v1/auth/register` | tạo user |
| `/api/v1/auth/login` | login |
| `/api/v1/auth/me` | me |
| `/api/v1/overview` | metric |
| `/api/v1/candidates/search` | local shortlist + remote rerank |
| `/api/v1/candidates/track` | build tracking request |
| `/api/v1/candidates/{id}/preview` | render JPEG preview with bbox |
| `/api/v1/queue/videos` | catalog queue |
| `/api/v1/queue/process-imports` | trigger manual Move/import |
| `/api/v1/tracking-artifacts/*` | serve video/manifest local fallback |

#### Business logic đặc biệt quan trọng

##### `rank_candidates()`

File: `metadata-service/app/service.py`

```python
def rank_candidates(session: Session, query_text: str, limit: int = 5) -> list[dict]:
    ...
    rows = session.scalars(statement).all()
    ...
    candidates = _local_prefilter_ranked_candidates(...)
    response = _post_tracking_json(
        "/api/v1/candidates/search",
        {
            "query_text": cleaned_query,
            "candidates": candidates,
            "limit": bounded_limit,
        },
    )
```

Ý nghĩa:

- metadata-service **không tự làm semantic rank cuối cùng**
- nó lấy toàn bộ candidates trong DB
- prefilter local
- gửi shortlist sang tracking-service để rerank bằng embedding model

##### `build_candidate_preview_image()`

Hàm này:

- resolve queue video source path
- mở video bằng OpenCV
- seek tới frame hợp lý
- vẽ bbox + overlay text
- lưu JPEG local

Đây là backend phục vụ requirement preview image/bounding box trên UI.

##### `build_tracking_video()`

Workspace hiện tại đã có nhánh logic mới cho selected candidate/global shortlist. Tuy nhiên cần phân biệt:

- metadata-service assemble request
- tracking-service mới là nơi build artifact video thật

#### Queue worker

`queue_worker.py` chạy dưới metadata-service image:

```python
command: ["python", "-m", "app.queue_worker"]
```

Nó dùng `QueueSyncService` trong `queue_runtime.py`.

### 5.4. Tracking Service

#### Entrypoint

- `backend/services/tracking-service/app/main.py`

Route quan trọng:

```python
@app.post("/api/v1/ingestion/process")
def ingestion_process(payload: VideoIngestionRequest) -> dict:
    return process_video_ingestion(payload.model_dump())

@app.post("/api/v1/ingestion/upload")
async def ingestion_upload(...):
    ...
    return process_video_ingestion(...)

@app.post("/api/v1/candidates/search")
def candidate_search(payload: CandidateSearchRequest) -> dict:
    return search_candidates_remote(payload.query_text, payload.candidates, payload.limit)

@app.post("/api/v1/candidates/track")
def candidate_track(payload: CandidateTrackRequest) -> dict:
    return build_tracking_video_remote(...)
```

#### Config

`tracking-service/app/config.py` là file rất quan trọng, vì nó quyết định:

- root paths
- Google Drive
- ffmpeg
- profile pipeline
- GPU profile
- Lightning endpoint

#### Luồng `process_video_query()`

Hàm này xử lý AI process path cho query trực tiếp trên video.

Nó có 2 mode:

1. local mode
2. remote LightningAI mode

Tóm tắt:

- chuẩn bị query embedding/runtime profile
- nếu không có remote base URL thì chạy local
- nếu có remote Lightning thì prepare `.h265` và gọi `gpu_client`

#### Luồng `process_video_query_worker()`

Đây là worker thực tế nhận multipart upload từ `/api/v1/ai/worker`.

Nó:

1. ingest video thành `people`
2. encode text query bằng OpenCLIP
3. score từng person
4. trả matched candidates + matched segments

#### Luồng `search_candidates_remote()`

Đây là top-k reranker cho website:

```python
def search_candidates_remote(query_text: str, candidates: list[dict], limit: int = 5) -> dict:
    query_embedding = _compute_query_embedding(cleaned_query)
    embedding_scores = _precompute_candidate_embedding_scores(query_embedding, candidates)
```

Model text dùng OpenCLIP:

```python
model, _, _preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
```

#### Luồng `build_tracking_video_remote()`

Hàm hiện tại trong workspace:

- chọn shortlist từ candidate đã gửi lên
- resolve source video cho từng candidate
- trích clip theo `matched_segments`
- dùng `OpenCV VideoWriter` ghép thành `mp4`

Điểm phải nói thẳng:

- đây **không phải full re-detection/retracking toàn bộ raw dataset**
- nó gần với `clip stitching/composition` dựa trên metadata/timeline có sẵn hơn
- workspace hiện tại có thêm nhánh `anchor_similarity_gpu`, nhưng vẫn chưa phải “rerun detector/tracker full dataset” theo nghĩa mạnh nhất

### 5.5. HTTP flow chuẩn

Một request điển hình:

```text
Frontend fetch
  -> API Gateway route
  -> Metadata service route
      -> service.py business logic
      -> PostgreSQL query / queue local artifact
      -> optional call tracking-service
          -> OpenCLIP / ingestion runtime / LightningAI
  -> JSON response
  -> Frontend render
```

### 5.6. Error handling

Hiện hệ thống chủ yếu dùng:

- `response.raise_for_status()` ở upstream HTTP
- `HTTPException` trong FastAPI
- `try/except` bao business route chính

Chưa có:

- exception middleware tập trung
- correlation id
- tracing phân tán
- structured logging đồng bộ giữa services

## 6. Database Deep Dive

## 6.1. Hệ DB đang dùng

Database chính:

- `PostgreSQL`

Không thấy Redis, MongoDB, Kafka, RabbitMQ trong runtime code hiện tại.

`docker-compose.edge.yml` có biến `CHROMA_DB_PATH`, nhưng tracking-service code hiện tại không cho thấy ChromaDB là thành phần chạy chính của web flow. Đây có vẻ là dư âm hoặc option chưa hoàn thiện.

## 6.2. Kết nối DB nằm ở đâu?

### Metadata service

- `backend/services/metadata-service/app/database.py`
- engine tạo từ `settings.database_url`

### Exchange/bootstrap

- `exchange.py` tự build DB URL
- có logic tự start PostgreSQL local nếu host là localhost và DB chưa lên

### Tracking service

- có giữ các biến PostgreSQL trong config
- nhưng web flow hiện tại DB business chính nằm ở metadata-service

## 6.3. ORM / query layer

ORM dùng:

- `SQLAlchemy ORM`
- dialect `postgresql+psycopg`

Không dùng raw ORM framework như Django, Prisma, TypeORM.

`models.py` là schema nguồn chính.

## 6.4. Schema hiện có

### `users`

Dùng cho auth/login.

Trường chính:

- `email` unique
- `full_name`
- `hashed_password`

### `video_assets`

Dùng cho managed video upload/reference theo user.

Đây là flow cũ/scaffold hơn là queue chính hiện nay.

### `video_queries`

Lưu user query theo video managed.

Đây cũng nghiêng về flow cũ/scaffold.

### `person_candidates`

Bảng quan trọng nhất cho search.

Lưu:

- `candidate_id`
- `camera_id`
- `video_id`
- `track_id`
- `human_key`
- `frame_idx`
- `search_text`
- `metadata_path`
- `raw_metadata` JSONB

`raw_metadata` là payload giàu dữ liệu nhất, chứa:

- bbox
- semantic attributes
- timeline
- embeddings
- world position
- caption/summary

### `queue_video_assets`

Bảng catalog cho queue video.

Lưu:

- `video_id`
- `queue_position`
- `available_link_video`
- `available_link_metadata`
- `drive_video_file_id`
- `drive_metadata_file_id`
- `local_video_path`
- `local_metadata_path`
- `raw_video_metadata`

## 6.5. Quan hệ giữa các bảng

| Quan hệ | Kiểu |
|---|---|
| `users -> video_assets` | 1-N |
| `users -> video_queries` | 1-N |
| `video_assets -> video_queries` | 1-N |
| `person_candidates` | gần như bảng độc lập, gắn logic theo `video_id` string |
| `queue_video_assets` | gần như bảng độc lập, join logic theo `video_id` string |

Điểm cần chú ý:

- `person_candidates.video_id` và `queue_video_assets.video_id` không phải foreign key chuẩn
- join được thực hiện ở tầng service bằng map `video_id`

Điều này linh hoạt nhưng tăng nguy cơ inconsistency.

## 6.6. Unique constraint / index

Từ `models.py`:

- `users.email` unique
- `video_assets.video_id` unique
- `video_queries.query_id` unique
- `person_candidates.candidate_id` unique
- `queue_video_assets.video_id` unique

Nhiều cột có `index=True` như:

- `camera_id`
- `video_id`
- `track_id`
- `human_key`

## 6.7. Migration nằm ở đâu?

**Không có Alembic migration.**

Schema hiện được tạo bởi:

```python
Base.metadata.create_all(bind=engine)
```

Hệ quả:

- không có versioned migration history
- không có rollback migration chuẩn
- thay đổi schema prod cần cực kỳ cẩn thận

## 6.8. Cách tạo DB từ đầu

### Cách chuẩn với Docker Compose

```bash
cd Multi-Camera-Person-Tracking-and-Re-Identification
docker compose --env-file ../secret/shared.env up -d postgres
docker compose --env-file ../secret/shared.env up -d metadata-service
```

Khi metadata-service start:

- nó sẽ tự chạy `Base.metadata.create_all(...)`

### Cách local helper

`start_postgres.py` có thể dựng PostgreSQL bằng Docker nếu chưa chạy:

```bash
cd Multi-Camera-Person-Tracking-and-Re-Identification
python start_postgres.py
```

Script này:

- thử kết nối DB
- nếu DB chết, thử start container postgres cũ
- nếu chưa có, tự `docker run postgres:16-alpine`

## 6.9. Cách seed dữ liệu

Có 4 đường chính:

1. `queue-worker` ingest từ Google Drive `Import_New`
2. `exchange.py` bootstrap từ source local/Drive
3. startup `sync_local_queue_state()` rehydrate từ local metadata JSON
4. `/api/v1/candidates/import-legacy`

### Seed bằng `exchange.py`

`exchange.py` là script “full pipeline integration/bootstrap”:

```python
parser.add_argument("--video-file", ...)
parser.add_argument("--input-dir", ...)
parser.add_argument("--batch", action="store_true")
parser.add_argument("--no-drive", action="store_true")
parser.add_argument("--no-lightning", action="store_true")
```

Nó làm:

1. chọn input `.h265/.hevc/.mp4`
2. upload Drive nếu không `--no-drive`
3. generate metadata
4. save PostgreSQL
5. tùy chọn gọi LightningAI step

### Seed bằng queue worker

Đây là đường production logic hơn:

- detect file trong `VinUni/Import_New`
- download local tmp
- gọi `tracking-service /api/v1/ingestion/upload`
- cập nhật local metadata + DB + upload `.h265` lên `Queue/.h265`

## 6.10. Cách query DB thực chiến

Đăng nhập PostgreSQL trong container:

```bash
docker compose exec postgres psql -U mcpt_user -d mcpt
```

Hoặc:

```sql
\dt
SELECT count(*) FROM users;
SELECT count(*) FROM person_candidates;
SELECT count(*) FROM queue_video_assets;
SELECT candidate_id, camera_id, track_id FROM person_candidates LIMIT 10;
SELECT video_id, queue_position, available_link_video FROM queue_video_assets ORDER BY queue_position;
```

## 6.11. Backup / restore cơ bản

Backup:

```bash
pg_dump -h localhost -p 5432 -U mcpt_user -d mcpt > mcpt_backup.sql
```

Restore:

```bash
psql -h localhost -p 5432 -U mcpt_user -d mcpt < mcpt_backup.sql
```

## 6.12. Điểm có nguy cơ inconsistency

### 1. Join bằng string thay vì FK

`person_candidates.video_id` <-> `queue_video_assets.video_id`

Nếu import/update xô lệch, candidate có thể không tìm được queue video.

### 2. Không có migration framework

Schema drift giữa local/prod rất dễ xảy ra.

### 3. Queue update và file system update không transactionally atomic

Ví dụ:

- upload Drive thành công
- ghi DB fail
- local metadata còn dang dở

=> rất dễ có orphan file hoặc orphan row.

### 4. Startup sync “only_if_empty”

Nếu DB đã có row nhưng local metadata mới thay đổi, startup sync không tự reconcile toàn bộ.

## 6.13. Redis / cache / vector DB?

- Không thấy Redis trong runtime chính
- Không thấy vector DB production path thật
- Search hiện dựa vào:
  - `person_candidates` trong PostgreSQL
  - embedding vector nhúng trong `raw_metadata`
  - rerank ở tracking-service bằng OpenCLIP/NumPy/Torch

Tức là repo hiện tại chưa phải kiến trúc vector DB riêng biệt.

## 7. AI / Model / ML / LLM Flow

## 7.1. Có model nào đang dùng?

Từ code hiện tại:

- OpenCLIP `ViT-B-32` pretrained `openai` cho text-image embedding/rerank
- legacy hospital pipeline trong `backend/legacy-engine`
- VLM metadata engine trong `src_vlm/vlm_engine.py`

Không thấy OpenAI/Anthropic/Gemini trong web flow inference chính.

Các API key ở root `.env.example` chủ yếu là generic template/hook logging, không phải engine AI chính của app này.

## 7.2. Model chạy ở đâu?

### Text query rerank

Chạy trong `tracking-service/app/service.py`

```python
import open_clip
model, _, _preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
```

Nếu có CUDA:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Video ingestion / metadata generation

Chạy qua:

- `tracking-service/app/ingestion_runtime.py`
- `backend/legacy-engine/src_vlm/hospital_pipeline.py`
- `backend/legacy-engine/src_vlm/vlm_engine.py`

### Remote LightningAI

`tracking-service` có thể gọi remote qua `gpu_client.py`.

`metadata-service/queue_runtime.py` cũng có thể gửi ingestion trực tiếp tới endpoint remote bằng:

- `/api/v1/ingestion/process`
- `/api/v1/ingestion/upload`

## 7.3. Prompting / prompt template?

Repo này không phải ứng dụng LLM chat điển hình.

Không có hệ prompt template rõ ràng như:

- system prompts
- prompt registry
- conversation memory

AI ở đây chủ yếu là:

- CV pipeline
- embedding/retrieval/rerank
- VLM caption/metadata

## 7.4. Input / output theo từng luồng AI

### A. Ingestion

Input:

- `source_path` hoặc `source_drive_file_id`
- metadata như `camera_id`, `recorded_start`

Output:

- `video`
- `people`
- `person_count`
- `compressed_path`
- `metadata_path`
- hardware profile
- acceleration state

Schema:

- `tracking-service/app/schemas.py -> VideoIngestionRequest/Response`

### B. Candidate search rerank

Input:

- `query_text`
- shortlist `candidates`

Output:

- `items` đã được score
- `matched_segments`

### C. Build tracking artifact

Input:

- `selected_candidate_id`
- candidate shortlist
- query text

Output:

- `artifact_id`
- `video_url`
- `manifest_url`
- manifest JSON

## 7.5. Có retrieval / embeddings / reranking không?

Có.

### Retrieval thực tế

- DB scan candidate rows
- local lexical/semantic prefilter trong metadata-service
- OpenCLIP embedding rerank trong tracking-service

### Embeddings ở đâu?

Embeddings không lưu trong vector DB riêng.

Chúng thường được giữ trong `raw_metadata` của `person_candidates` dưới các key như:

- `embedding_vector`
- `candidate_vector`
- `itself_features`

## 7.6. Có memory / session / streaming?

- không có conversation memory
- không có streaming response
- session chỉ là JWT auth user session

## 7.7. Có retry / timeout / fallback?

Có timeout ở HTTP level, ví dụ:

- `LIGHTNING_TIMEOUT_SECONDS`
- `tracking_request_timeout_seconds`

Có fallback cục bộ ở một số điểm:

- prefer local tracking-service nếu `TRACKING_SERVICE_PREFER_LOCAL=true`
- build query mode local nếu không có `LIGHTNING_API_BASE_URL`

Nhưng hệ thống **không có retry framework chuẩn** cho Lightning/Drive/DB.

## 8. Main Flow và Sub-flow chi tiết

## 8.1. Main flow A: user login

### Điểm bắt đầu

- frontend `page.tsx -> handleAuthSubmit()`

### Dữ liệu vào

- `identifier/password` hoặc `full_name/email/password`

### Xử lý

1. frontend POST tới `/api/v1/auth/login` hoặc `/register`
2. api-gateway proxy xuống metadata-service
3. metadata-service:
   - validate payload qua Pydantic schema
   - `authenticate_user()` hoặc `create_user()`
   - `create_access_token(user.email)`
4. JWT trả về frontend
5. frontend lưu `localStorage`

### Lỗi có thể gặp

- email duplicate
- password sai
- JWT secret thiếu
- DB chưa lên

### Cách debug

```bash
docker compose logs -f metadata-service
curl -X POST http://localhost:8000/api/v1/auth/login -H "Content-Type: application/json" -d "{\"identifier\":\"admin\",\"password\":\"admin\"}"
```

## 8.2. Main flow B: Import_New -> Queue tự động

### Điểm bắt đầu

- file `.h265/.hevc` xuất hiện trong Google Drive `VinUni/Import_New`
- hoặc user bấm `Move`

### File/hàm xử lý

- `frontend/app/page.tsx -> handleMove()`
- `metadata-service/app/main.py -> process_imports()`
- `metadata-service/app/queue_runtime.py -> QueueSyncService.process_import_queue()`

### Luồng chi tiết

1. `ensure_drive_layout()` resolve folder IDs
2. `list_import_files()` query file trong `Import_New`
3. `_process_import_drive_item()`:
   - download file từ Drive về local tmp
   - gọi `_request_tracking_processing_upload()`
4. `_request_tracking_processing_upload()` POST multipart tới:
   - `TRACKING_SERVICE_URL/api/v1/ingestion/upload`
5. tracking-service ingest video, trả metadata
6. queue runtime:
   - move local `.h265` sang queue local
   - ghi local metadata JSON
   - upload `.h265` lên Drive `Queue/.h265`
   - upsert `queue_video_assets`
   - upsert `person_candidates`
7. xóa file cũ nếu queue vượt `QUEUE_MAX_SIZE`

### Kết quả đầu ra

- Drive có `.h265` trong `Queue/.h265`
- DB có `queue_video_assets`
- DB có `person_candidates`
- local có metadata JSON + cached video file

### Edge cases

- endpoint remote trả 404/500
- OAuth token hết hạn
- thiếu folder ID
- metadata path ghi được nhưng DB commit fail

### Failure modes

- import file được download nhưng chưa move
- row DB bị thiếu dù file local tồn tại
- Drive upload thành công nhưng metadata local mất

## 8.3. Main flow C: text query -> top-k candidate

### Điểm bắt đầu

- frontend `handleSearch()`

### Luồng

1. frontend POST `/api/v1/candidates/search`
2. api-gateway proxy sang metadata-service
3. metadata-service `rank_candidates()`:
   - load toàn bộ `PersonCandidate`
   - tạo `queue_map`
   - local prefilter shortlist
   - POST shortlist sang tracking-service `/api/v1/candidates/search`
4. tracking-service:
   - compute text embedding bằng OpenCLIP
   - score từng candidate
   - normalize matched segments
   - sort
5. response trả lại frontend
6. frontend render top-k cards + preview image

### DB interaction

- metadata-service đọc `person_candidates`
- metadata-service đọc `queue_video_assets`

### External/model interaction

- tracking-service chạy OpenCLIP local hoặc trên GPU local nếu có CUDA

### Debug

```bash
curl -X POST http://localhost:8000/api/v1/candidates/search \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d "{\"query_text\":\"person with backpack\",\"limit\":5}"
```

## 8.4. Main flow D: chọn candidate -> build tracking video

### Điểm bắt đầu

- frontend `handleBuildTracking()`

### Luồng

1. frontend gửi:
   - `selected_candidate_id`
   - `candidate_ids`
   - `query_text`
2. metadata-service assemble payload
3. metadata-service gọi tracking-service `/api/v1/candidates/track`
4. tracking-service:
   - shortlist selected/global matches
   - resolve source video path/URL/Drive file
   - mở video bằng OpenCV
   - cắt clip theo `matched_segments`
   - ghi `mp4`
   - ghi manifest JSON
5. artifact URL trả lại frontend
6. frontend phát preview video output

### Điểm cần hiểu đúng

Code hiện tại không phải full “rerun detector/tracker trên toàn bộ dataset raw” theo nghĩa mạnh nhất.

Nó đang là:

- candidate-based global shortlist/rerank
- sau đó stitch clip từ source video

Nếu muốn đúng 100% yêu cầu “re-track object trên toàn bộ tập dữ liệu”, tracking-service sẽ cần một job GPU/global re-identification job nặng hơn nữa.

## 8.5. Main flow E: bootstrap DB bằng `exchange.py`

### Điểm bắt đầu

```bash
python exchange.py --input-dir ./videos --batch
```

### Luồng

1. chọn file video
2. upload Drive nếu bật
3. generate metadata
4. save PostgreSQL
5. optional LightningAI step

### Điểm mạnh

- tiện để batch rebuild DB
- tích hợp đủ Drive + DB + metadata pipeline

### Điểm yếu

- là script “integration/bootstrap”, không phải service runtime thuần
- có một số mô tả trong docstring cũ hơn thực tế service flow

## 9. Deployment / VPS / Server Running

## 9.1. Repo hiện đang được deploy theo kiểu gì?

Theo code và script hiện có, kiểu deploy chuẩn là:

- Docker trên VPS
- nginx chạy ở host
- reverse proxy:
  - `/` -> frontend
  - `/api/` -> api-gateway
- `tracking-service` local là optional
- production thường trỏ `TRACKING_SERVICE_URL` sang LightningAI public endpoint

Không thấy:

- PM2
- systemd unit file cho app chính
- Kubernetes
- Helm

Nhưng có:

- GitHub Actions deploy qua SCP + SSH

## 9.2. Script deploy hiện có

### `infra/vps/provision.sh`

Cài:

- docker
- docker compose plugin
- nginx
- certbot

### `infra/vps/deploy.sh`

```bash
docker compose --env-file "$SECRETS_ENV" down --remove-orphans || true
docker compose --env-file "$SECRETS_ENV" build metadata-service api-gateway frontend
docker compose --env-file "$SECRETS_ENV" up -d postgres metadata-service queue-worker api-gateway frontend
```

Nhận xét:

- không rebuild `tracking-service` vì mặc định prod dùng remote Lightning
- deploy script đang tập trung vào VPS web stack

### GitHub Actions

`.github/workflows/mcpt-ci-cd.yml`:

- build Docker stack
- copy app lên VPS
- SSH chạy `infra/vps/deploy.sh`

## 9.3. Hướng dẫn deploy VPS từ đầu

### Bước 1: chuẩn bị VPS

Khuyến nghị:

- Ubuntu 22.04
- tối thiểu 4 vCPU / 8 GB RAM cho web stack nhẹ
- nhiều hơn nếu chạy local tracking-service

### Bước 2: cài package hệ thống

```bash
apt update
apt install -y git
```

Clone repo:

```bash
mkdir -p /opt/mcpt
cd /opt/mcpt
git clone <repo-url> app
cd app/Multi-Camera-Person-Tracking-and-Re-Identification
```

### Bước 3: provision server

```bash
bash infra/vps/provision.sh
```

### Bước 4: chuẩn bị secret

Tạo thư mục secret:

```bash
mkdir -p /opt/mcpt/secrets/oauth
```

Copy:

- `shared.env`
- `oauth2_credentials.json`
- `oauth2_token.pickle`

Lưu ý:

- code mặc định mong đợi `secrets/`
- nếu server của bạn dùng `secret/`, phải sửa `MCPT_SECRETS_ROOT` hoặc `SECRETS_ENV`

### Bước 5: chỉnh env production

Ít nhất phải set:

- `POSTGRES_PASSWORD`
- `JWT_SECRET_KEY`
- `TRACKING_SERVICE_URL`
- `LIGHTNING_API_BASE_URL`
- `LIGHTNING_API_TOKEN`
- `GOOGLE_DRIVE_ENABLED`
- `GOOGLE_DRIVE_VINUNI_FOLDER_ID` hoặc `GOOGLE_DRIVE_ROOT_FOLDER_ID`
- `MCPT_OAUTH2_CREDENTIALS_FILE`
- `MCPT_OAUTH2_TOKEN_FILE`

### Bước 6: deploy app

```bash
cd /opt/mcpt/app/Multi-Camera-Person-Tracking-and-Re-Identification
bash infra/vps/deploy.sh "$(pwd)" "/opt/mcpt/secrets/shared.env"
```

### Bước 7: cấu hình nginx

```bash
bash infra/vps/configure_nginx.sh your-domain.example.com 3000 8000
```

Template nginx:

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:__API_GATEWAY_PORT__;
}

location / {
    proxy_pass http://127.0.0.1:__FRONTEND_PORT__;
}
```

### Bước 8: bật HTTPS

```bash
bash infra/vps/enable_https.sh your-domain.example.com your-email@example.com
```

### Bước 9: mở firewall

```bash
ufw allow 80
ufw allow 443
ufw enable
```

## 9.4. Chạy app bền vững trên VPS

Hiện repo thiên về:

- Docker restart policies
- nginx system service

Nếu muốn bền hơn:

- để `restart: unless-stopped` cho container chính
- backup volume `postgres_data`, `storage_data`
- tách Postgres ra container/server riêng nếu prod nghiêm túc

## 9.5. Deploy bằng Docker trên VPS

Build:

```bash
docker compose --env-file /opt/mcpt/secrets/shared.env build metadata-service api-gateway frontend
```

Run:

```bash
docker compose --env-file /opt/mcpt/secrets/shared.env up -d postgres metadata-service queue-worker api-gateway frontend
```

Xem logs:

```bash
docker compose logs -f metadata-service
docker compose logs -f queue-worker
docker compose logs -f api-gateway
docker compose logs -f frontend
```

Rollback cơ bản:

- checkout commit cũ
- rebuild image cũ
- `docker compose up -d --build`

## 9.6. Topology production khuyến nghị

```text
Internet
  -> Nginx host
      -> frontend container
      -> api-gateway container
          -> metadata-service container
              -> postgres container
              -> storage volume
              -> queue-worker container
          -> remote tracking-service / LightningAI
  -> Google Drive
```

## 9.7. Troubleshooting trên VPS

### App local chạy được, VPS lỗi

Kiểm tra:

- env có đủ không
- `MCPT_SECRETS_ROOT` đúng chưa
- nginx proxy đúng port chưa
- container có mount đúng repo root không

### DB connection refused

```bash
docker compose ps
docker compose logs postgres
docker compose exec postgres pg_isready -U mcpt_user -d mcpt
```

### Migration/schema mismatch

Vì không có Alembic, nếu code mới thêm field nhưng DB cũ không có cột, cần:

- tự ALTER TABLE
- hoặc drop/recreate schema có kiểm soát

### Reverse proxy sai

```bash
nginx -t
systemctl status nginx
journalctl -u nginx -f
```

### Lightning/SSL sai

Kiểm tra:

- `TRACKING_SERVICE_URL`
- `LIGHTNING_API_*`
- khả năng certificate chain/CloudSpaces endpoint

### Worker không chạy

```bash
docker compose logs -f queue-worker
```

Quan sát:

- có poll Import_New không
- có lỗi OAuth token không
- có lỗi 404/500 từ tracking-service không

## 10. Lệnh cụ thể và script discovery

## 10.1. Các lệnh quan trọng nhất nên nhớ

### Hằng ngày cho dev

```bash
docker compose --env-file ../secret/shared.env up --build
docker compose logs -f metadata-service
docker compose logs -f queue-worker
docker compose exec postgres psql -U mcpt_user -d mcpt
```

### Batch bootstrap

```bash
python exchange.py --input-dir ./videos --batch --output exchange_results.json
```

### Bootstrap queue từ source local

```bash
docker compose run --rm queue-worker python -m app.queue_worker --bootstrap --source-dir /workspace/data/... --once
```

### Chạy local GPU tracking-service

```bash
COMPOSE_PROFILES=local-gpu docker compose up -d --build tracking-service
```

### Edge Docker local GPU

```bash
docker compose -f docker-compose.edge.yml up --build
```

## 10.2. Những lệnh được khai báo ở đâu?

| Lệnh | Nơi khai báo |
|---|---|
| `next dev/build/start` | `frontend/package.json` |
| `uvicorn ...` | Dockerfiles từng service |
| `python -m app.queue_worker` | `docker-compose.yml` |
| deploy VPS | `infra/vps/deploy.sh` |
| provision VPS | `infra/vps/provision.sh` |
| CI/CD | `.github/workflows/mcpt-ci-cd.yml` |

## 10.3. Lệnh nào là chuẩn nhất?

| Mục tiêu | Lệnh chuẩn |
|---|---|
| chạy cả stack | `docker compose --env-file ... up --build` |
| debug từng backend service | `docker compose logs -f <service>` |
| bootstrap dữ liệu nhiều file | `exchange.py --batch` |
| ingest queue production-style | `queue-worker` |
| deploy VPS | `infra/vps/deploy.sh` |

## 10.4. Lệnh legacy / ít nên dùng

- chạy trực tiếp root `src/agent.py`: không phải app chính
- `ARCHITECTURE.md` mô tả SQLite/FAISS/vector flow khái niệm, không phản ánh đầy đủ website runtime hiện tại

## 11. Debugging, Logging, Observability

## 11.1. Log sinh ra ở đâu?

### Docker logs

Đây là nguồn log chính.

```bash
docker compose logs -f frontend
docker compose logs -f api-gateway
docker compose logs -f metadata-service
docker compose logs -f queue-worker
docker compose logs -f tracking-service
```

### Host nginx logs

```bash
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### Không thấy observability stack đầy đủ

Hiện repo không tích hợp sẵn:

- Sentry
- Prometheus
- Grafana
- OpenTelemetry

Logging chủ yếu là stdout/stderr.

## 11.2. Cách trace một request từ frontend đến model

Ví dụ query top-k:

1. Browser Network tab xem `POST /api/v1/candidates/search`
2. log `api-gateway`
3. log `metadata-service`
4. log `tracking-service`
5. nếu remote Lightning, kiểm tra endpoint remote / proxy log tương ứng

## 11.3. Debug frontend

Kiểm tra:

- `NEXT_PUBLIC_API_GATEWAY_URL`
- browser console
- network response body
- xem có stale bundle/browser cache không

## 11.4. Debug backend

### Metadata-service

Các lỗi thường gặp:

- DB chưa chạy
- JWT secret rỗng
- queue local path không tồn tại
- upstream tracking-service timeout/500

### Tracking-service

Các lỗi thường gặp:

- thiếu ffmpeg
- thiếu GPU/CUDA
- OpenCLIP không tải model
- remote source path không resolve được

## 11.5. Debug Google Drive

Kiểm tra:

- `GOOGLE_DRIVE_ENABLED=true`
- OAuth credentials/token path đúng
- root/VinUni folder ID đúng
- token còn valid

Các file chính:

- `shared_secret_runtime.py`
- `metadata-service/app/queue_runtime.py`
- `tracking-service/app/ingestion_runtime.py`

## 11.6. Debug DB

```bash
docker compose exec postgres psql -U mcpt_user -d mcpt -c "\\dt"
docker compose exec postgres psql -U mcpt_user -d mcpt -c "SELECT count(*) FROM person_candidates;"
docker compose exec postgres psql -U mcpt_user -d mcpt -c "SELECT count(*) FROM queue_video_assets;"
```

## 11.7. Debug AI/model call

### Check ingestion endpoint

```bash
curl -X POST http://localhost:8002/api/v1/ingestion/process \
  -H "Content-Type: application/json" \
  -d "{\"source_path\":\"/path/to/file.h265\",\"camera_id\":\"Camera_01\"}"
```

### Check pipeline config

```bash
curl http://localhost:8002/api/v1/pipeline/config
curl http://localhost:8002/api/v1/pipeline/hardware
```

## 12. Bảo mật, hiệu năng, vận hành thực tế

## 12.1. Security

### Điểm nhạy cảm

1. Secret thật đang được quản lý qua file env/OAuth local.
2. Token auth user nằm trong `localStorage`.
3. `GOOGLE_DRIVE_MAKE_PUBLIC=true` sẽ làm file Drive có thể public-read.
4. Một số env deploy/secret file trong workspace chứa giá trị nhạy cảm thật.

Khuyến nghị:

- tách secret ra secret manager hoặc ít nhất file ngoài repo
- rotate secret nếu repo từng bị lộ
- chuyển JWT sang secure cookie nếu muốn tăng an toàn frontend
- hạn chế `GOOGLE_DRIVE_MAKE_PUBLIC` nếu không thực sự cần

## 12.2. Performance bottleneck

### 1. `rank_candidates()` scan toàn bộ `PersonCandidate`

Hiện tại metadata-service đọc toàn bộ rows rồi mới shortlist. Với hàng chục/hàng trăm nghìn candidates, đây sẽ là bottleneck lớn.

### 2. Artifact build dùng OpenCV tuần tự

`build_tracking_video_remote()` đọc video/cắt frame/ghi video. Khi dataset lớn hoặc source nhiều, IO và CPU sẽ nặng.

### 3. Queue worker và Drive IO

Download/upload Drive là điểm chậm tự nhiên.

### 4. Không có cache

Không có Redis cache cho:

- overview
- candidate preview
- top-k query
- artifact manifest

## 12.3. Flow AI nào có thể chậm/tốn tài nguyên?

- ingestion full video
- OpenCLIP embedding trên candidate shortlist lớn
- global shortlist/anchor similarity khi số candidate nhiều

## 12.4. Giới hạn khi chạy trên 1 VPS

- metadata-service + queue-worker + frontend + api-gateway + postgres cùng máy
- nếu thêm tracking-service local GPU thì máy phải đủ mạnh
- storage volume local tăng nhanh
- không có queue broker, scale ngang khó

## 12.5. Nếu scale lên nhiều instance cần gì?

1. tách PostgreSQL ra managed DB hoặc máy riêng
2. tách storage local sang object storage
3. thêm hàng đợi thực như RabbitMQ/Kafka/Redis queue
4. đưa preview/artifact sang blob/object store
5. thêm migration framework
6. thêm metrics/tracing
7. candidate search nên chuyển sang hybrid index hoặc vector store chuyên dụng

## 12.6. Độ lệch giữa tài liệu và code

Đây là phần rất quan trọng khi onboarding:

### `ARCHITECTURE.md`

- mô tả kiến trúc khái niệm thiên về offline-first + FAISS + SQLite/PostgreSQL
- không khớp hoàn toàn với website runtime hiện nay

### `README.md`

- đúng ở mức service overview
- nhưng chưa phản ánh đầy đủ drift mới như metadata local/DB-first, preview image, selected candidate tracking flow

### Queue metadata

- README/doc cũ còn mô tả upload metadata lên Drive `Queue/Metadata`
- code `queue_runtime.py` hiện tại đã thiên về local metadata + `drive_metadata_file_id = None`
- nhưng `exchange.py` vẫn còn docstring và flow cũ hơn, có bước upload metadata lên Drive nếu bật Drive

Khi đọc repo, phải coi đây là dấu hiệu “hệ thống đang trong quá trình tái kiến trúc”.

## 13. Kết luận thực dụng

### 13.1. Cách nhanh nhất để chạy repo

```bash
cd Multi-Camera-Person-Tracking-and-Re-Identification
docker compose --env-file ../secret/shared.env up --build
```

Sau đó mở:

- `http://localhost:3000`
- `http://localhost:8000/api/v1/overview`

### 13.2. Cách đúng nhất để setup cho dev mới

1. đọc `shared_secret_runtime.py`
2. đọc `docker-compose.yml`
3. đọc `metadata-service/app/main.py`
4. đọc `metadata-service/app/queue_runtime.py`
5. đọc `tracking-service/app/main.py`
6. đọc `tracking-service/app/service.py`
7. đọc `frontend/app/page.tsx`

### 13.3. Cách tốt nhất để deploy lên VPS

1. dùng Docker Compose
2. dùng nginx host reverse proxy
3. để `tracking-service` remote qua LightningAI
4. giữ Postgres + storage trên VPS web stack
5. quản lý secret ngoài repo

### 13.4. Những lệnh quan trọng nhất cần nhớ

```bash
docker compose --env-file ../secret/shared.env up --build
docker compose logs -f metadata-service
docker compose logs -f queue-worker
docker compose exec postgres psql -U mcpt_user -d mcpt
python exchange.py --input-dir ./videos --batch
COMPOSE_PROFILES=local-gpu docker compose up -d --build tracking-service
```

### 13.5. Những file quan trọng nhất cần đọc đầu tiên

1. `shared_secret_runtime.py`
2. `Multi-Camera-Person-Tracking-and-Re-Identification/docker-compose.yml`
3. `backend/services/metadata-service/app/queue_runtime.py`
4. `backend/services/metadata-service/app/service.py`
5. `backend/services/tracking-service/app/service.py`
6. `backend/services/tracking-service/app/ingestion_runtime.py`
7. `frontend/app/page.tsx`
8. `infra/vps/deploy.sh`

### 13.6. Những điểm dễ lỗi nhất cần kiểm tra đầu tiên

1. secret path `secret/` vs `secrets/`
2. PostgreSQL host/password/db name
3. `TRACKING_SERVICE_URL` có thực sự reachable không
4. Google Drive OAuth file có đúng path không
5. `GOOGLE_DRIVE_VINUNI_FOLDER_ID` / `GOOGLE_DRIVE_ROOT_FOLDER_ID`
6. DB có dữ liệu `person_candidates` và `queue_video_assets` chưa
7. Lightning endpoint có đúng contract giữa:
   - `/api/v1/ingestion/process`
   - `/api/v1/ingestion/upload`
   - `/api/v1/ai/worker`
   - `/api/v1/candidates/search`
   - `/api/v1/candidates/track`

## 14. Các phần còn mơ hồ hoặc cần kiểm tra thêm

### 14.1. `secret/` vs `secrets/`

Workspace hiện tại dùng `secret/`, nhưng code mặc định nghiêng về `secrets/`. Đây là chỗ đầu tiên cần normalize.

### 14.2. Build tracking hiện chưa phải full global re-tracking

Code workspace đã tiến gần hơn tới selected-candidate global shortlist, nhưng nếu yêu cầu cuối là “re-track object full dataset bằng GPU nặng”, cần một job GPU chuyên biệt hơn.

### 14.3. Tài liệu khái niệm vs runtime hiện tại

`ARCHITECTURE.md` và một số docs cũ mang tính định hướng kiến trúc, không thể dùng một mình để triển khai/debug production hiện tại.

### 14.4. Migration/story schema dài hạn

Repo hiện chưa có chiến lược migration production-grade. Trước khi scale hoặc onboard team lớn, nên thêm Alembic và quy ước migration rõ ràng.

---

## Phụ lục A: Snippet code quan trọng

### A.1. Metadata service startup auto-create schema

```python
@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        sync_local_queue_state(session, only_if_empty=True)
```

### A.2. Queue import gọi ingestion upload

```python
endpoint = f"{endpoint_root}/api/v1/ingestion/upload"
with source_path.open("rb") as handle:
    files = {"file": (source_filename, handle, "video/h265")}
    with httpx.Client(timeout=float(settings.tracking_request_timeout_seconds)) as client:
        response = client.post(endpoint, data=data, files=files, headers=headers or None)
```

### A.3. Candidate search remote rerank

```python
def search_candidates_remote(query_text: str, candidates: list[dict], limit: int = 5) -> dict:
    query_embedding = _compute_query_embedding(cleaned_query)
    embedding_scores = _precompute_candidate_embedding_scores(query_embedding, candidates)
```

### A.4. Candidate preview vẽ bbox

```python
if bbox is not None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 120), 3)
```

### A.5. Frontend build tracking request

```tsx
const response = await apiFetch("/api/v1/candidates/track", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    selected_candidate_id: selectedCandidateId,
    candidate_ids: candidateResults.map((item) => item.candidate_id),
    query_text: candidateQuery,
    max_segments_per_candidate: 2,
  }),
});
```

