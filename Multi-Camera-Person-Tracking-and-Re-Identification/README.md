# MCPT Video AI Platform

Website quan ly video tich hop AI duoc tach theo kien truc microservice:

- `frontend/`: Next.js cho end-user
- `backend/services/api-gateway/`: FastAPI gateway cho domain/frontend
- `backend/services/metadata-service/`: FastAPI cho auth, video, text query, PostgreSQL
- `backend/services/tracking-service/`: AI orchestration service goi LightningAI GPU
- `backend/legacy-engine/`: legacy ReID code duoc giu lai de tai su dung
- `infra/postgres/`: khoi tao PostgreSQL

## Chuc nang da scaffold

- Dang ky va dang nhap tai khoan bang JWT
- Upload video vao local Docker volume hoac dang ky URL/path san co
- PostgreSQL chi luu metadata va duong dan `storage_path`
- Tao text query theo video
- API gateway goi AI service, AI service goi LightningAI endpoint
- Mock mode de local dev van chay du khi chua co GPU endpoint that
- Docker compose cho Next.js, FastAPI, PostgreSQL
- GitHub Actions scaffold cho CI/CD deploy VPS

## Kien truc

```text
Next.js
   |
   v
API Gateway (FastAPI)
   |
   +--> Metadata Service (FastAPI + PostgreSQL)
   |
   +--> Tracking / AI Service (FastAPI -> LightningAI GPU)
```

## Chay local

```bash
docker compose up --build
```

Mac dinh:

- Frontend: `http://localhost:3000`
- API Gateway: `http://localhost:8000`
- Metadata Service: `http://localhost:8001`
- Tracking Service: `http://localhost:8002`
- PostgreSQL: `localhost:5432`

## Bien moi truong quan trong

Xem [`.env.example`](./.env.example).

Canh bao:

- `TRACKING_USE_MOCK=true` se tra ket qua AI gia lap
- Muon goi LightningAI that, set:
  - `TRACKING_USE_MOCK=false`
  - `LIGHTNING_API_BASE_URL`
  - `LIGHTNING_API_ENDPOINT`
  - `LIGHTNING_API_TOKEN`

## Queue Google Drive va xu ly video

Kien truc queue da duoc tach thanh 2 vai tro:

- `queue-worker`: poll `VinUni/Import_New`, download file, goi `tracking-service`, upload `.h265` va metadata len `VinUni/Queue`, xu ly FIFO, cap nhat PostgreSQL.
- `tracking-service`: nhan job ingestion video qua `POST /api/v1/ingestion/process`, chiu trach nhiem detect / re-id / sinh metadata chi tiet.

API moi o metadata-service:

- `GET /api/v1/queue/videos`
- `POST /api/v1/queue/bootstrap`
- `POST /api/v1/queue/process-imports`

Chay bootstrap tu folder nguon local:

```bash
docker compose run --rm queue-worker python -m app.queue_worker --bootstrap --source-dir /workspace/data/NVIDIA_SmartSpaces/MTMC_Tracking_2025/val/Hospital_000/videos --once
```

Neu muon xoa source `.mp4` local sau khi da convert va dua len Queue Drive:

```bash
docker compose run --rm queue-worker python -m app.queue_worker --bootstrap --source-dir /workspace/data/NVIDIA_SmartSpaces/MTMC_Tracking_2025/val/Hospital_000/videos --delete-source --once
```

Dong bo `Import_New` lien tuc:

```bash
docker compose up -d queue-worker
```

Luu y:

- `GOOGLE_DRIVE_CREDENTIALS_FILE` phai tro dung file service account va folder Drive phai share quyen cho service account.
- Neu `TRACKING_USE_MOCK=false` va trong `tracking-service` co model/GPU day du, metadata se duoc sinh boi processor thuc te trong service nay.

## CI/CD VPS

Workflow mau nam tai [`.github/workflows/mcpt-ci-cd.yml`](../.github/workflows/mcpt-ci-cd.yml).

Can cung cap GitHub Secrets sau:

- `VPS_HOST`
- `VPS_PORT`
- `VPS_USERNAME`
- `VPS_SSH_KEY`
- `VPS_APP_DIR`

Pipeline hien tai:

1. Build Docker stack
2. Copy source len VPS
3. Chay `docker compose up -d --build` tren VPS

## Deploy production len `search-engine-119.smartnovi.tech`

Repo da co san bo script VPS va nginx host-level de deploy end-to-end:

- `infra/vps/provision.sh`: cai Docker, nginx, certbot
- `infra/vps/deploy.sh`: build va chay stack Docker
- `infra/vps/configure_nginx.sh`: tao virtual host nginx cho domain
- `infra/vps/enable_https.sh`: bat Let's Encrypt
- `infra/vps/.env.vps.example`: mau env production

Kien truc production khuyen nghi:

- VPS chay `frontend`, `api-gateway`, `metadata-service`, `queue-worker`, `postgres`
- `tracking-service` local la tuy chon va mac dinh khong bat
- `TRACKING_SERVICE_URL` tro sang LightningAI GPU public endpoint
- nginx tren host reverse proxy:
  - `/` -> Next.js frontend
  - `/api/` -> FastAPI api-gateway

Trinh tu deploy tren Ubuntu VPS:

```bash
apt update
apt install -y git
mkdir -p /opt/mcpt
cd /opt/mcpt
git clone <your-repo-url> app
cd app
bash infra/vps/provision.sh
mkdir -p secrets
cp infra/vps/.env.vps.example .env
```

Copy file service-account Google Drive vao:

```bash
/opt/mcpt/app/secrets/drive-sa.json
```

Sua `.env` production:

- `POSTGRES_PASSWORD`
- `JWT_SECRET_KEY`
- `LETSENCRYPT_EMAIL`
- `GOOGLE_DRIVE_VINUNI_FOLDER_ID`
- `LIGHTNING_API_TOKEN`
- `TRACKING_SERVICE_URL`
- `LIGHTNING_API_BASE_URL`

Sau do deploy app:

```bash
cd /opt/mcpt/app
bash infra/vps/deploy.sh
```

Cau hinh nginx cho domain:

```bash
cd /opt/mcpt/app
bash infra/vps/configure_nginx.sh search-engine-119.smartnovi.tech 13000 18000
```

Bat HTTPS:

```bash
cd /opt/mcpt/app
bash infra/vps/enable_https.sh search-engine-119.smartnovi.tech your-email@example.com
```

Kiem tra:

- `https://search-engine-119.smartnovi.tech`
- `https://search-engine-119.smartnovi.tech/api/v1/overview`

Ghi chu production:

- Port container tren VPS duoc bind vao `127.0.0.1` de chi co nginx host moi public ra Internet.
- Frontend production mac dinh goi API cung origin, khong con phu thuoc `localhost:8000`.
- Neu sau nay muon bat `tracking-service` local co GPU tren VPS, chay them:

```bash
COMPOSE_PROFILES=local-gpu docker compose up -d --build tracking-service
```

## Mo rong tiep theo

- Gan domain vao reverse proxy/Nginx tren VPS
- Chuyen local volume sang object storage (S3/MinIO)
- Kich hoat LightningAI GPU endpoint production
- Them migration Alembic va role-based access
- Xem tai lieu `../docs/ACCURACY_FIRST_REARCHITECTURE.md` de theo profile `accuracy_first`
