# TraceX-AI Docker Deployment Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VPS (Coolify)                                 │
│   tracex-ai.smartnovi.tech                                           │
│   ┌───────────────────────────────────────────────────────────────┐   │
│   │  frontend (Next.js)  :3000  ───► Traefik (HTTPS :443)       │   │
│   └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Browser calls directly
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     LightningAI (NVIDIA A100 80GB)                    │
│   https://8000-xxxx.cloudspaces.litng.ai                             │
│                                                                       │
│   ┌──────────────┐  ┌───────────────────┐  ┌────────────────────┐   │
│   │  postgres    │  │metadata-service  │  │  query-service     │   │
│   │    :5432     │  │     :8002         │  │     :8003          │   │
│   │  (internal)  │  │  GPU (50GB VRAM) │  │  GPU (15GB VRAM)   │   │
│   └──────────────┘  └───────────────────┘  └────────────────────┘   │
│                          │                                           │
│                          │ (internal bridge network: "internal")      │
│                          ▼                                           │
│                    ┌───────────────────┐                             │
│                    │  trace-service    │                             │
│                    │     :8004         │                             │
│                    │  GPU (15GB VRAM)  │                             │
│                    └───────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

**Traffic Flow:**
- User Browser → `tracex-ai.smartnovi.tech` (VPS) — serves frontend HTML/JS
- Frontend (browser) → `NEXT_PUBLIC_API_BASE_URL` = LightningAI URL + `/api/v1`
- Backend services communicate internally via Docker bridge network
- Database accessed only by backend services (not exposed publicly)

---

## Service Summary

| Service | Runs On | Port | GPU | Description |
|---------|---------|------|-----|-------------|
| `frontend` | VPS | 3000 | No | Next.js web app |
| `postgres` | LightningAI | 5432 | No | PostgreSQL 16 database |
| `metadata-service` | LightningAI | 8002 | Yes | Auth, queue, video metadata + SOTA AI |
| `query-service` | LightningAI | 8003 | Yes | Search, translation (SeamlessM4T) |
| `trace-service` | LightningAI | 8004 | Yes | Neural video reconstruction |

---

## Prerequisites

### On LightningAI
- Docker and Docker Compose installed
- NVIDIA GPU with nvidia-docker runtime
- Access to `/workspace/` directory

### On VPS (Coolify)
- Docker and Docker Compose installed
- Coolify configured with Traefik

---

## Environment Files

### LightningAI — `.env.lightningai`
Copy and configure:

```bash
cp .env.lightningai.example .env.lightningai
nano .env.lightningai
```

Required variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `COOLIFY_PUBLIC_URL` | LightningAI public URL | `https://8000-xxxx.cloudspaces.litng.ai` |
| `POSTGRES_PASSWORD` | Database password | (generate with `openssl rand -base64 32`) |
| `JWT_SECRET_KEY` | JWT signing key | (generate with `openssl rand -base64 64`) |
| `STORAGE_BASE_URL` | VPS domain for static files | `https://tracex-ai.smartnovi.tech/storage` |
| `BOOTSTRAP_ADMIN_EMAIL` | Initial admin email | `admin@tracex.example.com` |
| `BOOTSTRAP_ADMIN_PASSWORD` | Initial admin password | `Admin@123456` |

### VPS — `.env.vps`
Copy and configure:

```bash
cp .env.vps.example .env.vps
nano .env.vps
```

Required variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_BASE_URL` | LightningAI URL + `/api/v1` | `https://8000-xxxx.cloudspaces.litng.ai/api/v1` |

---

## Deployment Steps

### Step 1: Deploy Backend on LightningAI

SSH into LightningAI machine and run:

```bash
# Navigate to project
cd /path/to/TraceX-AI

# Create environment file
cp .env.lightningai.example .env.lightningai
nano .env.lightningai  # Fill in COOLIFY_PUBLIC_URL and secrets

# Create required directories (if not exist)
sudo mkdir -p /workspace/models \
    /workspace/datasets/MTMC_Tracking_2024/train \
    /workspace/a20-root \
    /workspace/storage/videos \
    /workspace/storage/tracking-output \
    /workspace/storage/traces \
    /workspace/storage/candidate-previews \
    /workspace/storage/cache \
    /workspace/secrets
sudo chmod -R 777 /workspace

# Deploy using script
./scripts/deploy-lightningai.sh

# Or manually:
docker compose -f backend/services/lightningai-compose.yml --env-file .env.lightningai build
docker compose -f backend/services/lightningai-compose.yml --env-file .env.lightningai up -d
```

### Step 2: Deploy Frontend on VPS

On VPS machine:

```bash
# Navigate to project
cd /path/to/TraceX-AI

# Create environment file
cp .env.vps.example .env.vps
nano .env.vps  # Set NEXT_PUBLIC_API_BASE_URL to your LightningAI URL + /api/v1

# Deploy using script
./scripts/deploy-vps.sh

# Or manually:
docker compose build frontend
docker compose --env-file .env.vps up -d
```

---

## Verify Deployment

### Check Backend Health (LightningAI)

```bash
# Check all services
docker compose -f backend/services/lightningai-compose.yml ps

# Check specific service logs
docker compose -f backend/services/lightningai-compose.yml logs -f metadata-service
docker compose -f backend/services/lightningai-compose.yml logs -f query-service
docker compose -f backend/services/lightningai-compose.yml logs -f trace-service

# Test health endpoints
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health
```

Expected output:
```json
{"status":"healthy","service":"metadata-service","gpu_warmup_done":true,...}
{"status":"healthy","service":"query-service","version":"2.0.0"}
{"status":"healthy","service":"trace-service","version":"2.0.0"}
```

### Check Database Connection

```bash
# Connect to postgres
docker compose -f backend/services/lightningai-compose.yml exec postgres psql -U mcpt_user -d video_tracking -c "\dt"
```

Expected: List of tables (users, videos, camera_locations, tracking_results, etc.)

### Check Frontend (VPS)

```bash
# Check frontend logs
docker compose logs -f frontend

# Test frontend
curl https://tracex-ai.smartnovi.tech

# Check browser console for API errors
# Navigate to frontend in browser
# Open DevTools → Console → Should see no API errors
# Check Network tab → API calls should return 200
```

---

## Update Services When Code Changes

### Update Backend (LightningAI)

```bash
cd /path/to/TraceX-AI

# Pull latest code
git pull origin main

# Rebuild and restart backend services
docker compose -f backend/services/lightningai-compose.yml --env-file .env.lightningai build
docker compose -f backend/services/lightningai-compose.yml --env-file .env.lightningai up -d

# Or use deploy script
./scripts/deploy-lightningai.sh
```

### Update Frontend (VPS)

```bash
cd /path/to/TraceX-AI

# Pull latest code
git pull origin main

# Rebuild and restart frontend
docker compose build frontend
docker compose --env-file .env.vps up -d

# Or use deploy script
./scripts/deploy-vps.sh
```

---

## Troubleshooting

### Backend Services Won't Start

```bash
# Check logs
docker compose -f backend/services/lightningai-compose.yml logs

# Check GPU availability
nvidia-smi

# Verify docker can see GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Frontend Can't Connect to Backend

```bash
# Verify NEXT_PUBLIC_API_BASE_URL is correct
cat .env.vps | grep NEXT_PUBLIC_API_BASE_URL

# Test backend directly from VPS
curl https://YOUR-LIGHTNINGAI-URL/health

# Check browser network tab for failed API calls
```

### Database Connection Issues

```bash
# Check postgres is running
docker compose -f backend/services/lightningai-compose.yml ps postgres

# Check postgres logs
docker compose -f backend/services/lightningai-compose.yml logs postgres

# Test connection from metadata-service
docker compose -f backend/services/lightningai-compose.yml exec metadata-service python -c \
  "from app.config import settings; print(f'Host: {settings.postgres_host}')"
```

### Reset Everything

```bash
# Stop all services
docker compose -f backend/services/lightningai-compose.yml down

# Remove volumes (WARNING: deletes all data!)
docker compose -f backend/services/lightningai-compose.yml down -v

# Restart fresh
docker compose -f backend/services/lightningai-compose.yml --env-file .env.lightningai up -d
```

---

## Port Reference

| Service | Internal Port | Exposed | Used By |
|---------|--------------|---------|---------|
| `postgres` | 5432 | No | metadata-service, query-service, trace-service |
| `metadata-service` | 8002 | 8002 | Frontend, query-service, trace-service |
| `query-service` | 8003 | 8003 | Frontend, metadata-service |
| `trace-service` | 8004 | 8004 | Frontend, metadata-service, query-service |
| `frontend` | 3000 | 3000 | Browser |

---

## Security Notes

- Never commit `.env` files to git
- Use strong passwords for `POSTGRES_PASSWORD` and `JWT_SECRET_KEY`
- The database port (5432) is NOT exposed publicly — only accessible within Docker network
- Backend ports (8002, 8003, 8004) are exposed via LightningAI's built-in proxy with HTTPS
- Consider enabling CORS restrictions in production

---

## File Structure

```
TraceX-AI/
├── docker-compose.yml                    # VPS: frontend only
├── .env.vps.example                     # VPS env template
├── .env.lightningai.example             # LightningAI env template
├── backend/
│   └── services/
│       ├── lightningai-compose.yml      # LightningAI: backend + postgres
│       ├── metadata-service/
│       │   └── Dockerfile
│       ├── query-service/
│       │   └── Dockerfile
│       └── trace-service/
│           └── Dockerfile
├── frontend/
│   └── Dockerfile
├── scripts/
│   ├── deploy-lightningai.sh            # Deploy backend
│   └── deploy-vps.sh                   # Deploy frontend
└── DEPLOY.md                           # This file
```
