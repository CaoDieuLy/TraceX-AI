# A20-App-119 Docker/Coolify Deployment

## Architecture

```
┌─────────────────┐
│     Frontend     │  Next.js (Port 3000)
│   (TraceX-AI)   │
└────────┬────────┘
         │
┌────────▼────────┐
│  Metadata Service │  FastAPI (Port 8001)
│   + PostgreSQL    │
└────────┬────────┘
         │ HTTP/gRPC
         ▼
┌─────────────────┐
│    AI Service    │  FastAPI + PyTorch (Port 8002)
│    (A100 GPU)    │  RF-DETR, DINOv2, VideoMAE, SigLIP2
└─────────────────┘
```

## Services

### 1. PostgreSQL
- Image: `postgres:16-alpine`
- Port: 5432
- Persistent volume: `postgres_data`

### 2. Metadata Service
- FastAPI backend - DB + API
- Port: 8001
- Calls AI service via HTTP

### 3. AI Service (A100 GPU)
- RF-DETR 2x-large person detector
- DINOv2 ViT-L/14 Re-ID
- VideoMAE action recognition
- SigLIP2 attribute classification
- Port: 8002

### 4. Frontend
- Next.js application
- Port: 3000

## Project Structure

```
backend/
├── services/
│   ├── api-gateway/        # (existing)
│   ├── metadata-service/    # FastAPI + PostgreSQL
│   └── ai-service/         # AI models on A100
│       ├── Dockerfile
│       ├── requirements.txt
│       └── app/
│           ├── main.py          # FastAPI entry
│           ├── config.py        # Settings
│           ├── detection/       # RF-DETR
│           ├── tracking/       # ByteTrack
│           ├── features/       # DINOv2, SigLIP2
│           ├── video/          # Frame sampling
│           └── core/
│               ├── types/       # BoundingBox, Frame
│               └── utils/       # Geometry, Image ops
├── config/                    # Camera calibration
└── storage/                   # Model weights
```

## Deployment with Coolify

### Step 1: Create PostgreSQL
```bash
# Coolify: New Database Resource
# Type: PostgreSQL 16
# Credentials from infra/env/backend.env
```

### Step 2: Deploy Metadata Service
```bash
# Coolify: New Application
# Build: Dockerfile
# Source: ./backend/services/metadata-service/
# Env: POSTGRES_HOST=<pg-host>
```

### Step 3: Deploy AI Service (GPU)
```bash
# Coolify: New Application (with GPU)
# Build: Dockerfile
# Source: ./backend/services/ai-service/
# GPU: Required (A100)
```

### Step 4: Deploy Frontend
```bash
# Coolify: New Application
# Build: Dockerfile
# Source: ./frontend/
```

## Environment Variables

See `infra/env/backend.env`

## Security

- Change all default passwords
- Set `JWT_SECRET_KEY` to secure random value
- Restrict CORS origins
- Use HTTPS via Traefik (Coolify)
