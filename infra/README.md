# infra/

Docker Compose stack cho VPS deployment.

## Chạy stack

```bash
# Sinh env files từ secrets/master.env trước
bash scripts/sync_secrets.sh

# Start tất cả services
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env up -d --build

# Xem status
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env ps
```

## Services

| Service | Port | Mô tả |
|---------|------|-------|
| `mcpt-frontend` | 3000 | Next.js web app |
| `mcpt-backend` | 8000 | API Gateway + Metadata Service + Queue Worker |
| `mcpt-ai-service` | 8001 | AI Search Service (internal) |
| `mcpt-postgres` | 5432 | PostgreSQL database |

## Env files

> Không edit trực tiếp — auto-generated bởi `scripts/sync_secrets.sh` từ `secrets/master.env`.

| File | Dùng bởi |
|------|---------|
| `env/backend.env` | `docker compose --env-file` |
| `env/ai.env` | ai_service container |
| `env/frontend.env` | frontend container |

## Deploy lên VPS

```bash
bash infra/vps/deploy.sh .
```
