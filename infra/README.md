# Hạ tầng (infra)

## Chạy toàn bộ stack

Từ thư mục gốc của repository:

```bash
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env up --build
```

- **frontend**: http://localhost:3000  
- **backend** (API): http://localhost:8000  
- **ai_service** (nội bộ): http://localhost:8001 — backend gọi qua biến `AI_SERVICE_URL`  
- **postgres**: localhost:5432 (có thể ghi đè bằng biến môi trường)

## Biến môi trường

Chỉnh các file trong `infra/env/`:

| File | Mục đích |
|------|----------|
| `backend.env` | Gateway + metadata đi kèm, DB, tracking/Lightning, JWT, Drive |
| `ai.env` | Dịch vụ AI: DB + tracking (cùng pipeline xếp hạng) |
| `frontend.env` | `NEXT_PUBLIC_API_BASE_URL` cho trình duyệt |

Thư mục secrets được mount chỉ đọc tại `/workspace/a20-root/secrets` và `/workspace/a20-root/secret` từ `secrets/` và `secret/` trong repo.

## Kiến trúc

- **backend**: FastAPI gateway + metadata API trong một container; gateway gọi **ai_service** qua HTTP cho `/search`.
- **ai_service**: FastAPI `POST /internal/search`; chạy `rank_candidates` (logic metadata được vendor + tracking từ xa).

Chi tiết vai trò từng thư mục (tiếng Việt): [docs/KIEN_TRUC_MODULAR_VI.md](../docs/KIEN_TRUC_MODULAR_VI.md).  
Lịch sử gỡ monolith: [docs/CHUYEN_DOI_CAU_TRUC.md](../docs/CHUYEN_DOI_CAU_TRUC.md).

## Deploy VPS (tùy chọn)

Trên máy chủ, thư mục ứng dụng phải chứa `backend/`, `frontend/`, `ai_service/`, `infra/`, `shared_secret_runtime.py` (đồng bộ từ CI hoặc `git pull`). Sau đó:

```bash
chmod +x infra/vps/deploy.sh
./infra/vps/deploy.sh "$(pwd)"
```
