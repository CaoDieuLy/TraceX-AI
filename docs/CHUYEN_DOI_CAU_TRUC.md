# Chuyển đổi cấu trúc (thay cho DEPRECATED trong monolith)

Toàn bộ mã từng nằm trong `Multi-Camera-Person-Tracking-and-Re-Identification/` đã được chuyển lên gốc repo:

- `backend/` — services (metadata, api-gateway, tracking-service, legacy-engine, …)
- `frontend/` — Next.js
- SQL init Postgres: `infra/postgres/init/`

Docker build context là **gốc repo**; xem `infra/README.md` và `docs/KIEN_TRUC_MODULAR_VI.md`.
