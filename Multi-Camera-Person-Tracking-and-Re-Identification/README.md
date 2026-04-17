# Multi-Camera Person Tracking and Re-Identification

Project này đã được tách từ dạng monolith sang kiến trúc microservice để dễ phát triển và triển khai hơn:

- `frontend/`: giao diện người dùng bằng Next.js
- `backend/services/api-gateway/`: FastAPI gateway cho frontend
- `backend/services/metadata-service/`: FastAPI + PostgreSQL cho metadata/search
- `backend/services/tracking-service/`: FastAPI bọc legacy tracking engine
- `backend/legacy-engine/`: code tracking/ReID hiện có, giữ nguyên để tương thích
- `infra/postgres/`: SQL khởi tạo PostgreSQL

## Cấu trúc thư mục

```text
Multi-Camera-Person-Tracking-and-Re-Identification/
|- frontend/
|- backend/
|  |- legacy-engine/
|  |- services/
|     |- api-gateway/
|     |- metadata-service/
|     |- tracking-service/
|- infra/
|  |- postgres/
|- docker-compose.yml
```

## Kiến trúc microservice

```text
Next.js Frontend
        |
        v
FastAPI API Gateway
   |            |
   v            v
Metadata Service   Tracking Service
   |                 |
   v                 v
PostgreSQL       Legacy ReID Engine
```

## Chạy bằng Docker Compose

```bash
docker compose up --build
```

Sau khi chạy:

- Frontend: `http://localhost:3000`
- API Gateway: `http://localhost:8000`
- Metadata Service: `http://localhost:8001`
- Tracking Service: `http://localhost:8002`
- PostgreSQL: `localhost:5432`

## Build Docker image riêng

```bash
docker build -t mcpt-frontend:latest ./frontend
docker build -t mcpt-api-gateway:latest ./backend/services/api-gateway
docker build -t mcpt-metadata-service:latest -f backend/services/metadata-service/Dockerfile .
docker build -t mcpt-tracking-service:latest -f backend/services/tracking-service/Dockerfile .
```

## Ghi chú vận hành

- `tracking-service` mặc định chạy `TRACKING_USE_MOCK=true` để container có thể lên nhanh mà chưa bắt buộc đủ weights/model runtime.
- Metadata cũ trong `backend/legacy-engine/data/metadata` có thể import vào PostgreSQL bằng nút trong frontend hoặc API `/api/v1/candidates/import-legacy`.
- Nếu muốn chạy inference thật, mount thêm weights/video vào `backend/legacy-engine` và đặt `TRACKING_USE_MOCK=false`.
