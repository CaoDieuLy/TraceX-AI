# Kiến trúc modular (sau khi gỡ thư mục monolith)

Mã nguồn ứng dụng nằm ở **gốc repo**:

| Thư mục | Vai trò |
|---------|---------|
| `frontend/` | Next.js 14: UI, gọi API qua `NEXT_PUBLIC_API_BASE_URL` (trình duyệt) và rewrite nội bộ (`INTERNAL_API_GATEWAY_URL` → `http://backend:8000` trong Docker). |
| `backend/` | **metadata-service**: Postgres, auth, video/candidate, queue, Drive, preview, và hàm **`rank_candidates`** (lọc cục bộ + *có thể* gọi tracking — dùng khi FE/API gọi thẳng metadata). **api-gateway**: CORS, forward metadata, **không còn gọi trực tiếp URL Lightning/tracking**; mọi gọi model/tracking từ gateway đi qua **`ai_service`** (`/internal/tracking/v1/...`). **legacy-engine**: thư viện cũ (metadata có thể import khi xử lý ảnh/preview cục bộ). |
| `ai_service/` | FastAPI cổng 8001: **`POST /internal/search`** (DB + `rank_candidates` + gọi tracking xa). **`GET/POST /internal/tracking/v1/...`**: proxy nội bộ tới `TRACKING_SERVICE_URL` (pipeline config, `ai/process`, `tracking/run`, artifacts) — **chỉ ai_service** nắm URL/token gọi Lightning. |
| `infra/` | `docker-compose.yml`, Dockerfiles, `env/*.env`, init Postgres, script VPS. |

## Luồng tìm kiếm (qua gateway)

1. Trình duyệt → **gateway** `POST /search`.
2. Gateway → **`ai_service` `POST /internal/search`**.
3. Nếu AI service lỗi, gateway fallback sang metadata `POST /api/v1/candidates/search` (vẫn có thể gọi tracking từ *bên trong* metadata — xem mục dưới).

## Tách BE vs AI (đã làm / còn lại)

- **Đã làm:** **api-gateway** không import và không dùng `TRACKING_SERVICE_URL` trực tiếp; chỉ dùng `AI_SERVICE_URL` cho search + các route proxy tracking tương ứng public API.
- **Còn lại (tuỳ chọn bước sau):** **metadata-service** vẫn chứa `rank_candidates` và `_post_tracking_json` — dùng khi client gọi thẳng `/api/v1/candidates/search` hoặc khi pipeline nội bộ cần. Muốn “một cửa” AI hoàn toàn: cho metadata endpoint đó HTTP gọi lại `ai_service` thay vì gọi tracking trực tiếp (refactor lớn hơn).

## Ghi chú

- Thư mục monolith cũ đã gỡ; build Docker dùng `backend/`, `frontend/`, `ai_service/` ở root.
- Deploy VPS: `infra/vps/deploy.sh` từ gốc repo sau khi đồng bộ mã.
- Biến **`TRACKING_SERVICE_URL`** và token Lightning vẫn cần trong **`infra/env/ai.env`** (và trong env của container backend cho **metadata**, vì metadata chạy chung image với gateway).
