# Repo Audit Report

Date: 2026-04-18
Repo: `A20-App-119`

## 1. Executive Summary

Repo này hiện là một workspace lai giữa:

- phần bài nộp/hạ tầng agent ở root repo
- một sản phẩm chính trong `Multi-Camera-Person-Tracking-and-Re-Identification/`
- một `legacy-engine` lớn vẫn còn được gọi trực tiếp từ service mới

Kiến trúc đang đi theo hướng đúng cho một hệ thống tìm người trong CCTV:

- `frontend` Next.js
- `api-gateway` FastAPI
- `metadata-service` FastAPI + PostgreSQL
- `tracking-service` FastAPI cho ingestion/tracking/query orchestration
- `queue-worker` đồng bộ Google Drive và quản lý FIFO queue

Nhưng mức độ hoàn thiện hiện tại là `hybrid / in-transition`:

- lớp microservice mới đã có cấu trúc rõ
- `tracking-service` đã có pipeline contract `accuracy_first`
- phần inference SOTA vẫn chủ yếu ở mức `profile/config contract`, chưa bundle model runtime đầy đủ
- local ingestion hiện vẫn phụ thuộc mạnh vào `backend/legacy-engine`

## 2. Phạm Vi Đọc Và Kiểm Tra

Đã quét toàn bộ cây file và đọc sâu các khu vực sau:

- root docs + root `src/` + `scripts/`
- `docker-compose.yml`
- `frontend/`
- `backend/services/api-gateway`
- `backend/services/metadata-service`
- `backend/services/tracking-service`
- `infra/`
- `backend/legacy-engine` và `backend/legacy-engine/src_vlm`

Kiểm tra đã chạy:

- thống kê file toàn repo
- quét dấu hiệu unfinished/mock/TODO
- `python -m py_compile` cho root source và 3 backend services
- kiểm tra `git status --short`

Kết quả:

- `py_compile`: pass cho các service chính và root source
- `git status --short`: clean

## 3. Thống Kê Nhanh

- Tổng file ngoài `.git`: `396`
- File trong app chính `Multi-Camera-Person-Tracking-and-Re-Identification`: `372`
- File Python: `153`
- File video `.mp4`: `72`
- File JSON: `40`
- File `.metadata`: `34`

Nhận xét:

- repo đang chứa cả code lẫn dữ liệu/video mẫu
- có cả `__pycache__` trong repo
- các thư mục top-level `src_vlm/`, `deep_sort/`, `torchreid/` dưới app chính gần như không còn source hữu ích, chủ yếu là `__pycache__` hoặc trống

## 4. Bản Đồ Repo

### 4.1 Root Repo

Các file đáng chú ý:

- `README.md`: mô tả bài toán CCTV person search và MVP
- `ARCHITECTURE.md`: mô tả kiến trúc logic offline-first
- `AGENTS.md`: quy định logging cho AI coding agents
- `src/`: một agent loop Python nhỏ dùng Anthropic
- `scripts/`: hook logger và submit AI logs

Nhận xét:

- root `src/agent.py` là demo harness riêng, không phải runtime của sản phẩm MCPT
- `src/tools.py` vẫn còn tool placeholder như `search_web`

### 4.2 App Chính

`Multi-Camera-Person-Tracking-and-Re-Identification/` là phần sản phẩm thực tế, gồm:

- `frontend/`
- `backend/services/api-gateway/`
- `backend/services/metadata-service/`
- `backend/services/tracking-service/`
- `backend/legacy-engine/`
- `infra/`
- `data/`, `videos/`

## 5. Kiến Trúc Thực Thi Hiện Tại

### 5.1 Frontend

Frontend là một dashboard Next.js đơn trang:

- auth login/register
- upload video hoặc khai báo `storage_url`
- chọn video
- chạy text query
- xem lịch sử query gần đây
- hiển thị overview metrics

Đặc điểm:

- toàn bộ UI chính nằm trong `frontend/app/page.tsx`
- giao diện đẹp, rõ ràng, nhưng hiện vẫn là dashboard MVP
- chưa thấy chia nhỏ thành nhiều route/module UI

### 5.2 API Gateway

`api-gateway` chủ yếu là thin proxy/orchestrator:

- forward auth header
- gọi `metadata-service` cho auth/video/query
- gọi `tracking-service` cho AI process
- endpoint `overview` ghép dữ liệu từ metadata + tracking pipeline config

Nhận xét:

- logic business ở đây ít
- role chính là public API façade cho frontend

### 5.3 Metadata Service

`metadata-service` là service lõi cho state nghiệp vụ:

- đăng ký / đăng nhập JWT
- quản lý `users`
- quản lý `video_assets`
- quản lý `video_queries`
- lưu `person_candidates`
- lưu `queue_video_assets`
- import metadata từ legacy JSON

Database:

- SQLAlchemy + PostgreSQL
- `Base.metadata.create_all()` chạy trực tiếp ở startup
- chưa có Alembic migration

Vai trò thực tế:

- là system of record cho user/video/query/candidate/queue
- chịu trách nhiệm map metadata từ AI pipeline sang schema tra cứu

### 5.4 Tracking Service

`tracking-service` đang có 3 vai trò khác nhau:

1. expose pipeline contract `accuracy_first`
2. xử lý query orchestration local/mock hoặc remote LightningAI
3. xử lý ingestion video và sinh metadata/candidate

Hai mode chính:

- `mock`: trả kết quả giả lập nhưng kèm pipeline config/hardware profile
- `remote`: upload video sang LightningAI endpoint

Điểm quan trọng:

- local ingestion không tự chạy full stack SOTA nội bộ
- nó đi vào `backend/legacy-engine` qua `legacy_workdir()` rồi gọi `src_vlm`

### 5.5 Queue Worker

`queue-worker` thực chất dùng image của `metadata-service` để:

- poll `Import_New` từ Google Drive
- download video
- gọi `tracking-service /api/v1/ingestion/process`
- upload `.h265` và metadata lên thư mục `Queue`
- lưu state queue trong PostgreSQL
- evict FIFO khi vượt `QUEUE_MAX_SIZE`

Đây là phần khá thực dụng và đã có luồng khép kín hơn nhiều so với phần query-time.

## 6. Luồng Dữ Liệu Thực Tế

### 6.1 Query Flow

Luồng query hiện tại:

1. Frontend gọi `POST /api/v1/video-queries/run`
2. API gateway tạo row query ở `metadata-service`
3. API gateway gọi `tracking-service /api/v1/ai/process`
4. `tracking-service`:
   - nếu `mock` hoặc không có Lightning base URL: trả demo response
   - nếu `remote`: chuẩn bị/copy/convert video rồi gọi LightningAI
5. API gateway patch lại `video_query` với AI response

Nhận xét:

- flow orchestration rõ
- nhưng query-time retrieval thật vẫn chưa nằm trọn trong service mới

### 6.2 Ingestion Flow

Luồng ingestion thực tế mạnh hơn query flow:

1. `queue-worker` hoặc API gọi `tracking-service /api/v1/ingestion/process`
2. `tracking-service` resolve profile + hardware config
3. `tracking-service` import `src_vlm` từ legacy engine
4. video được nén H.265
5. `hospital_pipeline._build_detected_people(...)` sinh:
   - `video_payload`
   - `people`
   - file metadata JSON
6. kết quả được đẩy về queue local hoặc Google Drive
7. `metadata-service` upsert `QueueVideoAsset` và `PersonCandidate`

Đây là chỗ repo hiện “chạy thật” nhiều nhất.

## 7. Vai Trò Của Legacy Engine

`backend/legacy-engine/` không chỉ để lưu tham khảo. Nó vẫn là runtime dependency thực tế.

Hiện đang được dùng cho:

- `src_vlm.vlm_engine`
- `src_vlm.video_ingestion`
- `src_vlm.hospital_pipeline`
- `src_vlm.vector_search`
- `src_vlm.tracker`

Ý nghĩa:

- microservice mới chưa thay thế hoàn toàn legacy engine
- bản chất hiện tại là “new service shell + old AI core”

Các thành phần vendored lớn trong legacy:

- YOLOv3/v4
- Deep SORT
- Torchreid
- Streamlit demo
- vector search với ChromaDB + sentence-transformers

## 8. Mức Độ Hoàn Thiện Theo Thành Phần

### 8.1 Phần khá hoàn chỉnh

- Docker compose cho stack cơ bản
- auth/video/query persistence
- queue import từ Google Drive
- ingest video và sinh metadata per-person
- report pipeline profile/hardware execution plan
- frontend dashboard MVP usable

### 8.2 Phần đang ở mức scaffold/design-ready

- `accuracy_first` profile: rất giàu metadata, nhưng chủ yếu là contract cấu hình
- detector/tracker/reid SOTA mới chưa được bundle thành runtime thật
- LightningAI path phụ thuộc external endpoint
- `run_tracking` trong service mới vẫn chỉ tạo clip/demo style overlay

### 8.3 Phần legacy còn đang gánh nghiệp vụ

- VLM metadata extraction
- video compression helper
- candidate generation
- vector search logic trong demo stack

## 9. Findings Quan Trọng

### 9.1 `docker compose up --build` mặc định chưa chắc chạy end-to-end như README mô tả

Lý do:

- `tracking-service` nằm sau profile `local-gpu`
- nhưng `api-gateway /api/v1/overview` luôn gọi `tracking-service`
- default URLs vẫn trỏ vào `http://tracking-service:8000`

Tác động:

- nếu không override `TRACKING_SERVICE_URL` sang endpoint remote hoặc bật profile `local-gpu`, overview/query có thể fail downstream

### 9.2 `tracking-service` local còn phụ thuộc nặng vào legacy dependencies chưa có trong requirements chính

`tracking-service/requirements.txt` không bundle rõ các thư viện mà legacy ingestion có thể cần, ví dụ:

- `transformers`
- `torch`
- `sentence-transformers`
- `chromadb`

Tác động:

- chế độ local thật rất dễ bị lệ thuộc vào môi trường ngoài hoặc image khác với requirements hiện tại

### 9.3 Repo có dấu hiệu lẫn cấu hình nhạy cảm trong file mẫu

`.env.example` đang chứa:

- `AI_LOG_API_KEY`
- `GOOGLE_DRIVE_VINUNI_FOLDER_ID`
- đường dẫn cụ thể tới file service account JSON

Tác động:

- không nên để file mẫu mang giá trị có vẻ là secret thật hoặc identifier production thật

### 9.4 Mặc định bảo mật còn yếu cho môi trường production

Ví dụ:

- `JWT_SECRET_KEY=change-me-in-production`
- `POSTGRES_PASSWORD=mcpt_password`
- `GOOGLE_DRIVE_MAKE_PUBLIC=true`

Tác động:

- nếu deploy thiếu hardening sẽ public asset và dùng secret mặc định

### 9.5 Repo đang hơi nặng và lẫn artefact build/runtime

Bao gồm:

- video `.mp4`
- `.metadata`
- `__pycache__`
- pyc files ở nhiều nơi

Tác động:

- repo khó review
- tăng clone size
- khó tách code và data

### 9.6 Root agent harness là một nhánh sản phẩm khác với app chính

`src/agent.py` + `src/tools.py` là mini agent loop riêng.

Tác động:

- dễ gây nhầm lẫn cho người mới vào repo
- không phải entry point của MCPT platform nhưng lại nằm ngay root

## 10. Risk Register

### High

- Stack local mặc định có nguy cơ thiếu `tracking-service` runtime ở luồng end-to-end.
- Secret/config production-like xuất hiện trong `.env.example`.
- Local AI pipeline phụ thuộc legacy code + dependency chain chưa được đóng gói trọn vẹn.

### Medium

- Chưa có migration strategy cho database ngoài `create_all`.
- Public Google Drive links bật mặc định.
- Frontend là một page lớn, chưa modularize, khó mở rộng.
- `backend/legacy-engine` và service mới cùng tồn tại lâu dài sẽ tăng chi phí bảo trì.

### Low

- Có artifact như `__pycache__` trong repo.
- Một số thư mục top-level dư thừa hoặc chỉ còn cache.
- Root repo chứa cả assignment tooling và product code.

## 11. Những Gì Đã Xác Thực Bằng Code

- `api-gateway` thực sự chỉ orchestration/proxy, không giữ nghiệp vụ chính.
- `metadata-service` là nơi lưu state thật.
- `tracking-service` hiện trả mock khi thiếu Lightning base URL hoặc khi `TRACKING_USE_MOCK=true`.
- ingestion local dùng `legacy_workdir()` để gọi code trong `backend/legacy-engine`.
- queue worker có FIFO eviction thật.
- frontend đang là dashboard MVP hoàn chỉnh về mặt flow cơ bản.

## 12. Khuyến Nghị Ưu Tiên

### Ưu tiên 1

- Làm rõ mode chạy chuẩn:
  - local full stack
  - remote LightningAI
  - mock/demo

- Sửa README và compose để mode mặc định chạy được thật theo đúng tài liệu.

### Ưu tiên 2

- Tách hẳn dependency/runtime cần cho local ingestion khỏi `legacy-engine`, hoặc đóng gói legacy thành module/service riêng có owner rõ ràng.

### Ưu tiên 3

- Dọn secret/config mẫu:
  - xoá token thật khỏi `.env.example`
  - thay folder IDs/path thật bằng placeholder
  - buộc secret production phải inject từ environment

### Ưu tiên 4

- Thêm migration bằng Alembic.

### Ưu tiên 5

- Tách code và data:
  - bỏ `__pycache__`
  - đưa video mẫu và metadata lớn ra storage riêng
  - giảm repo weight

## 13. Kết Luận

Đây không phải repo “đang dang dở hoàn toàn”, mà là repo đang ở giai đoạn chuyển giao thật sự:

- phần platform/microservice đã hình thành tương đối tốt
- phần AI core mới chưa fully productized
- legacy engine vẫn còn là xương sống cho ingestion/search nội bộ

Nếu mục tiêu gần hạn là demo hoặc pilot, repo này đã có nền đủ mạnh để vận hành một luồng ingest + query cơ bản.

Nếu mục tiêu là production sạch, maintainable, và reproducible, 3 việc cần làm sớm nhất là:

1. chuẩn hóa mode chạy mặc định
2. bóc/tách legacy runtime
3. dọn secret + hardening config
