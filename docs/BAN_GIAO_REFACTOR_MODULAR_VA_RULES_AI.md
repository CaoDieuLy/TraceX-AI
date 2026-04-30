# Ban giao refactor modular + strict AI rules

Tai lieu nay tong hop nhung gi da duoc sua trong qua trinh tach module, cac quy tac bat buoc cho team phat trien tiep theo, danh sach file nhay cam "khong duoc sua tuy tien", va bo lenh van hanh du an.

## 1) Tong quan thay doi da thuc hien

- Tach kien truc thanh 4 module ro rang tai root:
  - `frontend/`: Next.js UI.
  - `backend/`: API Gateway + Metadata Service (orchestration, auth, db, queue).
  - `ai_service/`: dau moi AI/noi bo va proxy tracking upstream.
  - `infra/`: Docker Compose, Dockerfiles, env, script deploy VPS.
- Xoa folder monolith cu `Multi-Camera-Person-Tracking-and-Re-Identification/`, cap nhat toan bo duong dan build/deploy ve root moi.
- Chuan hoa Docker stack trong `infra/docker-compose.yml`:
  - service: `postgres`, `ai_service`, `backend`, `frontend`.
  - them `healthcheck`, `depends_on: service_healthy`.
  - map `secret/` + `secrets/` vao container.
- Gateway khong goi truc tiep tracking/lightning nua; tat ca request tracking di qua `ai_service`.
- Search flow da theo kieu strict:
  - `frontend` -> `backend /search` -> `ai_service /internal/search` -> `metadata-service rank_candidates` -> `tracking_service /api/v1/candidates/search`.
  - khong fallback qua luong cu trong gateway.
- Tinh nang deploy VPS da bo sung:
  - `infra/vps/check-secrets.sh` (preflight),
  - `infra/vps/deploy.sh` (build + up),
  - `docs/DEPLOY_VPS_VI.md`.

## 2) Ranh gioi trach nhiem BE vs AI (bat buoc)

### Backend (`backend/`)
- Chi xu ly API, auth, validation, orchestration, metadata/db, queue.
- Duoc phep goi `ai_service` qua HTTP noi bo.
- Khong duoc nhung logic model runtime cot loi vao gateway.

### AI Service (`ai_service/`)
- La diem vao noi bo cho tim kiem AI (`/internal/search`).
- La proxy tracking/lightning (`/internal/tracking/v1/*`) cho gateway.
- Chiu trach nhiem bo tri luong AI upstream + xu ly loi upstream.

### Frontend (`frontend/`)
- Goi API thong qua base url env (`NEXT_PUBLIC_API_BASE_URL`) va rewrite/noi bo da cau hinh.
- Khong chua logic AI.

## 3) Strict pipeline rules (KHONG duoc pha vo)

Chi duoc phep 1 pipeline duy nhat:
- detector: `RF-DETR 2x-large`
- tracker: `OCMCTrack-style corrective cascade`
- ReID: `SOLIDER + KPR`
- semantic search: `ITSELF`
- evaluation: `TrackEval HOTA`
- tracklet metadata pipeline sau quality scoring: mot pipeline thong nhat cho `attribute`, `appearance`, `action`

Cam cac hanh vi sau:
- Them pipeline/model select moi.
- Them fallback sang model cu.
- Mo lai override hyperparameters tu payload/env theo kieu da bi khoa.

File enforce chinh:
- `backend/services/tracking-service/app/strict_pipeline.py`
- `backend/services/tracking-service/app/config.py`
- `backend/services/tracking-service/app/runtime.py`
- `backend/services/tracking-service/app/ingestion_runtime.py`
- `backend/services/metadata-service/app/service.py`
- `backend/services/tracking-service/app/schemas.py`
- `backend/services/tracking-service/app/gpu_client.py`

## 4) Danh sach file nhay cam "khong duoc sua tuy tien"

> Muc tieu: tranh lam vo strict pipeline hoac pha vo bien gioi BE/AI.

### Nhom A - Khoa pipeline (sửa cần lưu ý)
- `backend/services/tracking-service/app/strict_pipeline.py`
- `backend/services/tracking-service/app/runtime.py`
- `backend/services/tracking-service/app/ingestion_runtime.py`
- `backend/services/tracking-service/app/config.py`
- `backend/services/tracking-service/app/schemas.py`
- `backend/legacy-engine/torchreid/models/__init__.py`

### Nhom B - Bien gioi giao tiep BE <-> AI (khong duoc noi lai truc tiep)
- `backend/services/api-gateway/app/main.py`
- `backend/services/api-gateway/app/services/ai_client.py`
- `backend/services/api-gateway/app/config.py`
- `ai_service/aiapp/main.py`
- `ai_service/aiapp/tracking_upstream.py`
- `ai_service/aiapp/config.py`

### Nhom C - Van hanh/deploy (khong doi ten service, khong doi wiring tuy tien)
- `infra/docker-compose.yml`
- `infra/docker/Dockerfile.backend`
- `infra/docker/Dockerfile.ai_service`
- `infra/env/backend.env`
- `infra/env/ai.env`
- `infra/env/frontend.env`
- `infra/vps/check-secrets.sh`
- `infra/vps/deploy.sh`

## 5) Cac def quan trong da duoc comment de handover

Da chen comment/docstring truc tiep vao code cho cac ham quan trong:
- `ai_service/aiapp/main.py`
  - `_to_result_item`: chuan hoa payload candidate thanh schema on dinh tra ve gateway.
  - `internal_search`: dau vao search noi bo, goi rank + phan trang.
  - `internal_tracking_runtime_config`, `internal_tracking_ai_process`, `internal_tracking_run`, `internal_tracking_artifact`, `internal_tracking_artifact_manifest`: vai tro proxy tracking/lightning.
- `backend/services/tracking-service/app/strict_pipeline.py`
  - `get_strict_pipeline`: tra ve duy nhat 1 pipeline co dinh, khong expose co che chon stack.
- `backend/services/metadata-service/app/service.py`
  - `rank_candidates`: luong rank strict va nguyen tac khong fallback local khi upstream loi.

## 6) Lenh chay du an (local, dung thu tu)

Chay tai root repo cua stack deploy, noi co `backend/`, `frontend/`, `ai_service/`, `infra/`.

### Khoi dong/rebuild stack
```powershell
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env up -d --build
```

### Xem trang thai
```powershell
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env ps
```

### Xem log tong hop
```powershell
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env logs -f --tail=200
```

### Xem log tung service
```powershell
docker logs -f mcpt-backend
docker logs -f mcpt-ai-service
docker logs -f mcpt-frontend
docker logs -f mcpt-postgres
```

### Restart nhanh service
```powershell
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env restart backend ai_service frontend
```

### Dung stack va don orphan
```powershell
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env down --remove-orphans
```

### Xu ly conflict ten container neu bi trung ten cu
```powershell
docker rm -f mcpt-postgres mcpt-frontend mcpt-backend mcpt-ai-service
```

## 7) Lenh deploy VPS

Trong VPS (thu muc app root co `backend/`, `frontend/`, `ai_service/`, `infra/`):

```bash
bash infra/vps/check-secrets.sh .
bash infra/vps/deploy.sh .
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env ps
docker compose -f infra/docker-compose.yml --env-file infra/env/backend.env logs -f --tail=200
```

Luu y:
- Neu `FRONTEND_BIND_PORT=3000` bi trung cong, doi sang cong khac trong env.
- Neu search loi 500 do Lightning upstream (het credit/cold start), can khoi phuc upstream truoc.
- De chuc nang AI chay on dinh, dich vu Lightning/tracking can san sang 24/7 hoac co co che wakeup phu hop.

## 8) Checklist cho nguoi phat trien tiep theo

- Khong dua logic model cot loi ve gateway/frontend.
- Khong mo lai stack/fallback cu.
- Moi thay doi lien quan pipeline phai cap nhat tai lieu nay + test end-to-end `search`.
- Neu can doi ha tang docker/env, phai giu nguyen ten service network (`backend`, `ai_service`, `postgres`, `frontend`) de tranh vo wiring.

## 9) Cac thay doi nho FE (có thể sửa nhanh)

### Login API
- `POST /api/v1/auth/register`: dang hoat dong, tra 200 khi tao user hop le.
- `POST /api/v1/auth/login` (identifier = username): dang hoat dong, tra 200 khi thong tin dung.

### Home guide (khi chua search)
- Block huong dan trang `/home` duoc doc tu:
  - `HOME_GUIDE_TITLE`
  - `HOME_GUIDE_LINES`
- Nguon hien tai: `frontend/lib/content/homeGuide` (khong nam trong `constants.ts`).

### Top n + phan trang ket qua
- Label dropdown: `Top n`.
- Option hien tai: `Top 10`, `Top 20`, `Top 30`, `Top 40`, `Top 50`.
- Mac dinh: `Top 50`.
- Tong ket qua bi cap boi `TOP_N_MAX = 50`.
- Moi lan `Next` hien thi them theo block `10` (`GRID_BATCH_SIZE = 10`).
- Dong trang thai duoi cung:
  - `Top {start}-{end}` khi chua toi cuoi.
  - `Top End` khi het ket qua.

### File da dong bo logic tren
- `frontend/lib/constants.ts`
- `frontend/components/search/TopKSelect.tsx`
- `frontend/features/search/SearchContext.tsx`
- `frontend/features/home/HomeView.tsx`

### Noi dung da loi thoi / khong con dung
- `frontend/lib/mock/videos.ts` hien khong duoc dung trong luong search that (chi la du lieu mock de tham khao).

