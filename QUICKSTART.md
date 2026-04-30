# QUICKSTART — Chạy hệ thống từ máy cá nhân (không cần GPU)

Máy cá nhân chỉ cần: Python 3.11+, git, sshpass, browser.
GPU → LightningAI. Storage → Google Drive + VPS.

---

## Chuyển sang máy mới

**Trên máy cũ** — export secrets thành 1 file:
```bash
bash scripts/export_secrets.sh
# → mcpt_secrets_20260430.tar.gz
```

**Chuyển file đó sang máy mới** (LightningAI UI download, USB, SCP...):
```bash
# Trên máy mới:
git clone <REPO_URL> && cd A20-App-119
bash scripts/import_secrets.sh /path/to/mcpt_secrets_20260430.tar.gz
bash scripts/sync_secrets.sh --vps
```

**Nếu chuyển LightningAI Studio mới** — thêm bước download model weights (vì shared filesystem khác):
```bash
# Trong Studio mới, chạy trong terminal:
cd /teamspace/studios/this_studio/A20-App-119
python3 - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
import os

weights = "storage/model-weights"
os.makedirs(f"{weights}/rf-detr", exist_ok=True)
os.makedirs(f"{weights}/transreid-reid", exist_ok=True)

# RF-DETR 2XLarge (~485MB)
hf_hub_download("kaiyangzhou/osnet", "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth", local_dir=f"{weights}/transreid-reid")

# TransReID (~330MB)
hf_hub_download("umair894/KAT-ReID-MSMT17", "transformer_120.pth", local_dir=f"{weights}/transreid-reid")

# VideoMAE Large (~1.3GB)
snapshot_download("MCG-NJU/videomae-large", local_dir=f"{weights}/videomae-action", ignore_patterns=["*.msgpack","flax*","tf_*","pytorch_model.bin"])

# SigLIP2 — tự download qua open_clip lần đầu load
print("Done. SigLIP2 sẽ auto-download lần đầu start tracking service (~3.4GB)")
PY
```

---

## Bước 1 — Điền secrets (làm 1 lần)

```bash
cp secrets/master.env.example secrets/master.env
nano secrets/master.env
```

Các giá trị cần điền:

| Biến | Lấy ở đâu |
|------|-----------|
| `VPS_HOST` | IP VPS |
| `VPS_PASSWORD` | Password VPS root |
| `POSTGRES_PASSWORD` | Tự đặt (giữ nguyên nếu VPS đã có) |
| `JWT_SECRET_KEY` | Tự đặt random string |
| `LIGHTNING_API_BASE_URL` | LightningAI UI → API Builder → Settings → URL |
| `TRACKING_SERVICE_URL` | Giống trên |
| `LIGHTNING_API_TOKEN` | Tự đặt (phải match với Bearer token trong start script) |
| `GOOGLE_DRIVE_ROOT_FOLDER_ID` | URL Drive folder VinUni/ → lấy ID |
| `GOOGLE_DRIVE_SOURCE_STORAGE_FOLDER_ID` | URL Drive folder Storage/ → lấy ID |
| `NEXT_PUBLIC_API_GATEWAY_URL` | `http://<VPS_HOST>` |

---

## Bước 2 — Sync secrets ra tất cả nơi + deploy VPS

```bash
# Sync local files + push lên VPS + restart Docker services
bash scripts/sync_secrets.sh --vps
```

Nếu lần đầu deploy VPS (chưa có Docker stack):
```bash
ssh root@<VPS_HOST> "cd /opt/mcpt/A20-App-119 && git pull && bash infra/vps/deploy.sh ."
```

---

## Bước 3 — Start LightningAI API Builder

1. Vào LightningAI Studio → **API Builder** → `tracking-service`
2. Machine: **1 × L4**
3. On start command:
   ```
   bash A20-App-119/scripts/start_tracking_service_api_builder.sh
   ```
4. Click **Start** — Auto start = ON

Đợi ~3–5 phút để tải tất cả model weights (RF-DETR, TransReID, VideoMAE, SigLIP2).

Kiểm tra:
```bash
curl -H "Authorization: Bearer <LIGHTNING_API_TOKEN>" \
  https://8000-<HASH>.cloudspaces.litng.ai/health
# {"status":"ok","service":"tracking-service"}
```

---

## Bước 4 — Move video từ Drive Temp/ → Storage/

```bash
# Dry run xem trước
python3 move.py --dry-run

# Move thực
python3 move.py
```

Queue worker trên VPS tự động detect sau 30s. Hoặc trigger ngay:
```bash
ssh root@<VPS_HOST> "docker exec mcpt-backend python -m metadata_app.queue_worker --once"
```

---

## Bước 5 — Mở website

```
http://<VPS_HOST>
```

---

## Khi LightningAI URL thay đổi (sau mỗi lần restart)

```bash
# Cập nhật 2 dòng trong secrets/master.env:
# LIGHTNING_API_BASE_URL=https://8000-<NEW_HASH>.cloudspaces.litng.ai
# TRACKING_SERVICE_URL=https://8000-<NEW_HASH>.cloudspaces.litng.ai
nano secrets/master.env

# Sync + restart VPS services
bash scripts/sync_secrets.sh --vps
```

---

## Debug commands

```bash
# Xem log VPS
ssh root@<VPS_HOST> "docker logs -f mcpt-backend --tail=50"

# Check DB
ssh root@<VPS_HOST> "docker exec mcpt-postgres psql -U mcpt_user -d video_tracking \
  -c 'SELECT COUNT(*) FROM person_candidates;'"

# Test search API trực tiếp
curl -X POST http://<VPS_HOST>/search \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"query": "người áo xanh", "top_k": 5}'

# Check LightningAI GPU info
curl -H "Authorization: Bearer <LIGHTNING_API_TOKEN>" \
  https://8000-<HASH>.cloudspaces.litng.ai/api/v1/runtime-config/hardware \
  | python3 -m json.tool | grep -E "gpu_model|detector_batch"
```
