# Bao cao ky thuat repository deep dive - trang thai sau strict cleanup

Ngay cap nhat: 2026-04-26

## Ket luan hien tai

Repo da chuyen sang hop dong strict pipeline:

- detector: `RF-DETR 2x-large`
- tracker: `OCMCTrack-style corrective cascade`
- ReID: `SOLIDER + KPR`
- semantic search: `ITSELF`
- evaluation: `TrackEval HOTA`

Khong con cho phep local substitute runtime trong `backend/legacy-engine/src_vlm`.
Nhung endpoint can inference strict se fail-fast neu external strict runtime chua duoc cau hinh.

## Module chinh

- `frontend/`: Next.js UI.
- `backend/services/api-gateway/`: public gateway, khong goi tracking upstream truc tiep.
- `backend/services/metadata-service/`: auth, DB, queue, candidate catalog, shortlist.
- `ai_service/`: noi bo cho search va proxy tracking upstream.
- `backend/services/tracking-service/`: strict pipeline contract, artifact endpoints, external runtime boundary.
- `infra/`: Docker Compose, Dockerfiles, env, VPS scripts.

## Diem can chu y

- Strict model weights/runtimes khong nam trong repo nay.
- `tracking-service` khong duoc tu dong fallback sang implementation local cu.
- `metadata-service rank_candidates` van goi tracking upstream de semantic rank; upstream phai la runtime strict.
- Cac file env/secret khong duoc commit them.

## Viec nen lam tiep

1. Cau hinh external runtime thuc thi dung profile strict.
2. Bo sung healthcheck chi tiet cho runtime strict.
3. Them integration test cho `/search` voi upstream strict da san sang.
4. Neu can ingestion local, tao module moi dung dung stack strict thay vi khoi phuc runtime cu.
