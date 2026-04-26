# Repo Audit Report

Ngay cap nhat: 2026-04-26

## Scope

Bao cao nay duoc rut gon sau cleanup strict pipeline. Cac mo ta ve local substitute runtime cu da bi go bo khoi tai lieu nay.

## Kien truc hien tai

- `frontend/`: UI Next.js.
- `backend/services/api-gateway/`: public API facade.
- `backend/services/metadata-service/`: system of record cho user, video, queue, candidate.
- `ai_service/`: internal AI boundary cho search va tracking proxy.
- `backend/services/tracking-service/`: strict pipeline contract va boundary toi external runtime.
- `infra/`: Docker/VPS wiring.

## Strict pipeline

Chi mot pipeline duoc chap nhan:

- `RF-DETR 2x-large`
- `OCMCTrack-style corrective cascade`
- `SOLIDER + KPR`
- `ITSELF`
- `TrackEval HOTA`

Repo khong duoc them profile moi, khong mo override hyperparameter, va khong fallback sang runtime khac.

## Rủi ro con lai

- External strict runtime/weights khong nam trong repo.
- Search end-to-end phu thuoc tracking upstream san sang.
- Database van dung `Base.metadata.create_all()` thay vi migration chuan.
- Thu muc monolith cu con mot so artifact da track trong git.

## Khuyen nghi

1. Cau hinh va test upstream strict runtime.
2. Dung Alembic cho DB migration.
3. Don artifact monolith cu neu khong con can.
4. Them integration test cho `/search` va artifact endpoints.
