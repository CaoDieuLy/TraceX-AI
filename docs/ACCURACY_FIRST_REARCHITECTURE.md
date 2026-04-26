# Accuracy-First MTMC Rearchitecture

Last reviewed: 2026-04-17

## Goal

Replace the legacy `YOLOv3/v4 + Deep SORT + mars-small128/Torchreid + caption-only search`
stack with an orchestration contract that favors dense-scene recall, identity stability, and
fine-grained person retrieval.

## Verified external references used for the redesign

- RF-DETR was selected as the primary detector because it is presented as a real-time
  DETR family model with strong COCO accuracy and NMS-free inference.
  Source: `https://iclr.cc/virtual/2026/poster/10007257`
- For strict detector naming, this repo now targets the upstream RF-DETR API contract from
  `roboflow/rf-detr`, specifically `RFDETR2XLarge` / `rfdetr-2xlarge`, which upstream documents
  behind the `rfdetr[plus]` extra and the standard `model.predict(image, threshold=...)` API.
  Source: `https://github.com/roboflow/rf-detr`
- HOTA remains the primary tracking metric through TrackEval.
  Source: `https://github.com/JonathonLuiten/TrackEval`
- SOLIDER is kept as the main global Re-ID backbone candidate because its public benchmark table
  still shows it as a strong MSMT17 baseline.
  Source: `https://github.com/tinyvision/SOLIDER`
- KPR is used as the occlusion and multi-person ambiguity extension.
  Source: `https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/10180_ECCV_2024_paper.php`
- ITSELF is used as the semantic retrieval reference for fine-grained text-to-person search.
  Source: `https://openaccess.thecvf.com/content/WACV2026/html/Nguyen_ITSELF_Attention_Guided_Fine-Grained_Alignment_for_Vision-Language_Retrieval_WACV_2026_paper.html`
- OCMCTrack informs the corrective matching cascade in online MTMC.
  Source: `https://publica.fraunhofer.de/entities/publication/b8475da1-6fa2-4696-8d3d-78323b3d50cf`
- NVIDIA Sparse4D documentation is used as the reference point for world-coordinate integration.
  Source: `https://docs.nvidia.com/vss/3.1.0/warehouse-docs/Sparse4D.html`

## Important scope note

This repository now exposes a single strict pipeline contract and emits manifests that describe
the intended stack. The repo does not bundle RF-DETR, SOLIDER, KPR, ITSELF, or TrackEval
weights/runtimes. Those must be supplied by the external strict runtime before production
inference can match the declared contract.

## What changed in code

- `tracking-service` now exposes a structured runtime configuration instead of a generic mock flag.
- Tracking runs emit a `strict_tracking_manifest_v1` JSON artifact beside the output clip.
- The active stack is fixed to `RF-DETR 2x-large + OCMCTrack-style corrective cascade + SOLIDER + KPR + ITSELF + TrackEval HOTA`;
  alternate stacks and hyperparameter overrides are disabled.
- The strict contract now carries tuned hyperparameters for:
  - detector confidence and query budget
  - tracker association gates, occlusion handling, corrective buffer, and world-speed gate
  - Re-ID weighting and rerank parameters
  - semantic retrieval fetch depth and ranking weights
  - ingest FPS, minimum track length, minimum person area, and track IoU
- The active ingest contract after `move.py` is fixed to:
  - `.mp4` container with H.265 video payload
  - decode + sampling at `5 fps`
  - person detection with `RFDETR2XLarge`
  - local tracking per video, where each 10-minute clip is independent
  - tracklet quality scoring before the unified tracklet feature pipeline
- The tracking service now resolves hardware plans for `L4`, `T4`, `A100`, and `H100`.
- Each hardware plan declares:
  - GPU stream count
  - CPU decode and crop worker pools
  - bounded prefetch queues
  - detector, Re-ID, VLM, and embedding batch sizes
  - precision policy and transfer settings
- When `torch` is present, the tracking service now applies runtime tuning at startup/query time:
  - `torch.set_num_threads(...)`
  - `torch.set_num_interop_threads(...)`
  - TF32 enablement
  - `cudnn.benchmark`
- Docker defaults were tightened for single-GPU serving:
  - `UVICORN_WORKERS=1`
  - `gpus: all`
  - `shm_size: 8gb`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Metadata import now preserves richer search fields:
  - `appearance_summary`
  - `semantic_attributes`
  - `visibility_scores`
  - `world_position`
- Legacy vector search now reranks candidates with a lightweight multi-signal ensemble:
  cosine similarity + semantic token overlap + visibility confidence.

## Recommended next implementation steps

1. Mount detector and Re-ID artifacts behind the tracking service and replace the current
   manifest-only runtime with true inference.
2. Add camera calibration JSON files and world-projection logic for each production scene.
3. Wire TrackEval into CI to block regressions on HOTA, DetA, and AssA.
4. Replace sentence-only embeddings with ITSELF-compatible image-text embeddings during ingest.
5. Introduce corrective cascade storage so online associations can be revised retroactively.

## L4 recommendation

For the current `1x NVIDIA L4, 16 vCPU, 64 GB RAM` target machine, prefer:

- `1` GPU-serving process
- `3` CUDA streams
- `8` CPU decode workers
- `6` CPU crop workers
- detector batch size `20`
- Re-ID batch size `256`
- VLM batch size `24`

Do not scale `uvicorn` workers above `1` on a single 24 GB L4 unless you intentionally want
multiple isolated model replicas and have measured the VRAM tradeoff.
