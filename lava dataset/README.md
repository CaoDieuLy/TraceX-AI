# LAVA Multimodal Search Engine

This folder contains a full implementation path for Team 119:

- text-to-video moment retrieval
- image-to-video retrieval
- hybrid text + image search
- sparse + dense + CLIP fusion
- FastAPI backend
- Streamlit demo UI
- selective download from Hugging Face

The project is built around the LAVA traffic dataset and treats the problem as a search engine, not as an object detection training task.

## Why LAVA

LAVA is a strong fit for the traffic search problem because it already contains:

- traffic surveillance videos
- frame-level object annotations
- captions per object
- track ids across time

That is enough to build searchable moments directly from existing metadata, then upgrade to visual retrieval with CLIP once videos are available locally.

## Project layout

```text
lava dataset/
├── lava_search/
│   ├── api.py
│   ├── cli.py
│   ├── config.py
│   ├── dataset.py
│   ├── indexer.py
│   ├── models.py
│   ├── streamlit_app.py
│   ├── video_tools.py
│   └── vision.py
├── tests/
├── requirements.txt
└── .gitignore
```

## Install

```powershell
cd "D:\python ky 9\A20-App-119\lava dataset"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Recommended strongest setup:

- `sentence-transformers/all-mpnet-base-v2` for dense text search
- `ViT-L-14` CLIP for visual retrieval
- `laion2b_s32b_b82k` pretrained weights

## Input / output

### Supported query modes

- text only
- image only
- text + image hybrid

### Search output

Each result returns:

- `location`
- `split`
- `video_path`
- `start_frame`, `end_frame`
- `start_second`, `end_second`
- `captions`
- `track_id`
- `score`
- score breakdown by retrieval branch
- representative visual asset if available

Optional:

- export a short mp4 clip around the matched moment

## Full pipeline

### 1. Download selected data from Hugging Face

Labels only:

```powershell
python -m lava_search download --locations amsterdam --splits test
```

Labels + videos:

```powershell
python -m lava_search download --locations amsterdam --splits test --include-videos
```

### 1b. One-command bootstrap

For an end-to-end production-style setup, use:

```powershell
python -m lava_search bootstrap --locations amsterdam --splits test --include-videos --profile strongest
```

This command:

- downloads the selected split
- builds searchable moments
- extracts visual assets if videos exist
- creates the runtime manifest
- writes the final bundle under `artifacts/full/`

### 2. Prepare visual assets

This extracts a few representative full frames and object crops per moment.

```powershell
python -m lava_search prepare-assets --locations amsterdam --splits test
```

### 3. Build the strongest bundle

This builds:

- sparse TF-IDF branch
- dense sentence-transformer branch
- CLIP image branch
- hybrid-ready bundle metadata

```powershell
python -m lava_search build --profile strongest --locations amsterdam --splits test
```

The bundle is written by default to:

```text
lava dataset/artifacts/full/
```

### 4. Search by text

```powershell
python -m lava_search search --query-text "red car turning left" --top-k 5
```

JSON output:

```powershell
python -m lava_search search --query-text "pedestrian crossing the road" --json
```

### 5. Search by image

```powershell
python -m lava_search search --image-path ".\query.jpg" --top-k 5
```

### 6. Hybrid search

```powershell
python -m lava_search search --query-text "white truck near intersection" --image-path ".\query.jpg"
```

Custom branch weights:

```powershell
python -m lava_search search --query-text "red sedan" --weights "sparse=0.1,dense=0.4,clip_text=0.2,clip_image=0.3"
```

### 7. Export a demo clip

```powershell
python -m lava_search clip --query-text "red car turning left" --rank 1 --output ".\clips\result.mp4"
```

## Run as a service

### FastAPI

```powershell
python -m lava_search serve-api --host 127.0.0.1 --port 8000
```

Main endpoints:

- `GET /health`
- `GET /manifest`
- `POST /search`
- `POST /clip`

### Streamlit demo

```powershell
python -m lava_search demo --port 8501
```

## Profiles

- `strongest`: sparse + dense + CLIP
- `balanced`: same as strongest, for future tuning
- `text-only`: sparse + dense, no CLIP
- `lite`: sparse only

## Artifact layout

```text
artifacts/full/
├── bundle.json
├── moments.json
├── sparse_vectorizer.pkl
├── sparse_matrix.npz
├── dense_embeddings.npy
├── dense.index
├── clip_image_embeddings.npy
├── clip_moment_indices.npy
└── clip.index
```

Visual assets are stored under:

```text
artifacts/visual_assets/
```

## Notes

- If you want CLIP image retrieval, videos must be downloaded locally first.
- If your machine is weak, start with one location and one split.
- If you do not have `ffmpeg`, clip export will not work.
- If `faiss-cpu` is missing, the project still keeps `.npy` embeddings for scoring, but FAISS index files will not be written.

## Practical recommendation

For a strong but still realistic demo:

1. Download one location, one split, with videos.
2. Run `prepare-assets`.
3. Build with `--profile strongest`.
4. Demo text query first.
5. Demo image query second.
6. Export one short clip for presentation.

## Near real-time indexing

The project now includes polling-based reindexing for production-style operation.

Rebuild once:

```powershell
python -m lava_search rebuild --locations amsterdam --splits test --profile strongest
```

Watch the dataset directory and rebuild automatically:

```powershell
python -m lava_search watch-index --locations amsterdam --splits test --poll-seconds 10
```

This is near-real-time indexing, not a streaming pipeline. It is the practical version for this project scope.

## PowerShell shortcuts

Run the full pipeline and start the API:

```powershell
.\run_end_to_end.ps1 -Location amsterdam -Split test -Port 8000
```

Run the watcher in a second terminal:

```powershell
.\run_watch_index.ps1 -Location amsterdam -Split test -PollSeconds 10
```
