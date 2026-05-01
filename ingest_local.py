#!/usr/bin/env python3
"""
ingest_local.py — Local pipeline runner (bypass VPS backend).

Flow: Google Drive Storage/ → LightningAI async queue → PostgreSQL DB

Usage:
    python ingest_local.py              # process all pending videos
    python ingest_local.py --dry-run    # list videos only, no processing
    python ingest_local.py --clear-db   # truncate DB tables before run

Config: reads from secrets/master.env (auto-detected).
DB   : set DATABASE_URL env var, or uses POSTGRES_* from master.env.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Bootstrap: load secrets/master.env
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
MASTER_ENV = REPO_ROOT / "secrets" / "master.env"

from dotenv import load_dotenv
if MASTER_ENV.exists():
    load_dotenv(MASTER_ENV, override=True)
    print(f"[config] Loaded {MASTER_ENV}")
else:
    print(f"[warn] {MASTER_ENV} not found — using environment variables only")

# ---------------------------------------------------------------------------
# Config (read after dotenv; --lightning-url arg overrides TRACKING_SERVICE_URL)
# ---------------------------------------------------------------------------
LIGHTNING_URL    = os.environ["TRACKING_SERVICE_URL"].rstrip("/")
LIGHTNING_TOKEN  = os.environ.get("LIGHTNING_API_TOKEN", "")
DRIVE_FOLDER_ID  = os.environ["GOOGLE_DRIVE_SOURCE_STORAGE_FOLDER_ID"]
POLL_INTERVAL    = int(os.environ.get("QUEUE_POLL_INTERVAL_SECONDS", "30"))
MAX_WAIT_SECONDS = 1800   # 30 min per video — increase for long videos

import httpx
from urllib.parse import quote_plus

PG_HOST = os.environ.get("LOCAL_POSTGRES_HOST") or "localhost"  # override Docker service name
PG_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
PG_USER = os.environ.get("POSTGRES_USER", "mcpt_user")
PG_PASS = os.environ.get("POSTGRES_PASSWORD", "")
PG_DB   = os.environ.get("POSTGRES_DATABASE") or os.environ.get("POSTGRES_DB", "video_tracking")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql+psycopg2://{quote_plus(PG_USER)}:{quote_plus(PG_PASS)}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_local")

# ---------------------------------------------------------------------------
# DB setup (SQLAlchemy, mirrors metadata-service models)
# ---------------------------------------------------------------------------
from sqlalchemy import (
    Boolean, Column, DateTime, Integer, String, Text,
    create_engine, text
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, sessionmaker

class Base(DeclarativeBase):
    pass

def _now():
    return datetime.now(timezone.utc)

class PersonCandidate(Base):
    __tablename__ = "person_candidates"
    id           = Column(Integer, primary_key=True)
    candidate_id = Column(String(255), nullable=False, unique=True, index=True)
    camera_id    = Column(String(255), nullable=True, index=True)
    video_id     = Column(String(255), nullable=True, index=True)
    track_id     = Column(String(255), nullable=True, index=True)
    human_key    = Column(String(255), nullable=True, index=True)
    frame_idx    = Column(Integer, nullable=True)
    search_text  = Column(Text, nullable=True)
    metadata_path= Column(String(1024), nullable=True)
    raw_metadata = Column(JSONB, nullable=False, default=dict)
    created_at   = Column(DateTime(timezone=True), default=_now)
    updated_at   = Column(DateTime(timezone=True), default=_now, onupdate=_now)

class QueueVideoAsset(Base):
    __tablename__ = "queue_video_assets"
    id                    = Column(Integer, primary_key=True)
    video_id              = Column(String(255), nullable=False, unique=True, index=True)
    camera_id             = Column(String(255), nullable=True, index=True)
    title                 = Column(String(255), nullable=False)
    source_filename       = Column(String(255), nullable=True)
    source_mode           = Column(String(64), nullable=True)
    queue_position        = Column(Integer, nullable=False, index=True, default=0)
    storage_backend       = Column(String(64), default="google_drive")
    available_link_video  = Column(String(2048), nullable=False)
    available_link_metadata= Column(String(2048), nullable=True)
    drive_video_file_id   = Column(String(255), nullable=True)
    drive_metadata_file_id= Column(String(255), nullable=True)
    local_video_path      = Column(String(2048), nullable=True)
    local_metadata_path   = Column(String(2048), nullable=True)
    raw_video_metadata    = Column(JSONB, nullable=False, default=dict)
    created_at            = Column(DateTime(timezone=True), default=_now)
    updated_at            = Column(DateTime(timezone=True), default=_now, onupdate=_now)


def setup_db(clear: bool = False):
    engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
    Base.metadata.create_all(bind=engine)
    log.info("[db] Tables ready: %s", list(Base.metadata.tables.keys()))
    if clear:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE person_candidates, queue_video_assets RESTART IDENTITY CASCADE"))
        log.info("[db] Cleared person_candidates + queue_video_assets")
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

# ---------------------------------------------------------------------------
# Google Drive helpers
# ---------------------------------------------------------------------------
def build_drive_service():
    sys.path.insert(0, str(REPO_ROOT))
    # Override container paths → local secrets/ folder
    os.environ["MCPT_SECRETS_ROOT"] = str(REPO_ROOT / "secrets")
    os.environ["MCPT_SHARED_ENV_FILE"] = str(REPO_ROOT / "secrets" / "shared.env")
    os.environ["MCPT_OAUTH2_TOKEN_FILE"] = str(REPO_ROOT / "secrets" / "oauth" / "oauth2_token.pickle")
    os.environ["MCPT_OAUTH2_CREDENTIALS_FILE"] = str(REPO_ROOT / "secrets" / "oauth" / "oauth2_credentials.json")
    from shared_secret_runtime import build_google_drive_oauth_service
    return build_google_drive_oauth_service()


def list_drive_mp4s(service, folder_id: str) -> list[dict]:
    """Recursively list all .mp4 files under a Drive folder."""
    results: list[dict] = []
    _walk_folder(service, folder_id, results)
    return results


def _walk_folder(service, folder_id: str, out: list[dict]):
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageSize=200,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            if f["mimeType"] == "application/vnd.google-apps.folder":
                _walk_folder(service, f["id"], out)
            elif f["name"].endswith(".mp4"):
                out.append(f)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def public_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0"

# ---------------------------------------------------------------------------
# LightningAI async job helpers
# ---------------------------------------------------------------------------
def _headers() -> dict:
    if LIGHTNING_TOKEN:
        return {"Authorization": f"Bearer {LIGHTNING_TOKEN}"}
    return {}


def check_ready() -> bool:
    try:
        r = httpx.get(f"{LIGHTNING_URL}/health", headers=_headers(), timeout=10)
        data = r.json()
        ready = r.status_code == 200 and bool(data.get("ready", False))
        if not ready:
            log.debug("[lightning] health status=%s ready=%s warmup=%s", r.status_code, data.get("ready"), data.get("warmup", {}).get("status"))
        return ready
    except Exception as exc:
        log.warning("[lightning] health check error: %s: %s", type(exc).__name__, exc)
        return False


def wait_ready(timeout: int = 1800) -> None:
    """Wait up to 30 min for LightningAI warmup. Prints status every 60s."""
    log.info("[lightning] Waiting for LightningAI to be ready (timeout=%ds)...", timeout)
    deadline = time.monotonic() + timeout
    last_log = time.monotonic()
    while time.monotonic() < deadline:
        if check_ready():
            log.info("[lightning] Ready!")
            return
        elapsed = int(time.monotonic() - (deadline - timeout))
        if time.monotonic() - last_log >= 60:
            log.info("[lightning] Still warming up... elapsed=%ds", elapsed)
            last_log = time.monotonic()
        time.sleep(15)
    raise TimeoutError(f"LightningAI not ready after {timeout}s — check LightningAI terminal")


def submit_job(file_id: str, filename: str, camera_id: str | None) -> str:
    payload = {
        "source_url": public_url(file_id),
        "source_filename": filename,
        "camera_id": camera_id,
        "metadata": {"source_mode": "storage_ingest"},
    }
    for attempt in range(1, 5):  # retry up to 4x on 503 (queue full / startup)
        r = httpx.post(
            f"{LIGHTNING_URL}/api/v1/ingestion/process",
            json=payload,
            headers=_headers(),
            timeout=60,
        )
        if r.status_code == 503:
            wait = 15 * attempt
            log.warning("[submit] 503 queue full, retry %d/4 in %ds ...", attempt, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        break
    data = r.json()
    job_id = data.get("job_id")
    if not job_id:
        raise ValueError(f"No job_id in response: {data}")
    return job_id


def poll_job(job_id: str) -> dict:
    """Poll until done or failed. Returns full result dict."""
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL)
        try:
            r = httpx.get(
                f"{LIGHTNING_URL}/api/v1/ingestion/status/{job_id}",
                headers=_headers(),
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise RuntimeError(f"Job {job_id} not found (server restarted?) — resubmit needed") from exc
            log.warning("[poll] %s failed: %s — retrying", job_id, exc)
            continue
        except httpx.HTTPError as exc:
            log.warning("[poll] %s failed: %s — retrying", job_id, exc)
            continue
        status = data.get("status")
        log.info("[poll] job=%s status=%s", job_id, status)
        if status == "done":
            return data.get("result", {})
        if status == "failed":
            raise RuntimeError(
                f"Job {job_id} failed: {data.get('error_type')}: {data.get('error')}"
            )
    raise TimeoutError(f"Job {job_id} not done within {MAX_WAIT_SECONDS}s")

# ---------------------------------------------------------------------------
# DB save helpers
# ---------------------------------------------------------------------------
def save_result(session, result: dict, file: dict) -> int:
    """Persist ingestion result to DB. Returns number of candidates saved."""
    people: list[dict] = result.get("people", [])
    camera_id = result.get("video", {}).get("camera_id") or file.get("camera_id")
    source_filename = result.get("video", {}).get("source_filename") or file["name"]

    saved = 0
    for person in people:
        cid = person.get("candidate_id") or person.get("id")
        if not cid:
            continue
        existing = session.query(PersonCandidate).filter_by(candidate_id=cid).first()
        if existing:
            existing.raw_metadata = person
            existing.updated_at = _now()
        else:
            session.add(PersonCandidate(
                candidate_id  = cid,
                camera_id     = camera_id,
                video_id      = person.get("video_id"),
                track_id      = str(person.get("track_id", "")),
                human_key     = person.get("human_key"),
                frame_idx     = person.get("frame_idx"),
                search_text   = person.get("search_text") or person.get("description"),
                raw_metadata  = person,
            ))
            saved += 1
    session.commit()
    return saved

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def parse_camera_id(filename: str) -> str | None:
    """Extract cam_XX from filename like cam_01_2026-04-28_11-00.mp4"""
    import re
    m = re.match(r"(cam_\d+)", filename, re.IGNORECASE)
    return m.group(1) if m else None


def _select_shard_files(all_files: list[dict], *, shard: int, total_shards: int, shard_mode: str) -> list[dict]:
    if total_shards <= 1:
        return all_files
    if shard_mode == "interleaved":
        return all_files[shard::total_shards]

    total_files = len(all_files)
    base = total_files // total_shards
    remainder = total_files % total_shards
    start = shard * base + min(shard, remainder)
    length = base + (1 if shard < remainder else 0)
    end = start + length
    return all_files[start:end]


def run(
    dry_run: bool = False,
    clear_db: bool = False,
    shard: int = 0,
    total_shards: int = 1,
    shard_mode: str = "contiguous",
):
    SessionLocal = setup_db(clear=clear_db)

    log.info("[drive] Listing mp4 files in Storage folder %s ...", DRIVE_FOLDER_ID)
    drive = build_drive_service()
    all_files = list_drive_mp4s(drive, DRIVE_FOLDER_ID)
    all_files.sort(key=lambda f: f["name"])
    log.info("[drive] Found %d .mp4 files total", len(all_files))

    # Interleaved sharding: shard 0 → [0,3,6,...], shard 1 → [1,4,7,...], shard 2 → [2,5,8,...]
    # Balances load evenly regardless of camera/time distribution.
    files = _select_shard_files(
        all_files,
        shard=shard,
        total_shards=total_shards,
        shard_mode=shard_mode,
    )
    if total_shards > 1:
        log.info(
            "[shard %d/%d] mode=%s assigned %d/%d videos",
            shard,
            total_shards - 1,
            shard_mode,
            len(files),
            len(all_files),
        )

    if dry_run:
        for f in files:
            print(f"  {f['name']}  ({f.get('size', '?')} bytes)  id={f['id']}")
        return

    wait_ready()

    total_saved = 0
    for i, f in enumerate(files, 1):
        filename  = f["name"]
        file_id   = f["id"]
        camera_id = parse_camera_id(filename)

        with SessionLocal() as session:
            already_done = session.query(PersonCandidate).filter_by(video_id=filename).first()
        if already_done:
            log.info("[%d/%d] SKIP %s — already in DB", i, len(files), filename)
            continue

        log.info("[%d/%d] %s  camera=%s", i, len(files), filename, camera_id)

        try:
            job_id = submit_job(file_id, filename, camera_id)
            log.info("[%d/%d] Submitted job_id=%s", i, len(files), job_id)

            result = poll_job(job_id)
            people_count = result.get("person_count", 0)
            tracklet_count = result.get("video", {}).get("tracklet_count", "?")
            log.info("[%d/%d] Done: %d people / %s tracklets", i, len(files), people_count, tracklet_count)

            with SessionLocal() as session:
                saved = save_result(session, result, {"name": filename, "camera_id": camera_id})
            total_saved += saved
            log.info("[%d/%d] Saved %d new candidates (total so far: %d)", i, len(files), saved, total_saved)

        except Exception as exc:
            log.error("[%d/%d] FAILED %s: %s", i, len(files), filename, exc)
            continue

    log.info("[done] Shard %d/%d finished. Videos=%d, candidates saved=%d",
             shard, total_shards - 1, len(files), total_saved)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local Drive→LightningAI→DB pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parallel execution (3 GPU servers, 50 videos):
  TRACKING_SERVICE_URL=https://gpu-0 python ingest_local.py --shard 0 --total-shards 3
  TRACKING_SERVICE_URL=https://gpu-1 python ingest_local.py --shard 1 --total-shards 3
  TRACKING_SERVICE_URL=https://gpu-2 python ingest_local.py --shard 2 --total-shards 3
""",
    )
    parser.add_argument("--dry-run", action="store_true", help="List videos only, no processing")
    parser.add_argument("--clear-db", action="store_true", help="Truncate DB tables before run")
    parser.add_argument("--shard", type=int, default=0, metavar="INDEX",
                        help="0-based shard index for parallel execution (default: 0)")
    parser.add_argument("--total-shards", type=int, default=1, metavar="TOTAL",
                        help="Total parallel shards, e.g. 3 for three concurrent processes (default: 1)")
    parser.add_argument("--shard-mode", choices=("contiguous", "interleaved"), default="contiguous",
                        help="How to split files across shards (default: contiguous)")
    parser.add_argument("--lightning-url", default="", metavar="URL",
                        help="Override TRACKING_SERVICE_URL for this shard")
    args = parser.parse_args()

    if args.lightning_url:
        LIGHTNING_URL = args.lightning_url.rstrip("/")

    run(
        dry_run=args.dry_run,
        clear_db=args.clear_db,
        shard=args.shard,
        total_shards=args.total_shards,
        shard_mode=args.shard_mode,
    )
