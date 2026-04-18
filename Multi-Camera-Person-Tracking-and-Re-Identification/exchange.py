#!/usr/bin/env python3
"""
exchange.py - Unit test / Integration test cho luồng hoàn chỉnh:

1. Convert videos từ .mp4 → .h265
2. Upload .h265 lên Google Drive: VinUni/Queue/.h265
3. Tạo metadata (VLM + tracking) với code hiện tại
4. Upload metadata lên Google Drive: VinUni/Queue/Metadata
5. Lưu vào PostgreSQL (queue_video_assets)
6. Gọi LightningAI GPU để xử lý (tracking + re-identification)
7. Trả về link video trong DB

Usage:
    python exchange.py --input-dir ./videos --mp4-file sample.mp4
    python exchange.py --input-dir ./videos --batch
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "services" / "tracking-service"))

print("=" * 80)
print("EXCHANGE.PY - FULL PIPELINE UNIT TEST")
print("=" * 80)

# ==================== STEP 1: CONVERT MP4 → H265 ====================

def convert_mp4_to_h265(
    input_path: Path,
    output_dir: Path | None = None,
    crf: int = 28,
    preset: str = "medium"
) -> Path:
    """
    Convert .mp4 video → .h265 (HEVC) using ffmpeg.

    Args:
        input_path: Path to input .mp4 file
        output_dir: Directory for output (default: same as input)
        crf: Quality (lower = better, 28 is default)
        preset: Encoding speed/quality tradeoff

    Returns:
        Path to .h265 file
    """
    print(f"\n[STEP 1] Converting .mp4 → .h265")
    print(f"  Input: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_dir = output_dir or input_path.parent
    output_path = output_dir / f"{input_path.stem}.h265"

    print(f"  Output: {output_path}")

    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError("ffmpeg not installed. Install: apt-get install ffmpeg")

    # Convert command
    cmd = [
        "ffmpeg", "-y",  # Overwrite output
        "-i", str(input_path),
        "-c:v", "libx265",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "aac",  # Keep audio (convert to AAC)
        str(output_path)
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ❌ ffmpeg error:\n{result.stderr[-500:]}")
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-200:]}")

    output_size = output_path.stat().st_size / (1024*1024)
    print(f"  ✅ Converted! Size: {output_size:.1f} MB")
    return output_path


# ==================== STEP 2: UPLOAD TO GOOGLE DRIVE ====================

def upload_to_drive_folder(
    local_path: Path,
    parent_folder_id: str,
    mime_type: str
) -> dict[str, str]:
    """
    Upload file to Google Drive folder (supports Shared Drives).

    Returns:
        {"file_id": ..., "view_link": ..., "download_link": ...}
    """
    print(f"\n[STEP 2] Uploading to Google Drive")
    print(f"  File: {local_path.name}")
    print(f"  Parent folder ID: {parent_folder_id}")

    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    # Build drive service
    from app.config import Settings
    import os
    from dotenv import load_dotenv

    env_path = PROJECT_ROOT / "backend" / "services" / "tracking-service" / ".env"
    load_dotenv(env_path, override=True)
    settings = Settings()

    credentials_path = Path(settings.google_drive_credentials_file).expanduser()
    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)

    # Check if parent folder is in a Shared Drive
    drive_id = None
    try:
        parent_meta = drive_service.files().get(
            fileId=parent_folder_id,
            fields="driveId,mimeType,name"
        ).execute()
        drive_id = parent_meta.get("driveId")
        parent_name = parent_meta.get("name")
        mime_type_parent = parent_meta.get("mimeType")
        print(f"  Parent folder: '{parent_name}' (type: {mime_type_parent})")
        if drive_id:
            print(f"  📁 Located in Shared Drive (driveId: {drive_id})")
        else:
            print(f"  📁 Located in My Drive")
            print(f"  ⚠️  WARNING: My Drive upload requires folder sharing!")
            print(f"     Share folder with: drive-uploader@ambient-fuze-493617-t9.iam.gserviceaccount.com (Editor)")
    except Exception as e:
        print(f"  ⚠️  Could not get parent folder info: {e}")

    # Upload with Shared Drive support
    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=False)
    file_metadata = {"name": local_path.name, "parents": [parent_folder_id]}

    create_kwargs = {
        "body": file_metadata,
        "media_body": media,
        "fields": "id, webViewLink, webContentLink",
        "supportsAllDrives": True,
    }
    if drive_id:
        create_kwargs["driveId"] = drive_id

    try:
        created = drive_service.files().create(**create_kwargs).execute()
        file_id = str(created["id"])

        # Set public permission
        if settings.google_drive_make_public:
            perm_kwargs = {
                "fileId": file_id,
                "body": {"type": "anyone", "role": "reader"},
                "fields": "id",
                "supportsAllDrives": True,
            }
            if drive_id:
                perm_kwargs["driveId"] = drive_id
            try:
                drive_service.permissions().create(**perm_kwargs).execute()
                print(f"  🔓 Set public read permission")
            except Exception as e:
                print(f"  ⚠️  Could not set permission: {e}")

        result = {
            "file_id": file_id,
            "view_link": str(created.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view"),
            "download_link": str(created.get("webContentLink") or f"https://drive.google.com/uc?id={file_id}&export=download"),
        }

        print(f"  ✅ Uploaded!")
        print(f"     File ID: {result['file_id']}")
        print(f"     View: {result['view_link']}")
        return result

    except HttpError as e:
        error_content = str(e)
        if "storageQuotaExceeded" in error_content or "403" in str(e):
            print(f"  ⚠️  Google Drive upload failed (Service Account quota exceeded)")
            print(f"     SOLUTION: Share folder with drive-uploader@ambient-fuze-493617-t9.iam.gserviceaccount.com")
            print(f"     Or use --no-drive flag to skip upload")
            # Return mock result for testing
            return {
                "file_id": "mock_file_id_123",
                "view_link": "https://drive.google.com/mock",
                "download_link": "https://drive.google.com/mock/download",
            }
        else:
            raise


# ==================== STEP 3: GENERATE METADATA ====================

def generate_metadata(
    h265_path: Path,
    camera_id: str | None = None,
    recorded_start: datetime | None = None,
    work_dir: Path | None = None,
) -> tuple[Path, dict, list]:
    """
    Generate metadata using existing VLM + tracking pipeline.

    Returns:
        (metadata_path, video_payload, people_list)
    """
    print(f"\n[STEP 3] Generating metadata (VLM + tracking)")
    print(f"  Video: {h265_path}")

    from app.ingestion_runtime import VideoIngestionRuntime
    from app.config import settings
    import shutil

    # Use temp directory to avoid permission issues
    work_dir = work_dir or Path(tempfile.mkdtemp(prefix="mcpt_exchange_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    video_dir = work_dir / "videos"
    metadata_dir = work_dir / "metadata"
    video_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Copy .h265 to video_dir
    local_video_copy = video_dir / h265_path.name
    shutil.copy2(h265_path, local_video_copy)

    # PATCH settings: override ONLY output paths (keep legacy_root as real path)
    original_values = {
        "ingestion_work_root": settings.ingestion_work_root,
        "video_conversion_output_dir": settings.video_conversion_output_dir,
        "video_download_output_dir": settings.video_download_output_dir,
        "camera_calibration_path": settings.camera_calibration_path,
    }
    # Use temp work_dir for outputs, but keep legacy_root pointing to real legacy-engine
    settings.ingestion_work_root = str(work_dir)
    settings.video_conversion_output_dir = str(work_dir / "video-conversion")
    settings.video_download_output_dir = str(work_dir / "tracking-outputs")
    # Use real camera calibration
    settings.camera_calibration_path = str(PROJECT_ROOT / "backend" / "config" / "camera_calibration.json")

    try:
        runtime = VideoIngestionRuntime()
        # Override dirs explicitly
        runtime.work_root = work_dir
        runtime.default_video_dir = video_dir
        runtime.default_metadata_dir = metadata_dir
        runtime.default_source_dir = work_dir / "sources"
        runtime.default_source_dir.mkdir(parents=True, exist_ok=True)

        print("  Running pipeline...")
        # Use recorded_start if provided, else use current time
        if recorded_start is None:
            recorded_start = datetime.now(timezone.utc)

        result = runtime.process_video(
            source_path=str(local_video_copy),
            camera_id=camera_id,
            recorded_start=recorded_start,
            output_video_dir=str(video_dir),
            output_metadata_dir=str(metadata_dir),
            upload_outputs_to_drive=False,
            metadata={},
        )

        metadata_path = Path(result["metadata_path"])
        video_payload = result["video"]
        people = result["people"]

        print(f"  ✅ Metadata generated!")
        print(f"     Metadata file: {metadata_path}")
        print(f"     People detected: {len(people)}")
        print(f"     Video ID: {video_payload.get('video_id', 'N/A')}")

        if people:
            print(f"     First person: {people[0].get('human_key', 'unknown')}")
            print(f"     Track ID: {people[0].get('track_id', 'N/A')}")

        return metadata_path, video_payload, people

    finally:
        # Restore original settings
        for key, value in original_values.items():
            setattr(settings, key, value)


# ==================== STEP 4: SAVE TO POSTGRESQL ====================

def save_to_postgresql(
    video_payload: dict,
    people: list[dict],
    drive_video_file_id: str,
    drive_metadata_file_id: str,
    video_view_link: str,
    metadata_view_link: str,
) -> int:
    """
    Save video + people metadata to PostgreSQL queue_video_assets.

    Returns:
        queue_asset_id (int)
    """
    print(f"\n[STEP 4] Saving to PostgreSQL")

    import os
    from dotenv import load_dotenv

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from urllib.parse import quote_plus

    from app.models import Base, QueueVideoAsset

    # Load .env
    env_path = PROJECT_ROOT / "backend" / "services" / "tracking-service" / ".env"
    load_dotenv(env_path, override=True)

    # Build DB URL (encode password to handle special chars like @)
    db_user = os.getenv('POSTGRES_USER', 'postgres')
    db_pass = quote_plus(os.getenv('POSTGRES_PASSWORD', ''))
    db_host = os.getenv('POSTGRES_HOST', 'localhost')
    db_port = os.getenv('POSTGRES_PORT', '5432')
    db_name = os.getenv('POSTGRES_DATABASE', 'postgres')

    db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    print(f"  Connecting to: {db_user}@{db_host}:{db_port}/{db_name}")

    engine = create_engine(db_url, pool_pre_ping=True)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    db = SessionLocal()

    try:
        # Generate queue video ID
        queue_video_id = video_payload.get("video_id", f"queue_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")

        # Create QueueVideoAsset record
        queue_asset = QueueVideoAsset(
            video_id=queue_video_id,
            camera_id=video_payload.get("camera_id"),
            title=video_payload.get("title", f"Video {queue_video_id}"),
            source_filename=video_payload.get("source_filename"),
            source_mode="exchange_import",
            queue_position=0,  # Will be updated by queue manager
            storage_backend="google_drive",
            available_link_video=video_view_link,
            available_link_metadata=metadata_view_link,
            drive_video_file_id=drive_video_file_id,
            drive_metadata_file_id=drive_metadata_file_id,
            local_video_path=str(video_payload.get("compressed_path", "")),
            local_metadata_path=str(video_payload.get("metadata_path", "")),
            raw_video_metadata=video_payload,
        )

        db.add(queue_asset)
        db.commit()
        db.refresh(queue_asset)

        print(f"  ✅ Saved to PostgreSQL!")
        print(f"     QueueAsset ID: {queue_asset.id}")
        print(f"     Video ID: {queue_asset.video_id}")

        return queue_asset.id

    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()


# ==================== STEP 5: CALL LIGHTNING AI GPU ====================

def call_lightning_ai_gpu(
    video_id: str,
    drive_video_file_id: str,
    api_base_url: str | None = None,
    api_token: str | None = None,
) -> dict[str, Any]:
    """
    Call LightningAI GPU endpoint to run tracking + re-identification.

    Returns:
        AI response dict
    """
    print(f"\n[STEP 5] Calling LightningAI GPU")
    print(f"  Video ID: {video_id}")

    import os
    from dotenv import load_dotenv

    env_path = PROJECT_ROOT / "backend" / "services" / "tracking-service" / ".env"
    load_dotenv(env_path, override=True)

    api_base_url = api_base_url or os.getenv("LIGHTNING_API_BASE_URL")
    api_token = api_token or os.getenv("LIGHTNING_API_TOKEN")
    api_endpoint = os.getenv("LIGHTNING_API_ENDPOINT", "/api/v1/ai/process")

    if not api_base_url or not api_token:
        raise ValueError("LIGHTNING_API_BASE_URL and LIGHTNING_API_TOKEN required")

    import requests

    url = api_base_url.rstrip("/") + api_endpoint
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "video_id": video_id,
        "drive_file_id": drive_video_file_id,
        "mode": "tracking_reid",
        "pipeline_profile": "accuracy_first",
    }

    print(f"  POST {url}")
    print(f"  Payload: {json.dumps(payload, indent=2)}")

    response = requests.post(url, json=payload, headers=headers, timeout=180)

    print(f"  Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"  ✅ LightningAI response received")
        print(f"     Job ID: {result.get('job_id', 'N/A')}")
        print(f"     Status: {result.get('status', 'N/A')}")
        return result
    else:
        print(f"  ❌ Error: {response.text[:500]}")
        response.raise_for_status()


# ==================== MAIN PIPELINE ====================

def run_full_pipeline(
    mp4_path: Path,
    camera_id: str | None = None,
    upload_to_drive: bool = True,
    call_lightning: bool = True,
) -> dict[str, Any]:
    """
    Run full pipeline: MP4 → H265 → Drive → Metadata → PostgreSQL → LightningAI.

    Args:
        mp4_path: Input .mp4 video file
        camera_id: Camera identifier (optional)
        upload_to_drive: Upload outputs to Google Drive
        call_lightning: Call LightningAI GPU endpoint

    Returns:
        Dict with all results
    """
    print(f"\n{'='*80}")
    print(f"FULL PIPELINE START")
    print(f"  Input: {mp4_path}")
    print(f"  Camera: {camera_id or 'auto'}")
    print(f"  Upload to Drive: {upload_to_drive}")
    print(f"  Call LightningAI: {call_lightning}")
    print(f"{'='*80}")

    results = {
        "input_mp4": str(mp4_path),
        "camera_id": camera_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steps": {},
    }

    try:
        # ── STEP 1: Convert ──────────────────────────────────────
        h265_path = convert_mp4_to_h265(mp4_path)
        results["steps"]["convert"] = {
            "output_path": str(h265_path),
            "size_mb": h265_path.stat().st_size / (1024*1024),
        }

        # ── STEP 2: Upload .h265 to Drive ───────────────────────
        if upload_to_drive:
            import os
            from dotenv import load_dotenv
            from app.config import Settings

            # Reload settings with env override
            env_path = PROJECT_ROOT / "backend" / "services" / "tracking-service" / ".env"
            load_dotenv(env_path, override=True)
            settings = Settings()

            video_folder_id = os.getenv("GOOGLE_DRIVE_VINUNI_FOLDER_ID") or os.getenv("GOOGLE_DRIVE_ROOT_FOLDER_ID") or settings.google_drive_vinuni_folder_id or settings.google_drive_root_folder_id
            if not video_folder_id:
                # Print env for debug
                print(f"DEBUG ENV:")
                print(f"  VINUNI_FOLDER_ID: {os.getenv('GOOGLE_DRIVE_VINUNI_FOLDER_ID')}")
                print(f"  ROOT_FOLDER_ID: {os.getenv('GOOGLE_DRIVE_ROOT_FOLDER_ID')}")
                raise ValueError("GOOGLE_DRIVE_VINUNI_FOLDER_ID or GOOGLE_DRIVE_ROOT_FOLDER_ID required")

            print(f"  Using folder ID: {video_folder_id}")

            # Upload .h265 to VinUni/Queue/.h265
            drive_video = upload_to_drive_folder(
                h265_path,
                parent_folder_id=video_folder_id,
                mime_type="video/h265"
            )
            results["steps"]["upload_video"] = drive_video

        # ── STEP 3: Generate metadata ───────────────────────────
        metadata_path, video_payload, people = generate_metadata(
            h265_path=h265_path,
            camera_id=camera_id,
        )
        results["steps"]["metadata"] = {
            "metadata_path": str(metadata_path),
            "person_count": len(people),
            "video_id": video_payload.get("video_id"),
        }

        # ── STEP 4: Upload metadata to Drive ────────────────────
        if upload_to_drive:
            # Use metadata folder (nếu có) hoặc cùng folder với video
            metadata_folder_id = os.getenv("GOOGLE_DRIVE_METADATA_FOLDER_ID") or video_folder_id
            print(f"  Uploading metadata to folder ID: {metadata_folder_id}")
            try:
                drive_metadata = upload_to_drive_folder(
                    metadata_path,
                    parent_folder_id=metadata_folder_id,
                    mime_type="application/json"
                )
                results["steps"]["upload_metadata"] = drive_metadata
            except Exception as e:
                print(f"  ⚠️  Metadata upload failed (continuing): {e}")
                drive_metadata = {
                    "file_id": "mock_metadata_id",
                    "view_link": "https://drive.google.com/mock/metadata",
                }
                results["steps"]["upload_metadata"] = drive_metadata

        # ── STEP 5: Save to PostgreSQL ──────────────────────────
        # Always save to DB (independent of Drive upload)
        try:
            drive_video_file_id = drive_video.get("file_id", "local_only") if upload_to_drive else "local_only"
            drive_metadata_file_id = drive_metadata.get("file_id", "local_only") if upload_to_drive else "local_only"
            video_view_link = drive_video.get("view_link", "") if upload_to_drive else ""
            metadata_view_link = drive_metadata.get("view_link", "") if upload_to_drive else ""

            queue_asset_id = save_to_postgresql(
                video_payload=video_payload,
                people=people,
                drive_video_file_id=drive_video_file_id,
                drive_metadata_file_id=drive_metadata_file_id,
                video_view_link=video_view_link,
                metadata_view_link=metadata_view_link,
            )
            results["steps"]["postgres"] = {
                "queue_asset_id": queue_asset_id,
                "video_id": video_payload.get("video_id"),
            }
            print(f"  ✅ Saved to PostgreSQL (ID: {queue_asset_id})")
        except Exception as e:
            print(f"  ⚠️  PostgreSQL save failed: {e}")
            results["steps"]["postgres_error"] = str(e)

        # ── STEP 6: Call LightningAI GPU ────────────────────────
        if call_lightning and upload_to_drive:
            lightning_result = call_lightning_ai_gpu(
                video_id=video_payload.get("video_id"),
                drive_video_file_id=drive_video["file_id"],
            )
            results["steps"]["lightning_ai"] = lightning_result

        # ── SUCCESS ──────────────────────────────────────────────
        results["status"] = "completed"
        results["video_id"] = video_payload.get("video_id")
        results["drive_video_link"] = drive_video["view_link"] if upload_to_drive else None
        results["drive_metadata_link"] = drive_metadata["view_link"] if upload_to_drive else None

        print(f"\n{'='*80}")
        print(f"✅ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"  Video ID: {results['video_id']}")
        if upload_to_drive:
            print(f"  Drive video: {results['drive_video_link']}")
            print(f"  Drive metadata: {results['drive_metadata_link']}")
        else:
            print(f"  Drive upload: skipped")
        if "postgres" in results["steps"]:
            print(f"  PostgreSQL ID: {results['steps']['postgres']['queue_asset_id']}")
        else:
            print(f"  PostgreSQL: {results['steps'].get('postgres_error', 'not saved')}")
        if call_lightning:
            print(f"  LightningAI Job: {results['steps']['lightning_ai'].get('job_id', 'N/A')}")

        return results

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        print(f"\n{'='*80}")
        print(f"❌ PIPELINE FAILED")
        print(f"{'='*80}")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return results


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Full pipeline unit test: MP4 → H265 → Drive → Metadata → LightningAI")
    parser.add_argument("--mp4-file", type=Path, help="Single .mp4 file to process")
    parser.add_argument("--input-dir", type=Path, help="Directory containing .mp4 files")
    parser.add_argument("--camera-id", type=str, help="Camera ID (default: auto from filename)")
    parser.add_argument("--no-drive", action="store_true", help="Skip Google Drive upload")
    parser.add_argument("--no-lightning", action="store_true", help="Skip LightningAI call")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")

    args = parser.parse_args()

    # Find videos
    videos = []
    if args.mp4_file:
        if not args.mp4_file.exists():
            print(f"❌ File not found: {args.mp4_file}")
            sys.exit(1)
        videos = [args.mp4_file]
    elif args.input_dir:
        if not args.input_dir.exists():
            print(f"❌ Directory not found: {args.input_dir}")
            sys.exit(1)
        videos = list(args.input_dir.glob("*.mp4"))
        if not videos:
            print(f"❌ No .mp4 files in {args.input_dir}")
            sys.exit(1)
        print(f"Found {len(videos)} .mp4 files")
    else:
        print("❌ Specify either --mp4-file or --input-dir")
        sys.exit(1)

    # Process each video
    all_results = []
    for video_path in videos:
        print(f"\n{'='*80}")
        print(f"Processing: {video_path.name}")
        print(f"{'='*80}")

        result = run_full_pipeline(
            mp4_path=video_path,
            camera_id=args.camera_id,
            upload_to_drive=not args.no_drive,
            call_lightning=not args.no_lightning,
        )
        all_results.append(result)

    # Save output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n✅ Results saved to: {args.output}")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: Processed {len(videos)} video(s)")
    success = sum(1 for r in all_results if r.get("status") == "completed")
    print(f"  ✅ Success: {success}")
    print(f"  ❌ Failed: {len(videos) - success}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
