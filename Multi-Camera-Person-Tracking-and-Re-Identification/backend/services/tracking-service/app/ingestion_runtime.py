from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from .config import settings
from .cuda_runtime import configure_torch_runtime
from .execution_plan import resolve_execution_plan
from .legacy_runtime import legacy_workdir
from .pipeline_profiles import resolve_pipeline_profile


def _dict_or_empty(value: object) -> dict:
    return value if isinstance(value, dict) else {}


DRIVE_SCOPES = ("https://www.googleapis.com/auth/drive",)
ALREADY_COMPRESSED_SUFFIXES = {".h265", ".hevc"}


class VideoIngestionRuntime:
    def __init__(self) -> None:
        self.work_root = Path(settings.ingestion_work_root)
        self.default_video_dir = self.work_root / "videos"
        self.default_metadata_dir = self.work_root / "metadata"
        self.default_source_dir = self.work_root / "sources"
        self.default_video_dir.mkdir(parents=True, exist_ok=True)
        self.default_metadata_dir.mkdir(parents=True, exist_ok=True)
        self.default_source_dir.mkdir(parents=True, exist_ok=True)
        self._vlm_engine = None
        self._drive_service = None

    def _resolve_profile(self, metadata: dict | None = None) -> tuple[dict, dict, dict]:
        metadata = _dict_or_empty(metadata)
        profile_name = str(metadata.get("pipeline_profile") or settings.pipeline_profile).strip() or settings.pipeline_profile
        profile = resolve_pipeline_profile(profile_name, _dict_or_empty(metadata.get("hyperparameter_overrides")))
        hardware_profile, execution_plan = resolve_execution_plan(
            pipeline_profile=profile,
            gpu_profile_name=str(metadata.get("gpu_hardware_profile") or settings.gpu_hardware_profile),
            gpu_profile_overrides=_dict_or_empty(metadata.get("gpu_hardware_overrides")),
            gpu_count=settings.gpu_count,
            host_cpu_count=settings.host_cpu_count,
            host_ram_gb=settings.host_ram_gb,
        )
        return profile, hardware_profile, execution_plan

    def _get_vlm_engine(self):
        if self._vlm_engine is not None:
            return self._vlm_engine
        with legacy_workdir():
            from src_vlm.vlm_engine import VLM_Metadata_Engine

            self._vlm_engine = VLM_Metadata_Engine(use_mock=settings.tracking_use_mock)
        return self._vlm_engine

    def _build_drive_service(self):
        if not settings.google_drive_enabled:
            raise RuntimeError("Google Drive support is disabled on tracking-service.")
        if self._drive_service is not None:
            return self._drive_service
        credentials_path = Path(settings.google_drive_credentials_file).expanduser()
        if not credentials_path.exists():
            raise FileNotFoundError(f"Missing Google Drive credentials file: {credentials_path}")
        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_path),
            scopes=list(DRIVE_SCOPES),
        )
        self._drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        return self._drive_service

    @staticmethod
    def _drive_view_link(file_id: str) -> str:
        return f"https://drive.google.com/file/d/{file_id}/view"

    @staticmethod
    def _drive_download_link(file_id: str) -> str:
        return f"https://drive.google.com/uc?id={file_id}&export=download"

    def _ensure_public_read(self, file_id: str) -> None:
        if not settings.google_drive_make_public:
            return
        service = self._build_drive_service()
        try:
            service.permissions().create(
                fileId=file_id,
                body={"type": "anyone", "role": "reader"},
                fields="id",
            ).execute()
        except Exception:
            pass

    def _download_drive_file(self, file_id: str, target_path: Path) -> None:
        service = self._build_drive_service()
        request = service.files().get_media(fileId=file_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        import io
        with target_path.open("wb") as handle:
            fh = io.FileIO(handle.name, mode='wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _status, done = downloader.next_chunk()

    def _upload_to_drive(self, local_path: Path, parent_id: str, mime_type: str) -> dict[str, str]:
        service = self._build_drive_service()
        media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=False)
        created = service.files().create(
            body={"name": local_path.name, "parents": [parent_id]},
            media_body=media,
            fields="id, webViewLink, webContentLink",
        ).execute()
        file_id = str(created["id"])
        self._ensure_public_read(file_id)
        return {
            "file_id": file_id,
            "view_link": str(created.get("webViewLink") or self._drive_view_link(file_id)),
            "download_link": str(created.get("webContentLink") or self._drive_download_link(file_id)),
        }

    def _resolve_source(
        self,
        *,
        source_path: str | None,
        source_drive_file_id: str | None,
        source_filename: str | None,
    ) -> Path:
        if source_path:
            resolved_source = Path(source_path)
            if not resolved_source.exists():
                raise FileNotFoundError(f"Missing source video: {resolved_source}")
            return resolved_source
        if source_drive_file_id:
            filename = Path(source_filename or f"{source_drive_file_id}.mp4").name
            target_path = self.default_source_dir / filename
            self._download_drive_file(source_drive_file_id, target_path)
            return target_path
        raise ValueError("Provide either source_path or source_drive_file_id")

    @staticmethod
    def _normalize_output_name(source_path: Path, output_basename: str | None) -> str:
        if output_basename:
            suffix = Path(output_basename).suffix.lower()
            if suffix in ALREADY_COMPRESSED_SUFFIXES:
                return output_basename
            return f"{output_basename}{source_path.suffix}"
        return source_path.name

    def _prepare_compressed_video(
        self,
        *,
        source_path: Path,
        target_video_dir: Path,
        output_basename: str | None,
    ) -> Path:
        target_video_dir.mkdir(parents=True, exist_ok=True)
        if source_path.suffix.lower() in ALREADY_COMPRESSED_SUFFIXES:
            output_name = self._normalize_output_name(source_path, output_basename)
            target_path = target_video_dir / output_name
            if source_path.resolve() != target_path.resolve():
                shutil.copy2(source_path, target_path)
            return target_path

        with legacy_workdir():
            from src_vlm import video_ingestion

            output_name = output_basename or f"{source_path.stem}.h265"
            return Path(
                video_ingestion.compress_video(
                    str(source_path),
                    str(target_video_dir),
                    use_h265=True,
                    output_filename=output_name,
                )
            )

    def process_video(
        self,
        *,
        source_path: str | None,
        source_drive_file_id: str | None = None,
        source_filename: str | None = None,
        camera_id: str | None = None,
        recorded_start: datetime | None = None,
        output_video_dir: str | None = None,
        output_metadata_dir: str | None = None,
        output_basename: str | None = None,
        destination_video_folder_id: str | None = None,
        destination_metadata_folder_id: str | None = None,
        upload_outputs_to_drive: bool = False,
        metadata: dict | None = None,
    ) -> dict:
        metadata = _dict_or_empty(metadata)
        profile, hardware_profile, _execution_plan = self._resolve_profile(metadata)
        acceleration_state = configure_torch_runtime(
            allow_tf32=bool(hardware_profile.get("allow_tf32", True)),
            cudnn_benchmark=bool(hardware_profile.get("cudnn_benchmark", True)),
            host_cpu_count=settings.host_cpu_count,
        )

        resolved_source = self._resolve_source(
            source_path=source_path,
            source_drive_file_id=source_drive_file_id,
            source_filename=source_filename,
        )

        target_video_dir = Path(output_video_dir or self.default_video_dir)
        target_metadata_dir = Path(output_metadata_dir or self.default_metadata_dir)
        target_video_dir.mkdir(parents=True, exist_ok=True)
        target_metadata_dir.mkdir(parents=True, exist_ok=True)

        vlm_engine = self._get_vlm_engine()

        with legacy_workdir():
            from src_vlm import hospital_pipeline

            start_time = recorded_start or datetime.fromtimestamp(resolved_source.stat().st_mtime, tz=timezone.utc)
            camera_slug = hospital_pipeline._slug(camera_id or resolved_source.stem)
            compressed_path = self._prepare_compressed_video(
                source_path=resolved_source,
                target_video_dir=target_video_dir,
                output_basename=output_basename or f"{camera_slug}_{start_time.strftime('%Y%m%dT%H%M%SZ')}.h265",
            )
            video_payload, people = hospital_pipeline._build_detected_people(
                compressed_path=compressed_path,
                source_path=resolved_source,
                camera_id=camera_slug,
                recorded_start=start_time,
                vlm_engine=vlm_engine,
                metadata_dir=target_metadata_dir,
            )
            metadata_path = Path(video_payload["metadata_path"])

            video_payload["pipeline_profile"] = profile["profile"]
            video_payload["gpu_hardware_profile"] = hardware_profile
            video_payload["processing_backend"] = "tracking_service_local"
            if metadata.get("source_mode"):
                video_payload["source_mode"] = metadata.get("source_mode")
            for person in people:
                person.setdefault("pipeline_profile", profile["profile"])
                person.setdefault("reid_profile", str(profile["components"].get("reid", {}).get("name") or ""))
                person.setdefault("processing_backend", "tracking_service_local")
                if metadata.get("source_mode"):
                    person.setdefault("source_mode", metadata.get("source_mode"))
            hospital_pipeline._write_video_metadata(metadata_path, video_payload, people)

        processed_at = datetime.now(timezone.utc)
        drive_video = None
        drive_metadata = None
        if upload_outputs_to_drive:
            if not destination_video_folder_id or not destination_metadata_folder_id:
                raise ValueError("destination_video_folder_id and destination_metadata_folder_id are required when upload_outputs_to_drive=true")
            drive_video = self._upload_to_drive(compressed_path, destination_video_folder_id, "video/h265")
            drive_metadata = self._upload_to_drive(metadata_path, destination_metadata_folder_id, "application/json")

        return {
            "status": "completed",
            "processing_backend": "tracking_service_local",
            "pipeline_profile": profile["profile"],
            "gpu_hardware_profile": hardware_profile,
            "acceleration_state": acceleration_state,
            "source_path": str(resolved_source),
            "compressed_path": str(compressed_path),
            "metadata_path": str(metadata_path),
            "drive_video_file_id": (drive_video or {}).get("file_id"),
            "drive_metadata_file_id": (drive_metadata or {}).get("file_id"),
            "drive_video_link": (drive_video or {}).get("view_link"),
            "drive_metadata_link": (drive_metadata or {}).get("view_link"),
            "video": video_payload,
            "people": people,
            "person_count": len(people),
            "processed_at": processed_at,
        }
