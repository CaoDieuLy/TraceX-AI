from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from .config import settings
from .cuda_runtime import configure_torch_runtime
from .execution_plan import resolve_execution_plan
from .local_ingestion_pipeline import LocalVideoIngestionPipeline
from .strict_pipeline import get_strict_pipeline


def _dict_or_empty(value: object) -> dict:
    return value if isinstance(value, dict) else {}


INGESTION_VIDEO_SUFFIXES = {".mp4"}

_HERE = Path(__file__).resolve()
for candidate in [Path(os.getenv("A20_ROOT", "")).expanduser() if os.getenv("A20_ROOT", "").strip() else None, Path("/workspace/a20-root"), *_HERE.parents]:
    if candidate and (candidate / "shared_secret_runtime.py").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from shared_secret_runtime import build_google_drive_oauth_service  # noqa: E402


class VideoIngestionRuntime:
    def __init__(self) -> None:
        self.work_root = Path(settings.ingestion_work_root)
        self.default_video_dir = self.work_root / "videos"
        self.default_metadata_dir = self.work_root / "metadata"
        self.default_source_dir = self.work_root / "sources"
        self.default_video_dir.mkdir(parents=True, exist_ok=True)
        self.default_metadata_dir.mkdir(parents=True, exist_ok=True)
        self.default_source_dir.mkdir(parents=True, exist_ok=True)
        self._drive_service = None

    @staticmethod
    def _apply_execution_environment(execution_plan: dict) -> None:
        parallelism = execution_plan.get("parallelism", {})
        batching = execution_plan.get("batching", {})
        memory = execution_plan.get("memory", {})
        hardware = execution_plan.get("hardware", {})
        host_cpu_count = max(1, int(hardware.get("host_cpu_count") or 1))
        gpu_streams = max(1, int(parallelism.get("gpu_streams") or 1))

        os.environ["OMP_NUM_THREADS"] = str(host_cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(host_cpu_count)
        os.environ["OPENBLAS_NUM_THREADS"] = str(host_cpu_count)
        os.environ["NUMEXPR_NUM_THREADS"] = str(host_cpu_count)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = str(gpu_streams)
        os.environ["MCPT_GPU_STREAMS"] = str(gpu_streams)
        os.environ["MCPT_CPU_DECODE_WORKERS"] = str(parallelism.get("cpu_decode_workers") or host_cpu_count)
        os.environ["MCPT_CPU_CROP_WORKERS"] = str(parallelism.get("cpu_crop_workers") or host_cpu_count)
        os.environ["MCPT_METADATA_WORKERS"] = str(parallelism.get("metadata_workers") or max(host_cpu_count // 2, 1))
        os.environ["MCPT_PREFETCH_QUEUE_SIZE"] = str(parallelism.get("prefetch_queue_size") or 32)
        os.environ["MCPT_PARALLEL_VIDEO_JOBS"] = str(parallelism.get("parallel_video_jobs") or 1)
        os.environ["MCPT_DETECTOR_BATCH_SIZE"] = str(batching.get("detector_batch_size") or 8)
        os.environ["MCPT_REID_BATCH_SIZE"] = str(batching.get("reid_batch_size") or 64)
        os.environ["MCPT_VLM_BATCH_SIZE"] = str(batching.get("vlm_batch_size") or 8)
        os.environ["MCPT_EMBEDDING_BATCH_SIZE"] = str(batching.get("embedding_batch_size") or 64)
        if memory.get("allow_tf32", True):
            os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
        if memory.get("pin_memory", True):
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    def _resolve_pipeline(self, metadata: dict | None = None) -> tuple[dict, dict, dict]:
        metadata = _dict_or_empty(metadata)
        pipeline = get_strict_pipeline()
        detected_hardware, execution_plan = resolve_execution_plan(pipeline_spec=pipeline)
        return pipeline, detected_hardware, execution_plan

    def _build_drive_service(self):
        if not settings.google_drive_enabled:
            raise RuntimeError("Google Drive support is disabled on tracking-service.")
        if self._drive_service is not None:
            return self._drive_service
        self._drive_service = build_google_drive_oauth_service()
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
                supportsAllDrives=True,
            ).execute()
        except Exception:
            pass

    def _download_drive_file(self, file_id: str, target_path: Path) -> None:
        service = self._build_drive_service()
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
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
        media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
        created = service.files().create(
            body={"name": local_path.name, "parents": [parent_id]},
            media_body=media,
            fields="id, webViewLink, webContentLink",
            supportsAllDrives=True,
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
            if suffix in INGESTION_VIDEO_SUFFIXES:
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
        if source_path.suffix.lower() in INGESTION_VIDEO_SUFFIXES:
            output_name = self._normalize_output_name(source_path, output_basename)
            target_path = target_video_dir / output_name
            if source_path.resolve() != target_path.resolve():
                shutil.copy2(source_path, target_path)
            return target_path
        raise ValueError(
            "Only .mp4 inputs are supported in the production ingestion flow. "
            f"Got: {source_path.name}"
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
        pipeline, detected_hardware, execution_plan = self._resolve_pipeline(metadata)
        self._apply_execution_environment(execution_plan)
        configure_torch_runtime(
            allow_tf32=bool(detected_hardware.get("allow_tf32", True)),
            cudnn_benchmark=bool(detected_hardware.get("cudnn_benchmark", True)),
            host_cpu_count=int(execution_plan.get("hardware", {}).get("host_cpu_count") or detected_hardware.get("host_cpu_count") or 1),
        )

        resolved_source_path = self._resolve_source(
            source_path=source_path,
            source_drive_file_id=source_drive_file_id,
            source_filename=source_filename,
        )
        output_video_root = Path(output_video_dir) if output_video_dir else self.default_video_dir
        output_metadata_root = Path(output_metadata_dir) if output_metadata_dir else self.default_metadata_dir
        compressed_path = self._prepare_compressed_video(
            source_path=resolved_source_path,
            target_video_dir=output_video_root,
            output_basename=output_basename,
        )
        metadata_basename = compressed_path.stem if compressed_path.stem else resolved_source_path.stem
        metadata_path = output_metadata_root / f"{metadata_basename}.json"

        sample_fps = int(
            metadata.get("ingestion_contract", {})
            .get("decode_sampling", {})
            .get("sample_fps")
            or 5
        )
        pipeline_runner = LocalVideoIngestionPipeline(sample_fps=sample_fps)
        output = pipeline_runner.run(
            source_path=resolved_source_path,
            compressed_path=compressed_path,
            metadata_path=metadata_path,
            camera_id=camera_id,
            recorded_start=recorded_start,
            metadata=metadata,
        )
        response = output.to_response()
        response["detected_hardware"] = detected_hardware
        response["acceleration_state"] = configure_torch_runtime(
            allow_tf32=bool(detected_hardware.get("allow_tf32", True)),
            cudnn_benchmark=bool(detected_hardware.get("cudnn_benchmark", True)),
            host_cpu_count=int(execution_plan.get("hardware", {}).get("host_cpu_count") or detected_hardware.get("host_cpu_count") or 1),
        )

        if upload_outputs_to_drive and destination_video_folder_id and destination_metadata_folder_id:
            uploaded_video = self._upload_to_drive(compressed_path, destination_video_folder_id, "video/mp4")
            uploaded_metadata = self._upload_to_drive(metadata_path, destination_metadata_folder_id, "application/json")
            response["drive_video_file_id"] = uploaded_video["file_id"]
            response["drive_metadata_file_id"] = uploaded_metadata["file_id"]
            response["drive_video_link"] = uploaded_video["view_link"]
            response["drive_metadata_link"] = uploaded_metadata["view_link"]

        return response
