from __future__ import annotations

from html.parser import HTMLParser
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse

from .config import settings
from .cuda_runtime import configure_torch_runtime
from .execution_plan import resolve_execution_plan
from .local_ingestion_pipeline import LocalVideoIngestionPipeline
from .strict_pipeline import get_strict_pipeline


LOGGER = logging.getLogger(__name__)


def _dict_or_empty(value: object) -> dict:
    return value if isinstance(value, dict) else {}


INGESTION_VIDEO_SUFFIXES = {".mp4"}


class _GoogleDriveDownloadFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.action: str | None = None
        self.inputs: dict[str, str] = {}
        self._in_download_form = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "form" and values.get("id") == "download-form":
            self._in_download_form = True
            self.action = values.get("action") or None
            return
        if tag == "input" and self._in_download_form:
            name = values.get("name")
            if name:
                self.inputs[name] = values.get("value", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._in_download_form:
            self._in_download_form = False


class VideoIngestionRuntime:
    def __init__(self) -> None:
        self.work_root = Path(settings.ingestion_work_root)
        self.default_video_dir = self.work_root / "videos"
        self.default_metadata_dir = self.work_root / "metadata"
        self.default_source_dir = self.work_root / "sources"
        self.default_video_dir.mkdir(parents=True, exist_ok=True)
        self.default_metadata_dir.mkdir(parents=True, exist_ok=True)
        self.default_source_dir.mkdir(parents=True, exist_ok=True)

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

    @staticmethod
    def _download_public_url(url: str, target_path: Path) -> None:
        """Download from a public HTTP URL with Google Drive large-file redirect handling."""
        if not str(url or "").strip():
            raise ValueError("source_url is required for ingestion download")

        import requests as _req
        session = _req.Session()
        resp = session.get(url, stream=True, timeout=300)
        LOGGER.info(
            "Ingestion download started url_host=%s status=%s content_type=%s filename=%s",
            urlparse(str(resp.url)).netloc,
            resp.status_code,
            resp.headers.get("content-type"),
            target_path.name,
        )
        # Google Drive shows a virus-scan warning for files > 25 MB
        for key, value in resp.cookies.items():
            if key.startswith("download_warning"):
                separator = "&" if "?" in url else "?"
                resp = session.get(f"{url}{separator}confirm={value}", stream=True, timeout=300)
                LOGGER.info(
                    "Followed Google Drive download_warning cookie for %s; status=%s content_type=%s",
                    target_path.name,
                    resp.status_code,
                    resp.headers.get("content-type"),
                )
                break
        if "text/html" in str(resp.headers.get("content-type") or "").lower():
            parser = _GoogleDriveDownloadFormParser()
            parser.feed(resp.text)
            if parser.action and parser.inputs:
                download_url = f"{urljoin(resp.url, parser.action)}?{urlencode(parser.inputs)}"
                resp.close()
                resp = session.get(download_url, stream=True, timeout=300)
                LOGGER.info(
                    "Followed Google Drive download form for %s; status=%s content_type=%s",
                    target_path.name,
                    resp.status_code,
                    resp.headers.get("content-type"),
                )
            else:
                preview = str(resp.text or "")[:500].replace("\n", " ")
                raise RuntimeError(
                    "Google Drive returned an HTML download page without a usable download form. "
                    f"Preview: {preview}"
                )
        resp.raise_for_status()
        content_type = str(resp.headers.get("content-type") or "").lower()
        if "text/html" in content_type:
            preview = str(resp.text or "")[:500].replace("\n", " ")
            raise RuntimeError(
                "Google Drive returned HTML instead of video content after confirmation. "
                f"Content-Type: {content_type}. Preview: {preview}"
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        bytes_written = 0
        with target_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 17):  # 128 KB
                if chunk:
                    fh.write(chunk)
                    bytes_written += len(chunk)
        if bytes_written <= 0:
            raise RuntimeError(f"Downloaded empty source video: {target_path.name}")
        LOGGER.info("Ingestion download completed filename=%s bytes=%s", target_path.name, bytes_written)

    def _resolve_source(
        self,
        *,
        source_url: str,
        source_filename: str | None,
    ) -> Path:
        filename = Path(source_filename or "video.mp4").name
        target_path = self.default_source_dir / filename
        self._download_public_url(source_url, target_path)
        return target_path

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
            if (
                settings.ingestion_reuse_downloaded_mp4
                and source_path.parent == self.default_source_dir
                and target_video_dir == self.default_video_dir
                and target_path.name == source_path.name
            ):
                return source_path
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
        source_url: str,
        source_filename: str | None = None,
        camera_id: str | None = None,
        recorded_start: datetime | None = None,
        output_video_dir: str | None = None,
        output_metadata_dir: str | None = None,
        output_basename: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        metadata = _dict_or_empty(metadata)
        LOGGER.info(
            "Video ingestion started source_filename=%s camera_id=%s output_basename=%s",
            source_filename,
            camera_id,
            output_basename,
        )
        pipeline, detected_hardware, execution_plan = self._resolve_pipeline(metadata)
        self._apply_execution_environment(execution_plan)
        configure_torch_runtime(
            allow_tf32=bool(detected_hardware.get("allow_tf32", True)),
            cudnn_benchmark=bool(detected_hardware.get("cudnn_benchmark", True)),
            host_cpu_count=int(execution_plan.get("hardware", {}).get("host_cpu_count") or detected_hardware.get("host_cpu_count") or 1),
        )

        resolved_source_path = self._resolve_source(
            source_url=source_url,
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
            or settings.ingestion_default_sample_fps
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

        LOGGER.info(
            "Video ingestion completed source_filename=%s camera_id=%s people=%s metadata=%s",
            source_filename,
            camera_id,
            response.get("person_count"),
            response.get("metadata_path"),
        )
        return response
