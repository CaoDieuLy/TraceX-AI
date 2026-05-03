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


def _parse_recorded_start_from_filename(filename: str) -> "datetime | None":
    """
    Parse UTC datetime from MTMC-style filename.

    Input  : 'cam_01_2026-04-28_11-00.mp4'
    Output : datetime(2026, 4, 28, 11, 0, tzinfo=utc) or None
    """
    import re
    from datetime import timezone
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})", Path(filename).stem)
    if not m:
        return None
    try:
        return datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)), 0,
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None


def _extract_drive_file_id(url: str) -> str:
    """
    Extract Google Drive file ID from view or download URL.

    Input  : 'https://drive.google.com/file/d/{ID}/view' or '?id={ID}'
    Output : file ID string, or '' if not found
    """
    import re
    m = re.search(r"/d/([a-zA-Z0-9_-]{25,})", url) or re.search(r"[?&]id=([a-zA-Z0-9_-]{25,})", url)
    return m.group(1) if m else ""


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
        precision = execution_plan.get("precision", {})
        memory = execution_plan.get("memory", {})
        hardware = execution_plan.get("hardware", {})
        host_cpu_count = max(1, int(hardware.get("host_cpu_count") or 1))
        gpu_streams = max(1, int(parallelism.get("gpu_streams") or 1))
        detector_batch_size = int(batching.get("detector_batch_size") or 8)

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
        os.environ["MCPT_DETECTOR_BATCH_SIZE"] = str(detector_batch_size)
        os.environ["MCPT_REID_BATCH_SIZE"] = str(batching.get("reid_batch_size") or 64)
        os.environ["MCPT_VLM_BATCH_SIZE"] = str(batching.get("vlm_batch_size") or 8)
        os.environ["MCPT_EMBEDDING_BATCH_SIZE"] = str(batching.get("embedding_batch_size") or 64)
        os.environ["MCPT_DETECTOR_PRECISION"] = str(precision.get("detector") or "fp32")
        os.environ["MCPT_REID_PRECISION"] = str(precision.get("reid") or "fp32")
        os.environ["MCPT_VLM_PRECISION"] = str(precision.get("vlm") or "fp32")
        os.environ["MCPT_EMBEDDING_PRECISION"] = str(precision.get("embedding") or precision.get("reid") or "fp32")
        os.environ["MCPT_STREAM_BATCH_SIZE"] = str(max(detector_batch_size * 3, 150))
        os.environ.setdefault("MCPT_TRACKLET_STAGE_WORKERS", "1")
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
    def _resolve_drive_response(session: "requests.Session", url: str) -> "requests.Response":
        """Follow Google Drive redirects/confirmation forms and return a streaming video response."""
        import requests as _req
        resp = session.get(url, stream=True, timeout=60)
        # Large file virus-scan cookie
        for key, value in resp.cookies.items():
            if key.startswith("download_warning"):
                separator = "&" if "?" in url else "?"
                resp = session.get(f"{url}{separator}confirm={value}", stream=True, timeout=60)
                break
        # HTML confirmation form fallback
        if "text/html" in str(resp.headers.get("content-type") or "").lower():
            parser = _GoogleDriveDownloadFormParser()
            parser.feed(resp.text)
            if parser.action and parser.inputs:
                download_url = f"{urljoin(resp.url, parser.action)}?{urlencode(parser.inputs)}"
                resp.close()
                resp = session.get(download_url, stream=True, timeout=60)
            else:
                preview = str(resp.text or "")[:300].replace("\n", " ")
                raise RuntimeError(f"Drive returned HTML without download form. Preview: {preview}")
        resp.raise_for_status()
        if "text/html" in str(resp.headers.get("content-type") or "").lower():
            preview = str(resp.text or "")[:300].replace("\n", " ")
            raise RuntimeError(f"Drive returned HTML instead of video. Preview: {preview}")
        return resp

    def _stream_url_to_pipe(self, url: str, source_filename: str) -> str:
        """Linux only: stream URL into a named FIFO pipe so cv2 reads concurrently (no full-file download).
        Returns the pipe path as a string for cv2.VideoCapture()."""
        import os, threading
        import requests as _req

        stem = Path(source_filename).stem
        pipe_path = self.default_source_dir / f"{stem}.fifo"
        pipe_path.parent.mkdir(parents=True, exist_ok=True)
        pipe_str = str(pipe_path)
        if pipe_path.exists():
            pipe_path.unlink()
        os.mkfifo(pipe_str)

        def _writer() -> None:
            # Open write end FIRST so cv2 reader never blocks forever on EOF.
            # If download fails, fh is closed immediately → cv2 gets empty stream.
            try:
                fh = open(pipe_str, "wb")
            except Exception as exc:
                LOGGER.error("Pipe open failed for %s: %s", source_filename, exc)
                return
            try:
                session = _req.Session()
                resp = self._resolve_drive_response(session, url)
                bytes_sent = 0
                for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB
                    if chunk:
                        fh.write(chunk)
                        bytes_sent += len(chunk)
                LOGGER.info("Pipe stream complete filename=%s bytes=%s", source_filename, bytes_sent)
            except Exception as exc:
                LOGGER.error("Pipe writer failed for %s: %s", source_filename, exc)
            finally:
                fh.close()
                try:
                    pipe_path.unlink(missing_ok=True)
                except Exception:
                    pass

        threading.Thread(target=_writer, daemon=True, name=f"pipe-{stem}").start()
        return pipe_str

    @staticmethod
    def _download_with_retry(url: str, target_path: Path, *, max_attempts: int = 3) -> None:
        """Download from a public HTTP URL with retry + exponential backoff."""
        import requests as _req
        import time as _time

        if not str(url or "").strip():
            raise ValueError("source_url is required for ingestion download")

        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                session = _req.Session()
                resp = VideoIngestionRuntime._resolve_drive_response(session, url)
                LOGGER.info(
                    "Download started attempt=%s/%s url_host=%s filename=%s",
                    attempt, max_attempts,
                    urlparse(str(resp.url)).netloc,
                    target_path.name,
                )
                target_path.parent.mkdir(parents=True, exist_ok=True)
                bytes_written = 0
                with target_path.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 17):  # 128 KB
                        if chunk:
                            fh.write(chunk)
                            bytes_written += len(chunk)
                if bytes_written <= 0:
                    raise RuntimeError(f"Downloaded empty file: {target_path.name}")
                LOGGER.info("Download complete filename=%s bytes=%s", target_path.name, bytes_written)
                return
            except Exception as exc:
                last_exc = exc
                LOGGER.warning("Download attempt %s/%s failed for %s: %s", attempt, max_attempts, target_path.name, exc)
                if attempt < max_attempts:
                    _time.sleep(2 ** attempt)  # 2s, 4s, 8s
                    if target_path.exists():
                        target_path.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed after {max_attempts} attempts for {target_path.name}") from last_exc

    def _resolve_source(
        self,
        *,
        source_url: str,
        source_filename: str | None,
    ) -> Path:
        source_url = str(source_url or "").strip()
        # Local file path — return directly
        local_source = Path(source_url).expanduser()
        if local_source.exists():
            return local_source
        # HTTP URL: download to temp file.
        # cv2.VideoCapture requires a seekable file; MP4 moov atoms are often at
        # the end of the container, so named-pipe / stream-only approaches fail.
        filename = Path(source_filename or "video.mp4").name
        target_path = self.default_source_dir / filename
        self._download_with_retry(source_url, target_path)
        return target_path

    def _cleanup_source(self, resolved_source: Path | str) -> None:
        """Delete temp download / FIFO pipe after processing to free disk."""
        try:
            p = Path(str(resolved_source))
            if p.exists() and p.parent == self.default_source_dir:
                p.unlink(missing_ok=True)
                LOGGER.info("Cleaned up temp source: %s", p.name)
        except Exception as exc:
            LOGGER.warning("Cleanup failed for %s: %s", resolved_source, exc)

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

        # Parse recorded_start from filename if caller didn't provide it.
        # Filename format: cam_01_2026-04-28_11-00.mp4 → 2026-04-28T11:00:00Z
        if recorded_start is None and source_filename:
            recorded_start = _parse_recorded_start_from_filename(source_filename)
            if recorded_start:
                LOGGER.info("recorded_start parsed from filename: %s → %s", source_filename, recorded_start.isoformat())

        # Extract Drive file ID for clip_drive_urls annotation in the pipeline.
        # Priority: explicit metadata field → derived from source_url
        drive_video_file_id: str = (
            str(metadata.get("drive_video_file_id") or "").strip()
            or _extract_drive_file_id(str(source_url or ""))
        )

        resolved_source_path = self._resolve_source(
            source_url=source_url,
            source_filename=source_filename,
        )
        output_video_root = Path(output_video_dir) if output_video_dir else self.default_video_dir
        output_metadata_root = Path(output_metadata_dir) if output_metadata_dir else self.default_metadata_dir
        if isinstance(resolved_source_path, Path):
            compressed_path = self._prepare_compressed_video(
                source_path=resolved_source_path,
                target_video_dir=output_video_root,
                output_basename=output_basename,
            )
            metadata_basename = compressed_path.stem if compressed_path.stem else resolved_source_path.stem
        else:
            output_video_root.mkdir(parents=True, exist_ok=True)
            normalized_name = self._normalize_output_name(Path(source_filename or "stream.mp4"), output_basename)
            compressed_path = output_video_root / normalized_name
            metadata_basename = compressed_path.stem or "stream"
        metadata_path = output_metadata_root / f"{metadata_basename}.json"

        sample_fps = int(
            metadata.get("ingestion_contract", {})
            .get("decode_sampling", {})
            .get("sample_fps")
            or settings.ingestion_default_sample_fps
        )
        pipeline_runner = LocalVideoIngestionPipeline(sample_fps=sample_fps)
        try:
            output = pipeline_runner.run(
                source_path=resolved_source_path,
                compressed_path=compressed_path,
                metadata_path=metadata_path,
                camera_id=camera_id,
                recorded_start=recorded_start,
                metadata=metadata,
                drive_video_file_id=drive_video_file_id,
            )
        finally:
            # Always clean up FIFO pipe or temp download to prevent disk full
            self._cleanup_source(resolved_source_path)

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
