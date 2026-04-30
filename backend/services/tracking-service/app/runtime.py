from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
os.environ.setdefault("OPENCV_VIDEOCAPTURE_DEBUG", "0")
os.environ.setdefault("OPENCV_FFMPEG_DEBUG", "0")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "16")

import cv2

try:
    cv2.setLogLevel(getattr(cv2, "LOG_LEVEL_ERROR", 2))
except Exception:
    pass

from .config import settings
from .cuda_runtime import configure_torch_runtime
from .execution_plan import resolve_execution_plan
from .strict_pipeline import get_strict_pipeline


class StrictTrackerRuntime:
    def __init__(self) -> None:
        self.pipeline = get_strict_pipeline()

    def _resolve_pipeline(self, candidate_info: dict) -> dict:
        return get_strict_pipeline()

    def build_manifest(
        self,
        candidate_info: dict,
        source_video: str | None,
        output_video_path: Path,
        pipeline: dict,
        detected_hardware: dict,
        execution_plan: dict,
    ) -> dict:
        candidate = dict(candidate_info or {})
        frame_idx = int(candidate.get("frame_idx") or 0)
        bbox = candidate.get("bbox") or candidate.get("representative_bbox") or []
        return {
            "schema_version": "strict_tracking_manifest_v1",
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "runtime_mode": settings.tracking_runtime_mode,
            "pipeline_summary": pipeline["summary"],
            "selected_components": pipeline["components"],
            "runtime_defaults": pipeline.get("runtime_defaults", {}),
            "hyperparameters": pipeline.get("hyperparameters", {}),
            "detected_hardware": detected_hardware,
            "execution_plan": execution_plan,
            "candidate": {
                "candidate_id": candidate.get("candidate_id"),
                "camera_id": candidate.get("camera_id"),
                "video_id": candidate.get("video_id"),
                "track_id": candidate.get("track_id"),
                "human_key": candidate.get("human_key"),
                "frame_idx": frame_idx,
                "bbox": bbox,
                "world_position": candidate.get("world_position"),
                "top_point_projection": candidate.get("top_point_projection"),
            },
            "stage_instructions": {
                "detector": "Use RF-DETR when model artifacts are available. Avoid NMS post-processing in accuracy-first mode.",
                "tracker": "Apply geometry-aware association, then corrective cascade to revisit ambiguous links.",
                "reid": "Use SOLIDER global embeddings and KPR part embeddings when keypoints are available.",
                "semantic_search": "Index fine-grained attributes and rerank with multi-signal ranking ensemble.",
                "evaluation": "Validate with HOTA, DetA, and AssA via TrackEval before promoting changes.",
            },
            "artifacts": {
                "source_video": source_video,
                "output_video": str(output_video_path),
                "output_manifest": str(output_video_path.with_suffix(output_video_path.suffix + ".json")),
            },
        }

    def _extract_clip(self, video_path: Path, center_frame: int, output_path: Path, clip_duration: int = 10) -> bool:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            return False

        half_clip_frames = int(fps * clip_duration / 2)
        start_frame = max(0, int(center_frame) - half_clip_frames)
        end_frame = min(max(total_frames - 1, 0), int(center_frame) + half_clip_frames)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            cv2.putText(
                frame,
                f"Strict tracking | Frame {frame_idx}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (40, 220, 40),
                2,
            )
            if frame_idx == int(center_frame):
                cv2.putText(
                    frame,
                    "Target frame",
                    (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            writer.write(frame)

        cap.release()
        writer.release()
        return output_path.exists()

    def _resolve_source_video(self, candidate_info: dict, video_database_path: str) -> Path | None:
        candidate = dict(candidate_info or {})
        explicit_path = str(candidate.get("storage_path") or candidate.get("source_path") or "").strip()
        if explicit_path:
            path = Path(explicit_path)
            if path.exists():
                return path

        video_id = str(candidate.get("video_id") or "").strip()
        if not video_id:
            return None

        local_candidate = Path(video_database_path) / video_id
        if local_candidate.exists():
            return local_candidate

        compressed_candidate = Path("data/videos/compressed") / video_id
        if compressed_candidate.exists():
            return compressed_candidate

        return None

    def run(self, candidate_info: dict, video_database_path: str, output_dir: str) -> dict:
        candidate = dict(candidate_info or {})
        pipeline = self._resolve_pipeline(candidate)
        detected_hardware, execution_plan = resolve_execution_plan(pipeline_spec=pipeline)
        acceleration_state = configure_torch_runtime(
            allow_tf32=bool(detected_hardware.get("allow_tf32", True)),
            cudnn_benchmark=bool(detected_hardware.get("cudnn_benchmark", True)),
            host_cpu_count=int(detected_hardware.get("host_cpu_count") or 1),
        )
        source_video = self._resolve_source_video(candidate, video_database_path)
        video_id = str(candidate.get("video_id") or "candidate").strip() or "candidate"
        output_path = Path(output_dir) / f"strict_{Path(video_id).stem}.avi"

        if source_video is None:
            raise FileNotFoundError("Could not resolve source video for tracking output.")
        clip_created = self._extract_clip(source_video, int(candidate.get("frame_idx") or 0), output_path)
        if not clip_created:
            raise RuntimeError(f"Failed to create tracking clip from source video: {source_video}")

        manifest = self.build_manifest(
            candidate,
            str(source_video) if source_video else None,
            output_path,
            pipeline,
            detected_hardware,
            execution_plan,
        )
        manifest["acceleration_state"] = acceleration_state
        manifest_path = output_path.with_suffix(output_path.suffix + ".json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "output_path": str(output_path.resolve()),
            "relative_output_path": str(output_path),
            "manifest_path": str(manifest_path.resolve()),
            "relative_manifest_path": str(manifest_path),
            "exists": output_path.exists(),
            "detected_hardware": detected_hardware,
            "runtime_mode": settings.tracking_runtime_mode,
            "acceleration_state": acceleration_state,
        }
