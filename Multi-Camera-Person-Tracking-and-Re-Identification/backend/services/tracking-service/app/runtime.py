from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import settings
from .cuda_runtime import configure_torch_runtime
from .execution_plan import resolve_execution_plan
from .pipeline_profiles import resolve_pipeline_profile


def _dict_or_empty(value: object) -> dict:
    return value if isinstance(value, dict) else {}


class AccuracyFirstTrackerRuntime:
    def __init__(self) -> None:
        self.profile = resolve_pipeline_profile(settings.pipeline_profile)

    def _resolve_profile(self, candidate_info: dict) -> dict:
        candidate = dict(candidate_info or {})
        selected_profile = str(candidate.get("pipeline_profile") or settings.pipeline_profile).strip() or settings.pipeline_profile
        overrides = _dict_or_empty(candidate.get("hyperparameter_overrides"))
        return resolve_pipeline_profile(selected_profile, overrides)

    def build_manifest(
        self,
        candidate_info: dict,
        source_video: str | None,
        output_video_path: Path,
        profile: dict,
        hardware_profile: dict,
        execution_plan: dict,
    ) -> dict:
        candidate = dict(candidate_info or {})
        frame_idx = int(candidate.get("frame_idx") or 0)
        bbox = candidate.get("bbox") or candidate.get("representative_bbox") or []
        return {
            "schema_version": "accuracy_first_tracking_manifest_v1",
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "runtime_mode": settings.tracking_runtime_mode,
            "pipeline_profile": profile["profile"],
            "pipeline_summary": profile["summary"],
            "selected_components": profile["components"],
            "runtime_defaults": profile.get("runtime_defaults", {}),
            "hyperparameters": profile.get("hyperparameters", {}),
            "gpu_hardware_profile": hardware_profile,
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
            "execution_plan": {
                "detector": "Use RF-DETR profile when model artifacts are available. Avoid NMS post-processing in accuracy-first mode.",
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
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            cv2.putText(
                frame,
                f"Accuracy-first tracking | Frame {frame_idx}",
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

    def _write_demo_clip(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = 960, 540
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height))
        for frame_idx in range(150):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Accuracy-first runtime manifest only",
                (60, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Frame {frame_idx}",
                (60, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (40, 220, 40),
                2,
            )
            x = 80 + ((frame_idx * 7) % 700)
            cv2.rectangle(frame, (x, 220), (x + 110, 420), (0, 255, 0), 3)
            writer.write(frame)
        writer.release()

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
        profile = self._resolve_profile(candidate)
        hardware_profile, execution_plan = resolve_execution_plan(
            pipeline_profile=profile,
            gpu_profile_name=str(candidate.get("gpu_hardware_profile") or settings.gpu_hardware_profile),
            gpu_profile_overrides=_dict_or_empty(candidate.get("gpu_hardware_overrides")),
            gpu_count=settings.gpu_count,
            host_cpu_count=settings.host_cpu_count,
            host_ram_gb=settings.host_ram_gb,
        )
        acceleration_state = configure_torch_runtime(
            allow_tf32=bool(hardware_profile.get("allow_tf32", True)),
            cudnn_benchmark=bool(hardware_profile.get("cudnn_benchmark", True)),
            host_cpu_count=settings.host_cpu_count,
        )
        source_video = self._resolve_source_video(candidate, video_database_path)
        video_id = str(candidate.get("video_id") or "candidate").strip() or "candidate"
        output_path = Path(output_dir) / f"accuracy_first_{Path(video_id).stem}.mp4"

        if source_video is not None:
            clip_created = self._extract_clip(source_video, int(candidate.get("frame_idx") or 0), output_path)
        else:
            clip_created = False

        if not clip_created:
            self._write_demo_clip(output_path)

        manifest = self.build_manifest(
            candidate,
            str(source_video) if source_video else None,
            output_path,
            profile,
            hardware_profile,
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
            "pipeline_profile": profile["profile"],
            "gpu_hardware_profile": hardware_profile.get("gpu_model"),
            "runtime_mode": settings.tracking_runtime_mode,
            "acceleration_state": acceleration_state,
        }
