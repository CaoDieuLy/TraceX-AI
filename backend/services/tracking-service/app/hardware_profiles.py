from __future__ import annotations

import multiprocessing
import subprocess
from copy import deepcopy


HARDWARE_PROFILES: dict[str, dict] = {
    "cpu_only": {
        "gpu_model": "CPU-only host",
        "service_processes": 1,
        "gpu_streams": 0,
        "cpu_decode_workers": 4,
        "cpu_crop_workers": 3,
        "metadata_workers": 2,
        "prefetch_queue_size": 16,
        "detector_batch_size": 4,
        "reid_batch_size": 32,
        "vlm_batch_size": 4,
        "embedding_batch_size": 32,
        "detector_precision": "fp32",
        "reid_precision": "fp32",
        "vlm_precision": "fp32",
        "pin_memory": False,
        "non_blocking_transfers": False,
        "channels_last": False,
        "allow_tf32": False,
        "cudnn_benchmark": False,
        "ffmpeg_hevc_preset": "medium",
        "exchange_max_workers": 1,
        "torch_thread_cap": 16,
        "execution_notes": [
            "CPU execution profile only. Keep batches small and prioritize deterministic progress over throughput.",
        ],
    },
    "generic_gpu": {
        "gpu_model": "Generic NVIDIA GPU",
        "service_processes": 1,
        "gpu_streams": 2,
        "cpu_decode_workers": 6,
        "cpu_crop_workers": 4,
        "metadata_workers": 3,
        "prefetch_queue_size": 32,
        "detector_batch_size": 8,
        "reid_batch_size": 96,
        "vlm_batch_size": 8,
        "embedding_batch_size": 64,
        "detector_precision": "fp16",
        "reid_precision": "fp16",
        "vlm_precision": "fp16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "ffmpeg_hevc_preset": "p3",
        "exchange_max_workers": 2,
        "torch_thread_cap": 24,
        "execution_notes": [
            "Generic GPU profile when the exact accelerator is unknown.",
        ],
    },
    "t4": {
        "gpu_model": "NVIDIA T4",
        "service_processes": 1,
        "gpu_streams": 2,
        "cpu_decode_workers": 6,
        "cpu_crop_workers": 4,
        "metadata_workers": 3,
        "prefetch_queue_size": 48,
        "detector_batch_size": 12,
        "reid_batch_size": 128,
        "vlm_batch_size": 12,
        "embedding_batch_size": 96,
        "detector_precision": "fp16",
        "reid_precision": "fp16",
        "vlm_precision": "fp16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": False,
        "cudnn_benchmark": True,
        "ffmpeg_hevc_preset": "p2",
        "exchange_max_workers": 2,
        "torch_thread_cap": 24,
        "execution_notes": [
            "Favor smaller batches to protect latency and memory headroom.",
        ],
    },
    "l4": {
        "gpu_model": "NVIDIA L4",
        "service_processes": 1,
        "gpu_streams": 4,
        "cpu_decode_workers": 10,
        "cpu_crop_workers": 8,
        "metadata_workers": 6,
        "prefetch_queue_size": 96,
        "detector_batch_size": 24,
        "reid_batch_size": 320,
        "vlm_batch_size": 24,
        "embedding_batch_size": 160,
        "detector_precision": "fp16",
        "reid_precision": "fp16",
        "vlm_precision": "bf16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "ffmpeg_hevc_preset": "p1",
        "exchange_max_workers": 6,
        "torch_thread_cap": 32,
        "execution_notes": [
            "Use one GPU-serving process per physical GPU and saturate the card with multiple CUDA streams.",
            "Pipeline decode, crops, ReID, and VLM aggressively because the L4 has good video + tensor balance.",
        ],
    },
    "a10": {
        "gpu_model": "NVIDIA A10",
        "service_processes": 1,
        "gpu_streams": 3,
        "cpu_decode_workers": 8,
        "cpu_crop_workers": 6,
        "metadata_workers": 4,
        "prefetch_queue_size": 72,
        "detector_batch_size": 18,
        "reid_batch_size": 192,
        "vlm_batch_size": 16,
        "embedding_batch_size": 128,
        "detector_precision": "fp16",
        "reid_precision": "fp16",
        "vlm_precision": "fp16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "ffmpeg_hevc_preset": "p1",
        "exchange_max_workers": 3,
        "torch_thread_cap": 32,
        "execution_notes": [
            "A10 can sustain larger decode and ReID queues than T4 while keeping latency predictable.",
        ],
    },
    "a30": {
        "gpu_model": "NVIDIA A30",
        "service_processes": 1,
        "gpu_streams": 4,
        "cpu_decode_workers": 10,
        "cpu_crop_workers": 8,
        "metadata_workers": 5,
        "prefetch_queue_size": 96,
        "detector_batch_size": 28,
        "reid_batch_size": 320,
        "vlm_batch_size": 24,
        "embedding_batch_size": 160,
        "detector_precision": "bf16",
        "reid_precision": "bf16",
        "vlm_precision": "bf16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "ffmpeg_hevc_preset": "p1",
        "exchange_max_workers": 4,
        "torch_thread_cap": 32,
        "execution_notes": [
            "A30 is comfortable with larger fp16/bf16 batches and deeper prefetch queues.",
        ],
    },
    "a40": {
        "gpu_model": "NVIDIA A40",
        "service_processes": 1,
        "gpu_streams": 4,
        "cpu_decode_workers": 10,
        "cpu_crop_workers": 8,
        "metadata_workers": 5,
        "prefetch_queue_size": 96,
        "detector_batch_size": 28,
        "reid_batch_size": 320,
        "vlm_batch_size": 24,
        "embedding_batch_size": 160,
        "detector_precision": "fp16",
        "reid_precision": "fp16",
        "vlm_precision": "bf16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "ffmpeg_hevc_preset": "p1",
        "exchange_max_workers": 4,
        "torch_thread_cap": 32,
        "execution_notes": [
            "A40 profile mirrors A30-class throughput with slightly more VRAM headroom for queues.",
        ],
    },
    "l40": {
        "gpu_model": "NVIDIA L40",
        "service_processes": 1,
        "gpu_streams": 5,
        "cpu_decode_workers": 12,
        "cpu_crop_workers": 9,
        "metadata_workers": 6,
        "prefetch_queue_size": 128,
        "detector_batch_size": 36,
        "reid_batch_size": 448,
        "vlm_batch_size": 32,
        "embedding_batch_size": 224,
        "detector_precision": "bf16",
        "reid_precision": "bf16",
        "vlm_precision": "bf16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "ffmpeg_hevc_preset": "p1",
        "exchange_max_workers": 5,
        "torch_thread_cap": 40,
        "execution_notes": [
            "L40 can keep detector, ReID, and VLM separated on more streams without starving memory.",
        ],
    },
    "l40s": {
        "gpu_model": "NVIDIA L40S",
        "service_processes": 1,
        "gpu_streams": 6,
        "cpu_decode_workers": 12,
        "cpu_crop_workers": 10,
        "metadata_workers": 6,
        "prefetch_queue_size": 160,
        "detector_batch_size": 40,
        "reid_batch_size": 512,
        "vlm_batch_size": 40,
        "embedding_batch_size": 256,
        "detector_precision": "bf16",
        "reid_precision": "bf16",
        "vlm_precision": "bf16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "ffmpeg_hevc_preset": "p1",
        "exchange_max_workers": 6,
        "torch_thread_cap": 48,
        "execution_notes": [
            "L40S can sustain deep queues and larger batches while preserving latency headroom.",
        ],
    },
    "a6000": {
        "gpu_model": "NVIDIA RTX A6000",
        "service_processes": 1,
        "gpu_streams": 4,
        "cpu_decode_workers": 10,
        "cpu_crop_workers": 8,
        "metadata_workers": 5,
        "prefetch_queue_size": 96,
        "detector_batch_size": 28,
        "reid_batch_size": 320,
        "vlm_batch_size": 24,
        "embedding_batch_size": 160,
        "detector_precision": "fp16",
        "reid_precision": "fp16",
        "vlm_precision": "fp16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "ffmpeg_hevc_preset": "p1",
        "exchange_max_workers": 4,
        "torch_thread_cap": 32,
        "execution_notes": [
            "RTX A6000 is close to the A40 class for this workload.",
        ],
    },
    "rtx4090": {
        "gpu_model": "NVIDIA RTX 4090",
        "service_processes": 1,
        "gpu_streams": 5,
        "cpu_decode_workers": 12,
        "cpu_crop_workers": 9,
        "metadata_workers": 6,
        "prefetch_queue_size": 128,
        "detector_batch_size": 36,
        "reid_batch_size": 384,
        "vlm_batch_size": 32,
        "embedding_batch_size": 192,
        "detector_precision": "fp16",
        "reid_precision": "fp16",
        "vlm_precision": "fp16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "ffmpeg_hevc_preset": "p1",
        "exchange_max_workers": 5,
        "torch_thread_cap": 40,
        "execution_notes": [
            "Consumer Ada cards favor aggressive stream concurrency and fast NVENC presets.",
        ],
    },
    "v100": {
        "gpu_model": "NVIDIA V100",
        "service_processes": 1,
        "gpu_streams": 3,
        "cpu_decode_workers": 8,
        "cpu_crop_workers": 6,
        "metadata_workers": 4,
        "prefetch_queue_size": 72,
        "detector_batch_size": 20,
        "reid_batch_size": 224,
        "vlm_batch_size": 16,
        "embedding_batch_size": 128,
        "detector_precision": "fp16",
        "reid_precision": "fp16",
        "vlm_precision": "fp16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": False,
        "cudnn_benchmark": True,
        "ffmpeg_hevc_preset": "p2",
        "exchange_max_workers": 3,
        "torch_thread_cap": 24,
        "execution_notes": [
            "V100 remains strong for fp16 inference but lacks TF32 and newer video-flow niceties.",
        ],
    },
    "a100": {
        "gpu_model": "NVIDIA A100",
        "service_processes": 1,
        "gpu_streams": 6,
        "cpu_decode_workers": 14,
        "cpu_crop_workers": 10,
        "metadata_workers": 6,
        "prefetch_queue_size": 160,
        "detector_batch_size": 48,
        "reid_batch_size": 640,
        "vlm_batch_size": 40,
        "embedding_batch_size": 256,
        "detector_precision": "bf16",
        "reid_precision": "bf16",
        "vlm_precision": "bf16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "ffmpeg_hevc_preset": "p1",
        "exchange_max_workers": 6,
        "torch_thread_cap": 48,
        "execution_notes": [
            "A100 can sustain larger batches and deeper prefetch without compromising memory headroom.",
        ],
    },
    "h100": {
        "gpu_model": "NVIDIA H100",
        "service_processes": 1,
        "gpu_streams": 8,
        "cpu_decode_workers": 16,
        "cpu_crop_workers": 12,
        "metadata_workers": 8,
        "prefetch_queue_size": 192,
        "detector_batch_size": 64,
        "reid_batch_size": 768,
        "vlm_batch_size": 56,
        "embedding_batch_size": 320,
        "detector_precision": "bf16",
        "reid_precision": "bf16",
        "vlm_precision": "bf16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "ffmpeg_hevc_preset": "p1",
        "exchange_max_workers": 8,
        "torch_thread_cap": 64,
        "execution_notes": [
            "H100 should keep the detector, ReID, and VLM fully overlapped before increasing process count.",
        ],
    },
}

GPU_NAME_ALIASES: tuple[tuple[str, str], ...] = (
    ("nvidia l4", "l4"),
    ("tesla t4", "t4"),
    ("nvidia t4", "t4"),
    ("nvidia a10", "a10"),
    ("nvidia a30", "a30"),
    ("nvidia a40", "a40"),
    ("rtx a6000", "a6000"),
    ("nvidia a6000", "a6000"),
    ("rtx 4090", "rtx4090"),
    ("nvidia geforce rtx 4090", "rtx4090"),
    ("nvidia l40s", "l40s"),
    ("nvidia l40", "l40"),
    ("tesla v100", "v100"),
    ("nvidia v100", "v100"),
    ("nvidia a100", "a100"),
    ("nvidia h100", "h100"),
)


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def detect_gpu_inventory() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        gpu_names = []
    return {
        "gpu_names": gpu_names,
        "gpu_count": len(gpu_names),
    }


def infer_hardware_profile_name(profile_name: str | None = None) -> str:
    normalized = (profile_name or "").strip().lower()
    if normalized and normalized not in {"auto", "detect"}:
        if normalized in HARDWARE_PROFILES:
            return normalized
        for alias, resolved in GPU_NAME_ALIASES:
            if alias == normalized:
                return resolved

    inventory = detect_gpu_inventory()
    gpu_names = inventory["gpu_names"]
    if not gpu_names:
        return "cpu_only"

    first_gpu = gpu_names[0].lower()
    for alias, resolved in GPU_NAME_ALIASES:
        if alias in first_gpu:
            return resolved
    return "generic_gpu"


def _detect_host_cpu_count() -> int:
    return max(1, multiprocessing.cpu_count())


def resolve_hardware_profile(
    profile_name: str,
    overrides: dict | None = None,
    *,
    gpu_count: int | None = None,
    host_cpu_count: int | None = None,
    host_ram_gb: int | None = None,
) -> dict:
    resolved_name = infer_hardware_profile_name(profile_name)
    profile = deepcopy(HARDWARE_PROFILES.get(resolved_name) or HARDWARE_PROFILES["generic_gpu"])
    inventory = detect_gpu_inventory()
    detected_gpu_count = inventory["gpu_count"]
    detected_gpu_name = inventory["gpu_names"][0] if inventory["gpu_names"] else ""
    effective_gpu_count = max(0, gpu_count if gpu_count is not None else detected_gpu_count)
    effective_cpu_count = max(1, host_cpu_count if host_cpu_count is not None else _detect_host_cpu_count())

    profile["resolved_profile_name"] = resolved_name
    profile["detected_gpu_model"] = detected_gpu_name or profile.get("gpu_model", "")
    profile["detected_gpu_count"] = detected_gpu_count
    profile["effective_gpu_count"] = effective_gpu_count
    profile["effective_host_cpu_count"] = effective_cpu_count
    if host_ram_gb is not None:
        profile["effective_host_ram_gb"] = host_ram_gb

    if effective_gpu_count > 1:
        profile["service_processes"] = max(1, effective_gpu_count)
        profile["prefetch_queue_size"] = int(profile.get("prefetch_queue_size", 32)) * effective_gpu_count
        profile["exchange_max_workers"] = max(
            int(profile.get("exchange_max_workers", 1)),
            effective_gpu_count,
        )

    if effective_cpu_count < 12:
        profile["cpu_decode_workers"] = min(int(profile.get("cpu_decode_workers", 4)), max(effective_cpu_count - 2, 1))
        profile["cpu_crop_workers"] = min(int(profile.get("cpu_crop_workers", 3)), max(effective_cpu_count - 4, 1))
        profile["metadata_workers"] = min(int(profile.get("metadata_workers", 2)), max(effective_cpu_count - 6, 1))

    if overrides:
        profile = _deep_merge(profile, overrides)
    return profile
