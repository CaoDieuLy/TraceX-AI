from __future__ import annotations

from copy import deepcopy


HARDWARE_PROFILES: dict[str, dict] = {
    "l4": {
        "gpu_model": "NVIDIA L4",
        "service_processes": 1,
        "gpu_streams": 3,
        "cpu_decode_workers": 8,
        "cpu_crop_workers": 6,
        "metadata_workers": 4,
        "prefetch_queue_size": 64,
        "detector_batch_size": 20,
        "reid_batch_size": 256,
        "vlm_batch_size": 24,
        "embedding_batch_size": 128,
        "detector_precision": "fp16",
        "reid_precision": "fp16",
        "vlm_precision": "bf16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "execution_notes": [
            "Use one GPU-serving process to avoid model duplication on a 24 GB card.",
            "Keep detector, ReID, and VLM on separate CUDA streams when model runtimes permit it.",
            "Pipeline CPU decode and crop extraction ahead of GPU inference with bounded queues.",
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
        "execution_notes": [
            "Favor smaller batches to protect latency and memory headroom.",
        ],
    },
    "a100": {
        "gpu_model": "NVIDIA A100",
        "service_processes": 1,
        "gpu_streams": 4,
        "cpu_decode_workers": 10,
        "cpu_crop_workers": 8,
        "metadata_workers": 4,
        "prefetch_queue_size": 96,
        "detector_batch_size": 32,
        "reid_batch_size": 384,
        "vlm_batch_size": 32,
        "embedding_batch_size": 192,
        "detector_precision": "bf16",
        "reid_precision": "bf16",
        "vlm_precision": "bf16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "execution_notes": [
            "Can sustain larger batches and deeper prefetch without compromising memory headroom.",
        ],
    },
    "h100": {
        "gpu_model": "NVIDIA H100",
        "service_processes": 1,
        "gpu_streams": 5,
        "cpu_decode_workers": 12,
        "cpu_crop_workers": 10,
        "metadata_workers": 6,
        "prefetch_queue_size": 128,
        "detector_batch_size": 40,
        "reid_batch_size": 512,
        "vlm_batch_size": 48,
        "embedding_batch_size": 256,
        "detector_precision": "bf16",
        "reid_precision": "bf16",
        "vlm_precision": "bf16",
        "pin_memory": True,
        "non_blocking_transfers": True,
        "channels_last": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "execution_notes": [
            "Leave headroom for concurrent detector and VLM streams before raising process count.",
        ],
    },
}


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_hardware_profile(profile_name: str, overrides: dict | None = None) -> dict:
    resolved_name = (profile_name or "l4").strip().lower()
    profile = HARDWARE_PROFILES.get(resolved_name) or HARDWARE_PROFILES["l4"]
    if overrides:
        return _deep_merge(profile, overrides)
    return deepcopy(profile)
