from __future__ import annotations

import ctypes
import multiprocessing
import os
import subprocess
from copy import deepcopy


CPU_ONLY_DEFAULTS: dict[str, object] = {
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
    "exchange_max_workers": 1,
    "execution_notes": [
        "No CUDA device detected; keep batches small and use CPU-safe defaults.",
    ],
}


GENERIC_GPU_DEFAULTS: dict[str, object] = {
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
    "exchange_max_workers": 2,
    "execution_notes": [
        "CUDA device detected; use balanced defaults until a more specific GPU family is recognized.",
    ],
}


GPU_TUNING_RULES: tuple[tuple[tuple[str, ...], dict[str, object]], ...] = (
    (
        ("nvidia h100",),
        {
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
            "exchange_max_workers": 8,
            "execution_notes": [
                "H100 detected; keep detector, ReID, and VLM overlapped on multiple streams.",
            ],
        },
    ),
    (
        ("nvidia a100",),
        {
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
            "exchange_max_workers": 6,
            "execution_notes": [
                "A100 detected; run larger batches and deeper prefetch safely.",
            ],
        },
    ),
    (
        ("nvidia l40s",),
        {
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
            "exchange_max_workers": 6,
            "execution_notes": [
                "L40S detected; sustain deeper queues while keeping latency headroom.",
            ],
        },
    ),
    (
        ("nvidia l40", "rtx 4090", "nvidia geforce rtx 4090"),
        {
            "gpu_streams": 5,
            "cpu_decode_workers": 12,
            "cpu_crop_workers": 9,
            "metadata_workers": 6,
            "prefetch_queue_size": 128,
            "detector_batch_size": 36,
            "reid_batch_size": 384,
            "vlm_batch_size": 32,
            "embedding_batch_size": 192,
            "exchange_max_workers": 5,
            "execution_notes": [
                "High-end Ada/Lovelace GPU detected; favor aggressive stream concurrency.",
            ],
        },
    ),
    (
        ("nvidia l4", "nvidia a30", "nvidia a40", "rtx a6000", "nvidia a6000"),
        {
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
            "exchange_max_workers": 4,
            "execution_notes": [
                "Mid-high datacenter GPU detected; keep decode, ReID, and VLM parallelized.",
            ],
        },
    ),
    (
        ("nvidia a10", "tesla v100", "nvidia v100"),
        {
            "gpu_streams": 3,
            "cpu_decode_workers": 8,
            "cpu_crop_workers": 6,
            "metadata_workers": 4,
            "prefetch_queue_size": 72,
            "detector_batch_size": 18,
            "reid_batch_size": 192,
            "vlm_batch_size": 16,
            "embedding_batch_size": 128,
            "exchange_max_workers": 3,
            "allow_tf32": False,
            "execution_notes": [
                "Older or mid-range accelerator detected; keep batches moderate to protect latency.",
            ],
        },
    ),
    (
        ("tesla t4", "nvidia t4"),
        {
            "gpu_streams": 2,
            "cpu_decode_workers": 6,
            "cpu_crop_workers": 4,
            "metadata_workers": 3,
            "prefetch_queue_size": 48,
            "detector_batch_size": 12,
            "reid_batch_size": 128,
            "vlm_batch_size": 12,
            "embedding_batch_size": 96,
            "exchange_max_workers": 2,
            "allow_tf32": False,
            "execution_notes": [
                "T4 detected; use smaller batches and fewer streams to stay within memory limits.",
            ],
        },
    ),
)


def detect_gpu_inventory() -> dict[str, object]:
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


def detect_host_cpu_count() -> int:
    return max(1, multiprocessing.cpu_count())


def detect_host_ram_gb() -> int:
    try:
        import psutil  # type: ignore

        return max(1, int(psutil.virtual_memory().total / (1024**3)))
    except Exception:
        pass

    try:
        if hasattr(os, "sysconf"):
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            return max(1, int((page_size * pages) / (1024**3)))
    except Exception:
        pass

    try:
        if os.name == "nt":
            total_kb = ctypes.c_ulonglong()
            ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(total_kb))
            return max(1, int(total_kb.value / (1024**2)))
    except Exception:
        pass

    return 64


def _deep_merge(base: dict[str, object], update: dict[str, object]) -> dict[str, object]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _tuning_for_gpu_name(gpu_name: str) -> dict[str, object]:
    normalized = gpu_name.strip().lower()
    tuning = deepcopy(GENERIC_GPU_DEFAULTS)
    for aliases, update in GPU_TUNING_RULES:
        if any(alias in normalized for alias in aliases):
            tuning = _deep_merge(tuning, update)
            break
    tuning["gpu_model"] = gpu_name
    return tuning


def resolve_detected_hardware() -> dict[str, object]:
    inventory = detect_gpu_inventory()
    gpu_names = list(inventory.get("gpu_names") or [])
    gpu_count = int(inventory.get("gpu_count") or 0)
    host_cpu_count = detect_host_cpu_count()
    host_ram_gb = detect_host_ram_gb()

    if gpu_count <= 0:
        detected = deepcopy(CPU_ONLY_DEFAULTS)
        detected["detection_mode"] = "cpu_only"
    else:
        detected = _tuning_for_gpu_name(str(gpu_names[0]))
        detected["detection_mode"] = "auto_detected_gpu"

    detected["detected_gpu_names"] = gpu_names
    detected["gpu_count"] = gpu_count
    detected["host_cpu_count"] = host_cpu_count
    detected["host_ram_gb"] = host_ram_gb

    if gpu_count > 1:
        detected["service_processes"] = max(1, gpu_count)
        detected["prefetch_queue_size"] = int(detected.get("prefetch_queue_size", 32)) * gpu_count
        detected["exchange_max_workers"] = max(int(detected.get("exchange_max_workers", 1)), gpu_count)

    if host_cpu_count < 12:
        detected["cpu_decode_workers"] = min(int(detected.get("cpu_decode_workers", 4)), max(host_cpu_count - 2, 1))
        detected["cpu_crop_workers"] = min(int(detected.get("cpu_crop_workers", 3)), max(host_cpu_count - 4, 1))
        detected["metadata_workers"] = min(int(detected.get("metadata_workers", 2)), max(host_cpu_count // 2, 1))

    if host_ram_gb < 48:
        detected["prefetch_queue_size"] = max(int(detected.get("prefetch_queue_size", 32)) // 2, 16)
        detected["detector_batch_size"] = max(int(detected.get("detector_batch_size", 8)) // 2, 4)
        detected["reid_batch_size"] = max(int(detected.get("reid_batch_size", 64)) // 2, 32)
        detected["vlm_batch_size"] = max(int(detected.get("vlm_batch_size", 8)) // 2, 4)
        detected["embedding_batch_size"] = max(int(detected.get("embedding_batch_size", 64)) // 2, 32)

    return detected
