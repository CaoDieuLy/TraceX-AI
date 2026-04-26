from __future__ import annotations

from .hardware_profiles import resolve_hardware_profile


def build_execution_plan(*, pipeline_spec: dict, hardware_profile: dict, gpu_count: int, host_cpu_count: int, host_ram_gb: int) -> dict:
    physical_gpu_count = max(int(gpu_count), 0)
    service_processes = int(hardware_profile.get("service_processes", 1))
    if physical_gpu_count > 1:
        service_processes = max(service_processes, physical_gpu_count)
    gpu_streams = int(hardware_profile.get("gpu_streams", 2))
    scale = max(physical_gpu_count, 1)
    available_cpu = max(host_cpu_count - 2, 1)
    decode_workers = min(int(hardware_profile.get("cpu_decode_workers", 4)) * scale, available_cpu)
    crop_workers = min(int(hardware_profile.get("cpu_crop_workers", 4)) * scale, available_cpu)
    metadata_workers = min(int(hardware_profile.get("metadata_workers", 2)) * scale, max(available_cpu // 2, 1))
    prefetch_queue_size = int(hardware_profile.get("prefetch_queue_size", 32))
    if host_ram_gb < 48:
        prefetch_queue_size = max(prefetch_queue_size // 2, 16)
    parallel_video_jobs = min(
        int(hardware_profile.get("exchange_max_workers", 1)) * scale,
        max(host_cpu_count, 1),
    )

    return {
        "hardware": {
            "gpu_profile": hardware_profile.get("gpu_model"),
            "gpu_count": physical_gpu_count,
            "host_cpu_count": host_cpu_count,
            "host_ram_gb": host_ram_gb,
            "resolved_profile_name": hardware_profile.get("resolved_profile_name"),
        },
        "process_model": {
            "api_processes": 1,
            "gpu_service_processes": service_processes,
            "why_single_gpu_process": "Avoid duplicating large detector/ReID/VLM weights into multiple worker processes on the same card.",
        },
        "parallelism": {
            "gpu_streams": gpu_streams,
            "cpu_decode_workers": decode_workers,
            "cpu_crop_workers": crop_workers,
            "metadata_workers": metadata_workers,
            "prefetch_queue_size": prefetch_queue_size,
            "parallel_video_jobs": parallel_video_jobs,
        },
        "batching": {
            "detector_batch_size": int(hardware_profile.get("detector_batch_size", 8)),
            "reid_batch_size": int(hardware_profile.get("reid_batch_size", 64)),
            "vlm_batch_size": int(hardware_profile.get("vlm_batch_size", 8)),
            "embedding_batch_size": int(hardware_profile.get("embedding_batch_size", 64)),
        },
        "memory": {
            "pin_memory": bool(hardware_profile.get("pin_memory", True)),
            "non_blocking_transfers": bool(hardware_profile.get("non_blocking_transfers", True)),
            "channels_last": bool(hardware_profile.get("channels_last", True)),
            "allow_tf32": bool(hardware_profile.get("allow_tf32", True)),
            "cudnn_benchmark": bool(hardware_profile.get("cudnn_benchmark", True)),
        },
        "stage_placement": {
            "decode": "cpu_thread_pool",
            "crop_extraction": "cpu_thread_pool",
            "detector": f"gpu_stream_0_{hardware_profile.get('detector_precision', 'fp16')}",
            "reid": f"gpu_stream_1_{hardware_profile.get('reid_precision', 'fp16')}",
            "semantic_vlm": f"gpu_stream_2_{hardware_profile.get('vlm_precision', 'fp16')}" if gpu_streams >= 3 else "shared_gpu_stream",
            "geometry_and_rerank": "cpu_vectorized",
            "vector_db": "cpu_io_bound",
        },
        "bottleneck_controls": [
            "Double-buffer CPU decode and GPU inference.",
            "Batch crops before ReID and VLM inference instead of per-frame calls.",
            "Keep vector search post-processing on CPU to preserve GPU time for detector/ReID/VLM.",
            "Use one GPU process per physical GPU unless you have measured a better replication strategy.",
            "Fixed strict pipeline selected.",
        ],
        "notes": hardware_profile.get("execution_notes", []),
    }


def resolve_execution_plan(
    *,
    pipeline_spec: dict,
    gpu_profile_name: str,
    gpu_profile_overrides: dict | None,
    gpu_count: int,
    host_cpu_count: int,
    host_ram_gb: int,
) -> tuple[dict, dict]:
    hardware_profile = resolve_hardware_profile(
        gpu_profile_name,
        gpu_profile_overrides,
        gpu_count=gpu_count,
        host_cpu_count=host_cpu_count,
        host_ram_gb=host_ram_gb,
    )
    plan = build_execution_plan(
        pipeline_spec=pipeline_spec,
        hardware_profile=hardware_profile,
        gpu_count=gpu_count,
        host_cpu_count=host_cpu_count,
        host_ram_gb=host_ram_gb,
    )
    return hardware_profile, plan
