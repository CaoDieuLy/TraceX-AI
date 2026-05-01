from __future__ import annotations

from .hardware_runtime import resolve_detected_hardware


def build_execution_plan(*, pipeline_spec: dict, detected_hardware: dict) -> dict:
    physical_gpu_count = max(int(detected_hardware.get("gpu_count") or 0), 0)
    host_cpu_count = max(int(detected_hardware.get("host_cpu_count") or 1), 1)
    host_ram_gb = max(int(detected_hardware.get("host_ram_gb") or 1), 1)
    service_processes = int(detected_hardware.get("service_processes", 1))
    if physical_gpu_count > 1:
        service_processes = max(service_processes, physical_gpu_count)
    gpu_streams = int(detected_hardware.get("gpu_streams", 2))
    scale = max(physical_gpu_count, 1)
    available_cpu = max(host_cpu_count - 2, 1)
    decode_workers = min(int(detected_hardware.get("cpu_decode_workers", 4)) * scale, available_cpu)
    crop_workers = min(int(detected_hardware.get("cpu_crop_workers", 4)) * scale, available_cpu)
    metadata_workers = min(int(detected_hardware.get("metadata_workers", 2)) * scale, max(available_cpu // 2, 1))
    prefetch_queue_size = int(detected_hardware.get("prefetch_queue_size", 32))
    if host_ram_gb < 48:
        prefetch_queue_size = max(prefetch_queue_size // 2, 16)
    parallel_video_jobs = min(
        int(detected_hardware.get("exchange_max_workers", 1)) * scale,
        max(host_cpu_count, 1),
    )
    if physical_gpu_count > 0:
        detector_placement = f"gpu_stream_0_{detected_hardware.get('detector_precision', 'fp16')}"
        reid_placement = f"gpu_stream_1_{detected_hardware.get('reid_precision', 'fp16')}"
        semantic_placement = (
            f"gpu_stream_2_{detected_hardware.get('vlm_precision', 'fp16')}"
            if gpu_streams >= 3
            else "shared_gpu_stream"
        )
    else:
        detector_placement = "cpu_inference_path"
        reid_placement = "cpu_inference_path"
        semantic_placement = "cpu_inference_path"

    return {
        "hardware": {
            "gpu_model": detected_hardware.get("gpu_model"),
            "gpu_count": physical_gpu_count,
            "host_cpu_count": host_cpu_count,
            "host_ram_gb": host_ram_gb,
            "detection_mode": detected_hardware.get("detection_mode"),
            "detected_gpu_names": detected_hardware.get("detected_gpu_names", []),
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
            "detector_batch_size": int(detected_hardware.get("detector_batch_size", 8)),
            "reid_batch_size": int(detected_hardware.get("reid_batch_size", 64)),
            "vlm_batch_size": int(detected_hardware.get("vlm_batch_size", 8)),
            "embedding_batch_size": int(detected_hardware.get("embedding_batch_size", 64)),
        },
        "precision": {
            "detector": str(detected_hardware.get("detector_precision", "fp32")),
            "reid": str(detected_hardware.get("reid_precision", "fp32")),
            "vlm": str(detected_hardware.get("vlm_precision", "fp32")),
            "embedding": str(detected_hardware.get("reid_precision", "fp32")),
        },
        "memory": {
            "pin_memory": bool(detected_hardware.get("pin_memory", True)),
            "non_blocking_transfers": bool(detected_hardware.get("non_blocking_transfers", True)),
            "channels_last": bool(detected_hardware.get("channels_last", True)),
            "allow_tf32": bool(detected_hardware.get("allow_tf32", True)),
            "cudnn_benchmark": bool(detected_hardware.get("cudnn_benchmark", True)),
        },
        "stage_placement": {
            "decode": "cpu_thread_pool",
            "crop_extraction": "cpu_thread_pool",
            "detector": detector_placement,
            "reid": reid_placement,
            "semantic_vlm": semantic_placement,
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
        "notes": detected_hardware.get("execution_notes", []),
    }


def resolve_execution_plan(*, pipeline_spec: dict) -> tuple[dict, dict]:
    detected_hardware = resolve_detected_hardware()
    plan = build_execution_plan(
        pipeline_spec=pipeline_spec,
        detected_hardware=detected_hardware,
    )
    return detected_hardware, plan
