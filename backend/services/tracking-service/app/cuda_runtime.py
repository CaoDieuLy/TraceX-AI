from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=8)
def configure_torch_runtime(
    *,
    allow_tf32: bool,
    cudnn_benchmark: bool,
    host_cpu_count: int,
) -> dict:
    try:
        import torch
    except Exception:
        return {
            "torch_available": False,
            "cuda_available": False,
        }

    cpu_threads = max(1, min(int(host_cpu_count), 64))
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    try:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, min(cpu_threads // 2, 32)))
    except Exception:
        pass

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0

    if cuda_available:
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    return {
        "torch_available": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "configured_cpu_threads": cpu_threads,
        "allow_tf32": bool(allow_tf32),
        "cudnn_benchmark": bool(cudnn_benchmark),
    }
