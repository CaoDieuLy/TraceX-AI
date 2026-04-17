import glob
import multiprocessing
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
except Exception:  # pragma: no cover - optional dependency on lightweight images
    torch = None

def _detect_hardware():
    """Tự động phát hiện GPU hay CPU, trả về cấu hình tối ưu."""
    has_gpu = bool(torch and torch.cuda.is_available())
    cpu_cores = multiprocessing.cpu_count()
    
    # Kiểm tra ffmpeg có hỗ trợ nvenc không
    has_nvenc = False
    if has_gpu:
        try:
            result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5)
            has_nvenc = "hevc_nvenc" in result.stdout or "h264_nvenc" in result.stdout
        except Exception:
            pass
    
    if has_gpu and has_nvenc:
        mode = "GPU"
        codec_h265 = "hevc_nvenc"
        codec_h264 = "h264_nvenc"
        extra_args = ["-hwaccel", "cuda", "-preset", "p4", "-cq", "28"]
        # GPU encoding: mỗi encode session dùng ít CPU, có thể chạy nhiều hơn
        max_workers = min(cpu_cores * 2, 50)
    else:
        mode = "CPU"
        codec_h265 = "libx265"
        codec_h264 = "libx264"
        extra_args = ["-preset", "fast", "-crf", "28"]
        # CPU encoding: mỗi ffmpeg chiếm ~1-2 cores, chia đều
        max_workers = max(cpu_cores // 2, 2)
    
    print(f"[Ingestion] Hardware: {'🟢 GPU NVENC' if mode == 'GPU' else f'🔵 CPU ({cpu_cores} cores)'}")
    print(f"[Ingestion] Codec: {codec_h265} (H.265) / {codec_h264} (H.264)")
    print(f"[Ingestion] Max workers: {max_workers}")
    
    return {
        "mode": mode,
        "codec_h265": codec_h265,
        "codec_h264": codec_h264,
        "extra_args": extra_args,
        "max_workers": max_workers,
        "cpu_cores": cpu_cores,
    }

# Detect once at module load
_HW_CONFIG = None

def _get_hw_config():
    global _HW_CONFIG
    if _HW_CONFIG is None:
        _HW_CONFIG = _detect_hardware()
    return _HW_CONFIG

def compress_video(input_path, output_dir, use_h265=True, output_filename=None):
    """Nén video sang chuẩn H.264 hoặc H.265 — tự chọn GPU/CPU."""
    hw = _get_hw_config()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    filename = output_filename or os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)
    
    # Nếu file đã tồn tại và có kích thước hợp lệ, skip
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"[Ingestion] Skip (đã nén): {filename}")
        return output_path
    
    codec = hw["codec_h265"] if use_h265 else hw["codec_h264"]
    
    if hw["mode"] == "GPU":
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "cuda", "-i", input_path,
            "-c:v", codec, "-preset", "p4", "-cq", "28",
            output_path
        ]
    else:
        # CPU mode: giới hạn threads per ffmpeg process
        threads_per_worker = max(hw["cpu_cores"] // hw["max_workers"], 1)
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", codec, "-preset", "fast", "-crf", "28",
            "-threads", str(threads_per_worker),
            output_path
        ]
    
    print(f"[Ingestion] [{hw['mode']}] Nén {filename} → {codec}...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        return output_path
    except FileNotFoundError:
        # FFmpeg không có sẵn -> copy file gốc làm fallback
        print(f"[Ingestion] FFmpeg không tìm thấy, copy nguyên file: {filename}")
        shutil.copy2(input_path, output_path)
        return output_path
    except subprocess.TimeoutExpired:
        print(f"[Ingestion] Timeout nén {filename}, copy nguyên file.")
        shutil.copy2(input_path, output_path)
        return output_path
    except Exception as e:
        print(f"[Ingestion] Lỗi nén {filename}: {e}, copy nguyên file.")
        shutil.copy2(input_path, output_path)
        return output_path

def ingest_videos_parallel(input_dir, output_dir, max_workers=None):
    """Xử lý nén camera song song — tự điều chỉnh workers theo GPU/CPU."""
    hw = _get_hw_config()
    
    # Nếu không truyền max_workers, dùng giá trị tự detect
    if max_workers is None:
        max_workers = hw["max_workers"]
    else:
        # Clamp theo phần cứng thực tế
        if hw["mode"] == "CPU":
            max_workers = min(max_workers, hw["max_workers"])
    
    # Tìm đệ quy toàn bộ video
    videos = glob.glob(os.path.join(input_dir, "**/*.mp4"), recursive=True) + \
             glob.glob(os.path.join(input_dir, "**/*.avi"), recursive=True)
    if not videos:
        print("[Ingestion] Không tìm thấy video mới nào.")
        return []

    print(f"[Ingestion] [{hw['mode']}] Xử lý {len(videos)} videos với {max_workers} workers...")
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    # CPU mode: dùng ThreadPoolExecutor (ffmpeg là subprocess, không cần ProcessPool)
    # GPU mode: cũng ThreadPool vì ffmpeg subprocess tự dùng GPU
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compress_video, v, output_dir) for v in videos]
        for i, f in enumerate(futures):
            result = f.result()
            results.append(result)
            if (i + 1) % 5 == 0 or i == len(futures) - 1:
                print(f"[Ingestion] Tiến độ: {i+1}/{len(videos)} videos xong.")
    
    return results

if __name__ == "__main__":
    hw = _get_hw_config()
    print(f"\nTest mode: {hw['mode']}")
    ingest_videos_parallel("../data/videos/raw", "../data/videos/compressed")
