import os
from huggingface_hub import snapshot_download

print("Đang tải xuống dữ liệu thử nghiệm Load Test (khoảng 100 Camera) từ NVIDIA PhysicalAI-SmartSpaces...")

local_dir = "data/NVIDIA_SmartSpaces"
os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id="nvidia/PhysicalAI-SmartSpaces", 
    repo_type="dataset", 
    allow_patterns=[
        # Kéo loạt các thư mục Hospital từ tập validation và training của NVIDIA
        "MTMC_Tracking_2025/val/Hospital_*/videos/*.mp4",
        "MTMC_Tracking_2025/train/Hospital_*/videos/*.mp4",
        "MTMC_Tracking_2025/val/Hospital_*/*.json",
    ], 
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print(f"✅ Tải thành công Load Test Camera! Dữ liệu được lưu tại: {local_dir}")
