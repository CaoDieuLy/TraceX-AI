"""Test nhanh: Chạy VLM embedding cho 8 video, đo thời gian, in metadata."""
import time
import glob
import json

# Lấy 8 video đầu tiên
videos = sorted(glob.glob("data/NVIDIA_SmartSpaces/**/*.mp4", recursive=True))[:8]
print(f"📹 Tìm thấy {len(videos)} videos để test:\n")
for v in videos:
    print(f"  - {v}")

# Khởi tạo VLM Engine
print("\n⏳ Đang tải model BLIP-Large...")
t0 = time.time()
from src_vlm.vlm_engine import VLM_Metadata_Engine
engine = VLM_Metadata_Engine(use_mock=False)
t_load = time.time() - t0
print(f"✅ Model loaded in {t_load:.1f}s\n")

# Chạy embedding từng video
print("="*80)
all_metadata = []
for i, vp in enumerate(videos):
    t1 = time.time()
    frames, metadata = engine.process_video(vp)
    elapsed = time.time() - t1
    all_metadata.extend(metadata)
    print(f"\n📊 Video {i+1}/8: {vp.split('/')[-1]} — {elapsed:.1f}s — {len(metadata)} records")
    for m in metadata:
        print(f"   Frame {m['frame_idx']:>6} | {m['caption']}")

# Tổng kết
total_time = time.time() - t0
print("\n" + "="*80)
print(f"\n🏁 TỔNG KẾT:")
print(f"   Videos processed:  {len(videos)}")
print(f"   Total metadata:    {len(all_metadata)} records")
print(f"   Model load time:   {t_load:.1f}s")
print(f"   Total time:        {total_time:.1f}s")
print(f"   Avg per video:     {(total_time - t_load) / len(videos):.1f}s")
