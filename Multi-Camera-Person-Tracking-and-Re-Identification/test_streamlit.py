"""Streamlit Test: Chạy VLM embedding 8 video, hiển thị metadata + benchmark."""
import streamlit as st
import time
import glob
import torch
import multiprocessing
import cv2
import os
from PIL import Image
import numpy as np

st.set_page_config(page_title="VLM Benchmark Test", layout="wide")
st.title("🧪 Test VLM Embedding — 8 Videos")

hw_mode = "🟢 GPU CUDA" if torch.cuda.is_available() else f"🔵 CPU ({multiprocessing.cpu_count()} cores)"
st.caption(f"Hardware: {hw_mode} | PyTorch: {torch.__version__}")

# Tìm 8 video
videos = sorted(glob.glob("data/NVIDIA_SmartSpaces/**/*.mp4", recursive=True))[:8]
st.info(f"📹 Tìm thấy {len(videos)} videos")

for v in videos:
    st.text(f"  • {os.path.basename(v)}")

st.divider()

@st.cache_resource
def load_vlm():
    from src_vlm.vlm_engine import VLM_Metadata_Engine
    return VLM_Metadata_Engine(use_mock=False)

if st.button("▶️ Bắt đầu Embedding 8 Videos", type="primary"):
    
    # Load model
    status = st.empty()
    status.warning("⏳ Đang tải model BLIP-Large...")
    t0 = time.time()
    engine = load_vlm()
    t_load = time.time() - t0
    status.success(f"✅ Model loaded in {t_load:.1f}s")
    
    # Progress bar
    progress = st.progress(0, text="Đang xử lý...")
    
    # Benchmark tổng
    all_metadata = []
    timing_data = []
    t_start = time.time()
    
    for i, vp in enumerate(videos):
        vid_name = os.path.basename(vp)
        progress.progress((i) / len(videos), text=f"🔄 Đang xử lý {vid_name} ({i+1}/{len(videos)})...")
        
        t1 = time.time()
        frames, metadata = engine.process_video(vp)
        elapsed = time.time() - t1
        
        all_metadata.extend(metadata)
        timing_data.append({"video": vid_name, "time": elapsed, "records": len(metadata)})
        
        # Hiển thị kết quả từng video
        with st.expander(f"📊 Video {i+1}: {vid_name} — ⏱️ {elapsed:.1f}s — {len(metadata)} records", expanded=(i==0)):
            
            # Hiển thị frames + captions
            cols = st.columns(4)
            for j, m in enumerate(metadata):
                with cols[j % 4]:
                    # Trích frame thật
                    cap = cv2.VideoCapture(vp)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(m['frame_idx']))
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(Image.fromarray(frame_rgb), caption=f"Frame {m['frame_idx']}", use_container_width=True)
                    
                    st.markdown(f"**Caption:** _{m['caption']}_")
    
    progress.progress(1.0, text="✅ Hoàn tất!")
    total_time = time.time() - t_start
    
    # Tổng kết
    st.divider()
    st.header("🏁 Tổng kết Benchmark")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Videos", len(videos))
    col2.metric("Total Records", len(all_metadata))
    col3.metric("Total Time", f"{total_time:.1f}s")
    col4.metric("Avg/Video", f"{total_time/len(videos):.1f}s")
    
    # Bảng timing
    st.subheader("⏱️ Chi tiết thời gian")
    import pandas as pd
    df = pd.DataFrame(timing_data)
    st.dataframe(df, use_container_width=True)
    
    # Toàn bộ metadata
    st.subheader("📋 Toàn bộ Metadata đã sinh")
    for m in all_metadata:
        st.text(f"[{m['video_id']}] Frame {m['frame_idx']:>6} | {m['caption']}")
