import streamlit as st
import time
import torch
import multiprocessing
from src_vlm.vlm_engine import VLM_Metadata_Engine
from src_vlm.vector_search import VectorSearchEngine
from src_vlm.tracker import ReID_Tracker
from src_vlm.video_ingestion import ingest_videos_parallel

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Hospital VLM & ReID Demo", layout="wide")

st.title("🏥 Hệ thống Demo Day: Hospital Multi-Camera VLM & ReID")

# Hiển thị Hardware Info
hw_mode = "🟢 GPU CUDA" if torch.cuda.is_available() else f"🔵 CPU ({multiprocessing.cpu_count()} cores)"
st.caption(f"Hardware: {hw_mode} | PyTorch: {torch.__version__}")

# Tải các module (được cache để không load lại nhiều lần)
@st.cache_resource
def load_engines():
    # Khởi tạo instance VLM, Vector Search và ReID Tracker thật
    vlm = VLM_Metadata_Engine(use_mock=False)
    vc = VectorSearchEngine(db_path="data/db")
    tracker = ReID_Tracker(use_mock=False)
    return vlm, vc, tracker

vlm_engine, vector_search, tracker = load_engines()

# ---- Bước 1 & 2.1: Ingest Video mới ----
st.header("1. Cập nhật Camera Loop (Ingest 10 phút / 50 Cam)")
if st.button("Nạp Video Camera mới (Video Ingestion)"):
    with st.spinner(f"[{hw_mode}] Đang nén video song song..."):
        # max_workers=None → tự detect theo GPU/CPU
        compressed_videos = ingest_videos_parallel("data/NVIDIA_SmartSpaces", "data/videos/compressed", max_workers=None)
        st.success(f"Đã nén xong {len(compressed_videos)} video qua H.264/H.265")
        
    with st.spinner(f"[{hw_mode}] Đang chạy VLM (BLIP-Large) sinh Metadata..."):
        if not compressed_videos:
            st.warning("Chưa có video nào nén xong!")
        else:
            all_metadata = vlm_engine.process_videos_batch(compressed_videos[:100])
            vector_search.index_metadata(all_metadata)
            st.success(f"Đã bóc tách {len(all_metadata)} metadata records và Index vào VectorDB.")


# ---- Bước 2.2: User Query & Candidate Selection ----
st.header("2. Truy vấn Video (Tìm kiếm đối tượng)")
query = st.text_input("Nhập văn bản miêu tả cần tìm (Text Query):", "Bác sĩ mang hộp dụng cụ y tế")

if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "page" not in st.session_state:
    st.session_state.page = 0
if "fail_count" not in st.session_state:
    st.session_state.fail_count = 0

if st.button("Truy vấn Data (Search)"):
    st.session_state.candidates = vector_search.search_candidates(query, top_k=50)
    st.session_state.page = 0
    st.session_state.fail_count = 0

if st.session_state.candidates:
    start_idx = st.session_state.page * 10
    end_idx = start_idx + 10
    current_batch = st.session_state.candidates[start_idx:end_idx]
    
    st.subheader(f"Hiển thị Kết quả (Page {st.session_state.page + 1})")
    
    # Hiển thị k Frame Candidates
    cols = st.columns(5)
    import numpy as np
    from PIL import Image
    for idx, c in enumerate(current_batch):
        with cols[idx % 5]:
            import cv2
            import os
            
            # Khởi tạo giá trị mặc định nếu ko lấy được ảnh
            display_img = Image.fromarray(np.ones((150, 200, 3), dtype=np.uint8) * 150)
            
            video_path = os.path.join("data/videos/compressed", c['video_id'])
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(c['frame_idx']))
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    display_img = Image.fromarray(frame_rgb)
            
            st.image(display_img, caption=f"Cam: {c['video_id']} | Frame: {c['frame_idx']}")
            if st.button(f"Chọn Frame này", key=f"select_{c['id']}"):
                st.session_state.selected_candidate = c
                st.success(f"Đã chọn Candidate: {c['video_id']}")
    
    # Logic Phân trang (Fail case 50 frames)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Các Frame này chưa đúng, Tải lứa tiếp theo >>"):
            st.session_state.page += 1
            st.session_state.fail_count += 10
            if st.session_state.fail_count >= 50:
                st.error("❌ Đã duyệt qua 50 frames nhưng người dùng không thấy kết quả hợp lệ -> CASE FAIL.")
                st.session_state.candidates = []

# ---- Bước 2.3: Chạy ReID Tracker qua 50 Cam ----
import os
if "selected_candidate" in st.session_state and st.session_state.selected_candidate:
    st.header("3. Multi-Camera ReID Tracking")
    st.info(f"Đang Track đối tượng từ: {st.session_state.selected_candidate['video_id']} (Frame {st.session_state.selected_candidate['frame_idx']})")
    
    if st.button("Tiến hành Tracking trên Toàn tuyến"):
        with st.spinner("Đang cắt clip 10s và overlay ReID tracking..."):
            result_vid = tracker.run_tracking_on_candidate(
                st.session_state.selected_candidate, "data/videos/compressed", "data/videos/tracking_output"
            )
        st.success("✅ Tracking hoàn tất! Clip 10 giây đã sẵn sàng.")
        if result_vid and os.path.exists(result_vid):
            st.video(result_vid)
        else:
            st.warning("Không tìm thấy file video gốc để cắt clip.")
