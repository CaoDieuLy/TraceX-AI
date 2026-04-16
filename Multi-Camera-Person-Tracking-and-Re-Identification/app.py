import os
import multiprocessing

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch

from src_vlm.hospital_pipeline import (
    add_single_video,
    bootstrap_nvidia_hospital_dataset,
    current_pipeline_config,
    save_uploaded_video,
)
from src_vlm.tracker import ReID_Tracker
from src_vlm.vector_search import VectorSearchEngine
from src_vlm.vlm_engine import VLM_Metadata_Engine


st.set_page_config(page_title="Hospital VLM & ReID Demo", layout="wide")

st.title("Hospital Multi-Camera Human Retrieval")

hw_mode = "GPU CUDA" if torch.cuda.is_available() else f"CPU ({multiprocessing.cpu_count()} cores)"
st.caption(f"Hardware: {hw_mode} | PyTorch: {torch.__version__}")


@st.cache_resource
def load_engines():
    vlm = VLM_Metadata_Engine(use_mock=False)
    vector_db = VectorSearchEngine(db_path="data/db")
    tracker = ReID_Tracker(use_mock=False)
    return vlm, vector_db, tracker


def reset_search_state():
    st.session_state.candidates = []
    st.session_state.page = 0
    st.session_state.fail_count = 0
    st.session_state.selected_candidate = None
    st.session_state.tracking_result_video = None


def load_candidate_preview(candidate: dict):
    display_img = Image.fromarray(np.ones((180, 240, 3), dtype=np.uint8) * 150)
    video_path = os.path.join("data/videos/compressed", str(candidate.get("video_id", "")))
    if not os.path.exists(video_path):
        return display_img

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(candidate.get("frame_idx", 0)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return display_img

    bbox = candidate.get("bbox") or candidate.get("representative_bbox") or []
    if isinstance(bbox, list) and len(bbox) == 4:
        x, y, w, h = [int(value) for value in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 220, 20), 2)
        cv2.putText(
            frame,
            f"{candidate.get('camera_id', 'cam')} | track {candidate.get('track_id', '?')}",
            (max(10, x), max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 220, 20),
            2,
        )

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


vlm_engine, vector_search, tracker = load_engines()
pipeline_cfg = current_pipeline_config()

st.caption(
    " | ".join(
        [
            f"Text query model: {vector_search.model_name}",
            f"Metadata caption model: {vlm_engine.model_id}",
            f"Bootstrap detect: {pipeline_cfg['bootstrap_detection_mode']}",
            f"Add-video detect: {pipeline_cfg['incremental_detection_mode']}",
        ]
    )
)

if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "page" not in st.session_state:
    st.session_state.page = 0
if "fail_count" not in st.session_state:
    st.session_state.fail_count = 0
if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None
if "tracking_result_video" not in st.session_state:
    st.session_state.tracking_result_video = None


st.header("1. Camera Pipeline")
bootstrap_col, add_col = st.columns(2, gap="large")

with bootstrap_col:
    st.subheader("Rebuild 31 video nền")
    st.caption(
        "Convert 31 video NVIDIA Hospital sang H.265, sinh metadata chi tiết theo từng người từ ground truth, "
        "rồi rebuild VectorDB theo human candidates."
    )
    if st.button("Rebuild data cho 31 video NVIDIA Hospital", type="primary", width="stretch"):
        with st.spinner("Đang dựng lại 31 video nền và metadata per-person..."):
            bootstrap_summary = bootstrap_nvidia_hospital_dataset(vlm_engine, limit=31)
            index_summary = vector_search.rebuild_from_metadata_dir("data/metadata")
            reset_search_state()
        st.success(
            f"Đã rebuild {bootstrap_summary['processed_videos']} video, {bootstrap_summary['people_indexed']} person candidates."
        )
        with st.expander("Chi tiết rebuild", expanded=False):
            st.json({"bootstrap": bootstrap_summary, "index": index_summary})

with add_col:
    st.subheader("Add đúng 1 video mới")
    st.caption(
        "Chỉ ingest video vừa upload, convert sang H.265, detect người, sinh metadata per-person, "
        "update queue tối đa 32 video, rồi rebuild lại VectorDB từ metadata sẵn có."
    )
    uploaded_video = st.file_uploader(
        "Video mới",
        type=["mp4", "avi", "mov", "mkv", "hevc", "h265"],
        key="single_video_upload",
    )
    camera_id = st.text_input("Camera ID", value="Camera_32")
    recorded_start = st.text_input("Recorded start UTC", value="", placeholder="2026-01-01T00:10:00Z")
    if st.button("Add video mới", width="stretch"):
        if uploaded_video is None:
            st.warning("Chọn một video trước khi bấm add.")
        else:
            saved_path = save_uploaded_video(uploaded_video.getvalue(), uploaded_video.name)
            with st.spinner("Đang ingest video mới và cập nhật queue..."):
                ingest_summary = add_single_video(
                    source_path=saved_path,
                    vlm_engine=vlm_engine,
                    camera_id=camera_id or None,
                    recorded_start=recorded_start or None,
                )
                index_summary = vector_search.rebuild_from_metadata_dir("data/metadata")
                reset_search_state()
            st.success(
                f"Đã add video `{ingest_summary['video_id']}` với {ingest_summary['person_count']} human candidates."
            )
            if ingest_summary.get("evicted_video_ids"):
                st.info(f"Đã evict FIFO: {', '.join(ingest_summary['evicted_video_ids'])}")
            with st.expander("Chi tiết ingest", expanded=False):
                st.json({"ingest": ingest_summary, "index": index_summary})


st.header("2. Truy vấn Human Candidates")
query = st.text_input("Nhập mô tả người cần tìm", "bác sĩ mang hộp dụng cụ y tế")

if st.button("Search top-K human candidates"):
    st.session_state.candidates = vector_search.search_candidates(query, top_k=50)
    st.session_state.page = 0
    st.session_state.fail_count = 0

if st.session_state.candidates:
    start_idx = st.session_state.page * 10
    end_idx = start_idx + 10
    current_batch = st.session_state.candidates[start_idx:end_idx]
    st.subheader(f"Kết quả human candidates (Page {st.session_state.page + 1})")

    cols = st.columns(5)
    for idx, candidate in enumerate(current_batch):
        with cols[idx % 5]:
            st.image(load_candidate_preview(candidate), caption=f"{candidate.get('camera_id')} | frame {candidate.get('frame_idx')}")
            st.caption(
                f"Track {candidate.get('track_id')} | score {candidate.get('score', 0):.3f} | human_key {candidate.get('human_key')}"
            )
            st.caption(candidate.get("person_caption", candidate.get("search_text", "")))
            if st.button("Chọn human này", key=f"select_{candidate.get('id', idx)}"):
                st.session_state.selected_candidate = candidate
                st.success(f"Đã chọn human candidate: {candidate.get('candidate_id')}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Các candidate này chưa đúng, tải lứa tiếp theo >>"):
            st.session_state.page += 1
            st.session_state.fail_count += 10
            if st.session_state.fail_count >= 50:
                st.error("Đã duyệt qua 50 human candidates nhưng vẫn chưa có kết quả đúng.")
                st.session_state.candidates = []

    with col2:
        if current_batch:
            st.caption("Top-K hiện đang dedupe theo human_key để tránh lặp lại cùng một người ở nhiều frame.")

if st.session_state.selected_candidate:
    candidate = st.session_state.selected_candidate
    st.header("3. Candidate Details & Tracking")
    st.info(
        f"Đang xem candidate {candidate.get('candidate_id')} | Camera {candidate.get('camera_id')} | "
        f"Track {candidate.get('track_id')} | Frame {candidate.get('frame_idx')}"
    )
    st.image(load_candidate_preview(candidate), caption="Representative frame", width="stretch")
    st.caption(candidate.get("search_text", ""))
    with st.expander("Metadata chi tiết của candidate", expanded=False):
        st.json(candidate)

    if st.button("Tiến hành Tracking trên Toàn tuyến"):
        with st.spinner("Đang cắt clip 10s và overlay tracking..."):
            result_vid = tracker.run_tracking_on_candidate(
                candidate,
                "data/videos/compressed",
                "data/videos/tracking_output",
            )
        st.session_state.tracking_result_video = result_vid
        st.success("Tracking hoàn tất.")
        if result_vid and os.path.exists(result_vid):
            st.info(f"Đã tạo clip 10s tại: `{result_vid}`")
        else:
            st.warning("Không tìm thấy file video để cắt clip tracking.")

tracking_result_video = st.session_state.get("tracking_result_video")
if tracking_result_video and os.path.exists(tracking_result_video):
    st.header("4. Tracking Preview")
    st.success(f"Đã tạo clip 10s tại: `{tracking_result_video}`")
    st.video(tracking_result_video)
