import streamlit as st
import cv2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline.search_engine import SearchEngine
from src.utils.logger import logger

st.set_page_config(page_title="Retail Video Semantic Search", layout="wide")
st.title("🏬 Retail Video Semantic Search (RVSS)")
st.markdown("Tìm kiếm video cửa hàng bán lẻ bằng ngôn ngữ tự nhiên | Hỗ trợ nhiều dataset")

@st.cache_resource
def get_engine():
    return SearchEngine()

engine = get_engine()

# Sidebar
st.sidebar.header("Tùy chọn tìm kiếm")
top_k = st.sidebar.slider("Số kết quả trả về", 5, 50, 20)

# Dataset filter
datasets = ["all", "surveillance_retail", "physicalai", "retailaction", "mimex", "unknown"]
dataset_filter = st.sidebar.selectbox("Dataset nguồn", datasets, index=0)
if dataset_filter == "all":
    dataset_filter = None

# Store filter (auto-populate after first search)
store_filter = st.sidebar.text_input("Lọc theo cửa hàng (store_id)", placeholder="store_001")
if store_filter == "":
    store_filter = None

# Search mode
search_mode = st.sidebar.radio("Chế độ tìm kiếm", ["Single query", "AND (nhiều điều kiện)"])

# Main query input
if search_mode == "Single query":
    query = st.text_input("🔍 Nhập câu hỏi:", "no staff at counter")
    if st.button("Tìm kiếm"):
        with st.spinner("Đang tìm kiếm..."):
            results = engine.search(query, top_k=top_k, 
                                    store_filter=store_filter, 
                                    dataset_filter=dataset_filter)
        st.session_state['results'] = results
else:
    col1, col2 = st.columns(2)
    with col1:
        query1 = st.text_input("Điều kiện 1:", "no staff at counter")
    with col2:
        query2 = st.text_input("Điều kiện 2:", "many customers")
    if st.button("Tìm kiếm kết hợp"):
        with st.spinner("Đang tìm kiếm..."):
            results = engine.multi_and_search([query1, query2], final_k=top_k, 
                                               store_filter=store_filter)
        st.session_state['results'] = results

# Display results
if 'results' in st.session_state and st.session_state['results']:
    st.subheader(f"📸 Kết quả ({len(st.session_state['results'])} khung hình)")
    
    # Statistics
    col1, col2 = st.columns(2)
    with col1:
        store_counts = engine.group_by_store(st.session_state['results'])
        st.write("**Thống kê theo cửa hàng:**")
        for store, count in store_counts.items():
            st.write(f"- {store}: {count} lần xuất hiện")
    with col2:
        dataset_counts = engine.group_by_dataset(st.session_state['results'])
        st.write("**Thống kê theo dataset:**")
        for ds, count in dataset_counts.items():
            st.write(f"- {ds}: {count} lần xuất hiện")
    
    # Display results as cards
    cols = st.columns(2)
    for idx, res in enumerate(st.session_state['results']):
        payload = res['payload']
        score = res['score']
        video_path = payload['video_path']
        timestamp = payload['timestamp']
        store = payload['store_id']
        camera = payload['camera_id']
        dataset = payload.get('dataset', 'unknown')
        
        with cols[idx % 2]:
            with st.container():
                st.markdown(f"**🏪 {store} | 📷 {camera} | 📁 {dataset}**")
                st.markdown(f"⏱️ Thời điểm: {timestamp:.1f}s | 🎯 Độ tương đồng: {score:.3f}")
                # Try to display thumbnail
                if Path(video_path).exists():
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_no = int(timestamp * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, use_container_width=True)
                st.markdown(f"[🎬 Xem video]({video_path})")
                st.markdown("---")

st.markdown("---")
st.caption("RVSS v1.0 | Hỗ trợ: Surveillance for Retail Stores, PhysicalAI-SmartSpaces, RetailAction, MIMEX")
