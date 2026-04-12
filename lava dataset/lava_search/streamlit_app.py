from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from .config import default_bundle_root, project_root
from .indexer import load_bundle, search_index


def main() -> None:
    st.set_page_config(page_title="LAVA Search Demo", layout="wide")
    st.title("LAVA Multimodal Search Demo")

    bundle_dir = st.sidebar.text_input("Bundle directory", str(default_bundle_root()))
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=20, value=5)

    try:
        bundle = load_bundle(Path(bundle_dir))
        st.sidebar.success("Bundle loaded")
        st.sidebar.json(bundle.get("artifacts", {}))
    except Exception as exc:
        st.sidebar.error(str(exc))
        return

    query_text = st.text_input("Text query", value="red car turning left")
    uploaded = st.file_uploader("Optional image query", type=["jpg", "jpeg", "png"])
    run = st.button("Search", type="primary")

    image_path = None
    if uploaded is not None:
        temp_dir = project_root() / ".streamlit-cache"
        temp_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(uploaded.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=suffix, delete=False) as handle:
            handle.write(uploaded.read())
            image_path = handle.name

    if run:
        if not query_text and not image_path:
            st.warning("Provide text and/or image query.")
            return

        results = search_index(
            Path(bundle_dir),
            query_text=query_text or None,
            query_image_path=image_path,
            top_k=top_k,
        )

        for rank, item in enumerate(results, start=1):
            with st.container(border=True):
                st.subheader(f"#{rank} | score {item['score']:.4f}")
                st.write(
                    {
                        "location": item["location"],
                        "split": item["split"],
                        "seconds": [item["start_second"], item["end_second"]],
                        "captions": item["captions"],
                        "video_path": item["video_path"],
                        "score_breakdown": item.get("score_breakdown", {}),
                    }
                )
                visual = item.get("visual_path")
                if visual and Path(visual).exists():
                    st.image(visual, caption=visual, use_column_width=True)


if __name__ == "__main__":
    main()
