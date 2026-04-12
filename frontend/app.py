"""
Streamlit UI cho VLM video: nhập text hoặc ảnh → hiển thị video/ảnh + text trả lời.
Backend FastAPI: cấu hình VLM_API_URL và endpoint /infer (xem docstring gọi API).
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_API = os.getenv("VLM_API_URL", "http://127.0.0.1:8000")
DEFAULT_PATH = os.getenv("VLM_INFER_PATH", "/infer")


def _decode_b64_image(b64: str) -> bytes | None:
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


def call_infer(
    api_base: str,
    infer_path: str,
    query_text: str,
    image_bytes: bytes | None,
    image_name: str | None,
    image_mime: str | None,
    timeout: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """POST multipart tới FastAPI. Trả về (json_dict, error_message)."""
    url = api_base.rstrip("/") + (infer_path if infer_path.startswith("/") else f"/{infer_path}")
    data: dict[str, str] = {}
    if query_text.strip():
        data["query"] = query_text.strip()
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None
    if image_bytes and image_name:
        mime = image_mime or "application/octet-stream"
        files = [("image", (image_name, image_bytes, mime))]
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, data=data or None, files=files)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:500]}"
        return r.json(), None
    except httpx.ConnectError:
        return None, "Không kết nối được tới API. Hãy chạy FastAPI hoặc kiểm tra URL."
    except httpx.TimeoutException:
        return None, "Hết thời gian chờ phản hồi từ server."
    except Exception as e:
        return None, str(e)


def render_outputs(payload: dict[str, Any]) -> None:
    """Đọc JSON từ BE: hỗ trợ nhiều key tên thường gặp."""
    text = (
        payload.get("text")
        or payload.get("answer")
        or payload.get("answer_text")
        or payload.get("caption")
        or ""
    )
    if text:
        st.subheader("Văn bản trả lời")
        st.markdown(text)

    # Ảnh: URL, bytes base64, hoặc path tương đối (nếu BE trả URL đầy đủ)
    img_b64 = payload.get("image_base64") or payload.get("image_b64")
    img_url = payload.get("image_url") or payload.get("image")
    if img_b64:
        raw = _decode_b64_image(str(img_b64))
        if raw:
            st.subheader("Ảnh kết quả")
            st.image(raw)
    elif img_url and isinstance(img_url, str) and img_url.startswith(("http://", "https://")):
        st.subheader("Ảnh kết quả")
        st.image(img_url)

    vid_url = payload.get("video_url") or payload.get("video")
    if vid_url and isinstance(vid_url, str) and vid_url.startswith(("http://", "https://")):
        st.subheader("Video kết quả")
        st.video(vid_url)

    vid_b64 = payload.get("video_base64")
    if vid_b64:
        raw = base64.b64decode(str(vid_b64))
        st.subheader("Video kết quả")
        st.video(io.BytesIO(raw))


def main() -> None:
    st.set_page_config(page_title="VLM Video Search", layout="wide")
    st.title("Tìm kiếm / truy vấn video (VLM)")
    st.caption("Nhập mô tả bằng chữ và/hoặc tải ảnh — kết quả: video, ảnh hoặc văn bản từ backend FastAPI.")

    with st.sidebar:
        st.header("Cấu hình API")
        api_base = st.text_input("Base URL", value=DEFAULT_API, help="Ví dụ: http://127.0.0.1:8000")
        infer_path = st.text_input("Đường dẫn infer", value=DEFAULT_PATH, help="POST multipart, ví dụ /infer")
        timeout = st.number_input("Timeout (giây)", min_value=5.0, max_value=600.0, value=120.0, step=10.0)
        st.divider()
        st.markdown(
            "**Hợp đồng JSON gợi ý** (BE trả một hoặc nhiều trường): "
            "`text` | `answer_text`, `image_url`, `image_base64`, `video_url`, `video_base64`."
        )

    col_in, col_out = st.columns((1, 1))
    with col_in:
        st.subheader("Đầu vào")
        query = st.text_area(
            "Mô tả / truy vấn (text)",
            placeholder='Ví dụ: "Người phụ nữ mặc áo đỏ đi xe máy xanh"',
            height=120,
        )
        uploaded = st.file_uploader("Ảnh tham chiếu (tuỳ chọn)", type=["png", "jpg", "jpeg", "webp", "gif"])

    with col_out:
        st.subheader("Kết quả")

    submitted = st.button("Gửi truy vấn", type="primary")
    if not submitted:
        return

    if not query.strip() and not uploaded:
        st.warning("Vui lòng nhập ít nhất text hoặc tải một ảnh.")
        return

    image_bytes = uploaded.getvalue() if uploaded else None
    image_name = uploaded.name if uploaded else None
    image_mime = uploaded.type if uploaded else None

    with st.spinner("Đang gọi API…"):
        payload, err = call_infer(
            api_base=api_base,
            infer_path=infer_path,
            query_text=query,
            image_bytes=image_bytes,
            image_name=image_name,
            image_mime=image_mime,
            timeout=float(timeout),
        )

    with col_out:
        if err:
            st.error(err)
            st.info(
                "Khi BE chưa sẵn sàng, bạn vẫn có thể phát triển FastAPI với endpoint POST nhận "
                "`query` (form) và `image` (file), trả JSON như mô tả trong sidebar."
            )
            return
        if not isinstance(payload, dict):
            st.warning("Phản hồi không phải JSON object.")
            st.code(str(payload)[:2000])
            return
        render_outputs(payload)
        if not any(
            [
                payload.get("text") or payload.get("answer") or payload.get("answer_text") or payload.get("caption"),
                payload.get("image_base64") or payload.get("image_url") or payload.get("image"),
                payload.get("video_url") or payload.get("video") or payload.get("video_base64"),
            ]
        ):
            st.info("API trả về JSON nhưng không có trường kết quả quen thuộc. Raw:")
            st.json(payload)


if __name__ == "__main__":
    main()
