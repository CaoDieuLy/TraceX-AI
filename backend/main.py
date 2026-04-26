"""
API tối thiểu cho Streamlit frontend: nhận text/ảnh, trả JSON (text, image_*, video_*).
Logic VLM/AI: thay nội dung hàm build_infer_result() bằng gọi service của bạn.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VLM Video API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def build_infer_result(
    query: str | None,
    image_bytes: bytes | None,
    image_filename: str | None,
    image_content_type: str | None,
) -> dict[str, Any]:
    """
    TODO: gọi pipeline AI / service đã triển khai ở chỗ khác.
    Hiện trả stub để FE và contract API ổn định.
    """
    parts: list[str] = []
    if query:
        parts.append(f"Truy vấn text ({len(query)} ký tự) đã nhận.")
    if image_bytes:
        parts.append(
            f"Ảnh `{image_filename or 'upload'}` ({len(image_bytes)} bytes, {image_content_type or 'unknown'}) đã nhận."
        )
    text = " ".join(parts) if parts else "Không có đầu vào."
    text += " (Stub: chưa gắn module phân tích — trả về JSON mẫu.)"
    return {
        "text": text,
        # Khi có kết quả thật, điền một hoặc nhiều trường sau (URL phải là http/https cho Streamlit):
        # "image_url": "...",
        # "video_url": "...",
        # "image_base64": "...",
        # "video_base64": "...",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/infer")
async def infer(
    query: str | None = Form(None),
    image: UploadFile | None = File(None),
) -> dict[str, Any]:
    q = (query or "").strip()
    image_bytes: bytes | None = None
    fname: str | None = None
    ctype: str | None = None
    if image is not None:
        image_bytes = await image.read()
        fname = image.filename
        ctype = image.content_type
        if not image_bytes:
            image_bytes = None

    if not q and not image_bytes:
        raise HTTPException(status_code=400, detail="Cần ít nhất query hoặc file image.")

    return await build_infer_result(q or None, image_bytes, fname, ctype)
