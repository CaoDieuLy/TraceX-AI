# Worklog

Ghi lại các quyết định kỹ thuật, phân công, và brainstorming của nhóm.

> Cập nhật **bất cứ khi nào** nhóm ra quyết định kỹ thuật quan trọng hoặc thay đổi hướng đi.

---

## Template

### Quyết định kỹ thuật

```markdown
### [ADR-N] Tiêu đề quyết định — DD/MM/YYYY

**Bối cảnh:** Vấn đề cần giải quyết là gì?

**Các lựa chọn đã xem xét:**
- Option A: ...
- Option B: ...

**Quyết định:** Chọn option nào và tại sao.

**Hệ quả:** Những gì bị ảnh hưởng / trade-off.
```

### Phân công

```markdown
### Sprint N — DD/MM → DD/MM/YYYY

| Task | Người làm | Deadline | Trạng thái |
| ---- | --------- | -------- | ---------- |
|      |           |          |            |
```

### Brainstorming

```markdown
### Brainstorm: [Chủ đề] — DD/MM/YYYY

**Câu hỏi:** ...

**Các ý tưởng:**
- Ý tưởng 1: ...
- Ý tưởng 2: ...

**Kết luận:** ...
```

---

## Phân công nhóm (mẫu) — đề tài Semantic Video Search (AI20K-243)

**Thành viên:** Bùi Văn Đạt, Dương Văn Hiệp, Cao Diệu Ly.

**Bối cảnh:** phần đã làm (tuần 1–2): chốt bài toán CCTV / truy vấn ngôn ngữ tự nhiên; demo **Streamlit + FastAPI** end-to-end stub; nhận diện sai hướng **LaVA**; thử **Qwen + ViT** (nặng GPU); định hướng **React + API**, **Docker/cloud**, kết quả giai đoạn gần **ảnh + timestamp** (sau có thể clip ngắn); kiến trúc **offline indexing + online search** (query parser → retrieval → ReID → thumbnail/clip).

> Bảng dưới là **mẫu phân công** — nhóm có thể đổi deadline / đổi người sau họp đồng bộ; cập nhật cột **Trạng thái** khi xong.

### Sprint 2 — 08/04 → 12/04/2026

| Task                                                                                                                            | Người làm      | Deadline | Trạng thái |
| ------------------------------------------------------------------------------------------------------------------------------- | -------------- | -------- | ---------- |
| Mở rộng **FastAPI**: contract `/infer` trả danh sách hit mẫu (`thumbnail_url` / `timestamp` / `text`)                           | Bùi Văn Đạt    | 14/04    | ⏳ Chờ      |
| Pipeline **offline** PoC: decode / frame sampling → stub detection hoặc model nhẹ → ghi **metadata + index** (file hoặc SQLite) | Dương Văn Hiệp | 19/04    | ⏳ Chờ      |
| Thu thập / chuẩn hoá **dataset** (hoặc subset CCTV), checklist nhãn tối thiểu để đánh giá retrieval                             | Cao Diệu Ly    | 12/04    | ✅ Xong     |
| Khảo sát **nhẹ hóa model** (CLIP nhỏ / giảm batch / quantize) so với Qwen+ViT; ghi 1 trang so sánh VRAM                         | Cao Diệu Ly    | 16/04    | ⏳ Chờ      |
| **React** scaffold: màn truy vấn + gọi API (axios/fetch), CORS test với FastAPI                                                 | Bùi Văn Đạt    | 20/04    | ⏳ Chờ      |
| Giữ **Streamlit** demo ổn định; đồng bộ env `VLM_API_URL` với BE Docker khi có                                                  | Cao Diệu Ly    | 18/04    | ⏳ Chờ      |
| Họp sync kiến trúc (offline vs online), cập nhật **WORKLOG** + **JOURNAL** sau sprint                                           | Cả nhóm        | 20/04    | ⏳ Chờ      |

### Sprint 1 02/04 → 07/04/2026

| Task                                                                | Người làm | Deadline | Trạng thái |
| ------------------------------------------------------------------- | --------- | -------- | ---------- |
| Chốt đề tài, pain point, phạm vi MVP (tìm kiếm video ngữ nghĩa)     | Cả nhóm   | 06/04    | ✅ Xong     |
| Research bài toán, data, công cụ AI hỗ trợ (ChatGPT, Perplexity, …) | Cả nhóm   | 07/04    | ✅ Xong     |

---


