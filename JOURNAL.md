# Weekly Journal

Ghi lại hành trình xây dựng sản phẩm mỗi tuần — những gì đã làm, học được gì, AI giúp như thế nào.

> **Cập nhật mỗi cuối tuần** (trước khi tạo PR). Không cần dài, chỉ cần thật.

---


## Template

```markdown
## Tuần N — DD/MM/YYYY

### Đã làm
-

### Khó nhất tuần này
-

### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| Claude Code | | |

### Học được
-

### Nếu làm lại, sẽ làm khác
-

### Kế hoạch tuần tới
-
```

---

## Ví dụ

### Tuần 1 — 02/04/2026

**Thành viên:** Dương Văn Hiệp, Cao Diệu Ly, Bùi Văn Đạt

### Quá trình chọn đề tài
- **Khởi đầu:** Nhóm brainstorm theo hướng “AI + thị giác máy tính”, liệt kê vài hướng (giám sát an ninh, tìm kiếm trong video, hỗ trợ người khiếm thị, v.v.).
- **Tiêu chí gạn lọc:** (1) có pain point rõ trong đời sống / doanh nghiệp, (2) có thể làm MVP với dữ liệu tìm được hoặc thu thập được, (3) không vượt quá năng lực GPU và thời gian môn học.
- **Chốt hướng:** Tập trung **Search Engine / truy vấn video bằng ngôn ngữ tự nhiên** (gần với camera giám sát, người dùng mô tả đối tượng cần tìm(người) thay vì xem lại cả giờ footage). Đề tài gắn mã AI20K-243, định hướng VLM + embedding (ví dụ CLIP) trong tài liệu đề xuất ban đầu.


### Đã làm
- Chốt thành viên nhóm, chốt đề tài và phạm vi bài toán (tìm kiếm ngữ nghĩa trong video).
- Học cách xác định **bài toán AI** (problem framing), phân biệt demo “đẹp” và bài toán deploy được.

### Khó nhất tuần này
- Đề tài lúc đầu **chưa khóa chặt pain point**.
- Chưa có data thật đủ tốt để validate ngay từ đầu.
### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| ChatGPT | Tìm hiểu bài toán, nhiều góc nhìn, scope và pain point | Hiểu rộng hơn về không gian bài toán và rủi ro chọn sai model |
| Perplexity | Research bài toán, phương pháp và thị trường | Tìm được hướng bài toán có dữ liệu khả thi hơn |

### Học được
- Học cách xác định bài toán cho AI một cách có kiểm chứng (pain point, metric, dữ liệu).

### Nếu làm lại, sẽ làm khác
- Xác định pain point từ người dùng / kịch bản thực tế, không chỉ từ paper.

### Kế hoạch tuần tới (lúc đó)
- Tìm hiểu sâu bài toán thực tế, thu thập / mock data, thử pipeline gần đúng với CCTV hơn.

---

## Tuần 02 - 12/04/2026 

**Thành viên:** Dương Văn Hiệp, Cao Diệu Ly, Bùi Văn Đạt

### Đã làm
- Thu nhỏ scope sản phẩm, tập trung **một use case chính** (truy vấn bằng text, trả về đoạn video / kết quả liên quan) thay vì làm rộng.
- Code MVP demo với các tính năng cơ bản để validate ý tưởng; có thể chạy **end-to-end** ở mức đơn giản.
- **Thử nghiệm model:** hướng hiện tại dùng **Qwen** (xử lý ngôn ngữ / đa phương thức tùy cấu hình) kết hợp **ViT** (Vision Transformer) cho nhánh thị giác. Nhận thấy bộ đôi này **rất nặng**, chiếm nhiều **VRAM/GPU**, khó chạy ổn định trên máy cấu hình vừa phải → ảnh hưởng tốc độ thử nghiệm và khả năng demo liên tục.
- **Định hướng stack sản phẩm (web):** nhóm **chọn frontend React** để xây giao diện; **backend** xây dựng dạng **API** (phục vụ infer, dữ liệu, session người dùng) — tách bạch khỏi phần model để sau này thay đổi backbone hoặc tối ưu GPU dễ hơn. (Chi tiết triển khai có thể điều chỉnh theo repo và deadline.)

### Khó nhất tuần này
- **Dữ liệu:** tìm dataset / footage phù hợp bài toán CCTV-semantic search; data hiện có đôi khi **không đủ sát** hoặc **không đủ nhãn** → chất lượng output dao động, khó đánh giá đúng sai của model.
- **Di sản hướng LaVA / pipeline cũ:** vẫn phải dọn chỗ không khớp use case, đồng thời giữ tiến độ MVP.
- **Tài nguyên GPU:** Qwen + ViT khiến vòng lặp thử (train / infer) chậm và dễ OOM nếu không giảm batch hoặc không quantize / dùng model nhỏ hơn.

### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| ChatGPT | Hỗ trợ code MVP, tài liệu, thu nhỏ scope | Có bản demo tối thiểu để test ý tưởng |

### Sai lầm kỹ thuật ban đầu: chọn LaVA (Large Language and Vision Assistant)
- **Lý do hồi đó:** LaVA nổi, dễ tìm tutorial “hỏi đáp trên ảnh”, demo trông ấn tượng nên nhóm tạm chọn làm “model chính” mà chưa khóa kỹ **bài toán đích**.
- **Vấn đề:** LaVA mạnh ở kiểu **VQA / mô tả ảnh / hội thoại đa phương thức**, không phải pipeline tối ưu cho **truy vấn ngữ nghĩa trên video dài, cắt clip, index theo thời gian** như CCTV. Nhóm lỡ **lệch đối tượng xử lý**: code và thử nghiệm xoay quanh format “một ảnh + một câu hỏi”, trong khi sản phẩm cần “video + truy vấn + trả về đoạn video / khung thời gian / embedding”.
- **Hệ quả tới hiện tại:** Phải **refactor lại kiến trúc** (tách bước embed, index, search), một phần thời gian đã đổ vào hướng không khớp use case; vẫn còn **nợ kỹ thuật** (chuẩn hoá dữ liệu, format output, đồng bộ giữa prototype cũ và MVP mới). Đây là một trong những “lỗi gốc” khiến team vừa làm MVP vừa phải sửa lại tư duy model.


### Học được
- Xác định đúng **pain point** quan trọng hơn nhồi nhiều tính năng; MVP là làm **đúng lõi** trước.
- **Data** quyết định lớn chất lượng, nhất là bài toán video + VLM.
- Cần cân nhắc **ngân sách GPU** ngay từ khi chọn backbone, không chỉ nhìn benchmark.

### Nếu làm lại, sẽ làm khác
- Validate và **mock pipeline + data** trước khi khóa bộ model nặng.
- Lên plan rõ **bài toán → metric → kích thước model tối đa** trên GPU đang có.

### Kế hoạch tuần tới
- Hoàn thiện MVP (ổn định hơn, xử lý edge case cơ bản).
- Cải thiện hoặc thu thập bộ data sát bài toán hơn.
- Cân nhắc **nhẹ hóa model** (distill, model nhỏ hơn, hoặc tách bước embed/search) để giảm áp lực GPU; tiếp tục chỉnh **frontend React + backend API** cho luồng người dùng thật.


  
