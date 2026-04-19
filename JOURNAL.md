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


---

## Tuần 03 - 19/04/2026

**Thành viên:** Dương Văn Hiệp, Cao Diệu Ly, Bùi Văn Đạt

### Đã làm
- **Dương Văn Hiệp:** thử nghiệm thêm các đề xuất CPU-based theo góp ý office hour; tiếp tục tối ưu lưu trữ; tìm kiếm và thử các mô hình detection / embedding khác nhau để trích xuất metadata tốt hơn; xây dựng metric đánh giá, tối ưu độ chính xác; unit test và nối các luồng xử lý để chạy được pipeline.
- **Bùi Văn Đạt:** xây dựng frontend bằng React; nghiên cứu hướng giải quyết mới; tìm nguồn dataset; nghiên cứu cách lấy, lưu dataset và mô phỏng streaming data; thiết kế luồng kết nối backend, frontend và model.
- **Cao Diệu Ly:** phát triển lại architecture theo user và workflow mới; research và điều chỉnh architecture theo scope mới; triển khai thử nghiệm; research VLM hiệu quả hơn; đánh giá chất lượng tracklet sau tracking; tìm kiếm, setup data và dựng DB; lựa chọn model tracking tối ưu theo IDF1.
- Nhóm chuyển trọng tâm từ prototype rời rạc sang pipeline rõ hơn: data / DB → model detection, tracking, embedding → backend API → frontend React.
- Bắt đầu đánh giá hệ thống bằng metric cụ thể hơn thay vì chỉ nhìn output demo; đặc biệt quan tâm metadata, tracklet, IDF1 và khả năng chạy trên tài nguyên hạn chế.

### Chi tiết daily standup
- **13/04/2026:** Hiệp thử đề xuất CPU-based theo góp ý office hour; Đạt xây dựng FE bằng React; Ly phát triển architecture theo user và workflow mới.
- **14/04/2026:** Hiệp điều chỉnh theo hướng dẫn mentor về tối ưu lưu trữ; Đạt tìm hướng giải quyết bài toán mới; Ly research và thay đổi architecture theo scope mới.
- **15/04/2026:** Hiệp tìm mô hình embedding để trích xuất metadata tốt hơn; Đạt nghiên cứu nguồn dataset; Ly triển khai thử nghiệm.
- **16/04/2026:** Hiệp xây metric đánh giá và tối ưu độ chính xác; Ly research VLM hiệu quả hơn và đánh giá tracklet sau tracking; Đạt nghiên cứu cách lấy / lưu dataset, mô phỏng streaming data.
- **17/04/2026:** Hiệp thử nghiệm các model detection, embedding khác nhau; Ly đưa ra phương án cải thiện architecture mới; Đạt tiếp tục mô phỏng theo hướng streaming data.
- **18/04/2026:** Hiệp tiếp tục thử model embedding tốt hơn; Đạt thiết kế luồng BE, FE với model; Ly tìm kiếm, setup data và viết lại architecture.
- **19/04/2026:** Hiệp unit test và nối các luồng để chạy; Đạt tiếp tục thiết kế luồng FE / BE; Ly tìm và dựng DB, lựa chọn model track tối ưu IDF1.

### Khó nhất tuần này
- Vừa phải đổi kiến trúc theo scope mới, vừa phải giữ tiến độ triển khai thử nghiệm.
- Dataset và cách lưu / mô phỏng streaming data vẫn là phần tốn thời gian vì ảnh hưởng trực tiếp tới pipeline backend, frontend và model.
- Cần cân bằng giữa độ chính xác của detection / embedding / tracking và khả năng chạy trên CPU hoặc tài nguyên GPU hạn chế.

### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| ChatGPT | Hỗ trợ brainstorm architecture, gợi ý hướng model / metric, rà soát pipeline và viết tài liệu | Có thêm hướng điều chỉnh architecture, metric đánh giá và cách diễn đạt journal rõ hơn |

### Học được
- Pipeline video search cần được thiết kế theo luồng dữ liệu trước: ingest / lưu trữ / tracking / embedding / search / hiển thị, không chỉ chọn model trước.
- Metric như IDF1, chất lượng tracklet và độ chính xác metadata giúp nhóm đánh giá thực tế hơn so với chỉ xem kết quả demo.
- Frontend, backend và model cần thống nhất contract sớm để tránh mỗi phần phát triển theo một giả định khác nhau.

### Nếu làm lại, sẽ làm khác
- Chốt format dataset, metadata và API contract sớm hơn trước khi thử nhiều model.
- Tách rõ thử nghiệm model, mô phỏng streaming data và phần giao diện để dễ đo tiến độ từng mảng.

### Kế hoạch tuần tới
- Hoàn thiện pipeline chạy được ổn định hơn từ data / DB đến model, backend và frontend.
- Tiếp tục so sánh model detection, embedding và tracking theo metric đã chọn.
- Chuẩn hoá dataset, metadata và API contract để phục vụ demo end-to-end.
