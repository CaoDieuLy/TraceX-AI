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

#### Đã làm
- Chốt thành viên nhóm, chốt đề tài.

#### Khó nhất tuần này
- Đề tài chưa xác nhận được pain point.

#### AI tool đã dùng

#### Học được
- Học cách xác định Bài toán cho AI.
#### Nếu làm lại, sẽ làm khác
- Xác định pain point mà mọi người xung quanh mà mọi người gặp phải tìm bài toán để giải quyết.

#### Kế hoạch tuần tới
- Tìm hiểu bài toán thực tế. 
  
#### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| ChatGpt | Tìm hiểu về bài toán| Phát hiện bài toán ở nhiều góc độ, hiểu về scope và pain point của nó|
| Perplexity | Research bài toán, thị trường các phương pháp xử lý bài toán | Tìm được bài toán có dữ liệu hợp lý |

---

## Tuần 02 - 11/04/2026 

**Thành viên:** Dương Văn Hiệp, Cao Diệu Ly, Bùi Văn Đạt

### Đã làm
- Thu nhỏ scope sản phẩm, tập trung vào 1 use case chính thay vì làm rộng
- Code MVP demo với các tính năng cơ bản để validate ý tưởng
- Có thể chạy demo end-to-end (dù còn đơn giản)

### Khó nhất tuần này
- Khó khăn lớn nhất là tìm dữ liệu phù hợp
- Data hiện tại không đủ chính xác hoặc không sát với bài toán → ảnh hưởng trực tiếp đến chất lượng output của sản phẩm
- Mất khá nhiều thời gian thử sai với các nguồn data khác nhau

### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| ChatGPT | Hỗ trợ code MVP, tìm tài liệu, gợi ý cách thu nhỏ scope | Hoàn thành được bản demo tối thiểu để test ý tưởng |

### Học được
- Quan trọng nhất là xác định đúng **pain point**, không cố giải quyết nhiều vấn đề cùng lúc
- Làm MVP không phải là làm ít tính năng, mà là làm đúng thứ quan trọng nhất trước
- Data ảnh hưởng rất lớn đến chất lượng sản phẩm, đặc biệt với các bài toán liên quan đến AI

### Nếu làm lại, sẽ làm khác
- Dành thời gian validate và tìm data trước khi bắt đầu code
- Chủ động lên plan rõ ràng hơn để tránh làm lan man
- Có thể bắt đầu bằng mock data để test flow trước, rồi mới tìm data thật

### Kế hoạch tuần tới
- Hoàn thiện MVP (ổn định hơn, xử lý edge case cơ bản)
- Tìm hoặc xây dựng bộ data tốt hơn phục vụ bài toán
- Bắt đầu test với một số case thực tế