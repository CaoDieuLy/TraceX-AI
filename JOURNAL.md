# Weekly Journal
init commit
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

### Tuần 1 — 31/03/2026

**Thành viên:** Nguyễn Văn A, Trần Thị B, Lê Văn C

#### Đã làm
- Setup project TypeScript + cấu hình `.env`
- Xây dựng agent loop cơ bản: nhận input → gọi Claude API → in output
- Thêm tool `search_web` đầu tiên (dùng Brave Search API)
- Viết README cho repo nhóm

#### Khó nhất tuần này
- Tool call response của Claude trả về sai format — mất 2 tiếng debug mới phát hiện ra thiếu `"type": "tool_result"` trong message history.
- Lần đầu dùng TypeScript nên type error khá nhiều, phải học cách dùng `as` và generic.

#### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| Claude Code | Giải thích Anthropic tool use API, debug message format | Giải quyết được bug trong 15 phút |
| Cursor | Autocomplete TypeScript types | Tiết kiệm khoảng 30% thời gian gõ |

#### Học được
- Tool use trong Claude hoạt động theo vòng lặp: model gọi tool → app trả kết quả → model tiếp tục. Cần giữ đúng message history.
- `zod` rất hữu ích để validate tool input schema.
- Nên đặt timeout cho API call ngay từ đầu, không để sau mới thêm.

#### Nếu làm lại, sẽ làm khác
- Setup TypeScript strict mode ngay từ đầu thay vì thêm sau (refactor mệt hơn).
- Viết unit test cho `parseToolCall()` trước khi tích hợp vào agent loop.

#### Kế hoạch tuần tới
- Thêm tool `read_file` và `write_file`
- Implement memory: lưu conversation history vào file JSON
- Thử chạy agent giải 1 bài tập thực tế

---

### Tuần 2 — 08/04/2026

**Thành viên:** Bùi Văn Đạt, Dương Văn Hiệp, Cao Diệu Ly

#### Đã làm
- Xác định scope cho bài toán.
- Chốt xử dụng kĩ thuật Zero-shot learning.
- Chốt chặt phạm vi đề bài và tính năng sẽ làm 

#### Khó nhất tuần này
- Việc chọn bối cảnh nào để phạm vi bài toán hợp lý và khả thi hơn?

#### AI tool đã dùng
| Tool | Dùng để làm gì | Kết quả |
|---|---|---|
| ChatGpt | Tìm hiểu về bài toán| Phát hiện bài toán ở nhiều góc độ, hiểu về scope và pain point của nó|

#### Học được
- Học được các kiến trúc AI, Prompt Engineering & Tool Calling
- Pain point cần xác thực với nhu cầu thực tế không nên áp đặt thiên ý cá nhân
#### Nếu làm lại, sẽ làm khác
- Dùng AI nhiều hơn để xác nhận pain point. 

#### Kế hoạch tuần này 
- Xây dựng bài toán.
- Xác nhận tool, công cụ làm bài toán 
- Tìm data thị trước
  
