# Starter Code App

A template for building AI Agents in Python.

## Structure

```
├── src/
│   ├── agent.py        # Main agent loop
│   ├── tools.py        # Tool definitions
│   └── config.py       # Configuration
├── scripts/
│   ├── setup_hooks.sh  # One-time hook installer
│   ├── log_hook.py     # AI tool hook handler
│   └── submit_log.py   # Submits logs on git push
├── requirements.txt
├── .env.example
├── AGENTS.md           # Rules for using AI coding agents
├── JOURNAL.md          # Weekly journal — product journey & learnings
└── WORKLOG.md          # Technical decisions, task assignments, brainstorming
```

## Getting Started

### 1. Clone and setup

```bash
git clone <repo-url>
cd <repo>

# Install git pre-push hook (required, run once)
bash scripts/setup_hooks.sh
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your `ANTHROPIC_API_KEY`. The `AI_LOG_*` variables are pre-filled.

### 3. Run

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
# or: venv\Scripts\activate    # Windows

pip install -r requirements.txt
python -m src.agent
```

## Weekly Journal

Update **[JOURNAL.md](./JOURNAL.md)** at the end of every week to document your product-building journey:

- Features shipped
- AI tools used and how they helped
- Hardest problem of the week and how you solved it
- What you'd do differently
- Plan for next week

> JOURNAL.md **must be updated** before each PR. It is your learning record for the course.

## Worklog

Update **[WORKLOG.md](./WORKLOG.md)** whenever your team makes a technical decision or changes direction:

- **Technical decisions** — why did you choose this approach over alternatives?
- **Task assignments** — who does what, by when
- **Brainstorming** — options considered, pros/cons, conclusion
- **Important bugs** — root cause and fix

See each file for the format and examples.

## AI Logging

Prompts and tool calls are **automatically logged** when you use any supported AI tool (Claude Code, Cursor, Codex, Gemini, Copilot). No manual steps needed after running `setup_hooks.sh`.

See [AGENTS.md](./AGENTS.md) for details.
# Thông tin chung (Project Identity)
- Tên dự án: Search Engine Video Realtime.
- Mã đề tài: AI20K-243.
- Lĩnh vực: VLM.
- Thành viên nhóm: Bùi Văn Đạt, Dương Văn Hiệp, Cao Diệu Ly.

# Mô tả bài toán (Problem Statement & Solution)
- Vấn đề: Các hệ thống camera giám sát (CCTV) hiện nay tạo ra hàng petabyte dữ liệu mỗi ngày. Khi cần tìm một đối tượng cụ thể (ví dụ: "Người phụ nữ mặc áo đỏ đi xe máy xanh"), nhân viên an ninh phải xem lại (replay) thủ công hàng giờ đồng hồ, dẫn đến hiệu suất thấp và dễ bỏ lỡ thời điểm vàng.

- Giải pháp: Xây dựng hệ thống Semantic Video Search.Cho phép người dùng truy vấn bằng văn bản tự nhiên.
- Đối tượng người dùng (User Persona): Quản lý, bảo vệ tòa nhà,...

# Tech Stack & AI Techniques
- AI Techniques: VLM, rezo-shot, Sử dụng CLIP.
- Framework/Library: PyTorch, FastAPI (Backend), Streamlit/Next.js (Frontend).
- Database & Vector DB: Mysql

# Yêu cầu sản phẩm tối thiểu (MVP)
- Ô tìm kiếm cho phép nhập: "Người đàn ông đeo ba lô" và trả về các đoạn video tương ứng.

# Kiến trúc hệ thống (System Architecture)


# Hướng dẫn cài đặt (Installation)
