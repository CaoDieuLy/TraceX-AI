# Retail Video Semantic Search (RVSS)

Hệ thống tìm kiếm video cửa hàng bán lẻ bằng ngôn ngữ tự nhiên, hỗ trợ **nhiều dataset**:
- Surveillance for Retail Stores (Kaggle)
- PhysicalAI-SmartSpaces (NVIDIA)
- RetailAction (FiftyOne/HuggingFace)
- MIMEX (fine-grained product classification)

## Tính năng
- ✅ Tìm kiếm zero-shot, không cần huấn luyện
- ✅ Hỗ trợ nhiều dataset, tự động convert về cùng định dạng
- ✅ Tìm kiếm AND (nhiều điều kiện) qua code
- ✅ Lọc theo cửa hàng, theo dataset nguồn
- ✅ Thống kê kết quả theo store và dataset
- ✅ Giao diện Streamlit hiển thị thumbnail + video link

## Cài đặt

```bash
# Clone repository
git clone <repo>
cd retail_video_search

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Cài dependencies
pip install -r requirements.txt
```

## Khởi động Qdrant
```bash
docker-compose up -d
```

## Tải và xử lý dữ liệu

Cách 1: Tải dataset tự động
```bash
python scripts/download_datasets.py --all
```

Cách 2: Chạy toàn bộ pipeline
```bash
python scripts/run_pipeline.py --download
```

Cách 3: Chỉ convert và index (nếu đã có video)
```bash
python src/pipeline/index_videos.py
```

## Chạy giao diện tìm kiếm
```bash
streamlit run src/app.py
```
Mở trình duyệt tại http://localhost:8501

## Cấu trúc dữ liệu đầu vào
Sau khi convert, video được lưu theo cấu trúc:
```
data/videos/
  {store_id}/
    {camera_id}/
      {date}.mp4
```

Ví dụ:
```
data/videos/
  store_001/
    counter/2025-04-01.mp4
    seating/2025-04-01.mp4
  warehouse_000/
    camera_001/2025-04-01.mp4
  store_mimex/
    product_chocolate/chocolate.mp4
```

## Truy vấn mẫu
- "no staff at counter" - Quầy không có nhân viên
- "many customers" - Đông khách
- "person wearing red shirt" - Người mặc áo đỏ
- "empty store" - Cửa hàng trống
- "product on shelf" - Sản phẩm trên kệ

## Kiểm thử
```bash
pytest tests/
```

## Giấy phép
MIT

## 🚀 Hướng dẫn chạy nhanh

1. **Clone project** và tạo cấu trúc thư mục
2. **Cài dependencies**: `pip install -r requirements.txt`
3. **Khởi động Qdrant**: `docker-compose up -d`
4. **Tải dataset** (tùy chọn): `python scripts/download_datasets.py --all`
5. **Chạy pipeline**: `python scripts/run_pipeline.py`
6. **Mở trình duyệt** tại `http://localhost:8501`

## 📊 Dataset support matrix

| Dataset | Source | Format | Auto-download | Converted to video |
|---------|--------|--------|---------------|-------------------|
| Surveillance for Retail Stores | Kaggle | Frames (jpg) | ✅ (via kagglehub) | ✅ |
| PhysicalAI-SmartSpaces | HuggingFace | MP4 | ✅ (via huggingface_hub) | ✅ |
| RetailAction | FiftyOne | MP4 | ✅ (via fiftyone) | ✅ |
| MIMEX | Paper | Images | ⚠️ (manual) | ✅ (slideshow) |

## 💡 Lưu ý

- **Không cần GPU**: CLIP ViT-B/32 chạy tốt trên CPU
- **Dung lượng**: PhysicalAI-SmartSpaces có thể lên đến 3TB, chỉ tải subset nếu cần
- **Tìm kiếm AND**: Hỗ trợ tìm giao của nhiều điều kiện (ví dụ: "no staff" AND "many customers")
- **Mở rộng**: Có thể thêm dataset mới bằng cách implement thêm method trong `DatasetConverter`
