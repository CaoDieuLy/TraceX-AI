"""
Hướng dẫn tải dataset mẫu từ Kaggle.
Không tự động tải vì yêu cầu đăng nhập.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.config import config

def print_instructions():
    print("=" * 60)
    print("Hướng dẫn tải dataset mẫu cho Retail Video Semantic Search")
    print("=" * 60)
    print("1. Truy cập Kaggle: https://www.kaggle.com/datasets")
    print("2. Tải dataset 'Surveillance for Retail Stores' (competition data)")
    print("   - Giải nén vào thư mục tạm")
    print("3. Tạo cấu trúc thư mục theo hướng dẫn:")
    print(f"   {config.video_root}/")
    print("     store_001/counter/2025-04-01.mp4  (từ sequence 1)")
    print("     store_001/seating/2025-04-01.mp4  (từ sequence 2)")
    print("4. Hoặc sử dụng video tự quay / tải từ Pexels")
    print("5. Sau khi có video, chạy: python src/pipeline/index_videos.py")
    print("6. Chạy frontend: streamlit run src/app.py")

if __name__ == "__main__":
    print_instructions()
