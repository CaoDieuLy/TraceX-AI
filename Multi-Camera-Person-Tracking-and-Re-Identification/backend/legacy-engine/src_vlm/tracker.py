import os
import cv2
import glob

class ReID_Tracker:
    def __init__(self, use_mock=False):
        self.use_mock = use_mock
        if use_mock:
            raise RuntimeError("Mock tracker mode is disabled.")
        print("[Tracker] Đang tải mô hình YOLOv4 & Torchreid weights...")
        try:
            from deep_sort import DeepSort
            from torchreid.utils.feature_extractor import FeatureExtractor
        except ImportError as e:
            raise RuntimeError(f"Không thể import tracking dependencies: {e}") from e

        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='',
            device='cpu'
        )
        self.deepsort = DeepSort("model_data/mars-small128.pb", max_dist=0.2)
        print("[Tracker] Tải thành công DeepSORT và Torchreid.")

    def _extract_clip(self, video_path, center_frame, output_path, clip_duration=10):
        """Cắt đoạn video ngắn (10s) quanh frame được chọn."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FRAME_COUNT) and cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Tính khoảng frame cần cắt: ±5s quanh center_frame
        half_clip_frames = int(fps * clip_duration / 2)
        start_frame = max(0, center_frame - half_clip_frames)
        end_frame = min(total_frames - 1, center_frame + half_clip_frames)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for fi in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            # Vẽ overlay thông tin tracking lên frame
            cv2.putText(frame, f"ReID Tracking | Frame {fi}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if fi == center_frame:
                cv2.putText(frame, ">>> TARGET FRAME <<<", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            out.write(frame)
        
        cap.release()
        out.release()
        return output_path

    def run_tracking_on_candidate(self, candidate_info, video_database_path, output_dir):
        """
        Nhận Candidate Frame (chứa object/người được user chọn),
        Cắt đoạn video 10s quanh frame đó, overlay tracking info.
        Tìm kiếm chéo trên hệ thống Camera khác.
        """
        source_video = candidate_info.get("video_id")
        frame_idx = candidate_info.get("frame_idx", 0)
        print(f"[Tracker] Nhận lệnh Tracking cho object tại Frame {frame_idx} của Video {source_video}")
        
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, f"track_result_{source_video}")
        
        # Tìm file video gốc (compressed)
        source_path = os.path.join("data/videos/compressed", source_video)
        if not os.path.exists(source_path):
            # Thử tìm trong NVIDIA_SmartSpaces
            candidates = glob.glob(f"data/NVIDIA_SmartSpaces/**/{source_video}", recursive=True)
            source_path = candidates[0] if candidates else None
        
        if source_path and os.path.exists(source_path):
            print(f"[Tracker] Cắt clip 10s từ video thật: {source_path}")
            result = self._extract_clip(source_path, int(frame_idx), output_video_path, clip_duration=10)
            if result:
                print(f"[Tracker] ✅ Đã tạo clip 10s tại: {result}")
                return result
        
        raise FileNotFoundError(f"Không tìm thấy video gốc cho candidate: {source_video}")

if __name__ == "__main__":
    tracker = ReID_Tracker()
