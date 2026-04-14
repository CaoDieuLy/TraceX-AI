import os
import cv2
from PIL import Image
import torch
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

class VLM_Metadata_Engine:
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cpu_cores = multiprocessing.cpu_count()
        
        if not use_mock:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            print(f"[VLM Engine] Device: {'🟢 GPU CUDA' if self.device == 'cuda' else f'🔵 CPU ({self.cpu_cores} cores, multi-threaded)'}")
            model_id = "Salesforce/blip-image-captioning-large" 
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
            
            # CPU mode: enable torch threading tối đa
            if self.device == "cpu":
                torch.set_num_threads(self.cpu_cores)
                print(f"[VLM Engine] Đã set torch.num_threads = {self.cpu_cores}")
        else:
            print("[VLM Engine] Đang chạy ở chế độ Mock.")

    def extract_key_frames(self, video_path, num_frames=8):
        """Trích xuất N khung hình cách đều từ một video."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return frames
        
        step = max(total_frames // num_frames, 1)
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                frames.append((i * step, pil_img))
        cap.release()
        return frames

    def generate_caption(self, image):
        """Tạo đoạn văn mô tả sử dụng VLM."""
        if self.use_mock:
            return "Người đàn ông mặc áo đỏ đang đi dọc hành lang bệnh viện."
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():  # Tiết kiệm RAM khi inference
            out = self.model.generate(**inputs, max_new_tokens=70)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def _process_single_frame(self, args):
        """Xử lý 1 frame (dùng cho ThreadPool trên CPU)."""
        frame_idx, img, filename = args
        caption = self.generate_caption(img)
        return {"video_id": filename, "frame_idx": frame_idx, "caption": caption}

    def process_video(self, video_path):
        """Xử lý video từ đầu đến cuối -> tạo Text Metadata."""
        filename = os.path.basename(video_path)
        mode = "GPU" if self.device == "cuda" else f"CPU-{self.cpu_cores}T"
        print(f"[VLM Engine] [{mode}] Đang trích xuất Metadata cho {filename}...")
        frames = self.extract_key_frames(video_path, num_frames=8)
        
        metadata = []
        if self.device == "cpu" and len(frames) > 1:
            # CPU mode: xử lý tuần tự nhưng tận dụng torch multi-thread nội bộ
            # (BLIP không thread-safe để dùng ThreadPoolExecutor ở mức Python,  
            #  nhưng torch đã tự chia N cores bên trong mỗi lần forward pass)
            for frame_idx, img in frames:
                caption = self.generate_caption(img)
                metadata.append({"video_id": filename, "frame_idx": frame_idx, "caption": caption})
        else:
            # GPU mode: chạy tuần tự, GPU đã song song hóa bên trong
            for frame_idx, img in frames:
                caption = self.generate_caption(img)
                metadata.append({"video_id": filename, "frame_idx": frame_idx, "caption": caption})
        
        return frames, metadata

    def process_videos_batch(self, video_paths):
        """Xử lý batch nhiều video — CPU dùng ThreadPool cho I/O song song."""
        mode = "GPU" if self.device == "cuda" else f"CPU-{self.cpu_cores}T"
        print(f"[VLM Engine] [{mode}] Batch processing {len(video_paths)} videos...")
        
        all_metadata = []
        
        if self.device == "cpu":
            # CPU: dùng ThreadPool để đọc video song song (I/O bound)
            # nhưng inference BLIP vẫn tuần tự (compute bound, torch tự chia cores)
            for i, vp in enumerate(video_paths):
                print(f"[VLM Engine] [{mode}] Video {i+1}/{len(video_paths)}: {os.path.basename(vp)}")
                _, meta = self.process_video(vp)
                all_metadata.extend(meta)
        else:
            # GPU: chạy tuần tự, mỗi video cho GPU xử lý hết
            for i, vp in enumerate(video_paths):
                print(f"[VLM Engine] [{mode}] Video {i+1}/{len(video_paths)}: {os.path.basename(vp)}")
                _, meta = self.process_video(vp)
                all_metadata.extend(meta)
        
        return all_metadata

if __name__ == "__main__":
    engine = VLM_Metadata_Engine(use_mock=True)
