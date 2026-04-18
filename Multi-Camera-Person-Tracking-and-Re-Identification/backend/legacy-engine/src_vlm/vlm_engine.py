import os
import cv2
from PIL import Image
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
except Exception:  # pragma: no cover - optional dependency in service containers
    torch = None

class VLM_Metadata_Engine:
    def __init__(self, use_mock=False):
        self.use_mock = use_mock
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.cpu_cores = multiprocessing.cpu_count()
        self.model_id = "Salesforce/blip-image-captioning-large"

        if use_mock:
            raise RuntimeError("Mock VLM mode is disabled for production ingestion.")
        if torch is None:
            raise RuntimeError("Torch is required for VLM metadata generation.")
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except Exception as exc:
            raise RuntimeError("Transformers is required for VLM metadata generation.") from exc
        print(f"[VLM Engine] Device: {'🟢 GPU CUDA' if self.device == 'cuda' else f'🔵 CPU ({self.cpu_cores} cores, multi-threaded)'}")
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

        if self.device == "cpu":
            torch.set_num_threads(self.cpu_cores)
            print(f"[VLM Engine] torch.num_threads = {self.cpu_cores}")

    def _extract_single_frame(self, args):
        """Worker: trích 1 frame từ video (chạy trong ThreadPool)."""
        video_path, frame_pos = args
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return (frame_pos, Image.fromarray(frame_rgb))
        return None

    def extract_key_frames(self, video_path, num_frames=8):
        """Trích xuất N khung hình — song song bằng ThreadPool (I/O bound)."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0:
            return []
        
        step = max(total_frames // num_frames, 1)
        frame_positions = [i * step for i in range(num_frames)]
        
        # Song song trích frame bằng ThreadPool
        with ThreadPoolExecutor(max_workers=min(num_frames, self.cpu_cores)) as pool:
            results = list(pool.map(
                self._extract_single_frame,
                [(video_path, pos) for pos in frame_positions]
            ))
        
        return [r for r in results if r is not None]

    def generate_captions_batch(self, images):
        """BATCH inference: gom tất cả ảnh, forward 1 lần duy nhất."""
        # Processor xử lý batch ảnh cùng lúc
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=70)
        
        captions = self.processor.batch_decode(out, skip_special_tokens=True)
        return captions

    def generate_caption(self, image):
        """Single image caption (backward compatible)."""
        return self.generate_captions_batch([image])[0]
    
    def process_video(self, video_path):
        """Xử lý 1 video end-to-end — song song trích frame + batch VLM."""
        filename = os.path.basename(video_path)
        mode = "GPU" if self.device == "cuda" else f"CPU-{self.cpu_cores}T"
        print(f"[VLM Engine] [{mode}] Processing {filename}...")
        
        # Song song: trích 8 frames cùng lúc
        frames = self.extract_key_frames(video_path, num_frames=8)
        if not frames:
            return [], []
        
        # Batch: gom 8 ảnh → 1 lần forward pass duy nhất
        frame_indices = [f[0] for f in frames]
        frame_images = [f[1] for f in frames]
        
        captions = self.generate_captions_batch(frame_images)
        
        metadata = []
        for idx, caption in zip(frame_indices, captions):
            metadata.append({
                "video_id": filename,
                "frame_idx": idx,
                "caption": caption
            })
        
        return frames, metadata

    def process_videos_batch(self, video_paths, max_parallel_videos=2):
        """Batch xử lý nhiều video — song song trích frame, batch VLM."""
        mode = "GPU" if self.device == "cuda" else f"CPU-{self.cpu_cores}T"
        print(f"[VLM Engine] [{mode}] Batch: {len(video_paths)} videos, parallel={max_parallel_videos}")
        
        all_metadata = []
        
        # Xử lý song song nhiều video bằng ThreadPool
        # (mỗi video đã tự song song trích frame + batch inference bên trong)
        with ThreadPoolExecutor(max_workers=max_parallel_videos) as pool:
            futures = {pool.submit(self.process_video, vp): vp for vp in video_paths}
            for i, future in enumerate(futures):
                vp = futures[future]
                _, meta = future.result()
                all_metadata.extend(meta)
                print(f"[VLM Engine] [{mode}] Done {i+1}/{len(video_paths)}: {os.path.basename(vp)}")
        
        return all_metadata

if __name__ == "__main__":
    engine = VLM_Metadata_Engine(use_mock=False)
