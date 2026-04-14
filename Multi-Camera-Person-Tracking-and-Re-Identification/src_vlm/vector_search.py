import os
import chromadb
from chromadb.utils import embedding_functions

class VectorSearchEngine:
    def __init__(self, db_path="../data/db"):
        print("[VectorSearch] Đang khởi tạo kết nối Vector Database (ChromaDB)...")
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Đổi sang mô hình Multilingual (Hỗ trợ Tiếng Việt trỏ tới Video Tiếng Anh/Đa ngữ)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        try:
            self.collection = self.client.get_or_create_collection(
                name="hospital_video_metadata",
                embedding_function=self.embedding_fn
            )
        except ValueError as e:
            print(f"[VectorSearch] Phát hiện xung đột (Conflict) trong cấu hình Collection. Cài đặt lại...\nChi tiết: {e}")
            self.client.delete_collection(name="hospital_video_metadata")
            self.collection = self.client.get_or_create_collection(
                name="hospital_video_metadata",
                embedding_function=self.embedding_fn
            )
        self.doc_id_counter = self.collection.count()

    def index_metadata(self, metadata_list):
        """Đưa danh sách đoạn caption (metadata) vào hệ thống tìm kiếm."""
        if not metadata_list: return
        
        docs = []
        metadatas = []
        ids = []
        for meta in metadata_list:
            docs.append(meta["caption"])
            metadatas.append({
                "video_id": meta["video_id"],
                "frame_idx": meta["frame_idx"]
            })
            ids.append(f"doc_{self.doc_id_counter}")
            self.doc_id_counter += 1
            
        self.collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[VectorSearch] Đã index {len(metadata_list)} records thành công.")

    def search_candidates(self, query_text, top_k=10):
        """Truy vấn Top K frame. Trả về Candidates hợp lệ cho User Selection."""
        print(f"[VectorSearch] Truy vấn: '{query_text}'...")
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        candidates = []
        if results['metadatas'] and len(results['metadatas']) > 0:
            for idx, meta in enumerate(results['metadatas'][0]):
                candidates.append({
                    "id": results['ids'][0][idx],
                    "video_id": meta["video_id"],
                    "frame_idx": meta["frame_idx"],
                    "score": results['distances'][0][idx] if 'distances' in results else 0
                })
        return candidates

if __name__ == "__main__":
    db = VectorSearchEngine()
