import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.core.clip_encoder import CLIPEncoder
from src.core.vector_store import QdrantStore
from typing import List, Dict

class SearchEngine:
    def __init__(self):
        self.encoder = CLIPEncoder()
        self.store = QdrantStore()
    
    def search(self, query: str, top_k: int = 20, 
               store_filter: str = None, dataset_filter: str = None) -> List[Dict]:
        """Single query search."""
        vec = self.encoder.encode_text(query)
        return self.store.search(vec.tolist(), top_k=top_k, 
                                  store_filter=store_filter, dataset_filter=dataset_filter)
    
    def multi_and_search(self, queries: List[str], top_k_per_query: int = 30, 
                         final_k: int = 20, store_filter: str = None) -> List[Dict]:
        """Intersection of multiple queries (AND logic)."""
        if not queries:
            return []
        all_results = []
        for q in queries:
            res = self.search(q, top_k=top_k_per_query, store_filter=store_filter)
            all_results.append({r["payload"]["video_path"] + str(r["payload"]["timestamp"]): r for r in res})
        common_keys = set(all_results[0].keys())
        for d in all_results[1:]:
            common_keys &= set(d.keys())
        combined = []
        for key in common_keys:
            scores = [all_results[i][key]["score"] for i in range(len(queries))]
            min_score = min(scores)
            combined.append((min_score, all_results[0][key]["payload"]))
        combined.sort(key=lambda x: x[0], reverse=True)
        return [{"score": s, "payload": p} for s, p in combined[:final_k]]
    
    def group_by_store(self, results: List[Dict]) -> dict:
        """Count results per store."""
        from collections import defaultdict
        counts = defaultdict(int)
        for r in results:
            store = r["payload"]["store_id"]
            counts[store] += 1
        return dict(counts)
    
    def group_by_dataset(self, results: List[Dict]) -> dict:
        """Count results per dataset source."""
        from collections import defaultdict
        counts = defaultdict(int)
        for r in results:
            dataset = r["payload"].get("dataset", "unknown")
            counts[dataset] += 1
        return dict(counts)
