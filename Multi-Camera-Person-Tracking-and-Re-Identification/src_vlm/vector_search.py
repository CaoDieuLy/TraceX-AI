from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import chromadb

from .hospital_pipeline import DEFAULT_TEXT_MODEL, METADATA_DIR


@lru_cache(maxsize=2)
def _load_sentence_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class VectorSearchEngine:
    def __init__(self, db_path="../data/db", model_name: str = DEFAULT_TEXT_MODEL):
        print("[VectorSearch] Đang khởi tạo kết nối Vector Database (ChromaDB)...")
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection_name = "hospital_video_metadata"
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            return self.client.get_or_create_collection(name=self.collection_name)
        except ValueError as exc:
            print(f"[VectorSearch] Conflict collection, reset lại. Chi tiết: {exc}")
            self.client.delete_collection(name=self.collection_name)
            return self.client.get_or_create_collection(name=self.collection_name)

    def reset_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = _load_sentence_model(self.model_name)
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        return embeddings.astype("float32").tolist()

    def index_metadata(self, metadata_list, reset: bool = False):
        """Index person-track metadata thay vì caption theo frame."""
        if not metadata_list:
            return {"indexed_count": 0, "collection_count": self.collection.count()}
        if reset:
            self.reset_collection()

        documents = []
        embeddings = []
        metadatas = []
        ids = []
        accepted_items = []

        pending_vectors = []
        for meta in metadata_list:
            text = str(meta.get("search_text") or meta.get("caption") or "").strip()
            if not text:
                continue
            candidate_id = str(meta.get("candidate_id") or meta.get("id") or f"candidate_{len(ids)}")
            metadata_path = str(meta.get("metadata_path") or "")
            vector = meta.get("candidate_vector") or []
            ids.append(candidate_id)
            documents.append(text)
            metadatas.append(
                {
                    "candidate_id": candidate_id,
                    "video_id": str(meta.get("video_id") or ""),
                    "camera_id": str(meta.get("camera_id") or ""),
                    "track_id": str(meta.get("track_id") or ""),
                    "human_key": str(meta.get("human_key") or candidate_id),
                    "frame_idx": int(meta.get("frame_idx") or 0),
                    "metadata_path": metadata_path,
                }
            )
            pending_vectors.append(vector)
            accepted_items.append(meta)

        missing_indexes = [index for index, vector in enumerate(pending_vectors) if not vector]
        if missing_indexes:
            generated = self._embed_texts([documents[index] for index in missing_indexes])
            for missing_index, vector in zip(missing_indexes, generated):
                pending_vectors[missing_index] = vector
                accepted_items[missing_index]["candidate_vector"] = vector
                accepted_items[missing_index]["vector_model"] = self.model_name

        embeddings = pending_vectors
        if not ids:
            return {"indexed_count": 0, "collection_count": self.collection.count()}

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        print(f"[VectorSearch] Đã index {len(ids)} candidates thành công.")
        return {"indexed_count": len(ids), "collection_count": self.collection.count()}

    def rebuild_from_metadata_dir(self, metadata_dir: str | Path | None = None):
        metadata_root = Path(metadata_dir or METADATA_DIR)
        metadata_root.mkdir(parents=True, exist_ok=True)
        self.reset_collection()

        people_to_index = []
        updated_payloads: list[tuple[Path, dict]] = []
        for metadata_path in sorted(metadata_root.glob("*.json")):
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            people = payload.get("people") or []
            if not isinstance(people, list):
                continue
            for person in people:
                if not isinstance(person, dict):
                    continue
                person["metadata_path"] = str(metadata_path)
                people_to_index.append(person)
            updated_payloads.append((metadata_path, payload))

        result = self.index_metadata(people_to_index, reset=False)

        by_path: dict[str, dict[str, dict]] = {}
        for person in people_to_index:
            metadata_path = str(person.get("metadata_path") or "")
            candidate_id = str(person.get("candidate_id") or "")
            if metadata_path and candidate_id:
                by_path.setdefault(metadata_path, {})[candidate_id] = person

        for metadata_path, payload in updated_payloads:
            replacements = by_path.get(str(metadata_path), {})
            updated_people = []
            for person in payload.get("people") or []:
                if not isinstance(person, dict):
                    continue
                candidate_id = str(person.get("candidate_id") or "")
                updated_people.append(replacements.get(candidate_id, person))
            payload["people"] = updated_people
            metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "indexed_count": result["indexed_count"],
            "collection_count": result["collection_count"],
            "metadata_files": len(updated_payloads),
            "text_model": self.model_name,
        }

    def _read_candidate_from_file(self, metadata_path: str, candidate_id: str) -> dict | None:
        if not metadata_path:
            return None
        path = Path(metadata_path)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        for person in payload.get("people", []):
            if isinstance(person, dict) and str(person.get("candidate_id")) == candidate_id:
                return person
        return None

    def search_candidates(self, query_text, top_k=10):
        """Trả về top-K human candidates, dedupe theo human_key thay vì frame."""
        query = str(query_text or "").strip()
        if not query:
            return []
        if self.collection.count() == 0:
            return []

        query_embedding = self._embed_texts([query])[0]
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(int(top_k) * 8, int(top_k)),
            include=["metadatas", "distances", "documents"],
        )

        metadatas = raw.get("metadatas", [[]])
        distances = raw.get("distances", [[]])
        documents = raw.get("documents", [[]])
        ids = raw.get("ids", [[]])
        candidates = []
        seen_humans: set[str] = set()

        for index, meta in enumerate(metadatas[0] if metadatas else []):
            candidate_id = str(meta.get("candidate_id") or ids[0][index])
            full_candidate = self._read_candidate_from_file(str(meta.get("metadata_path") or ""), candidate_id)
            if full_candidate is None:
                full_candidate = {
                    "candidate_id": candidate_id,
                    "video_id": meta.get("video_id"),
                    "camera_id": meta.get("camera_id"),
                    "track_id": meta.get("track_id"),
                    "frame_idx": meta.get("frame_idx"),
                    "search_text": documents[0][index] if documents and documents[0] else "",
                }
            human_key = str(full_candidate.get("human_key") or meta.get("human_key") or candidate_id)
            if human_key in seen_humans:
                continue
            seen_humans.add(human_key)
            distance = float(distances[0][index]) if distances and distances[0] else 0.0
            full_candidate["id"] = candidate_id
            full_candidate["score"] = round(1.0 / (1.0 + max(distance, 0.0)), 6)
            full_candidate["distance"] = round(distance, 6)
            full_candidate["text_model"] = self.model_name
            full_candidate.setdefault("search_text", documents[0][index] if documents and documents[0] else "")
            candidates.append(full_candidate)
            if len(candidates) >= top_k:
                break
        return candidates


if __name__ == "__main__":
    db = VectorSearchEngine()
    print(f"Current text model: {db.model_name}")
