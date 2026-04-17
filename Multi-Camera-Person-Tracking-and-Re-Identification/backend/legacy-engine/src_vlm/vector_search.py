from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import chromadb

from .hospital_pipeline import DEFAULT_TEXT_MODEL, METADATA_DIR


DEFAULT_SEARCH_HYPERPARAMETERS = {
    "fetch_multiplier": 10,
    "score_weights": {
        "embedding": 0.58,
        "semantic_overlap": 0.22,
        "visibility": 0.12,
        "world_position": 0.08,
    },
    "minimum_semantic_overlap": 0.08,
}


@lru_cache(maxsize=2)
def _load_sentence_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class VectorSearchEngine:
    def __init__(self, db_path="../data/db", model_name: str = DEFAULT_TEXT_MODEL, hyperparameters: dict | None = None):
        print("[VectorSearch] Đang khởi tạo kết nối Vector Database (ChromaDB)...")
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.hyperparameters = hyperparameters or DEFAULT_SEARCH_HYPERPARAMETERS
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

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if len(token) >= 2}

    def _semantic_overlap(self, query_text: str, candidate: dict) -> float:
        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return 0.0
        candidate_tokens = set()
        candidate_tokens.update(self._tokenize(candidate.get("search_text") or ""))
        candidate_tokens.update(self._tokenize(candidate.get("appearance_summary") or ""))
        for attribute in candidate.get("semantic_attributes") or []:
            candidate_tokens.update(self._tokenize(attribute))
        if not candidate_tokens:
            return 0.0
        overlap = len(query_tokens & candidate_tokens)
        return float(overlap / max(len(query_tokens), 1))

    def _visibility_bonus(self, candidate: dict) -> float:
        scores = candidate.get("visibility_scores")
        if not isinstance(scores, dict) or not scores:
            return 0.0
        numeric_scores = [float(value) for value in scores.values() if isinstance(value, (int, float))]
        if not numeric_scores:
            return 0.0
        return float(sum(numeric_scores) / len(numeric_scores))

    def _world_position_bonus(self, candidate: dict) -> float:
        world_position = candidate.get("world_position")
        if not isinstance(world_position, dict) or not world_position:
            return 0.0
        if any(world_position.get(axis) is not None for axis in ("x", "y", "z")):
            return 1.0
        return 0.0

    def _ranking_ensemble_score(self, query_text: str, candidate: dict, distance: float) -> float:
        weights = self.hyperparameters.get("score_weights", {})
        cosine_score = 1.0 / (1.0 + max(float(distance), 0.0))
        semantic_score = self._semantic_overlap(query_text, candidate)
        visibility_score = self._visibility_bonus(candidate)
        world_position_score = self._world_position_bonus(candidate)
        return round(
            (cosine_score * float(weights.get("embedding", 0.58)))
            + (semantic_score * float(weights.get("semantic_overlap", 0.22)))
            + (visibility_score * float(weights.get("visibility", 0.12)))
            + (world_position_score * float(weights.get("world_position", 0.08))),
            6,
        )

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
        fetch_multiplier = int(self.hyperparameters.get("fetch_multiplier", DEFAULT_SEARCH_HYPERPARAMETERS["fetch_multiplier"]))
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(int(top_k) * fetch_multiplier, int(top_k)),
            include=["metadatas", "distances", "documents"],
        )

        metadatas = raw.get("metadatas", [[]])
        distances = raw.get("distances", [[]])
        documents = raw.get("documents", [[]])
        ids = raw.get("ids", [[]])
        candidates = []
        fallback_candidates = []
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
            full_candidate["score"] = self._ranking_ensemble_score(query, full_candidate, distance)
            full_candidate["distance"] = round(distance, 6)
            full_candidate["semantic_overlap"] = round(self._semantic_overlap(query, full_candidate), 6)
            full_candidate["world_position_score"] = round(self._world_position_bonus(full_candidate), 6)
            full_candidate["text_model"] = self.model_name
            full_candidate.setdefault("search_text", documents[0][index] if documents and documents[0] else "")
            minimum_semantic_overlap = float(
                self.hyperparameters.get(
                    "minimum_semantic_overlap",
                    DEFAULT_SEARCH_HYPERPARAMETERS["minimum_semantic_overlap"],
                )
            )
            if full_candidate["semantic_overlap"] < minimum_semantic_overlap:
                fallback_candidates.append(full_candidate)
                continue
            candidates.append(full_candidate)
        if not candidates and fallback_candidates:
            candidates = fallback_candidates
        candidates.sort(key=lambda item: (-float(item.get("score") or 0.0), float(item.get("distance") or 0.0)))
        return candidates[:top_k]


if __name__ == "__main__":
    db = VectorSearchEngine()
    print(f"Current text model: {db.model_name}")
