from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid
from typing import List, Dict, Any
from src.utils.config import config
from src.utils.logger import logger

class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
        self.collection_name = config.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            logger.info(f"Creating collection {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            # Create payload indexes for filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="store_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="camera_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="dataset",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def insert_batch(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=v, payload=p)
            for v, p in zip(vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Inserted {len(points)} frames")

    def search(self, query_vector: List[float], top_k: int = 20, 
               store_filter: str = None, dataset_filter: str = None) -> List[Dict]:
        """Search similar vectors with optional filters."""
        must_conditions = []
        if store_filter:
            must_conditions.append(
                models.FieldCondition(key="store_id", match=models.MatchValue(value=store_filter))
            )
        if dataset_filter:
            must_conditions.append(
                models.FieldCondition(key="dataset", match=models.MatchValue(value=dataset_filter))
            )
        filter_cond = models.Filter(must=must_conditions) if must_conditions else None
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter_cond,
            with_payload=True,
        )
        return [{"score": hit.score, "payload": hit.payload} for hit in results]
