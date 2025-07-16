from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from models.models import DocumentModel
from datetime import datetime
import os
from dotenv import load_dotenv
import uuid

load_dotenv()
class VectorStore:
    def __init__(self, collection_name: str = "AI_store", vector_size: int = 384):
        self.client = QdrantClient(url="https://da4b5f90-0522-487e-9082-a0884bb63b81.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key=os.getenv("QDRANT_API_KEY"))
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.author_user_map = {}

    def init_collection(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def _get_or_create_user_id(self, author: str) -> str:
        if author not in self.author_user_map:
            self.author_user_map[author] = f"user_{uuid.uuid4().hex[:8]}"
        return self.author_user_map[author]

    def ingest(self, documents: List[DocumentModel], vectors: List[List[float]]):
        points = []
        doc_uuid = f"doc_{uuid.uuid4().hex[:8]}"

        for i, (doc, vector) in enumerate(zip(documents, vectors)):
            user_id = self._get_or_create_user_id(doc.researcher)
            payload = {
                "text": doc.text,
                "researcher": doc.researcher,
                "user_id": user_id,
                "document_id": doc_uuid,
                "chunk_id": f"chunk_{i}",
                "time": datetime.now(),
                "pdf_path": doc.pdf_path,
                "findings": doc.findings
            }

            point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
            points.append(point)

        if points:
            try:
                self.client.upsert(collection_name=self.collection_name, points=points)
            except Exception as e:
                raise RuntimeError(f"Failed to upsert points into collection '{self.collection_name}': {e}")