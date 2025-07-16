from typing import List, Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient
from configs.configs import SIMILARITY_THRESHOLD, QDRANT_URL
from qdrant_client.http.exceptions import UnexpectedResponse
from dotenv import load_dotenv
import os

load_dotenv()

def vector_search_with_filter(query_vector: List[float], collection_name: str = "AI_store", limit: int = 5,
                              keyword: Optional[str] = None):
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=os.getenv("QDRANT_API_KEY"))
    try:
        filters = None
        if keyword:
            filters = Filter(
                must=[
                    FieldCondition(key="text", match=MatchValue(value=keyword))
                ]
            )
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=filters
        )
        return results

    except UnexpectedResponse as e:
        raise RuntimeError(f"Qdrant query failed: {str(e)}") from e
    except Exception as e:
        raise RuntimeError("An unexpected error occurred during vector search.") from e


def is_vector_similar(query_vector: List[float], collection_name: str = "AI_store") -> bool:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=os.getenv("QDRANT_API_KEY")
        )

        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            score_threshold=SIMILARITY_THRESHOLD,
            limit=1,
            with_payload=False
        )
        return bool(results.points)

def is_collection_exist(collection_name: str):
    all_collections = list_vector_stores()
    if collection_name in all_collections:
        return True
    return False

def list_vector_stores() -> list:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=os.getenv("QDRANT_API_KEY"))
    collections_info = client.get_collections()
    return [collection.name for collection in collections_info.collections]
