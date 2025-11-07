from typing import List, Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue, PayloadSchemaType
from qdrant_client import QdrantClient
from configs.configs import SIMILARITY_THRESHOLD, QDRANT_URL
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Filter, FieldCondition, Match
from dotenv import load_dotenv
import os

load_dotenv()


def vector_search_with_filter(
    query_vector: List[float],
    collection_name: str = "AI_store",
    limit: int = 5,
    keyword: Optional[str] = None,
):
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    try:
        # Ensure the collection exists before indexing
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        if collection_name not in collection_names:
            raise RuntimeError(
                f"Collection '{collection_name}' does not exist in Qdrant!"
            )

        # Ensure keyword indexes exist for filtering fields
        for field in ["researcher", "filename"]:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                print(f"✅ Created missing index for field: {field}")
            except Exception:
                # Likely already exists — ignore quietly
                pass

        # Apply researcher filter if provided
        filters = None
        if keyword:
            filters = Filter(
                must=[FieldCondition(key="researcher", match=MatchValue(value=keyword))]
            )
            print(f"🔍 Applying filter: researcher = {keyword}")

        # Perform the filtered vector search
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=filters,
        )

        return results

    except UnexpectedResponse as e:
        raise RuntimeError(f"Qdrant query failed: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred during vector search: {str(e)}"
        ) from e


def are_vectors_similar(
    query_vectors: List[List[float]], collection_name: str = "AI_store"
) -> List[bool]:
    client = QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY"))

    search_queries = [
        models.QueryRequest(
            query=vector,
            score_threshold=SIMILARITY_THRESHOLD,
            limit=1,
            with_payload=False,
        )
        for vector in query_vectors
    ]

    search_results = client.query_batch_points(
        collection_name=collection_name,
        requests=search_queries,
    )

    return [bool(result.points) for result in search_results]


def is_vector_similar(
    query_vector: List[float], collection_name: str = "AI_store"
) -> bool:
    client = QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY"))

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        score_threshold=SIMILARITY_THRESHOLD,
        limit=1,
        with_payload=False,
    )
    return bool(results.points)


def is_collection_exist(collection_name: str):
    all_collections = list_vector_stores()
    if collection_name in all_collections:
        return True
    return False


def list_vector_stores() -> list:
    client = QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY"))
    collections_info = client.get_collections()
    return [collection.name for collection in collections_info.collections]


def extract_metadata(collection_name: str, findings: Optional[str] = None):
    """Extract distinct researcher metadata from Qdrant collection."""

    client = QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY"))

    # Build filter if findings provided

    query_filter = None
    if findings:
        query_filter = Filter(
            must=[FieldCondition(key="findings", match=Match(value=findings))]
        )
    # Get points from collection
    result = client.scroll(
        collection_name=collection_name, scroll_filter=query_filter, with_payload=True
    )

    print(result)
    # Extract distinct metadata
    seen = set()
    metadata = []

    for point in result[0]:
        researcher = point.payload.get("researcher", "")
        findings_text = point.payload.get("findings", "")

        key = (researcher, findings_text)
        if key not in seen:
            seen.add(key)
            metadata.append({"researcher": researcher, "findings": findings_text})

    return metadata


def extract_from_all_collections(findings: Optional[str] = None):
    """Extract metadata from all collections that match findings."""

    # Get all collection names
    client = QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY"))

    collections_info = client.get_collections()
    collections = [collection.name for collection in collections_info.collections]
    results = {}
    for collection in collections:
        try:
            metadata = extract_metadata(collection, findings)
            if metadata:  # Only include if has results
                results[collection] = metadata
        except:
            continue

    return results


if __name__ == "__main__":

    print(vector_search_with_filter())