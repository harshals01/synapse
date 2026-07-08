from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from app import config
from app.logger import get_logger

logger = get_logger()

# Initialize Qdrant client
if config.QDRANT_URL == ":memory:":
    qdrant_client = QdrantClient(":memory:")
else:
    _client_kwargs: dict = {"url": config.QDRANT_URL}
    if config.QDRANT_API_KEY:
        _client_kwargs["api_key"] = config.QDRANT_API_KEY
    qdrant_client = QdrantClient(**_client_kwargs)

logger.info(f"QdrantClient instance type: {type(qdrant_client)}")
logger.info(f"QdrantClient has 'search': {hasattr(qdrant_client, 'search')}")
logger.info(f"QdrantClient attributes: {[attr for attr in dir(qdrant_client) if not attr.startswith('_')]}")


def ensure_collection_exists(collection_name: str) -> None:
    """Create the Qdrant collection if it does not already exist."""
    try:
        qdrant_client.get_collection(collection_name)
        logger.info(f"Collection '{collection_name}' is ready.")
    except Exception:
        logger.info(f"Collection '{collection_name}' not found. Creating...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=384,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        # Full-text index on 'combined' field enables keyword search via scroll filter
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="combined",
            field_schema=qdrant_models.TextIndexParams(
                type="text",
                tokenizer=qdrant_models.TokenizerType.WORD,
                lowercase=True,
            ),
        )
        logger.info(f"Collection '{collection_name}' created successfully.")


# Attempt collection setup at startup; log a warning on failure instead of crashing.
try:
    ensure_collection_exists(config.INDEX_NAME)
except Exception as _e:
    logger.warning(
        f"Qdrant startup check failed: {_e}. "
        "Verify QDRANT_URL and QDRANT_API_KEY are correctly set."
    )


def semantic_search(collection_name: str, query_vector: list, top_k: int) -> dict:
    """Return top-K documents ranked by cosine similarity to the query vector."""
    try:
        # Use the modern query_points API which is supported on all recent qdrant-client versions
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
        )
        hits = [
            {"_id": str(p.id), "_score": p.score, "_source": p.payload}
            for p in results.points
        ]
        return {"hits": {"hits": hits}}
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return {"hits": {"hits": []}}


def keyword_search(collection_name: str, query: str) -> dict:
    """Return documents matching the query text on the 'combined' payload field."""
    try:
        results, _ = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="combined",
                        match=qdrant_models.MatchText(text=query),
                    )
                ]
            ),
            limit=100,
        )
        hits = [
            {"_id": str(p.id), "_score": 1.0, "_source": p.payload}
            for p in results
        ]
        return {"hits": {"hits": hits}}
    except Exception as e:
        logger.error(f"Keyword search failed: {e}")
        return {"hits": {"hits": []}}

