from app.services.embedding_service import get_embedding
from app.services.qdrant_service import (
    semantic_search,
    keyword_search,
    get_latest_source_file,
)
from app.services.qdrant_service import qdrant_models
from app.utils.query_utils import extract_entity
from app.config import INDEX_NAME


def _build_filter(document_filter: str, logger) -> "qdrant_models.Filter | None":
    """Resolve document_filter to a Qdrant Filter or None (all documents)."""
    if document_filter == "all":
        return None

    if document_filter == "latest":
        source_file = get_latest_source_file(INDEX_NAME)
        if not source_file:
            logger.warning("document_filter='latest' but no documents found in index; searching all.")
            return None
        logger.info(f"document_filter='latest' resolved to '{source_file}'")
        document_filter = source_file  # fall through to exact-match below

    return qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="source_file",
                match=qdrant_models.MatchValue(value=document_filter),
            )
        ]
    )


def run_search(query, top_k, logger, document_filter: str = "latest"):
    logger.info(f"Running search for query: {query} | scope: {document_filter}")

    query_vector = get_embedding(query)
    qdrant_filter = _build_filter(document_filter, logger)

    vector_response = semantic_search(INDEX_NAME, query_vector, top_k, query_filter=qdrant_filter)
    vector_hits = vector_response["hits"]["hits"]

    keyword_hits = []
    if extract_entity(query):
        logger.info("Keyword search enabled")
        keyword_response = keyword_search(INDEX_NAME, query, query_filter=qdrant_filter)
        keyword_hits = keyword_response["hits"]["hits"]
    else:
        logger.info("Keyword search skipped")

    return vector_hits, keyword_hits