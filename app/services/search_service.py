import os
from app.services.embedding_service import get_embedding
from app.services.qdrant_service import (
    semantic_search,
    keyword_search,
    get_latest_source_file,
    qdrant_models,
)
from app.utils.query_utils import extract_entity
from app.config import INDEX_NAME


def _user_condition(user_id: str) -> "qdrant_models.FieldCondition":
    """Return a Qdrant FieldCondition that matches points belonging to user_id."""
    return qdrant_models.FieldCondition(
        key="user_id",
        match=qdrant_models.MatchValue(value=user_id),
    )


def _build_filter(
    document_filter: str, user_id: str, logger
) -> "qdrant_models.Filter":
    """Resolve document_filter into a Qdrant Filter that always includes the user_id condition.

    The user_id condition is mandatory — it is always injected regardless of scope.
    """
    must = [_user_condition(user_id)]

    if not document_filter or document_filter == "all":
        # Search across all of this user's documents
        return qdrant_models.Filter(must=must)

    if document_filter == "latest":
        source_file = get_latest_source_file(INDEX_NAME, user_id=user_id)
        if not source_file:
            logger.warning(
                f"document_filter='latest' but no timestamped document found for user '{user_id}'; "
                "searching all of user's documents."
            )
            return qdrant_models.Filter(must=must)
        logger.info(f"document_filter='latest' resolved to '{source_file}' for user '{user_id}'")
        document_filter = source_file

    clean_filename = os.path.basename(document_filter.strip())
    must.append(
        qdrant_models.FieldCondition(
            key="source_file",
            match=qdrant_models.MatchValue(value=clean_filename),
        )
    )
    return qdrant_models.Filter(must=must)


def run_search(query, top_k, logger, document_filter: str = "latest", user_id: str = "default_user"):
    logger.info(f"Running search | scope: {document_filter} | user: {user_id}")

    query_vector = get_embedding(query)
    qdrant_filter = _build_filter(document_filter, user_id, logger)

    vector_response = semantic_search(INDEX_NAME, query_vector, top_k, query_filter=qdrant_filter)
    vector_hits = vector_response["hits"]["hits"]

    # Fallback 1: if scoped search returned 0 hits, retry with all user's documents
    if not vector_hits and document_filter != "all":
        logger.warning(
            f"Scoped search for '{document_filter}' returned 0 vector hits. "
            "Retrying with all of user's documents as fallback."
        )
        fallback_filter = _build_filter("all", user_id, logger)
        vector_response = semantic_search(INDEX_NAME, query_vector, top_k, query_filter=fallback_filter)
        vector_hits = vector_response["hits"]["hits"]

    # Fallback 2: if still 0 hits and user_id is not 'default_user', retry searching 'default_user' scope
    if not vector_hits and user_id != "default_user":
        logger.warning(
            f"Search for user '{user_id}' returned 0 vector hits. "
            "Retrying with 'default_user' scope as fallback."
        )
        fallback_filter = _build_filter("all", "default_user", logger)
        vector_response = semantic_search(INDEX_NAME, query_vector, top_k, query_filter=fallback_filter)
        vector_hits = vector_response["hits"]["hits"]

    # Fallback 3: if still 0 hits, retry with unconstrained search (recovers legacy/untagged points)
    if not vector_hits:
        logger.warning("Search returned 0 hits with user filters. Retrying without user_id filter.")
        vector_response = semantic_search(INDEX_NAME, query_vector, top_k, query_filter=None)
        vector_hits = vector_response["hits"]["hits"]

    keyword_hits = []
    if extract_entity(query):
        logger.info("Keyword search enabled")
        keyword_response = keyword_search(INDEX_NAME, query, query_filter=qdrant_filter)
        keyword_hits = keyword_response["hits"]["hits"]
        if not keyword_hits and user_id != "default_user":
            fallback_filter = _build_filter("all", "default_user", logger)
            keyword_response = keyword_search(INDEX_NAME, query, query_filter=fallback_filter)
            keyword_hits = keyword_response["hits"]["hits"]
    else:
        logger.info("Keyword search skipped")

    return vector_hits, keyword_hits
