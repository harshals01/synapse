from app.services.embedding_service import get_embedding
from app.services.qdrant_service import semantic_search, keyword_search
from app.utils.query_utils import extract_entity
from app.config import INDEX_NAME


def run_search(query, top_k, logger):
    logger.info(f"Running search for query: {query}")

    query_vector = get_embedding(query)

    vector_response = semantic_search(INDEX_NAME, query_vector, top_k)
    vector_hits = vector_response["hits"]["hits"]

    keyword_hits = []
    if extract_entity(query):
        logger.info("Keyword search enabled")
        keyword_response = keyword_search(INDEX_NAME, query)
        keyword_hits = keyword_response["hits"]["hits"]
    else:
        logger.info("Keyword search skipped")

    return vector_hits, keyword_hits