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

logger.info(f"Qdrant client initialized — URL: {config.QDRANT_URL}")


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
        # Keyword index on source_file for fast per-document filtering
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="source_file",
            field_schema=qdrant_models.KeywordIndexParams(type="keyword"),
        )
        # Datetime index on ingested_at for latest-document resolution
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="ingested_at",
            field_schema=qdrant_models.KeywordIndexParams(type="keyword"),
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


def semantic_search(
    collection_name: str,
    query_vector: list,
    top_k: int,
    query_filter: qdrant_models.Filter | None = None,
) -> dict:
    """Return top-K documents ranked by cosine similarity to the query vector."""
    try:
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )
        hits = [
            {"_id": str(p.id), "_score": p.score, "_source": p.payload}
            for p in results.points
        ]
        return {"hits": {"hits": hits}}
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return {"hits": {"hits": []}}


def keyword_search(
    collection_name: str,
    query: str,
    query_filter: qdrant_models.Filter | None = None,
) -> dict:
    """Return documents matching the query text on the 'combined' payload field."""
    try:
        text_condition = qdrant_models.FieldCondition(
            key="combined",
            match=qdrant_models.MatchText(text=query),
        )
        if query_filter is not None:
            # Merge the text condition with any existing must-clauses
            existing_must = list(query_filter.must or [])
            combined_filter = qdrant_models.Filter(must=existing_must + [text_condition])
        else:
            combined_filter = qdrant_models.Filter(must=[text_condition])

        results, _ = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=combined_filter,
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


def get_latest_source_file(collection_name: str) -> str | None:
    """Return the source_file name of the most recently ingested document.

    Only considers points that have a valid non-empty 'ingested_at' timestamp string.
    If no points have a valid 'ingested_at', returns None so search falls back safely.
    """
    try:
        results, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None

        valid_points = []
        for p in results:
            if not p.payload:
                continue
            sf = p.payload.get("source_file")
            ts = p.payload.get("ingested_at")
            if sf and isinstance(sf, str) and ts and isinstance(ts, str) and ts.strip():
                valid_points.append((ts, sf))

        if not valid_points:
            logger.warning("No points with valid 'ingested_at' timestamp found in index.")
            return None

        # Return the source_file of the newest ingested_at timestamp
        valid_points.sort(key=lambda x: x[0], reverse=True)
        return valid_points[0][1]
    except Exception as e:
        logger.error(f"get_latest_source_file failed: {e}")
        return None


def list_ingested_documents(collection_name: str) -> list[dict]:
    """Return a summary list of every distinct document in the collection."""
    try:
        results, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        doc_map: dict[str, dict] = {}
        for p in results:
            if not p.payload:
                continue
            fname = p.payload.get("source_file")
            if not fname or not isinstance(fname, str):
                continue
            ts = p.payload.get("ingested_at")
            ts_str = ts if isinstance(ts, str) else ""

            if fname not in doc_map:
                doc_map[fname] = {
                    "source_file": fname,
                    "chunk_count": 0,
                    "ingested_at": ts_str,
                }
            doc_map[fname]["chunk_count"] += 1
            if ts_str and ts_str > doc_map[fname]["ingested_at"]:
                doc_map[fname]["ingested_at"] = ts_str

        return sorted(doc_map.values(), key=lambda d: d["ingested_at"] or "", reverse=True)
    except Exception as e:
        logger.error(f"list_ingested_documents failed: {e}")
        return []


def delete_documents(
    collection_name: str,
    file_name: str | None = None,
    clear_all: bool = False,
) -> int:
    """Delete points from the collection.

    Args:
        collection_name: Target Qdrant collection.
        file_name: If provided, deletes only points whose source_file matches.
        clear_all: If True, deletes ALL points in the collection.

    Returns:
        Number of points deleted (approximated via count difference).
    """
    try:
        before = qdrant_client.count(collection_name=collection_name).count
        if clear_all:
            qdrant_client.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(must=[])
                ),
            )
        elif file_name:
            qdrant_client.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="source_file",
                                match=qdrant_models.MatchValue(value=file_name),
                            )
                        ]
                    )
                ),
            )
        after = qdrant_client.count(collection_name=collection_name).count
        return before - after
    except Exception as e:
        logger.error(f"delete_documents failed: {e}")
        return 0

