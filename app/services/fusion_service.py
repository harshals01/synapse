from collections import defaultdict

def rrf_fusion(vector_hits, keyword_hits, k=60):
    scores = defaultdict(float)
    doc_map = {}

    # Build rankings
    rankings = [
        [hit["_id"] for hit in keyword_hits],
        [hit["_id"] for hit in vector_hits]
    ]

    # Store document data
    for hit in keyword_hits + vector_hits:
        doc_map[hit["_id"]] = hit["_source"]

    # Apply RRF
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1 / (k + rank + 1)

    # Final sorted results
    return sorted(
        [{"score": score, "source": doc_map[doc_id]}
         for doc_id, score in scores.items()],
        key=lambda x: x["score"],
        reverse=True
    )
    