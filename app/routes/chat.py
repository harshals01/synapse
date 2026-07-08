from fastapi import APIRouter
from app.models.request_models import ChatRequest
from app.logger import get_logger
from app.services.search_service import run_search
from app.services.fusion_service import rrf_fusion
from app.services.llm_service import call_llm
from app.utils.query_utils import rewrite_query_with_context
from app.config import LOW_CONFIDENCE_THRESHOLD, MAX_CONTEXT_DOCS

router = APIRouter()
logger = get_logger()


@router.post("/chat")
def chat(req: ChatRequest):
    try:
        if not req.messages:
            return {"error": "No messages provided"}

        messages = req.messages

        raw_query = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            ""
        )
        logger.info(f"Raw Query: {raw_query}")

        user_query = rewrite_query_with_context(messages, logger)
        user_query_cleaned = user_query.lower().strip()
        logger.info(f"Final Query: {user_query_cleaned}")

        vector_hits, keyword_hits = run_search(user_query_cleaned, req.top_k, logger)
        logger.info(f"Retrieved {len(vector_hits)} vector hits and {len(keyword_hits)} keyword hits.")

        sorted_hits = rrf_fusion(vector_hits, keyword_hits)

        max_score = sorted_hits[0]["score"] if sorted_hits else 0
        logger.info(f"Hybrid Max Score: {max_score} (Threshold: {LOW_CONFIDENCE_THRESHOLD})")

        if max_score < LOW_CONFIDENCE_THRESHOLD:
            return {"reply": "No relevant context found.", "llm_called": False}

        top_hits = sorted_hits[:MAX_CONTEXT_DOCS]

        context_blocks = [
            f"Document {i + 1}:\n{hit['source']['combined'].replace('<br>', chr(10))}"
            for i, hit in enumerate(top_hits)
        ]
        retrieved_context = "\n\n".join(context_blocks)

        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in messages[-4:-1]
        ]
        chat_history.append({"role": "user", "content": user_query})

        payload = {
            "messages": [
                {"role": "system", "content": "Answer using only the provided context. Do not hallucinate."},
                {"role": "system", "content": f"Context:\n{retrieved_context}"},
            ] + chat_history
        }

        reply = call_llm(payload, logger)

        return {
            "reply": reply,
            "llm_called": True,
            "hits_used": len(top_hits),
            "max_score": max_score,
        }

    except Exception as e:
        logger.error(str(e))
        return {"error": str(e)}

