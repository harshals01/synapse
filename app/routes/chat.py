from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import require_api_key
from app import config
from app.config import LOW_CONFIDENCE_THRESHOLD, MAX_CONTEXT_DOCS
from app.logger import get_logger
from app.models.request_models import ChatRequest
from app.services.fusion_service import rrf_fusion
from app.services.llm_service import call_llm
from app.services.search_service import run_search
from app.utils.query_utils import rewrite_query_with_context

router = APIRouter()
logger = get_logger()


@router.post("/chat", dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest):
    try:
        messages = [m.model_dump() for m in req.messages]

        raw_query = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        logger.info(f"Query received ({len(raw_query)} chars)")

        user_query = rewrite_query_with_context(messages, logger)
        user_query_cleaned = user_query.lower().strip()
        logger.info(f"Final Query: {user_query_cleaned}")

        vector_hits, keyword_hits = run_search(user_query_cleaned, req.top_k, logger)
        logger.info(
            f"Retrieved {len(vector_hits)} vector hits and {len(keyword_hits)} keyword hits."
        )

        sorted_hits = rrf_fusion(vector_hits, keyword_hits)

        max_score = sorted_hits[0]["score"] if sorted_hits else 0
        logger.info(f"Hybrid Max Score: {max_score} (Threshold: {LOW_CONFIDENCE_THRESHOLD})")

        if max_score < LOW_CONFIDENCE_THRESHOLD:
            return {"reply": "No relevant context found.", "llm_called": False}

        top_hits = sorted_hits[:MAX_CONTEXT_DOCS]

        MAX_CONTEXT_CHARS = 8000
        context_blocks = []
        accumulated_len = 0

        for i, hit in enumerate(top_hits):
            combined_text = (
                hit.get("source", {}).get("combined", "").replace("<br>", "\n").strip()
            )
            if not combined_text:
                continue
            block = f"Document {i + 1}:\n{combined_text}"
            if accumulated_len + len(block) > MAX_CONTEXT_CHARS and context_blocks:
                break
            context_blocks.append(block)
            accumulated_len += len(block)

        retrieved_context = "\n\n".join(context_blocks)

        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in messages[-(config.CHAT_HISTORY_WINDOW + 1):-1]
            if m.get("content", "").strip()
        ]
        chat_history.append({"role": "user", "content": user_query})

        system_prompt = (
            "Answer using only the provided context. Do not hallucinate.\n\n"
            f"Context:\n{retrieved_context}"
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt}
            ] + chat_history
        }

        reply = call_llm(payload, logger)

        return {
            "reply": reply,
            "llm_called": True,
            "hits_used": len(context_blocks),
            "max_score": max_score,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unhandled exception in POST /chat")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your request.",
        )
