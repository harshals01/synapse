import os
import asyncio
import uuid
from datetime import datetime, timezone
from functools import partial

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from qdrant_client.http.models import PointStruct

from app.auth import require_api_key, get_current_user_id
from app.config import INDEX_NAME, INGEST_BATCH_SIZE
from app.logger import get_logger
from app.services.embedding_service import get_embeddings_batch
from app.services.qdrant_service import ensure_collection_exists, qdrant_client
from app.services.text_processor import chunk_text, extract_text_from_bytes

router = APIRouter()
logger = get_logger()

_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "applications/vnd.pdf",
    "text/pdf",
    "text/x-pdf",
    "application/octet-stream",
}
_MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("/ingest", dependencies=[Depends(require_api_key)])
async def ingest_pdf(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """
    Accept a PDF upload, extract text, chunk it with sliding overlap,
    embed via HF Serverless API, and upsert vectors into Qdrant.
    Returns ingestion statistics on completion.
    """
    is_valid_type = file.content_type in _ALLOWED_CONTENT_TYPES
    is_pdf_ext = file.filename and file.filename.lower().endswith(".pdf")

    if not (is_valid_type or is_pdf_ext):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    raw_bytes = await file.read()

    if len(raw_bytes) > _MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 20 MB size limit.")

    logger.info(f"[INGEST] Processing: {file.filename} ({len(raw_bytes):,} bytes)")

    text = extract_text_from_bytes(raw_bytes)
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text found. Scanned or image-only PDFs are not supported without OCR.",
        )

    chunks = chunk_text(text)
    logger.info(f"[INGEST] {len(chunks)} chunks generated from '{file.filename}'.")

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, partial(ensure_collection_exists, INDEX_NAME))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector store unavailable: {e}")

    points: list[PointStruct] = []
    failed_batches: list[dict] = []
    ingested_at = datetime.now(timezone.utc).isoformat()

    batches = [
        (batch_start, chunks[batch_start : batch_start + INGEST_BATCH_SIZE])
        for batch_start in range(0, len(chunks), INGEST_BATCH_SIZE)
    ]
    sem = asyncio.Semaphore(4)

    async def process_batch(batch_start: int, batch: list[str]):
        async with sem:
            try:
                vectors: list[list[float]] = await loop.run_in_executor(
                    None, partial(get_embeddings_batch, batch)
                )
                return (batch_start, batch, vectors, None)
            except Exception as e:
                logger.error(f"[INGEST] Embedding failed at batch index {batch_start}: {e}")
                return (batch_start, batch, None, str(e))

    results = await asyncio.gather(*(process_batch(b_start, b_chunks) for b_start, b_chunks in batches))

    for batch_start, batch, vectors, error in results:
        if error or not vectors:
            failed_batches.append({"batch_start": batch_start, "error": error or "Empty vectors"})
            continue
        for i, (chunk, vector) in enumerate(zip(batch, vectors)):
            chunk_index = batch_start + i
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"doc_{file.filename}_{chunk_index}",
                )
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "combined": chunk,
                        "source_file": os.path.basename(file.filename or ""),
                        "ingested_at": ingested_at,
                        "user_id": user_id,
                    },
                )
            )

    if not points and failed_batches:
        first_err = failed_batches[0]["error"]
        logger.error(f"[INGEST] Ingestion failed completely for '{file.filename}': {first_err}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding API call failed during PDF processing: {first_err}",
        )

    if points:
        try:
            await loop.run_in_executor(
                None,
                partial(qdrant_client.upsert, collection_name=INDEX_NAME, points=points),
            )
            logger.info(f"[INGEST] Upserted {len(points)} points into '{INDEX_NAME}'.")
        except Exception as e:
            logger.exception("[INGEST] Qdrant upsert failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Vector store write error: {e}",
            )

    return {
        "status": "success",
        "file": file.filename,
        "chunks_indexed": len(points),
        "batches_failed": len(failed_batches),
        "errors": failed_batches if failed_batches else None,
    }
