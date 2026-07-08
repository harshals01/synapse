import asyncio
import io
import uuid
from functools import partial

from fastapi import APIRouter, File, HTTPException, UploadFile
from pypdf import PdfReader
from qdrant_client.http.models import PointStruct

from app.config import INDEX_NAME, INGEST_BATCH_SIZE
from app.logger import get_logger
from app.services.embedding_service import get_embeddings_batch
from app.services.qdrant_service import ensure_collection_exists, qdrant_client

router = APIRouter()
logger = get_logger()

_ALLOWED_CONTENT_TYPES = {"application/pdf"}
_MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload, extract text, chunk it, embed via HF Serverless API,
    and upsert vectors into Qdrant. Returns ingestion statistics.
    """
    # --- Validation ---
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    raw_bytes = await file.read()

    if len(raw_bytes) > _MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 20 MB size limit.")

    logger.info(f"[INGEST] Processing: {file.filename} ({len(raw_bytes)} bytes)")

    # --- Step 1: Extract text ---
    text = _extract_text(raw_bytes)
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text found. Scanned/image-only PDFs are not supported.",
        )

    # --- Step 2: Chunk text ---
    chunks = _chunk_text(text)
    logger.info(f"[INGEST] {len(chunks)} chunks generated from '{file.filename}'.")

    # --- Step 3: Ensure collection exists ---
    try:
        ensure_collection_exists(INDEX_NAME)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector store unavailable: {e}")

    # --- Step 4: Batch embed + build PointStructs ---
    loop = asyncio.get_event_loop()
    points: list[PointStruct] = []
    failed_batches: list[dict] = []

    for batch_start in range(0, len(chunks), INGEST_BATCH_SIZE):
        batch = chunks[batch_start: batch_start + INGEST_BATCH_SIZE]
        try:
            # Run blocking sync call in thread executor to keep the event loop free
            vectors: list[list[float]] = await loop.run_in_executor(
                None, partial(get_embeddings_batch, batch)
            )
        except Exception as e:
            logger.error(f"[INGEST] Embedding failed at batch index {batch_start}: {e}")
            failed_batches.append({"batch_start": batch_start, "error": str(e)})
            continue

        for i, (chunk, vector) in enumerate(zip(batch, vectors)):
            chunk_index = batch_start + i
            # Deterministic UUID: re-uploading the same file overwrites rather than duplicates
            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"doc_{file.filename}_{chunk_index}",
            ))
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={"combined": chunk, "source_file": file.filename},
            ))

    # --- Step 5: Bulk write to Qdrant ---
    if points:
        try:
            qdrant_client.upsert(collection_name=INDEX_NAME, points=points)
            logger.info(f"[INGEST] Upserted {len(points)} points into '{INDEX_NAME}'.")
        except Exception as e:
            logger.error(f"[INGEST] Qdrant upsert failed: {e}")
            raise HTTPException(status_code=500, detail=f"Vector store write error: {e}")

    return {
        "status": "success",
        "file": file.filename,
        "chunks_indexed": len(points),
        "batches_failed": len(failed_batches),
        "errors": failed_batches if failed_batches else None,
    }


def _extract_text(raw_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n\n".join(pages)


def _chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 70) -> list[str]:
    """Split document text into overlapping chunks using a paragraph-sentence heuristic."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length: int = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > chunk_size:
            sentences = para.replace(". ", ".\n").split("\n")
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if current_length + len(sentence) > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, current_length = [], 0
                current_chunk.append(sentence)
                current_length += len(sentence)
        else:
            if current_length + len(para) > chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk, current_length = [], 0
            current_chunk.append(para)
            current_length += len(para)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks
