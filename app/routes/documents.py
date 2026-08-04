from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.auth import require_api_key
from app.config import INDEX_NAME
from app.logger import get_logger
from app.services.qdrant_service import (
    delete_documents,
    list_ingested_documents,
    get_latest_source_file,
)

router = APIRouter()
logger = get_logger()


@router.get("/documents", dependencies=[Depends(require_api_key)])
def get_documents():
    """List all documents currently indexed in Qdrant, ordered by most recently ingested first."""
    try:
        docs = list_ingested_documents(INDEX_NAME)
        latest = get_latest_source_file(INDEX_NAME)
        return {
            "total_documents": len(docs),
            "latest_document": latest,
            "documents": docs,
        }
    except Exception:
        logger.exception("Unhandled exception in GET /documents")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document list.",
        )


@router.delete("/documents", dependencies=[Depends(require_api_key)])
def remove_documents(
    file_name: str | None = Query(
        None, description="Delete only chunks belonging to this filename."
    ),
    clear_all: bool = Query(
        False, description="If true, deletes ALL points in the index."
    ),
):
    """Delete one document's vectors or wipe the entire index.

    - ``?file_name=report.pdf``  removes only that document's chunks.
    - ``?clear_all=true``        removes all points from the collection.
    """
    if not file_name and not clear_all:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either file_name or clear_all=true.",
        )
    try:
        deleted = delete_documents(
            INDEX_NAME, file_name=file_name, clear_all=clear_all
        )
        return {
            "status": "success",
            "deleted_chunks": deleted,
            "file_name": file_name,
            "clear_all": clear_all,
        }
    except Exception:
        logger.exception("Unhandled exception in DELETE /documents")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete documents.",
        )
