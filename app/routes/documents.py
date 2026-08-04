from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.auth import require_api_key, get_current_user_id
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
def get_documents(user_id: str = Depends(get_current_user_id)):
    """List documents indexed by the calling user, ordered by most recently ingested first."""
    try:
        docs = list_ingested_documents(INDEX_NAME, user_id=user_id)
        latest = get_latest_source_file(INDEX_NAME, user_id=user_id)
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
    user_id: str = Depends(get_current_user_id),
    file_name: str | None = Query(
        None, description="Delete only chunks belonging to this filename."
    ),
    clear_all: bool = Query(
        False, description="If true, deletes ALL of the calling user's points in the index."
    ),
):
    """Delete one of the calling user's documents or wipe all of their indexed data.

    - ``?file_name=report.pdf``  removes only that document's chunks.
    - ``?clear_all=true``        removes all points belonging to the calling user.
    """
    if not file_name and not clear_all:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either file_name or clear_all=true.",
        )
    try:
        deleted = delete_documents(
            INDEX_NAME, user_id=user_id, file_name=file_name, clear_all=clear_all
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
