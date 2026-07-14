"""
CLI utility — Create or recreate the Qdrant collection for the RAG knowledge base.

Usage
-----
    # Check if collection exists (create if missing):
    python -m app.services.create_index

    # Force-delete and recreate the collection (DESTROYS ALL DATA):
    python -m app.services.create_index --recreate
"""
import argparse

from app.config import INDEX_NAME
from app.services.qdrant_service import ensure_collection_exists, qdrant_client


def recreate_collection() -> None:
    """Delete and recreate the Qdrant collection, wiping all existing data."""
    if not INDEX_NAME:
        print("[ERROR] INDEX_NAME is not configured.")
        return

    # Delete if exists
    try:
        qdrant_client.get_collection(INDEX_NAME)
        print(f"[*] Collection '{INDEX_NAME}' exists. Deleting...")
        qdrant_client.delete_collection(INDEX_NAME)
        print(f"[+] Collection '{INDEX_NAME}' deleted.")
    except Exception:
        pass  # Collection did not exist — nothing to delete

    # Delegate creation to the single source of truth in qdrant_service
    ensure_collection_exists(INDEX_NAME)
    print(f"[SUCCESS] Collection '{INDEX_NAME}' recreated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create/Recreate the Qdrant collection.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Force-delete and recreate the collection (DESTROYS ALL DATA).",
    )
    args = parser.parse_args()

    if args.recreate:
        recreate_collection()
    else:
        # Delegate to ensure_collection_exists — same logic used at server startup
        ensure_collection_exists(INDEX_NAME)
        print(f"[+] Collection '{INDEX_NAME}' is ready.")
