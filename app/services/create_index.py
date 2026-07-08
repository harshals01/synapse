import argparse
from app.config import INDEX_NAME
from app.services.qdrant_service import qdrant_client
from qdrant_client.http import models as qdrant_models

def recreate_collection():
    if not INDEX_NAME:
        print("[ERROR] INDEX_NAME is not configured in app/config.py")
        return

    # Delete if exists
    try:
        qdrant_client.get_collection(INDEX_NAME)
        print(f"[*] Collection '{INDEX_NAME}' already exists. Deleting it first...")
        qdrant_client.delete_collection(INDEX_NAME)
        print(f"[+] Collection '{INDEX_NAME}' deleted successfully.")
    except Exception:
        pass

    # Create collection
    print(f"[*] Creating Qdrant collection '{INDEX_NAME}'...")
    try:
        qdrant_client.create_collection(
            collection_name=INDEX_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=384,
                distance=qdrant_models.Distance.COSINE
            )
        )
        # Create full-text payload index
        qdrant_client.create_payload_index(
            collection_name=INDEX_NAME,
            field_name="combined",
            field_schema=qdrant_models.TextIndexParams(
                type="text",
                tokenizer=qdrant_models.TokenizerType.WORD,
                lowercase=True
            )
        )
        print(f"[SUCCESS] Qdrant collection '{INDEX_NAME}' created successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to create collection '{INDEX_NAME}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create/Recreate Qdrant collections.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the collection (deletes all existing data)."
    )
    args = parser.parse_args()

    if args.recreate:
        recreate_collection()
    else:
        # Check if exists
        try:
            qdrant_client.get_collection(INDEX_NAME)
            print(f"[+] Qdrant collection '{INDEX_NAME}' already exists.")
        except Exception:
            recreate_collection()
