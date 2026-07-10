"""
CLI ingestion script — indexes a local PDF file into the Qdrant vector store.

Usage
-----
    python -m app.services.ingest_service --file path/to/document.pdf

Text extraction and chunking are delegated to the shared ``text_processor``
module so CLI and API ingestion produce identical chunk boundaries and overlap.
"""
import argparse
import os
import uuid

from qdrant_client.http.models import PointStruct

from app.config import INDEX_NAME
from app.services.embedding_service import get_embedding
from app.services.qdrant_service import qdrant_client
from app.services.text_processor import chunk_text, extract_text_from_path


def index_chunks(chunks: list[str], file_path: str) -> None:
    """Generate embeddings for text chunks and upload them to Qdrant in bulk."""
    print("Generating embeddings and indexing into Qdrant...")
    points = []

    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk)

        point_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"doc_{os.path.basename(file_path)}_{i}")
        )

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={"combined": chunk, "source_file": os.path.basename(file_path)},
            )
        )

        if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
            print(f"    Processed {i + 1}/{len(chunks)} chunks…")

    try:
        qdrant_client.upsert(collection_name=INDEX_NAME, points=points)
        print(f"Successfully indexed {len(points)} chunks into '{INDEX_NAME}'.")
    except Exception as e:
        print(f"Error during bulk upload to Qdrant: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a PDF file into the RAG Qdrant vector store."
    )
    parser.add_argument("--file", type=str, required=True, help="Path to the PDF file.")
    args = parser.parse_args()

    try:
        raw_text = extract_text_from_path(args.file)
        if not raw_text.strip():
            print("[ERROR] No extractable text found in the PDF.")
            raise SystemExit(1)

        document_chunks = chunk_text(raw_text)
        print(f"Generated {len(document_chunks)} chunks from '{args.file}'.")

        index_chunks(document_chunks, args.file)
        print("\n[SUCCESS] Ingestion completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {e}")
        raise SystemExit(1)
