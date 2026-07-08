import os
import uuid
import argparse
from pypdf import PdfReader
from qdrant_client.http.models import PointStruct
from app.config import INDEX_NAME
from app.services.qdrant_service import qdrant_client
from app.services.embedding_service import get_embedding

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text page-by-page from the target PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    
    print(f"[*] Extracting text from: {pdf_path}")
    reader = PdfReader(pdf_path)
    full_text = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text.append(text)
            
    return "\n\n".join(full_text)

def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 70) -> list:
    """Split text into overlapping chunks using a recursive paragraph-sentence heuristic."""
    print(f"[*] Chunking document text (Target size: {chunk_size} chars, overlap: {chunk_overlap} chars)")
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If a single paragraph is too large, split it into sentences
        if len(para) > chunk_size:
            sentences = para.replace(". ", ".\n").split("\n")
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if current_length + len(sentence) > chunk_size:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        # Keep trailing sentences as overlap
                        overlap_chars = 0
                        overlap_chunk = []
                        for s in reversed(current_chunk):
                            if overlap_chars + len(s) < chunk_overlap:
                                overlap_chunk.insert(0, s)
                                overlap_chars += len(s)
                            else:
                                break
                        current_chunk = overlap_chunk
                        current_length = sum(len(s) for s in current_chunk)
                current_chunk.append(sentence)
                current_length += len(sentence)
        else:
            if current_length + len(para) > chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    # Keep overlap
                    overlap_chars = 0
                    overlap_chunk = []
                    for p in reversed(current_chunk):
                        if overlap_chars + len(p) < chunk_overlap:
                            overlap_chunk.insert(0, p)
                            overlap_chars += len(p)
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_length = sum(len(p) for p in current_chunk)
            current_chunk.append(para)
            current_length += len(para)
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    print(f"[+] Generated {len(chunks)} chunks.")
    return chunks

def index_chunks(chunks: list, file_path: str):
    """Generate embeddings for text chunks and upload them to Qdrant in bulk."""
    print("[*] Generating embeddings and indexing into Qdrant...")
    points = []
    
    for i, chunk in enumerate(chunks):
        # Generate 384-dimension vector from Hugging Face Inference API
        vector = get_embedding(chunk)
        
        # Generate a deterministic valid UUID for Qdrant compatibility
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"doc_{os.path.basename(file_path)}_{i}"))
        
        # Prepare Qdrant point structure
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "combined": chunk
            }
        )
        points.append(point)
        
        # Progress logging
        if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
            print(f"    - Processed {i + 1}/{len(chunks)} chunks...")
            
    # Bulk write to Qdrant
    try:
        qdrant_client.upsert(
            collection_name=INDEX_NAME,
            points=points
        )
        print(f"[+] Successfully indexed {len(points)} chunks into Qdrant collection '{INDEX_NAME}'!")
    except Exception as e:
        print(f"[!] Error: encountered errors during bulk upload to Qdrant: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF file into the RAG Qdrant vector store.")
    parser.add_argument("--file", type=str, required=True, help="Path to the PDF file to ingest.")
    args = parser.parse_args()
    
    try:
        # 1. Parse text from file
        raw_text = extract_text_from_pdf(args.file)
        
        # 2. Divide text into chunks
        document_chunks = chunk_text(raw_text)
        
        # 3. Embed and Bulk Index
        index_chunks(document_chunks, args.file)
        
        print("\n[SUCCESS] Ingestion completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {str(e)}")
