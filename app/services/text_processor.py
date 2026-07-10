"""
Shared PDF text extraction and overlapping chunk utilities.

Both the HTTP ingest route (app/routes/ingest.py) and the CLI ingestion
script (app/services/ingest_service.py) import from this module to guarantee
identical preprocessing behaviour regardless of ingestion pathway.
"""
import io

from pypdf import PdfReader


# ── Text Extraction ───────────────────────────────────────────────────────────

def extract_text_from_bytes(raw_bytes: bytes) -> str:
    """
    Extract all selectable text from a PDF supplied as raw bytes.
    Used by the API route where the file arrives as an in-memory upload.
    Returns an empty string if the PDF contains no extractable text.
    """
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n\n".join(pages)


def extract_text_from_path(pdf_path: str) -> str:
    """
    Extract all selectable text from a PDF supplied as a filesystem path.
    Used by CLI scripts where the file is read from disk.
    """
    with open(pdf_path, "rb") as fh:
        return extract_text_from_bytes(fh.read())


# ── Text Chunking ─────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 70,
) -> list[str]:
    """
    Split document text into overlapping chunks using a paragraph-sentence
    heuristic with a sliding overlap window.

    Algorithm
    ---------
    1.  Split the text on double-newlines to obtain paragraphs.
    2.  Accumulate paragraphs into the current chunk while they fit within
        ``chunk_size``.
    3.  When a paragraph does not fit, flush the current chunk, then seed
        the next chunk with the trailing portion of the flushed content that
        fits within ``chunk_overlap`` characters (sliding window).
    4.  Paragraphs wider than ``chunk_size`` are further split on sentence
        boundaries ('. '), applying the same overlap logic at sentence level.

    Parameters
    ----------
    text:         Full document text string.
    chunk_size:   Target maximum character length per chunk (default 700).
    chunk_overlap: Number of characters from the end of the previous chunk
                   carried into the start of the next chunk (default 70).

    Returns
    -------
    A list of text chunk strings.
    """
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length: int = 0

    def _flush(parts: list[str], join_str: str) -> tuple[list[str], int]:
        """
        Append the current chunk to ``chunks``, then build and return
        the overlap seed for the next chunk.
        """
        chunks.append(join_str.join(parts))

        overlap_chars = 0
        overlap_parts: list[str] = []
        for part in reversed(parts):
            if overlap_chars + len(part) <= chunk_overlap:
                overlap_parts.insert(0, part)
                overlap_chars += len(part)
            else:
                break
        seed_length = sum(len(p) for p in overlap_parts)
        return overlap_parts, seed_length

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > chunk_size:
            # Oversized paragraph → split on sentence boundaries
            sentences = para.replace(". ", ".\n").split("\n")

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if current_length + len(sentence) > chunk_size and current_chunk:
                    current_chunk, current_length = _flush(current_chunk, " ")

                current_chunk.append(sentence)
                current_length += len(sentence)

        else:
            if current_length + len(para) > chunk_size and current_chunk:
                current_chunk, current_length = _flush(current_chunk, "\n\n")

            current_chunk.append(para)
            current_length += len(para)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks
