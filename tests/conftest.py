"""
Pytest fixtures shared across the test suite.
Provides a minimal valid PDF (with extractable text) built from raw bytes —
no external PDF library required beyond pypdf which is already a project dependency.
"""
import pytest


def _create_minimal_pdf() -> bytes:
    """
    Build a minimal valid PDF-1.4 document with a single page containing
    extractable text. Byte offsets in the xref table are computed dynamically
    so the output is always structurally correct regardless of content changes.
    """
    content_text = "Test document for RAG ingestion pipeline."
    content_stream: bytes = (
        f"BT /F1 12 Tf 72 720 Td ({content_text}) Tj ET".encode()
    )

    # PDF object bodies
    obj1 = b"<</Type /Catalog /Pages 2 0 R>>"
    obj2 = b"<</Type /Pages /Kids [3 0 R] /Count 1>>"
    obj3 = (
        b"<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>>"
    )
    obj4_dict: bytes = f"<</Length {len(content_stream)}>>".encode()
    obj5 = b"<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>"

    # Build body and record each object's byte offset
    body = b"%PDF-1.4\n"
    offsets: dict[int, int] = {}

    for obj_id, obj_body in [(1, obj1), (2, obj2), (3, obj3)]:
        offsets[obj_id] = len(body)
        body += f"{obj_id} 0 obj\n".encode() + obj_body + b"\nendobj\n"

    # Object 4 has a stream
    offsets[4] = len(body)
    body += (
        b"4 0 obj\n"
        + obj4_dict
        + b"\nstream\n"
        + content_stream
        + b"\nendstream\nendobj\n"
    )

    # Object 5
    offsets[5] = len(body)
    body += b"5 0 obj\n" + obj5 + b"\nendobj\n"

    # Cross-reference table
    xref_offset = len(body)
    xref = b"xref\n0 6\n"
    xref += b"0000000000 65535 f \n"
    for obj_id in [1, 2, 3, 4, 5]:
        xref += f"{offsets[obj_id]:010d} 00000 n \n".encode()

    trailer = (
        b"trailer\n<</Size 6 /Root 1 0 R>>\nstartxref\n"
        + str(xref_offset).encode()
        + b"\n%%EOF"
    )

    return body + xref + trailer


@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    """
    Minimal valid PDF with extractable text.
    Scoped to the session so it is generated once and reused across all tests.
    """
    return _create_minimal_pdf()


@pytest.fixture
def sample_pdf_path(tmp_path, sample_pdf_bytes: bytes) -> str:
    """
    Write the sample PDF to a temporary file and return its absolute path.
    The file is automatically removed after each test.
    """
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(sample_pdf_bytes)
    return str(pdf_file)
