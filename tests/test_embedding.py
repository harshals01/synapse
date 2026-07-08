"""
Smoke tests for app/services/embedding_service.py.
All external HTTP calls are mocked — no real API key required.
"""
import time
from unittest.mock import MagicMock, patch

import pytest

import app.services.embedding_service as svc


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_ok(vectors: list) -> MagicMock:
    """Return a mock requests.Response with status 200 and the given JSON body."""
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = vectors
    return m


def _mock_error(status: int, body: dict | str = "", headers: dict | None = None) -> MagicMock:
    m = MagicMock()
    m.status_code = status
    m.text = str(body)
    m.headers = headers or {}
    m.json.return_value = body if isinstance(body, dict) else {}
    return m


VECTOR_384 = [0.1] * 384


# ── Single embedding tests ─────────────────────────────────────────────────────

class TestGetEmbedding:
    def test_returns_384_dim_vector(self):
        with patch("app.services.embedding_service.requests.post",
                   return_value=_mock_ok([VECTOR_384])):
            result = svc.get_embedding("What is Kubernetes?")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_nested_list_response_unwrapped(self):
        """HF API can return [[vec]] — the outer list must be unwrapped."""
        with patch("app.services.embedding_service.requests.post",
                   return_value=_mock_ok([[VECTOR_384]])):
            result = svc.get_embedding("test")
        assert len(result) == 384

    def test_oversized_chunk_is_truncated_before_sending(self):
        long_text = "a" * 3000
        with patch("app.services.embedding_service.requests.post",
                   return_value=_mock_ok([VECTOR_384])) as mock_post:
            svc.get_embedding(long_text)
        sent = mock_post.call_args.kwargs["json"]["inputs"]
        assert len(sent) == svc._MAX_CHARS_PER_CHUNK

    def test_retries_on_429_then_succeeds(self):
        rate_limited = _mock_error(429, headers={"Retry-After": "0"})
        success = _mock_ok([VECTOR_384])
        with patch("app.services.embedding_service.requests.post",
                   side_effect=[rate_limited, success]):
            with patch("app.services.embedding_service.time.sleep"):
                result = svc.get_embedding("test")
        assert len(result) == 384

    def test_retries_on_503_then_succeeds(self):
        cold_start = _mock_error(503, body={"estimated_time": 0})
        success = _mock_ok([VECTOR_384])
        with patch("app.services.embedding_service.requests.post",
                   side_effect=[cold_start, success]):
            with patch("app.services.embedding_service.time.sleep"):
                result = svc.get_embedding("test")
        assert len(result) == 384

    def test_raises_runtime_error_after_max_retries_exhausted(self):
        rate_limited = _mock_error(429, headers={"Retry-After": "0"})
        with patch("app.services.embedding_service.requests.post",
                   return_value=rate_limited):
            with patch("app.services.embedding_service.time.sleep"):
                with pytest.raises(RuntimeError, match="failed after"):
                    svc.get_embedding("test")

    def test_raises_value_error_on_terminal_4xx(self):
        bad_request = _mock_error(400, body="Bad input")
        with patch("app.services.embedding_service.requests.post",
                   return_value=bad_request):
            with pytest.raises(ValueError, match="HTTP 400"):
                svc.get_embedding("test")

    def test_raises_timeout_error_after_repeated_timeouts(self):
        with patch("app.services.embedding_service.requests.post",
                   side_effect=__import__("requests").exceptions.Timeout):
            with patch("app.services.embedding_service.time.sleep"):
                with pytest.raises(TimeoutError):
                    svc.get_embedding("test")


# ── Batch embedding tests ──────────────────────────────────────────────────────

class TestGetEmbeddingsBatch:
    def test_returns_one_vector_per_input(self):
        batch_vectors = [VECTOR_384, VECTOR_384, VECTOR_384]
        with patch("app.services.embedding_service.requests.post",
                   return_value=_mock_ok(batch_vectors)):
            result = svc.get_embeddings_batch(["a", "b", "c"])
        assert len(result) == 3
        assert all(len(v) == 384 for v in result)

    def test_truncates_oversized_chunks_in_batch(self):
        long_texts = ["x" * 3000, "y" * 2500]
        with patch("app.services.embedding_service.requests.post",
                   return_value=_mock_ok([VECTOR_384, VECTOR_384])) as mock_post:
            svc.get_embeddings_batch(long_texts)
        sent_inputs = mock_post.call_args.kwargs["json"]["inputs"]
        assert all(len(t) <= svc._MAX_CHARS_PER_CHUNK for t in sent_inputs)

    def test_falls_back_to_serial_on_413(self):
        too_large = _mock_error(413)
        serial_ok = _mock_ok([VECTOR_384])
        # First call is batch (413), subsequent calls are serial (one per text)
        with patch("app.services.embedding_service.requests.post",
                   side_effect=[too_large, serial_ok, serial_ok]):
            result = svc.get_embeddings_batch(["text1", "text2"])
        assert len(result) == 2

    def test_falls_back_to_serial_after_timeout_exhaustion(self):
        import requests as req_lib
        serial_ok = _mock_ok([VECTOR_384])
        side_effects = [req_lib.exceptions.Timeout] * svc.config.MAX_EMBED_RETRIES + [serial_ok, serial_ok]
        with patch("app.services.embedding_service.requests.post",
                   side_effect=side_effects):
            with patch("app.services.embedding_service.time.sleep"):
                result = svc.get_embeddings_batch(["text1", "text2"])
        assert len(result) == 2
