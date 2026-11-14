import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import app.services.embedding_service as svc



VECTOR_384 = [0.1] * 384
NP_VECTOR_384 = np.array(VECTOR_384, dtype=np.float32)
NP_BATCH_384 = np.array([VECTOR_384, VECTOR_384, VECTOR_384], dtype=np.float32)


def _mock_model(single_vector=NP_VECTOR_384, batch_vectors=NP_BATCH_384) -> MagicMock:
    """Return a mock SentenceTransformer whose encode() returns controlled numpy arrays."""
    mock = MagicMock()

    def _encode(text_or_texts, batch_size=32, normalize_embeddings=True):
        if isinstance(text_or_texts, list):
            return np.array([single_vector] * len(text_or_texts), dtype=np.float32)
        return single_vector

    mock.encode.side_effect = _encode
    return mock



class TestGetEmbedding:
    def test_returns_384_dim_vector(self):
        with patch("app.services.embedding_service.get_model", return_value=_mock_model()):
            result = svc.get_embedding("What is Kubernetes?")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_returns_floats(self):
        with patch("app.services.embedding_service.get_model", return_value=_mock_model()):
            result = svc.get_embedding("test")
        assert all(isinstance(v, float) for v in result)

    def test_empty_string_returns_384_dim_vector(self):
        with patch("app.services.embedding_service.get_model", return_value=_mock_model()):
            result = svc.get_embedding("")
        assert len(result) == 384

    def test_model_encode_called_with_normalize(self):
        mock = _mock_model()
        with patch("app.services.embedding_service.get_model", return_value=mock):
            svc.get_embedding("hello")
        mock.encode.assert_called_once_with("hello", normalize_embeddings=True)



class TestGetEmbeddingsBatch:
    def test_returns_one_vector_per_input(self):
        with patch("app.services.embedding_service.get_model", return_value=_mock_model()):
            result = svc.get_embeddings_batch(["a", "b", "c"])
        assert len(result) == 3
        assert all(len(v) == 384 for v in result)

    def test_empty_list_returns_empty(self):
        result = svc.get_embeddings_batch([])
        assert result == []

    def test_each_vector_is_list_of_floats(self):
        with patch("app.services.embedding_service.get_model", return_value=_mock_model()):
            result = svc.get_embeddings_batch(["x", "y"])
        assert all(isinstance(v, list) for v in result)
        assert all(isinstance(f, float) for v in result for f in v)

    def test_model_encode_called_with_batch_size_and_normalize(self):
        mock = _mock_model()
        texts = ["alpha", "beta", "gamma"]
        with patch("app.services.embedding_service.get_model", return_value=mock):
            svc.get_embeddings_batch(texts)
        mock.encode.assert_called_once_with(texts, batch_size=32, normalize_embeddings=True)



class TestGetModel:
    def test_model_loaded_only_once(self):
        """get_model() must return the same instance on repeated calls."""
        svc._model = None  
        with patch(
            "app.services.embedding_service.SentenceTransformer",
            return_value=_mock_model(),
        ) as mock_cls:
            svc.get_model()
            svc.get_model()
        mock_cls.assert_called_once()
        svc._model = None  
