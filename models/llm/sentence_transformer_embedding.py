"""Free sentence-transformers embedding model (replaces OpenAI text-embedding-3-small)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger

# Default model - high quality and free
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Alternative: "sentence-transformers/all-mpnet-base-v2" (better quality, slower)


class SentenceTransformerEmbedding:
    """Local sentence-transformer embedding model (free, no API needed)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """Initialize sentence-transformer model.

        Args:
            model_name: HuggingFace model name or local path
            device: 'cuda', 'cpu', or None for auto-detect
            normalize: Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.normalize = normalize
        self._model = None
        self._device = device

    @property
    def model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

            logger.info("Loading sentence-transformer model: {}", self.model_name)
            self._model = SentenceTransformer(
                self.model_name,
                device=self._device,
                cache_folder="./models_cache/sentence_transformers",
            )
            logger.info("Model loaded successfully (device: {})", self._model.device)
        return self._model

    def embed_text(
        self,
        input_text: Union[str, List[str]],
        *,
        normalize: Optional[bool] = None,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Union[List[float], List[List[float]]]:
        """Create vector embeddings for text strings.

        Args:
            input_text: Single string or list of strings
            normalize: Override instance normalize setting
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Single embedding (list of floats) or list of embeddings
        """
        single_input = isinstance(input_text, str)
        texts = [input_text] if single_input else list(input_text)

        # Clean texts
        cleaned = [t.replace("\n", " ").strip() for t in texts]

        # Encode
        embeddings = self.model.encode(
            cleaned,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize if normalize is not None else self.normalize,
            convert_to_numpy=True,
            convert_to_tensor=False,
        )

        result = embeddings.astype(np.float32).tolist()
        return result[0] if single_input else result

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries for semantic search.

        Args:
            queries: List of query strings

        Returns:
            numpy array of shape (num_queries, embedding_dim)
        """
        return self.model.encode(
            queries,
            batch_size=32,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

    def encode_corpus(
        self, corpus: List[str], batch_size: int = 32
    ) -> np.ndarray:
        """Encode corpus documents for semantic search.

        Args:
            corpus: List of document strings
            batch_size: Batch size for encoding

        Returns:
            numpy array of shape (num_docs, embedding_dim)
        """
        return self.model.encode(
            corpus,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=True,
        )


# ----------------------------------------------------------------------
# Convenience factory function
# ----------------------------------------------------------------------
def create_embedding_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> SentenceTransformerEmbedding:
    """Create an embedding model instance.

    Args:
        model_name: Model name (defaults to DEFAULT_MODEL)
        device: Device to use ('cuda', 'cpu', or None)
        **kwargs: Additional arguments to SentenceTransformerEmbedding

    Returns:
        SentenceTransformerEmbedding instance
    """
    if model_name is None:
        # Map from old OpenAI model names to sentence-transformer equivalents
        model_name = DEFAULT_MODEL

    return SentenceTransformerEmbedding(
        model_name=model_name,
        device=device,
        **kwargs,
    )
