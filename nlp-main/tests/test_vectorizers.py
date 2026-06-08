"""Testes do pacote vectorizers."""

import os

import numpy as np
import pytest

from aibox.nlp.core import TrainableVectorizer, Vectorizer
from aibox.nlp.factory import available_vectorizers, get_vectorizer
from aibox.nlp.typing import ArrayLike, TextArrayLike
from aibox.nlp.vectorizers.fasttext_word_vectorizer import (
    FTAggregationStrategy,
    FTTokenizerStrategy,
)


class DummyVectorizer(Vectorizer):
    def __init__(self, state: int):
        self.state = state

    def _vectorize(self, text: str, **kwargs) -> ArrayLike:
        return np.array([self.state], dtype=np.float32)


def _get_default_configs(vectorizer_cls: str) -> tuple[tuple, dict]:
    if vectorizer_cls == "vanillaAEVectorizer":
        return (DummyVectorizer(42), DummyVectorizer(56)), dict(
            encoder_network_hidden_sizes=[64], decoder_network_hidden_sizes=[64]
        )
    return tuple(), dict()


@pytest.mark.skipif(
    os.environ.get("TEST_EXPENSIVE_AIBOX_NLP", None) is None,
    reason="'TEST_EXPENSIVE_AIBOX_NLP' is unset.",
)
@pytest.mark.parametrize("vectorizer_cls", available_vectorizers())
@pytest.mark.parametrize(
    "text",
    [
        "Esse é um texto de exemplo.",
        ["Teste com múltiplos textos."] * 20,
        np.array(["Teste com array de múltiplos textos."] * 20),
    ],
)
def test_vectorizer(vectorizer_cls: str, text: TextArrayLike):
    # Get vectorizer
    args, kwargs = _get_default_configs(vectorizer_cls)
    vectorizer = get_vectorizer(vectorizer_cls, *args, **kwargs)

    # Maybe it's trainable?
    if isinstance(vectorizer, TrainableVectorizer):
        vectorizer.fit(["Esse é um texto de treinamento."])

    # Try to vectorize input
    for kind in ["numpy", "torch"]:
        vectorizer.vectorize(text, vector_type=kind)


@pytest.mark.skipif(
    os.environ.get("TEST_EXPENSIVE_AIBOX_NLP", None) is None,
    reason="'TEST_EXPENSIVE_AIBOX_NLP' is unset.",
)
@pytest.mark.parametrize("aggregation", FTAggregationStrategy)
@pytest.mark.parametrize("tokenizer", FTTokenizerStrategy)
def test_fasttext_vectorizer(
    aggregation: FTAggregationStrategy, tokenizer: FTTokenizerStrategy
):
    # Initialize vectorizer
    vectorizer = get_vectorizer(
        "fasttextWordVectorizer", aggregation=aggregation, tokenizer=tokenizer
    )

    # Vectorize text
    out = vectorizer.vectorize("Esse é um texto de exemplo.")

    # Assertions
    if aggregation == FTAggregationStrategy.NONE:
        assert len(out.shape) == 2
        assert out.shape[-1] == 50
    elif aggregation == FTAggregationStrategy.AVERAGE:
        assert out.shape == (50,)
