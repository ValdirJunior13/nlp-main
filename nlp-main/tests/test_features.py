"""Testes do pacote features."""

import os

import pytest

from aibox.nlp.data.datasets.essay_br import DatasetEssayBR
from aibox.nlp.data.datasets.portuguese_narrative_essays import (
    DatasetPortugueseNarrativeEssays,
)
from aibox.nlp.factory import available_extractors, get_extractor

_SIMILARITY_EXTRACTOR = {
    "bertSimilarityBR",
    "fuzzySimilarity",
    "nilcSimilarityBR",
    "tfidfSimilarity",
}


def _config_for_cls(extractor_cls: str) -> dict:
    if extractor_cls in _SIMILARITY_EXTRACTOR:
        return dict(reference_text="Esse é o texto de referência.")

    return dict()


@pytest.mark.skipif(
    os.environ.get("TEST_EXPENSIVE_AIBOX_NLP", None) is None,
    reason="'TEST_EXPENSIVE_AIBOX_NLP' is unset.",
)
@pytest.mark.parametrize("extractor_cls", available_extractors())
@pytest.mark.parametrize(
    "text",
    [
        "Esse é um texto de exemplo.",
        "Exemplo com parágrafos.\nEsse é um outro parágrafo. Essa é mais uma sentença.\nEsse um outro parágrafo.",
        *DatasetEssayBR.load_raw(extended=True)
        .sample(1, random_state=42)
        .text.tolist(),
        *DatasetPortugueseNarrativeEssays.load_raw()
        .sample(1, random_state=42)
        .essay.tolist(),
    ],
)
def test_extractors(text: str, extractor_cls: str):
    extractor = get_extractor(extractor_cls, _config_for_cls(extractor_cls))
    extractor.extract(text)


@pytest.mark.skipif(
    os.environ.get("TEST_EXPENSIVE_AIBOX_NLP", None) is None,
    reason="'TEST_EXPENSIVE_AIBOX_NLP' is unset.",
)
@pytest.mark.parametrize("extractor_cls", _SIMILARITY_EXTRACTOR)
def test_reference_text_setter(extractor_cls: str):
    extractor = get_extractor(extractor_cls, dict(reference_text="Texto inicial."))
    extractor.reference_text = "Trocando o texto antes da chamada ao método extract."
    extractor.extract("Texto de exemplo.")
    extractor.reference_text = "Trocando o texto após a chamada ao método extract."
    extractor.extract("Outro texto de exemplo.")
