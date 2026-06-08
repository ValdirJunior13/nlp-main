"""Teste do pacote
pipelines.
"""

import os

import pytest

from aibox.nlp.core import Pipeline
from aibox.nlp.pipelines.cohmetrix_extratrees_classification import (
    CohMetrixExtraTreesClassification,
)
from aibox.nlp.pipelines.tfidf_extratrees_classification import (
    TFIDFExtraTreesClassification,
)


@pytest.mark.skipif(
    os.environ.get("TEST_EXPENSIVE_AIBOX_NLP", None) is None,
    reason="'TEST_EXPENSIVE_AIBOX_NLP' is unset.",
)
@pytest.mark.parametrize(
    "pipeline",
    [
        CohMetrixExtraTreesClassification(random_state=42),
        TFIDFExtraTreesClassification(random_state=42),
    ],
)
def test_pipelines(pipeline: Pipeline):
    # Fit pipeline
    pipeline.fit(["Esse é um texto simples de exemplo.", "Esse é outro."], [0, 1])

    # Predict
    pipeline.predict(["Esse é um texto de testes."])
