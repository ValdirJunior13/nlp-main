"""Testes que garantem
que classes serializáveis
suportam pickle.
"""

import io
import os
import pickle

import joblib
import pytest

from aibox.nlp.factory import (
    available_estimators,
    available_extractors,
    available_vectorizers,
    get_estimator,
    get_extractor,
    get_vectorizer,
)


def _test_serialization(serializer, obj):
    # Save
    with io.BytesIO() as buffer:
        serializer.dump(obj, buffer)
        data = buffer.getvalue()

    # Load
    with io.BytesIO(data) as buffer:
        _ = serializer.load(buffer)


@pytest.mark.skipif(
    os.environ.get("TEST_EXPENSIVE_AIBOX_NLP", None) is None,
    reason="'TEST_EXPENSIVE_AIBOX_NLP' is unset.",
)
@pytest.mark.parametrize("serializer", [pickle, joblib])
@pytest.mark.parametrize("extractor", available_extractors())
def test_extractors(serializer, extractor: str):
    _test_serialization(
        serializer,
        get_extractor(
            extractor,
            (
                dict(reference_text="Texto de referência.")
                if "similarity" in extractor.lower()
                else dict()
            ),
        ),
    )


@pytest.mark.skipif(
    os.environ.get("TEST_EXPENSIVE_AIBOX_NLP", None) is None,
    reason="'TEST_EXPENSIVE_AIBOX_NLP' is unset.",
)
@pytest.mark.parametrize("serializer", [pickle, joblib])
@pytest.mark.parametrize("vectorizer", available_vectorizers())
def test_vectorizers(serializer, vectorizer: str):
    _test_serialization(serializer, get_vectorizer(vectorizer))


@pytest.mark.skipif(
    os.environ.get("TEST_EXPENSIVE_AIBOX_NLP", None) is None,
    reason="'TEST_EXPENSIVE_AIBOX_NLP' is unset.",
)
@pytest.mark.parametrize("serializer", [pickle, joblib])
@pytest.mark.parametrize("estimator", available_estimators())
def test_estimators(serializer, estimator: str):
    # Instanciando e treinando estimador
    estimator = get_estimator(estimator)
    estimator.fit([[1.0], [2.0], [3.0]], [1, 2, 3])

    _test_serialization(serializer, estimator)
