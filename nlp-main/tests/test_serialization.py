"""Testes do pacote serialization."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aibox.nlp.core import Estimator, Pipeline, Vectorizer
from aibox.nlp.serialization.pipeline import load_pipeline, save_pipeline
from aibox.nlp.typing import ArrayLike


class DummyVectorizer(Vectorizer):
    def __init__(self, state: int):
        self.state = state

    def _vectorize(self, text: str, **kwargs) -> ArrayLike:
        return np.array([self.state], dtype=np.float32)


class DummyEstimator(Estimator):
    def __init__(self, random_state: int):
        super().__init__(random_state=random_state)

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        return np.array([self.random_state], dtype=np.float32)

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs): ...

    @property
    def hyperparameters(self):
        return dict(seed=self.random_state)

    @property
    def params(self):
        return dict(state=self.random_state)


@pytest.mark.parametrize("state", [42, 0, 241])
def test_pipeline_serialization(state: int):
    # Construct pipeline
    pipeline = Pipeline(
        DummyVectorizer(state + 1), DummyEstimator(state + 2), name="DummyPipeline"
    )

    # Fit pipeline
    pipeline.fit(["Não utilizado."], [0])

    # Serialize
    with tempfile.TemporaryDirectory() as dirname:
        # Construct output path
        output = Path(dirname).joinpath("pipeline.joblib")

        # Save pipeline
        assert not output.exists()
        save_pipeline(pipeline, output)
        assert output.exists()

        # Load pipeline
        loaded_pipeline = load_pipeline(output)

    # Guarantee consistency
    assert pipeline.vectorizer.state == loaded_pipeline.vectorizer.state
    assert pipeline.estimator.params == loaded_pipeline.estimator.params
    assert (
        pipeline.estimator.hyperparameters == loaded_pipeline.estimator.hyperparameters
    )
    assert np.allclose(
        pipeline.predict(["Não utilizado."]),
        loaded_pipeline.predict(["Não utilizado."]),
    )
