"""Testes do pacote experiments."""

import os

import numpy as np
import pandas as pd
import pytest

from aibox.nlp.core import Dataset, Estimator, Metric, Pipeline, Vectorizer
from aibox.nlp.experiments.simple_experiment import SimpleExperiment
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


class DummyDataset(Dataset):
    def __init__(self, as_reg: bool):
        self._df = pd.DataFrame(
            dict(text=["Não utilizado 1.", "Não utilizado 2."], target=[0, 1])
        )

        if as_reg:
            self._df["target"] = self._df.target.astype(np.float32)

    def to_frame(self):
        return self._df

    def cv_splits(self, k: int, stratified: bool, seed: int) -> list[pd.DataFrame]: ...

    def train_test_split(
        self, frac_train: float, stratified: bool, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._df, self._df


class DummyMetric(Metric):
    def name(self) -> str:
        return "Dummy"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray[np.float32]:
        return y_pred.sum()


@pytest.mark.skipif(
    os.environ.get("TEST_EXPENSIVE_AIBOX_NLP", None) is None,
    reason="'TEST_EXPENSIVE_AIBOX_NLP' is unset.",
)
@pytest.mark.parametrize("metric_maximize", [True, False])
@pytest.mark.parametrize("is_reg", [True, False])
@pytest.mark.parametrize("disable_cache", [True, False])
@pytest.mark.parametrize(
    "embeddings_dict",
    [
        None,
        {"DummyVectorizer": pd.DataFrame({"text": "Não utilizado 1.", 0: [1]})},
        {
            "DummyVectorizer": pd.DataFrame(
                {"text": [f"Não utilizado {i}." for i in range(1, 3)], 0: [1, 2]}
            )
        },
    ],
)
def test_simple_experiment(
    is_reg: bool,
    metric_maximize: bool,
    disable_cache: bool,
    embeddings_dict: dict[str, pd.DataFrame],
):
    # Initialize experiment
    experiment = SimpleExperiment(
        pipelines=[
            Pipeline(DummyVectorizer(42), DummyEstimator(80), name="Pipeline 1"),
            Pipeline(DummyVectorizer(43), DummyEstimator(81), name="Pipeline 2"),
        ],
        dataset=DummyDataset(is_reg),
        metrics=[DummyMetric()],
        criteria_best=DummyMetric(),
        criteria_maximize=metric_maximize,
        seed=8080,
        force_disable_cache=disable_cache,
        embeddings_dict=embeddings_dict,
    )

    # Expected results
    expected_best = f"Pipeline {1 + int(metric_maximize)}"
    expected_best_metric = 80 + int(metric_maximize)
    size_feature_df = 0

    # If has initial cache, will return the same cache
    #   or augmented version (vectorizer is the same)
    if embeddings_dict:
        emb_dict_size = 1
    else:
        emb_dict_size = 1 - int(disable_cache)

    # Run experiment
    result = experiment.run()

    # Assertions
    assert result.extras is not None
    assert np.allclose(result.best_metrics["Dummy"], expected_best_metric)
    assert result.best_pipeline.name == expected_best
    assert len(result.extras.df_features) == size_feature_df
    assert len(result.extras.embeddings) == emb_dict_size
