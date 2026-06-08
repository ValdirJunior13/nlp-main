"""Wrapper para estimadores do :py:mod:`sklearn`."""

import numpy as np

from aibox.nlp.core.estimator import Estimator
from aibox.nlp.typing import ArrayLike


class SklearnEstimator(Estimator):
    """Wrapper para estimadores scikit-like.

    :param estimator: instância de um estimador scikit-like.
        Deve possuir os métodos `fit(...)`, `predict(...)`,
        `get_params(...)` e atributo `params`.
    """

    def __init__(self, estimator) -> None:
        """Construtor."""
        self._estimator = estimator

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        del kwargs

        preds = self._estimator.predict(X)
        return np.array(preds)

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs):
        del kwargs

        self._estimator.fit(X, y)

    @property
    def hyperparameters(self) -> dict:
        return self.params

    @property
    def params(self) -> dict:
        return self._estimator.get_params()
