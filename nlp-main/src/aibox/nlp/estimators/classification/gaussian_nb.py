"""Classificador Naive Bayes Gaussiano."""

import numpy as np
from sklearn.naive_bayes import GaussianNB as _GNB

from aibox.nlp.core import Estimator
from aibox.nlp.typing import ArrayLike


class GaussianNBClassifier(Estimator):
    """Classificador Gaussian Naive Bayes (NB). Essa
    classe é um wrapper de :py:class:`sklearn.naive_bayes.GaussianNB`.
    """

    def __init__(
        self,
        random_state: int | None = None,
    ):
        super().__init__(random_state=random_state)
        self._gnb = _GNB()

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        del kwargs

        preds = self._gnb.predict(X)
        return np.array(preds)

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs):
        del kwargs

        self._gnb.fit(X, y)

    @property
    def hyperparameters(self) -> dict:
        return dict(random_state=self.random_state)

    @property
    def params(self) -> dict:
        params = self._gnb.get_params()
        return {k: v for k, v in params.items() if k not in self.hyperparameters}
