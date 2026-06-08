"""Classificador AdaBoost."""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier as _AdaBoostClassifier

from aibox.nlp.core import Estimator
from aibox.nlp.typing import ArrayLike


class AdaBoostClassifier(Estimator):
    """Ensemble de árvores de decisão. Essa classe é
    um wrapper de :py:class:`sklearn.ensemble.AdaBoostClassifier`.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        larning_rate: float = 1.0,
        random_state: int | None = None,
    ):
        super().__init__(random_state=random_state)
        self._hyperparams = dict(
            n_estimators=n_estimators,
            learning_rate=larning_rate,
            random_state=self.random_state,
        )

        self._ada = _AdaBoostClassifier(**self._hyperparams)

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        del kwargs

        preds = self._ada.predict(X)
        return np.array(preds)

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs):
        del kwargs

        self._ada.fit(X, y)

    @property
    def hyperparameters(self) -> dict:
        return self._hyperparams

    @property
    def params(self) -> dict:
        params = self._ada.get_params()

        return {k: v for k, v in params.items() if k not in self.hyperparameters}
