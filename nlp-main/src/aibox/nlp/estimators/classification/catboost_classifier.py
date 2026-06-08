"""Classificador do catboost."""

import numpy as np
from catboost import CatBoostClassifier as _CatBoostClassifier

from aibox.nlp.core.estimator import Estimator
from aibox.nlp.typing import ArrayLike


class CatBoostClassifier(Estimator):
    """Classificador CatBoost. Essa classe
    é um wrapper de `catboost.CatBoostClassifier
    <https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier>`_.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.01,
        random_state: int | None = None,
    ):
        super().__init__(random_state=random_state)
        self._hyperparams = dict(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=random_state,
        )

        self._catboost_classifier = _CatBoostClassifier(**self._hyperparams)

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        del kwargs
        preds = self._catboost_classifier.predict(X)
        preds = np.array(preds)

        # Maybe remove inner dimensions?
        if len(preds.shape) > 1:
            preds = np.squeeze(preds, axis=-1)

        return preds

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs):
        del kwargs
        self._catboost_classifier.fit(X, y, silent=True)

    @property
    def hyperparameters(self) -> dict:
        return self._hyperparams

    @property
    def params(self) -> dict:
        params = self._catboost_classifier.get_all_params()
        return {k: v for k, v in params.items() if k not in self.hyperparameters}
