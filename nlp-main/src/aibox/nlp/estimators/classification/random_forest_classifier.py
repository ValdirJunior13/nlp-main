"""Classificador Random Forest."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier as _RFClassifier

from aibox.nlp.core import Estimator
from aibox.nlp.typing import ArrayLike


class RandomForestClassifier(Estimator):
    """Ensemble de árvores de decisão. Essa classe é
    um wrapper de :py:class:`sklearn.ensemble.RandomForestClassifier`.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_features: str | None = "sqrt",
        bootstrap: bool = False,
        class_weight: str | dict = None,
        random_state: int | None = None,
    ):
        super().__init__(random_state=random_state)
        self._hyperparams = dict(
            n_estimators=n_estimators,
            criterion=criterion,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=class_weight,
            random_state=self.random_state,
        )

        self._rf = _RFClassifier(verbose=0, warm_start=False, **self._hyperparams)

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        del kwargs

        preds = self._rf.predict(X)
        return np.array(preds)

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs):
        del kwargs

        self._rf.fit(X, y)

    @property
    def hyperparameters(self) -> dict:
        return self._hyperparams

    @property
    def params(self) -> dict:
        params = self._rf.get_params()

        return {k: v for k, v in params.items() if k not in self.hyperparameters}
