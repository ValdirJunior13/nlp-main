"""Regressor Catboost."""

import numpy as np
from catboost import CatBoostRegressor as _CatBoostRegressor

from aibox.nlp.core.estimator import Estimator
from aibox.nlp.typing import ArrayLike


class CatBoostRegressor(Estimator):
    """Regressor CatBoost. Essa classe é
    um wrapper de :py:class:`catboost.CatBoostRegressor`.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.01,
        random_state: int | None = None,
    ):
        super().__init__(random_state=random_state)
        self._hyperparams = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=self.random_state,
            loss_function="RMSE",
        )
        self._catboost_regressor = _CatBoostRegressor(**self._hyperparams)

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        del kwargs

        preds = self._catboost_regressor.predict(X)
        return np.array(preds)

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs):
        del kwargs

        self._catboost_regressor.fit(X, y, silent=True)

    @property
    def hyperparameters(self) -> dict:
        return self._hyperparams

    @property
    def params(self) -> dict:
        params = self._catboost_regressor.get_all_params()
        return {k: v for k, v in params.items() if k not in self.hyperparameters}
