"""Regressor SVR."""

import numpy as np
from sklearn.svm import SVR as _SVR

from aibox.nlp.core import Estimator
from aibox.nlp.typing import ArrayLike


class SVR(Estimator):
    """Classificador Support-Vector Regression (SVR). Essa
    classe é um wrapper de :py:class:`sklearn.svm.SVR`.
    """

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=0.001,
        cache_size=200,
        max_iter=-1,
        shrinking=True,
        random_state: int | None = None,
    ):
        super().__init__(random_state=random_state)
        self._hyperparams = dict(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            tol=tol,
            cache_size=cache_size,
            verbose=False,
            max_iter=max_iter,
        )

        self._svr = _SVR(**self._hyperparams)

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        del kwargs

        preds = self._svr.predict(X)
        return np.array(preds)

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs):
        del kwargs

        self._svr.fit(X, y)

    @property
    def hyperparameters(self) -> dict:
        return dict(**self._hyperparams, random_state=self.random_state)

    @property
    def params(self) -> dict:
        params = self._svr.get_params()
        return {k: v for k, v in params.items() if k not in self.hyperparameters}
