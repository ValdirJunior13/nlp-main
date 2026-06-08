"""Classificador utilizando k-NN."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as _KNClassifier

from aibox.nlp.core import Estimator
from aibox.nlp.typing import ArrayLike


class KNeighborsClassifier(Estimator):
    """Classificar baseado nos vizinhos mais próximos. Essa classe é
    um wrapper de :py:class:`sklearn.neighbors.KNeighborsClassifier`.
    """

    def __init__(
        self,
        n_neighbors: int = 2,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: float = 2.0,
        metric: str = "minkowski",
        random_state: int | None = None,
    ):
        super().__init__(random_state=random_state)
        self._hyperparams = dict(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
        )

        self._kn = _KNClassifier(**self._hyperparams)

    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        del kwargs

        preds = self._kn.predict(X)
        return np.array(preds)

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs):
        del kwargs

        self._kn.fit(X, y)

    @property
    def hyperparameters(self) -> dict:
        return dict(**self._hyperparams, random_state=self.random_state)

    @property
    def params(self) -> dict:
        params = self._kn.get_params()

        return {k: v for k, v in params.items() if k not in self.hyperparameters}
