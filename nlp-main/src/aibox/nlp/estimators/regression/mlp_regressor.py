"""Regressor baseado na arquitetura MLP."""

import typing

from aibox.nlp.estimators.generic.mlp import MLPEstimator


class MLPRegressor(MLPEstimator):
    """Regressor MLP.

    Para informações sobre a classe acesse
    :py:class:`~aibox.nlp.estimators.generic.mlp.MLPEstimator`.
    """

    def __init__(
        self,
        hidden_layers: list[int] = [256, 128, 64],
        epochs: int = 10,
        dropout_prob: float | list[float | None] | None = None,
        learning_rate: float = 1e-4,
        optim_params: dict = dict(),
        optim: typing.Literal["adam", "adamw", "rmsprop", "adagrad", "sgd"] = "adamw",
        regression_ensure_bounds=False,
        random_state: int | None = None,
        train_batch_size: int = 64,
        device: str = None,
    ):
        super().__init__(
            hidden_layers=hidden_layers,
            kind="regressor",
            epochs=epochs,
            dropout_prob=dropout_prob,
            learning_rate=learning_rate,
            optim_params=optim_params,
            optim=optim,
            regression_ensure_bounds=regression_ensure_bounds,
            train_batch_size=train_batch_size,
            random_state=random_state,
            device=device,
        )
