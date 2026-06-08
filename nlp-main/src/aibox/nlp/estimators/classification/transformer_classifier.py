"""Classificador baseado na arquitetura Transformer."""

from aibox.nlp.estimators.generic.transformers import TransformerEstimator


class TransformerClassifier(TransformerEstimator):
    """Classificador Transformer.

    Para informações sobre a classe acesse
    :py:class:`~aibox.nlp.estimators.generic.transformers.TransformerEstimator`.
    """

    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        epochs: int = 2,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        random_state: int | None = None,
        do_lower_case: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            do_lower_case=do_lower_case,
            random_state=random_state,
            regression_ensure_bounds=False,
            kind="classifier",
        )
