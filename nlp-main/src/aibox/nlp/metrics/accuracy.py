"""Implementação da métrica de acurácia."""

import numpy as np

from aibox.nlp.core import Metric

from . import utils


class Accuracy(Metric):
    """Métrica para cálculo da Acurácia."""

    @utils.to_float32_array
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray[np.float32]:
        return (y_true == y_pred).sum() / len(y_true)

    def name(self) -> str:
        return "Accuracy"
