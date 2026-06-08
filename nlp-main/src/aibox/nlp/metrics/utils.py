"""Funções utilitárias para o cálculo das métricas."""

from typing import Any, Callable

import numpy as np


def to_float32_array(
    fn: Callable[[Any, np.ndarray, np.ndarray], np.ndarray[np.float32]],
) -> Callable[[Any, np.ndarray, np.ndarray], np.ndarray[np.float32]]:
    """Wrapper que recebe um Callable
    e retorna um Callabel que converte
    as saídas para NumPy Array de floats.

    :param fn: função para ser envolopada.

    :return: nova função que converte a saída
        array de float32.
    """

    def wrapper(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray[np.float32]:
        out = fn(self, y_true=y_true, y_pred=y_pred)

        if not isinstance(out, np.ndarray):
            out = np.array(out, dtype=np.float32)

        return out.astype(np.float32)

    return wrapper
