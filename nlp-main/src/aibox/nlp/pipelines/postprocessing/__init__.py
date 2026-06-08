"""Módulo com funções de pós-processamento para pipelines."""

import numpy as np


def round_to_integer(x: np.ndarray) -> np.ndarray[np.int32]:
    """Arrendondamento para inteiro mais próximo.

    :param x: array de entrada.

    :return: array de inteiros.
    """
    return np.round(x).astype(np.int32)
