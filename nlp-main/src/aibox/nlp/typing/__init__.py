"""Definições comuns de type aliases."""

from os import PathLike
from typing import Callable, TypeAlias

import numpy as np
import pandas as pd
import torch

#: Define a noção de array-like numéricos.
ArrayLike: TypeAlias = (
    list[int]
    | list[float]
    | np.ndarray[np.int32]
    | np.ndarray[np.float32]
    | torch.Tensor
)

#: Define array-like de textos.
TextArrayLike: TypeAlias = list[str] | np.ndarray[np.str_]

#: Define uma função de pós-processamento.
PostProcessing: TypeAlias = Callable[[np.ndarray], np.ndarray]

#: Define valores que podem ser convertidos para :py:class:`pandas.DataFrame`
DataFrameLike: TypeAlias = pd.DataFrame | PathLike | str
