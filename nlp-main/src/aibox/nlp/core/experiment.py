"""Interface para um experimento."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import Dataset
from .pipeline import Pipeline


@dataclass(frozen=True)
class ExperimentResult:
    """Dataclass para resultados da execução de um experimento.

    :param best_pipeline: melhor pipeline de acordo
        com o critério.
    :param best_metrics: métricas para a melhor pipeline. Chaves
        representam o nome da métrica.
    :param best_pipeline_test_predictions: predições no conjunto de testes
        para a melhor pipeline.
    :param train_df: conjunto de treinamento utilizado.
    :param test_df: conjunto de testes utilizado.
    :param metrics_history: métricas de cada uma das pipelines treinadas.
        Chaves são o nome da pipeline.
    :param pipeline_history: histórico de pipelines treinadas.
    :param extras: informações extras que podem ser retornadas
        por experimentos específicos.
    """

    best_pipeline: Pipeline
    best_metrics: dict[str, np.ndarray]
    best_pipeline_test_predictions: np.ndarray
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    metrics_history: dict[str, dict[str, np.ndarray]]
    pipeline_history: dict[str, Pipeline]
    extras: object | None = None


@dataclass(frozen=True)
class ExperimentConfiguration:
    """Dataclass com a configuração de um experimento.

    :param dataset: dataset a ser utilizado.
    :param metrics: métricas a serem calculadas
    :param best_criteria: métrica a ser utilizada
        como melhor critério.
    :param extras: configurações extras que podem
        ser utilizadas por experimentos específicos.
    """

    dataset: Dataset
    metrics: list[str]
    best_criteria: str
    extras: object | None = None


class Experiment(ABC):
    """Experimento de classificação/regressão com
    uma ou mais pipelines.
    """

    @abstractmethod
    def run(self) -> ExperimentResult:
        """Executa o experimento e retorna
        os resultados.

        :return: resultados do experimento.
        """

    @abstractmethod
    def config(self) -> ExperimentConfiguration:
        """Retorna as configurações desse experimento.

        :return: configuração do experimento.
        """
