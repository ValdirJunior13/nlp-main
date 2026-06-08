"""Interface para estimadores."""

import random
from abc import ABC, abstractmethod

import numpy as np

from aibox.nlp.typing import ArrayLike


class Estimator(ABC):
    """Estimador.

    :param random_state: seed randômica utilizada
        para inicialização do estimador. Caso não seja passada,
        uma seed randômica é escolhida automaticamente.

    Um estimador é um objeto que pode ser
    treinado para resolver problemas de
    classificação ou regressão single-label.

    .. code-block:: python

        # Exemplo de uso (utilizar classes concretas):
        estimator = Estimator(random_state=42)

        # Realizar o treinamento
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        estimator.fit(X, y)

        # Utilizar para predições
        estimator.predict(y)
        # Out: np.array([1.0, 2.0, 3.0])
    """

    def __init__(self, random_state: int | None = None) -> None:
        if random_state is None:
            random_state = random.randint(0, 99999)

        self._seed = random_state

    @abstractmethod
    def predict(self, X: ArrayLike, **kwargs) -> np.ndarray:
        """Realiza a predição utilizando os parâmetros
        atuais do modelo.

        :param X: dados de entrada com shape (n_samples,
            n_features).
        :param `**kwargs`: parâmetros extras que podem
            ser passados para alguns estimadores.

        :return: predições com shape (n_samples,).
        """

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> None:
        """Realiza o treinamento do estimador.

        :param X: features no formato (n_samples, n_features).
        :param y: saída esperada com formato (n_samples,)
        :param `**kwargs`: parâmetros extras que podem
            ser passados para alguns estimadores.
        """

    @property
    @abstractmethod
    def hyperparameters(self) -> dict:
        """Hiper-parâmetros do modelo. Inclui
        a seed randômica. Estrutura do dicionário
        varia entre diferentes estimadores.

        :return: dicionário de hiper-parâmetros.
        """

    @property
    @abstractmethod
    def params(self) -> dict:
        """Retorna um dicionário com os parâmetros
        para esse estimador.

        Os parâmetros retornados descrevem totalmente o
        estado do modelo (e,g. pesos de uma rede,
        superfícies de decisão, estrutura da árvore de
        decisão, etc).

        :return: parâmetros do estimador.
        """

    @property
    def random_state(self) -> int:
        """Seed randômica utilizada pelo
        estimador.

        :return: seed.
        """
        return self._seed
