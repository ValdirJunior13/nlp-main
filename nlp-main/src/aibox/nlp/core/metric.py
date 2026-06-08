"""Interface básica para o cálculo de métricas."""

from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """Essa é a interface para uma métrica.

    Toda métrica recebe os valores reais e os
    preditos por algum estimador e retorna
    uma numpy array com os resultados.

    .. code-block:: python

        import numpy as np
        from aibox.nlp.core.metric import Metric

        # Exemplo de uso para classes concretas
        metric = Metric()
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 0, 2])

        # Calculando métricas
        metric.compute(y_true, y_pred)
        # Out: array(0.6666667)

    """

    @abstractmethod
    def name(self) -> str:
        """Nome dessa métrica, toda
        métrica possui um nome único.

        Se dois instâncias de uma métrica possuem o mesmo nome, o valor
        do método :py:meth:`compute` é o mesmo para ambas instâncias dada
        as mesmas configurações e entradas.

        :return: nome identificador dessa métrica.
        """

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray[np.float32]:
        """Computa o valor dessa métrica para as
        entradas recebidas.

        :param y_true: valores ground-truth com formato (n_samples,).
        :param y_pred: valores preditos por algum estimator com formato (n_samples,).

        :return: array com os valores da métrica. Shape depende da métrica.
        """

    def __repr__(self) -> str:
        """Representação de uma
        métrica como string.

        :return: nome da classe seguido pelo
            nome da métrica.
        """
        return f"{self.__class__.__name__}: {self.name()}"

    def __eq__(self, other) -> bool:
        """Função de igualdade.

        Duas métricas são iguais se possuem
        o mesmo nome.

        :param other: outro objeto.

        :return: se a métrica é igual ao outro
            objeto.
        """
        if isinstance(other, Metric):
            return self.name() == other.name()

        return False
