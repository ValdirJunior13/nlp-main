"""Entidades de extração de características."""

from abc import ABC, abstractmethod

import numpy as np
import torch

from .vectorizer import Vectorizer


class FeatureSet(ABC):
    """Representa um conjunto de características
    para um texto.

    :param `**kw_features`: conjunto de pares de
        característica e valor (:py:type:`float`).

    Todas as características possuem um nome e todos
    os resultados são ordenados seguindo a ordem
    lexicográfica.
    """

    def __init__(self, **kw_features):
        """Construtor."""

    @abstractmethod
    def as_dict(self) -> dict[str, float]:
        """Retorna os valores das características
        desse conjunto para um dado texto.

        :return: características contidas nesse
            FeatureSet para um dado texto.
        """

    def as_numpy(self) -> np.ndarray[np.float32]:
        """Retorna as características como uma
        NumPy array. Os valores de cada índice são
        correspondentes às características na ordem
        de :py:meth:`names`.

        :return: array de np.float32 representando os
            valores das características.
        """
        return np.array(list(self.as_dict().values()), dtype=np.float32)

    def as_tensor(self, device: str | None = None) -> torch.Tensor:
        """Retorna as características como um
        tensor. Os valores de cada índice são
        correspondentes às características na ordem
        de :py:meth:`names`.

        :param device: dispositivo de armazenamento.
        :type device: str, opcional

        :return: Tensor do torch representado os valores das
                características.
        """
        tensor = torch.from_numpy(self.as_numpy())

        if device is not None:
            tensor = tensor.to(device)

        return tensor

    def names(self) -> list[str]:
        """Retorna os nomes das características
        em ordem lexicográfica. Todos os outros
        métodos apresentam os valores conforme
        essa ordem.

        :return: nome das características desse conjunto.
        """
        return list(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())


class FeatureExtractor(Vectorizer):
    """Representa um extrator de características,
    que possibilita extrair um conjunto de características
    de um texto passado como entrada.

    Todo extrator de características é um :py:class:`~aibox.nlp.core.vectorizer.Vectorizer`,
    ou seja, permite converter um texto para uma representação
    numérica.

    .. code-block:: python

        from aibox.nlp.core.feature_extractor import FeatureSet, FeatureExtractor

        # Exemplo de uso para classes concretas
        extractor = FeatureExtractor()
        text = "Texto de exemplo"

        # Extração de características
        features: FeatureSet = extractor.extract(text)

        # Lendo as features como um dicionário
        print(features.as_dict())
    """

    @property
    @abstractmethod
    def feature_set(self) -> type[FeatureSet]:
        """Retorna a classe que contém o conjunto
        de características retornado por esse extrator.

        :return: classe do conjunto de características
            retornado por esse extrator.
        """
        return FeatureSet

    @abstractmethod
    def extract(self, text: str, **kwargs) -> FeatureSet:
        """Realiza a extração de características
        para o texto de entrada.

        :param text: texto.
        :param `**kwargs`: argumentos extras que pode
            ser utilizados por alguns extratores para
            controlar o processo de extração.

        :return: características para o texto de entrada.
        """

    def _vectorize(self, text: str, **kwargs) -> np.ndarray:
        """Vetorização do texto com base no método
        :py:meth:`extract`.

        :param text: texto.
        :param `**kwargs`: argumentos extras a serem
            passados para :py:meth:`extract`.

        :return: características como uma array.
        """
        feature_set = self.extract(text, **kwargs)
        return feature_set.as_numpy()
