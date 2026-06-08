"""Agregação de características e extratores."""

import logging
from typing import Iterable

import numpy as np
import torch

from aibox.nlp.core import FeatureExtractor, FeatureSet
from aibox.nlp.typing import ArrayLike, TextArrayLike

logger = logging.getLogger(__name__)


class AggregatedFeatures(FeatureSet):
    """Conjunto de características agregadas.

    Essa classe permite que características oriundas de múltiplos
    extratores sejam tratadas como sendo de um único extrator.

    O conjunto de features é a união das features passadas como
    argumento.

    :param `*features`: conjuntos de características
        a serem agregados.
    :param `**kw_features`: conjunto de características como
        keyword arguments (valores devem ser float).
    """

    def __init__(self, *features: FeatureSet, **kw_features):
        self._features = features
        self._kw_features = kw_features

    def as_dict(self) -> dict[str, float]:
        # Dicionário a partir dos features set
        combined_dict = {k: v for fs in self._features for k, v in fs.as_dict().items()}

        # Atualizando com as features
        #   passadas como keyword arguments
        for k, v in self._kw_features.items():
            assert k not in combined_dict
            combined_dict[k] = float(v)

        sorted_dict = dict(sorted(combined_dict.items(), key=lambda x: x[0]))
        return sorted_dict

    @property
    def features_sets(self) -> Iterable[FeatureSet]:
        """Características base presentes
        no objeto.

        :return: características base.
        """
        return self._features


class AggregatedFeatureExtractor(FeatureExtractor):
    """Agregação de extratores de características.

    :param `*extractors`: extratores de características.
    """

    def __init__(self, *extractors) -> None:
        self._extractors = extractors

        # Variável auxiliar, possui a ordem que
        #   as características de cada extrator
        #   devem ser organizadas para seguirem
        #   a ordem lexicográfica.
        #
        # Isso é, dada a ordem padrão que se
        #   obtém ao concatenar as saídas dos
        #   extratores sequencialmente, essa
        #   ordenação converte para a ordem
        #   lexicográfica.
        #
        # === Exemplo ===
        # Características:  "a" "c" "b" "d"
        # Índices:           0   1   2   3
        # index_order:       0   2   1   3
        self._index_order = None
        self._set_index_order()

    @property
    def feature_set(self) -> type[AggregatedFeatures]:
        return AggregatedFeatures

    @property
    def extractors(self) -> list[FeatureExtractor]:
        """Extratores presentes na agregação.

        :return: extratores.
        """
        return self._extractors

    def extract(self, text: str, **kwargs) -> AggregatedFeatures:
        del kwargs

        features = [e.extract(text) for e in self._extractors]
        return AggregatedFeatures(*features)

    def _set_index_order(self):
        text = "Texto utiliário."
        features = [e.extract(text) for e in self._extractors]
        names = np.array([f_ for f in features for f_ in f.names()])
        self._index_order = np.argsort(names)
        assert (names[self._index_order] == AggregatedFeatures(*features).names()).all()

    def _batch_vectorize(self, texts: TextArrayLike, **kwargs):
        def _convert_to_numpy(arr: ArrayLike):
            if isinstance(arr, list):
                return np.array(arr, dtype=np.float32)

            if isinstance(arr, torch.Tensor):
                return arr.numpy()

            return arr

        # Lista de np arrays com shapes (n_texts, <variable>)
        logger.info("Running batch vectorize for %d extractors.", len(self.extractors))
        embedding_by_extractor = [
            _convert_to_numpy(e._batch_vectorize(texts, **kwargs))
            for e in self.extractors
        ]

        # Concatenando todos vetores no eixo mais interno,
        #   novo shape: (n_texts, total_features)
        arr = np.concatenate(embedding_by_extractor, axis=-1)

        # Reorganizando ordem mais interna para seguir a ordem
        #   dada por names
        arr = arr[:, self._index_order]

        return arr
