"""Cacheamento na extração de características."""

from abc import ABC, abstractmethod

import numpy as np

from aibox.nlp.core import FeatureExtractor, FeatureSet
from aibox.nlp.typing import TextArrayLike


class FeatureCache(ABC):
    """Interface básica para um cacheador
    de características.
    """

    @abstractmethod
    def get(self, text: str) -> FeatureSet | None:
        """Obtém o feature set para esse texto,
        caso exista no cache, ou retorna None.

        :param text: texto.

        :return: conjunto de características
            armazenadas ou None.
        """

    @abstractmethod
    def save(self, text: str, data: FeatureSet, overwrite: bool = False) -> bool:
        """Adiciona uma entrada no cache. O conjunto de características
        não precisa ser igual ao das demais entradas.

        :param text: texto.
        :param data: conjunto de características.
        :param overwrite: se devemos sobrescrever caso
            o texto já esteja no cache.

        :return: indica se foi realizado o salvamento ou não.
        """

    def as_dict(self) -> dict[str, FeatureSet]:
        """Retorna esse cache como
        um dicionário de textos para
        :py:class`~aibox.nlp.core.FeatureSet`.

        :return: dicionário com os textos cacheados.
        """


class DictFeatureCache(FeatureCache):
    def __init__(self, max_limit: int = -1):
        """Construtor. Pode ter um tamanho
        máximo para a quantidade de entradas
        armazenadas no cache.

        :param max_limit: quantidade máxima
            de entradas para armazenar. Se menor ou
            igual a 0, não são aplicados limites.
        """
        self._max = max_limit
        self._cache: dict[str, FeatureSet] = dict()

    def get(self, text: str) -> FeatureSet | None:
        return self._cache.get(text, None)

    def save(self, text: str, data: FeatureSet, overwrite: bool = False) -> bool:
        if not overwrite and text in self._cache:
            return False

        self._cache[text] = data
        self._prune_to_limit()
        return True

    def as_dict(self) -> dict[str, FeatureSet]:
        return self._cache.copy()

    def _prune_to_limit(self):
        mem_size = len(self._cache)

        if self._max <= 0 or mem_size <= self._max:
            return

        keys = list(self._cache)
        diff = mem_size - self._max

        for k in keys[0:diff]:
            del self._cache[k]


class CachedExtractor(FeatureExtractor):
    """Extrator de características com memória.

    :param extractor: extrator de características base.
    :param memory: memória auxiliar.

    Essa classe tenta primeiro obter uma entrada da memória,
    caso não existe, realiza a extração de características e
    salva no cache.
    """

    def __init__(self, extractor: FeatureExtractor, memory: FeatureCache | None = None):
        if memory is None:
            memory = DictFeatureCache()

        self._extractor = extractor
        self._memory = memory

    @property
    def feature_set(self) -> type[FeatureSet]:
        return self._extractor.feature_set

    @property
    def memory(self) -> FeatureCache:
        return self._memory

    @property
    def extractor(self) -> FeatureExtractor:
        return self._extractor

    def extract(self, text: str) -> FeatureSet:
        features = self._memory.get(text)

        if features is None:
            features = self._extractor.extract(text)
            _ = self._memory.save(text, features)

        return features

    def _batch_vectorize(self, texts: TextArrayLike, **kwargs):
        def _as_numpy(v):
            if v is not None:
                return v.as_numpy()
            return v

        # Coletar embeddings já conhecidos e coletar os índices
        #   dos não conhecidos
        embeddings = [_as_numpy(self._memory.get(t)) for t in texts]
        unk_indices = [i for i, emb in enumerate(embeddings) if emb is None]
        texts = np.array(texts, dtype=np.str_)
        unk_texts = texts[unk_indices]

        # Vetorizar os demais textos sem se preocupar
        #   em salvar resultados na memória.
        if len(unk_texts) > 0:
            unk_embeddings = self._extractor.vectorize(
                unk_texts, vector_type="numpy", **kwargs
            )

            # Atualizar embeddings
            for idx, emb in zip(unk_indices, unk_embeddings):
                embeddings[idx] = emb

            # Reconstruir conjunto de características
            #   e salvar na memória.
            sample_features = self._extractor.extract("Esse é um texto de exemplo.")
            names = sample_features.names()
            for text, emb in zip(unk_texts, unk_embeddings):
                assert emb.shape == (len(names),)
                _ = self._memory.save(
                    text.item(),
                    self._extractor.feature_set(**{k: v for k, v in zip(names, emb)}),
                )

        # Retornar conjunto total vetorizado
        return embeddings
