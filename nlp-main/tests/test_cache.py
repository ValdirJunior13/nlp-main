"""Testes unitários para
o cacheamento de vetorizadores
e extratores de características.
"""

import operator
from dataclasses import dataclass

import numpy as np
import pytest

from aibox.nlp.cache.features import CachedExtractor, DictFeatureCache
from aibox.nlp.cache.mixed_feature_cache import MixedFeatureCache
from aibox.nlp.cache.vectorizers import CachedVectorizer, DictVectorizerCache
from aibox.nlp.core import FeatureExtractor, Vectorizer
from aibox.nlp.features.utils import DataclassFeatureSet


@dataclass(frozen=True)
class DummyFeatureSet(DataclassFeatureSet):
    feature_a: float
    feature_b: float
    feature_c: float


@dataclass(frozen=True)
class DummyFeatureSetB(DataclassFeatureSet):
    feature_d: float
    feature_e: float


class DummyExtractor(FeatureExtractor):
    @property
    def feature_set(self) -> DummyFeatureSet:
        return DummyFeatureSet

    def extract(self, text: str, **kwargs) -> DummyFeatureSet:
        return DummyFeatureSet(feature_a=1.0, feature_b=2.0, feature_c=3.0)


class DummyVectorizer(Vectorizer):
    def _vectorize(self, text: str, **kwargs) -> np.array:
        return np.array([0.0, 1.0, 2.0])


@pytest.mark.parametrize(
    "target_cls,data,eq",
    [
        (DictFeatureCache, DummyFeatureSet(1.0, 2.0, 3.0), operator.eq),
        (DictVectorizerCache, np.array([0.0, 1.0, 3.0]), np.array_equal),
    ],
)
@pytest.mark.parametrize("limit", [-1, 0, 4])
def test_dict_cache(target_cls, data, eq, limit: int):
    # Inicializando o cache com o limite definido
    cache = target_cls(max_limit=limit)
    text = "A"

    # Salvando inicialmente um texto
    assert cache.save(text, data)
    assert eq(cache.get(text), data)

    # Sobrescrita de dados
    assert not cache.save(text, data)
    assert cache.save(text, data, overwrite=True)

    # Teste de limite de cache
    if limit > 0:
        for i in range(limit + 1):
            assert cache.save(str(i), data)
        assert cache.get(text) is None
        assert eq(cache.get(str(limit)), data)


@pytest.mark.parametrize(
    "mem_cls,cached_cls,ext_cls,data,eq",
    [
        (
            DictFeatureCache,
            CachedExtractor,
            DummyExtractor,
            DummyFeatureSet(4.0, 5.0, 6.0),
            operator.eq,
        ),
        (
            DictVectorizerCache,
            CachedVectorizer,
            DummyVectorizer,
            np.array([3.0, 4.0, 5.0]),
            np.array_equal,
        ),
    ],
)
def test_cached_obj(mem_cls, cached_cls, ext_cls, data, eq):
    # Inicializando dados e memória
    memory = mem_cls()
    text = "A"

    # Inicializando extrator
    obj = cached_cls(ext_cls(), memory)
    try:
        method = obj.extract
    except AttributeError:
        method = obj.vectorize

    # Testando que escreve para memória
    method(text)
    assert memory.get(text) is not None

    # Testando que lê da memória
    memory.save(text, data, overwrite=True)
    assert eq(method(text), data)

    # Testando que funciona corretamente
    #   com batches
    texts = ["B", "C", "D"]
    assert all(memory.get(t) is None for t in texts)
    obj.vectorize(texts, num_workers=0)
    assert all(memory.get(t) is not None for t in texts)


@pytest.mark.parametrize(
    "target_features",
    [
        {"feature_a", "feature_b"},
        {"feature_d", "feature_e"},
        {"feature_a", "feature_e"},
    ],
)
def test_mixed_feature_cache(target_features: set[str]):
    # Inicializar memória com "schema"
    memory = MixedFeatureCache(target_features, initial_cache=None, max_limit=-1)
    data1 = DummyFeatureSet(1.0, 2.0, 3.0)
    data2 = DummyFeatureSetB(4.0, 5.0)
    text = "A"

    # Salvando dados de 1
    memory.save(text, data1)
    should_have_cache = all(t in data1.names() for t in target_features)
    assert (memory.get(text) is not None) == should_have_cache

    # Salvando dados de 2
    memory.save(text, data2)
    should_have_cache = should_have_cache or all(
        t in data2.names() + data1.names() for t in target_features
    )
    assert (memory.get(text) is not None) == should_have_cache

    # Garantindo que o retorno possui os dados requisitados
    assert all(t in memory.get(text).names() for t in target_features)
