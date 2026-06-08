"""Esse é um teste para garantir
que os identificadores das features são
únicos em toda a biblioteca.
"""

import importlib
import inspect
import pkgutil
from dataclasses import fields

import pytest

import aibox.nlp.features.portuguese
from aibox.nlp.core import FeatureSet
from aibox.nlp.features.utils import DataclassFeatureSet
from aibox.nlp.features.utils.aggregator import AggregatedFeatures
from aibox.nlp.features.utils.dict_feature_set import DictFeatureSet

_UTILITY_FEATURE_SETS = {DataclassFeatureSet}
_SKIP_MODULES = {"utils"}


def test_no_duplicated_features():
    """Realiza um teste para garantir que
    todos os identificadores de características
    são únicos.
    """
    global_ids = []

    # Supõem que a estrutura do pacote `nlpbox.features.portuguese` é:
    # __init__.py
    # feature1.py
    # feature2.py
    # feature3.py
    # subpackage
    #   non_feature.py
    # feature4.py
    # ....
    modules_info = [
        m for m in pkgutil.walk_packages(aibox.nlp.features.portuguese.__path__)
    ]

    for module_info in modules_info:
        name = module_info.name
        if name in _SKIP_MODULES:
            # Alguns módulos não devem ser checados
            continue

        # Importando módulo
        module = importlib.import_module(f"aibox.nlp.features.portuguese.{name}")

        # Coletando todas as classes presentes nesse módulo
        classes = [m for _, m in inspect.getmembers(module, predicate=inspect.isclass)]

        # Caso esse módulo não possua classe,
        #   não precisamos testar.
        if len(classes) < 1:
            continue

        # Coletando todas as classes que implementam um FeatureSet
        # OBS:. excluímos FeatureSet's utilitários que podem ter
        #   sido importados
        feature_sets = [
            c
            for c in classes
            if (c not in _UTILITY_FEATURE_SETS) and issubclass(c, FeatureSet)
        ]

        # Condição da biblioteca: 1 módulo só pode ter 1 feature set
        assert len(feature_sets) == 1, f"{name}, {feature_sets}"

        # Obtemos o feature set
        fs = feature_sets[0]

        # Obtemos os identificadores de features
        ids = [field.name for field in fields(fs)]

        # Garantimos que não existe duplicada
        assert len(ids) == len(set(ids))

        # Salvamos os nomes de features desse feature set na lista
        #   de identificadores global.
        global_ids.extend(ids)

    # Garantimos que não existem IDs duplicados
    assert len(global_ids) == len(set(global_ids))


@pytest.mark.parametrize("fscls", [AggregatedFeatures, DictFeatureSet])
@pytest.mark.parametrize(
    "kw_features,names",
    [
        (dict(b=0.4, a=1.0), ["a", "b"]),
        (dict(c=1.0, b=2.0, d=3.0, e=4.0), ["b", "c", "d", "e"]),
    ],
)
def test_feature_set_construction(
    fscls: type[FeatureSet], kw_features: dict[str, float], names: list[str]
):
    features = fscls(**kw_features)
    as_dict = features.as_dict()
    assert len(features) == len(kw_features)
    assert all(k in as_dict for k in kw_features)
    assert names == list(as_dict)


@pytest.mark.parametrize(
    "features", [[DictFeatureSet(a=2.0, c=3.0), DictFeatureSet(e=5.0)], []]
)
@pytest.mark.parametrize(
    "kw_features",
    [
        dict(b=0.4),
        dict(),
    ],
)
def test_aggregated_features(features: list[FeatureSet], kw_features: dict[str, float]):
    # Construindo features que devem estar presentes
    names = [name for f in features for name in f.names()]
    names += list(kw_features)
    names = sorted(names)

    # Caso onde não temos features
    if not names:
        return

    data = AggregatedFeatures(*features, **kw_features)
    as_dict = data.as_dict()
    assert len(data) == len(names)
    assert names == list(as_dict)


@pytest.mark.parametrize("base", [dict(a=1.0), dict()])
@pytest.mark.parametrize(
    "kw_features",
    [
        dict(b=0.4),
        dict(),
    ],
)
def test_dict_features(base: dict[str, float], kw_features: dict[str, float]):
    # Construindo features que devem estar presentes
    names = sorted(list(base) + list(kw_features))

    # Caso onde não temos features
    if not names:
        return

    data = DictFeatureSet(base, **kw_features)
    as_dict = data.as_dict()
    assert len(data) == len(names)
    assert names == list(as_dict)
