"""Obtenção de extratores
de características através de nomes.
"""

from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils.aggregator import AggregatedFeatureExtractor

from .class_registry import get_class


def get_extractor(
    features: str | list[str], configs: dict | list[dict] = None
) -> FeatureExtractor:
    """Obtém um extrator de características para
    todas as características na lista `features`.

    :param features: lista com o nome dos extratores de características.
    :param configs: parâmetros para serem passados aos construtores
        dos extratores de características.

    :return: Extrator de características.
    """
    if not isinstance(features, list):
        features = [features]

    if configs is None:
        configs = [dict() for _ in features]
    elif isinstance(configs, dict):
        configs = [configs]

    features = list(features)
    configs = list(configs)
    assert len(configs) == len(features)

    extractors = []
    for feature, config in zip(features, configs):
        extractor = get_class(feature)
        assert issubclass(
            extractor, FeatureExtractor
        ), "Esse nome não corresponde a um extrator de características."
        extractors.append(extractor(**config))

    return (
        AggregatedFeatureExtractor(*extractors)
        if len(extractors) > 1
        else extractors[0]
    )
