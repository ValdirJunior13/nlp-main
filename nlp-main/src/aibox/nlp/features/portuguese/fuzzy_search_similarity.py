"""Características de similaridade
entre textos baseadas no FuzzySearch.
"""

from dataclasses import dataclass

import fuzzysearch
from fuzzywuzzy import fuzz

from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet


@dataclass(frozen=True)
class FuzzySearchSimilarityFeatures(DataclassFeatureSet):
    """Características de similaridade
    do `fuzzysarch <https://github.com/taleinat/fuzzysearch>`_.
    """

    fuzz_ratio: float
    fuzz_partial_ratio: float
    fuzz_token_sort_ratio: float
    fuzz_token_set_ratio: float
    fuzz_partial_token_set_ratio: float
    fuzz_partial_token_sort_ratio: float
    fuzzysearch_near_matches: float
    fuzz_wratio: float


class FuzzySearchSimilarityExtractor(FeatureExtractor):
    """Extrator de características de similaridade.

    :param reference_text: texto de referência para
        cálculo de similaridade.

    Exemplo de uso em
    :py:class:`~aibox.nlp.core.feature_extractor.FeatureExtractor`
    """

    def __init__(self, reference_text: str):
        self._ref_text = reference_text
        self._features = {
            "fuzz_ratio": fuzz.ratio,
            "fuzz_partial_ratio": fuzz.partial_ratio,
            "fuzz_token_sort_ratio": fuzz.token_sort_ratio,
            "fuzz_token_set_ratio": fuzz.token_set_ratio,
            "fuzz_partial_token_set_ratio": fuzz.partial_token_set_ratio,
            "fuzz_partial_token_sort_ratio": fuzz.partial_token_sort_ratio,
            "fuzzysearch_near_matches": self._n_near_matches,
            "fuzz_wratio": fuzz.WRatio,
        }

    @property
    def feature_set(self) -> type[FuzzySearchSimilarityFeatures]:
        return FuzzySearchSimilarityFeatures

    @property
    def reference_text(self) -> str:
        return self._ref_text

    @reference_text.setter
    def reference_text(self, value: str):
        self._ref_text = value

    def extract(self, text: str, **kwargs) -> FuzzySearchSimilarityFeatures:
        del kwargs

        return FuzzySearchSimilarityFeatures(
            **{k: float(f(text, self._ref_text)) for k, f in self._features.items()}
        )

    @staticmethod
    def _n_near_matches(t, p) -> float:
        near = fuzzysearch.find_near_matches(t, p, max_l_dist=10)
        return float(len(near))
