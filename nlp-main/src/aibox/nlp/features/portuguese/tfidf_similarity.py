"""Características de similaridade
entre textos baseadas no TF-IDF.
"""

from dataclasses import dataclass

import polyfuzz
from polyfuzz.models import TFIDF, RapidFuzz

from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet


@dataclass(frozen=True)
class TFIDFSimilarityFeatures(DataclassFeatureSet):
    """Características de similaridade baseadas
    no TF-IDF.
    """

    rapid_fuzz: float
    tf_idf_ngram1: float
    tf_idf_ngram2: float
    tf_idf_ngram3: float
    tf_idf_ngram4: float
    tf_idf_ngram_all: float


class TFIDFSimilarityExtractor(FeatureExtractor):
    """Extrator de características de similaridade
    baseado no TF-IDF.

    :param reference_text: texto de referência para o cálculo
        de similaridade.

    Exemplo de uso em
    :py:class:`~aibox.nlp.core.feature_extractor.FeatureExtractor`
    """

    def __init__(self, reference_text: str) -> None:
        self._ref_text = reference_text
        self._models = {
            "rapid_fuzz",
            "tf_idf_ngram1",
            "tf_idf_ngram2",
            "tf_idf_ngram3",
            "tf_idf_ngram4",
            "tf_idf_ngram_all",
        }
        self._model = polyfuzz.PolyFuzz(
            [
                RapidFuzz(n_jobs=1, model_id="rapid_fuzz"),
                TFIDF(n_gram_range=(1, 1), model_id="tf_idf_ngram1"),
                TFIDF(n_gram_range=(2, 2), model_id="tf_idf_ngram2"),
                TFIDF(n_gram_range=(3, 3), model_id="tf_idf_ngram3"),
                TFIDF(n_gram_range=(4, 4), model_id="tf_idf_ngram4"),
                TFIDF(n_gram_range=(1, 5), model_id="tf_idf_ngram_all"),
            ]
        )

    @property
    def reference_text(self) -> str:
        return self._ref_text

    @reference_text.setter
    def reference_text(self, value: str) -> str:
        self._ref_text = value

    @property
    def feature_set(self) -> type[TFIDFSimilarityFeatures]:
        return TFIDFSimilarityFeatures

    def extract(self, text: str, **kwargs) -> TFIDFSimilarityFeatures:
        del kwargs

        self._model.match([text], [self._ref_text])
        matches = self._model.get_matches()
        return TFIDFSimilarityFeatures(
            **{k: matches[k]["Similarity"].item() for k in self._models}
        )
