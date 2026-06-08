"""Características clássicas de similaridade
entre textos adotadas pelo NILC-Metrix.
"""

from dataclasses import dataclass

import spacy
from gensim.models import KeyedVectors

from aibox.nlp import resources
from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet


@dataclass(frozen=True)
class NILCSimilarityFeatures(DataclassFeatureSet):
    """Características de similaridade
    baseadas no `NILC-Metrix <http://fw.nilc.icmc.usp.br:23380/nilcmetrix>`_.

    :param similarity_jaccard: similaridade Jaccard entre os
        tokens do texto de referência e texto de entrada.
    :param similarity_dice: similaridade dice entre os
        tokens do texto de referência e texto de entrada.
    :param similarity_cosine_cbow: similaridade cosseno
        com o Word2Vec CBOW do NILC-Metrix.
    :param similarity_word_movers_cbow: similaridade Word Movers
        com o Word2Vec CBOW do NILC-Metrix.
    """

    similarity_jaccard: float
    similarity_dice: float
    similarity_cosine_cbow: float
    similarity_word_movers_cbow: float


class NILCSimilarityExtractor(FeatureExtractor):
    """Extrator de características de similaridade
    baseadas no NILC-Metrix.

    :param reference_text: texto de referência para
        similaridade.
    :param nlp: modelo do spaCy a ser utilizado.

    Exemplo de uso em
    :py:class:`~aibox.nlp.core.feature_extractor.FeatureExtractor`
    """

    def __init__(self, reference_text: str, nlp: spacy.Language = None):
        self._wv = None
        self._ref_doc = None
        self._ref_tokens = None
        self._ref_text = reference_text
        self._nlp = nlp

    @property
    def feature_set(self) -> type[NILCSimilarityFeatures]:
        return NILCSimilarityFeatures

    @property
    def reference_text(self) -> str:
        return self._ref_doc.text

    @reference_text.setter
    def reference_text(self, value: str) -> None:
        self._maybe_load_models()
        self._ref_doc = self._nlp(value)
        self._ref_tokens = [t.text for t in self._ref_doc]

    def extract(self, text: str, **kwargs) -> NILCSimilarityFeatures:
        del kwargs

        self._maybe_load_models()
        doc = self._nlp(text)
        text_tokens = [t.text for t in doc]

        jaccard = self.jaccard(text_tokens, self._ref_tokens)
        dice = self.dice(text_tokens, self._ref_tokens)
        cos_cbow = 0.0
        word_movers_cbow = 0.0
        x = self._validate(text_tokens)
        y = self._validate(self._ref_tokens)

        try:
            cos_cbow = self._cosine_similarity(x, y)
        except Exception:
            pass

        try:
            word_movers_cbow = self._word_movers_distance(x, y)
        except Exception:
            pass

        return NILCSimilarityFeatures(
            similarity_jaccard=jaccard,
            similarity_dice=dice,
            similarity_cosine_cbow=cos_cbow,
            similarity_word_movers_cbow=word_movers_cbow,
        )

    @staticmethod
    def jaccard(tokens_x: list[str], tokens_y: list[str]) -> float:
        """Similaridade de jaccard.

        :param tokens_x: lista de sentenças.
        :param tokens_y: lista de sentenças.

        :return: similaridade.
        """
        x, y = set(tokens_x), set(tokens_y)
        if not x and not y:
            return 0.0
        return len(x & y) / len(x | y)

    @staticmethod
    def dice(tokens_x: list[str], tokens_y: list[str]) -> float:
        """Similaridade de dice.

        :param tokens_x: lista de sentenças.
        :param tokens_y: lista de sentenças.

        :return: similaridade.
        """
        x, y = set(tokens_x), set(tokens_y)
        if not x and not y:
            return 0.0
        return (2 * len(x & y)) / (len(x) + len(y))

    def _validate(self, sent: list[str]) -> list[str]:
        return [t for t in sent if t in self._wv]

    def _cosine_similarity(self, x: list[str], y: list[str]) -> float:
        return self._wv.n_similarity(x, y)

    def _word_movers_distance(self, x: list[str], y: list[str]) -> float:
        x = " ".join(x)
        y = " ".join(y)
        return self._wv.wmdistance(x, y)

    def _maybe_load_models(self):
        if self._nlp is None:
            self._nlp = spacy.load("pt_core_news_md")

        if self._wv is None:
            root_dir = resources.path("external/nilc-word2vec50.v1")
            word2vec_path = root_dir.joinpath("cbow_s50.bin")
            self._wv = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)

        if self._ref_doc is None:
            self._ref_doc = self._nlp(self._ref_text)

        if self._ref_tokens is None:
            self._ref_tokens = [t.text for t in self._ref_doc]
