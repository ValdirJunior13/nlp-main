"""Características relacionadas com
regência verbal e nominal.
"""

import json
from dataclasses import dataclass

import spacy

from aibox.nlp import resources
from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet


@dataclass(frozen=True)
class RegencyFeatures(DataclassFeatureSet):
    """Características de regência verbal
    e nominal.

    Para informações sobre as características, acessar as referências.

    Referências:
        `[1] <https://doi.org/10.5753/wapla.2023.236084>`_:
        Silva Filho, M. W. da, Nascimento, A. C. A.,
        Miranda, P., Rodrigues, L., Cordeiro, T., Isotani,
        S., Bittencourt, I. I., & Mello, R. F. (2023).
        Automated Formal Register Scoring of Student Narrative
        Essays Written in Portuguese. In Anais do II Workshop
        de Aplicações Práticas de Learning Analytics em
        Instituições de Ensino no Brasil (WAPLA 2023)
        (pp. 1–11). Workshop de Aplicações Práticas
        de Learning Analytics em Instituições de Ensino
        no Brasil. Sociedade Brasileira de Computação.
    """

    verb_regency_score: float
    nominal_regency_score: float


class RegencyExtractor(FeatureExtractor):
    """Extrator de características de regência.

    :param nlp: modelo do spaCy para ser utilizado. Defaults
        to "pt_core_news_md".

    Exemplo de uso em
    :py:class:`~aibox.nlp.core.feature_extractor.FeatureExtractor`
    """

    def __init__(self, nlp: spacy.Language = None):
        self._nlp = nlp

        root_dir = resources.path("dictionary/morph-checker.v1")
        verbs_path = root_dir.joinpath("verb_pattern.txt")
        regencies_path = root_dir.joinpath("vregence_dict.json")
        with verbs_path.open() as f1, regencies_path.open() as f2:
            self._verbs = set(
                map(lambda line: str(line).replace("\n", ""), f1.readlines())
            )
            self._verb_regencies = {
                key: set(value) for key, value in json.load(f2).items()
            }

        root_dir = resources.path("dictionary/nominal-regency.v1")
        regencies_path = root_dir.joinpath("nominal_regency_dict.json")
        with regencies_path.open() as f:
            self._name_regencies = {
                key: set(value) for key, value in json.load(f).items()
            }
            self._names = set(self._name_regencies.keys())

    @property
    def feature_set(self) -> type[RegencyFeatures]:
        return RegencyFeatures

    def extract(self, text: str) -> RegencyFeatures:
        self._maybe_load_models()
        doc = self._nlp(text)
        score_verb = self._score(
            *self._check_regency(doc, self._verbs, self._verb_regencies)
        )
        score_nominal = self._score(
            *self._check_regency(doc, self._names, self._name_regencies)
        )
        return RegencyFeatures(
            verb_regency_score=score_verb, nominal_regency_score=score_nominal
        )

    def _maybe_load_models(self):
        if self._nlp is None:
            self._nlp = spacy.load("pt_core_news_md")

    @staticmethod
    def _score(hits, errors) -> float:
        total = hits + errors
        return hits / total if total else 1.0

    @staticmethod
    def _check_regency(doc: spacy.tokens.Doc, word_set, regencies) -> tuple[int, int]:
        errors = 0
        matches = 0

        for token in doc[:-1]:
            # Check whether the token is a verb and has regency
            if not RegencyExtractor._has_regency(token, word_set):
                # if not, continue
                continue

            # Convert both lemma and next token to lower case
            lemma = token.lemma_.lower()
            next_token = doc[token.i + 1].text.lower()

            if next_token in regencies[lemma]:
                # If the following token is in the list of possible
                #   pronouns, it is a hit.
                matches += 1
            else:
                # Otherwise, it's an error/miss
                errors += 1

        return errors, matches

    @staticmethod
    def _has_regency(t: spacy.tokens.Token, target) -> bool:
        return t.lemma_.lower() in target
