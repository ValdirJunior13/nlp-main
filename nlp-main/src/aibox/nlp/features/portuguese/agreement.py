"""Características relacionadas
com concordância verbal e nominal.
"""

from dataclasses import dataclass

import spacy

from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet
from aibox.nlp.lazy_loading import lazy_import
from aibox.nlp.lazy_loading.patches import get_cogroo

langtool = lazy_import("language_tool_python")


@dataclass(frozen=True)
class AgreementFeatures(DataclassFeatureSet):
    """Características de concordância verbal
    e nominal.

    :param verb_agreement_score: score entre [0, 1] que
        define a fração de acertos com relação ao total
        de instância de concordância verbal.
    :param nominal_agreement_score: score entre [0, 1] que
        define a fração de acertos com relação ao total
        de instância de concordância nominal.

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

    verb_agreement_score: float
    nominal_agreement_score: float


class AgreementExtractor(FeatureExtractor):
    """Extrator de características de concordância (verbal e nominal).

    :param nlp: modelo do spaCy a ser utilizado.
    :param cogroo: instância do CoGrOO.
    :param tool: instância do LanguageTool.

    Esse extrator utilizar o LanguageTool e CoGrOO para calcular
    *scores* sobre o uso correto das regras de concordância.

    Exemplo de uso em
    :py:class:`~aibox.nlp.core.feature_extractor.FeatureExtractor`
    """

    def __init__(self, nlp: spacy.Language = None, cogroo=None, tool=None):
        self._nlp = nlp
        self._cogroo = cogroo
        self._tool = tool
        self._va_matcher = None
        self._va_clauses = ["Number", "Person"]
        self._cogroo_rules = {
            "xml:17",
            "xml:21",
            "xml:25",
            "xml:38",
            "xml:40",
            "xml:92",
            "xml:95",
            "xml:103",
            "xml:104",
            "xml:105",
            "xml:114",
            "xml:115",
            "xml:124",
        }
        self._langtool_rules = {
            "TODOS_NUMBER_AGREEMENT",
            "CUJA_CUJO_MASCULINO_FEMININO",
            "GENERAL_NUMBER_AGREEMENT_ERRORS",
        }

    @property
    def feature_set(self) -> type[AgreementFeatures]:
        return AgreementFeatures

    def extract(self, text: str, **kwargs) -> AgreementFeatures:
        del kwargs

        # Se modelos não foram carregados, carregar.
        self._maybe_load_models()

        va_score = 0.0
        na_score = 0.0

        # Calculando nota de concordância verbal
        h, e = self._va_check(text)
        t = h + e

        if t > 0:
            va_score = h / t

        # Calculando nota de concordância nominal
        n_erros, n_rules = self._na_check(text)
        na_score = 1.0 - (n_erros / n_rules)

        return AgreementFeatures(
            verb_agreement_score=va_score, nominal_agreement_score=na_score
        )

    def _va_check(self, text: str) -> tuple[int, int]:
        doc = self._nlp(text)
        errors = 0
        matches = 0

        for span in self._va_matcher(doc, as_spans=True):
            subj = span[0].morph.to_dict()
            verb = span[1].morph.to_dict()

            # Verb agreement must occur in person and number
            for k in self._va_clauses:
                if k not in subj or k not in verb:
                    continue

                if subj[k] != verb[k]:
                    # If subject and verb doesn't match, increment error count
                    errors += 1
                else:
                    # Otherwise, increment hits/matches count
                    matches += 1

        return errors, matches

    def _na_check(self, text: str) -> tuple[int, int]:
        def _get_n_mistakes(fn, rules, fn_id):
            try:
                all_mistakes = fn()
            except Exception:
                return 0

            mistakes = filter(lambda m: m in rules, map(fn_id, all_mistakes))
            mistakes = set(mistakes)
            return len(mistakes)

        n_cogroo_mistakes = _get_n_mistakes(
            lambda: self._cogroo.grammar_check(text).mistakes,
            self._cogroo_rules,
            lambda m: m.rule_id,
        )
        n_langtool_mistakes = _get_n_mistakes(
            lambda: self._tool.check(text), self._langtool_rules, lambda m: m.ruleId
        )

        total_mistakes = n_langtool_mistakes + n_cogroo_mistakes
        total_rules = len(self._cogroo_rules) + len(self._langtool_rules)

        return total_mistakes, total_rules

    def _maybe_load_models(self):
        if self._nlp is None:
            self._nlp = spacy.load("pt_core_news_md")

        if self._cogroo is None:
            self._cogroo = get_cogroo()

        if self._tool is None:
            self._tool = langtool.LanguageTool("pt-BR")

        if self._va_matcher is None:
            self._va_matcher = spacy.matcher.Matcher(self._nlp.vocab)
            self._va_matcher.add(
                "verb", [[{"POS": {"IN": ["PRON", "NOUN", "PROPN"]}}, {"POS": "VERB"}]]
            )

    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        for k in {"_tool", "_cogroo"}:
            d[k] = None
        return d
