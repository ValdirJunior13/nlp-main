"""Características relacionadas com ortografia."""

import re
from dataclasses import dataclass

import spacy

from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet
from aibox.nlp.lazy_loading import lazy_import

langtool = lazy_import("language_tool_python")


@dataclass(frozen=True)
class OrtographyFeatures(DataclassFeatureSet):
    """Características de domínio ortográfico.

    Para mais informações sobre as características, acessar
    as referências.

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

    ortography_score: float


class OrthographyExtractor(FeatureExtractor):
    """Extrator de características ortográficas.

    :param tool: instância do LanguageTool.

    Exemplo de uso em
    :py:class:`~aibox.nlp.core.feature_extractor.FeatureExtractor`
    """

    def __init__(self, tool: langtool.LanguageTool = None):
        self._tool = tool
        self._tokenizer_pattern = re.compile(r"\s+")
        self._tokenizer = spacy.tokenizer.Tokenizer(
            spacy.blank("pt").vocab, token_match=self._tokenizer_pattern.match
        )
        self._rules = {
            "Encontrado possível erro de ortografia.",
            "Palavras estrangeiras com diacríticos",
            "Uso de apóstrofe para palavras no plural",
            "Femininos irregulares",
            "Erro ortográfico: Abreviaturas da internet",
            "Palavras raras facilmente confundidas",
            "Palavras raras: Capitalização de nomes geográficos",
        }

    @property
    def feature_set(self) -> type[OrtographyFeatures]:
        return OrtographyFeatures

    def extract(self, text: str, **kwargs) -> OrtographyFeatures:
        del kwargs

        self._maybe_load_models()

        # Limpeza do texto (pontuação, whitespace, etc)
        text = self._cleaner(text)

        # Obter tokens (palavras)
        tokens = [token.text for token in self._tokenizer(text)]

        # Inicializando features
        score = 0.0

        # Calcular os erros ortográficos presentes no texto
        errors = self._check(text)
        score = 1.0 - (len(errors) / len(tokens))

        return OrtographyFeatures(ortography_score=score)

    def _check(self, text: str) -> list[dict[str, str]]:
        # Realizar uma checagem no texto utilizando o LanguageTool
        matches = self._tool.check(text)

        # Lista dos erros
        errors = []

        for match in matches:
            # Caso sejam encontrados erros de ortografia:
            if match.message in self._rules:
                error_dict = {}
                correct_token = ""

                # Obter token/palavra original errôneo
                offset = match.offset
                length = match.errorLength
                token = text[offset : offset + length]

                # Buscar se existe um candidato à substituição
                if len(match.replacements) > 0:
                    correct_token = match.replacements[0]

                # Adicionar informações no dicionário para esse erro
                error_dict["token"] = token
                error_dict["correct_token"] = correct_token

                # Adicionar esse erro na lista
                errors.append(error_dict)

        return errors

    def _maybe_load_models(self):
        if self._tool is None:
            self._tool = langtool.LanguageTool("pt-BR")

    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        for k in {"_tool"}:
            d[k] = None
        return d

    @staticmethod
    def _cleaner(text: str):
        text = text.strip()
        text = re.sub(r"[!\.,—]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text[0].upper() + text[1:]
