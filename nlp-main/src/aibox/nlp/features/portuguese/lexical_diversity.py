"""Características de diversidade
léxica.
"""

import re
from collections import Counter
from dataclasses import dataclass

import spacy
import TRUNAJOD.ttr
import unidecode
from lexical_diversity import lex_div as ld
from spacy.tokens import Doc, Span

from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet


@dataclass(frozen=True)
class LexicalDiversityFeatures(DataclassFeatureSet):
    """Características de diversidade léxida..

    Para mais informações sobre as características, checar
    as referências.

    Referências:
        `[1] <https://doi.org/10.5753/sbie.2022.224736>`_:
        Oliveira, H., Miranda, P., Isotani, S., Santos,
        J., Cordeiro, T., Bittencourt, I. I., &
        Ferreira Mello, R. (2022). Estimando Coesão
        Textual em Redações no Contexto do ENEM Utilizando
        Modelos de Aprendizado de Máquina. In Anais do
        XXXIII Simpósio Brasileiro de Informática na
        Educação (SBIE 2022) (pp. 883–894). Simpósio
        Brasileiro de Informática na Educação.
        Sociedade Brasileira de Computação - SBC.
    """

    ratio_verb: float
    ratio_noun: float
    ratio_propn: float
    ratio_adj: float
    ratio_adv: float
    ratio_adp: float
    ratio_aux: float
    ratio_det: float
    yule_k: float
    hapax_legomena: float
    guiraud_index: float
    pos_dissimilarity: float
    lexical_density: float
    lexical_diversity_mtld: float
    lexical_diversity: float


class LexicalDiversityExtractor(FeatureExtractor):
    """Extrator de características de diversidade
    léxica.

    :param nlp: modelo do spaCy a ser utilizado. Defaults
        to "pt_core_news_md".

    Exemplo de uso em
    :py:class:`~aibox.nlp.core.feature_extractor.FeatureExtractor`
    """

    def __init__(self, nlp: spacy.Language = None):
        self._nlp = nlp

    @property
    def feature_set(self) -> type[LexicalDiversityFeatures]:
        return LexicalDiversityFeatures

    def extract(self, text: str, **kwargs) -> LexicalDiversityFeatures:
        del kwargs

        self._maybe_load_models()
        doc = self._nlp(text)
        tokens = [
            t.lower_
            for sent in doc.sents
            for t in sent
            if t.pos_ not in {"PUNCT", "SYM"}
        ]

        yule_k = 0
        hapax_legomena = 0
        guiraud_index = 0
        pos_dissimilarity = 0
        lexical_density = 0
        lexical_diversity_mtld = 0
        lexical_diversity = 0
        dict_ratio_pos = {
            "verb": 0,
            "noun": 0,
            "propn": 0,
            "adj": 0,
            "adv": 0,
            "adp": 0,
            "aux": 0,
            "det": 0,
        }

        if len(tokens) > 0:
            counter_tokens = Counter(tokens)
            freq_tokens = {}

            for token, freq in counter_tokens.items():
                freq_tokens[freq] = freq_tokens.get(freq, []) + [token]

            sum_frequencies = 0

            for freq, tokens_freq in freq_tokens.items():
                sum_frequencies += freq**2 * len(tokens_freq)

            total_tokens = len(tokens)
            sum_frequencies -= total_tokens
            yule_k = 10**4 * (sum_frequencies / total_tokens**2)
            content_tokens = [
                unidecode.unidecode(t.lemma_.lower())
                for t in doc
                if t.pos_ not in {"PUNCT", "SYM", "SPACE"}
            ]
            hapax_legomena = len(
                [t for t in content_tokens if content_tokens.count(t) == 1]
            )
            guiraud_index = len(set(content_tokens)) / len(content_tokens)
            pos_dissimilarity = self._compute_pos_dissimilarity(doc)
            lexical_density = self._compute_lexical_density(doc)

            try:
                lexical_diversity_mtld = TRUNAJOD.ttr.lexical_diversity_mtld(doc)
            except ZeroDivisionError:
                lexical_diversity_mtld = 0

            lexical_diversity = self._compute_lexical_diversity(doc)
            dict_ratio_pos = self._compute_pos_dist(doc)

        dict_features = {
            "yule_k": yule_k,
            "hapax_legomena": hapax_legomena,
            "guiraud_index": guiraud_index,
            "pos_dissimilarity": pos_dissimilarity,
            "lexical_density": lexical_density,
            "lexical_diversity_mtld": lexical_diversity_mtld,
            "lexical_diversity": lexical_diversity,
        }
        dict_features.update(dict_ratio_pos)

        return LexicalDiversityFeatures(**dict_features)

    def _maybe_load_models(self):
        if self._nlp is None:
            self._nlp = spacy.load("pt_core_news_md")

    @staticmethod
    def _compute_pos_dissimilarity(doc: Doc) -> float:
        """Método que computa a dissimilaridade do texto baseada na
        distribuição das classes gramaticais.

        :param doc: texto.

        :return: grau de dissimilaridade.
        """

        sent_pos_dist = []
        cont_sents = 0

        for sent in doc.sents:
            pos_dist = LexicalDiversityExtractor._pos_distribution(sent)
            sent_pos_dist.append(pos_dist)
            cont_sents += 1

        dissimilarity = 0
        sentences_size = cont_sents - 1 if cont_sents >= 2 else cont_sents

        if sentences_size == 0:
            return 0

        for i in range(len(sent_pos_dist) - 1):
            common_adj_tags = set(sent_pos_dist[i].keys()) | set(
                sent_pos_dist[i + 1].keys()
            )
            difference = 0
            totals = 0
            for pos in common_adj_tags:
                pos_dist_value = sent_pos_dist[i].get(pos, 0)
                pos_dist_value_next = sent_pos_dist[i + 1].get(pos, 0)
                difference += abs(pos_dist_value - pos_dist_value_next)
                totals += pos_dist_value + pos_dist_value_next
            dissimilarity += difference / totals
        return dissimilarity / sentences_size

    @staticmethod
    def _compute_lexical_density(doc: Doc) -> float:
        """Método que computa a proporção de algumas
        classes gramaticais no texto.

        :param doc: texto.

        :return: densidade léxica.
        """
        target = "VERB|AUX|ADJ|NOUN|PROPN|ADV"
        return LexicalDiversityExtractor._pos_ratio(doc, target)

    @staticmethod
    def _pos_distribution(sentence: Span) -> dict:
        """Método que computa a distribuição de
        classes gramáticas.

        :param sentence: sentença.

        :return: dicionário com a distribuição.
        """

        distribution = {}
        for token in sentence:
            distribution[token.pos_] = distribution.get(token.pos_, 0) + 1
        return distribution

    @staticmethod
    def _pos_ratio(doc: Doc, pos_types: str) -> float:
        """Método que computa a proporção de classes gramaticais.

        :param doc: Doc com o texto.
        :param pos_types: regex com as tags POS.

        :return: proporção.
        """

        pos_regex = re.compile(pos_types)
        total_words = 0
        total_pos_tags = 0
        for token in doc:
            if token.pos_ not in {"PUNCT", "SYM", "SPACE"}:
                total_words += 1
                if pos_regex.search(token.tag_):
                    total_pos_tags += 1
        if total_words == 0:
            return 0
        return total_pos_tags / total_words

    @staticmethod
    def _compute_lexical_diversity(doc: Doc) -> float:
        """Método que computa a diversidade léxica
        com base na medida TTR e a proporção de alguns
        tipos de palavras.

        :param doc: texto.

        :return: diversidade léxica.
        """

        all_tokens = []
        pos_dict = {}
        for sent in doc.sents:
            for t in sent:
                if t.is_alpha and not t.is_stop:
                    all_tokens.append(t.lemma_.lower())
                    pos_dict[t.pos_] = pos_dict.get(t.pos_, 0) + 1

        try:
            return ld.ttr(all_tokens)
        except ZeroDivisionError:
            return 0

    @staticmethod
    def _compute_pos_dist(doc: Doc) -> dict:
        """Método que gera um dicionário com a
        distribuição de classes gramaticais.

        :pram doc: texto.

        :return: distribuição das classes gramaticais.
        """
        pos_dict = {}
        vocab_size = 0

        for t in doc:
            if t.is_alpha and not t.is_stop:
                pos_label = t.pos_.lower()
                pos_dict[pos_label] = pos_dict.get(pos_label, 0) + 1
                vocab_size += 1

        ratio_pos_dict = {}
        pos_tags = ["verb", "noun", "propn", "adj", "adv", "adp", "aux", "det"]

        for pos_tag in pos_tags:
            if pos_tag in pos_dict:
                ratio_pos_dict["ratio_" + pos_tag] = pos_dict[pos_tag] / vocab_size
            else:
                ratio_pos_dict["ratio_" + pos_tag] = 0
        return ratio_pos_dict
