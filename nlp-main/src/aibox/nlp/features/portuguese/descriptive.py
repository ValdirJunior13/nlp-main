"""Características descritivas do texto."""

from dataclasses import dataclass

import numpy as np
import spacy
from pyphen import Pyphen
from spacy import tokens

from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet


@dataclass(frozen=True)
class DescriptiveFeatures(DataclassFeatureSet):
    """Características descritivas sobre o
    texto.

    :param total_paragraphs: total de parágrafos.
    :param total_sentences: total de sentenças.
    :param sentences_per_paragraph: quantidade média de
        sentenças por parágrafo.
    :param syllables_per_content_word: quantidade de
        sílabas por palavras.
    :param total_words: total de palavras.
    :param words_per_sentence: quantidade média
        de palavras por sentença.
    :param sentence_length_max: tamanho máximo de
        um sentença.
    :param sentence_length_min: tamanho minimo
        de uma sentença.
    :param sentence_length_std: desvio padrão
        do tamanho das sentenças.
    :param total_stopwords: total de stopwords
        no texto.
    :param stopwords_ratio: razão entre a quantidade
        de stopwords e as palavras do texto.
    """

    total_paragraphs: float
    total_sentences: float
    sentences_per_paragraph: float
    syllables_per_content_word: float
    total_words: float
    words_per_sentence: float
    sentence_length_max: float
    sentence_length_min: float
    sentence_length_std: float
    total_stopwords: float
    stopwords_ratio: float


class DescriptiveExtractor(FeatureExtractor):
    """Extrator de características descritivas.

    :param nlp: modelo do spaCy a ser utilizado. Defaults
        to "pt_core_news_md".

    Exemplo de uso em
    :py:class:`~aibox.nlp.core.feature_extractor.FeatureExtractor`
    """

    def __init__(self, nlp: spacy.Language | None = None):
        self._nlp = nlp
        self._dict_pyphen = Pyphen(lang="pt-BR")

    @property
    def feature_set(self) -> type[DescriptiveFeatures]:
        return DescriptiveFeatures

    def extract(self, text: str, **kwargs) -> DescriptiveFeatures:
        del kwargs

        self._maybe_load_models()
        doc = self._nlp(text)
        sentences = [sent for sent in doc.sents]
        words = [word.text for word in doc if not word.is_punct]

        total_paragraphs = len(text.split("\n"))
        total_sentences = len(sentences)
        sentences_per_paragraph = total_sentences / total_paragraphs
        syllables_per_content_word = self._compute_syllables_per_content_word(doc)
        total_words = len(words)
        words_per_sentence = total_words / total_sentences
        sent_statistics = self._compute_sentence_length_max(sentences)
        sent_len_max, sent_len_min, sent_len_std = sent_statistics
        stopwords_features = self._compute_stopwords_features(doc)

        features = {
            "total_paragraphs": total_paragraphs,
            "total_sentences": total_sentences,
            "sentences_per_paragraph": sentences_per_paragraph,
            "syllables_per_content_word": syllables_per_content_word,
            "total_words": total_words,
            "words_per_sentence": words_per_sentence,
            "sentence_length_max": sent_len_max,
            "sentence_length_min": sent_len_min,
            "sentence_length_std": sent_len_std,
        }

        features.update(stopwords_features)

        return DescriptiveFeatures(**features)

    def _compute_syllables_per_content_word(self, doc: spacy.tokens.Doc) -> float:
        """Método que computa a taxa de sílabas por palavra,
        considerando apenas as palavras com conteúdo.

        :param doc: objeto Doc com o texto.
        """
        content_pos_tags = {"PROPN", "NOUN", "VERB", "ADV", "ADJ"}
        content_words = [
            word.text
            for word in doc
            if not word.is_punct and word.pos_ in content_pos_tags
        ]
        if len(content_words) == 0:
            return 0
        total_syllables = 0
        for word in content_words:
            syllables = self._dict_pyphen.inserted(word).split("-")
            total_syllables += len(syllables)
        return total_syllables / len(content_words)

    def _compute_sentence_length_max(
        self, sentences: list[tokens.Span]
    ) -> tuple[int, int, float]:
        """Método que computa características
        relacionadas ao tamanho das sentenças do texto.

        :param sentences: Lista de sentenças extraídas
            do objeto Doc contendo o texto

        :return: tamanho da maior sentença, tamanho da menor sentença
            e o desvio padrão do tamanho das sentenças do texto.
        """
        sentences_length = []
        if len(sentences) == 0:
            return 0, 0, 0
        for sent in sentences:
            tokens = [token for token in sent if not token.is_punct]
            sentences_length.append(len(tokens))
        return max(sentences_length), min(sentences_length), np.std(sentences_length)

    def _compute_stopwords_features(self, doc: tokens.Doc) -> dict[str, float]:
        """Método que computa características relacionadas
        a quantidade de stopwords no texto.

        :param doc: objeto Doc com o texto.

        :return: dicionário com características.
        """
        all_tokens = [
            token for token in doc if token.pos_ != "PUNCT" and token.pos_ != "SPACE"
        ]
        stopwords_features = {"total_stopwords": 0, "stopwords_ratio": 0}
        if len(all_tokens) == 0:
            return stopwords_features
        stopwords = [token for token in all_tokens if token.is_stop]
        stopwords_features["total_stopwords"] = len(stopwords)
        stopwords_features["stopwords_ratio"] = len(stopwords) / len(all_tokens)
        return stopwords_features

    def _maybe_load_models(self):
        if self._nlp is None:
            self._nlp = spacy.load("pt_core_news_md")
