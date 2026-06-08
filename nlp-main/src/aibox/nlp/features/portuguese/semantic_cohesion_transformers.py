"""Características de coesão semântica
usando Sentence Transformers.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import spacy
from gensim.matutils import cossim, full2sparse, sparse2full
from scipy.linalg import pinv
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc

from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet
from aibox.nlp.typing import TextArrayLike


@dataclass(frozen=True)
class SemanticFeaturesTransformers(DataclassFeatureSet):
    """Características de coesão semântica versão
    SentenceTransformer.
    """

    lsa_adj_mean_embedding: float
    lsa_adj_std_embedding: float
    lsa_all_mean_embedding: float
    lsa_all_std_embedding: float
    lsa_givenness_mean_embedding: float
    lsa_givenness_std_embedding: float
    lsa_paragraph_mean_embedding: float
    lsa_paragraph_std_embedding: float
    lsa_span_mean_embedding: float
    lsa_span_std_embedding: float


class SemanticExtractorTransformers(FeatureExtractor):
    """Extrator de características de coesão semãntica
    versão SentenceTransformer.

    :param nlp: modelo do spaCy para ser utilizado. Defaults
        to "pt_core_news_md".
    :param model: modelo do SentenceTransformer. Defaults
        to "ricardo-filho/bert-base-portuguese-cased-nli-assin-2".
    :param dims: quantidade de dimensões dos embeddings.
    :param device: dispositivo padrão a set utilizado.

    Exemplo de uso em
    :py:class:`~aibox.nlp.core.feature_extractor.FeatureExtractor`
    """

    def __init__(
        self,
        nlp: spacy.Language | None = None,
        model: SentenceTransformer = None,
        dims: int = None,
        device: str | None = None,
    ):
        self._nlp = nlp
        self._model = model
        self._device = device
        self._dims = None

    @property
    def feature_set(self) -> type[SemanticFeaturesTransformers]:
        return SemanticFeaturesTransformers

    def extract(self, text: str, **kwargs) -> SemanticFeaturesTransformers:
        del kwargs

        self._maybe_load_models()
        doc = self._nlp(text)
        sentences = [sent.text for sent in doc.sents]
        features = {
            "lsa_adj_mean_embedding": 0,
            "lsa_adj_std_embedding": 0,
            "lsa_all_mean_embedding": 0,
            "lsa_all_std_embedding": 0,
            "lsa_givenness_mean_embedding": 0,
            "lsa_givenness_std_embedding": 0,
            "lsa_paragraph_mean_embedding": 0,
            "lsa_paragraph_std_embedding": 0,
            "lsa_span_mean_embedding": 0,
            "lsa_span_std_embedding": 0,
        }

        if len(sentences) > 1:
            adj_sent_mean, adj_sent_std = self._compute_adjacent_sentences(sentences)
            all_sent_mean, all_sent_std = self._compute_all_sentences(sentences)
            givenness_mean, givenness_std = self._compute_sentences_givenness(sentences)
            paragraph_mean, paragraph_std = self._compute_paragraphs(doc)
            span_mean, span_std = self._compute_span(sentences)
            features["lsa_adj_mean_embedding"] = adj_sent_mean
            features["lsa_adj_std_embedding"] = adj_sent_std
            features["lsa_all_mean_embedding"] = all_sent_mean
            features["lsa_all_std_embedding"] = all_sent_std
            features["lsa_givenness_mean_embedding"] = givenness_mean
            features["lsa_givenness_std_embedding"] = givenness_std
            features["lsa_paragraph_mean_embedding"] = paragraph_mean
            features["lsa_paragraph_std_embedding"] = paragraph_std
            features["lsa_span_mean_embedding"] = span_mean
            features["lsa_span_std_embedding"] = span_std

        return SemanticFeaturesTransformers(**features)

    def _maybe_load_models(self):
        if self._nlp is None:
            self._nlp = spacy.load("pt_core_news_md")

        if self._model is None:
            model_name = "ricardo-filho/bert-base-portuguese-cased-nli-assin-2"
            self._model = SentenceTransformer(model_name, device=self._device)
            self._dims = 768

    def _compute_adjacent_sentences(self, sentences: list[str]) -> tuple[float, float]:
        """Método que extrai a coesão local usando a
        similaridade entre pares adjacentes de sentenças no texto.

        :param sentences: lista de sentenças do texto

        :return: média e desvio padrão da similaridade
        """
        pairs = [(sentences[i], sentences[i + 1]) for i in range(len(sentences) - 1)]
        mean, std = self._compute_text(pairs)
        return mean, std

    def _compute_all_sentences(self, sentences: list[str]) -> tuple[float, float]:
        """Método que extrai coesão global usando a similaridade
        de todos os pares possíveis de sentenças do texto.

        :param sentences: lista de sentenças do texto

        :return: média e desvio padrão da similaridade
        """
        pairs = list(combinations(sentences, 2))
        mean, std = self._compute_text(pairs)
        return mean, std

    def _compute_sentences_givenness(self, sentences: list[str]) -> tuple[float, float]:
        """Método que extrai o quanto de informação dada existe
        em cada sentença de um texto, comparando com o
        conteúdo de informação anterior no texto.

        :param sentences: lista de sentenças do texto

        :return: média e desvio padrão da similaridade
        """
        pairs = [
            (sentences[i], "".join(sentences[:i])) for i in range(1, len(sentences))
        ]
        mean, std = self._compute_text(pairs)
        return mean, std

    def _compute_paragraphs(self, doc: Doc) -> tuple[float, float]:
        """Método que extrai a semelhança de um parágrafo
        com os outros parágrafos do texto.

        :param doc: texto.

        :return: média e desvio padrão da similaridade
        """
        paragraphs = [
            line.strip() for line in doc.text.split("\n") if line and not line.isspace()
        ]
        if len(paragraphs) <= 1:
            return 0, 0
        pairs = [(paragraphs[i], paragraphs[i + 1]) for i in range(len(paragraphs) - 1)]
        mean, std = self._compute_text(pairs)
        return mean, std

    def _compute_span(self, sentences: list[str]) -> tuple[float, float]:
        """Método que extrai o span da sentença. O span
        de uma sentença é uma forma de medir a proximidade
        entre uma sentença e o contexto que a precede.

        :param sentences: lista de sentenças do texto.

        :return: média e desvio padrão da similaridade
        """
        mean, std = 0, 0

        if len(sentences) < 2:
            return mean, std

        spans = np.zeros(len(sentences) - 1)

        for i in range(1, len(sentences)):
            past_sentences = sentences[:i]
            span_dimensions = len(past_sentences)

            if span_dimensions > self._dims - 1:
                beginning = past_sentences[0 : span_dimensions - self._dims]
                past_sentences[0] = beginning

            past_sentences_vectors = [
                sparse2full(self._get_vector(sentence), self._dims)
                for sentence in past_sentences
            ]

            current_sentence_vector = sparse2full(
                self._get_vector(sentences[i]), self._dims
            )
            current_sentence_array = np.array(current_sentence_vector).reshape(
                self._dims, 1
            )

            past_sentences_vectors_trans = np.array(past_sentences_vectors).T

            projection_matrix = np.dot(
                np.dot(
                    past_sentences_vectors_trans,
                    pinv(
                        np.dot(
                            past_sentences_vectors_trans.T, past_sentences_vectors_trans
                        )
                    ),
                ),
                past_sentences_vectors_trans.T,
            )

            projection = np.dot(projection_matrix, current_sentence_array).ravel()

            spans[i - 1] = cossim(
                full2sparse(current_sentence_vector), full2sparse(projection)
            )

        mean, std = self._get_mean_std(spans)

        return mean, std

    def _compute_text(self, pairs: list[tuple[str, str]]) -> tuple[float, float]:
        """Método usado para extrair similaridade
        entre os pares de sentenças.

        :param pairs: lista de pares de sentenças, ou parágrafos.

        :return: média e desvio padrão das similaridades
            entre todos os pares.
        """
        similarities = [
            self._compute_similarity(sent1, sent2) for sent1, sent2 in pairs
        ]
        if len(similarities) == 0:
            return 0, 0
        mean, std = self._get_mean_std(similarities)
        return round(mean, 5), round(std, 5)

    def _compute_similarity(self, sentence1: str, sentence2: str) -> float:
        """Método que calcula a similaridade entre duas sentenças."""
        return cossim(self._get_vector(sentence1), self._get_vector(sentence2))

    def _get_vector(self, sentence: str) -> list[tuple[int, float]]:
        """Método que extrai os embeddings das de uma sentença.

        :return: embeddings da sentença no formato de
            Bag of Words do gensim.
        """
        embeddings = self._model.encode(sentence)
        return list(enumerate(embeddings))

    @staticmethod
    def _get_mean_std(similarities: list[float]) -> tuple[float, float]:
        """Método que calcula a média e o desvio padrão
        de uma lista de similaridades.
        """
        arr = np.array(similarities)
        mean = arr.mean()
        std = arr.std()
        return mean, std

    def _batch_vectorize(self, texts: TextArrayLike, **kwargs):
        # TODO: investigar hang que ocorre quando utilizamos
        #   o multiprocessing
        kwargs["n_workers"] = 0
        return super()._batch_vectorize(texts, **kwargs)
