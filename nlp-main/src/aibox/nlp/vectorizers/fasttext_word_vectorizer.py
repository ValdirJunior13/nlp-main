"""Vetorizador de palavras baseado nos modelos do Fasttext."""

import re
from enum import Enum

import fasttext
import numpy as np
import spacy

from aibox.nlp import resources
from aibox.nlp.core import Vectorizer


class FTAggregationStrategy(Enum):
    """Estratégias para agregação dos word
    embeddings.

    :param NONE: nenhuma agregação é realizada.
        O vetorizador produz uma saída com formato
        (n_words, dims).
    :param AVERAGE: média de todas os word embeddings.
        O vetorizador produz uma saída com formato (dims,).
    """

    NONE = "none"
    AVERAGE = "average"


class FTTokenizerStrategy(Enum):
    """Estratégias de tokenização do texto.

    :param SPACY: utiliza o spaCy para tokenização.
    :param REGEX: utiliza um regex.
    """

    SPACY = "spacy"
    REGEX = "regex"


class FasttextWordVectorizer(Vectorizer):
    """Vetorização a nível de palavra baseada no FastText.

    :param aggregation: estratégia de agregação dos word
        embeddings.
    :param tokenizer: estratégia de tokenização.
    :param nlp: modelo do spacy a ser utilizado se tokenização
        através do spaCy (ignorado do contrário).
    :param regex_tokenizer: regex a ser utilizado se tokenização
        por regex (ignorado do contrário).
    :param language: linguagem do modelo.
    :param dims: dimensões do embedding.

    São utilizados os modelos pré-treinados do FastText. Atualmente,
    apenas a linguagem "pt" com 50 dimensões é suportada.

    O processo de vetorização ocorre da seguinte forma:
        1. Tokenização utilizando a estratégia selecionada;
        2. Cálculo dos word embeddings para cada token;
        3. Agregação utilizando a estratégia selecionada;

    .. code-block:: python

        from aibox.nlp.vectorizers.fasttext_word_vectorizer import FasttextWordVectorizer

        # Instanciando
        vectorizer = FasttextWordVectorizer()
        text = "Esse é um texto de exemplo"

        # Obtendo os vetores para cada palavra do texto.
        vectorizer.vectorize(text).shape
        # Out: (6, 50)
    """

    def __init__(
        self,
        aggregation: FTAggregationStrategy | str = FTAggregationStrategy.NONE,
        tokenizer: FTTokenizerStrategy | str = FTTokenizerStrategy.SPACY,
        nlp: spacy.Language | None = None,
        regex_tokenizer: str = r"\s+",
        language: str = "pt",
        dims: int = 50,
    ):
        """Construtor."""
        assert language in {"pt"}
        assert dims in {50}

        if isinstance(aggregation, str):
            aggregation = next(
                a for a in FTAggregationStrategy if aggregation == a.value
            )

        if isinstance(tokenizer, str):
            tokenizer = next(t for t in FTTokenizerStrategy if tokenizer == t.value)

        if tokenizer == FTTokenizerStrategy.SPACY and nlp is None:
            nlp = spacy.load("pt_core_news_md")

        # Armazenando variáveis auxiliares
        self._agg = aggregation
        self._tkn = tokenizer
        self._nlp = nlp
        self._tkn_regex = regex_tokenizer
        self._dims = dims

        # Carregando  modelo
        self._ft = None

    def _vectorize(self, text: str):
        self._maybe_load_models()
        words = self._tokenize(text)
        word_vectors = [self._ft.get_word_vector(w) for w in words]
        return self._maybe_aggregate(np.array(word_vectors))

    def _tokenize(self, text: str) -> list[str]:
        if self._tkn == FTTokenizerStrategy.REGEX:
            return re.split(self._tkn_regex, text)
        return [
            token.text
            for token in self._nlp(text)
            if not token.is_punct and not token.is_space
        ]

    def _maybe_aggregate(self, embeddings: np.ndarray) -> np.ndarray:
        if self._agg == FTAggregationStrategy.NONE:
            return embeddings
        return np.mean(embeddings, axis=0)

    def _maybe_load_models(self):
        # Obtendo caminho para o modelo
        root = resources.path(f"embeddings/fasttext-cc-{self._dims}.v1")
        model_path = root.joinpath(f"cc.pt.{self._dims}.bin").absolute()

        # Carregando o modelo
        self._ft = fasttext.load_model(str(model_path))

    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        for k in {"_ft"}:
            d[k] = None
        return d
