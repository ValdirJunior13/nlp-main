"""Vetorizador TF-IDF."""

from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTFIDF

from aibox.nlp.core import TrainableVectorizer
from aibox.nlp.typing import ArrayLike, TextArrayLike


class TFIDFVectorizer(TrainableVectorizer):
    """Vetorizador TF-IDF.

    .. code-block:: python

        from aibox.nlp.vectorizers.tfidf_vectorizer import TFIDFVectorizer

        # Instanciando
        vectorizer = TFIDFVectorizer()
        train = ["Texto 1", "Texto 2", "Texto 3"]
        text = "Texto de teste"

        # Treinando
        vectorizer.fit(train)

        # Obtendo representação
        vectorizer.vectorize(text)
        # Out: array([1.])
    """

    def __init__(self) -> None:
        self._tfidf = SklearnTFIDF()

    def _vectorize(self, text: str, **kwargs):
        return self._batch_vectorize([text], **kwargs)[0]

    def _batch_vectorize(self, texts: TextArrayLike, **kwargs):
        del kwargs
        sparse_matrix = self._tfidf.transform(texts)
        arr = sparse_matrix.toarray()
        return arr

    def fit(self, X: TextArrayLike, y: ArrayLike = None, **kwargs) -> None:
        del kwargs
        self._tfidf.fit(X)
