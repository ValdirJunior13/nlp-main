"""Vetorizador identidade."""

from aibox.nlp.core import Vectorizer
from aibox.nlp.typing import TextArrayLike


class IdentityVectorizer(Vectorizer):
    """Vetorizador identidade.

    Essa é uma classe especial que implementa o
    operador identidade para vetorizadores.

    Ou seja, ela retorna os mesmos argumentos
    recebidos como entrada e na prática é uma
    exceção aos vetorizadores (que devem produzir
    uma representação numérica dos textos recebidos
    como entrada).

    Essa classe serve o único propósito de permitir
    que abordagens de Deep Learning que aprendem
    representações implícitas possam ser utilizados
    seguindo a estrutura já existente da biblioteca.

    Por ser um caso especial, essa classe não pode ser
    instanciada através dos métodos de
    :py:mod:`aibox.nlp.factory`.
    """

    def vectorize(
        self,
        text: str | TextArrayLike,
        vector_type: str = None,
        device: str | None = None,
        **kwargs,
    ) -> str | TextArrayLike:
        """Retorna o mesmo valor passado
        como argumento.

        :return: ``text``.
        """
        return text

    def _vectorize(self, text: str, **kwargs):
        return text

    def _batch_vectorize(self, texts: str, **kwargs):
        return texts
