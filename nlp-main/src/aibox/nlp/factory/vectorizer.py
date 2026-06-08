"""Obtenção de vetorizadores
através de um nome.
"""

from aibox.nlp.core import Vectorizer

from .class_registry import get_class


def get_vectorizer(vectorizer: str, *args, **config) -> Vectorizer:
    """Obtém um vetorizador dado um nome.

    :param vectorizer: nome do vetorizador.
    :param args: configurações posicionais para
        o vetorizador.
    :param config: configurações desse vetorizador.

    :return: vetorizador.
    """
    vectorizer = get_class(vectorizer)
    assert issubclass(
        vectorizer, Vectorizer
    ), "Esse nome não corresponde à um vetorizador."
    return vectorizer(*args, **config)
