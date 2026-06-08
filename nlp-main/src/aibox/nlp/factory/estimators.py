"""Obtenção de estimadores através de nomes."""

from aibox.nlp.core import Estimator

from .class_registry import get_class


def get_estimator(estimator: str, **config) -> Estimator:
    """Obtém um estimador.

    :param estimador: nome do estimador.
    :param `**config`: configurações desse estimador.


    :return: estimador.
    """
    estimator = get_class(estimator)
    assert issubclass(estimator, Estimator), "Esse nome não corresponde à um estimador."
    return estimator(**config)
