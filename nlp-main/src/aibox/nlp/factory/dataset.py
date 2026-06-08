"""Obtenção de datasets através de nomes."""

from aibox.nlp.core import Dataset

from .class_registry import get_class


def get_dataset(ds: str, **config) -> Dataset:
    """Obtém um dataset dado o nome.

    :param dataset: nome do dataset.
    :param config: configuração do dataset.

    :return: dataset.
    """
    ds = get_class(ds)
    assert issubclass(ds, Dataset), "Esse nome não corresponde à um dataset."
    return ds(**config)
