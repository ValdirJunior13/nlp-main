"""Construção e obtenção de pipelines."""

from aibox.nlp.core import Estimator, Pipeline, Vectorizer

from .class_registry import get_class


def get_pipeline(pipeline: str, pipeline_config: dict = dict()) -> Pipeline:
    """Carrega uma pipeline com o nome passado.

    :param pipeline: nome da pipeline.
    :param pipeline_config: configuração da pipeline.

    :return: pipeline.
    """
    pipeline = get_class(pipeline)(**pipeline_config)
    assert isinstance(pipeline, Pipeline)
    return pipeline


def make_pipeline(
    vectorizer: str,
    estimator: str,
    vectorizer_config: dict = dict(),
    estimator_config: dict = dict(),
) -> Pipeline:
    """Constrói uma pipeline dado o vetorizador
    e estimador a serem utilizados.

    :param vectorizer: nome do vetorizador.
    :param estimator: nome do estimador.
    :param vectorizer_config: configurações do
        vetorizador (passadas ao construtor).
    :param estimator_config: configurações do
        estimador (passadas ao construtor).

    :return: pipeline com os parâmetros selecionados.
    """
    vectorizer = get_class(vectorizer)(**vectorizer_config)
    estimator = get_class(estimator)(**estimator_config)
    assert isinstance(vectorizer, Vectorizer)
    assert isinstance(estimator, Estimator)
    return Pipeline(vectorizer, estimator)
