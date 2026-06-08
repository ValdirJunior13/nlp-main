"""Serialização de pipelines."""

from pathlib import Path
from typing import Literal

import joblib

from aibox.nlp.core import Pipeline


def save_pipeline(
    pipeline: Pipeline, save_path: str | Path, method: Literal["joblib"] = "joblib"
):
    """Realiza o salvamento de uma pipeline
    para o disco.

    :param pipeline: instância da pipeline.
    :param save_path: caminho de salvamento.
    :param method: método de serialização, atualmente
        apenas o joblib é suportado.
    """
    del method
    joblib.dump(pipeline, save_path)


def load_pipeline(path: str | Path) -> Pipeline:
    """Realiza o carregamento de uma pipeline.

    :param path: caminho para pipeline serializada.

    :return: pipeline carregada.
    """
    obj = joblib.load(path)
    assert isinstance(obj, Pipeline)

    return obj
