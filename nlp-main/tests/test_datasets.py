"""Testes do pacote de
datasets da biblioteca.
"""

import pandas as pd
import pytest

from aibox.nlp.data.datasets import DatasetDF
from aibox.nlp.factory import get_dataset
from aibox.nlp.factory.class_registry import get_class


@pytest.mark.parametrize("extended", [True, False])
@pytest.mark.parametrize(
    "target_competence", [f"C{i}" for i in range(1, 6)] + ["score"]
)
def test_essay_br(extended: bool, target_competence: str):
    ds = get_dataset("essayBR", extended=extended, target_competence=target_competence)
    assert ds.is_extended == extended
    assert ds.competence == target_competence
    assert ds.to_frame().shape == (4570 if not extended else 6564, 9)


@pytest.mark.parametrize("clean_tags", [True, False])
@pytest.mark.parametrize(
    "target_competence",
    [
        "cohesion",
        "thematic_coherence",
        "formal_register",
        "narrative_rhetorical_structure",
    ],
)
def test_portuguese_narrative_essays(clean_tags: bool, target_competence: str):
    ds = get_dataset(
        "narrativeEssaysBR", clean_tags=clean_tags, target_competence=target_competence
    )
    assert ds.competence == target_competence
    assert ds.to_frame().shape[0] == 1235


@pytest.mark.parametrize(
    "target_competence",
    [
        "cohesion",
        "thematic_coherence",
        "formal_register",
        "narrative_rhetorical_structure",
    ],
)
def test_dataset_df_load_from_kaggle(target_competence: str):
    ds = DatasetDF.load_from_kaggle(
        "moesiof/portuguese-narrative-essays",
        "essay",
        target_competence,
        ["train.csv", "test.csv", "validation.csv"],
    )
    assert ds.to_frame().shape[0] == 1235


@pytest.mark.parametrize(
    "data",
    [
        dict(idx=[0, 1, 2, 3, 4]),
        (dict(idx=[0, 1, 2]), dict(idx=[3, 4])),
        [dict(idx=[0, 1]), dict(idx=[2, 3]), dict(idx=[4])],
    ],
)
def test_dataset_df_custom_splits(data: dict | tuple[dict, dict] | list[dict]):
    def _dict_to_df(v: dict) -> pd.DataFrame:
        df = pd.DataFrame(v)
        df["text"] = df.idx.map(lambda v: f"Texto {v + 1}")
        df["target"] = df.idx + 1
        return df

    train, test = None, None
    splits = None
    if isinstance(data, tuple):
        data = tuple(map(_dict_to_df, data))
        train, test = data
    elif isinstance(data, list):
        splits = data = list(map(_dict_to_df, data))
    else:
        data = _dict_to_df(data)

    # Instanciando dataset
    ds = DatasetDF(data, text_column="text", target_column="target")

    # Assertions
    if train is not None:
        ds_train, ds_test = ds.train_test_split(0.0, False, 42)
        for v, e in zip([ds_train, ds_test], [train, test]):
            assert len(v) == len(e)
            assert v.text.isin(e.text).all()

    if splits is not None:
        ds_splits = ds.cv_splits(0, False, 42)
        for v, e in zip(ds_splits, splits):
            assert len(v) == len(e)
            assert v.text.isin(e.text).all()


@pytest.mark.parametrize(
    "dataset_identifier,config",
    [
        *[("essayBR", dict(extended=e)) for e in [True, False]],
        ("narrativeEssaysBR", dict()),
    ],
)
def test_load_raw(dataset_identifier: str, config: dict):
    cls = get_class(dataset_identifier)
    df = cls.load_raw(**config)
    assert len(df) > 1
    assert len(df.columns) > 1
