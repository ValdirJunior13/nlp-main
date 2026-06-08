"""Dataset Portuguese Narrative Essays
apresentado no PROPOR 2024.
"""

import pandas as pd

from aibox.nlp.core import Dataset

from . import utils
from .dataset_df import DatasetDF


class DatasetPortugueseNarrativeEssays(Dataset):
    """`Portuguese Narrative Essays <https://kaggle.com/datasets/moesiof/portuguese-narrative-essays>`_.

    :param target_competence: competência ('cohesion',
        'thematic_coherence', 'formal_register', 'narrative_rhetorical_structure').
    :param clean_tags: se devem ser removidas as tags de anotação.

    Essa classe apenas suporta splits (CV ou train/test) estratificados.
    """

    def __init__(self, target_competence: str, clean_tags: bool = True):
        """Construtor."""
        # Carregamento do Dataset
        self._target = target_competence
        self._df = DatasetDF.load_from_kaggle(
            "moesiof/portuguese-narrative-essays",
            text_column="essay",
            target_column=target_competence,
            files_to_load=["train.csv", "test.csv", "validation.csv"],
        ).to_frame()

        # Reorganizando a ordem do DataFrame: text, target
        #   vem primeiro e depois as colunas faltantes.
        cols = list(self._df.columns)
        cols.remove("text")
        cols.remove("target")
        self._df = self._df[["text", "target"] + cols]

        # Remoção de tags
        if clean_tags:
            self._df = self._remove_tags(self._df)

    @property
    def competence(self) -> str:
        """Competência target.

        :return: competência.
        """
        return self._target

    def to_frame(self):
        return self._df.copy()

    def cv_splits(self, k: int, stratified: bool, seed: int) -> list[pd.DataFrame]:
        del stratified
        return utils.stratified_splits_clf(df=self._df, k=k, seed=seed)

    def train_test_split(
        self, frac_train: float, stratified: bool, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        del stratified
        return utils.train_test_clf(df=self._df, frac_train=frac_train, seed=seed)

    def _remove_tags(self, df: pd.DataFrame, copy: bool = False) -> pd.DataFrame:
        if copy:
            df = df.copy()

        # Well-formed tags with format [<LETTER_OR_SYMBOL>]
        tag_regex = r"(\[[PpSsTtXx?]\])"

        # Well-formed tags with format {<LETTER_OR_SYMBOL>}
        tag_regex += r"|({[ptx?]})"

        # Well-formed tags [LT] or [LC]
        tag_regex += r"|(\[L[TC]\])"

        # Well-formed tags with format [lt] or [lc]
        tag_regex += r"|(\[l[tc]\])"

        # Variant with a trailing space
        tag_regex += r"|(\[ P\])"

        # Mixed closing/opening symbol
        tag_regex += r"|(\[[PX?]\})"
        tag_regex += r"|(\{?\])"

        # Remove tags
        df.text = df.text.str.replace(tag_regex, "", regex=True)

        return df

    @classmethod
    def load_raw(self) -> pd.DataFrame:
        """Carregamento dos dados crus do dataset. Nenhuma
        limpeza ou conversão de estrutura de colunas é
        realizada.

        O objetivo desse método é permitir que o dataset seja
        carregado as-is sem escolha de competência ou outras
        limpezas.

        :return: Portuguese Narrative Essays.
        """
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        return pd.concat(
            [
                kagglehub.dataset_load(
                    KaggleDatasetAdapter.PANDAS,
                    "moesiof/portuguese-narrative-essays",
                    f,
                )
                for f in ["train.csv", "test.csv", "validation.csv"]
            ],
            axis=0,
            ignore_index=True,
        )
