"""Adapter para representar
:py:class:`~pandas.DataFrame`'s
como Datasets."""

import logging
from typing import Any, Callable

try:
    from typing import Self
except ImportError:
    # Self was added on Python 3.11
    from typing import TypeVar

    Self = TypeVar("Self", bound="DatasetDF")


import pandas as pd
from pandas.api import types

from aibox.nlp.core import Dataset
from aibox.nlp.typing import DataFrameLike

from . import utils

logger = logging.getLogger(__name__)


class DatasetDF(Dataset):
    """Dataset a partir de um :py:class:`~pandas.DataFrame`.

    :param df: conjunto de dados que devem ser utilizados
        pelo dataset. Se é um valor único, ele é considerado
        o conjunto completo dos dados. Se tupla, o primeiro
        argumento é considerado para treinamento e o segundo
        para testes (retornados por :py:meth:`train_test_split`),
        o conjunto completo é a concatenação. Se lista, os valores
        são considerados como splits (retornados por :py:meth:`cv_splits`)
        e o conjunto completo é a concatenação.
    :param text_column: coluna que possui os textos.
    :param target_column: coluna com os valores target.
    :param copy: se devemos armazenar uma cópia do ou não.
    :param drop_others: se devemos remover outras
        colunas que não sejam as de texto e target.
    :param target_mapper: função de mapeamento para coluna target.
        Deve ser utilizada caso algum pré-processamento deva ser
        aplicado para converter a coluna para valores numéricos
        (default=não realiza mapeamento).

    Observações:

        - Os métodos :py:meth:`train_test_split` e :py:meth:`cv_splits`
        | respeitam os argumentos passados como entrada. Isso é, se o
        | conjunto de treino e testes foi passado, ele será retornado
        | pela respectiva função. Do contrário, uma amostragem do conjunto
        | completo é realizada.
    """

    def __init__(
        self,
        df: DataFrameLike | tuple[DataFrameLike, DataFrameLike] | list[DataFrameLike],
        text_column: str,
        target_column: str,
        copy: bool = True,
        drop_others: bool = False,
        target_mapper: Callable[[Any], int | float] = None,
    ):
        """Construtor."""
        # Variáveis auxiliares
        train, test = None, None
        splits = None

        # Parse do argumento df
        if isinstance(df, tuple):
            train, test = tuple(map(self._maybe_load_df, df))
            df = pd.concat([train, test], axis=0, ignore_index=True)
        elif isinstance(df, list):
            splits = list(map(self._maybe_load_df, df))
            df = pd.concat(splits, axis=0, ignore_index=True)
        else:
            df = self._maybe_load_df(df)

        # Checks de consistência
        assert text_column in df.columns, "Coluna não encontrada."
        assert target_column in df.columns, "Coluna não encontrada."
        assert len(df) > 0, "DataFrame não pode ser vazio."

        # Se o DataFrame original não deve ser alterado
        if copy:
            df = df.copy()

        # Talvez seja necessário aplicar algum mapeamento
        #   na coluna de target.
        if target_mapper is not None:
            df[target_column] = df[target_column].map(target_mapper)

        # Renomeando colunas
        self._df = df.rename(columns={text_column: "text", target_column: "target"})

        # Se demais devem ser removidos
        if drop_others:
            columns = set(self._df.columns.tolist())
            columns.remove("text")
            columns.remove("target")
            self._df.drop(columns=columns, inplace=True)

        # Checando consistência
        has_duplicates = self._df.text.duplicated().any()
        has_na_text = self._df.text.isnull().any()
        is_numeric = types.is_numeric_dtype(self._df.target.dtype)
        has_na_target = self._df.target.isnull().any()
        assert not has_na_text, "Não devem existir textos NULL."
        assert not has_duplicates, "Não devem existir textos duplicados."
        assert not has_na_target, "Não devem existir targets NULL."
        assert is_numeric, 'Coluna "target" deve ser numérica.'

        # Convertendo train, test e splits
        if train is not None:
            train = df[df.text.isin(train[text_column])]

        if test is not None:
            test = df[df.text.isin(test[text_column])]

        if splits is not None:
            for i in range(len(splits)):
                splits[i] = df[df.text.isin(splits[i][text_column])]

        # Armazenando valores
        self._train = train
        self._test = test
        self._splits = splits

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def to_frame(self):
        return self._df.copy()

    def cv_splits(self, k: int, stratified: bool, seed: int) -> list[pd.DataFrame]:
        if self._splits is not None:
            logging.warning(
                "CV splits have been define by init (k=%d). "
                "Ignored function arguments (k=%d,stratified=%s,seed=%d).",
                len(self._splits),
                k,
                stratified,
                seed,
            )
            return self._splits

        if stratified and self._is_classification():
            return utils.stratified_splits_clf(df=self._df, k=k, seed=seed)

        return utils.splits(df=self._df, k=k, seed=seed)

    def train_test_split(
        self, frac_train: float, stratified: bool, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._train is not None:
            logging.warning(
                "Train and test splits have been define by init (frac_train=%.2f). "
                "Ignored function arguments (frac_train=%.2f,stratified=%s,seed=%d).",
                len(self._train) / len(self._df),
                frac_train,
                stratified,
                seed,
            )
            return self._train, self._test

        if stratified and self._is_classification():
            return utils.train_test_clf(df=self._df, frac_train=frac_train, seed=seed)

        return utils.train_test(df=self._df, frac_train=frac_train, seed=seed)

    def _is_classification(self) -> bool:
        return types.is_integer_dtype(self._df.target.dtype)

    @staticmethod
    def _maybe_load_df(value: DataFrameLike) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value

        return pd.read_csv(value)

    @classmethod
    def load_from_kaggle(
        cls,
        ds_name: str,
        text_column: str,
        target_column: str,
        files_to_load: str | list[str],
        drop_others: bool = False,
        target_mapper: Callable[[Any], int | float] = None,
    ) -> Self:
        """Carrega um dataset a partir de um identificador de
        dataset no Kaggle. Se o dataset é privado ou requer
        permissão do usuário, é necessário ter realizado login
        com `kagglehub.login()`.


        :param ds_name: identificador do dataset no Kaggle.
        :param text_column: coluna que contém textos.
        :param target_column: coluna que contém os targets.
        :param *files_to_load: uma ou mais strings com quais
            arquivos devem ser carregados (o DataFrame final
            é uma concatenação linha a linha).
        :param drop_others: se demais colunas devem ser
            removidas (default=False).
        :param target_mapper: função
            de mapeamento da coluna target.

        :return: instância com os dados do dataset do Kaggle.
        """
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        df = pd.concat(
            [
                kagglehub.dataset_load(
                    KaggleDatasetAdapter.PANDAS,
                    ds_name,
                    f,
                )
                for f in files_to_load
            ],
            axis=0,
            ignore_index=True,
        )

        return cls(
            df,
            text_column=text_column,
            target_column=target_column,
            target_mapper=target_mapper,
            drop_others=drop_others,
            copy=False,
        )
