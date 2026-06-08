"""Essay-BR (versão original e estendida)
com redações do Ensino Médio.
"""

import pandas as pd

from aibox.nlp import resources
from aibox.nlp.core import Dataset

from . import utils


class DatasetEssayBR(Dataset):
    """Essay-BR original e estendido.

    :param extended: se devemos utilizar a versão estendida.
    :param target_competence: competência ('C1', 'C2', 'C3',
        'C4', 'C5' ou 'score').


    As versões utilizadas pela biblioteca se encontram
    disponíveis nos repositórios originais do GitHub:
        - https://github.com/rafaelanchieta/essay/tree/master/essay-br
            - Commit: da35364a0e213310ce83e55a613fbaa58d134bd3
        - https://github.com/lplnufpi/essay-br/tree/main/extended-corpus
            - Commit: fb6391a79cbb12dff877eb442c2a31caa7f00c77

    São aplicados alguns pós-processamentos visto que os dados originais
    possuem redações duplicadas e/ou faltantes.

    Essa classe apenas suporta splits (CV ou train/test) estratificados.
    """

    def __init__(self, extended: bool, target_competence: str):
        # Carregamento do dataset
        self._extended = extended
        self._target = target_competence
        self._df = self.load_raw(self._extended)

        # Garantindo que o DataFrame possui os dados
        #   necessários.
        assert self._target in self._df.columns
        assert "text" in self._df.columns

        # Pós-processamentos
        # 1. Remoação de vazios (NaN, Nulls, etc)
        self._df.dropna(ignore_index=True, inplace=True)

        # 2. Remoção de redações duplicadas
        self._df.drop_duplicates(subset="text", ignore_index=True, inplace=True)

        # Adicionando nova coluna com o target
        self._df["target"] = self._df[self._target]

        # Reorganizando a ordem do DataFrame: text, target
        #   vem primeiro e depois as colunas faltantes.
        cols = list(self._df.columns)
        cols.remove("text")
        cols.remove("target")
        self._df = self._df[["text", "target"] + cols]

    @property
    def competence(self) -> str:
        """Competência target.

        :return: competência.
        """
        return self._target

    @property
    def is_extended(self) -> bool:
        """Se a versão carregada é estendida
        ou original.

        :return: versão estendida ou original.
        """
        return self._extended

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

    @classmethod
    def load_raw(self, extended: bool) -> pd.DataFrame:
        """Carregamento dos dados crus do dataset. Nenhuma
        limpeza ou conversão de estrutura de colunas é
        realizada.

        O objetivo desse método é permitir que o dataset seja
        carregado as-is sem escolha de competência ou outras
        limpezas.

        :param extended: se deve ser carregada
            a versão estendida do dataset.

        :return: Essay-BR versão original ou estendida.
        """
        target_resource = "essay-br-extended" if extended else "essay-br"
        root_dir = resources.path(f"datasets/{target_resource}.v1")
        return pd.read_csv(root_dir.joinpath("dataset.csv"))
