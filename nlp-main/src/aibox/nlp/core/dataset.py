"""Interface básica para datasets."""

from abc import ABC, abstractmethod

import pandas as pd


class Dataset(ABC):
    """Classe abstrata representa um Dataset
    para classificação ou regressão. Essa classe
    supõe que os dados passados já foram limpos
    e/ou processados.

    Classes concretas definem construtores bem
    como métodos adicionais.
    """

    @abstractmethod
    def to_frame(self) -> pd.DataFrame:
        """Converte esse dataset para
        um DataFrame (cópia) com as colunas:
            1. text (str): textos;
            2. target (numérico): label;

        O DataFrame pode ter colunas adicionais.

        :return: representação desse dataset como um
            DataFrame.
        """

    @abstractmethod
    def cv_splits(self, k: int, stratified: bool, seed: int) -> list[pd.DataFrame]:
        """Retorna splits para serem utilizados. Esse método
        particiona o dataset em `k` partes aleatórias de tamanho
        similar.

        :param k: quantidade de splits.
        :param stratified: se os splits devem ser estratificados.
        :param seed: seed randômica para geração de splits. É
            garantido que uma mesma seed gere os mesmos splits.

        :return: Lista com `k` DataFrames.
        """

    @abstractmethod
    def train_test_split(
        self, frac_train: float, stratified: bool, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Obtém os conjuntos de treino e teste desse Dataset como
        DataFrames.

        :param frac_train: fração de amostras para treinamento.
        :param stratified: se cada split deve ser estratificado.
        :param seed: seed randômica para geração de splits. É
            garantido que uma mesma seed gere os mesmos splits.

        :return: tupla (train, test).
        """
