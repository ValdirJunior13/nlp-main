"""Interface básica para pipelines."""

import numpy as np

from aibox.nlp.typing import ArrayLike, PostProcessing, TextArrayLike

from .estimator import Estimator
from .vectorizer import TrainableVectorizer, Vectorizer


class Pipeline:
    """Essa é a interface básica para uma
    pipeline.

    :param vectorizer: estratégia de vetorização dessa pipeline.
        Pode ser treinável ou não.
    :param estimator: estimador dessa pipeline.
    :param postprocessing: estratégia de pós-processamento das saídas
        do estimador. Valor padrão é identidade (i.e., no-op). A estratégia
        de pós-processamento deve respeitar o shape (n_samples,) como entrada
        e saída. Dtypes podem ser distintos.
    :type postprocessing: Callable[[np.ndarray], np.ndarray], opcional
    :param name: nome dessa pipeline. Quando não passado, um nome automático
        é gerado com base nos demais argumentos.
    :type name: str, opcional

    Todas as etapas de uma pipeline são sequenciais, isto é,
    a saída de uma etapa é entrada para a próxima.

    Toda pipeline é composta por 3 componentes:
        1. Vetorizador
        2. Estimador
        3. Pós-processamento

    Quando o método :py:meth:`fit` é invocado em uma pipeline,
    o seguinte processo ocorre para cada componente treinável `T`:
        1. Treinamos `T` fazendo `T.fit(X, y)`;
        2. Calculamos o novo valor de `X = T.predict(X)`;
        3. Passamos o novo `X` e o mesmo `y` para a próxima etapa treinável;
    """

    def __init__(
        self,
        vectorizer: Vectorizer,
        estimator: Estimator,
        postprocessing: PostProcessing = None,
        name: str | None = None,
    ):
        """Construtor."""
        if postprocessing is None:
            postprocessing = self._identity

        if name is None:
            name = self._generate_name(vectorizer, estimator)

        self._vectorizer = vectorizer
        self._estimator = estimator
        self._postprocessing = postprocessing
        self._name = name

    def predict(self, X: TextArrayLike, **kwargs) -> np.ndarray:
        """Realiza a predição utilizando os parâmetros
        atuais da pipeline.

        O comportamento desse método não é definido caso seja
        chamado antes do treinamento (i.e., :py:meth:`fit`).

        :param X: array-like de strings com formato (n_samples, <any>).
        :param `**kwargs`: configurações estras para o estimador ou
            vetorizador.

        :return: array com as predições para cada amostra.
        :rtype: np.ndarray
        """
        # Obtemos a representação vetorial para cada um dos
        #   textos
        X_ = self.vectorizer.vectorize(X, **kwargs)

        # Calculamos as predições do estimador
        preds = self.estimator.predict(X_, **kwargs)

        # Aplicamos o pós processamento
        preds = self._postprocessing(preds)

        return preds

    def fit(self, X: TextArrayLike, y: ArrayLike, **kwargs) -> None:
        """Realiza o treinamento da pipeline
        utilizando as entradas `X` com os targets
        `y`.

        :param X: array-like de strings com formato (n_samples, <any>).
        :param y: array-like com formato (n_samples,).
        :param `**kwargs`: configurações estras para o estimador ou
            vetorizador.
        """
        # Caso o vetorizador seja treinável
        if isinstance(self.vectorizer, TrainableVectorizer):
            self.vectorizer.fit(X, y, **kwargs)

        # Obtemos a representação vetorial para todos textos
        X_ = self.vectorizer.vectorize(X, **kwargs)

        # Treinamos o estimador utilizando os vetores
        self.estimator.fit(X_, y, **kwargs)

    @property
    def vectorizer(self) -> Vectorizer:
        """Retorna o vetorizador dessa pipeline.

        :return: vetorizador dessa pipeline.
        :rtype: Vectorizer
        """
        return self._vectorizer

    @property
    def estimator(self) -> Estimator:
        """Retorna o estimador utilizado
        nessa pipeline.

        :return: estimador dessa pipeline.
        """
        return self._estimator

    @property
    def name(self) -> str:
        """Retorna o nome dessa pipeline.

        :return: nome da pipeline.
        """
        return self._name

    def postprocessing(self, y: np.ndarray) -> np.ndarray:
        """Método de pós-processamento da pipeline.

        :param y: array-like com formato (n_samples,).
        :type y: np.ndarray

        :return: array com mesmo formato após função
            de pós-processamento.
        """
        return self._postprocessing(y)

    def _identity(self, x):
        return x

    @staticmethod
    def _generate_name(vectorizer: Vectorizer, estimator: Estimator) -> str:
        # Obtendo nome da classe do estimador
        estimator_name = estimator.__class__.__name__

        # Se for um agregado de features, obtemos o nome
        #   individual de cada uma
        extractors = getattr(vectorizer, "extractors", None)
        if extractors:
            vectorizer_name = "_".join(v.__class__.__name__ for v in extractors)
        else:
            vectorizer_name = vectorizer.__class__.__name__

        # Obtemos os parâmetros do estimador
        estimator_params = "_".join(
            str(v)
            for v in estimator.hyperparameters.values()
            if not isinstance(v, dict)
        )

        # Construímos o nome final da pipeline
        name = "_".join(
            [
                vectorizer_name,
                estimator_name,
                estimator_params,
                f"seed_{estimator.random_state}",
            ]
        )
        return name
