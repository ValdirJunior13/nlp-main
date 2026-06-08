"""Construção e obtenção de experimentos."""

import logging
from typing import ClassVar, Literal

import numpy as np

from aibox.nlp.core import Dataset, Experiment, Metric, Pipeline
from aibox.nlp.experiments.simple_experiment import SimpleExperiment
from aibox.nlp.pipelines import postprocessing as pp
from aibox.nlp.typing import PostProcessing

from .class_registry import get_class, registry
from .dataset import get_dataset
from .feature_extractor import get_extractor
from .metric import get_metric
from .vectorizer import get_vectorizer

logger = logging.getLogger(__name__)


class SimpleExperimentBuilder:
    """Builder para :py:class:`~aibox.nlp.experiments.simple_experiment.SimpleExperiment`.

    Essa classe implementa o padrão builder para construção de
    um experimento de forma programaticamente, abstraindo consideravelmente
    o processo manual. Exemplo de uso:

    .. code-block:: python

        from aibox.nlp.factory.experiment import SimpleExperimentBuilder

        # Inicializando o builder
        builder = SimpleExperimentBuilder()

        # Definindo a seed
        builder.seed(42)

        # Selecionando o dataset e tipo do problema
        builder.dataset("essayBR", extended=False, target_competence="C1").classification()

        # Adicionando métricas
        builder.add_metric("precision", average="weighted")
        builder.add_metric("recall", average="weighted")
        builder.add_metric("f1", average="weighted")
        builder.add_metric("kappa")
        builder.add_metric("neighborKappa")

        # Definindo a métrica para melhor critério
        builder.best_criteria("precision", average="weighted")

        # Pipeline de features
        builder.add_feature_pipeline(
            features=["descriptiveBR"],
            estimators=["svm", "xgbClf"],
            names=["svm+features", "xgb+features"],
        )

        # Pipeline com outras estratégias de vetorização
        builder.add_vectorizer_pipeline(
            "tfidfVectorizer",
            estimators=["etreesClf", "lgbmClf"],
            names=["etrees+tfidf", "lgbm+tfidf"],
            estimators_configs=[dict(n_estimators=20), dict(n_estimators=20)],
        )

        builder.add_vectorizer_pipeline(
            "bertVectorizer",
            estimators=["svm", "etreesClf", "lgbmClf", "xgbClf"],
            names=["svm+bert", "etrees+bert", "lgbm+bert", "xgb+bert"])

        # Construindo experimento:
        experiment = builder.build()
    """

    _SKIP_ESTIMATOR: ClassVar[set[str]] = {"lstmClf", "lstmReg"}
    _IMPLICIT_REPRESENTATION_ESTIMATORS: ClassVar[set[str]] = {
        "transformerClf",
        "transformerReg",
    }

    def __init__(self) -> None:
        self._ds: Dataset = None
        self._criteria: Metric = None
        self._maximize_criteria = None
        self._pipelines: list[Pipeline] = []
        self._metrics: list[Metric] = []
        self._seed = None
        self._rng = None
        self._problem = None

    def add_feature_pipeline(
        self,
        features: str | list[str],
        estimators: str | list[str],
        names: str | list[str],
        postprocessing: PostProcessing | list[PostProcessing] = None,
        features_configs: dict | list[dict] = None,
        estimators_configs: dict | list[dict] = None,
    ) -> "SimpleExperimentBuilder":
        """Adiciona uma ou mais pipelines baseada em características. Se
        forem passados mais que um estimador, serão construídas pipelines
        com o mesmo conjunto de features mas com cada estimador.

        :param features: característica ou lista de características.
        :param estimator: estimador ou lista de estimadores.
        :param features_configs: configurações dos extratores de características.

        :return: self.
        """
        if self._seed is None:
            logger.info("Inicialize a seed randômica primeiro.")
            return

        features = self._maybe_convert_to_list(features)
        estimators = self._maybe_convert_to_list(estimators)
        names = self._maybe_convert_to_list(names)
        assert all(
            e not in self._IMPLICIT_REPRESENTATION_ESTIMATORS for e in estimators
        )

        if features_configs is None:
            features_configs = [dict()] * len(features)
        features_configs = self._maybe_convert_to_list(features_configs)

        if estimators_configs is None:
            estimators_configs = [dict()] * len(names)
        estimators_configs = self._maybe_convert_to_list(estimators_configs)

        if postprocessing is None:
            postprocessing = [None] * len(names)
        postprocessing = self._maybe_convert_to_list(postprocessing)

        assert len(estimators) == len(names)
        extractor = get_extractor(features, features_configs)
        for name, e, c, p in zip(names, estimators, estimators_configs, postprocessing):
            seed = self._estimator_seed()
            estimator = get_class(e)(**c, random_state=seed)
            self._pipelines.append(
                Pipeline(
                    vectorizer=extractor,
                    estimator=estimator,
                    postprocessing=p,
                    name=name,
                )
            )

        return self

    def add_vectorizer_pipeline(
        self,
        vectorizer: str,
        estimators: str | list[str],
        names: str | list[str],
        postprocessing: PostProcessing | list[PostProcessing] = None,
        vectorizer_config: dict = dict(),
        estimators_configs: dict | list[dict] = None,
    ) -> "SimpleExperimentBuilder":
        """Adiciona uma ou mais pipelines baseadas no vetorizador. Se
        forem passados mais que um estimador, serão construídas pipelines
        com o mesmo vetorizador mas com cada estimador.

        :param vectorizer: vetorizador ou lista de vetorizadores.
        :param estimators: estimador ou lista de estimadores.
        :param names: nome(s) do estimador(es).
        :param postprocessing: pós-processamentos.
        :param vectorizer_config: configuração do vetorizador.
        :param estimators_configs: configuração(ões) do estimador(es).

        :return: self.
        """
        if self._seed is None:
            logger.info("Inicialize a seed randômica primeiro.")
            return

        estimators = self._maybe_convert_to_list(estimators)
        names = self._maybe_convert_to_list(names)
        if vectorizer != "__identity":
            assert all(
                e not in self._IMPLICIT_REPRESENTATION_ESTIMATORS for e in estimators
            )

        if estimators_configs is None:
            estimators_configs = [dict()] * len(names)
        estimators_configs = self._maybe_convert_to_list(estimators_configs)

        if postprocessing is None:
            postprocessing = [None] * len(names)
        postprocessing = self._maybe_convert_to_list(postprocessing)

        assert len(estimators) == len(names)
        vectorizer = get_vectorizer(vectorizer, **vectorizer_config)

        for name, e, c, p in zip(names, estimators, estimators_configs, postprocessing):
            seed = self._estimator_seed()
            estimator = get_class(e)(**c, random_state=seed)
            self._pipelines.append(
                Pipeline(
                    vectorizer=vectorizer,
                    estimator=estimator,
                    postprocessing=p,
                    name=name,
                )
            )
        return self

    def add_implicit_vectorizer_pipeline(
        self,
        estimators: str | list[str],
        names: str | list[str],
        postprocessing: PostProcessing | list[PostProcessing] = None,
        estimators_configs: dict | list[dict] = None,
    ) -> "SimpleExperimentBuilder":
        """Adiciona uma ou mais pipelines baseada em Deep Learning que
        aprendem vetorização de forma implícita (e.g., Transformers).

        :param estimators: estimador ou lista de estimadores.
        :param names: nome(s) do estimador(es).
        :param postprocessing: pós-processamentos.
        :param estimators_configs: configuração(ões) do estimador(es).

        :return: self.
        """
        assert (
            estimators in self._IMPLICIT_REPRESENTATION_ESTIMATORS
        ), "Esse estimador não aprende representações para textos."
        return self.add_vectorizer_pipeline(
            vectorizer="__identity",
            estimators=estimators,
            names=names,
            postprocessing=postprocessing,
            estimators_configs=estimators_configs,
            vectorizer_config=dict(),
        )

    def add_metric(self, metric: str, **metric_config) -> "SimpleExperimentBuilder":
        """Adiciona uma métrica para o experimento caso
        ela não tenha sido adicionada anteriormente.

        :param metric: métrica.
        :param `**metric_config`: configurações da métrica.

        :return: self.
        """
        m = get_metric(metric, **metric_config)
        if m not in self._metrics:
            self._metrics.append(m)

        return self

    def best_criteria(
        self, metric: str, maximize: bool = True, **metric_config
    ) -> "SimpleExperimentBuilder":
        """Define a métrica para selecionar
        a melhor pipeline.

        :param metric: métrica.
        :param maximize: se essa métrica deve ser maximizada.
        :param `**metric_config`: configurações da métrica.

        :return: self.
        """
        self._criteria = get_metric(metric, **metric_config)
        if self._criteria not in self._metrics:
            self._metrics.append(self._criteria)
        self._maximize_criteria = maximize
        return self

    def dataset(self, ds: str, **ds_config) -> "SimpleExperimentBuilder":
        """Define o dataset para
        os experimentos.

        :param ds: dataset.
        :param `**ds_config`: configurações do dataset.

        :return: self.
        """
        self._ds = get_dataset(ds, **ds_config)
        return self

    def seed(self, seed: int) -> "SimpleExperimentBuilder":
        """Define a seed para o experimento.

        :param seed: seed.

        :return: self.
        """
        self._seed = seed

        # Inicializando RNG para criar as seeds
        #   dos estimadores. Por garantia,
        #   utilizamos uma seed diferente da passada
        #   para o experimento.
        self._rng = np.random.default_rng(self._seed + 1)

        return self

    def classification(self) -> "SimpleExperimentBuilder":
        """Define que esse é um experimento
        de classificação.

        :return: self.
        """
        self._problem = "classification"
        return self

    def regression(self) -> "SimpleExperimentBuilder":
        """Define que esse é um experimento
        de regressão.

        :return: self.
        """
        self._problem = "regression"
        return self

    def custom_dataset(self, ds: Dataset) -> "SimpleExperimentBuilder":
        """Adiciona uma instância de um dataset.

        :param ds: dataset.

        :return: self.
        """
        self._ds = ds
        return self

    def build(self, **kwargs) -> Experiment:
        """Constrói o experimento com as informações
        coletadas e limpa as informações coletadas.

        :param `**kwargs`: configurações extras
            passadas ao construtor de
            :py:class:`~aibox.nlp.experiments.simple_experiment.SimpleExperiment`.

        :return: experimento.
        """
        # Construção do experimento
        experiment = SimpleExperiment(
            pipelines=self._pipelines,
            dataset=self._ds,
            criteria_best=self._criteria,
            criteria_maximize=self._maximize_criteria,
            metrics=self._metrics,
            seed=self._seed,
            keep_all_pipelines=False,
            problem=self._problem,
            **kwargs,
        )

        # Reset do estado do builder
        self._ds: Dataset = None
        self._criteria: Metric = None
        self._pipelines: list[Pipeline] = []
        self._metrics: list[Metric] = []
        self._seed = None
        self._rng = None
        self._problem = None

        # Retornando o experimento
        return experiment

    @classmethod
    def features_experiment(
        cls,
        seed: int,
        problem: Literal["classification", "regression"],
        include_reg_as_clf: bool = True,
    ) -> "SimpleExperimentBuilder":
        """Retorna uma instância pré-inicializada do builder com
        todas as pipelines utilizando todas características disponíveis.

        :param seed: seed randômica.
        :param problem: tipo do problema.
        :param include_reg_as_clf: Se estimadores voltados
            à regressão devem ser adicionados para classificação
            utilizando :py:meth:`np.round`.

        :return: builder.
        """
        # Inicializando builder
        builder = cls()
        builder.seed(seed)

        # Obtendo o nome de todas as características
        features = [k for k in registry.features_br if "similarity" not in k.lower()]

        # Selecionando estimadores do tipo
        #   esperado (i.e., clf ou reg)
        target = f"{problem}."
        estimators = [
            k
            for k, v in registry.estimators.items()
            if target in v
            and (k not in cls._SKIP_ESTIMATOR)
            and (k not in cls._IMPLICIT_REPRESENTATION_ESTIMATORS)
        ]
        names = [f"all_features+{e}" for e in estimators]

        # Adicionando esses estimadores no experimento
        builder.add_feature_pipeline(features, estimators, names)

        # Atualizando problema do builder
        if problem == "regression":
            # Adicionando métricas de regressão
            builder.add_metric("MAE")
            builder.add_metric("RMSE")
            builder.add_metric("R2")
            builder.add_metric("MSE")
            builder.regression()
        else:
            # Adicionando métricas de classificação
            builder.add_metric("precision")
            builder.add_metric("precision", average="weighted")
            builder.add_metric("recall")
            builder.add_metric("recall", average="weighted")
            builder.add_metric("f1")
            builder.add_metric("f1", average="weighted")
            builder.add_metric("kappa")
            builder.classification()

            # Adicionando regressores para
            #   classificação
            if include_reg_as_clf:
                estimators = [
                    k
                    for k, v in registry.estimators.items()
                    if "regression." in v and k not in cls._SKIP_ESTIMATOR
                ]
                names = [f"all_features+{e}" for e in estimators]
                postprocessing = [pp.round_to_integer]
                postprocessing = postprocessing * len(estimators)
                builder.add_feature_pipeline(
                    features, estimators, names, postprocessing
                )

        return builder

    def _maybe_convert_to_list(self, obj) -> list:
        if not isinstance(obj, list):
            return [obj]

        return obj

    def _estimator_seed(self) -> int:
        return self._rng.integers(0, 99999, endpoint=True).item()
