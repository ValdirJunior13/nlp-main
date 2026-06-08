"""Experimentos sequenciais simples."""

import logging
import operator
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aibox.nlp.cache.features import CachedExtractor
from aibox.nlp.cache.mixed_feature_cache import MixedFeatureCache
from aibox.nlp.cache.vectorizers import (
    CachedVectorizer,
    DictVectorizerCache,
    TrainableCachedVectorizer,
)
from aibox.nlp.core import (
    Dataset,
    Experiment,
    ExperimentConfiguration,
    ExperimentResult,
    FeatureExtractor,
    Metric,
    Pipeline,
    TrainableVectorizer,
    Vectorizer,
)
from aibox.nlp.features.utils.aggregator import AggregatedFeatureExtractor
from aibox.nlp.lazy_loading import lazy_import
from aibox.nlp.vectorizers.identity import IdentityVectorizer

pandas_types = lazy_import("pandas.api.types")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimpleExperimentExtras:
    """Dados extras desse tipo de experimento.

    :param df_features: conjunto de características
        extraídas para cada um dos textos do
        dataset. Possui uma coluna `text` com o
        texto e possui o valor de características nas
        demais colunas.
    :param embeddings: conjunto de embeddings (textos
        vetorizados) que não são características. Cada
        entrada no dicionário é um DataFrame com uma
        coluna `text` com o texto e demais colunas representam
        dimensões do embedding (chave é um identificador para
        o vetorizador).
    :param run_duration: duração total da execução
        do experimento em segundos.
    """

    df_features: pd.DataFrame
    embeddings: dict[str, pd.DataFrame]
    run_duration: float


class SimpleExperiment(Experiment):
    """Classe para experimentos simples e
    sequenciais.

    :param pipelines: pipelines que devem ser testadas.
    :param dataset: que deve ser utilizado.
    :param metrics: métricas que devem ser
        calculadas.
    :param criteria_best: métrica que é utilizada
        para escolher a melhor pipeline.
    :param criteria_maximize: se a métrica critério
        deve ser maximizada (true) ou minimizada (false).
    :param seed: seed randômica utilizada.
    :param frac_train: fração do conjunto a ser utilizado
        para treinamento.
    :param keep_all_pipelines: se todas pipelines
        devem ser guardadas.
    :param cache_limit: limite de cacheamento
        da vetorização entre as diferentes pipelines. Se
        <= 0, todos os textos são cacheados.
    :param problem: 'classification', 'regression' ou
        None. Caso None, inferir do dataset (default=None).
    :param stratified_ds: se devemos utilizar splits de train
        e test estratificados.
    :param n_workers: quantidade de workers que devem ser utilizados
        para vetorização em batch de vetorizadores que utilizam
        `multiprocessing`.
    :param features_df: DataFrame com características
        pré-extraídas. O DataFrame precisa
        ter uma coluna 'text' e todas as demais colunas são
        relativas à uma característica existente na biblioteca.
    :param embeddings_dict: dicionário com embeddings pré-extraídos.
        Cada chave deve ser um identificador de vetorizador (i.e.,
        nome da classe). Cada valor deve ser um DataFrame onde a
        primeira coluna corresponde ao texto ("text") e demais colunas
        correspondem a representação vetorial desse texto de forma
        ordenada (i.e., 1ª dimensão do embedding, 2ª dimensão, etc).
    :param force_disable_cache: desativar completamento o
        cacheamento de extratores/vetorizadores. Se desativado,
        pode aumentar consideravelmente o custo computacional.

    Essa classe permite a execução de múltiplas pipelines de
    forma a buscar a que possui o melhor valor da métrica
    critério. Por padrão, o experimento busca maximizar a
    métrica critério (``criteria_maximize=True``).

    O método de avaliação é o holdout. Ou seja, um conjunto
    de treino e testes é definido de antemão e é utilizado
    por todas as pipelines.

    Para tornar a execução mais eficiente, durante o processo
    de treinamento das pipelines um cache é automaticamente
    configurado. Dessa forma, se duas pipelines utilizam o mesmo
    conjunto de características, o processo de vetorização só
    ocorre uma vez e é reaproveitado.

    De forma similar, pipelines com outras estratégias de
    vetorização também utilizam um cache. Se duas pipelines
    possuem o mesmo vetorizador (i.e., mesma classe), os
    embeddings são extraídos apenas uma vez e reaproveitados.

    É importante ressaltar que o cache não leva em conta configurações
    específicas dos extratores/vetorizadores. Isso quer dizer que se
    duas pipelines utilizam o mesmo extrator/vetorizador
    mas com configurações distintas, o comportamento do cache é não
    definido (i.e., o cache mantém apenas a versão vetorizada primeiro).
    Em tais casos, o recomendado é desabilitar o cache de características
    por completo com o parâmetro ``force_disable_cache=True``.

    Dessa forma, é necessário cautela ao se utilizar os parâmetros
    ``features_df`` e ``embeddings_dict``.
    """

    def __init__(
        self,
        pipelines: list[Pipeline],
        dataset: Dataset,
        metrics: list[Metric],
        criteria_best: Metric,
        criteria_maximize: bool = True,
        seed: int = 8990,
        frac_train: float = 0.8,
        keep_all_pipelines: bool = False,
        cache_limit: int | None = 0,
        problem: str | None = None,
        stratified_ds: bool = True,
        n_workers: int = 1,
        features_df: pd.DataFrame | None = None,
        embeddings_dict: dict[str, pd.DataFrame] | None = None,
        force_disable_cache: bool = False,
    ):
        """Construtor."""
        if problem is None:
            dtype = dataset.to_frame().target.dtype
            assert pandas_types.is_numeric_dtype(dtype)
            problem = (
                "classification"
                if pandas_types.is_integer_dtype(dtype)
                else "regression"
            )

        if embeddings_dict is None:
            embeddings_dict = dict()
        else:
            # Convertendo dicionário de embeddings para cache inicial
            data = dict()
            for vec_id, df in embeddings_dict.items():
                # Copiando DataFrame
                df = df.copy()

                # Criando coluna com valor dos embeddings
                df["embeddings"] = df.iloc[:, 1:].values.tolist()

                # Convertendo para dicionário de cache inicial
                initial_cache = {t: v for t, v in zip(df.text, df.embeddings)}

                # Criando cache pré inicializado desse vetorizador
                data[vec_id] = DictVectorizerCache(
                    cache_limit, initial_cache=initial_cache
                )

            embeddings_dict = data

        def _pipeline_priority(p: Pipeline) -> int:
            vectorizer = p.vectorizer

            if isinstance(vectorizer, AggregatedFeatureExtractor):
                return len(vectorizer.extractors)

            return 1

        # Obtendo as caches iniciais
        initial_features = self._df_to_dict(features_df)

        # Instanciando e fazendo sort das pipelines
        self._pipelines = pipelines
        self._pipelines = sorted(self._pipelines, key=_pipeline_priority)

        # Variáveis auxiliares
        self._seed = seed
        self._keep_all_pipelines = keep_all_pipelines
        self._dataset = dataset
        self._metrics = metrics
        self._best_criteria = criteria_best
        self._best_comparator = operator.gt if criteria_maximize else operator.lt
        self._problem = problem
        self._cache_limit = cache_limit
        self._frac_train = frac_train
        self._feature_cache = MixedFeatureCache(
            target_features=None,
            initial_cache=initial_features,
            max_limit=self._cache_limit,
        )
        self._vectorizer_cache = embeddings_dict
        self._stratified = stratified_ds
        self._disable_cache = force_disable_cache
        self._n_workers = n_workers
        self._validate()

    def run(self) -> ExperimentResult:
        """Executa o experimento com todas
        as pipelines.

        :return: resultado do experimento, retorna um objeto
            :py:class:`SimpleExperimentExtras`
            na chave `extras` de
            :py:attr:`~aibox.nlp.core.experiment.ExperimentResult`.
        """
        logger.info("Setting up experiment...")
        best_pipeline: Pipeline = None
        best_metrics = None
        best_test_predictions = None
        metrics_history = dict()
        pipeline_history = dict()
        rng = np.random.default_rng(self._seed)

        logger.info("Obtaining train and test split...")
        seed_splits = rng.integers(low=0, high=9999, endpoint=True)
        train, test = self._dataset.train_test_split(
            frac_train=self._frac_train, seed=seed_splits, stratified=self._stratified
        )
        X_train, y_train = train.text.to_numpy(), train.target.to_numpy()
        X_test, y_test = test.text.to_numpy(), test.target.to_numpy()
        logger.info("Train has %d samples, Test has %d samples.", len(train), len(test))

        def _update_best(pipeline, metrics, predictions):
            nonlocal best_pipeline
            nonlocal best_metrics
            nonlocal best_test_predictions
            best_pipeline = pipeline
            best_metrics = metrics
            best_test_predictions = predictions

        logger.info("Run started.")
        run_start = time.perf_counter()
        i = 0
        n_pipelines = len(self._pipelines)

        while self._pipelines:
            i += 1

            # Obtendo pipeline
            pipeline = self._pipelines.pop()
            name = pipeline.name
            logger.info('Started pipeline "%s" (%d/%d)', name, i, n_pipelines)

            # Obtendo pipeline a ser treinada
            maybe_cached_pipeline = self._maybe_cached_pipeline(pipeline)

            # Treinamento da pipeline
            maybe_cached_pipeline.fit(X_train, y_train, n_workers=self._n_workers)

            # Predições
            predictions = maybe_cached_pipeline.predict(
                X_test, n_workers=self._n_workers
            )

            # Cálculo das métricas
            metrics_result = {
                m.name(): m.compute(y_true=y_test, y_pred=predictions)
                for m in self._metrics
            }

            # Calculando melhor pipeline
            criteria = self._best_criteria.name()
            if best_pipeline is None or self._best_comparator(
                metrics_result[criteria], best_metrics[criteria]
            ):
                _update_best(pipeline, metrics_result, predictions)

            # Armazenando resultados das métricas
            #   no histórico.
            metrics_history[name] = metrics_result

            # Caso a pipeline deva ser guardada no
            #   histórico, salvamos ela.
            if self._keep_all_pipelines:
                pipeline_history[name] = pipeline

        run_duration = time.perf_counter() - run_start
        logger.info(
            "Run finished in %.2f seconds.\n" "Best pipeline: %s",
            run_duration,
            best_pipeline.name,
        )

        # Criando o objeto usado em "extras"
        extras = SimpleExperimentExtras(
            df_features=self._features_df(),
            embeddings=self._embeddings_to_dict_of_df(),
            run_duration=run_duration,
        )

        # Retornando resultados
        return ExperimentResult(
            best_pipeline=best_pipeline,
            best_metrics=best_metrics,
            best_pipeline_test_predictions=best_test_predictions,
            train_df=train,
            test_df=test,
            metrics_history=metrics_history,
            pipeline_history=pipeline_history,
            extras=extras,
        )

    def config(self) -> ExperimentConfiguration:
        return ExperimentConfiguration(
            dataset=self._dataset,
            metrics=self._metrics,
            best_criteria=self._best_criteria,
            extras=dict(
                problem=self._problem,
                keep_all_pipelines=self._keep_all_pipelines,
                fraction_train=self._frac_train,
                stratified_ds=self._stratified,
                cache_limit=self._cache_limit,
                force_disable_cache=self._disable_cache,
                n_workers=self._n_workers,
            ),
        )

    def _validate(self):
        """Realiza uma validação nos
        componentes da classe.
        """
        # Tem que existir pipelines
        assert len(self._pipelines) > 0

        # Não podem existir pipelines duplicadas
        names = [p.name for p in self._pipelines]
        assert len(names) == len(set(names)), names

        # Não podem existir métricas duplicadas
        metrics_names = list(m.name() for m in self._metrics)
        assert len(metrics_names) == len(set(metrics_names))

    def _maybe_cached_pipeline(self, pipeline: Pipeline) -> Pipeline:
        #  Se cache desativado, retorna a pipeline
        #   original. Se vetorização for identidade,
        #   nada a cachear.
        if self._disable_cache or isinstance(pipeline.vectorizer, IdentityVectorizer):
            return pipeline

        # Caso seja um extrator de características,
        #   atualizamos para utilizar uma versão cacheada.
        if isinstance(pipeline.vectorizer, FeatureExtractor):
            # Coletando informações sobre quais features são
            #   extraídas pelo vetorizador
            sample_features = pipeline.vectorizer.extract("Texto de exemplo.")

            # Atualizar a memória para retornar
            #   apenas essas características.
            self._feature_cache.target_features = set(sample_features.as_dict().keys())

            # Instanciando um novo extrator com cache.
            cached_extractor = CachedExtractor(pipeline.vectorizer, self._feature_cache)

            # Nova pipeline que compartilha o mesmo estimador,
            #   vetorizador e pós-processamento que a original.
            return Pipeline(
                vectorizer=cached_extractor,
                estimator=pipeline.estimator,
                postprocessing=pipeline.postprocessing,
                name=pipeline.name,
            )

        # Do contrário, retornamos a pipeline
        #   com um cache genérico.
        target_cls = CachedVectorizer
        if isinstance(pipeline.vectorizer, TrainableVectorizer):
            target_cls = TrainableCachedVectorizer

        vec_id = self._vectorizer_id(pipeline.vectorizer)
        if vec_id not in self._vectorizer_cache:
            self._vectorizer_cache[vec_id] = DictVectorizerCache(self._cache_limit)

        cached_vectorizer = target_cls(
            vectorizer=pipeline.vectorizer, memory=self._vectorizer_cache[vec_id]
        )

        return Pipeline(
            vectorizer=cached_vectorizer,
            estimator=pipeline.estimator,
            postprocessing=pipeline.postprocessing,
            name=pipeline.name,
        )

    def _features_df(self) -> pd.DataFrame:
        # Inicializando variável
        df = pd.DataFrame(dict(text=[]))

        # Coletando memória como dicionário
        memory_dict = self._feature_cache.as_dict()

        # Caso existam dados
        if len(memory_dict) > 0:
            memory_dict = {k: v.as_dict() for k, v in memory_dict.items()}

            # Construindo DataFrame
            df = pd.DataFrame(memory_dict).T.reset_index(names="text")

            # Removendo colunas que não possuem valor
            #   para todos os textos
            df = df.dropna(axis=1)

        return df

    def _embeddings_to_dict_of_df(self) -> dict[str, pd.DataFrame]:
        data = dict()
        for k, v in self._vectorizer_cache.items():
            data[k] = pd.DataFrame(v.as_dict()).T.reset_index(names="text")
        return data

    def _vectorizer_id(self, v: Vectorizer) -> str:
        return v.__class__.__name__

    def _df_to_dict(self, df: pd.DataFrame | None) -> dict:
        if df is not None:
            keys = df.text.to_list()
            values = df.drop(columns="text").to_dict(orient="records")
            return dict(zip(keys, values))

        return dict()
