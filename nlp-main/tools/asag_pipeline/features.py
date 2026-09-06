from __future__ import annotations

import multiprocessing as mp
import queue
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import DatasetBundle, normalize_answer


EMBEDDING_MODELS = {
    "minilm": "paraphrase-multilingual-MiniLM-L12-v2",
    "bertimbau": "neuralmind/bert-base-portuguese-cased",
}
POS_RELEVANTES = {"NOUN", "VERB", "ADJ", "ADV"}

# Uma resposta pode legitimamente gastar vários minutos no conjunto completo,
# sobretudo na primeira execução que aquece os modelos. O watchdog é reiniciado
# sempre que a próxima métrica começa; portanto, o limite abaixo detecta uma
# função individual bloqueada, e não limita o tempo total da resposta.
COHMETRIX_FEATURE_TIMEOUT_SECONDS = 60
COHMETRIX_STARTUP_TIMEOUT_SECONDS = 120
COHMETRIX_CHECKPOINT_EVERY = 10
COHMETRIX_MAX_CONSECUTIVE_TIMEOUTS = 3

# O CohMetrix-BR 0.1.5 tenta calcular estas métricas com um modelo CBOW
# externo. A implementação faz download por HTTP sem timeout e usa a API
# antiga ``wv.vocab`` sobre um ``KeyedVectors`` do Gensim 4. Nos arquivos
# históricos de Anderson, as três colunas já contêm exclusivamente os valores
# de fallback abaixo. Mantê-los preserva o esquema e a comparabilidade dos
# experimentos, sem introduzir informação artificial nem bloquear a execução.
COHMETRIX_ANDERSON_FALLBACKS = {
    "wrdfrqc": 1.0,
    "wrdfrqa": 0.0,
    "wrdfrqmc": 0.0,
}
COHMETRIX_COMPATIBILITY_PROFILE = "anderson_fallback_v1"


def _anderson_compatible_rdl2(values: dict[str, float]) -> float:
    """Reproduz o RDL2 presente nos quatro caches históricos de Anderson.

    A fórmula publicada inclui ``22.205 * WRDFRQmc``. Como o extrator usado
    nos experimentos históricos devolveu WRDFRQmc=0 para todas as respostas,
    o termo é nulo. A expressão abaixo foi validada nos 19.279 registros não
    nulos de Anderson (erro máximo absoluto inferior a 1.4e-13).
    """
    return (
        -45.032
        + 52.230 * values["crfcwo1"]
        + 61.306 * values["synstruta"]
        + 22.205 * values["wrdfrqmc"]
    )


def build_combined_text(df: pd.DataFrame) -> list[str]:
    return (
        df["question_text"].fillna("").astype(str)
        + "\n Resposta: "
        + df["answer_text"].fillna("").astype(str)
    ).tolist()


class FeatureResources:
    """Carrega modelos uma vez e reutiliza resultados durante toda a bateria."""

    def __init__(self, embedding_key: str, spacy_model: str = "pt_core_news_lg"):
        if embedding_key not in EMBEDDING_MODELS:
            raise ValueError(f"Embedding desconhecido: {embedding_key}")
        self.embedding_key = embedding_key
        self.embedding_model_name = EMBEDDING_MODELS[embedding_key]
        self.spacy_model = spacy_model
        self._nlp = None
        self._stopwords: set[str] | None = None
        self._embedding_model = None
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._preprocessing_cache: dict[str, dict[str, str]] = {}

    def _load_nlp(self) -> None:
        if self._nlp is not None:
            return
        import nltk
        import spacy
        from nltk.corpus import stopwords

        try:
            stopwords.words("portuguese")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        self._stopwords = set(stopwords.words("portuguese"))
        self._nlp = spacy.load(self.spacy_model, disable=["ner", "parser"])

    def _load_embedding_model(self) -> None:
        if self._embedding_model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._embedding_model = SentenceTransformer(self.embedding_model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load_embedding_model()
        missing = list(dict.fromkeys(text for text in texts if text not in self._embedding_cache))
        if missing:
            vectors = self._embedding_model.encode(missing, show_progress_bar=True)
            for text, vector in zip(missing, vectors):
                self._embedding_cache[text] = np.asarray(vector, dtype=float)
        return np.vstack([self._embedding_cache[text] for text in texts])

    def preprocess_variants(self, texts: list[str], batch_size: int = 64) -> dict[str, list[str]]:
        self._load_nlp()
        missing = list(dict.fromkeys(text for text in texts if text not in self._preprocessing_cache))
        if missing:
            for text, doc in zip(
                missing,
                self._nlp.pipe(missing, batch_size=batch_size, n_process=1),
            ):
                tokens_cru: list[str] = []
                tokens_stop: list[str] = []
                tokens_lemma: list[str] = []
                tokens_lemma_pos: list[str] = []
                for token in doc:
                    if not token.is_alpha:
                        continue
                    lower = token.text.lower()
                    tokens_cru.append(lower)
                    if token.is_stop or lower in self._stopwords:
                        continue
                    tokens_stop.append(lower)
                    lemma = token.lemma_.lower()
                    tokens_lemma.append(lemma)
                    if token.pos_ in POS_RELEVANTES:
                        tokens_lemma_pos.append(lemma)
                self._preprocessing_cache[text] = {
                    "cru": " ".join(tokens_cru),
                    "stopwords": " ".join(tokens_stop),
                    "stopwords_lemma": " ".join(tokens_lemma),
                    "stopwords_lemma_pos": " ".join(tokens_lemma_pos),
                }
        keys = ("cru", "stopwords", "stopwords_lemma", "stopwords_lemma_pos")
        return {key: [self._preprocessing_cache[text][key] for text in texts] for key in keys}


def _tfidf_frames(
    train_texts: list[str],
    test_texts: list[str],
    max_features: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vectorizer = TfidfVectorizer(max_features=max_features)
    train_matrix = vectorizer.fit_transform(train_texts).toarray()
    test_matrix = vectorizer.transform(test_texts).toarray()
    columns = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    return pd.DataFrame(train_matrix, columns=columns), pd.DataFrame(test_matrix, columns=columns)


def _candidate_file(root: Path, filename: str, preferred_dir: str | None = None) -> Path | None:
    candidates = []
    if preferred_dir:
        candidates.append(root / preferred_dir / filename)
    candidates.append(root / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob(filename), key=lambda p: (len(p.parts), str(p)))
    return matches[0] if matches else None


def _legacy_cohmetrix(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"resposta_original", "nota_original"}
    if not required.issubset(raw.columns):
        raise ValueError(f"{path.name} não possui {sorted(required)}")
    raw["_key"] = raw["resposta_original"].map(normalize_answer)
    raw = raw.drop_duplicates("_key")
    frame = df.copy()
    frame["_key"] = frame["answer_text"].map(normalize_answer)
    merged = frame.merge(raw, on="_key", how="left", suffixes=("", "_coh"))
    remove = {
        "resposta_original", "nota_original", "answer_text", "grade", "_key",
        "question_text", "question_number", "answer_normalized", "answer_key",
        "duplicate_count", "source_row", "record_id", "source", "grade_source", "id",
        "question_id",
    }
    feature_columns = [column for column in raw.columns if column not in remove]
    numeric = merged[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    numeric.columns = [f"coh_{column}" for column in numeric.columns]
    return numeric.reset_index(drop=True)


def _cohmetrix_worker(request_queue, response_queue) -> None:
    """Executa o Coh-Metrix isoladamente para permitir timeout real no Windows.

    O wrapper ``aibox`` chama ``nltk.download`` em toda resposta. Aqui usamos a
    mesma compreensão de métricas implementada por esse wrapper, diretamente
    sobre ``cohmetrixBR.features.FEATURES``. As três métricas de frequência
    defeituosas mantêm os valores constantes observados nos caches de Anderson.
    """
    try:
        response_queue.put(
            ("startup_progress", None, "importando cohmetrixBR.features")
        )
        from cohmetrixBR import features as coh_features

        feature_functions = tuple(coh_features.FEATURES)
        response_queue.put(
            (
                "startup_progress",
                None,
                f"{len(feature_functions)} métricas carregadas; "
                f"{len(COHMETRIX_ANDERSON_FALLBACKS)} em modo de "
                "compatibilidade Anderson e RDL2 derivado",
            )
        )
        response_queue.put(("ready", None, None))
    except BaseException as exc:  # devolve falha de inicialização ao processo pai
        response_queue.put(
            ("init_error", None, f"{type(exc).__name__}: {exc}")
        )
        return

    while True:
        task = request_queue.get()
        if task is None:
            return
        task_id, text = task
        try:
            values = {}
            total_features = len(feature_functions)
            for position, function in enumerate(feature_functions, start=1):
                feature_name = function.__name__.lower()
                response_queue.put(
                    (
                        "feature_progress",
                        task_id,
                        (position, total_features, feature_name),
                    )
                )
                if feature_name in COHMETRIX_ANDERSON_FALLBACKS:
                    values[feature_name] = COHMETRIX_ANDERSON_FALLBACKS[
                        feature_name
                    ]
                elif feature_name == "rdl2":
                    values[feature_name] = _anderson_compatible_rdl2(values)
                else:
                    values[feature_name] = float(function(text))
            response_queue.put(("result", task_id, values))
        except BaseException as exc:
            response_queue.put(
                ("error", task_id, f"{type(exc).__name__}: {exc}")
            )


class _CohMetrixProcess:
    """Worker reiniciável que protege a execução contra bloqueios internos."""

    def __init__(
        self,
        feature_timeout_seconds: int = COHMETRIX_FEATURE_TIMEOUT_SECONDS,
        startup_timeout_seconds: int = COHMETRIX_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        self.feature_timeout_seconds = feature_timeout_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self._context = mp.get_context("spawn")
        self._process = None
        self._request_queue = None
        self._response_queue = None
        self._task_id = 0

    def _terminate(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=10)
        self._process = None
        for process_queue in (self._request_queue, self._response_queue):
            if process_queue is not None:
                try:
                    process_queue.cancel_join_thread()
                    process_queue.close()
                except (OSError, ValueError):
                    pass
        self._request_queue = None
        self._response_queue = None

    def _start(self) -> None:
        self._terminate()
        self._request_queue = self._context.Queue()
        self._response_queue = self._context.Queue()
        self._process = self._context.Process(
            target=_cohmetrix_worker,
            args=(self._request_queue, self._response_queue),
            daemon=True,
        )
        self._process.start()
        deadline = time.monotonic() + self.startup_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate()
                raise TimeoutError(
                    "O processo do Coh-Metrix não inicializou dentro de "
                    f"{self.startup_timeout_seconds}s."
                )
            try:
                kind, _, payload = self._response_queue.get(timeout=remaining)
            except queue.Empty as exc:
                self._terminate()
                raise TimeoutError(
                    "O processo do Coh-Metrix não inicializou dentro de "
                    f"{self.startup_timeout_seconds}s."
                ) from exc
            if kind == "startup_progress":
                print(f"\n      Inicialização: {payload}", flush=True)
                continue
            if kind == "ready":
                return
            self._terminate()
            raise RuntimeError(f"Falha ao inicializar Coh-Metrix: {payload}")

    def extract(self, text: str) -> dict[str, float]:
        if self._process is None or not self._process.is_alive():
            self._start()
        self._task_id += 1
        task_id = self._task_id
        self._request_queue.put((task_id, text))
        last_feature = "antes da primeira métrica"
        last_position = 0
        total_features = 0
        deadline = time.monotonic() + self.feature_timeout_seconds

        def timeout_error() -> TimeoutError:
            feature_location = (
                f"{last_position}/{total_features} '{last_feature}'"
                if total_features
                else last_feature
            )
            return TimeoutError(
                f"A métrica {feature_location} não terminou em "
                f"{self.feature_timeout_seconds}s."
            )

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate()
                raise timeout_error()
            try:
                kind, response_task_id, payload = self._response_queue.get(
                    timeout=remaining
                )
            except queue.Empty as exc:
                self._terminate()
                raise timeout_error() from exc
            if response_task_id != task_id:
                self._terminate()
                raise RuntimeError("Resposta inesperada do processo Coh-Metrix.")
            if kind == "feature_progress":
                position, total, feature_name = payload
                last_position = position
                total_features = total
                last_feature = feature_name
                deadline = time.monotonic() + self.feature_timeout_seconds
                print(
                    f"\r      Métrica {position}/{total}: {feature_name}",
                    end="",
                    flush=True,
                )
                continue
            if kind == "error":
                raise RuntimeError(payload)
            if kind != "result":
                raise RuntimeError(f"Mensagem inesperada do Coh-Metrix: {kind}")
            return payload

    def close(self) -> None:
        if self._process is not None and self._process.is_alive():
            try:
                self._request_queue.put(None)
                self._process.join(timeout=10)
            except (OSError, ValueError):
                pass
        self._terminate()


def _write_cohmetrix_cache(cache: pd.DataFrame, cache_path: Path) -> None:
    """Grava em arquivo temporário para não corromper o cache numa interrupção."""
    temporary_path = cache_path.with_name(f"{cache_path.name}.tmp")
    cache.to_csv(temporary_path, index=False)
    temporary_path.replace(cache_path)


def _append_cache_rows(
    cache: pd.DataFrame,
    rows: list[dict],
    cache_path: Path,
) -> pd.DataFrame:
    if not rows:
        return cache
    new_rows = pd.DataFrame(rows)
    updated = (
        new_rows.reset_index(drop=True)
        if cache.empty
        else pd.concat([cache, new_rows], ignore_index=True)
    )
    updated = updated.drop_duplicates("answer_normalized", keep="last")
    _write_cohmetrix_cache(updated, cache_path)
    rows.clear()
    return updated


def _generated_cohmetrix(
    df: pd.DataFrame,
    cache_path: Path,
    project_root: Path,
) -> pd.DataFrame:
    del project_root  # mantido na assinatura para compatibilidade com as chamadas atuais
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cache = pd.read_csv(cache_path, keep_default_na=False)
    else:
        cache = pd.DataFrame(
            columns=[
                "answer_normalized",
                "answer_text_cache",
                "cohmetrix_status",
                "cohmetrix_error",
                "cohmetrix_seconds",
                "cohmetrix_compatibility_profile",
            ]
        )

    if "cohmetrix_status" in cache.columns:
        reusable_cache = cache[
            cache["cohmetrix_status"].isin({"ok", "texto_invalido"})
        ]
    else:
        reusable_cache = cache
    # Falhas e timeouts ficam documentados no CSV, mas voltam a ser tentados
    # numa execução posterior. Somente resultados válidos são reutilizados.
    known = set(
        reusable_cache.get("answer_normalized", pd.Series(dtype=str))
        .dropna()
        .astype(str)
    )
    unique_answers = (
        df[["answer_normalized", "answer_text"]]
        .drop_duplicates("answer_normalized")
        .assign(answer_normalized=lambda frame: frame["answer_normalized"].astype(str))
        .loc[lambda x: ~x["answer_normalized"].isin(known)]
    )
    if not unique_answers.empty:
        worker = _CohMetrixProcess()
        pending_rows: list[dict] = []
        total = len(unique_answers)
        consecutive_timeouts = 0
        successful_seconds = 0.0
        successful_count = 0
        print(
            f"    Coh-Metrix: {total} respostas únicas pendentes; "
            f"checkpoint a cada {COHMETRIX_CHECKPOINT_EVERY}; "
            f"timeout por métrica={COHMETRIX_FEATURE_TIMEOUT_SECONDS}s.",
            flush=True,
        )
        try:
            for position, row in enumerate(
                unique_answers.itertuples(index=False), start=1
            ):
                text = str(row.answer_text)
                item = {
                    "answer_normalized": str(row.answer_normalized),
                    "answer_text_cache": text,
                    "cohmetrix_error": "",
                    "cohmetrix_compatibility_profile": (
                        COHMETRIX_COMPATIBILITY_PROFILE
                    ),
                }
                item_started_at = time.perf_counter()
                print(
                    f"\r    Coh-Metrix: processando {position}/{total}...",
                    end="",
                    flush=True,
                )
                if len(text.strip()) < 2 or not any(char.isalpha() for char in text):
                    item["cohmetrix_status"] = "texto_invalido"
                    consecutive_timeouts = 0
                else:
                    try:
                        item.update(worker.extract(text))
                        item["cohmetrix_status"] = "ok"
                        consecutive_timeouts = 0
                    except TimeoutError as exc:
                        item["cohmetrix_status"] = "timeout"
                        item["cohmetrix_error"] = str(exc)
                        consecutive_timeouts += 1
                    except Exception as exc:
                        item["cohmetrix_status"] = f"falha:{type(exc).__name__}"
                        item["cohmetrix_error"] = str(exc)[:1000]
                        consecutive_timeouts = 0
                item["cohmetrix_seconds"] = round(
                    time.perf_counter() - item_started_at, 3
                )
                if item["cohmetrix_status"] == "ok":
                    successful_seconds += item["cohmetrix_seconds"]
                    successful_count += 1
                pending_rows.append(item)

                checkpoint = (
                    position % COHMETRIX_CHECKPOINT_EVERY == 0
                    or position == total
                    or item["cohmetrix_status"] != "ok"
                )
                if checkpoint:
                    cache = _append_cache_rows(cache, pending_rows, cache_path)
                    if successful_count:
                        average = successful_seconds / successful_count
                        eta_text = f"{average * (total - position) / 60:.1f} min"
                    else:
                        eta_text = "indisponível até a primeira resposta válida"
                    error_suffix = ""
                    if item["cohmetrix_error"]:
                        error_suffix = (
                            " | erro="
                            + str(item["cohmetrix_error"]).replace("\n", " ")[:180]
                        )
                    print(
                        f"\r    Coh-Metrix: {position}/{total} | "
                        f"status={item['cohmetrix_status']} | "
                        f"tempo={item['cohmetrix_seconds']:.1f}s | "
                        f"ETA={eta_text} | cache salvo{error_suffix}",
                        flush=True,
                    )

                if consecutive_timeouts >= COHMETRIX_MAX_CONSECUTIVE_TIMEOUTS:
                    raise RuntimeError(
                        "Coh-Metrix interrompido após "
                        f"{consecutive_timeouts} timeouts consecutivos. "
                        f"Consulte o cache '{cache_path}' para identificar "
                        "as respostas afetadas."
                    )
        except KeyboardInterrupt:
            cache = _append_cache_rows(cache, pending_rows, cache_path)
            print(f"\n    Coh-Metrix interrompido; cache preservado em: {cache_path}")
            raise
        finally:
            worker.close()
            if pending_rows:
                cache = _append_cache_rows(cache, pending_rows, cache_path)

    requested_keys = set(df["answer_normalized"].astype(str))
    requested_cache = cache[
        cache["answer_normalized"].astype(str).isin(requested_keys)
    ]
    if "cohmetrix_status" in requested_cache.columns:
        status_counts = requested_cache["cohmetrix_status"].value_counts().to_dict()
        summary = ", ".join(
            f"{status}={count}" for status, count in sorted(status_counts.items())
        )
        print(f"    Resumo Coh-Metrix no split: {summary or 'sem registros'}")

    merged = df[["answer_normalized"]].merge(cache, on="answer_normalized", how="left")
    remove = {
        "answer_normalized",
        "answer_text_cache",
        "cohmetrix_status",
        "cohmetrix_error",
        "cohmetrix_seconds",
        "cohmetrix_compatibility_profile",
    }
    features = [column for column in merged.columns if column not in remove]
    numeric = merged[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    numeric.columns = [f"coh_{column}" for column in numeric.columns]
    return numeric.reset_index(drop=True)


def load_cohmetrix(
    bundle: DatasetBundle,
    dataset_root: Path,
    project_root: Path,
    cache_dir: Path,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if mode == "off":
        return None
    if bundle.spec.source == "anderson":
        train_path = _candidate_file(
            dataset_root,
            f"cohmetrix_{bundle.spec.scenario}_train.csv",
            "anderson",
        )
        test_path = _candidate_file(
            dataset_root,
            f"cohmetrix_{bundle.spec.scenario}_test.csv",
            "anderson",
        )
        if not train_path or not test_path:
            if mode == "generate":
                raise FileNotFoundError("Coh-Metrix Anderson ausente; preserve os caches históricos.")
            print(
                f"AVISO: cache Coh-Metrix Anderson ausente para {bundle.spec.scenario}; "
                "cenários Coh-Metrix serão omitidos."
            )
            return None
        return _legacy_cohmetrix(bundle.train, train_path), _legacy_cohmetrix(bundle.test, test_path)

    cache_path = cache_dir / "cohmetrix_galhardi.csv"
    if mode == "existing" and not cache_path.exists():
        raise FileNotFoundError(
            "Cache Coh-Metrix Galhardi ausente em "
            f"'{cache_path}'. Informe o cache correto com --cache-dir, execute "
            "uma vez com --cohmetrix generate ou use --cohmetrix off para "
            "omitir explicitamente essa representação."
        )
    return (
        _generated_cohmetrix(bundle.train, cache_path, project_root),
        _generated_cohmetrix(bundle.test, cache_path, project_root),
    )


def build_representations(
    bundle: DatasetBundle,
    resources: FeatureResources,
    profile: str,
    dataset_root: Path,
    project_root: Path,
    cache_dir: Path,
    cohmetrix_mode: str = "existing",
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    train_texts = build_combined_text(bundle.train)
    test_texts = build_combined_text(bundle.test)

    # Valida/gera primeiro a etapa mais frágil e demorada. Assim, uma falha do
    # Coh-Metrix não desperdiça o tempo de TF-IDF e embeddings.
    print("    Representações: carregando Coh-Metrix...", flush=True)
    coh = load_cohmetrix(
        bundle,
        dataset_root=dataset_root,
        project_root=project_root,
        cache_dir=cache_dir,
        mode=cohmetrix_mode,
    )
    if coh is not None:
        coh = coh[0].align(coh[1], join="outer", axis=1, fill_value=0)

    print("    Representações: pré-processando textos para TF-IDF...", flush=True)
    train_variants = resources.preprocess_variants(train_texts)
    test_variants = resources.preprocess_variants(test_texts)
    labels = {
        "cru": "Apenas TF-IDF (cru)",
        "stopwords": "Apenas TF-IDF (stopwords)",
        "stopwords_lemma": "Apenas TF-IDF (stopwords+lemma)",
        "stopwords_lemma_pos": "Apenas TF-IDF (stopwords+lemma+POS)",
    }
    tfidf: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    print("    Representações: vetorizando TF-IDF...", flush=True)
    for key in labels:
        tfidf[key] = _tfidf_frames(train_variants[key], test_variants[key])

    print(
        f"    Representações: gerando/reutilizando embeddings {resources.embedding_key}...",
        flush=True,
    )
    train_embeddings = resources.encode(train_texts)
    test_embeddings = resources.encode(test_texts)
    embedding_columns = [f"emb_{i}" for i in range(train_embeddings.shape[1])]
    emb_train = pd.DataFrame(train_embeddings, columns=embedding_columns)
    emb_test = pd.DataFrame(test_embeddings, columns=embedding_columns)

    processed_train, processed_test = tfidf["stopwords_lemma_pos"]

    if profile == "paper_like":
        representations = {
            "TFIDF": (processed_train, processed_test),
            "Embeddings": (emb_train, emb_test),
        }
        if coh is not None:
            representations["CohMetrix"] = coh
        return representations

    if profile != "full":
        raise ValueError(f"Perfil desconhecido: {profile}")

    representations = {labels[key]: value for key, value in tfidf.items()}
    representations["Apenas Embeddings"] = (emb_train, emb_test)
    representations["TF-IDF (processado) + Embeddings"] = (
        pd.concat([processed_train, emb_train], axis=1),
        pd.concat([processed_test, emb_test], axis=1),
    )
    if coh is not None:
        coh_train, coh_test = coh
        representations["Apenas Coh-Metrix"] = (coh_train, coh_test)
        representations["TF-IDF (processado) + Coh-Metrix"] = (
            pd.concat([processed_train, coh_train], axis=1),
            pd.concat([processed_test, coh_test], axis=1),
        )
        representations["Coh-Metrix + Embeddings"] = (
            pd.concat([coh_train, emb_train], axis=1),
            pd.concat([coh_test, emb_test], axis=1),
        )
        representations["Tudo (TF-IDF processado + Coh-Metrix + Emb)"] = (
            pd.concat([processed_train, coh_train, emb_train], axis=1),
            pd.concat([processed_test, coh_test, emb_test], axis=1),
        )
    return representations
