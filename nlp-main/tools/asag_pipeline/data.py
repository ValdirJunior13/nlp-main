from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


TARGET_QUESTION_NUMBERS = (1, 9, 11, 12)
REQUIRED_COLUMNS = {"question_id", "question_text", "answer_text", "grade"}


@dataclass(frozen=True)
class DatasetSpec:
    source: str
    scenario: str
    question_numbers: tuple[int, ...] = TARGET_QUESTION_NUMBERS
    grade_policy: str = "primary"
    duplicate_policy: str = "preserve"
    test_size: float = 0.30
    random_state: int = 42
    grouped_candidates: int = 512


@dataclass
class DatasetBundle:
    spec: DatasetSpec
    train: pd.DataFrame
    test: pd.DataFrame
    fold: str = "holdout"
    test_question_number: int | None = None
    audit: dict | None = None


def normalize_answer(value: object) -> str:
    """Normalização conservadora: Unicode, minúsculas e espaços."""
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def _find_file(root: Path, candidates: Iterable[str], preferred_dir: str | None = None) -> Path:
    candidates = tuple(candidates)
    search_roots = []
    if preferred_dir:
        search_roots.append(root / preferred_dir)
    search_roots.append(root)

    for search_root in search_roots:
        for name in candidates:
            direct = search_root / name
            if direct.exists():
                return direct

    matches: list[Path] = []
    for name in candidates:
        matches.extend(root.rglob(name))
    if not matches:
        raise FileNotFoundError(
            f"Nenhum dos arquivos {candidates} foi encontrado dentro de '{root}'."
        )
    matches = sorted(set(matches), key=lambda p: (len(p.parts), str(p)))
    return matches[0]


def _load_questions(dataset_root: Path) -> pd.DataFrame:
    path = _find_file(
        dataset_root,
        ("questionsv2.csv", "questions.csv"),
        preferred_dir="galhardi",
    )
    questions = pd.read_csv(path)
    expected = {"question_id", "question_text"}
    missing = expected - set(questions.columns)
    if missing:
        raise ValueError(f"{path.name} não possui as colunas: {sorted(missing)}")
    questions = questions[["question_id", "question_text"]].drop_duplicates("question_id")
    questions = questions.reset_index(drop=True)
    questions["question_number"] = np.arange(1, len(questions) + 1)
    return questions


def _validate_core(df: pd.DataFrame, label: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{label} não possui as colunas obrigatórias: {sorted(missing)}")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        nulls = df[list(REQUIRED_COLUMNS)].isna().sum()
        nulls = nulls[nulls > 0].to_dict()
        raise ValueError(f"{label} possui valores ausentes nas colunas principais: {nulls}")
    invalid_grades = sorted(set(df["grade"].unique()) - {0, 1, 2, 3})
    if invalid_grades:
        raise ValueError(f"{label} possui notas fora do intervalo 0..3: {invalid_grades}")


def _canonicalize(
    df: pd.DataFrame,
    source: str,
    questions: pd.DataFrame,
    grade_source: str,
) -> pd.DataFrame:
    result = df.copy().reset_index(drop=True)
    if "question_text" not in result.columns:
        result = result.merge(
            questions[["question_id", "question_text"]],
            on="question_id",
            how="left",
            validate="many_to_one",
        )
    if "question_number" not in result.columns:
        result = result.merge(
            questions[["question_id", "question_number"]],
            on="question_id",
            how="left",
            validate="many_to_one",
        )

    _validate_core(result, source)
    if result["question_number"].isna().any():
        unknown = sorted(result.loc[result["question_number"].isna(), "question_id"].unique())
        raise ValueError(f"IDs de questão sem mapeamento em questions.csv: {unknown}")

    if "source_row" not in result.columns:
        result["source_row"] = np.arange(1, len(result) + 1)
    if "record_id" not in result.columns:
        result["record_id"] = [f"{source}:{i}" for i in result["source_row"]]

    result["source"] = source
    result["grade_source"] = grade_source
    result["answer_normalized"] = result["answer_text"].map(normalize_answer)
    result["answer_key"] = (
        result["question_id"].astype(str) + "::" + result["answer_normalized"]
    )
    counts = result["answer_key"].value_counts()
    result["duplicate_count"] = result["answer_key"].map(counts).astype(int)
    result["question_number"] = result["question_number"].astype(int)
    return result


def load_anderson(dataset_root: Path, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scenario not in {"df_11", "df_12"}:
        raise ValueError(f"Cenário Anderson inválido: {scenario}")
    questions = _load_questions(dataset_root)
    train_path = _find_file(
        dataset_root,
        (f"{scenario}_train.csv",),
        preferred_dir="anderson",
    )
    test_path = _find_file(
        dataset_root,
        (f"{scenario}_test.csv",),
        preferred_dir="anderson",
    )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["source_row"] = train.get("id", pd.Series(np.arange(1, len(train) + 1)))
    test["source_row"] = test.get("id", pd.Series(np.arange(1, len(test) + 1)))
    train["record_id"] = scenario + ":" + train["source_row"].astype(str)
    test["record_id"] = scenario + ":" + test["source_row"].astype(str)
    return (
        _canonicalize(train, "anderson", questions, "primary"),
        _canonicalize(test, "anderson", questions, "primary"),
    )


def load_galhardi(dataset_root: Path, grade_policy: str = "primary") -> pd.DataFrame:
    questions = _load_questions(dataset_root)
    if grade_policy == "primary":
        candidates = ("student_answers_and_grades_v2.csv",)
    elif grade_policy == "other":
        candidates = ("student_answers_and_grades_v2_other_graders.csv",)
    else:
        raise ValueError(
            "grade_policy deve ser 'primary' ou 'other'. "
            "Consenso não é aplicado automaticamente porque os arquivos não têm um ID comum."
        )
    path = _find_file(dataset_root, candidates, preferred_dir="galhardi")
    data = pd.read_csv(path)
    data["source_row"] = np.arange(1, len(data) + 1)
    data["record_id"] = f"galhardi_{grade_policy}:" + data["source_row"].astype(str)
    return _canonicalize(data, "galhardi", questions, grade_policy)


def _row_split_per_question(df: pd.DataFrame, spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("question_id", sort=True):
        try:
            train, test = train_test_split(
                group,
                test_size=spec.test_size,
                random_state=spec.random_state,
                stratify=group["grade"],
            )
        except ValueError:
            train, test = train_test_split(
                group,
                test_size=spec.test_size,
                random_state=spec.random_state,
            )
        train_parts.append(train)
        test_parts.append(test)
    return _shuffle(pd.concat(train_parts), spec.random_state), _shuffle(
        pd.concat(test_parts), spec.random_state
    )


def _grade_distribution(values: pd.Series) -> np.ndarray:
    # Cenários deduplicados usam a média das notas atribuídas à mesma
    # pergunta+resposta. Para equilibrar o split, essas médias contínuas são
    # convertidas apenas aqui para a classe ordinal mais próxima. O alvo usado
    # pelos regressores continua sendo a média original, sem arredondamento.
    numeric = pd.to_numeric(values, errors="coerce").clip(0, 3)
    ordinal = np.floor(numeric + 0.5).astype(int)
    counts = ordinal.value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0)
    return counts.to_numpy(dtype=float)


def _deduplicate_question_answers(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém um registro por pergunta+resposta e agrega notas pela média.

    A média é adequada ao problema de regressão e evita escolher
    arbitrariamente uma nota quando textos idênticos receberam avaliações
    diferentes. As quantidades originais e os conflitos ficam preservados em
    colunas de auditoria.
    """
    rows: list[pd.Series] = []
    ordered = df.sort_values(["question_number", "answer_key", "source_row"])
    for _, group in ordered.groupby("answer_key", sort=True):
        representative = group.iloc[0].copy()
        grades = pd.to_numeric(group["grade"], errors="raise").astype(float)
        representative["grade"] = float(grades.mean())
        representative["dedup_original_rows"] = int(len(group))
        representative["grade_observations"] = int(len(grades))
        representative["grade_unique_count"] = int(grades.nunique())
        representative["grade_std_within_key"] = float(grades.std(ddof=0))
        representative["grade_conflict"] = bool(grades.nunique() > 1)
        representative["grade_aggregation"] = "mean"
        rows.append(representative)

    deduplicated = pd.DataFrame(rows).reset_index(drop=True)
    deduplicated["duplicate_count"] = deduplicated["dedup_original_rows"].astype(int)
    deduplicated["record_id"] = [
        f"{source}:dedup:{index}"
        for index, source in enumerate(deduplicated["source"], start=1)
    ]
    return deduplicated


def _stable_deduplicated_split_per_question(
    df: pd.DataFrame,
    spec: DatasetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split reproduzível e independente das notas de cada avaliador.

    Após a deduplicação cada answer_key é única. Ordenar pelas chaves antes do
    sorteio garante que primary e other recebam as mesmas respostas em treino
    e teste quando possuem o mesmo conjunto de textos.
    """
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("question_id", sort=True):
        group = group.sort_values("answer_key").reset_index(drop=True)
        train, test = train_test_split(
            group,
            test_size=spec.test_size,
            random_state=spec.random_state,
            shuffle=True,
        )
        train_parts.append(train)
        test_parts.append(test)
    train = _shuffle(pd.concat(train_parts), spec.random_state)
    test = _shuffle(pd.concat(test_parts), spec.random_state)
    overlap = set(train["answer_key"]) & set(test["answer_key"])
    if overlap:
        raise RuntimeError(
            f"Split deduplicado inválido: {len(overlap)} chaves sobrepostas."
        )
    return train, test


def compare_galhardi_graders(
    dataset_root: Path,
    question_numbers: tuple[int, ...] = TARGET_QUESTION_NUMBERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara avaliadores no nível de pergunta+resposta única.

    Os arquivos não expõem um ID comum de avaliação individual. Por isso a
    unidade comparável e explicitamente auditável é a chave normalizada
    pergunta+resposta. Duplicatas internas de cada avaliador são consolidadas
    pela média antes da comparação.
    """
    frames: dict[str, pd.DataFrame] = {}
    for evaluator in ("primary", "other"):
        data = load_galhardi(dataset_root, evaluator)
        data = data[data["question_number"].isin(question_numbers)].copy()
        deduplicated = _deduplicate_question_answers(data)
        frames[evaluator] = deduplicated

    primary_columns = [
        "answer_key", "question_number", "question_id", "question_text",
        "answer_text", "grade", "grade_observations", "grade_unique_count",
        "grade_conflict", "grade_std_within_key",
    ]
    other_columns = [
        "answer_key", "question_number", "question_id", "grade",
        "grade_observations", "grade_unique_count", "grade_conflict",
        "grade_std_within_key",
    ]
    details = frames["primary"][primary_columns].merge(
        frames["other"][other_columns],
        on=["answer_key", "question_number", "question_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_primary", "_other"),
    )
    details = details.rename(
        columns={
            "grade_primary": "Nota_Media_Primary",
            "grade_other": "Nota_Media_Other",
            "grade_observations_primary": "N_Observacoes_Primary",
            "grade_observations_other": "N_Observacoes_Other",
            "grade_unique_count_primary": "N_Notas_Unicas_Primary",
            "grade_unique_count_other": "N_Notas_Unicas_Other",
            "grade_conflict_primary": "Conflito_Interno_Primary",
            "grade_conflict_other": "Conflito_Interno_Other",
            "grade_std_within_key_primary": "Desvio_Interno_Primary",
            "grade_std_within_key_other": "Desvio_Interno_Other",
        }
    )
    details["Diferenca_Other_Menos_Primary"] = (
        details["Nota_Media_Other"] - details["Nota_Media_Primary"]
    )
    details["Diferenca_Absoluta"] = details[
        "Diferenca_Other_Menos_Primary"
    ].abs()
    details["Nota_Ordinal_Primary"] = np.floor(
        details["Nota_Media_Primary"].clip(0, 3) + 0.5
    ).astype(int)
    details["Nota_Ordinal_Other"] = np.floor(
        details["Nota_Media_Other"].clip(0, 3) + 0.5
    ).astype(int)
    details["Acordo_Ordinal_Exato"] = (
        details["Nota_Ordinal_Primary"] == details["Nota_Ordinal_Other"]
    )

    def summarize(frame: pd.DataFrame, label: str) -> dict:
        primary = frame["Nota_Media_Primary"].astype(float)
        other = frame["Nota_Media_Other"].astype(float)
        ordinal_primary = frame["Nota_Ordinal_Primary"].astype(int)
        ordinal_other = frame["Nota_Ordinal_Other"].astype(int)
        try:
            kappa = float(
                cohen_kappa_score(
                    ordinal_primary,
                    ordinal_other,
                    labels=[0, 1, 2, 3],
                    weights="quadratic",
                )
            )
        except ValueError:
            kappa = np.nan
        return {
            "Escopo": label,
            "Questao": (
                int(frame["question_number"].iloc[0])
                if frame["question_number"].nunique() == 1
                else np.nan
            ),
            "N_Pares_Unicos_Comparados": len(frame),
            "Media_Primary": float(primary.mean()),
            "Media_Other": float(other.mean()),
            "Vies_Other_Menos_Primary": float((other - primary).mean()),
            "MAE_Entre_Avaliadores": float(np.abs(other - primary).mean()),
            "RMSE_Entre_Avaliadores": float(np.sqrt(np.mean((other - primary) ** 2))),
            "Correlacao_Pearson": float(primary.corr(other, method="pearson")),
            "Correlacao_Spearman": float(primary.corr(other, method="spearman")),
            "Kappa_Quadratico_Ordinal": kappa,
            "Acordo_Ordinal_Exato": float((ordinal_primary == ordinal_other).mean()),
            "Diferenca_Ordinal_Ate_1": float(
                (np.abs(ordinal_other - ordinal_primary) <= 1).mean()
            ),
            "Unidade_Comparacao": "pergunta+resposta normalizada; nota média por avaliador",
        }

    summaries = [summarize(details, "Todas as questões")]
    for question_number, group in details.groupby("question_number", sort=True):
        summaries.append(summarize(group, f"Questão {int(question_number)}"))
    summary = pd.DataFrame(summaries)
    details = details.sort_values(
        ["question_number", "Diferenca_Absoluta"], ascending=[True, False]
    ).reset_index(drop=True)
    return summary, details


def _grouped_split_one_question(
    group: pd.DataFrame,
    spec: DatasetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group["answer_key"].nunique() < 2:
        raise ValueError(
            f"Questão {group['question_number'].iloc[0]} tem menos de dois grupos de resposta."
        )

    splitter = GroupShuffleSplit(
        n_splits=spec.grouped_candidates,
        test_size=spec.test_size,
        random_state=spec.random_state,
    )
    target_distribution = _grade_distribution(group["grade"])
    best_score = float("inf")
    best_indices: tuple[np.ndarray, np.ndarray] | None = None

    for train_idx, test_idx in splitter.split(group, groups=group["answer_key"]):
        test = group.iloc[test_idx]
        size_error = abs(len(test) / len(group) - spec.test_size)
        grade_error = np.abs(_grade_distribution(test["grade"]) - target_distribution).mean()
        score = (2.0 * size_error) + grade_error
        if score < best_score:
            best_score = score
            best_indices = (train_idx, test_idx)

    if best_indices is None:
        raise RuntimeError("Não foi possível construir o split agrupado.")
    train_idx, test_idx = best_indices
    return group.iloc[train_idx].copy(), group.iloc[test_idx].copy()


def _grouped_split_per_question(df: pd.DataFrame, spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("question_id", sort=True):
        train, test = _grouped_split_one_question(group.reset_index(drop=True), spec)
        train_parts.append(train)
        test_parts.append(test)
    train = _shuffle(pd.concat(train_parts), spec.random_state)
    test = _shuffle(pd.concat(test_parts), spec.random_state)
    overlap = set(train["answer_key"]) & set(test["answer_key"])
    if overlap:
        raise RuntimeError(f"Split agrupado inválido: {len(overlap)} chaves sobrepostas.")
    return train, test


def _shuffle(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def audit_bundle(bundle: DatasetBundle) -> dict:
    train = bundle.train
    test = bundle.test
    exact_train = set(zip(train["question_id"], train["answer_text"]))
    exact_test = set(zip(test["question_id"], test["answer_text"]))
    norm_overlap = set(train["answer_key"]) & set(test["answer_key"])
    exact_overlap = exact_train & exact_test
    all_data = pd.concat([train, test], ignore_index=True)
    duplicated = all_data.duplicated("answer_key", keep=False)
    if "dedup_original_rows" in all_data.columns:
        original_total = int(all_data["dedup_original_rows"].sum())
        removed_by_deduplication = original_total - len(all_data)
        conflicting_groups = int(all_data["grade_conflict"].fillna(False).sum())
        conflicting_original_rows = int(
            all_data.loc[
                all_data["grade_conflict"].fillna(False), "dedup_original_rows"
            ].sum()
        )
        grade_aggregation = "media_das_notas_por_pergunta_resposta"
    else:
        original_total = len(all_data)
        removed_by_deduplication = 0
        conflicting_groups = np.nan
        conflicting_original_rows = np.nan
        grade_aggregation = "nota_original"

    def counts_by_question(frame: pd.DataFrame) -> str:
        counts = frame["question_number"].value_counts().sort_index().to_dict()
        return str({int(question): int(count) for question, count in counts.items()})

    question_map = (
        all_data[["question_number", "question_id"]]
        .drop_duplicates()
        .sort_values("question_number")
    )
    question_map_text = str(
        {
            int(row.question_number): str(row.question_id)
            for row in question_map.itertuples(index=False)
        }
    )

    return {
        "Fonte": bundle.spec.source,
        "Cenario_Dados": bundle.spec.scenario,
        "Fold": bundle.fold,
        "Questao_Teste": bundle.test_question_number,
        "Avaliador": bundle.spec.grade_policy,
        "Politica_Duplicatas": bundle.spec.duplicate_policy,
        "Random_State": bundle.spec.random_state,
        "N_Total": len(all_data),
        "N_Total_Original": original_total,
        "Linhas_Removidas_Deduplicacao": removed_by_deduplication,
        "Grupos_Nota_Conflitante": conflicting_groups,
        "Linhas_Originais_Conflitantes": conflicting_original_rows,
        "Criterio_Nota_Deduplicada": grade_aggregation,
        "N_Treino": len(train),
        "N_Teste": len(test),
        "N_Questoes_Total": all_data["question_id"].nunique(),
        "N_Questoes_Treino": train["question_id"].nunique(),
        "N_Questoes_Teste": test["question_id"].nunique(),
        "Questoes_Treino": ",".join(map(str, sorted(train["question_number"].unique()))),
        "Questoes_Teste": ",".join(map(str, sorted(test["question_number"].unique()))),
        "Mapa_Questao_Numero_ID": question_map_text,
        "Respostas_Por_Questao_Total": counts_by_question(all_data),
        "Respostas_Por_Questao_Treino": counts_by_question(train),
        "Respostas_Por_Questao_Teste": counts_by_question(test),
        "Pares_Unicos": all_data["answer_key"].nunique(),
        "Linhas_Em_Grupos_Duplicados": int(duplicated.sum()),
        "Excesso_Duplicatas": int(len(all_data) - all_data["answer_key"].nunique()),
        "Leakage_Chaves_Exatas": len(exact_overlap),
        "Leakage_Chaves_Normalizadas": len(norm_overlap),
        "Linhas_Teste_Afetadas": int(test["answer_key"].isin(norm_overlap).sum()),
        "Media_Nota_Treino": float(train["grade"].mean()),
        "Media_Nota_Teste": float(test["grade"].mean()),
        "Distribuicao_Notas_Treino": str(train["grade"].value_counts().sort_index().to_dict()),
        "Distribuicao_Notas_Teste": str(test["grade"].value_counts().sort_index().to_dict()),
    }


def build_datasets(dataset_root: Path, spec: DatasetSpec) -> list[DatasetBundle]:
    dataset_root = Path(dataset_root).resolve()
    if spec.source == "anderson":
        train, test = load_anderson(dataset_root, spec.scenario)
        bundle = DatasetBundle(spec=spec, train=train, test=test)
        bundle.audit = audit_bundle(bundle)
        return [bundle]

    if spec.source != "galhardi":
        raise ValueError(f"Fonte desconhecida: {spec.source}")

    data = load_galhardi(dataset_root, spec.grade_policy)
    data = data[data["question_number"].isin(spec.question_numbers)].copy()
    found = set(data["question_number"].unique())
    missing = set(spec.question_numbers) - found
    if missing:
        raise ValueError(f"Questões solicitadas não encontradas: {sorted(missing)}")

    bundles: list[DatasetBundle] = []
    if spec.scenario == "four_questions_row":
        train, test = _row_split_per_question(data, spec)
        bundles.append(DatasetBundle(spec=spec, train=train, test=test, fold="holdout_linha"))
    elif spec.scenario == "four_questions_grouped":
        train, test = _grouped_split_per_question(data, spec)
        bundles.append(DatasetBundle(spec=spec, train=train, test=test, fold="holdout_agrupado"))
    elif spec.scenario == "four_questions_deduplicated":
        deduplicated = _deduplicate_question_answers(data)
        train, test = _stable_deduplicated_split_per_question(deduplicated, spec)
        bundles.append(
            DatasetBundle(
                spec=spec,
                train=train,
                test=test,
                fold="holdout_deduplicado",
            )
        )
    elif spec.scenario == "leave_one_question_out":
        for question_number in spec.question_numbers:
            test = data[data["question_number"] == question_number].copy().reset_index(drop=True)
            train = data[data["question_number"] != question_number].copy().reset_index(drop=True)
            bundles.append(
                DatasetBundle(
                    spec=spec,
                    train=train,
                    test=test,
                    fold=f"test_q{question_number}",
                    test_question_number=question_number,
                )
            )
    elif spec.scenario == "leave_one_question_out_deduplicated":
        deduplicated = _deduplicate_question_answers(data)
        for question_number in spec.question_numbers:
            test = deduplicated[
                deduplicated["question_number"] == question_number
            ].copy().reset_index(drop=True)
            train = deduplicated[
                deduplicated["question_number"] != question_number
            ].copy().reset_index(drop=True)
            bundles.append(
                DatasetBundle(
                    spec=spec,
                    train=train,
                    test=test,
                    fold=f"test_q{question_number}_dedup",
                    test_question_number=question_number,
                )
            )
    else:
        raise ValueError(f"Cenário Galhardi inválido: {spec.scenario}")

    for bundle in bundles:
        bundle.audit = audit_bundle(bundle)
        if spec.scenario != "four_questions_row" and bundle.audit["Leakage_Chaves_Normalizadas"]:
            raise RuntimeError(
                f"Leakage detectado em {bundle.fold}: "
                f"{bundle.audit['Leakage_Chaves_Normalizadas']} chaves."
            )
    return bundles
