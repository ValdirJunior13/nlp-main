from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .data import DatasetBundle
from .features import EMBEDDING_MODELS, FeatureResources, build_representations


@dataclass(frozen=True)
class ExperimentOptions:
    profile: str = "full"
    embedding: str = "minilm"
    cohmetrix_mode: str = "existing"
    save_predictions: bool = False


def _models(profile: str) -> dict[str, object]:
    if profile == "paper_like":
        models: dict[str, object] = {
            "SVR": SVR(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Árvore de Decisão": DecisionTreeRegressor(random_state=42),
        }
    elif profile == "full":
        models = {
            "Regressão Linear": LinearRegression(),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "SVR": SVR(),
            "Árvore de Decisão": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
            "Rede Neural (MLP)": MLPRegressor(random_state=42, max_iter=1000),
        }
    else:
        raise ValueError(f"Perfil desconhecido: {profile}")

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(random_state=42)
    except ImportError:
        print("AVISO: xgboost não está instalado; o modelo XGBoost será ignorado.")
    return models


def expected_configuration_count(profile: str) -> int:
    """Quantidade esperada quando todas as representações estão disponíveis."""
    representation_count = {"full": 10, "paper_like": 3}.get(profile)
    if representation_count is None:
        raise ValueError(f"Perfil desconhecido: {profile}")
    return representation_count * len(_models(profile))


def _tree_model(name: str) -> bool:
    return name in {
        "Árvore de Decisão",
        "Random Forest",
        "HistGradientBoosting",
        "XGBoost",
    }


def _run_id(bundle: DatasetBundle, options: ExperimentOptions) -> str:
    parts = [
        bundle.spec.source,
        bundle.spec.scenario,
        bundle.fold,
        bundle.spec.grade_policy,
        options.embedding,
        options.profile,
    ]
    return "__".join(str(part).replace(" ", "_") for part in parts)


def run_experiment(
    bundle: DatasetBundle,
    resources: FeatureResources,
    options: ExperimentOptions,
    dataset_root: Path,
    project_root: Path,
    cache_dir: Path,
    checkpoint_path: Path | None = None,
    prediction_checkpoint_path: Path | None = None,
    resume: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("  -> Preparando representações...", flush=True)
    representations = build_representations(
        bundle=bundle,
        resources=resources,
        profile=options.profile,
        dataset_root=dataset_root,
        project_root=project_root,
        cache_dir=cache_dir,
        cohmetrix_mode=options.cohmetrix_mode,
    )
    models = _models(options.profile)
    y_train = bundle.train["grade"].to_numpy()
    y_test = bundle.test["grade"].to_numpy()
    audit = bundle.audit or {}
    run_id = _run_id(bundle, options)

    result_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    completed_results: set[tuple[str, str]] = set()
    completed_predictions: set[tuple[str, str]] = set()

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        previous = pd.read_csv(checkpoint_path)
        previous = previous[previous.get("Run_ID", pd.Series(dtype=str)) == run_id]
        if not previous.empty:
            result_rows.extend(previous.to_dict("records"))
            completed_results = set(
                zip(previous["Cenário"].astype(str), previous["Modelo"].astype(str))
            )
            print(
                f"  -> Retomada: {len(completed_results)} ajustes já concluídos.",
                flush=True,
            )

    if (
        resume
        and options.save_predictions
        and prediction_checkpoint_path is not None
        and prediction_checkpoint_path.exists()
    ):
        previous_predictions = pd.read_csv(prediction_checkpoint_path)
        previous_predictions = previous_predictions[
            previous_predictions.get("Run_ID", pd.Series(dtype=str)) == run_id
        ]
        if not previous_predictions.empty:
            prediction_frames.append(previous_predictions)
            completed_predictions = set(
                zip(
                    previous_predictions["Cenário"].astype(str),
                    previous_predictions["Modelo"].astype(str),
                )
            )

    def save_checkpoints() -> None:
        if checkpoint_path is not None and result_rows:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            current_results = pd.DataFrame(result_rows).drop_duplicates(
                ["Run_ID", "Cenário", "Modelo"], keep="last"
            )
            temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
            current_results.to_csv(temporary, index=False)
            temporary.replace(checkpoint_path)
        if (
            options.save_predictions
            and prediction_checkpoint_path is not None
            and prediction_frames
        ):
            prediction_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            current_predictions = pd.concat(prediction_frames, ignore_index=True)
            current_predictions = current_predictions.drop_duplicates(
                ["Run_ID", "Cenário", "Modelo", "Record_ID"], keep="last"
            )
            temporary = prediction_checkpoint_path.with_name(
                prediction_checkpoint_path.name + ".tmp"
            )
            current_predictions.to_csv(temporary, index=False)
            temporary.replace(prediction_checkpoint_path)

    representation_count = len(representations)
    model_count = len(models)
    for representation_index, (
        representation,
        (x_train_frame, x_test_frame),
    ) in enumerate(representations.items(), start=1):
        if x_train_frame.shape[1] == 0:
            print(f"AVISO: representação sem colunas ignorada: {representation}")
            continue
        x_train_frame, x_test_frame = x_train_frame.align(
            x_test_frame,
            join="outer",
            axis=1,
            fill_value=0,
        )
        x_train_frame = x_train_frame.apply(pd.to_numeric, errors="coerce").fillna(0)
        x_test_frame = x_test_frame.apply(pd.to_numeric, errors="coerce").fillna(0)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_frame)
        x_test_scaled = scaler.transform(x_test_frame)
        x_train_raw = x_train_frame.to_numpy()
        x_test_raw = x_test_frame.to_numpy()

        for model_index, (model_name, model_template) in enumerate(
            models.items(), start=1
        ):
            configuration_key = (representation, model_name)
            result_ready = configuration_key in completed_results
            prediction_ready = (
                not options.save_predictions
                or configuration_key in completed_predictions
            )
            if result_ready and prediction_ready:
                print(
                    f"  [{representation_index}/{representation_count}] "
                    f"[{model_index}/{model_count}] {representation} | "
                    f"{model_name}: reutilizado",
                    flush=True,
                )
                continue

            if _tree_model(model_name):
                x_train, x_test = x_train_raw, x_test_raw
            else:
                x_train, x_test = x_train_scaled, x_test_scaled

            print(
                f"  [{representation_index}/{representation_count}] "
                f"[{model_index}/{model_count}] {representation} | "
                f"{model_name}: treinando...",
                flush=True,
            )
            started_at = time.perf_counter()
            model = clone(model_template)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            common = {
                "Run_ID": run_id,
                "Fonte": bundle.spec.source,
                "Dataset_Base": "pt-asag-2018-v2",
                "Cenario_Dados": bundle.spec.scenario,
                "Fold": bundle.fold,
                "Questao_Teste": bundle.test_question_number,
                "Avaliador": bundle.spec.grade_policy,
                "Politica_Duplicatas": bundle.spec.duplicate_policy,
                "Embedding": options.embedding,
                "Embedding_Modelo": EMBEDDING_MODELS[options.embedding],
                "Perfil": options.profile,
                "Dataset": bundle.spec.scenario if bundle.spec.source == "galhardi" else bundle.spec.scenario,
                "Cenário": representation,
                "Modelo": model_name,
                "N_Treino": len(bundle.train),
                "N_Teste": len(bundle.test),
                "Leakage_Exato": audit.get("Leakage_Chaves_Exatas", 0),
                "Leakage_Normalizado": audit.get("Leakage_Chaves_Normalizadas", 0),
                "Linhas_Teste_Afetadas": audit.get("Linhas_Teste_Afetadas", 0),
                "Random_State": bundle.spec.random_state,
            }
            new_result = {
                **common,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R²": float(r2),
            }
            result_rows = [
                row
                for row in result_rows
                if (str(row.get("Cenário")), str(row.get("Modelo")))
                != configuration_key
            ]
            result_rows.append(new_result)
            completed_results.add(configuration_key)

            if options.save_predictions:
                new_predictions = pd.DataFrame(
                    {
                        **{key: value for key, value in common.items() if key not in {"N_Treino", "N_Teste"}},
                        "Record_ID": bundle.test["record_id"].to_numpy(),
                        "Question_Number": bundle.test["question_number"].to_numpy(),
                        "Question_ID": bundle.test["question_id"].to_numpy(),
                        "Answer_Text": bundle.test["answer_text"].to_numpy(),
                        "Nota_Real": y_test,
                        "Nota_Prevista": predictions,
                        "Erro_Absoluto": np.abs(y_test - predictions),
                    }
                )
                prediction_frames = [
                    frame
                    for frame in prediction_frames
                    if not (
                        "Cenário" in frame.columns
                        and "Modelo" in frame.columns
                        and len(frame)
                        and str(frame["Cenário"].iloc[0]) == representation
                        and str(frame["Modelo"].iloc[0]) == model_name
                    )
                ]
                prediction_frames.append(new_predictions)
                completed_predictions.add(configuration_key)

            save_checkpoints()
            elapsed = time.perf_counter() - started_at
            print(
                f"      concluído em {elapsed:.1f}s | "
                f"MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f} | "
                "checkpoint salvo",
                flush=True,
            )

    results = pd.DataFrame(result_rows)
    if not results.empty:
        results = results.sort_values(["RMSE", "MAE"]).reset_index(drop=True)
        if checkpoint_path is not None:
            results.to_csv(checkpoint_path, index=False)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    if not predictions.empty:
        predictions = predictions.drop_duplicates(
            ["Run_ID", "Cenário", "Modelo", "Record_ID"], keep="last"
        ).reset_index(drop=True)
    return results, predictions
