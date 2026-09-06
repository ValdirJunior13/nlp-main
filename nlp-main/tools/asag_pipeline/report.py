from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TFIDF_VARIANTS = [
    "Apenas TF-IDF (cru)",
    "Apenas TF-IDF (stopwords)",
    "Apenas TF-IDF (stopwords+lemma)",
    "Apenas TF-IDF (stopwords+lemma+POS)",
]

MODEL_DISPLAY_MAP = {
    "SVM": "SVR",
    "RF": "Random Forest",
    "DT": "Árvore de Decisão",
    "XGB": "XGBoost",
}

REPRESENTATION_DISPLAY_MAP = {
    "TFIDF": "Apenas TF-IDF (processado)",
    "Embeddings": "Apenas Embeddings",
    "CohMetrix": "Apenas Coh-Metrix",
}


def standardize_report_labels(results: pd.DataFrame) -> pd.DataFrame:
    """Padroniza somente a apresentação; não altera os CSVs experimentais."""
    standardized = results.copy()
    standardized["Modelo"] = standardized["Modelo"].replace(MODEL_DISPLAY_MAP)
    standardized["Cenário"] = standardized["Cenário"].replace(
        REPRESENTATION_DISPLAY_MAP
    )
    return standardized


def _embedding_used(representation: object, embedding: object) -> str:
    if "emb" not in str(representation).lower():
        return "N/A"
    return str(embedding)


def create_ablation(results: pd.DataFrame) -> pd.DataFrame:
    filtered = results[results["Cenário"].isin(TFIDF_VARIANTS)].copy()
    if filtered.empty:
        return pd.DataFrame()
    index = [
        "Fonte", "Cenario_Dados", "Fold", "Avaliador", "Embedding", "Perfil", "Modelo"
    ]
    pivot = filtered.pivot_table(index=index, columns="Cenário", values="R²", aggfunc="first")
    pivot = pivot.reindex(columns=TFIDF_VARIANTS)
    pivot.columns = ["Cru", "Stopwords", "Stopwords+Lemma", "Stopwords+Lemma+POS"]
    pivot["Delta_R2_POS_menos_Cru"] = pivot["Stopwords+Lemma+POS"] - pivot["Cru"]
    return pivot.reset_index()


def create_best_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results.copy()
    group_columns = [
        "Fonte", "Cenario_Dados", "Fold", "Questao_Teste", "Avaliador",
        "Embedding", "Perfil", "Modelo",
    ]
    valid = results.dropna(subset=["RMSE"]).copy()
    indices = valid.groupby(group_columns, dropna=False)["RMSE"].idxmin()
    columns = group_columns + [
        "Cenário", "N_Treino", "N_Teste", "Leakage_Normalizado",
        "Linhas_Teste_Afetadas", "MAE", "RMSE", "R²",
    ]
    return valid.loc[indices, columns].sort_values(group_columns).reset_index(drop=True)


def create_loq_summary(results: pd.DataFrame) -> pd.DataFrame:
    loq = results[
        results["Cenario_Dados"].astype(str).str.startswith("leave_one_question_out")
    ].copy()
    if loq.empty:
        return pd.DataFrame()
    group_columns = [
        "Cenario_Dados", "Avaliador", "Embedding", "Perfil", "Cenário", "Modelo"
    ]
    summary = loq.groupby(group_columns)[["MAE", "RMSE", "R²"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().sort_values("RMSE_mean")


def create_executive_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    valid = results.dropna(subset=["RMSE"]).copy()
    is_loq = valid["Cenario_Dados"].astype(str).str.startswith(
        "leave_one_question_out"
    )
    non_loq = valid[~is_loq]
    group_columns = [
        "Fonte", "Cenario_Dados", "Fold", "Questao_Teste", "Avaliador"
    ]
    indices = non_loq.groupby(group_columns, dropna=False)["RMSE"].idxmin()
    columns = group_columns + [
        "Politica_Duplicatas", "Embedding", "Perfil", "Cenário", "Modelo",
        "N_Treino", "N_Teste", "Leakage_Normalizado", "Linhas_Teste_Afetadas",
        "MAE", "RMSE", "R²",
    ]
    summary = non_loq.loc[indices, columns].copy()

    loq_summary = create_loq_summary(valid)
    if not loq_summary.empty:
        loq_best_indices = loq_summary.groupby(
            ["Cenario_Dados", "Avaliador"], dropna=False
        )["RMSE_mean"].idxmin()
        loq_best = loq_summary.loc[loq_best_indices].copy()
        loq_rows = pd.DataFrame(
            {
                "Fonte": "galhardi",
                "Cenario_Dados": loq_best["Cenario_Dados"],
                "Fold": "média dos folds",
                "Questao_Teste": np.nan,
                "Avaliador": loq_best["Avaliador"],
                "Politica_Duplicatas": np.where(
                    loq_best["Cenario_Dados"].str.contains("deduplicated"),
                    "deduplicar_pergunta_resposta_media_notas_questao_inteira",
                    "questao_inteira",
                ),
                "Embedding": loq_best["Embedding"],
                "Perfil": loq_best["Perfil"],
                "Cenário": loq_best["Cenário"],
                "Modelo": loq_best["Modelo"],
                "N_Treino": np.nan,
                "N_Teste": np.nan,
                "Leakage_Normalizado": 0,
                "Linhas_Teste_Afetadas": 0,
                "MAE": loq_best["MAE_mean"],
                "RMSE": loq_best["RMSE_mean"],
                "R²": loq_best["R²_mean"],
            }
        )
        summary = pd.concat([summary, loq_rows], ignore_index=True)

    summary["Embedding_Usado"] = [
        _embedding_used(representation, embedding)
        for representation, embedding in zip(summary["Cenário"], summary["Embedding"])
    ]
    summary["Status_Leakage"] = np.where(
        summary["Leakage_Normalizado"].fillna(0).gt(0),
        "ATENÇÃO — há leakage",
        "OK — sem leakage",
    )

    def observation(row: pd.Series) -> str:
        scenario = str(row["Cenario_Dados"])
        if row["Leakage_Normalizado"] > 0:
            return "Baseline histórico; não usar como estimativa livre de vazamento."
        if "deduplicated" in scenario:
            return "Um par pergunta+resposta; conflitos de nota agregados pela média."
        if scenario == "four_questions_grouped":
            return "Duplicatas preservadas no mesmo split; métricas ponderam repetições."
        if scenario.startswith("leave_one_question_out"):
            return "Generalização para questão não vista."
        return "Baseline histórico reproduzido."

    summary["Observação"] = summary.apply(observation, axis=1)
    summary["Experimento"] = (
        summary["Fonte"].astype(str)
        + " / "
        + summary["Cenario_Dados"].astype(str)
        + " / "
        + summary["Avaliador"].astype(str)
    )
    final_columns = [
        "Experimento", "Fonte", "Cenario_Dados", "Fold", "Questao_Teste",
        "Avaliador", "Politica_Duplicatas", "Embedding_Usado", "Perfil",
        "Cenário", "Modelo", "N_Treino", "N_Teste", "Leakage_Normalizado",
        "Linhas_Teste_Afetadas", "Status_Leakage", "MAE", "RMSE", "R²",
        "Observação",
    ]
    return summary[final_columns].sort_values(
        ["Fonte", "Cenario_Dados", "Avaliador"]
    ).reset_index(drop=True)


def create_evaluator_comparison(results: pd.DataFrame) -> pd.DataFrame:
    galhardi = results[
        results["Fonte"].eq("galhardi")
        & results["Avaliador"].isin(["primary", "other"])
    ].copy()
    if galhardi.empty or galhardi["Avaliador"].nunique() < 2:
        return pd.DataFrame()

    galhardi["Embedding_Usado"] = [
        _embedding_used(representation, embedding)
        for representation, embedding in zip(
            galhardi["Cenário"], galhardi["Embedding"]
        )
    ]
    group_columns = [
        "Cenario_Dados", "Fold", "Questao_Teste", "Embedding_Usado",
        "Perfil", "Cenário", "Modelo",
    ]
    # TF-IDF e Coh-Metrix isolados são repetidos pelo orquestrador para cada
    # embedding. Na comparação eles aparecem uma única vez com Embedding=N/A.
    compact = galhardi.drop_duplicates(
        group_columns + ["Avaliador"], keep="first"
    )
    pivot = compact.pivot_table(
        index=group_columns,
        columns="Avaliador",
        values=["N_Treino", "N_Teste", "MAE", "RMSE", "R²"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{evaluator}" for metric, evaluator in pivot.columns]
    comparison = pivot.reset_index()
    required = [
        "MAE_primary", "MAE_other", "RMSE_primary", "RMSE_other",
        "R²_primary", "R²_other",
    ]
    if not set(required).issubset(comparison.columns):
        return pd.DataFrame()
    comparison = comparison.dropna(subset=required)
    comparison["Delta_MAE_other_menos_primary"] = (
        comparison["MAE_other"] - comparison["MAE_primary"]
    )
    comparison["Delta_RMSE_other_menos_primary"] = (
        comparison["RMSE_other"] - comparison["RMSE_primary"]
    )
    comparison["Delta_R²_other_menos_primary"] = (
        comparison["R²_other"] - comparison["R²_primary"]
    )

    def comparability(scenario: object) -> str:
        scenario = str(scenario)
        if "deduplicated" in scenario:
            return "Alta — mesmas respostas; split independente das notas"
        if scenario.startswith("leave_one_question_out"):
            return "Moderada — mesma questão; repetições podem diferir"
        return "Limitada — split agrupado pode variar por avaliador"

    comparison["Comparabilidade"] = comparison["Cenario_Dados"].map(comparability)
    return comparison.sort_values(
        ["Cenario_Dados", "Fold", "RMSE_primary", "RMSE_other"]
    ).reset_index(drop=True)


def _get_article_metrics(results: pd.DataFrame, equivalent: object) -> tuple[float, float]:
    if pd.isna(equivalent) or str(equivalent).strip() == "-":
        return np.nan, np.nan
    parts = str(equivalent).split("+")
    if len(parts) != 2:
        return np.nan, np.nan
    representation, model = (part.strip() for part in parts)
    model_map = {
        "SVM": "SVR",
        "RF": "Random Forest",
        "DT": "Árvore de Decisão",
        "XGB": "XGBoost",
    }
    model = model_map.get(model, model)
    if representation == "TFIDF":
        mask = (
            results["Modelo"].eq(model)
            & results["Cenário"].str.contains("TF-IDF", na=False)
            & ~results["Cenário"].str.contains("Embeddings|Coh-Metrix", na=False)
        )
    elif representation == "Embeddings":
        mask = results["Modelo"].eq(model) & results["Cenário"].eq("Apenas Embeddings")
    else:
        return np.nan, np.nan
    subset = results[mask]
    if subset.empty:
        return np.nan, np.nan
    best = subset.sort_values("RMSE").iloc[0]
    return float(best["MAE"]), float(best["RMSE"])


def _load_article_comparisons(
    reference_path: Path,
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not reference_path.exists():
        print(f"AVISO: referência do artigo não encontrada: {reference_path}")
        return None
    excel = pd.ExcelFile(reference_path)
    names = set(excel.sheet_names)
    expected = {"5. Comparação Completa Split1", "6. Comparação Completa Split2"}
    if not expected.issubset(names):
        print("AVISO: planilha de referência não contém as abas de comparação esperadas.")
        return None

    comparisons = []
    for sheet, dataset in [
        ("5. Comparação Completa Split1", "df_12"),
        ("6. Comparação Completa Split2", "df_11"),
    ]:
        comparison = pd.read_excel(excel, sheet_name=sheet, header=2)
        if not comparison.empty:
            comparison.columns = comparison.iloc[0]
            comparison = comparison.drop(0).reset_index(drop=True)
        subset = results[
            results["Fonte"].eq("anderson")
            & results["Cenario_Dados"].eq(dataset)
            & results["Embedding"].eq("bertimbau")
            & results["Perfil"].eq("full")
        ]
        if "Equivalente (Meu)" in comparison.columns:
            for index, row in comparison.iterrows():
                mae, rmse = _get_article_metrics(subset, row.get("Equivalente (Meu)"))
                if np.isnan(mae):
                    continue
                comparison.at[index, "MAE (Meu)"] = round(mae, 4)
                rmse_column = "RMSE (Meu)" if "RMSE (Meu)" in comparison.columns else "RMSE (meu)"
                comparison.at[index, rmse_column] = round(rmse, 4)
                if "MAE (artigo)" in comparison.columns and pd.notna(row.get("MAE (artigo)")):
                    comparison.at[index, "Diferença (MAE)"] = round(
                        mae - float(row["MAE (artigo)"]), 4
                    )
        comparisons.append(comparison)
    return comparisons[0], comparisons[1]


def _write_sheet(writer, dataframe: pd.DataFrame, name: str, title: str) -> None:
    dataframe = dataframe.copy()
    dataframe.to_excel(writer, sheet_name=name, index=False, startrow=2)
    workbook = writer.book
    worksheet = writer.sheets[name]
    title_format = workbook.add_format(
        {"bold": True, "font_size": 14, "font_color": "#1F4E78"}
    )
    header_format = workbook.add_format(
        {
            "bg_color": "#2E4053",
            "font_color": "#FFFFFF",
            "bold": True,
            "align": "center",
            "valign": "vcenter",
            "bottom": 1,
            "bottom_color": "#A6A6A6",
        }
    )
    number_format = workbook.add_format({"num_format": "0.0000"})
    integer_format = workbook.add_format({"num_format": "0"})
    percentage_format = workbook.add_format({"num_format": "0.0%"})
    worksheet.write(0, 0, title, title_format)
    if dataframe.shape[1] == 0:
        worksheet.write(2, 0, "Sem dados para esta seleção.")
        worksheet.set_column(0, 0, 32)
        worksheet.hide_gridlines(2)
        return
    integer_like_columns = {
        "Questao", "Questao_Teste", "Random_State", "N_Total", "N_Total_Original",
        "N_Treino", "N_Teste", "N_Questoes_Total", "N_Questoes_Treino",
        "N_Questoes_Teste", "Pares_Unicos", "Linhas_Em_Grupos_Duplicados",
        "Excesso_Duplicatas", "Linhas_Removidas_Deduplicacao",
        "Grupos_Nota_Conflitante", "Linhas_Originais_Conflitantes",
        "Leakage_Exato", "Leakage_Normalizado", "Leakage_Chaves_Exatas",
        "Leakage_Chaves_Normalizadas", "Linhas_Teste_Afetadas",
        "N_Treino_primary", "N_Treino_other", "N_Teste_primary",
        "N_Teste_other",
    }
    for column_number, column_name in enumerate(dataframe.columns):
        worksheet.write(2, column_number, str(column_name), header_format)
        sample = dataframe[column_name].head(300)
        lengths = sample.map(
            lambda value: 0 if pd.isna(value) else len(str(value))
        )
        max_value_length = int(lengths.max()) if not lengths.empty else 0
        max_length = max(len(str(column_name)), max_value_length)
        width = min(max(max_length + 2, 12), 42)
        if column_name in {"Acordo_Ordinal_Exato", "Diferenca_Ordinal_Ate_1"}:
            worksheet.set_column(
                column_number, column_number, width, percentage_format
            )
        elif (
            pd.api.types.is_integer_dtype(dataframe[column_name])
            or column_name in integer_like_columns
        ):
            worksheet.set_column(column_number, column_number, width, integer_format)
        elif pd.api.types.is_numeric_dtype(dataframe[column_name]):
            worksheet.set_column(column_number, column_number, width, number_format)
        else:
            worksheet.set_column(column_number, column_number, width)
    worksheet.freeze_panes(3, 0)
    worksheet.autofilter(2, 0, 2 + len(dataframe), max(len(dataframe.columns) - 1, 0))
    first_data_row = 3
    last_data_row = 2 + len(dataframe)
    if last_data_row >= first_data_row:
        for metric in [
            "MAE", "RMSE", "R²", "MAE_mean", "RMSE_mean", "R²_mean",
            "Delta_MAE_other_menos_primary",
            "Delta_RMSE_other_menos_primary",
            "Delta_R²_other_menos_primary",
        ]:
            if metric not in dataframe.columns:
                continue
            column = dataframe.columns.get_loc(metric)
            higher_is_better = "R²" in metric
            worksheet.conditional_format(
                first_data_row,
                column,
                last_data_row,
                column,
                {"type": "3_color_scale", "min_color": "#F8696B", "mid_color": "#FFEB84", "max_color": "#63BE7B"}
                if higher_is_better
                else {"type": "3_color_scale", "min_color": "#63BE7B", "mid_color": "#FFEB84", "max_color": "#F8696B"},
            )
        for leakage_column in [
            "Leakage_Exato", "Leakage_Normalizado", "Leakage_Chaves_Exatas",
            "Leakage_Chaves_Normalizadas", "Linhas_Teste_Afetadas",
        ]:
            if leakage_column not in dataframe.columns:
                continue
            column = dataframe.columns.get_loc(leakage_column)
            worksheet.conditional_format(
                first_data_row,
                column,
                last_data_row,
                column,
                {
                    "type": "cell",
                    "criteria": ">",
                    "value": 0,
                    "format": workbook.add_format(
                        {"bg_color": "#F4CCCC", "font_color": "#9C0006", "bold": True}
                    ),
                },
            )
    worksheet.hide_gridlines(2)


def _add_executive_chart(writer, summary: pd.DataFrame) -> None:
    if summary.empty or "Experimento" not in summary or "RMSE" not in summary:
        return
    worksheet = writer.sheets["0. Resumo Executivo"]
    workbook = writer.book
    first_row = 3
    last_row = first_row + len(summary) - 1
    category_column = summary.columns.get_loc("Experimento")
    rmse_column = summary.columns.get_loc("RMSE")
    chart = workbook.add_chart({"type": "bar"})
    chart.add_series(
        {
            "name": "RMSE do melhor resultado",
            "categories": [
                "0. Resumo Executivo", first_row, category_column,
                last_row, category_column,
            ],
            "values": [
                "0. Resumo Executivo", first_row, rmse_column,
                last_row, rmse_column,
            ],
            "fill": {"color": "#4472C4"},
            "border": {"none": True},
            "data_labels": {"value": True, "num_format": "0.0000"},
        }
    )
    chart.set_title({"name": "Melhor RMSE por experimento"})
    chart.set_x_axis({"name": "RMSE — menor é melhor", "num_format": "0.00"})
    chart.set_y_axis({"name": ""})
    chart.set_legend({"none": True})
    chart.set_style(10)
    chart.set_size({"width": 980, "height": max(320, 42 * len(summary))})
    worksheet.insert_chart(5 + len(summary), 0, chart)


def generate_excel_report(
    results_csv: Path,
    audit_csv: Path,
    output_path: Path,
    article_reference: Path | None = None,
    evaluator_agreement_summary_csv: Path | None = None,
    evaluator_agreement_details_csv: Path | None = None,
) -> Path:
    results = pd.read_csv(results_csv)
    audit = pd.read_csv(audit_csv) if audit_csv.exists() else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_results = standardize_report_labels(results)
    executive = create_executive_summary(display_results)
    best = create_best_results(display_results)
    ablation = create_ablation(display_results)
    loq = display_results[
        display_results["Cenario_Dados"].astype(str).str.startswith(
            "leave_one_question_out"
        )
    ].copy()
    loq_summary = create_loq_summary(display_results)
    evaluator_comparison = create_evaluator_comparison(display_results)
    evaluator_agreement_summary = (
        pd.read_csv(evaluator_agreement_summary_csv)
        if evaluator_agreement_summary_csv is not None
        and evaluator_agreement_summary_csv.exists()
        else pd.DataFrame()
    )
    evaluator_agreement_details = (
        pd.read_csv(evaluator_agreement_details_csv)
        if evaluator_agreement_details_csv is not None
        and evaluator_agreement_details_csv.exists()
        else pd.DataFrame()
    )
    article = (
        _load_article_comparisons(article_reference, results)
        if article_reference is not None
        else None
    )

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        _write_sheet(
            writer,
            executive,
            "0. Resumo Executivo",
            "Resumo executivo — melhor resultado de cada experimento",
        )
        _write_sheet(writer, display_results, "1. Resultados", "Resultados consolidados dos experimentos")
        _write_sheet(writer, best, "2. Melhores", "Melhor representação por cenário, modelo e embedding")
        _write_sheet(writer, ablation, "3. Ablação", "Ablação do pré-processamento TF-IDF")
        _write_sheet(writer, loq, "4. LOQ Detalhado", "Leave-One-Question-Out — resultados por fold")
        _write_sheet(writer, loq_summary, "5. LOQ Resumo", "Leave-One-Question-Out — média e desvio-padrão")
        _write_sheet(writer, audit, "6. Auditoria", "Auditoria dos datasets e splits")
        _write_sheet(
            writer,
            evaluator_agreement_summary,
            "7. Acordo Avaliadores",
            "Concordância das notas — pergunta+resposta única",
        )
        _write_sheet(
            writer,
            evaluator_comparison,
            "8. Modelos Avaliadores",
            "Comparação de desempenho — avaliador principal versus demais avaliadores",
        )
        _write_sheet(
            writer,
            evaluator_agreement_details,
            "9. Notas Avaliadores",
            "Comparação detalhada das notas por pergunta+resposta única",
        )
        _add_executive_chart(writer, executive)
        if article is not None:
            _write_sheet(
                writer,
                article[0],
                "10. Artigo Split1",
                "Comparação com o artigo — Anderson df_12 / BERTimbau",
            )
            _write_sheet(
                writer,
                article[1],
                "11. Artigo Split2",
                "Comparação com o artigo — Anderson df_11 / BERTimbau",
            )
    return output_path
