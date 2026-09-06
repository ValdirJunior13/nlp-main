from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd

from asag_pipeline.data import DatasetSpec, build_datasets, compare_galhardi_graders
from asag_pipeline.experiment import (
    ExperimentOptions,
    expected_configuration_count,
    run_experiment,
)
from asag_pipeline.features import FeatureResources, load_cohmetrix
from asag_pipeline.report import (
    create_ablation,
    create_best_results,
    create_evaluator_comparison,
    create_executive_summary,
    create_loq_summary,
    generate_excel_report,
    standardize_report_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa os experimentos Anderson e Galhardi e consolida os resultados."
    )
    parser.add_argument("--dataset-dir", type=Path, help="Pasta dataset do projeto.")
    parser.add_argument("--output-dir", type=Path, help="Pasta de saída.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Pasta compartilhada dos caches. Por padrão usa CACHE dentro de "
            "--output-dir. Pode apontar para um cache gerado em outra execução."
        ),
    )
    parser.add_argument(
        "--fontes",
        nargs="+",
        choices=["anderson", "galhardi"],
        default=["anderson", "galhardi"],
    )
    parser.add_argument(
        "--cenarios",
        nargs="+",
        choices=[
            "df_11",
            "df_12",
            "four_questions_row",
            "four_questions_grouped",
            "four_questions_deduplicated",
            "leave_one_question_out",
            "leave_one_question_out_deduplicated",
        ],
        help="Por padrão, executa todos os cenários compatíveis com as fontes selecionadas.",
    )
    parser.add_argument(
        "--embeddings",
        nargs="+",
        choices=["minilm", "bertimbau"],
        default=["minilm", "bertimbau"],
    )
    parser.add_argument(
        "--perfis",
        nargs="+",
        choices=["full", "paper_like"],
        default=["full"],
    )
    parser.add_argument(
        "--avaliadores",
        nargs="+",
        choices=["primary", "other"],
        default=["primary"],
    )
    parser.add_argument(
        "--questoes",
        nargs="+",
        type=int,
        default=[1, 9, 11, 12],
        help="Números das questões Galhardi. Padrão: 1 9 11 12.",
    )
    parser.add_argument(
        "--cohmetrix",
        choices=["off", "existing", "generate"],
        default="existing",
        help="existing reutiliza caches; generate cria o cache Galhardi quando necessário.",
    )
    parser.add_argument("--previsoes", action="store_true", help="Salva predições por instância.")
    parser.add_argument(
        "--retomar",
        action="store_true",
        help="Reutiliza checkpoints por representação/modelo já concluídos.",
    )
    parser.add_argument("--somente-auditoria", action="store_true")
    parser.add_argument(
        "--somente-cohmetrix",
        action="store_true",
        help="Gera/valida o cache Coh-Metrix sem carregar embeddings ou treinar modelos.",
    )
    parser.add_argument("--sem-excel", action="store_true")
    parser.add_argument("--referencia-artigo", type=Path)
    return parser.parse_args()


def resolve_dataset_dir(project_root: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        return supplied.resolve()
    candidates = [project_root / "dataset", project_root.parent / "dataset"]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "A pasta dataset não foi encontrada. Use --dataset-dir CAMINHO_DA_PASTA."
    )


def build_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    requested = set(args.cenarios or [])
    specs: list[DatasetSpec] = []
    if "anderson" in args.fontes:
        for scenario in ("df_11", "df_12"):
            if not requested or scenario in requested:
                specs.append(
                    DatasetSpec(
                        source="anderson",
                        scenario=scenario,
                        grade_policy="primary",
                        duplicate_policy="baseline_historico",
                    )
                )
    if "galhardi" in args.fontes:
        for evaluator in args.avaliadores:
            candidates = [
                ("four_questions_row", "linha_baseline"),
                ("four_questions_grouped", "grupo_pergunta_resposta"),
                (
                    "four_questions_deduplicated",
                    "deduplicar_pergunta_resposta_media_notas_split_estavel",
                ),
                ("leave_one_question_out", "questao_inteira"),
                (
                    "leave_one_question_out_deduplicated",
                    "deduplicar_pergunta_resposta_media_notas_questao_inteira",
                ),
            ]
            for scenario, duplicate_policy in candidates:
                # Mantém o comportamento padrão anterior. Os cenários
                # deduplicados são adicionais e executados quando solicitados
                # explicitamente em --cenarios.
                default_safe_scenario = scenario in {
                    "four_questions_grouped",
                    "leave_one_question_out",
                }
                if (not requested and default_safe_scenario) or scenario in requested:
                    specs.append(
                        DatasetSpec(
                            source="galhardi",
                            scenario=scenario,
                            question_numbers=tuple(args.questoes),
                            grade_policy=evaluator,
                            duplicate_policy=duplicate_policy,
                        )
                    )
    return specs


def safe_name(value: object) -> str:
    return str(value).replace(" ", "_").replace("/", "-")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    dataset_root = resolve_dataset_dir(project_root, args.dataset_dir)
    output_root = (args.output_dir or project_root / "outputs").resolve()
    result_dir = output_root / "results"
    audit_dir = output_root / "audits"
    prediction_dir = output_root / "predictions"
    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else output_root / "cache"
    )
    for directory in (result_dir, audit_dir, prediction_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_root}")
    print(f"Saídas: {output_root}")
    print(f"Cache: {cache_dir}")

    specs = build_specs(args)
    bundles = []
    audits: list[dict] = []
    errors: list[dict] = []
    for spec in specs:
        try:
            generated = build_datasets(dataset_root, spec)
            bundles.extend(generated)
            audits.extend(bundle.audit for bundle in generated if bundle.audit)
            for bundle in generated:
                leakage = bundle.audit["Leakage_Chaves_Normalizadas"]
                print(
                    f"AUDITORIA {spec.source}/{spec.scenario}/{bundle.fold}: "
                    f"treino={len(bundle.train)}, teste={len(bundle.test)}, leakage={leakage}"
                )
        except Exception as exc:
            errors.append(
                {
                    "Etapa": "dataset",
                    "Fonte": spec.source,
                    "Cenario_Dados": spec.scenario,
                    "Erro": f"{type(exc).__name__}: {exc}",
                }
            )
            traceback.print_exc()

    audit_csv = audit_dir / "auditoria_datasets.csv"
    pd.DataFrame(audits).to_csv(audit_csv, index=False)
    print(f"Auditoria salva: {audit_csv}")
    grader_agreement_summary_csv: Path | None = None
    grader_agreement_details_csv: Path | None = None
    if (
        "galhardi" in args.fontes
        and {"primary", "other"}.issubset(set(args.avaliadores))
    ):
        try:
            agreement_summary, agreement_details = compare_galhardi_graders(
                dataset_root,
                tuple(args.questoes),
            )
            grader_agreement_summary_csv = (
                audit_dir / "comparacao_notas_avaliadores_resumo.csv"
            )
            grader_agreement_details_csv = (
                audit_dir / "comparacao_notas_avaliadores_detalhada.csv"
            )
            agreement_summary.to_csv(grader_agreement_summary_csv, index=False)
            agreement_details.to_csv(grader_agreement_details_csv, index=False)
            print(
                "Comparação de notas dos avaliadores salva: "
                f"{grader_agreement_summary_csv}"
            )
        except Exception as exc:
            errors.append(
                {
                    "Etapa": "comparacao_avaliadores",
                    "Fonte": "galhardi",
                    "Cenario_Dados": "pergunta_resposta_unica",
                    "Erro": f"{type(exc).__name__}: {exc}",
                }
            )
            traceback.print_exc()
    if args.somente_auditoria:
        if errors:
            pd.DataFrame(errors).to_csv(output_root / "erros_execucao.csv", index=False)
        return

    if args.somente_cohmetrix:
        if args.cohmetrix == "off":
            raise ValueError(
                "--somente-cohmetrix requer --cohmetrix generate ou existing."
            )
        galhardi_bundles = [
            bundle for bundle in bundles if bundle.spec.source == "galhardi"
        ]
        if not galhardi_bundles:
            raise ValueError(
                "--somente-cohmetrix requer ao menos um cenário da fonte galhardi."
            )
        for bundle in galhardi_bundles:
            label = f"{bundle.spec.scenario}/{bundle.fold}"
            print(f"\nGERANDO/VALIDANDO COH-METRIX: {label}")
            try:
                load_cohmetrix(
                    bundle=bundle,
                    dataset_root=dataset_root,
                    project_root=project_root,
                    cache_dir=cache_dir,
                    mode=args.cohmetrix,
                )
            except Exception as exc:
                errors.append(
                    {
                        "Etapa": "cohmetrix",
                        "Fonte": bundle.spec.source,
                        "Cenario_Dados": bundle.spec.scenario,
                        "Fold": bundle.fold,
                        "Erro": f"{type(exc).__name__}: {exc}",
                    }
                )
                pd.DataFrame(errors).to_csv(
                    output_root / "erros_execucao.csv", index=False
                )
                raise
        if errors:
            pd.DataFrame(errors).to_csv(output_root / "erros_execucao.csv", index=False)
        print(f"\nCache Coh-Metrix concluído: {cache_dir / 'cohmetrix_galhardi.csv'}")
        return

    all_results: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    resources_by_embedding: dict[str, FeatureResources] = {}

    for embedding in args.embeddings:
        resources = resources_by_embedding.setdefault(embedding, FeatureResources(embedding))
        for profile in args.perfis:
            options = ExperimentOptions(
                profile=profile,
                embedding=embedding,
                cohmetrix_mode=args.cohmetrix,
                save_predictions=args.previsoes,
            )
            for bundle in bundles:
                label = (
                    f"{bundle.spec.source}/{bundle.spec.scenario}/{bundle.fold}/"
                    f"{embedding}/{profile}"
                )
                print(f"\n{'=' * 80}\nEXECUTANDO {label}\n{'=' * 80}")
                filename = "resultados__" + "__".join(
                    map(
                        safe_name,
                        [
                            bundle.spec.source,
                            bundle.spec.scenario,
                            bundle.fold,
                            bundle.spec.grade_policy,
                            embedding,
                            profile,
                        ],
                    )
                ) + ".csv"
                result_path = result_dir / filename
                prediction_name = filename.replace("resultados__", "predicoes__")
                prediction_path = prediction_dir / prediction_name
                try:
                    if args.retomar and result_path.exists() and not args.previsoes:
                        existing_results = pd.read_csv(result_path)
                        completed = existing_results.drop_duplicates(
                            ["Cenário", "Modelo"]
                        )
                        expected = expected_configuration_count(profile)
                        if len(completed) >= expected:
                            print(
                                f"  -> Bloco completo reutilizado: "
                                f"{len(completed)}/{expected} ajustes.",
                                flush=True,
                            )
                            all_results.append(existing_results)
                            continue

                    results, predictions = run_experiment(
                        bundle=bundle,
                        resources=resources,
                        options=options,
                        dataset_root=dataset_root,
                        project_root=project_root,
                        cache_dir=cache_dir,
                        checkpoint_path=result_path,
                        prediction_checkpoint_path=(
                            prediction_path if args.previsoes else None
                        ),
                        resume=args.retomar,
                    )
                    results.to_csv(result_path, index=False)
                    all_results.append(results)
                    if not predictions.empty:
                        predictions.to_csv(prediction_path, index=False)
                        all_predictions.append(predictions)
                except Exception as exc:
                    errors.append(
                        {
                            "Etapa": "experimento",
                            "Fonte": bundle.spec.source,
                            "Cenario_Dados": bundle.spec.scenario,
                            "Fold": bundle.fold,
                            "Embedding": embedding,
                            "Perfil": profile,
                            "Erro": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    traceback.print_exc()

    if not all_results:
        if errors:
            pd.DataFrame(errors).to_csv(output_root / "erros_execucao.csv", index=False)
        raise RuntimeError("Nenhum resultado foi gerado. Consulte erros_execucao.csv.")

    consolidated = pd.concat(all_results, ignore_index=True)
    consolidated_csv = result_dir / "resultados_consolidados.csv"
    consolidated.to_csv(consolidated_csv, index=False)
    display_results = standardize_report_labels(consolidated)
    executive_summary = create_executive_summary(display_results)
    executive_summary.to_csv(result_dir / "resumo_executivo.csv", index=False)
    best_results = create_best_results(display_results)
    best_results.to_csv(result_dir / "comparacao_melhores_cenarios.csv", index=False)
    evaluator_comparison = create_evaluator_comparison(display_results)
    evaluator_comparison.to_csv(
        result_dir / "comparacao_avaliadores.csv", index=False
    )
    ablation = create_ablation(display_results)
    ablation.to_csv(result_dir / "ablacao_tfidf.csv", index=False)
    distance = ablation[
        ablation.get("Modelo", pd.Series(dtype=str)).isin(
            ["KNN", "SVR", "Regressão Linear"]
        )
    ] if not ablation.empty else ablation
    distance.to_csv(result_dir / "analise_modelos_distancia.csv", index=False)
    loq_summary = create_loq_summary(display_results)
    loq_summary.to_csv(result_dir / "resumo_leave_one_question_out.csv", index=False)

    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            prediction_dir / "predicoes_consolidadas.csv", index=False
        )
    if errors:
        pd.DataFrame(errors).to_csv(output_root / "erros_execucao.csv", index=False)

    print(f"\nResultados consolidados: {consolidated_csv}")
    if not args.sem_excel:
        report_path = output_root / "Relatorio_TCC_Consolidado.xlsx"
        generate_excel_report(
            results_csv=consolidated_csv,
            audit_csv=audit_csv,
            output_path=report_path,
            article_reference=args.referencia_artigo,
            evaluator_agreement_summary_csv=grader_agreement_summary_csv,
            evaluator_agreement_details_csv=grader_agreement_details_csv,
        )
        print(f"Relatório Excel: {report_path}")


if __name__ == "__main__":
    main()
