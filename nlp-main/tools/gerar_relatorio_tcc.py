from __future__ import annotations

import argparse
from pathlib import Path

from asag_pipeline.report import generate_excel_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenera somente o relatório Excel do TCC.")
    parser.add_argument(
        "--resultados",
        type=Path,
        default=Path("outputs/results/resultados_consolidados.csv"),
    )
    parser.add_argument(
        "--auditoria",
        type=Path,
        default=Path("outputs/audits/auditoria_datasets.csv"),
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("outputs/Relatorio_TCC_Consolidado.xlsx"),
    )
    parser.add_argument(
        "--acordo-avaliadores",
        type=Path,
        default=Path("outputs/audits/comparacao_notas_avaliadores_resumo.csv"),
    )
    parser.add_argument(
        "--notas-avaliadores",
        type=Path,
        default=Path("outputs/audits/comparacao_notas_avaliadores_detalhada.csv"),
    )
    parser.add_argument("--referencia-artigo", type=Path)
    args = parser.parse_args()
    output = generate_excel_report(
        results_csv=args.resultados,
        audit_csv=args.auditoria,
        output_path=args.saida,
        article_reference=args.referencia_artigo,
        evaluator_agreement_summary_csv=args.acordo_avaliadores,
        evaluator_agreement_details_csv=args.notas_avaliadores,
    )
    print(f"Relatório gerado: {output}")


if __name__ == "__main__":
    main()
