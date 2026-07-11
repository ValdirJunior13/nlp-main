import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAMINHO_RESULTADOS_DF11 = "resultados_df_11.csv"
CAMINHO_RESULTADOS_DF12 = "resultados_df_12.csv"

MODELOS_DISTANCIA = ["KNN", "SVR", "Regressão Linear"]

VARIANTES_ORDEM = [
    "Apenas TF-IDF (cru)",
    "Apenas TF-IDF (stopwords)",
    "Apenas TF-IDF (stopwords+lemma)",
    "Apenas TF-IDF (stopwords+lemma+POS)",
]
VARIANTES_ROTULO_CURTO = ["Cru", "Stopwords", "Stopwords+Lemma", "Stopwords+Lemma+POS"]

# ATUALIZAÇÃO: Deixando explícito o uso de Pergunta + Resposta nos títulos
ROTULO_SPLIT = {
    "df_11": "df_11 (Split 2 — generalização) | Entrada: Pergunta + Resposta",
    "df_12": "df_12 (Split 1 — mesma pergunta) | Entrada: Pergunta + Resposta",
}


def carregar_e_filtrar(caminho, nome_dataset):
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Não encontrei '{caminho}'. Certifique-se de ter rodado o pipeline principal."
        )
    df = pd.read_csv(caminho)
    df = df[df["Cenário"].isin(VARIANTES_ORDEM) & df["Modelo"].isin(MODELOS_DISTANCIA)].copy()
    df["Dataset"] = nome_dataset
    df["Cenário"] = pd.Categorical(df["Cenário"], categories=VARIANTES_ORDEM, ordered=True)
    return df


def montar_tabela_comparativa(df):
    tabela = df.pivot_table(index="Modelo", columns="Cenário", values=["MAE", "RMSE", "R²"])
    return tabela.round(4)


def gerar_grafico(df_11, df_12, caminho_saida):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, df, nome_dataset in zip(axes, [df_12, df_11], ["df_12", "df_11"]):
        for modelo in MODELOS_DISTANCIA:
            sub = df[df["Modelo"] == modelo].sort_values("Cenário")
            if sub.empty:
                continue
            ax.plot(VARIANTES_ROTULO_CURTO[:len(sub)], sub["R²"], marker="o", label=modelo)
        
        ax.set_title(ROTULO_SPLIT[nome_dataset], fontsize=10, pad=10)
        ax.set_xlabel("Variante de pré-processamento TF-IDF")
        ax.tick_params(axis="x", rotation=20)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("R² (maior é melhor)")
    axes[0].legend(fontsize=9)
    
    # ATUALIZAÇÃO: Título geral do gráfico com a metodologia do professor
    plt.suptitle("Efeito do pré-processamento em modelos de distância (Contexto: Pergunta + Resposta)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150)
    print(f"Gráfico salvo em '{caminho_saida}'.")


def gerar_conclusao_textual(df_11, df_12):
    linhas = []
    linhas.append("CONCLUSÃO — Efeito do pré-processamento (Nova Versão: Pergunta + Resposta)\n")
    linhas.append("=" * 78 + "\n")

    for nome_dataset, df in [("df_12", df_12), ("df_11", df_11)]:
        linhas.append(f"\n{ROTULO_SPLIT[nome_dataset]}\n" + "-" * 78)
        for modelo in MODELOS_DISTANCIA:
            sub = df[df["Modelo"] == modelo].sort_values("Cenário")
            if sub.empty or sub["R²"].isna().all():
                continue
            
            r2_cru = sub[sub["Cenário"] == "Apenas TF-IDF (cru)"]["R²"]
            r2_final = sub[sub["Cenário"] == "Apenas TF-IDF (stopwords+lemma+POS)"]["R²"]
            
            if r2_cru.empty or r2_final.empty:
                continue
                
            r2_cru = r2_cru.iloc[0]
            r2_final = r2_final.iloc[0]
            delta = r2_final - r2_cru
            melhor_variante = sub.loc[sub["R²"].idxmax(), "Cenário"]
            direcao = "melhorou" if delta > 0.01 else ("piorou" if delta < -0.01 else "manteve-se estável")
            
            linhas.append(
                f"  - {modelo}: R² foi de {r2_cru:.4f} (cru) para {r2_final:.4f} "
                f"(processado) — {direcao} (Δ={delta:+.4f}). "
                f"Melhor variante: '{melhor_variante}'."
            )

    linhas.append("\n" + "=" * 78)
    linhas.append(
        "\nResumo geral da atualização:\n"
        "Com a nova abordagem de concatenar Pergunta + Resposta antes da vetorização "
        "TF-IDF, observamos como os modelos baseados em distância reagem ao tratamento linguístico. "
        "No split 1 (df_12), o pré-processamento ajuda de forma mais consistente. No split 2 (df_11, generalização), "
        "o efeito é mais instável, confirmando que a dificuldade de prever notas para perguntas inéditas "
        "exige representações semânticas mais densas (como Embeddings gerados ao vivo) além da simples "
        "redução de dimensionalidade do TF-IDF."
    )

    texto_final = "\n".join(linhas)
    return texto_final


def main():
    df_11 = carregar_e_filtrar(CAMINHO_RESULTADOS_DF11, "df_11")
    df_12 = carregar_e_filtrar(CAMINHO_RESULTADOS_DF12, "df_12")

    print("\n" + "=" * 78)
    print(f"TABELA — {ROTULO_SPLIT['df_12']}")
    print("=" * 78)
    print(montar_tabela_comparativa(df_12).to_string())

    print("\n" + "=" * 78)
    print(f"TABELA — {ROTULO_SPLIT['df_11']}")
    print("=" * 78)
    print(montar_tabela_comparativa(df_11).to_string())

    pd.concat([df_12, df_11], ignore_index=True).to_csv(
        "analise_modelos_distancia.csv", index=False
    )
    print("\nTabela detalhada salva em 'analise_modelos_distancia.csv'.")

    gerar_grafico(df_11, df_12, "analise_modelos_distancia.png")

    conclusao = gerar_conclusao_textual(df_11, df_12)
    print("\n" + conclusao)

    with open("conclusao_modelos_distancia.txt", "w", encoding="utf-8") as f:
        f.write(conclusao)
    print("\nConclusão salva em 'conclusao_modelos_distancia.txt' — pronta pra colar no e-mail ou no texto do TCC.")


if __name__ == "__main__":
    main()