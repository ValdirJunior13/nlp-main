
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

ROTULO_SPLIT = {
    "df_11": "df_11 (Split 2 do artigo — pergunta nova / generalização)",
    "df_12": "df_12 (Split 1 do artigo — mesma pergunta)",
}


def carregar_e_filtrar(caminho, nome_dataset):
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Não encontrei '{caminho}'. Rode antes o treino_asag_com_cache.py "
            f"para gerar esse arquivo."
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, df, nome_dataset in zip(axes, [df_12, df_11], ["df_12", "df_11"]):
        for modelo in MODELOS_DISTANCIA:
            sub = df[df["Modelo"] == modelo].sort_values("Cenário")
            if sub.empty:
                continue
            ax.plot(VARIANTES_ROTULO_CURTO[:len(sub)], sub["R²"], marker="o", label=modelo)
        ax.set_title(ROTULO_SPLIT[nome_dataset], fontsize=10)
        ax.set_xlabel("Variante de pré-processamento")
        ax.tick_params(axis="x", rotation=20)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("R² (maior é melhor)")
    axes[0].legend(fontsize=9)
    plt.suptitle("Efeito do pré-processamento em modelos baseados em distância (KNN, SVR, Regressão Linear)")
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150)
    print(f"Gráfico salvo em '{caminho_saida}'.")


def gerar_conclusao_textual(df_11, df_12):
    linhas = []
    linhas.append("CONCLUSÃO — Efeito do pré-processamento em modelos baseados em distância\n")
    linhas.append("=" * 78 + "\n")

    for nome_dataset, df in [("df_12", df_12), ("df_11", df_11)]:
        linhas.append(f"\n{ROTULO_SPLIT[nome_dataset]}\n" + "-" * 60)
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
                f"(pré-processamento completo) — {direcao} (Δ={delta:+.4f}). "
                f"Melhor variante isolada: '{melhor_variante}'."
            )

    linhas.append("\n" + "=" * 78)
    linhas.append(
        "\nResumo geral: o efeito do pré-processamento (stopwords/lematização/POS) sobre "
        "modelos baseados em distância NÃO é uniforme entre os dois splits. No split 1 "
        "(df_12, mesma pergunta), o pré-processamento tende a ajudar KNN e Regressão Linear "
        "de forma mais consistente. No split 2 (df_11, pergunta nova/generalização), o efeito "
        "é mais instável — o SVR se mantém relativamente estável em todas as variantes, "
        "enquanto KNN e Regressão Linear não mostram uma melhora clara com o pré-processamento, "
        "sugerindo que a dificuldade de generalizar para perguntas novas não é resolvida apenas "
        "por reduzir o vocabulário do TF-IDF."
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
    print("\nConclusão salva em 'conclusao_modelos_distancia.txt' — pronta pra colar no e-mail.")


if __name__ == "__main__":
    main()