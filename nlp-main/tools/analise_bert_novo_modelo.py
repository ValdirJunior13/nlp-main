"""
ASAG - Pipeline Principal (v2: pergunta + resposta como entrada)
=========================================================================

MUDANÇA em relação à versão anterior (sugestão do orientador, e-mail de
10/07/2026): antes o modelo só via a resposta do aluno (answer_text). Agora
concatenamos pergunta + resposta:

    question_answer = question_text + "\\n Resposta: " + answer_text

Isso afeta:
  - TF-IDF: agora vetoriza question_answer (pré-processado), não answer_text.
  - Embeddings: agora são gerados AO VIVO em cima de question_answer, porque o
    cache antigo (caracteristicas_embeddings.csv) foi calculado só sobre
    answer_text e não serve mais aqui. Isso é rápido (segundos), diferente do
    Coh-Metrix.

O que NÃO muda:
  - Coh-Metrix continua vindo do cache (cohmetrix_{dataset}_{split}.csv), que
    foi extraído em cima de answer_text isolado. Recalcular isso levaria ~4
    dias de novo, então mantemos como está. ISSO É UMA LIMITAÇÃO A DISCUTIR
    COM O ORIENTADOR: as features de Coh-Metrix avaliam só a resposta, mas
    TF-IDF/Embeddings agora avaliam pergunta+resposta. Se ele quiser Coh-Metrix
    também com pergunta+resposta, vai precisar rodar a extração de novo
    (mesmo custo de tempo de antes).

------------------------------------------------------------------------------
INSTALAÇÃO
------------------------------------------------------------------------------
pip install numpy pandas scikit-learn nltk spacy xgboost sentence-transformers
python -m spacy download pt_core_news_lg

------------------------------------------------------------------------------
COMO RODAR
------------------------------------------------------------------------------
python analise_asag_mvp.py
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
import spacy
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    TEM_XGBOOST = True
except ImportError:
    TEM_XGBOOST = False
    print("AVISO: xgboost não instalado. Rode: pip install xgboost")

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.abspath(os.path.join(DIRETORIO_ATUAL, ".."))
PASTA_DATASET = os.path.join(DIRETORIO_RAIZ, "dataset")

LISTA_DE_DATASETS = ["df_11", "df_12"]
MODELO_SPACY = "pt_core_news_lg"
MODELO_EMBEDDINGS = "neuralmind/bert-base-portuguese-cased"

POS_RELEVANTES = {"NOUN", "VERB", "ADJ", "ADV"}


# ------------------------------------------------------------------
# 1. Recursos
# ------------------------------------------------------------------
def carregar_recursos():
    print("Baixando lista de stopwords (nltk)...")
    nltk.download('stopwords', quiet=True)
    stop_words_pt = set(stopwords.words('portuguese'))

    print(f"Carregando modelo spaCy '{MODELO_SPACY}' para lematização/POS...")
    nlp_pt = spacy.load(MODELO_SPACY, disable=["ner", "parser"])

    print(f"Carregando modelo de embeddings '{MODELO_EMBEDDINGS}'...")
    modelo_ia = SentenceTransformer(MODELO_EMBEDDINGS)

    print("Recursos prontos!")
    return stop_words_pt, nlp_pt, modelo_ia


def limpar_para_merge(texto):
    return re.sub(r'\s+', ' ', str(texto).lower()).strip()


# ------------------------------------------------------------------
# 2. Construção do texto combinado (pergunta + resposta)
# ------------------------------------------------------------------
def construir_texto_combinado(df):
    """
    Cria a coluna 'question_answer' = pergunta + resposta, conforme sugestão
    do orientador. Exige que o DataFrame tenha 'question_text' e 'answer_text'.
    """
    if "question_text" not in df.columns:
        raise KeyError(
            "Coluna 'question_text' não encontrada. Confira se o CSV de origem "
            "tem essa coluna (df_11_train.csv/df_12_train.csv já têm por padrão)."
        )
    df = df.copy()
    df["question_answer"] = (
        df["question_text"].fillna("").astype(str)
        + "\n Resposta: "
        + df["answer_text"].fillna("").astype(str)
    )
    return df


# ------------------------------------------------------------------
# 3. Pré-processamento linguístico (só para TF-IDF)
# ------------------------------------------------------------------
def gerar_variantes_preprocessamento(textos, nlp_pt, stop_words_pt, batch_size=64, n_process=1):
    textos = [str(t) if pd.notna(t) else "" for t in textos]

    saida = {
        "cru": [],
        "stopwords": [],
        "stopwords_lemma": [],
        "stopwords_lemma_pos": [],
    }

    for doc in nlp_pt.pipe(textos, batch_size=batch_size, n_process=n_process):
        tokens_cru, tokens_stop, tokens_lemma, tokens_lemma_pos = [], [], [], []

        for tok in doc:
            if not tok.is_alpha:
                continue

            tokens_cru.append(tok.text.lower())

            eh_stopword = tok.is_stop or tok.text.lower() in stop_words_pt
            if eh_stopword:
                continue

            tokens_stop.append(tok.text.lower())
            tokens_lemma.append(tok.lemma_.lower())

            if tok.pos_ in POS_RELEVANTES:
                tokens_lemma_pos.append(tok.lemma_.lower())

        saida["cru"].append(" ".join(tokens_cru))
        saida["stopwords"].append(" ".join(tokens_stop))
        saida["stopwords_lemma"].append(" ".join(tokens_lemma))
        saida["stopwords_lemma_pos"].append(" ".join(tokens_lemma_pos))

    return saida


# ------------------------------------------------------------------
# 4. Carregamento + merge por dataset/split
# ------------------------------------------------------------------
def carregar_e_juntar(nome_dataset, split, pasta=PASTA_DATASET):
    caminho = os.path.join(pasta, f"{nome_dataset}_{split}.csv")
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Não encontrei {caminho}")

    df = pd.read_csv(caminho)
    df = construir_texto_combinado(df)
    df['chave_merge'] = df['answer_text'].apply(limpar_para_merge)

    # --- Coh-Metrix (baseado em answer_text isolado - ver aviso no topo do arquivo) ---
    caminho_coh = os.path.join(pasta, f"cohmetrix_{nome_dataset}_{split}.csv")
    coh_features = None
    if os.path.exists(caminho_coh):
        coh_raw = pd.read_csv(caminho_coh)
        coh_raw['chave_merge'] = coh_raw['resposta_original'].apply(limpar_para_merge)
        coh_dedup = coh_raw.drop_duplicates(subset='chave_merge')

        df_coh = df.merge(coh_dedup, on='chave_merge', how='left')
        colunas_remover = [
            'resposta_original', 'nota_original', 'answer_text', 'grade',
            'chave_merge', 'question_text', 'question_answer', 'id', 'question_id',
        ]
        colunas_features = [c for c in coh_dedup.columns if c not in colunas_remover]
        coh_features = df_coh[colunas_features].fillna(0)
        coh_features.index = df.index

    return df, coh_features


# ------------------------------------------------------------------
# 5. Pipeline principal por dataset
# ------------------------------------------------------------------
def executar_experimento_asag(nome_dataset, nlp_pt, stop_words_pt, modelo_ia):
    print("\n" + "=" * 70)
    print(f"INICIANDO PIPELINE PARA: {nome_dataset} (texto = pergunta + resposta)")
    print("=" * 70)

    df_train, coh_train = carregar_e_juntar(nome_dataset, "train")
    df_test, coh_test = carregar_e_juntar(nome_dataset, "test")

    usar_cohmetrix = coh_train is not None and coh_test is not None
    if not usar_cohmetrix:
        print(" -> AVISO: Coh-Metrix não encontrado para este dataset. Rodando sem essas features.")

    textos_train = df_train["question_answer"].tolist()
    textos_test = df_test["question_answer"].tolist()
    y_train = df_train["grade"]
    y_test = df_test["grade"]

    # ---------- Embeddings AO VIVO (pergunta + resposta) ----------
    print(" -> Gerando embeddings (pergunta + resposta) — ao vivo, não usa mais o cache antigo...")
    emb_train = modelo_ia.encode(textos_train, show_progress_bar=False)
    emb_test = modelo_ia.encode(textos_test, show_progress_bar=False)
    cols_emb = [f"emb_{i}" for i in range(emb_train.shape[1])]
    df_emb_train = pd.DataFrame(emb_train, columns=cols_emb)
    df_emb_test = pd.DataFrame(emb_test, columns=cols_emb)

    # ---------- Variantes de pré-processamento para TF-IDF (pergunta + resposta) ----------
    print(" -> Gerando variantes de pré-processamento (stopwords / lemma / POS)...")
    variantes_train = gerar_variantes_preprocessamento(textos_train, nlp_pt, stop_words_pt)
    variantes_test = gerar_variantes_preprocessamento(textos_test, nlp_pt, stop_words_pt)

    nomes_variantes = {
        "cru": "Apenas TF-IDF (cru)",
        "stopwords": "Apenas TF-IDF (stopwords)",
        "stopwords_lemma": "Apenas TF-IDF (stopwords+lemma)",
        "stopwords_lemma_pos": "Apenas TF-IDF (stopwords+lemma+POS)",
    }

    print(" -> Vetorizando TF-IDF para cada variante...")
    tfidf_por_variante = {}
    for chave, rotulo in nomes_variantes.items():
        vec = TfidfVectorizer(max_features=1000)
        X_tr = pd.DataFrame(vec.fit_transform(variantes_train[chave]).toarray(),
                             columns=vec.get_feature_names_out())
        X_te = pd.DataFrame(vec.transform(variantes_test[chave]).toarray(),
                             columns=vec.get_feature_names_out())
        tfidf_por_variante[chave] = (X_tr, X_te)

    df_tfidf_train, df_tfidf_test = tfidf_por_variante["stopwords_lemma_pos"]

    df_emb_train = df_emb_train.reset_index(drop=True)
    df_emb_test = df_emb_test.reset_index(drop=True)
    if usar_cohmetrix:
        coh_train = coh_train.reset_index(drop=True)
        coh_test = coh_test.reset_index(drop=True)

    # ---------- Cenários ----------
    cenarios = {}
    for chave, rotulo in nomes_variantes.items():
        cenarios[rotulo] = tfidf_por_variante[chave]

    cenarios["Apenas Embeddings"] = (df_emb_train, df_emb_test)
    cenarios["TF-IDF (processado) + Embeddings"] = (
        pd.concat([df_tfidf_train, df_emb_train], axis=1),
        pd.concat([df_tfidf_test, df_emb_test], axis=1),
    )

    if usar_cohmetrix:
        cenarios["Apenas Coh-Metrix"] = (coh_train, coh_test)
        cenarios["TF-IDF (processado) + Coh-Metrix"] = (
            pd.concat([df_tfidf_train, coh_train], axis=1),
            pd.concat([df_tfidf_test, coh_test], axis=1),
        )
        cenarios["Coh-Metrix + Embeddings"] = (
            pd.concat([coh_train, df_emb_train], axis=1),
            pd.concat([coh_test, df_emb_test], axis=1),
        )
        cenarios["Tudo (TF-IDF processado + Coh-Metrix + Emb)"] = (
            pd.concat([df_tfidf_train, coh_train, df_emb_train], axis=1),
            pd.concat([df_tfidf_test, coh_test, df_emb_test], axis=1),
        )

    # ---------- Modelos ----------
    print(" -> Treinando e avaliando modelos...")
    modelos = {
        "Regressão Linear": LinearRegression(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(),
        "Árvore de Decisão": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "Rede Neural (MLP)": MLPRegressor(random_state=42, max_iter=1000),
    }
    if TEM_XGBOOST:
        modelos["XGBoost"] = XGBRegressor(random_state=42)

    # Modelos baseados em árvore não precisam de StandardScaler (pedido do orientador,
    # e-mail de 10/07/2026): eles particionam o espaço por limiares em cada variável
    # isoladamente, então normalizar não muda o resultado, só adiciona uma etapa à toa.
    # XGBoost entra no mesmo grupo por também ser baseado em árvores (gradient boosting).
    MODELOS_SEM_ESCALA = {"Árvore de Decisão", "Random Forest", "HistGradientBoosting", "XGBoost"}

    resultados = []
    melhor_r2 = -float('inf')
    dados_melhor_modelo = {}

    for nome_cenario, (X_train_c, X_test_c) in cenarios.items():
        if X_train_c.shape[1] == 0:
            print(f"    * Cenário '{nome_cenario}' ficou sem colunas. Pulando.")
            continue

        colunas = X_train_c.columns

        scaler = StandardScaler()
        X_tr_escalado = pd.DataFrame(scaler.fit_transform(X_train_c), columns=colunas)
        X_te_escalado = pd.DataFrame(scaler.transform(X_test_c), columns=colunas)

        # Versão sem escala, para os modelos baseados em árvore
        X_tr_cru = X_train_c.reset_index(drop=True)
        X_te_cru = X_test_c.reset_index(drop=True)

        for nome_modelo, modelo_base in modelos.items():
            if nome_modelo in MODELOS_SEM_ESCALA:
                X_tr_sc, X_te_sc = X_tr_cru, X_te_cru
            else:
                X_tr_sc, X_te_sc = X_tr_escalado, X_te_escalado

            modelo = clone(modelo_base)
            modelo.fit(X_tr_sc, y_train)
            y_pred = modelo.predict(X_te_sc)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            resultados.append({
                "Dataset": nome_dataset, "Cenário": nome_cenario, "Modelo": nome_modelo,
                "MAE": mae, "RMSE": rmse, "R²": r2
            })

            if r2 > melhor_r2:
                melhor_r2 = r2
                dados_melhor_modelo = {
                    'modelo': modelo, 'nome_modelo': nome_modelo, 'cenario': nome_cenario,
                    'y_pred': y_pred, 'y_test': y_test, 'df_ref': df_test
                }

    df_resultados = pd.DataFrame(resultados).sort_values(by="R²", ascending=False).reset_index(drop=True)
    df_resultados.to_csv(os.path.join(DIRETORIO_ATUAL, f"resultados_{nome_dataset}_bertimbau.csv"), index=False)

    print(f"\nConcluído! Resultados oficiais para {nome_dataset}:")
    print(df_resultados.head(10).round(4).to_string(index=False))
    if dados_melhor_modelo:
        print(f"Melhor Modelo: {dados_melhor_modelo['nome_modelo']} "
              f"({dados_melhor_modelo['cenario']}) | R²: {melhor_r2:.4f}\n")

    return df_resultados, dados_melhor_modelo


# ------------------------------------------------------------------
# 6. Tabela de ablação
# ------------------------------------------------------------------
def tabela_ablacao_tfidf(df_resultados):
    variantes_ordem = [
        "Apenas TF-IDF (cru)",
        "Apenas TF-IDF (stopwords)",
        "Apenas TF-IDF (stopwords+lemma)",
        "Apenas TF-IDF (stopwords+lemma+POS)",
    ]
    df_abl = df_resultados[df_resultados["Cenário"].isin(variantes_ordem)].copy()
    df_abl["Cenário"] = pd.Categorical(df_abl["Cenário"], categories=variantes_ordem, ordered=True)
    tabela = df_abl.pivot_table(index="Modelo", columns="Cenário", values=["MAE", "RMSE", "R²"])
    return tabela.round(4)


# ------------------------------------------------------------------
# 7. Execução
# ------------------------------------------------------------------
def main():
    stop_words_pt, nlp_pt, modelo_ia = carregar_recursos()

    todas_as_tabelas = []
    melhores_modelos = {}

    print("Iniciando a bateria de experimentos...")

    for dataset in LISTA_DE_DATASETS:
        try:
            tabela_resultado, melhor_modelo = executar_experimento_asag(
                dataset, nlp_pt, stop_words_pt, modelo_ia)

            if tabela_resultado is not None and not tabela_resultado.empty:
                todas_as_tabelas.append(tabela_resultado)
                melhores_modelos[dataset] = melhor_modelo

                print(f"\n--- Ablação de pré-processamento TF-IDF: {dataset} ---")
                print(tabela_ablacao_tfidf(tabela_resultado).to_string())

        except Exception as e:
            print(f"Erro crítico ao processar {dataset}: {e}")
            import traceback
            traceback.print_exc()

    if todas_as_tabelas:
        relatorio_final = pd.concat(todas_as_tabelas, ignore_index=True)

        print("\n" + "=" * 70)
        print("RANKING GERAL DE TODOS OS EXPERIMENTOS (Top 20)")
        print("=" * 70)
        print(relatorio_final.sort_values(by="R²", ascending=False).head(20).round(4).to_string(index=False))

        caminho_relatorio = os.path.join(DIRETORIO_ATUAL, "relatorio_geral_experimentos_asag_bertimbau.csv")
        relatorio_final.to_csv(caminho_relatorio, index=False)
        print(f"Relatório geral salvo em '{caminho_relatorio}'")
    else:
        print("\nNenhum resultado foi gerado. Verifique os logs acima para entender onde falhou.")


if __name__ == "__main__":
    main()