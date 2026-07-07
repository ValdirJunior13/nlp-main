import os
import re
import warnings

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
import spacy

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
CAMINHO_EMBEDDINGS = os.path.join(DIRETORIO_ATUAL, "caracteristicas_embeddings.csv")

LISTA_DE_DATASETS = ["df_11", "df_12"]
MODELO_SPACY = "pt_core_news_lg"

POS_RELEVANTES = {"NOUN", "VERB", "ADJ", "ADV"}

def carregar_recursos():
    print("Baixando lista de stopwords (nltk)...")
    nltk.download('stopwords', quiet=True)
    stop_words_pt = set(stopwords.words('portuguese'))

    print(f"Carregando modelo spaCy '{MODELO_SPACY}' para lematização/POS...")
    nlp_pt = spacy.load(MODELO_SPACY, disable=["ner", "parser"])
    print("Modelo spaCy pronto para uso!")

    if not os.path.exists(CAMINHO_EMBEDDINGS):
        raise FileNotFoundError(
            f"Não encontrei '{CAMINHO_EMBEDDINGS}'. Verifique se ele está na pasta raiz do projeto."
        )

    df_emb_completo = pd.read_csv(CAMINHO_EMBEDDINGS)
    colunas_emb = [c for c in df_emb_completo.columns
                   if c not in ("resposta_original", "nota_original")]
    print(f"Embeddings carregados: {len(df_emb_completo)} respostas, "
          f"{len(colunas_emb)} dimensões.")

    df_emb_completo['chave_merge'] = df_emb_completo['resposta_original'].apply(limpar_para_merge)
    df_emb_dedup = df_emb_completo.drop_duplicates(subset='chave_merge')

    return stop_words_pt, nlp_pt, df_emb_dedup, colunas_emb


def limpar_para_merge(texto):
    return re.sub(r'\s+', ' ', str(texto).lower()).strip()

def gerar_variantes_preprocessamento(textos, nlp_pt, stop_words_pt, batch_size=64, n_process=1):
    """
    Gera 4 variantes de texto a partir da lista `textos`:
      - cru:                  sem nenhum tratamento (além de manter só letras)
      - stopwords:             remove stopwords
      - stopwords_lemma:       remove stopwords + lematiza
      - stopwords_lemma_pos:   remove stopwords + lematiza + filtra classe gramatical
                                (mantém só NOUN, VERB, ADJ, ADV)
    """
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

def carregar_e_juntar(nome_dataset, split, df_emb_dedup, colunas_emb, pasta=PASTA_DATASET):
    caminho = os.path.join(pasta, f"{nome_dataset}_{split}.csv")
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Não encontrei {caminho}")

    df = pd.read_csv(caminho)
    df['chave_merge'] = df['answer_text'].apply(limpar_para_merge)

    caminho_coh = os.path.join(pasta, f"cohmetrix_{nome_dataset}_{split}.csv")
    coh_features = None
    if os.path.exists(caminho_coh):
        coh_raw = pd.read_csv(caminho_coh)
        coh_raw['chave_merge'] = coh_raw['resposta_original'].apply(limpar_para_merge)
        coh_dedup = coh_raw.drop_duplicates(subset='chave_merge')

        df_coh = df.merge(coh_dedup, on='chave_merge', how='left')
        colunas_remover = ['resposta_original', 'nota_original', 'answer_text', 'grade', 'chave_merge']
        colunas_features = [c for c in coh_dedup.columns if c not in colunas_remover]
        coh_features = df_coh[colunas_features].fillna(0)
        coh_features.index = df.index

    df_emb_merged = df.merge(df_emb_dedup, on='chave_merge', how='left')
    n_sem_embedding = df_emb_merged[colunas_emb[0]].isna().sum()
    if n_sem_embedding > 0:
        print(f"    AVISO [{nome_dataset}_{split}]: {n_sem_embedding} de {len(df)} "
              f"respostas não encontraram embedding correspondente (preenchidas com 0).")
    emb_features = df_emb_merged[colunas_emb].fillna(0)
    emb_features.index = df.index

    return df, coh_features, emb_features


def executar_experimento_asag(nome_dataset, nlp_pt, stop_words_pt, df_emb_dedup, colunas_emb):
    print("\n" + "=" * 70)
    print(f"INICIANDO PIPELINE PARA: {nome_dataset}")
    print("=" * 70)

    df_train, coh_train, df_emb_train = carregar_e_juntar(
        nome_dataset, "train", df_emb_dedup, colunas_emb)
    df_test, coh_test, df_emb_test = carregar_e_juntar(
        nome_dataset, "test", df_emb_dedup, colunas_emb)

    usar_cohmetrix = coh_train is not None and coh_test is not None
    if not usar_cohmetrix:
        print(" -> AVISO: Coh-Metrix não encontrado para este dataset. Rodando sem essas features.")

    textos_train = df_train["answer_text"].fillna("").astype(str).tolist()
    textos_test = df_test["answer_text"].fillna("").astype(str).tolist()
    y_train = df_train["grade"]
    y_test = df_test["grade"]

    # ---------- Variantes de pré-processamento para TF-IDF ----------
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

    resultados = []
    melhor_r2 = -float('inf')
    dados_melhor_modelo = {}

    for nome_cenario, (X_train_c, X_test_c) in cenarios.items():
        if X_train_c.shape[1] == 0:
            print(f"    * Cenário '{nome_cenario}' ficou sem colunas. Pulando.")
            continue

        scaler = StandardScaler()
        colunas = X_train_c.columns
        X_tr_sc = pd.DataFrame(scaler.fit_transform(X_train_c), columns=colunas)
        X_te_sc = pd.DataFrame(scaler.transform(X_test_c), columns=colunas)

        for nome_modelo, modelo_base in modelos.items():
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
    df_resultados.to_csv(os.path.join(DIRETORIO_ATUAL, f"resultados_{nome_dataset}.csv"), index=False)

    print(f"\nConcluído! Resultados oficiais para {nome_dataset}:")
    print(df_resultados.head(10).round(4).to_string(index=False))
    if dados_melhor_modelo:
        print(f"Melhor Modelo: {dados_melhor_modelo['nome_modelo']} "
              f"({dados_melhor_modelo['cenario']}) | R²: {melhor_r2:.4f}\n")

    return df_resultados, dados_melhor_modelo

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

def main():
    stop_words_pt, nlp_pt, df_emb_dedup, colunas_emb = carregar_recursos()

    todas_as_tabelas = []
    melhores_modelos = {}

    print("Iniciando a bateria de experimentos...")

    for dataset in LISTA_DE_DATASETS:
        try:
            tabela_resultado, melhor_modelo = executar_experimento_asag(
                dataset, nlp_pt, stop_words_pt, df_emb_dedup, colunas_emb)

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
        
        caminho_relatorio = os.path.join(DIRETORIO_ATUAL, "relatorio_geral_experimentos_asag.csv")
        relatorio_final.to_csv(caminho_relatorio, index=False)
        print(f"Relatório geral salvo em '{caminho_relatorio}'")
    else:
        print("\nNenhum resultado foi gerado. Verifique os logs acima para entender onde falhou.")


if __name__ == "__main__":
    main()