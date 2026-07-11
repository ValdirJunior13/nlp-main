"""
ASAG - App Streamlit (Dashboard + Pipelines de Treino)
=========================================================================

Reúne, numa interface única, tudo que você já tinha em notebooks separados —
EXCETO a extração de Coh-Metrix (que leva ~4 dias e continua sendo feita
separadamente pelo extracao_asag_tcc.ipynb).

Abas:
  1. Dashboard          -> explora os CSVs de resultados que você já gerou
  2. Rodar Treino        -> roda a ablação de pré-processamento + combinações
                            (equivalente ao analise_asag_mvp.ipynb), ao vivo
  3. Estilo Artigo        -> roda as representações isoladas
                            (equivalente ao analise_semelhante_mvp.ipynb), ao vivo
  4. Modelos de Distância -> KNN / SVR / Regressão Linear, com conclusão automática

------------------------------------------------------------------------------
INSTALAÇÃO
------------------------------------------------------------------------------
pip install streamlit pandas numpy scikit-learn nltk spacy xgboost matplotlib --break-system-packages
python -m spacy download pt_core_news_lg

------------------------------------------------------------------------------
COMO RODAR
------------------------------------------------------------------------------
streamlit run app.py

Rode a partir da pasta "tools" (mesma onde estão os outros scripts/notebooks),
para que os caminhos padrão (../dataset, caracteristicas_embeddings.csv) funcionem.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="ASAG - Painel do TCC", layout="wide")

# ------------------------------------------------------------------
# CONSTANTES
# ------------------------------------------------------------------
POS_RELEVANTES = {"NOUN", "VERB", "ADJ", "ADV"}
MODELOS_DISTANCIA = ["KNN", "SVR", "Regressão Linear"]
LISTA_DE_DATASETS = ["df_11", "df_12"]
ROTULO_DATASET = {
    "df_11": "df_11 — Split 2 do artigo (pergunta nova / generalização)",
    "df_12": "df_12 — Split 1 do artigo (mesma pergunta)",
}
VARIANTES_ORDEM = [
    "Apenas TF-IDF (cru)",
    "Apenas TF-IDF (stopwords)",
    "Apenas TF-IDF (stopwords+lemma)",
    "Apenas TF-IDF (stopwords+lemma+POS)",
]
VARIANTES_CURTO = ["Cru", "Stopwords", "Stopwords+Lemma", "Stopwords+Lemma+POS"]

# Valores do artigo (Mello et al., LAK25), coluna PT_ASAG, Tables 5 e 6
ARTIGO_SPLIT1 = {
    ("TFIDF", "SVM"): (0.43, 0.81), ("TFIDF", "RF"): (0.37, 0.73),
    ("TFIDF", "DT"): (0.47, 0.84), ("TFIDF", "XGB"): (0.41, 0.77),
    ("BERT", "SVM"): (0.49, 0.89), ("BERT", "RF"): (0.67, 1.09),
    ("BERT", "DT"): (0.67, 1.09), ("BERT", "XGB"): (0.47, 0.86),
}
ARTIGO_SPLIT2 = {
    ("TFIDF", "SVM"): (0.69, 1.12), ("TFIDF", "RF"): (0.67, 1.09),
    ("TFIDF", "DT"): (0.63, 1.02), ("TFIDF", "XGB"): (0.66, 1.08),
    ("BERT", "SVM"): (0.74, 1.16), ("BERT", "RF"): (0.79, 1.21),
    ("BERT", "DT"): (0.79, 1.21), ("BERT", "XGB"): (0.72, 1.14),
}


def limpar_para_merge(texto):
    return re.sub(r"\s+", " ", str(texto).lower()).strip()


# ------------------------------------------------------------------
# RECURSOS PESADOS (cacheados: só carregam uma vez por sessão)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando spaCy (pt_core_news_lg)...")
def carregar_spacy():
    import spacy
    return spacy.load("pt_core_news_lg", disable=["ner", "parser"])


@st.cache_resource(show_spinner="Carregando stopwords...")
def carregar_stopwords():
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("portuguese"))


@st.cache_data(show_spinner="Carregando embeddings pré-calculados...")
def carregar_embeddings(caminho):
    df = pd.read_csv(caminho)
    colunas_emb = [c for c in df.columns if c not in ("resposta_original", "nota_original")]
    df["chave_merge"] = df["resposta_original"].apply(limpar_para_merge)
    df_dedup = df.drop_duplicates(subset="chave_merge")
    return df_dedup, colunas_emb


def gerar_variantes_preprocessamento(textos, nlp_pt, stop_words_pt, batch_size=64):
    textos = [str(t) if pd.notna(t) else "" for t in textos]
    saida = {"cru": [], "stopwords": [], "stopwords_lemma": [], "stopwords_lemma_pos": []}

    for doc in nlp_pt.pipe(textos, batch_size=batch_size):
        tokens_cru, tokens_stop, tokens_lemma, tokens_lemma_pos = [], [], [], []
        for tok in doc:
            if not tok.is_alpha:
                continue
            tokens_cru.append(tok.text.lower())
            if tok.is_stop or tok.text.lower() in stop_words_pt:
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


def preprocessar_para_tfidf(textos, nlp_pt, stop_words_pt, batch_size=64):
    textos = [str(t) if pd.notna(t) else "" for t in textos]
    saida = []
    for doc in nlp_pt.pipe(textos, batch_size=batch_size):
        tokens = []
        for tok in doc:
            if not tok.is_alpha or tok.is_stop or tok.text.lower() in stop_words_pt:
                continue
            if tok.pos_ not in POS_RELEVANTES:
                continue
            tokens.append(tok.lemma_.lower())
        saida.append(" ".join(tokens))
    return saida


def carregar_e_juntar(nome_dataset, split, pasta_dataset, df_emb_dedup, colunas_emb):
    caminho = os.path.join(pasta_dataset, f"{nome_dataset}_{split}.csv")
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Não encontrei {caminho}")

    df = pd.read_csv(caminho)
    df["chave_merge"] = df["answer_text"].apply(limpar_para_merge)

    caminho_coh = os.path.join(pasta_dataset, f"cohmetrix_{nome_dataset}_{split}.csv")
    coh_features = None
    if os.path.exists(caminho_coh):
        coh_raw = pd.read_csv(caminho_coh)
        coh_raw["chave_merge"] = coh_raw["resposta_original"].apply(limpar_para_merge)
        coh_dedup = coh_raw.drop_duplicates(subset="chave_merge")
        df_coh = df.merge(coh_dedup, on="chave_merge", how="left")
        colunas_remover = ["resposta_original", "nota_original", "answer_text", "grade", "chave_merge"]
        colunas_features = [c for c in coh_dedup.columns if c not in colunas_remover]
        coh_features = df_coh[colunas_features].fillna(0)
        coh_features.index = df.index

    df_emb_merged = df.merge(df_emb_dedup, on="chave_merge", how="left")
    n_sem_emb = df_emb_merged[colunas_emb[0]].isna().sum()
    if n_sem_emb > 0:
        st.caption(f"⚠️ {nome_dataset}_{split}: {n_sem_emb} de {len(df)} respostas sem embedding correspondente (preenchidas com 0).")
    emb_features = df_emb_merged[colunas_emb].fillna(0)
    emb_features.index = df.index

    return df, coh_features, emb_features


# ------------------------------------------------------------------
# ABA 1 — DASHBOARD
# ------------------------------------------------------------------
def aba_dashboard():
    st.header("📊 Dashboard de Resultados")
    st.caption("Explore os CSVs já gerados pelos outros notebooks/scripts.")

    col1, col2 = st.columns(2)
    with col1:
        arquivo_geral = st.file_uploader("relatorio_geral_experimentos_asag.csv", type="csv", key="up_geral")
    with col2:
        arquivo_pesquisa = st.file_uploader("relatorio_geral_experimentos_pesquisa_semelhante.csv", type="csv", key="up_pesq")

    if arquivo_geral is None and os.path.exists("relatorio_geral_experimentos_asag.csv"):
        arquivo_geral = "relatorio_geral_experimentos_asag.csv"
    if arquivo_pesquisa is None and os.path.exists("relatorio_geral_experimentos_pesquisa_semelhante.csv"):
        arquivo_pesquisa = "relatorio_geral_experimentos_pesquisa_semelhante.csv"

    if arquivo_geral is not None:
        geral = pd.read_csv(arquivo_geral)
        geral["DatasetRotulo"] = geral["Dataset"].map(ROTULO_DATASET).fillna(geral["Dataset"])

        st.subheader("Resultados completos")
        c1, c2, c3 = st.columns(3)
        with c1:
            datasets_sel = st.multiselect("Dataset", geral["DatasetRotulo"].unique(),
                                           default=list(geral["DatasetRotulo"].unique()))
        with c2:
            cenarios_sel = st.multiselect("Cenário", geral["Cenário"].unique(),
                                           default=list(geral["Cenário"].unique()))
        with c3:
            modelos_sel = st.multiselect("Modelo", geral["Modelo"].unique(),
                                          default=list(geral["Modelo"].unique()))

        filtrado = geral[
            geral["DatasetRotulo"].isin(datasets_sel) &
            geral["Cenário"].isin(cenarios_sel) &
            geral["Modelo"].isin(modelos_sel)
        ].sort_values("R²", ascending=False)

        st.dataframe(filtrado[["DatasetRotulo", "Cenário", "Modelo", "MAE", "RMSE", "R²"]]
                     .round(4), use_container_width=True, height=350)

        st.subheader("Top 10 por R²")
        top10 = filtrado.head(10)
        fig, ax = plt.subplots(figsize=(9, 4))
        rotulos = top10["Cenário"] + " | " + top10["Modelo"]
        ax.barh(rotulos, top10["R²"], color="#2E86C1")
        ax.invert_yaxis()
        ax.set_xlabel("R²")
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Ablação de pré-processamento (TF-IDF)")
        dataset_ablacao = st.selectbox("Split para ablação", list(ROTULO_DATASET.values()), key="dataset_abl")
        nome_ds = [k for k, v in ROTULO_DATASET.items() if v == dataset_ablacao][0]
        sub_abl = geral[geral["Dataset"] == nome_ds]
        sub_abl = sub_abl[sub_abl["Cenário"].isin(VARIANTES_ORDEM)]
        if not sub_abl.empty:
            pivot = sub_abl.pivot_table(index="Modelo", columns="Cenário", values="R²")
            pivot = pivot.reindex(columns=VARIANTES_ORDEM)
            st.dataframe(pivot.round(4), use_container_width=True)
    else:
        st.info("Envie ou coloque na pasta atual o arquivo 'relatorio_geral_experimentos_asag.csv' para ver o dashboard.")

    if arquivo_pesquisa is not None:
        st.subheader("Representações isoladas (estilo artigo) + comparação")
        pesquisa = pd.read_csv(arquivo_pesquisa)
        st.dataframe(pesquisa.sort_values(["Dataset", "MAE"]).round(4), use_container_width=True, height=300)

        st.markdown("**Comparação com o artigo (Tables 5/6, coluna PT_ASAG)**")
        split_escolhido = st.radio("Split", ["Split 1 (mesma pergunta)", "Split 2 (pergunta nova)"], horizontal=True)
        dataset_alvo = "df_12" if split_escolhido.startswith("Split 1") else "df_11"
        tabela_artigo = ARTIGO_SPLIT1 if split_escolhido.startswith("Split 1") else ARTIGO_SPLIT2

        linhas = []
        for (feat, modelo), (mae_art, rmse_art) in tabela_artigo.items():
            feat_seu = "Embeddings" if feat == "BERT" else feat
            linha = pesquisa[(pesquisa["Dataset"] == dataset_alvo) &
                              (pesquisa["Features"] == feat_seu) &
                              (pesquisa["Model"] == modelo)]
            if linha.empty:
                continue
            mae_seu = float(linha["MAE"].iloc[0])
            linhas.append({
                "Representação (artigo)": feat, "Modelo": modelo,
                "MAE artigo": mae_art, "MAE seu": mae_seu,
                "Diferença": round(mae_seu - mae_art, 4),
                "Quem ganhou": "Você" if mae_seu < mae_art else "Artigo",
            })
        if linhas:
            df_comp = pd.DataFrame(linhas)

            def cor_linha(row):
                cor = "#d5f5e3" if row["Quem ganhou"] == "Você" else "#fadbd8"
                return [f"background-color: {cor}"] * len(row)

            st.dataframe(df_comp.style.apply(cor_linha, axis=1), use_container_width=True)
    else:
        st.info("Envie ou coloque na pasta atual o arquivo 'relatorio_geral_experimentos_pesquisa_semelhante.csv' para ver a comparação com o artigo.")


# ------------------------------------------------------------------
# ABA 2 — RODAR TREINO (ablação + combinações)
# ------------------------------------------------------------------
def executar_experimento_asag(nome_dataset, pasta_dataset, nlp_pt, stop_words_pt, df_emb_dedup, colunas_emb):
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
        tem_xgb = True
    except ImportError:
        tem_xgb = False

    df_train, coh_train, df_emb_train = carregar_e_juntar(nome_dataset, "train", pasta_dataset, df_emb_dedup, colunas_emb)
    df_test, coh_test, df_emb_test = carregar_e_juntar(nome_dataset, "test", pasta_dataset, df_emb_dedup, colunas_emb)
    usar_coh = coh_train is not None and coh_test is not None

    textos_train = df_train["answer_text"].fillna("").astype(str).tolist()
    textos_test = df_test["answer_text"].fillna("").astype(str).tolist()
    y_train, y_test = df_train["grade"], df_test["grade"]

    variantes_train = gerar_variantes_preprocessamento(textos_train, nlp_pt, stop_words_pt)
    variantes_test = gerar_variantes_preprocessamento(textos_test, nlp_pt, stop_words_pt)

    nomes_variantes = {
        "cru": "Apenas TF-IDF (cru)", "stopwords": "Apenas TF-IDF (stopwords)",
        "stopwords_lemma": "Apenas TF-IDF (stopwords+lemma)",
        "stopwords_lemma_pos": "Apenas TF-IDF (stopwords+lemma+POS)",
    }
    tfidf_por_variante = {}
    for chave, rotulo in nomes_variantes.items():
        vec = TfidfVectorizer(max_features=1000)
        X_tr = pd.DataFrame(vec.fit_transform(variantes_train[chave]).toarray(), columns=vec.get_feature_names_out())
        X_te = pd.DataFrame(vec.transform(variantes_test[chave]).toarray(), columns=vec.get_feature_names_out())
        tfidf_por_variante[chave] = (X_tr, X_te)

    df_tfidf_train, df_tfidf_test = tfidf_por_variante["stopwords_lemma_pos"]
    df_emb_train, df_emb_test = df_emb_train.reset_index(drop=True), df_emb_test.reset_index(drop=True)
    if usar_coh:
        coh_train, coh_test = coh_train.reset_index(drop=True), coh_test.reset_index(drop=True)

    cenarios = {rotulo: tfidf_por_variante[chave] for chave, rotulo in nomes_variantes.items()}
    cenarios["Apenas Embeddings"] = (df_emb_train, df_emb_test)
    cenarios["TF-IDF (processado) + Embeddings"] = (
        pd.concat([df_tfidf_train, df_emb_train], axis=1), pd.concat([df_tfidf_test, df_emb_test], axis=1))
    if usar_coh:
        cenarios["Apenas Coh-Metrix"] = (coh_train, coh_test)
        cenarios["TF-IDF (processado) + Coh-Metrix"] = (
            pd.concat([df_tfidf_train, coh_train], axis=1), pd.concat([df_tfidf_test, coh_test], axis=1))
        cenarios["Coh-Metrix + Embeddings"] = (
            pd.concat([coh_train, df_emb_train], axis=1), pd.concat([coh_test, df_emb_test], axis=1))
        cenarios["Tudo (TF-IDF processado + Coh-Metrix + Emb)"] = (
            pd.concat([df_tfidf_train, coh_train, df_emb_train], axis=1),
            pd.concat([df_tfidf_test, coh_test, df_emb_test], axis=1))

    modelos = {
        "Regressão Linear": LinearRegression(), "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(), "Árvore de Decisão": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "Rede Neural (MLP)": MLPRegressor(random_state=42, max_iter=1000),
    }
    if tem_xgb:
        modelos["XGBoost"] = XGBRegressor(random_state=42)

    resultados = []
    barra = st.progress(0.0, text=f"Treinando modelos para {nome_dataset}...")
    total = len(cenarios) * len(modelos)
    i = 0
    for nome_cenario, (X_tr_c, X_te_c) in cenarios.items():
        if X_tr_c.shape[1] == 0:
            continue
        scaler = StandardScaler()
        colunas = X_tr_c.columns
        X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr_c), columns=colunas)
        X_te_sc = pd.DataFrame(scaler.transform(X_te_c), columns=colunas)
        for nome_modelo, modelo_base in modelos.items():
            modelo = clone(modelo_base)
            modelo.fit(X_tr_sc, y_train)
            y_pred = modelo.predict(X_te_sc)
            resultados.append({
                "Dataset": nome_dataset, "Cenário": nome_cenario, "Modelo": nome_modelo,
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R²": r2_score(y_test, y_pred),
            })
            i += 1
            barra.progress(i / total, text=f"{nome_dataset}: {nome_cenario} / {nome_modelo}")
    barra.empty()

    return pd.DataFrame(resultados).sort_values("R²", ascending=False).reset_index(drop=True)


def aba_rodar_treino():
    st.header("🚀 Rodar Treino (ablação de pré-processamento + combinações)")
    st.caption("Reaproveita Coh-Metrix e Embeddings já extraídos. Não recalcula a extração de 4 dias.")

    pasta_dataset = st.text_input("Pasta com os datasets (../dataset)", value=os.path.join("..", "dataset"))
    caminho_embeddings = st.text_input("Caminho do caracteristicas_embeddings.csv", value="caracteristicas_embeddings.csv")

    if st.button("▶️ Rodar treino completo (df_11 e df_12)", type="primary"):
        if not os.path.exists(caminho_embeddings):
            st.error(f"Não encontrei '{caminho_embeddings}'.")
            return
        nlp_pt = carregar_spacy()
        stop_words_pt = carregar_stopwords()
        df_emb_dedup, colunas_emb = carregar_embeddings(caminho_embeddings)

        todas_tabelas = {}
        for dataset in LISTA_DE_DATASETS:
            with st.spinner(f"Processando {dataset}..."):
                df_resultado = executar_experimento_asag(
                    dataset, pasta_dataset, nlp_pt, stop_words_pt, df_emb_dedup, colunas_emb)
                todas_tabelas[dataset] = df_resultado
                st.session_state[f"resultados_{dataset}"] = df_resultado

            st.subheader(f"Resultados — {ROTULO_DATASET[dataset]}")
            st.dataframe(df_resultado.head(10).round(4), use_container_width=True)
            st.download_button(f"⬇️ Baixar resultados_{dataset}.csv",
                                df_resultado.to_csv(index=False), file_name=f"resultados_{dataset}.csv")

        relatorio_final = pd.concat(todas_tabelas.values(), ignore_index=True)
        st.session_state["relatorio_geral"] = relatorio_final
        st.subheader("Ranking geral (Top 20)")
        st.dataframe(relatorio_final.sort_values("R²", ascending=False).head(20).round(4), use_container_width=True)
        st.download_button("⬇️ Baixar relatorio_geral_experimentos_asag.csv",
                            relatorio_final.to_csv(index=False), file_name="relatorio_geral_experimentos_asag.csv")


# ------------------------------------------------------------------
# ABA 3 — ESTILO ARTIGO (representações isoladas)
# ------------------------------------------------------------------
def executar_pesquisa_semelhante(nome_dataset, pasta_dataset, nlp_pt, stop_words_pt, df_emb_dedup, colunas_emb):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.base import clone
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    try:
        from xgboost import XGBRegressor
        tem_xgb = True
    except ImportError:
        tem_xgb = False

    df_train, coh_train, df_emb_train = carregar_e_juntar(nome_dataset, "train", pasta_dataset, df_emb_dedup, colunas_emb)
    df_test, coh_test, df_emb_test = carregar_e_juntar(nome_dataset, "test", pasta_dataset, df_emb_dedup, colunas_emb)
    usar_coh = coh_train is not None and coh_test is not None

    textos_train = df_train["answer_text"].fillna("").astype(str).tolist()
    textos_test = df_test["answer_text"].fillna("").astype(str).tolist()
    y_train, y_test = df_train["grade"], df_test["grade"]

    texto_proc_train = preprocessar_para_tfidf(textos_train, nlp_pt, stop_words_pt)
    texto_proc_test = preprocessar_para_tfidf(textos_test, nlp_pt, stop_words_pt)
    vec = TfidfVectorizer(max_features=1000)
    df_tfidf_train = pd.DataFrame(vec.fit_transform(texto_proc_train).toarray(), columns=vec.get_feature_names_out())
    df_tfidf_test = pd.DataFrame(vec.transform(texto_proc_test).toarray(), columns=vec.get_feature_names_out())

    df_emb_train, df_emb_test = df_emb_train.reset_index(drop=True), df_emb_test.reset_index(drop=True)
    if usar_coh:
        coh_train, coh_test = coh_train.reset_index(drop=True), coh_test.reset_index(drop=True)

    representacoes = {"TFIDF": (df_tfidf_train, df_tfidf_test), "Embeddings": (df_emb_train, df_emb_test)}
    if usar_coh:
        representacoes["CohMetrix"] = (coh_train, coh_test)

    modelos = {"SVM": SVR(), "RF": RandomForestRegressor(n_estimators=100, random_state=42),
               "DT": DecisionTreeRegressor(random_state=42)}
    if tem_xgb:
        modelos["XGB"] = XGBRegressor(random_state=42)

    resultados = []
    barra = st.progress(0.0, text=f"Treinando (estilo artigo) para {nome_dataset}...")
    total = len(representacoes) * len(modelos)
    i = 0
    for nome_repr, (X_tr_r, X_te_r) in representacoes.items():
        if X_tr_r.shape[1] == 0:
            continue
        scaler = StandardScaler()
        colunas = X_tr_r.columns
        X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr_r), columns=colunas)
        X_te_sc = pd.DataFrame(scaler.transform(X_te_r), columns=colunas)
        for nome_modelo, modelo_base in modelos.items():
            modelo = clone(modelo_base)
            modelo.fit(X_tr_sc, y_train)
            y_pred = modelo.predict(X_te_sc)
            resultados.append({
                "Dataset": nome_dataset, "Features": nome_repr, "Model": nome_modelo,
                "MAE": round(mean_absolute_error(y_test, y_pred), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                "R2": round(r2_score(y_test, y_pred), 4),
            })
            i += 1
            barra.progress(i / total, text=f"{nome_dataset}: {nome_repr} / {nome_modelo}")
    barra.empty()
    return pd.DataFrame(resultados).sort_values("MAE").reset_index(drop=True)


def aba_estilo_artigo():
    st.header("🎯 Estilo Artigo (representações isoladas)")
    st.caption("TF-IDF, Embeddings e Coh-Metrix testados isoladamente, com SVM/RF/DT/XGB — igual à metodologia do artigo.")

    pasta_dataset = st.text_input("Pasta com os datasets", value=os.path.join("..", "dataset"), key="pasta2")
    caminho_embeddings = st.text_input("Caminho dos embeddings", value="caracteristicas_embeddings.csv", key="emb2")

    if st.button("▶️ Rodar (estilo artigo)", type="primary"):
        if not os.path.exists(caminho_embeddings):
            st.error(f"Não encontrei '{caminho_embeddings}'.")
            return
        nlp_pt = carregar_spacy()
        stop_words_pt = carregar_stopwords()
        df_emb_dedup, colunas_emb = carregar_embeddings(caminho_embeddings)

        tabelas = []
        for dataset in LISTA_DE_DATASETS:
            with st.spinner(f"Processando {dataset}..."):
                df_resultado = executar_pesquisa_semelhante(
                    dataset, pasta_dataset, nlp_pt, stop_words_pt, df_emb_dedup, colunas_emb)
                tabelas.append(df_resultado)
            st.subheader(ROTULO_DATASET[dataset])
            st.dataframe(df_resultado.round(4), use_container_width=True)

        relatorio = pd.concat(tabelas, ignore_index=True)
        st.session_state["relatorio_pesquisa_semelhante"] = relatorio
        st.download_button("⬇️ Baixar relatorio_geral_experimentos_pesquisa_semelhante.csv",
                            relatorio.to_csv(index=False),
                            file_name="relatorio_geral_experimentos_pesquisa_semelhante.csv")


# ------------------------------------------------------------------
# ABA 4 — MODELOS DE DISTÂNCIA
# ------------------------------------------------------------------
def aba_modelos_distancia():
    st.header("📏 Modelos Baseados em Distância (KNN, SVR, Regressão Linear)")
    st.caption("Usa os resultados da aba 'Rodar Treino' (se já rodou nesta sessão) ou arquivos enviados abaixo.")

    df_11 = st.session_state.get("resultados_df_11")
    df_12 = st.session_state.get("resultados_df_12")

    if df_11 is None or df_12 is None:
        c1, c2 = st.columns(2)
        with c1:
            up11 = st.file_uploader("resultados_df_11.csv", type="csv", key="up11")
        with c2:
            up12 = st.file_uploader("resultados_df_12.csv", type="csv", key="up12")
        if up11 is not None:
            df_11 = pd.read_csv(up11)
        if up12 is not None:
            df_12 = pd.read_csv(up12)

    if df_11 is None or df_12 is None:
        st.info("Rode o treino na aba anterior ou envie os dois arquivos CSV acima.")
        return

    def filtrar(df, nome_dataset):
        d = df[df["Cenário"].isin(VARIANTES_ORDEM) & df["Modelo"].isin(MODELOS_DISTANCIA)].copy()
        d["Dataset"] = nome_dataset
        d["Cenário"] = pd.Categorical(d["Cenário"], categories=VARIANTES_ORDEM, ordered=True)
        return d

    d11 = filtrar(df_11, "df_11")
    d12 = filtrar(df_12, "df_12")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, df, nome_dataset in zip(axes, [d12, d11], ["df_12", "df_11"]):
        for modelo in MODELOS_DISTANCIA:
            sub = df[df["Modelo"] == modelo].sort_values("Cenário")
            if sub.empty:
                continue
            ax.plot(VARIANTES_CURTO[:len(sub)], sub["R²"], marker="o", label=modelo)
        ax.set_title(ROTULO_DATASET[nome_dataset], fontsize=9)
        ax.set_xlabel("Variante")
        ax.tick_params(axis="x", rotation=20)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("R²")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Conclusão automática")
    for nome_dataset, df in [("df_12", d12), ("df_11", d11)]:
        st.markdown(f"**{ROTULO_DATASET[nome_dataset]}**")
        for modelo in MODELOS_DISTANCIA:
            sub = df[df["Modelo"] == modelo].sort_values("Cenário")
            if sub.empty or sub["R²"].isna().all():
                continue
            r2_cru = sub[sub["Cenário"] == "Apenas TF-IDF (cru)"]["R²"]
            r2_final = sub[sub["Cenário"] == "Apenas TF-IDF (stopwords+lemma+POS)"]["R²"]
            if r2_cru.empty or r2_final.empty:
                continue
            delta = float(r2_final.iloc[0]) - float(r2_cru.iloc[0])
            direcao = "melhorou ✅" if delta > 0.01 else ("piorou ❌" if delta < -0.01 else "estável ➖")
            st.write(f"- {modelo}: R² {r2_cru.iloc[0]:.4f} → {r2_final.iloc[0]:.4f} ({direcao}, Δ={delta:+.4f})")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    st.title("📚 ASAG — Painel do TCC")
    st.caption("Automatic Short Answer Grading — dashboard e pipelines de treino (sem a extração Coh-Metrix, que continua sendo feita separadamente).")

    aba1, aba2, aba3, aba4 = st.tabs(["📊 Dashboard", "🚀 Rodar Treino", "🎯 Estilo Artigo", "📏 Modelos de Distância"])
    with aba1:
        aba_dashboard()
    with aba2:
        aba_rodar_treino()
    with aba3:
        aba_estilo_artigo()
    with aba4:
        aba_modelos_distancia()


if __name__ == "__main__":
    main()