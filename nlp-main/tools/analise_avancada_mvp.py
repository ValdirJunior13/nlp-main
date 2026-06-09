import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sentence_transformers import SentenceTransformer
from sklearn.base import clone
 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import nltk
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
 
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
 

nltk.download('stopwords', quiet=True)
stop_words_pt = stopwords.words('portuguese')

tfidf_vec = TfidfVectorizer(max_features=1000, stop_words=stop_words_pt)
DATASET = "df_11"
 
PASTA = os.path.join(os.path.dirname(__file__), "..", "dataset")
 
print(f"1. Carregando os datasets de treino e teste ({DATASET})...")
 
df_train = pd.read_csv(os.path.join(PASTA, f"{DATASET}_train.csv"))
df_test  = pd.read_csv(os.path.join(PASTA, f"{DATASET}_test.csv"))
 
COL_TEXTO = "answer_text"
COL_NOTA  = "grade"
 
print(f"   Treino: {len(df_train)} amostras | Teste: {len(df_test)} amostras")
 
textos_train = df_train[COL_TEXTO].fillna("").astype(str).tolist()
textos_test  = df_test[COL_TEXTO].fillna("").astype(str).tolist()
 
y_train = df_train[COL_NOTA].reset_index(drop=True)
y_test  = df_test[COL_NOTA].reset_index(drop=True)
 
print("2. Gerando Embeddings (Sentence Transformers)...")
modelo_ia = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
 
emb_train = modelo_ia.encode(textos_train, show_progress_bar=True)
emb_test  = modelo_ia.encode(textos_test,  show_progress_bar=True)
 
cols_emb = [f"emb_{i}" for i in range(emb_train.shape[1])]
df_emb_train = pd.DataFrame(emb_train, columns=cols_emb)
df_emb_test  = pd.DataFrame(emb_test,  columns=cols_emb)
 
print("2.5. Gerando TF-IDF...")
tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_vec.fit(textos_train) 
 
cols_tfidf = tfidf_vec.get_feature_names_out()
df_tfidf_train = pd.DataFrame(tfidf_vec.transform(textos_train).toarray(), columns=cols_tfidf)
df_tfidf_test  = pd.DataFrame(tfidf_vec.transform(textos_test).toarray(),  columns=cols_tfidf)

print("3. Verificando features Coh-Metrix...")

ficheiro_cohmetrix = os.path.join(os.path.dirname(__file__), "caracteristicas_cohmetrix_BKP.csv")
usar_cohmetrix = os.path.exists(ficheiro_cohmetrix)

if usar_cohmetrix:
    df_coh = pd.read_csv(ficheiro_cohmetrix)
    colunas_drop     = ['resposta_original', 'nota_original']
    colunas_features = [c for c in df_coh.columns if c not in colunas_drop]
    df_coh_dedup = df_coh.drop_duplicates(subset='resposta_original').reset_index(drop=True)

    df_train_coh = df_train.merge(
        df_coh_dedup, left_on='answer_text', right_on='resposta_original', how='inner'
    ).reset_index(drop=True)

    df_test_coh = df_test.merge(
        df_coh_dedup, left_on='answer_text', right_on='resposta_original', how='inner'
    ).reset_index(drop=True)

    coh_train   = df_train_coh[colunas_features].fillna(0).reset_index(drop=True)
    coh_test    = df_test_coh[colunas_features].fillna(0).reset_index(drop=True)
    y_train_coh = df_train_coh['grade'].reset_index(drop=True)
    y_test_coh  = df_test_coh['grade'].reset_index(drop=True)

    textos_train_coh = df_train_coh['answer_text'].fillna("").astype(str).tolist()
    textos_test_coh  = df_test_coh['answer_text'].fillna("").astype(str).tolist()

    emb_train_coh = modelo_ia.encode(textos_train_coh, show_progress_bar=False)
    emb_test_coh  = modelo_ia.encode(textos_test_coh,  show_progress_bar=False)
    df_emb_train_coh = pd.DataFrame(emb_train_coh, columns=cols_emb)
    df_emb_test_coh  = pd.DataFrame(emb_test_coh,  columns=cols_emb)

    df_tfidf_train_coh = pd.DataFrame(tfidf_vec.transform(textos_train_coh).toarray(), columns=cols_tfidf)
    df_tfidf_test_coh  = pd.DataFrame(tfidf_vec.transform(textos_test_coh).toarray(),  columns=cols_tfidf)

    print(f"   Coh-Metrix: {len(coh_train)} treino / {len(coh_test)} teste (após merge pelo texto)")
    print(f"   Features Coh-Metrix: {len(colunas_features)}")
else:
    print("   AVISO: Coh-Metrix não encontrado.")

print("4. Preparando cenários de características (Ablation Study)...")
if usar_cohmetrix:
    cenarios = {
        "Apenas TF-IDF": (
            df_tfidf_train, df_tfidf_test, y_train, y_test, df_test),
        "Apenas Embeddings": (
            df_emb_train, df_emb_test, y_train, y_test, df_test),
        "Apenas Coh-Metrix": (
            coh_train, coh_test, y_train_coh, y_test_coh, df_test_coh),
        "TF-IDF + Embeddings": (
            pd.concat([df_tfidf_train, df_emb_train], axis=1),
            pd.concat([df_tfidf_test,  df_emb_test],  axis=1),
            y_train, y_test, df_test),
        "TF-IDF + Coh-Metrix": (
            pd.concat([df_tfidf_train_coh, coh_train], axis=1),
            pd.concat([df_tfidf_test_coh,  coh_test],  axis=1),
            y_train_coh, y_test_coh, df_test_coh),
        "Coh-Metrix + Embeddings": (
            pd.concat([coh_train, df_emb_train_coh], axis=1),
            pd.concat([coh_test,  df_emb_test_coh],  axis=1),
            y_train_coh, y_test_coh, df_test_coh),
        "Tudo (TF-IDF + Coh-Metrix + Emb)": (
            pd.concat([df_tfidf_train_coh, coh_train, df_emb_train_coh], axis=1),
            pd.concat([df_tfidf_test_coh,  coh_test,  df_emb_test_coh],  axis=1),
            y_train_coh, y_test_coh, df_test_coh),
    }
else:
    cenarios = {
        "Apenas TF-IDF": (
            df_tfidf_train, df_tfidf_test, y_train, y_test, df_test),
        "Apenas Embeddings": (
            df_emb_train, df_emb_test, y_train, y_test, df_test),
        "TF-IDF + Embeddings": (
            pd.concat([df_tfidf_train, df_emb_train], axis=1),
            pd.concat([df_tfidf_test,  df_emb_test],  axis=1),
            y_train, y_test, df_test),
    }
 
print("\n" + "-"*55)
print("RESUMO DOS CENÁRIOS:")
for nome, (Xtr, Xte, *_) in cenarios.items():
    print(f"  - {nome}: {Xtr.shape[1]} features | {len(Xtr)} amostras treino")
print("-"*55 + "\n")
 
modelos = {
    "Regressão Linear":     LinearRegression(),
    "KNN":                  KNeighborsRegressor(n_neighbors=5),
    "SVR":                  SVR(),
    "Árvore de Decisão":    DecisionTreeRegressor(random_state=42),
    "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    "Rede Neural (MLP)":    MLPRegressor(random_state=42, max_iter=1000),
}
 
print("5. Treinando e avaliando modelos...\n")
 
resultados_gerais     = []
melhor_r2_global      = -float('inf')
melhor_modelo_global  = None
nome_melhor_modelo    = ""
melhor_cenario_global = ""
previsoes_melhor      = None
X_test_melhor         = None
y_test_melhor_ref     = None
df_test_melhor        = None
 
for nome_cenario, (X_train_c, X_test_c, y_tr, y_te, df_te) in cenarios.items():
    print(f"=== Cenário: {nome_cenario} ===")
 
    X_train_c = X_train_c.reset_index(drop=True)
    X_test_c  = X_test_c.reset_index(drop=True)
    y_tr      = y_tr.reset_index(drop=True)
    y_te      = y_te.reset_index(drop=True)
 
    for nome_modelo, modelo_base in modelos.items():
        modelo = clone(modelo_base)
        modelo.fit(X_train_c, y_tr)
        y_pred = modelo.predict(X_test_c)
 
        mae  = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2   = r2_score(y_te, y_pred)
 
        resultados_gerais.append({
            "Dataset": DATASET,
            "Cenário": nome_cenario,
            "Modelo":  nome_modelo,
            "MAE":     mae,
            "RMSE":    rmse,
            "R²":      r2,
        })
 
        if r2 > melhor_r2_global:
            melhor_r2_global      = r2
            melhor_modelo_global  = modelo
            nome_melhor_modelo    = nome_modelo
            melhor_cenario_global = nome_cenario
            previsoes_melhor      = y_pred
            X_test_melhor         = X_test_c
            y_test_melhor_ref     = y_te
            df_test_melhor        = df_te
 
print("\n" + "="*55)
print(f"RESULTADO FINAL — Dataset: {DATASET}")
print("="*55)
 
df_resultados = pd.DataFrame(resultados_gerais)
df_resultados = df_resultados.sort_values(by="R²", ascending=False).reset_index(drop=True)
 
pd.set_option('display.max_rows', None)
print(df_resultados.to_string(
    formatters={'MAE': '{:.4f}'.format, 'RMSE': '{:.4f}'.format, 'R²': '{:.4f}'.format}
))
 
nome_resultado = f"resultados_{DATASET}.csv"
df_resultados.to_csv(nome_resultado, index=False)
print(f"\nResultados salvos em '{nome_resultado}'")
 
print("\n" + "="*55)
print(f"ANÁLISE DE ERROS — Vencedor: {nome_melhor_modelo} | {melhor_cenario_global}")
print("="*55)
 
df_analise = pd.DataFrame({
    'Texto':         df_test_melhor[COL_TEXTO].reset_index(drop=True),
    'Real':          y_test_melhor_ref.values,
    'Previsto':      previsoes_melhor,
    'Erro_Absoluto': np.abs(y_test_melhor_ref.values - previsoes_melhor),
})
 
piores = df_analise.sort_values(by='Erro_Absoluto', ascending=False).head(5)
for _, linha in piores.iterrows():
    print(f"\n  PROFESSOR: {linha['Real']} | IA: {linha['Previsto']:.2f} (Erro: {linha['Erro_Absoluto']:.2f})")
    print(f"  RESPOSTA: {linha['Texto']}")
 
print("\n" + "="*55)
print("IMPORTÂNCIA DAS FEATURES (Top 10)")
print("="*55)
 
try:
    resultado_imp = permutation_importance(
        melhor_modelo_global, X_test_melhor, y_test_melhor_ref,
        n_repeats=5, random_state=42, n_jobs=-1
    )
    importancias = pd.Series(resultado_imp.importances_mean, index=X_test_melhor.columns)
    top_10 = importancias.sort_values(ascending=False).head(10)
 
    plt.figure(figsize=(10, 6))
    top_10.sort_values().plot(kind='barh', color='#1f77b4')
    plt.title(f"Top 10 Features — {nome_melhor_modelo} ({melhor_cenario_global}) | {DATASET}")
    plt.xlabel("Permutation Importance")
    plt.tight_layout()
 
    nome_grafico = f"resultado_importancia_{DATASET}.png"
    plt.savefig(nome_grafico, dpi=300)
    print(f"Gráfico salvo como '{nome_grafico}'.")
 
except Exception as e:
    print(f"Não foi possível gerar o gráfico: {e}")