import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt

print("1. Carregando os dados de Backup do Coh-Metrix...")
df_cohmetrix = pd.read_csv("caracteristicas_cohmetrix_BKP.csv")
print(f"Total de respostas para o MVP: {len(df_cohmetrix)}")

print("\n2. Gerando Embeddings para os textos... (Isso transforma as frases em matemática)")
modelo_ia = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
textos = df_cohmetrix['resposta_original'].fillna("").astype(str).tolist()
embeddings = modelo_ia.encode(textos, show_progress_bar=True)

df_embeddings = pd.DataFrame(embeddings)
df_embeddings.columns = [f"emb_{i}" for i in range(df_embeddings.shape[1])]

print("\n3. Juntando as métricas gramaticais com as matrizes semânticas...")
X_cohmetrix = df_cohmetrix.drop(columns=['resposta_original', 'nota_original']).fillna(0)
X_final = pd.concat([X_cohmetrix, df_embeddings], axis=1)
y = df_cohmetrix['nota_original']

print("4. Separando dados para treinar a IA (80% treino, 20% teste)...")
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

print("5. Treinando o modelo (HistGradientBoosting)...")
modelo_ml = HistGradientBoostingClassifier(random_state=42)
modelo_ml.fit(X_train, y_train)


y_pred = modelo_ml.predict(X_test)
acc = accuracy_score(y_test, y_pred)
qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic') 

print("\n" + "="*50)
print("🏆 RESULTADOS DA PROVA DE CONCEITO (MVP) 🏆")
print("="*50)
print(f"Acurácia: {acc*100:.2f}% (Exatidão na previsão da nota)")
print(f"Métrica QWK: {qwk:.4f} (Nível de concordância com o professor humano)")
print("="*50)

print("\n7. Gerando o gráfico para a sua entrega de hoje...")
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
plt.title(f"Matriz de Confusão do Modelo (Métrica QWK: {qwk:.2f})")
plt.xlabel("Nota Prevista pela IA")
plt.ylabel("Nota Verdadeira do Professor")


nome_imagem = "resultado_mvp_matriz_confusao.png"
plt.savefig(nome_imagem, dpi=300, bbox_inches='tight')
print(f"✅ Gráfico salvo com sucesso como '{nome_imagem}'.")
print("Anexe esta imagem e os resultados no seu relatório de hoje!")