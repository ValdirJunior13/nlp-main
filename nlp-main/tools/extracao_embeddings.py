import os
import glob
import pandas as pd
import kagglehub
from sentence_transformers import SentenceTransformer

# 1. O Caçador de Datasets do Kaggle
print("A descarregar/carregar a pasta do dataset do Kaggle...")
pasta_dataset = kagglehub.dataset_download("lucasbgalhardi/pt-asag-2018")

arquivos_dados = glob.glob(os.path.join(pasta_dataset, "**", "*.csv"), recursive=True)
arquivos_dados.extend(glob.glob(os.path.join(pasta_dataset, "**", "*.xlsx"), recursive=True))

if not arquivos_dados:
    raise FileNotFoundError("Nenhum ficheiro de dados (.csv ou .xlsx) foi encontrado no dataset!")

print("\nFicheiros encontrados no Kaggle:")
caminho_do_arquivo = None

for arquivo in arquivos_dados:
    nome = os.path.basename(arquivo).lower()
    print(f" -> {nome}")
    if "concept" not in nome and "question" not in nome:
        caminho_do_arquivo = arquivo

if not caminho_do_arquivo:
    caminho_do_arquivo = arquivos_dados[0]

print(f"\n✅ Ficheiro selecionado para a extração: {os.path.basename(caminho_do_arquivo)}")

if caminho_do_arquivo.endswith('.csv'):
    df = pd.read_csv(caminho_do_arquivo)
else:
    df = pd.read_excel(caminho_do_arquivo)

COLUNA_RESPOSTA = 'answer_text'  
COLUNA_NOTA = 'grade'       

respostas_alunos = df[COLUNA_RESPOSTA].fillna("").astype(str).tolist()
notas_originais = df[COLUNA_NOTA].tolist()

print("\nA carregar o modelo de Inteligência Artificial...")
print("(Se for a primeira vez, fará o download do modelo da Hugging Face)")
modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("\nA transformar os textos em matrizes matemáticas (Embeddings)...")
embeddings = modelo.encode(respostas_alunos, show_progress_bar=True)

print("\nA guardar os dados no ficheiro CSV...")
df_embeddings = pd.DataFrame(embeddings)

df_embeddings['resposta_original'] = respostas_alunos
df_embeddings['nota_original'] = notas_originais

df_embeddings.to_csv("caracteristicas_embeddings.csv", index=False)
print("🎉 Sucesso absoluto! O ficheiro 'caracteristicas_embeddings.csv' foi gerado na diretoria tools.")