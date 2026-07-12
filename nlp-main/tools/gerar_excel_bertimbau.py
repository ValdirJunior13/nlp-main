import pandas as pd
import numpy as np
import os

# Nomes dos arquivos de entrada e saída
ARQUIVO_BERTIMBAU = "relatorio_geral_experimentos_asag_bertimbau.csv"
ARQUIVO_REF = "relatorio_geral_organizado_v3 (2).xlsx"
ARQUIVO_SAIDA = "Relatorio_TCC_BERTimbau_Organizado.xlsx"

print("Carregando resultados do BERTimbau...")
df_bertimbau = pd.read_csv(ARQUIVO_BERTIMBAU)

# Separando os datasets
df_11 = df_bertimbau[df_bertimbau['Dataset'] == 'df_11'].copy()
df_12 = df_bertimbau[df_bertimbau['Dataset'] == 'df_12'].copy()

print("Carregando arquivo de referência do artigo...")
xls_orig = pd.ExcelFile(ARQUIVO_REF)
comp1 = pd.read_excel(xls_orig, sheet_name="5. Comparação Completa Split1", header=2)
comp2 = pd.read_excel(xls_orig, sheet_name="6. Comparação Completa Split2", header=2)

comp1.columns = comp1.iloc[0]
comp1 = comp1.drop(0).reset_index(drop=True)
comp2.columns = comp2.iloc[0]
comp2 = comp2.drop(0).reset_index(drop=True)

map_modelos = {'SVM': 'SVR', 'RF': 'Random Forest', 'DT': 'Árvore de Decisão', 'XGB': 'XGBoost'}

def get_my_metrics(df_res, equivalent_str):
    if pd.isna(equivalent_str) or equivalent_str == '-': return np.nan, np.nan
    parts = equivalent_str.split('+')
    if len(parts) != 2: return np.nan, np.nan
    
    rep, mod = parts[0].strip(), parts[1].strip()
    my_mod = map_modelos.get(mod, mod)
    
    if rep == "TFIDF":
        sub = df_res[(df_res['Modelo'] == my_mod) & (df_res['Cenário'].str.contains('TF-IDF')) & (~df_res['Cenário'].str.contains('Embeddings|Coh-Metrix'))]
    elif rep == "Embeddings":
        sub = df_res[(df_res['Modelo'] == my_mod) & (df_res['Cenário'] == 'Apenas Embeddings')]
    else:
        return np.nan, np.nan

    if sub.empty: return np.nan, np.nan
    best = sub.sort_values('RMSE').iloc[0]
    return best['MAE'], best['RMSE']

print("Atualizando comparações com os dados do BERTimbau...")
for comp, df_res in zip([comp1, comp2], [df_12, df_11]):
    for i, row in comp.iterrows():
        if not pd.isna(row.get('Equivalente (Meu)')):
            mae, rmse = get_my_metrics(df_res, row['Equivalente (Meu)'])
            if not pd.isna(mae):
                comp.at[i, 'MAE (Meu)'] = round(mae, 4)
                rmse_col = 'RMSE (Meu)' if 'RMSE (Meu)' in comp.columns else 'RMSE (meu)'
                comp.at[i, rmse_col] = round(rmse, 4)
                if 'MAE (artigo)' in comp.columns and not pd.isna(row['MAE (artigo)']):
                    comp.at[i, 'Diferença (MAE)'] = round(mae - row['MAE (artigo)'], 4)

def criar_ablacao(df):
    variantes = ["Apenas TF-IDF (cru)", "Apenas TF-IDF (stopwords)", "Apenas TF-IDF (stopwords+lemma)", "Apenas TF-IDF (stopwords+lemma+POS)"]
    df_abl = df[df['Cenário'].isin(variantes)].copy()
    if df_abl.empty: return pd.DataFrame()
    piv = df_abl.pivot_table(index='Modelo', columns='Cenário', values='R²').reindex(columns=variantes)
    piv.columns = ['Cru', 'Stopwords', 'Stopwords+Lemma', 'Stopwords+Lemma+POS']
    piv['Δ R² (POS − Cru)'] = piv['Stopwords+Lemma+POS'] - piv['Cru']
    return piv.reset_index()

print("Formatando e salvando a nova planilha...")
writer = pd.ExcelWriter(ARQUIVO_SAIDA, engine='xlsxwriter')
workbook = writer.book

# --- DEFINIÇÃO DAS CORES E FORMATOS ---
formato_titulo = workbook.add_format({'bold': True, 'font_size': 11})
formato_cabecalho = workbook.add_format({
    'bg_color': '#2E4053',
    'font_color': '#FFFFFF',
    'bold': True,
    'border': 1
})
formato_celula = workbook.add_format({'border': 1})
formato_numero = workbook.add_format({'border': 1, 'num_format': '0.0000'})

def formatar_aba(df, aba_nome, linha_inicio, titulo=""):
    ws = writer.sheets[aba_nome]
    if titulo:
        ws.write(linha_inicio - 2, 0, titulo, formato_titulo)
    for col_num, value in enumerate(df.columns.values):
        ws.write(linha_inicio, col_num, str(value), formato_cabecalho)
    for row_num in range(len(df)):
        for col_num in range(len(df.columns)):
            val = df.iloc[row_num, col_num]
            if pd.isna(val):
                ws.write(linha_inicio + 1 + row_num, col_num, "", formato_celula)
            elif isinstance(val, (int, float)):
                ws.write(linha_inicio + 1 + row_num, col_num, val, formato_numero)
            else:
                ws.write(linha_inicio + 1 + row_num, col_num, val, formato_celula)

# --- ABA 1: Resultados Completos ---
df_completos = pd.concat([df_11.assign(Dataset="df_11 (Split 2 artigo)"), df_12.assign(Dataset="df_12 (Split 1 artigo)")], ignore_index=True)
writer.book.add_worksheet('1. Resultados Completos')
formatar_aba(df_completos, '1. Resultados Completos', 2, "Resultados Totais - BERTimbau (Pergunta + Resposta)")
writer.sheets['1. Resultados Completos'].set_column('A:B', 35)
writer.sheets['1. Resultados Completos'].set_column('C:D', 20)
writer.sheets['1. Resultados Completos'].set_column('E:G', 12)

# --- ABA 2: Ablação ---
abl_12 = criar_ablacao(df_12)
abl_11 = criar_ablacao(df_11)
writer.book.add_worksheet('2. Ablação Pré-processamento')
writer.sheets['2. Ablação Pré-processamento'].write(0, 0, "Ablação TF-IDF - BERTimbau", formato_titulo)
if not abl_12.empty:
    formatar_aba(abl_12, '2. Ablação Pré-processamento', 3, "df_12 (Split 1 artigo)")
if not abl_11.empty:
    start_11 = len(abl_12) + 7 if not abl_12.empty else 3
    formatar_aba(abl_11, '2. Ablação Pré-processamento', start_11, "df_11 (Split 2 artigo)")
writer.sheets['2. Ablação Pré-processamento'].set_column('A:F', 20)

# --- ABA 3: Modelos Distância ---
writer.book.add_worksheet('3. Modelos de Distância')
writer.sheets['3. Modelos de Distância'].write(0, 0, "Modelos de Distância - BERTimbau", formato_titulo)
if not abl_12.empty:
    dist_12 = abl_12[abl_12['Modelo'].isin(["KNN", "SVR", "Regressão Linear"])]
    formatar_aba(dist_12, '3. Modelos de Distância', 3, "df_12 (Split 1)")
if not abl_11.empty:
    dist_11 = abl_11[abl_11['Modelo'].isin(["KNN", "SVR", "Regressão Linear"])]
    start_11_dist = len(dist_12) + 7 if not abl_12.empty else 3
    formatar_aba(dist_11, '3. Modelos de Distância', start_11_dist, "df_11 (Split 2)")
writer.sheets['3. Modelos de Distância'].set_column('A:F', 20)

# --- ABA 5 e 6: Comparações ---
# Pulando a aba 4 pois o script gerou resultados consolidados
writer.book.add_worksheet('5. Comparação Completa Split1')
writer.book.add_worksheet('6. Comparação Completa Split2')
formatar_aba(comp1, '5. Comparação Completa Split1', 2, "Comparação com Artigo (Split 1) - BERTimbau")
formatar_aba(comp2, '6. Comparação Completa Split2', 2, "Comparação com Artigo (Split 2) - BERTimbau")
writer.sheets['5. Comparação Completa Split1'].set_column('A:H', 20)
writer.sheets['6. Comparação Completa Split2'].set_column('A:H', 20)

writer.close()
print(f"Sucesso! Excel gerado: '{ARQUIVO_SAIDA}'")