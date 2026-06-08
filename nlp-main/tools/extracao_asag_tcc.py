import sys
import os
import glob
import pandas as pd
import kagglehub
from joblib import Parallel, delayed
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../src")) 

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

extrator_global = None

def extrair_metricas(texto, nota):
    global extrator_global
    try:
        if extrator_global is None:
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            from aibox.nlp.features.portuguese.cohmetrix import CohMetrixExtractor
            extrator_global = CohMetrixExtractor()
        
        texto_str = str(texto) if pd.notna(texto) else ""
        if len(texto_str.strip()) < 2:
            return None
            
        resultado = extrator_global.extract(texto_str)
        metricas = resultado.as_dict()
        metricas['resposta_original'] = texto_str
        metricas['nota_original'] = nota
        return metricas
    except Exception as e:
        return None

if __name__ == '__main__':
    print("A carregar a pasta do dataset do Kaggle...")
    pasta_dataset = kagglehub.dataset_download("lucasbgalhardi/pt-asag-2018")

    arquivos_dados = glob.glob(os.path.join(pasta_dataset, "**", "*.csv"), recursive=True)
    arquivos_dados.extend(glob.glob(os.path.join(pasta_dataset, "**", "*.xlsx"), recursive=True))

    if not arquivos_dados:
        raise FileNotFoundError("Nenhum ficheiro de dados encontrado!")
    caminho_do_arquivo = None
    for arquivo in arquivos_dados:
        nome = os.path.basename(arquivo).lower()
        if "concept" not in nome and "question" not in nome:
            caminho_do_arquivo = arquivo

    if not caminho_do_arquivo:
        caminho_do_arquivo = arquivos_dados[0]

    print(f"\n✅ Ficheiro selecionado: {os.path.basename(caminho_do_arquivo)}")

    df = pd.read_csv(caminho_do_arquivo) if caminho_do_arquivo.endswith('.csv') else pd.read_excel(caminho_do_arquivo)

    COLUNA_RESPOSTA = 'answer_text'  
    COLUNA_NOTA = 'grade'               

    print("\n🚀 Iniciando MODO TURBO (Utilizando múltiplos núcleos do processador)...")
    print("Isto vai dividir o trabalho e guardar backups a cada lote de 50 textos.")

    todas_as_features = []
    tamanho_do_lote = 50
    for i in range(0, len(df), tamanho_do_lote):
        lote = df.iloc[i:i+tamanho_do_lote]
    
        resultados_do_lote = Parallel(n_jobs=4)(
            delayed(extrair_metricas)(row[COLUNA_RESPOSTA], row[COLUNA_NOTA]) 
            for _, row in lote.iterrows()
        )
        resultados_limpos = [r for r in resultados_do_lote if r is not None]
        todas_as_features.extend(resultados_limpos)
        
        print(f"[{len(todas_as_features)}] textos processados... (Guardando Backup)")
        pd.DataFrame(todas_as_features).to_csv("caracteristicas_cohmetrix_BKP.csv", index=False)

    df_features = pd.DataFrame(todas_as_features)
    df_features.to_csv("caracteristicas_cohmetrix.csv", index=False)

    if os.path.exists("caracteristicas_cohmetrix_BKP.csv"):
        os.remove("caracteristicas_cohmetrix_BKP.csv")
        
    print("\n🎉 Sucesso ABSOLUTO! O ficheiro 'caracteristicas_cohmetrix.csv' foi gerado e o tempo foi otimizado.")