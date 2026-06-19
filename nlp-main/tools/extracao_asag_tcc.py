import sys
import os
import glob
import warnings
import pandas as pd
from joblib import Parallel, delayed

# Ignorar os avisos matemáticos (como divisão por zero) para não travar o processo
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuração de diretórios
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../src")) 
pasta_dataset = os.path.abspath(os.path.join(current_dir, "../dataset"))

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

extrator_global = None

def extrair_metricas(texto, nota):
    global extrator_global
    
    texto_str = str(texto) if pd.notna(texto) else ""
    texto_limpo = texto_str.strip()
    
    # 1. Cria SEMPRE o dicionário base para NUNCA perder a linha no dataset final
    metricas = {
        'resposta_original': texto_str,
        'nota_original': nota
    }

    try:
        if extrator_global is None:
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            from aibox.nlp.features.portuguese.cohmetrix import CohMetrixExtractor
            extrator_global = CohMetrixExtractor()
        
        # 2. Rede de Segurança: Se não tiver letras (ex: só pontuação ou números)
        if len(texto_limpo) < 2 or not any(c.isalpha() for c in texto_limpo):
            # Retorna só o dicionário base (as colunas do Coh-Metrix ficarão NaN e depois 0)
            return metricas
            
        resultado = extrator_global.extract(texto_str)
        metricas_extraidas = resultado.as_dict()
        
        # 3. Junta as métricas extraídas ao dicionário base
        metricas.update(metricas_extraidas)
        
        return metricas
        
    except Exception as e:
        # 4. Em caso de erro severo da ferramenta, também não quebra o loop!
        return metricas

if __name__ == '__main__':
    print("🚀 Iniciando MODO TURBO para arquivos separados (Treino e Teste)...")
    
    # Lista exata dos arquivos que queremos processar
    arquivos_para_processar = [
        "df_11_train.csv",
        "df_11_test.csv",
        "df_12_train.csv",
        "df_12_test.csv"
    ]
    
    COLUNA_RESPOSTA = 'answer_text'  
    COLUNA_NOTA = 'grade'              
    tamanho_do_lote = 50

    for nome_arquivo in arquivos_para_processar:
        caminho_arquivo = os.path.join(pasta_dataset, nome_arquivo)
        nome_saida = f"cohmetrix_{nome_arquivo}"
        caminho_saida = os.path.join(pasta_dataset, nome_saida)
        arquivo_backup = os.path.join(pasta_dataset, f"BKP_{nome_saida}")
        
        if not os.path.exists(caminho_arquivo):
            print(f"⚠️ Ignorando: {nome_arquivo} não foi encontrado na pasta dataset.")
            continue
            
        if os.path.exists(caminho_saida):
            print(f"✅ Ignorando: {nome_saida} já foi extraído anteriormente.")
            continue

        print(f"\n" + "="*50)
        print(f"📂 Processando: {nome_arquivo}")
        print("="*50)

        df = pd.read_csv(caminho_arquivo)
        todas_as_features = []

        for i in range(0, len(df), tamanho_do_lote):
            lote = df.iloc[i:i+tamanho_do_lote]
        
            resultados_do_lote = Parallel(n_jobs=-1)( 
                delayed(extrair_metricas)(row[COLUNA_RESPOSTA], row[COLUNA_NOTA]) 
                for _, row in lote.iterrows()
            )
            
            # Agora os resultados limpos contêm TODAS as linhas, nunca descartamos nada
            resultados_limpos = [r for r in resultados_do_lote if r is not None]
            todas_as_features.extend(resultados_limpos)
            
            print(f"  [{len(todas_as_features)}/{len(df)}] textos processados... (Guardando Backup)")
            pd.DataFrame(todas_as_features).to_csv(arquivo_backup, index=False)

        # Salva o arquivo final 
        df_features = pd.DataFrame(todas_as_features)
        df_features.to_csv(caminho_saida, index=False)

        # Limpa o backup
        if os.path.exists(arquivo_backup):
            os.remove(arquivo_backup)
            
        print(f"🎉 Concluído! Arquivo '{nome_saida}' gerado com sucesso.")

    print("\n🏁 EXTRAÇÃO TOTAL FINALIZADA! Pode voltar para o seu script de Machine Learning.")