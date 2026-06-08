import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../../../"))

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

if current_dir in sys.path:
    sys.path.remove(current_dir)

try:
    import lexical_diversity
except ImportError:
    pass
finally:
    sys.path.insert(0, current_dir)
# ---------------------------------------------------

from cohmetrix import CohMetrixExtractor

# Dicionário com a explicação de cada métrica
METRIC_DESC = {
    # --- Descritivas (DES) ---
    "despc": "Nº de Parágrafos",
    "despc2": "Nº de Parágrafos (Alt)",
    "despl": "Média de Sentenças por Parágrafo",
    "despld": "Desvio Padrão Sentenças/Parágrafo",
    "dessc": "Nº de Sentenças",
    "dessl": "Média de Palavras por Sentença",
    "dessld": "Desvio Padrão Palavras/Sentença",
    "deswc": "Nº Total de Palavras",
    "deswlsy": "Média de Sílabas por Palavra",
    "deswlsyd": "Desvio Padrão Sílabas/Palavra",
    "deswllt": "Média de Letras por Palavra",
    "deswlltd": "Desvio Padrão Letras/Palavra",

    # --- Coesão Referencial (CRF) ---
    "crfno1": "Sobreposição de Substantivos (Sentenças Adjacentes)",
    "crfao1": "Sobreposição de Argumentos (Sentenças Adjacentes)",
    "crfso1": "Sobreposição de Raiz/Stem (Sentenças Adjacentes)",
    "crfnoa": "Sobreposição de Substantivos (Todas as Sentenças)",
    "crfaoa": "Sobreposição de Argumentos (Todas as Sentenças)",
    "crfsoa": "Sobreposição de Raiz/Stem (Todas as Sentenças)",
    "crfcwo1": "Sobreposição de Palavras de Conteúdo (Adjacentes)",
    "crfcwo1d": "Desvio Padrão Sobreposição Conteúdo (Adjacentes)",
    "crfcwoa": "Sobreposição de Palavras de Conteúdo (Todas)",
    "crfcwoad": "Desvio Padrão Sobreposição Conteúdo (Todas)",

    # --- Diversidade Léxica (LD) ---
    "ldttrc": "Type-Token Ratio (Palavras de Conteúdo)",
    "ldttra": "Type-Token Ratio (Todas as Palavras)",
    "ldmtlda": "Índice MTLD (Diversidade Léxica Textual)",
    "ldvocda": "Índice VOCD (Curva de Vocabulário)",

    # --- Conectivos (CNC) ---
    "cncadc": "Incidência de Conectivos Aditivos",
    "cncadd": "Incidência de Conectivos Aditivos (Geral)",
    "cncall": "Incidência Total de Conectivos",
    "cncalter": "Incidência de Conectivos Alternativos",
    "cnccaus": "Incidência de Conectivos Causais",
    "cnccomp": "Incidência de Conectivos Comparativos",
    "cncconce": "Incidência de Conectivos Concessivos",
    "cncconclu": "Incidência de Conectivos Conclusivos",
    "cnccondi": "Incidência de Conectivos Condicionais",
    "cncconfor": "Incidência de Conectivos de Conformidade",
    "cncconse": "Incidência de Conectivos Consecutivos",
    "cncexpli": "Incidência de Conectivos Explicativos",
    "cncfinal": "Incidência de Conectivos Finais",
    "cncinte": "Incidência de Conectivos de Intensidade", # Ou similar
    "cnclogic": "Incidência de Conectivos Lógicos",
    "cncneg": "Incidência de Conectivos Negativos",
    "cncpos": "Incidência de Conectivos Positivos",
    "cncprop": "Incidência de Conectivos Proporcionais",
    "cnctemp": "Incidência de Conectivos Temporais",

    # --- Modelo Situacional (SM) ---
    "smintep": "Coesão Temporal (Geral)",
    "smintep_sentence": "Coesão Temporal (Sentença)",
    "sminter": "Coesão Temporal (Intermediária)",
    "smcauswn": "Coesão Causal (WordNet)",

    # --- Sintaxe (SYN) ---
    "synle": "Embeddings à Esquerda (Complexidade)",
    "synnp": "Complexidade de Sintagmas Nominais",
    "synmedpos": "Distância de Edição (POS Tags)",
    "synmedlem": "Distância de Edição (Lemas)",
    "synmedwrd": "Distância de Edição (Palavras)",
    "synstruta": "Similaridade Estrutural (A)",
    "synstrutt": "Similaridade Estrutural (T)",

    # --- Densidade de Sintagmas (DR) ---
    "drnp": "Densidade de Sintagmas Nominais",
    "drvp": "Densidade de Sintagmas Verbais",
    "drap": "Densidade de Sintagmas Adjetivos/Adv",
    "drpp": "Densidade de Sintagmas Preposicionais",
    "drpval": "Densidade de Verbos (Valência)", # Possível interpretação
    "drneg": "Densidade de Negações",
    "drgerund": "Densidade de Gerúndios",
    "drinf": "Densidade de Infinitivos",

    # --- Informação da Palavra (WRD) ---
    "wrdnoun": "Incidência de Substantivos",
    "wrdverb": "Incidência de Verbos",
    "wrdadj": "Incidência de Adjetivos",
    "wrdadv": "Incidência de Advérbios",
    "wrdpro": "Incidência de Pronomes",
    "wrdprp1s": "Pronomes Pessoais 1ª Pess. Sing (eu)",
    "wrdprp1p": "Pronomes Pessoais 1ª Pess. Plural (nós)",
    "wrdprp2": "Pronomes Pessoais 2ª Pessoa",
    "wrdprp2s": "Pronomes Pessoais 2ª Pess. Sing",
    "wrdprp2p": "Pronomes Pessoais 2ª Pess. Plural",
    "wrdprp3s": "Pronomes Pessoais 3ª Pess. Sing",
    "wrdprp3p": "Pronomes Pessoais 3ª Pess. Plural",
    "wrdfrqc": "Frequência de Palavras de Conteúdo",
    "wrdfrqa": "Frequência de Todas as Palavras",
    "wrdfrqmc": "Frequência Média (Conteúdo)",
    "wrdaoac": "Idade de Aquisição (Estimada)",
    "wrdfamc": "Familiaridade das Palavras",
    "wrdcncc": "Concretude das Palavras",
    "wrdimgc": "Imaginabilidade das Palavras",
    "wrdmeac": "Significância (Meaningfulness)",

    # --- Legibilidade (RD) ---
    "rdfre": "Legibilidade Flesch (Facilidade)",
    "rdfkgl": "Escolaridade Flesch-Kincaid (Ano escolar)",
    "rdl2": "Legibilidade L2 (Segunda Língua)"
}

def avaliar_texto(texto):
    print("\n" + "="*80)
    print(f"{'ANÁLISE TEXTO':^80}")
    print("="*80)
    
    try:
        # 1. Extração
        extrator = CohMetrixExtractor()
        resultado = extrator.extract(texto)
        metricas = resultado.as_dict()
        
        # 2. Exibição do Texto
        print(f"\n[Texto Analisado]:\n\"{texto.strip()[:100]}...\" (e mais {len(texto)-100} caracteres)")
        print("-" * 80)
        print(f"\n[Resultados - {len(metricas)} métricas encontradas]:")
        print(f"{'MÉTRICA (SIGLA)':<25} | {'VALOR':<10} | {'DESCRIÇÃO'}")
        print("-" * 80)
        
        # 3. Exibição das métricas com descrição
        # Agrupa chaves desconhecidas se houver
        chaves_ordenadas = sorted(metricas.keys())
        
        for sigla in chaves_ordenadas:
            valor = metricas[sigla]
            # Busca a descrição no dicionário, se não achar usa "Métrica CohMetrix"
            descricao = METRIC_DESC.get(sigla, "Métrica Específica CohMetrix")
            
            # Formatação: Sigla (25 chars) | Valor (10 chars, 4 casas) | Descrição
            print(f"{sigla:<25} | {valor:<10.4f} | {descricao}")
            
        print("=" * 80 + "\n")
        return metricas

    except Exception as e:
        print(f"Erro durante a extração: {e}")
        import traceback
        traceback.print_exc()
        return {}

# Exemplo de uso
texto_exemplo = """
O Coh-Metrix é um sistema computacional que possui diferentes 
medidas de análise textual incluindo legibilidade, coerência 
e coesão textual. Essas medidas permitem uma análise mais 
profunda de diferentes tipos de textos educacionais como redações, 
respostas de perguntas abertas, mensagens em fóruns educacionais, 
entre outros. Este protótipo apresenta uma uma API web com a adaptação 
das medidas do Coh-Metrix para a língua portuguesa do Brasil.
"""

texto_exemplo2 = """
O cachorro azul decidiu que a matemática era mais rápida às 
quartas-feiras, enquanto a janela pensava em viagens longas. 
Não havia motivo para o relógio cantar, mas mesmo assim o chão 
respondeu com silêncio alto. As ideias correram para trás, porque 
o amanhã já tinha acabado ontem na prateleira do vento.

No segundo parágrafo, a chuva escreveu cartas para ninguém e o lápis 
esqueceu como andar. Algumas palavras estavam cansadas de ser frases, 
então viraram cadeiras sem pernas. O resultado foi um som quadrado, que 
não explica nada, mas continua andando para o lado.
"""

if __name__ == "__main__":
    avaliar_texto(texto_exemplo)
    avaliar_texto(texto_exemplo2)