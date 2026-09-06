# Pipeline ASAG — Anderson e Galhardi

Esta refatoração mantém a execução simples: um único arquivo executa os cenários,
gera CSVs individuais, consolida os resultados e cria o relatório Excel.

## Estrutura esperada dos dados

Os arquivos brutos não são alterados. A estrutura recomendada é:

```text
dataset/
  anderson/
    df_11_train.csv
    df_11_test.csv
    df_12_train.csv
    df_12_test.csv
    cohmetrix_df_11_train.csv
    cohmetrix_df_11_test.csv
    cohmetrix_df_12_train.csv
    cohmetrix_df_12_test.csv

  galhardi/
    questions.csv                 # ou questionsv2.csv
    student_answers_and_grades_v2.csv
    student_answers_and_grades_v2_other_graders.csv
```

O carregador também procura os arquivos recursivamente, portanto pequenas
diferenças de subpastas não exigem alteração do código.

## Instalação

Use o mesmo ambiente virtual do projeto atual. Caso necessário:

```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_lg
```

O Coh-Metrix gerado pelo projeto continua dependendo do pacote local `aibox`
existente na pasta `src` do projeto original.

## Execução completa

```bash
python executar_experimentos.py
```

Por padrão, a execução inclui:

- Anderson `df_11` e `df_12`;
- Galhardi nas questões 1, 9, 11 e 12;
- split 70/30 agrupado por pergunta+resposta;
- quatro folds Leave-One-Question-Out;
- MiniLM e BERTimbau;
- perfil completo de representações e modelos;
- avaliador principal;
- reutilização dos arquivos Coh-Metrix já existentes.

O split Galhardi 70/30 por linha fica disponível apenas como diagnóstico, pois
pode separar respostas idênticas entre treino e teste. Para executá-lo
explicitamente:

```bash
python executar_experimentos.py \
  --fontes galhardi \
  --cenarios four_questions_row
```

## Execuções menores

Somente auditoria, sem carregar spaCy ou modelos de embeddings:

```bash
python executar_experimentos.py --somente-auditoria
```

Somente Galhardi e MiniLM:

```bash
python executar_experimentos.py --fontes galhardi --embeddings minilm
```

Somente Leave-One-Question-Out:

```bash
python executar_experimentos.py \
  --fontes galhardi \
  --cenarios leave_one_question_out \
  --embeddings minilm
```

As questões também são parametrizáveis. O loop Leave-One-Question-Out será
criado automaticamente para cada número informado:

```bash
python executar_experimentos.py \
  --fontes galhardi \
  --cenarios leave_one_question_out \
  --questoes 1 9 11 12
```

Executar também o perfil semelhante ao artigo:

```bash
python executar_experimentos.py --perfis full paper_like
```

Usar o arquivo de outros avaliadores:

```bash
python executar_experimentos.py \
  --fontes galhardi \
  --avaliadores other
```

Gerar Coh-Metrix para Galhardi quando o cache ainda não existir:

```bash
python executar_experimentos.py --cohmetrix generate
```

Essa opção pode demorar. Depois da primeira geração, use o padrão
`--cohmetrix existing`.

## Arquivo de referência do artigo

Para atualizar também as abas de comparação do script anterior:

```bash
python executar_experimentos.py \
  --referencia-artigo "relatorio_geral_organizado_v3 (2).xlsx"
```

As comparações do artigo usam somente Anderson + BERTimbau + perfil completo,
preservando o significado original das abas Split 1 e Split 2.

## Saídas

```text
outputs/
  audits/
    auditoria_datasets.csv

  results/
    resultados__<execucao>.csv
    resultados_consolidados.csv
    comparacao_melhores_cenarios.csv
    ablacao_tfidf.csv
    analise_modelos_distancia.csv
    resumo_leave_one_question_out.csv

  predictions/
    predicoes__<execucao>.csv       # somente com --previsoes
    predicoes_consolidadas.csv

  cache/
    cohmetrix_galhardi.csv

  erros_execucao.csv                # criado apenas se houver erros
  Relatorio_TCC_Consolidado.xlsx
```

## Colunas principais do consolidado

- `Fonte`: Anderson ou Galhardi;
- `Cenario_Dados`: forma de preparação e divisão;
- `Fold`: holdout ou questão deixada para teste;
- `Avaliador`: arquivo de notas utilizado;
- `Politica_Duplicatas`: baseline, linha, grupo ou questão inteira;
- `Embedding`: MiniLM ou BERTimbau;
- `Perfil`: completo ou semelhante ao artigo;
- `Cenário`: representação textual/linguística;
- `Modelo`: regressor;
- `Leakage_Normalizado`: chaves pergunta+resposta presentes nos dois splits;
- `MAE`, `RMSE` e `R²`: métricas do experimento.

## Compatibilidade com os experimentos anteriores

- Os CSVs Anderson são carregados sem refazer seus splits.
- O `df_12` continua executável mesmo com leakage, que fica registrado na saída.
- Modelos e hiperparâmetros foram mantidos.
- TF-IDF continua sendo ajustado somente no treino.
- StandardScaler continua sendo ajustado somente no treino.
- Modelos baseados em árvores continuam recebendo as características sem escala.
- Os scripts antigos podem permanecer no projeto durante a validação.

A auditoria também registra o mapa entre número e ID de cada questão, a
quantidade de respostas por questão, as distribuições de notas no treino e no
teste, duplicatas e a quantidade de linhas de teste afetadas por sobreposição.

Antes de substituir os resultados históricos, compare os CSVs antigos com os
novos para Anderson. Diferenças podem ocorrer por versões de bibliotecas,
modelo BERTimbau carregado por `SentenceTransformer` ou disponibilidade do
XGBoost.
