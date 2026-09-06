# Mudanças aplicadas

## Dados

- carregador único para Anderson e Galhardi;
- mapeamento explícito entre número da questão e `question_id`;
- esquema canônico compartilhado;
- auditoria de ausências, duplicatas e leakage;
- preservação dos splits históricos Anderson;
- reconstrução do split Galhardi 70/30 por linha;
- novo split agrupado por pergunta+resposta;
- Leave-One-Question-Out genérico;
- suporte separado para notas principais e outros avaliadores.

O split por linha é opt-in. A execução padrão do Galhardi usa somente o split
agrupado e Leave-One-Question-Out, ambos com sobreposição normalizada zero.

## Representações

- quatro variantes de TF-IDF preservadas;
- MiniLM e BERTimbau configuráveis no mesmo executor;
- cache em memória para não recalcular o mesmo texto durante uma bateria;
- Coh-Metrix histórico Anderson preservado;
- geração opcional de cache Coh-Metrix para Galhardi;
- combinações do perfil completo preservadas;
- perfil semelhante ao artigo preservado.

## Modelos

- LinearRegression;
- KNeighborsRegressor;
- SVR;
- DecisionTreeRegressor;
- RandomForestRegressor;
- HistGradientBoostingRegressor;
- MLPRegressor;
- XGBRegressor quando instalado.

Nenhum hiperparâmetro existente foi alterado.

## Resultados

- CSV por execução;
- CSV consolidado;
- comparação dos melhores cenários;
- ablação TF-IDF e recorte dos modelos de distância;
- resumo LOQ com média e desvio-padrão;
- CSV de auditoria;
- predições opcionais por instância;
- relatório Excel consolidado;
- comparação opcional com a planilha do artigo;
- arquivo de erros, sem ocultar falhas de execução.
