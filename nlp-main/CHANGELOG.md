# Changelog

Todas mudanças pertinentes a biblioteca serão documentadas aqui. O formato utiliazdo se baseia no [_Keep a Changelog_](https://keepachangelog.com/en/1.1.0/) e esse projeto adota o [_Semantic Versioning_](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

> Sem modificações.

## [v0.0.0](https://github.com/aiboxlab/nlp/releases/tag/v0.0.0) - 2024-07-22

### Added

-  Versão _alpha_ da biblioteca;
-  Código fonte utilizado em múltiplas publicações do AiBox Lab de forma estruturada (módulos, pacotes, testes unitários iniciais);

## [v0.0.1](https://github.com/aiboxlab/nlp/releases/tag/v0.0.1) - 2024-09-14

### Added

- Suporte inicial para o Python 3.11;

## [v0.0.2](https://github.com/aiboxlab/nlp/releases/tag/v0.0.2) - 2024-09-14

### Added

- Parâmetro para limpeza de tags do dataset `PortugueseNarrativeEssays`;

### Removed

- Remoção de suporte ao Python <3.10;

## [v0.0.3](https://github.com/aiboxlab/nlp/releases/tag/v0.0.3) - 2025-05-16

### Changed

-  Atualização da biblioteca garantindo compatibilidade com Google Colab/Kaggle;
-  `pyproject.toml` agora define a versão mínima Python com 3.10;
-  Biblioteca não suporta oficialmente Python 3.12+;

## [v0.0.4](https://github.com/aiboxlab/nlp/releases/tag/v0.0.4) - 2025-05-16

### Fixed

- Correção de erros nos notebooks de exemplo para Kaggle/Colab (e.g., badges, instalção de dependências, etc);

## [v0.0.5](https://github.com/aiboxlab/nlp/releases/tag/v0.0.5) - 2025-05-16

### Added

- Esteiras CI/CD para geração de releases e publicação no PyPI/GH releases;

### Fixed

- Instruções de instalação no README;
- Adição de `try-cathch` em `aibox.nlp.resources`: garantia que em caso de parada de execução pelo usuário/sistema durante o download resulta em remoção dos arquivos parcialmente baixados;

## [v0.0.6](https://github.com/aiboxlab/nlp/releases/tag/v0.0.6) - 2025-05-18

### Fixed

- Nome da classe de regressor de LSTM (`LSTMClassifier` -> `LSTMRegressor`);

## [v0.1.0](https://github.com/aiboxlab/nlp/releases/tag/v0.1.0) - 2025-05-18

### Added

- Suporte para carregamento de `DataFrame`'s através de path-like em `DatasetDF`;
    - Suporte para arquivos `CSV`;
    - Adição de parâmetro que permite converter coluna `target` para valor numérico;
    - Método de classe `load_from_kaggle(...)` que utiliza o `kagglehub` para carregamento de datasets;
- Pacote `aibox.nlp.factory` agora possui função para carregamento de métricas, vetorizadores, e datasets a partir de nomes;

### Changed

- `kagglehub>=0.3.0` agora faz parte dos requisitos básicos da biblioteca;


## [v0.1.1](https://github.com/aiboxlab/nlp/releases/tag/v0.1.1) - 2025-05-27

### Added

- Documentação inicial de toda API;
- Requirements (`dev`) agora possui Sphinx e tema RTD para Sphinx;
- Biblioteca agora possui autor e mantainer em `pyproject.toml`;
- Testes unitários iniciais para todas funcionalidades;
- Método de classe `load_from_kaggle(...)` para `DatasetDF`;
- Sub-pacote `aibox.nlp.typing` com aliases comuns utilizados na biblioteca;

### Changed

- `LSTMEstimator` agora possui novo parâmetro para controle do `dtype`;
- `LSTMEstimator` agora lida corretamente com entradas que não possuem dimensão de sequência;
    - Conversão automática de `(n_samples, n_features)` para `(n_samples, 1, n_features)`;
- Mudanças em alguns docstrings da biblioteca;
- Mudanças em alguns type hints para refletir as premissas da biblioteca;
- `DatasetPortugueseNarrativeEssays` agora utiliza o método `load_from_kaggle` para carregamento do dataset da fonte pública oficial;

### Removed

- Remoção do sub-pacote `aibox.nlp.models`: não era utilizado pela biblioteca;
- Remoção do módulo `aibox.nlp.features.portuguese.punctuation`: arquivo vazio;

## [v0.2.0](https://github.com/aiboxlab/nlp/releases/tag/v0.2.0) - 2025-05-28

### Added

- Configurações de tokenização e agregação em `FasttextWordVectorizer` (https://github.com/aiboxlab/nlp/issues/90);
- Novo parâmetro `criteria_maximize` para controlar se a melhor métrica deve ser maximizada ou minimizada em `SimpleExperiment`;
- Novo parâmetro `embeddings_dict` para inicializar cache de vetorizadores em `SimpleExperiment`;
- Novo campo `embeddings` em `SimpleExperimentExtras` com embeddings extraídos durante execução de `SimpleExperiment`;
- Suporte a cache inicial em `DictVectorizerCache`;
- Novo parâmetro `maximize` em `SimpleExperimentBuilder.best_criteria(...)`;

### Changed

- Cache de vetorizadores em `SimpleExperiment` só utiliza o nome da classe como identificador (previamente utilizava nome da classe + `id` do objeto);

## [v0.2.1](https://github.com/aiboxlab/nlp/releases/tag/v0.2.1) - 2025-05-28

### Fixed

- Regressão em `get_extractor` (`aibox.nlp.factory`): correção na construção de extratores agregados;
    - Typo no código fazia com que extratores agregados fossem compostos apenas pelo primeiro extrator;
    - Adição de novo caso de teste para evitar esse mesmo erro;
- Regressão `ReferentialCohesionExtractor`: tentativa de acesso à atributo não existente (`compute_adj_arg_ovl`);
    - Typo no nome do método (presença de `_`) impedia o funcionamento correto da classe;
    - Adição de novos casos de teste com textos do `essayBR` e `narrativeEssaysBR` para todas as features;
- Shape de `y` e saída da rede em `LSTMEstimator` para o caso de regressão (https://github.com/aiboxlab/nlp/issues/97);


## [v0.3.0](https://github.com/aiboxlab/nlp/releases/tag/v0.3.0) - 2025-05-30

### Added

- Adição de método `load_raw` para `DatasetEssayBR` e `DatasetPortugueseNarrativeEssays`;
    - Agora é possível carregar os dados as-is dos datasets como `DataFrame`;
- Adição de atributo `is_extended` para `DatasetEssayBR`;
- Adição de vetorização em batch (https://github.com/aiboxlab/nlp/issues/89);
    - Agora `vectorize(...)` aceita um `ArrayLike` de textos;
    - `AggregatedExtractor` permite a vetorização em batch respeitando as regras de cada extrator base;
    - `CachedExtractor` permite a vetorização em batch respeitando o cache (não é possível reaproveitar vetorização entre textos do mesmo batch);
    - `CachedVectorizer` e `TrainableCachedVectorizer` permitem a vetorização em batch;
- Adição de método `len` para `FeatureSet`;
- Novo sub-pacote `lazy_loading.patches` que contém facilidades de import lazy com aplicação de patches;
    - Para se obter uma instância do `cogroo4py` são necessárias algumas etapas de patch no `jpype` e nos loggers;

### Changed

- Redução de logs de download do `nltk` em `CohMetrixExtractor`;
- Redução de logs desnecessários para `AgreementExtractor` causados pelo `cogroo4py`;
- Todos extratores em `aibox.nlp.features` são pickle-safe;
    - Alguns componentes do extratores não são salvos durante `pickle` e são instanciados no momento de un-`pickle`;
- `DatasetDF.load_from_kaggle` agora retorna `DataFrame` numerado de `0..n-1`;
    - `ignore_index=True` em `pd.concat`;
    - `DatasetPortugueseNarrativeEssays.load_raw` tem o mesmo comportamento;

## [v0.3.1](https://github.com/aiboxlab/nlp/releases/tag/v0.3.1) - 2025-05-30

### Fixed

- Trechos da biblioteca assumem que um `FeatureSet` pode ser construído via `**kwargs`, agora todas implementações de `FeatureSet` seguem essa convenção;
    - `AggregatedFeatures` e `DictFeatureSet` foram atualizadas para permitir esse comportamento;
    - Testes unitários adicionados;
    - Uma discussão sobre esse convenção se encontra em https://github.com/aiboxlab/nlp/issues/105;

## [v0.3.2](https://github.com/aiboxlab/nlp/releases/tag/v0.3.2) - 2025-05-31

### Changed

- Refatoração das dependências opcionais e adição de URLs do projeto em `pyproject.toml`;

## [v0.3.3](https://github.com/aiboxlab/nlp/releases/tag/v0.3.3) - 2025-05-31

### Fixed

- Algumas modificaçõees nas URLs não haviam sido incluídas na release anterior, podendo gerar bugs;
- Typos em exemplos;


## [v0.4.0](https://github.com/aiboxlab/nlp/releases/tag/v0.4.0) - 2025-06-03

### Added

- Extratores de características agora possuem atributo `feature_set` que retorna a classe retornada pelo método `extract`;
- Conjunto de características agora aceitam `**kwargs` no `__init__`;
- `BERTVectorizer` agora aceita `batch_size` nos `**kwargs` de `vectorize` (https://github.com/aiboxlab/nlp/issues/109);
- `DatasetDF` agora permite conjuntos customizados (https://github.com/aiboxlab/nlp/issues/110);
  - Conjuntos de <treino, teste> ou splits podem ser definidos durante inicialização;
  - Um detalhe importante é que a classe ignora os parâmetros passados pelo usuário caso tais conjuntos sejam pré-definidos;
- Adicionada métrica de acurácia: `Accuracy`;
- Adicionados novos estimadores de classificação e regressão (https://github.com/aiboxlab/nlp/issues/92);
    - `AdaBoost{Classifier,Regressor}`;
    - `GaussianNBClassifier`;
    - `KNeighbors{Classifier,Regressor}`;
    - `MLP{Classifier,Regressor}`;
    - `Transformer{Classifier,Regressor}`;
        - Esse estimador aprende representações vetoriais (embeddings) de forma implícita e deve ser utilizado com o `IdentityVectorizer` nas pipelines;

### Changed

- Extratores de características e vetorizadores que requerem modelos só realizam o carregamento quando necessário (https://github.com/aiboxlab/nlp/issues/103);
    - Essa modificação busca otimizar a configuração padrão da biblioteca quando são utilizados extratores cuja vetorização em batch utiliza `multiprocesing`;
    - Com essa mudança, o consumo de memória deveria ser reduzido consideravelmente no cenário supracitado (i.e., o processo principal não executa extração e portanto não carrega modelos);
- Todos estimadores agora retornam a seed randômica na propriedade `hyperparameters`;
- Seleção da seed randômica garante que o valor é um inteiro (anteriormente, alguns métodos factory geravam `np.int64`);
- `SimpleExperiment` agora utiliza `n_workers=1` como valor padrão;
    - A funcionalidade de execução em paralelo com `multiprocessing` pode gerar um aumento considerável de memória que só vale a pena quando o `Dataset` é grande o suficiente;
    - Visto que a biblioteca, nos próprios exemplos, trabalha com dados relativamente pequenos essa mudança deve tornar a funcionalidade mais user-friendly (i.e., usuário seleciona quando realiza a ativação);
- `Vectorizer` agora utiliza `n_workers=min(2,os.cpu_count)` como valor padrão (i.e., `n_workers` não definido);
    - Similar a mudança em `SimpleExperiment`;
    - A funcionalidade é ativa por padrão, todavia utilizando uma quantidade de workers menor para evitar estouros de memória;
- Dispositivo padrão para classes que possuem argumento `device` agora é `None`;
    - Anteriormente, algumas classes utilizavam a string `cpu`;
    - Essa modificação permite que usuário possa controlar o dispositivo padrão global através de `torch.set_default_device`;

## [v0.4.1](https://github.com/aiboxlab/nlp/releases/tag/v0.4.1) - 2025-06-05

### Fixed

- Alguns extratores de características baseados em similaridade necessitam do modelo estar carregado no momento da troca do texto de referência através do `setter`;
    - Extratores afetados e corrigidos: `BERTSimilarityExtractor`, `NILCSimilarityExtractor`;
    - Correção agora garante que tais extratores possuem seus modelos carregados no momento da troca de texto;

## [v0.5.0](https://github.com/aiboxlab/nlp/releases/tag/v0.5.0) - 2025-09-28

### Added

- Adiciona novo vetorizador baseado em Multi-View AutoEncoders;

### Changed

- Atualiza gerenciador de pacotes padrão para `uv`;
    - Remoção de antigos `requirements`;
- Pequenas melhorias nos códigos;

## [v0.5.1](https://github.com/aiboxlab/nlp/releases/tag/v0.5.1) - 2025-10-04

### Fixed

- Multi-View AutoEncoder deveria suportar vetorizadores treináveis;
    - Correção adiciona treinamento se o vetorizador for treinável;
