# Como contribuir à `aibox-nlp` 👐

> [!NOTE]
> Agradecemos pelo interesse em melhorar a biblioteca! Pedimos que leia atentamente as próximas seções sobre o processo de contribuição.

## Relato de Erros/Bugs

1. Garanta que o bug ainda não foi reportado procurando nas [Issues](https://github.com/aiboxlab/nlp/issues) do GitHub.
2. Caso você não tenha encontrado nenhum relato, [abra uma nova _issue_](https://github.com/aiboxlab/nlp/issues/new) utilizando o template de _bug report_. Não se esqueça de adicionar detalhes sobre seu sistema operacional, ambiente de execução e versões.
3. Se possível, tente realizar reproduzir o mesmo bug em um _ambiente de execução limpo_ (i.e., inicializar um novo ambiente virtual e instalar a biblioteca).

##  Tem alguma dúvida sobre a biblioteca que não é respondida na [Wiki](https://github.com/aiboxlab/nlp/wiki)?

- Abra [_issue_](https://github.com/aiboxlab/nlp/issues/new) com as tags `help wanted` e/ou `question`!
- A Wiki e exemplos da biblioteca costumam ser atualizados sempre que alguma nova funcionalidade é adicionada.
- Também é pertinente checar o [`CHANGELOG`](https://github.com/aiboxlab/nlp/blob/main/CHANGELOG.md) caso esteja buscando pelo suporte (ou remoção de suporte) à alguma funcionalidade.

## Você escreveu um _patch_ que corrige um bug/erro?

- Abra um Pull Request com o patch.
- Garanta que a descrição do Pull Request é concissa e descreve todas as modificações realizadas bem como o erro que tais mudanças resolvem.
- Adicione uma referência para quaisquer _issues_ relacionadas (se aplicável).
- Garanta que as modificações não removeram suporte para versões antigas do Python;
  - O script [`tools/ensure_compatibility.py`](https://github.com/aiboxlab/nlp/blob/main/tools/ensure_compatibility.py) realiza checagens simples executando todos os testes e exemplos da biblioteca em ambientes de execução limpos.
  - É comum que mudanças relacionadas ao sistema de _type hinting_ ou sintaxe  (e.g., `match`) sejam incompatíveis entre certas versões do Python.

## Você gostaria de uma nova funcionalidade ou alteração de uma existente?

- Sugira essa funcionalidade/mudança nas [Issues](https://github.com/aiboxlab/nlp/issues) do GitHub utilizando o template adequado.
- Se você gostaria de trabalhar nessa funcionalidade, informe na Issue (e.g., "Posso trabalhar nessa funcionalidade").
- Aguarde um _feedback_ da equipe de manutenção da biblioteca.
  - A equipe irá definir se essa é uma funcionalidade que pode ser suportada pela biblioteca ou não;
  - Caso você tenha interesse em trabalhar nessa funcionalidade, sinta-se à vontade para inicar os trabalhos caso a sugestão seja acatada pela equipe;

## Qual workflow adotado pela biblioteca?

- A biblioteca utiliza o GitHub Workflow com branches de candidatos a release;
  - Isso quer dizer que a branch `main` busca ser sempre estável e com a versão mais atualizada da biblioteca (comumento chamada de `nightly` ou `git`);
  - Além disso, sempre que a `main` evoluiu suficiente para se tornar uma release, uma nova branch de candidato a release é criada;
- Quando existir necessidade de um `hotfix` que não possa esperar a próxima release, uma nova branch é criada do commit da última release e nela são introduzidas as correções;
  - Uma vez que o `hotfix` seja implementado e testado, uma nova release é gerada diretamente dessa branch;
  - Posteriormente, um rebase é realizado na `main` com as modificações da última release;

---
Obrigado! :heart: :heart: :heart:

_Equipe AiBox Lab_
