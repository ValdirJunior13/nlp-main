"""Estimador baseado na arquitetura MLP."""

import typing

import torch
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch import nn

from aibox.nlp.core import Estimator
from aibox.nlp.typing import ArrayLike


class MLPEstimator(Estimator):
    """Estimador MLP genérico.

    :param hidden_layers: número de features para
        cada camada escondida da rede.
    :param kind: se é uma rede de classificação
        ou regressão.
    :param random_state: seed.
    :param activation_fn: função de ativação para as camadas
        escondidas. Defaults to :py:class:`torch.nn.ReLU`.
    :param output_activation: função de ativação para a camada
        de saída. Defaults to :py:class:`torch.nn.Linear`.
    :param epochs: quantidade de épocas a ser utilizadas
        no momento do treinamento.
    :param dropout_prob: probabilidade de dropout. Se não for None,
        introduz uma camada de dropout com a aquela probabilidade após
        as camadas escondidas. É possível controlar quais camadas devem
        possuir dropout passando uma lista.
    :param learning_rate: taxa de aprendizado do otimizador.
    :param optim_params: parâmetros extras a serem passados
        para o otimizador além da taxa de aprendizado.
    :param optim: otimizador a ser utilizado.
    :param regression_ensure_bounds: se os limites devem
        ser respeitados em redes de regressão.
    :param train_batch_size: tamanho do batch para treinamento.
    :param device: dispositivo que o modelo deve ser armazenado.
    :param dtype: dtype esperado pela rede. Conversões são realizadas
        automaticamente.

    O modelo é construído na primeira chamada ao  método :py:meth:`fit`.
    A arquitetura consiste em:

        1. ``len(hidden_layers)`` camadas escondidas com ativação ``activation_fn`` e
        | camada de dropout se ``dropout_prob is not None``;
        2. Uma camada totalmente conectada com ativação linear;

    A saída do modelo depende se a rede é de classificação (logits)
    ou regressão (valor numérico).
    """

    _OPTIMIZER: typing.ClassVar[dict] = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "rmsprop": optim.RMSprop,
        "adagrad": optim.Adagrad,
        "sgd": optim.SGD,
    }

    def __init__(
        self,
        hidden_layers: list[int],
        kind: typing.Literal["classifier", "regressor"],
        random_state: int | None = None,
        activation_fn: nn.Module = nn.ReLU,
        dropout_prob: float | list[float | None] | None = 0.05,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        optim_params: dict | None = None,
        optim: typing.Literal["adam", "adamw", "rmsprop", "adagrad", "sgd"] = "adamw",
        regression_ensure_bounds: bool = False,
        train_batch_size: int = 256,
        device: str = None,
        dtype=torch.float32,
    ):
        """Construtor."""
        # Atualmente, a implementação não utiliza
        #   o Random State.
        super().__init__(random_state=random_state)

        if isinstance(dropout_prob, list):
            assert len(dropout_prob) == len(hidden_layers)
        else:
            dropout_prob = [dropout_prob] * len(hidden_layers)

        if optim_params is None:
            optim_params = dict()

        # Armazenando dispositivo
        self._device = device
        self._dtype = dtype

        # Armazenando hiperparâmetros
        self._hyperparams = {
            "hidden_layers": hidden_layers,
            "dropout_prob": dropout_prob,
            "activation_fn": activation_fn.__class__.__name__.lower(),
            "output_activation": "linear",
            "train_batch_size": train_batch_size,
            "epochs": epochs,
            "optimizer": optim,
            "learning_rate": learning_rate,
            "regression_ensure_bounds": regression_ensure_bounds,
        }

        self._hyperparams.update({f"optimizer_{k}": v for k, v in optim_params.items()})

        # Inicializando o otimizador
        self._optim = self._OPTIMIZER[optim]
        self._lr = learning_rate
        self._epochs = epochs
        self._optim_params = optim_params

        # Inicializando o critério de otimização
        if kind == "classifier":
            self._criterion = nn.CrossEntropyLoss()
        else:
            self._criterion = nn.MSELoss()

        # Variáveis relacionadas com o modelo
        self._model = None
        self._encoder = None
        self._activation_fn = activation_fn
        self._dropout_prob = dropout_prob
        self._kind = kind
        self._batch_size = train_batch_size
        self._ensure_bounds = regression_ensure_bounds
        self._max_y, self._min_y = None, None

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> None:
        del kwargs

        # Convertendo entrada para tensor
        X = torch.tensor(X, dtype=self._dtype)
        assert len(X.shape) == 2

        # Talvez converter "y" para um intervalo [0, N]
        y = self._maybe_convert_label(y, direction="from")
        y = torch.tensor(
            y,
            device=self._device,
            dtype=self._dtype if self._kind == "regressor" else torch.int64,
        )
        self._max_y = y.max()
        self._min_y = y.min()

        # Se regressão, atualizar shape de y com 1 dimensão
        #   interna: (batch_size,) -> (batch_size, 1)
        if self._kind == "regressor":
            y = torch.unsqueeze(y, dim=-1)

        # Criando o modelo
        self._create_model(input_size=X.size(dim=1), y=y)

        # Criando dataset
        ds = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=self._batch_size)
        del X, y

        # Criando otimizador
        self._optim = self._optim(
            self._model.parameters(), lr=self._lr, **self._optim_params
        )

        # Treinamento do modelo
        for _ in range(self._epochs):
            for data in loader:
                inputs, targets = data

                # Resetando o acúmulo de gradientes
                self._optim.zero_grad()

                # Obtendo a saída do modelo
                outputs = self._model(inputs)

                # Calculando o forward e backward-pass
                #   da função objetiva
                loss = self._criterion(outputs, targets)
                loss.backward()

                # Atualizar pesos com base na saída
                self._optim.step()

                # Removendo do escopo do for
                del outputs, loss

    def predict(self, X: ArrayLike, **kwargs) -> None:
        del kwargs

        # Convertendo entrada para tensor
        X = torch.tensor(X, dtype=self._dtype)

        # Obtendo saídas do modelo
        with torch.no_grad():
            preds = self._model(X)

        # Possivelmente convertendo
        #   para classificação/regressão
        preds = self._maybe_update_preds(preds)

        # Convertendo para NumPy
        preds = preds.cpu().numpy()

        # Possivelmente converter para os labels
        #   vistos durante treino
        preds = self._maybe_convert_label(preds, direction="to")

        # Retornar predições
        return preds

    def _maybe_update_preds(self, preds: torch.Tensor) -> torch.Tensor:
        if self._kind == "classifier":
            # Se for classificação, devemos obter
            #   o máximo da saída linear.
            _, preds = torch.max(preds, dim=1)
        elif self._ensure_bounds:
            # Do contrário, se for regressão
            #   e devemos garantir limites
            #   realizamos o clip.
            preds = torch.squeeze(preds, dim=1)
            preds = torch.clip(preds, min=self._min_y, max=self._max_y)

        # Talvez possua uma dimensão a mais?
        if len(preds.shape) > 1:
            preds = preds.squeeze(dim=-1)

        return preds

    def _create_model(self, input_size, y):
        if self._kind == "classifier":
            out = torch.unique(y, dim=0).size(dim=0)
        else:
            out = 1

        hidden_layers = self._hyperparams["hidden_layers"]
        dropout = self._hyperparams["dropout_prob"]
        self._model = nn.Sequential(
            *[
                layer
                for idx in range(len(hidden_layers))
                for layer in [
                    nn.Linear(
                        input_size if idx == 0 else hidden_layers[idx - 1],
                        hidden_layers[idx],
                        dtype=self._dtype,
                    )
                ]
                + ([nn.Dropout(dropout[idx])] if dropout[idx] is not None else [])
                + [self._activation_fn()]
            ],
            nn.Linear(hidden_layers[-1], out, dtype=self._dtype),
        )

    def _maybe_convert_label(self, y: ArrayLike, direction: str) -> ArrayLike:
        if self._kind == "regressor":
            # Não precisamos realizar conversão
            #   se for regressão.
            return y

        if direction == "from":
            self._encoder = LabelEncoder()
            self._encoder.fit(y)
            return self._encoder.transform(y)

        return self._encoder.inverse_transform(y)

    @property
    def hyperparameters(self) -> dict:
        return dict(**self._hyperparams, random_state=self.random_state)

    @property
    def params(self) -> dict:
        return self._hyperparams
