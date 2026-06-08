"""Classes abstratas
para vetorizadores baseados
em Autoencoders.
"""

import random
from abc import abstractmethod
from collections import OrderedDict
from functools import cached_property
from typing import ClassVar, Literal

import torch
from torch import nn, optim

from aibox.nlp.core import TrainableVectorizer, Vectorizer
from aibox.nlp.typing import ArrayLike, TextArrayLike


class AEVectorizerABC(TrainableVectorizer):
    """Classe base para vetorizadores baseados
    em Autoencoders.

    Vetorizadores baseados em Autoencoders aprendem
    novas características a partir de outras
    já existentes.
    """

    _OPTIMIZER: ClassVar[dict] = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "rmsprop": optim.RMSprop,
        "adagrad": optim.Adagrad,
        "sgd": optim.SGD,
    }

    def __init__(
        self,
        *vectorizers: Vectorizer,
        latent_size: int = 50,
        random_state: int = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        optim_params: dict = dict(),
        optimizer: Literal["adam", "adamw", "rmsprop", "adagrad", "sgd"] = "adamw",
        train_batch_size: int = 256,
        device: str = None,
        dtype=torch.float32,
    ):
        for v in vectorizers:
            if not isinstance(v, Vectorizer):
                raise ValueError(f"Only Vectorizers are supported: {type(v)}")

        if random_state is None:
            random_state = random.randint(0, 99999)

        self._vectorizers = list(vectorizers)
        self._latent_size = latent_size
        self._optim = self._OPTIMIZER[optimizer]
        self._optim_params = optim_params
        self._epochs = epochs
        self._seed = random_state
        self._batch_size = train_batch_size
        self._lr = learning_rate
        self._hyperparams = {
            "train_batch_size": train_batch_size,
            "epochs": epochs,
            "optimizer": optim,
            "learning_rate": learning_rate,
        }
        self._criterion = nn.MSELoss()
        self._device = device
        self._dtype = dtype

    @property
    def random_state(self) -> int:
        return self._seed

    @property
    def epochs(self) -> int:
        return self._epochs

    @property
    def latent_size(self) -> int:
        return self._latent_size

    @property
    def vectorizers(self) -> list[Vectorizer]:
        return self._vectorizers

    @property
    def hyperparameters(self) -> dict:
        return dict(**self._hyperparams, random_state=self.random_state)

    @cached_property
    def model(self) -> nn.Module:
        return nn.Sequential(
            OrderedDict([("encoder", self.encoder), ("decoder", self.decoder)])
        )

    @property
    @abstractmethod
    def encoder(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def decoder(self) -> nn.Module:
        pass

    def fit(self, X: TextArrayLike, y: ArrayLike = None, **kwargs) -> None:
        del y

        # Convertendo entrada para tensor
        X = self._get_input_vectors(X, is_train=True, **kwargs)

        # Constrói o modelo dinamicamente
        self._build_model(X)

        # Criando dataset
        ds = torch.utils.data.TensorDataset(*X)
        loader = torch.utils.data.DataLoader(ds, batch_size=self._batch_size)
        del X

        # Criando otimizador
        self._optim = self._optim(
            self.model.parameters(), lr=self._lr, **self._optim_params
        )

        # Treinamento do modelo
        for _ in range(self._epochs):
            for data in loader:
                self._training_step(*data)

    def _training_step(self, *inputs: torch.Tensor):
        # Resetando o acúmulo de gradientes
        self._optim.zero_grad()

        # Obtendo a saída do modelo
        outputs = self.model(inputs)

        # Calculando o forward e backward-pass
        #   da função objetiva
        losses = [self._criterion(i, o) for i, o in zip(inputs, outputs)]
        loss = torch.stack(losses).mean()
        loss.backward()

        # Atualizar pesos com base na saída
        self._optim.step()

    def _get_input_vectors(
        self, X: TextArrayLike, is_train: bool = False, **kwargs
    ) -> list[torch.Tensor]:
        if is_train:
            for v in self.vectorizers:
                if isinstance(v, TrainableVectorizer):
                    v.fit(X)

        return [
            v.vectorize(X, vector_type="torch", device=self._device, **kwargs).to(
                dtype=self._dtype
            )
            for v in self.vectorizers
        ]

    def _vectorize(self, text: str, **kwargs):
        return self._batch_vectorize([text], **kwargs)[0]

    def _batch_vectorize(self, texts: TextArrayLike, **kwargs):
        with torch.no_grad():
            return self.encoder(self._get_input_vectors(texts, **kwargs))

    @abstractmethod
    def _build_model(self, X: list[torch.Tensor]):
        pass
