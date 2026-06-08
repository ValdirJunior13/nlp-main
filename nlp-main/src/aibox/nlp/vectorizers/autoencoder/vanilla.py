"""Autoencoder single
e multi-view padrão.
"""

from typing import Literal, Iterable

import torch
from torch import nn

from aibox.nlp.core import Vectorizer

from .base import AEVectorizerABC


class MultiViewEncoder(nn.Module):
    """Multi-view encoder com suporte
    para n_views >= 1.

    Cada view é processada por uma rede
    própria. As saídas das redes de cada
    view são concatenadas e passadas como
    entrada para uma camada totalmente
    conectada que representa o espaço
    latente.
    """

    def __init__(
        self,
        views_networks: list[nn.Module] | nn.Module,
        views_networks_output_sizes: list[int] | int,
        latent_size: int,
        latent_activation: nn.Module | None = None,
    ):
        """Construtor.

        Args:
            views_networks: redes para cada view.
            views_networks_output_sizes: tamanho de
                saída de cada view.
            latent_size: tamanho do espaço latente.
            latent_activation: função de ativação
                do espaço latente.
        """
        if isinstance(views_networks, nn.Module):
            views_networks = [views_networks]

        if isinstance(views_networks_output_sizes, int):
            views_networks_output_sizes = [views_networks_output_sizes]

        if len(views_networks) != len(views_networks_output_sizes):
            raise ValueError("Views networks and output sizes don't match.")

        latent = [
            nn.Linear(
                in_features=sum(views_networks_output_sizes), out_features=latent_size
            )
        ]

        if latent_activation:
            latent.append(latent_activation)

        super().__init__()
        self.networks = nn.ModuleList(views_networks)
        self.latent = nn.Sequential(*latent)

    def forward(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        outputs = [net(xv) for xv, net in zip(x, self.networks)]
        view_concat = torch.concat(outputs, dim=-1)
        return self.latent(view_concat)


class MultiViewDecoder(nn.Module):
    """Multi-view decoder com suporte
    para n_views >= 1.

    Utiliza a saída do espaço latente
    como entrada para várias redes que
    reconstroem cada view.
    """

    def __init__(
        self,
        views_networks: list[nn.Module] | nn.Module,
        latent_size: int,
    ):
        """Construtor.

        Args:
            views_networks: redes para cada view.
            views_networks_output_sizes: tamanho de
                saída de cada view.
            latent_size: tamanho do espaço latente.
            latent_activation: função de ativação
                do espaço latente.
        """
        if isinstance(views_networks, nn.Module):
            views_networks = [views_networks]

        super().__init__()
        self.networks = nn.ModuleList(views_networks)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor]:
        return tuple(net(latent) for net in self.networks)


class AEVectorizer(AEVectorizerABC):

    def __init__(
        self,
        *vectorizers: Vectorizer,
        encoder_network_hidden_sizes: list[int],
        decoder_network_hidden_sizes: list[int],
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
        super().__init__(
            *vectorizers,
            latent_size=latent_size,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            optim_params=optim_params,
            optimizer=optimizer,
            train_batch_size=train_batch_size,
            device=device,
            dtype=dtype,
        )

        self._encoder_network_sizes = encoder_network_hidden_sizes
        self._decoder_network_sizes = decoder_network_hidden_sizes
        self._encoder = None
        self._decoder = None

    @property
    def encoder(self) -> nn.Module:
        if self._encoder is None:
            raise ValueError("Model hasn't been built yet.")

        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        if self._decoder is None:
            raise ValueError("Model hasn't been built yet.")

        return self._decoder

    def _build_model(self, X: list[torch.Tensor]):
        n_views = len(X)

        def _views_networks(
            sizes: list[int], latent_size: int = None, out_sizes: list[int] = None
        ) -> list[nn.Module]:
            networks = []
            for view in range(n_views):
                layers = []

                in_features = latent_size if latent_size else X[view].shape[-1]
                for size in sizes:
                    layers.append(nn.Linear(in_features, size))
                    layers.append(nn.ReLU())
                    in_features = size

                if out_sizes:
                    layers.append(nn.Linear(in_features, out_sizes[view]))

                networks.append(nn.Sequential(*layers))

            return networks

        self._encoder = MultiViewEncoder(
            views_networks=_views_networks(self._encoder_network_sizes),
            views_networks_output_sizes=[self._encoder_network_sizes[-1]] * n_views,
            latent_size=self.latent_size,
        )

        out_sizes = [x.shape[-1] for x in X]
        self._decoder = MultiViewDecoder(
            views_networks=_views_networks(
                self._decoder_network_sizes, self.latent_size, out_sizes
            ),
            latent_size=self.latent_size,
        )
