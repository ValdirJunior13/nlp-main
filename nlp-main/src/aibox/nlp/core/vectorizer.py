"""Interface para vetorizadores."""

import logging
import multiprocessing
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import tqdm

from aibox.nlp.typing import ArrayLike, TextArrayLike

logger = logging.getLogger(__name__)


class Vectorizer(ABC):
    """Interface para vetorizadores.

    Um vetorizador consegue converter textos (str)
    para uma representação numérica (vetor e/ou tensor).

    .. code-block:: python

        from aibox.nlp.core import Vectorizer

        # Exemplo de uso para classes concretas
        vectorizer = Vectorizer()
        text = "Esse é um texto de exemplo."

        # Realizando a vetorização
        vectorizer.vectorize(text, "numpy")
    """

    def vectorize(
        self,
        text: str | TextArrayLike,
        vector_type: str = "numpy",
        device: str | None = None,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:
        """Método para vetorização de textos. A vetorização de múltiplos
        textos é realizada de forma paralela sempre que possível.

        Aceita os campos `n_workers` (default=`min(4, cpu_count)`)
        e `show_bar` (default=`true`) quando array-like de string.
        Demais parâmetros são passados para :py:meth:`_vectorize`.

        `n_workers` é utilizado quando a implementação utiliza
        :py:mod:`multiprocessing`. Caso `n_workers <= 1`, um `for`.

        :param text: texto ou textos de entrada.
        :param vector_type: tipo do vetor de saída ('numpy
            ou 'torch').
        :type vector_type: str, opcional
        :param device: dispositivo para armazenamento do tensor Torch. Padrão é
            CPU.
        :type device: str, opcional.
        :param `**kwargs`: parâmetros extras que podem ser utilizados
            por alguns vetorizadores para controlar o processo de
            vetorização.

        :return: representação numérica do texto.
        """
        # Obtendo representação vetorial
        vectorize_fn = (
            self._vectorize if isinstance(text, str) else self._batch_vectorize
        )
        text_vector = vectorize_fn(text, **kwargs)
        is_np = isinstance(text_vector, np.ndarray)
        is_torch = isinstance(text_vector, torch.Tensor)

        if not is_np and not is_torch:
            # Por padrão, convertemos para NumPy
            text_vector = np.array(text_vector, dtype=np.float32)

        # Caso seja necessário um tensor, convertemos
        if (vector_type == "torch") and not is_torch:
            text_vector = torch.from_numpy(text_vector)

            if device is not None:
                text_vector = text_vector.to(device)

        # Caso seja necessário uma ndarray, convertemos
        if (vector_type == "numpy") and is_torch:
            text_vector = text_vector.numpy()

        return text_vector

    def _batch_vectorize(self, texts: TextArrayLike, **kwargs) -> ArrayLike:
        """Método privado para vetorização de múltiplos
        textos.

        Aceita os campos `n_workers` (default=`min(4, cpu_count)`)
        e `show_bar` (default=`true`) quando array-like de string.
        Demais parâmetros são passados para :py:meth:`_vectorize`.

        `n_workers` é utilizado quando a implementação utiliza
        :py:mod:`multiprocessing`. Caso `n_workers <= 1`, um
        `for` é utilizado sem :py:mod:`multiprocessing`.

        :param texts: textos a serem vetorizados.
        :param `**kwargs`: parâmetros extras.

        :return: representação numérica dos textos.
        """
        # Convertendo texsts para lista
        if not isinstance(texts, list):
            texts = texts.tolist()

        # Obtendo configurações do multiprocessing
        mp = multiprocessing.get_context("spawn")
        n_texts = len(texts)
        n_workers = kwargs.pop("n_workers", min(2, os.cpu_count()))

        # Garantindo que n_workers <= n_texts
        n_workers = min(n_workers, n_texts)

        # Obtendo configurações do tqdm
        show_bar = kwargs.pop("show_bar", True)

        # Casos degenerados
        if n_texts == 1:
            return [self._vectorize(texts[0], **kwargs)]

        # Caso apenas 1 worker ou nenhum,
        #   executar single-process
        if n_workers <= 1:
            logger.info(
                "Single worker detected. Running in current process for `%s`.",
                self.__class__.__name__,
            )
            out = []
            for text in tqdm.tqdm(
                texts, desc="Vetorização de textos", disable=not show_bar
            ):
                out.append(self._vectorize(text, **kwargs))
            return out

        # 1 texto por worker
        if n_texts == n_workers:
            logger.warning(
                "1 text per worker detected (texts=%d, workers=%d). "
                "Expect degraded performance (either reduce number of "
                "workers or use a single worker).",
                n_texts,
                n_workers,
            )

        # n_workers < n_texts, sempre
        chunk_size = n_texts // n_workers
        logger.info(
            "Started %d workers with chunk_size=%d (texts=%d) for `%s`.",
            n_workers,
            chunk_size,
            n_texts,
            self.__class__.__name__,
        )
        with mp.Pool(n_workers) as pool:
            out = list(
                tqdm.tqdm(
                    pool.imap(
                        self._vectorize_dict_kwarg,
                        zip(texts, [kwargs] * n_texts),
                        chunksize=chunk_size,
                    ),
                    desc="Vetorização de textos",
                    total=n_texts,
                    disable=not show_bar,
                )
            )

        return out

    @abstractmethod
    def _vectorize(self, text: str, **kwargs) -> ArrayLike:
        """Método privado para vetorização do texto
        e retorno de um array-like qualquer (e.g., lista,
        tupla, ndarray, torch.Tensor, etc).

        :param text: texto que deve ser vetorizado.
        :param `**kwargs`: parâmetros extras que podem ser utilizados
            por alguns vetorizadores para controlar o processo de
            vetorização.

        :return: representação numérica do texto.
        """

    def _vectorize_dict_kwarg(self, v: tuple[str, dict]) -> ArrayLike:
        text, kwargs = v
        return self._vectorize(text, **kwargs)


class TrainableVectorizer(Vectorizer):
    """Representação de um vetorizador
    treinável (e.g., TF-IDF, BERT).

    Esse é um vetorizador que requer treinamento
    antes de ser utilizável diretamente. Apesar de
    possuir um método :py:meth:`fit`, não deve ser
    confundido com :py:class:`~aibox.nlp.core.estimator.Estimator`.

    O comportamento do método :py:meth:`vectorize` não é
    definido caso o vetorizador não tenha sido treinado.

    .. code-block:: python

        from aibox.nlp.core import TrainableVectorizer

        # Exemplo de uso para classes concretas
        vectorizer = TrainableVectorizer()
        train = ["Texto de treinamento 1.", "Texto de treinamento 2."]
        text = "Esse é um texto de exemplo."

        # Treinamento da classe
        vectorizer.fit(train)

        # Realizando a vetorização
        vectorizer.vectorize(text, "numpy")
    """

    def fit(self, X: TextArrayLike, y: None = None, **kwargs) -> None:
        """Método para treinamento do vetorizador. O valor de `y`
        não é utilizado, só é mantido por consistência da interface
        `fit(X, y)`.

        :param X: array-like de strings com formato (n_samples,).
        :param y: desconsiderado. Existe para compatibilidade com
            outras classes que implementam o método com mesmo nome.
        :param `**kwargs`: configurações extras que alguns vetorizadores
            treináveis podem utilizar.
        """
