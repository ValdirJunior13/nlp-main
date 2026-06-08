"""Estimador baseado na arquitetura Transformer."""

import logging
import typing

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from aibox.nlp.core import Estimator
from aibox.nlp.typing import ArrayLike, TextArrayLike

logger = logging.getLogger(__name__)


class TransformerEstimator(Estimator):
    """Estimador genérico baseado na arquitetura
    Transformers utilizando o :py:mod:`huggingface`.

    :param kind: se é uma rede de classificação ou
        regressão.
    :param model_name: nome do modelo base. Defaults
        to "neuralmind/bert-base-portuguese-cased".
    :param max_seq_len: tamanho máximo de uma sentença (
        é realizado truncamento).
    :param epochs: quantidade de épocas de treinamento.
    :param batch_size: tamanho do batch utilizado para treino
        e avaliação.
    :param learning_rate: taxa de aprendizado passada para
        o otimizador.
    :param random_state: seed.
    :param do_lower_case: se a tokenização deve converter
        para caixa baixa.
    :param regression_ensure_bounds: caso rede seja de regressão,
        se os valores preditos devem ser clipados para o máximo
        e mínimo encontrados durante treinamento.
    """

    def __init__(
        self,
        kind: typing.Literal["classifier", "regressor"],
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        max_seq_len: int = 512,
        epochs: int = 2,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        random_state: int | None = None,
        do_lower_case: bool = False,
        regression_ensure_bounds: bool = False,
    ):

        super().__init__(random_state=random_state)
        self._hyperparams = dict(
            trainer=dict(
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                log_level="warning",
                logging_strategy="no",
                save_strategy="no",
                seed=self.random_state,
                output_dir=f"{model_name}-finetuned",
                report_to="none",
            ),
            tokenizer=dict(do_lower_case=do_lower_case),
            model=model_name,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            random_state=self.random_state,
            regression_ensure_bounds=regression_ensure_bounds,
        )

        # Variáveis auxiliares
        self._train_args = TrainingArguments(**self._hyperparams["trainer"])
        self._model = None
        self._tokenizer = None
        self._trainer = None
        self._label2id, self._id2label = None, None
        self._is_classifier = kind == "classifier"
        self._ensure_bounds = regression_ensure_bounds

    def predict(self, X: ArrayLike | TextArrayLike, **kwargs) -> np.ndarray:
        del kwargs

        # Garantindo que `X` são textos
        X = self._maybe_convert_to_list(X)
        if not isinstance(X[0], str):
            logger.warning(
                "`X` is not TextArrayLike (type=%s), forced conversion to `str`. "
                "Results are unpredictable. Ensure `predict` is called with `TextArrayLike` "
                "to avoid errors or undefined behavior.",
                type(X[0]),
            )
            X = [str(x) for x in X]

        # Execução do modelo
        with torch.no_grad():
            outputs = []
            for batch in tqdm(
                self._chunks(X, self._hyperparams["batch_size"]), desc="Batches"
            ):
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self._hyperparams["max_seq_len"],
                )
                inputs = inputs.to(self._model.device)
                outputs.append(self._model(**inputs).logits)

            output = torch.cat(outputs, dim=0)

        if self._is_classifier:
            # Logits
            output = output.argmax(dim=-1)
        elif self._ensure_bounds:
            output = torch.clip(output, min=self._min_y, max=self._max_y)

        output = output.squeeze(dim=-1)
        output = output.cpu().numpy()
        return self._maybe_id2label(output)

    def fit(self, X: ArrayLike | TextArrayLike, y: ArrayLike, **kwargs):
        del kwargs

        # Garantindo que `X` são textos
        X = self._maybe_convert_to_list(X)
        if not isinstance(X[0], str):
            logger.warning(
                "`X` is not TextArrayLike (type=%s), forced conversion to `str`. "
                "Results are unpredictable. Ensure `predict` is called with `TextArrayLike` "
                "to avoid errors or undefined behavior.",
                type(X[0]),
            )
            X = [str(x) for x in X]
        y = np.array(y, dtype=np.int32 if self._is_classifier else np.float32)

        # Configurando para classificação/regressão
        if self._is_classifier:
            labels = np.unique(y).tolist()
            n_labels = len(labels)
            self._label2id = {labels[i]: i for i in range(n_labels)}
            self._id2label = {i: labels[i] for i in range(n_labels)}
        else:
            n_labels = 1
            self._max_y = y.max()
            self._min_y = y.min()

        # Inicializando Trainer e modelos
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._hyperparams["model"],
            num_labels=n_labels,
            problem_type=(
                "regression"
                if not self._is_classifier
                else "single_label_classification"
            ),
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._hyperparams["model"], **self._hyperparams["tokenizer"]
        )

        # Criando dataset
        ds = Dataset.from_dict(dict(text=X, label=self._maybe_label2id(y)))
        ds = ds.map(
            lambda v: self._tokenizer(
                v["text"], truncation=True, max_length=self._hyperparams["max_seq_len"]
            ),
            batched=True,
        )

        # Criando collator
        collator = DataCollatorWithPadding(tokenizer=self._tokenizer)

        # Criando trainer
        trainer = Trainer(
            model=self._model,
            args=self._train_args,
            train_dataset=ds,
            processing_class=self._tokenizer,
            data_collator=collator,
        )

        # Realizando treinamento
        trainer.train()

    def _maybe_label2id(self, v: np.ndarray):
        if self._label2id is None:
            return v

        return np.vectorize(self._label2id.get)(v)

    def _maybe_id2label(self, v: np.ndarray):
        if self._id2label is None:
            return v

        return np.vectorize(self._id2label.get)(v)

    @staticmethod
    def _maybe_convert_to_list(X: TextArrayLike | ArrayLike) -> list:
        if isinstance(X, np.ndarray) or isinstance(X, torch.Tensor):
            return X.tolist()
        return X

    @staticmethod
    def _chunks(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]

    @property
    def hyperparameters(self) -> dict:
        return self._hyperparams

    @property
    def params(self) -> dict:
        return self._hyperparams
