"""Características de similaridade
entre textos baseadas no BERT.
"""

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet


@dataclass(frozen=True)
class BERTSimilarityFeatures(DataclassFeatureSet):
    """Características de similaridade.

    :param bert_similarity_cosine: similaridade cosseno
        entre os embeddings do BERT para dois textos.
    """

    bert_similarity_cosine: float


class BERTSimilarityExtractor(FeatureExtractor):
    """Extrator de similaridade entre textos baseado
    nos Embeddings do BERT.

    :param reference_text: texto de referência para
        calcular similaridade.
    :param bert_model: modelo do BERT a ser adotado. Defaults
        to "neuralmind/bert-base-portuguese-cased".
    :param device: dispositivo a ser utilizado.

    Exemplo de uso em
    :py:class:`~aibox.nlp.core.feature_extractor.FeatureExtractor`
    """

    def __init__(
        self,
        reference_text: str,
        bert_model: SentenceTransformer = None,
        device: str | None = None,
    ) -> None:
        self._model = bert_model
        self._ref_text = reference_text
        self._ref_embdedings = None
        self._device = device

    @property
    def feature_set(self) -> type[BERTSimilarityFeatures]:
        return BERTSimilarityFeatures

    @property
    def reference_text(self) -> str:
        return self._ref_text

    @reference_text.setter
    def reference_text(self, value: str) -> str:
        self._maybe_load_models()
        self._ref_text = value
        self._ref_embdedings = self._model.encode(
            [self._ref_text.lower()], convert_to_tensor=True
        )

    def extract(self, text: str, **kwargs) -> BERTSimilarityFeatures:
        del kwargs

        self._maybe_load_models()
        embeddings = self._model.encode([text.lower()], convert_to_tensor=True)
        similarity = cos_sim(embeddings, self._ref_embdedings).item()
        return BERTSimilarityFeatures(bert_similarity_cosine=similarity)

    def _maybe_load_models(self):
        if self._model is None:
            model_name = "neuralmind/bert-base-portuguese-cased"
            self._model = SentenceTransformer(model_name, device=self._device)

        if self._ref_embdedings is None:
            self._ref_embdedings = self._model.encode(
                [self._ref_text.lower()], convert_to_tensor=True
            )
