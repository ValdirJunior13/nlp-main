"""Pipeline de classificação utilizando o
CohMetrix com um ensemble de Extremely
Randomized Trees.
"""

from aibox.nlp.core.pipeline import Pipeline
from aibox.nlp.estimators.classification.extra_trees_classifier import (
    ExtraTreesClassifier,
)
from aibox.nlp.features.portuguese.cohmetrix import CohMetrixExtractor


class CohMetrixExtraTreesClassification(Pipeline):
    """Pipeline com características do CohMetrix e
    :py:class:`~aibox.nlp.estimators.classification.extra_trees_classifier.ExtraTreesClassifier`.

    :param random_state: seed.
    :param etree_config: configuração pro classificador.
    """

    def __init__(self, random_state: int | None = None, etree_config: dict = None):
        if etree_config is None:
            etree_config = dict()

        super().__init__(
            vectorizer=CohMetrixExtractor(),
            estimator=ExtraTreesClassifier(**etree_config, random_state=random_state),
            name=f"cohmetrix_etree_clf_seed{random_state}",
        )
