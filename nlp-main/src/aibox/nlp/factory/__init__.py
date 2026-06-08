"""Interface factory/builder para criação de entidades da biblioteca."""

from .class_registry import (
    available_datasets,
    available_estimators,
    available_extractors,
    available_metrics,
    available_vectorizers,
)
from .dataset import get_dataset
from .estimators import get_estimator
from .feature_extractor import get_extractor
from .metric import get_metric
from .pipeline import get_pipeline, make_pipeline
from .vectorizer import get_vectorizer
