"""Testes das métricas
da biblioteca.
"""

import numpy as np
import pytest

from aibox.nlp.factory import get_metric


@pytest.mark.parametrize(
    "metric_clf,metric_config,y_true,y_pred,expected",
    [
        (
            "R2",
            None,
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array(1.0),
        ),
        *[
            (m, None, np.array([1.0]), np.array([1.0]), np.array(0.0))
            for m in ["MAE", "RMSE", "MSE"]
        ],
        *[
            (
                m,
                dict(average=avg),
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                np.array(expected),
            )
            for avg, expected in zip(
                [None, "micro", "macro", "weighted"], [[1.0, 1.0, 1.0], 1.0, 1.0, 1.0]
            )
            for m in ["precision", "recall", "f1"]
        ],
    ],
)
def test_metrics(
    metric_clf: str,
    metric_config: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    expected: np.ndarray,
):
    # Get metric
    if metric_config is None:
        metric_config = dict()
    metric = get_metric(metric_clf, **metric_config)

    # Find value
    value = metric.compute(y_true, y_pred)

    # Guarantee expected is float32
    expected = expected.astype(np.float32)

    # Assertions
    assert value.shape == expected.shape
    assert value.dtype == expected.dtype
    assert np.allclose(value, expected)
