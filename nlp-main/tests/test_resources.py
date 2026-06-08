"""Teste do pacote
resources.
"""

import pytest

from aibox.nlp import resources


@pytest.mark.parametrize(
    "resource",
    [
        "dictionary/connectives-list.v1",
        "dictionary/verb-conjugation.v1",
    ],
)
def test_resource_get(resource: str):
    p = resources.path(resource)

    # Assert resource exists
    assert p.exists()
    assert len(list(p.iterdir())) > 0
