"""Conjunto de características
como um :py:class:`~dataclasses.dataclass`.
"""

import dataclasses

from aibox.nlp.core import FeatureSet


class DataclassFeatureSet(FeatureSet):
    """Implementação de um :py:class:`~aibox.nlp.core.feture_extractor.FeatureSet`
    que supõe que a classe base é um :py:class:`~dataclasses.dataclass` (i.e., possui
    um método :py:meth:`~dataclasses.dataclass.asdict`).
    """

    def as_dict(self) -> dict[str, float]:
        unordered_dict = dataclasses.asdict(self)
        lexical_sorted_dict = dict(sorted(unordered_dict.items(), key=lambda x: x[0]))
        return lexical_sorted_dict
