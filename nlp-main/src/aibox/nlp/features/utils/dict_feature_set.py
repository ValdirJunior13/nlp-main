"""Conjunto de características
como um :py:class:`dict`.
"""

from aibox.nlp.core import FeatureSet


class DictFeatureSet(FeatureSet):
    """Implementação de um
    :py:class:`~aibox.nlp.core.feature_extractor.FeatureSet`
    a partir de um dicionário qualquer.

    O conjunto de features é a união das features passadas como
    argumento.

    :param data: dicionário com mapeamento das
        features.
    :param `**kw_features`: features passadas como
        keyword argumets.
    """

    def __init__(self, data: dict[str, float] = None, **kw_features):
        self._d = dict() if data is None else data
        for k, v in kw_features.items():
            assert k not in self._d
            self._d[k] = float(v)

    def as_dict(self) -> dict[str, float]:
        lexical_sorted_dict = dict(sorted(self._d.items(), key=lambda x: x[0]))
        return lexical_sorted_dict
