"""Cache para múltiplos extratores de características."""

from aibox.nlp.cache.features import FeatureCache
from aibox.nlp.core import FeatureSet
from aibox.nlp.features.utils.dict_feature_set import DictFeatureSet


class MixedFeatureCache(FeatureCache):
    """Cache características com diferentes
    schemas.

    :param target_features: conjunto de características
        target (i.e., retornadas no :py:meth:`get`).
    :param initial_cache: cache inicial. Mapeia textos
        para um conjunto de características.
    :param max_limit: limite de entradas no cache. Se <= 0,
        nenhum limite é aplicado.

    Essa classe permite que múltiplos extratores de características
    compartilhem de um mesmo cache/memória.

    É crucial que quando um extrator deseje utilizar a memória,
    o atributo :py:attr:`target_features` seja atualizada com o
    conjunto de características esperado.
    """

    def __init__(
        self,
        target_features: set[str],
        initial_cache: dict[str, dict[str, float]] = None,
        max_limit: int = 0,
    ):
        if initial_cache is None:
            initial_cache = dict()

        self._features = target_features
        self._cache: dict[str, dict[str, float]] = initial_cache
        self._max = max_limit

    @property
    def target_features(self) -> set[str]:
        """Conjunto de características esperado
        ao se utilizar o método :py:meth:`get`.
        """
        return self._features.copy()

    @target_features.setter
    def target_features(self, value: set[str]) -> None:
        self._features = value

    def get(self, text: str) -> FeatureSet | None:
        if text not in self._cache:
            # Texto ainda não foi cacheado
            return None

        entry = self._cache[text]
        current_features = set(entry.keys())

        if not self._features.issubset(current_features):
            # Texto foi cacheado mas não possui
            #   todas as features.
            return None

        return DictFeatureSet({k: v for k, v in entry.items() if k in self._features})

    def save(self, text: str, data: FeatureSet, overwrite: bool = False) -> bool:
        data_dict = data.as_dict()

        if text not in self._cache:
            self._cache[text] = data_dict
        else:
            # Caso não se deva sobrescrever e o cache possuir alguma das features
            #   presente em `data`, retornamos false
            if not overwrite and any(k in self._cache for k in data_dict):
                return False

            self._cache[text].update(data_dict)

        return True

    def as_dict(self) -> dict[str, FeatureSet]:
        return {k: DictFeatureSet(v) for k, v in self._cache.items()}

    def _prune_to_limit(self):
        mem_size = len(self._cache)

        if self._max <= 0 or mem_size <= self._max:
            return

        keys = list(self._cache)
        diff = mem_size - self._max

        for k in keys[0:diff]:
            del self._cache[k]
