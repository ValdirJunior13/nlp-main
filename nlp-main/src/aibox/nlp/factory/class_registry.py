"""Implementação de um Class Registry,
permitindo que classes sejam representadas
como uma string.
"""

import importlib
import json

try:
    from importlib.resources import files
except ImportError:
    # Python < 3.9 doesn't have the
    #   same files(...) method.
    # Instead, we use the one provided
    # by the importlib_resources library
    from importlib_resources import files


class Registry:
    """Registro de identificadores de entidades.

    Essa é uma classe utilitária utilizada para os
    métodos factory. Esse registro contém uma lista
    de todos as entidades da biblioteca.

    .. literalinclude:: /registry.json
        :language: json
        :caption: Registro de entidades da biblioteca.
    """

    def __init__(self) -> None:
        p = files("aibox.nlp.factory").joinpath("registry.json")
        with p.open("r", encoding="utf-8") as f:
            self._reg = json.load(f)

        self._reg["global"] = dict()
        global_prefix = "aibox.nlp."
        for key, prefix in zip(
            ["features_br", "vectorizers", "estimators", "metrics", "datasets"],
            [
                "features.portuguese.{0}",
                "vectorizers.{0}",
                "estimators.{0}",
                "metrics.{0}",
                "data.datasets.{0}",
            ],
        ):
            # k é o ID da classe
            # v é o caminho relativo
            for k, v in self._reg[key].items():
                # full_v contém o caminho completo desde
                #   a raiz do pacote até a classe.
                full_v = global_prefix + prefix.format(v)

                # Garantindo que essa chave é única
                assert k not in self._reg["global"]

                # Adicionamos esse novo caminho na chave global
                #   do registro.
                self._reg["global"][k] = full_v

    @property
    def estimators(self) -> dict[str, str]:
        """Mapeamento de identificador de estimadores
        para suas respectivas classes.
        """
        return self.get_registry_for("estimators")

    @property
    def datasets(self) -> dict[str, str]:
        """Mapeamento de identificador de datasets
        para suas respectivas classes.
        """
        return self.get_registry_for("datasets")

    @property
    def features_br(self) -> dict[str, str]:
        """Mapeamento de identificador de extratores
        de características para o português brasileiro
        para suas respectivas classes.
        """
        return self.get_registry_for("features_br")

    @property
    def metrics(self) -> dict[str, str]:
        """Mapeamento de identificador de métricas
        para suas respectivas classes.
        """
        return self.get_registry_for("metrics")

    @property
    def vectorizers(self) -> dict[str, str]:
        """Mapeamento de identificador de vetorizadores
        para suas respectivas classes.
        """
        return self.get_registry_for("vectorizers")

    def get_registry_for(self, kind: str) -> dict[str, str]:
        """Retorna as entradas do registro para
        um dado tipo.

        :param kind: tipo das entradas.

        :return: registro para esse tipo.
        """
        return self._reg[kind].copy()

    def get_path(self, identifier: str) -> str:
        """Dado um identificador para uma classe,
        retorna o caminho até essa classe seguindo
        o padrão de import de Python (e.g.,
        package_a.package_b.module_c.ClassD)

        :param identifier: identificador.

        :return: caminho até a classe.
        """
        return self._reg["global"][identifier]


#: Singleton do registro de entidades.
registry = Registry()


def get_class(key: str) -> type:
    """Retorna a classe do identificador
    recebido como argumento.

    :param key: identificador.

    :return: classe associada ao identificador.
    """
    # Obtendo nome do pacote e classe
    class_path = registry.get_path(key)
    splits = class_path.rsplit(".", 1)
    module_name = splits[0]
    class_name = splits[1]

    # Carregando módulo
    module = importlib.import_module(module_name)

    # Obtendo classe dentro desse módulo
    cls = getattr(module, class_name)

    return cls


def available_datasets() -> list[str]:
    """Retorna uma lista dos datasets
    disponíveis.

    :return: datasets disponíveis.
    """
    return list(registry.datasets)


def available_extractors() -> list[str]:
    """Retorna uma lista dos extratores de
    características disponíveis.

    :return: extratores disponíveis.
    """
    return list(registry.features_br)


def available_metrics() -> list[str]:
    """Retorna uma lista das métricas
    disponíveis.

    :return: métricas disponíveis.
    """
    return list(registry.metrics)


def available_vectorizers() -> list[str]:
    """Retorna uma lista dos vetorizadores
    disponíveis.

    :return: vetorizadores disponíveis.
    """
    return list(registry.vectorizers)


def available_estimators() -> list[str]:
    """Retorna uma lista dos estimadores
    disponíveis.

    :return: estimadores disponíveis.
    """
    return list(registry.estimators)
