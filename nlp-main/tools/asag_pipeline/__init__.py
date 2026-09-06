"""Pipeline reutilizável para os experimentos do TCC de ASAG."""

from .data import DatasetBundle, DatasetSpec, build_datasets

__all__ = ["DatasetBundle", "DatasetSpec", "build_datasets"]
