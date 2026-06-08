"""Imports e instanciação
de objetos que precisam de
patch.
"""

from aibox.nlp.lazy_loading import lazy_import


def get_cogroo():
    import pkgutil
    from pathlib import Path

    cogroo4py = lazy_import("cogroo4py.cogroo")
    import jpype

    # Inicializar JVM com configurações esperadas
    if not jpype.isJVMStarted():
        cogroo_root = Path(pkgutil.get_loader("cogroo4py.jpype_config").path).parent
        jpype.startJVM(
            jpype.getDefaultJVMPath(),
            "--enable-native-access=ALL-UNNAMED",
            "-Dorg.slf4j.simpleLogger.defaultLogLevel=off",
            "-Dlog4j.rootLogger=OFF",
            classpath=str(cogroo_root.joinpath("jars", "*")),
        )

    # Desativando loggers
    import jpype.imports
    from org.apache.logging.log4j import Level, LogManager
    from org.apache.logging.log4j.core.config import Configurator

    Configurator.setAllLevels(LogManager.getRootLogger().getName(), Level.OFF)
    startJVM = jpype.startJVM
    jpype.startJVM = lambda *args, **kwargs: None

    # Obtendo instância do CoGrOO
    cogroo = cogroo4py.Cogroo()

    # Desfazendo patch
    jpype.startJVM = startJVM

    return cogroo
