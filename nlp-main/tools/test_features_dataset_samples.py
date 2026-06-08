

import tqdm
import inspect
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np

from aibox.nlp.factory import available_extractors, get_dataset, get_extractor
from aibox.nlp.factory.class_registry import get_class


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_default_kwargs_for(identifier: str) -> dict:
    # Obter classe do identificador
    cls = get_class(identifier)
    kwargs = dict()
    assert inspect.isclass(cls)

    # Obter assinatura do __init__
    signature = inspect.signature(cls.__init__)

    # Checar todos os parâmetros do __init__
    for param in signature.parameters.values():
        name = param.name

        # Alguns parâmetros não precisam ser checados
        if name in {"self", "args", "kwargs"}:
            continue

        # Se não possui valor default, adicionar um
        if param.default == inspect.Parameter.empty:
            if name == "reference_text":
                kwargs["reference_text"] = (
                    "Esse é um texto padrão para cálculo de similaridade.\n"
                    "Ele possui alguns parágrafos e diferentes sentenças. "
                    "O objetivo é permite que extratores de características "
                    "baseados em similaridade funcionem corretamente.\n"
                    "Por ser um texto artificial, não contém estruturas "
                    "e/ou características que um texto real possui."
                )
            elif name == "target_competence":
                if identifier == "essayBR":
                    value = "C1"
                elif identifier == "narrativeEssaysBR":
                    value = "cohesion"
                else:
                    raise ValueError(f"Dataset '{identifier}' desconhecido.")
                kwargs["target_competence"] = value
            elif name == "extended":
                kwargs["extended"] = True
            else:
                raise ValueError(
                    f"Valor de parâmetro '{name}' (classe {cls}) é desconhecido."
                )

    return kwargs


if __name__ == "__main__":
    # Criando argparser
    parser = ArgumentParser(
        prog=__name__,
        description="Executa todos extratores em amostras do dataset selecionado.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # Adicionando argumentos
    parser.add_argument("dataset", type=str, help="Identificador do dataset.")
    parser.add_argument("-n", type=int, help="Quantidade de amostras.", default=20)
    parser.add_argument(
        "-s", "--seed", type=int, help="Seed randômica para obter amostras.", default=42
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        help="Quantidade de workers para vetorização em batch.",
        default=8,
    )

    # Realizando parse
    args = parser.parse_args()

    # Obtendo amostras
    sampĺes = (
        get_dataset(args.dataset, **get_default_kwargs_for(args.dataset))
        .to_frame()
        .sample(n=args.n, random_state=args.seed)
    )

    # Realizando extração de características
    #   utilizando chunks de extratores
    arrays = []
    extractors = available_extractors()
    chunk_size = 5
    for chunk in tqdm.tqdm(
        chunks(extractors, chunk_size),
        desc="Chunk extratores",
        total=len(extractors) // chunk_size,
    ):
        print("Extratores:", chunk)
        extractor = get_extractor(
            chunk,
            [get_default_kwargs_for(e) for e in chunk],
        )

        # Realizando extração de características para todos
        #   os textos
        arrays.append(extractor.vectorize(sampĺes.text.values, n_workers=args.workers))

    # Estatísticas das características extraídas
    arr = np.concatenate(arrays, axis=-1)
    print("Quantidade de características:", arr.shape[1])
    print(f"NaNs (%): {100 * np.isnan(arr).sum() / arr.shape[0]:.2f}")
    print(f"Mean (std): {arr.mean():.2f} ({arr.std():.2f})")
