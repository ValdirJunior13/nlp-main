"""Características do CohMetrix-BR."""

from dataclasses import dataclass

from cohmetrixBR import features

from aibox.nlp.core.feature_extractor import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet


@dataclass(frozen=True)
class CohMetrixFeatures(DataclassFeatureSet):
    """Essa classe possui todas as características
    disponibilizadas pelo CohMetrix BR.

    Para uma descrição de cada característica checar
    as referências:
        `[1] <https://doi.org/10.5753/cbie.wcbie.2020.179>`_: Camelo, R., Justino, S., & Mello,
        R. F. L. de. (2020). Coh-Metrix PT-BR: Uma API
        web de análise textual para a educação.
        In Anais Estendidos do IX Congresso Brasileiro
        de Informática na Educação (CBIE 2020)
        (pp. 179–186). Anais Estendidos do Congresso
        Brasileiro de Informática na Educação.
        Sociedade Brasileira de Computação.
    """

    despc: float
    despc2: float
    despl: float
    despld: float
    dessc: float
    dessl: float
    dessld: float
    deswc: float
    deswlsy: float
    deswlsyd: float
    deswllt: float
    deswlltd: float
    crfno1: float
    crfao1: float
    crfso1: float
    crfnoa: float
    crfaoa: float
    crfsoa: float
    crfcwo1: float
    crfcwo1d: float
    crfcwoa: float
    crfcwoad: float
    ldttrc: float
    ldttra: float
    ldmtlda: float
    ldvocda: float
    cncadc: float
    cncadd: float
    cncall: float
    cncalter: float
    cnccaus: float
    cnccomp: float
    cncconce: float
    cncconclu: float
    cnccondi: float
    cncconfor: float
    cncconse: float
    cncexpli: float
    cncfinal: float
    cncinte: float
    cnclogic: float
    cncneg: float
    cncpos: float
    cncprop: float
    cnctemp: float
    smintep: float
    smintep_sentence: float
    sminter: float
    smcauswn: float
    synle: float
    synnp: float
    synmedpos: float
    synmedlem: float
    synmedwrd: float
    synstruta: float
    synstrutt: float
    drnp: float
    drvp: float
    drap: float
    drpp: float
    drpval: float
    drneg: float
    drgerund: float
    drinf: float
    wrdnoun: float
    wrdverb: float
    wrdadj: float
    wrdadv: float
    wrdpro: float
    wrdprp1s: float
    wrdprp1p: float
    wrdprp2: float
    wrdprp2s: float
    wrdprp2p: float
    wrdprp3s: float
    wrdprp3p: float
    wrdfrqc: float
    wrdfrqa: float
    wrdfrqmc: float
    wrdaoac: float
    wrdfamc: float
    wrdcncc: float
    wrdimgc: float
    wrdmeac: float
    rdfre: float
    rdfkgl: float
    rdl2: float


class CohMetrixExtractor(FeatureExtractor):
    """Extrator de características do CohMetrix-BR.

    Exemplo de uso em
    :py:class:`~aibox.nlp.core.feature_extractor.FeatureExtractor`
    """

    @property
    def feature_set(self) -> type[CohMetrixFeatures]:
        return CohMetrixFeatures

    def extract(self, text: str, **kwargs) -> CohMetrixFeatures:
        del kwargs

        # CohMetrix-BR requer alguns recursos do NLTK
        self._maybe_download_resources()

        return CohMetrixFeatures(
            **{f.__name__.lower(): float(f(text)) for f in features.FEATURES}
        )

    def _maybe_download_resources(self):
        import nltk

        nltk.download("punkt_tab", quiet=True)
