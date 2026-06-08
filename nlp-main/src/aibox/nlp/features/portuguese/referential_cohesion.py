"""Características relacionadas
à Coesão Referencial.
"""

import math
from dataclasses import dataclass

import spacy
from nltk.stem.snowball import SnowballStemmer

from aibox.nlp.core import FeatureExtractor
from aibox.nlp.features.utils import DataclassFeatureSet


@dataclass(frozen=True)
class ReferentialCohesion(DataclassFeatureSet):
    """Características de coesão referencial.

    Para mais informações sobre as características, checar
    as referências.

    Referências:
        `[1] <https://doi.org/10.1145/3576050.3576152>`_:
        Oliveira, H., Ferreira Mello, R., Barreiros Rosa,
        B. A., Rakovic, M., Miranda, P., Cordeiro, T., Isotani,
        S., Bittencourt, I., & Gasevic, D. (2023). Towards
        explainable prediction of essay cohesion in
        Portuguese and English. In LAK23: 13th International
        Learning Analytics and Knowledge Conference (pp.
        509–519). LAK 2023: 13th International Learning
        Analytics and Knowledge Conference. ACM.
    """

    adj_arg_ovl: float
    adj_cw_ovl: float
    adj_stem_ovl: float
    arg_ovl: float
    adjacent_refs: float
    coreference_pronoun_ratio: float
    anaphoric_refs: float
    stem_ovl: float
    demonstrative_pronoun_ratio: float


class ReferentialCohesionExtractor(FeatureExtractor):
    """Extrator de características relacionadas
    à coesão referencial.

    :param nlp: modelo do spaCy a ser utilizado. Defaults
        to "pt_core_news_md".
    :param stemmer: stemmer do NLTK a ser utilizado. Defaults
        to :py:class:`~nltk.stem.snowball.SnowballStemmer`.

    Exemplo de uso em
    :py:class:`~aibox.nlp.core.feature_extractor.FeatureExtractor`
    """

    def __init__(
        self, nlp: spacy.Language | None = None, stemmer: SnowballStemmer | None = None
    ):
        self._nlp = nlp
        self._stemmer = stemmer

    @property
    def feature_set(self) -> type[ReferentialCohesion]:
        return ReferentialCohesion

    def extract(self, text: str, **kwargs) -> ReferentialCohesion:
        del kwargs

        self._maybe_load_models()
        doc = self._nlp(text)
        sentences = [sent.text for sent in doc.sents]
        features = {
            "adj_arg_ovl": 0,
            "adj_cw_ovl": 0,
            "adj_stem_ovl": 0,
            "arg_ovl": 0,
            "adjacent_refs": 0,
            "coreference_pronoun_ratio": 0,
            "anaphoric_refs": 0,
            "stem_ovl": 0,
            "demonstrative_pronoun_ratio": 0,
        }
        if len(sentences) <= 1:
            return ReferentialCohesion(**features)

        features["adj_arg_ovl"] = self.compute_adj_arg_ovl(doc)
        features["adj_cw_ovl"] = self.compute_adj_cw_ovl(doc)
        features["adj_stem_ovl"] = self.compute_adj_stem_ovl(doc)
        features["arg_ovl"] = self.compute_arg_ovl(doc)
        features["adjacent_refs"] = self.compute_adjacent_refs(doc)
        features["coreference_pronoun_ratio"] = self.compute_coreference_pronoun_ratio(
            doc
        )
        features["anaphoric_refs"] = self.compute_anaphoric_refs(doc)
        features["stem_ovl"] = self.compute_stem_ovl(doc)
        features["demonstrative_pronoun_ratio"] = (
            self.compute_demonstrative_pronoun_ratio(doc)
        )

        return ReferentialCohesion(**features)

    def compute_adj_arg_ovl(self, doc) -> float:
        """Método que computa a quantidade média de referentes
        (substantivos e pronomes) que se repetem nos pares de
        sentenças adjacentes do texto.
        """
        assert doc is not None, "Error DOC is None"
        sentences = [sent for sent in doc.sents]
        if len(sentences) <= 1:
            return 0
        tokens_dict = {}
        referents_pos = {"NOUN", "PROPN", "PRON"}
        for key, sentence in enumerate(sentences):
            tokens_dict[key] = set(
                [
                    token.text
                    for token in sentence
                    if not token.is_punct and token.pos_ in referents_pos
                ]
            )
        count_repetitions = 0
        total_sents_pairs = len(tokens_dict) - 1
        for i in range(total_sents_pairs):
            repetitions = [
                token for token in tokens_dict[i] if token in tokens_dict[i + 1]
            ]
            count_repetitions += len(repetitions)
        return count_repetitions / total_sents_pairs

    def compute_adj_cw_ovl(self, doc) -> float:
        """Método que computa a quantidade média de palavras
        de conteúdo (substantivos, verbos, adjetivos e advérbios)
        que se repetem nos pares de sentenças adjacentes do texto.
        """
        assert doc is not None, "Error DOC is None"
        sentences = [sent for sent in doc.sents]
        tokens_dict = {}
        if len(sentences) <= 1:
            return 0
        content_words_pos = {"NOUN", "PROPN", "ADJ", "ADV", "VERB"}
        for key, sentence in enumerate(sentences):
            tokens_dict[key] = [
                token.text
                for token in sentence
                if not token.is_punct and token.pos_ in content_words_pos
            ]
        count_repetitions = 0
        total_sents_pairs = len(tokens_dict) - 1
        for i in range(total_sents_pairs):
            repetitions = [
                token for token in tokens_dict[i] if token in tokens_dict[i + 1]
            ]
            count_repetitions += len(repetitions)
        return count_repetitions / total_sents_pairs

    def compute_adj_stem_ovl(self, doc) -> float:
        """Método que computa a quantidade média de radicais de
        palavras de conteúdo (substantivos, verbos, adjetivos e
        advérbios) que se repetem nos pares de sentenças adjacentes
        do texto.
        """
        assert doc is not None, "Error DOC is None"
        sentences = [sent for sent in doc.sents]
        if len(sentences) <= 1:
            return 0
        tokens_dict = {}
        content_words_pos = {"NOUN", "PROPN", "ADJ", "ADV", "VERB"}
        for key, sentence in enumerate(sentences):
            tokens = [
                token.text
                for token in sentence
                if not token.is_punct and token.pos_ in content_words_pos
            ]
            radicals = [self._stemmer.stem(token) for token in tokens]
            tokens_dict[key] = radicals
        count_repetitions = 0
        total_sents_pairs = len(tokens_dict) - 1
        for i in range(total_sents_pairs):
            repetitions = [
                token for token in tokens_dict[i] if token in tokens_dict[i + 1]
            ]
            count_repetitions += len(repetitions)
        return count_repetitions / total_sents_pairs

    def compute_arg_ovl(self, doc) -> float:
        """Método que computa a quantidade média de referentes
        (substantivos ou pronomes) que se repetem nos pares de
        sentenças do texto.
        """
        assert doc is not None, "Error DOC is None"
        sentences = [sent for sent in doc.sents]
        if len(sentences) <= 1:
            return 0
        tokens_dict = {}
        referents_pos = {"NOUN", "PROPN", "PRON"}
        for key, sentence in enumerate(sentences):
            tokens_dict[key] = [
                token.text
                for token in sentence
                if not token.is_punct and token.pos_ in referents_pos
            ]
        count_repetitions = 0
        total_sentences = len(sentences)
        total_sents_pairs = math.factorial(total_sentences) / (
            math.factorial(4) * math.factorial(2)
        )
        for i in range(total_sentences):
            for j in range(i + 1, total_sentences):
                repetitions = [
                    token for token in tokens_dict[i] if token in tokens_dict[j]
                ]
                count_repetitions += len(repetitions)
        return count_repetitions / total_sents_pairs

    def compute_adjacent_refs(self, doc) -> float:
        """Método que computa a média de candidatos a referente,
        na sentença anterior, por pronome anafórico
        """
        assert doc is not None, "Error DOC is None"
        sentences = [sent for sent in doc.sents]
        all_referents = {}
        all_pronouns = {}
        referents_pos = {"NOUN", "PROPN"}
        for key, sentence in enumerate(sentences):
            all_referents[key] = [
                [token.morph.get("Gender"), token.morph.get("Number")]
                for token in sentence
                if not token.is_punct and token.pos_ in referents_pos
            ]
            all_pronouns[key] = [
                [token.morph.get("Gender"), token.morph.get("Number")]
                for token in sentence
                if not token.is_punct and token.pos_ == "PRON"
            ]
        total_candidates = 0
        anaphoric_pron = len(all_pronouns) - 1
        if anaphoric_pron == 0:
            return 0
        for i in range(1, len(sentences)):
            if len(all_pronouns[i]) > 0 and len(all_referents[i - 1]) > 0:
                for pronouns in all_pronouns[i]:
                    result = all_referents[i - 1].count(pronouns)
                    total_candidates += result
        return total_candidates / anaphoric_pron

    def compute_anaphoric_refs(self, doc) -> float:
        """Método que computa a média de candidatos a referente,
        em até 5 sentenças anteriores, por pronome anafórico.
        """
        assert doc is not None, "Error DOC is None"
        sentences = [sent for sent in doc.sents]
        referents_sentences = {}
        pronouns_sentences = {}
        all_pronouns = []
        referents_pos = {"NOUN", "PROPN"}
        for key, sentence in enumerate(sentences):
            referents_sentences[key] = [
                [token.morph.get("Gender"), token.morph.get("Number")]
                for token in sentence
                if not token.is_punct and token.pos_ in referents_pos
            ]
            pronouns_sentences[key] = [
                [token.morph.get("Gender"), token.morph.get("Number")]
                for token in sentence
                if not token.is_punct and token.pos_ == "PRON"
            ]
            pronouns_sent = [
                token.text
                for token in sentence
                if not token.is_punct and token.pos_ == "PRON"
            ]
            all_pronouns.extend(pronouns_sent)
        if len(all_pronouns) == 0:
            return 0
        total_candidates = 0
        for i in range(1, len(sentences)):
            if len(pronouns_sentences[i]) > 0:
                limit = i - 6 if i - 6 >= -1 else -1
                for j in range(i - 1, limit, -1):
                    if len(referents_sentences[j]) > 0:
                        for pronouns in pronouns_sentences[i]:
                            result = referents_sentences[j].count(pronouns)
                            total_candidates += result
        return total_candidates / len(all_pronouns)

    def compute_stem_ovl(self, doc) -> float:
        """Método que computa a quantidade média de
        radicais de palavras de conteúdo (substantivos, verbos, adjetivos
        e advérbios) que se repetem nos pares de sentenças do texto.
        """
        assert doc is not None, "Error DOC is None"
        sentences = [sent for sent in doc.sents]
        tokens_sentences = []
        content_words_pos = {"NOUN", "PROPN", "ADJ", "ADV", "VERB"}
        for sentence in sentences:
            words = [
                token.text
                for token in sentence
                if not token.is_punct and token.pos_ in content_words_pos
            ]
            stems = [self._stemmer.stem(token) for token in words]
            if len(stems) > 0:
                tokens_sentences.append(stems)
        total_overlaps = 0
        q = len(tokens_sentences)
        if q < 2:
            return 0
        total_pairs = math.factorial(q) / (math.factorial(q - 2) * math.factorial(2))
        if total_pairs == 0:
            return 0
        for i in range(q):
            for j in range(i + 1, q):
                repetitions = []
                for tokens in tokens_sentences[i]:
                    if tokens in tokens_sentences[j]:
                        repetitions.extend(tokens)
                total_overlaps += len(repetitions)
        return total_overlaps / total_pairs

    def compute_coreference_pronoun_ratio(self, doc) -> float:
        """Método que computa a média de candidatos a referente,
        na sentença anterior, por pronome anafórico
        do caso reto (ele, ela, eles e elas).
        """
        assert doc is not None, "Error DOC is None"
        sentences = [sent for sent in doc.sents]
        if len(sentences) <= 1:
            return 0
        referents_pos = {"NOUN", "PROPN", "PRON"}
        all_referents = {}
        all_pronouns = {}
        straight_pronouns = ["ele", "ela", "eles", "elas"]
        all_straight_pronouns = []
        for key, sentence in enumerate(sentences):
            all_referents[key] = [
                [token.morph.get("Gender"), token.morph.get("Number")]
                for token in sentence
                if not token.is_punct and token.pos_ in referents_pos
            ]
            all_pronouns[key] = [
                [token.morph.get("Gender"), token.morph.get("Number")]
                for token in sentence
                if not token.is_punct
                and token.pos_ == "PRON"
                and token.lower_ in straight_pronouns
            ]
            all_straight_pronouns.extend(all_pronouns[key])
        total_candidates = 0
        anaphoric_pron = len(all_straight_pronouns)
        if anaphoric_pron == 0:
            return 0
        for i in range(1, len(sentences)):
            if len(all_pronouns[i]) > 0 and len(all_referents[i - 1]) > 0:
                for pronouns in all_pronouns[i]:
                    result = all_referents[i - 1].count(pronouns)
                    total_candidates += result
        return total_candidates / anaphoric_pron

    def compute_demonstrative_pronoun_ratio(self, doc) -> float:
        """Método que computa a média de candidatos a referente,
        na sentença anterior, por pronome demonstrativo
        anafórico ('esse', 'essa', 'esses', 'essas', 'desse',
        'dessa', 'desses', 'dessas').
        """
        assert doc is not None, "Error DOC is None"
        sentences = [sent for sent in doc.sents]
        if len(sentences) <= 1:
            return 0
        referents_sentences = {}
        dem_pronouns_sentences = {}
        referents_pos = {"NOUN", "PROPN", "PRON"}
        dem_pronouns = [
            "esse",
            "essa",
            "esses",
            "essas",
            "desse",
            "dessa",
            "desses",
            "dessas",
        ]
        founded_dem_pronouns = []
        for key, sentence in enumerate(sentences):
            referents_sentences[key] = [
                [token.morph.get("Gender"), token.morph.get("Number")]
                for token in sentence
                if not token.is_punct and token.pos_ in referents_pos
            ]
            dem_pronouns_sentences[key] = [
                [token.morph.get("Gender"), token.morph.get("Number")]
                for token in sentence
                if not token.is_punct and token.lower_ in dem_pronouns
            ]
            founded_dem_pronouns.extend(dem_pronouns_sentences[key])
        total_founded_dem_pronouns = len(founded_dem_pronouns)
        if total_founded_dem_pronouns == 0:
            return 0
        total_candidates = 0
        for i in range(1, len(sentences)):
            if (
                len(dem_pronouns_sentences[i]) > 0
                and len(referents_sentences[i - 1]) > 0
            ):
                for pronouns in dem_pronouns_sentences[i]:
                    result = referents_sentences[i - 1].count(pronouns)
                    total_candidates += result
        return total_candidates / total_founded_dem_pronouns

    def _maybe_load_models(self):
        if self._nlp is None:
            self._nlp = spacy.load("pt_core_news_md")

        if self._stemmer is None:
            self._stemmer = SnowballStemmer(language="portuguese")
