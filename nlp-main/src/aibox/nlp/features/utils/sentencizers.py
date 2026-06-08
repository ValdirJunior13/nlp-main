"""Utilidades para extração de sentenças."""

import spacy


def spacy_sentencizer(text: str, nlp: spacy.Language = None) -> list[spacy.tokens.Span]:
    if nlp is None:
        nlp = spacy.load("pt_core_news_md")

    doc = nlp(text)

    return [s for s in doc.sents if len(s.text.strip()) > 0]
