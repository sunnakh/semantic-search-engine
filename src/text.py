"""Word tokenization shared by the lexical retriever and claim audit."""

import re
import unicodedata

WORDS = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return WORDS.findall(unicodedata.normalize("NFKC", text).casefold())
