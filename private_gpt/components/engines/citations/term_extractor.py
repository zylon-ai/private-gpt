import re
from typing import Any

from injector import singleton
from langdetect import detect
from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

SUPPORTED_LANGUAGES = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
}

# Mapping langdetect codes to our supported codes
LANG_DETECT_MAP = {
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
}


@singleton
class TextAnalyzer:
    """Create an analizer for text that allow to retrieve unique terms."""

    def __init__(
        self,
        languages: list[str] | None = None,
    ):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(
            stopwords.words(fileids=(languages or SUPPORTED_LANGUAGES.values()))
        )

    def _clean_text(self, text: str, **kwargs: Any) -> str:
        text = text.lower()
        # Only keep characters, hyphens, numbers and points
        text = re.sub(r"[^a-z0-9\-\.]", " ", text)
        return " ".join(text.split())

    def detect_language(self, text: str) -> str | None:
        """Detect the language of the input text.

        Args:
            text: Text to detect language for

        Returns:
            Language code

        Raises:
            ValueError: If detected language is not supported
        """
        try:
            detected = str(detect(text))
            if detected not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Detected language '{detected}' is not supported.")
            return LANG_DETECT_MAP.get(detected)
        except Exception:
            return None

    def _lemmatize_word(self, word: str, lang: str | None = None, **kwargs: Any) -> str:
        if not word:
            return word

        # NLTK doesn't support lemma for some languages
        if lang and lang != "eng":
            return word

        # Get the part of speech
        lang = lang or "eng"
        pos = pos_tag([word], lang=lang)[0][1]

        # Convert Penn Treebank tag to WordNet POS tag
        tag = {
            "N": "n",  # noun
            "V": "v",  # verb
            "R": "r",  # adverb
            "J": "a",  # adjective
        }.get(pos[0], "n")

        # Lemmatize with the POS tag
        return str(self.lemmatizer.lemmatize(word, tag))

    def process_words(self, words: list[str], **kwargs: Any) -> list[str]:
        # Lemmatize each word
        words = [self._lemmatize_word(word, **kwargs) for word in words]

        # Strip content
        words = [word.strip() for word in words]

        return words

    def filter_words(
        self,
        words: list[str],
        **kwargs: Any,
    ) -> list[str]:
        # Skip any that doesn't have any letters
        words = [word for word in words if any(char.isalpha() for char in word)]

        # Skip any word that contains something different to letters
        words = [word for word in words if re.match(r"^[a-zA-Z]+$", word)]

        # Skip any stopwords
        words = [word for word in words if word not in self.stop_words]

        # Skip any words that are too short
        min_length = kwargs.get("min_length")
        if min_length is not None:
            words = [word for word in words if len(word) >= min_length]

        # Skip any words that are too long
        max_length = kwargs.get("max_length")
        if max_length is not None:
            words = [word for word in words if len(word) <= max_length]

        return words

    def get_terms(self, text: str, lang: str | None = None, **kwargs: Any) -> set[str]:
        # Detect language
        # lang = lang or self.detect_language(text)

        # Clean text before processing
        cleaned = self._clean_text(text, **kwargs)

        # Tokenize, process and filter words
        words = word_tokenize(cleaned)
        words = self.process_words(words, lang=lang, **kwargs)
        words = self.filter_words(words, **kwargs)

        # Return unique words
        return set(words)

    def score_term(self, term: str, text: str) -> float:
        cleaned = self._clean_text(text)
        if term not in cleaned:
            return 0.0

        # Score based on position in lines
        lines = cleaned.split("\n")
        score = 0.0

        for line in lines:
            if term in line:
                # Higher score for terms at start of line or after table delimiter
                if line.strip().startswith(term):
                    score = max(score, 0.8)
                elif "|" in line and term in line.split("|")[0].strip():
                    score = max(score, 0.6)
                else:
                    score = max(score, 0.4)

        return score

    def get_unique_terms(
        self,
        texts: list[str],
        max_terms: int = 5,
        min_length: int | None = None,
        max_length: int | None = None,
        langs: set[str] | None = None,
        **kwargs: Any,
    ) -> list[list[str]]:
        # Validate if current languages are supported
        if langs:
            for lang in langs:
                supported = False
                for supported_lang in SUPPORTED_LANGUAGES:
                    if supported_lang in lang:
                        supported = True
                        break
                if not supported:
                    return []

        # Get terms for each text
        all_terms = [
            self.get_terms(text, min_length=min_length, max_length=max_length, **kwargs)
            for text in texts
        ]

        # Find unique terms
        unique_terms = []
        for i, terms in enumerate(all_terms):
            other_terms = set().union(
                *(term_set for j, term_set in enumerate(all_terms) if j != i)
            )
            unique = terms - other_terms

            # Score and sort unique terms
            scored = [(term, self.score_term(term, texts[i])) for term in unique]
            best_terms = [
                term
                for term, score in sorted(scored, key=lambda x: x[1], reverse=True)
                if score > 0.2
            ][:max_terms]

            unique_terms.append(best_terms)

        return unique_terms
