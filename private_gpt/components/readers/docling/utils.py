# Language mapping dictionaries
LANG_TO_EASYOCR: dict[str, str] = {
    "ar-SA": "ar",  # Arabic
    "bg-BG": "bg",  # Bulgarian
    "zh-CN": "ch_sim",  # Simplified Chinese
    "zh-TW": "ch_tra",  # Traditional Chinese
    "cs-CZ": "cs",  # Czech
    "da-DK": "da",  # Danish
    "nl-NL": "nl",  # Dutch
    "en-US": "en",  # English
    "en-GB": "en",  # English
    "fi-FI": "fi",  # Finnish
    "fr-FR": "fr",  # French
    "de-DE": "de",  # German
    "el-GR": "el",  # Greek
    "he-IL": "he",  # Hebrew
    "hi-IN": "hi",  # Hindi
    "hu-HU": "hu",  # Hungarian
    "id-ID": "id",  # Indonesian
    "it-IT": "it",  # Italian
    "ja-JP": "ja",  # Japanese
    "ko-KR": "ko",  # Korean
    "ms-MY": "ms",  # Malay
    "no-NO": "no",  # Norwegian
    "pl-PL": "pl",  # Polish
    "pt-PT": "pt",  # Portuguese
    "pt-BR": "pt",  # Portuguese (Brazil)
    "ro-RO": "ro",  # Romanian
    "ru-RU": "ru",  # Russian
    "es-ES": "es",  # Spanish
    "sv-SE": "sv",  # Swedish
    "th-TH": "th",  # Thai
    "tr-TR": "tr",  # Turkish
    "uk-UA": "uk",  # Ukrainian
    "vi-VN": "vi",  # Vietnamese
}

LANG_TO_TESSERACT: dict[str, str] = {
    "ar-SA": "ara",  # Arabic
    "bg-BG": "bul",  # Bulgarian
    "zh-CN": "chi_sim",  # Simplified Chinese
    "zh-TW": "chi_tra",  # Traditional Chinese
    "cs-CZ": "ces",  # Czech
    "da-DK": "dan",  # Danish
    "nl-NL": "nld",  # Dutch
    "en-US": "eng",  # English
    "en-GB": "eng",  # English
    "fi-FI": "fin",  # Finnish
    "fr-FR": "fra",  # French
    "de-DE": "deu",  # German
    "el-GR": "ell",  # Greek
    "he-IL": "heb",  # Hebrew
    "hi-IN": "hin",  # Hindi
    "hu-HU": "hun",  # Hungarian
    "id-ID": "ind",  # Indonesian
    "it-IT": "ita",  # Italian
    "ja-JP": "jpn",  # Japanese
    "ko-KR": "kor",  # Korean
    "ms-MY": "msa",  # Malay
    "no-NO": "nor",  # Norwegian
    "pl-PL": "pol",  # Polish
    "pt-PT": "por",  # Portuguese
    "pt-BR": "por",  # Portuguese (Brazil)
    "ro-RO": "ron",  # Romanian
    "ru-RU": "rus",  # Russian
    "es-ES": "spa",  # Spanish
    "sv-SE": "swe",  # Swedish
    "th-TH": "tha",  # Thai
    "tr-TR": "tur",  # Turkish
    "uk-UA": "ukr",  # Ukrainian
    "vi-VN": "vie",  # Vietnamese
}

LANG_TO_RAPIDOCR: dict[str, str] = {
    "zh-CN": "chinese",  # Simplified Chinese
    "zh-TW": "chinese",  # Traditional Chinese
    "en-US": "english",  # English
    "en-GB": "english",  # English
}


def convert_to_easyocr_lang(lang: str) -> str:
    """Convert language code to EasyOCR format.

    Args:
        lang: Language code in format like 'en-US', 'es-ES'

    Returns:
        Language code in EasyOCR format (e.g., 'en', 'es')

    Raises:
        ValueError: If language is not supported
    """
    try:
        return LANG_TO_EASYOCR[lang]
    except KeyError as e:
        raise ValueError(f"Language {lang} not supported by EasyOCR") from e


def convert_to_tesseract_lang(lang: str) -> str:
    """Convert language code to Tesseract format.

    Args:
        lang: Language code in format like 'en-US', 'es-ES'

    Returns:
        Language code in Tesseract format (e.g., 'eng', 'spa')

    Raises:
        ValueError: If language is not supported
    """
    try:
        return LANG_TO_TESSERACT[lang]
    except KeyError as e:
        raise ValueError(f"Language {lang} not supported by Tesseract") from e


def convert_to_rapidocr_lang(lang: str) -> str:
    """Convert language code to RapidOCR format.

    Args:
        lang: Language code in format like 'en-US', 'es-ES'

    Returns:
        Language name in RapidOCR format (e.g., 'english', 'chinese')

    Raises:
        ValueError: If language is not supported
    """
    try:
        return LANG_TO_RAPIDOCR[lang]
    except KeyError as e:
        raise ValueError(f"Language {lang} not supported by RapidOCR") from e


def convert_to_ocrmac_lang(lang: str) -> str:
    """Convert language code to OCRMac format.

    Args:
        lang: Language code in format like 'en-US', 'es-ES'

    Returns:
        Language code in OCRMac format (usually same as input)

    Raises:
        ValueError: If language is not supported
    """
    try:
        # Do nothing
        return lang
    except KeyError as e:
        raise ValueError(f"Language {lang} not supported by OCRMac") from e
