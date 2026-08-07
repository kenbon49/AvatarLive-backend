def clean_text(text, language):
    if language == "EN":
        from . import english as language_module
    elif language == "ZH_MIX_EN":
        from . import chinese_mix as language_module
    else:
        raise ValueError(f"Unsupported language: {language}")

    normalized = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(normalized)
    return normalized, phones, tones, word2ph
