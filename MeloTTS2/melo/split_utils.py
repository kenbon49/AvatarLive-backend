import re


def split_sentence(text, min_len=10, language_str="EN"):
    if language_str == "EN":
        return split_sentences_latin(text)
    return split_sentences_zh(text, min_len)


def split_sentences_latin(text, desired_length=256, max_length=512):
    text = re.sub(r"[。！？；]", ".", text)
    text = re.sub(r"[，]", ",", text)
    text = re.sub(r"[<>()[\]\"«»]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if current and len(current) + len(sentence) + 1 > max_length:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
        if len(current) >= desired_length:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def split_sentences_zh(text, min_len=10):
    text = re.sub(r"[。！？；]", ".", text)
    text = re.sub(r"[，]", ",", text)
    text = re.sub(r"[\n\t ]+", " ", text)
    sentences = [part.strip() for part in re.split(r"(?<=[,.!?;])", text) if part.strip()]
    merged = []
    buffer = ""
    for sentence in sentences:
        buffer = f"{buffer} {sentence}".strip()
        if len(buffer) > min_len:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged and len(buffer) <= 2:
            merged[-1] = f"{merged[-1]} {buffer}"
        else:
            merged.append(buffer)
    return merged
