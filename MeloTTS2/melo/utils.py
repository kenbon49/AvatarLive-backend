import json
from pathlib import Path

import torch

from . import commons
from .text import cleaned_text_to_sequence, get_bert
from .text.cleaner import clean_text


def get_text_for_tts_infer(
    text, language, hps, device, symbol_to_id, bert_feature=None
):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    phones, tones, language_ids = cleaned_text_to_sequence(
        phones, tones, language, symbol_to_id
    )

    if hps.data.add_blank:
        phones = commons.intersperse(phones, 0)
        tones = commons.intersperse(tones, 0)
        language_ids = commons.intersperse(language_ids, 0)
        word2ph = [count * 2 for count in word2ph]
        word2ph[0] += 1

    feature = (
        bert_feature(norm_text, word2ph)
        if bert_feature is not None
        else get_bert(norm_text, word2ph, language, device)
    )
    if feature.shape[-1] != len(phones):
        raise RuntimeError(
            f"BERT length {feature.shape[-1]} does not match phone length {len(phones)}"
        )
    if language == "ZH_MIX_EN":
        bert = torch.zeros(1024, len(phones))
        ja_bert = feature
    else:
        bert = torch.zeros(1024, len(phones))
        ja_bert = feature

    return (
        bert,
        ja_bert,
        torch.LongTensor(phones),
        torch.LongTensor(tones),
        torch.LongTensor(language_ids),
    )


def get_hparams_from_file(config_path):
    with Path(config_path).open(encoding="utf-8") as file:
        return HParams(**json.load(file))


class HParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = HParams(**value)
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def __iter__(self):
        return iter(self.__dict__)

    def __contains__(self, key):
        return hasattr(self, key)
