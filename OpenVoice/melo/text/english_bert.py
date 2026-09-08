from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


MODEL_PATH = Path(__file__).resolve().parents[2] / "checkpoints_v2" / "melotts" / "bert_model" / "english_bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = None


def load_model(device=None):
    global model
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if model is None:
        model = AutoModel.from_pretrained(
            MODEL_PATH, local_files_only=True
        ).to(device)
        model.eval()
    elif model.device != device:
        model = model.to(device)
    return model


def get_bert_feature(text, word2ph, device=None):
    bert_model = load_model(device)

    with torch.inference_mode():
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(bert_model.device) for key, value in inputs.items()}
        result = bert_model(**inputs, output_hidden_states=True)
        features = result.hidden_states[-3][0]

    if inputs["input_ids"].shape[-1] != len(word2ph):
        raise RuntimeError("English tokenizer length does not match word-to-phone mapping")
    return torch.cat(
        [features[index].repeat(count, 1) for index, count in enumerate(word2ph)],
        dim=0,
    ).T
