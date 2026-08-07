from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


MODEL_PATH = Path(__file__).resolve().parents[2] / "bert_model" / "multilingual"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = None


def get_bert_feature(text, word2ph, device=None):
    global model
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if model is None:
        model = AutoModel.from_pretrained(
            MODEL_PATH, local_files_only=True
        ).to(device)
        model.eval()

    with torch.inference_mode():
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        result = model(**inputs, output_hidden_states=True)
        features = result.hidden_states[-3][0]

    if features.shape[0] != len(word2ph):
        raise RuntimeError("Multilingual tokenizer length does not match word-to-phone mapping")
    return torch.cat(
        [features[index].repeat(count, 1) for index, count in enumerate(word2ph)],
        dim=0,
    ).T
