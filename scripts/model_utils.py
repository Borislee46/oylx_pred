import os

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

torch.classes.__path__ = []


def get_model(model_name="intfloat/multilingual-e5-large-instruct", use_quantization=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 兼容 src/service 与 src/services 两种放置路径
    e5_local_path = None
    for cand in (
        os.path.join(project_root, "src", "service", "multilingual-e5-large-instruct"),
        os.path.join(project_root, "src", "services", "multilingual-e5-large-instruct"),
    ):
        if os.path.exists(os.path.join(cand, "config.json")):
            e5_local_path = cand
            break

    if os.path.exists(e5_local_path):
        model = SentenceTransformer(e5_local_path)

        if use_quantization and not torch.cuda.is_available():
            transformer = model._first_module().auto_model
            quantized_transformer = torch.quantization.quantize_dynamic(
                transformer, {torch.nn.Linear}, dtype=torch.qint8
            )
            model._first_module().auto_model = quantized_transformer
        elif torch.cuda.is_available():
            model = model.to(torch.device("cuda"))

        return model

    model = SentenceTransformer(model_name)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    return model


def compute_embeddings_batch_local(texts, model, batch_size=32):
    if not texts:
        return []

    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing Embeddings"):
            batch_texts = texts[i : i + batch_size]
            embeddings = model.encode(
                batch_texts,
                convert_to_tensor=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            all_embeddings.extend(embeddings.cpu())

    return all_embeddings
