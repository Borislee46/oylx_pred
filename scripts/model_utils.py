import os

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

torch.classes.__path__ = []


def get_model(model_name="intfloat/multilingual-e5-large-instruct", use_quantization=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    e5_local_path = os.path.join(project_root, "src", "services", "multilingual-e5-large-instruct")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing Embeddings"):
            batch_texts = texts[i : i + batch_size]
            embeddings = model.encode(
                batch_texts,
                convert_to_tensor=True,
                batch_size=batch_size,
                device=device,
                normalize_embeddings=True,
            )
            all_embeddings.extend(embeddings.cpu())

    return all_embeddings


def compute_similarity_matrix(embeddings, model_device="cuda"):
    device = torch.device(model_device if torch.cuda.is_available() else "cpu")
    embedding_tensor = torch.stack(embeddings).to(device)
    similarity_matrix = torch.mm(embedding_tensor, embedding_tensor.t()).cpu().numpy()
    del embedding_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return similarity_matrix
