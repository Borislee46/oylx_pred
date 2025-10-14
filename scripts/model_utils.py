import logging
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
        logging.info(f"尝试从路径加载本地 E5 模型: {e5_local_path}")
        try:
            model = SentenceTransformer(e5_local_path)

            if (
                use_quantization
                and hasattr(torch, "quantization")
                and not torch.cuda.is_available()
            ):
                logging.info("应用量化（仅限CPU）...")
                try:
                    transformer = model._first_module().auto_model
                    quantized_transformer = torch.quantization.quantize_dynamic(
                        transformer, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    model._first_module().auto_model = quantized_transformer
                    logging.info("模型量化成功。")
                except Exception as e_quant:
                    logging.warning(f"模型量化失败: {str(e_quant)}. 使用原始模型。")
            elif torch.cuda.is_available():
                logging.info("检测到CUDA，使用GPU加速。")
                model = model.to(torch.device("cuda"))

            logging.info(f"成功加载本地 E5 模型: {e5_local_path}.")
            return model
        except Exception as e_load:
            logging.warning(f"加载本地 E5 模型失败: {str(e_load)}. 尝试在线模型...")

    logging.warning(f"本地 E5 模型加载失败或不存在。尝试加载在线模型: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))
            logging.info("在线模型使用GPU加速。")
        logging.info(f"成功加载在线模型: {model_name}")
        return model
    except Exception as e_final_online:
        logging.error(f"最终加载在线模型 {model_name} 失败: {str(e_final_online)}")
        raise RuntimeError(f"无法加载任何指定的模型: {model_name}") from e_final_online


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
