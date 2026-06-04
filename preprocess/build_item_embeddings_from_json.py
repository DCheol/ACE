import argparse
import json
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

# try:
from dotenv import load_dotenv
# except ImportError:  # optional dependency
#     load_dotenv = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. toys")
    parser.add_argument("--input_path", type=str, default="dataset", help="Root directory containing dataset folders")
    parser.add_argument("--output_path", type=str, default="dataset", help="Where to write the embedding pickle")
    parser.add_argument(
        "--item_str_name",
        type=str,
        default="item_str.json",
        help="Name of the item string json file inside the dataset folder",
    )
    parser.add_argument(
        "--id_map_name",
        type=str,
        default="id_map.json",
        help="Name of the id map json file inside the dataset folder",
    )
    parser.add_argument(
        "--llm_encoder",
        type=str,
        default="openai_text_embedding_3_large",
        help="Output pickle name without extension",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="text-embedding-3-large",
        help="OpenAI embedding model name",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL, e.g. http://localhost:8000/v1 for vLLM",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Optional API key override. If omitted, uses OPENAI_API_KEY from the environment.",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable L2 normalization of each embedding vector",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_ordered_texts(item_str_dict, item2id):
    ordered_texts = [""] * len(item2id)
    for item_id, idx in item2id.items():
        row = int(idx) - 1
        ordered_texts[row] = item_str_dict.get(item_id, "")

    missing = [i for i, text in enumerate(ordered_texts) if text == ""]
    if missing:
        raise ValueError(
            f"Found {len(missing)} empty item texts after alignment. "
            f"Check whether item_str.json covers all items in id_map.json."
        )
    return ordered_texts


def make_openai_client(base_url=None, api_key=None):
    from openai import OpenAI

    env_base_url = os.getenv("OPENAI_BASE_URL")
    env_api_key = os.getenv("OPENAI_API_KEY")

    base_url = base_url or env_base_url
    api_key = api_key or env_api_key

    if base_url:
        resolved_key = api_key or "EMPTY"
        return OpenAI(base_url=base_url, api_key=resolved_key)

    if api_key:
        return OpenAI(api_key=api_key)

    return OpenAI()


def encode_with_openai(sentences, model_name):
    response = encode_with_openai.client.embeddings.create(model=model_name, input=sentences)
    batch_emb = [item.embedding for item in response.data]
    return np.asarray(batch_emb, dtype=np.float32)


encode_with_openai.client = None


def l2_normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return embeddings / norms


def generate_embeddings(ordered_texts, model_name, batch_size, normalize):
    embeddings = []
    with tqdm(total=len(ordered_texts), desc="Embedding items", unit="item") as pbar:
        start = 0
        while start < len(ordered_texts):
            sentences = ordered_texts[start : start + batch_size]
            batch_embeddings = encode_with_openai(sentences, model_name)
            if normalize:
                batch_embeddings = l2_normalize(batch_embeddings)
            embeddings.append(batch_embeddings)
            start += batch_size
            pbar.update(len(sentences))

    return np.concatenate(embeddings, axis=0)


def main():
    # if load_dotenv is not None:
    load_dotenv()

    args = parse_args()
    dataset_dir = os.path.join(args.input_path, args.dataset)
    item_str_path = os.path.join(dataset_dir, args.item_str_name)
    id_map_path = os.path.join(dataset_dir, args.id_map_name)

    if not os.path.exists(item_str_path):
        raise FileNotFoundError(f"Missing item string file: {item_str_path}")
    if not os.path.exists(id_map_path):
        raise FileNotFoundError(f"Missing id map file: {id_map_path}")

    item_str_dict = load_json(item_str_path)
    id_map = load_json(id_map_path)
    item2id = id_map["item2id"]

    ordered_texts = build_ordered_texts(item_str_dict, item2id)
    encode_with_openai.client = make_openai_client(base_url=args.base_url, api_key=args.api_key)
    print(
        "Embedding target:",
        getattr(encode_with_openai.client, "base_url", None) or "default OpenAI endpoint",
    )
    embeddings = generate_embeddings(
        ordered_texts=ordered_texts,
        model_name=args.openai_model,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
    )

    out_dir = os.path.join(args.output_path, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.llm_encoder}.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {out_path}")
    print(f"shape={embeddings.shape}, dtype={embeddings.dtype}")


if __name__ == "__main__":
    main()
