import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from preprocess.utils.dataset import build_id_map_from_inter
from preprocess.utils.text import build_item_str_from_atomic_item


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. beauty")
    parser.add_argument("--input_path", type=str, default="dataset", help="Root path containing dataset folders")
    parser.add_argument("--output_path", type=str, default="dataset", help="Where to write json outputs")
    parser.add_argument("--user_k", type=int, default=5, help="User k-core threshold used for id_map generation")
    parser.add_argument("--item_k", type=int, default=5, help="Item k-core threshold used for id_map generation")
    parser.add_argument(
        "--item_fields",
        type=str,
        default=None,
        help="Optional comma-separated list of item fields to include in item_str.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = os.path.join(args.input_path, args.dataset)
    inter_path = os.path.join(dataset_dir, f"{args.dataset}.inter")
    item_path = os.path.join(dataset_dir, f"{args.dataset}.item")

    if not os.path.exists(inter_path):
        raise FileNotFoundError(f"Missing interaction file: {inter_path}")
    if not os.path.exists(item_path):
        raise FileNotFoundError(f"Missing item file: {item_path}")

    out_dir = os.path.join(args.output_path, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    id_map_path = os.path.join(out_dir, "id_map.json")
    item_str_path = os.path.join(out_dir, "item_str.json")

    item_fields = None
    if args.item_fields:
        item_fields = [field.strip() for field in args.item_fields.split(",") if field.strip()]

    id_map = build_id_map_from_inter(
        inter_path,
        id_map_path,
        user_k_core_threshold=args.user_k,
        item_k_core_threshold=args.item_k,
    )
    build_item_str_from_atomic_item(
        item_path,
        output_path=item_str_path,
        dataset_name=args.dataset,
        fields=item_fields,
        item_filter=set(id_map["item2id"].keys()),
    )

    print(f"Wrote {id_map_path}")
    print(f"Wrote {item_str_path}")


if __name__ == "__main__":
    main()
