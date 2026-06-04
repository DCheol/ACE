# ACE: Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation

This repository provides preprocessing scripts and RecBole experiment code for LLM-enhanced sequential recommendation.

The overall flow is:

```text
.item / .inter
    ↓
id_map.json / item_str.json
    ↓
{text_encoder}.pkl
    ↓
RecBole experiment
```

---

## Dataset Structure
You can download preprocessed dataset here: [link](https://drive.google.com/drive/folders/1B5tUP9VWcY4jGiLBwmq76rxr0yq8PUu8?usp=drive_link)

Place the dataset files under `dataset/<dataset>/`.

For example:

```text
dataset/
└── toys/
    ├── toys.item
    └── toys.inter
```

After preprocessing, the directory will look like:

```text
dataset/
└── toys/
    ├── toys.item
    ├── toys.inter
    ├── id_map.json
    ├── item_str.json
    └── openai_text_embedding_3_large.pkl
```

If you do not have `.item` and `.inter` files, you can download RecBole's preprocessed files here:

[Google Drive folder](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI)

---

## Step 1. Build Atomic JSON Files

```bash
python3 preprocess/build_atomic_jsons.py \
  --dataset toys \
  --input_path dataset \
  --output_path dataset \
  --user_k 5 \
  --item_k 5
```

This creates:

```text
dataset/toys/id_map.json
dataset/toys/item_str.json
```

`id_map.json` is used to align item ids with embedding rows.
`item_str.json` contains item text built from the `.item` file.

---

## Step 2. Build Item Embeddings

```bash
python3 preprocess/build_item_embeddings_from_json.py \
  --dataset toys \
  --input_path dataset \
  --output_path dataset \
  --llm_encoder openai_text_embedding_3_large \
  --openai_model text-embedding-3-large
```

This creates:

```text
dataset/toys/openai_text_embedding_3_large.pkl
```

The embedding vectors are saved as `float32` and L2-normalized by default.

The script can also use an OpenAI-compatible local embedding server such as vLLM.
In our experiments, item embeddings were generated with a vLLM-served embedding model.

---

## Step 3. Run RecBole Experiments

```bash
python run_recbole.py \
  --model ACE_SASRec \
  --dataset toys \
  --lr 0.0005 \
  --reg_weight 50 \
  --hidden_size 128 \
  --scale 0.1 \
  --text_encoder openai_text_embedding_3_large
```

You can also check `run.sh` for an end-to-end example.

---

## Common Arguments

Frequently tuned arguments include:

```text
lr
hidden_size
reg_weight
scale
text_encoder
```

For different backbone models, you may also tune the number of layers, number of heads, dropout values, or loss type.

---

## Notes

* `build_atomic_jsons.py` and `build_item_embeddings_from_json.py` are the main preprocessing scripts.
* `--text_encoder` should match the embedding file name.
* For example, `--text_encoder openai_text_embedding_3_large` expects:

```text
dataset/toys/openai_text_embedding_3_large.pkl
```

* If you use a local embedding server, make sure `OPENAI_BASE_URL` points to the `/v1` endpoint.
