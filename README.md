# ACE: Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation

This repo contains the code for preprocessing Amazon-style sequential recommendation datasets, building item embeddings, and running RecBole experiments.

## Overview

The recommended flow is:

1. Prepare `.item` and `.inter`
2. Build `id_map.json` and `item_str.json`
3. Build `{llm_encoder}.pkl`
4. Run RecBole experiments

The `test.sh` file shows a typical end-to-end experiment command.

## Data Files

For preprocessing, you need:

- `<dataset>.item`
- `<dataset>.inter`

If you do not already have these files, download Recbole's preprocessed file:

[Google Drive folder](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI)

Place them under:

```text
dataset/<dataset>/<dataset>.item
dataset/<dataset>/<dataset>.inter
```

## Step 1: Build `id_map.json` and `item_str.json`

Use:

```bash
python3 preprocess/build_atomic_jsons.py \
  --dataset toys \
  --input_path dataset \
  --output_path dataset \
  --user_k 5 \
  --item_k 5
```

This creates:

- `dataset/<dataset>/id_map.json`
- `dataset/<dataset>/item_str.json`

Notes:

- `id_map.json` is used to align item rows in the embedding file.
- `item_str.json` is the text representation built from the `.item` file.
- `user_k` and `item_k` control the k-core filtering used while building the mapping.

## Step 2: Build `{llm_encoder}.pkl`

Use:

```bash
python3 preprocess/build_item_embeddings_from_json.py \
  --dataset toys \
  --input_path dataset \
  --output_path dataset \
  --llm_encoder openai_text_embedding_3_large \
  --openai_model text-embedding-3-large
```

This creates:

- `dataset/<dataset>/<llm_encoder>.pkl`

### OpenAI / vLLM configuration

The embedding script supports both OpenAI and OpenAI-compatible servers such as vLLM.

Use environment variables or CLI flags:

```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

Or pass them directly:

```bash
python3 preprocess/build_item_embeddings_from_json.py \
  --dataset toys \
  --input_path dataset \
  --output_path dataset \
  --llm_encoder openai_text_embedding_3_large \
  --openai_model text-embedding-3-large \
  --base_url http://127.0.0.1:8000/v1
```

Embedding vectors are saved as `float32` and L2-normalized by default.

## Optional shortcut

If preprocessing is inconvenient, prebuilt processed files may be provided on Google Drive. In that case, download the prepared dataset folder instead of rebuilding it locally.

## Step 3: Run experiments

The main experiment entry point is:

```bash
python run_recbole.py --model ACE_SASRec \
  --dataset=toys \
  --lr=0.0005 \
  --reg_weight=50 \
  --hidden_size=128 \
  --scale=0.1 \
  --text_encoder=openai_text_embedding_3_large
```

You can also check `test.sh` for a ready-to-run example.

## Common tuning knobs

The main knobs you may want to tune are:

- `lr`
- `hidden_size`
- `reg_weight`
- `scale`
- `text_encoder`

For different models, you may also want to tune:

- number of layers
- number of heads
- dropout values
- loss type

## Notes

- `build_atomic_jsons.py` and `build_item_embeddings_from_json.py` are the key preprocessing scripts for this repo.
- The RecBole model code reads `id_map.json` and `{llm_encoder}.pkl` during training.
- If you are using a local embedding server, make sure `OPENAI_BASE_URL` points to the correct `/v1` endpoint.
