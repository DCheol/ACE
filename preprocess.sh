# preprocessing. make item_id.json & item_str.json
python preprocess/build_atomic_jsons.py \
    --dataset beauty \
    --input_path dataset \
    --output_path dataset \
    --user_k 5 \
    --item_k 5
   # --item_fields


# OpenAI
python preprocess/build_item_embeddings_from_json.py \
  --dataset beauty \
  --input_path dataset \
  --output_path dataset \
  --llm_encoder openai_text_embedding_3_large \
  --openai_model text-embedding-3-large

# # vLLM
# python preprocess/build_item_embeddings_from_json.py \
#   --dataset beauty \
#   --input_path dataset \
#   --output_path dataset \
#   --llm_encoder your-embed-model-name \
#   --openai_model your-embed-model-name \
#   --base_url http://localhost:8000/v1