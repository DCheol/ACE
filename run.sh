python run_recbole.py --model ACE_SASRec \
    --dataset=beauty \
    --lr=0.0005 \
    --reg_weight=50 \
    --hidden_size=128 \
    --scale=0.1 \
    --aug=False \
    --text_encoder="openai_text_embedding_3_large"