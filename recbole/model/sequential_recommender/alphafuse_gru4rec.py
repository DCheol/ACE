# -*- coding: utf-8 -*-
# @Time   : 2020/8/17 19:38
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE:
# @Time   : 2020/8/19, 2020/10/2
# @Author : Yupeng Hou, Yujie Lu
# @Email  : houyupeng@ruc.edu.cn, yujielu1998@gmail.com

r"""
GRU4Rec
################################################

Reference:
    Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.

"""

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

import numpy as np
import json
import pickle
from sklearn.decomposition import PCA
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class Alphafuse_GRU4Rec(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, dataset):
        super(Alphafuse_GRU4Rec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        self.init_type = config["init_type"]

        # define layers and loss
        
        ## LLM embedding load
        id_map = json.load(open(f'./dataset/{config["dataset"]}/id_map.json', "r"))["item2id"]
        loaded_feat = pickle.load(open(f'./dataset/{config["dataset"]}/{config["text_encoder"]}.pkl', "rb"))
        mapped_feat = np.zeros((self.n_items, loaded_feat.shape[1]), dtype=np.float32)
        for i, token in enumerate(dataset.field2id_token['item_id']):
            if token == '[PAD]': continue
            token_idx = int(id_map[token])-1
            mapped_feat[i] = loaded_feat[token_idx]
        
        
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        self.item_embedding = Item_Embedding(mapped_feat, self.n_items, self.embedding_size, init_type=self.init_type, ID_dim=int(self.hidden_size/2), scale=None, padding_idx=0)
        

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding)):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output
    
    
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores

    
class Item_Embedding(torch.nn.Module):
    def __init__(self, language_embs, item_num, embedding_size, init_type="uniform", ID_dim=64, scale=None, padding_idx=0):
        super(Item_Embedding, self).__init__()
        print("Item_Embedding init")
        
        cal_scale = 40
        self.nullity = ID_dim
        print(f"self.nullity: {self.nullity}, embedding_size: {embedding_size}, init_type: {init_type}, scale: {scale}")
        ### init ID embeddings ###
        ### init ID embeddings ###
        ### init ID embeddings ###
        self.ID_embeddings = nn.Embedding(
            num_embeddings=item_num+1, 
            # embedding_dim=embedding_size-ID_dim,
            embedding_dim=ID_dim,
            # padding_idx=0
        )
        if init_type == "uniform":
            nn.init.uniform_(self.ID_embeddings.weight, a=0.0, b=1.0)
        elif init_type == "normal":
            nn.init.normal_(self.ID_embeddings.weight, 0, 1)
        elif init_type == "zeros":
            nn.init.zeros_(self.ID_embeddings.weight)
        elif init_type == "ortho":
            nn.init.orthogonal_(self.ID_embeddings.weight, gain=1.0)
        elif init_type == "xavier":
            nn.init.xavier_uniform_(self.ID_embeddings.weight, gain=1.0)
        elif init_type == "sparse":
            nn.init.sparse_(self.ID_embeddings.weight, 0.01, std=1)
        else:
            raise NotImplementedError("This kind of init for ID embeddings is not implemented yet.")
        
        ### init LLM embeddings ###
        ### init LLM embeddings ###
        ### init LLM embeddings ###
        language_embs = language_embs[1:,:] * cal_scale
        self.language_mean = np.mean(language_embs, axis=0)
        
        cov = np.cov(language_embs - self.language_mean, rowvar=False)
        U, S, _ = np.linalg.svd(cov, full_matrices=False)
        
        Projection_matrix = U[...,:embedding_size]
        Diagnals = np.sqrt(1/S)[:embedding_size]
        # Diagnals = 0.1*np.sqrt(1/S)[:embedding_size]
        
        Projection_matrix = Projection_matrix.dot(np.diag(Diagnals)) # V_{\lamda} into V_1
        clipped_language_embs = (language_embs-self.language_mean).dot(Projection_matrix)
    
        padding_emb = np.random.rand(clipped_language_embs.shape[1])  # padding ID embedding, padding_idx=0
        clipped_language_embs = np.vstack([padding_emb, clipped_language_embs]) # (self.item_num, 128)
        self.language_embeddings = torch.nn.Embedding.from_pretrained(
            torch.tensor(clipped_language_embs,dtype=torch.float32),
            freeze=True,
            padding_idx=0
        )
        
        
        
    ### @property
    def __call__(self, item_ids):
        language_embs = self.language_embeddings(item_ids)
        ID_embs = self.ID_embeddings(item_ids)
        fuse_embs = language_embs.clone()
        fuse_embs[...,-self.nullity:] = language_embs[...,-self.nullity:] + ID_embs
        return fuse_embs
    
    @property
    def weight(self):
        # 0 ~ item_num (padding 포함)까지 한번에 임베딩 조회
        all_ids = torch.arange(self.language_embeddings.num_embeddings, device=self.language_embeddings.weight.device)
        return self(all_ids)  # __call__(all_ids)를 통해 fuse된 임베딩 행렬 반환