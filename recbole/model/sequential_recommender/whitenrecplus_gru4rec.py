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

import os
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


class WhitenRecPlus_GRU4Rec(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, dataset):
        super(WhitenRecPlus_GRU4Rec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
       
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
        self.item_embedding = WhitenRecPlus_Item_Embedding(config, mapped_feat, self.n_items, self.embedding_size, scale=None, padding_idx=0, )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
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
            item_ids = torch.arange(self.n_items, device=seq_output.device)
            test_item_emb = self.item_embedding(item_ids)
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
        item_ids = torch.arange(self.n_items, device=seq_output.device)
        test_items_emb = self.item_embedding(item_ids)
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores


class WhitenRecPlus_Item_Embedding(torch.nn.Module):
    def __init__(self, config, F, item_num, hidden_size, scale=1, padding_idx=0, ):
        super(WhitenRecPlus_Item_Embedding, self).__init__()
        print("WhitenRecPlus_Item_Embedding init")
        self.initializer_range  = config["initializer_range"]
        self.group_num = config["group_num"]
        # from IPython import embed; embed()
        
        # if config["Centerise"]==True:
        F = F[1:]
        F = F - F.mean(axis=0, keepdims=True)
                
        # from IPython import embed; embed()
        
        self.dataset_name = config["dataset"]
        self.hidden_size = config["hidden_size"] 
        
        U, S, Vt = np.linalg.svd(F, full_matrices=False)
        
        # Apply ZCA transform
        N = F.shape[0]
        llm_dim = F.shape[1]
        ZCA_full = U@Vt
        ZCA_group = self.group_zca_whitening(F, group_num=self.group_num)

        self.item_embedding_full = nn.Embedding(item_num, llm_dim, padding_idx=0)
        self.item_embedding_group = nn.Embedding(item_num, llm_dim, padding_idx=0)

        X_full = torch.zeros((item_num, llm_dim), dtype=torch.float32)
        X_group = torch.zeros((item_num, llm_dim), dtype=torch.float32)

        X_full[1:] = torch.from_numpy(ZCA_full).float()
        X_group[1:] = torch.from_numpy(ZCA_group).float()

        # # Apply PCA transform
        # X = U

        with torch.no_grad():
            self.item_embedding_full.weight.copy_(X_full)
            self.item_embedding_group.weight.copy_(X_group)
            self.item_embedding_full.weight.requires_grad = False
            self.item_embedding_group.weight.requires_grad = False

        # === 4️⃣ Shared MLP adapter (projection head) ===
        self.adapter = nn.Sequential(
            nn.Linear(llm_dim, int(llm_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(llm_dim / 2), self.hidden_size)
        )



    ### @property
    def forward(self, item_ids):
        emb_full = self.item_embedding_full(item_ids)
        emb_group = self.item_embedding_group(item_ids)
        
        # Pass both through shared adapter and sum
        item_embedding = self.adapter(emb_full) + self.adapter(emb_group)
        return item_embedding
    
    @property
    def weight(self):
        # 0 ~ item_num (padding 포함)까지 한번에 임베딩 조회
        all_ids = torch.arange(self.item_embedding_full.num_embeddings, device=self.item_embedding_full.weight.device)
        item_embedding = self.forward(all_ids)
        return item_embedding

    def group_zca_whitening(self, F, group_num, eps=1e-5):
        """Split features into G groups and apply ZCA whitening per group"""
        N, D = F.shape
        assert D % group_num == 0, "D must be divisible by G"
        gdim = D // group_num
        outs = []
        for g in range(group_num):
            start = g * gdim
            end = (g + 1) * gdim
            Fg = F[:, start:end]
            Ug, Sg, Vtg = np.linalg.svd(Fg, full_matrices=False)
            Fg_white = Ug @ Vtg  # ZCA within group
            outs.append(Fg_white)
        F_out = np.concatenate(outs, axis=1)
        return F_out