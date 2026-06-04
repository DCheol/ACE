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
import json
import torch
import numpy as np
import os
import pickle
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class ACE_GRU4Rec(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, dataset):
        super(ACE_GRU4Rec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
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


        id_map = json.load(open(f'./dataset/{config["dataset"]}/id_map.json', "r"))["item2id"]
        loaded_feat = pickle.load(open(f'./dataset/{config["dataset"]}/{config["text_encoder"]}.pkl', "rb"))
        mapped_feat = np.zeros((self.n_items, loaded_feat.shape[1]), dtype=np.float32)
        for i, token in enumerate(dataset.field2id_token['item_id']):
            if token == '[PAD]': continue
            token_idx = int(id_map[token])-1
            mapped_feat[i] = loaded_feat[token_idx]
        self.item_embedding = ACE_Item_Embedding(config, mapped_feat, self.n_items, self.hidden_size, scale=config['scale'], padding_idx=0,)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
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


class ACE_Item_Embedding(torch.nn.Module):
    def __init__(self, config, F, item_num, hidden_size, scale=1, padding_idx=0, ):
        super(ACE_Item_Embedding, self).__init__()
        print("ACE_Item_Embedding init")
        # from IPython import embed; embed()
        
        pca_desc = ""
        if config["Centerise"]==True:
            F_mean = F[1:].mean(axis=0, keepdims=True)
            F[1:] = F[1:] - F_mean
            pca_desc = "_PCA"
        
        self.trade_off = config["trade_off"]
        self.reg_weight = config["reg_weight"]
        
        
        
        self.dataset_name = config["dataset"]
        self.hidden_size = config["hidden_size"] 

        self.dataset_name = config["dataset"]
        self.hidden_size = config["hidden_size"] 
        
        if not os.path.exists(f'./tmp/{self.dataset_name}_F_U.npy'):
            F = F[1:]
            U, S, Vt = np.linalg.svd(F, full_matrices=False)
            np.save(f'./tmp/{self.dataset_name}_F_U.npy', U)
            np.save(f'./tmp/{self.dataset_name}_F_S.npy', S)
            np.save(f'./tmp/{self.dataset_name}_F_Vt.npy', Vt)
        else:
            print("load saved SVD files")
            U = np.load(f'./tmp/{self.dataset_name}_F_U.npy')
            S = np.load(f'./tmp/{self.dataset_name}_F_S.npy')
            Vt = np.load(f'./tmp/{self.dataset_name}_F_Vt.npy')
        
        U = U[:, :self.hidden_size]
        S = S[:self.hidden_size]
        
        # X = U*S/np.sqrt(F.shape[0]-1)
        # from IPython import embed; embed()
        if self.reg_weight < 0:
            X = U*S
        else:
            X = U*np.sqrt(S**2/(S**2 + self.reg_weight))

        if config["norm"]:
            X_scaled = (X) / (X.std() + 1e-6) * scale
        else:
            X_scaled = X
            # X_scaled = X * scale
        
        # from IPython import embed; embed()
        print(f"S.sum(): {S.sum():.6f}")
        print(f"X_scaled.mean(): {X_scaled.mean():.6f}")
        print(f"X_scaled.std(): {X_scaled.std():.6f}")
        print(f"1/X.std(): {(1/X.std()):.6f}")
        print(f"X_scaled.norm: {(np.linalg.norm(X_scaled)):.6f}")
        print(f"X_scaled.norm.mean(): {(np.linalg.norm(X_scaled, axis=1).mean()):.6f}")
        
        self.item_embedding = nn.Embedding(
            item_num, hidden_size, padding_idx=0
        )        
        X_full = torch.zeros((item_num, self.hidden_size), dtype=torch.float,
                            device=self.item_embedding.weight.device)
        X_full[1:] = torch.from_numpy(X_scaled).float()
        with torch.no_grad():
            self.item_embedding.weight.copy_(X_full)
        
    ### @property
    def __call__(self, item_ids):
        item_embedding = self.item_embedding(item_ids)
        return item_embedding
    
    @property
    def weight(self):
        # 0 ~ item_num (padding 포함)까지 한번에 임베딩 조회
        all_ids = torch.arange(self.item_embedding.num_embeddings, device=self.item_embedding.weight.device)
        
        item_embedding = self(all_ids)
        return item_embedding