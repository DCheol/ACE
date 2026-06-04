# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import os
import numpy as np
import json
import pickle
import csv
from sklearn.decomposition import PCA
import torch.nn.functional as F
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix

class WhitenRec_SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(WhitenRec_SASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        
        ## LLM embedding load
        id_map = json.load(open(f'./dataset/{config["dataset"]}/id_map.json', "r"))["item2id"]
        loaded_feat = pickle.load(open(f'./dataset/{config["dataset"]}/{config["text_encoder"]}.pkl', "rb"))
        mapped_feat = np.zeros((self.n_items, loaded_feat.shape[1]), dtype=np.float32)
        for i, token in enumerate(dataset.field2id_token['item_id']):
            if token == '[PAD]': continue
            token_idx = int(id_map[token])-1
            mapped_feat[i] = loaded_feat[token_idx]
        self.step=0
        self.epoch=0
        self.desc = config['desc']
        self.llmemb = torch.tensor(mapped_feat[1:,:])
        self.log_filename = f"tmp/{self.desc}.csv"
        self.epoch_loggin_value = {}
        # if config["target_std"] is not None:
        #     self.target_std = config["target_std"]
        #     mapped_feat[1:,:] = (mapped_feat[1:,:] - mapped_feat[1:,:].mean()) / (mapped_feat[1:,:].std() + 1e-8) * self.target_std
            
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")



        # parameters initialization
        self.apply(self._init_weights)

        # self.item_embedding = LAE_Item_Embedding(config, mapped_feat, self.n_items, self.hidden_size, scale=config['scale'], padding_idx=0, )
        # self.item_embedding = Graph_Filtered_Item_Embedding(config, mapped_feat.T, self.n_items, self.hidden_size, scale=config['scale'], padding_idx=0, )
        self.item_embedding = WhitenRec_Item_Embedding(config, mapped_feat, self.n_items, self.hidden_size, scale=config['scale'], padding_idx=0, )


    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif module is self.position_embedding:
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        # from IPython import embed; embed()
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
            self.step += 1
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
    # def full_sort_predict(self, interaction, positive_i, positive_u):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]

        return scores
    

class WhitenRec_Item_Embedding(torch.nn.Module):
    def __init__(self, config, F, item_num, hidden_size, scale=1, padding_idx=0, ):
        super(WhitenRec_Item_Embedding, self).__init__()
        print("WhitenRec_Item_Embedding init")
        self.initializer_range  = config["initializer_range"]
        # from IPython import embed; embed()
        
        # if config["Centerise"]==True:
        F = F[1:]
        F = F - F.mean(axis=0, keepdims=True)
                
        # from IPython import embed; embed()
        
        self.dataset_name = config["dataset"]
        self.hidden_size = config["hidden_size"]

        # np.save(f'./tmp/{self.dataset_name}_F_centerized_U.npy', U)
        # np.save(f'./tmp/{self.dataset_name}_F_centerized_S.npy', S)
        # np.save(f'./tmp/{self.dataset_name}_F_centerized_Vt.npy', Vt)
        if os.path.exists(f'./tmp/{self.dataset_name}_F_centerized_U.npy'):
            print("Loading precomputed SVD components...")
            U = np.load(f'./tmp/{self.dataset_name}_F_centerized_U.npy')
            S = np.load(f'./tmp/{self.dataset_name}_F_centerized_S.npy')
            Vt = np.load(f'./tmp/{self.dataset_name}_F_centerized_Vt.npy')
        else:
            U, S, Vt = np.linalg.svd(F, full_matrices=False)
            np.save(f'./tmp/{self.dataset_name}_F_centerized_U.npy', U)
            np.save(f'./tmp/{self.dataset_name}_F_centerized_S.npy', S)
            np.save(f'./tmp/{self.dataset_name}_F_centerized_Vt.npy', Vt)
        # Apply ZCA transform
        # X = U@Vt
        N = F.shape[0]
        X = U@Vt
        # # Apply PCA transform
        # X = U

        llm_dim = X.shape[1]

        self.item_embedding = nn.Embedding(
            # item_num, hidden_size, padding_idx=0
            item_num, llm_dim, padding_idx=0
        )        
        X_full = torch.zeros((item_num, llm_dim), dtype=torch.float,
                            device=self.item_embedding.weight.device)
        X_full[1:] = torch.from_numpy(F).float()

        with torch.no_grad():
            self.item_embedding.weight.copy_(X_full)
            self.item_embedding.weight.requires_grad = False

        self.adapter = nn.Sequential(
            nn.Linear(llm_dim, int(llm_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(llm_dim / 2), self.hidden_size))
        

    ### @property
    def forward(self, item_ids):
        item_embedding = self.item_embedding(item_ids)
        item_embedding = self.adapter(item_embedding)
        return item_embedding
    
    @property
    def weight(self):
        # 0 ~ item_num (padding 포함)까지 한번에 임베딩 조회
        all_ids = torch.arange(self.item_embedding.num_embeddings, device=self.item_embedding.weight.device)
        
        item_embedding = self.forward(all_ids)
        return item_embedding
