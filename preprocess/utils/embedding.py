import os, torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

def generate_item_sparse_features(item_metadata_dict, item2idx, word2idx, feature_list):
    rows, cols, data = [], [], []

    for item_id, features in item_metadata_dict.items():
        for feature in features.keys():
            if feature not in feature_list:
                continue

            feature_value = features[feature]
            words = set()  # To avoid duplicates per item

            if isinstance(feature_value, str):
                words = set(feature_value.lower().split())
            elif isinstance(feature_value, list):
                for v in feature_value:
                    if isinstance(v, list):
                        words.update(w.lower() for w in v)
                    else:
                        words.add(v.lower())
                print("feature as list")
            elif isinstance(feature_value, (int, float)):
                words = {str(feature_value)}

            for word in words:
                if word in word2idx:
                    rows.append(item2idx[item_id])
                    cols.append(word2idx[word])
                    data.append(1)  # Multi-hot encoding
        
        return rows, cols, data


def generate_item_nvemb_embedding_with_metadata(item_metatext_dict, item2index, text_encoder, dataset, output_path, batch_size=5,
                                                features=["title", "category", "brand", "description"]):    
    items = list(item_metatext_dict.keys())
    texts = list(item_metatext_dict.values())
    # from IPython import embed ; embed()
    
    # order_texts = [[0]] * len(items)
    order_texts = [[0]] * len(item2index)
    # int_item2index = {int(k):int(v) for k, v in item2index.items()}
    for item_id, text in zip(items, texts):
        try:
            order_texts[int(item2index[item_id])] = text
            # order_texts[int(int_item2index[item_id])] = text

        except:
            pass


    for idx, item_id in enumerate(order_texts):
        if item_id == [0]:
            print(idx)
        assert item_id != [0]
    
    query_prefix_text = ""
    query_prefix_list = []

    for feature in features:
        str_1 = f"{feature}: " + "{" + feature+"}"
        query_prefix_list.append(str_1)

    query_prefix_text = " ; ".join(query_prefix_list)
    query_prefix = f"Produce an embedding of this item that can be used to recommend similar or complementary items. \
                    The item's information includes: \n{query_prefix_text}"
    
    embeddings = []
    start = 0

    order_texts_len = len(order_texts)
    with torch.no_grad():
        with tqdm(total=order_texts_len, desc="Processing", unit="item") as pbar:
            while start < order_texts_len:
                sentences = order_texts[start: start + batch_size]
                torch.cuda.empty_cache()

                query_embeddings = text_encoder.encode(sentences, instruction=query_prefix, max_length=32768) # max_length=32768
                query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
                tensor_embeddings = query_embeddings.detach().cpu().numpy()
                
                embeddings.append(tensor_embeddings)

                del query_embeddings
                
                start += batch_size
                pbar.update(batch_size)
                
        embeddings = np.concatenate(embeddings, axis=0)

        file_prefix = os.path.join(output_path, dataset)
        file = os.path.join(file_prefix, f'nvembv2.{"_".join(features)}')
        # file = os.path.join(file_prefix, f'sbert.{"_".join(features)}')
        embeddings.tofile(file)

    del embeddings


def generate_item_sbert_embedding_with_metadata(item_metatext_dict, item2index, text_encoder, dataset, output_path, batch_size=5,
                                                features=["title", "category", "brand", "description"]):    
    items = list(item_metatext_dict.keys())
    texts = list(item_metatext_dict.values())
    
    # order_texts = [[0]] * len(items)
    order_texts = [[0]] * len(item2index)
    # int_item2index = {int(k):int(v) for k, v in item2index.items()}
    for item_id, text in zip(items, texts):
        try:
            order_texts[int(item2index[item_id])] = text
            # order_texts[int(int_item2index[item_id])] = text

        except:
            pass


    for idx, item_id in enumerate(order_texts):
        if item_id == [0]:
            print(idx)
        assert item_id != [0]
    
    query_prefix_text = ""
    query_prefix_list = []

    for feature in features:
        str_1 = f"{feature}: " + "{" + feature+"}"
        query_prefix_list.append(str_1)

    # query_prefix_text = " ; ".join(query_prefix_list)
    # query_prefix = f"Produce an embedding of this item that can be used to recommend similar or complementary items. \
    #                 The item's information includes: \n{query_prefix_text}"
    
    embeddings = []
    start = 0

    order_texts_len = len(order_texts)
    with torch.no_grad():
        with tqdm(total=order_texts_len, desc="Processing", unit="item") as pbar:
            while start < order_texts_len:
                sentences = order_texts[start: start + batch_size]
                torch.cuda.empty_cache()

                query_embeddings = text_encoder.encode(sentences, convert_to_tensor=True, max_length=32768) # max_length=32768
                query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
                tensor_embeddings = query_embeddings.detach().cpu().numpy()
                
                embeddings.append(tensor_embeddings)

                del query_embeddings
                
                start += batch_size
                pbar.update(batch_size)
                
        embeddings = np.concatenate(embeddings, axis=0)

        file_prefix = os.path.join(output_path, dataset)
        # file = os.path.join(file_prefix, f'nvembv2.{"_".join(features)}')
        file = os.path.join(file_prefix, f'sbert.{"_".join(features)}')
        embeddings.tofile(file)

    del embeddings

def generate_item_nvemb_embedding_with_metadata_another_instruction(args, item_text_list, additional_text_dict, item2index, sbert_model, batch_size=5, comb=["title", "category", "brand", "description"], instruction=0 ,word_drop_ratio=-1):
    abbreviate_comb = {"title":"T", "category":"C", "brand":"B", "description":"D"}
    abbreviate_str = ""
    
    items = []
    texts = []
    for item, texts_dict in additional_text_dict.items():
        text = []
        # print(texts_dict)
        for k in texts_dict.keys():
            text.append(f"{k}: {texts_dict[k]}")
        # print(texts_dict.keys())
        text = " ; ".join(text)
        items.append(item)
        texts.append(text)
    
    # a,b = zip(*additional_text_dict)
    
    # # based on rating_inters
    # items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    # based on item set
    for item, text in zip(items, texts):
        # print(item2index[item])
        order_texts[int(item2index[item])] = text
    for text in order_texts:
        assert text != [0]

    # query_prefix_text = "title: {title} ; categories: {categories} ; brand: {brand}"
    # query_prefix = f"Generate an embedding that captures the overall characteristics of the item for personalized recommendation. The item's information includes: \n{query_prefix_text}"
    
    query_prefix_list=[]
    query_prefix_text=""
    for c in comb:
        abbreviate_str += abbreviate_comb[c]
        str_1 = f"{c}: "+"{" +c+"}"
        query_prefix_list.append(str_1)
        
    query_prefix_text = " ; ".join(query_prefix_list)
    if instruction == 0:
        query_prefix = f"The item's information includes: \n{query_prefix_text}"
    elif instruction == 1:
        query_prefix = f"Generate an embedding that captures the overall characteristics of the item for personalized recommendation. The item's information includes: \n{query_prefix_text}"
    elif instruction == 2:
        query_prefix = ""
    elif instruction == 3:
        query_prefix = f"Generate an embedding that captures the overall characteristics of the item for personalized recommendation. Prioritize encoding the item's title as the most important feature, while using other information as supplementary context. The item's information includes: \n{query_prefix_text}"
    
    
    embeddings = []
    start = 0
    
    # cur = 0 
    # while start < len(order_texts):
    #     if start/len(order_texts)*100 > cur:
    #         cur += 1
    #         print("*", end=" ")
        
    #     sentences = order_texts[start: start + batch_size]
    order_texts_len = len(order_texts)
    with tqdm(total=order_texts_len, desc="Processing", unit="item") as pbar:
        while start < order_texts_len:
            sentences = order_texts[start: start + batch_size]
            
            # 처리 로직 (예: 모델 입력, 변환, 저장 등)
            query_embeddings = sbert_model.encode(sentences, instruction=query_prefix, max_length=4096)
            query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
            tensor_embeddings = query_embeddings.detach().cpu()
            # tensor_embeddings = torch.tensor(batch_embeddings).detach().cpu()
            
            embeddings.append(tensor_embeddings)
            
            start += batch_size
            pbar.update(batch_size)  # tqdm 진행률 업데이트
            
            
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('\nEmbeddings shape: ', embeddings.shape)

    file_prefix = os.path.join(args.output_path, args.dataset)
    file = os.path.join(file_prefix, f'nvemb.{instruction}_{abbreviate_str}')
    embeddings.tofile(file)


import os, re, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction import text
from scipy.sparse import lil_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import os, json
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction import text   # scikit‑learn 기본 불용어

def generate_item_multihot_embedding(
    item_metatext_dict: dict,
    item2index: dict,
    dataset: str,
    output_path: str,
    min_df: int = 2,
    stop_words = None,    # 불용어 set
):
    """
    공백 단위(split) 토큰화 → stopword 제거 → multi‑hot(uint8) 생성 → .bin + .shape.json 저장
    """
    from sklearn.feature_extraction import text 
    # stop_words = text.ENGLISH_STOP_WORDS
    # 0) 불용어 세팅
    if stop_words is None:
        stop_words = set(text.ENGLISH_STOP_WORDS)
    stop_words = {w.lower() for w in stop_words}

    # 1) item 순서대로 텍스트 배열
    items = list(item_metatext_dict.keys())
    texts = list(item_metatext_dict.values())
    # from IPython import embed ; embed()
    
    # order_texts = [[0]] * len(items)
    order_texts = [[0]] * len(item2index)
    # int_item2index = {int(k):int(v) for k, v in item2index.items()}
    for item_id, text in zip(items, texts):
        try:
            order_texts[int(item2index[item_id])] = text
            # order_texts[int(int_item2index[item_id])] = text

        except:
            pass

    for idx, item_id in enumerate(order_texts):
        if item_id == [0]:
            print(idx)
        assert item_id != [0]

    # 2) vocab 수집 (공백 기준 + stopword 제거 + min_df 적용)
    counter = Counter()
    for txt in order_texts:
        tokens = set(txt.split()) - stop_words
        counter.update(tokens)

    vocab = {tok for tok, c in counter.items() if c >= min_df}
    vocab2idx = {tok: i for i, tok in enumerate(sorted(vocab))}
    V, N = len(vocab2idx), len(order_texts)
    print(f"[multi‑hot] vocab={V:,}, items={N:,}")

    # 3) multi‑hot dense 행렬
    X = np.zeros((N, V), dtype=np.float32)
    for r, txt in enumerate(tqdm(order_texts, desc="build multi‑hot")):
        for tok in set(txt.split()) - stop_words:
            c = vocab2idx.get(tok)
            if c is not None:
                X[r, c] = 1

    # 4) 저장 (.bin + .shape.json)
    file_prefix = os.path.join(output_path, dataset)
    os.makedirs(file_prefix, exist_ok=True)

    bin_path   = os.path.join(file_prefix, f"multihot_minDf{min_df}.bin")
    shape_path = bin_path + ".shape.json"

    X.tofile(bin_path)
    with open(shape_path, "w") as f:
        json.dump({"rows": N, "cols": V, "dtype": "uint8"}, f)

    print("✓ multi‑hot saved:", bin_path)
    print("✓ shape saved  :", shape_path)
    # multi_hot = np.fromfile(bin_path, dtype=meta["dtype"]).reshape(meta["rows"], meta["cols"])

import os, json
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_item_tfidf_embedding(
    item_metatext_dict: dict,     # {item_id: "combined text"}
    item2index: dict,             # {item_id: idx}
    dataset: str,
    output_path: str,
    max_features: int = 50000,    # 상위 N개 단어만 사용
    stop_words = None,  # stopword set | "english" | None
):
    """
    공백 기준 토큰화 → stopword 제거 → TF‑IDF(float32) 행렬 생성
    저장:  <output_path>/<dataset>/tfidf_{max_features}.bin
          + .shape.json  (rows, cols, dtype)
          + .vocab.json  (token → col‑idx)
    """

    # 0) stopword 세팅
    from sklearn.feature_extraction import text 

    if stop_words is None:
        stop_words = set(text.ENGLISH_STOP_WORDS)      # 영어 기본
    elif isinstance(stop_words, str):
        # e.g. "english" 그대로 넘긴 경우
        stop_words = set(text.ENGLISH_STOP_WORDS) if stop_words.lower() == "english" else set()
    else:
        stop_words = {w.lower() for w in stop_words}

    # -------- 1. item 순서 맞춰 텍스트 배열 --------
    # 1) item 순서대로 텍스트 배열
    items = list(item_metatext_dict.keys())
    texts = list(item_metatext_dict.values())
    # from IPython import embed ; embed()
    
    # order_texts = [[0]] * len(items)
    order_texts = [[0]] * len(item2index)
    # int_item2index = {int(k):int(v) for k, v in item2index.items()}
    for item_id, text in zip(items, texts):
        try:
            order_texts[int(item2index[item_id])] = text
            # order_texts[int(int_item2index[item_id])] = text

        except:
            pass

    for idx, item_id in enumerate(order_texts):
        if item_id == [0]:
            print(idx)
        assert item_id != [0]

    # -------- 2. whitespace tokenizer 함수 --------
    whitespace_tokenizer = lambda s: [tok for tok in s.split() if tok not in stop_words]

    vectorizer = TfidfVectorizer(
        tokenizer=whitespace_tokenizer,      # 공백 기준
        preprocessor=lambda x: x,            # 이미 소문자 처리함
        lowercase=False,                     # 추가 소문자화 없음
        max_features=max_features,
        dtype=np.float32,
        norm="l2",                           # 기본 L2 정규화
    )

    print(f"[tf‑idf] fitting… (max_features={max_features})")
    X_sparse = vectorizer.fit_transform(tqdm(order_texts, desc="tf‑idf fit"))
    X = X_sparse.toarray().astype(np.float32)   # dense → float32

    N, V = X.shape
    print(f"[tf‑idf] shape = {N} × {V}")

    # -------- 3. 저장 (.bin + .shape.json + .vocab.json) --------
    file_prefix = os.path.join(output_path, dataset)
    os.makedirs(file_prefix, exist_ok=True)

    bin_path   = os.path.join(file_prefix, f"tfidf_{max_features}.bin")
    shape_path = bin_path + ".shape.json"
    vocab_path = bin_path + ".vocab.json"

    X.tofile(bin_path)
    with open(shape_path, "w") as f:
        json.dump({"rows": N, "cols": V, "dtype": "float32"}, f)
    # with open(vocab_path, "w") as f:
    #     json.dump(vectorizer.vocabulary_, f)

    print("✓ tf‑idf saved :", bin_path)
    print("✓ shape saved :", shape_path)
    print("✓ vocab saved :", vocab_path)



class ImageDataset(Dataset):
    def __init__(self, paths, proc):
        self.paths = paths
        self.proc  = proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if path is None or not os.path.exists(path):
            # 이미지가 없을 땐 검은색 placeholder 이미지를 생성
            # (224×224 사이즈는 processor 설정에 따라 조정하세요)
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        else:
            img = Image.open(path).convert("RGB")
            
        # HuggingFace processor를 사용해 [1,3,224,224] 텐서 얻기
        proc_out = self.proc(images=img, return_tensors="pt")
        pixel_values = proc_out["pixel_values"].squeeze(0)  # → [3,224,224]
        return pixel_values, idx
        # return img, idx
        # img = Image.open(self.paths[idx]).convert("RGB")
        # return img, self.paths[idx]
    
    
    
    
def generate_img_embedding(item_metatext_dict, item2index, model, processor, dataset, output_path, batch_size=5):    
    items = list(item_metatext_dict.keys())
    img_path = list(item_metatext_dict.values())
    

    # order_url = [[0]] * len(items)
    # order_url = [[0]] * len(item2index)
    order_url = [None] * len(item2index)
    # int_item2index = {int(k):int(v) for k, v in item2index.items()}
    # from IPython import embed ; embed()
    for item_id, text in zip(items, img_path):
        try:
            order_url[int(item2index[item_id])] = text
            # order_url[int(int_item2index[item_id])] = text
        except:
            pass

    
    missing = [i for i,p in enumerate(order_url) if p is None]
    if missing:
        print(f"[Warning] 이미지가 없는 인덱스 {len(missing)}개: {missing[:5]} …")
    
    
    img_dataset = ImageDataset(order_url, processor)
    loader  = DataLoader(img_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    embeddings_list = []
    model.eval()
    device = next(model.parameters()).device
    num_batches = len(loader)
    with torch.no_grad():
        for pixel_batch, idxs in tqdm(loader, desc="Extracting embeddings", total=num_batches):
            # pixel_batch: [B,3,224,224]
            pixel_batch = pixel_batch.to(device)
            outputs = model(pixel_values=pixel_batch)
            embs = outputs.last_hidden_state[:, 0]          # [B, D]
            embs = embs / embs.norm(p=2, dim=-1, keepdim=True)
            embeddings_list.append(embs.cpu().numpy())
            
    # 4) 하나의 (N, D) 배열로 합침
    embeddings = np.concatenate(embeddings_list, axis=0)  # shape == (n_items, D)

    # from IPython import embed ; embed()
    file_prefix = os.path.join(output_path, dataset)
    # file = os.path.join(file_prefix, f'emb3large.{"_".join(features)}')
    file = os.path.join(file_prefix, f'dinov2.img')
    # file = os.path.join(file_prefix, f'ada2.{"_".join(features)}')
    # file = os.path.join(file_prefix, f'sbert.{"_".join(features)}')
    embeddings.tofile(file)

    del embeddings