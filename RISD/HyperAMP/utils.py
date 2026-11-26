import os
import sys
from pathlib import Path

import dhg
import esm
import ankh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from time import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA_DIR

# for V1: directly fitting reward function
def reward_score(value):
    if value == '/':
        score = 0
    else:
        value = float(value)
        # score = value
        score = 1 / (1+ pow(10, (value - 2)))
    return score

# data.csv: No./seq/MIC
def read_dataset(path):
    try:
        file = open(DATA_DIR / path, "r")
    except:
        print('Data path Not Found')
    X_seq = []
    target_keys = []
    y = []
    for line in file:
        values = line.split()
        target_keys.append(values[0])
        X_seq.append(values[1])
        y.append(reward_score(values[2]))
    file.close()

    df_data = pd.DataFrame(zip(target_keys, X_seq, y))
    df_data.rename(columns={0:'name', 1: 'seq', 2:'label'}, inplace=True)
    print('The number of peptides: ', len(df_data))
    return df_data

# node feature from raw esm
def get_esm(peptide_name, input_seq, model=None, alphabet=None, device=None):
    emb_path = DATA_DIR / 'esm2'
    emb_file = os.path.join(emb_path, peptide_name + '.pt')
    if os.path.exists(emb_file):
        esm = torch.load(emb_file)
        esm_emb = esm['representations'][33]
    else:
        print('ESM File Not Found')
        esm_emb = get_esm_for_test(peptide_name, input_seq, model, alphabet, device)
    return esm_emb

# directly extract esm embeddings from pretrained model
def get_esm_for_test(peptide_name, input_seq, model, alphabet, device):   
    batch_converter = alphabet.get_batch_converter()
    data = [(peptide_name, input_seq),]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    # Extract per-residue representations (on device)
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
    esm_emb = torch.squeeze(results["representations"][33])
    esm_emb = esm_emb.detach().cpu()[1:-1]
    return esm_emb

# node feature from ankh embedding (768/1536)
def get_ankh(peptide_name, input_seq, model, tokenizer, device):
    emb_path = DATA_DIR
    emb_file = os.path.join(emb_path, peptide_name + '.pt')
    if os.path.exists(emb_file):
        ankh_emb = torch.load(emb_file)
    else:
        # print('Ankh File Not Found')
        ankh_emb = get_ankh_for_test(peptide_name, input_seq, model, tokenizer, device)
    return ankh_emb

# extract Ankh-base embedding (768) directly
def get_ankh_for_test(peptide_name, input_seq, model, tokenizer, device):
    splitted_seq = list(input_seq)
    with torch.inference_mode():
        ids = tokenizer.batch_encode_plus(
            [splitted_seq], add_special_tokens=True,
            padding=True, is_split_into_words=True,
            return_tensors="pt"
        )
        embedding = model(
            input_ids=ids["input_ids"].to(device),
            attention_mask=ids["attention_mask"].to(device),
        )[0]
        embedding = embedding[0].detach().cpu().numpy()[0:-1]
    return embedding


def get_ankh_batch(peptide_names, input_seqs, model, tokenizer, device, batch_size=128):
    embeddings = [None for _ in input_seqs]
    missing_indices = []
    missing_names = []
    missing_seqs = []

    emb_path = DATA_DIR
    for i, (peptide_name, seq) in enumerate(zip(peptide_names, input_seqs)):
        emb_file = os.path.join(emb_path, peptide_name + '.pt')
        if os.path.exists(emb_file):
            embeddings[i] = torch.load(emb_file)
        else:
            missing_indices.append(i)
            missing_names.append(peptide_name)
            missing_seqs.append(seq)

    for start in range(0, len(missing_seqs), batch_size):
        batch_indices = missing_indices[start:start + batch_size]
        batch_seqs = missing_seqs[start:start + batch_size]
        split_batch = [list(seq) for seq in batch_seqs]
        with torch.inference_mode():
            ids = tokenizer.batch_encode_plus(
                split_batch,
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            batch_embedding = model(
                input_ids=ids["input_ids"].to(device),
                attention_mask=ids["attention_mask"].to(device),
            )[0]
        batch_embedding = batch_embedding.detach().cpu().numpy()
        for row_idx, seq, original_idx in zip(range(len(batch_seqs)), batch_seqs, batch_indices):
            embeddings[original_idx] = batch_embedding[row_idx, :len(seq)]

    return embeddings

# node feature from ProtT5 embedding (1024)
def get_protT5(peptide_name, input_seq):
    emb_path = DATA_DIR / 'protT5-XL-uni50'
    emb_file = os.path.join(emb_path, peptide_name + '.pt')
    if os.path.exists(emb_file):
        protT5_emb = torch.load(emb_file)
    else:
        print('ProtT5 File Not Found')
    return protT5_emb

df_k2 = pd.read_pickle(DATA_DIR / 'tf-idf' / 'k2_tfidf.pkl')
df_k3 = pd.read_pickle(DATA_DIR / 'tf-idf' / 'k3_tfidf.pkl')
df_k4 = pd.read_pickle(DATA_DIR / 'tf-idf' / 'k4_tfidf.pkl')
df_ls = [df_k2, df_k3, df_k4]
# k-mer tf-idf value
def tf_idf(peptide_name, seq, edge, k):
    df_tfidf = df_ls[k-2]
    edge = edge.lower()
    try:
        eweight = df_tfidf[edge][peptide_name]
    except:
        # print('No eweight of ' + edge.upper() + ' in ' + peptide_name)
        eweight = calc_tfidf(peptide_name, seq, edge, k)
        # eweight = 0.1   # just to avoid 0
    return eweight

# on-the-fly update tf-idf edge weights when testing
k_idf = pd.read_pickle(DATA_DIR / 'tf-idf' / 'k_idf.pkl')
def calc_tfidf(peptide_name, seq, edge, k):
    vectorizer = CountVectorizer(ngram_range=(k,k), analyzer='char')
    X = vectorizer.fit_transform([seq])
    word = vectorizer.get_feature_names_out()

    tf = TfidfTransformer(use_idf=False).fit_transform(X).toarray().tolist()
    df_tf = pd.DataFrame(tf, columns = word, index = [peptide_name])
    dic_idf = k_idf[k-2][k]
    
    tf_value = df_tf[edge][peptide_name]
    if dic_idf.get(edge, 0) == 0:
        idf_value = 1
    else:
        idf_value = dic_idf[edge]
    tfidf_value = tf_value * idf_value + 1
    return tfidf_value


# transform to hypergraph (k-mer) and get parameters
def get_hypergraph(peptide_name, seq, esm_emb):
    num = esm_emb.shape[0]
    def get_paras(k):
        elist = []
        eweight = []
        for i in range(num-k+1):
            elist.append(list(range(i, i+k)))
            edge = seq[i:i+k]
            ew = tf_idf(peptide_name, seq, edge, k) # preprocess to calculate tf-idf, save in a file
            # ew = 1
            eweight.append(ew)
        hg = dhg.Hypergraph(num, elist, eweight)
        return hg.L_HGNN
    para2 = get_paras(2)
    para3 = get_paras(3)
    para4 = get_paras(4)
    return [para2, para3, para4]

# Transform to hypergraph.
# The caller provides the local Ankh model, tokenizer, and runtime device.
class load_data(data.Dataset):
    def __init__(self, idxs, labels, df, premodel, tokenizer, device):
        super(load_data, self).__init__()
        self.labels = labels
        self.idxs = idxs
        self.df = df

        self.device = device
        self.premodel = premodel
        self.tokenizer = tokenizer
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.idxs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.idxs[index]   
        name = self.df.iloc[index]['name']
        seq = self.df.iloc[index]['seq']
        y = self.labels[index]
        emb = get_ankh(name, seq, self.premodel, self.tokenizer, self.device)   # get_esm / get_ankh
        L_HGNN = get_hypergraph(name, seq, emb)
        return emb, L_HGNN, y


def build_batched_samples(df, premodel, tokenizer, device, embedding_batch_size=128):
    names = df["name"].astype(str).tolist()
    seqs = df["seq"].astype(str).tolist()
    labels = df["label"].tolist()
    embeddings = get_ankh_batch(names, seqs, premodel, tokenizer, device, batch_size=embedding_batch_size)
    samples = []
    for name, seq, label, emb in zip(names, seqs, labels, embeddings):
        L_HGNN = get_hypergraph(name, seq, emb)
        samples.append((emb, L_HGNN, label))
    return samples

def pad_sparse(X, max_length):
    sparse = X.sparse_dim()
    dense = X.dense_dim()
    X_pad = X.sparse_resize_((max_length, max_length), sparse, dense)
    return X_pad

# collate and padding to 40
def collate_padding(batch):
    emb_list = []
    params_list = []
    lab_list = []

    for sample in batch:
        emb = torch.tensor(sample[0])
        params = sample[1]
        label = sample[2]

        max_length = 40
        seq_len = emb.shape[0]
        if seq_len < max_length:
            zero = np.zeros((max_length - seq_len, emb.shape[1]))
            zero = torch.from_numpy(zero)
            emb = torch.cat((emb, zero), 0)

        emb_list.append(emb)
        lab_list.append(label)

        para_list = []
        for para in params:
            para = pad_sparse(para, 40)

            # 关键修改：把 sparse tensor 转成 dense tensor
            para = para.to_dense()

            para_list.append(para)

        params_list.append(para_list)

    emb = torch.stack(emb_list, dim=0)
    lab = torch.tensor(lab_list)

    # params_list: [batch, 3, 40, 40]
    params_k2 = torch.stack([x[0] for x in params_list], dim=0)
    params_k3 = torch.stack([x[1] for x in params_list], dim=0)
    params_k4 = torch.stack([x[2] for x in params_list], dim=0)
    params = [params_k2, params_k3, params_k4]

    return [emb, params, lab]

# Evaluation metrics (regression)
def evalute(preds, labels):

    spearman = spearmanr(labels, preds)[0]
    pearson = pearsonr(labels, preds)[0]
    r2 = r2_score(labels, preds)
    rmse = mean_squared_error(labels, preds, squared = False) 
    # squared: If True returns MSE value, if False returns RMSE value.

    return spearman, pearson, r2, rmse
