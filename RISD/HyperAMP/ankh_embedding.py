

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import ANKH_BASE_DIR, DATA_DIR, configure_environment, get_device

configure_environment()

import torch
import pathlib
import random
import ankh
import pandas as pd
from tqdm.auto import tqdm
from modlamp.core import read_fasta
from transformers import AutoTokenizer, T5EncoderModel

seed = 1234
torch.manual_seed(seed)
random.seed(seed)
device = get_device()
print('Available device:', device)

# To load Ankh model:
tokenizer = AutoTokenizer.from_pretrained(ANKH_BASE_DIR, local_files_only=True)
model = T5EncoderModel.from_pretrained(ANKH_BASE_DIR, local_files_only=True)
model.eval()
model.to(device=device)

def preprocess_dataset(sequences, max_length=None):
    if max_length is None:
        max_length = len(max(training_sequences, key=lambda x: len(x)))
    splitted_sequences = [list(seq[:max_length]) for seq in sequences]
    return splitted_sequences

def embed_dataset(model, sequences, shift_left = 0, shift_right = -1, names = None):
    # inputs_embedding = []
    output_dir = DATA_DIR / 'ankhbase' / 'QP23'
    with torch.no_grad():
        for (i, sample) in tqdm(enumerate(sequences)):
            ids = tokenizer.batch_encode_plus([sample], add_special_tokens=True, 
                                              padding=True, is_split_into_words=True, 
                                              return_tensors="pt")
            embedding = model(input_ids=ids['input_ids'].to(device))[0]
            embedding = embedding[0].detach().cpu().numpy()[shift_left:shift_right]
            # save files
            name = names[i]
            output_file = (output_dir / f"{name}.pt")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embedding, output_file)
            print('save embed files for ', name)
            # inputs_embedding.append(embedding)
'''
# for fasta input file
input_fasta = '../data/seq.fasta'
seq_name = read_fasta(input_fasta)

training_sequences = preprocess_dataset(seq_name[0], max_length = 40)
embed_dataset(model, training_sequences, names = seq_name[1])
'''
# for plain seq input file
df_seq = pd.read_csv(DATA_DIR / 'seqs.txt', sep=' ', header=None)
seq = df_seq.iloc[:, 1].values.tolist()
name = df_seq.iloc[:, 0].values.tolist()

training_sequences = preprocess_dataset(seq, max_length = 40)
embed_dataset(model, training_sequences, names = name)
