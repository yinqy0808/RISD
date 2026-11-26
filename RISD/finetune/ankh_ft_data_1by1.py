import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from modlamp.core import read_fasta

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA_DIR, MASKED_FINETUNE_DATA

random.seed(1234)

input_file = DATA_DIR / '9-7316.fasta'
seq_name = read_fasta(input_file)
seqs = seq_name[0]

masking_ratio = 0.2
seqs_mask = []
rev_mask = []

for i, seq in enumerate(seqs):
    n_mask = round(len(seq) * masking_ratio)
    seqls = list(seq)
    seqid = list(range(len(seqls)))
    mask_aa = sorted(random.sample(seqid, n_mask))

    rev_seqls = []       # initial reverse masking seq
    for k, n in enumerate(mask_aa):
        n = int(n)
        rev_seqls.append(f"<extra_id_{k}>")
        rev_seqls.append(seqls[n])
        seqls[n] = f"<extra_id_{k}>"    # replace masked aa to masking token
    seqs_mask.append(''.join(token for token in seqls))
    rev_mask.append(''.join(token for token in rev_seqls))

# save as df
df = pd.DataFrame(zip(seqs, seqs_mask, rev_mask), columns=['seq', 'source', 'target'])
df.to_csv(MASKED_FINETUNE_DATA, index=False)
