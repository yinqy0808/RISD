import os
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import ANKH_BASE_DIR, FINETUNED_MODEL_DIR, MASKED_FINETUNE_DATA, configure_environment

configure_environment()

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

seed = 1234
torch.manual_seed(seed)
random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Available device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

learning_rate = 3e-4
weight_decay = 5e-4
batch_size = 16
epoches = 2

tokenizer = AutoTokenizer.from_pretrained(ANKH_BASE_DIR, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(ANKH_BASE_DIR, local_files_only=True)

model.gradient_checkpointing_enable()
model.config.use_cache = False
model.to(device=device)

max_length = 101

df_all = pd.read_csv(MASKED_FINETUNE_DATA)
df = df_all[["source", "target"]]


class load_data(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, source_text, target_text):
        self.tokenizer = tokenizer
        self.data = dataframe.reset_index(drop=True)
        self.max_length = max_length
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = " ".join(str(self.source_text[index]).split())
        target_text = " ".join(str(self.target_text[index]).split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_ids = target["input_ids"].squeeze(0)
        return {
            "source_ids": source["input_ids"].squeeze(0).to(dtype=torch.long),
            "source_mask": source["attention_mask"].squeeze(0).to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


train_set = load_data(df, tokenizer, max_length, "source", "target")
train_dataloader = DataLoader(
    train_set,
    shuffle=True,
    batch_size=batch_size,
    pin_memory=(device.type == "cuda"),
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

model.train()
for epoch in range(epoches):
    loss_record = []
    optimizer.zero_grad(set_to_none=True)

    for idx, data in enumerate(tqdm(train_dataloader)):
        y_ids = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y_ids.contiguous().clone().detach()
        y_ids[y_ids == tokenizer.pad_token_id] = -100

        ids = data["source_ids"].to(device, dtype=torch.long, non_blocking=True)
        mask = data["source_mask"].to(device, dtype=torch.long, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            outputs = model(input_ids=ids, attention_mask=mask, labels=y_ids)
            loss = outputs.loss

        loss_record.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if idx % 10 == 0:
            print(f"\nEpoch {epoch} loss in iteration {idx}: {loss.item():.6f}")

    with open("loss_record.txt", "a") as f:
        f.writelines(str(loss_record) + "\n")

os.makedirs(FINETUNED_MODEL_DIR, exist_ok=True)
model.save_pretrained(FINETUNED_MODEL_DIR)
tokenizer.save_pretrained(FINETUNED_MODEL_DIR)
print("Model saved to:", FINETUNED_MODEL_DIR)
