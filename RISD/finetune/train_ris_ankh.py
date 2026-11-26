#!/usr/bin/env python3
import argparse
import os
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    ANKH_BASE_DIR,
    RIS_FINETUNE_DATA,
    RIS_FINETUNED_MODEL_DIR,
    SEED,
    configure_environment,
    get_device,
)

configure_environment()

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


SPECIAL_TOKENS = ["<replace>", "<insert>"]


class RISFineTuneDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        source_text = " ".join(str(row["source"]).split())
        target_text = " ".join(str(row["target"]).split())

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

        return {
            "source_ids": source["input_ids"].squeeze(0).to(dtype=torch.long),
            "source_mask": source["attention_mask"].squeeze(0).to(dtype=torch.long),
            "target_ids": target["input_ids"].squeeze(0).to(dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune local Ankh for RIS replace/insert infilling.")
    parser.add_argument("--data", type=Path, default=RIS_FINETUNE_DATA)
    parser.add_argument("--base-model", type=Path, default=ANKH_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=RIS_FINETUNED_MODEL_DIR)
    parser.add_argument("--max-length", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()

    if args.log_file is None:
        args.log_file = args.output_dir / "train_ris_ankh.log"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"Loading base Ankh model: {args.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, local_files_only=True)
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added special tokens: {SPECIAL_TOKENS}")

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device=device)

    df = pd.read_csv(args.data)
    required_cols = {"source", "target", "task"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.data}: {sorted(missing)}")
    print(f"Training rows: {len(df)}")
    print("Task counts:", df["task"].value_counts().to_dict())

    dataset = RISFineTuneDataset(df, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    with open(args.log_file, "w", encoding="utf-8") as log_f:
        log_f.write("epoch,iteration,loss\n")

    model.train()
    for epoch in range(args.epochs):
        loss_record = []
        optimizer.zero_grad(set_to_none=True)
        for idx, batch in enumerate(tqdm(dataloader, desc=f"epoch {epoch + 1}/{args.epochs}")):
            y_ids = batch["target_ids"].to(device, dtype=torch.long)
            y_ids = y_ids.contiguous().clone().detach()
            y_ids[y_ids == tokenizer.pad_token_id] = -100
            ids = batch["source_ids"].to(device, dtype=torch.long, non_blocking=True)
            mask = batch["source_mask"].to(device, dtype=torch.long, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(input_ids=ids, attention_mask=mask, labels=y_ids)
                loss = outputs.loss

            loss_record.append(float(loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if idx % 10 == 0:
                print(f"Epoch {epoch + 1} iteration {idx}: loss={loss.item():.6f}")
            with open(args.log_file, "a", encoding="utf-8") as log_f:
                log_f.write(f"{epoch + 1},{idx},{loss.item():.8f}\n")

        print(f"Epoch {epoch + 1} mean loss: {sum(loss_record) / max(len(loss_record), 1):.6f}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Model saved to:", args.output_dir)
    print("Training log:", args.log_file)


if __name__ == "__main__":
    main()
