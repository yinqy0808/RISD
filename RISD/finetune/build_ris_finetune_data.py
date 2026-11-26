#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA_DIR, RIS_FINETUNE_DATA, SEED


SENTINEL = "<extra_id_0>"
REPLACE_PREFIX = "<replace>"
INSERT_PREFIX = "<insert>"


def read_fasta(path: Path) -> list[str]:
    seqs: list[str] = []
    current: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    seqs.append("".join(current))
                    current = []
            else:
                current.append(line.upper())
        if current:
            seqs.append("".join(current))
    return seqs


def replace_sample(seq: str, pos: int) -> dict:
    source = f"{REPLACE_PREFIX} {seq[:pos]}{SENTINEL}{seq[pos + 1:]}"
    target = f"{SENTINEL}{seq[pos]}"
    return {
        "task": "replace",
        "seq": seq,
        "position": pos,
        "source": source,
        "target": target,
    }


def insert_sample(seq: str, pos: int) -> dict:
    source = f"{INSERT_PREFIX} {seq[:pos]}{SENTINEL}{seq[pos + 1:]}"
    target = f"{SENTINEL}{seq[pos]}"
    return {
        "task": "insert",
        "seq": seq,
        "position": pos,
        "source": source,
        "target": target,
    }


def build_rows(
    seqs: list[str],
    max_seq_len: int,
    replace_per_insert: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict] = []
    for seq in seqs:
        if len(seq) < 3 or len(seq) > max_seq_len:
            continue
        if any(aa not in "ACDEFGHIKLMNPQRSTVWY" for aa in seq):
            continue

        replace_positions = list(range(len(seq)))
        rng.shuffle(replace_positions)
        for pos in replace_positions[:replace_per_insert]:
            rows.append(replace_sample(seq, pos))

        internal_positions = list(range(1, len(seq) - 1))
        if internal_positions:
            rows.append(insert_sample(seq, rng.choice(internal_positions)))

    return pd.DataFrame(rows, columns=["task", "seq", "position", "source", "target"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RIS multi-task Ankh fine-tuning data.")
    parser.add_argument("--input-fasta", type=Path, default=DATA_DIR / "9-7316.fasta")
    parser.add_argument("--output-csv", type=Path, default=RIS_FINETUNE_DATA)
    parser.add_argument("--max-seq-len", type=int, default=100)
    parser.add_argument("--replace-per-insert", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seqs = read_fasta(args.input_fasta)
    df = build_rows(
        seqs=seqs,
        max_seq_len=args.max_seq_len,
        replace_per_insert=args.replace_per_insert,
        seed=args.seed,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    counts = df["task"].value_counts().to_dict()
    print(f"Loaded sequences: {len(seqs)}")
    print(f"Saved rows: {len(df)} -> {args.output_csv}")
    print(f"Task counts: {counts}")


if __name__ == "__main__":
    main()
