import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import HYPERAMP_DIR, HYPERAMP_MODEL_PATH

if str(HYPERAMP_DIR) not in sys.path:
    sys.path.insert(0, str(HYPERAMP_DIR))

from predict import test

try:
    import Levenshtein
except Exception:
    Levenshtein = None


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AA_ALPHABET)}
ID_TO_AA = {i + 1: aa for i, aa in enumerate(AA_ALPHABET)}
SENTINEL = "<extra_id_0>"
REPLACE_PREFIX = "<replace>"
INSERT_PREFIX = "<insert>"


def encode_aa_sequences(seqs, max_length: int, device):
    arr = np.zeros((len(seqs), max_length), dtype=np.int64)
    lengths = []
    for i, seq in enumerate(seqs):
        clipped = seq[:max_length]
        lengths.append(len(clipped))
        for j, aa in enumerate(clipped):
            arr[i, j] = AA_TO_ID.get(aa, 0)
    return torch.tensor(arr, dtype=torch.long, device=device), torch.tensor(lengths, dtype=torch.long, device=device)


def build_replace_source(seq: str, pos: int) -> str:
    return f"{REPLACE_PREFIX} {seq[:pos]}{SENTINEL}{seq[pos + 1:]}"


def build_insert_source(seq: str, gap_pos: int) -> str:
    return f"{INSERT_PREFIX} {seq[:gap_pos + 1]}{SENTINEL}{seq[gap_pos + 1:]}"


def extract_first_amino_acid(text: str, forbidden: set[str] | None = None) -> str | None:
    forbidden = forbidden or set()
    text = re.sub(r"<extra_id_\d+>", "", text)
    for char in text:
        aa = char.upper()
        if aa in AA_ALPHABET and aa not in forbidden:
            return aa
    return None


def generate_residues_batch(
    model,
    tokenizer,
    device,
    sources: list[str],
    forbidden: set[str] | list[set[str]] | None = None,
    max_attempts: int = 3,
    max_length: int = 8,
    num_beams: int = 3,
    temperature: float = 2.0,
    batch_size: int = 256,
) -> list[str]:
    if not sources:
        return []

    if forbidden is None:
        forbidden_sets = [set() for _ in sources]
    elif isinstance(forbidden, set):
        forbidden_sets = [forbidden for _ in sources]
    else:
        forbidden_sets = forbidden

    results: list[str | None] = [None for _ in sources]
    pending = list(range(len(sources)))

    for _ in range(max_attempts):
        if not pending:
            break
        for start in range(0, len(pending), batch_size):
            chunk_indices = pending[start:start + batch_size]
            chunk_sources = [sources[i] for i in chunk_indices]
            with torch.inference_mode():
                ids = tokenizer.batch_encode_plus(
                    chunk_sources,
                    add_special_tokens=True,
                    padding=True,
                    is_split_into_words=False,
                    return_tensors="pt",
                )
                generated = model.generate(
                    input_ids=ids["input_ids"].to(device),
                    attention_mask=ids["attention_mask"].to(device),
                    temperature=temperature,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=temperature > 0,
                )
            decoded = tokenizer.batch_decode(
                generated,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for source_idx, text in zip(chunk_indices, decoded):
                aa = extract_first_amino_acid(text, forbidden=forbidden_sets[source_idx])
                if aa is not None:
                    results[source_idx] = aa
        pending = [i for i in pending if results[i] is None]

    for i, aa in enumerate(results):
        if aa is None:
            choices = [candidate for candidate in AA_ALPHABET if candidate not in forbidden_sets[i]]
            results[i] = random.choice(choices)
    return [aa for aa in results if aa is not None]


def has_three_repeat(seq: str) -> bool:
    return any(seq[i] == seq[i + 1] == seq[i + 2] for i in range(max(0, len(seq) - 2)))


def has_low_complexity(seq: str) -> bool:
    if len(seq) < 8:
        return False
    counts = {aa: seq.count(aa) for aa in set(seq)}
    dominant_fraction = max(counts.values()) / float(len(seq))
    unique_fraction = len(counts) / float(len(seq))
    return dominant_fraction >= 0.5 or unique_fraction <= 0.25


def length_penalty(seq: str, target_upper: int, max_length: int) -> float:
    return max(0, len(seq) - target_upper) / float(max_length)


def valid_sequence(seq: str, max_length: int = 40) -> bool:
    return 0 < len(seq) <= max_length and all(aa in AA_ALPHABET for aa in seq)


def build_score_df(seqs):
    target_keys = [f"seq{i}" for i, _ in enumerate(seqs)]
    labels = [0 for _ in seqs]
    return pd.DataFrame(zip(target_keys, seqs, labels), columns=["name", "seq", "label"])


_SCORE_CACHE: dict[str, float] = {}


def clear_score_cache():
    _SCORE_CACHE.clear()


def _clip_hyperamp_score(pred) -> float:
    pred = float(pred)
    if pred < 0:
        return 0.0
    if pred > 1:
        return 1.0
    return pred


def scoring_function(seqs):
    seqs = [str(seq) for seq in seqs]
    missing = []
    seen_missing = set()
    for seq in seqs:
        if seq not in _SCORE_CACHE and seq not in seen_missing:
            missing.append(seq)
            seen_missing.add(seq)

    if missing:
        df_data = build_score_df(missing)
        preds = test(df_data, str(HYPERAMP_MODEL_PATH))
        for seq, pred in zip(missing, preds):
            _SCORE_CACHE[seq] = _clip_hyperamp_score(pred)

    return np.array([_SCORE_CACHE[seq] for seq in seqs], dtype=np.float32)


def write_fasta(records, fasta_path: Path, id_prefix: str = "seq"):
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, "w", encoding="utf-8") as f:
        for i, seq in enumerate(records, start=1):
            f.write(f">{id_prefix}_{i:05d}\n{seq}\n")


def save_step_record(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _levenshtein_distance(source, target):
    if Levenshtein is not None:
        return Levenshtein.distance(source, target)
    if source == target:
        return 0
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (source_char != target_char),
                )
            )
        previous = current
    return previous[-1]


def levenshtein(source_ls, new_ls):
    if len(source_ls) != len(new_ls):
        print("the seq numbers in input lists not match")
        return []
    return [_levenshtein_distance(source_ls[i], new_ls[i]) for i in range(len(source_ls))]


def mean_pairwise_levenshtein(seqs: list[str]) -> float:
    if len(seqs) < 2:
        return 0.0
    total = 0
    count = 0
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            total += _levenshtein_distance(seqs[i], seqs[j])
            count += 1
    return total / count
