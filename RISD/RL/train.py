#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from time import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    DATA_DIR,
    RIS_FINETUNED_MODEL_DIR,
    RL_TRAIN_OUTPUT_DIR,
    configure_environment,
    get_device,
    seed_everything,
)

configure_environment()

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging as hf_logging

from runner import RunConfig, load_high_score_amp_like_from_labeled, run_agent, setup_logger

hf_logging.set_verbosity_error()


def read_sequences(path: Path, limit: int | None = None) -> list[str]:
    df = pd.read_csv(path, header=None)
    seqs = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    if limit is not None:
        seqs = seqs[:limit]
    return seqs


def load_finetuned_model(device, model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    model.to(device=device)
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RIS replace/insert/stop policy with HyperAMP reward.")
    parser.add_argument("--model-dir", type=Path, default=RIS_FINETUNED_MODEL_DIR)
    parser.add_argument("--random-file", type=Path, default=DATA_DIR / "combined966.txt")
    parser.add_argument("--random-limit", type=int, default=966)
    parser.add_argument(
        "--amp-like-file",
        type=Path,
        default=DATA_DIR / "train.txt",
        help="Labeled HyperAMP training file; only pos_* rows with numeric pMIC are used as AMP-like seeds.",
    )
    parser.add_argument("--amp-like-top-k", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=RL_TRAIN_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--max-insert-per-episode", type=int, default=2)
    parser.add_argument("--max-delete-per-episode", type=int, default=1)
    parser.add_argument("--min-length-after-delete", type=int, default=8)
    parser.add_argument("--operation-warmup-epochs", type=int, default=2)
    parser.add_argument("--counterfactual-top-k", type=int, default=3)
    parser.add_argument("--residue-candidate-k", type=int, default=5)
    parser.add_argument("--min-edit-steps-before-stop", type=int, default=2)
    parser.add_argument("--stop-score-threshold", type=float, default=0.85)
    parser.add_argument("--stop-margin", type=float, default=0.02)
    parser.add_argument("--stop-patience", type=int, default=2)
    parser.add_argument("--insert-rollout-gamma", type=float, default=0.5)
    parser.add_argument("--planner-operation-weight", type=float, default=0.1)
    parser.add_argument("--planner-position-weight", type=float, default=0.05)
    parser.add_argument("--reward-alpha", type=float, default=0.5)
    parser.add_argument("--beta-replace", type=float, default=0.01)
    parser.add_argument("--beta-insert", type=float, default=0.01)
    parser.add_argument("--beta-delete", type=float, default=0.005)
    parser.add_argument("--length-penalty-weight", type=float, default=0.02)
    parser.add_argument("--repeat-penalty-weight", type=float, default=0.02)
    parser.add_argument("--target-length-upper", type=int, default=35)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything()
    device = get_device()

    logger = setup_logger(args.output_dir, "ampainter_ris_train")
    logger.info("Available device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    logger.info("Loading RIS Ankh model: %s", args.model_dir)
    model, tokenizer = load_finetuned_model(device, args.model_dir)

    random_seqs = read_sequences(args.random_file, limit=args.random_limit)
    amp_like = load_high_score_amp_like_from_labeled(
        args.amp_like_file,
        args.amp_like_top_k,
        max_length=args.max_length,
        logger=logger,
        output_path=args.output_dir / "amp_like_pool.csv",
    )
    initial_seq = random_seqs + amp_like
    logger.info("Random training sequences: %s from %s", len(random_seqs), args.random_file)
    logger.info("Total policy training sequences: %s", len(initial_seq))

    config = RunConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        iterations=args.iterations,
        early_stop=args.early_stop,
        trainable=True,
        save_checkpoint=True,
        reward_alpha=args.reward_alpha,
        beta_replace=args.beta_replace,
        beta_insert=args.beta_insert,
        beta_delete=args.beta_delete,
        length_penalty_weight=args.length_penalty_weight,
        repeat_penalty_weight=args.repeat_penalty_weight,
        target_length_upper=args.target_length_upper,
        max_insert_per_episode=args.max_insert_per_episode,
        max_delete_per_episode=args.max_delete_per_episode,
        min_length_after_delete=args.min_length_after_delete,
        entropy_coef=args.entropy_coef,
        operation_warmup_epochs=args.operation_warmup_epochs,
        counterfactual_top_k=args.counterfactual_top_k,
        residue_candidate_k=args.residue_candidate_k,
        min_edit_steps_before_stop=args.min_edit_steps_before_stop,
        stop_score_threshold=args.stop_score_threshold,
        stop_margin=args.stop_margin,
        stop_patience=args.stop_patience,
        insert_rollout_gamma=args.insert_rollout_gamma,
        planner_operation_weight=args.planner_operation_weight,
        planner_position_weight=args.planner_position_weight,
    )

    t0 = time()
    run_agent(model, tokenizer, initial_seq, device, config, logger)
    logger.info("Total runtime: %.4fs", time() - t0)
