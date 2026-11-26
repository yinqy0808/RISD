import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

RL_DIR = Path(__file__).resolve().parent
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))

from func import save_step_record, valid_sequence, write_fasta
from network import Agent, physicochemical_scores, shaped_sequence_score


AA_BASIC = set("KR")
AA_HYDROPHOBIC = set("AVILMFWY")
TARGET_LENGTH = 20


@dataclass
class RunConfig:
    output_dir: Path
    max_length: int = 40
    learning_rate: float = 1e-3
    batch_size: int = 128
    n_steps: int = 20
    iterations: int = 8
    early_stop: int = 5
    trainable: bool = True
    agent_checkpoint: Path | None = None
    save_checkpoint: bool = True
    reward_alpha: float = 0.5
    beta_replace: float = 0.01
    beta_insert: float = 0.01
    beta_delete: float = 0.005
    length_penalty_weight: float = 0.02
    repeat_penalty_weight: float = 0.02
    target_length_upper: int = 35
    max_insert_per_episode: int = 2
    max_delete_per_episode: int = 1
    min_length_after_delete: int = 8
    entropy_coef: float = 0.001
    operation_warmup_epochs: int = 2
    counterfactual_top_k: int = 3
    min_edit_steps_before_stop: int = 2
    stop_score_threshold: float = 0.85
    stop_margin: float = 0.02
    stop_patience: int = 2
    insert_rollout_gamma: float = 0.5
    planner_operation_weight: float = 0.1
    planner_position_weight: float = 0.05
    residue_candidate_k: int = 5


def setup_logger(output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def _batch_count(num_items, batch_size):
    return (num_items + batch_size - 1) // batch_size


def _write_step_score(path: Path, step_log):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "step,score,shaped_reward,loss,replace_count,insert_count,delete_count,stop_count,"
            "valid_stop_count,avg_length,avg_edit_count,avg_charge,avg_kr_frac,avg_hyd_frac,"
            "avg_pro_frac,avg_hydrophobic_moment,avg_max_aa_frac,replace_pro_frac,insert_pro_frac,"
            "delete_pro_frac,stop_pro_frac\n"
        )
        for row in step_log:
            f.write(
                f"{row['step']},{row['score']},{row['shaped_reward']},{row['loss']},"
                f"{row['replace_count']},{row['insert_count']},{row['delete_count']},{row['stop_count']},"
                f"{row['valid_stop_count']},{row['avg_length']},{row['avg_edit_count']},"
                f"{row['avg_charge']},{row['avg_kr_frac']},{row['avg_hyd_frac']},{row['avg_pro_frac']},"
                f"{row['avg_hydrophobic_moment']},{row['avg_max_aa_frac']},{row['replace_pro_frac']},{row['insert_pro_frac']},"
                f"{row['delete_pro_frac']},{row['stop_pro_frac']}\n"
            )


def _residue_fraction(seq: str, residues: set[str]) -> float:
    if not seq:
        return 0.0
    return sum(aa in residues for aa in seq) / len(seq)


def _max_residue_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    return max(seq.count(aa) for aa in set(seq)) / len(seq)


def _add_candidate_sort_features(df: pd.DataFrame) -> pd.DataFrame:
    candidates = df.copy()
    candidates["seq"] = candidates["seq"].astype(str)
    candidates["length"] = candidates["seq"].map(len)
    candidates["basic_frac"] = candidates["seq"].map(lambda seq: _residue_fraction(seq, AA_BASIC))
    candidates["hyd_frac"] = candidates["seq"].map(lambda seq: _residue_fraction(seq, AA_HYDROPHOBIC))
    candidates["pro_frac"] = candidates["seq"].map(lambda seq: _residue_fraction(seq, {"P"}))
    candidates["charge"] = candidates["seq"].map(lambda seq: physicochemical_scores(seq)["charge"])
    candidates["hydrophobic_moment"] = candidates["seq"].map(lambda seq: physicochemical_scores(seq)["hydrophobic_moment"])
    candidates["shaped_score"] = candidates.apply(
        lambda row: shaped_sequence_score(row["seq"], float(row["hyperamp_score"])),
        axis=1,
    )
    candidates["physicochemical_score"] = candidates["seq"].map(lambda seq: physicochemical_scores(seq)["physicochemical_score"])
    candidates["max_aa_frac"] = candidates["seq"].map(_max_residue_fraction)
    candidates["unique_frac"] = candidates["seq"].map(lambda seq: len(set(seq)) / len(seq) if seq else 0.0)
    candidates["length_good"] = candidates["length"].between(18, 24)
    candidates["charge_good"] = candidates["charge"] >= 6
    candidates["pro_good"] = candidates["pro_frac"] <= 0.15
    candidates["basic_good"] = candidates["basic_frac"] >= 0.32
    candidates["hyd_good"] = candidates["hyd_frac"] >= 0.35
    candidates["moment_good"] = candidates["hydrophobic_moment"].between(0.35, 0.70)
    candidates["low_complexity_penalty"] = (
        (candidates["max_aa_frac"] >= 0.5) | (candidates["unique_frac"] <= 0.25)
    )
    candidates["length_distance"] = (candidates["length"] - TARGET_LENGTH).abs()
    return candidates


def _load_step_records(output_dir: Path, max_length: int) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for step_dir in sorted(output_dir.glob("step_*")):
        record_path = step_dir / "record.csv"
        if not record_path.exists():
            continue
        df = pd.read_csv(record_path)
        if "seq" not in df.columns or "hyperamp_score" not in df.columns:
            continue
        if "source_step" not in df.columns:
            try:
                df["source_step"] = int(step_dir.name.split("_")[-1])
            except ValueError:
                df["source_step"] = len(records) + 1
        if "source_rank" not in df.columns:
            df["source_rank"] = range(1, len(df) + 1)
        records.append(df)

    if not records:
        return pd.DataFrame()

    candidates = pd.concat(records, ignore_index=True)
    candidates["_source_order"] = range(len(candidates))
    candidates = candidates[candidates["seq"].map(lambda seq: valid_sequence(str(seq), max_length=max_length))].copy()
    return _add_candidate_sort_features(candidates)


def _save_candidate_outputs(output_dir: Path, max_length: int):
    candidates = _load_step_records(output_dir, max_length=max_length)
    if candidates.empty:
        return

    if "likelihood" not in candidates.columns:
        candidates["likelihood"] = float("inf")

    candidates = candidates.sort_values(
        [
            "hyperamp_score",
            "shaped_score",
            "charge_good",
            "basic_good",
            "pro_good",
            "hyd_good",
            "moment_good",
            "length_good",
            "low_complexity_penalty",
            "length_distance",
            "likelihood",
            "source_step",
            "source_rank",
            "_source_order",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)
    candidates.insert(0, "candidate_id", [f"cand_{i + 1:06d}" for i in range(len(candidates))])
    candidates = candidates.drop(columns=["_source_order"])
    candidates.to_csv(output_dir / "all_candidates.csv", index=False)
    write_fasta(candidates["seq"].tolist(), output_dir / "all_candidates.fasta", id_prefix="cand")


def _summarize_operations(df: pd.DataFrame) -> dict:
    edits = df[~df["is_initial"].astype(bool)].copy()
    counts = edits["operation"].value_counts().to_dict()
    def mean_col(column: str) -> float:
        return float(edits[column].mean()) if len(edits) and column in edits.columns else 0.0

    def op_mean(operation: str, column: str) -> float:
        if column not in edits.columns:
            return 0.0
        op_df = edits[edits["operation"] == operation]
        return float(op_df[column].mean()) if len(op_df) else 0.0

    return {
        "replace_count": int(counts.get("replace", 0)),
        "insert_count": int(counts.get("insert", 0)),
        "delete_count": int(counts.get("delete", 0)),
        "stop_count": int(counts.get("stop", 0)),
        "valid_stop_count": int(
            ((edits["operation"] == "stop") & edits["stop_physically_allowed"].astype(bool)).sum()
        )
        if "stop_physically_allowed" in edits.columns
        else 0,
        "avg_length": float(edits["length"].mean()) if len(edits) else 0.0,
        "avg_edit_count": float(edits["edit_count"].mean()) if len(edits) else 0.0,
        "avg_shaped_reward": mean_col("reward"),
        "avg_charge": mean_col("charge"),
        "avg_kr_frac": mean_col("kr_frac"),
        "avg_hyd_frac": mean_col("hyd_frac"),
        "avg_pro_frac": mean_col("pro_frac"),
        "avg_hydrophobic_moment": mean_col("hydrophobic_moment"),
        "avg_max_aa_frac": mean_col("max_aa_frac"),
        "replace_pro_frac": op_mean("replace", "pro_frac"),
        "insert_pro_frac": op_mean("insert", "pro_frac"),
        "delete_pro_frac": op_mean("delete", "pro_frac"),
        "stop_pro_frac": op_mean("stop", "pro_frac"),
    }


def run_agent(model, tokenizer, batchs_seqs, device, config: RunConfig, logger=None):
    logger = logger or setup_logger(config.output_dir, "ampainter_ris")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time()

    agent = Agent(model, tokenizer, config.max_length, device)
    if config.agent_checkpoint is not None:
        logger.info("Loading RIS policy checkpoint: %s", config.agent_checkpoint)
        agent.policy.load_state_dict(torch.load(config.agent_checkpoint, map_location=device))

    optimizer = torch.optim.Adam(agent.policy.parameters(), lr=config.learning_rate) if config.trainable else None
    if config.trainable:
        agent.policy.train()
    else:
        agent.policy.eval()

    step_log: list[dict] = []
    best_score = -float("inf")
    early_stop_count = 0

    logger.info(
        "Start RIS run | trainable=%s | inputs=%s | steps=%s | iterations=%s | batch_size=%s | max_insert=%s | max_delete=%s",
        config.trainable,
        len(batchs_seqs),
        config.n_steps,
        config.iterations,
        config.batch_size,
        config.max_insert_per_episode,
        config.max_delete_per_episode,
    )
    logger.info(
        "Reward config | alpha=%s beta_replace=%s beta_insert=%s beta_delete=%s length_penalty=%s repeat_penalty=%s entropy=%s",
        config.reward_alpha,
        config.beta_replace,
        config.beta_insert,
        config.beta_delete,
        config.length_penalty_weight,
        config.repeat_penalty_weight,
        config.entropy_coef,
    )
    logger.info(
        "Planner config | top_k=%s residue_candidate_k=%s min_delete_len=%s min_stop_step=%s stop_threshold=%s stop_margin=%s stop_patience=%s insert_rollout_gamma=%s planner_op_weight=%s planner_pos_weight=%s",
        config.counterfactual_top_k,
        config.residue_candidate_k,
        config.min_length_after_delete,
        config.min_edit_steps_before_stop,
        config.stop_score_threshold,
        config.stop_margin,
        config.stop_patience,
        config.insert_rollout_gamma,
        config.planner_operation_weight,
        config.planner_position_weight,
    )
    if config.trainable and config.operation_warmup_epochs > 0:
        logger.info("Start operation supervised warm-up | epochs=%s", config.operation_warmup_epochs)
        for warmup_epoch in range(config.operation_warmup_epochs):
            warmup_losses = []
            warmup_counts = {"replace": 0, "insert": 0, "delete": 0, "stop": 0}
            batch_num = _batch_count(len(batchs_seqs), config.batch_size)
            for bn in tqdm(range(batch_num), desc=f"warm-up {warmup_epoch + 1}", leave=False):
                start_idx = bn * config.batch_size
                end_idx = min((bn + 1) * config.batch_size, len(batchs_seqs))
                batch_seqs = batchs_seqs[start_idx:end_idx]
                warmup_loss, counts = agent.operation_warmup_step(
                    batch_seqs=batch_seqs,
                    iterations=config.iterations,
                    max_insert_per_episode=config.max_insert_per_episode,
                    max_delete_per_episode=config.max_delete_per_episode,
                    min_length_after_delete=config.min_length_after_delete,
                    counterfactual_top_k=config.counterfactual_top_k,
                    reward_alpha=config.reward_alpha,
                    beta_replace=config.beta_replace,
                    beta_insert=config.beta_insert,
                    beta_delete=config.beta_delete,
                    length_penalty_weight=config.length_penalty_weight,
                    repeat_penalty_weight=config.repeat_penalty_weight,
                    target_length_upper=config.target_length_upper,
                    min_edit_steps_before_stop=config.min_edit_steps_before_stop,
                    stop_score_threshold=config.stop_score_threshold,
                    stop_margin=config.stop_margin,
                    insert_rollout_gamma=config.insert_rollout_gamma,
                    residue_candidate_k=config.residue_candidate_k,
                    stop_gate_strict=False,
                    planner_position_weight=config.planner_position_weight,
                )
                optimizer.zero_grad()
                warmup_loss.backward()
                optimizer.step()
                warmup_losses.append(float(warmup_loss.detach().item()))
                for key, value in counts.items():
                    warmup_counts[key] = warmup_counts.get(key, 0) + int(value)
            logger.info(
                "Warm-up epoch %s/%s | loss=%.4f | pseudo_labels replace=%s insert=%s delete=%s stop=%s",
                warmup_epoch + 1,
                config.operation_warmup_epochs,
                float(np.mean(warmup_losses)) if warmup_losses else float("nan"),
                warmup_counts.get("replace", 0),
                warmup_counts.get("insert", 0),
                warmup_counts.get("delete", 0),
                warmup_counts.get("stop", 0),
            )

    for step in tqdm(range(config.n_steps), desc="RIS steps"):
        mean_train_loss = []
        step_final_seqs: list[str] = []
        step_final_scores: list[float] = []
        step_likelihood: list[float] = []
        step_trajectories: list[pd.DataFrame] = []

        batch_num = _batch_count(len(batchs_seqs), config.batch_size)
        for bn in tqdm(range(batch_num), desc=f"step {step + 1}", leave=False):
            start_idx = bn * config.batch_size
            end_idx = min((bn + 1) * config.batch_size, len(batchs_seqs))
            batch_seqs = batchs_seqs[start_idx:end_idx]

            final_seqs, loss_terms, nll_terms, trajectory, final_scores = agent.run_episode(
                batch_seqs=batch_seqs,
                iterations=config.iterations,
                outer_step=step + 1,
                batch_num=bn,
                batch_start_index=start_idx,
                reward_alpha=config.reward_alpha,
                beta_replace=config.beta_replace,
                beta_insert=config.beta_insert,
                beta_delete=config.beta_delete,
                length_penalty_weight=config.length_penalty_weight,
                repeat_penalty_weight=config.repeat_penalty_weight,
                target_length_upper=config.target_length_upper,
                max_insert_per_episode=config.max_insert_per_episode,
                max_delete_per_episode=config.max_delete_per_episode,
                min_length_after_delete=config.min_length_after_delete,
                counterfactual_top_k=config.counterfactual_top_k,
                min_edit_steps_before_stop=config.min_edit_steps_before_stop,
                stop_score_threshold=config.stop_score_threshold,
                stop_margin=config.stop_margin,
                stop_patience=config.stop_patience,
                insert_rollout_gamma=config.insert_rollout_gamma,
                residue_candidate_k=config.residue_candidate_k,
                stop_gate_strict=not config.trainable,
                planner_operation_weight=config.planner_operation_weight,
                planner_position_weight=config.planner_position_weight,
                entropy_coef=config.entropy_coef,
                record_trajectory=True,
            )

            loss = loss_terms.mean()
            if config.trainable:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_train_loss.append(float(loss.detach().item()))

            step_final_seqs.extend(final_seqs)
            step_final_scores.extend([float(x) for x in final_scores])
            step_likelihood.extend([float(x) for x in nll_terms.detach().cpu().numpy()])
            step_trajectories.append(trajectory)

        step_df = pd.DataFrame(
            {
                "seq": step_final_seqs,
                "hyperamp_score": step_final_scores,
                "likelihood": step_likelihood,
            }
        )
        step_df = _add_candidate_sort_features(step_df)
        step_df["source_step"] = step + 1
        step_df["source_rank"] = range(1, len(step_df) + 1)
        step_dir = config.output_dir / f"step_{step + 1:03d}"
        save_step_record(step_df, step_dir / "record.csv")

        step_traj = pd.concat(step_trajectories, ignore_index=True)
        step_traj.to_csv(step_dir / "trajectory.csv", index=False)

        mean_loss = float(np.array(mean_train_loss).mean()) if mean_train_loss else float("nan")
        op_summary = _summarize_operations(step_traj)
        step_summary = {
            "step": step + 1,
            "score": float(np.mean(step_final_scores)),
            "shaped_reward": float(step_df["shaped_score"].mean()) if len(step_df) else 0.0,
            "loss": mean_loss,
            **op_summary,
        }
        step_log.append(step_summary)

        time_elapsed = (time() - start_time) / 3600
        time_left = time_elapsed * ((config.n_steps - step - 1) / (step + 1))
        logger.info(
            "Step %s/%s | mean_score=%.4f | shaped=%.4f | loss=%s | replace=%s insert=%s delete=%s stop=%s valid_stop=%s | avg_len=%.2f | charge=%.2f KR=%.3f hyd=%.3f pro=%.3f moment=%.3f | elapsed=%.2fh | left=%.2fh",
            step + 1,
            config.n_steps,
            step_summary["score"],
            step_summary["shaped_reward"],
            f"{mean_loss:.4f}" if not np.isnan(mean_loss) else "nan",
            op_summary["replace_count"],
            op_summary["insert_count"],
            op_summary["delete_count"],
            op_summary["stop_count"],
            op_summary["valid_stop_count"],
            op_summary["avg_length"],
            op_summary["avg_charge"],
            op_summary["avg_kr_frac"],
            op_summary["avg_hyd_frac"],
            op_summary["avg_pro_frac"],
            op_summary["avg_hydrophobic_moment"],
            time_elapsed,
            time_left,
        )
        logger.info("Top final samples by HyperAMP:")
        preview = step_df.sort_values("hyperamp_score", ascending=False).head(5)
        for _, row in preview.iterrows():
            logger.info(
                "  score=%6.4f shaped=%6.4f pro=%.3f KR=%.3f hyd=%.3f moment=%.3f likelihood=%6.2f seq=%s",
                row["hyperamp_score"],
                row["shaped_score"],
                row["pro_frac"],
                row["basic_frac"],
                row["hyd_frac"],
                row["hydrophobic_moment"],
                row["likelihood"],
                row["seq"],
            )

        _write_step_score(config.output_dir / "step_score.csv", step_log)

        if config.trainable and config.save_checkpoint:
            if step_summary["shaped_reward"] > best_score:
                best_score = step_summary["shaped_reward"]
                torch.save(agent.policy.state_dict(), config.output_dir / "RIS_Agent.ckpt")
                logger.info("Saved best RIS checkpoint with mean_shaped_reward %.4f", best_score)
                early_stop_count = 0
            else:
                early_stop_count += 1
            if early_stop_count >= config.early_stop:
                logger.info("Policy is not improving, so we halt the training session.")
                break

    _save_candidate_outputs(config.output_dir, max_length=config.max_length)
    logger.info("RIS run finished | outputs=%s", config.output_dir)
    return agent


def load_high_score_amp_like_from_labeled(
    path: Path,
    top_k: int,
    max_length: int = 40,
    logger=None,
    output_path: Path | None = None,
) -> list[str]:
    if top_k <= 0 or path is None or not path.exists():
        return []

    df = pd.read_csv(path, sep="\t", header=None, names=["id", "seq", "pmic"], dtype=str)
    df["id"] = df["id"].fillna("").str.strip()
    df["seq"] = df["seq"].fillna("").str.strip().str.upper()
    df["pmic"] = pd.to_numeric(df["pmic"], errors="coerce")

    pos = df[df["id"].str.startswith("pos_") & df["pmic"].notna()].copy()
    pos = pos[pos["seq"].map(lambda seq: valid_sequence(str(seq), max_length=max_length))]
    pos = pos.sort_values("pmic", ascending=False).drop_duplicates(subset=["seq"], keep="first")
    selected = pos.head(top_k).reset_index(drop=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        selected[["id", "seq", "pmic"]].to_csv(output_path, index=False)

    if logger:
        logger.info(
            "Loaded AMP-like pMIC seeds: %s/%s positives from %s",
            len(selected),
            len(pos),
            path,
        )
        if len(selected):
            logger.info(
                "AMP-like pMIC range: %.4f - %.4f",
                float(selected["pmic"].min()),
                float(selected["pmic"].max()),
            )
            logger.info("Saved AMP-like seed pool: %s", output_path)

    return selected["seq"].tolist()
