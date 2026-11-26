import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

RL_DIR = Path(__file__).resolve().parent
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))

from func import (
    build_insert_source,
    build_replace_source,
    encode_aa_sequences,
    generate_residues_batch,
    has_low_complexity,
    has_three_repeat,
    length_penalty,
    levenshtein,
    scoring_function,
)


OP_REPLACE = 0
OP_INSERT = 1
OP_DELETE = 2
OP_STOP = 3
OP_NAMES = {
    OP_REPLACE: "replace",
    OP_INSERT: "insert",
    OP_DELETE: "delete",
    OP_STOP: "stop",
}

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_BASIC = set("KR")
AA_HYDROPHOBIC = set("AVILMFWY")
AA_AROMATIC = set("FWY")
AA_ACIDIC = set("DE")
TARGET_LENGTH_CENTER = 20
HYDROPHOBICITY = {
    "A": 0.62,
    "C": 0.29,
    "D": -0.90,
    "E": -0.74,
    "F": 1.19,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "K": -1.50,
    "L": 1.06,
    "M": 0.64,
    "N": -0.78,
    "P": 0.12,
    "Q": -0.85,
    "R": -2.53,
    "S": -0.18,
    "T": -0.05,
    "V": 1.08,
    "W": 0.81,
    "Y": 0.26,
}

SHAPING_WEIGHTS = {
    "kr": 0.23,
    "hyd": 0.04,
    "charge": 0.18,
    "length": 0.04,
    "moment": 0.02,
    "pro": 0.10,
    "complexity": 0.05,
}


class RISPolicy(nn.Module):
    def __init__(
        self,
        max_length: int = 40,
        hidden: int = 128,
        state_dim: int = 6,
        operation_prior: tuple[float, float, float, float] = (0.70, 0.15, 0.05, 0.10),
    ):
        super().__init__()
        self.max_length = max_length
        self.hidden = hidden
        self.aa_emb = nn.Embedding(21, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(max_length, hidden)
        self.encoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.operation_head = nn.Sequential(
            nn.Linear(hidden + state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
        )
        self.replace_head = nn.Linear(hidden, 1)
        self.delete_head = nn.Linear(hidden, 1)
        self.insert_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._init_operation_prior(operation_prior)

    def _init_operation_prior(self, prior):
        prior_tensor = torch.tensor(prior, dtype=torch.float32)
        prior_tensor = prior_tensor / prior_tensor.sum()
        final_layer = self.operation_head[-1]
        if isinstance(final_layer, nn.Linear):
            with torch.no_grad():
                final_layer.bias.copy_(torch.log(prior_tensor))

    def forward(self, aa_ids, lengths, state_features):
        batch_size, max_length = aa_ids.shape
        pos_ids = torch.arange(max_length, device=aa_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.aa_emb(aa_ids) + self.pos_emb(pos_ids)
        hidden = self.encoder(hidden)

        valid = (torch.arange(max_length, device=aa_ids.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        pooled = (hidden * valid.unsqueeze(-1)).sum(dim=1) / lengths.clamp(min=1).unsqueeze(1).float()

        operation_logits = self.operation_head(torch.cat([pooled, state_features], dim=1))
        replace_logits = self.replace_head(hidden).squeeze(-1)
        delete_logits = self.delete_head(hidden).squeeze(-1)

        left = hidden[:, :-1, :]
        right = hidden[:, 1:, :]
        insert_logits = self.insert_head(torch.cat([left, right], dim=-1)).squeeze(-1)
        return operation_logits, replace_logits, insert_logits, delete_logits


def _masked_logits(logits, mask):
    return logits.masked_fill(~mask, -1e9)


def _sample_masked(logits, mask):
    samples = []
    log_probs = []
    entropies = []
    for row_logits, row_mask in zip(logits, mask):
        if not bool(row_mask.any()):
            samples.append(torch.tensor(0, device=logits.device, dtype=torch.long))
            log_probs.append(torch.tensor(0.0, device=logits.device))
            entropies.append(torch.tensor(0.0, device=logits.device))
            continue
        dist = Categorical(logits=_masked_logits(row_logits, row_mask))
        sample = dist.sample()
        samples.append(sample)
        log_probs.append(dist.log_prob(sample))
        entropies.append(dist.entropy())
    return torch.stack(samples), torch.stack(log_probs), torch.stack(entropies)


def _topk_masked_indices(logits, mask, k: int):
    rows = []
    for row_logits, row_mask in zip(logits, mask):
        valid_idx = torch.nonzero(row_mask, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            rows.append([])
            continue
        valid_logits = row_logits[valid_idx]
        top_k = min(int(k), int(valid_idx.numel()))
        chosen = torch.topk(valid_logits, k=top_k).indices
        rows.append(valid_idx[chosen].detach().cpu().tolist())
    return rows


def _log_prob_at(logits, mask, index: int):
    dist = Categorical(logits=_masked_logits(logits, mask))
    return dist.log_prob(torch.tensor(index, dtype=torch.long, device=logits.device)), dist.entropy()


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _residue_fraction(seq: str, residues: set[str]) -> float:
    if not seq:
        return 0.0
    return sum(aa in residues for aa in seq) / len(seq)


def physicochemical_features(seq: str) -> dict[str, float]:
    seq = str(seq)
    length = len(seq)
    counts = {aa: seq.count(aa) for aa in AA_ALPHABET}
    if length == 0:
        return {
            "length": 0.0,
            "charge": 0.0,
            "kr_frac": 0.0,
            "hyd_frac": 0.0,
            "arom_frac": 0.0,
            "pro_frac": 0.0,
            "gly_frac": 0.0,
            "acid_frac": 0.0,
            "hydrophobic_moment": 0.0,
            "max_aa_frac": 0.0,
            "unique_frac": 0.0,
        }
    charge = counts["K"] + counts["R"] + 0.1 * counts["H"] - counts["D"] - counts["E"]
    angle = math.radians(100)
    moment_x = sum(HYDROPHOBICITY.get(aa, 0.0) * math.cos(i * angle) for i, aa in enumerate(seq))
    moment_y = sum(HYDROPHOBICITY.get(aa, 0.0) * math.sin(i * angle) for i, aa in enumerate(seq))
    return {
        "length": float(length),
        "charge": float(charge),
        "kr_frac": _residue_fraction(seq, AA_BASIC),
        "hyd_frac": _residue_fraction(seq, AA_HYDROPHOBIC),
        "arom_frac": _residue_fraction(seq, AA_AROMATIC),
        "pro_frac": counts["P"] / float(length),
        "gly_frac": counts["G"] / float(length),
        "acid_frac": _residue_fraction(seq, AA_ACIDIC),
        "hydrophobic_moment": math.sqrt(moment_x * moment_x + moment_y * moment_y) / float(length),
        "max_aa_frac": max(counts.values()) / float(length),
        "unique_frac": len(set(seq)) / float(length),
    }


def _length_score(length: float) -> float:
    if 18 <= length <= 24:
        return 1.0
    if 16 <= length < 18 or 24 < length <= 26:
        return 0.5
    return 0.0


def _moment_score(moment: float) -> float:
    if moment < 0.35:
        return _clip(moment / 0.35)
    if moment < 0.40:
        return 0.8 + 0.2 * ((moment - 0.35) / 0.05)
    if moment <= 0.65:
        return 1.0
    if moment <= 0.80:
        return 1.0 - 0.3 * ((moment - 0.65) / 0.15)
    return 0.3


def _kr_score(kr_frac: float) -> float:
    if kr_frac < 0.25:
        return _clip(kr_frac / 0.25)
    if kr_frac < 0.32:
        return 0.75 + 0.25 * ((kr_frac - 0.25) / 0.07)
    if kr_frac <= 0.45:
        return 1.0
    if kr_frac <= 0.55:
        return 0.7
    return 0.3


def _hyd_score(hyd_frac: float) -> float:
    if hyd_frac < 0.35:
        return _clip(hyd_frac / 0.35)
    if hyd_frac <= 0.50:
        return 1.0
    if hyd_frac <= 0.60:
        return 0.7
    return 0.3


def _charge_score(charge: float) -> float:
    if charge < 4:
        return _clip(charge / 4)
    if charge <= 9:
        return 1.0
    if charge <= 11:
        return 0.8
    return 0.4


def _pro_penalty(pro_frac: float) -> float:
    if pro_frac <= 0.08:
        return 0.0
    if pro_frac <= 0.15:
        return (pro_frac - 0.08) / 0.07
    return 1.0 + 2.0 * (pro_frac - 0.15)


def _complexity_penalty(max_aa_frac: float) -> float:
    if max_aa_frac <= 0.40:
        return 0.0
    if max_aa_frac <= 0.50:
        return (max_aa_frac - 0.40) / 0.10
    return 1.5


def physicochemical_scores(seq: str) -> dict[str, float]:
    features = physicochemical_features(seq)
    scores = {
        "length_score": _length_score(features["length"]),
        "kr_score": _kr_score(features["kr_frac"]),
        "hyd_score": _hyd_score(features["hyd_frac"]),
        "charge_score": _charge_score(features["charge"]),
        "moment_score": _moment_score(features["hydrophobic_moment"]),
        "pro_penalty": _pro_penalty(features["pro_frac"]),
        "complexity_penalty_value": _complexity_penalty(features["max_aa_frac"]),
    }
    physicochemical_score = (
        SHAPING_WEIGHTS["kr"] * scores["kr_score"]
        + SHAPING_WEIGHTS["hyd"] * scores["hyd_score"]
        + SHAPING_WEIGHTS["charge"] * scores["charge_score"]
        + SHAPING_WEIGHTS["length"] * scores["length_score"]
        + SHAPING_WEIGHTS["moment"] * scores["moment_score"]
        - SHAPING_WEIGHTS["pro"] * scores["pro_penalty"]
        - SHAPING_WEIGHTS["complexity"] * scores["complexity_penalty_value"]
    )
    scores["physicochemical_score"] = float(physicochemical_score)
    scores.update(features)
    return scores


def shaped_sequence_score(seq: str, hyperamp_score: float) -> float:
    return float(hyperamp_score) + physicochemical_scores(seq)["physicochemical_score"]


class Agent:
    def __init__(self, model, tokenizer, max_length, device):
        self.max_length = max_length
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.policy = RISPolicy(max_length=self.max_length)
        self.policy.to(self.device)

    def _operation_mask(
        self,
        lengths,
        insert_counts,
        delete_counts,
        max_insert_per_episode,
        max_delete_per_episode,
        min_length_after_delete,
    ):
        mask = torch.ones((len(lengths), 4), dtype=torch.bool, device=self.device)
        mask[:, OP_INSERT] = (
            (lengths >= 3)
            & (lengths < self.max_length)
            & (insert_counts < max_insert_per_episode)
        )
        mask[:, OP_DELETE] = (
            (lengths > min_length_after_delete)
            & (delete_counts < max_delete_per_episode)
        )
        return mask

    def _replace_position_mask(self, lengths):
        pos = torch.arange(self.max_length, device=self.device).unsqueeze(0)
        return pos < lengths.unsqueeze(1)

    def _insert_position_mask(self, lengths):
        pos = torch.arange(self.max_length - 1, device=self.device).unsqueeze(0)
        return pos < (lengths - 1).clamp(min=0).unsqueeze(1)

    def _delete_position_mask(self, lengths, min_length_after_delete):
        pos = torch.arange(self.max_length, device=self.device).unsqueeze(0)
        return (pos < lengths.unsqueeze(1)) & (lengths.unsqueeze(1) > min_length_after_delete)

    def _reward_value(
        self,
        seq: str,
        parent_seq: str,
        score: float,
        old_score: float,
        operation: str,
        reward_alpha: float,
        beta_replace: float,
        beta_insert: float,
        beta_delete: float,
        length_penalty_weight: float,
        repeat_penalty_weight: float,
        target_length_upper: int,
    ) -> float:
        improvement = max(float(score - old_score), 0.0)
        replace_cost = beta_replace if operation == "replace" else 0.0
        insert_cost = beta_insert if operation == "insert" else 0.0
        delete_cost = beta_delete if operation == "delete" else 0.0
        len_penalty = length_penalty_weight * length_penalty(seq, target_length_upper, self.max_length)
        rep_penalty = repeat_penalty_weight if (has_three_repeat(seq) or has_low_complexity(seq)) else 0.0
        shaped = shaped_sequence_score(seq, score)
        delete_bonus = 0.0
        if operation == "delete":
            old_features = physicochemical_features(parent_seq)
            new_features = physicochemical_features(seq)
            pro_reduction = max(0.0, old_features["pro_frac"] - new_features["pro_frac"])
            complexity_reduction = max(0.0, old_features["max_aa_frac"] - new_features["max_aa_frac"])
            if old_features["pro_frac"] > 0.15:
                delete_bonus = 0.02 * pro_reduction + 0.01 * complexity_reduction
        return float(
            shaped
            + reward_alpha * improvement
            + delete_bonus
            - replace_cost
            - insert_cost
            - delete_cost
            - len_penalty
            - rep_penalty
        )

    def _stop_gate(self, seq: str, score: float, strict: bool) -> bool:
        features = physicochemical_features(seq)
        if strict:
            return (
                float(score) >= 0.90
                and 18 <= features["length"] <= 24
                and features["pro_frac"] <= 0.15
                and features["kr_frac"] >= 0.32
                and features["charge"] >= 6
                and features["hyd_frac"] >= 0.35
                and features["max_aa_frac"] <= 0.45
            )
        return (
            float(score) >= 0.85
            and features["length"] >= 16
            and features["pro_frac"] <= 0.18
            and features["kr_frac"] >= 0.25
            and features["charge"] >= 5
            and features["hyd_frac"] >= 0.32
        )

    def _counterfactual_candidates(
        self,
        seqs,
        current_scores,
        replace_position_lists,
        insert_position_lists,
        delete_position_lists,
        insert_allowed,
        delete_allowed,
        inner_step,
        reward_alpha,
        beta_replace,
        beta_insert,
        beta_delete,
        length_penalty_weight,
        repeat_penalty_weight,
        target_length_upper,
        min_edit_steps_before_stop,
        stop_score_threshold,
        stop_margin,
        insert_rollout_gamma,
        residue_candidate_k,
        stop_gate_strict,
    ):
        specs: list[dict] = []
        score_jobs: list[tuple[int, str, int]] = []
        score_seqs: list[str] = []
        edit_jobs: list[tuple[int, str, int]] = []
        edit_sources: list[str] = []

        for j, seq in enumerate(seqs):
            parent_score = float(current_scores[j])
            candidates = {
                "replace": [],
                "insert": [],
                "delete": [],
                "stop": [
                    {
                        "op_id": OP_STOP,
                        "seq": seq,
                        "position": -1,
                        "generated_aa": "",
                        "score": parent_score,
                    }
                ],
            }

            for replace_pos in replace_position_lists[j]:
                replace_pos = int(replace_pos)
                for _ in range(max(1, int(residue_candidate_k))):
                    edit_jobs.append((j, "replace", replace_pos))
                    edit_sources.append(build_replace_source(seq, replace_pos))

            if bool(insert_allowed[j]):
                for insert_pos in insert_position_lists[j]:
                    insert_pos = int(insert_pos)
                    for _ in range(max(1, int(residue_candidate_k))):
                        edit_jobs.append((j, "insert", insert_pos))
                        edit_sources.append(build_insert_source(seq, insert_pos))

            if bool(delete_allowed[j]):
                for delete_pos in delete_position_lists[j]:
                    delete_pos = int(delete_pos)
                    delete_seq = seq[:delete_pos] + seq[delete_pos + 1:]
                    candidates["delete"].append(
                        {
                            "op_id": OP_DELETE,
                            "seq": delete_seq,
                            "position": delete_pos,
                            "generated_aa": "",
                            "score": None,
                        }
                    )
                    score_jobs.append((j, "delete", len(candidates["delete"]) - 1))
                    score_seqs.append(delete_seq)

            specs.append({"parent_seq": seq, "parent_score": parent_score, "candidates": candidates})

        edit_aas = generate_residues_batch(self.model, self.tokenizer, self.device, edit_sources)
        seen_edit_candidates: set[tuple[int, str, int, str]] = set()

        def add_edit_candidate(spec_idx: int, op_name: str, pos: int, aa: str) -> None:
            key = (spec_idx, op_name, int(pos), aa)
            if key in seen_edit_candidates:
                return
            seen_edit_candidates.add(key)
            seq = specs[spec_idx]["parent_seq"]
            if op_name == "replace":
                edited_seq = seq[:pos] + aa + seq[pos + 1:]
                op_id = OP_REPLACE
            else:
                edited_seq = seq[:pos + 1] + aa + seq[pos + 1:]
                op_id = OP_INSERT
            specs[spec_idx]["candidates"][op_name].append(
                {
                    "op_id": op_id,
                    "seq": edited_seq,
                    "position": pos,
                    "generated_aa": aa,
                    "score": None,
                }
            )
            candidate_idx = len(specs[spec_idx]["candidates"][op_name]) - 1
            score_jobs.append((spec_idx, op_name, candidate_idx))
            score_seqs.append(edited_seq)

        for aa, (spec_idx, op_name, pos) in zip(edit_aas, edit_jobs):
            add_edit_candidate(spec_idx, op_name, pos, aa)

        for spec_idx, op_name, pos in sorted(set(edit_jobs)):
            for aa in ("K", "R"):
                add_edit_candidate(spec_idx, op_name, pos, aa)

        if score_seqs:
            scored = scoring_function(score_seqs)
            for score, (spec_idx, op_name, candidate_idx) in zip(scored, score_jobs):
                specs[spec_idx]["candidates"][op_name][candidate_idx]["score"] = float(score)

        rollout_jobs: list[tuple[int, int]] = []
        rollout_positions: list[int] = []
        rollout_sources: list[str] = []
        rollout_seqs: list[str] = []
        if insert_rollout_gamma > 0:
            for spec_idx, spec in enumerate(specs):
                for candidate_idx, candidate in enumerate(spec["candidates"]["insert"]):
                    insert_seq = candidate["seq"]
                    insert_pos = int(candidate["position"])
                    local_positions = sorted(
                        {
                            max(0, insert_pos - 1),
                            min(len(insert_seq) - 1, insert_pos),
                            min(len(insert_seq) - 1, insert_pos + 1),
                        }
                    )
                    for replace_pos in local_positions:
                        rollout_jobs.append((spec_idx, candidate_idx))
                        rollout_positions.append(replace_pos)
                        rollout_sources.append(build_replace_source(insert_seq, replace_pos))
            rollout_aas = generate_residues_batch(self.model, self.tokenizer, self.device, rollout_sources)
            for aa, (spec_idx, candidate_idx), replace_pos in zip(rollout_aas, rollout_jobs, rollout_positions):
                insert_seq = specs[spec_idx]["candidates"]["insert"][candidate_idx]["seq"]
                rollout_seq = insert_seq[:replace_pos] + aa + insert_seq[replace_pos + 1:]
                rollout_seqs.append(rollout_seq)
            if rollout_seqs:
                rollout_scores = scoring_function(rollout_seqs)
                best_by_candidate: dict[tuple[int, int], tuple[float, str]] = {}
                for score, seq, key in zip(rollout_scores, rollout_seqs, rollout_jobs):
                    score = float(score)
                    if key not in best_by_candidate or score > best_by_candidate[key][0]:
                        best_by_candidate[key] = (score, seq)
                for (spec_idx, candidate_idx), (rollout_score, rollout_seq) in best_by_candidate.items():
                    specs[spec_idx]["candidates"]["insert"][candidate_idx]["rollout_score"] = rollout_score
                    specs[spec_idx]["candidates"]["insert"][candidate_idx]["rollout_seq"] = rollout_seq

        for spec in specs:
            old_score = spec["parent_score"]
            for op_name, candidates in spec["candidates"].items():
                for candidate in candidates:
                    phys = physicochemical_scores(candidate["seq"])
                    candidate["shaped_score"] = shaped_sequence_score(candidate["seq"], float(candidate["score"]))
                    candidate["physicochemical_score"] = phys["physicochemical_score"]
                    candidate["pro_frac"] = phys["pro_frac"]
                    candidate["kr_frac"] = phys["kr_frac"]
                    candidate["hyd_frac"] = phys["hyd_frac"]
                    candidate["charge"] = phys["charge"]
                    candidate["max_aa_frac"] = phys["max_aa_frac"]
                    candidate["pro_penalty"] = phys["pro_penalty"]
                    candidate["complexity_penalty_value"] = phys["complexity_penalty_value"]
                    candidate["reward"] = self._reward_value(
                        seq=candidate["seq"],
                        parent_seq=spec["parent_seq"],
                        score=float(candidate["score"]),
                        old_score=old_score,
                        operation=op_name,
                        reward_alpha=reward_alpha,
                        beta_replace=beta_replace,
                        beta_insert=beta_insert,
                        beta_delete=beta_delete,
                        length_penalty_weight=length_penalty_weight,
                        repeat_penalty_weight=repeat_penalty_weight,
                        target_length_upper=target_length_upper,
                    )
                    candidate["planner_reward"] = candidate["reward"]
                    if op_name == "insert" and "rollout_score" in candidate:
                        rollout_reward = self._reward_value(
                            seq=candidate["rollout_seq"],
                            parent_seq=candidate["seq"],
                            score=float(candidate["rollout_score"]),
                            old_score=float(candidate["score"]),
                            operation="replace",
                            reward_alpha=reward_alpha,
                            beta_replace=beta_replace,
                            beta_insert=beta_insert,
                            beta_delete=beta_delete,
                            length_penalty_weight=length_penalty_weight,
                            repeat_penalty_weight=repeat_penalty_weight,
                            target_length_upper=target_length_upper,
                        )
                        candidate["planner_reward"] = candidate["reward"] + insert_rollout_gamma * max(
                            0.0,
                            rollout_reward - float(candidate["score"]),
                        )

            actions = {}
            if spec["candidates"]["replace"]:
                actions["replace"] = max(spec["candidates"]["replace"], key=lambda item: item["planner_reward"])
            if spec["candidates"]["insert"]:
                actions["insert"] = max(spec["candidates"]["insert"], key=lambda item: item["planner_reward"])
            if spec["candidates"]["delete"]:
                actions["delete"] = max(spec["candidates"]["delete"], key=lambda item: item["planner_reward"])

            stop_action = spec["candidates"]["stop"][0]
            edit_rewards = [action["planner_reward"] for action in actions.values()]
            best_edit_reward = max(edit_rewards) if edit_rewards else -float("inf")
            stop_physically_allowed = self._stop_gate(
                stop_action["seq"],
                float(stop_action["score"]),
                strict=stop_gate_strict,
            )
            stop_allowed = (
                stop_physically_allowed
                and
                inner_step >= min_edit_steps_before_stop
                and (
                    old_score >= stop_score_threshold
                    or best_edit_reward <= stop_action["planner_reward"] + stop_margin
                )
            )
            if stop_allowed:
                actions["stop"] = stop_action

            rewards = [action["planner_reward"] for action in actions.values()]
            baseline = float(np.mean(rewards))
            for action in actions.values():
                action["advantage"] = float(action["planner_reward"] - baseline)
            spec["actions"] = actions
            spec["stop_allowed"] = stop_allowed
            spec["stop_physically_allowed"] = stop_physically_allowed
            spec["stop_reward"] = stop_action["reward"]
            spec["best_edit_reward"] = best_edit_reward
            spec["best_operation"] = max(actions, key=lambda name: actions[name]["planner_reward"])

        return specs

    def operation_warmup_step(
        self,
        batch_seqs,
        iterations=8,
        max_insert_per_episode=2,
        max_delete_per_episode=1,
        min_length_after_delete=8,
        counterfactual_top_k=3,
        reward_alpha=0.5,
        beta_replace=0.01,
        beta_insert=0.01,
        beta_delete=0.005,
        length_penalty_weight=0.02,
        repeat_penalty_weight=0.02,
        target_length_upper=35,
        min_edit_steps_before_stop=2,
        stop_score_threshold=0.8,
        stop_margin=0.02,
        insert_rollout_gamma=0.5,
        residue_candidate_k=5,
        stop_gate_strict=False,
        planner_position_weight=0.05,
    ):
        current_seqs = list(batch_seqs)
        current_scores = scoring_function(current_seqs)
        insert_counts = np.zeros(len(current_seqs), dtype=np.int64)
        delete_counts = np.zeros(len(current_seqs), dtype=np.int64)

        aa_ids, lengths = encode_aa_sequences(current_seqs, self.max_length, self.device)
        insert_count_tensor = torch.tensor(insert_counts, dtype=torch.long, device=self.device)
        delete_count_tensor = torch.tensor(delete_counts, dtype=torch.long, device=self.device)
        state_features = torch.tensor(
            [
                [
                    min_edit_steps_before_stop / float(iterations),
                    len(seq) / float(self.max_length),
                    0.0,
                    0.0,
                    float(current_scores[i]),
                    0.0,
                ]
                for i, seq in enumerate(current_seqs)
            ],
            dtype=torch.float32,
            device=self.device,
        )

        op_logits, replace_logits, insert_logits, delete_logits = self.policy(aa_ids, lengths, state_features)
        op_mask = self._operation_mask(
            lengths,
            insert_count_tensor,
            delete_count_tensor,
            max_insert_per_episode,
            max_delete_per_episode,
            min_length_after_delete,
        )
        replace_mask = self._replace_position_mask(lengths)
        insert_mask = self._insert_position_mask(lengths)
        delete_mask = self._delete_position_mask(lengths, min_length_after_delete)

        replace_position_lists = _topk_masked_indices(replace_logits.detach(), replace_mask, counterfactual_top_k)
        insert_position_lists = _topk_masked_indices(insert_logits.detach(), insert_mask, counterfactual_top_k)
        delete_position_lists = _topk_masked_indices(delete_logits.detach(), delete_mask, counterfactual_top_k)
        specs = self._counterfactual_candidates(
            seqs=current_seqs,
            current_scores=current_scores,
            replace_position_lists=replace_position_lists,
            insert_position_lists=insert_position_lists,
            delete_position_lists=delete_position_lists,
            insert_allowed=op_mask[:, OP_INSERT].detach().cpu().numpy(),
            delete_allowed=op_mask[:, OP_DELETE].detach().cpu().numpy(),
            inner_step=min_edit_steps_before_stop,
            reward_alpha=reward_alpha,
            beta_replace=beta_replace,
            beta_insert=beta_insert,
            beta_delete=beta_delete,
            length_penalty_weight=length_penalty_weight,
            repeat_penalty_weight=repeat_penalty_weight,
            target_length_upper=target_length_upper,
            min_edit_steps_before_stop=min_edit_steps_before_stop,
            stop_score_threshold=stop_score_threshold,
            stop_margin=stop_margin,
            insert_rollout_gamma=insert_rollout_gamma,
            residue_candidate_k=residue_candidate_k,
            stop_gate_strict=stop_gate_strict,
        )

        labels = torch.tensor(
            [spec["actions"][spec["best_operation"]]["op_id"] for spec in specs],
            dtype=torch.long,
            device=self.device,
        )
        dynamic_op_mask = torch.zeros_like(op_mask)
        for i, spec in enumerate(specs):
            for action in spec["actions"].values():
                dynamic_op_mask[i, action["op_id"]] = True
        op_loss = F.cross_entropy(_masked_logits(op_logits, dynamic_op_mask), labels)
        position_losses = []
        for i, spec in enumerate(specs):
            best_action = spec["actions"][spec["best_operation"]]
            if spec["best_operation"] == "replace":
                pos_log_prob, _ = _log_prob_at(replace_logits[i], replace_mask[i], int(best_action["position"]))
                position_losses.append(-pos_log_prob)
            elif spec["best_operation"] == "insert":
                pos_log_prob, _ = _log_prob_at(insert_logits[i], insert_mask[i], int(best_action["position"]))
                position_losses.append(-pos_log_prob)
            elif spec["best_operation"] == "delete":
                pos_log_prob, _ = _log_prob_at(delete_logits[i], delete_mask[i], int(best_action["position"]))
                position_losses.append(-pos_log_prob)
        if position_losses:
            position_loss = torch.stack(position_losses).mean()
        else:
            position_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        loss = op_loss + planner_position_weight * position_loss
        counts = {name: 0 for name in OP_NAMES.values()}
        for spec in specs:
            counts[spec["best_operation"]] += 1
        return loss, counts

    def run_episode(
        self,
        batch_seqs,
        iterations,
        outer_step,
        batch_num,
        batch_start_index,
        reward_alpha=0.5,
        beta_replace=0.01,
        beta_insert=0.01,
        beta_delete=0.005,
        length_penalty_weight=0.02,
        repeat_penalty_weight=0.02,
        target_length_upper=35,
        max_insert_per_episode=2,
        max_delete_per_episode=1,
        min_length_after_delete=8,
        counterfactual_top_k=3,
        min_edit_steps_before_stop=2,
        stop_score_threshold=0.8,
        stop_margin=0.02,
        stop_patience=2,
        insert_rollout_gamma=0.5,
        residue_candidate_k=5,
        stop_gate_strict=False,
        planner_operation_weight=0.1,
        planner_position_weight=0.05,
        entropy_coef=0.001,
        record_trajectory=True,
    ):
        batch_size = len(batch_seqs)
        current_seqs = list(batch_seqs)
        initial_scores = scoring_function(current_seqs)
        current_scores = initial_scores.copy()
        insert_counts = np.zeros(batch_size, dtype=np.int64)
        delete_counts = np.zeros(batch_size, dtype=np.int64)
        edit_counts = np.zeros(batch_size, dtype=np.int64)
        terminated = np.zeros(batch_size, dtype=bool)
        stop_streaks = np.zeros(batch_size, dtype=np.int64)
        loss_terms = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        nll_terms = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        trajectory_rows: list[dict] = []

        input_ids = [batch_start_index + i for i in range(batch_size)]
        if record_trajectory:
            for i, seq in enumerate(current_seqs):
                phys = physicochemical_scores(seq)
                trajectory_rows.append(
                    {
                        "input_id": input_ids[i],
                        "outer_step": outer_step,
                        "batch_num": batch_num,
                        "inner_step": 0,
                        "operation": "init",
                        "position": -1,
                        "generated_aa": "",
                        "parent_seq": "",
                        "seq": seq,
                        "hyperamp_score": float(current_scores[i]),
                        "reward": float(current_scores[i]),
                        "shaped_score": shaped_sequence_score(seq, float(current_scores[i])),
                        "physicochemical_score": phys["physicochemical_score"],
                        "charge": phys["charge"],
                        "kr_frac": phys["kr_frac"],
                        "hyd_frac": phys["hyd_frac"],
                        "arom_frac": phys["arom_frac"],
                        "pro_frac": phys["pro_frac"],
                        "gly_frac": phys["gly_frac"],
                        "acid_frac": phys["acid_frac"],
                        "hydrophobic_moment": phys["hydrophobic_moment"],
                        "max_aa_frac": phys["max_aa_frac"],
                        "unique_frac": phys["unique_frac"],
                        "pro_penalty": phys["pro_penalty"],
                        "complexity_penalty_value": phys["complexity_penalty_value"],
                        "length": len(seq),
                        "insert_count": int(insert_counts[i]),
                        "delete_count": int(delete_counts[i]),
                        "edit_count": int(edit_counts[i]),
                        "likelihood": 0.0,
                        "advantage": 0.0,
                        "terminated": False,
                        "is_initial": True,
                    }
                )

        for inner_step in range(1, iterations + 1):
            active_indices = [i for i in range(batch_size) if not terminated[i]]
            if not active_indices:
                break

            active_seqs = [current_seqs[i] for i in active_indices]
            active_scores = np.array([current_scores[i] for i in active_indices], dtype=np.float32)
            active_initial_scores = np.array([initial_scores[i] for i in active_indices], dtype=np.float32)
            active_insert_counts = np.array([insert_counts[i] for i in active_indices], dtype=np.int64)
            active_delete_counts = np.array([delete_counts[i] for i in active_indices], dtype=np.int64)

            aa_ids, lengths = encode_aa_sequences(active_seqs, self.max_length, self.device)
            insert_count_tensor = torch.tensor(active_insert_counts, dtype=torch.long, device=self.device)
            delete_count_tensor = torch.tensor(active_delete_counts, dtype=torch.long, device=self.device)
            state_features = torch.tensor(
                [
                    [
                        inner_step / float(iterations),
                        len(seq) / float(self.max_length),
                        active_insert_counts[j] / float(max_insert_per_episode),
                        active_delete_counts[j] / float(max_delete_per_episode),
                        float(active_scores[j]),
                        float(active_scores[j] - active_initial_scores[j]),
                    ]
                    for j, seq in enumerate(active_seqs)
                ],
                dtype=torch.float32,
                device=self.device,
            )

            op_logits, replace_logits, insert_logits, delete_logits = self.policy(aa_ids, lengths, state_features)
            base_op_mask = self._operation_mask(
                lengths,
                insert_count_tensor,
                delete_count_tensor,
                max_insert_per_episode,
                max_delete_per_episode,
                min_length_after_delete,
            )
            replace_mask = self._replace_position_mask(lengths)
            insert_mask = self._insert_position_mask(lengths)
            delete_mask = self._delete_position_mask(lengths, min_length_after_delete)

            replace_position_lists = _topk_masked_indices(replace_logits.detach(), replace_mask, counterfactual_top_k)
            insert_position_lists = _topk_masked_indices(insert_logits.detach(), insert_mask, counterfactual_top_k)
            delete_position_lists = _topk_masked_indices(delete_logits.detach(), delete_mask, counterfactual_top_k)

            specs = self._counterfactual_candidates(
                seqs=active_seqs,
                current_scores=active_scores,
                replace_position_lists=replace_position_lists,
                insert_position_lists=insert_position_lists,
                delete_position_lists=delete_position_lists,
                insert_allowed=base_op_mask[:, OP_INSERT].detach().cpu().numpy(),
                delete_allowed=base_op_mask[:, OP_DELETE].detach().cpu().numpy(),
                inner_step=inner_step,
                reward_alpha=reward_alpha,
                beta_replace=beta_replace,
                beta_insert=beta_insert,
                beta_delete=beta_delete,
                length_penalty_weight=length_penalty_weight,
                repeat_penalty_weight=repeat_penalty_weight,
                target_length_upper=target_length_upper,
                min_edit_steps_before_stop=min_edit_steps_before_stop,
                stop_score_threshold=stop_score_threshold,
                stop_margin=stop_margin,
                insert_rollout_gamma=insert_rollout_gamma,
                residue_candidate_k=residue_candidate_k,
                stop_gate_strict=stop_gate_strict,
            )

            op_mask = torch.zeros_like(base_op_mask)
            for j, spec in enumerate(specs):
                for action in spec["actions"].values():
                    op_mask[j, action["op_id"]] = True
            planner_labels = torch.tensor(
                [spec["actions"][spec["best_operation"]]["op_id"] for spec in specs],
                dtype=torch.long,
                device=self.device,
            )
            planner_operation_loss = F.cross_entropy(
                _masked_logits(op_logits, op_mask),
                planner_labels,
                reduction="none",
            )
            planner_position_losses = []
            for j, spec in enumerate(specs):
                best_action = spec["actions"][spec["best_operation"]]
                if spec["best_operation"] == "replace":
                    pos_log_prob, _ = _log_prob_at(replace_logits[j], replace_mask[j], int(best_action["position"]))
                    planner_position_losses.append(-pos_log_prob)
                elif spec["best_operation"] == "insert":
                    pos_log_prob, _ = _log_prob_at(insert_logits[j], insert_mask[j], int(best_action["position"]))
                    planner_position_losses.append(-pos_log_prob)
                elif spec["best_operation"] == "delete":
                    pos_log_prob, _ = _log_prob_at(delete_logits[j], delete_mask[j], int(best_action["position"]))
                    planner_position_losses.append(-pos_log_prob)
                else:
                    planner_position_losses.append(torch.tensor(0.0, dtype=torch.float32, device=self.device))
            planner_position_loss = torch.stack(planner_position_losses)
            operations, op_log_probs, op_entropies = _sample_masked(op_logits, op_mask)

            step_log_probs = []
            step_entropies = []
            step_operations = []
            step_positions = []
            generated_aas = []
            new_active_seqs = []
            new_active_scores = []
            selected_rewards = []
            selected_advantages = []

            for j, original_index in enumerate(active_indices):
                op = int(operations[j].detach().cpu().item())
                op_name = OP_NAMES[op]
                selected = specs[j]["actions"][op_name]
                log_prob = op_log_probs[j]
                entropy = op_entropies[j]

                if op == OP_REPLACE:
                    pos_log_prob, pos_entropy = _log_prob_at(replace_logits[j], replace_mask[j], int(selected["position"]))
                    log_prob = log_prob + pos_log_prob
                    entropy = entropy + pos_entropy
                    edit_counts[original_index] += int(selected["seq"] != current_seqs[original_index])
                    stop_streaks[original_index] = 0
                elif op == OP_INSERT:
                    pos_log_prob, pos_entropy = _log_prob_at(insert_logits[j], insert_mask[j], int(selected["position"]))
                    log_prob = log_prob + pos_log_prob
                    entropy = entropy + pos_entropy
                    insert_counts[original_index] += 1
                    edit_counts[original_index] += 1
                    stop_streaks[original_index] = 0
                elif op == OP_DELETE:
                    pos_log_prob, pos_entropy = _log_prob_at(delete_logits[j], delete_mask[j], int(selected["position"]))
                    log_prob = log_prob + pos_log_prob
                    entropy = entropy + pos_entropy
                    delete_counts[original_index] += 1
                    edit_counts[original_index] += 1
                    stop_streaks[original_index] = 0
                elif op == OP_STOP:
                    stop_streaks[original_index] += 1
                    if stop_streaks[original_index] >= stop_patience:
                        terminated[original_index] = True
                else:
                    raise ValueError(f"Unknown operation id: {op}")

                current_seqs[original_index] = selected["seq"]
                current_scores[original_index] = float(selected["score"])
                new_active_seqs.append(selected["seq"])
                new_active_scores.append(float(selected["score"]))
                step_log_probs.append(log_prob)
                step_entropies.append(entropy)
                step_operations.append(op_name)
                step_positions.append(int(selected["position"]))
                generated_aas.append(selected["generated_aa"])
                selected_rewards.append(float(selected["reward"]))
                selected_advantages.append(float(selected["advantage"]))

            advantage_tensor = torch.tensor(selected_advantages, dtype=torch.float32, device=self.device)
            log_prob_tensor = torch.stack(step_log_probs)
            entropy_tensor = torch.stack(step_entropies)
            active_loss = (-log_prob_tensor * advantage_tensor.detach()) - entropy_coef * entropy_tensor
            active_loss = (
                active_loss
                + planner_operation_weight * planner_operation_loss
                + planner_position_weight * planner_position_loss
            )
            for j, original_index in enumerate(active_indices):
                loss_terms[original_index] = loss_terms[original_index] + active_loss[j]
                nll_terms[original_index] = nll_terms[original_index] + (-log_prob_tensor[j].detach())

            if record_trajectory:
                parent_seqs = [spec["parent_seq"] for spec in specs]
                distances = levenshtein(parent_seqs, new_active_seqs)
                for j, original_index in enumerate(active_indices):
                    selected_action = specs[j]["actions"][step_operations[j]]
                    trajectory_rows.append(
                        {
                            "input_id": input_ids[original_index],
                            "outer_step": outer_step,
                            "batch_num": batch_num,
                            "inner_step": inner_step,
                            "operation": step_operations[j],
                            "position": int(step_positions[j]),
                            "generated_aa": generated_aas[j],
                            "parent_seq": parent_seqs[j],
                            "seq": new_active_seqs[j],
                            "hyperamp_score": float(new_active_scores[j]),
                            "reward": float(selected_rewards[j]),
                            "shaped_score": float(selected_action["shaped_score"]),
                            "physicochemical_score": float(selected_action["physicochemical_score"]),
                            "charge": float(selected_action["charge"]),
                            "kr_frac": float(selected_action["kr_frac"]),
                            "hyd_frac": float(selected_action["hyd_frac"]),
                            "arom_frac": float(physicochemical_features(new_active_seqs[j])["arom_frac"]),
                            "pro_frac": float(selected_action["pro_frac"]),
                            "gly_frac": float(physicochemical_features(new_active_seqs[j])["gly_frac"]),
                            "acid_frac": float(physicochemical_features(new_active_seqs[j])["acid_frac"]),
                            "hydrophobic_moment": float(physicochemical_features(new_active_seqs[j])["hydrophobic_moment"]),
                            "max_aa_frac": float(selected_action["max_aa_frac"]),
                            "unique_frac": float(physicochemical_features(new_active_seqs[j])["unique_frac"]),
                            "pro_penalty": float(selected_action["pro_penalty"]),
                            "complexity_penalty_value": float(selected_action["complexity_penalty_value"]),
                            "advantage": float(selected_advantages[j]),
                            "planner_operation_loss": float(planner_operation_loss[j].detach().cpu().item()),
                            "planner_position_loss": float(planner_position_loss[j].detach().cpu().item()),
                            "cf_replace_reward": float(specs[j]["actions"]["replace"]["reward"]),
                            "cf_insert_reward": float(specs[j]["actions"]["insert"]["reward"]) if "insert" in specs[j]["actions"] else np.nan,
                            "cf_delete_reward": float(specs[j]["actions"]["delete"]["reward"]) if "delete" in specs[j]["actions"] else np.nan,
                            "cf_stop_reward": float(specs[j]["stop_reward"]),
                            "cf_replace_shaped": float(specs[j]["actions"]["replace"]["shaped_score"]),
                            "cf_insert_shaped": float(specs[j]["actions"]["insert"]["shaped_score"]) if "insert" in specs[j]["actions"] else np.nan,
                            "cf_delete_shaped": float(specs[j]["actions"]["delete"]["shaped_score"]) if "delete" in specs[j]["actions"] else np.nan,
                            "cf_stop_shaped": float(specs[j]["candidates"]["stop"][0]["shaped_score"]),
                            "cf_best_operation": specs[j]["best_operation"],
                            "stop_allowed": bool(specs[j]["stop_allowed"]),
                            "stop_physically_allowed": bool(specs[j]["stop_physically_allowed"]),
                            "best_edit_reward": float(specs[j]["best_edit_reward"]),
                            "stop_streak": int(stop_streaks[original_index]),
                            "length": len(new_active_seqs[j]),
                            "insert_count": int(insert_counts[original_index]),
                            "delete_count": int(delete_counts[original_index]),
                            "edit_count": int(edit_counts[original_index]),
                            "likelihood": float(nll_terms[original_index].detach().cpu().item()),
                            "distance_from_parent": int(distances[j]),
                            "terminated": bool(terminated[original_index]),
                            "is_initial": False,
                        }
                    )

        trajectory = pd.DataFrame(trajectory_rows)
        return current_seqs, loss_terms, nll_terms, trajectory, current_scores
