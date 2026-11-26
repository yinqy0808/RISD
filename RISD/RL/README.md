# RIS workflow

This folder implements the replace / insert / delete / stop AMPainter extension.

Main entry points:

- `train.py`: trains the RIS policy network with HyperAMP-guided physicochemical shaped reward.
- `evolve.py`: generates candidates with the trained RIS policy and HyperAMP-guided physicochemical shaped reward.
- `select_topk.py`: selects deduplicated top50/top100/top200 from the existing `all_candidates.csv` order.

Key constraints:

- `max_length <= 40`
- one-residue insertion only
- internal-gap insertion only
- `max_insert_per_episode = 2`
- `max_delete_per_episode = 1`
- `min_length_after_delete = 8`
- `residue_candidate_k = 5`
- high-score AMP-like policy seeds are selected from `data/train.txt` positive rows (`pos_*`) by measured pMIC
- MBC and AMPread are not used during training, generation, or topK selection
- MBC, AMPread, and diversity are evaluated after selection and are not part of the selection score

Training logic:

- each active sequence builds best-of-K action-level counterfactual candidates for replace, insert, and delete
- replace and insert generate multiple residue candidates per selected position and rerank them by shaped reward
- insert planner value includes a lightweight local 2-step rollout: insert, then best nearby replace
- delete removes one residue directly, does not call the RIS Ankh generator, and is scored by HyperAMP like other edits
- delete receives a small bonus when it reduces Pro-richness or low complexity
- stop joins the action competition only after HyperAMP and physicochemical stop gates are satisfied
- policy-gradient updates use selected-action advantage over the available action rewards, not raw positive reward
- stop terminates that sequence only after consecutive stop decisions reach the configured patience
- before RL, the operation head and valid position heads are warmed up with best-of-K planner pseudo-labels
- during RL, the loss keeps a continuous planner imitation term for the operation head and the planner best position
- checkpoints are selected by mean shaped reward, not mean HyperAMP alone
- `KR_score` reaches full reward at `KR_frac >= 0.32`, so the policy is encouraged past the weak 0.25 boundary
- hydrophobic moment is logged, used as a HyperAMP-tie candidate ordering signal, and added as a small shaped reward term
- `length_score` gives full reward to the broad 18-24 AMP length range

Shaped reward:

```text
shaped_reward = HyperAMP
              + 0.23 * KR_score
              + 0.18 * charge_score
              + 0.04 * hyd_score
              + 0.04 * length_score
              + 0.02 * moment_score
              - 0.10 * pro_penalty
              - 0.05 * complexity_penalty
```

Final candidate pool:

```text
all_candidates.csv is built by concatenating every step_xxx/record.csv.
Duplicates are kept.
```

Final candidate ordering:

```text
1. hyperamp_score desc
2. shaped_score desc
3. charge_good desc
4. basic_good desc
5. pro_good desc
6. hyd_good desc
7. moment_good desc
8. length_good desc
9. low_complexity_penalty asc
10. abs(length - 20) asc
11. likelihood asc
12. source_step asc
13. source_rank asc
```

TopK selection:

```text
top50/top100/top200 preserve the ordered all_candidates.csv ranking, keep each sequence once, and then take topK.
```
