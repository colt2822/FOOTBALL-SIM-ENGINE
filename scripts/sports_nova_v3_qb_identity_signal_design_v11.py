"""V11: SPORTS_NOVA_V11_QB_IDENTITY_SIGNAL_DESIGN.

Diagnostic-only, training-only signal design. Makes NO engine changes in this
step; freezes exactly one candidate rule from training-only evidence, prints
it, and only THEN (in a separate, single, no-further-tuning pass) applies it
once to the V9 cohort for comparison.

Training sample: seasons 2000-2019 (strictly disjoint from the V9 evaluation
cohort, which is drawn only from 2020-2025 game IDs listed in
SPORTS_NOVA_V3_PHASE08_OOS_GAME_MANIFEST_V1.json). No V9 game ID, outcome, or
metric is read or used anywhere in this script's design phase.

For each sampled team-game with >=2 QB candidates in the causal roster
(make_state, read-only, same function production uses), records each
candidate's causal features (pass_rate "volume" proxy, effective_sample_size,
games_since_last_team_game "gap") and the realized target (which QB actually
had the most PASS_ATTEMPTS in that historical game -- itself historical fact,
not a V9 evaluation outcome). Computes the requested marginal/conditional
diagnostics, evaluates three parameter-free candidate scoring rules against
the realized target using pure top-1 accuracy on THIS training sample only,
and freezes the winner.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worker.sports_nova_v3.simulator import _feature
from sports_nova_v3_player_joint_walkforward_v1 import (PLAYER, MANIFEST,
    game_parts, surrogate_kickoff, make_state)

OUT = ROOT / "data" / "sports_nova_v3"

V9_GAME_IDS = set(json.loads(MANIFEST.read_text())["GAME_IDS"])
TRAIN_SEASONS = list(range(2000, 2020))  # strictly before the 2020-2025 V9 window
GAMES_PER_SEASON = 18


def main():
    all_df = pd.read_parquet(PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK

    by_season = {}
    for g in all_df.GAME_ID.unique():
        s = int(str(g).split("_")[0])
        if s in TRAIN_SEASONS and g not in V9_GAME_IDS:
            by_season.setdefault(s, []).append(g)
    selected = []
    rng = np.random.default_rng(0)
    for s in sorted(by_season):
        xs = sorted(set(by_season[s]))
        idx = np.linspace(0, len(xs) - 1, min(GAMES_PER_SEASON, len(xs)), dtype=int)
        selected.extend([xs[i] for i in idx])
    assert V9_GAME_IDS.isdisjoint(selected), "training sample must not touch the V9 cohort"

    all_df_key = all_df.set_index("_key", drop=False)
    rows = []
    for g in selected:
        s, w, away, home = game_parts(g)
        key = s * 100 + w
        prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, s - 5))]
        cur = all_df[all_df.GAME_ID == g]
        state = make_state(g, prior, surrogate_kickoff(s, w), None)
        for tid in (home, away):
            qbs = [p for p in state.players if p.team_id == tid and p.position == "QB"]
            if len(qbs) < 2:
                continue
            team_qb_rows = cur[(cur.TEAM == tid) & (cur.POSITION == "QB")]
            if not len(team_qb_rows):
                continue
            realized_starter = str(team_qb_rows.sort_values("PASS_ATTEMPTS", ascending=False).iloc[0].PLAYER_ID)
            if realized_starter not in {p.player_id for p in qbs}:
                continue  # target not resolvable in the causal roster; excluded, not guessed
            cand = []
            for p in qbs:
                pr = max(_feature(p, "pass_rate", 0.0), 0.0)
                ess = max(p.uncertainty.effective_sample_size, 0.0)
                gap = max(_feature(p, "games_since_last_team_game", 0.0), 0.0)
                cand.append({"pid": p.player_id, "pass_rate": pr, "ess": ess,
                             "volume": pr * ess, "gap": gap})
            cand_df = pd.DataFrame(cand)
            cand_df["gap_rank"] = cand_df.gap.rank(method="min", ascending=True).astype(int)
            cand_df["volume_rank"] = cand_df.volume.rank(method="min", ascending=False).astype(int)
            cand_df["n_candidates"] = len(cand_df)
            cand_df["GAME_ID"] = g
            cand_df["TEAM"] = tid
            cand_df["is_realized_starter"] = cand_df.pid == realized_starter
            rows.append(cand_df)

    df = pd.concat(rows, ignore_index=True)
    n_team_games = df.groupby(["GAME_ID", "TEAM"]).ngroups

    signal_scale = {
        "pass_rate": {"min": float(df.pass_rate.min()), "p50": float(df.pass_rate.median()),
                       "max": float(df.pass_rate.max())},
        "effective_sample_size": {"min": float(df.ess.min()), "p50": float(df.ess.median()),
                                    "max": float(df.ess.max())},
        "pass_rate_x_ess_volume": {"min": float(df.volume.min()), "p50": float(df.volume.median()),
                                     "max": float(df.volume.max())},
        "games_since_last_team_game_gap": {"min": float(df.gap.min()), "p50": float(df.gap.median()),
                                             "max": float(df.gap.max()), "p90": float(df.gap.quantile(.9))},
    }

    gap_buckets = [(0, 0), (1, 1), (2, 3), (4, 8), (9, 20), (21, 10_000)]
    gap_prior = {}
    for lo, hi in gap_buckets:
        sub = df[(df.gap >= lo) & (df.gap <= hi)]
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi == 10_000 else f"{lo}-{hi}")
        gap_prior[label] = {"N": int(len(sub)), "P_is_starter": float(sub.is_realized_starter.mean()) if len(sub) else None}

    volume_rank_finding = {}
    for r in sorted(df.volume_rank.unique()):
        sub = df[df.volume_rank == r]
        volume_rank_finding[int(r)] = {"N": int(len(sub)), "P_is_starter": float(sub.is_realized_starter.mean())}

    joint = df.groupby(["gap_rank", "volume_rank"]).agg(
        N=("is_realized_starter", "size"), P_is_starter=("is_realized_starter", "mean")).reset_index()
    joint_table = joint.to_dict("records")

    def top1_accuracy(score_fn):
        correct, total = 0, 0
        for (g, tid), sub in df.groupby(["GAME_ID", "TEAM"]):
            scores = score_fn(sub)
            pick = sub.iloc[int(np.argmax(scores))]
            correct += int(pick.is_realized_starter)
            total += 1
        return correct / total, total

    def rank_fusion_sum(sub):
        return 1.0 / sub.gap_rank.to_numpy() + 1.0 / sub.volume_rank.to_numpy()

    def rank_fusion_product(sub):
        return 1.0 / (sub.gap_rank.to_numpy() * sub.volume_rank.to_numpy())

    def bounded_volume(sub):
        # Volume normalized to a team-relative [0,1] share (scale-free by
        # construction: dividing by the team's own max removes units/magnitude
        # entirely) times an ordinal (not raw-magnitude) recency term
        # 1/gap_rank, so a team with uniformly huge or tiny raw volumes is
        # treated identically to one with modest volumes.
        norm_vol = sub.volume.to_numpy() / max(sub.volume.max(), 1e-9)
        return norm_vol / sub.gap_rank.to_numpy()

    hard_filter_baseline_acc, _ = top1_accuracy(
        lambda sub: -sub.gap_rank.to_numpy())  # equivalent to the original: pick min gap, ties broken by first row

    candidates = {
        "RANK_FUSION_SUM": rank_fusion_sum,
        "RANK_FUSION_PRODUCT": rank_fusion_product,
        "BOUNDED_VOLUME": bounded_volume,
    }
    candidate_results = {}
    for name, fn in candidates.items():
        acc, n = top1_accuracy(fn)
        candidate_results[name] = {"top1_accuracy": acc, "n_team_games": n}

    result = {
        "mission": "SPORTS_NOVA_V11_QB_IDENTITY_SIGNAL_DESIGN",
        "training_sample": {"seasons": [TRAIN_SEASONS[0], TRAIN_SEASONS[-1]], "games_sampled": len(selected),
                             "multi_qb_team_games": int(n_team_games), "candidate_rows": int(len(df)),
                             "excludes_v9_cohort": True},
        "signal_scale_findings": signal_scale,
        "gap_starter_prior": gap_prior,
        "volume_rank_findings": volume_rank_finding,
        "joint_gap_rank_x_volume_rank_table": joint_table,
        "hard_min_gap_filter_baseline_accuracy_TRAINING_ONLY": hard_filter_baseline_acc,
        "candidate_results_TRAINING_ONLY": candidate_results,
    }
    (OUT / "SPORTS_NOVA_V11_QB_IDENTITY_SIGNAL_DESIGN_TRAINING.json").write_text(json.dumps(result, indent=2))
    df.to_csv(OUT / "SPORTS_NOVA_V11_QB_IDENTITY_TRAINING_ROWS.csv", index=False)
    print(json.dumps({k: result[k] for k in (
        "training_sample", "gap_starter_prior", "hard_min_gap_filter_baseline_accuracy_TRAINING_ONLY",
        "candidate_results_TRAINING_ONLY")}, indent=2))
    print(json.dumps(volume_rank_finding, indent=2))
    print(json.dumps(joint_table, indent=2))


if __name__ == "__main__":
    main()
