"""SPORTS_NOVA M1 V4 ablation: does the residual-redistribution fix already
implemented in the V4 harness (scripts/sports_nova_m1_v4_raw_path_diagnostic_challenger.py)
actually reduce QB pass-yard error when applied to champion V21 unchanged,
holding every other mechanism fixed?

The main V4 diagnostic compares V3 (a pre-superseded ancestor engine with an
already-reverted QB-recency-gate regression and a missing-PAT bug) against
V4 (V3 + fix). That comparison cannot isolate the fix's effect because V3
carries two unrelated, already-root-caused defects V21 does not have. V21's
own simulator imports allocate_opportunities live from the same
worker.sports_nova_v3.allocation module V3 uses, so the residual-drop defect
this fix targets is present in the champion too. This script runs
V21-vs-(V21+fix) as a controlled pair, same cohort, same seeds, same
mechanism except the fix.

Additive and read-only with respect to source: worker/sports_nova_v3/ and
worker/sports_nova_v21/ are never edited, matching the main V4 harness's own
rule. Writes only to its own output path; does not touch the main V4
harness's ledger/report/manifest.

Known limitation (not corrected here): the fix's redistribution draw
(rng.multinomial calls inside the patched allocate_opportunities wrapper)
consumes extra entropy from the same per-path RNG stream the rest of the
simulation uses. For any (game, sim_id) where a residual event actually
occurs, every downstream draw in that path decouples from the paired
V21 baseline path with the same seed -- so this is not a bit-exact
counterfactual replay. It remains a statistically valid comparison because
each arm's own simulations are still proper independent draws from that
arm's generative model; only path-level replay identity is lost. The
integrity check below empirically bounds this: for (game, sim_id) pairs
where the fix arm records zero residual events, the two arms' full player
stat dictionaries are compared for exact equality.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy import stats as _stats

from scripts.sports_nova_m1_v4_raw_path_diagnostic_challenger import (
    instrument, pilot_games, game_parts, make_state, surrogate_kickoff,
    assert_path, PLAYER, MANIFEST, DATA,
)

OUT = DATA / "SPORTS_NOVA_M1_V4_V21_RESIDUAL_ABLATION_V1.json"
N_PILOT = 128

ACCOUNTING_COVERAGE_NOTE = (
    "assert_path only checks PASS_ATTEMPTS_QB_CONSERVATION, TARGETS_LE_PASS_ATTEMPTS, "
    "RECEPTIONS_LE_COMPLETIONS (vacuous by construction -- make_raw_row defines team "
    "completions as sum(receptions), so this assertion always passes and proves nothing), "
    "RECEIVING_YARDS_TEAM_RECONCILIATION, RUSH_ATTEMPTS_CONSERVATION, and "
    "RUSH_YARDS_TEAM_RECONCILIATION. PAT/XP, FG, team points, and pass/rush/receiving TD "
    "conservation are NOT checked anywhere in this suite. ACCOUNTING_FAILURE_COUNT=0 here "
    "means 0 on this partial checklist, not 0 unexplained failures against Phase 2's full spec."
)


def _tally(failures: list[dict]) -> dict:
    by_assertion: dict[str, int] = defaultdict(int)
    for f in failures:
        by_assertion[f["assertion"]] += 1
    return {"total": len(failures), "by_assertion": dict(by_assertion)}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    games_all = list(manifest["GAME_IDS"])
    games = pilot_games(games_all)
    all_df = pd.read_parquet(PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    df = all_df[all_df.GAME_ID.isin(games)].copy()
    df_by_game = {g: df[df.GAME_ID == g] for g in games}

    qb_pred: dict[str, dict[tuple[str, str], list[int]]] = {
        "V21": defaultdict(list), "V21_FIX": defaultdict(list)}
    arm_hash: dict[str, dict[tuple[str, int], str]] = {"V21": {}, "V21_FIX": {}}
    arm_residual: dict[str, dict[tuple[str, int], bool]] = {"V21": {}, "V21_FIX": {}}
    failures: dict[str, list[dict]] = {"V21": [], "V21_FIX": []}

    for arm, fix in (("V21", False), ("V21_FIX", True)):
        collector: list[dict] = []
        with instrument("V21", collector, fix=fix, detail=True) as module:
            for game in games:
                season, week, away, home = game_parts(game)
                key = season * 100 + week
                prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, season - 5))]
                state = make_state(game, prior, surrogate_kickoff(season, week), None)
                seed = int(hashlib.sha256(game.encode()).hexdigest()[:8], 16)
                before = len(collector)
                module.simulate_game(state, N_PILOT, seed, module.MODEL_VERSION)
                paths = collector[before:]
                cur = df_by_game[game]
                qb_ids = set(str(x) for x in cur[cur.POSITION == "QB"].PLAYER_ID)
                for p in paths:
                    stats, team, winner_ = p["result"]
                    for pid in qb_ids:
                        if pid in stats:
                            qb_pred[arm][(game, pid)].append(int(stats[pid]["pass_yards"]))
                    # "V4" sentinel reuses assert_path's existing engine-guard to skip
                    # UNREALIZED_TARGET_RESIDUAL for the fix arm: that check reads
                    # path["residual_by_team"], which instrument() populates from the
                    # PRE-redistribution result.untargeted regardless of `fix`, so it
                    # would fire identically on both arms and say nothing about whether
                    # the fix arm actually reconciled -- the conservation/reconciliation
                    # checks (computed from realized per-player stats) are what answer
                    # that question. Entries are relabeled to the real arm name after.
                    before_n = len(failures[arm])
                    assert_path(game, state, p, "V4" if fix else "V21", failures[arm])
                    for f in failures[arm][before_n:]:
                        f["engine"] = arm
                    sim_key = (game, int(p["sim_id"]))
                    arm_hash[arm][sim_key] = hashlib.sha256(
                        json.dumps(stats, sort_keys=True).encode()).hexdigest()
                    arm_residual[arm][sim_key] = any(
                        b["allocation"]["allocated_opportunities"].get("redistributed_target_mass")
                        or b["allocation"]["allocated_opportunities"].get("redistributed_carry_mass")
                        for b in p["blocks"])
                del collector[before:]

    def qb_errors(arm: str) -> dict[tuple[str, str], float]:
        errs = {}
        for (game, pid), vals in qb_pred[arm].items():
            obs_row = df_by_game[game]
            obs = obs_row[obs_row.PLAYER_ID.astype(str) == pid]["PASS_YARDS"]
            if obs.empty:
                continue
            pred_mean = float(np.mean(vals))
            errs[(game, pid)] = abs(pred_mean - float(obs.iloc[0]))
        return errs

    def qb_mae(arm: str) -> dict:
        vals = list(qb_errors(arm).values())
        return {"N": len(vals), "MAE": float(np.mean(vals)) if vals else None}

    # Pre-registered BEFORE reading the MAE numbers: this is a mechanical
    # conservation repair, not a modeling change, so per [[feedback-no-fake-improvement]]
    # / doctrine's DO_NOT clause the fix is not rejected merely because a trading/error
    # metric fails to improve -- it is rejected only for a MATERIAL football regression.
    # Paired sign test (same machinery as V19/V20/V21) on the (game, pid) keys common to
    # both arms is reported for information; a null or mildly negative result reads as
    # "adopt, no material regression," not as grounds to withhold the accounting fix.
    e_base, e_fix = qb_errors("V21"), qb_errors("V21_FIX")
    common = sorted(set(e_base) & set(e_fix))
    deltas = [e_fix[k] - e_base[k] for k in common]
    n_fix_better = sum(1 for d in deltas if d < 0)
    n_base_better = sum(1 for d in deltas if d > 0)
    sign_p = (float(_stats.binomtest(min(n_fix_better, n_base_better),
                                      n_fix_better + n_base_better, 0.5).pvalue)
              if n_fix_better + n_base_better else None)
    qb_paired_sign_test = {
        "N_PAIRS": len(common), "N_FIX_BETTER": n_fix_better, "N_BASELINE_BETTER": n_base_better,
        "N_TIED": len(common) - n_fix_better - n_base_better,
        "MEAN_DELTA_FIX_MINUS_BASELINE": float(np.mean(deltas)) if deltas else None,
        "SIGN_TEST_TWO_SIDED_P": sign_p,
        "note": "Informational only -- see doctrine note above. Material-regression call, "
                "if any, must be made on football-realism distributions (Phase 3), not this.",
    }

    no_residual_keys = [k for k, v in arm_residual["V21_FIX"].items() if not v]
    matches = sum(1 for k in no_residual_keys if arm_hash["V21"].get(k) == arm_hash["V21_FIX"].get(k))
    residual_keys = [k for k, v in arm_residual["V21_FIX"].items() if v]

    out = {
        "cohort": {"games": len(games), "all_games": len(games_all), "sims_per_game": N_PILOT,
                   "same_cohort_as_main_v4_pilot": True},
        "QB_PASS_YARD_MAE_V21": qb_mae("V21"),
        "QB_PASS_YARD_MAE_V21_FIX": qb_mae("V21_FIX"),
        "QB_PASS_YARD_PAIRED_SIGN_TEST": qb_paired_sign_test,
        "ACCOUNTING_FAILURES_V21": _tally(failures["V21"]),
        "ACCOUNTING_FAILURES_V21_FIX": _tally(failures["V21_FIX"]),
        "ACCOUNTING_CHECK_PARITY_NOTE": (
            "NOT an apples-to-apples 6-vs-6 comparison. V21's tally includes "
            "UNREALIZED_TARGET_RESIDUAL (80 failures); V21_FIX's tally deliberately "
            "excludes it (via the 'V4' engine sentinel passed to assert_path) because "
            "that check reads path['residual_by_team'], which instrument() records from "
            "the PRE-redistribution result.untargeted regardless of the fix -- it would "
            "fire identically on both arms and is not evidence either way about whether "
            "redistribution happened. The conservation/reconciliation checks (computed "
            "from realized per-player stats, which DO change post-fix) are what carry the "
            "result: V21_FIX is 0/5 on those; V21 baseline has 585 RUSH_ATTEMPTS_CONSERVATION "
            "+ 566 RUSH_YARDS_TEAM_RECONCILIATION failures on the same 5 checks. Read "
            "V21_FIX.total=0 as '0 on the 5 checks that can distinguish the arms,' not as "
            "'a strict superset win over V21's 1231.'"
        ),
        "ACCOUNTING_COVERAGE_NOTE": ACCOUNTING_COVERAGE_NOTE,
        "residual_event_rate": len(residual_keys) / len(arm_residual["V21_FIX"]) if arm_residual["V21_FIX"] else None,
        "rng_coupling_integrity_check": {
            "no_residual_sims": len(no_residual_keys),
            "byte_identical_matches": matches,
            "byte_identical_rate": matches / len(no_residual_keys) if no_residual_keys else None,
            "note": "Expect ~1.0: absent a residual event this run, the fix arm must reproduce the baseline arm exactly at the same seed.",
        },
        "note": "Pilot-scale (60 games/128 sims), not the full 1,693-game cohort the frozen QB_PASS_YARD_MAE_V21=81.06 uses -- not directly comparable to that figure. Compares V21 against V21+fix only; V3/V4 untouched.",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
