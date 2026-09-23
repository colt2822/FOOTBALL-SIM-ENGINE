"""V2.1 final scoring. TEST 2020-2025 is read exactly once, with the DEV-frozen
candidate. No selection happens here."""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C
from . import experiment as E
from . import features as F
from . import features_v21 as V21
from . import firewall as FW
from . import metrics as M
from . import models as Mo

NOW = datetime.now(timezone.utc).isoformat()
REGIME = dict(hetero=True, resid_mode="prior_oos", calibration="isotonic")
PROB_COL = "p_raw"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_prev_game_share_offset(full: pd.DataFrame) -> dict:
    """`prev_game_attempt_share` must come from the player's prior appearance.

    An off-by-one here would leak the current game's team attempt total, which
    the |r|>0.90 correlation scan is too blunt to catch.
    """
    df = full.sort_values(["player_id", "kickoff_timestamp_utc"])
    viol = checked = 0
    for _, sub in df.groupby("player_id"):
        if len(sub) < 3:
            continue
        own = sub["own_share_this_game"].values
        prev = sub["prev_game_attempt_share"].values
        for i in range(1, len(sub)):
            if np.isfinite(prev[i]) and np.isfinite(own[i - 1]):
                checked += 1
                if abs(prev[i] - own[i - 1]) > 1e-9:
                    viol += 1
                # must NOT equal the current game's own share unless coincidental
        if checked > 30000:
            break
    return {"rows_checked": checked, "violations": viol, "PASS": viol == 0}


def check_offseason_state_separation(el: pd.DataFrame) -> dict:
    """A season opener and an in-season absence must not share a feature vector
    on the state block."""
    op = el[(el.days_rest >= 29.9) & (el.week == 1)]
    ab = el[(el.days_rest >= 29.9) & (el.week > 1) &
            (el.crossed_season_boundary == 0)]
    return {
        "season_opener_n": int(len(op)),
        "in_season_absence_n": int(len(ab)),
        "opener_in_season_days_rest_all_nan": bool(op.in_season_days_rest.isna().all()),
        "opener_team_games_missed_all_zero": bool((op.team_games_missed == 0).all()),
        "absence_offseason_gap_all_zero": bool((ab.offseason_gap_days == 0).all()),
        "absence_team_games_missed_mean": float(ab.team_games_missed.mean()),
        "PASS": bool(op.in_season_days_rest.isna().all()
                     and (op.team_games_missed == 0).all()
                     and (ab.offseason_gap_days == 0).all()),
    }


def bucket_table(oos: pd.DataFrame, el: pd.DataFrame, seasons=None) -> dict:
    j = oos.join(el[["days_rest", "crossed_season_boundary", "career_lead_share",
                     "prior_season_lead_games"]], rsuffix="_x")
    if seasons is not None:
        j = j[j.oos_season.isin(seasons)]
    out = {}
    for k, m in V21.buckets(j).items():
        s = j[m]
        if len(s) < 20:
            continue
        blk = M.score_block(s[C.TARGET].values, s.pred.values,
                            s[PROB_COL].values, C.PROB_THRESHOLD)
        blk["BIAS"] = float((s.pred - s[C.TARGET]).mean())
        out[k] = blk
    return out


def main():
    t0 = time.time()
    sel = json.loads((C.ARTIFACTS / "SPORTS_NOVA_V2_1_DEV_SELECTION.json").read_text())
    best = sel["SELECTED_CANDIDATE"]
    feats_v21 = sel["SELECTED_FEATURES"]
    print(f"frozen V2.1 candidate: {best} ({len(feats_v21)} features)", flush=True)

    el, led, full = V21.build_v21_panel(return_full=True)

    fw = FW.run_all(el, full)
    fw["prev_game_share_offset"] = check_prev_game_share_offset(full)
    fw["offseason_state_separation"] = check_offseason_state_separation(el)
    fw["TEMPORAL_FIREWALL"] = "PASS" if all(
        v.get("PASS") for v in fw.values() if isinstance(v, dict)) else "FAIL"
    print("firewall:", fw["TEMPORAL_FIREWALL"], flush=True)

    runs = {}
    for label, feats in (("CURRENT_V2", V21.CANDIDATES["A_current_v2"]),
                         ("V2_1", feats_v21)):
        oos, folds = E.run_candidate(el, Mo.make_blend(feats, weight=0.5), **REGIME)
        assert all(f.cutoff_ok for f in folds)
        runs[label] = oos
        print(f"  {label} done", flush=True)

    result = {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_1_WALK_FORWARD_RESULTS",
        "GENERATED_AT": NOW,
        "SELECTED_CANDIDATE": best,
        "SELECTION_RULE": sel["SELECTION_RULE"],
        "PASSED_GUARDRAILS": sel["PASSED_GUARDRAILS"],
        "N_FEATURES": {"CURRENT_V2": len(V21.CANDIDATES["A_current_v2"]),
                       "V2_1": len(feats_v21)},
        "NEW_FEATURES_ADDED": [f for f in feats_v21
                               if f not in V21.CANDIDATES["A_current_v2"]],
        "TEMPORAL_FIREWALL": fw,
        "ROWS": {"eligible": int(len(el)), "oos_n": int(len(runs["V2_1"]))},
    }
    for label, oos in runs.items():
        result[label] = {
            "PERIODS": E.score_all_periods(oos, PROB_COL),
            "BUCKETS_POOLED": bucket_table(oos, el),
            "BUCKETS_TEST": bucket_table(oos, el, C.TEST_SEASONS),
            "BUCKETS_DEV": bucket_table(oos, el, C.DEV_SEASONS),
        }

    v2t = result["CURRENT_V2"]["PERIODS"]["TEST_2020_2025"]
    v21t = result["V2_1"]["PERIODS"]["TEST_2020_2025"]
    result["IMPROVEMENT_TEST_OVERALL"] = E.improvement(v2t, v21t)
    result["OVERALL_V2_1_BEATS_CURRENT_V2"] = bool(
        v21t["MAE"] < v2t["MAE"] and v21t["BRIER"] < v2t["BRIER"])

    bt2, bt21 = result["CURRENT_V2"]["BUCKETS_TEST"], result["V2_1"]["BUCKETS_TEST"]
    result["BUCKET_DELTAS_TEST"] = {
        k: {"N": bt21[k]["N"],
            "MAE_v2": bt2[k]["MAE"], "MAE_v2_1": bt21[k]["MAE"],
            "MAE_DELTA": round(bt21[k]["MAE"] - bt2[k]["MAE"], 4),
            "BIAS_v2": round(bt2[k]["BIAS"], 3),
            "BIAS_v2_1": round(bt21[k]["BIAS"], 3),
            "ABS_BIAS_DELTA": round(abs(bt21[k]["BIAS"]) - abs(bt2[k]["BIAS"]), 3)}
        for k in bt21 if k in bt2}

    with open(C.ARTIFACTS / "_v21_state.pkl", "wb") as fh:
        pickle.dump({"el": el, "runs": runs, "result": result, "feats": feats_v21}, fh)
    p = C.ARTIFACTS / "SPORTS_NOVA_V2_1_WALK_FORWARD_RESULTS.json"
    p.write_text(json.dumps(result, indent=2, default=str))

    print("\n=== TEST 2020-2025 overall ===")
    for k in ("MAE", "RMSE", "BRIER", "LOGLOSS", "ECE"):
        print(f"  {k:8s} V2={v2t[k]:.5f}  V2.1={v21t[k]:.5f}")
    print("\n=== TEST buckets (MAE / bias) ===")
    for k, v in result["BUCKET_DELTAS_TEST"].items():
        print(f"  {k:34s} n={v['N']:5d}  MAE {v['MAE_v2']:6.2f}->{v['MAE_v2_1']:6.2f} "
              f"({v['MAE_DELTA']:+.2f})   bias {v['BIAS_v2']:+7.2f}->{v['BIAS_v2_1']:+7.2f}")
    print(f"\ntotal {round(time.time()-t0,1)}s")


if __name__ == "__main__":
    main()
