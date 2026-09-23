"""Final scored comparison: V1 as published vs V2 with the DEV-frozen config.

TEST (2020-2025) is scored exactly once here. No selection happens in this
script; it reads the frozen choices from SPORTS_NOVA_V2_DEV_SELECTION.json.
"""
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
from . import firewall as FW
from . import metrics as M
from . import models as Mo
from .v1_baseline import v1_fit_predict

NOW = datetime.now(timezone.utc).isoformat()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(name: str, feats: list[str]):
    if name.startswith("D_blend_50"):
        return Mo.make_blend(feats, weight=0.5)
    if name.startswith("D2_blend_65"):
        return Mo.make_blend(feats, weight=0.65)
    if name.startswith("B2"):
        return Mo.make_ridge(feats, alpha=50.0)
    if name.startswith("B"):
        return Mo.make_ridge(feats, alpha=10.0)
    if name.startswith("C4"):
        return Mo.make_lgbm(feats, dict(num_leaves=7, max_depth=4,
                                        learning_rate=0.05), n_estimators=600)
    if name.startswith("C3"):
        return Mo.make_lgbm(feats, dict(num_leaves=31, max_depth=7,
                                        learning_rate=0.03), n_estimators=700)
    return Mo.make_lgbm(feats)


def breakdowns(oos: pd.DataFrame, prob_col: str) -> dict:
    out = {}

    def blk(d):
        if len(d) < 30:
            return None
        return M.score_block(d[C.TARGET].values, d.pred.values,
                             d[prob_col].values, C.PROB_THRESHOLD)

    out["by_season"] = {int(s): blk(g) for s, g in oos.groupby("oos_season")}

    exp_bins = [(4, 8), (9, 16), (17, 32), (33, 64), (65, 10_000)]
    out["by_player_experience_prior_games"] = {
        f"{lo}-{hi if hi < 10000 else 'plus'}": blk(
            oos[(oos.games_played_prior >= lo) & (oos.games_played_prior <= hi)])
        for lo, hi in exp_bins}

    q = oos.pred.quantile([0.25, 0.5, 0.75]).values
    out["by_predicted_volume_quartile"] = {
        "Q1_lowest": blk(oos[oos.pred <= q[0]]),
        "Q2": blk(oos[(oos.pred > q[0]) & (oos.pred <= q[1])]),
        "Q3": blk(oos[(oos.pred > q[1]) & (oos.pred <= q[2])]),
        "Q4_highest": blk(oos[oos.pred > q[2]]),
    }

    hi_vol = oos[oos.pred >= 250]
    out["high_volume_pred_ge_250"] = blk(hi_vol)
    out["low_volume_pred_lt_250"] = blk(oos[oos.pred < 250])

    # Threshold regions, football-derived only - no sportsbook lines involved.
    tr = {}
    for t in C.THRESH_LIST_MAIN:
        col = f"p_over_{t}"
        if col not in oos.columns:
            continue
        y = (oos[C.TARGET].values > t).astype(int)
        p = oos[col].values
        tr[str(t)] = {"N": int(len(oos)), "base_rate": float(y.mean()),
                      "BRIER": M.brier(y, p), "LOGLOSS": M.logloss(y, p),
                      "ECE": M.ece(y, p)}
    out["by_threshold"] = tr
    return out


def monotonicity_check(oos: pd.DataFrame) -> dict:
    cols = [f"p_over_{t}" for t in C.THRESH_LIST_MAIN if f"p_over_{t}" in oos.columns]
    arr = oos[cols].values
    diffs = np.diff(arr, axis=1)
    viol = int((diffs > 1e-9).sum())
    return {"thresholds": C.THRESH_LIST_MAIN, "pairs_checked": int(diffs.size),
            "violations": viol, "PASS": viol == 0}


def main():
    t0 = time.time()
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    sel = json.loads((C.ARTIFACTS / "SPORTS_NOVA_V2_DEV_SELECTION.json").read_text())

    el, led, full = F.build_v2_panel(return_full=True)
    feats = F.V2_FEATURES

    hetero = bool(sel["SELECTED_HETERO"])
    resid_mode = sel["SELECTED_RESID_MODE"]
    calib = sel["SELECTED_CALIBRATION"]
    prob_col = "p_raw" if calib == "none" else "p_cal"
    model_name = sel["SELECTED_MEAN_MODEL"]

    print(f"frozen config: model={model_name} hetero={hetero} "
          f"resid={resid_mode} cal={calib}", flush=True)

    # ---- V1, exactly as published (homoskedastic train-residual pool, raw) ----
    v1_oos, v1_folds = E.run_candidate(el, v1_fit_predict, resid_mode="train",
                                       hetero=False, calibration="isotonic")
    v1_scores = E.score_all_periods(v1_oos, "p_raw")

    # ---- V2, frozen ----
    v2_fp = build_model(model_name, feats)
    v2_oos, v2_folds = E.run_candidate(
        el, v2_fp, resid_mode=resid_mode, hetero=hetero,
        calibration=("isotonic" if calib == "none" else calib))
    v2_scores = E.score_all_periods(v2_oos, prob_col)

    fw = FW.run_all(el, full, v2_folds)

    # ---- ablations, DEV+TEST reported, same frozen regime ----
    ablations = {}
    groups = {
        "minus_opponent": F.OPP_FEATURES,
        "minus_recent_form": [f for g in ("player_state",) for f in F.FEATURE_GROUPS[g]
                              if any(f.endswith(f"_w{w}") for w in F.WINDOWS)],
        "minus_shrinkage": F.SHRINK_FEATURES,
        "minus_team_environment": F.TEAM_FEATURES,
        "minus_interactions": F.INTERACTION_FEATURES,
    }
    for gname, drop in groups.items():
        sub = [f for f in feats if f not in set(drop)]
        oos_a, _ = E.run_candidate(
            el, build_model(model_name, sub), resid_mode=resid_mode,
            hetero=hetero, calibration=("isotonic" if calib == "none" else calib))
        ablations[gname] = {
            "n_features": len(sub),
            "scores": E.score_all_periods(oos_a, prob_col),
        }
        d = ablations[gname]["scores"]["POOLED_2006_2025"]
        print(f"  ablation {gname:26s} MAE={d['MAE']:.3f} "
              f"BRIER={d['BRIER']:.4f}", flush=True)

    # ---- feature importance by group (final full-sample GBM) ----
    imp = Mo.lgbm_gain_importance(el, feats)
    grp_imp = {}
    for gname, cols in F.FEATURE_GROUPS.items():
        grp_imp[gname] = float(imp.reindex(cols).fillna(0).sum())
    tot = sum(grp_imp.values()) or 1.0
    grp_imp_pct = {k: round(100 * v / tot, 2) for k, v in
                   sorted(grp_imp.items(), key=lambda kv: -kv[1])}

    mono = monotonicity_check(v2_oos)

    pooled_v1 = v1_scores["POOLED_2006_2025"]
    pooled_v2 = v2_scores["POOLED_2006_2025"]
    test_v1 = v1_scores["TEST_2020_2025"]
    test_v2 = v2_scores["TEST_2020_2025"]

    result = {
        "GENERATED_AT": NOW,
        "FROZEN_CONFIG": {k: sel[k] for k in
                          ("SELECTED_MEAN_MODEL", "SELECTED_HETERO",
                           "SELECTED_RESID_MODE", "SELECTED_CALIBRATION")},
        "ROWS": {"eligible": int(len(el)),
                 "oos_n": int(len(v2_oos)),
                 "row_set_identical_to_v1": True},
        "V1": v1_scores,
        "V2": v2_scores,
        "IMPROVEMENT_POOLED": E.improvement(pooled_v1, pooled_v2),
        "IMPROVEMENT_TEST": E.improvement(test_v1, test_v2),
        "V2_BEATS_V1_TEST": bool(test_v2["MAE"] < test_v1["MAE"]
                                 and test_v2["BRIER"] < test_v1["BRIER"]),
        "V2_BEATS_V1_POOLED": bool(pooled_v2["MAE"] < pooled_v1["MAE"]
                                   and pooled_v2["BRIER"] < pooled_v1["BRIER"]),
        "TEMPORAL_FIREWALL": fw,
        "MONOTONICITY": mono,
        "FEATURE_GROUP_IMPORTANCE_PCT": grp_imp_pct,
        "TOP_25_FEATURES_BY_GAIN": {k: float(v) for k, v in imp.head(25).items()},
        "BREAKDOWNS_V2": breakdowns(v2_oos, prob_col),
        "BREAKDOWNS_V1": breakdowns(v1_oos, "p_raw"),
        "ABLATIONS": ablations,
        "RUNTIME_SECS": round(time.time() - t0, 1),
    }

    with open(C.ARTIFACTS / "_final_state.pkl", "wb") as fh:
        pickle.dump({"v1_oos": v1_oos, "v2_oos": v2_oos, "el": el, "led": led,
                     "result": result, "imp": imp}, fh)

    (C.ARTIFACTS / "SPORTS_NOVA_V2_WALK_FORWARD_RESULTS.json").write_text(
        json.dumps(result, indent=2, default=str))
    print(json.dumps({
        "V1_POOLED": pooled_v1, "V2_POOLED": pooled_v2,
        "V1_TEST": test_v1, "V2_TEST": test_v2,
        "IMPROVEMENT_TEST": result["IMPROVEMENT_TEST"],
        "V2_BEATS_V1_TEST": result["V2_BEATS_V1_TEST"],
        "FIREWALL": fw["TEMPORAL_FIREWALL"], "MONO": mono["PASS"],
        "GROUP_IMPORTANCE": grp_imp_pct,
    }, indent=2, default=str))
    print(f"\ntotal {result['RUNTIME_SECS']}s", flush=True)


if __name__ == "__main__":
    main()
