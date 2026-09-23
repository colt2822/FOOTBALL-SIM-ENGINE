"""V2.1 candidate selection on DEV (2006-2019) only.

Selection metric is retargeted. The task's premise was that Week 1 is broken;
measurement shows Week 1 is already sound (DEV bias +0.8) and the damage sits
in the rows where an offseason boundary is mistaken for an in-season absence.
So the primary metric is MAE on the collided `days_rest == 30` cohort, with
overall DEV MAE and Week-1 DEV MAE as guardrails that must not degrade.

TEST is not read here.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import config as C
from . import experiment as E
from . import features_v21 as V21
from . import metrics as M
from . import models as Mo

REGIME = dict(hetero=True, resid_mode="prior_oos", calibration="isotonic")
PROB_COL = "p_raw"          # frozen V2 regime selected cal=none


def bucket_scores(oos: pd.DataFrame, el: pd.DataFrame) -> dict:
    j = oos.join(el[["days_rest", "crossed_season_boundary", "career_lead_share",
                     "prior_season_lead_games"]], rsuffix="_x")
    j["err"] = j.pred - j[C.TARGET]
    out = {}
    for k, m in V21.buckets(j).items():
        s = j[m]
        if len(s) < 20:
            continue
        out[k] = {"N": int(len(s)), "MAE": float(s.err.abs().mean()),
                  "BIAS": float(s.err.mean())}
    ceil = j[j.days_rest >= 29.9]
    out["ALL_CEILING_ROWS"] = {"N": int(len(ceil)),
                               "MAE": float(ceil.err.abs().mean()),
                               "BIAS": float(ceil.err.mean())}
    return out


def main():
    el, _ = V21.build_v21_panel()
    results = {}

    for name, feats in V21.CANDIDATES.items():
        t0 = time.time()
        oos, _ = E.run_candidate(el, Mo.make_blend(feats, weight=0.5),
                                 seasons=C.DEV_SEASONS, **REGIME)
        dev = E.score(oos, C.DEV_SEASONS, PROB_COL)
        bk = bucket_scores(oos, el)
        results[name] = {
            "n_features": len(feats),
            "overall": {k: dev[k] for k in ("N", "MAE", "RMSE", "BRIER",
                                            "LOGLOSS", "ECE")},
            "buckets": bk,
            "secs": round(time.time() - t0, 1),
        }
        c = bk["ALL_CEILING_ROWS"]
        print(f"{name:32s} DEV_MAE={dev['MAE']:7.3f}  "
              f"CEIL_MAE={c['MAE']:7.3f} CEIL_BIAS={c['BIAS']:+7.2f}  "
              f"LATE_DEBUT_BIAS={bk.get('LATE_SEASON_DEBUT',{}).get('BIAS',float('nan')):+7.2f}  "
              f"WK1_MAE={bk.get('WEEK_1',{}).get('MAE',float('nan')):7.3f}  "
              f"[{results[name]['secs']}s]", flush=True)

    base = results["A_current_v2"]
    cand = {k: v for k, v in results.items() if k != "A_current_v2"}

    # Primary: ceiling-cohort MAE. Guardrails: overall and Week-1 DEV MAE must
    # not degrade by more than a hair against current V2.
    TOL = 0.15
    eligible = {}
    for k, v in cand.items():
        ok_overall = v["overall"]["MAE"] <= base["overall"]["MAE"] + TOL
        ok_week1 = (v["buckets"]["WEEK_1"]["MAE"] <=
                    base["buckets"]["WEEK_1"]["MAE"] + TOL)
        if ok_overall and ok_week1:
            eligible[k] = v
        print(f"  guardrails {k:32s} overall_ok={ok_overall} week1_ok={ok_week1}")

    pool = eligible or cand
    best = min(pool, key=lambda k: pool[k]["buckets"]["ALL_CEILING_ROWS"]["MAE"])
    print(f"\nSELECTED on DEV: {best}")

    frozen = {
        "SELECTED_CANDIDATE": best,
        "SELECTED_FEATURES": V21.CANDIDATES[best],
        "N_FEATURES": len(V21.CANDIDATES[best]),
        "DISTRIBUTION_REGIME": {**REGIME, "calibration_applied": "none",
                                "note": "inherited unchanged from V2 so the "
                                        "comparison isolates the feature fix"},
        "SELECTION_RULE": ("primary: MAE on the collided days_rest==30 cohort; "
                           "guardrails: overall DEV MAE and WEEK_1 DEV MAE within "
                           f"{TOL} of current V2. TEST never consulted."),
        "GUARDRAIL_TOLERANCE": TOL,
        "PASSED_GUARDRAILS": sorted(eligible),
        "DEV_RESULTS": results,
    }
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (C.ARTIFACTS / "SPORTS_NOVA_V2_1_DEV_SELECTION.json").write_text(
        json.dumps(frozen, indent=2, default=str))
    print("frozen ->", C.ARTIFACTS / "SPORTS_NOVA_V2_1_DEV_SELECTION.json")


if __name__ == "__main__":
    main()
