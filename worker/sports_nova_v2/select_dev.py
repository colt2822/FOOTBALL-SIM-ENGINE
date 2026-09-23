"""DEV-only selection (2006-2019).

Everything discretionary is decided here and written to a frozen config:
model family, LightGBM hyperparameters, blend weight, residual regime and
calibration method. TEST is not read in this script.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import config as C
from . import experiment as E
from . import features as F
from . import models as Mo
from .v1_baseline import v1_fit_predict


def _fmt(name, dev):
    return (f"{name:34s} MAE={dev['MAE']:7.3f} RMSE={dev['RMSE']:7.3f} "
            f"BRIER={dev['BRIER']:.4f} LL={dev['LOGLOSS']:.4f} "
            f"ECE={dev['ECE']:.4f} CRPS={dev['CRPS']:7.3f}")


def main():
    el, _ = F.build_v2_panel()
    feats = F.V2_FEATURES
    no_shrink = [f for f in feats if f not in F.SHRINK_FEATURES]
    results = {}

    candidates = {
        "A_v1_ridge_4feat": (v1_fit_predict, dict(hetero=False, resid_mode="train",
                                                  calibration="none")),
        "B_ridge_v2_full": (Mo.make_ridge(feats, alpha=10.0), {}),
        "B2_ridge_v2_a50": (Mo.make_ridge(feats, alpha=50.0), {}),
        "C_lgbm_noshrink": (Mo.make_lgbm(no_shrink), {}),
        "C2_lgbm_full": (Mo.make_lgbm(feats), {}),
        "C3_lgbm_full_deep": (Mo.make_lgbm(feats, dict(num_leaves=31, max_depth=7,
                                                       learning_rate=0.03),
                                           n_estimators=700), {}),
        "C4_lgbm_full_shallow": (Mo.make_lgbm(feats, dict(num_leaves=7, max_depth=4,
                                                          learning_rate=0.05),
                                              n_estimators=600), {}),
        "D_blend_50": (Mo.make_blend(feats, weight=0.5), {}),
        "D2_blend_65": (Mo.make_blend(feats, weight=0.65), {}),
    }

    for name, (fp, kw) in candidates.items():
        t0 = time.time()
        opts = dict(hetero=True, resid_mode="train", calibration="isotonic")
        opts.update(kw)
        cal = opts.pop("calibration")
        oos, _ = E.run_candidate(el, fp, seasons=C.DEV_SEASONS,
                                 calibration=("isotonic" if cal == "none" else cal),
                                 **opts)
        prob_col = "p_raw" if cal == "none" else "p_cal"
        dev = E.score(oos, C.DEV_SEASONS, prob_col)
        results[name] = {"dev": dev, "opts": {**opts, "calibration": cal},
                         "secs": round(time.time() - t0, 1)}
        print(_fmt(name, dev), f" [{results[name]['secs']}s]", flush=True)

    # Distribution / calibration regime, evaluated on the winning mean model.
    best_mean = min((k for k in results if k != "A_v1_ridge_4feat"),
                    key=lambda k: results[k]["dev"]["MAE"])
    print(f"\nbest mean model on DEV MAE: {best_mean}\n")
    fp = candidates[best_mean][0]

    regimes = {}
    for hetero in (False, True):
        for resid_mode in ("train", "prior_oos"):
            for calib in ("none", "platt", "isotonic"):
                key = f"hetero={int(hetero)}|resid={resid_mode}|cal={calib}"
                oos, _ = E.run_candidate(
                    el, fp, seasons=C.DEV_SEASONS, hetero=hetero,
                    resid_mode=resid_mode,
                    calibration=("isotonic" if calib == "none" else calib))
                col = "p_raw" if calib == "none" else "p_cal"
                dev = E.score(oos, C.DEV_SEASONS, col)
                regimes[key] = dev
                print(_fmt(key, dev), flush=True)

    best_regime = min(regimes, key=lambda k: regimes[k]["LOGLOSS"])
    print(f"\nbest distribution regime on DEV LOGLOSS: {best_regime}")

    h, r, c = best_regime.split("|")
    frozen = {
        "SELECTED_MEAN_MODEL": best_mean,
        "SELECTED_HETERO": bool(int(h.split("=")[1])),
        "SELECTED_RESID_MODE": r.split("=")[1],
        "SELECTED_CALIBRATION": c.split("=")[1],
        "DEV_SEASONS": C.DEV_SEASONS,
        "TEST_SEASONS": C.TEST_SEASONS,
        "candidate_dev_results": {k: v["dev"] for k, v in results.items()},
        "regime_dev_results": regimes,
        "selection_rule": ("mean model by DEV MAE; distribution regime by DEV "
                           "LOGLOSS; TEST never consulted"),
    }
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (C.ARTIFACTS / "SPORTS_NOVA_V2_DEV_SELECTION.json").write_text(
        json.dumps(frozen, indent=2, default=str))
    print("\nfrozen ->", C.ARTIFACTS / "SPORTS_NOVA_V2_DEV_SELECTION.json")


if __name__ == "__main__":
    main()
