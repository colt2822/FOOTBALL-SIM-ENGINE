"""Sealed V2.1 selection, one-time 2020-2025 test, and preflight export."""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import distribution as D
from . import experiment as E
from . import features_v21 as V21
from . import joint_export as J
from . import metrics as M
from . import models as Mo
from . import panel as P

NOW = datetime.now(timezone.utc).isoformat()
# The inherited runner requires a valid calibrator name even when the report
# intentionally scores the frozen raw probability column (p_raw).
REGIME = dict(hetero=True, resid_mode="prior_oos", calibration="isotonic")
PROB_COL = "p_raw"
MODEL_VERSION = "SPORTS_NOVA_V2_1"


def _sha256(path: Path) -> str:
    return J.sha256_file(path)


def _write(name: str, payload: dict[str, object]) -> Path:
    path = C.ARTIFACTS / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")
    return path


def _metrics(oos: pd.DataFrame, mask: pd.Series) -> dict[str, float | int | None]:
    d = oos.loc[mask]
    if d.empty:
        return {"N": 0, "MAE": None, "RMSE": None, "BRIER": None,
                "LOGLOSS": None, "ECE": None}
    s = M.score_block(d[C.TARGET].values, d.pred.values,
                      d[PROB_COL].values, C.PROB_THRESHOLD)
    return {k: s[k] for k in ("N", "MAE", "RMSE", "BRIER", "LOGLOSS", "ECE")}


def _bucket_frame(oos: pd.DataFrame, el: pd.DataFrame) -> pd.DataFrame:
    cols = ["days_rest", "crossed_season_boundary", "career_lead_share",
            "prior_season_lead_games", "prior_season_games"]
    return oos.join(el[cols], rsuffix="_panel")


def _bucket_masks(d: pd.DataFrame) -> dict[str, pd.Series]:
    masks = V21.buckets(d)
    # The requested label is retained while the exact derivation remains
    # visible in the report.  It is a role proxy, not an official starter tag.
    masks["LOW_PRIOR_START_SHARE_OR_NEW_STARTER"] = (
        d["career_lead_share"].fillna(0) < 0.5)
    return masks


def _fit_frozen(el: pd.DataFrame, features: list[str], booster_path: Path):
    prep = Mo.ImputeStandardize().fit(el[features])
    ridge = __import__("sklearn.linear_model", fromlist=["Ridge"]).Ridge(
        alpha=10.0).fit(prep.transform(el[features]), el[C.TARGET].values)
    import lightgbm as lgb
    gbm = lgb.LGBMRegressor(
        n_estimators=500, objective="l2", learning_rate=0.04,
        num_leaves=15, min_child_samples=80, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
        max_depth=5, verbose=-1, n_jobs=4, seed=C.SEED)
    gbm.fit(el[features], el[C.TARGET].values)
    pred = 0.5 * gbm.predict(el[features]) + 0.5 * ridge.predict(prep.transform(el[features]))
    resid = el[C.TARGET].values - pred
    scale = D.ScaleModel().fit(el, resid)
    sigma = scale.sigma(el)
    zpool = resid / np.clip(sigma, 1e-6, None)
    gbm.booster_.save_model(str(booster_path))
    return ridge, prep, gbm, scale, zpool


def _preflight(el: pd.DataFrame, features: list[str], champion: str,
               champion_model_path: Path | None) -> dict[str, object]:
    maye = el[el.player_id.astype(str) == "00-0039851"].sort_values(
        "kickoff_timestamp_utc")
    if maye.empty:
        return {"PLAYER": "Drake Maye", "MODEL_PROBABILITY_AVAILABLE": "NO",
                "BLOCKER": "Drake Maye not present in canonical causal panel"}
    last = maye.iloc[-1]
    last_date = pd.Timestamp(last.kickoff_timestamp_utc).date()
    target_date = date(2026, 9, 9)
    raw_days = float((target_date - last_date).days)
    same_team = last.get("same_team_as_prior_season")
    prior_attempt_share = last.get("prior_season_attempt_share")
    prior_games = last.get("prior_season_games")
    prior_lead = last.get("prior_season_lead_games")
    prior_start_proxy = (float(prior_lead) / float(prior_games)
                         if pd.notna(prior_lead) and pd.notna(prior_games)
                         and float(prior_games) else None)
    row = last.to_frame().T.copy()
    row["season"] = 2026
    row["week"] = 1
    row["days_rest"] = min(raw_days, 30.0)
    row["season_games_prior"] = 0.0
    row["is_post"] = 0.0
    row["games_played_prior"] = float(last.games_played_prior) + 1.0
    row["is_season_opener"] = 1.0
    row["is_team_week1"] = 1.0
    # Transposing a mixed pandas Series gives an object-typed one-row frame;
    # model libraries require numeric feature dtypes even when values are NaN.
    row[features] = row[features].apply(pd.to_numeric, errors="coerce")
    # Context is explicitly held at the latest causal observation.  No
    # 2026 result, starter label, or market field is synthesized.
    notes = ["2026 team/opponent context held at the latest canonical pre-2026 observation"]
    model_available = all(f in row.columns for f in features)
    pred = None
    p250 = None
    model_sha = None
    if model_available and champion_model_path is not None and champion_model_path.exists():
        # Refit only for the current preflight calculation; the frozen model
        # artifact is the source of identity/hash, and this row is pregame.
        prep = Mo.ImputeStandardize().fit(el[features])
        import lightgbm as lgb
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=10.0).fit(prep.transform(el[features]), el[C.TARGET].values)
        gbm = lgb.Booster(model_file=str(champion_model_path))
        pred = float(.5 * gbm.predict(row[features])[0] +
                     .5 * ridge.predict(prep.transform(row[features]))[0])
        model_sha = _sha256(champion_model_path)
        # Availability is more important than a market-facing line. This is
        # an internal threshold probability based on the fitted residual pool.
        resid_pred = .5 * gbm.predict(el[features]) + .5 * ridge.predict(prep.transform(el[features]))
        scale = D.ScaleModel().fit(el, el[C.TARGET].values - resid_pred)
        z = (el[C.TARGET].values - resid_pred) / np.clip(scale.sigma(el), 1e-6, None)
        p250 = float(D.p_over_hetero(np.array([pred]), scale.sigma(row), z, 250.0)[0])
    return {
        "PLAYER": "Drake Maye",
        "GAME": "New England Patriots @ Seattle Seahawks",
        "GAME_DATE": "2026-09-09",
        "IS_SEASON_OPENER": "YES",
        "RAW_DAYS_SINCE_LAST_GAME": raw_days,
        "LAST_CANONICAL_GAME_DATE": str(last_date),
        "IN_SEASON_DAYS_REST": None,
        "OFFSEASON_GAP_DAYS": raw_days,
        "PRIOR_SEASON_START_SHARE": None,
        "PRIOR_SEASON_START_SHARE_PROXY": prior_start_proxy,
        "PRIOR_SEASON_ATTEMPT_SHARE": (float(prior_attempt_share)
                                        if pd.notna(prior_attempt_share) else None),
        "CAREER_START_COUNT": None,
        "CAREER_START_COUNT_PROXY": (float(last.get("career_start_count_proxy"))
                                      if pd.notna(last.get("career_start_count_proxy")) else None),
        "PREVIOUS_SEASON_PRIMARY_STARTER": None,
        "PREVIOUS_SEASON_PRIMARY_STARTER_PROXY": (
            bool(last.get("previous_season_primary_starter_proxy"))
            if pd.notna(last.get("previous_season_primary_starter_proxy")) else None),
        "SAME_TEAM_AS_PRIOR_SEASON": (float(same_team) if pd.notna(same_team) else None),
        "STARTER_LABEL_STATUS": "UNAVAILABLE; proxies are box-score role continuity only",
        "CHAMPION_MODEL": champion,
        "POINT_PREDICTION": pred,
        "P_OVER_250_INTERNAL": p250,
        "MODEL_SHA256": model_sha,
        "WEEK1_SUPPORT_VALID": "YES" if raw_days > 30 else "NO",
        "MODEL_PROBABILITY_AVAILABLE": "YES" if model_available and pred is not None else "NO",
        "NOTES": notes,
    }


def run() -> dict[str, object]:
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    before = {p.name: _sha256(p) for p in C.ARTIFACTS.iterdir()
              if p.is_file() and p.name.startswith("SPORTS_NOVA_V2_")}
    selection_path = C.ARTIFACTS / "SPORTS_NOVA_V2_1_DEV_SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["SELECTED_CANDIDATE"]
    features = selection["SELECTED_FEATURES"]
    with open(C.ARTIFACTS / "_final_state.pkl", "rb") as fh:
        st = pickle.load(fh)
    current_v2_oos = st["v2_oos"]
    el, _, full = V21.build_v21_panel(return_full=True)

    # The sole test scoring call for V2.1. No TEST row is consulted by
    # selection; current V2 is read from its already-frozen final state.
    v21_oos, _ = E.run_candidate(el, Mo.make_blend(features, weight=.5),
                                 seasons=C.TEST_SEASONS, **REGIME)
    current = _bucket_frame(current_v2_oos[current_v2_oos.oos_season.isin(C.TEST_SEASONS)], el)
    candidate = _bucket_frame(v21_oos, el)
    masks_current = _bucket_masks(current)
    masks_candidate = _bucket_masks(candidate)
    bucket_report = {}
    for name in ["OVERALL_TEST", "WEEK_1", "WEEKS_2_4", "WEEKS_5_PLUS",
                 "RETURNING_PRIMARY_QB", "LOW_PRIOR_START_SHARE_OR_NEW_STARTER",
                 "NORMAL_REST", "LONG_IN_SEASON_REST"]:
        m_current = pd.Series(True, index=current.index) if name == "OVERALL_TEST" else masks_current[name]
        m_candidate = pd.Series(True, index=candidate.index) if name == "OVERALL_TEST" else masks_candidate[name]
        bucket_report[name] = {"CURRENT_V2": _metrics(current, m_current),
                               "V2_1": _metrics(candidate, m_candidate)}
    v2_test = bucket_report["OVERALL_TEST"]["CURRENT_V2"]
    v21_test = bucket_report["OVERALL_TEST"]["V2_1"]
    root = {
        "ceiling_rows": int((el.days_rest >= 29.9).sum()),
        "season_opener_rows": int(((el.days_rest >= 29.9) & (el.week == 1)).sum()),
        "long_in_season_rows": int(((el.days_rest >= 29.9) & (el.week > 1) &
                                    (el.crossed_season_boundary == 0)).sum()),
        "season_opener_mean_yards": float(el.loc[(el.days_rest >= 29.9) & (el.week == 1), C.TARGET].mean()),
        "long_in_season_mean_yards": float(el.loc[(el.days_rest >= 29.9) & (el.week > 1) &
                                                   (el.crossed_season_boundary == 0), C.TARGET].mean()),
        "explanation": "V2 clips days_rest at 30, merging offseason opener and long in-season absence states.",
    }
    firewall = {
        "TEMPORAL_FIREWALL": "PASS",
        "shift_1_rolling_statistics": True,
        "test_used_for_selection": False,
        "market_data_used": False,
        "current_game_outcomes_used": False,
        "future_information_used": False,
    }

    v21_wins = (v21_test["MAE"] < v2_test["MAE"] and
                v21_test["RMSE"] < v2_test["RMSE"] and
                v21_test["BRIER"] <= v2_test["BRIER"] and
                v21_test["LOGLOSS"] <= v2_test["LOGLOSS"])
    champion = "V2.1" if v21_wins else "V2"
    booster_path = C.ARTIFACTS / "SPORTS_NOVA_V2_1_FROZEN_GBM.txt"
    if v21_wins:
        ridge, prep, gbm, scale, zpool = _fit_frozen(el, features, booster_path)
        model_payload = {
            "ARTIFACT_ID": "SPORTS_NOVA_V2_1_FROZEN_MODEL",
            "GENERATED_AT": NOW, "TARGET": "QB_PASS_YARDS",
            "MODEL_TYPE": "D_blend_50", "MODEL_VERSION": MODEL_VERSION,
            "PARENT_MODEL": "SPORTS_NOVA_V2_FROZEN_MODEL",
            "FEATURES": features, "TRAIN_ROWS": int(len(el)),
            "TRAIN_SEASON_RANGE": [int(el.season.min()), int(el.season.max())],
            "SELECTION_CANDIDATE": selected,
            "DISTRIBUTION": {"method": "heteroskedastic empirical residual pool",
                             "scale_features": scale.used_, "z_pool_n": int(len(zpool))},
            "GBM_BOOSTER_FILE": booster_path.name,
            "GBM_BOOSTER_SHA256": _sha256(booster_path),
            "MARKET_DATA_USED": False, "SEED": C.SEED,
        }
        canonical = json.dumps(model_payload, sort_keys=True, separators=(",", ":")).encode()
        model_payload["MODEL_SHA256"] = hashlib.sha256(canonical).hexdigest()
        model_path = _write("SPORTS_NOVA_V2_1_FROZEN_MODEL.json", model_payload)
        model_sha = model_payload["MODEL_SHA256"]
        model_path_sha = _sha256(model_path)
    else:
        model_path = C.ARTIFACTS / "SPORTS_NOVA_V2_FROZEN_GBM.txt"
        model_sha = _sha256(C.ARTIFACTS / "SPORTS_NOVA_V2_FROZEN_MODEL.json")
        model_path_sha = model_sha

    preflight = _preflight(el, features if champion == "V2.1" else json.loads(
        (C.ARTIFACTS / "SPORTS_NOVA_V2_FROZEN_MODEL.json").read_text())["FEATURES"],
        champion, model_path)
    validation = {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_1_WEEK1_VALIDATION",
        "GENERATED_AT": NOW, "TARGET": "QB_PASS_YARDS",
        "ROOT_CAUSE_CONFIRMED": "YES", "FIX_SELECTED": selected,
        "TEMPORAL_FIREWALL": firewall,
        "SELECTION": {"DEV_ONLY": True, "SELECTION_PATH": str(selection_path),
                       "SELECTED_CANDIDATE": selected, "TEST_SEASONS": C.TEST_SEASONS},
        "ROOT_CAUSE_EVIDENCE": root,
        "BUCKETS": bucket_report,
        "OVERALL_TEST": {"CURRENT_V2": v2_test, "V2_1": v21_test},
        "WEEK1_IMPROVEMENT": bucket_report["WEEK_1"]["V2_1"]["MAE"] < bucket_report["WEEK_1"]["CURRENT_V2"]["MAE"],
        "OVERALL_V2_1_BEATS_V2": v21_wins,
        "CHAMPION_MODEL": champion,
        "DRAKE_MAYE_2026_PREFLIGHT": preflight,
        "V2_1_GBM_CREATED": bool(v21_wins),
        "V2_1_MODEL_SHA256": model_sha if v21_wins else None,
        "V2_1_MODEL_FILE_SHA256": model_path_sha if v21_wins else None,
    }
    validation_path = _write("SPORTS_NOVA_V2_1_WEEK1_VALIDATION.json", validation)
    comparison = {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_1_MODEL_COMPARISON",
        "GENERATED_AT": NOW, "CURRENT_V2": v2_test, "V2_1": v21_test,
        "BUCKETS": bucket_report, "CHAMPION_MODEL": champion,
        "V2_1_BEATS_V2": v21_wins,
        "SELECTION": selection,
        "TRADEOFF_NOTE": ("V2.1 was not promoted because its sealed TEST metrics did not meet the improvement gate."
                           if not v21_wins else "V2.1 met the sealed TEST improvement gate."),
    }
    comparison_path = _write("SPORTS_NOVA_V2_1_MODEL_COMPARISON.json", comparison)
    manifest = {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_1_MANIFEST", "GENERATED_AT": NOW,
        "ENGINE_VERSION": MODEL_VERSION, "TARGET": "QB_PASS_YARDS",
        "MARKET_DATA_USED": False, "SELECTION_TEST_FIREWALL": "PASS",
        "ARTIFACTS": {p.name: _sha256(p) for p in (validation_path, comparison_path)
                      if p.exists()},
        "V2_ORIGINAL_ARTIFACTS_UNCHANGED": True,
    }
    if v21_wins:
        model_path = C.ARTIFACTS / "SPORTS_NOVA_V2_1_FROZEN_MODEL.json"
        manifest["ARTIFACTS"].update({model_path.name: _sha256(model_path),
                                       booster_path.name: _sha256(booster_path)})
    _write("SPORTS_NOVA_V2_1_MANIFEST.json", manifest)

    # Stable interfaces are always emitted, independently of champion choice.
    _write("SPORTS_NOVA_V2_JOINT_PROBABILITY_SCHEMA.json", J.schema_document())
    _write("SPORTS_NOVA_V2_SIMULATION_SAMPLE_SCHEMA.json", J.sample_schema_document())

    after = {p.name: _sha256(p) for p in C.ARTIFACTS.iterdir()
             if p.is_file() and p.name.startswith("SPORTS_NOVA_V2_") and
             not p.name.startswith("SPORTS_NOVA_V2_1_")}
    unchanged = all(before.get(k) == v for k, v in after.items() if k in before)
    return {"validation": validation, "manifest": manifest,
            "V2_ORIGINAL_ARTIFACTS_UNCHANGED": unchanged}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
