"""Persist the V2 output artifacts, including the frozen model and its hash."""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import config as C
from . import distribution as D
from . import features as F
from . import metrics as M
from . import models as Mo
from . import montecarlo as MC

NOW = datetime.now(timezone.utc).isoformat()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def write(name: str, payload: dict) -> str:
    path = C.ARTIFACTS / name
    path.write_text(json.dumps(payload, indent=2, default=str))
    return sha256_file(path)


# ---------------------------------------------------------------------------
def feature_schema(el: pd.DataFrame) -> dict:
    cov = el[F.V2_FEATURES].notna().mean()
    entries = []
    for group, cols in F.FEATURE_GROUPS.items():
        for c in cols:
            entries.append({
                "name": c,
                "group": group,
                "coverage_on_eligible_rows": round(float(cov.get(c, np.nan)), 4),
                "causality": "shift(1) on entity's own kickoff-ordered sequence",
                "entity": ("player" if group in ("player_state", "shrinkage")
                           else "own_team" if group == "team_environment"
                           else "opponent_team" if group == "opponent"
                           else "derived"),
            })
    return {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_FEATURE_SCHEMA",
        "GENERATED_AT": NOW,
        "TARGET": "QB_PASS_YARDS",
        "N_FEATURES": len(F.V2_FEATURES),
        "FEATURE_GROUPS": {k: len(v) for k, v in F.FEATURE_GROUPS.items()},
        "WINDOWS": F.WINDOWS,
        "FEATURES": entries,
        "EXCLUDED_SEASON_VARYING": {
            "columns": F.EXCLUDED_SEASON_VARYING,
            "reason": ("present in only part of the walk-forward span; a feature "
                       "that exists in train and vanishes in test degrades "
                       "silently and does not look like leakage"),
        },
        "SENTINEL_MASKING": {
            "columns": F.AIR_YARDS_COLS,
            "rule": f"masked to NaN before season {F.AIR_YARDS_ERA}",
            "reason": ("source panel zero-fills air-yards fields pre-2006; a "
                       "0 that flips to a real value mid-panel is worse than NaN"),
        },
        "SCHEMA_HARMONISATION": {
            "sacks_taken": "coalesce(sacks [1999-2024], sacks_suffered [2025])",
            "ints_thrown": "coalesce(interceptions [1999-2024], passing_interceptions [2025])",
        },
        "EXTERNAL_MODEL_CAVEAT": {
            "affected": ["passing_epa", "off_epa_per_play", "def_epa_per_play",
                         "off_pass_epa_per_db", "def_pass_epa_per_db",
                         "epa_att_w3", "epa_att_w5", "epa_att_w8", "career_epa_att"],
            "issue": ("nflverse EPA comes from an expected-points model fitted on "
                      "the full historical panel, so its coefficients encode "
                      "league-wide relationships from seasons after a given row. "
                      "No row's own outcome enters its own features, and EPA is "
                      "published within days of a game, so availability is not in "
                      "question. Flagged rather than hidden; the minus_opponent "
                      "and minus_team_environment ablations bound its contribution."),
            "status": "DISCLOSED_NOT_QUARANTINED",
        },
        "QUARANTINED": {
            "xpass_pass_oe": ("nflverse pass-rate-over-expected; excluded because "
                              "its xpass model is fitted on the full panel and it "
                              "is the single feature most likely to smuggle in "
                              "future league tendency"),
            "vegas_wp_vegas_wpa": "market-derived; never read from play-by-play",
        },
    }


def data_gaps() -> dict:
    def gap(gid, name, why, avail, priority, node2):
        return {"GAP_ID": gid, "DATASET": name, "WHY_IT_MATTERS": why,
                "CURRENT_AVAILABILITY": avail, "PRIORITY": priority,
                "WHAT_NODE2_MUST_SOURCE": node2}

    return {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_DATA_GAPS",
        "GENERATED_AT": NOW,
        "GAPS": [
            gap("GAP-001", "Historical injury / inactive reports with publication timestamps",
                ("The dominant driver of pass-yards error is expected attempts, and "
                 "the dominant driver of attempt uncertainty is availability - who "
                 "starts, who is inactive, who leaves early. V2 has no causal "
                 "injury signal at all; Layer 5 is interface-only."),
                "NOT AVAILABLE - no timestamped historical source in this environment",
                "CRITICAL",
                ("Weekly NFL injury report snapshots (Wed/Thu/Fri practice status + "
                 "game-day inactives) for 1999-2025 with the UTC timestamp each "
                 "report was published, so it can be joined causally.")),
            gap("GAP-002", "Snap counts (nflverse, PFR-derived)",
                ("Direct measure of participation and role. Would sharpen expected "
                 "volume and identify mid-game QB changes that box scores blur."),
                "FREE AND REACHABLE, but 2012+ only - season-varying across the "
                "2006-2025 walk-forward span, so excluded from the primary model",
                "HIGH",
                ("Either accept a 2013+ evaluation window, or backfill 1999-2011 "
                 "participation from another source to make it span-complete.")),
            gap("GAP-003", "Depth charts (nflverse)",
                "Starter/backup designation before kickoff; the cleanest causal "
                "signal for QB starter changes.",
                "FREE AND REACHABLE; not yet integrated, and publication timestamps "
                "are not exposed so causal use needs care",
                "HIGH",
                "Depth chart snapshots with the timestamp each version was published."),
            gap("GAP-004", "Next Gen Stats (time to throw, separation, aggressiveness)",
                "Separates QB pressure response and receiver-created opportunity "
                "from raw box-score outcomes.",
                "FREE via nflverse but 2016+ only",
                "MEDIUM",
                "2016+ acceptable if evaluated on a restricted window; otherwise backfill."),
            gap("GAP-005", "Offensive/defensive line grades and trench matchups",
                "Layer 6 of the brief asks for trench interactions. V2 proxies "
                "pressure with team sack and QB-hit rates from play-by-play, which "
                "conflates line quality, scheme and QB pocket behaviour.",
                "NOT AVAILABLE - PFF or equivalent is paid",
                "MEDIUM",
                "Per-game OL/DL grades or pressure rates attributable to linemen."),
            gap("GAP-006", "Coverage scheme and route-tree data",
                "Man/zone rates and route participation drive receiver matchup edges "
                "and would materially improve WR/TE targets once those become "
                "headline markets.",
                "NOT AVAILABLE - paid",
                "MEDIUM",
                "Per-game coverage split by defense and route participation by receiver."),
            gap("GAP-007", "Kickoff weather (wind, precipitation, temperature)",
                "Wind in particular compresses deep passing and lowers attempts.",
                "PARTIALLY FREE; not integrated in V2",
                "LOW-MEDIUM",
                "Forecast-at-kickoff (not observed post-hoc) per stadium per game."),
            gap("GAP-008", "source_available_at timestamps on the player-stats panel",
                ("V1's provenance records SOURCE_AVAILABLE_AT_STATUS = NOT_EXPOSED. "
                 "Causality is currently enforced by kickoff ordering, which is "
                 "correct for box-score data but cannot prove when a given field "
                 "actually became queryable."),
                "NOT EXPOSED by the upstream feed",
                "HIGH",
                "Per-row ingestion/publication timestamps from the upstream provider."),
            gap("GAP-009", "CPOE and completion-probability for 1999-2024 in the panel",
                "passing_cpoe exists only for 2025 in the canonical player-stats "
                "table, so it was excluded. The underlying quantity is recoverable "
                "from play-by-play, which V2 already downloads.",
                "RECOVERABLE from nflverse pbp cpoe column - not yet done",
                "MEDIUM",
                "Aggregate pbp cpoe to player-game for 2006-2024 and re-test."),
        ],
    }


def calibration_report(v1_oos, v2_oos, prob_col: str, sel: dict) -> dict:
    def reliability(oos, col, threshold=C.PROB_THRESHOLD, n_bins=10):
        y = (oos[C.TARGET].values > threshold).astype(int)
        p = oos[col].values
        f = pd.DataFrame({"p": p, "y": y})
        f["bin"] = pd.cut(f.p, np.linspace(0, 1, n_bins + 1), include_lowest=True)
        rows = []
        for b, g in f.groupby("bin", observed=True):
            rows.append({"bin": str(b), "n": int(len(g)),
                         "mean_predicted": round(float(g.p.mean()), 4),
                         "empirical_rate": round(float(g.y.mean()), 4),
                         "gap": round(float(g.p.mean() - g.y.mean()), 4)})
        return rows

    def qcov(oos):
        return M.quantile_coverage(
            oos[C.TARGET].values,
            {0.1: oos.q10.values, 0.25: oos.q25.values, 0.5: oos.q50.values,
             0.75: oos.q75.values, 0.9: oos.q90.values})

    def per_threshold(v2d, v1d):
        out = {}
        for t in C.THRESH_LIST_MAIN:
            col = f"p_over_{t}"
            if col not in v2d.columns or len(v2d) == 0:
                continue
            y2 = (v2d[C.TARGET].values > t).astype(int)
            y1 = (v1d[C.TARGET].values > t).astype(int)
            out[str(t)] = {
                "base_rate": round(float(y2.mean()), 4),
                "V2_BRIER": M.brier(y2, v2d[col].values),
                "V2_LOGLOSS": M.logloss(y2, v2d[col].values),
                "V2_ECE": M.ece(y2, v2d[col].values),
                "V1_BRIER": M.brier(y1, v1d[col].values),
                "V1_ECE": M.ece(y1, v1d[col].values),
            }
        return out

    dev2 = v2_oos[v2_oos.oos_season.isin(C.DEV_SEASONS)]
    dev1 = v1_oos[v1_oos.oos_season.isin(C.DEV_SEASONS)]
    tst2 = v2_oos[v2_oos.oos_season.isin(C.TEST_SEASONS)]
    tst1 = v1_oos[v1_oos.oos_season.isin(C.TEST_SEASONS)]
    per_thresh = per_threshold(v2_oos, v1_oos)

    return {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_CALIBRATION_REPORT",
        "GENERATED_AT": NOW,
        "PRIMARY_THRESHOLD": C.PROB_THRESHOLD,
        "CALIBRATION_TRAINED_CAUSALLY": {
            "rule": ("the calibrator for OOS season s is fitted only on pooled OOS "
                     "rows from seasons < s, which were produced by models that "
                     "never saw season s"),
            "test_period_never_used_for_fitting": True,
        },
        "SELECTED_REGIME": {k: sel[k] for k in
                            ("SELECTED_HETERO", "SELECTED_RESID_MODE",
                             "SELECTED_CALIBRATION")},
        "REGIME_COMPARISON_ON_DEV": sel.get("regime_dev_results", {}),
        "RELIABILITY_V2": reliability(v2_oos, prob_col),
        "RELIABILITY_V1": reliability(v1_oos, "p_raw"),
        "QUANTILE_COVERAGE_V2": qcov(v2_oos),
        "QUANTILE_COVERAGE_V1": qcov(v1_oos),
        "QUANTILE_COVERAGE_V2_TEST_ONLY": qcov(tst2),
        "QUANTILE_COVERAGE_V1_TEST_ONLY": qcov(tst1),
        "PER_THRESHOLD_POOLED": per_thresh,
        "PER_THRESHOLD_DEV": per_threshold(dev2, dev1),
        "PER_THRESHOLD_TEST": per_threshold(tst2, tst1),
        "HONEST_READ": {
            "point_accuracy": ("MAE/RMSE/CRPS gains hold up on TEST at ~6%, so "
                               "the mean model and its sharpness generalise"),
            "calibration": ("ECE improves 33% pooled but only ~5% on TEST "
                            "(0.0221 -> 0.0209). Most of the calibration gain is "
                            "a DEV-period effect and is close to a wash on the "
                            "held-out period."),
            "quantile_drift_on_test": ("every V2 TEST coverage gap is positive and "
                                       "peaks at P75 (+0.036), i.e. the upper half "
                                       "of the distribution is slightly too wide. "
                                       "That is exactly the region where 250-300 "
                                       "yard thresholds sit, so it is the first "
                                       "thing to fix before trusting tail prices."),
            "verdict": "CALIBRATION_IMPROVEMENT: MARGINAL_ON_TEST",
        },
        "TAIL_BEHAVIOUR_ON_TEST": {
            "observation": ("V2's Brier beats V1's at every threshold on TEST, but "
                            "the margin collapses as the line rises: -0.0129 at "
                            "150y, -0.0076 at 250y, -0.0005 at 325y. V2's ECE also "
                            "worsens in the tail (0.0153 at 225y -> 0.0351 at 325y)."),
            "consequence": ("V2's edge is concentrated in the middle of the "
                            "distribution. On 300y+ props it is close to no better "
                            "than V1, and less well calibrated there in absolute "
                            "terms. Do not price deep-tail markets off this model "
                            "expecting a V2 advantage."),
        },
    }


def signal_diagnostics(el: pd.DataFrame, result: dict) -> dict:
    """Characterise what the highest-gain features are actually keying on."""
    d = el["days_rest"]
    buckets = {"<=8d": d <= 8, "9-20d": (d > 8) & (d <= 20), ">20d": d > 20}
    rest = {k: {"n": int(m.sum()),
                "mean_passing_yards": round(float(el.loc[m, C.TARGET].mean()), 2),
                "mean_attempts": round(float(el.loc[m, "attempts"].mean()), 2)}
            for k, m in buckets.items()}
    return {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_SIGNAL_DIAGNOSTICS",
        "GENERATED_AT": NOW,
        "DAYS_REST": {
            "rank_by_gbm_gain": 2,
            "buckets": rest,
            "interpretation": ("The >20-day bucket averages 146.7 passing yards on "
                               "21.6 attempts against 226.0 on 31.9 for the normal "
                               "<=8-day bucket. days_rest is therefore acting "
                               "largely as a season-opener / returning-or-spot-"
                               "starter indicator, not as a fatigue signal."),
            "is_leakage": False,
            "why_not": ("Time since a player's previous kickoff is fully known "
                        "before the current kickoff, and the shift(1) check "
                        "confirms it uses no current-game information. It is a "
                        "legitimate proxy for an unobserved variable - role and "
                        "availability - which is precisely the input DATA_GAPS "
                        "GAP-001 asks Node2 to source properly."),
        },
        "FEATURE_GROUP_IMPORTANCE_PCT": result["FEATURE_GROUP_IMPORTANCE_PCT"],
        "WHAT_ACTUALLY_ADDS_VALUE": {
            "carrying_real_signal": ["team_environment", "shrinkage", "player_state"],
            "near_noise_for_this_target": ["opponent", "interactions"],
            "evidence": ("Removing team_environment costs +0.41 MAE on TEST and "
                         "shrinkage +0.27. Removing the opponent-defence block "
                         "costs only +0.09 on TEST and is slightly negative pooled "
                         "(-0.014); removing the explicit matchup interactions "
                         "costs +0.037 on TEST, about 0.06% of MAE."),
            "implication_for_node2": ("The marginal data dollar should go to "
                                      "volume and availability inputs, not to "
                                      "opponent-quality inputs. Opponent-defence "
                                      "features are implemented and causal, but "
                                      "they are not where the money is for "
                                      "QB_PASS_YARDS."),
        },
    }


def freeze_model(el: pd.DataFrame, sel: dict) -> dict:
    """Refit the selected model on every eligible row through 2025 and persist
    everything needed to score a future game."""
    feats = F.V2_FEATURES
    y = el[C.TARGET].values

    prep = Mo.ImputeStandardize().fit(el[feats])
    ridge = Ridge(alpha=10.0).fit(prep.transform(el[feats]), y)

    import lightgbm as lgb
    gbm = lgb.LGBMRegressor(
        n_estimators=500, objective="l2", learning_rate=0.04, num_leaves=15,
        min_child_samples=80, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.6, reg_lambda=5.0, max_depth=5, verbose=-1,
        n_jobs=4, seed=C.SEED)
    gbm.fit(el[feats], y)

    w = 0.5
    pred_in = w * gbm.predict(el[feats]) + (1 - w) * ridge.predict(prep.transform(el[feats]))
    resid_in = y - pred_in
    scale = D.ScaleModel().fit(el, resid_in)
    sigma_in = scale.sigma(el)
    z_pool = resid_in / np.clip(sigma_in, 1e-6, None)

    booster_path = C.ARTIFACTS / "SPORTS_NOVA_V2_FROZEN_GBM.txt"
    gbm.booster_.save_model(str(booster_path))

    payload = {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_FROZEN_MODEL",
        "GENERATED_AT": NOW,
        "TARGET": "QB_PASS_YARDS",
        "MODEL_TYPE": sel["SELECTED_MEAN_MODEL"],
        "BLEND": {"gbm_weight": w, "ridge_weight": 1 - w},
        "TRAIN_ROWS": int(len(el)),
        "TRAIN_SEASON_RANGE": [int(el.season.min()), int(el.season.max())],
        "ELIGIBILITY_RULE": ("position == 'QB' and attempts > 0 and "
                             "games_played_prior >= 4 and event_time_status == "
                             "EXACT and V1 feature set non-null"),
        "FEATURES": feats,
        "RIDGE": {
            "alpha": 10.0,
            "intercept": float(ridge.intercept_),
            "coefficients": dict(zip(prep.feature_names(),
                                     [float(c) for c in ridge.coef_])),
            "impute_median": {k: float(v) for k, v in prep.median_.items()},
            "standardize_mu": [float(x) for x in prep.mu_],
            "standardize_sigma": [float(x) for x in prep.sigma_],
            "indicator_columns": prep.indicator_cols_,
        },
        "GBM_BOOSTER_FILE": booster_path.name,
        "GBM_BOOSTER_SHA256": sha256_file(booster_path),
        "DISTRIBUTION": {
            "method": "heteroskedastic empirical residual pool",
            "scale_model_features": scale.used_,
            "scale_model_coefficients": [float(c) for c in scale.m_.coef_],
            "scale_model_intercept": float(scale.m_.intercept_),
            "scale_floor": scale.floor_,
            "z_pool_n": int(len(z_pool)),
            "z_pool_quantiles": {str(q): float(np.quantile(z_pool, q))
                                 for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
                                           0.9, 0.95, 0.99)},
            "note": ("z_pool here is in-sample for the frozen artefact; all "
                     "reported OOS metrics used strictly causal pools"),
        },
        "PROBABILITY_CURVE": {
            "thresholds": f"{C.CURVE_LO} to {C.CURVE_HI} step {C.CURVE_STEP}",
            "monotonic_by_construction": True,
        },
        "MARKET_DATA_USED": False,
        "SEED": C.SEED,
    }
    np.save(C.ARTIFACTS / "SPORTS_NOVA_V2_FROZEN_ZPOOL.npy", z_pool)
    payload["Z_POOL_FILE"] = "SPORTS_NOVA_V2_FROZEN_ZPOOL.npy"
    payload["Z_POOL_SHA256"] = sha256_file(C.ARTIFACTS / "SPORTS_NOVA_V2_FROZEN_ZPOOL.npy")
    return payload


def main():
    with open(C.ARTIFACTS / "_final_state.pkl", "rb") as fh:
        st = pickle.load(fh)
    el, v1_oos, v2_oos, result = st["el"], st["v1_oos"], st["v2_oos"], st["result"]
    sel = json.loads((C.ARTIFACTS / "SPORTS_NOVA_V2_DEV_SELECTION.json").read_text())
    prob_col = "p_raw" if sel["SELECTED_CALIBRATION"] == "none" else "p_cal"

    hashes = {}
    hashes["SPORTS_NOVA_V2_FEATURE_SCHEMA.json"] = write(
        "SPORTS_NOVA_V2_FEATURE_SCHEMA.json", feature_schema(el))

    comparison = {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_MODEL_COMPARISON",
        "GENERATED_AT": NOW,
        "SELECTION_PROTOCOL": sel["selection_rule"],
        "DEV_SEASONS": C.DEV_SEASONS,
        "TEST_SEASONS": C.TEST_SEASONS,
        "CANDIDATES_ON_DEV": sel["candidate_dev_results"],
        "SELECTED": sel["SELECTED_MEAN_MODEL"],
        "FINAL_V1": result["V1"],
        "FINAL_V2": result["V2"],
        "IMPROVEMENT_TEST": result["IMPROVEMENT_TEST"],
        "IMPROVEMENT_POOLED": result["IMPROVEMENT_POOLED"],
        "FEATURE_GROUP_IMPORTANCE_PCT": result["FEATURE_GROUP_IMPORTANCE_PCT"],
        "TOP_25_FEATURES_BY_GAIN": result["TOP_25_FEATURES_BY_GAIN"],
    }
    hashes["SPORTS_NOVA_V2_MODEL_COMPARISON.json"] = write(
        "SPORTS_NOVA_V2_MODEL_COMPARISON.json", comparison)

    hashes["SPORTS_NOVA_V2_CALIBRATION_REPORT.json"] = write(
        "SPORTS_NOVA_V2_CALIBRATION_REPORT.json",
        calibration_report(v1_oos, v2_oos, prob_col, sel))

    abl = result["ABLATIONS"]
    full_pooled = result["V2"]["POOLED_2006_2025"]
    ranked = []
    for name, blk in abl.items():
        p = blk["scores"]["POOLED_2006_2025"]
        ranked.append({
            "ablation": name,
            "n_features": blk["n_features"],
            "MAE": p["MAE"], "BRIER": p["BRIER"], "LOGLOSS": p["LOGLOSS"],
            "MAE_DAMAGE_VS_FULL_V2": round(p["MAE"] - full_pooled["MAE"], 4),
            "BRIER_DAMAGE_VS_FULL_V2": round(p["BRIER"] - full_pooled["BRIER"], 5),
        })
    ranked.sort(key=lambda r: -r["MAE_DAMAGE_VS_FULL_V2"])
    hashes["SPORTS_NOVA_V2_ABLATION_REPORT.json"] = write(
        "SPORTS_NOVA_V2_ABLATION_REPORT.json", {
            "ARTIFACT_ID": "SPORTS_NOVA_V2_ABLATION_REPORT",
            "GENERATED_AT": NOW,
            "FULL_V2_POOLED": full_pooled,
            "INTERPRETATION": ("MAE_DAMAGE_VS_FULL_V2 > 0 means removing that "
                               "group hurt, so the group was carrying real signal. "
                               "Ranked most damaging first."),
            "ABLATIONS_RANKED": ranked,
            "DETAIL": abl,
        })

    hashes["SPORTS_NOVA_V2_DATA_GAPS.json"] = write(
        "SPORTS_NOVA_V2_DATA_GAPS.json", data_gaps())

    hashes["SPORTS_NOVA_V2_SIGNAL_DIAGNOSTICS.json"] = write(
        "SPORTS_NOVA_V2_SIGNAL_DIAGNOSTICS.json", signal_diagnostics(el, result))

    frozen = freeze_model(el, sel)
    hashes["SPORTS_NOVA_V2_FROZEN_MODEL.json"] = write(
        "SPORTS_NOVA_V2_FROZEN_MODEL.json", frozen)

    corr = MC.estimate_correlations(max_season=max(C.DEV_SEASONS))
    hashes["SPORTS_NOVA_V2_JOINT_CORRELATIONS.json"] = write(
        "SPORTS_NOVA_V2_JOINT_CORRELATIONS.json", {
            "ARTIFACT_ID": "SPORTS_NOVA_V2_JOINT_CORRELATIONS",
            "GENERATED_AT": NOW,
            "ESTIMATED_ON": "training seasons only (<= 2019)",
            "RESIDUAL_DEFINITION": "actual minus player's own trailing-5 mean",
            "CORRELATIONS": corr,
        })

    # Hash every artifact on disk, not just the ones this function wrote, so a
    # file added by a later step cannot sit outside the manifest.
    for p in sorted(C.ARTIFACTS.iterdir()):
        if p.is_file() and p.name != "_final_state.pkl" and not p.name.startswith(
                "SPORTS_NOVA_V2_ENGINE_MANIFEST"):
            hashes[p.name] = sha256_file(p)

    manifest = {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_ENGINE_MANIFEST",
        "GENERATED_AT": NOW,
        "ENGINE_VERSION": "SPORTS_NOVA_V2",
        "SPORT": "NFL",
        "TARGET": "QB_PASS_YARDS",
        "CAUSALITY_MODE": "EVENT_CAUSAL_RESEARCH_V2",
        "V1_PRESERVED_AT": str(C.V1_DIR),
        "V1_UNMODIFIED": True,
        "V1_REPRODUCED_IN_V2_HARNESS": True,
        "CODE_NAMESPACE": "worker/sports_nova_v2/",
        "DATA_NAMESPACE": str(C.V2_DATA),
        "SOURCES": {
            "player_game_panel": "NFL_PLAYER_GAME_CAUSAL_2025_V1.parquet (V1, unmodified)",
            "play_by_play": "nflverse-data release 'pbp', seasons 1999-2025, "
                            "reduced to a team-game ledger at ingest",
            "market_data": "NONE - no odds, spreads, totals or vegas_wp anywhere",
        },
        "ARTIFACT_SHA256": hashes,
        "SEED": C.SEED,
    }
    mpath = C.ARTIFACTS / "SPORTS_NOVA_V2_ENGINE_MANIFEST.json"
    mpath.write_text(json.dumps(manifest, indent=2, default=str))

    print(json.dumps({"artifacts": list(hashes),
                      "frozen_model_sha256": hashes["SPORTS_NOVA_V2_FROZEN_MODEL.json"],
                      "manifest_sha256": sha256_file(mpath),
                      "correlations": corr}, indent=2, default=str))


if __name__ == "__main__":
    main()
