"""Freeze V2.1 and run the 2026 Week-1 feature-support preflight.

Two things happen here:

1. The DEV-selected V2.1 model is refitted on every eligible row through the
   2025 season and persisted with complete scoring parameters. V2's artifact
   omitted the scale model's impute/standardise constants, which made its sigma
   irreproducible from the artifact alone; that is fixed here.

2. A pregame feature row is constructed for the 2026 Week-1 game to prove the
   feature pipeline has real support there. Synthetic *placeholder* rows carry
   no outcome data - they exist only so the shift(1) trailing windows resolve
   against completed 2025 games.

No odds are fetched. The schedule is read with market columns excluded.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import config as C
from . import distribution as D
from . import features as F
from . import features_v21 as V21
from . import models as Mo
from . import panel as P

NOW = datetime.now(timezone.utc).isoformat()
SCHED_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
             "schedules/games.parquet")
SCHED_COLS = ["game_id", "season", "game_type", "week", "gameday", "gametime",
              "weekday", "away_team", "home_team", "location", "stadium", "roof"]
TARGET_GAME = "2026_01_NE_SEA"
TARGET_PLAYER = "Drake Maye"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
def load_schedule_2026() -> pd.DataFrame:
    raw = urllib.request.urlopen(SCHED_URL, timeout=180).read()
    df = pd.read_parquet(io.BytesIO(raw), columns=SCHED_COLS)
    P._assert_no_market_columns(df)
    return df[df.season == 2026]


def build_preflight_panel(sched: pd.DataFrame):
    """Historical panel plus placeholder rows for 2026 Week 1."""
    causal = P.load_event_causal()
    canon = P._canon_map()

    wk1 = sched[(sched.week == 1) & (sched.game_type == "REG")].copy()
    kick = pd.to_datetime(wk1.gameday.astype(str) + " " + wk1.gametime.astype(str))
    # Schedule times are US Eastern; September is EDT (UTC-4).
    wk1["kickoff_utc"] = (kick + pd.Timedelta(hours=4)).dt.tz_localize("UTC")

    # One placeholder row per team so every 2026 Week-1 team's trailing windows
    # resolve. No outcome fields are populated.
    rows = []
    for _, g in wk1.iterrows():
        for team, opp, home in ((g.home_team, g.away_team, True),
                                (g.away_team, g.home_team, False)):
            rows.append({"season": 2026, "week": 1, "season_type": "REG",
                         "recent_team": team, "opponent_team": opp,
                         "team_canon": canon.get(team), "opp_canon": canon.get(opp),
                         "kickoff_utc": g.kickoff_utc, "is_home": home,
                         "game_id_2026": g.game_id})
    team_rows = pd.DataFrame(rows)

    # --- player placeholder: the QB we are scoring ---
    qb_hist = causal[(causal.position == "QB") & (causal.attempts > 0)].copy()
    qb_hist["kickoff_timestamp_utc"] = pd.to_datetime(
        qb_hist["m_kickoff_timestamp_utc"], utc=True)
    pid = qb_hist[qb_hist.player_display_name == TARGET_PLAYER].player_id.iloc[-1]
    tgt = team_rows[team_rows.game_id_2026 == TARGET_GAME]
    ne = tgt[tgt.recent_team == "NE"].iloc[0]

    ph = {c: np.nan for c in qb_hist.columns}
    ph.update({
        "player_id": pid, "player_display_name": TARGET_PLAYER, "position": "QB",
        "season": 2026, "week": 1, "season_type": "REG",
        "recent_team": ne.recent_team, "opponent_team": ne.opponent_team,
        "kickoff_timestamp_utc": ne.kickoff_utc,
        "m_kickoff_timestamp_utc": ne.kickoff_utc,
        "m_is_home": bool(ne.is_home), "event_time_status": "EXACT",
        "m_canonical_game_id": TARGET_GAME,
    })
    qb = pd.concat([qb_hist, pd.DataFrame([ph])], ignore_index=True)
    qb = qb.sort_values(["player_id", "kickoff_timestamp_utc"]).reset_index(drop=True)

    # V1 feature construction, unchanged.
    qb = F.mask_air_yards_era(qb)
    qb = F.harmonize_schema_split(qb)
    g = qb.groupby("player_id")
    qb["games_played_prior"] = g.cumcount()
    qb["career_prior_pass_yds"] = g["passing_yards"].transform(
        lambda s: s.shift(1).expanding().mean())
    qb["career_prior_attempts"] = g["attempts"].transform(
        lambda s: s.shift(1).expanding().mean())
    qb["trailing4_pass_yds"] = g["passing_yards"].transform(
        lambda s: s.shift(1).rolling(4).mean())
    qb["trailing4_attempts"] = g["attempts"].transform(
        lambda s: s.shift(1).rolling(4).mean())
    qb["season_td_pass_yds"] = qb.groupby(["player_id", "season"])[
        "passing_yards"].transform(lambda s: s.shift(1).expanding().mean())
    qb["season_td_pass_yds"] = qb["season_td_pass_yds"].fillna(qb["career_prior_pass_yds"])
    qb["team_canon"] = qb["recent_team"].map(canon)
    qb["opp_canon"] = qb["opponent_team"].map(canon)

    qb = F.add_player_state(qb)
    qb = F.add_shrinkage(qb)

    # --- team ledger with 2026 Week-1 placeholders ---
    kickoffs = (qb.groupby(["season", "week", "team_canon"])["kickoff_timestamp_utc"]
                .min().rename("kickoff").reset_index())
    led = F.load_team_game_ledger(kickoffs)
    metric_cols = [c for c in led.columns
                   if c not in ("season", "week", "season_type", "game_id",
                                "team", "team_canon", "kickoff")]
    ph_led = pd.DataFrame({
        "season": 2026, "week": 1, "season_type": "REG",
        "game_id": team_rows.game_id_2026.values,
        "team": team_rows.recent_team.values,
        "team_canon": team_rows.team_canon.values,
        "kickoff": team_rows.kickoff_utc.values})
    for c in metric_cols:
        ph_led[c] = np.nan
    led = pd.concat([led, ph_led[led.columns]], ignore_index=True)
    led = F.add_team_trailing(led)

    qb = F.attach_context(qb, led)
    qb = F.add_interactions(qb)

    causal_ext = pd.concat([causal, pd.DataFrame([{
        "player_id": pid, "position": "QB", "season": 2026, "week": 1,
        "recent_team": "NE", "attempts": np.nan}])], ignore_index=True)
    qb = V21.add_role_continuity(qb, causal_ext)
    return qb, pid


# ---------------------------------------------------------------------------
def freeze_v21(el: pd.DataFrame, feats: list[str], sel: dict) -> dict:
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

    pred_in = 0.5 * gbm.predict(el[feats]) + 0.5 * ridge.predict(prep.transform(el[feats]))
    resid_in = y - pred_in
    scale = D.ScaleModel().fit(el, resid_in)
    z_pool = resid_in / np.clip(scale.sigma(el), 1e-6, None)

    booster = C.ARTIFACTS / "SPORTS_NOVA_V2_1_FROZEN_GBM.txt"
    gbm.booster_.save_model(str(booster))
    zp = C.ARTIFACTS / "SPORTS_NOVA_V2_1_FROZEN_ZPOOL.npy"
    np.save(zp, z_pool)

    return {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_1_FROZEN_MODEL",
        "GENERATED_AT": NOW,
        "ENGINE_VERSION": "SPORTS_NOVA_V2_1",
        "SUPERSEDES": "SPORTS_NOVA_V2 (not overwritten; V2 artifacts remain intact)",
        "TARGET": "QB_PASS_YARDS",
        "MODEL_TYPE": f"{sel['SELECTED_CANDIDATE']} / blend_50",
        "FIX_SUMMARY": ("adds is_season_opener and is_team_week1 so an offseason "
                        "gap and an in-season 30-day absence are no longer the "
                        "same point in feature space"),
        "TRAIN_ROWS": int(len(el)),
        "TRAIN_SEASON_RANGE": [int(el.season.min()), int(el.season.max())],
        "FEATURES": feats,
        "RIDGE": {
            "alpha": 10.0, "intercept": float(ridge.intercept_),
            "coefficients": dict(zip(prep.feature_names(),
                                     [float(c) for c in ridge.coef_])),
            "impute_median": {k: float(v) for k, v in prep.median_.items()},
            "standardize_mu": [float(x) for x in prep.mu_],
            "standardize_sigma": [float(x) for x in prep.sigma_],
            "indicator_columns": prep.indicator_cols_,
        },
        "GBM_BOOSTER_FILE": booster.name,
        "GBM_BOOSTER_SHA256": sha256_file(booster),
        "BLEND": {"gbm_weight": 0.5, "ridge_weight": 0.5},
        "DISTRIBUTION": {
            "method": "heteroskedastic empirical residual pool",
            "scale_model_features": scale.used_,
            "scale_model_coefficients": [float(c) for c in scale.m_.coef_],
            "scale_model_intercept": float(scale.m_.intercept_),
            "scale_floor": float(scale.floor_),
            "scale_sigma_multiplier": float(np.sqrt(np.pi / 2)),
            # These were missing from V2's artifact, which made sigma
            # irreproducible from the artifact alone.
            "scale_impute_median": {k: float(v) for k, v in scale.prep_.median_.items()},
            "scale_standardize_mu": [float(x) for x in scale.prep_.mu_],
            "scale_standardize_sigma": [float(x) for x in scale.prep_.sigma_],
            "scale_indicator_columns": scale.prep_.indicator_cols_,
            "z_pool_file": zp.name, "z_pool_sha256": sha256_file(zp),
            "z_pool_n": int(len(z_pool)),
        },
        "MARKET_DATA_USED": False,
        "SEED": C.SEED,
    }, (prep, ridge, gbm, scale, z_pool)


def main():
    sel = json.loads((C.ARTIFACTS / "SPORTS_NOVA_V2_1_DEV_SELECTION.json").read_text())
    feats = sel["SELECTED_FEATURES"]

    el, _ = V21.build_v21_panel()
    frozen, (prep, ridge, gbm, scale, z_pool) = freeze_v21(el, feats, sel)
    fp = C.ARTIFACTS / "SPORTS_NOVA_V2_1_FROZEN_MODEL.json"
    fp.write_text(json.dumps(frozen, indent=2, default=str))
    print("frozen V2.1 ->", fp)
    print("FROZEN_MODEL_SHA256:", sha256_file(fp))

    # ---- preflight ----
    sched = load_schedule_2026()
    qb, pid = build_preflight_panel(sched)
    row = qb[(qb.player_id == pid) & (qb.season == 2026) & (qb.week == 1)]
    assert len(row) == 1, f"expected one placeholder row, got {len(row)}"
    r = row.iloc[0]

    eligible_ok = bool(r.games_played_prior >= C.MIN_PRIOR_GAMES and
                       row[C.V1_FEATURES].notna().all(axis=1).iloc[0])
    missing = [f for f in feats if not np.isfinite(pd.to_numeric(
        pd.Series([r.get(f)]), errors="coerce").iloc[0])]

    # A NaN is only a support problem if historical Week-1 rows have that
    # feature populated. Season-to-date features are NaN in week 1 by
    # construction, and the model was trained on exactly that pattern.
    hist_w1_rows = el[(el.week == 1) & (el.season >= 2006)]
    hist_w1_cov = hist_w1_rows[feats].notna().mean()
    expected_nan = [f for f in missing if hist_w1_cov.get(f, 1.0) < 0.05]
    unexpected_nan = [f for f in missing if hist_w1_cov.get(f, 1.0) >= 0.05]

    X = row[feats]
    pred = float(0.5 * gbm.predict(X)[0] + 0.5 * ridge.predict(prep.transform(X))[0])
    sigma = float(scale.sigma(row)[0])
    sample = sigma * z_pool
    stats = D.summary_stats(pred, sample)
    curve = D.dense_curve(pred, sample)
    keys = sorted(float(k) for k in curve)
    vals = [curve[str(round(k, 1))] for k in keys]
    mono = sum(1 for i in range(len(vals) - 1) if vals[i] < vals[i + 1] - 1e-12)

    hist_w1 = el[(el.week == 1) & (el.season >= 2006)]
    report_feats = ["days_rest", "is_season_opener", "is_team_week1",
                    "season_games_prior", "games_played_prior",
                    "trailing4_pass_yds", "trailing4_attempts",
                    "career_prior_pass_yds", "season_td_pass_yds",
                    "pass_yds_w5", "attempts_w5", "ypa_w5",
                    "pass_yds_w8_shrunk", "exp_attempts", "exp_pass_yds",
                    "off_plays_w5", "off_pass_rate_w5", "off_points_s2d",
                    "def_pass_yds_w5", "def_pass_epa_per_db_w5",
                    "prev_game_attempt_share", "prev_game_was_team_lead",
                    "career_lead_games", "career_lead_share",
                    "prior_season_attempt_share", "prior_season_lead_games",
                    "prior_season_games", "same_team_as_prior_game",
                    "same_team_as_prior_season", "team_games_missed",
                    "offseason_gap_days", "in_season_days_rest",
                    "crossed_season_boundary"]
    fv = {}
    for f in report_feats:
        v = r.get(f)
        try:
            fv[f] = None if pd.isna(v) else round(float(v), 4)
        except (TypeError, ValueError):
            fv[f] = str(v)

    preflight = {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_1_WEEK1_2026_PREFLIGHT",
        "GENERATED_AT": NOW,
        "GAME": TARGET_GAME, "MATCHUP": "NE @ SEA",
        "KICKOFF_UTC": str(r.kickoff_timestamp_utc),
        "PLAYER": TARGET_PLAYER, "PLAYER_ID": str(pid),
        "SCHEDULE_SOURCE": "nflverse schedules; market columns excluded at read",
        "ODDS_FETCHED": False,
        "PREGAME_FEATURE_VALUES": fv,
        "V1_ELIGIBILITY_SATISFIED": eligible_ok,
        "FEATURES_REQUIRED": len(feats),
        "N_FEATURES_NAN": len(missing),
        "NAN_EXPECTED_FOR_WEEK1": {
            "features": expected_nan, "count": len(expected_nan),
            "reason": ("season-to-date team features are NaN for every week-1 row, "
                       "historical and live alike; the model was trained on this "
                       "exact pattern and LightGBM handles NaN natively")},
        "NAN_UNEXPECTED": {
            "features": unexpected_nan, "count": len(unexpected_nan),
            "reason": "populated for historical week-1 rows but missing here - "
                      "these would be genuine support gaps"},
        "HISTORICAL_WEEK1_SUPPORT": {
            "training_week1_rows_2006_2025": int(len(hist_w1)),
            "week1_mean_passing_yards": round(float(hist_w1.passing_yards.mean()), 2),
            "week1_mean_attempts": round(float(hist_w1.attempts.mean()), 2),
            "note": ("Week-1 rows are a well-populated, in-distribution regime. "
                     "The earlier Node4 claim that Week-1 starters map onto a "
                     "148-yard backup cohort was wrong: that 148 figure blended "
                     "796 season openers (221.6 yds) with 1,035 in-season "
                     "absences (91.8 yds)."),
        },
        "MODEL_OUTPUT": {
            "POINT_PREDICTION": round(pred, 2), "SIGMA": round(sigma, 2),
            **{k: round(v, 2) for k, v in stats.items()},
            "P_OVER_MAIN": {f"P_OVER_{t}": round(float((pred + sample > t).mean()), 4)
                            for t in C.THRESH_LIST_MAIN},
            "CURVE_MONOTONIC_VIOLATIONS": mono,
        },
        "WEEK1_SUPPORT_VALID": bool(eligible_ok and len(unexpected_nan) == 0),
        "MODEL_PROBABILITY_AVAILABLE": bool(eligible_ok and len(unexpected_nan) == 0),
        "LIMITATION": ("No causal injury/availability data. The model assumes the "
                       "listed QB starts and plays a normal workload; it cannot "
                       "know a pregame inactive or an early exit."),
    }
    pp = C.ARTIFACTS / "SPORTS_NOVA_V2_1_WEEK1_2026_PREFLIGHT.json"
    pp.write_text(json.dumps(preflight, indent=2, default=str))
    print(json.dumps({k: preflight[k] for k in
                      ("GAME", "KICKOFF_UTC", "PLAYER", "V1_ELIGIBILITY_SATISFIED",
                       "N_FEATURES_NAN", "NAN_EXPECTED_FOR_WEEK1", "NAN_UNEXPECTED",
                       "WEEK1_SUPPORT_VALID", "MODEL_PROBABILITY_AVAILABLE",
                       "MODEL_OUTPUT")}, indent=2, default=str))
    print("\nkey pregame features:")
    for k in ("days_rest", "is_season_opener", "is_team_week1", "team_games_missed",
              "offseason_gap_days", "in_season_days_rest", "prev_game_attempt_share",
              "prev_game_was_team_lead", "career_lead_games", "career_lead_share",
              "prior_season_attempt_share", "prior_season_lead_games",
              "same_team_as_prior_season", "trailing4_pass_yds", "attempts_w5"):
        print(f"  {k:32s} {fv.get(k)}")


if __name__ == "__main__":
    main()
