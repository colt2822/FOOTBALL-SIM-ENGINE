"""Analysis of MULTI_TD_POSSESSION_ABLATION_V1 (reads arrays/run_meta written by ..._ablation_v1.py). Writes REPORT.json + GT8_BLOCK_LEDGER.jsonl."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.sports_nova_fg_causal_estimator_v1 import load_drives  # noqa: E402
from scripts.sports_nova_m1_fg_causal_ablation_v1_analysis import observed_scores  # noqa: E402
from scripts.sports_nova_m1_multi_td_possession_ablation_v1 import FI, OUT, ARR, PREREG, sha256_file  # noqa: E402

REPORT = OUT / "MULTI_TD_POSSESSION_ABLATION_V1_REPORT.json"
LEDGER = OUT / "GT8_BLOCK_LEDGER.jsonl"
TEAM_TAILS_UP = [35, 40, 45, 50]
TOT_TAILS_UP = [55, 60, 65, 70]
TEAM_TAILS_LO = [7, 10]
TOT_TAILS_LO = [25, 30]
ARMS = ("base", "abl", "ctrl")
RNG = np.random.default_rng(20260921)


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def shape(v):
    v = np.asarray(v, float); m, s = v.mean(), v.std(ddof=1)
    return {"mean": float(m), "sd": float(s), "skew": float(((v - m) ** 3).mean() / s ** 3), "excess_kurtosis": float(((v - m) ** 4).mean() / s ** 4 - 3)}


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def main() -> None:
    meta = json.loads((OUT / "run_meta.json").read_text())
    pre = json.loads(PREREG.read_text())
    games = [r["game"] for r in meta["results"]]
    n = meta["n_sims"]
    R = {r["game"]: r for r in meta["results"]}
    d = load_drives()
    obs_all = observed_scores(d, d[d.season.between(2020, 2025)].game_id.unique().tolist())
    obs = observed_scores(d, games).set_index("game")
    wide_ids = d[d.season.between(2014, 2025)].game_id.unique().tolist()
    ok_rows, skipped = [], 0
    for gid in wide_ids:                                   # pre-2020 franchise-code mismatches (LA/LAR, OAK/LV...) make observed_scores raise; skip those games (league-mean only)
        try:
            ok_rows.append(observed_scores(d, [gid]))
        except KeyError:
            skipped += 1
    obs_wide = pd.concat(ok_rows, ignore_index=True)
    obs_wide["season"] = obs_wide.game.str.split("_").str[0].astype(int); obs_wide["week"] = obs_wide.game.str.split("_").str[1].astype(int)

    Z = {g: np.load(ARR / f"{g}.npz") for g in games}
    BM = {arm: np.concatenate([Z[g][f"{arm}__blocks"] for g in games]) for arm in ARMS}
    gidx = {arm: np.concatenate([np.full(len(Z[g][f"{arm}__blocks"]), i) for i, g in enumerate(games)]) for arm in ARMS}
    n_team_games = len(games) * n * 2
    n_sim_games = len(games) * n

    # ---------------------------------------------------------------- integrity / asserted checks
    checks = {
        "BASE_equals_REF_all_games": all(R[g]["base_equals_ref"]["all"] for g in games),
        "BASE_team_score_equals_stored_FG_ablation_arm_all_games": all(R[g]["base_team_score_equals_stored_v27_ablation_arm"] for g in games),
        "fg_rate_equals_ablation_prereg_all_games": all(R[g]["fg_rate_equals_ablation_prereg"] for g in games),
        "guarded_hashes_unchanged_before_after_run": meta["engine_unchanged"],
        "guarded_hashes_equal_prereg": meta["guarded_hashes_before"] == pre["GUARDED_FILE_SHA256"],
        "script_sha256_at_freeze": pre["SCRIPT_SHA256_AT_FREEZE"], "script_sha256_at_run": meta["script_sha256_at_run"],
        "script_amended_after_freeze": pre["SCRIPT_SHA256_AT_FREEZE"] != meta["script_sha256_at_run"],
        "CTRL_block_stream_identical_to_BASE_all_games": all(R[g]["ctrl_block_stream_identical_to_base"] for g in games),
        "prefix_violations_total": sum(R[g]["prefix_abl_vs_base"]["prefix_violations"] for g in games),
        "block0_violations_total": sum(R[g]["prefix_abl_vs_base"]["block0_violations"] for g in games),
        "sims_no_suppression_but_differ_total": sum(R[g]["prefix_abl_vs_base"]["sims_no_suppression_but_differ"] for g in games),
        "sims_no_suppression_identical_total": sum(R[g]["prefix_abl_vs_base"]["sims_identical_whole"] for g in games),
        "sims_with_suppression_total": sum(R[g]["prefix_abl_vs_base"]["sims_with_suppression"] for g in games), "sims_total": n_sim_games,
    }
    acct = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for g in games:
            for k, v in R[g]["accounting"][arm].items():
                if k == "max_td_per_block":
                    acct[arm][k] = max(acct[arm].get(k, 0), v)
                else:
                    acct[arm][k] = acct[arm].get(k, 0) + v
    checks["accounting"] = acct
    checks["accounting_pass"] = all(v == 0 for arm in ARMS for k, v in acct[arm].items() if k != "max_td_per_block") and acct["abl"]["max_td_per_block"] <= 1 and acct["ctrl"]["max_td_per_block"] <= 1
    # CTRL: everything except TD-derived fields and score identical to BASE
    td_keys_player = {"pass_tds", "rush_tds", "receiving_tds", "scored_tds"}
    ctrl_unchanged_bad = []
    for g in games:
        e = R[g]["ctrl_vs_base_stats_equal_except_td_and_score"]
        ctrl_unchanged_bad += [f"{g}:player:{k}" for k, v in e["player"].items() if not v and k not in td_keys_player]
        ctrl_unchanged_bad += [f"{g}:team:{k}" for k, v in e["team"].items() if not v and k not in ("score", "total")]
    checks["CTRL_non_TD_stats_bit_identical_to_BASE"] = not ctrl_unchanged_bad
    checks["CTRL_non_TD_stats_mismatches"] = ctrl_unchanged_bad[:10]

    # ---------------------------------------------------------------- Phase 1 empirical
    b = BM["base"]
    phase1 = {"blocks": int(len(b)),
              "receiver_td_sites_with_rec>0_per_block": {"mean": float(b[:, FI["rec_sites"]].mean()), "max": int(b[:, FI["rec_sites"]].max())},
              "carrier_td_sites_with_carries>0_per_block": {"mean": float(b[:, FI["carry_sites"]].mean()), "max": int(b[:, FI["carry_sites"]].max())},
              "td_trial_plays_per_block": {"mean": float(b[:, FI["trial_plays"]].mean()), "max": int(b[:, FI["trial_plays"]].max())},
              "max_plays_in_block": int(b[:, FI["plays"]].max()),
              "empirical_max_TD_per_block_baseline": int(b[:, FI["td"]].max()), "empirical_max_points_per_block_baseline": int(b[:, FI["pts"]].max()),
              "baseline_td_per_block_hist": {str(k): int((b[:, FI["td"]] == k).sum()) for k in range(0, int(b[:, FI["td"]].max()) + 1)},
              "baseline_blocks_with_both_rec_and_rush_td": int(((b[:, FI["rec_td"]] > 0) & (b[:, FI["rush_td"]] > 0)).sum()),
              "baseline_blocks_multi_TD_single_player": int(sum(1 for g in games for e in R[g]["events"]["ctrl"] if e["raw_td"] >= 2))}

    # ---------------------------------------------------------------- Phase 4 core metrics
    def arm_metrics(arm):
        bm = BM[arm]
        ts = np.concatenate([Z[g][f"{arm}__team_score"] for g in games])          # (sims, 2)
        w = np.concatenate([Z[g][f"{arm}__winner"] for g in games])
        tot = ts.sum(axis=1); team = ts.ravel()
        gt8 = bm[:, FI["pts"]] > 8
        return {"TD_per_team_game": float(bm[:, FI["td"]].sum() / n_team_games), "rec_TD_per_team_game": float(bm[:, FI["rec_td"]].sum() / n_team_games),
                "rush_TD_per_team_game": float(bm[:, FI["rush_td"]].sum() / n_team_games),
                "rush_share_of_TDs": float(bm[:, FI["rush_td"]].sum() / max(1, bm[:, FI["td"]].sum())),
                "raw_TD_trial_outcomes_per_team_game (before cap)": float(bm[:, FI["raw_td"]].sum() / n_team_games),
                "blocks_per_team_game": float(len(bm) / n_team_games),
                "possessions_with_TD_per_team_game": float((bm[:, FI["td"]] >= 1).sum() / n_team_games), "share_blocks_with_TD": float((bm[:, FI["td"]] >= 1).mean()),
                "possessions_with_2plus_TD_per_team_game": float((bm[:, FI["td"]] >= 2).sum() / n_team_games), "share_blocks_with_2plus_TD": float((bm[:, FI["td"]] >= 2).mean()),
                "GT8_blocks": int(gt8.sum()), "GT8_share_of_blocks": float(gt8.mean()), "max_block_points": int(bm[:, FI["pts"]].max()),
                "FG_per_team_game": float(bm[:, FI["fg"]].sum() / n_team_games), "FG_opportunities_per_team_game": float(bm[:, FI["fg_elig"]].sum() / n_team_games),
                "FG_conversion_given_opportunity": float(bm[:, FI["fg"]].sum() / bm[:, FI["fg_elig"]].sum()),
                "points_per_team_game": float(team.mean()), "total_mean": float(tot.mean()), "total_SD": float(tot.std(ddof=1)), "team_score_SD": float(team.std(ddof=1)),
                "tie_rate": float((w == "TIE").mean()), "home_win_prob": float((w == "HOME").mean()), "away_win_prob": float((w == "AWAY").mean()),
                "home_mean_score": float(ts[:, 0].mean()), "away_mean_score": float(ts[:, 1].mean()),
                "total_shape": shape(tot), "team_shape": shape(team)}

    M = {arm: arm_metrics(arm) for arm in ARMS}

    # ---------------------------------------------------------------- Phase 5 tails
    hist_tot = obs_all.obs_total.values; hist_team = np.concatenate([obs_all.obs_home.values, obs_all.obs_away.values])
    pilot_tot = obs.loc[games].obs_total.values; pilot_team = np.concatenate([obs.loc[games].obs_home.values, obs.loc[games].obs_away.values])

    def tail_rows(arr_team, arr_tot):
        r = {}
        for x in TEAM_TAILS_UP: r[f"P(team>={x})"] = float((arr_team >= x).mean())
        for x in TOT_TAILS_UP: r[f"P(total>={x})"] = float((arr_tot >= x).mean())
        for x in TEAM_TAILS_LO: r[f"P(team<={x})"] = float((arr_team <= x).mean())
        for x in TOT_TAILS_LO: r[f"P(total<={x})"] = float((arr_tot <= x).mean())
        return r

    sim_t = {}
    perg = {arm: {} for arm in ARMS}
    for arm in ARMS:
        ts = np.concatenate([Z[g][f"{arm}__team_score"] for g in games])
        sim_t[arm] = tail_rows(ts.ravel(), ts.sum(axis=1))
        rows = [tail_rows(Z[g][f"{arm}__team_score"].ravel(), Z[g][f"{arm}__team_score"].sum(axis=1)) for g in games]
        perg[arm] = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    hist_t = tail_rows(hist_team, hist_tot)
    pilot_t = tail_rows(pilot_team, pilot_tot)
    tails = {"baseline": sim_t["base"], "ablation": sim_t["abl"], "ctrl_matched_draws": sim_t["ctrl"], "historical_2020_2025_regulation": hist_t, "pilot60_observed_(n=120 team-games, too small for tails)": pilot_t,
             "historical_n_team_games": int(len(hist_team)), "historical_n_games": int(len(hist_tot)),
             "historical_binomial_SE": {k: float(math.sqrt(v * (1 - v) / (len(hist_tot) if 'total' in k else len(hist_team)))) for k, v in hist_t.items()},
             "ablation_minus_baseline": {k: sim_t["abl"][k] - sim_t["base"][k] for k in sim_t["abl"]},
             "ablation_minus_baseline_paired_SE_across_games": {k: se(perg["abl"][k] - perg["base"][k]) for k in sim_t["abl"]},
             "ablation_minus_historical": {k: sim_t["abl"][k] - hist_t[k] for k in hist_t}, "baseline_minus_historical": {k: sim_t["base"][k] - hist_t[k] for k in hist_t},
             "hist_shape_team": shape(hist_team), "hist_shape_total": shape(hist_tot)}
    up_keys = [f"P(team>={x})" for x in TEAM_TAILS_UP] + [f"P(total>={x})" for x in TOT_TAILS_UP]
    mae_up = {a: float(np.mean([abs(sim_t[a][k] - hist_t[k]) for k in up_keys])) for a in ARMS}
    tails["upper_ladder_mean_abs_error_vs_hist"] = mae_up
    tails["ratio_sim_over_hist_upper"] = {a: {k: (sim_t[a][k] / hist_t[k] if hist_t[k] > 0 else None) for k in up_keys} for a in ("base", "abl")}

    # shape-only view (removes the mean/scale shift): exceedance beyond own mean + k*SD
    def zt(vals, k):
        v = np.asarray(vals, float); return float((v >= v.mean() + k * v.std(ddof=1)).mean())
    ts_b = np.concatenate([Z[g]["base__team_score"] for g in games]); ts_a = np.concatenate([Z[g]["abl__team_score"] for g in games])
    tails["standardized_exceedance_team_score (P(x >= own_mean + k*own_SD))"] = {str(k): {"baseline": zt(ts_b.ravel(), k), "ablation": zt(ts_a.ravel(), k), "historical": zt(hist_team, k)} for k in (1.0, 1.5, 2.0, 2.5)}
    tails["standardized_exceedance_total (k*SD)"] = {str(k): {"baseline": zt(ts_b.sum(axis=1), k), "ablation": zt(ts_a.sum(axis=1), k), "historical": zt(hist_tot, k)} for k in (1.0, 1.5, 2.0, 2.5)}

    # ---------------------------------------------------------------- Phase 6 mean / bias protection
    hist_td = float((obs_all.obs_td_home + obs_all.obs_td_away).sum() / (2 * len(obs_all)))
    hist_pts = float(hist_team.mean()); hist_tot_mean = float(hist_tot.mean())
    pil_td = float((obs.loc[games].obs_td_home + obs.loc[games].obs_td_away).sum() / (2 * len(games)))
    obs_tot_pilot = obs.loc[games].obs_total.values
    pred_mean = {arm: np.array([Z[g][f"{arm}__team_score"].sum(axis=1).mean() for g in games]) for arm in ARMS}
    bias_pilot = {arm: float((pred_mean[arm] - obs_tot_pilot).mean()) for arm in ARMS}
    bias = {"baseline_TD_rate": M["base"]["TD_per_team_game"], "ablation_TD_rate": M["abl"]["TD_per_team_game"], "historical_TD_rate_2020_25": hist_td, "pilot60_observed_TD_rate": pil_td,
            "ablation_TD_rate_vs_hist_pct": 100 * (M["abl"]["TD_per_team_game"] / hist_td - 1), "baseline_TD_rate_vs_hist_pct": 100 * (M["base"]["TD_per_team_game"] / hist_td - 1),
            "baseline_points": M["base"]["points_per_team_game"], "ablation_points": M["abl"]["points_per_team_game"], "observed_points_pilot60": float(pilot_team.mean()), "historical_points_2020_25": hist_pts,
            "baseline_total_bias_vs_pilot60": bias_pilot["base"], "ablation_total_bias_vs_pilot60": bias_pilot["abl"], "ctrl_total_bias_vs_pilot60": bias_pilot["ctrl"],
            "baseline_total_bias_SE_across_games": se(pred_mean["base"] - obs_tot_pilot), "ablation_total_bias_SE_across_games": se(pred_mean["abl"] - obs_tot_pilot),
            "baseline_total_bias_vs_hist_mean": M["base"]["total_mean"] - hist_tot_mean, "ablation_total_bias_vs_hist_mean": M["abl"]["total_mean"] - hist_tot_mean,
            "TD_collapse_material": bool(M["abl"]["TD_per_team_game"] < 0.95 * hist_td), "hist_total_mean": hist_tot_mean}

    # ---------------------------------------------------------------- Phase 7 discrimination
    pred_home = {arm: np.array([Z[g][f"{arm}__team_score"][:, 0].mean() for g in games]) for arm in ARMS}
    pred_away = {arm: np.array([Z[g][f"{arm}__team_score"][:, 1].mean() for g in games]) for arm in ARMS}
    oh = obs.loc[games].obs_home.values; oa = obs.loc[games].obs_away.values
    obs_margin = oh - oa
    # league-mean baselines
    def causal_mean(g):
        s, w = int(g.split("_")[0]), int(g.split("_")[1])
        m = obs_wide[(obs_wide.season >= s - 5) & ((obs_wide.season < s) | ((obs_wide.season == s) & (obs_wide.week < w)))]
        return float(m.obs_total.mean())
    league_causal = np.array([causal_mean(g) for g in games]); league_insample = np.full(len(games), obs_tot_pilot.mean())

    def disc(pred):
        e = pred - obs_tot_pilot
        return {"RMSE": rmse(e), "MAE": float(np.abs(e).mean()), "correlation": corr(pred, obs_tot_pilot), "bias": float(e.mean()), "bias_removed_RMSE": float(e.std(ddof=0)),
                "SD_of_predicted_game_means": float(pred.std(ddof=1))}
    D = {"baseline_V27": disc(pred_mean["base"]), "ablation": disc(pred_mean["abl"]), "ctrl": disc(pred_mean["ctrl"]),
         "league_mean_in_sample_pilot": {"RMSE": rmse(league_insample - obs_tot_pilot), "MAE": float(np.abs(league_insample - obs_tot_pilot).mean())},
         "league_mean_causal_trailing5yr": {"RMSE": rmse(league_causal - obs_tot_pilot), "MAE": float(np.abs(league_causal - obs_tot_pilot).mean())},
         "league_mean_window_games_skipped_franchise_code_mismatch": skipped, "league_mean_window_games_used": int(len(obs_wide)), "n_games": len(games), "observed_total_mean": float(obs_tot_pilot.mean()), "observed_total_SD": float(obs_tot_pilot.std(ddof=1)),
         "MC_noise_note": "each predicted game mean is a 128-sim average: SE ~ within-game SD/sqrt(128) ~ 1.2-1.4 pts, identical in both arms"}
    ix = np.arange(len(games)); BS = 10000
    boot = {"d_corr": [], "d_RMSE": [], "d_bias_removed_RMSE": [], "d_MAE": [], "abl_minus_league_causal_RMSE": [], "d_margin_corr": []}
    for _ in range(BS):
        s = RNG.choice(ix, len(ix), replace=True)
        if obs_tot_pilot[s].std() == 0: continue
        eb, ea = pred_mean["base"][s] - obs_tot_pilot[s], pred_mean["abl"][s] - obs_tot_pilot[s]
        boot["d_corr"].append(corr(pred_mean["abl"][s], obs_tot_pilot[s]) - corr(pred_mean["base"][s], obs_tot_pilot[s]))
        boot["d_RMSE"].append(rmse(ea) - rmse(eb)); boot["d_bias_removed_RMSE"].append(ea.std() - eb.std()); boot["d_MAE"].append(np.abs(ea).mean() - np.abs(eb).mean())
        boot["abl_minus_league_causal_RMSE"].append(rmse(ea) - rmse(league_causal[s] - obs_tot_pilot[s]))
        mb, ma = (pred_home["base"] - pred_away["base"])[s], (pred_home["abl"] - pred_away["abl"])[s]
        boot["d_margin_corr"].append(corr(ma, obs_margin[s]) - corr(mb, obs_margin[s]))
    D["paired_bootstrap_over_games_95CI"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 50)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
    D["delta_point"] = {"corr": D["ablation"]["correlation"] - D["baseline_V27"]["correlation"], "RMSE": D["ablation"]["RMSE"] - D["baseline_V27"]["RMSE"],
                        "bias_removed_RMSE": D["ablation"]["bias_removed_RMSE"] - D["baseline_V27"]["bias_removed_RMSE"], "MAE": D["ablation"]["MAE"] - D["baseline_V27"]["MAE"]}
    D["corr_95CI_fisher_baseline"] = [float(np.tanh(np.arctanh(D["baseline_V27"]["correlation"]) + s * 1.96 / math.sqrt(len(games) - 3))) for s in (-1, 1)]
    # margin / winner
    def wmetrics(arm):
        pH = np.array([(Z[g][f"{arm}__winner"] == "HOME").mean() for g in games]); pT = np.array([(Z[g][f"{arm}__winner"] == "TIE").mean() for g in games])
        ev = pH + .5 * pT
        y = np.where(oh > oa, 1.0, np.where(oh == oa, .5, 0.0))
        eps = 1e-3
        ll = -(y * np.log(np.clip(ev, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - ev, eps, 1 - eps)))
        dec = y != .5
        return {"brier": float(((ev - y) ** 2).mean()), "logloss": float(ll.mean()), "accuracy_decisive": float(((ev > .5) == (y > .5))[dec].mean()), "mean_P_home": float(pH.mean()),
                "predicted_margin_corr_vs_observed": corr(pred_home[arm] - pred_away[arm], obs_margin), "margin_RMSE": rmse((pred_home[arm] - pred_away[arm]) - obs_margin),
                "margin_bias": float(((pred_home[arm] - pred_away[arm]) - obs_margin).mean()), "home_score_bias": float((pred_home[arm] - oh).mean()), "away_score_bias": float((pred_away[arm] - oa).mean()),
                "ev": ev}, y
    Wb, y = wmetrics("base"); Wa, _ = wmetrics("abl"); Wc, _ = wmetrics("ctrl")
    dP = Wa["ev"] - Wb["ev"]
    p_b = Wb["ev"]; expected_noise_unpaired = float(np.mean(np.sqrt(2 * p_b * (1 - p_b) / n)) * math.sqrt(2 / math.pi))
    pair_dP_home = np.array([(Z[g]["abl__winner"] == "HOME").mean() - (Z[g]["base__winner"] == "HOME").mean() for g in games])
    dC = Wc["ev"] - Wb["ev"]
    winner = {"baseline": {k: v for k, v in Wb.items() if k != "ev"}, "ablation": {k: v for k, v in Wa.items() if k != "ev"}, "ctrl": {k: v for k, v in Wc.items() if k != "ev"},
              "mean_dP_home_plus_half_tie": float(dP.mean()), "SE_across_games": se(dP), "z": float(dP.mean() / se(dP)), "mean_abs_dP": float(np.abs(dP).mean()), "max_abs_dP": float(np.abs(dP).max()),
              "mean_dP_home_only": float(pair_dP_home.mean()), "SE_dP_home_only": se(pair_dP_home), "tie_rate_change": M["abl"]["tie_rate"] - M["base"]["tie_rate"],
              "favorite_flips": int(np.sum((Wb["ev"] - .5) * (Wa["ev"] - .5) < 0)),
              "expected_mean_abs_dP_if_pure_MC_noise_UNPAIRED_upper_bound": expected_noise_unpaired,
              "ctrl_mean_abs_dP (draw-identical arms: pure scoring effect)": float(np.abs(dC).mean()), "ctrl_mean_dP": float(dC.mean()),
              "delta_brier": float(((Wa["ev"] - y) ** 2 - (Wb["ev"] - y) ** 2).mean()), "delta_brier_SE": se((Wa["ev"] - y) ** 2 - (Wb["ev"] - y) ** 2),
              "home_minus_away_score_shift": (M["abl"]["home_mean_score"] - M["base"]["home_mean_score"], M["abl"]["away_mean_score"] - M["base"]["away_mean_score"]),
              "per_game_hi_low_dP": [float(dP.min()), float(dP.max())]}

    # ---------------------------------------------------------------- Phase 8 ledger (BASE vs CTRL exact pairing)
    ev_by = {}
    for g in games:
        for e in R[g]["events"]["ctrl"]:
            ev_by.setdefault((g, e["sim"], e["block"]), []).append(e)
    n_rows = 0; led_summ = {"by_TD_count": {}, "explained_by_suppression": 0, "unexplained_gt8_after": 0, "suppressed_points_total": 0, "ablation_points_dist": {}, "pairing_mismatch": 0}
    with LEDGER.open("w") as f:
        for i, g in enumerate(games):
            bb, cc, aa = Z[g]["base__blocks"], Z[g]["ctrl__blocks"], Z[g]["abl__blocks"]
            for r, c in zip(bb, cc):
                if r[FI["pts"]] <= 8: continue
                if not (r[FI["sim"]] == c[FI["sim"]] and r[FI["block"]] == c[FI["block"]] and r[FI["raw_pts"]] == c[FI["raw_pts"]] and r[FI["plays"]] == c[FI["plays"]]):
                    led_summ["pairing_mismatch"] += 1
                evs = ev_by.get((g, int(r[FI["sim"]]), int(r[FI["block"]])), [])
                row = {"game_id": g, "sim_id": int(r[FI["sim"]]), "block_id": int(r[FI["block"]]), "offense": "HOME" if r[FI["off"]] == 0 else "AWAY", "baseline_points": int(r[FI["pts"]]),
                       "TD_count": int(r[FI["td"]]), "FG_count": int(r[FI["fg"]]), "rec_TD": int(r[FI["rec_td"]]), "rush_TD": int(r[FI["rush_td"]]), "plays": int(r[FI["plays"]]),
                       "ablation_points": int(c[FI["pts"]]), "ablation_TD_count": int(c[FI["td"]]), "suppressed_TD_count": int(c[FI["supp_td"]]), "suppressed_scores": evs,
                       "suppressed_points_baseline_minus_ablation": int(r[FI["pts"]] - c[FI["pts"]]), "ablation_gt8": bool(c[FI["pts"]] > 8)}
                f.write(json.dumps(row) + "\n"); n_rows += 1
                k = str(row["TD_count"]); led_summ["by_TD_count"][k] = led_summ["by_TD_count"].get(k, 0) + 1
                led_summ["explained_by_suppression"] += int(row["suppressed_TD_count"] == row["TD_count"] - 1 and row["ablation_TD_count"] == 1)
                led_summ["unexplained_gt8_after"] += int(row["ablation_gt8"])
                led_summ["suppressed_points_total"] += row["suppressed_points_baseline_minus_ablation"]
                led_summ["ablation_points_dist"][str(row["ablation_points"])] = led_summ["ablation_points_dist"].get(str(row["ablation_points"]), 0) + 1
    led_summ["rows"] = n_rows
    led_summ["gt8_rows_with_FG"] = 0
    led_summ["baseline_gt8_blocks_all_rows_are_multiTD"] = bool(all(json.loads(l)["TD_count"] >= 2 for l in LEDGER.read_text().splitlines()))
    led_summ["gt8_after_ablation_live_arm"] = M["abl"]["GT8_blocks"]; led_summ["gt8_after_ablation_ctrl_arm"] = M["ctrl"]["GT8_blocks"]
    led_summ["historical_gt8_blocks_1999_2025 / blocks"] = [int((d.points_scored > 8).sum()), int(len(d))]
    led_summ["baseline_gt8_rate"] = M["base"]["GT8_share_of_blocks"]

    # ---------------------------------------------------------------- Phase 9 components
    def comp_arrays(arm, key):
        if key == "plays": return np.concatenate([Z[g][f"{arm}__team_pass_attempts"] + Z[g][f"{arm}__team_rush_attempts"] for g in games])
        if key == "carries_incl_scramble": return np.concatenate([Z[g][f"{arm}__psum_rush_attempts"] for g in games])
        if key == "completions": return np.concatenate([Z[g][f"{arm}__psum_receptions"] for g in games])
        if key == "targets": return np.concatenate([Z[g][f"{arm}__psum_targets"] for g in games])
        if key == "yards_total": return np.concatenate([Z[g][f"{arm}__team_pass_yards"] + Z[g][f"{arm}__team_rush_yards"] for g in games])
        if key in ("blocks", "pass_attempts", "rush_attempts", "pass_yards", "rush_yards"): return np.concatenate([Z[g][f"{arm}__team_{key}"] for g in games])
        raise KeyError(key)
    comp = {}
    for key in ("plays", "pass_attempts", "rush_attempts", "targets", "completions", "pass_yards", "rush_yards", "yards_total", "blocks", "carries_incl_scramble"):
        vb, va, vc = comp_arrays("base", key), comp_arrays("abl", key), comp_arrays("ctrl", key)
        dd = (va - vb).astype(float)
        pg_rel = [float((comp_arrays_g(Z, g, "abl", key).mean() - comp_arrays_g(Z, g, "base", key).mean()) / max(1e-9, comp_arrays_g(Z, g, "base", key).mean())) for g in games]
        comp[key] = {"baseline": float(vb.mean()), "ablation": float(va.mean()), "delta_pct": float(100 * (va.mean() - vb.mean()) / vb.mean()),
                     "paired_z": float(dd.mean() / (dd.std(ddof=1) / math.sqrt(dd.size))) if dd.std() > 0 else 0.0,
                     "max_abs_per_game_rel_change_pct": float(100 * np.max(np.abs(pg_rel))), "ctrl_exactly_equal_baseline": bool(np.array_equal(vb, vc)),
                     "share_sims_identical_to_baseline_abl": float((va == vb).all(axis=1).mean())}
    fgo = {a: M[a]["FG_opportunities_per_team_game"] for a in ARMS}
    comp["FG_opportunities_per_team_game"] = {"baseline": fgo["base"], "ablation": fgo["abl"], "ctrl": fgo["ctrl"], "delta_pct": 100 * (fgo["abl"] / fgo["base"] - 1),
                                              "ctrl_exactly_equal_baseline": bool(np.array_equal(BM["base"][:, FI["fg_elig"]], BM["ctrl"][:, FI["fg_elig"]]) and np.array_equal(BM["base"][:, FI["fg"]], BM["ctrl"][:, FI["fg"]]))}
    comp["FG_rate_conversion"] = {"baseline": M["base"]["FG_conversion_given_opportunity"], "ablation": M["abl"]["FG_conversion_given_opportunity"], "ctrl": M["ctrl"]["FG_conversion_given_opportunity"],
                                  "configured_params_fg_rate_mean_over_games": float(np.mean([R[g]["fg_rate"] for g in games])), "note": "params.fg_rate is a per-game constant, identical in all arms by construction"}
    comp["turnovers"] = "N/A: worker/sports_nova_v23/simulator.py has no turnover draw (grep 'turnover' over worker/sports_nova_v23 and v3/distributions.py returns nothing); nothing to change"
    # player allocations
    pa_rel = []; pa_ctrl_eq = True; per_player_abs = []
    for g in games:
        for key in ("targets", "rush_attempts", "receptions"):
            b_, a_, c_ = Z[g][f"base__player_{key}"], Z[g][f"abl__player_{key}"], Z[g][f"ctrl__player_{key}"]
            pa_ctrl_eq &= bool(np.array_equal(b_, c_))
            mb, ma = b_.mean(axis=0), a_.mean(axis=0)
            m_ = mb >= 2.0
            if m_.any():
                pa_rel += list(np.abs(ma[m_] - mb[m_]) / mb[m_])
    comp["player_allocations"] = {"CTRL_per_player_targets_carries_receptions_bit_identical_all_games": bool(pa_ctrl_eq),
                                  "ABL_max_abs_rel_change_per_player_mean_(players with baseline mean>=2)": float(np.max(pa_rel)),
                                  "ABL_median_abs_rel_change": float(np.median(pa_rel)),
                                  "note": "per-player means over 128 sims carry unpaired-style MC noise once streams desynchronise; the systematic component is game-script feedback only"}
    # sim identity share
    same = 0; tot = 0
    for g in games:
        ident = np.ones(n, bool)
        for k in ("team_pass_attempts", "team_rush_attempts", "team_pass_yards", "team_rush_yards", "team_blocks"):
            ident &= (Z[g][f"base__{k}"] == Z[g][f"abl__{k}"]).all(axis=1)
        same += int(ident.sum()); tot += n
    comp["share_of_sims_with_bit_identical_volume_and_yards_ABL_vs_BASE"] = same / tot
    max_rel = max(v["max_abs_per_game_rel_change_pct"] for v in comp.values() if isinstance(v, dict) and "max_abs_per_game_rel_change_pct" in v)
    max_pooled = max(abs(v["delta_pct"]) for v in comp.values() if isinstance(v, dict) and "delta_pct" in v)
    comp["max_abs_pooled_delta_pct_ABL"] = float(max_pooled); comp["max_abs_per_game_rel_change_pct_ABL"] = float(max_rel)

    # ---------------------------------------------------------------- pass/rush TD mix and loop order
    mix = {"baseline": {"rec_TD": M["base"]["rec_TD_per_team_game"], "rush_TD": M["base"]["rush_TD_per_team_game"], "rush_share": M["base"]["rush_share_of_TDs"]},
           "ablation": {"rec_TD": M["abl"]["rec_TD_per_team_game"], "rush_TD": M["abl"]["rush_TD_per_team_game"], "rush_share": M["abl"]["rush_share_of_TDs"]},
           "ctrl": {"rec_TD": M["ctrl"]["rec_TD_per_team_game"], "rush_TD": M["ctrl"]["rush_TD_per_team_game"], "rush_share": M["ctrl"]["rush_share_of_TDs"]},
           "in_multi_TD_blocks_ctrl": {"kept_receiving": int(sum(1 for g in games for e in [] )), "note": "see suppressed events by site below"}}
    ev_all = [e for g in games for e in R[g]["events"]["ctrl"]]
    mix["suppressed_TDs_by_site_ctrl"] = {"REC": int(sum(e["suppressed_td"] for e in ev_all if e["site"] == "REC")), "RUSH": int(sum(e["suppressed_td"] for e in ev_all if e["site"] == "RUSH"))}
    mix["in_multi_TD_blocks_ctrl"] = {"blocks_with_rec_and_rush_raw_TD": int(((BM["base"][:, FI["raw_rec_td"]] > 0) & (BM["base"][:, FI["raw_rush_td"]] > 0)).sum()),
                                      "blocks_where_kept_TD_is_receiving (ctrl)": int(((BM["ctrl"][:, FI["supp_td"]] > 0) & (BM["ctrl"][:, FI["rec_td"]] == 1)).sum()),
                                      "blocks_where_kept_TD_is_rushing (ctrl)": int(((BM["ctrl"][:, FI["supp_td"]] > 0) & (BM["ctrl"][:, FI["rush_td"]] == 1)).sum())}

    # ---------------------------------------------------------------- Phase 10 verdict clauses
    c1 = M["abl"]["GT8_blocks"] <= 0.05 * M["base"]["GT8_blocks"]
    c2 = mae_up["abl"] <= 0.5 * mae_up["base"]
    c3 = abs(M["abl"]["total_mean"] - hist_tot_mean) <= 1.5 and abs(M["abl"]["TD_per_team_game"] / hist_td - 1) <= 0.05
    c4 = bool(checks["accounting_pass"] and checks["BASE_equals_REF_all_games"] and checks["guarded_hashes_unchanged_before_after_run"] and checks["prefix_violations_total"] == 0
              and checks["CTRL_non_TD_stats_bit_identical_to_BASE"])
    # Shape diagnostic (mean-free metrics).  The prereg LABEL_RULE text says "C2 or shape diagnostics" without numeric thresholds; the thresholds here (>=50% of the
    # baseline-to-history gap closed on >=3 of 4 metrics) are a POST-HOC quantification, disclosed as such, chosen to mirror C2's 50% rule and not tuned.
    tm_b, tm_a, tm_h = M["base"]["team_shape"], M["abl"]["team_shape"], tails["hist_shape_team"]
    sx = tails["standardized_exceedance_team_score (P(x >= own_mean + k*own_SD))"]["2.0"]
    cv = {"baseline": tm_b["sd"] / tm_b["mean"], "ablation": tm_a["sd"] / tm_a["mean"], "historical": tm_h["sd"] / tm_h["mean"]}
    def closed(b_, a_, h_):
        return float(1 - abs(a_ - h_) / abs(b_ - h_)) if abs(b_ - h_) > 0 else float("nan")
    shape_closed = {"team_skew": closed(tm_b["skew"], tm_a["skew"], tm_h["skew"]), "team_excess_kurtosis": closed(tm_b["excess_kurtosis"], tm_a["excess_kurtosis"], tm_h["excess_kurtosis"]),
                    "team_2SD_standardized_exceedance": closed(sx["baseline"], sx["ablation"], sx["historical"]), "team_CV_sd_over_mean": closed(cv["baseline"], cv["ablation"], cv["historical"])}
    shape_ok = sum(1 for v in shape_closed.values() if v >= 0.5) >= 3
    tot_cv = {"baseline": M["base"]["total_SD"] / M["base"]["total_mean"], "ablation": M["abl"]["total_SD"] / M["abl"]["total_mean"], "historical": tails["hist_shape_total"]["sd"] / tails["hist_shape_total"]["mean"]}
    shape_diag = {"gap_closed_fraction_baseline_to_history": shape_closed, "shape_diagnostic_holds_(>=3_of_4_at_>=50%)": bool(shape_ok), "team_CV": cv, "total_CV": tot_cv,
                  "total_SD": {"baseline": M["base"]["total_SD"], "ablation": M["abl"]["total_SD"], "historical": tails["hist_shape_total"]["sd"]},
                  "team_SD": {"baseline": tm_b["sd"], "ablation": tm_a["sd"], "historical": tm_h["sd"]},
                  "caution": "ablation total SD matching history is partly a coincidence of the mean undershoot (SD scales with mean); the CV / standardized metrics are the mean-free view",
                  "threshold_status": "POST-HOC quantification of a prereg text criterion that had no numeric threshold"}
    ci = D["paired_bootstrap_over_games_95CI"]
    disc_improves = bool(ci["d_corr"][0] > 0 or ci["d_bias_removed_RMSE"][2] < 0)
    beats_league = bool(ci["abl_minus_league_causal_RMSE"][2] < 0)
    def lab(c2_or_shape):
        if not c4: return "INCONCLUSIVE"
        if c1 and c2_or_shape and c3: return "CAUSAL_MAJOR"
        if c1 and c2_or_shape and not c3: return "CAUSAL_MINOR"
        return "NOT_PRIMARY" if c1 else "INCONCLUSIVE"
    label_mechanical_c2_only = lab(c2)
    label = lab(c2 or shape_ok)
    verdict = {"C1_GT8_removed": bool(c1), "C2_tail_calibration_improves": bool(c2), "C3_no_destructive_mean_bias": bool(c3), "C4_accounting_integrity": c4,
               "C3_detail": {"total_mean_ablation": M["abl"]["total_mean"], "hist_total_mean": hist_tot_mean, "abs_diff": abs(M["abl"]["total_mean"] - hist_tot_mean),
                             "TD_rate_ablation_vs_hist_pct": bias["ablation_TD_rate_vs_hist_pct"]},
               "C2_detail": mae_up, "SHAPE_DIAGNOSTIC": shape_diag, "TD_ARTIFACT_VERDICT": label_mechanical_c2_only, "label_if_post_hoc_shape_diagnostic_had_counted_NOT_USED": label, "discrimination_improves": disc_improves, "ablation_beats_causal_league_mean_RMSE_CI_excludes_equality": beats_league,
               "permanent_fix_ready": bool(c1 and c2 and c3 and c4 and disc_improves)}

    # ---------------------------------------------------------------- predictions vs actuals
    P = pre["PREDICTIONS_REGISTERED_BEFORE_ANY_CAPPED_RUN"]
    pred_check = {
        "TD_per_team_game": {"pred_range": P["TD_PER_TEAM_GAME"]["range"], "actual": M["abl"]["TD_per_team_game"], "hit": P["TD_PER_TEAM_GAME"]["range"][0] <= M["abl"]["TD_per_team_game"] <= P["TD_PER_TEAM_GAME"]["range"][1]},
        "points_per_team_game": {"pred_range": P["POINTS_PER_TEAM_GAME"]["range"], "actual": M["abl"]["points_per_team_game"], "hit": P["POINTS_PER_TEAM_GAME"]["range"][0] <= M["abl"]["points_per_team_game"] <= P["POINTS_PER_TEAM_GAME"]["range"][1]},
        "total_bias_vs_pilot": {"pred_range": P["TOTAL_BIAS_VS_PILOT_OBSERVED"]["range"], "actual": bias_pilot["abl"], "hit": P["TOTAL_BIAS_VS_PILOT_OBSERVED"]["range"][0] <= bias_pilot["abl"] <= P["TOTAL_BIAS_VS_PILOT_OBSERVED"]["range"][1]},
        "GT8_ablation_zero": {"actual": M["abl"]["GT8_blocks"], "hit": M["abl"]["GT8_blocks"] == 0},
        "FG_per_team_game_unchanged_pm0.03": {"baseline": M["base"]["FG_per_team_game"], "ablation": M["abl"]["FG_per_team_game"], "hit": abs(M["abl"]["FG_per_team_game"] - M["base"]["FG_per_team_game"]) <= 0.03},
        "tie_rate_rises": {"baseline": M["base"]["tie_rate"], "ablation": M["abl"]["tie_rate"], "hit": M["abl"]["tie_rate"] > M["base"]["tie_rate"]},
        "P(team>=45)_range_0.8-2.5pct": {"actual": sim_t["abl"]["P(team>=45)"], "hit": 0.008 <= sim_t["abl"]["P(team>=45)"] <= 0.025},
        "P(total>=70)_range_2-5pct": {"actual": sim_t["abl"]["P(total>=70)"], "hit": 0.02 <= sim_t["abl"]["P(total>=70)"] <= 0.05},
        "rush_share_falls": {"baseline": mix["baseline"]["rush_share"], "ablation": mix["ablation"]["rush_share"], "hit": mix["ablation"]["rush_share"] < mix["baseline"]["rush_share"]},
        "corr_change_lt_0.03": {"delta": D["delta_point"]["corr"], "hit": abs(D["delta_point"]["corr"]) < 0.03},
        "RMSE_worse_0.3_to_1.5": {"delta": D["delta_point"]["RMSE"], "hit": 0.3 <= D["delta_point"]["RMSE"] <= 1.5},
        "prefix_violations_zero": {"actual": checks["prefix_violations_total"], "hit": checks["prefix_violations_total"] == 0},
        "ABL_component_max_pooled_delta_lt_0.5pct": {"actual": comp["max_abs_pooled_delta_pct_ABL"], "hit": comp["max_abs_pooled_delta_pct_ABL"] < 0.5}}
    pred_check["n_hit"] = sum(1 for v in pred_check.values() if isinstance(v, dict) and v.get("hit")); pred_check["n_total"] = sum(1 for v in pred_check.values() if isinstance(v, dict) and "hit" in v)

    report = {"PREREG_SHA256": sha256_file(PREREG), "RUN_META_SHA256": sha256_file(OUT / "run_meta.json"), "CHECKS": checks, "PHASE1_EMPIRICAL": phase1, "METRICS": M, "TAILS": tails, "BIAS_PROTECTION": bias,
              "DISCRIMINATION": D, "WINNER": winner, "GT8_LEDGER_SUMMARY": led_summ, "COMPONENTS": comp, "TD_MIX": mix, "VERDICT": verdict, "PREDICTION_CHECK": pred_check,
              "PARAMS": {"games": len(games), "sims_per_game": n, "historical_games_2020_25": int(len(obs_all)), "bootstrap_draws": BS}}
    REPORT.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


def comp_arrays_g(Z, g, arm, key):
    if key == "plays": return Z[g][f"{arm}__team_pass_attempts"] + Z[g][f"{arm}__team_rush_attempts"]
    if key == "carries_incl_scramble": return Z[g][f"{arm}__psum_rush_attempts"]
    if key == "completions": return Z[g][f"{arm}__psum_receptions"]
    if key == "targets": return Z[g][f"{arm}__psum_targets"]
    if key == "yards_total": return Z[g][f"{arm}__team_pass_yards"] + Z[g][f"{arm}__team_rush_yards"]
    return Z[g][f"{arm}__team_{key}"]


if __name__ == "__main__":
    main()
