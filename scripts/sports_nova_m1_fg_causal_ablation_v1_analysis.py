"""Analysis of the FG-only causal ablation (reads arrays written by ..._ablation_v1.py)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sports_nova_fg_causal_estimator_v1 import load_drives, sha256_file  # noqa: E402

OUT = ROOT / "data" / "sports_nova_v3" / "FG_CAUSAL_ABLATION_V1"
ARR = OUT / "arrays"
TOT_LADDER = [33.5, 37.5, 41.5, 44.5, 47.5, 51.5, 55.5]
TEAM_LADDER = [13.5, 17.5, 20.5, 23.5, 27.5, 30.5]
MARGIN_LADDER = [0.5, 3.5, 6.5, 10.5, 13.5]     # P(home margin >= x) and P(away margin >= x)
UNRELATED = ["team_blocks", "plays", "team_pass_attempts", "team_rush_attempts", "team_pass_yards",
             "team_rush_yards", "psum_targets", "psum_receptions", "carries_incl_scramble", "td_trials"]


def observed_scores(d: pd.DataFrame, games: list[str]) -> pd.DataFrame:
    """Regulation final scores rebuilt from drive-level score changes (offense + defensive scores)."""
    rows = []
    for g in games:
        x = d[d.game_id == g].sort_values("source_drive_number")
        prev_end = None; ot_start = None
        for i, r in enumerate(x.itertuples()):
            if prev_end is not None and r.start_game_seconds_remaining > prev_end + 60:
                ot_start = i; break
            prev_end = r.end_game_seconds_remaining
        reg = x if ot_start is None else x.iloc[:ot_start]
        parts = g.split("_"); away, home = parts[2], parts[3]
        sc = {home: 0.0, away: 0.0}; fg = {home: 0, away: 0}; td = {home: 0, away: 0}
        for r in reg.itertuples():
            off = r.offense_team; de = away if off == home else home
            pts_off = float(r.points_scored)
            delta = float(r.end_score_diff - r.start_score_diff)
            sc[off] += pts_off; sc[de] += pts_off - delta
            fg[off] += int(r.result == "FIELD_GOAL"); td[off] += int(r.touchdowns)
        rows.append({"game": g, "home": home, "away": away, "obs_home": sc[home], "obs_away": sc[away],
                     "obs_total": sc[home] + sc[away], "ot": ot_start is not None,
                     "obs_fg_home": fg[home], "obs_fg_away": fg[away], "obs_td_home": td[home], "obs_td_away": td[away]})
    return pd.DataFrame(rows)


def ok_recon(d, obs, sample=None):
    """Sanity: reconstructed regulation home-away diff == last regulation drive's end_score_diff."""
    bad = 0
    for r in obs.itertuples():
        x = d[d.game_id == r.game].sort_values("source_drive_number")
        # Only valid for non-OT games (OT drives change the final diff).
        if r.ot:
            continue
        last = x.iloc[-1]
        diff_off = float(last.end_score_diff)
        diff_home = diff_off if last.offense_team == r.home else -diff_off
        bad += int(abs(diff_home - (r.obs_home - r.obs_away)) > 1e-9)
    return bad


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def game_summary(z, arm, n):
    h, a = z[f"{arm}__team_score"][:, 0], z[f"{arm}__team_score"][:, 1]
    tot = h + a
    w = z[f"{arm}__winner"]
    pH, pA, pT = float((w == "HOME").mean()), float((w == "AWAY").mean()), float((w == "TIE").mean())
    return {"mean_home_score": float(h.mean()), "mean_away_score": float(a.mean()), "mean_total": float(tot.mean()),
            "home_win_prob": pH, "away_win_prob": pA, "tie_prob": pT,
            "fg_made_home": float(z[f"{arm}__rec_fg"][:, 0].mean()), "fg_made_away": float(z[f"{arm}__rec_fg"][:, 1].mean()),
            "td_count": float(z[f"{arm}__rec_td"].sum(axis=1).mean()),
            "score_sd_home": float(h.std(ddof=1)), "score_sd_away": float(a.std(ddof=1)), "score_sd_total": float(tot.std(ddof=1)),
            "total_quantiles": {q: float(np.quantile(tot, q / 100)) for q in (5, 25, 50, 75, 95)}}


def ladders(z, arm):
    h, a = z[f"{arm}__team_score"][:, 0], z[f"{arm}__team_score"][:, 1]
    tot = h + a
    return {"total_over": {str(x): float((tot > x).mean()) for x in TOT_LADDER},
            "home_team_over": {str(x): float((h > x).mean()) for x in TEAM_LADDER},
            "away_team_over": {str(x): float((a > x).mean()) for x in TEAM_LADDER},
            "home_margin_ge": {str(x): float(((h - a) >= x).mean()) for x in MARGIN_LADDER},
            "away_margin_ge": {str(x): float(((a - h) >= x).mean()) for x in MARGIN_LADDER}}


def per_team_game(z, arm, key):
    v = z[f"{arm}__{key}"]
    return v


def component(z, arm, name):
    if name == "plays":
        return z[f"{arm}__team_pass_attempts"] + z[f"{arm}__team_rush_attempts"]
    if name == "carries_incl_scramble":
        return z[f"{arm}__psum_rush_attempts"]
    if name == "td_trials":
        return z[f"{arm}__psum_receptions"] + z[f"{arm}__psum_rush_attempts"]
    return z[f"{arm}__{name}"]


def main() -> None:
    meta = json.loads((OUT / "run_meta.json").read_text())
    pre = json.loads((OUT / "FG_CAUSAL_ABLATION_V1_PREREG.json").read_text())
    games = [r["game"] for r in meta["results"]]
    d = load_drives()
    obs = observed_scores(d, games).set_index("game")
    all_games = d[d.season.between(2020, 2025)].game_id.unique().tolist()
    obs_all = observed_scores(d, all_games)
    recon_bad = ok_recon(d, obs_all)

    per_game = {}
    agg = {a: {"fg": [], "pts_team": [], "blocks": [], "elig": [], "plays": []} for a in ("base", "abl")}
    comp = {k: {"base": [], "abl": []} for k in UNRELATED}
    comp_diff = {k: [] for k in UNRELATED}          # per-sim paired differences pooled over games
    identical_sims = 0; total_sims = 0
    ident_keys = ["team_pass_attempts", "team_rush_attempts", "team_pass_yards", "team_rush_yards", "team_blocks"]
    acct_tot = {"base": {}, "abl": {}}
    pool = {a: {"tot": [], "h": [], "a": [], "win": []} for a in ("base", "abl")}
    gt8_all = []
    z_stat = {a: {"tot": [], "h": [], "a": []} for a in ("base", "abl")}
    for r in meta["results"]:
        g = r["game"]; n = r["n_sims"]
        z = np.load(ARR / f"{g}.npz")
        o = obs.loc[g]
        rec = {}
        for arm in ("base", "abl"):
            s = game_summary(z, arm, n); s["ladders"] = ladders(z, arm)
            rec[arm] = s
            agg[arm]["fg"].append(z[f"{arm}__rec_fg"].mean(axis=0))
            agg[arm]["pts_team"].append(z[f"{arm}__team_score"].mean(axis=0))
            agg[arm]["blocks"].append(z[f"{arm}__rec_nblk"].mean(axis=0))
            agg[arm]["elig"].append(z[f"{arm}__rec_elig"].mean(axis=0))
            agg[arm]["plays"].append(z[f"{arm}__rec_plays"].mean(axis=0))
            h, a = z[f"{arm}__team_score"][:, 0], z[f"{arm}__team_score"][:, 1]
            pool[arm]["tot"].append(h + a); pool[arm]["h"].append(h); pool[arm]["a"].append(a)
            pool[arm]["win"].append(z[f"{arm}__winner"])
            z_stat[arm]["tot"].append((o.obs_total - (h + a).mean()) / (h + a).std(ddof=1))
            z_stat[arm]["h"].append((o.obs_home - h.mean()) / h.std(ddof=1))
            z_stat[arm]["a"].append((o.obs_away - a.mean()) / a.std(ddof=1))
        for k in UNRELATED:
            b, ab = component(z, "base", k), component(z, "abl", k)
            comp[k]["base"].append(b.mean(axis=0)); comp[k]["abl"].append(ab.mean(axis=0))
            comp_diff[k].append((ab - b).sum(axis=1) / 2.0)          # per-sim mean of the two team values
        same = np.ones(n, bool)
        for k in ident_keys:
            same &= (z[f"base__{k}"] == z[f"abl__{k}"]).all(axis=1)
        identical_sims += int(same.sum()); total_sims += n
        delta = {"delta_FG": float((rec["abl"]["fg_made_home"] + rec["abl"]["fg_made_away"]) - (rec["base"]["fg_made_home"] + rec["base"]["fg_made_away"])),
                 "delta_home_score": rec["abl"]["mean_home_score"] - rec["base"]["mean_home_score"],
                 "delta_away_score": rec["abl"]["mean_away_score"] - rec["base"]["mean_away_score"],
                 "delta_total": rec["abl"]["mean_total"] - rec["base"]["mean_total"],
                 "delta_home_win_prob": rec["abl"]["home_win_prob"] - rec["base"]["home_win_prob"],
                 "delta_tie_prob": rec["abl"]["tie_prob"] - rec["base"]["tie_prob"],
                 "delta_score_SD_total": rec["abl"]["score_sd_total"] - rec["base"]["score_sd_total"]}
        tot_diff = (z["abl__team_score"].sum(axis=1) - z["base__team_score"].sum(axis=1)).astype(float)
        delta["delta_total_paired_SE"] = float(tot_diff.std(ddof=1) / math.sqrt(n))
        delta["ladder_probability_changes"] = {
            grp: {k: rec["abl"]["ladders"][grp][k] - rec["base"]["ladders"][grp][k] for k in rec["abl"]["ladders"][grp]}
            for grp in rec["abl"]["ladders"]}
        for arm in ("base", "abl"):
            for kk, v in r[f"acct_{arm}"].items():
                acct_tot[arm][kk] = acct_tot[arm].get(kk, 0) + int(v)
            for row in r[f"gt8_{arm}"]:
                gt8_all.append({"arm": arm, "game": g, "simulation_id": row["sim"], "block_id": row["block"],
                                "offense": row["offense"], "score_value": row["pts"],
                                "scoring_components": {"touchdowns": row["tds"], "td_points": 6 * row["tds"],
                                                       "implied_pat_points": row["implied_pats"],
                                                       "pass_td_players": row["pass_td_players"],
                                                       "rush_td_players": row["rush_td_players"],
                                                       "max_single_player_tds": row["max_player_tds"], "fg": 0},
                                "plays": row["plays"],
                                "code_path": "worker/sports_nova_v23/simulator.py:78-110 (per-receiver + per-carrier TD loops each add 6*td+PAT to block_points; no once-per-possession cap)",
                                "classification": "MULTIPLE_SCORE_ARTIFACT" if row["tds"] >= 2 else "UNKNOWN"})
        per_game[g] = {"observed": {k: (float(v) if not isinstance(v, (bool, np.bool_)) else bool(v)) for k, v in o.items() if k not in ("home", "away")},
                       "baseline": {k: v for k, v in rec["base"].items() if k != "ladders"},
                       "ablation": {k: v for k, v in rec["abl"].items() if k != "ladders"}, "paired": delta}
    (OUT / "GT8_BLOCK_LEDGER.jsonl").write_text("\n".join(json.dumps(x) for x in gt8_all))
    n_games = len(games)

    def tg(arm, key):   # mean per team-game across games
        return float(np.mean(np.concatenate([np.asarray(x).ravel() for x in agg[arm][key]])))

    hist = pre["HISTORICAL_REFERENCE_2020_2025"]
    obs_pilot = obs.loc[games]
    obs_fg_pilot = float((obs_pilot.obs_fg_home + obs_pilot.obs_fg_away).sum() / (2 * n_games))
    obs_td_pilot = float((obs_pilot.obs_td_home + obs_pilot.obs_td_away).sum() / (2 * n_games))
    obs_pts_pilot = float((obs_pilot.obs_home + obs_pilot.obs_away).sum() / (2 * n_games))

    def wmetrics(arm):
        pH = np.array([per_game[g][{"base": "baseline", "abl": "ablation"}[arm]]["home_win_prob"] for g in games])
        pT = np.array([per_game[g][{"base": "baseline", "abl": "ablation"}[arm]]["tie_prob"] for g in games])
        ev = pH + 0.5 * pT
        y = np.array([1.0 if obs.loc[g].obs_home > obs.loc[g].obs_away else (0.5 if obs.loc[g].obs_home == obs.loc[g].obs_away else 0.0) for g in games])
        brier = (ev - y) ** 2
        eps = 1e-3
        ll = -(y * np.log(np.clip(ev, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - ev, eps, 1 - eps)))
        dec = y != 0.5
        acc = float(((ev > .5) == (y > .5))[dec].mean())
        return ev, y, brier, ll, acc

    ev_b, y, br_b, ll_b, acc_b = wmetrics("base")
    ev_a, _, br_a, ll_a, acc_a = wmetrics("abl")

    def tot_err(arm):
        key = {"base": "baseline", "abl": "ablation"}[arm]
        pred = np.array([per_game[g][key]["mean_total"] for g in games])
        ob = np.array([per_game[g]["observed"]["obs_total"] for g in games])
        e = pred - ob
        ph = np.array([per_game[g][key]["mean_home_score"] for g in games]); pa = np.array([per_game[g][key]["mean_away_score"] for g in games])
        oh = np.array([per_game[g]["observed"]["obs_home"] for g in games]); oa = np.array([per_game[g]["observed"]["obs_away"] for g in games])
        te = np.concatenate([ph - oh, pa - oa])
        return {"total_bias": float(e.mean()), "total_bias_SE": se(e), "total_MAE": float(np.abs(e).mean()), "total_RMSE": float(np.sqrt((e ** 2).mean())),
                "team_total_bias": float(te.mean()), "team_total_MAE": float(np.abs(te).mean()), "team_total_RMSE": float(np.sqrt((te ** 2).mean())),
                "home_bias": float((ph - oh).mean()), "away_bias": float((pa - oa).mean()), "obs_mean_total": float(ob.mean()),
                "obs_regulation_ties_or_ot": int(sum(per_game[g]["observed"]["ot"] for g in games))}

    te_b, te_a = tot_err("base"), tot_err("abl")
    d_tot = np.array([per_game[g]["paired"]["delta_total"] for g in games])
    d_ph = np.array([per_game[g]["paired"]["delta_home_win_prob"] for g in games])
    flips = int(sum(((per_game[g]["baseline"]["home_win_prob"] + .5 * per_game[g]["baseline"]["tie_prob"]) - .5) *
                    ((per_game[g]["ablation"]["home_win_prob"] + .5 * per_game[g]["ablation"]["tie_prob"]) - .5) < 0 for g in games))

    # ---- pooled distribution and overcorrection ----
    def pooled(arm):
        t = np.concatenate(pool[arm]["tot"]); hh = np.concatenate(pool[arm]["h"]); aa = np.concatenate(pool[arm]["a"]); w = np.concatenate(pool[arm]["win"])
        team = np.concatenate([hh, aa])
        return {"total_mean": float(t.mean()), "total_sd_pooled": float(t.std(ddof=1)), "team_score_sd_pooled": float(team.std(ddof=1)),
                "mean_within_game_total_sd": float(np.mean([per_game[g][{"base": "baseline", "abl": "ablation"}[arm]]["score_sd_total"] for g in games])),
                "mean_within_game_team_sd": float(np.mean([[per_game[g][{"base": "baseline", "abl": "ablation"}[arm]]["score_sd_home"], per_game[g][{"base": "baseline", "abl": "ablation"}[arm]]["score_sd_away"]] for g in games])),
                "total_q": {q: float(np.quantile(t, q / 100)) for q in (1, 5, 25, 50, 75, 95, 99)},
                "P_total_ge_70": float((t >= 70).mean()), "P_team_ge_45": float((team >= 45).mean()), "P_team_le_3": float((team <= 3).mean()),
                "tie_rate": float((w == "TIE").mean())}
    pb, pa_ = pooled("base"), pooled("abl")
    hist_tot = obs_all.obs_total.values; hist_team = np.concatenate([obs_all.obs_home.values, obs_all.obs_away.values])
    hist_dist = {"n_games": int(len(obs_all)), "total_mean": float(hist_tot.mean()), "total_sd": float(hist_tot.std(ddof=1)), "team_score_mean": float(hist_team.mean()),
                 "team_score_sd": float(hist_team.std(ddof=1)), "total_q": {q: float(np.quantile(hist_tot, q / 100)) for q in (1, 5, 25, 50, 75, 95, 99)},
                 "P_total_ge_70": float((hist_tot >= 70).mean()), "P_team_ge_45": float((hist_team >= 45).mean()), "P_team_le_3": float((hist_team <= 3).mean()),
                 "regulation_tie_rate": float((obs_all.obs_home == obs_all.obs_away).mean()), "ot_games_detected": int(obs_all.ot.sum()),
                 "reconstruction_check_bad_games": recon_bad}
    zvar = {arm: {k: float(np.var(v, ddof=1)) for k, v in z_stat[arm].items()} for arm in z_stat}
    zmean = {arm: {k: float(np.mean(v)) for k, v in z_stat[arm].items()} for arm in z_stat}

    comp_report = {}
    for k in UNRELATED:
        b = np.concatenate([x.ravel() for x in comp[k]["base"]]); a = np.concatenate([x.ravel() for x in comp[k]["abl"]])
        dd = np.concatenate(comp_diff[k]); n_all = len(dd)
        comp_report[k] = {"baseline": float(b.mean()), "ablation": float(a.mean()), "delta": float(a.mean() - b.mean()),
                          "delta_pct": float(100 * (a.mean() - b.mean()) / b.mean()), "paired_sim_SE": float(dd.std(ddof=1) / math.sqrt(n_all)),
                          "z": float(dd.mean() / (dd.std(ddof=1) / math.sqrt(n_all))) if dd.std() > 0 else 0.0}
    unrelated_ok = all(abs(v["delta_pct"]) < 1.0 for v in comp_report.values())

    fg_b, fg_a = tg("base", "fg"), tg("abl", "fg")
    gap = obs_fg_pilot - fg_b
    gap_hist = hist["made_fg_per_team_game"] - fg_b
    gt8_by_arm = {arm: [x for x in gt8_all if x["arm"] == arm] for arm in ("base", "abl")}
    blocks_total = {arm: float(np.sum([np.sum(agg[arm]["blocks"][i]) for i in range(n_games)]) * 128) for arm in ("base", "abl")}
    multi_td_blocks = {}
    for arm in ("base", "abl"):
        tot = 0; nb = 0
        for g in games:
            z = np.load(ARR / f"{g}.npz"); tot += int(z[f"{arm}__rec_multi_td"].sum()); nb += int(z[f"{arm}__rec_nblk"].sum())
        multi_td_blocks[arm] = {"blocks_with_ge2_TD": tot, "all_blocks": nb, "rate": tot / nb}
    hist_gt8 = int((d.points_scored > 8).sum())

    report = {
        "PREREG_SHA256": sha256_file(OUT / "FG_CAUSAL_ABLATION_V1_PREREG.json"),
        "RUN_META_SHA256": sha256_file(OUT / "run_meta.json"),
        "COHORT": {"games": n_games, "sims_per_game": 128, "engine": "worker.sports_nova_v23.simulator (V25/V26 scoring path)"},
        "FG_ESTIMATOR": {"per_game_rate_min_median_max": [min(v["rate"] for v in pre["PER_GAME_ESTIMATES"].values()),
                                                         float(np.median([v["rate"] for v in pre["PER_GAME_ESTIMATES"].values()])),
                                                         max(v["rate"] for v in pre["PER_GAME_ESTIMATES"].values())],
                         "n_eligible_min_max": [min(v["n_eligible"] for v in pre["PER_GAME_ESTIMATES"].values()), max(v["n_eligible"] for v in pre["PER_GAME_ESTIMATES"].values())],
                         "leakage_pass": all(v["leakage_pass"] for v in pre["PER_GAME_ESTIMATES"].values())},
        "OBSERVED_PILOT": {"FG_per_team_game": obs_fg_pilot, "TD_per_team_game": obs_td_pilot, "points_per_team_game": obs_pts_pilot, "ot_games": int(obs_pilot.ot.sum())},
        "HISTORICAL_2020_2025": hist, "HISTORICAL_DIST": hist_dist,
        "BASELINE": {"FG_per_team_game": fg_b, "points_per_team_game": tg("base", "pts_team"), "points_per_game": 2 * tg("base", "pts_team"),
                     "blocks_per_team_game": tg("base", "blocks"), "eligible_D3_blocks_per_team_game": tg("base", "elig"), "plays_in_blocks_per_team_game": tg("base", "plays"),
                     "errors": te_b, "winner": {"brier": float(br_b.mean()), "brier_SE": se(br_b), "logloss": float(ll_b.mean()), "accuracy_decisive": acc_b},
                     "pooled": pb, "z_variance_obs_vs_sim": zvar["base"], "z_mean": zmean["base"]},
        "ABLATION": {"FG_per_team_game": fg_a, "points_per_team_game": tg("abl", "pts_team"), "points_per_game": 2 * tg("abl", "pts_team"),
                     "blocks_per_team_game": tg("abl", "blocks"), "eligible_D3_blocks_per_team_game": tg("abl", "elig"), "plays_in_blocks_per_team_game": tg("abl", "plays"),
                     "errors": te_a, "winner": {"brier": float(br_a.mean()), "brier_SE": se(br_a), "logloss": float(ll_a.mean()), "accuracy_decisive": acc_a},
                     "pooled": pa_, "z_variance_obs_vs_sim": zvar["abl"], "z_mean": zmean["abl"]},
        "PAIRED": {"delta_FG_per_team_game": fg_a - fg_b, "delta_points_per_team_game": tg("abl", "pts_team") - tg("base", "pts_team"),
                   "delta_total_mean": float(d_tot.mean()), "delta_total_SE_across_games": se(d_tot),
                   "delta_total_error_bias": te_a["total_bias"] - te_b["total_bias"],
                   "delta_total_MAE": te_a["total_MAE"] - te_b["total_MAE"], "delta_team_total_bias": te_a["team_total_bias"] - te_b["team_total_bias"],
                   "delta_team_total_MAE": te_a["team_total_MAE"] - te_b["team_total_MAE"],
                   "winner": {"mean_abs_delta_home_win_prob": float(np.abs(d_ph).mean()), "max_abs_delta_home_win_prob": float(np.abs(d_ph).max()),
                              "mean_delta_home_win_prob": float(d_ph.mean()), "sd_delta_home_win_prob": float(d_ph.std(ddof=1)),
                              "favorite_flips": flips, "delta_brier": float((br_a - br_b).mean()), "delta_brier_SE": se(br_a - br_b),
                              "delta_logloss": float((ll_a - ll_b).mean()), "delta_tie_prob_mean": float(np.mean([per_game[g]["paired"]["delta_tie_prob"] for g in games])),
                              "quantiles_delta_home_win_prob": {q: float(np.quantile(d_ph, q / 100)) for q in (0, 10, 25, 50, 75, 90, 100)}},
                   "mean_ladder_probability_changes": {grp: {k: float(np.mean([per_game[g]["paired"]["ladder_probability_changes"][grp][k] for g in games])) for k in per_game[games[0]]["paired"]["ladder_probability_changes"][grp]}
                                                      for grp in per_game[games[0]]["paired"]["ladder_probability_changes"]}},
        "FG_REPAIR": {"gap_closed_vs_pilot_observed": float((fg_a - fg_b) / gap), "gap_closed_vs_2020_25": float((fg_a - fg_b) / gap_hist), "residual_vs_2020_25": hist["made_fg_per_team_game"] - fg_a,
                      "residual_vs_pilot_observed": obs_fg_pilot - fg_a},
        "UNRELATED_COMPONENTS": comp_report, "UNRELATED_COMPONENTS_UNCHANGED_under_1pct": unrelated_ok,
        "PATH_ALIGNMENT": {"sims_with_identical_team_volume_and_yards": identical_sims, "total_sims": total_sims, "share": identical_sims / total_sims},
        "ACCOUNTING": {"base": acct_tot["base"], "abl": acct_tot["abl"], "pass": all(v == 0 for a in acct_tot.values() for v in a.values())},
        "MULTI_TD_BLOCKS": multi_td_blocks,
        "GT8": {"base_count": len(gt8_by_arm["base"]), "abl_count": len(gt8_by_arm["abl"]),
                "classification_counts": {arm: {c: sum(1 for x in gt8_by_arm[arm] if x["classification"] == c) for c in ("VALID_AGGREGATED_BLOCK", "MULTIPLE_SCORE_ARTIFACT", "UNKNOWN")} for arm in gt8_by_arm},
                "td_count_distribution_abl": {str(k): sum(1 for x in gt8_by_arm["abl"] if x["scoring_components"]["touchdowns"] == k) for k in range(1, 8)},
                "single_receiver_or_rusher_multi_td_share_abl": float(np.mean([x["scoring_components"]["max_single_player_tds"] >= 2 for x in gt8_by_arm["abl"]])) if gt8_by_arm["abl"] else None,
                "historical_gt8_blocks_1999_2025": hist_gt8, "historical_blocks": int(len(d)), "historical_gt8_rate": hist_gt8 / len(d)},
        "PLAIN_EQUALS_RECORDED_BASELINE_ALL": all(r["plain_equals_recorded_baseline"] for r in meta["results"]),
        "ENGINE_UNCHANGED": meta["engine_unchanged"],
        "PER_GAME": per_game,
    }
    (OUT / "FG_CAUSAL_ABLATION_V1_REPORT.json").write_text(json.dumps(report, indent=1, default=float))
    keep = {k: v for k, v in report.items() if k != "PER_GAME"}
    print(json.dumps(keep, indent=1, default=float))


if __name__ == "__main__":
    main()
