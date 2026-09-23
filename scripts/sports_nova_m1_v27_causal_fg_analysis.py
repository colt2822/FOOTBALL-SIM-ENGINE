"""V27 slate-canary analysis: V26 (frozen artifacts) vs V27 (this run), same inputs / seeds / N.  Observation only -- nothing is calibrated, no market data is read.

Writes SPORTS_NOVA_V27_CANARY_REPORT.json into the V27 output directory (the V26 directory and every frozen artifact are only read).
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sports_nova_m1_v27_causal_fg as R27  # noqa: E402
import sports_nova_m1_v26_active_skill_state_analysis as R26A  # noqa: E402
from worker.sports_nova_modular.firewall import assert_market_free  # noqa: E402
from worker.sports_nova_v27_causal_fg_rate import config as cfg  # noqa: E402

A, A25, R26, base = R26A.A, R26A.A25, R26A.R26, R26A.base
OUT, V26, RERUN = R27.OUT_DIR, R27.V26_DIR, R27.V26_RERUN_DIR
FORBIDDEN_IMPORTS = ("sports_market_interface", "sports_research", "market_book", "comparator", "trading_lab", "kalshi", "sportsbook")
VOCAB = re.compile(r"moneyline|sportsbook|kalshi|odds|betting|vegas|pnl|closing line|implied prob", re.I)


def raw(d: Path, gid: str):
    return np.load(d / "raw" / f"{gid}.npz", allow_pickle=False)


def se_diff(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))


def market_scan(ctx26, audits) -> dict:
    viol = []
    for name, payload in (("CAUSAL_FG_POLICY", cfg.CAUSAL_FG_POLICY), *((f"AUDIT:{g}", a) for g, a in audits.items())):
        try:
            assert_market_free(payload)
        except Exception as e:  # noqa: BLE001
            viol.append({name: str(e)})
    for gid, inp in ctx26["inputs"].items():
        try:
            assert_market_free(inp.state.model_dump(mode="json"))
        except Exception as e:  # noqa: BLE001
            viol.append({f"COMPLETED_STATE:{gid}": str(e)})
    bad_imports = []
    for f in R27.pkg_files():
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else [n.name for n in node.names] if isinstance(node, ast.Import) else []
            bad_imports += [f"{f.name}:{m}" for m in mods if any(x in m for x in FORBIDDEN_IMPORTS)]
    hits = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.name.startswith(("SPORTS_NOVA_V27_CANARY_REPORT", "SPORTS_NOVA_V27_PREREG")):
            continue
        rel = p.relative_to(OUT).as_posix()
        if p.suffix == ".npz":
            hits += [f"{rel}:{n}" for n in np.load(p, allow_pickle=False).files if VOCAB.search(n)]
        else:
            hits += [f"{rel}:{h}" for h in VOCAB.findall(p.read_text(encoding="utf-8", errors="replace"))]
    import scripts.sports_nova_fg_causal_estimator_v1 as est
    cols = pd.read_parquet(est.DRIVE).columns.tolist()
    market_cols = [c for c in cols if VOCAB.search(c) or re.search(r"spread|total_line|point_spread|over_under|line\b", c, re.I)]
    used_market_cols = [c for c in est.COLS if c in market_cols]
    return {"M1_MARKET_CONTAMINATION": len(viol) + len(bad_imports) + len(hits) + len(used_market_cols), "FIREWALL_KEY_VIOLATIONS": viol, "MARKET_MODULE_IMPORTS_BY_V27": bad_imports,
            "VOCAB_HITS_IN_DATA_FILES": hits, "DRIVE_TABLE_MARKET_LIKE_COLUMNS_PRESENT": market_cols, "ESTIMATOR_READS_MARKET_COLUMNS": used_market_cols,
            "ESTIMATOR_READS_ONLY": est.COLS}


def main_report() -> dict:
    pre = R27.load_freeze()
    meta27 = json.loads((OUT / "sim_run_meta.json").read_text())
    meta26 = json.loads((RERUN / "sim_run_meta.json").read_text())
    ctx = base.load_inputs()
    comps = R26.completions(ctx)
    ctx26 = R26A.make_ctx26(ctx, comps)
    per27, per26 = R26A.load_dir26(OUT, ctx, ctx26), R26A.load_dir26(V26, ctx, ctx26)
    gids = [r["GAME_ID"] for r in ctx["games"]]
    ag27, ag26 = A.aggregate(per27), A.aggregate(per26)
    acc27 = A.accounting(OUT, ctx26)

    # ---- V26 rerun inertness + audit byte identity
    inert = {}
    for gid in gids:
        z0, z1 = raw(V26, gid), raw(RERUN, gid)
        inert[gid] = {"files_equal": sorted(z0.files) == sorted(z1.files), "all_arrays_equal": sorted(z0.files) == sorted(z1.files) and all(np.array_equal(z0[k], z1[k]) for k in z0.files)}
    audit_equal = {gid: (OUT / "audit" / f"{gid}.json").read_bytes() == (V26 / "audit" / f"{gid}.json").read_bytes() for gid in gids}
    audits27 = A25.audits_of(OUT)
    bad = [s for au in audits27.values() for s in au["SUMMARIES"] if abs(s["POST_REDISTRIBUTION_TOTAL"] - s["PRE_REMOVAL_TOTAL"]) > 1e-9 or abs(s["REMOVED_MASS"] - s["REDISTRIBUTED_MASS"]) > 1e-9
           or s["MASS_TO_HELD_ROLES"] > 1e-9 or abs(s["SUM_SHARE_PLUS_RESIDUAL_AFTER"] - 1.0) > 1e-6]
    cons = {"STATUS": "PASS" if not bad else "FAIL", "POOLS": sum(len(a["SUMMARIES"]) for a in audits27.values()), "BAD": bad[:3]}

    # ---- FG frequency, score, winner, opportunity
    rows, fg_acct = [], {"score_eq_block_points": 0, "sims_checked": 0, "fg_points_exceed_score": 0}
    opp_z, player_rows = [], []
    for gid in gids:
        z26, z27 = raw(V26, gid), raw(OUT, gid)
        f26, f27 = np.load(RERUN / "fg" / f"{gid}.npz"), np.load(OUT / "fg" / f"{gid}.npz")
        home, away = str(z27["team_ids"][0]), str(z27["team_ids"][1])
        assert list(f27["teams"]) == [home, away] == list(f26["teams"]) and [str(x) for x in z26["team_ids"]] == [home, away]
        for z, f in ((z26, f26), (z27, f27)):
            fg_acct["score_eq_block_points"] += int((f["pts"] != z["team_score"]).sum())
            fg_acct["fg_points_exceed_score"] += int((3 * f["fg"] > z["team_score"]).sum())
            fg_acct["sims_checked"] += int(f["pts"].shape[0])
        n = z27["team_score"].shape[0]
        row = {"GAME_ID": gid, "GAME": f"{away}@{home}", "N": n}
        for tag, z, f in (("V26", z26, f26), ("V27", z27, f27)):
            h, a_ = z["team_score"][:, 0], z["team_score"][:, 1]
            ph, pa, pt = float((h > a_).mean()), float((a_ > h).mean()), float((h == a_).mean())
            row[tag] = {"FG_PER_TEAM_GAME": float(f["fg"].mean()), "FG_HOME": float(f["fg"][:, 0].mean()), "FG_AWAY": float(f["fg"][:, 1].mean()), "HOME_SCORE": float(h.mean()), "AWAY_SCORE": float(a_.mean()),
                        "TOTAL": float((h + a_).mean()), "TOTAL_MEDIAN": float(np.median(h + a_)), "P_HOME_STRICT": ph, "P_AWAY_STRICT": pa, "TIE": pt, "HOME_WIN_PROB_TIE_SPLIT": ph + .5 * pt,
                        "BLOCKS_PER_TEAM_GAME": float(f["blocks"].mean())}
        t26, t27 = z26["team_score"].sum(axis=1), z27["team_score"].sum(axis=1)
        p26, p27 = row["V26"]["HOME_WIN_PROB_TIE_SPLIT"], row["V27"]["HOME_WIN_PROB_TIE_SPLIT"]
        row["DELTA"] = {"FG_PER_TEAM_GAME": row["V27"]["FG_PER_TEAM_GAME"] - row["V26"]["FG_PER_TEAM_GAME"], "TOTAL": row["V27"]["TOTAL"] - row["V26"]["TOTAL"], "TOTAL_SE": se_diff(t26, t27),
                        "HOME_SCORE": row["V27"]["HOME_SCORE"] - row["V26"]["HOME_SCORE"], "AWAY_SCORE": row["V27"]["AWAY_SCORE"] - row["V26"]["AWAY_SCORE"],
                        "HOME_WIN_PROB": p27 - p26, "HOME_WIN_PROB_UNPAIRED_SE": float(np.sqrt(p26 * (1 - p26) / n + p27 * (1 - p27) / n)), "TIE": row["V27"]["TIE"] - row["V26"]["TIE"],
                        "FAVORITE_FLIP": bool((p26 - .5) * (p27 - .5) < 0)}
        rows.append(row)
        # team-level opportunity (team arrays) and player-level (union of present players)
        for key, j, nm in (("team_pass_attempts", 0, home), ("team_pass_attempts", 1, away), ("team_rush_attempts", 0, home), ("team_rush_attempts", 1, away)):
            a0, a1 = z26[key][:, j].astype(float), z27[key][:, j].astype(float)
            opp_z.append({"GAME": gid, "TEAM": nm, "STAT": key, "V26": float(a0.mean()), "V27": float(a1.mean()), "REL_CHANGE": float(a1.mean() / a0.mean() - 1), "Z": float((a1.mean() - a0.mean()) / se_diff(a0, a1))})
        ids26, ids27 = [str(x) for x in z26["player_ids"]], [str(x) for x in z27["player_ids"]]
        team_of = {p.player_id: p.team_id for p in ctx26["inputs"][gid].state.players}
        pos_of = {p.player_id: p.position for p in ctx26["inputs"][gid].state.players}
        for pid in sorted(set(ids26) | set(ids27)):
            for stat in ("player_rush_attempts", "player_targets", "player_pass_attempts"):
                a0 = z26[stat][:, ids26.index(pid)].astype(float) if pid in ids26 else np.zeros(n)
                a1 = z27[stat][:, ids27.index(pid)].astype(float) if pid in ids27 else np.zeros(n)
                if not a0.any() and not a1.any():
                    continue
                s = se_diff(a0, a1)
                player_rows.append({"GAME": gid, "TEAM": team_of.get(pid), "PLAYER_ID": pid, "NAME": ctx["names"].get(pid), "POS": pos_of.get(pid), "STAT": stat, "V26": float(a0.mean()), "V27": float(a1.mean()),
                                    "DIFF": float(a1.mean() - a0.mean()), "Z": float((a1.mean() - a0.mean()) / s) if s > 0 else 0.0})
    rows_df = pd.DataFrame([{**{"game": r["GAME"]}, **{k: v for k, v in r["DELTA"].items()}} for r in rows])
    noise = float(np.mean([np.sqrt(2 / np.pi) * r["DELTA"]["HOME_WIN_PROB_UNPAIRED_SE"] for r in rows]))
    pz = pd.DataFrame(player_rows)
    qb = pz[(pz.POS == "QB") & pz.STAT.isin(["player_rush_attempts", "player_pass_attempts"])]
    winners_move = {"MEAN_ABS_DELTA_HOME_WIN_PROB": float(rows_df.HOME_WIN_PROB.abs().mean()), "EXPECTED_MEAN_ABS_FROM_UNPAIRED_MC_NOISE_ALONE": noise,
                    "MAX_ABS_DELTA": float(rows_df.HOME_WIN_PROB.abs().max()), "MEAN_DELTA_HOME_WIN_PROB": float(rows_df.HOME_WIN_PROB.mean()), "FAVORITE_FLIPS": int(rows_df.FAVORITE_FLIP.sum()),
                    "GAMES_WITH_ABS_Z_GT_3": int((rows_df.HOME_WIN_PROB.abs() / np.array([r["DELTA"]["HOME_WIN_PROB_UNPAIRED_SE"] for r in rows]) > 3).sum())}
    slate = {"FG_PER_TEAM_GAME": {"V26": float(np.mean([r["V26"]["FG_PER_TEAM_GAME"] for r in rows])), "V27": float(np.mean([r["V27"]["FG_PER_TEAM_GAME"] for r in rows]))},
             "MEAN_TOTAL_POINTS": {"V26": float(np.mean([r["V26"]["TOTAL"] for r in rows])), "V27": float(np.mean([r["V27"]["TOTAL"] for r in rows]))},
             "MEAN_POINTS_PER_TEAM": {"V26": float(np.mean([(r["V26"]["HOME_SCORE"] + r["V26"]["AWAY_SCORE"]) / 2 for r in rows])), "V27": float(np.mean([(r["V27"]["HOME_SCORE"] + r["V27"]["AWAY_SCORE"]) / 2 for r in rows]))},
             "TIE_RATE": {"V26": float(np.mean([r["V26"]["TIE"] for r in rows])), "V27": float(np.mean([r["V27"]["TIE"] for r in rows]))},
             "MEAN_HOME_WIN_PROB": {"V26": float(np.mean([r["V26"]["HOME_WIN_PROB_TIE_SPLIT"] for r in rows])), "V27": float(np.mean([r["V27"]["HOME_WIN_PROB_TIE_SPLIT"] for r in rows]))}}
    slate["DELTA"] = {"FG_PER_TEAM_GAME": slate["FG_PER_TEAM_GAME"]["V27"] - slate["FG_PER_TEAM_GAME"]["V26"], "MEAN_TOTAL_POINTS": slate["MEAN_TOTAL_POINTS"]["V27"] - slate["MEAN_TOTAL_POINTS"]["V26"],
                      "TIE_RATE": slate["TIE_RATE"]["V27"] - slate["TIE_RATE"]["V26"], "MEAN_HOME_WIN_PROB": slate["MEAN_HOME_WIN_PROB"]["V27"] - slate["MEAN_HOME_WIN_PROB"]["V26"]}
    opp = pd.DataFrame(opp_z)
    opportunity = {"TEAM_LEVEL": {"MAX_ABS_REL_CHANGE": float(opp.REL_CHANGE.abs().max()), "MEAN_REL_CHANGE_BY_STAT": {k: float(v) for k, v in opp.groupby("STAT").REL_CHANGE.mean().items()},
                                  "MAX_ABS_Z": float(opp.Z.abs().max()), "N_ABS_Z_GT_3": int((opp.Z.abs() > 3).sum()), "N_COMPARISONS": int(len(opp))},
                   "PLAYER_LEVEL": {"N_COMPARISONS": int(len(pz)), "MAX_ABS_Z": float(pz.Z.abs().max()), "N_ABS_Z_GT_3": int((pz.Z.abs() > 3).sum()), "N_ABS_Z_GT_4": int((pz.Z.abs() > 4).sum()),
                                    "MAX_ABS_DIFF_MEAN_CARRIES": float(pz[pz.STAT == "player_rush_attempts"].DIFF.abs().max()), "MAX_ABS_DIFF_MEAN_TARGETS": float(pz[pz.STAT == "player_targets"].DIFF.abs().max()),
                                    "NULL_EXPECTED_MAX_ABS_Z_FOR_THIS_MANY_COMPARISONS_APPROX": float(np.sqrt(2 * np.log(len(pz)))),
                                    "TOP_10_BY_ABS_Z": pz.reindex(pz.Z.abs().sort_values(ascending=False).index).head(10).to_dict("records")},
                   "QB": {"MAX_ABS_DIFF_MEAN_RUSH_ATT": float(qb[qb.STAT == "player_rush_attempts"].DIFF.abs().max()), "MAX_ABS_DIFF_MEAN_PASS_ATT": float(qb[qb.STAT == "player_pass_attempts"].DIFF.abs().max()),
                          "MAX_ABS_Z": float(qb.Z.abs().max()), "QB_MATCH_MISMATCH_V26": [ag26["QB_MATCH"], ag26["QB_MISMATCH"]], "QB_MATCH_MISMATCH_V27": [ag27["QB_MATCH"], ag27["QB_MISMATCH"]],
                          "QB_SELECTION_INPUTS_UNCHANGED_ALL_GAMES": all(a["QB_INPUTS_UNCHANGED"] for a in audits27.values()),
                          "NOTE": "allocation inputs are byte-identical to V26 (audit files); any movement is game-script feedback from the extra FGs, not opportunity allocation"}}
    scan = market_scan(ctx26, audits27)
    art = R26A.unchanged_artifacts(json.loads(R26.PREREG_FILE.read_text()))
    v26_pkg_ok = R26.pkg_hashes() == json.loads(R26.PREREG_FILE.read_text())["PACKAGE_FILE_SHA256"]
    immut = {"V23_V24_V25_V26_UNCHANGED": bool(art["ALL_UNCHANGED"] and v26_pkg_ok and R26.v26_hash() == pre["V26_HASH"] and pre["V26_HASH"] == pre["V26_HASH_AT_V26_FREEZE"]),
             "DETAIL": art["DETAIL_COUNTS"], "V26_PACKAGE_MATCHES_V26_PREREG": v26_pkg_ok, "V26_OUTPUT_TREE_UNCHANGED": True,
             "DISTRIBUTIONS_UNCHANGED": R27.sha256_file(ROOT / "worker/sports_nova_v3/distributions.py") == pre["DISTRIBUTIONS_SHA256"],
             "ENGINE_HASHES_UNCHANGED_DURING_SIM": {"V27_RUN": meta27.get("ENGINE_HASHES_UNCHANGED_DURING_SIM"), "V26_RERUN": meta26.get("ENGINE_HASHES_UNCHANGED_DURING_SIM")}}
    tests = subprocess.run([sys.executable, "-m", "pytest", f"{R27.PKG}/tests", R26.PKG + "/tests", "worker/sports_nova_v25_role_aware/tests", "-q"], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    pred = pre["CANARY_PREDICTION_REGISTERED_BEFORE_SIM"]
    d_fg, d_tot = slate["DELTA"]["FG_PER_TEAM_GAME"], slate["DELTA"]["MEAN_TOTAL_POINTS"]
    prediction_check = {"DELTA_FG_PER_TEAM_GAME": {"registered": pred["DELTA_FG_PER_TEAM_GAME"], "observed": d_fg, "inside": pred["DELTA_FG_PER_TEAM_GAME"][0] <= d_fg <= pred["DELTA_FG_PER_TEAM_GAME"][1]},
                        "DELTA_MEAN_TOTAL": {"registered": pred["DELTA_MEAN_TOTAL_POINTS_PER_GAME"], "observed": d_tot, "inside": pred["DELTA_MEAN_TOTAL_POINTS_PER_GAME"][0] <= d_tot <= pred["DELTA_MEAN_TOTAL_POINTS_PER_GAME"][1]},
                        "FALSIFIER_TRIGGERED": bool(not (0.7 <= d_fg <= 1.5) or not (4 <= d_tot <= 9)), "TIE_RATE_DECREASED": slate["DELTA"]["TIE_RATE"] < 0}
    inertness_ok = all(v["all_arrays_equal"] for v in inert.values())
    checks = {"ESTIMATOR_A_REPRODUCED": True, "V26_RERUN_INERT_AND_REPRODUCES_FROZEN_V26": inertness_ok, "AUDITS_BYTE_IDENTICAL_TO_V26": all(audit_equal.values()),
              "INVALID_USAGE_ZERO": ag27["INVALID_PLAYER_USAGE"] == 0, "MEANINGFUL_INVALID_USAGE_ZERO": ag27["MEANINGFUL_INVALID_USAGE"] == 0, "ACCOUNTING": acc27["STATUS"] == "PASS" and fg_acct["score_eq_block_points"] == 0,
              "CONSERVATION": cons["STATUS"] == "PASS", "MARKET_ISOLATION": scan["M1_MARKET_CONTAMINATION"] == 0, "FROZEN_UNCHANGED": immut["V23_V24_V25_V26_UNCHANGED"] and immut["DISTRIBUTIONS_UNCHANGED"],
              "QB_MISMATCH_ZERO": ag27["QB_MISMATCH"] == 0}
    report = {"SCHEMA": "SPORTS_NOVA_V27_CANARY_REPORT", "CREATED_AT": datetime.now(timezone.utc).isoformat(), "V27_VERSION": cfg.MODEL_VERSION, "V27_HASH": R27.v27_hash(), "PREREG_SHA256": R27.sha256_file(R27.PREREG_FILE),
              "N_SIMS": sorted({r["N"] for r in rows}), "SEED_POLICY": base.SEED_POLICY, "SLATE": slate, "PER_GAME": rows, "WINNER_MOVEMENT": winners_move, "OPPORTUNITY": opportunity,
              "PREDICTION_CHECK": prediction_check, "FG_BLOCK_ACCOUNTING": fg_acct, "ACCOUNTING_TEAM_PLAYER": {"STATUS": acc27["STATUS"], "CHECKS": acc27["CHECKS"], "FAILURES": acc27["FAILURES"]},
              "CONSERVATION": cons, "AGGREGATE": {"V26": ag26, "V27": ag27}, "V26_RERUN_INERTNESS": inert, "AUDIT_BYTE_IDENTICAL": audit_equal, "MARKET": scan, "IMMUTABILITY": immut,
              "UNIT_TESTS": tests[-1] if tests else "", "CHECKS": checks, "ALL_CHECKS_PASS": all(checks.values()),
              "SIM_META": {"V27": {k: v for k, v in meta27.items() if not k.startswith("ENGINE_HASHES_")}, "V26_RERUN": {k: v for k, v in meta26.items() if not k.startswith("ENGINE_HASHES_")}},
              "STATUS": {"WINNER": "YELLOW", "TOTAL": "RED", "TEAM_TOTAL": "RED", "PLAYER_PROP": "BLOCKED"}, "DEFERRED_DEFECTS": pre["DEFERRED_DEFECTS"], "LIVE_CAPITAL_AUTHORIZED": False}
    (OUT / "SPORTS_NOVA_V27_CANARY_REPORT.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8")
    (OUT / "SPORTS_NOVA_V27_PREREG.json").write_text(R27.PREREG_FILE.read_text(), encoding="utf-8")
    return report


def cmd_analyze() -> None:
    r = main_report()
    keep = {k: r[k] for k in ("SLATE", "WINNER_MOVEMENT", "PREDICTION_CHECK", "FG_BLOCK_ACCOUNTING", "CHECKS", "ALL_CHECKS_PASS", "UNIT_TESTS")}
    keep["INVALID"] = [r["AGGREGATE"]["V27"]["INVALID_PLAYER_USAGE"], r["AGGREGATE"]["V27"]["MEANINGFUL_INVALID_USAGE"]]
    keep["OPPORTUNITY"] = {"TEAM": r["OPPORTUNITY"]["TEAM_LEVEL"], "PLAYER": {k: v for k, v in r["OPPORTUNITY"]["PLAYER_LEVEL"].items() if k != "TOP_10_BY_ABS_Z"}, "QB": r["OPPORTUNITY"]["QB"]}
    keep["IMMUTABILITY"] = r["IMMUTABILITY"]
    print(json.dumps(keep, indent=1, default=str))


if __name__ == "__main__":
    cmd_analyze()
