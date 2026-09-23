"""V23-vs-V24 analysis + the four mission artifacts.  Imported by sports_nova_m1_v24_roster_eligibility.py `analyze`.

Measurement is identical for both versions: base.analyze_game / base.sim_class (the V23 baseline runner's own definitions).
The V23 side is first REPRODUCED from its persisted per-game files and must equal the published baseline (615 / 276 / 31.2% / 81.0% / 0-6-8)
before any V24 number is trusted.
"""
from __future__ import annotations

import ast
import difflib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import sports_nova_m1_v24_roster_eligibility as R
from worker.sports_nova_modular.firewall import assert_market_free

base, cfg, ROOT, OUT_DIR, V23_DIR = R.base, R.cfg, R.ROOT, R.OUT_DIR, R.V23_DIR
STATS = base.STAT_NAMES


def load_dir(d: Path, ctx: dict) -> dict:
    per = {}
    for rec in ctx["games"]:
        gid = rec["GAME_ID"]
        g = json.loads((d / "games" / f"{gid}.json").read_text())
        inp = ctx["inputs"][gid]
        a = base.analyze_game(rec, inp, g, ctx)
        smoke = rec["M1_ADAPTER"]["SMOKE"].get("M1_STATE_HASH")
        input_hash = base._state_hash(inp.state)
        hash_ok = input_hash == smoke == g["STATE_HASH"]
        sc, reasons = base.sim_class(rec["DATA_QUALITY"]["CLASS"], a, a["QB_INTEGRITY"], hash_ok, len(a["PANEL_OUT_PLAYERS_WITH_SIM_USAGE"]))
        a.update({"SIM_CLASS": sc, "SIM_CLASS_REASONS": reasons, "INPUT_HASH_OK": hash_ok})
        per[gid] = (rec, inp, g, a)
    return per


def aggregate(per: dict) -> dict:
    order = list(per)
    teams = [(gid, t) for gid in order for t in (per[gid][2]["AWAY"], per[gid][2]["HOME"])]
    comb = [per[gid][3]["TEAMS"][t]["NON_ACTIVE_ROSTER_SHARE"] for gid, t in teams]
    pools = [per[gid][3]["TEAMS"][t][k] for gid, t in teams for k in ("NON_ACTIVE_CARRY_SHARE", "NON_ACTIVE_TARGET_SHARE")]
    cats: dict = {}
    for gid in order:
        for k, v in per[gid][3]["INVALID_PLAYER_USAGE_BY_CATEGORY"].items():
            cats[k] = cats.get(k, 0) + v
    qb = [q for gid in order for q in per[gid][3]["QB_INTEGRITY"]]
    return {"INVALID_PLAYER_USAGE": sum(per[g][3]["INVALID_PLAYER_USAGE_COUNT"] for g in order),
            "MEANINGFUL_INVALID_USAGE": sum(per[g][3]["INVALID_PLAYER_USAGE_MEANINGFUL_COUNT"] for g in order),
            "AVG_NONACTIVE_SHARE": float(np.mean(comb)), "MAX_NONACTIVE_SHARE": float(max(pools)),
            "WORST_GAME": max(order, key=lambda g: per[g][3]["MAX_TEAM_NONACTIVE_SHARE"]),
            "INVALID_BY_CATEGORY": cats,
            "SIM_CLASS": {k: sum(1 for g in order if per[g][3]["SIM_CLASS"] == k) for k in ("SIM_GREEN", "SIM_YELLOW", "SIM_RED")},
            "CONTAMINATION_CLASS": {k: sum(1 for g in order if per[g][3]["CONTAMINATION_CLASS"] == k) for k in ("CLEAN", "LOW", "MATERIAL", "SEVERE", "CRITICAL")},
            "QB_MATCH": sum(1 for q in qb if q["MATCH"]), "QB_MISMATCH": sum(1 for q in qb if not q["MATCH"]),
            "QB_MATCH_VIA_FALLBACK": sum(1 for q in qb if q["MATCH"] and q["MATCH_BASIS"] != "PANEL_IDENTITY_INSTALLED"),
            "PANEL_OUT_USED": sum(len(per[g][3]["PANEL_OUT_PLAYERS_WITH_SIM_USAGE"]) for g in order)}


def usage_split(per: dict) -> dict:
    """Invalid usage split by mechanism: allocation (carries/targets) vs QB-selection (pass attempts only; QB logic is out of scope)."""
    alloc, qbsel, rows = 0, 0, []
    for gid, (_, _, _, a) in per.items():
        for r in a["INVALID_PLAYERS"]:
            if r["SIM_MEAN_CARRIES"] > 0 or r["SIM_MEAN_TARGETS"] > 0:
                alloc += 1
                rows.append({"GAME": gid, **{k: r[k] for k in ("TEAM", "NAME", "POSITION", "ROSTER_CATEGORY", "SIM_MEAN_CARRIES", "SIM_MEAN_TARGETS")}})
            else:
                qbsel += 1
                rows.append({"GAME": gid, "MECHANISM": "QB_SELECTION_PASS_ATTEMPTS_ONLY", **{k: r[k] for k in ("TEAM", "NAME", "POSITION", "ROSTER_CATEGORY", "SIM_MEAN_PASS_ATT")}})
    return {"ALLOCATION_USAGE_COUNT": alloc, "QB_SELECTION_ONLY_COUNT": qbsel, "ROWS": rows}


def accounting(d: Path, ctx: dict) -> dict:
    """sum(player rush_attempts)==team rush_attempts and sum(targets)==team pass_attempts, every sim, ALL players incl. QB (scrambles are credited to the QB)."""
    fails, checks = [], 0
    for rec in ctx["games"]:
        gid = rec["GAME_ID"]
        z = np.load(d / "raw" / f"{gid}.npz", allow_pickle=False)
        team_of = {p.player_id: p.team_id for p in ctx["inputs"][gid].state.players}
        pids = [str(x) for x in z["player_ids"]]
        for j, tid in enumerate(z["team_ids"]):
            idx = [i for i, p in enumerate(pids) if team_of.get(p) == str(tid)]
            for pk, tk in (("player_rush_attempts", "team_rush_attempts"), ("player_targets", "team_pass_attempts")):
                checks += 1
                bad = int((z[pk][:, idx].sum(axis=1) != z[tk][:, j]).sum())
                if bad:
                    fails.append({"GAME": gid, "TEAM": str(tid), "CHECK": f"{pk}=={tk}", "FAILING_SIMS": bad})
    return {"CHECKS": checks, "FAILURES": fails, "STATUS": "PASS" if not fails else "FAIL"}


def median_se(a, b, rng, reps=300):
    da = [np.median(rng.choice(a, a.size)) for _ in range(reps)]
    db = [np.median(rng.choice(b, b.size)) for _ in range(reps)]
    return float(np.sqrt(np.var(da) + np.var(db)))


def football(per23, per24, ctx) -> dict:
    rng = np.random.default_rng(20260920)
    games = []
    for gid in per23:
        g23, g24 = per23[gid][2], per24[gid][2]
        z23, z24 = (np.load(d / "raw" / f"{gid}.npz", allow_pickle=False) for d in (V23_DIR, OUT_DIR))
        h = list(z23["team_ids"]).index(g23["HOME"]), list(z23["team_ids"]).index(g23["AWAY"])
        tot = lambda z: z["team_score"][:, h[0]] + z["team_score"][:, h[1]]
        mar = lambda z: z["team_score"][:, h[0]] - z["team_score"][:, h[1]]
        row = {"GAME": f"{g23['AWAY']}@{g23['HOME']}", "GAME_ID": gid}
        for nm, f, key in (("TOTAL", tot, "TOTAL_POINTS"), ("MARGIN", mar, "MARGIN_HOME_MINUS_AWAY")):
            v23, v24 = f(z23), f(z24)
            row[nm] = {"V23_MEDIAN": g23[key]["median"], "V24_MEDIAN": g24[key]["median"], "DELTA": g24[key]["median"] - g23[key]["median"],
                       "SE_OF_DELTA": median_se(v23, v24, rng), "V23_MEAN": g23[key]["mean"], "V24_MEAN": g24[key]["mean"],
                       "V23_P10_P90": [g23[key]["P10"], g23[key]["P90"]], "V24_P10_P90": [g24[key]["P10"], g24[key]["P90"]]}
        row["SCORE_MEDIAN"] = {t: {"V23": g23[k]["median"], "V24": g24[k]["median"]} for t, k in ((g23["AWAY"], "SCORE_AWAY"), (g23["HOME"], "SCORE_HOME"))}
        row["HOME_WIN_PROB"] = {"V23": g23["HOME_WIN_PROB_TIE_SPLIT"], "V24": g24["HOME_WIN_PROB_TIE_SPLIT"]}
        row["TEAM_ATTEMPTS_MEAN"] = {t: {k: {"V23": g23["TEAM"][t][k]["mean"], "V24": g24["TEAM"][t][k]["mean"],
                                              "PCT": 100.0 * (g24["TEAM"][t][k]["mean"] / g23["TEAM"][t][k]["mean"] - 1.0)}
                                          for k in ("pass_attempts", "rush_attempts")} for t in (g23["AWAY"], g23["HOME"])}
        row["PLAYER_CONCENTRATION"] = {}
        for t in (g23["AWAY"], g23["HOME"]):
            c = {}
            for lab, per in (("V23", per23), ("V24", per24)):
                pl = per[gid][3]["TEAMS"][t]["PLAYERS"]
                tc, tt = max(pl, key=lambda r: r["SIM_CARRY_SHARE"]), max(pl, key=lambda r: r["SIM_TARGET_SHARE"])
                c[lab] = {"TOP_CARRY": [tc["NAME"], tc["SIM_CARRY_SHARE"], tc["ROSTER_CATEGORY"]], "TOP_TARGET": [tt["NAME"], tt["SIM_TARGET_SHARE"], tt["ROSTER_CATEGORY"]]}
            row["PLAYER_CONCENTRATION"][t] = c
        games.append(row)
    rules = R.REGRESSION_RULES
    m23 = float(np.mean([r["TOTAL"]["V23_MEDIAN"] for r in games]))
    m24 = float(np.mean([r["TOTAL"]["V24_MEDIAN"] for r in games]))
    trig = []
    if m23 - m24 > rules["SLATE_MEAN_OF_MEDIAN_TOTALS_DROP_POINTS"]:
        trig.append({"RULE": "SLATE_MEAN_OF_MEDIAN_TOTALS_DROP", "V23": m23, "V24": m24})
    for r in games:
        if abs(r["TOTAL"]["DELTA"]) > rules["GAME_MEDIAN_TOTAL_SHIFT_POINTS"]:
            trig.append({"RULE": "GAME_MEDIAN_TOTAL_SHIFT", "GAME": r["GAME"], "DELTA": r["TOTAL"]["DELTA"], "SE": r["TOTAL"]["SE_OF_DELTA"]})
        for t, d in r["TEAM_ATTEMPTS_MEAN"].items():
            for k, v in d.items():
                if abs(v["PCT"]) > rules["TEAM_MEAN_ATTEMPTS_SHIFT_PCT"]:
                    trig.append({"RULE": "TEAM_MEAN_ATTEMPTS_SHIFT", "GAME": r["GAME"], "TEAM": t, "STAT": k, "PCT": v["PCT"]})
    dt = [r["TOTAL"]["DELTA"] for r in games]
    dm = [r["MARGIN"]["DELTA"] for r in games]
    return {"GAMES": games, "SLATE_MEAN_OF_MEDIAN_TOTALS": {"V23": m23, "V24": m24, "DELTA": m24 - m23},
            "TOTAL_DISTRIBUTION_CHANGE": {"MEAN_DELTA": float(np.mean(dt)), "MEAN_ABS_DELTA": float(np.mean(np.abs(dt))), "MAX_ABS_DELTA": float(np.max(np.abs(dt))),
                                          "MEAN_SE": float(np.mean([r["TOTAL"]["SE_OF_DELTA"] for r in games]))},
            "MARGIN_DISTRIBUTION_CHANGE": {"MEAN_DELTA": float(np.mean(dm)), "MEAN_ABS_DELTA": float(np.mean(np.abs(dm))), "MAX_ABS_DELTA": float(np.max(np.abs(dm))),
                                           "MEAN_SE": float(np.mean([r["MARGIN"]["SE_OF_DELTA"] for r in games]))},
            "REGRESSION_TRIGGERS": trig, "REGRESSION_RULES": rules}


def load_audits() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted((OUT_DIR / "audit").glob("*.json"))}


def replacement_tables(audits: dict, per24: dict) -> dict:
    flagged, qb_rows, top = [], [], []
    for gid, au in audits.items():
        realized = {}
        for t in (per24[gid][2]["AWAY"], per24[gid][2]["HOME"]):
            for r in per24[gid][3]["TEAMS"][t]["PLAYERS"]:
                realized[r["PLAYER_ID"]] = r
        for r in au["ROWS"]:
            rr = realized.get(r["PLAYER_ID"], {})
            slim = {"GAME": gid, "TEAM": r["TEAM"], "KIND": r["KIND"], "PLAYER_ID": r["PLAYER_ID"], "NAME": r["NAME"], "POSITION": r["POSITION"],
                    "PLAYER_HISTORY_SAMPLE": r["PLAYER_HISTORY_SAMPLE"], "PRE_REPAIR_SHARE": r["PRE_REPAIR_SHARE"], "POST_REPAIR_SHARE": r["POST_REPAIR_SHARE"],
                    "SHARE_DELTA": r["SHARE_DELTA"], "UNCERTAINTY_FLAG": r["UNCERTAINTY_FLAG"],
                    "SIM_SHARE_V24": rr.get("SIM_CARRY_SHARE" if r["KIND"] == "carry" else "SIM_TARGET_SHARE")}
            if r["UNCERTAINTY_FLAG"]:
                flagged.append(slim)
            if r["POSITION"] == "QB" and r["KIND"] == "carry" and r["SHARE_DELTA"] > 0.02:
                qb_rows.append(slim)
            if r["SHARE_DELTA"] > 0.05:
                top.append(slim)
    key = lambda r: -r["POST_REPAIR_SHARE"]
    return {"FLAGGED_REPLACEMENTS": sorted(flagged, key=key), "QB_CARRY_SHARE_INFLATED_OVER_2PT": sorted(qb_rows, key=lambda r: -r["SHARE_DELTA"]),
            "LARGE_INCREASES_OVER_5PT": sorted(top, key=lambda r: -r["SHARE_DELTA"]),
            "COUNTS": {"FLAGGED": len(flagged), "CONCENTRATION_ABOVE_THRESHOLD": sum("CONCENTRATION_ABOVE_THRESHOLD" in r["UNCERTAINTY_FLAG"] for r in flagged),
                       "THIN_HISTORY_ROLE_GROWTH": sum("THIN_HISTORY_ROLE_GROWTH" in r["UNCERTAINTY_FLAG"] for r in flagged),
                       "QB_CARRY_INFLATED_OVER_2PT": len(qb_rows), "LARGE_INCREASES_OVER_5PT": len(top)}}


def diagnosis(ctx, audits) -> dict:
    """Why did replacement land on quarterbacks?  (A) QB carry history dominates the surviving pool, (B) missing arrivals -- position-filtered, ACT roster only."""
    import pandas as pd
    ros = pd.read_parquet(R.base.PANEL_DIR / "sources" / "roster_weekly_2026.parquet")
    ros = ros[ros.week == ctx["roster_week"]]
    pos = {str(r.gsis_id): r.position for r in ros.itertuples() if r.gsis_id}
    absent = {k: 0 for k in ("QB", "RB", "WR", "TE")}
    rb_absent_teams = []
    for rec in ctx["games"]:
        instate = {p.player_id for p in ctx["inputs"][rec["GAME_ID"]].state.players}
        for team in (rec["AWAY_TEAM"], rec["HOME_TEAM"]):
            miss = {k: [pid for pid, (t, st) in ctx["roster_idx"].items() if t == team and st == "ACT" and pos.get(pid) == k and pid not in instate] for k in absent}
            for k in absent:
                absent[k] += len(miss[k])
            if miss["RB"]:
                rb_absent_teams.append({"TEAM": team, "N_RB_ABSENT": len(miss["RB"]), "NAMES": [ctx["names"].get(p) or str(ros[ros.gsis_id == p].full_name.iloc[0]) for p in miss["RB"]]})
    qb_frac = []
    for gid, a in audits.items():
        for team in sorted({r["TEAM"] for r in a["ROWS"]}):
            surv = [r for r in a["ROWS"] if r["TEAM"] == team and r["KIND"] == "carry" and r["POST_REPAIR_SHARE"] > 0]
            tot = sum(r["PRE_REPAIR_SHARE"] for r in surv) or 1.0
            qb_frac.append({"GAME": gid, "TEAM": team, "QB_FRACTION_OF_SURVIVING_PRE_MASS": sum(r["PRE_REPAIR_SHARE"] for r in surv if r["POSITION"] == "QB") / tot,
                            "QB_POST_CARRY_SHARE": sum(r["POST_REPAIR_SHARE"] for r in surv if r["POSITION"] == "QB")})
    comp = {}
    for gid, team in (("2026_02_IND_KC", "KC"), ("2026_02_JAX_DEN", "JAX")):
        rows = [r for r in audits[gid]["ROWS"] if r["TEAM"] == team and r["KIND"] == "carry"]
        comp[team] = {"PRE_POOL_TOP": [(r["NAME"], r["POSITION"], round(r["PRE_REPAIR_SHARE"], 3), r["REASON_CODE"] or "ELIGIBLE") for r in sorted(rows, key=lambda r: -r["PRE_REPAIR_SHARE"])[:8]],
                      "SURVIVORS": [(r["NAME"], r["POSITION"], int(r["PLAYER_HISTORY_SAMPLE"]), round(r["PRE_REPAIR_SHARE"], 3), round(r["POST_REPAIR_SHARE"], 3))
                                    for r in sorted([r for r in rows if r["POST_REPAIR_SHARE"] > 0], key=lambda r: -r["POST_REPAIR_SHARE"])[:6]]}
    doubtful = [{"GAME": gid, "TEAM": r["TEAM"], "NAME": r["NAME"], "POSITION": r["POSITION"], "KIND": r["KIND"], "PRE_REPAIR_SHARE": r["PRE_REPAIR_SHARE"],
                 "POST_REPAIR_SHARE": r["POST_REPAIR_SHARE"], "AVAILABILITY_WEIGHT": r["AVAILABILITY_WEIGHT"]} for gid, a in audits.items() for r in a["ROWS"] if r["AVAILABILITY_WEIGHT"] != 1.0]
    return {"MECHANISM": "Pro-rata redistribution behaved exactly as specified; the defect is upstream. The frozen carry pool contains QB carry history (scrambles included) which V23 credits AGAIN via "
                         "selection.scrambles, and the ACT running backs that remain have 1-17 games of history and 0.4-2% shares. When departed RBs (KC: Pacheco/Hunt/Edwards-Helaire = 63% of the pool) "
                         "are removed, the QB is the largest remaining historical mass and receives most of it. Missing arrivals are NOT the driver (see ACT_ROSTER_SKILL_PLAYERS_ABSENT_FROM_M1_STATE).",
            "ACT_ROSTER_SKILL_PLAYERS_ABSENT_FROM_M1_STATE": absent, "TEAMS_WITH_ACT_RB_ABSENT_FROM_STATE": rb_absent_teams,
            "QB_MASS_IN_SURVIVING_CARRY_POOL": {"TEAMS_ABOVE_25PCT": sum(1 for r in qb_frac if r["QB_FRACTION_OF_SURVIVING_PRE_MASS"] > 0.25), "N_TEAMS": len(qb_frac),
                                                "TEAMS_QB_POST_SHARE_ABOVE_30PCT": sum(1 for r in qb_frac if r["QB_POST_CARRY_SHARE"] > 0.30),
                                                "MEDIAN_QB_FRACTION": float(np.median([r["QB_FRACTION_OF_SURVIVING_PRE_MASS"] for r in qb_frac])), "PER_TEAM": qb_frac},
            "COMPOSITION_EXAMPLES": comp, "DOUBTFUL_WEIGHT_FIRED_ON_SLATE": doubtful}


def sea_ari(per23, per24, audits, ctx) -> dict:
    gid = "2026_02_SEA_ARI"
    names, ridx = ctx["names"], ctx["roster_idx"]
    by = lambda per, t: {r["PLAYER_ID"]: r for r in per[gid][3]["TEAMS"][t]["PLAYERS"]}
    out = {"GAME_ID": gid, "TEAMS": {}}
    for t in ("ARI", "SEA"):
        a23, a24 = by(per23, t), by(per24, t)
        rows = []
        for pid in sorted(set(a23) | set(a24), key=lambda p: -(a24.get(p, {}).get("SIM_MEAN_CARRIES", 0) + a23.get(p, {}).get("SIM_MEAN_CARRIES", 0))):
            r23, r24 = a23.get(pid, {}), a24.get(pid, {})
            if max(r23.get("SIM_MEAN_CARRIES", 0), r24.get("SIM_MEAN_CARRIES", 0)) < 0.05:
                continue
            aud = next((x for x in audits[gid]["ROWS"] if x["PLAYER_ID"] == pid and x["KIND"] == "carry" and x["TEAM"] == t), None)
            rows.append({"NAME": names.get(pid), "POSITION": (r24 or r23).get("POSITION"), "ROSTER": ridx.get(pid), "V23_MEAN_CARRIES": r23.get("SIM_MEAN_CARRIES", 0.0),
                         "V24_MEAN_CARRIES": r24.get("SIM_MEAN_CARRIES", 0.0), "V23_CARRY_SHARE": r23.get("SIM_CARRY_SHARE", 0.0), "V24_CARRY_SHARE": r24.get("SIM_CARRY_SHARE", 0.0),
                         "PLAYER_HISTORY_SAMPLE": aud["PLAYER_HISTORY_SAMPLE"] if aud else None, "REASON_CODE": (audits[gid]["ROWS"] and next(
                             (x["REASON_CODE"] for x in audits[gid]["ROWS"] if x["PLAYER_ID"] == pid), None)),
                         "UNCERTAINTY_FLAG": aud["UNCERTAINTY_FLAG"] if aud else []})
        out["TEAMS"][t] = {"CARRIES_BY_PLAYER": rows, "POOL_MEAN_CARRIES": {"V23": per23[gid][3]["TEAMS"][t]["SIM_POOL_MEAN_CARRIES"], "V24": per24[gid][3]["TEAMS"][t]["SIM_POOL_MEAN_CARRIES"]},
                           "NON_ACTIVE_CARRY_SHARE": {"V23": per23[gid][3]["TEAMS"][t]["NON_ACTIVE_CARRY_SHARE"], "V24": per24[gid][3]["TEAMS"][t]["NON_ACTIVE_CARRY_SHARE"]},
                           "NON_ACTIVE_TARGET_SHARE": {"V23": per23[gid][3]["TEAMS"][t]["NON_ACTIVE_TARGET_SHARE"], "V24": per24[gid][3]["TEAMS"][t]["NON_ACTIVE_TARGET_SHARE"]}}
    checks = {}
    for nm in ("James Conner", "Trey Benson", "Zach Charbonnet"):
        pid = next((p for p, n in names.items() if n == nm and ridx.get(p, (None,))[0] in ("ARI", "SEA")), None)
        r24 = next((r for t in ("ARI", "SEA") for r in per24[gid][3]["TEAMS"][t]["PLAYERS"] if r["PLAYER_ID"] == pid), None)
        r23 = next((r for t in ("ARI", "SEA") for r in per23[gid][3]["TEAMS"][t]["PLAYERS"] if r["PLAYER_ID"] == pid), None)
        checks[nm] = {"PLAYER_ID": pid, "ROSTER_TEAM_STATUS": ridx.get(pid), "PANEL_MARKS_INELIGIBLE": bool(pid and ridx.get(pid, (None, "ACT"))[1] != "ACT"),
                      "V23_MEAN_CARRIES": r23["SIM_MEAN_CARRIES"] if r23 else None, "V23_MEAN_TARGETS": r23["SIM_MEAN_TARGETS"] if r23 else None,
                      "V24_MEAN_CARRIES": r24["SIM_MEAN_CARRIES"] if r24 else 0.0, "V24_MEAN_TARGETS": r24["SIM_MEAN_TARGETS"] if r24 else 0.0,
                      "V24_P_ANY_TD": r24["SIM_P_ANY_TD"] if r24 else 0.0}
    out["REQUIRED_PLAYER_CHECKS"] = checks
    out["REQUIRED_CHECK_PASS"] = all((c["V24_MEAN_CARRIES"] == 0 and c["V24_MEAN_TARGETS"] == 0) for c in checks.values() if c["PANEL_MARKS_INELIGIBLE"])
    return out


def market_scan(ctx, audits) -> dict:
    """M1_MARKET_CONTAMINATION = key-level firewall violations + market-module imports by the V24 package + vocabulary hits in persisted DATA files."""
    viol = []
    for name, payload in (("ELIGIBILITY_POLICY", cfg.ELIGIBILITY_POLICY), ("DOUBTFUL_POLICY", cfg.DOUBTFUL_POLICY), ("REDISTRIBUTION_POLICY", cfg.REDISTRIBUTION_POLICY),
                          *((f"AUDIT:{g}", a) for g, a in audits.items())):
        try:
            assert_market_free(payload)
        except Exception as e:  # noqa: BLE001
            viol.append({name: str(e)})
    for gid, inp in ctx["inputs"].items():
        try:
            assert_market_free(inp.state.model_dump(mode="json"))
        except Exception as e:  # noqa: BLE001
            viol.append({f"STATE:{gid}": str(e)})
    bad_imports = []
    forbidden = ("sports_market_interface", "sports_research", "market_book", "comparator", "trading_lab", "kalshi", "sportsbook")
    for f in R.pkg_files():
        for node in ast.walk(ast.parse(f.read_text())):
            mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else [n.name for n in node.names] if isinstance(node, ast.Import) else []
            bad_imports += [f"{f.name}:{m}" for m in mods if any(x in m for x in forbidden)]
    vocab = re.compile(r"moneyline|sportsbook|kalshi|odds|betting|vegas|pnl|closing line|implied prob", re.I)
    data_hits = []
    for p in sorted(OUT_DIR.rglob("*")):
        if not p.is_file() or p.suffix == ".npz" and False:
            continue
        rel = str(p.relative_to(OUT_DIR)).replace("\\", "/")
        if p.suffix == ".npz":
            names = list(np.load(p, allow_pickle=False).files)
            data_hits += [f"{rel}:{n}" for n in names if vocab.search(n)]
        elif rel.startswith(("games/", "audit/")) or p.name == "sim_run_meta.json":
            data_hits += [f"{rel}:{h}" for h in vocab.findall(p.read_text(encoding="utf-8", errors="replace"))]
    total = len(viol) + len(bad_imports) + len(data_hits)
    return {"M1_MARKET_CONTAMINATION": total, "FIREWALL_KEY_VIOLATIONS": viol, "MARKET_MODULE_IMPORTS_BY_V24": bad_imports, "VOCAB_HITS_IN_DATA_FILES": data_hits,
            "SCOPE": "policies + 14 audits + 14 repaired-input states (key firewall); V24 package imports; every raw/games/audit/meta file (vocabulary + npz array names)",
            "INPUTS": "panel V2 only (roster_weekly, injuries, causal player-game panel); no market source is read",
            "TERMINAL_PATH_TOUCHES_V24": False, "ABSURD_BOOK_ISOLATION_TEST": "NOT_REQUIRED: terminal not wired to V24 and V24 has no market input surface (simulate_game accepts only PregameState + RosterSnapshot)"}



def qb_rush_realism(per23, per24, ctx) -> dict:
    """POST-HOC diagnostic (added after seeing the replacement tables; disclosed as such): realized mean QB rush attempts vs the 2019-2025 historical range."""
    import pandas as pd
    df, _ = R.base.adapter.load_panel(R.base.PANEL_DIR / "sources" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.live.parquet")
    q = df[(df.SEASON >= 2019) & (df.SEASON <= 2025) & (df.GAME_TYPE == "REG") & (df.POSITION == "QB")]
    g = q.groupby(["PLAYER_ID", "SEASON"]).agg(n=("GAME_ID", "count"), ra=("RUSH_ATTEMPTS", "sum")).reset_index()
    g = g[g.n >= 6]
    pg = g.ra / g.n
    hist = {"QB_SEASONS": int(len(g)), "MEAN": float(pg.mean()), "P99": float(pg.quantile(.99)), "MAX": float(pg.max()), "SAMPLE": "QB-seasons 2019-2025 REG, >=6 games, mean rush attempts per game"}
    def rows(per):
        out = []
        for gid, (_, _, g_, a) in per.items():
            for t in (g_["AWAY"], g_["HOME"]):
                for r in g_["PLAYERS"]:
                    if r["TEAM"] == t and r["POSITION"] == "QB":
                        out.append({"GAME": gid, "TEAM": t, "NAME": ctx["names"].get(r["PLAYER_ID"]), "MEAN_RUSH_ATTEMPTS": r["MEAN"]["rush_attempts"], "MEAN_RUSH_YARDS": r["MEAN"]["rush_yards"], "P_ANY_TD": r["P_ANY_TD"]})
        return out
    r23, r24 = rows(per23), rows(per24)
    teams = lambda rs: {(r["GAME"], r["TEAM"]) for r in rs}
    tot = lambda rs: {k: sum(r["MEAN_RUSH_ATTEMPTS"] for r in rs if (r["GAME"], r["TEAM"]) == k) for k in teams(rs)}
    t23, t24 = tot(r23), tot(r24)
    over = lambda rs: sorted([r for r in rs if r["MEAN_RUSH_ATTEMPTS"] > hist["MAX"]], key=lambda r: -r["MEAN_RUSH_ATTEMPTS"])
    return {"HISTORICAL": hist, "SLATE_MEAN_QB_RUSH_ATTEMPTS_PER_TEAM": {"V23": float(np.mean(list(t23.values()))), "V24": float(np.mean(list(t24.values())))},
            "QBS_ABOVE_HISTORICAL_MAX": {"V23": over(r23), "V24": over(r24)},
            "TOP8_V24": sorted(r24, key=lambda r: -r["MEAN_RUSH_ATTEMPTS"])[:8], "TOP8_V23": sorted(r23, key=lambda r: -r["MEAN_RUSH_ATTEMPTS"])[:8],
            "CAVEAT": "post-hoc: added after inspecting the replacement tables; used only to inform the promotion judgement, not to alter any frozen rule. "
                      "PLAYER_HISTORY_SAMPLE for QBs is the 4-game recency window used by make_state, so THIN_HISTORY_ROLE_GROWTH flags on QBs are confounded by that construction; "
                      "the QB flag is kept as frozen and the realized-usage comparison here is the cleaner evidence."}


def promotion(result, qbr, fb) -> dict:
    n23, n24 = len(qbr["QBS_ABOVE_HISTORICAL_MAX"]["V23"]), len(qbr["QBS_ABOVE_HISTORICAL_MAX"]["V24"])
    frozen = bool(result["TECHNICAL_PASS"] and result["VALIDITY_GAIN_FROZEN_RULE"] and not result["REGRESSION_TRIGGERED"])
    new_absurd = n24 > n23
    promote = bool(frozen and not new_absurd)
    return {"PROMOTE_V24": "YES" if promote else "NO", "FROZEN_RULES_SATISFIED": frozen,
            "JUDGEMENT_OVERRIDE": bool(frozen and not promote),
            "REASON": ("all frozen promotion rules satisfied and no new football-invalid replacement" if promote else
                       f"roster validity is fully repaired and team-level football behaviour is unchanged (frozen rules satisfied: {frozen}), but the frozen pro-rata redistribution "
                       f"pushes removed carry mass onto quarterbacks: {n24} QBs exceed the historical maximum QB rush-attempt rate (V23: {n23}), slate mean QB rush attempts per team "
                       f"{qbr['SLATE_MEAN_QB_RUSH_ATTEMPTS_PER_TEAM']['V23']:.1f} -> {qbr['SLATE_MEAN_QB_RUSH_ATTEMPTS_PER_TEAM']['V24']:.1f}. The brief requires replacement allocation to be materially "
                       "more football-valid than V23; that is not established, so V24 is NOT promoted. This is a conservative override of the mechanically-satisfied frozen rule, not a rule edit."),
            "V23_REMAINS_CHAMPION": not promote}


def code_manifest() -> dict:
    files = {}
    for f in R.pkg_files() + sorted((ROOT / R.PKG / "tests").glob("*.py")):
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        tree = ast.parse(f.read_text())
        files[rel] = {"SHA256": R.sha256_file(f), "LINES": len(f.read_text().splitlines()),
                      "FUNCTIONS": [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]}
    return files


def code_patch() -> str:
    parts = []
    for f in R.pkg_files() + sorted((ROOT / R.PKG / "tests").glob("*.py")):
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        parts.append("".join(difflib.unified_diff([], f.read_text().splitlines(True), "/dev/null", f"b/{rel}")))
    return "\n".join(parts)


def cmd_analyze() -> None:
    ctx = base.load_inputs()
    assert ctx["raw_sha"] == base.PANEL_SHA_EXPECTED
    freeze = json.loads(R.FREEZE_FILE.read_text())
    meta = json.loads((OUT_DIR / "sim_run_meta.json").read_text())
    now_hashes, before = base.engine_hashes(), json.loads(R.V23_BEFORE.read_text())
    v23_files_now = {k: R.sha256_file(ROOT / k) for k in before["FILES"]}
    v23_unchanged = v23_files_now == before["FILES"] and bool(meta.get("ENGINE_HASHES_UNCHANGED_DURING_SIM"))
    man = json.loads((base.PANEL_DIR / "MANIFEST.json").read_text())
    v23_panel_match = all(now_hashes.get(k) == v for k, v in man["M1_ADAPTER_CODE_SHA256"].items() if "sports_nova_v23" in k)
    v24_unchanged_since_freeze = R.v24_hash() == freeze["V24_HASH"]

    per23, per24 = load_dir(V23_DIR, ctx), load_dir(OUT_DIR, ctx)
    a23, a24 = aggregate(per23), aggregate(per24)
    published = json.loads((V23_DIR / "SUNDAY_2026_09_20_M1_BASELINE_V1.json").read_text()) if (V23_DIR / "SUNDAY_2026_09_20_M1_BASELINE_V1.json").exists() else {}
    reproduced = (a23["INVALID_PLAYER_USAGE"] == 615 and a23["MEANINGFUL_INVALID_USAGE"] == 276 and round(a23["AVG_NONACTIVE_SHARE"], 3) == 0.312
                  and round(a23["MAX_NONACTIVE_SHARE"], 3) == 0.810 and a23["SIM_CLASS"] == {"SIM_GREEN": 0, "SIM_YELLOW": 6, "SIM_RED": 8})
    split23, split24 = usage_split(per23), usage_split(per24)
    acc23, acc24 = accounting(V23_DIR, ctx), accounting(OUT_DIR, ctx)
    audits = load_audits()
    fb = football(per23, per24, ctx)
    repl = replacement_tables(audits, per24)
    sea = sea_ari(per23, per24, audits, ctx)
    scan = market_scan(ctx, audits)
    qbr = qb_rush_realism(per23, per24, ctx)
    diag = diagnosis(ctx, audits)

    fallbacks = sorted({f for a in audits.values() for f in a["ZERO_SURVIVOR_FALLBACKS"]})
    summ_ok = all(abs(s["SUM_SHARE_PLUS_RESIDUAL_AFTER"] - 1) <= cfg.SHARE_TOLERANCE and (not s["N_SURVIVORS"] or abs(s["EFFECTIVE_SUM_AFTER"] - 1) <= cfg.SHARE_TOLERANCE)
                  and abs(s["REMOVED_MASS"] - s["REDISTRIBUTED_MASS"]) <= cfg.SHARE_TOLERANCE for a in audits.values() for s in a["SUMMARIES"])
    mass_ok = summ_ok and not fallbacks and acc24["STATUS"] == "PASS"
    qb_inputs_ok = all(a["QB_INPUTS_UNCHANGED"] for a in audits.values())
    qb_rows = lambda per: [(q["TEAM"], q["MATCH"], q["ENGINE_PRIMARY_QB_ID"], q["ENGINE_IDENTITY_INSTALLED"]) for g in per for q in per[g][3]["QB_INTEGRITY"]]
    qb_same = qb_rows(per23) == qb_rows(per24)
    qb_unchanged = qb_inputs_ok and qb_same and v23_unchanged
    cat = a24["INVALID_BY_CATEGORY"]
    gates = {"INVALID_INELIGIBLE_PLAYER_USAGE": a24["INVALID_PLAYER_USAGE"], "WRONG_TEAM_PLAYER_USAGE": cat.get("ON_OTHER_TEAM_ROSTER", 0),
             "OUT_PLAYER_USAGE": a24["PANEL_OUT_USED"], "IR_INELIGIBLE_USAGE": cat.get("RES", 0),
             "TEAM_PLAYER_ACCOUNTING": acc24["STATUS"], "OPPORTUNITY_MASS_CONSERVATION": "PASS" if mass_ok else "FAIL",
             "QB_LOGIC_UNCHANGED": qb_unchanged, "M1_MARKET_CONTAMINATION": scan["M1_MARKET_CONTAMINATION"], "V23_UNCHANGED": v23_unchanged}
    technical_pass = (gates["INVALID_INELIGIBLE_PLAYER_USAGE"] == 0 and gates["WRONG_TEAM_PLAYER_USAGE"] == 0 and gates["OUT_PLAYER_USAGE"] == 0 and gates["IR_INELIGIBLE_USAGE"] == 0
                      and gates["TEAM_PLAYER_ACCOUNTING"] == "PASS" and gates["OPPORTUNITY_MASS_CONSERVATION"] == "PASS" and qb_unchanged and gates["M1_MARKET_CONTAMINATION"] == 0
                      and v23_unchanged and v24_unchanged_since_freeze)
    reduction = 1.0 - a24["AVG_NONACTIVE_SHARE"] / a23["AVG_NONACTIVE_SHARE"]
    validity_gain = reduction >= 0.99 and acc24["STATUS"] == "PASS"
    regression = bool(fb["REGRESSION_TRIGGERS"])
    promo = promotion(result_pre := {"TECHNICAL_PASS": technical_pass, "VALIDITY_GAIN_FROZEN_RULE": validity_gain, "REGRESSION_TRIGGERED": regression}, qbr, fb)
    now = datetime.now(timezone.utc).isoformat()
    tests = subprocess.run([sys.executable, "-m", "pytest", f"{R.PKG}/tests", "-q"], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()[-1]
    m = re.search(r"(\d+) passed", tests)
    n_pass, n_fail = int(m.group(1)) if m else 0, int((re.search(r"(\d+) failed", tests) or [0, 0])[1])

    result = {"GATES": gates, "TECHNICAL_PASS": technical_pass, "VALIDITY_GAIN_FROZEN_RULE": validity_gain, "AVG_NONACTIVE_SHARE_REDUCTION": reduction,
              "REGRESSION_TRIGGERED": regression, "V23_HARNESS_REPRODUCED": reproduced, "V24_UNCHANGED_SINCE_FREEZE": v24_unchanged_since_freeze, "PROMOTION": promo, "QB_RUSH_REALISM_POST_HOC": qbr}
    common = {"CREATED_AT": now, "V24_VERSION": cfg.MODEL_VERSION, "V24_HASH": R.v24_hash(), "IMPLEMENTATION_PACKAGE": R.PKG, "PANEL_SHA256": ctx["raw_sha"],
              "SEED_POLICY": base.SEED_POLICY, "N_SIMS_PER_GAME": sorted({g["N_SIMS"] for _, _, g, _ in per24.values()}),
              "ELIGIBILITY_POLICY_HASH": cfg.ELIGIBILITY_POLICY_HASH, "REDISTRIBUTION_POLICY_HASH": cfg.REDISTRIBUTION_POLICY_HASH, "DOUBTFUL_POLICY_HASH": cfg.DOUBTFUL_POLICY_HASH,
              "POLICY_FREEZE_SHA256": R.sha256_file(R.FREEZE_FILE), "POLICY_FROZEN_AT": freeze["FROZEN_AT"], "SIM_STARTED_AT": meta["STARTED_AT"]}
    fnames = {k: v["FUNCTIONS"] for k, v in code_manifest().items() if "/tests/" not in k}
    sea_v = lambda per, t: per["2026_02_SEA_ARI"][3]["TEAMS"][t]
    top_repl = {t: sorted([r for r in sea["TEAMS"][t]["CARRIES_BY_PLAYER"] if r["V24_CARRY_SHARE"] > 0.05], key=lambda r: -r["V24_CARRY_SHARE"]) for t in ("ARI", "SEA")}
    qbmass = diag["QB_MASS_IN_SURVIVING_CARRY_POOL"]
    OUTPUT = {"SPORTS_NOVA_V24_STATUS": "PARTIAL" if technical_pass and promo["PROMOTE_V24"] == "NO" else ("PASS" if technical_pass else "FAIL"),
              "STATUS_MEANING": "PARTIAL = every technical gate passes (roster validity, accounting, QB/market/V23 integrity) but V24 is not promoted: replacement allocation is not more football-valid than V23",
              "V23_UNCHANGED": v23_unchanged, "V24_VERSION": cfg.MODEL_VERSION, "V24_HASH": R.v24_hash(),
              "FILES_CHANGED": {"MODIFIED": [], "ADDED": sorted(fnames) + ["scripts/sports_nova_m1_v24_roster_eligibility.py", "scripts/sports_nova_m1_v24_roster_eligibility_analysis.py", "scripts/sports_nova_v24_policy_derivation.py"]},
              "FUNCTIONS_CHANGED": {"MODIFIED_EXISTING": [], "NEW": fnames},
              "ELIGIBILITY_POLICY_HASH": cfg.ELIGIBILITY_POLICY_HASH, "REDISTRIBUTION_POLICY_HASH": cfg.REDISTRIBUTION_POLICY_HASH, "DOUBTFUL_POLICY_HASH": cfg.DOUBTFUL_POLICY_HASH,
              "TESTS_PASS": n_pass, "TESTS_FAIL": n_fail,
              "INVALID_PLAYER_USAGE_V23": a23["INVALID_PLAYER_USAGE"], "INVALID_PLAYER_USAGE_V24": a24["INVALID_PLAYER_USAGE"],
              "MEANINGFUL_INVALID_USAGE_V23": a23["MEANINGFUL_INVALID_USAGE"], "MEANINGFUL_INVALID_USAGE_V24": a24["MEANINGFUL_INVALID_USAGE"],
              "AVG_NONACTIVE_SHARE_V23": a23["AVG_NONACTIVE_SHARE"], "AVG_NONACTIVE_SHARE_V24": a24["AVG_NONACTIVE_SHARE"],
              "MAX_NONACTIVE_SHARE_V23": a23["MAX_NONACTIVE_SHARE"], "MAX_NONACTIVE_SHARE_V24": a24["MAX_NONACTIVE_SHARE"],
              "NONACTIVE_METRIC_CAVEAT": "measures the fix by its own definition (V24 zeroes exactly the set the metric counts); independent evidence is the 56/56 all-player accounting pass, the football deltas inside bootstrap SE, and the QB realism check",
              "SEA_ARI_V23_CONTAMINATION": {t: {"NON_ACTIVE_CARRY_SHARE": sea_v(per23, t)["NON_ACTIVE_CARRY_SHARE"], "NON_ACTIVE_TARGET_SHARE": sea_v(per23, t)["NON_ACTIVE_TARGET_SHARE"]} for t in ("ARI", "SEA")},
              "SEA_ARI_V24_CONTAMINATION": {t: {"NON_ACTIVE_CARRY_SHARE": sea_v(per24, t)["NON_ACTIVE_CARRY_SHARE"], "NON_ACTIVE_TARGET_SHARE": sea_v(per24, t)["NON_ACTIVE_TARGET_SHARE"]} for t in ("ARI", "SEA")},
              "SEA_ARI_REPLACEMENT_CONCENTRATION": {t: [{"NAME": r["NAME"], "POSITION": r["POSITION"], "V24_CARRY_SHARE": r["V24_CARRY_SHARE"], "HISTORY_GAMES": r["PLAYER_HISTORY_SAMPLE"], "FLAGS": r["UNCERTAINTY_FLAG"]} for r in top_repl[t]] for t in top_repl},
              "SIM_GREEN_V24": a24["SIM_CLASS"]["SIM_GREEN"], "SIM_YELLOW_V24": a24["SIM_CLASS"]["SIM_YELLOW"], "SIM_RED_V24": a24["SIM_CLASS"]["SIM_RED"],
              "SIM_CLASS_CAVEAT": "sim_class() does not read opportunity concentration or QB carry realism, so the SIM_GREEN count overstates football validity",
              "SIM_GREEN_WITH_QB_ABOVE_HISTORICAL_MAX": sorted({r["GAME"] for r in qbr["QBS_ABOVE_HISTORICAL_MAX"]["V24"] if per24[r["GAME"]][3]["SIM_CLASS"] == "SIM_GREEN"}),
              "QB_MATCH_COUNT": a24["QB_MATCH"], "QB_MISMATCH_COUNT": a24["QB_MISMATCH"], "CIN_HOU": "UNRESOLVED: Burrow GAME_TIME_DECISION/LOW, engine fallback, not turned into confirmed starter truth",
              "TEAM_PLAYER_ACCOUNTING": gates["TEAM_PLAYER_ACCOUNTING"], "OPPORTUNITY_MASS_CONSERVATION": gates["OPPORTUNITY_MASS_CONSERVATION"], "M1_MARKET_CONTAMINATION": gates["M1_MARKET_CONTAMINATION"],
              "TOTAL_DISTRIBUTION_CHANGE": fb["TOTAL_DISTRIBUTION_CHANGE"], "MARGIN_DISTRIBUTION_CHANGE": fb["MARGIN_DISTRIBUTION_CHANGE"], "SLATE_MEAN_OF_MEDIAN_TOTALS": fb["SLATE_MEAN_OF_MEDIAN_TOTALS"],
              "PROMOTE_V24": promo["PROMOTE_V24"],
              "REAL_BLOCKER": "Replacement mass lands on quarterbacks: the frozen carry pool holds QB carry history (already double-counted by V23's separate scramble credit) and the surviving ACT RBs have 1-17 games / 0.4-2% shares, "
                              f"so the QB is the largest surviving historical mass (>25% of it) on {qbmass['TEAMS_ABOVE_25PCT']}/{qbmass['N_TEAMS']} teams (KC: Mahomes 24.5 mean carries vs 7.7 in V23; historical QB-season max 11.7). "
                              "Missing arrivals are not the cause (13 ACT RBs absent slate-wide; KC 1, a fullback).",
              "NEXT_SINGLE_ACTION": "Pre-register and build V25 = V24 with QBs excluded as RECIPIENTS of removed designed-carry mass (QB carry share held at its pre-repair value; QB rushing stays on the existing scramble path), "
                                    "redistributing removed RB/WR/TE mass pro-rata among eligible non-QB players only; rerun the same 14 games and compare V25 vs V24 vs V23 on QB rush attempts, accounting and the frozen regression rules. "
                                    "Treat the V23 QB carry/scramble double-count (slate mean 8.3 QB rush attempts per team vs 3.5 historical) as a separate, later fix.",
              "HISTORICAL_VALIDATION": "DATA_LIMITED: no point-in-time roster-status ledger; V24 not validated against outcomes"}
    result["OUTPUT"] = OUTPUT
    result["DIAGNOSIS"] = diag
    per_game = {gid: {"GAME": f"{per24[gid][2]['AWAY']}@{per24[gid][2]['HOME']}", "V23_SIM_CLASS": per23[gid][3]["SIM_CLASS"], "V24_SIM_CLASS": per24[gid][3]["SIM_CLASS"],
                      "V24_SIM_CLASS_REASONS": per24[gid][3]["SIM_CLASS_REASONS"], "V24_CONTAMINATION_CLASS": per24[gid][3]["CONTAMINATION_CLASS"],
                      "V23_MAX_NONACTIVE": per23[gid][3]["MAX_TEAM_NONACTIVE_SHARE"], "V24_MAX_NONACTIVE": per24[gid][3]["MAX_TEAM_NONACTIVE_SHARE"],
                      "V24_INVALID_USAGE": per24[gid][3]["INVALID_PLAYER_USAGE_COUNT"], "INPUT_HASH_OK": per24[gid][3]["INPUT_HASH_OK"],
                      "EXCLUDED_BY_REASON": {k: len(v) for k, v in audits[gid]["EXCLUDED_BY_REASON"].items()}} for gid in per24}

    (OUT_DIR / "V24_CODE_DIFF.patch").write_text(code_patch(), encoding="utf-8")
    manifest = {"SCHEMA": "SPORTS_NOVA_V24_VERSION_MANIFEST", **common, "BASELINE_V23_VERSION": "sports_nova_v23.drive_block.residual_allocation_fix.1",
                "V23_UNCHANGED": v23_unchanged, "V23_FILE_SHA256_BEFORE": before["FILES"], "V23_FILE_SHA256_AFTER": v23_files_now, "V23_ENGINE_HASH_BEFORE": before["V23_ENGINE_HASH"],
                "V23_MATCHES_PANEL_MANIFEST": v23_panel_match, "V24_FILES": code_manifest(), "V24_ENGINE_LOADED_FILE_HASHES": now_hashes,
                "FILES_CHANGED": {"V23_ENGINE_FILES_MODIFIED": [], "NEW_V24_PACKAGE_FILES": sorted(k for k in code_manifest() if "/tests/" not in k),
                                  "NEW_TEST_FILES": sorted(k for k in code_manifest() if "/tests/" in k), "NEW_RUNNER_FILES": ["scripts/sports_nova_m1_v24_roster_eligibility.py",
                                  "scripts/sports_nova_m1_v24_roster_eligibility_analysis.py", "scripts/sports_nova_v24_policy_derivation.py"]},
                "EXACT_DIFF": {"FILE": "V24_CODE_DIFF.patch", "SHA256": R.sha256_file(OUT_DIR / "V24_CODE_DIFF.patch"),
                               "NOTE": "V24 is a NEW package; every line is added. No V23/V21/V19/V3 file is edited (hashes equal before/after). V24 calls V23 simulate_game unchanged on a share-repaired PregameState."},
                "UNCHANGED_BY_CONSTRUCTION": ["play volume", "play selection", "efficiency/yardage draws", "scoring", "QB selection (_qb_shares)", "RNG contract", "calibration"],
                "UNIT_TESTS": {"SUMMARY": tests, "PASS": n_pass, "FAIL": n_fail},
                "RUNNER_FILE_SHA256": {f: R.sha256_file(ROOT / f) for f in ("scripts/sports_nova_m1_v24_roster_eligibility.py", "scripts/sports_nova_m1_v24_roster_eligibility_analysis.py", "scripts/sports_nova_v24_policy_derivation.py")},
                "RUNNER_NOTE": "the sim runner gained a `resume` mode after the freeze (a background shell kill left 2 of 14 games unrun; per-game sims are independent and seeded, so resuming is equivalent). V24_HASH covers the engine package only.",
                "CONCURRENT_IMPLEMENTATION_NOTE": "A Codex agent was concurrently building a different implementation under worker/sports_nova_v24/ with the same version string. "
                                                  "This implementation lives in worker/sports_nova_v24_roster_eligibility/ (relative imports; rename-safe). Codex's earlier attempt is archived at "
                                                  "data/sports_nova_v3/sunday_2026_09_20/ARCHIVE_CODEX_V24_PRIOR_ATTEMPT/ with sha256 lists."}
    report = {"SCHEMA": "SPORTS_NOVA_V24_ROSTER_ELIGIBILITY_REPORT", **common, **result, "V23_AGGREGATE": a23, "V24_AGGREGATE": a24,
              "INVALID_USAGE_SPLIT": {"V23": {k: v for k, v in split23.items() if k != "ROWS"}, "V24": split24},
              "ACCOUNTING": {"V23": acc23, "V24": acc24}, "ZERO_SURVIVOR_FALLBACKS": fallbacks, "QB": {"INPUTS_UNCHANGED_ALL_GAMES": qb_inputs_ok, "QB_ROWS_IDENTICAL_V23_V24": qb_same,
                             "QB_MATCH": a24["QB_MATCH"], "QB_MISMATCH": a24["QB_MISMATCH"], "QB_MATCH_VIA_ENGINE_FALLBACK": a24["QB_MATCH_VIA_FALLBACK"],
                             "CIN_HOU": [q for q in per24["2026_02_CIN_HOU"][3]["QB_INTEGRITY"]]},
              "SEA_ARI": sea, "REPLACEMENT": repl,  "MARKET": scan,
              "EXCLUSIONS_BY_REASON_SLATE": {r: sum(len(a["EXCLUDED_BY_REASON"].get(r, [])) for a in audits.values()) for r in cfg.REASON_CODES},
              "HISTORICAL_VALIDATION": "DATA_LIMITED: no point-in-time historical roster-status ledger exists in the repo, so V24 has NOT been validated against outcomes; "
                                       "this run measures roster validity + accounting + simulator side effects only",
              "FREEZE": {"REGRESSION_RULES": R.REGRESSION_RULES, "PROMOTION_RULES": R.PROMOTION_RULES}}
    baseline = {"SCHEMA": "SUNDAY_2026_09_20_V24_BASELINE", **common, "OUTPUT": OUTPUT, "AGGREGATE": a24, "GATES": gates, "PROMOTION": promo, "PER_GAME": per_game,
                "FOOTBALL": {k: fb[k] for k in ("SLATE_MEAN_OF_MEDIAN_TOTALS", "TOTAL_DISTRIBUTION_CHANGE", "MARGIN_DISTRIBUTION_CHANGE")},
                "RAW_NOTE": "raw M1-derived research outputs; NOT calibrated, NOT actionable; no market comparison performed"}
    for nm, obj in (("SPORTS_NOVA_V24_ROSTER_ELIGIBILITY_REPORT.json", report), ("SUNDAY_2026_09_20_V24_BASELINE.json", baseline), ("V24_VERSION_MANIFEST.json", manifest)):
        (OUT_DIR / nm).write_text(json.dumps(obj, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    (OUT_DIR / "FOOTBALL_COMPARISON.json").write_text(json.dumps(fb, indent=1, default=float), encoding="utf-8")
    (OUT_DIR / "SUNDAY_2026_09_20_V23_V24_DIFF.md").write_text(render_md(common, a23, a24, gates, per_game, fb, repl, sea, split24, result, reproduced), encoding="utf-8")
    print(json.dumps({"GATES": gates, "TECHNICAL_PASS": technical_pass, "V23": {k: a23[k] for k in ("INVALID_PLAYER_USAGE", "MEANINGFUL_INVALID_USAGE", "AVG_NONACTIVE_SHARE", "MAX_NONACTIVE_SHARE", "SIM_CLASS")},
                      "V24": {k: a24[k] for k in ("INVALID_PLAYER_USAGE", "MEANINGFUL_INVALID_USAGE", "AVG_NONACTIVE_SHARE", "MAX_NONACTIVE_SHARE", "SIM_CLASS", "QB_MATCH", "QB_MISMATCH")},
                      "REPRODUCED_V23": reproduced, "REPL_COUNTS": repl["COUNTS"], "REGRESSION_TRIGGERS": len(fb["REGRESSION_TRIGGERS"]),
                      "SLATE_TOTALS": fb["SLATE_MEAN_OF_MEDIAN_TOTALS"], "SEA_ARI_REQUIRED_PASS": sea["REQUIRED_CHECK_PASS"], "TESTS": [n_pass, n_fail],
                      "USAGE_SPLIT_V24": {k: v for k, v in split24.items() if k != "ROWS"}}, indent=1, default=float))


def pct(x): return f"{100 * x:.1f}%"


def render_md(common, a23, a24, gates, per_game, fb, repl, sea, split24, result, reproduced) -> str:
    promo, qbr = result["PROMOTION"], result["QB_RUSH_REALISM_POST_HOC"]
    O = result["OUTPUT"]
    L = [f"# SPORTS_NOVA V24 roster-eligibility repair: V23 vs V24 (Sunday 2026-09-20, panel V2, N=5000/game)", "",
         f"## READ THIS FIRST: status **{O['SPORTS_NOVA_V24_STATUS']}**, PROMOTE_V24 = **{O['PROMOTE_V24']}**", "",
         "V24 fully repairs roster validity and passes every technical gate, but it is **not** promoted: removed carry mass lands on quarterbacks. "
         f"In IND@KC, Mahomes goes to 24.5 mean carries (V23: 7.7; historical QB-season max 11.7). The SIM green count below does not see this: {O['SIM_CLASS_CAVEAT']}. "
         f"SIM_GREEN games that still contain a QB above the historical maximum: {O['SIM_GREEN_WITH_QB_ABOVE_HISTORICAL_MAX']}.", "",
         f"**Real blocker:** {O['REAL_BLOCKER']}", "", f"**Next single action:** {O['NEXT_SINGLE_ACTION']}", "",
         f"V24 `{common['V24_VERSION']}` hash `{common['V24_HASH']}`; package `{common['IMPLEMENTATION_PACKAGE']}`. Policies frozen {common['POLICY_FROZEN_AT']} (before the sim started {common['SIM_STARTED_AT']}).",
         f"V23 harness reproduced from persisted files: **{reproduced}** (615 / 276 / 31.2% / 81.0% / 0-6-8). No market data used or compared. Raw research outputs, not calibrated, not actionable.", "",
         "## Contamination", "", "| metric | V23 | V24 |", "|---|---|---|"]
    for k, lab in (("INVALID_PLAYER_USAGE", "invalid player usage (count)"), ("MEANINGFUL_INVALID_USAGE", "meaningful invalid usage")):
        L.append(f"| {lab} | {a23[k]} | {a24[k]} |")
    L += [f"| avg non-active share (team combined; measures the fix by its own definition: V24 zeroes exactly this set) | {pct(a23['AVG_NONACTIVE_SHARE'])} | {pct(a24['AVG_NONACTIVE_SHARE'])} |",
          f"| max non-active pool share | {pct(a23['MAX_NONACTIVE_SHARE'])} ({a23['WORST_GAME']}) | {pct(a24['MAX_NONACTIVE_SHARE'])} ({a24['WORST_GAME']}) |",
          f"| SIM green/yellow/red (does not read concentration or QB realism) | {a23['SIM_CLASS']['SIM_GREEN']}/{a23['SIM_CLASS']['SIM_YELLOW']}/{a23['SIM_CLASS']['SIM_RED']} | {a24['SIM_CLASS']['SIM_GREEN']}/{a24['SIM_CLASS']['SIM_YELLOW']}/{a24['SIM_CLASS']['SIM_RED']} |",
          f"| QB match / mismatch | {a23['QB_MATCH']}/{a23['QB_MISMATCH']} | {a24['QB_MATCH']}/{a24['QB_MISMATCH']} |", "",
          f"V24 remaining invalid usage by mechanism: allocation (carries/targets) {split24['ALLOCATION_USAGE_COUNT']}; QB-selection pass attempts only {split24['QB_SELECTION_ONLY_COUNT']} (QB logic is out of scope).", "",
          "## Gates", "", "| gate | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in gates.items()]
    L += ["", f"Technical pass: **{result['TECHNICAL_PASS']}**. Frozen validity-gain rule: {result['VALIDITY_GAIN_FROZEN_RULE']} (avg non-active share reduction {pct(result['AVG_NONACTIVE_SHARE_REDUCTION'])}).", "",
          "## Per game", "", "| game | V23 | V24 | V24 max non-active | V24 invalid | exclusions |", "|---|---|---|---|---|---|"]
    for gid, r in per_game.items():
        L.append(f"| {r['GAME']} | {r['V23_SIM_CLASS']} | {r['V24_SIM_CLASS']} | {pct(r['V24_MAX_NONACTIVE'])} | {r['V24_INVALID_USAGE']} | {r['EXCLUDED_BY_REASON']} |")
    L += ["", "## Football behaviour (V24 minus V23; SE = bootstrap SE of the median difference)", "",
          f"Slate mean of median totals: V23 {fb['SLATE_MEAN_OF_MEDIAN_TOTALS']['V23']:.2f} -> V24 {fb['SLATE_MEAN_OF_MEDIAN_TOTALS']['V24']:.2f} (delta {fb['SLATE_MEAN_OF_MEDIAN_TOTALS']['DELTA']:+.2f}). "
          f"Total delta: mean {fb['TOTAL_DISTRIBUTION_CHANGE']['MEAN_DELTA']:+.2f}, mean |delta| {fb['TOTAL_DISTRIBUTION_CHANGE']['MEAN_ABS_DELTA']:.2f}, max |delta| {fb['TOTAL_DISTRIBUTION_CHANGE']['MAX_ABS_DELTA']:.1f}, typical SE {fb['TOTAL_DISTRIBUTION_CHANGE']['MEAN_SE']:.2f}. "
          f"Margin delta: mean |delta| {fb['MARGIN_DISTRIBUTION_CHANGE']['MEAN_ABS_DELTA']:.2f}, max {fb['MARGIN_DISTRIBUTION_CHANGE']['MAX_ABS_DELTA']:.1f}, typical SE {fb['MARGIN_DISTRIBUTION_CHANGE']['MEAN_SE']:.2f}.",
          f"Regression triggers (frozen rules): **{len(fb['REGRESSION_TRIGGERS'])}** {fb['REGRESSION_TRIGGERS'][:6]}", "",
          "| game | total V23 -> V24 (d, SE) | margin V23 -> V24 (d, SE) |", "|---|---|---|"]
    for r in fb["GAMES"]:
        L.append(f"| {r['GAME']} | {r['TOTAL']['V23_MEDIAN']:.0f} -> {r['TOTAL']['V24_MEDIAN']:.0f} ({r['TOTAL']['DELTA']:+.1f}, {r['TOTAL']['SE_OF_DELTA']:.2f}) | {r['MARGIN']['V23_MEDIAN']:+.0f} -> {r['MARGIN']['V24_MEDIAN']:+.0f} ({r['MARGIN']['DELTA']:+.1f}, {r['MARGIN']['SE_OF_DELTA']:.2f}) |")
    L += ["", "## SEA@ARI targeted regression (carries by player, mean per sim)", ""]
    for t, blk in sea["TEAMS"].items():
        L += [f"**{t}** non-active carry share V23 {pct(blk['NON_ACTIVE_CARRY_SHARE']['V23'])} -> V24 {pct(blk['NON_ACTIVE_CARRY_SHARE']['V24'])}", "",
              "| player | pos | roster | V23 carries (share) | V24 carries (share) | history games | flags |", "|---|---|---|---|---|---|---|"]
        for r in blk["CARRIES_BY_PLAYER"]:
            L.append(f"| {r['NAME']} | {r['POSITION']} | {r['ROSTER']} | {r['V23_MEAN_CARRIES']:.2f} ({pct(r['V23_CARRY_SHARE'])}) | {r['V24_MEAN_CARRIES']:.2f} ({pct(r['V24_CARRY_SHARE'])}) | {r['PLAYER_HISTORY_SAMPLE']} | {', '.join(r['UNCERTAINTY_FLAG'])} |")
        L.append("")
    L += ["Required checks (panel-ineligible => zero V24 usage): " + json.dumps({k: [v["V24_MEAN_CARRIES"], v["V24_MEAN_TARGETS"]] for k, v in sea["REQUIRED_PLAYER_CHECKS"].items()}) + f" -> {sea['REQUIRED_CHECK_PASS']}", "",
          "## Flagged replacements (not capped)", "", f"Counts: {repl['COUNTS']}", "", "| game | team | kind | player | pos | history games | pre | post | delta | flags |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in repl["FLAGGED_REPLACEMENTS"][:40]:
        L.append(f"| {r['GAME'][8:]} | {r['TEAM']} | {r['KIND']} | {r['NAME']} | {r['POSITION']} | {r['PLAYER_HISTORY_SAMPLE']:.0f} | {pct(r['PRE_REPAIR_SHARE'])} | {pct(r['POST_REPAIR_SHARE'])} | {r['SHARE_DELTA']:+.3f} | {', '.join(r['UNCERTAINTY_FLAG'])} |")
    L += ["", "## Diagnosis (why replacement lands on QBs)", "", result["DIAGNOSIS"]["MECHANISM"], "",
          f"QB share of surviving carry mass above 25% on {result['DIAGNOSIS']['QB_MASS_IN_SURVIVING_CARRY_POOL']['TEAMS_ABOVE_25PCT']}/{result['DIAGNOSIS']['QB_MASS_IN_SURVIVING_CARRY_POOL']['N_TEAMS']} teams; "
          f"ACT roster skill players absent from the M1 state (position-filtered): {result['DIAGNOSIS']['ACT_ROSTER_SKILL_PLAYERS_ABSENT_FROM_M1_STATE']}.", "",
          "Doubtful weight (0.025) fired on: " + "; ".join(f"{d['NAME']} {d['KIND']} {d['PRE_REPAIR_SHARE']:.3f}->{d['POST_REPAIR_SHARE']:.4f}" for d in result["DIAGNOSIS"]["DOUBTFUL_WEIGHT_FIRED_ON_SLATE"]), "",
          "## Promotion", "", f"**PROMOTE_V24 = {promo['PROMOTE_V24']}** (frozen rules satisfied: {promo['FROZEN_RULES_SATISFIED']}; judgement override: {promo['JUDGEMENT_OVERRIDE']})", "", promo["REASON"], "",
          f"QB rush realism (POST-HOC diagnostic): historical QB-season mean rush attempts per game 2019-2025: mean {qbr['HISTORICAL']['MEAN']:.1f}, max {qbr['HISTORICAL']['MAX']:.1f}. "
          f"Slate mean QB rush attempts per team V23 {qbr['SLATE_MEAN_QB_RUSH_ATTEMPTS_PER_TEAM']['V23']:.1f} -> V24 {qbr['SLATE_MEAN_QB_RUSH_ATTEMPTS_PER_TEAM']['V24']:.1f}.", "",
          "| QB (V24 top) | team | V24 mean rush att | V24 rush yds | P(any TD) |", "|---|---|---|---|---|"] + [f"| {r['NAME']} | {r['TEAM']} | {r['MEAN_RUSH_ATTEMPTS']:.1f} | {r['MEAN_RUSH_YARDS']:.0f} | {r['P_ANY_TD']:.2f} |" for r in qbr["TOP8_V24"]] + ["", qbr["CAVEAT"]]
    L += ["", "### QB carry-share inflation (QB position, carry share up by more than 2 points)", "", "| game | team | QB | history games | pre | post |", "|---|---|---|---|---|---|"]
    for r in repl["QB_CARRY_SHARE_INFLATED_OVER_2PT"][:30]:
        L.append(f"| {r['GAME'][8:]} | {r['TEAM']} | {r['NAME']} | {r['PLAYER_HISTORY_SAMPLE']:.0f} | {pct(r['PRE_REPAIR_SHARE'])} | {pct(r['POST_REPAIR_SHARE'])} |")
    return "\n".join(L) + "\n"
