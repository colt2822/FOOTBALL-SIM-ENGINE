"""V23 vs V24 vs V25 replay analysis + mirror-gate decision.  Imported by sports_nova_m1_v25_role_aware.py `analyze`.

Measurement is identical for all three versions (base.analyze_game / base.sim_class).  All rules used here are in SPORTS_NOVA_V25_PREREG.json,
frozen before any V25 simulation output existed.
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

import sports_nova_m1_v24_roster_eligibility_analysis as A
import sports_nova_m1_v25_role_aware as R25
from worker.sports_nova_modular.firewall import assert_market_free

base, cfg, ROOT = R25.base, R25.cfg, R25.ROOT
OUT_DIR, V23_DIR, V24_DIR = R25.OUT_DIR, R25.V23_DIR, R25.V24_DIR
ENV = R25.PATH_CLASSIFICATION_RULES["ENVELOPES"]
QB_MAX = ENV["QB_MEAN_RUSH_ATT_PER_GAME_HISTORICAL_MAX"]
TOL = 1e-9


def audits_of(d: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted((d / "audit").glob("*.json"))}


def raw(d: Path, gid: str):
    return np.load(d / "raw" / f"{gid}.npz", allow_pickle=False)


def player_rows(per, ctx):
    """flat rows: version-independent per-player sim summary."""
    out = {}
    for gid, (_, _, g, a) in per.items():
        for t in (g["AWAY"], g["HOME"]):
            for r in a["TEAMS"][t]["PLAYERS"]:
                out[(gid, t, r["PLAYER_ID"])] = r
    return out


def qb_table(per, ctx):
    rows = []
    for gid, (_, _, g, a) in per.items():
        for t in (g["AWAY"], g["HOME"]):
            team_rush = g["TEAM"][t]["rush_attempts"]["mean"]
            for r in g["PLAYERS"]:
                if r["TEAM"] == t and r["POSITION"] == "QB" and r["MEAN"]["rush_attempts"] > 0.005:
                    rows.append({"GAME": gid, "TEAM": t, "PLAYER_ID": r["PLAYER_ID"], "NAME": ctx["names"].get(r["PLAYER_ID"]),
                                 "MEAN_RUSH_ATTEMPTS": r["MEAN"]["rush_attempts"], "TEAM_MEAN_RUSH_ATTEMPTS": team_rush,
                                 "QB_SHARE_OF_TEAM_RUSH": r["MEAN"]["rush_attempts"] / team_rush if team_rush else 0.0})
    return rows


def mean_se(z, pid, key):
    j = [str(x) for x in z["player_ids"]].index(pid)
    v = z[key][:, j].astype(float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def classify_paths(ctx, per23, per25, aud25, aud24_unused=None):
    out = []
    names = ctx["names"]
    for team, kind, nm in R25.EIGHT_PATHS:
        gid = next((g for g, (_, _, gg, _) in per25.items() if team in (gg["AWAY"], gg["HOME"])), None)
        pid = next((r["PLAYER_ID"] for r in aud25[gid]["ROWS"] if r["TEAM"] == team and r["NAME"] == nm), None)
        row = next((r for r in aud25[gid]["ROWS"] if r["PLAYER_ID"] == pid and r["KIND"] == kind and r["TEAM"] == team), None)
        rec = {"PATH": f"{team}:{kind}:{nm}", "GAME": gid, "PLAYER_ID": pid}
        if row is None:
            rec.update({"STATUS": "UNSUPPORTED", "WHY": "player has no share in that pool in the V25 audit"})
            out.append(rec)
            continue
        z23, z25 = raw(V23_DIR, gid), raw(OUT_DIR, gid)
        key = "player_rush_attempts" if kind == "carry" else "player_targets"
        m23, s23 = mean_se(z23, pid, key)
        m25, s25 = mean_se(z25, pid, key)
        env = ENV["TEAM_SEASON_MAX_SINGLE_PLAYER_SHARE_P99"][kind]
        post, pre, v23e, v24e = row["V25_POST_SHARE"], row["PRE_FULL_POOL_SHARE"], row["V23_EFFECTIVE_SHARE"], row["V24_EQUIVALENT_SHARE"]
        thin = row["PLAYER_HISTORY_SAMPLE"] < 8
        held = row["ROLE_CLASS"] == "HELD"
        se = float(np.hypot(s23, s25))
        if held:
            resolved = post <= pre + TOL and m25 <= m23 + 2 * se
            status = "MECHANICALLY_RESOLVED" if resolved else "UNSUPPORTED"
            why = ("QB held at its no-removal share (received none of the removed non-QB mass); V25 sim mean <= V23 mean + 2SE" if resolved else
                   "held role above its no-removal share or above V23 sim mean")
        elif row["ELIGIBLE"] and post <= env and not thin:
            status, why = "SUPPORTED", "recipient role, eligible, state share inside the historical single-player envelope, history >= 8 games"
        elif row["ELIGIBLE"]:
            status = "AMBIGUOUS"
            why = "; ".join(x for x in (f"state share {post:.3f} > envelope {env}" if post > env else "", f"thin history ({row['PLAYER_HISTORY_SAMPLE']:.0f} games)" if thin else "") if x)
            why += " -- role-valid pro-rata mass on a short surviving pool; V25 does not change this"
        else:
            status, why = "UNSUPPORTED", "ineligible player"
        rec.update({"ROLE": row["ROLE"], "ROLE_CLASS": row["ROLE_CLASS"], "STATUS": status, "WHY": why, "HISTORY_SAMPLE": row["PLAYER_HISTORY_SAMPLE"],
                    "STATE_SHARE": {"PRE_FULL_POOL": pre, "V23_EFFECTIVE": v23e, "V24_EQUIVALENT": v24e, "V25_POST": post},
                    "SIM_MEAN": {"V23": m23, "V25": m25, "SE_DIFF": se, "STAT": key},
                    "SIM_SHARE_V25": next((r["SIM_CARRY_SHARE" if kind == "carry" else "SIM_TARGET_SHARE"] for r in per25[gid][3]["TEAMS"][team]["PLAYERS"] if r["PLAYER_ID"] == pid), None),
                    "ENVELOPE": env, "V24_STATE_SHARE_WAS_ABOVE_ENVELOPE": v24e > env})
        out.append(rec)
    return out


def new_concentration(aud25):
    new, held_up = [], []
    for gid, au in aud25.items():
        for r in au["ROWS"]:
            env = ENV["TEAM_SEASON_MAX_SINGLE_PLAYER_SHARE_P99"][r["KIND"]]
            if r["ROLE_CLASS"] == "HELD" and r["V25_POST_SHARE"] > r["PRE_FULL_POOL_SHARE"] + TOL:
                held_up.append({"GAME": gid, **{k: r[k] for k in ("TEAM", "KIND", "NAME", "ROLE")}})
            if r["ROLE_CLASS"] == "RECIPIENT" and r["ELIGIBLE"] and r["V25_POST_SHARE"] > env and r["V24_EQUIVALENT_SHARE"] <= env:
                new.append({"GAME": gid, **{k: r[k] for k in ("TEAM", "KIND", "NAME", "ROLE", "PLAYER_HISTORY_SAMPLE")}, "V24_EQ": r["V24_EQUIVALENT_SHARE"], "V25": r["V25_POST_SHARE"], "ENVELOPE": env})
    return {"NEW_SEVERE_CONCENTRATION": new, "HELD_ROLE_INCREASES": held_up}


def top_concentrations(per23, per24, per25, ctx, kind, n=10):
    skey = "SIM_CARRY_SHARE" if kind == "carry" else "SIM_TARGET_SHARE"
    idx = [player_rows(p, ctx) for p in (per23, per24, per25)]
    rows = []
    for k, r in idx[2].items():
        rows.append({"GAME": k[0], "TEAM": k[1], "NAME": r["NAME"], "POSITION": r["POSITION"], "V23": idx[0].get(k, {}).get(skey, 0.0), "V24": idx[1].get(k, {}).get(skey, 0.0), "V25": r[skey]})
    return sorted(rows, key=lambda r: -r["V25"])[:n]


def sea_ari(per23, per24, per25, aud25, ctx):
    gid = "2026_02_SEA_ARI"
    ridx, names = ctx["roster_idx"], ctx["names"]
    out = {"GAME_ID": gid, "TEAMS": {}}
    for t in ("ARI", "SEA"):
        rows = []
        pl = [player_rows(p, ctx) for p in (per23, per24, per25)]
        audit_rows = [r for r in aud25[gid]["ROWS"] if r["TEAM"] == t]
        pids = {r["PLAYER_ID"] for r in audit_rows} | {k[2] for p in pl for k in p if k[0] == gid and k[1] == t}
        for pid in pids:
            vals = [p.get((gid, t, pid), {}) for p in pl]
            if max(v.get("SIM_MEAN_CARRIES", 0) + v.get("SIM_MEAN_TARGETS", 0) for v in vals) < 0.05 and not any(r["PLAYER_ID"] == pid and (r["V25_POST_SHARE"] > 0.005) for r in audit_rows):
                continue
            cr = next((r for r in audit_rows if r["PLAYER_ID"] == pid and r["KIND"] == "carry"), {})
            tr = next((r for r in audit_rows if r["PLAYER_ID"] == pid and r["KIND"] == "target"), {})
            rows.append({"NAME": names.get(pid), "POSITION": next((v["POSITION"] for v in vals if v), None), "ROLE": (cr or tr).get("ROLE"), "ROSTER": ridx.get(pid),
                         "HISTORY_SAMPLE": (cr or tr).get("PLAYER_HISTORY_SAMPLE"),
                         "CARRY": {"STATE_PRE_FULL": cr.get("PRE_FULL_POOL_SHARE", 0.0), "STATE_V23": cr.get("V23_EFFECTIVE_SHARE", 0.0), "STATE_V24": cr.get("V24_EQUIVALENT_SHARE", 0.0), "STATE_V25": cr.get("V25_POST_SHARE", 0.0),
                                   "SIM_MEAN": [v.get("SIM_MEAN_CARRIES", 0.0) for v in vals], "SIM_SHARE": [v.get("SIM_CARRY_SHARE", 0.0) for v in vals]},
                         "TARGET": {"STATE_PRE_FULL": tr.get("PRE_FULL_POOL_SHARE", 0.0), "STATE_V23": tr.get("V23_EFFECTIVE_SHARE", 0.0), "STATE_V24": tr.get("V24_EQUIVALENT_SHARE", 0.0), "STATE_V25": tr.get("V25_POST_SHARE", 0.0),
                                    "SIM_MEAN": [v.get("SIM_MEAN_TARGETS", 0.0) for v in vals], "SIM_SHARE": [v.get("SIM_TARGET_SHARE", 0.0) for v in vals]},
                         "REASON_CODE": (cr or tr).get("REASON_CODE"), "P_ANY_TD_V25": vals[2].get("SIM_P_ANY_TD", 0.0)})
        out["TEAMS"][t] = sorted(rows, key=lambda r: -(r["CARRY"]["SIM_MEAN"][2] + r["TARGET"]["SIM_MEAN"][2]))
    out["SIM_MEAN_ORDER"] = "[V23, V24, V25]"
    return out


def football(per23, per25):
    rng = np.random.default_rng(20260920)
    games = []
    for gid in per23:
        g23, g25 = per23[gid][2], per25[gid][2]
        z23, z25 = raw(V23_DIR, gid), raw(OUT_DIR, gid)
        h = list(z23["team_ids"]).index(g23["HOME"]), list(z23["team_ids"]).index(g23["AWAY"])
        tot = lambda z: z["team_score"][:, h[0]] + z["team_score"][:, h[1]]
        mar = lambda z: z["team_score"][:, h[0]] - z["team_score"][:, h[1]]
        row = {"GAME": f"{g23['AWAY']}@{g23['HOME']}", "GAME_ID": gid}
        for nm, f, key in (("TOTAL", tot, "TOTAL_POINTS"), ("MARGIN", mar, "MARGIN_HOME_MINUS_AWAY")):
            row[nm] = {"V23_MEDIAN": g23[key]["median"], "V25_MEDIAN": g25[key]["median"], "DELTA": g25[key]["median"] - g23[key]["median"],
                       "SE_OF_DELTA": A.median_se(f(z23), f(z25), rng), "V23_MEAN": g23[key]["mean"], "V25_MEAN": g25[key]["mean"]}
        row["HOME_WIN_PROB"] = {"V23": g23["HOME_WIN_PROB_TIE_SPLIT"], "V25": g25["HOME_WIN_PROB_TIE_SPLIT"]}
        row["TEAM_ATTEMPTS_MEAN"] = {t: {k: {"V23": g23["TEAM"][t][k]["mean"], "V25": g25["TEAM"][t][k]["mean"], "PCT": 100.0 * (g25["TEAM"][t][k]["mean"] / g23["TEAM"][t][k]["mean"] - 1.0)}
                                         for k in ("pass_attempts", "rush_attempts")} for t in (g23["AWAY"], g23["HOME"])}
        games.append(row)
    rules = R25.REGRESSION_RULES
    m23, m25 = float(np.mean([r["TOTAL"]["V23_MEDIAN"] for r in games])), float(np.mean([r["TOTAL"]["V25_MEDIAN"] for r in games]))
    trig = []
    if m23 - m25 > rules["SLATE_MEAN_OF_MEDIAN_TOTALS_DROP_POINTS"]:
        trig.append({"RULE": "SLATE_MEAN_OF_MEDIAN_TOTALS_DROP", "V23": m23, "V25": m25})
    for r in games:
        if abs(r["TOTAL"]["DELTA"]) > rules["GAME_MEDIAN_TOTAL_SHIFT_POINTS"]:
            trig.append({"RULE": "GAME_MEDIAN_TOTAL_SHIFT", "GAME": r["GAME"], "DELTA": r["TOTAL"]["DELTA"], "SE": r["TOTAL"]["SE_OF_DELTA"]})
        for t, d in r["TEAM_ATTEMPTS_MEAN"].items():
            for k, v in d.items():
                if abs(v["PCT"]) > rules["TEAM_MEAN_ATTEMPTS_SHIFT_PCT"]:
                    trig.append({"RULE": "TEAM_MEAN_ATTEMPTS_SHIFT", "GAME": r["GAME"], "TEAM": t, "STAT": k, "PCT": v["PCT"]})
    dt = [r["TOTAL"]["DELTA"] for r in games]
    dw = [r["HOME_WIN_PROB"]["V25"] - r["HOME_WIN_PROB"]["V23"] for r in games]
    return {"GAMES": games, "SLATE_MEAN_OF_MEDIAN_TOTALS": {"V23": m23, "V25": m25, "DELTA": m25 - m23},
            "TOTAL_DISTRIBUTION_CHANGE": {"MEAN_DELTA": float(np.mean(dt)), "MAX_ABS_DELTA": float(np.max(np.abs(dt))), "MEAN_SE": float(np.mean([r["TOTAL"]["SE_OF_DELTA"] for r in games]))},
            "HOME_WIN_PROB_CHANGE": {"MEAN_ABS": float(np.mean(np.abs(dw))), "MAX_ABS": float(np.max(np.abs(dw)))},
            "REGRESSION_TRIGGERS": trig}


def market_scan(ctx, audits):
    viol = []
    for name, payload in (("ELIGIBILITY_POLICY", cfg.ELIGIBILITY_POLICY), ("DOUBTFUL_POLICY", cfg.DOUBTFUL_POLICY), ("ROLE_POLICY", cfg.ROLE_POLICY),
                          ("REDISTRIBUTION_POLICY", cfg.REDISTRIBUTION_POLICY), *((f"AUDIT:{g}", a) for g, a in audits.items())):
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
    for f in R25.pkg_files():
        for node in ast.walk(ast.parse(f.read_text())):
            mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else [n.name for n in node.names] if isinstance(node, ast.Import) else []
            bad_imports += [f"{f.name}:{m}" for m in mods if any(x in m for x in forbidden)]
    vocab = re.compile(r"moneyline|sportsbook|kalshi|odds|betting|vegas|pnl|closing line|implied prob", re.I)
    hits = []
    for p in sorted(OUT_DIR.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(OUT_DIR)).replace("\\", "/")
        if p.suffix == ".npz":
            hits += [f"{rel}:{n}" for n in np.load(p, allow_pickle=False).files if vocab.search(n)]
        elif rel.startswith(("games/", "audit/")) or p.name == "sim_run_meta.json":
            hits += [f"{rel}:{h}" for h in vocab.findall(p.read_text(encoding="utf-8", errors="replace"))]
    return {"M1_MARKET_CONTAMINATION": len(viol) + len(bad_imports) + len(hits), "FIREWALL_KEY_VIOLATIONS": viol, "MARKET_MODULE_IMPORTS_BY_V25": bad_imports, "VOCAB_HITS_IN_DATA_FILES": hits}


def cmd_analyze() -> None:
    ctx = base.load_inputs()
    meta = json.loads((OUT_DIR / "sim_run_meta.json").read_text())
    prereg = json.loads(R25.PREREG_FILE.read_text())
    assert prereg["V25_HASH"] == R25.v25_hash() == meta["V25_HASH"], "V25 package changed since freeze"
    per23, per24, per25 = (A.load_dir(d, ctx) for d in (V23_DIR, V24_DIR, OUT_DIR))
    ag = [A.aggregate(p) for p in (per23, per24, per25)]
    acc = [A.accounting(d, ctx) for d in (V23_DIR, V24_DIR, OUT_DIR)]
    aud25 = audits_of(OUT_DIR)
    scan = market_scan(ctx, aud25)
    reproduced = (ag[0]["INVALID_PLAYER_USAGE"], ag[0]["MEANINGFUL_INVALID_USAGE"]) == (615, 276)
    # mass conservation over every pool of every game
    cons_fail = [s for au in aud25.values() for s in au["SUMMARIES"]
                 if abs(s["POST_REDISTRIBUTION_TOTAL"] - s["PRE_REMOVAL_TOTAL"]) > 1e-9 or abs(s["REMOVED_MASS"] - s["REDISTRIBUTED_MASS"]) > 1e-9 or s["MASS_TO_HELD_ROLES"] > 1e-9]
    cons = "PASS" if not cons_fail else "FAIL"
    qb23, qb24, qb25 = (qb_table(p, ctx) for p in (per23, per24, per25))
    over = lambda rs: sorted([r for r in rs if r["MEAN_RUSH_ATTEMPTS"] > QB_MAX], key=lambda r: -r["MEAN_RUSH_ATTEMPTS"])
    key = lambda r: (r["GAME"], r["TEAM"], r["PLAYER_ID"])
    d23 = {key(r): r for r in qb23}
    infl = [{"NAME": r["NAME"], "TEAM": r["TEAM"], "V23": d23.get(key(r), {}).get("MEAN_RUSH_ATTEMPTS", 0.0), "V25": r["MEAN_RUSH_ATTEMPTS"],
             "DELTA": r["MEAN_RUSH_ATTEMPTS"] - d23.get(key(r), {}).get("MEAN_RUSH_ATTEMPTS", 0.0)} for r in qb25]
    team_tot = lambda rs: float(np.mean(list({(r["GAME"], r["TEAM"]): r["QB_SHARE_OF_TEAM_RUSH"] for r in rs}.values())))
    worst = max(qb25, key=lambda r: r["MEAN_RUSH_ATTEMPTS"])
    worst_share = max(qb25, key=lambda r: r["QB_SHARE_OF_TEAM_RUSH"])
    state_qb = {"QB_STATE_CARRY_INCREASES_VS_NO_REMOVAL": sum(1 for au in aud25.values() for r in au["ROWS"] if r["ROLE"] == "QB" and r["KIND"] == "carry" and r["V25_POST_SHARE"] > r["PRE_FULL_POOL_SHARE"] + TOL),
                "QB_STATE_CARRY_INCREASES_V24_VS_NO_REMOVAL": sum(1 for au in aud25.values() for r in au["ROWS"] if r["ROLE"] == "QB" and r["KIND"] == "carry" and r["V24_EQUIVALENT_SHARE"] > r["PRE_FULL_POOL_SHARE"] + 0.02)}
    qbr = {"HISTORICAL_MAX_MEAN_RUSH_ATT": QB_MAX,
           "SLATE_MEAN_QB_RUSH_ATTEMPTS_PER_TEAM": {"V23": float(np.mean([sum(r["MEAN_RUSH_ATTEMPTS"] for r in qb23 if (r["GAME"], r["TEAM"]) == k) for k in {(r["GAME"], r["TEAM"]) for r in qb23}])),
                                                     "V24": float(np.mean([sum(r["MEAN_RUSH_ATTEMPTS"] for r in qb24 if (r["GAME"], r["TEAM"]) == k) for k in {(r["GAME"], r["TEAM"]) for r in qb24}])),
                                                     "V25": float(np.mean([sum(r["MEAN_RUSH_ATTEMPTS"] for r in qb25 if (r["GAME"], r["TEAM"]) == k) for k in {(r["GAME"], r["TEAM"]) for r in qb25}]))},
           "QB_SHARE_OF_TEAM_RUSH_MEAN": {"V23": team_tot(qb23), "V24": team_tot(qb24), "V25": team_tot(qb25)},
           "QBS_ABOVE_HISTORICAL_MAX": {"V23": [(r["NAME"], round(r["MEAN_RUSH_ATTEMPTS"], 2)) for r in over(qb23)], "V24": [(r["NAME"], round(r["MEAN_RUSH_ATTEMPTS"], 2)) for r in over(qb24)],
                                        "V25": [(r["NAME"], round(r["MEAN_RUSH_ATTEMPTS"], 2)) for r in over(qb25)]},
           "WORST_QB_V25": {"NAME": worst["NAME"], "TEAM": worst["TEAM"], "MEAN_RUSH_ATTEMPTS": worst["MEAN_RUSH_ATTEMPTS"]},
           "WORST_QB_SHARE_V25": {"NAME": worst_share["NAME"], "TEAM": worst_share["TEAM"], "SHARE_OF_TEAM_RUSH": worst_share["QB_SHARE_OF_TEAM_RUSH"]},
           "QB_CARRY_MEAN_BY_PLAYER": sorted([{"NAME": r["NAME"], "TEAM": r["TEAM"], "V23": d23.get(key(r), {}).get("MEAN_RUSH_ATTEMPTS", 0.0),
                                               "V24": next((x["MEAN_RUSH_ATTEMPTS"] for x in qb24 if key(x) == key(r)), 0.0), "V25": r["MEAN_RUSH_ATTEMPTS"]} for r in qb25], key=lambda r: -r["V25"]),
           "QB_CARRY_SHARE_BY_TEAM": sorted([{"GAME": r["GAME"], "TEAM": r["TEAM"], "NAME": r["NAME"], "V23": d23.get(key(r), {}).get("QB_SHARE_OF_TEAM_RUSH", 0.0),
                                              "V24": next((x["QB_SHARE_OF_TEAM_RUSH"] for x in qb24 if key(x) == key(r)), 0.0), "V25": r["QB_SHARE_OF_TEAM_RUSH"]} for r in qb25], key=lambda r: -r["V25"]),
           "QB_CARRY_INFLATION_VS_V23": sorted(infl, key=lambda r: -r["DELTA"]),
           "STATE_LEVEL": state_qb,
           "SEVERE_FLAGS_V25": [f"{r['NAME']} {r['MEAN_RUSH_ATTEMPTS']:.1f} > {QB_MAX:.1f}" for r in over(qb25)],
           "CAVEATS": ["QB_CARRY_INFLATION_VS_V23 ~ 0 is TRUE BY CONSTRUCTION (QB held at its no-removal share); the independent evidence is QBS_ABOVE_HISTORICAL_MAX V24 -> V25 and where the mass landed",
                       "V25 does NOT fix V23's inherited QB double count (designed-carry share incl. scrambles + scrambles credited again): slate QB rush attempts per team stay above the 3.5 historical mean"]}
    # where the removed mass landed (independent evidence): state-level, recipient side
    landed = {"REMOVED_CARRY_MASS_TO_QB_V24": float(sum(max(0.0, r["V24_EQUIVALENT_SHARE"] - r["PRE_FULL_POOL_SHARE"]) for au in aud25.values() for r in au["ROWS"] if r["ROLE"] == "QB" and r["KIND"] == "carry")),
              "REMOVED_CARRY_MASS_TO_QB_V25": float(sum(max(0.0, r["V25_POST_SHARE"] - r["PRE_FULL_POOL_SHARE"]) for au in aud25.values() for r in au["ROWS"] if r["ROLE"] == "QB" and r["KIND"] == "carry"))}
    paths = classify_paths(ctx, per23, per25, aud25)
    pcount = {k: sum(1 for p in paths if p["STATUS"] == k) for k in ("SUPPORTED", "MECHANICALLY_RESOLVED", "AMBIGUOUS", "UNSUPPORTED")}
    nc = new_concentration(aud25)
    fb = football(per23, per25)
    sea = sea_ari(per23, per24, per25, aud25, ctx)
    tops = {"TOP10_CARRY_CONCENTRATIONS": top_concentrations(per23, per24, per25, ctx, "carry"), "TOP10_TARGET_CONCENTRATIONS": top_concentrations(per23, per24, per25, ctx, "target")}
    # gates
    tech = {"invalid_usage": ag[2]["INVALID_PLAYER_USAGE"], "meaningful_invalid_usage": ag[2]["MEANINGFUL_INVALID_USAGE"], "nonactive_share": ag[2]["AVG_NONACTIVE_SHARE"],
            "mass_conservation": cons, "accounting": acc[2]["STATUS"], "qb_match": ag[2]["QB_MATCH"], "qb_mismatch": ag[2]["QB_MISMATCH"], "market_contamination": scan["M1_MARKET_CONTAMINATION"]}
    tech_ok = tech["invalid_usage"] == 0 and tech["meaningful_invalid_usage"] == 0 and cons == "PASS" and tech["accounting"] == "PASS" and tech["qb_mismatch"] == 0 and tech["market_contamination"] == 0
    qb_removed = state_qb["QB_STATE_CARRY_INCREASES_VS_NO_REMOVAL"] == 0 and not nc["HELD_ROLE_INCREASES"] and len(over(qb25)) <= len(over(qb23))
    paths_ok = all(p["STATUS"] in ("SUPPORTED", "MECHANICALLY_RESOLVED") for p in paths)
    prop_ready = bool(tech_ok and paths_ok and not nc["NEW_SEVERE_CONCENTRATION"] and qb_removed)
    game_ready = bool(tech_ok and qb_removed and not [n for n in nc["NEW_SEVERE_CONCENTRATION"] if n["ROLE"] not in cfg.RECIPIENT_ROLES] and not fb["REGRESSION_TRIGGERS"])
    tests = subprocess.run([sys.executable, "-m", "pytest", f"{R25.PKG}/tests", "-q"], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    gates = {"TECHNICAL": tech, "TECHNICAL_OK": tech_ok, "PATHS_ALL_SUPPORTED_OR_RESOLVED": paths_ok, "NEW_SEVERE_CONCENTRATION_COUNT": len(nc["NEW_SEVERE_CONCENTRATION"]), "QB_REDISTRIBUTION_INFLATION_REMOVED": qb_removed,
             "REGRESSION_TRIGGERS": fb["REGRESSION_TRIGGERS"], "SPORTS_NOVA_RESEARCH_MIRROR_READY": "YES" if prop_ready else "NO", "GAME_LEVEL_MIRROR_READY": "YES" if game_ready else "NO"}
    report = {"SCHEMA": "SPORTS_NOVA_V25_REPORT", "CREATED_AT": datetime.now(timezone.utc).isoformat(), "V25_VERSION": cfg.MODEL_VERSION, "V25_HASH": R25.v25_hash(), "PREREG_SHA256": R25.sha256_file(R25.PREREG_FILE),
              "V23_HARNESS_REPRODUCED": reproduced, "AGGREGATE": {"V23": ag[0], "V24": ag[1], "V25": ag[2]}, "ACCOUNTING": {"V23": acc[0]["STATUS"], "V24": acc[1]["STATUS"], "V25": acc[2]["STATUS"], "V25_CHECKS": acc[2]["CHECKS"], "V25_FAILURES": acc[2]["FAILURES"]},
              "MASS_CONSERVATION": {"STATUS": cons, "POOLS_CHECKED": sum(len(a["SUMMARIES"]) for a in aud25.values()), "FAILURES": cons_fail[:5]},
              "MARKET": scan, "QB_REALISM": qbr, "MASS_LANDING": landed, "EIGHT_PATHS": paths, "EIGHT_PATH_COUNTS": pcount, "NEW_CONCENTRATION": nc, **tops, "SEA_ARI": sea, "FOOTBALL_V25_VS_V23": fb,
              "POOLS_CHANGED": sum(1 for a in aud25.values() for s in a["SUMMARIES"] if s["CHANGED"]), "POOLS_TOTAL": sum(len(a["SUMMARIES"]) for a in aud25.values()), "GATES": gates,
              "TESTS": tests[-1] if tests else "", "ENGINE_HASHES_UNCHANGED_DURING_SIM": meta.get("ENGINE_HASHES_UNCHANGED_DURING_SIM"), "V23_REMAINS_CHAMPION": True, "V25_HISTORICAL_PROMOTION": False, "LIVE_CAPITAL_AUTHORIZED": False}
    (OUT_DIR / "SPORTS_NOVA_V25_REPORT.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8")
    (OUT_DIR / "SPORTS_NOVA_V25_PREREG.json").write_text(R25.PREREG_FILE.read_text(), encoding="utf-8")
    print(json.dumps({"TECH": tech, "GATES": {k: v for k, v in gates.items() if k not in ("TECHNICAL", "REGRESSION_TRIGGERS")}, "PATHS": pcount,
                      "QB": {k: qbr[k] for k in ("SLATE_MEAN_QB_RUSH_ATTEMPTS_PER_TEAM", "QBS_ABOVE_HISTORICAL_MAX", "WORST_QB_V25", "WORST_QB_SHARE_V25", "STATE_LEVEL")},
                      "LANDED": landed, "REPRO": reproduced, "TESTS": report["TESTS"]}, indent=1, default=str))
