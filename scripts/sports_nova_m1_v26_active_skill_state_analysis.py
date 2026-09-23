"""V23 vs V24 vs V25 vs V26 replay analysis + player-prop / game-level gate decision.  Imported by sports_nova_m1_v26_active_skill_state.py `analyze`.

Measurement is identical for all four versions (base.analyze_game / base.sim_class).  Every rule used here is in SPORTS_NOVA_V26_PREREG.json,
frozen before any V26 simulation output existed.  The portable-share counterfactual is arithmetic on the state only: it is never simulated and never a V26 output.
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

import sports_nova_m1_v24_roster_eligibility_analysis as A
import sports_nova_m1_v25_role_aware_analysis as A25
import sports_nova_m1_v26_active_skill_state as R26
from worker.sports_nova_modular.firewall import assert_market_free
from worker.sports_nova_sunday_panel import adapter
from worker.sports_nova_v25_role_aware.eligibility import game_eligibility
from worker.sports_nova_v25_role_aware.redistribution import assert_repair_invariants, repair_pregame_state
from worker.sports_nova_v26_active_skill_state.completion import expected_active_skill
from worker.sports_nova_v26_active_skill_state.config import RECIPIENT_ROLES

R25, base, cfg, ROOT = R26.R25, R26.base, R26.cfg, R26.ROOT
OUT_DIR, V23_DIR, V24_DIR, V25_DIR = R26.OUT_DIR, R26.V23_DIR, R26.V24_DIR, R26.V25_DIR
ENV = R25.PATH_CLASSIFICATION_RULES["ENVELOPES"]
ENV_SHARE = ENV["TEAM_SEASON_MAX_SINGLE_PLAYER_SHARE_P99"]
QB_MAX = ENV["QB_MEAN_RUSH_ATT_PER_GAME_HISTORICAL_MAX"]
TOL = 1e-9
NAMED = [("MIA", "target", "De'Von Achane"), ("ARI", "carry", "Bam Knight"), ("SEA", "carry", "George Holani"), ("HOU", "carry", "Woody Marks"), ("WAS", "target", "Terry McLaurin")]


# ---------------------------------------------------------------------------------------------- loading
def make_ctx26(ctx, comps):
    inputs = {}
    for gid, inp in ctx["inputs"].items():
        inputs[gid] = adapter.M1GameInput(gid, comps[gid][2].state if gid in comps else inp.state, inp.resolutions, inp.missing, inp.notes, inp.alignment)
    return dict(ctx, inputs=inputs)


def load_dir26(d, ctx, ctx26):
    """Like A.load_dir, but shares/positions come from the COMPLETED state while the input-hash check keeps using the RAW state (what the batch reports)."""
    per = {}
    for rec in ctx["games"]:
        gid = rec["GAME_ID"]
        g = json.loads((d / "games" / f"{gid}.json").read_text())
        a = base.analyze_game(rec, ctx26["inputs"][gid], g, ctx26)
        smoke = rec["M1_ADAPTER"]["SMOKE"].get("M1_STATE_HASH")
        hash_ok = base._state_hash(ctx["inputs"][gid].state) == smoke == g["STATE_HASH"]
        sc, reasons = base.sim_class(rec["DATA_QUALITY"]["CLASS"], a, a["QB_INTEGRITY"], hash_ok, len(a["PANEL_OUT_PLAYERS_WITH_SIM_USAGE"]))
        a.update({"SIM_CLASS": sc, "SIM_CLASS_REASONS": reasons, "INPUT_HASH_OK": hash_ok})
        per[gid] = (rec, ctx["inputs"][gid], g, a)
    return per


# ---------------------------------------------------------------------------------------------- V26 vs V25 sim identity
def sim_identity(ctx):
    out = {}
    for rec in ctx["games"]:
        gid = rec["GAME_ID"]
        z25, z26 = A25.raw(V25_DIR, gid), A25.raw(OUT_DIR, gid)
        ids25, ids26 = [str(x) for x in z25["player_ids"]], [str(x) for x in z26["player_ids"]]
        common = [i for i in ids25 if i in set(ids26)]
        i25, i26 = [ids25.index(i) for i in common], [ids26.index(i) for i in common]
        diffs, identical = {}, True
        for k in z25.files:
            if k in ("player_ids", "team_ids", "winners"):
                d = 0.0 if np.array_equal(z25[k], z26[k]) else 1.0
            elif k.startswith("player_"):
                d = float(np.abs(z25[k][:, i25].astype(float) - z26[k][:, i26].astype(float)).max())
            else:
                d = float(np.abs(z25[k].astype(float) - z26[k].astype(float)).max())
            diffs[k] = d
            identical &= d == 0.0
        extra = [i for i in ids26 if i not in set(ids25)]
        extra_usage = float(sum(np.abs(z26[k][:, [ids26.index(i) for i in extra]].astype(float)).sum() for k in z26.files if k.startswith("player_") and k != "player_ids" and extra))
        out[gid] = {"BIT_IDENTICAL_COMMON_PLAYERS_AND_TEAMS": bool(identical), "MAX_ABS_DIFF": max(diffs.values()), "ADDED_PLAYER_COUNT_IN_OUTPUT": len(extra),
                    "ADDED_PLAYERS_TOTAL_SIMULATED_USAGE": extra_usage,
                    "DIFFERING_KEYS": [k for k, v in diffs.items() if v != 0.0]}
    return out


# ---------------------------------------------------------------------------------------------- thin pools
def historical_reference(panel_df):
    df = panel_df[(panel_df.SEASON.between(2019, 2025)) & (panel_df.GAME_TYPE == "REG") & (panel_df.POSITION.isin(RECIPIENT_ROLES))]
    out = {}
    for kind, col in (("carry", "RUSH_ATTEMPTS"), ("target", "TARGETS")):
        tops, top2s, effs = [], [], []
        for _, g in df.groupby(["SEASON", "TEAM"]):
            v = g.groupby("PLAYER_ID")[col].sum().clip(lower=0).to_numpy(dtype=float)
            if v.sum() <= 0:
                continue
            s = np.sort(v / v.sum())[::-1]
            tops.append(s[0]); top2s.append(s[:2].sum()); effs.append(1.0 / float((s ** 2).sum()))
        q = lambda a, p: float(np.quantile(a, p))
        out[kind] = {"TEAM_SEASONS": len(tops), "TOP_SHARE": {"median": q(tops, .5), "p90": q(tops, .9), "p99": q(tops, .99)}, "TOP_2_SHARE": {"median": q(top2s, .5), "p90": q(top2s, .9)},
                     "EFFECTIVE_POOL_SIZE": {"median": q(effs, .5), "p10": q(effs, .1), "min": float(min(effs))}}
    return out


def pool_table(aud, states_snaps, hist):
    """one row per game x team x kind from a version's audit rows (V25_POST_SHARE = the share handed to the V23 draw)."""
    rows = []
    for gid, au in aud.items():
        state, snap = states_snaps[gid]
        elig = game_eligibility(state, snap)
        for team in (state.home.team_id, state.away.team_id):
            n_elig = sum(1 for p in state.players if p.team_id == team and elig[p.player_id].eligible and elig[p.player_id].role in RECIPIENT_ROLES)
            for kind in ("carry", "target"):
                pr = [r for r in au["ROWS"] if r["TEAM"] == team and r["KIND"] == kind]
                post = {r["PLAYER_ID"]: r["V25_POST_SHARE"] for r in pr if r["V25_POST_SHARE"] > 0}
                s = np.sort(np.array(list(post.values())))[::-1]
                rec_post = [(r["NAME"], r["V25_POST_SHARE"], r["PLAYER_HISTORY_SAMPLE"]) for r in pr if r["ROLE_CLASS"] == "RECIPIENT" and r["V25_POST_SHARE"] > 0]
                top = max(rec_post, key=lambda x: x[1]) if rec_post else (None, 0.0, 0.0)
                rows.append({"GAME": gid, "TEAM": team, "KIND": kind, "eligible_players": n_elig, "surviving_players": len(rec_post),
                             "top_share": float(s[0]) if s.size else 0.0, "top_2_share": float(s[:2].sum()) if s.size else 0.0,
                             "effective_pool_size": float(1.0 / (s ** 2).sum()) if s.size else 0.0, "top_recipient": top[0], "top_recipient_share": top[1],
                             "top_recipient_history_games": top[2], "top_recipient_above_p99_envelope": bool(top[1] > ENV_SHARE[kind]),
                             "above_hist_p99_top_share": bool(float(s[0]) > hist[kind]["TOP_SHARE"]["p99"]) if s.size else False,
                             "historical_reference": {"TOP_SHARE_MEDIAN": hist[kind]["TOP_SHARE"]["median"], "TOP_SHARE_P99": hist[kind]["TOP_SHARE"]["p99"],
                                                      "EFF_POOL_MEDIAN": hist[kind]["EFFECTIVE_POOL_SIZE"]["median"], "EFF_POOL_P10": hist[kind]["EFFECTIVE_POOL_SIZE"]["p10"]}})
    return rows


def pool_summary(rows):
    return {"POOLS": len(rows), "POOLS_TOP_RECIPIENT_ABOVE_P99_ENVELOPE": sum(r["top_recipient_above_p99_envelope"] for r in rows),
            "POOLS_ABOVE_HIST_P99_TOP_SHARE": sum(r["above_hist_p99_top_share"] for r in rows),
            "POOLS_EFFECTIVE_SIZE_BELOW_HIST_P10": sum(r["effective_pool_size"] < r["historical_reference"]["EFF_POOL_P10"] for r in rows),
            "MEAN_EFFECTIVE_POOL_SIZE": float(np.mean([r["effective_pool_size"] for r in rows])), "MEAN_TOP_SHARE": float(np.mean([r["top_share"] for r in rows])),
            "MEAN_SURVIVING_PLAYERS": float(np.mean([r["surviving_players"] for r in rows])), "MEAN_ELIGIBLE_PLAYERS": float(np.mean([r["eligible_players"] for r in rows]))}


# ---------------------------------------------------------------------------------------------- paths
def classify(ctx, per23, per_new, aud_new, dir_new, label):
    out = []
    for team, kind, nm in R26.PATHS_AUDITED:
        gid = next((g for g, (_, _, gg, _) in per_new.items() if team in (gg["AWAY"], gg["HOME"])), None)
        pid = next((r["PLAYER_ID"] for r in aud_new[gid]["ROWS"] if r["TEAM"] == team and r["NAME"] == nm), None)
        row = next((r for r in aud_new[gid]["ROWS"] if r["PLAYER_ID"] == pid and r["KIND"] == kind and r["TEAM"] == team), None)
        rec = {"PATH": f"{team}:{kind}:{nm}", "GAME": gid, "PLAYER_ID": pid}
        if row is None:
            rec.update({"STATUS": "UNSUPPORTED", "WHY": f"player has no share in that pool in the {label} audit"})
            out.append(rec)
            continue
        z23, zn = A25.raw(V23_DIR, gid), A25.raw(dir_new, gid)
        key = "player_rush_attempts" if kind == "carry" else "player_targets"
        m23, s23 = A25.mean_se(z23, pid, key)
        mn, sn = A25.mean_se(zn, pid, key)
        env = ENV_SHARE[kind]
        post, pre = row["V25_POST_SHARE"], row["PRE_FULL_POOL_SHARE"]
        thin = row["PLAYER_HISTORY_SAMPLE"] < 8
        held = row["ROLE_CLASS"] == "HELD"
        se = float(np.hypot(s23, sn))
        if held:
            ok = post <= pre + TOL and mn <= m23 + 2 * se
            status, why = ("MECHANICALLY_RESOLVED", "QB held at its no-removal share; sim mean <= V23 mean + 2SE") if ok else ("UNSUPPORTED", "held role above its no-removal share or V23 sim mean")
        elif row["ELIGIBLE"] and post <= env and not thin:
            status, why = "SUPPORTED", "recipient role, eligible, state share inside the historical single-player envelope, history >= 8 games"
        elif row["ELIGIBLE"]:
            status = "AMBIGUOUS"
            why = "; ".join(x for x in (f"state share {post:.3f} > envelope {env}" if post > env else "", f"thin history ({row['PLAYER_HISTORY_SAMPLE']:.0f} games)" if thin else "") if x)
        else:
            status, why = "UNSUPPORTED", "ineligible player"
        rec.update({"ROLE": row["ROLE"], "STATUS": status, "WHY": why, "HISTORY_SAMPLE": row["PLAYER_HISTORY_SAMPLE"], "STATE_SHARE_POST": post, "STATE_SHARE_PRE_FULL_POOL": pre,
                    "SIM_MEAN": {"V23": m23, label: mn, "SE_DIFF": se, "STAT": key}, "ENVELOPE": env,
                    "SIM_SHARE": next((r["SIM_CARRY_SHARE" if kind == "carry" else "SIM_TARGET_SHARE"] for r in per_new[gid][3]["TEAMS"][team]["PLAYERS"] if r["PLAYER_ID"] == pid), None)})
        out.append(rec)
    return out


# ---------------------------------------------------------------------------------------------- market scan
def market_scan(ctx26, audits):
    viol = []
    for name, payload in (("COMPLETION_POLICY", cfg.COMPLETION_POLICY), *((f"AUDIT:{g}", a) for g, a in audits.items())):
        try:
            assert_market_free(payload)
        except Exception as e:  # noqa: BLE001
            viol.append({name: str(e)})
    for gid, inp in ctx26["inputs"].items():
        try:
            assert_market_free(inp.state.model_dump(mode="json"))
        except Exception as e:  # noqa: BLE001
            viol.append({f"COMPLETED_STATE:{gid}": str(e)})
    bad_imports, forbidden = [], ("sports_market_interface", "sports_research", "market_book", "comparator", "trading_lab", "kalshi", "sportsbook")
    for f in R26.pkg_files():
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else [n.name for n in node.names] if isinstance(node, ast.Import) else []
            bad_imports += [f"{f.name}:{m}" for m in mods if any(x in m for x in forbidden)]
    vocab, hits = re.compile(r"moneyline|sportsbook|kalshi|odds|betting|vegas|pnl|closing line|implied prob", re.I), []
    for p in sorted(OUT_DIR.rglob("*")):
        if not p.is_file() or p.name.startswith(("SPORTS_NOVA_V26_REPORT", "THIN_POOL", "STATE_COMPLETENESS")):
            continue
        rel = str(p.relative_to(OUT_DIR)).replace("\\", "/")
        if p.suffix == ".npz":
            hits += [f"{rel}:{n}" for n in np.load(p, allow_pickle=False).files if vocab.search(n)]
        elif rel.startswith(("games/", "audit/")) or p.name == "sim_run_meta.json":
            hits += [f"{rel}:{h}" for h in vocab.findall(p.read_text(encoding="utf-8", errors="replace"))]
    return {"M1_MARKET_CONTAMINATION": len(viol) + len(bad_imports) + len(hits), "FIREWALL_KEY_VIOLATIONS": viol, "MARKET_MODULE_IMPORTS_BY_V26": bad_imports, "VOCAB_HITS_IN_DATA_FILES": hits}


def unchanged_artifacts(prereg):
    now = {"V25_PACKAGE": R26.v25_files_unchanged(),
           "V23_FILES": {k: R26.sha256_file(ROOT / k) == v for k, v in json.loads(R25.V23_BEFORE.read_text())["FILES"].items()},
           "V24_ROSTER_ELIGIBILITY": {k: R26.sha256_file(ROOT / k) == v for k, v in prereg["V24_ROSTER_ELIGIBILITY_FILES_AT_FREEZE"].items()},
           "V24_CODEX": {k: R26.sha256_file(ROOT / k) == v for k, v in prereg["V24_CODEX_FILES_AT_FREEZE"].items()},
           "V25_MIRROR": {k: R26.sha256_file(ROOT / k) == v for k, v in prereg["V25_MIRROR_FILES_AT_FREEZE"].items()}}
    return {"ALL_UNCHANGED": all(all(v.values()) for v in now.values()), "DETAIL_COUNTS": {k: [sum(v.values()), len(v)] for k, v in now.items()}}


# ---------------------------------------------------------------------------------------------- main
def cmd_analyze() -> None:
    ctx = base.load_inputs()
    meta = json.loads((OUT_DIR / "sim_run_meta.json").read_text())
    prereg = json.loads(R26.PREREG_FILE.read_text())
    assert prereg["V26_HASH"] == R26.v26_hash() == meta["V26_HASH"], "V26 package changed since freeze"
    comps = R26.completions(ctx)
    cf_comps = R26.completions(ctx, share_basis=cfg.SHARE_BASIS_COUNTERFACTUAL)
    ctx26 = make_ctx26(ctx, comps)
    per23, per24, per25 = (A.load_dir(d, ctx) for d in (V23_DIR, V24_DIR, V25_DIR))
    per26 = load_dir26(OUT_DIR, ctx, ctx26)
    pers = [per23, per24, per25, per26]
    ag = [A.aggregate(p) for p in pers]
    acc = [A.accounting(V23_DIR, ctx), A.accounting(V24_DIR, ctx), A.accounting(V25_DIR, ctx), A.accounting(OUT_DIR, ctx26)]
    aud25, aud26 = A25.audits_of(V25_DIR), A25.audits_of(OUT_DIR)

    def conservation(aud):
        bad = [s for au in aud.values() for s in au["SUMMARIES"] if abs(s["POST_REDISTRIBUTION_TOTAL"] - s["PRE_REMOVAL_TOTAL"]) > 1e-9 or abs(s["REMOVED_MASS"] - s["REDISTRIBUTED_MASS"]) > 1e-9
               or s["MASS_TO_HELD_ROLES"] > 1e-9 or abs(s["SUM_SHARE_PLUS_RESIDUAL_AFTER"] - 1.0) > 1e-6]
        return ("PASS" if not bad else "FAIL"), sum(len(a["SUMMARIES"]) for a in aud.values()), bad[:3]
    cons25, cons26 = conservation(aud25), conservation(aud26)
    v24_report = json.loads((V24_DIR / "SPORTS_NOVA_V24_ROSTER_ELIGIBILITY_REPORT.json").read_text())
    scan = market_scan(ctx26, aud26)
    identity = sim_identity(ctx)
    audit = R26.build_audit(ctx, comps)

    # ---- real-data tests T1..T10
    exp_missing_valid = []
    for gid, (snap, prior, comp) in comps.items():
        in_state = {(p.team_id, p.player_id) for p in comp.state.players}
        for r in comp.rows:
            if r["REASON"] != "NO_HISTORY" and r["ACTION"] != cfg.ACTION_BLOCKED_DUPLICATE and (r["TEAM"], r["PLAYER_ID"]) not in in_state:
                exp_missing_valid.append((gid, r["NAME"] if "NAME" in r else r["PLAYER_ID"]))
    added_wrong_team = [(gid, r["PLAYER_ID"]) for gid, (snap, prior, comp) in comps.items() for r in comp.rows
                        if r["ACTION"] in (cfg.ACTION_ADDED, cfg.ACTION_ADDED_ZERO_SHARE) and snap.roster.get(r["PLAYER_ID"], (None,))[0] != r["TEAM"]]
    dup_ids = [gid for gid, (_, _, comp) in comps.items() if len({p.player_id for p in comp.state.players}) != len(comp.state.players)]
    ineligible_positive = [(gid, r["NAME"]) for gid, au in aud26.items() for r in au["ROWS"] if not r["ELIGIBLE"] and r["V25_POST_SHARE"] > 0]
    game_rec = {x["GAME_ID"]: x for x in ctx["games"]}
    out_in_expected = [(gid, pid) for gid, (snap, prior, comp) in comps.items() for t in comp.teams for pid in R26.known_out(game_rec[gid])
                       if pid in expected_active_skill(snap, t, frozenset())]
    qb25 = {(g, r["TEAM"], r["PLAYER_ID"], r["KIND"]): r["V25_POST_SHARE"] for g, a in aud25.items() for r in a["ROWS"] if r["ROLE"] == "QB"}
    qb26 = {(g, r["TEAM"], r["PLAYER_ID"], r["KIND"]): r["V25_POST_SHARE"] for g, a in aud26.items() for r in a["ROWS"] if r["ROLE"] == "QB"}
    qb_share_diff = max((abs(qb26.get(k, 0.0) - v) for k, v in qb25.items()), default=0.0)
    qb_share_changed = sorted({(k[0], k[1], k[3]) for k, v in qb25.items() if abs(qb26.get(k, 0.0) - v) > 1e-12})
    art = unchanged_artifacts(prereg)
    tests = subprocess.run([sys.executable, "-m", "pytest", f"{R26.PKG}/tests", "worker/sports_nova_v25_role_aware/tests", "-q"], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    T = {"T1_all_active_valid_history_present": {"PASS": not exp_missing_valid, "MISSING_VALID": exp_missing_valid},
         "T2_out_ineligible_remain_zero": {"PASS": not ineligible_positive and not out_in_expected and ag[3]["INVALID_PLAYER_USAGE"] == 0, "INELIGIBLE_WITH_POSITIVE_SHARE": ineligible_positive, "OUT_IN_EXPECTED": out_in_expected,
                                          "INVALID_USAGE": ag[3]["INVALID_PLAYER_USAGE"]},
         "T3_qb_identical_to_v25": {"PASS": all(a["QB_INPUTS_UNCHANGED"] for a in aud26.values()) and ag[3]["QB_MISMATCH"] == 0, "QB_SELECTION_INPUTS_UNCHANGED_ALL_GAMES": all(a["QB_INPUTS_UNCHANGED"] for a in aud26.values()),
                                    "MAX_QB_STATE_SHARE_DIFF_V26_VS_V25": qb_share_diff, "QB_STATE_SHARE_CHANGED_POOLS": qb_share_changed},
         "T4_v25_redistribution_unchanged": {"PASS": all(R26.v25_files_unchanged().values()) and "def role_aware_values" not in "".join(f.read_text(encoding="utf-8") for f in R26.pkg_files()),
                                             "V25_FILES_BYTE_IDENTICAL": R26.v25_files_unchanged(),
                                             "V26_CONTAINS_NO_REDISTRIBUTION_CODE": "def role_aware_values" not in "".join(f.read_text(encoding="utf-8") for f in R26.pkg_files())},
         "T5_no_mass_created_or_lost": {"PASS": cons26[0] == "PASS", "POOLS": cons26[1]}, "T6_accounting": {"PASS": acc[3]["STATUS"] == "PASS", "CHECKS": acc[3]["CHECKS"], "FAILURES": acc[3]["FAILURES"]},
         "T7_market_contamination_zero": {"PASS": scan["M1_MARKET_CONTAMINATION"] == 0, "COUNT": scan["M1_MARKET_CONTAMINATION"]},
         "T8_no_cross_team_insertion": {"PASS": not added_wrong_team and not dup_ids, "ADDED_ON_WRONG_TEAM": added_wrong_team, "DUPLICATE_IDENTITY_GAMES": dup_ids},
         "T9_missing_history_blocked_safely": {"PASS": all(r["ACTION"] == cfg.ACTION_BLOCKED_COLD_START and (r["TEAM"], r["PLAYER_ID"]) not in {(p.team_id, p.player_id) for p in comps[r["GAME"]][2].state.players}
                                                                for r in audit["MISSING_PLAYERS"] if r["REASON"] == "NO_HISTORY"),
                                               "COLD_START_POLICY_REQUIRED_PLAYERS": sum(1 for r in audit["MISSING_PLAYERS"] if r["REASON"] == "NO_HISTORY")},
         "T10_v23_v24_artifacts_unchanged": {"PASS": art["ALL_UNCHANGED"], "DETAIL_COUNTS": art["DETAIL_COUNTS"]},
         "UNIT_TESTS": tests[-1] if tests else ""}
    tests_pass = all(v["PASS"] for k, v in T.items() if isinstance(v, dict) and "PASS" in v)

    # ---- QB behaviour
    qb_t = [A25.qb_table(p, ctx) for p in (per23, per24, per25, per26)]
    key = lambda r: (r["GAME"], r["TEAM"], r["PLAYER_ID"])
    d = [{key(r): r for r in rs} for rs in qb_t]
    over = lambda rs: sorted([r for r in rs if r["MEAN_RUSH_ATTEMPTS"] > QB_MAX], key=lambda r: -r["MEAN_RUSH_ATTEMPTS"])
    team_mean = lambda rs: float(np.mean([sum(r["MEAN_RUSH_ATTEMPTS"] for r in rs if (r["GAME"], r["TEAM"]) == k) for k in {(r["GAME"], r["TEAM"]) for r in rs}]))
    qb_rows = sorted([{"NAME": r["NAME"], "TEAM": r["TEAM"], "V23": d[0].get(key(r), {}).get("MEAN_RUSH_ATTEMPTS", 0.0), "V24": d[1].get(key(r), {}).get("MEAN_RUSH_ATTEMPTS", 0.0),
                       "V25": d[2].get(key(r), {}).get("MEAN_RUSH_ATTEMPTS", 0.0), "V26": r["MEAN_RUSH_ATTEMPTS"], "V26_MINUS_V25": r["MEAN_RUSH_ATTEMPTS"] - d[2].get(key(r), {}).get("MEAN_RUSH_ATTEMPTS", 0.0)}
                      for r in qb_t[3]], key=lambda r: -r["V26"])
    qbr = {"HISTORICAL_MAX_MEAN_RUSH_ATT": QB_MAX, "SLATE_MEAN_QB_RUSH_ATTEMPTS_PER_TEAM": {v: team_mean(rs) for v, rs in zip(("V23", "V24", "V25", "V26"), qb_t)},
           "QBS_ABOVE_HISTORICAL_MAX": {v: [(r["NAME"], round(r["MEAN_RUSH_ATTEMPTS"], 2)) for r in over(rs)] for v, rs in zip(("V23", "V24", "V25", "V26"), qb_t)},
           "MAX_ABS_QB_RUSH_MEAN_CHANGE_V26_VS_V25": max(abs(r["V26_MINUS_V25"]) for r in qb_rows), "QB_ROWS": qb_rows,
           "QB_MATCH_MISMATCH": {v: (a["QB_MATCH"], a["QB_MISMATCH"]) for v, a in zip(("V23", "V24", "V25", "V26"), ag)},
           "STATE_LEVEL_QB_CARRY_INCREASES_VS_NO_REMOVAL_V26": sum(1 for au in aud26.values() for r in au["ROWS"] if r["ROLE"] == "QB" and r["KIND"] == "carry" and r["V25_POST_SHARE"] > r["PRE_FULL_POOL_SHARE"] + TOL),
           "INHERITED_NOT_FIXED": "V23 QB scramble double count (deferred)"}

    # ---- thin pools
    hist = historical_reference(ctx["panel_df"])
    ss25 = {gid: (ctx["inputs"][gid].state, comps[gid][0]) for gid in comps}
    ss26 = {gid: (comps[gid][2].state, comps[gid][0]) for gid in comps}
    tp25, tp26 = pool_table(aud25, ss25, hist), pool_table(aud26, ss26, hist)
    # counterfactual (portable) -- diagnostic only, never simulated
    cf_aud = {}
    for gid, (snap, prior, comp) in cf_comps.items():
        rep, au = repair_pregame_state(comp.state, snap)
        assert_repair_invariants(comp.state, rep, au)
        cf_aud[gid] = {"ROWS": [dict(r, NAME=ctx["names"].get(r["PLAYER_ID"])) for r in au.rows], "SUMMARIES": au.summaries}
    tpcf = pool_table(cf_aud, {gid: (cf_comps[gid][2].state, cf_comps[gid][0]) for gid in cf_comps}, hist)
    pools = {"HISTORICAL_REFERENCE": hist, "V25_BEFORE": pool_summary(tp25), "V26_AFTER": pool_summary(tp26), "COUNTERFACTUAL_PORTABLE_NOT_SIMULATED": pool_summary(tpcf),
             "ROWS_V25": tp25, "ROWS_V26": tp26, "ROWS_COUNTERFACTUAL": tpcf,
             "MAX_ABS_TOP_SHARE_CHANGE_V26_VS_V25": max(abs(a["top_share"] - b["top_share"]) for a, b in zip(tp25, tp26)),
             "POOLS_WITH_ANY_TOP_SHARE_CHANGE": [f'{a["GAME"]}:{a["TEAM"]}:{a["KIND"]}' for a, b in zip(tp25, tp26) if abs(a["top_share"] - b["top_share"]) > 1e-9]}
    named = []
    for team, kind, nm in NAMED:
        gid = next(g for g, (_, _, gg, _) in per26.items() if team in (gg["AWAY"], gg["HOME"]))
        pid = next((r["PLAYER_ID"] for r in aud26[gid]["ROWS"] if r["TEAM"] == team and r["NAME"] == nm), None)
        sel = lambda au: next((r for r in au[gid]["ROWS"] if r["PLAYER_ID"] == pid and r["KIND"] == kind and r["TEAM"] == team), {})
        r25, r26, rcf = sel(aud25), sel(aud26), sel(cf_aud)
        skey, mkey = ("SIM_CARRY_SHARE", "SIM_MEAN_CARRIES") if kind == "carry" else ("SIM_TARGET_SHARE", "SIM_MEAN_TARGETS")
        pl = lambda per: next((r for r in per[gid][3]["TEAMS"][team]["PLAYERS"] if r["PLAYER_ID"] == pid), {})
        ctm = aud26[gid]["COMPLETION_TEAMS"][team]
        miss = [{"NAME": r["NAME"], "ROLE": r["ROLE"], "REASON": r["REASON"], "ACTION": r["ACTION"], "TARGETS_THIS_TEAM": r["TARGETS_THIS_TEAM"], "CARRIES_THIS_TEAM": r["CARRIES_THIS_TEAM"],
                 "TARGETS_ALL_TEAMS": r["TARGETS_ALL_TEAMS"], "CARRIES_ALL_TEAMS": r["CARRIES_ALL_TEAMS"]} for r in aud26[gid]["COMPLETION_ROWS"] if r["TEAM"] == team]
        named.append({"PATH": f"{team}:{kind}:{nm}", "GAME": gid, "STATE_SHARE": {"PRE_FULL_POOL": r26.get("PRE_FULL_POOL_SHARE"), "V23_EFFECTIVE": r26.get("V23_EFFECTIVE_SHARE"), "V24_EQUIVALENT": r26.get("V24_EQUIVALENT_SHARE"),
                                                                                  "V25": r25.get("V25_POST_SHARE"), "V26": r26.get("V25_POST_SHARE"), "COUNTERFACTUAL_PORTABLE_NOT_SIMULATED": rcf.get("V25_POST_SHARE")},
                      "SIM_MEAN": {"V23": pl(per23).get(mkey), "V24": pl(per24).get(mkey), "V25": pl(per25).get(mkey), "V26": pl(per26).get(mkey)},
                      "SIM_SHARE": {"V23": pl(per23).get(skey), "V25": pl(per25).get(skey), "V26": pl(per26).get(skey)},
                      "TEAM_ACTIVE_SKILL": {"expected": ctm["ACTIVE_SKILL_EXPECTED"], "previously_present": ctm["PRESENT_BEFORE"], "missing": ctm["MISSING"], "added": ctm["ADDED"], "blocked": ctm["BLOCKED"]},
                      "MISSING_PLAYERS_ON_TEAM": miss, "ENVELOPE": ENV_SHARE[kind]})
    # ---- paths / gates
    paths25 = classify(ctx, per23, per25, aud25, V25_DIR, "V25")
    paths26 = classify(ctx, per23, per26, aud26, OUT_DIR, "V26")
    pc = lambda ps: {k: sum(1 for p in ps if p["STATUS"] == k) for k in ("SUPPORTED", "MECHANICALLY_RESOLVED", "AMBIGUOUS", "UNSUPPORTED")}
    nc = A25.new_concentration(aud26)
    fb = A25_football(per23, per26)
    severe_pools = [f'{r["GAME"]}:{r["TEAM"]}:{r["KIND"]}:{r["top_recipient"]}={r["top_recipient_share"]:.3f}' for r in tp26 if r["top_recipient_above_p99_envelope"]]
    tech = {"invalid_usage": ag[3]["INVALID_PLAYER_USAGE"], "meaningful_invalid_usage": ag[3]["MEANINGFUL_INVALID_USAGE"], "mass_conservation": cons26[0], "accounting": acc[3]["STATUS"],
            "qb_match": ag[3]["QB_MATCH"], "qb_mismatch": ag[3]["QB_MISMATCH"], "market_contamination": scan["M1_MARKET_CONTAMINATION"]}
    tech_ok = tech["invalid_usage"] == 0 and tech["meaningful_invalid_usage"] == 0 and tech["mass_conservation"] == "PASS" and tech["accounting"] == "PASS" and tech["qb_mismatch"] == 0 and tech["market_contamination"] == 0
    complete_valid = T["T1_all_active_valid_history_present"]["PASS"]
    qb_removed = qbr["STATE_LEVEL_QB_CARRY_INCREASES_VS_NO_REMOVAL_V26"] == 0 and not nc["HELD_ROLE_INCREASES"] and len(over(qb_t[3])) <= len(over(qb_t[0]))
    paths_ok = all(p["STATUS"] in ("SUPPORTED", "MECHANICALLY_RESOLVED") for p in paths26)
    prop_ready = bool(tech_ok and tests_pass and complete_valid and paths_ok and not severe_pools and not nc["NEW_SEVERE_CONCENTRATION"] and qb_removed)
    game_ready = bool(tech_ok and tests_pass and qb_removed and not [n for n in nc["NEW_SEVERE_CONCENTRATION"] if n["ROLE"] not in cfg.RECIPIENT_ROLES] and not fb["REGRESSION_TRIGGERS"])
    unresolved_total = audit["TOTAL"]["blocked_unresolved"]
    report = {"SCHEMA": "SPORTS_NOVA_V26_REPORT", "CREATED_AT": datetime.now(timezone.utc).isoformat(), "V26_VERSION": cfg.MODEL_VERSION, "V26_HASH": R26.v26_hash(), "PREREG_SHA256": R26.sha256_file(R26.PREREG_FILE),
              "ACTIVE_SKILL_STATE": audit["TOTAL"], "MISSING_BY_ROLE": audit["MISSING_BY_ROLE"], "MISSING_BY_REASON": audit["MISSING_BY_REASON"], "BY_ACTION": audit["BY_ACTION"],
              "MACHINERY_VERIFICATION_GAMES_PASSED": len(audit["MACHINERY_VERIFICATION"]), "TESTS": T, "TESTS_ALL_PASS": tests_pass,
              "SIM_IDENTITY_V26_VS_V25": identity, "SIM_IDENTICAL_GAMES": [g for g, v in identity.items() if v["BIT_IDENTICAL_COMMON_PLAYERS_AND_TEAMS"]],
              "SIM_DIFFERING_GAMES": [g for g, v in identity.items() if not v["BIT_IDENTICAL_COMMON_PLAYERS_AND_TEAMS"]],
              "AGGREGATE": {v: a for v, a in zip(("V23", "V24", "V25", "V26"), ag)},
              "ACCOUNTING": {v: a["STATUS"] for v, a in zip(("V23", "V24", "V25", "V26"), acc)},
              "CONSERVATION": {"V23": "N/A (no redistribution)", "V24": "NOT_RECORDED (V24 emitted no conservation audit; not recoverable without re-running V24)", "V25": cons25[0], "V26": cons26[0], "V26_POOLS": cons26[1]},
              "MARKET": scan, "QB_REALISM": qbr, "THIN_POOLS": {k: v for k, v in pools.items() if not k.startswith("ROWS")}, "NAMED_CASES": named, "PATHS": {"V25": paths25, "V26": paths26},
              "PATH_COUNTS": {"V25": pc(paths25), "V26": pc(paths26)}, "NEW_CONCENTRATION": nc, "SEVERE_THIN_POOL_ARTIFACTS": severe_pools,
              "FOOTBALL_V26_VS_V23": fb, "GATES": {"TECHNICAL": tech, "TECHNICAL_OK": tech_ok, "ACTIVE_SKILL_STATE_COMPLETE_FOR_VALID_HISTORY": complete_valid,
                                                    "ACTIVE_SKILL_STATE_COMPLETE_OVERALL": unresolved_total == 0, "UNRESOLVED_COLD_START_OR_DUPLICATE": unresolved_total,
                                                    "ALL_NINE_PATHS_SUPPORTED_OR_RESOLVED": paths_ok, "NEW_SEVERE_CONCENTRATION_COUNT": len(nc["NEW_SEVERE_CONCENTRATION"]), "QB_REDISTRIBUTION_INFLATION_REMOVED": qb_removed,
                                                    "SEVERE_THIN_POOL_ARTIFACT_COUNT": len(severe_pools), "REGRESSION_TRIGGERS": fb["REGRESSION_TRIGGERS"],
                                                    "PLAYER_PROP_MIRROR_READY": "YES" if prop_ready else "NO", "GAME_LEVEL_MIRROR_READY": "YES" if game_ready else "NO"},
              "MARKET_LABELS": R26.GATE_RULES_V26["MARKET_LABELS"], "DEFERRED_DEFECTS": R26.DEFERRED, "ENGINE_HASHES_UNCHANGED_DURING_SIM": meta.get("ENGINE_HASHES_UNCHANGED_DURING_SIM"),
              "V23_REMAINS_CHAMPION": True, "V25_HISTORICAL_PROMOTION": False, "V26_HISTORICAL_PROMOTION": False, "LIVE_CAPITAL_AUTHORIZED": False}
    (OUT_DIR / "SPORTS_NOVA_V26_REPORT.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8")
    (OUT_DIR / "THIN_POOL_DIAGNOSTICS.json").write_text(json.dumps(pools, indent=1, sort_keys=True, default=str), encoding="utf-8")
    (OUT_DIR / "SPORTS_NOVA_V26_PREREG.json").write_text(R26.PREREG_FILE.read_text(), encoding="utf-8")
    print(json.dumps({"ACTIVE": audit["TOTAL"], "TECH": tech, "TESTS_ALL_PASS": tests_pass, "T": {k: (v["PASS"] if isinstance(v, dict) and "PASS" in v else v) for k, v in T.items()},
                      "IDENTICAL_GAMES": len(report["SIM_IDENTICAL_GAMES"]), "DIFFERING": report["SIM_DIFFERING_GAMES"], "PATH_COUNTS": report["PATH_COUNTS"],
                      "POOLS": {k: pools[k] for k in ("V25_BEFORE", "V26_AFTER", "COUNTERFACTUAL_PORTABLE_NOT_SIMULATED", "MAX_ABS_TOP_SHARE_CHANGE_V26_VS_V25", "POOLS_WITH_ANY_TOP_SHARE_CHANGE")},
                      "GATES": {k: v for k, v in report["GATES"].items() if k not in ("TECHNICAL", "REGRESSION_TRIGGERS")}, "REGRESSION": fb["REGRESSION_TRIGGERS"]}, indent=1, default=str))


def A25_football(per23, per26):
    rng = np.random.default_rng(20260920)
    games = []
    for gid in per23:
        g23, g26 = per23[gid][2], per26[gid][2]
        z23, z26 = A25.raw(V23_DIR, gid), A25.raw(OUT_DIR, gid)
        h = list(z23["team_ids"]).index(g23["HOME"]), list(z23["team_ids"]).index(g23["AWAY"])
        tot = lambda z: z["team_score"][:, h[0]] + z["team_score"][:, h[1]]
        mar = lambda z: z["team_score"][:, h[0]] - z["team_score"][:, h[1]]
        row = {"GAME": f"{g23['AWAY']}@{g23['HOME']}", "GAME_ID": gid}
        for nm, f, k in (("TOTAL", tot, "TOTAL_POINTS"), ("MARGIN", mar, "MARGIN_HOME_MINUS_AWAY")):
            row[nm] = {"V23_MEDIAN": g23[k]["median"], "V26_MEDIAN": g26[k]["median"], "DELTA": g26[k]["median"] - g23[k]["median"], "SE_OF_DELTA": A.median_se(f(z23), f(z26), rng)}
        row["HOME_WIN_PROB"] = {"V23": g23["HOME_WIN_PROB_TIE_SPLIT"], "V26": g26["HOME_WIN_PROB_TIE_SPLIT"]}
        row["TEAM_ATTEMPTS_MEAN"] = {t: {k: {"V23": g23["TEAM"][t][k]["mean"], "V26": g26["TEAM"][t][k]["mean"], "PCT": 100.0 * (g26["TEAM"][t][k]["mean"] / g23["TEAM"][t][k]["mean"] - 1.0)}
                                         for k in ("pass_attempts", "rush_attempts")} for t in (g23["AWAY"], g23["HOME"])}
        games.append(row)
    rules = R25.REGRESSION_RULES
    m23, m26 = float(np.mean([r["TOTAL"]["V23_MEDIAN"] for r in games])), float(np.mean([r["TOTAL"]["V26_MEDIAN"] for r in games]))
    trig = []
    if m23 - m26 > rules["SLATE_MEAN_OF_MEDIAN_TOTALS_DROP_POINTS"]:
        trig.append({"RULE": "SLATE_MEAN_OF_MEDIAN_TOTALS_DROP", "V23": m23, "V26": m26})
    for r in games:
        if abs(r["TOTAL"]["DELTA"]) > rules["GAME_MEDIAN_TOTAL_SHIFT_POINTS"]:
            trig.append({"RULE": "GAME_MEDIAN_TOTAL_SHIFT", "GAME": r["GAME"], "DELTA": r["TOTAL"]["DELTA"]})
        for t, dd in r["TEAM_ATTEMPTS_MEAN"].items():
            for k, v in dd.items():
                if abs(v["PCT"]) > rules["TEAM_MEAN_ATTEMPTS_SHIFT_PCT"]:
                    trig.append({"RULE": "TEAM_MEAN_ATTEMPTS_SHIFT", "GAME": r["GAME"], "TEAM": t, "STAT": k, "PCT": v["PCT"]})
    dt = [r["TOTAL"]["DELTA"] for r in games]
    return {"GAMES": games, "SLATE_MEAN_OF_MEDIAN_TOTALS": {"V23": m23, "V26": m26, "DELTA": m26 - m23}, "TOTAL_DISTRIBUTION_CHANGE": {"MEAN_DELTA": float(np.mean(dt)), "MAX_ABS_DELTA": float(np.max(np.abs(dt)))},
            "REGRESSION_TRIGGERS": trig}
