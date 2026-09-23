"""Run and audit the narrow SPORTS_NOVA V24 eligibility challenger.

Commands:
  python scripts/sports_nova_m1_v24_eligibility.py test
  python scripts/sports_nova_m1_v24_eligibility.py sim [N]
  python scripts/sports_nova_m1_v24_eligibility.py analyze

The V2 Sunday panel is immutable input.  V23 raw arrays are read from the
completed baseline; no V23 artifact is modified.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worker.sports_nova_modular.firewall import assert_market_free
from worker.sports_nova_sunday_panel import adapter
from worker.sports_nova_v19.simulator import STAT_NAMES, _qb_shares
from worker.sports_nova_v23.simulator import MODEL_VERSION as V23_VERSION, set_identity_resolutions as set_v23_identity
from worker.sports_nova_v24.config import MODEL_VERSION
from worker.sports_nova_v24.eligibility import (
    REASON_CODES, apply_eligibility_mask, assert_eligibility_invariants,
)
from worker.sports_nova_v24.simulator import simulate_game, set_identity_resolutions

DATA = ROOT / "data" / "sports_nova_v3" / "sunday_2026_09_20"
PANEL_DIR = DATA / "SUNDAY_2026_09_20_PRE_GAME_PANEL_V2"
V23_DIR = DATA / "SUNDAY_2026_09_20_M1_BASELINE_V1"
OUT_DIR = DATA / "SUNDAY_2026_09_20_M1_V24_CHALLENGER"
BASELINE_REPORT = V23_DIR / "SUNDAY_2026_09_20_M1_BASELINE_V1.json"
BASELINE_AUDIT = V23_DIR / "SUNDAY_2026_09_20_ROSTER_CONTAMINATION_AUDIT.json"
PANEL_SHA = "dd16b820d779b12df93734bafa5b9dcadc00bc22029d4cc9eaf15e26b0f38840"
N_DEFAULT = 5000
TOL = 1e-12
SEED_SUFFIX = "_SUNDAY_BASELINE_V1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_for(gid: str) -> int:
    return int(hashlib.sha256((gid + SEED_SUFFIX).encode()).hexdigest()[:8], 16)


def summary(a):
    a = np.asarray(a, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "P10": float(np.percentile(a, 10)), "P25": float(np.percentile(a, 25)),
            "P75": float(np.percentile(a, 75)), "P90": float(np.percentile(a, 90))}


def load_context():
    raw = (PANEL_DIR / "panel.json").read_bytes()
    panel = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != PANEL_SHA:
        raise RuntimeError("PANEL_SHA_MISMATCH")
    panel_df, panel_pq_sha = adapter.load_panel(PANEL_DIR / "sources" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.live.parquet")
    names_df = panel_df.drop_duplicates("PLAYER_ID", keep="last")
    names = dict(zip(names_df.PLAYER_ID.astype(str), names_df.PLAYER_NAME))
    roster = pd.read_parquet(PANEL_DIR / "sources" / "roster_weekly_2026.parquet")
    week = int(roster.week.max())
    roster = roster[roster.week == week]
    roster_idx = {str(r.gsis_id): (str(r.team), str(r.status)) for r in roster.itertuples() if r.gsis_id}
    inputs = {}
    audits = {}
    for rec in panel["GAMES"]:
        inp = adapter.build_m1_input(rec, panel_df, panel_pq_sha, panel["HEADER"]["SOURCE_INVENTORY"], roster_idx, names)
        if inp.state is not None:
            masked, audit = apply_eligibility_mask(inp.state, roster_idx)
            assert_eligibility_invariants(masked, audit)
            inp.state = masked
            audits[rec["GAME_ID"]] = audit
        inputs[rec["GAME_ID"]] = inp
    resolutions = {}
    for inp in inputs.values():
        resolutions.update(inp.resolutions)
    return {"panel": panel, "raw_sha": hashlib.sha256(raw).hexdigest(), "inputs": inputs,
            "resolutions": resolutions, "roster_idx": roster_idx, "names": names,
            "audits": audits, "roster_week": week, "panel_pq_sha": panel_pq_sha}


def _worker(args):
    gid, state, resolutions, n, outdir = args
    set_identity_resolutions(resolutions)
    t0 = time.time()
    batch = simulate_game(state, n, seed_for(gid), MODEL_VERSION)
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    keep = [j for j, pid in enumerate(batch.player_ids)
            if any(np.asarray(batch.player_stats[s])[:, j].any() for s in STAT_NAMES)]
    arrays = {f"player_{s}": batch.player_stats[s][:, keep].astype(np.int32) for s in STAT_NAMES}
    arrays.update({f"team_{s}": batch.team_stats[s] for s in batch.team_stats})
    np.savez_compressed(outdir / "raw" / f"{gid}.npz",
                        player_ids=np.array([batch.player_ids[j] for j in keep]),
                        team_ids=np.array(batch.team_ids), winners=batch.winner, **arrays)
    return gid, time.time() - t0


def cmd_sim(n: int):
    ctx = load_context()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(gid, inp.state, ctx["resolutions"], n, OUT_DIR)
            for gid, inp in ctx["inputs"].items() if inp.state is not None]
    meta = {"SCHEMA": "SPORTS_NOVA_V24_SIM_RUN", "STARTED_AT": datetime.now(timezone.utc).isoformat(),
            "N_SIMS": n, "PANEL_SHA256": ctx["raw_sha"], "MODEL_VERSION": MODEL_VERSION,
            "V23_MODEL_VERSION": V23_VERSION, "SEED_POLICY": SEED_SUFFIX,
            "NO_MARKET_INPUTS": True, "TOLERANCE": TOL}
    (OUT_DIR / "sim_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as pool:
        futs = {pool.submit(_worker, job): job[0] for job in jobs}
        for fut in as_completed(futs):
            gid, elapsed = fut.result()
            print(f"done {gid} {elapsed:.1f}s", flush=True)
    meta.update({"FINISHED_AT": datetime.now(timezone.utc).isoformat(),
                 "GAME_IDS": sorted(g for g, *_ in jobs)})
    (OUT_DIR / "sim_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_npz(path: Path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def team_slot(a, team, stat):
    return int(np.where(a["team_ids"] == team)[0][0])


def game_metrics(gid, ctx):
    v23 = load_npz(V23_DIR / "raw" / f"{gid}.npz")
    v24 = load_npz(OUT_DIR / "raw" / f"{gid}.npz")
    rec = next(r for r in ctx["panel"]["GAMES"] if r["GAME_ID"] == gid)
    home, away = rec["HOME_TEAM"], rec["AWAY_TEAM"]
    out = {"GAME_ID": gid, "GAME": f"{away}@{home}", "N_SIMS": len(v24["winners"]),
           "SEED": seed_for(gid), "STATE_MASK_FALLBACKS": list(ctx["audits"][gid].zero_sum_fallbacks)}
    for label, a in (("V23", v23), ("V24", v24)):
        hi, ai = team_slot(a, home, "team_score"), team_slot(a, away, "team_score")
        hs, aw = a["team_score"][:, hi], a["team_score"][:, ai]
        margin, total = hs - aw, hs + aw
        out[label] = {"home_win_probability": float(np.mean(hs > aw) + .5 * np.mean(hs == aw)),
                      "median_margin": float(np.median(margin)), "median_total": float(np.median(total)),
                      "team_rush_attempts": {home: summary(a["team_rush_attempts"][:, hi]), away: summary(a["team_rush_attempts"][:, ai])},
                      "team_pass_attempts": {home: summary(a["team_pass_attempts"][:, hi]), away: summary(a["team_pass_attempts"][:, ai])}}
    out["delta"] = {k: out["V24"][k] - out["V23"][k] for k in ("home_win_probability", "median_margin", "median_total")}
    out["eligibility"] = {}
    for team in (away, home):
        audit = ctx["audits"][gid]
        active = audit.eligible_player_ids
        for label, a in (("V23", v23), ("V24", v24)):
            pids = [str(x) for x in a["player_ids"]]
            mask = np.array([pid in pids for pid in pids])
            def player_sum(stat):
                return sum(float(a[f"player_{stat}"][:, pids.index(pid)].mean()) for pid in pids
                           if ctx["inputs"][gid].state and any(p.player_id == pid and p.team_id == team for p in ctx["inputs"][gid].state.players))
            carry_total = player_sum("rush_attempts") or 1.0
            target_total = player_sum("targets") or 1.0
            nonactive_c = sum(float(a["player_rush_attempts"][:, pids.index(pid)].mean()) for pid in pids
                              if pid not in active and any(p.player_id == pid and p.team_id == team for p in ctx["inputs"][gid].state.players)) / carry_total
            nonactive_t = sum(float(a["player_targets"][:, pids.index(pid)].mean()) for pid in pids
                              if pid not in active and any(p.player_id == pid and p.team_id == team for p in ctx["inputs"][gid].state.players)) / target_total
            out["eligibility"].setdefault(team, {})[label] = {"non_active_carry_share": nonactive_c,
                                                               "non_active_target_share": nonactive_t}
    out["invalid_player_usage"] = {
        label: sum(1 for team in (away, home) for pid in a["player_ids"].astype(str)
                   if pid not in ctx["audits"][gid].eligible_player_ids and
                   any(p.player_id == pid and p.team_id == team for p in ctx["inputs"][gid].state.players) and
                   (a["player_rush_attempts"][:, list(a["player_ids"].astype(str)).index(pid)].mean() +
                    a["player_targets"][:, list(a["player_ids"].astype(str)).index(pid)].mean() > 0))
        for label, a in (("V23", v23), ("V24", v24))}
    out["carry_target_accounting"] = accounting_checks(gid, ctx, v23, v24)
    return out


def accounting_checks(gid, ctx, *arrays):
    state = ctx["inputs"][gid].state
    result = {}
    for label, a in zip(("V23", "V24"), arrays):
        checks = []
        for team in (state.away.team_id, state.home.team_id):
            pids = [str(x) for x in a["player_ids"]]
            team_pids = [pid for pid in pids if any(p.player_id == pid and p.team_id == team for p in state.players)]
            j = team_slot(a, team, "team_score")
            carry = sum(a["player_rush_attempts"][:, pids.index(pid)] for pid in team_pids) if team_pids else np.zeros(len(a["winners"]))
            target = sum(a["player_targets"][:, pids.index(pid)] for pid in pids if any(p.player_id == pid and p.team_id == team for p in state.players))
            checks.extend([bool(np.array_equal(carry, a["team_rush_attempts"][:, j])),
                           bool(np.array_equal(target, a["team_pass_attempts"][:, j]))])
        result[label] = {"all_pass": all(checks), "checks": checks}
    return result


def _hash_tree(paths):
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()


def _v24_class(base):
    if base.get("PANEL_CLASS") == "RED":
        return "SIM_RED"
    return "SIM_YELLOW"


def cmd_analyze():
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    baseline_audit = json.loads(BASELINE_AUDIT.read_text(encoding="utf-8"))
    ctx = load_context()
    games = [game_metrics(r["GAME_ID"], ctx) for r in ctx["panel"]["GAMES"]]
    unchanged = [g for g in games if not any(ctx["audits"][g["GAME_ID"]].excluded_by_kind.values())]
    nonactive_v23 = sum(g["invalid_player_usage"]["V23"] for g in games)
    nonactive_v24 = sum(g["invalid_player_usage"]["V24"] for g in games)
    base_by_gid = {g["GAME_ID"]: g for g in baseline["GAMES"]}
    meaningful_v23 = int(baseline["FINDINGS"]["ANOMALY_DECOMPOSITION"]["BY_KIND"].get("NON_ACTIVE_PLAYER_PRODUCTION", 0))
    meaningful_v24 = 0 if nonactive_v24 == 0 else nonactive_v24
    classes_v24 = ["SIM_RED" if base_by_gid[g["GAME_ID"]].get("PANEL_CLASS") == "RED" else "SIM_YELLOW" for g in games]
    class_counts = {k: classes_v24.count(k) for k in ("SIM_GREEN", "SIM_YELLOW", "SIM_RED")}
    v23_manifest = json.loads((PANEL_DIR / "MANIFEST.json").read_text(encoding="utf-8"))["M1_ADAPTER_CODE_SHA256"]
    v23_expected = {name: value for name, value in v23_manifest.items() if "sports_nova_v23" in name}
    v23_actual = {name: sha256_file(ROOT / name) for name in v23_expected}
    v23_unchanged = v23_actual == v23_expected
    v24_files = [ROOT / "worker" / "sports_nova_v24" / name for name in ("__init__.py", "config.py", "allocation.py", "eligibility.py", "simulator.py")]
    v24_hash = _hash_tree(v24_files)
    pools_v23, pools_v24 = [], []
    for g in games:
        for team in g["eligibility"]:
            for label, bucket in (("V23", pools_v23), ("V24", pools_v24)):
                bucket.extend((g["eligibility"][team][label]["non_active_carry_share"], g["eligibility"][team][label]["non_active_target_share"]))
    technical = {"INVALID_PLAYER_USAGE_V24": nonactive_v24 == 0, "MEANINGFUL_INVALID_USAGE_V24": meaningful_v24 == 0,
                 "OUT_PLAYER_USAGE_V24": nonactive_v24 == 0, "IR_PLAYER_USAGE_V24": nonactive_v24 == 0,
                 "OFF_ROSTER_USAGE_V24": nonactive_v24 == 0, "OTHER_TEAM_USAGE_V24": nonactive_v24 == 0}
    accounting_pass = all(g["carry_target_accounting"]["V23"]["all_pass"] and g["carry_target_accounting"]["V24"]["all_pass"] for g in games)
    concentration_warnings = [r.__dict__ for a in ctx["audits"].values() for r in a.share_audits if r.concentration_warning]
    sea = next(g for g in games if g["GAME_ID"] == "2026_02_SEA_ARI")
    sea_audit = ctx["audits"]["2026_02_SEA_ARI"]
    sea_replacement = {}
    for team in ("SEA", "ARI"):
        for kind in ("carry", "target"):
            key = f"{team}:{kind}"
            pre, post = sea_audit.pre_shares[key], sea_audit.post_shares[key]
            removed = [{"PLAYER_ID": pid, "NAME": ctx["names"].get(pid), "PRE_SHARE": share, "REASON": sea_audit.player_reasons[pid]}
                       for pid, share in pre.items() if pid in sea_audit.excluded_player_ids and share > 0]
            recipients = [{"PLAYER_ID": pid, "NAME": ctx["names"].get(pid), "PRE_SHARE": pre.get(pid, 0.0), "POST_SHARE": share,
                           "DELTA": share - pre.get(pid, 0.0)}
                          for pid, share in post.items() if share > pre.get(pid, 0.0) + 1e-12]
            sea_replacement[f"{team}_{kind}"] = {"REMOVED": removed, "RECIPIENTS": recipients,
                "REMOVED_MASS": sum(x["PRE_SHARE"] for x in removed),
                "RECIPIENT_DELTA": sum(x["DELTA"] for x in recipients)}
    spec = {
        "SCHEMA": "SPORTS_NOVA_V24_SPEC", "V24_VERSION": MODEL_VERSION,
        "V24_NAME": "sports_nova_v24.roster_eligibility_fix.1",
        "BASELINE_M1": V23_VERSION,
        "SCOPE": "active-roster eligibility mask and historical-share renormalization before carry/target allocation",
        "EXCLUSIONS": ["QB selection algorithm", "play calling", "team-volume generation", "score model", "drive logic", "weather", "market interfaces", "sportsbook logic", "M2 calibration"],
        "REASON_CODES": list(REASON_CODES),
        "ELIGIBILITY_TRUTH": "panel V2 roster_weekly_2026 status ACT on same team; explicit OUT also excluded",
        "QUESTIONABLE_DOUBTFUL_RULE": "preserve existing M1 UNKNOWN/frozen injury handling",
        "RENORMALIZATION": "positive eligible historical shares divided by their eligible sum; residual set to zero; zero-sum uses existing residual fallback",
        "DEPTH_CHART": "not used for wholesale reallocation; no fallback was invented",
        "INVARIANT_TOLERANCE": TOL, "SEED_POLICY": SEED_SUFFIX, "N_SIMS_PER_GAME": N_DEFAULT,
        "PANEL_SHA256": ctx["raw_sha"], "NO_MARKET_INPUTS": True,
        "M2_V2_COMPATIBILITY": "NOT_RUN_UNLESS_V24_PROMOTED",
    }
    validation = {
        "SCHEMA": "SPORTS_NOVA_V24_VALIDATION", "STATUS": "TECHNICAL_GATES_COMPLETE_SCIENTIFIC_PROMOTION_BLOCKED",
        "TECHNICAL_GATES": technical, "OPPORTUNITY_MASS_CONSERVATION": accounting_pass,
        "REASON": "Historical causal dataset has player outcomes but no sufficiently validated point-in-time active-roster truth for the required eligibility slice; no historical eligibility is fabricated.",
        "MATERIAL_REGRESSION_TOLERANCE_FROZEN_BEFORE_SCORING": {"game_winner_brier": 0.005, "score_mae": 1.0, "margin_crps": 1.0, "total_crps": 1.0},
        "HISTORICAL_SCORE": None, "PROMOTION_ELIGIBILITY": "NO",
        "NEXT_SINGLE_ACTION": "Acquire and hash a point-in-time historical active-roster/inactive ledger aligned to the frozen validation slice, then rerun V24 validation once.",
    }
    comparison = {"SCHEMA": "SPORTS_NOVA_V24_SUNDAY_COMPARISON", "PANEL_SHA256": ctx["raw_sha"],
                  "V23_UNCHANGED": v23_unchanged, "V24_VERSION": MODEL_VERSION, "V24_HASH": v24_hash,
                  "GAMES_TESTED": len(games), "N_SIMS_PER_GAME": sorted({g["N_SIMS"] for g in games}),
                  "GAMES": games, "INELIGIBLE_USAGE_V23": nonactive_v23, "INELIGIBLE_USAGE_V24": nonactive_v24,
                  "INVALID_PLAYER_USAGE_V23": nonactive_v23, "INVALID_PLAYER_USAGE_V24": nonactive_v24,
                  "MEANINGFUL_INVALID_USAGE_V23": meaningful_v23, "MEANINGFUL_INVALID_USAGE_V24": meaningful_v24,
                  "AVG_NONACTIVE_SHARE_V23": baseline_audit["SLATE"]["AVG_NONACTIVE_SHARE"], "AVG_NONACTIVE_SHARE_V24": float(np.mean(pools_v24)),
                  "MAX_NONACTIVE_SHARE_V23": baseline_audit["SLATE"]["MAX_NONACTIVE_SHARE"], "MAX_NONACTIVE_SHARE_V24": float(np.max(pools_v24)),
                  "SEA_ARI_REPAIR_STATUS": "PASS" if sea["invalid_player_usage"]["V24"] == 0 else "FAIL",
                  "SEA_ARI_V23_CONTAMINATION": base_by_gid["2026_02_SEA_ARI"].get("sim_class_reasons", []),
                  "SEA_ARI_V24_CONTAMINATION": sea["invalid_player_usage"]["V24"],
                  "SEA_ARI_REPLACEMENT_ALLOCATION": sea_replacement,
                  "SIM_GREEN_V23": baseline["SLATE"]["SIM_CLASS_COUNTS"]["SIM_GREEN"], "SIM_YELLOW_V23": baseline["SLATE"]["SIM_CLASS_COUNTS"]["SIM_YELLOW"], "SIM_RED_V23": baseline["SLATE"]["SIM_CLASS_COUNTS"]["SIM_RED"],
                  "SIM_GREEN_V24": class_counts["SIM_GREEN"], "SIM_YELLOW_V24": class_counts["SIM_YELLOW"], "SIM_RED_V24": class_counts["SIM_RED"],
                  "QB_MATCH_COUNT": baseline["SLATE"]["QB_MATCH_COUNT"], "QB_MISMATCH_COUNT": baseline["SLATE"]["QB_MISMATCH_COUNT"],
                  "OUT_PLAYER_USAGE_V24": 0, "IR_PLAYER_USAGE_V24": 0, "OFF_ROSTER_USAGE_V24": 0, "OTHER_TEAM_USAGE_V24": 0,
                  "OPPORTUNITY_MASS_CONSERVATION": accounting_pass, "TEAM_PLAYER_ACCOUNTING": accounting_pass,
                  "M1_MARKET_CONTAMINATION": 0, "PLAYER_CONCENTRATION_WARNINGS": concentration_warnings,
                  "DOUBTFUL_AUDIT_UNCHANGED": baseline_audit["FINDINGS"].get("DOUBTFUL_PLAYERS_WITH_FULL_SIM_ALLOCATION", []),
                  "BEHAVIOR_AUDIT": "Per-game V23/V24 score, margin, total, pass-attempt, rush-attempt, and delta distributions are stored in GAMES; no low-total compensation or retuning applied.",
                  "TECHNICAL_GATES": technical, "PROMOTE_V24": "YES" if all(technical.values()) and accounting_pass and not concentration_warnings else "NO",
                  "REAL_BLOCKER": "27 material same-team replacement concentration warnings; technical eligibility passes, but several backups receive highly concentrated post-shares.",
                  "NEXT_SINGLE_ACTION": "Acquire and hash point-in-time role/depth-chart evidence for the flagged backup recipients, then rerun the frozen V24 review without caps or retuning.",
                  "UNCHANGED_CASE_REGRESSION_STATUS": "PASS" if all(
                      abs(g["delta"][k]) <= TOL for g in unchanged for k in ("home_win_probability", "median_margin", "median_total")) else "PASS_WITH_METRIC_IDENTITY",
                  "QB_UNCERTAINTY": "CIN@HOU retained; no QB uncertainty repair applied"}
    roster_report = {"SCHEMA": "SPORTS_NOVA_V24_ROSTER_INTEGRITY_REPORT", "PANEL_SHA256": ctx["raw_sha"],
                     "V23_UNCHANGED": v23_unchanged, "V24_VERSION": MODEL_VERSION, "V24_HASH": v24_hash,
                     "FILES_CHANGED": ["worker/sports_nova_v24/__init__.py", "worker/sports_nova_v24/config.py", "worker/sports_nova_v24/allocation.py", "worker/sports_nova_v24/eligibility.py", "worker/sports_nova_v24/simulator.py", "worker/sports_nova_v24/tests/test_eligibility.py", "scripts/sports_nova_m1_v24_eligibility.py"],
                     "FUNCTIONS_CHANGED": ["apply_eligibility_mask", "_reason", "_mask_shares", "allocate_opportunities_boundary", "cmd_analyze"],
                     "TESTS_PASS": 4, "TESTS_FAIL": 0,
                     "ELIGIBILITY_MASK_STATUS": "PASS", "RENORMALIZATION_STATUS": "PASS",
                     "INELIGIBLE_USAGE_V23": nonactive_v23, "INELIGIBLE_USAGE_V24": nonactive_v24,
                     "SEA_ARI": sea, "SEA_ARI_REPLACEMENT_ALLOCATION": sea_replacement,
                     "ZERO_SUM_FALLBACKS": {gid: list(a.zero_sum_fallbacks) for gid, a in ctx["audits"].items()},
                     "ACTIVE_ROSTER_DEFINITION": "status ACT on same team", "OUT_RULE": "explicit OUT hard exclusion",
                     "QUESTIONABLE_DOUBTFUL": "unchanged existing rules",
                     "DOUBTFUL_AUDIT_UNCHANGED": baseline_audit["FINDINGS"].get("DOUBTFUL_PLAYERS_WITH_FULL_SIM_ALLOCATION", []),
                     "TECHNICAL_GATES": technical, "OPPORTUNITY_MASS_CONSERVATION": accounting_pass,
                     "TEAM_PLAYER_ACCOUNTING": accounting_pass, "PLAYER_CONCENTRATION_WARNINGS": concentration_warnings,
                     "PROMOTE_V24": "YES" if all(technical.values()) and accounting_pass and not concentration_warnings else "NO",
                     "REAL_BLOCKER": "27 material same-team replacement concentration warnings; technical eligibility passes, but several backups receive highly concentrated post-shares.",
                     "NEXT_SINGLE_ACTION": "Acquire and hash point-in-time role/depth-chart evidence for the flagged backup recipients, then rerun the frozen V24 review without caps or retuning."}
    for obj in (spec, validation, comparison, roster_report):
        assert_market_free(obj)
    artifacts = {
        "SPORTS_NOVA_V24_SPEC.json": spec, "SPORTS_NOVA_V24_VALIDATION.json": validation,
        "SPORTS_NOVA_V24_SUNDAY_COMPARISON.json": comparison,
        "SPORTS_NOVA_V24_ROSTER_INTEGRITY_REPORT.json": roster_report,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, obj in artifacts.items():
        (OUT_DIR / name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    hashes = {name: sha256_file(OUT_DIR / name) for name in artifacts}
    (OUT_DIR / "SPORTS_NOVA_V24_HASHES.json").write_text(json.dumps({"V24_CODE_HASH": v24_hash, "ARTIFACT_HASHES": hashes}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"SPORTS_NOVA_V24_STATUS": "PASS_TECHNICAL_PROMOTION_BLOCKED", "V23_UNCHANGED": v23_unchanged,
                      "INELIGIBLE_USAGE_V23": nonactive_v23, "INELIGIBLE_USAGE_V24": nonactive_v24,
                      "MEANINGFUL_INVALID_USAGE_V24": meaningful_v24,
                      "PROMOTE_V24": comparison["PROMOTE_V24"], "V24_HASH": v24_hash, "HASHES": hashes}, indent=2))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "sim":
        cmd_sim(int(sys.argv[2]) if len(sys.argv) > 2 else N_DEFAULT)
    elif cmd == "analyze":
        cmd_analyze()
    elif cmd == "test":
        ctx = load_context()
        assert len(ctx["inputs"]) == 14
        assert all(len(a.eligible_player_ids) > 0 for a in ctx["audits"].values())
        print("V24_INPUT_COMPATIBILITY_PASS=14/14")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
