"""SPORTS_NOVA V26 active-skill state completion: audit -> freeze -> sim -> analyze (14-game Sunday slate, panel V2, N=5000, same seeds as V23/V24/V25).

  python scripts/sports_nova_m1_v26_active_skill_state.py audit           # Phase 1-2: expected vs present, missing-reason classes (read-only)
  python scripts/sports_nova_m1_v26_active_skill_state.py freeze          # SPORTS_NOVA_V26_PREREG.json; refuses to run if any V26 sim output exists
  python scripts/sports_nova_m1_v26_active_skill_state.py sim [N] [resume]
  python scripts/sports_nova_m1_v26_active_skill_state.py analyze

V23, both V24 packages and V25 are never modified.  Inputs come only from the immutable panel V2 directory (via the V23 baseline runner's load_inputs()).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sports_nova_m1_v25_role_aware as R25  # noqa: E402  (V25 runner: reused for inputs/snapshots, not modified)
from worker.sports_nova_modular.firewall import assert_market_free  # noqa: E402
from worker.sports_nova_sunday_panel import adapter  # noqa: E402
from worker.sports_nova_v26_active_skill_state import config as cfg  # noqa: E402
from worker.sports_nova_v26_active_skill_state import simulator as v26  # noqa: E402
from worker.sports_nova_v26_active_skill_state.completion import complete_state, verify_machinery  # noqa: E402

base = R25.base
PKG = "worker/sports_nova_v26_active_skill_state"
DATA = base.DATA
V23_DIR, V24_DIR, V25_DIR = R25.V23_DIR, R25.V24_DIR, R25.OUT_DIR
OUT_DIR = DATA / "SUNDAY_2026_09_20_M1_V26_ACTIVE_SKILL_STATE"
PREREG_DIR = DATA / "V26_PREREG"
PREREG_FILE = PREREG_DIR / "SPORTS_NOVA_V26_PREREG.json"
N_DEFAULT = 5000
PATHS_AUDITED = R25.EIGHT_PATHS + [("WAS", "target", "Terry McLaurin")]


def sha256_file(p: Path) -> str:
    return base.sha256_file(p)


def pkg_files() -> list[Path]:
    return sorted(p for p in (ROOT / PKG).glob("*.py"))


def pkg_hashes() -> dict:
    return {f"{PKG}/{p.name}": sha256_file(p) for p in pkg_files()}


def v26_hash() -> str:
    return hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(pkg_hashes().items())).encode()).hexdigest()


def v25_files_unchanged() -> dict:
    return {k: sha256_file(ROOT / k) == v for k, v in cfg.V25_PACKAGE_FILE_SHA256.items()}


def known_out(rec: dict) -> frozenset:
    return frozenset(p["PLAYER_ID"] for s in ("AWAY", "HOME") for p in rec["INJURY_STATE"][s]["VALUE"]["PLAYERS"] if p["M1_AVAILABILITY"] == "OUT")


def prior_for(rec: dict, ctx: dict):
    return adapter._prior(ctx["panel_df"], rec["SEASON"], rec["WEEK"])


def completions(ctx: dict, share_basis: str = cfg.SHARE_BASIS_OFFICIAL) -> dict:
    """gid -> (snapshot, prior, CompletionResult) for every game with a state."""
    out = {}
    for rec in ctx["games"]:
        gid, inp = rec["GAME_ID"], ctx["inputs"][rec["GAME_ID"]]
        if inp.state is None:
            continue
        snap, prior = R25.snapshot_for(rec, ctx), prior_for(rec, ctx)
        out[gid] = (snap, prior, complete_state(inp.state, snap, prior, known_out=known_out(rec), share_basis=share_basis))
    return out


# ---------------------------------------------------------------------------------------------- audit (Phase 1-2)
def build_audit(ctx: dict, comps: dict) -> dict:
    names = ctx["names"]
    per_team, rows, verify = [], [], {}
    for rec in ctx["games"]:
        gid = rec["GAME_ID"]
        snap, prior, comp = comps[gid]
        verify[gid] = verify_machinery(ctx["inputs"][gid].state, prior)
        for t, c in comp.teams.items():
            per_team.append({"GAME": gid, "TEAM": t, "active_skill_expected": c["ACTIVE_SKILL_EXPECTED"], "active_skill_in_M1_before": c["PRESENT_BEFORE"],
                             "missing_from_M1": c["MISSING"], "successfully_added": c["ADDED"], "blocked": c["BLOCKED"],
                             "reason_missing": {}})
        for r in comp.rows:
            rows.append(dict(r, GAME=gid, NAME=names.get(r["PLAYER_ID"])))
    idx = {(p["GAME"], p["TEAM"]): p for p in per_team}
    for r in rows:
        d = idx[(r["GAME"], r["TEAM"])]["reason_missing"]
        d[r["REASON"]] = d.get(r["REASON"], 0) + 1
    by_reason, by_action, by_role = {}, {}, {}
    for r in rows:
        by_reason[r["REASON"]] = by_reason.get(r["REASON"], 0) + 1
        by_action[r["ACTION"]] = by_action.get(r["ACTION"], 0) + 1
        by_role[r["ROLE"]] = by_role.get(r["ROLE"], 0) + 1
    pos_other = 0
    for gid, (snap, prior, comp) in comps.items():
        for p in ctx["inputs"][gid].state.players:
            if p.position == "OTHER" and snap.roster.get(p.player_id, (None, None)) == (p.team_id, "ACT") and \
                    next((c for c in snap.positions.get(p.player_id, (None, None)) if c in cfg.RECIPIENT_ROLES), None):
                pos_other += 1
    exp = sum(t["active_skill_expected"] for t in per_team)
    prev = sum(t["active_skill_in_M1_before"] for t in per_team)
    added = sum(t["successfully_added"] for t in per_team)
    return {"SCHEMA": "SPORTS_NOVA_V26_STATE_COMPLETENESS_AUDIT", "ACTIVE_SKILL_SET_RULE": cfg.COMPLETION_POLICY["ACTIVE_SKILL_SET"],
            "TOTAL": {"expected": exp, "previously_present": prev, "missing": exp - prev, "added": added, "blocked_unresolved": exp - prev - added},
            "MISSING_BY_ROLE": by_role, "MISSING_BY_REASON": by_reason, "BY_ACTION": by_action,
            "PRESENT_BUT_STATE_POSITION_OTHER_ROLE_RESOLVED_FROM_ROSTER": pos_other,
            "MACHINERY_VERIFICATION": verify, "PER_TEAM": per_team, "MISSING_PLAYERS": rows}


def cmd_audit() -> None:
    ctx = base.load_inputs()
    comps = completions(ctx)
    a = build_audit(ctx, comps)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "STATE_COMPLETENESS_AUDIT.json").write_text(json.dumps(a, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({k: a[k] for k in ("TOTAL", "MISSING_BY_ROLE", "MISSING_BY_REASON", "BY_ACTION", "PRESENT_BUT_STATE_POSITION_OTHER_ROLE_RESOLVED_FROM_ROSTER")}, indent=1))


# ---------------------------------------------------------------------------------------------- rules frozen BEFORE any V26 sim output
PATH_RULES_V26 = dict(R25.PATH_CLASSIFICATION_RULES,
                      SOURCE_OF_TAXONOMY="V25's own deterministic taxonomy, unchanged, extended with one extra audited path (WAS:target:Terry McLaurin); frozen before any V26 sim output exists",
                      EXTRA_PATH="WAS:target:Terry McLaurin (flagged in the V26 brief; the gate requires it too, which is STRICTER than V25's eight-path gate)")
THIN_POOL_RULES = {
    "POOL": "one team x {carry, target} pool exactly as handed to the V23 draw (V25 post shares, QB/held roles included, sums to 1)",
    "eligible_players": "state players with recipient role (RB/FB/WR/TE) that are eligible under V25 eligibility (ACT, on this team, not OUT), zero-share included",
    "surviving_players": "recipient-role players with post share > 0", "top_share/top_2_share": "largest and two largest single-player post shares in the pool",
    "effective_pool_size": "1 / sum(post_share^2) over the pool",
    "historical_reference": "2019-2025 REG team-season distribution of the same statistics over non-QB RB/FB/WR/TE players (median, p90, p99 of top share; median, p10 of effective pool size), computed from the panel itself",
    "SEVERE_THIN_POOL_ARTIFACT": "any eligible recipient whose V26 state share exceeds the frozen single-player p99 envelope (carry 0.838 / target 0.328) -> player-prop gate cannot pass",
    "NO_CAPS": True}
GATE_RULES_V26 = {
    "PLAYER_PROP_MIRROR_READY": "YES iff invalid_usage==0 AND meaningful_invalid_usage==0 AND mass_conservation==PASS AND accounting==PASS AND qb_mismatch==0 AND market_contamination==0 "
                                "AND active_skill_state_complete_for_valid_history AND all NINE audited paths in {SUPPORTED, MECHANICALLY_RESOLVED, RESOLVED} AND no SEVERE_THIN_POOL_ARTIFACT AND NEW_SEVERE_CONCENTRATION==0. "
                                "Identical to V25's gate, not weakened; the ninth path and the thin-pool clause only tighten it.",
    "GAME_LEVEL_MIRROR_READY": "V25's pre-registered sub-gate, unchanged (technical gates + QB inflation removed + no held-role concentration + V26-vs-V23 shift inside the V24 regression rules)",
    "MARKET_LABELS": {"WINNER": "FORWARD_RESEARCH", "TOTAL": "KNOWN_V23_LOW_TOTALS_BIAS; DO_NOT_INTERPRET_DISAGREEMENT_AS_EDGE", "TEAM_TOTAL": "RESEARCH_ONLY"},
    "NOT_A_PROMOTION": "V23 remains champion; neither flag promotes V25 or V26; M2 uncalibrated for both"}
DEFERRED = ["QB_SCRAMBLE_DOUBLE_COUNT", "LOW_TOTALS_BIAS", "OVERTIME_MODEL", "M2_RECALIBRATION"]


def dir_hashes(d: Path, pattern: str = "*.py") -> dict:
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(d.glob(pattern))}


def mirror_hashes() -> dict:
    m = V25_DIR / "MIRROR"
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(m.rglob("*")) if p.is_file()} if m.exists() else {}


def cmd_freeze() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.rglob("*.npz")):
        raise SystemExit("REFUSING: V26 simulation output already exists; the pre-registration must precede it")
    if PREREG_FILE.exists():
        raise SystemExit(f"REFUSING: {PREREG_FILE.name} already exists (a rule change ships as a new version, never an edit)")
    PREREG_DIR.mkdir(parents=True, exist_ok=True)
    unchanged25 = v25_files_unchanged()
    assert all(unchanged25.values()), [k for k, v in unchanged25.items() if not v]
    assert R25.v25_hash() == cfg.V25_HASH, "V25 hash drifted"
    before = json.loads(R25.V23_BEFORE.read_text())["FILES"]
    unchanged23 = {k: sha256_file(ROOT / k) == v for k, v in before.items()}
    assert all(unchanged23.values()), [k for k, v in unchanged23.items() if not v]
    ctx = base.load_inputs()
    comps = completions(ctx)
    audit = build_audit(ctx, comps)
    positive = [{"GAME": r["GAME"], "TEAM": r["TEAM"], "NAME": r["NAME"], "TARGETS_THIS_TEAM": r["TARGETS_THIS_TEAM"], "CARRIES_THIS_TEAM": r["CARRIES_THIS_TEAM"]}
                for r in audit["MISSING_PLAYERS"] if r["ACTION"] in (cfg.ACTION_ADDED, cfg.ACTION_ADDED_ZERO_SHARE) and (r["TARGETS_THIS_TEAM"] > 0 or r["CARRIES_THIS_TEAM"] > 0)]
    t = subprocess.run([sys.executable, "-m", "pytest", f"{PKG}/tests", "-q"], cwd=ROOT, capture_output=True, text=True)
    tail = t.stdout.strip().splitlines()[-1] if t.stdout.strip() else t.stderr[-200:]
    rec = {"SCHEMA": "SPORTS_NOVA_V26_PREREG", "FROZEN_AT": datetime.now(timezone.utc).isoformat(), "MODEL_VERSION": cfg.MODEL_VERSION, "IMPLEMENTATION_PACKAGE": PKG,
           "PACKAGE_FILE_SHA256": pkg_hashes(), "V26_HASH": v26_hash(), "COMPLETION_POLICY": cfg.COMPLETION_POLICY, "COMPLETION_POLICY_HASH": cfg.COMPLETION_POLICY_HASH,
           "V25_FILES_BYTE_IDENTICAL_TO_V25_FREEZE": unchanged25, "V25_HASH": cfg.V25_HASH,
           "V23_HASHES_BEFORE_FILE": R25.V23_BEFORE.relative_to(ROOT).as_posix(), "V23_HASHES_BEFORE_SHA256": sha256_file(R25.V23_BEFORE), "V23_FILES_UNCHANGED_AT_FREEZE": True,
           "V24_ROSTER_ELIGIBILITY_FILES_AT_FREEZE": dir_hashes(ROOT / "worker/sports_nova_v24_roster_eligibility"), "V24_CODEX_FILES_AT_FREEZE": dir_hashes(ROOT / "worker/sports_nova_v24"),
           "V25_MIRROR_FILES_AT_FREEZE": mirror_hashes(),
           "AUDIT_AT_FREEZE": {k: audit[k] for k in ("TOTAL", "MISSING_BY_ROLE", "MISSING_BY_REASON", "BY_ACTION")},
           "PREDICTION_REGISTERED_BEFORE_SIM": {
               "ADDED_PLAYERS_WITH_POSITIVE_MACHINERY_SHARE": positive,
               "STATEMENT": "Only added players with volume on their own team's rows can change a share vector; all others enter at share 0 and cannot receive a draw. "
                            "V26 simulations are therefore predicted bit-identical to V25 (same seeds) in every game without such a player, and V26 will not materially change any thin-pool concentration. "
                            "The portable-share arithmetic is reported as a diagnostic counterfactual only.",
               "THIN_POOL_CLAUSE_EXPECTED": "NOT_RESOLVED"},
           "PATH_CLASSIFICATION_RULES": PATH_RULES_V26, "PATHS_AUDITED": [list(x) for x in PATHS_AUDITED], "THIN_POOL_RULES": THIN_POOL_RULES, "GATE_RULES": GATE_RULES_V26,
           "QB_REALISM_RULES": R25.QB_REALISM_RULES, "REGRESSION_RULES": R25.REGRESSION_RULES, "DEFERRED_DEFECTS": DEFERRED,
           "PANEL_SHA256": base.PANEL_SHA_EXPECTED, "SEED_POLICY": base.SEED_POLICY, "N_SIMS_PER_GAME": N_DEFAULT, "DATE": "2026-09-20", "GAMES": 14,
           "ADVISOR": "one advisor pass before implementation: the thin-pool clause is decided by arithmetic (4 of 5 flagged teams have no missing skill player with volume); "
                      "IDENTITY_JOIN needs a portability policy that does not exist; do not edit make_state; chain V26 before V25",
           "UNIT_TESTS_AT_FREEZE": tail, "V26_SIM_OUTPUT_EXISTED_AT_FREEZE": False}
    assert_market_free({k: rec[k] for k in ("SCHEMA", "MODEL_VERSION", "PACKAGE_FILE_SHA256", "V26_HASH", "COMPLETION_POLICY_HASH", "AUDIT_AT_FREEZE")})
    PREREG_FILE.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("V26_HASH", "COMPLETION_POLICY_HASH", "UNIT_TESTS_AT_FREEZE")}, indent=1))
    print("POSITIVE-SHARE ADDITIONS:", json.dumps(positive))


# ---------------------------------------------------------------------------------------------- sim
def sim_worker_v26(args):
    gid, state, resolutions, n, outdir, snap, comp = args
    base.MODEL_VERSION = cfg.MODEL_VERSION
    base.simulate_game = lambda st, nn, seed, mv: v26.simulate_game(st, nn, seed, mv, roster=snap, completion=comp)
    return base.sim_worker((gid, state, resolutions, n, outdir))


def cmd_sim(n: int, resume: bool = False) -> None:
    if not PREREG_FILE.exists():
        raise SystemExit("run `freeze` first: SPORTS_NOVA_V26_PREREG.json must exist before any V26 simulation")
    freeze = json.loads(PREREG_FILE.read_text())
    assert freeze["V26_HASH"] == v26_hash(), "V26 package changed after the freeze"
    assert all(v25_files_unchanged().values()), "V25 package changed"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audit").mkdir(exist_ok=True)
    before = base.engine_hashes()
    ctx = base.load_inputs()
    assert ctx["raw_sha"] == base.PANEL_SHA_EXPECTED, "PANEL SHA MISMATCH"
    res = base.all_resolutions(ctx["inputs"])
    comps = completions(ctx)
    jobs, names = [], ctx["names"]
    for rec in ctx["games"]:
        gid, inp = rec["GAME_ID"], ctx["inputs"][rec["GAME_ID"]]
        if inp.state is None:
            print(f"SKIP {gid}: {inp.missing}", flush=True)
            continue
        snap, prior, comp = comps[gid]
        repaired, audit = v26.prepare_state(inp.state, snap, prior, known_out=known_out(rec))[1:]
        (OUT_DIR / "audit" / f"{gid}.json").write_text(json.dumps({
            "GAME_ID": gid, "CHANGED": audit.changed, "COMPLETION_CHANGED": comp.changed, "RAW_STATE_HASH": comp.raw_state_hash, "COMPLETED_STATE_HASH": comp.completed_state_hash,
            "REPAIRED_STATE_HASH": audit.repaired_state_hash, "QB_INPUTS_UNCHANGED": audit.qb_inputs_unchanged, "COMPLETION_TEAMS": comp.teams,
            "COMPLETION_ROWS": [dict(r, NAME=names.get(r["PLAYER_ID"])) for r in comp.rows],
            "EXCLUDED_BY_REASON": {k: [dict(v, NAME=names.get(v["PLAYER_ID"])) for v in vs] for k, vs in audit.excluded().items()},
            "SUMMARIES": audit.summaries, "ROWS": [dict(r, NAME=names.get(r["PLAYER_ID"])) for r in audit.rows]}, indent=1, sort_keys=True), encoding="utf-8")
        if resume and (OUT_DIR / "raw" / f"{gid}.npz").exists() and (OUT_DIR / "games" / f"{gid}.json").exists():
            continue
        jobs.append((gid, inp.state, res, n, OUT_DIR, snap, comp))
    base.set_identity_resolutions(res)
    gid0 = "2026_02_IND_KC"
    snap0, _, comp0 = comps[gid0]
    args0 = (ctx["inputs"][gid0].state, 40, base.seed_for(gid0), cfg.MODEL_VERSION)
    x = v26.simulate_game(*args0, roster=snap0, completion=comp0)
    y = v26.simulate_game(*args0, roster=snap0, completion=comp0)
    det = all(np.array_equal(x.player_stats[s], y.player_stats[s]) for s in base.STAT_NAMES)
    print(f"determinism smoke (N=40 x2, IND_KC): {det}", flush=True)
    prior_meta = json.loads((OUT_DIR / "sim_run_meta.json").read_text()) if resume and (OUT_DIR / "sim_run_meta.json").exists() else {}
    (OUT_DIR / "sim_run_meta.json").write_text(json.dumps({
        "RESUMED_AT": datetime.now(timezone.utc).isoformat() if resume else None, "FIRST_STARTED_AT": prior_meta.get("FIRST_STARTED_AT") or prior_meta.get("STARTED_AT"),
        "STARTED_AT": prior_meta.get("STARTED_AT") or datetime.now(timezone.utc).isoformat(), "N_SIMS": n, "MODEL_VERSION": cfg.MODEL_VERSION, "V26_HASH": v26_hash(),
        "ENGINE_HASHES_BEFORE": before, "DETERMINISM_SMOKE": det, "PANEL_SHA256": ctx["raw_sha"], "SEED_POLICY": base.SEED_POLICY,
        "PREREG_SHA256": sha256_file(PREREG_FILE)}, indent=1), encoding="utf-8")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(sim_worker_v26, j): j[0] for j in jobs}
        for f in as_completed(futs):
            gid, el = f.result()
            print(f"done {gid} {el:.0f}s wall={time.time() - t0:.0f}s", flush=True)
    after = base.engine_hashes()
    meta = json.loads((OUT_DIR / "sim_run_meta.json").read_text())
    meta.update({"FINISHED_AT": datetime.now(timezone.utc).isoformat(), "ENGINE_HASHES_AFTER": after, "ENGINE_HASHES_UNCHANGED_DURING_SIM": before == after, "WALL_SEC": time.time() - t0})
    (OUT_DIR / "sim_run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("SIM COMPLETE; engine unchanged during run:", before == after, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "audit":
        cmd_audit()
    elif cmd == "freeze":
        cmd_freeze()
    elif cmd == "sim":
        cmd_sim(int(sys.argv[2]) if len(sys.argv) > 2 else N_DEFAULT, resume=len(sys.argv) > 3 and sys.argv[3] == "resume")
    elif cmd == "analyze":
        from sports_nova_m1_v26_active_skill_state_analysis import cmd_analyze
        cmd_analyze()
    else:
        raise SystemExit(__doc__)
