"""SPORTS_NOVA V24 roster-eligibility repair: freeze -> sim -> analyze (14-game Sunday slate, panel V2, N=5000, same seed policy as V23).

  python scripts/sports_nova_m1_v24_roster_eligibility.py freeze     # pre-registration record; refuses to run if any V24 sim output exists
  python scripts/sports_nova_m1_v24_roster_eligibility.py sim [N]
  python scripts/sports_nova_m1_v24_roster_eligibility.py analyze
  python scripts/sports_nova_m1_v24_roster_eligibility.py scan

V23 is never modified.  Inputs come only from the immutable panel V2 directory (via the V23 baseline runner's own load_inputs()).
The V23 runner is imported as `base` and reused for sim summaries and contamination analysis so V23 and V24 are measured identically.
"""
from __future__ import annotations

import ast
import difflib
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

import sports_nova_sunday_m1_baseline_v1 as base  # noqa: E402  (V23 baseline runner: reused, not modified)
from worker.sports_nova_modular.firewall import assert_market_free  # noqa: E402
from worker.sports_nova_v24_roster_eligibility import config as cfg  # noqa: E402
from worker.sports_nova_v24_roster_eligibility import simulator as v24  # noqa: E402
from worker.sports_nova_v24_roster_eligibility.eligibility import RosterSnapshot  # noqa: E402

PKG = "worker/sports_nova_v24_roster_eligibility"
DATA = base.DATA
V23_DIR = DATA / "SUNDAY_2026_09_20_M1_BASELINE_V1"
OUT_DIR = DATA / "SUNDAY_2026_09_20_M1_V24_ROSTER_ELIGIBILITY"
PREREG = DATA / "V24_PREREG"
FREEZE_FILE = PREREG / "V24_POLICY_FREEZE.json"
V23_BEFORE = PREREG / "V23_HASHES_BEFORE.json"
N_DEFAULT = 5000
# Pre-registered BEFORE any V24 simulation output exists (recorded verbatim in the freeze file):
REGRESSION_RULES = {
    "SLATE_MEAN_OF_MEDIAN_TOTALS_DROP_POINTS": 1.5,      # V24 slate mean of median totals below V23's by more than this = regression
    "GAME_MEDIAN_TOTAL_SHIFT_POINTS": 4.0,                # any game's median total moves by more than this
    "TEAM_MEAN_ATTEMPTS_SHIFT_PCT": 5.0,                  # any team's mean pass or rush attempts moves by more than this percent
    "NOISE_NOTE": "V23 and V24 use the same per-game seeds, but the RNG stream shifts wherever the allocation vector length changes, so "
                  "differences contain Monte-Carlo noise; bootstrap SE of the median difference is reported next to every delta",
}
PROMOTION_RULES = {
    "TECHNICAL_GATES": ["INVALID_INELIGIBLE_PLAYER_USAGE==0", "WRONG_TEAM_PLAYER_USAGE==0", "OUT_PLAYER_USAGE==0", "IR_INELIGIBLE_USAGE==0",
                        "TEAM_PLAYER_ACCOUNTING==PASS", "OPPORTUNITY_MASS_CONSERVATION==PASS", "QB_LOGIC_UNCHANGED", "M1_MARKET_CONTAMINATION==0", "V23_UNCHANGED"],
    "MATERIALLY_MORE_FOOTBALL_VALID": "non-eligible simulated usage eliminated (>=99% reduction in average non-eligible share) AND no new critical accounting defect "
                                      "AND every UNCERTAINTY_FLAG'd replacement is disclosed",
    "NO_SEVERE_UNEXPLAINED_REGRESSION": "none of REGRESSION_RULES triggers, or every trigger has a stated mechanism",
    "NOTE": "PROMOTE_V24=YES only if all of the above; otherwise NO and V23 stays champion",
}


def sha256_file(p: Path) -> str:
    return base.sha256_file(p)


def pkg_files() -> list[Path]:
    return sorted(p for p in (ROOT / PKG).glob("*.py"))


def pkg_hashes() -> dict:
    return {f"{PKG}/{p.name}": sha256_file(p) for p in pkg_files()}


def v24_hash() -> str:
    return hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(pkg_hashes().items())).encode()).hexdigest()


def snapshot_for(rec: dict, ctx: dict) -> RosterSnapshot:
    status = {}
    for side in ("AWAY", "HOME"):
        for p in rec["INJURY_STATE"][side]["VALUE"]["PLAYERS"]:
            if p["GAME_STATUS"] in ("OUT", "DOUBTFUL", "QUESTIONABLE"):
                status[p["PLAYER_ID"]] = p["GAME_STATUS"]
    assert int(rec["WEEK"]) == ctx["roster_week"], "roster week != game week"
    inv = ctx["inv"]
    return RosterSnapshot(roster=ctx["roster_idx"], game_status=status, roster_week=ctx["roster_week"],
                          source_sha256={k: inv[k]["sha256"] for k in ("roster_weekly_2026.parquet", "injuries_2026.parquet") if k in inv})


# ---------------------------------------------------------------------------------------------- freeze
def cmd_freeze() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.rglob("*.npz")):
        raise SystemExit("REFUSING: V24 simulation output already exists; the pre-registration must precede it")
    if FREEZE_FILE.exists():
        raise SystemExit(f"REFUSING: {FREEZE_FILE.name} already exists (a rule change ships as a new version, never an edit)")
    t = subprocess.run([sys.executable, "-m", "pytest", f"{PKG}/tests", "-q"], cwd=ROOT, capture_output=True, text=True)
    tail = t.stdout.strip().splitlines()[-1] if t.stdout.strip() else t.stderr[-200:]
    rec = {"SCHEMA": "SPORTS_NOVA_V24_POLICY_FREEZE", "FROZEN_AT": datetime.now(timezone.utc).isoformat(), "MODEL_VERSION": cfg.MODEL_VERSION,
           "IMPLEMENTATION_PACKAGE": PKG, "PACKAGE_FILE_SHA256": pkg_hashes(), "V24_HASH": v24_hash(),
           "V23_HASHES_BEFORE_FILE": str(V23_BEFORE.relative_to(ROOT)).replace("\\", "/"), "V23_HASHES_BEFORE_SHA256": sha256_file(V23_BEFORE),
           "ELIGIBILITY_POLICY": cfg.ELIGIBILITY_POLICY, "ELIGIBILITY_POLICY_HASH": cfg.ELIGIBILITY_POLICY_HASH,
           "REDISTRIBUTION_POLICY": cfg.REDISTRIBUTION_POLICY, "REDISTRIBUTION_POLICY_HASH": cfg.REDISTRIBUTION_POLICY_HASH,
           "DOUBTFUL_POLICY": cfg.DOUBTFUL_POLICY, "DOUBTFUL_POLICY_HASH": cfg.DOUBTFUL_POLICY_HASH,
           "DOUBTFUL_EVIDENCE_FILE": "V24_PREREG/DOUBTFUL_EVIDENCE.json", "DOUBTFUL_EVIDENCE_SHA256": sha256_file(PREREG / "DOUBTFUL_EVIDENCE.json"),
           "REGRESSION_RULES": REGRESSION_RULES, "PROMOTION_RULES": PROMOTION_RULES,
           "PANEL_SHA256": base.PANEL_SHA_EXPECTED, "SEED_POLICY": base.SEED_POLICY, "N_SIMS_PER_GAME": N_DEFAULT,
           "UNIT_TESTS_AT_FREEZE": tail, "V24_SIM_OUTPUT_EXISTED_AT_FREEZE": False}
    assert_market_free(rec)
    FREEZE_FILE.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("V24_HASH", "ELIGIBILITY_POLICY_HASH", "REDISTRIBUTION_POLICY_HASH", "DOUBTFUL_POLICY_HASH", "UNIT_TESTS_AT_FREEZE")}, indent=1))


# ---------------------------------------------------------------------------------------------- sim
def sim_worker_v24(args):
    gid, state, resolutions, n, outdir, snap = args
    base.MODEL_VERSION = cfg.MODEL_VERSION
    base.simulate_game = lambda st, nn, seed, mv: v24.simulate_game(st, nn, seed, mv, roster=snap)
    return base.sim_worker((gid, state, resolutions, n, outdir))


def cmd_sim(n: int, resume: bool = False) -> None:
    if not FREEZE_FILE.exists():
        raise SystemExit("run `freeze` first: the policy pre-registration must exist before any V24 simulation")
    freeze = json.loads(FREEZE_FILE.read_text())
    assert freeze["V24_HASH"] == v24_hash(), "V24 package changed after the freeze"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audit").mkdir(exist_ok=True)
    before = base.engine_hashes()
    ctx = base.load_inputs()
    assert ctx["raw_sha"] == base.PANEL_SHA_EXPECTED, "PANEL SHA MISMATCH"
    res = base.all_resolutions(ctx["inputs"])
    jobs = []
    for rec in ctx["games"]:
        gid, inp = rec["GAME_ID"], ctx["inputs"][rec["GAME_ID"]]
        if inp.state is None:
            print(f"SKIP {gid}: {inp.missing}", flush=True)
            continue
        snap = snapshot_for(rec, ctx)
        repaired, audit = v24.prepare_state(inp.state, snap)
        names = ctx["names"]
        (OUT_DIR / "audit" / f"{gid}.json").write_text(json.dumps({
            "GAME_ID": gid, "CHANGED": audit.changed, "RAW_STATE_HASH": audit.raw_state_hash, "REPAIRED_STATE_HASH": audit.repaired_state_hash,
            "QB_INPUTS_UNCHANGED": audit.qb_inputs_unchanged, "ZERO_SURVIVOR_FALLBACKS": audit.zero_survivor_fallbacks,
            "EXCLUDED_BY_REASON": {k: [dict(v, NAME=names.get(v["PLAYER_ID"])) for v in vs] for k, vs in audit.excluded().items()},
            "SUMMARIES": audit.summaries,
            "ROWS": [dict(r, NAME=names.get(r["PLAYER_ID"])) for r in audit.rows]}, indent=1, sort_keys=True), encoding="utf-8")
        if resume and (OUT_DIR / "raw" / f"{gid}.npz").exists() and (OUT_DIR / "games" / f"{gid}.json").exists():
            continue          # per-game sims are independent and seeded per game: resuming reproduces exactly what an uninterrupted run would
        jobs.append((gid, inp.state, res, n, OUT_DIR, snap))
    base.set_identity_resolutions(res)
    rec0 = next(r for r in ctx["games"] if r["GAME_ID"] == "2026_02_IND_KC")
    args0 = (ctx["inputs"]["2026_02_IND_KC"].state, 40, base.seed_for("2026_02_IND_KC"), cfg.MODEL_VERSION)
    x = v24.simulate_game(*args0, roster=snapshot_for(rec0, ctx))
    y = v24.simulate_game(*args0, roster=snapshot_for(rec0, ctx))
    det = all(np.array_equal(x.player_stats[s], y.player_stats[s]) for s in base.STAT_NAMES)
    print(f"determinism smoke (N=40 x2, IND_KC): {det}", flush=True)
    prior = json.loads((OUT_DIR / "sim_run_meta.json").read_text()) if resume and (OUT_DIR / "sim_run_meta.json").exists() else {}
    (OUT_DIR / "sim_run_meta.json").write_text(json.dumps({"RESUMED_AT": datetime.now(timezone.utc).isoformat() if resume else None, "FIRST_STARTED_AT": prior.get("FIRST_STARTED_AT") or prior.get("STARTED_AT"),
        "STARTED_AT": prior.get("STARTED_AT") or datetime.now(timezone.utc).isoformat(), "N_SIMS": n, "MODEL_VERSION": cfg.MODEL_VERSION, "V24_HASH": v24_hash(),
        "ENGINE_HASHES_BEFORE": before, "DETERMINISM_SMOKE": det, "PANEL_SHA256": ctx["raw_sha"], "SEED_POLICY": base.SEED_POLICY,
        "POLICY_FREEZE_SHA256": sha256_file(FREEZE_FILE)}, indent=1), encoding="utf-8")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(sim_worker_v24, j): j[0] for j in jobs}
        for f in as_completed(futs):
            gid, el = f.result()
            print(f"done {gid} {el:.0f}s wall={time.time() - t0:.0f}s", flush=True)
    after = base.engine_hashes()
    meta = json.loads((OUT_DIR / "sim_run_meta.json").read_text())
    meta.update({"FINISHED_AT": datetime.now(timezone.utc).isoformat(), "ENGINE_HASHES_AFTER": after,
                 "ENGINE_HASHES_UNCHANGED_DURING_SIM": before == after, "WALL_SEC": time.time() - t0})
    (OUT_DIR / "sim_run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("SIM COMPLETE; engine unchanged during run:", before == after, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "freeze":
        cmd_freeze()
    elif cmd == "sim":
        cmd_sim(int(sys.argv[2]) if len(sys.argv) > 2 else N_DEFAULT, resume=len(sys.argv) > 3 and sys.argv[3] == "resume")
    elif cmd == "analyze":
        from sports_nova_m1_v24_roster_eligibility_analysis import cmd_analyze
        cmd_analyze()
    else:
        raise SystemExit(__doc__)
