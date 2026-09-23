"""SPORTS_NOVA V25 role-aware redistribution: freeze -> sim -> analyze (14-game Sunday slate, panel V2, N=5000, same seeds as V23/V24).

  python scripts/sports_nova_m1_v25_role_aware.py freeze         # SPORTS_NOVA_V25_PREREG.json; refuses to run if any V25 sim output exists
  python scripts/sports_nova_m1_v25_role_aware.py sim [N] [resume]
  python scripts/sports_nova_m1_v25_role_aware.py analyze
  python scripts/sports_nova_m1_v25_role_aware.py mirror [GAME_ID ...]   # research-only NOVA-vs-Kalshi comparison (only if MIRROR gate allows)

V23 and both V24 packages are never modified.  Inputs come only from the immutable panel V2 directory (via the V23 baseline runner's load_inputs()).
"""
from __future__ import annotations

import hashlib
import json
import shutil
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

import sports_nova_m1_v24_roster_eligibility as R24  # noqa: E402  (V24 runner: reused for inputs/snapshot, not modified)
from worker.sports_nova_modular.firewall import assert_market_free  # noqa: E402
from worker.sports_nova_v25_role_aware import config as cfg  # noqa: E402
from worker.sports_nova_v25_role_aware import simulator as v25  # noqa: E402
from worker.sports_nova_v25_role_aware.eligibility import RosterSnapshot  # noqa: E402

base = R24.base
PKG = "worker/sports_nova_v25_role_aware"
DATA = base.DATA
V23_DIR = DATA / "SUNDAY_2026_09_20_M1_BASELINE_V1"
V24_DIR = R24.OUT_DIR
OUT_DIR = DATA / "SUNDAY_2026_09_20_M1_V25_ROLE_AWARE"
PREREG_DIR = DATA / "V25_PREREG"
PREREG_FILE = PREREG_DIR / "SPORTS_NOVA_V25_PREREG.json"
V23_BEFORE = R24.V23_BEFORE
N_DEFAULT = 5000

# ----------------------------------------------------------------------------------------------------------------------------- frozen rules
EIGHT_PATHS = [("HOU", "carry", "Woody Marks"), ("GB", "carry", "Jordan Love"), ("NO", "carry", "Alvin Kamara"), ("JAX", "carry", "Trevor Lawrence"),
               ("MIA", "target", "De'Von Achane"), ("ARI", "carry", "Bam Knight"), ("SEA", "carry", "George Holani"), ("KC", "carry", "Patrick Mahomes")]
PATH_CLASSIFICATION_RULES = {
    "SOURCE_OF_TAXONOMY": "the brief's supported=19/unsupported=8 taxonomy exists only in Codex's V24 (8d994158) artifacts and its definition is not in them; this is V25's OWN deterministic taxonomy, "
                          "frozen before any V25 simulation output exists",
    "INPUTS": "V25 audit row for the path (state-level shares, deterministic) + V25 simulated mean carries/targets and the historical envelopes below",
    "ENVELOPES": {"TEAM_SEASON_MAX_SINGLE_PLAYER_SHARE_P99": {"carry": 0.838, "target": 0.328}, "QB_MEAN_RUSH_ATT_PER_GAME_HISTORICAL_MAX": 11.733333333333333,
                  "SOURCE": "2019-2025 REG team-seasons / QB-seasons (>=6 games), computed before 2026 data; carried in V24 config CONTEXT_ONLY and the V24 report"},
    "THIN_HISTORY": "PLAYER_HISTORY_SAMPLE < 8 games",
    "MECHANICALLY_RESOLVED": "path role is QB (a held role) AND V25 state share <= no-removal share + 1e-9 (no removed non-QB mass received) AND V25 simulated mean rush attempts <= V23 simulated mean + 2 SE",
    "SUPPORTED": "path role is a recipient role AND eligible AND V25 state share <= the historical envelope for its kind AND not THIN_HISTORY",
    "AMBIGUOUS": "recipient role AND eligible, but V25 state share > envelope OR THIN_HISTORY; mass is role-valid but rests on a thin/short surviving pool -- needs new pregame role evidence, NOT resolved by V25",
    "UNSUPPORTED": "role-invalid mass (held role above its no-removal share) OR ineligible player with usage",
    "ORDER": "MECHANICALLY_RESOLVED / SUPPORTED are tested first, then AMBIGUOUS, then UNSUPPORTED",
}
QB_REALISM_RULES = {"SEVERE_QB_FLAG": "QB V25 mean rush attempts per game > historical max 11.7333", "WORST_QB": "max mean rush attempts among QBs, and max QB carry share of team pool",
                    "V23_DOUBLE_COUNT_NOT_FIXED": "QB rush attempts include V23's inherited scramble double count; V25 removes amplification, not the baseline"}
NEW_CONCENTRATION_RULE = ("NEW_SEVERE_CONCENTRATION: a RECIPIENT-role player-path whose V25 state share exceeds the historical envelope for its kind while its V24-equivalent state share did NOT. "
                          "Held-role increases must be 0.")
MIRROR_GATE_RULES = {
    "SPORTS_NOVA_RESEARCH_MIRROR_READY": "YES iff invalid_usage==0 AND meaningful_invalid_usage==0 AND mass_conservation==PASS AND accounting==PASS AND qb_mismatch==0 AND market_contamination==0 "
                                          "AND all eight paths in {SUPPORTED, MECHANICALLY_RESOLVED} AND NEW_SEVERE_CONCENTRATION count==0 AND QB redistribution inflation removed (no QB above V23 no-removal share). "
                                          "Otherwise NO for the player-prop scope.",
    "GAME_LEVEL_MIRROR_READY": "separate, declared here before results: the same six technical gates AND QB inflation removed AND no NEW_SEVERE_CONCENTRATION on a QB/held role AND "
                               "the V25-vs-V23 game-level distribution shift is inside the frozen V24 regression rules (slate mean of median totals drop <= 1.5, any game median total shift <= 4.0, "
                               "any team mean attempts shift <= 5%). Authorizes ONLY winner/total/team-total research comparison. Player props stay gated by the rule above.",
    "NOT_A_PROMOTION": "neither flag promotes V25; V23 remains the formal champion; M2 is uncalibrated for V25",
}
REGRESSION_RULES = R24.REGRESSION_RULES


def sha256_file(p: Path) -> str:
    return base.sha256_file(p)


def pkg_files() -> list[Path]:
    return sorted(p for p in (ROOT / PKG).glob("*.py"))


def pkg_hashes() -> dict:
    return {f"{PKG}/{p.name}": sha256_file(p) for p in pkg_files()}


def v25_hash() -> str:
    return hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(pkg_hashes().items())).encode()).hexdigest()


def position_index(ctx) -> dict:
    if "positions" not in ctx:
        import pandas as pd
        ros = pd.read_parquet(base.PANEL_DIR / "sources" / "roster_weekly_2026.parquet")
        ctx["positions"] = {str(r.gsis_id): (str(r.position), str(r.depth_chart_position)) for r in ros.itertuples() if r.gsis_id}
    return ctx["positions"]


def snapshot_for(rec: dict, ctx: dict) -> RosterSnapshot:
    s = R24.snapshot_for(rec, ctx)
    src = dict(s.source_sha256)
    src["roster_weekly_2026.parquet(position,depth_chart_position)"] = src.get("roster_weekly_2026.parquet", "")
    return RosterSnapshot(roster=s.roster, game_status=s.game_status, roster_week=s.roster_week, source_sha256=src, positions=position_index(ctx))


def fingerprint_conflicts() -> dict:
    """Fingerprint (not modify) the two V24 implementations and the frozen V23 engine so V25 provenance is unambiguous."""
    mine = sorted((ROOT / "worker/sports_nova_v24_roster_eligibility").glob("*.py"))
    codex = sorted((ROOT / "worker/sports_nova_v24").glob("*.py"))
    codex_tree = hashlib.sha256("".join(f"{p.relative_to(ROOT).as_posix()}:{sha256_file(p)}\n" for p in codex if p.name in
                                        ("__init__.py", "config.py", "allocation.py", "eligibility.py", "simulator.py")).encode()).hexdigest()
    arch = PREREG_DIR / "ARCHIVE_CODEX_V24_CODE_AT_V25_FREEZE"
    arch.mkdir(parents=True, exist_ok=True)
    for p in codex:
        shutil.copy2(p, arch / p.name)
    return {"V24_ROSTER_ELIGIBILITY_PACKAGE": {"PATH": "worker/sports_nova_v24_roster_eligibility", "V24_HASH_RECORDED": "491d536cce16b8d3758ae683f5a9944b094d446abd4e464b7953cff6981434fb",
                                               "V24_HASH_RECOMPUTED": R24.v24_hash(), "FILES": {p.name: sha256_file(p) for p in mine}},
            "V24_CODEX_PACKAGE": {"PATH": "worker/sports_nova_v24", "V24_HASH_IN_BRIEF": "8d994158cd5fd4798f4d7476174623c842a965b5eb31dd223a4dfa9e2b16ada7",
                                  "RECOMPUTED_OVER_5_MODULE_FILES_SHA256_OF_NAME_HASH_LINES": codex_tree, "FILES": {p.name: sha256_file(p) for p in codex},
                                  "ARCHIVE_COPY": str(arch.relative_to(ROOT)).replace("\\", "/"), "ARTIFACT_DIR": "data/sports_nova_v3/sunday_2026_09_20/SUNDAY_2026_09_20_M1_V24_CHALLENGER"},
            "BRIEF_V24_IS": "the Codex implementation (hash 8d994158 is in its SPORTS_NOVA_V24_HASHES.json; its 27 concentration warnings equal KNOWN_V24.concentration_warnings=27)",
            "V25_DERIVES_FROM": "worker/sports_nova_v24_roster_eligibility (491d536c): same eligibility rules, same Doubtful weight, same frozen-share pro-rata mechanics for recipients; "
                                "equivalence to Codex's mask: both zero ineligible shares and renormalize pro-rata over survivors with residual dropped",
            "V25_IMPORTS_NOTHING_FROM_EITHER_V24": True}


# ---------------------------------------------------------------------------------------------- freeze
def cmd_freeze() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.rglob("*.npz")):
        raise SystemExit("REFUSING: V25 simulation output already exists; the pre-registration must precede it")
    if PREREG_FILE.exists():
        raise SystemExit(f"REFUSING: {PREREG_FILE.name} already exists (a rule change ships as a new version, never an edit)")
    PREREG_DIR.mkdir(parents=True, exist_ok=True)
    before = json.loads(V23_BEFORE.read_text())["FILES"]
    unchanged = {k: sha256_file(ROOT / k) == v for k, v in before.items()}
    assert all(unchanged.values()), [k for k, v in unchanged.items() if not v]
    t = subprocess.run([sys.executable, "-m", "pytest", f"{PKG}/tests", "-q"], cwd=ROOT, capture_output=True, text=True)
    tail = t.stdout.strip().splitlines()[-1] if t.stdout.strip() else t.stderr[-200:]
    rec = {"SCHEMA": "SPORTS_NOVA_V25_PREREG", "FROZEN_AT": datetime.now(timezone.utc).isoformat(), "MODEL_VERSION": cfg.MODEL_VERSION,
           "IMPLEMENTATION_PACKAGE": PKG, "PACKAGE_FILE_SHA256": pkg_hashes(), "V25_HASH": v25_hash(),
           "V23_HASHES_BEFORE_FILE": str(V23_BEFORE.relative_to(ROOT)).replace("\\", "/"), "V23_HASHES_BEFORE_SHA256": sha256_file(V23_BEFORE), "V23_FILES_UNCHANGED_AT_FREEZE": True,
           "CONFLICT_FINGERPRINTS": fingerprint_conflicts(),
           "ELIGIBILITY_POLICY": cfg.ELIGIBILITY_POLICY, "ELIGIBILITY_POLICY_HASH": cfg.ELIGIBILITY_POLICY_HASH,
           "DOUBTFUL_POLICY": cfg.DOUBTFUL_POLICY, "DOUBTFUL_POLICY_HASH": cfg.DOUBTFUL_POLICY_HASH,
           "ROLE_POLICY": cfg.ROLE_POLICY, "ROLE_POLICY_HASH": cfg.ROLE_POLICY_HASH,
           "REDISTRIBUTION_POLICY": cfg.REDISTRIBUTION_POLICY, "REDISTRIBUTION_POLICY_HASH": cfg.REDISTRIBUTION_POLICY_HASH,
           "SCOPE_A_DESIGNED_CARRY_RECIPIENTS": "QB (and non-skill) excluded as recipients of removed carry mass; QB scramble path untouched; the architecture cannot separate QB_DESIGNED_RUSH from "
                                                "QB_SCRAMBLE inside the frozen carry share, so redistribution alone is isolated (QB held at its no-removal share)",
           "SCOPE_B_TARGET_RECIPIENTS": "same family rule on the target pool; 16 QBs hold <=0.23% target share on this slate, so B is nearly a no-op in effect but the rule is applied uniformly",
           "PATH_CLASSIFICATION_RULES": PATH_CLASSIFICATION_RULES, "EIGHT_PATHS": [list(x) for x in EIGHT_PATHS],
           "QB_REALISM_RULES": QB_REALISM_RULES, "NEW_CONCENTRATION_RULE": NEW_CONCENTRATION_RULE, "MIRROR_GATE_RULES": MIRROR_GATE_RULES,
           "REGRESSION_RULES": REGRESSION_RULES, "PANEL_SHA256": base.PANEL_SHA_EXPECTED, "SEED_POLICY": base.SEED_POLICY, "N_SIMS_PER_GAME": N_DEFAULT, "DATE": "2026-09-20", "GAMES": 14,
           "ADVISOR": "one Opus-class advisor pass before implementation; findings applied: build on 491d536c not 8d994158; QB-only fix cannot resolve the 5 non-QB paths; "
                      "MIRROR_READY not to be redefined to pass; measure game-level impact; V23 QB double count remains",
           "OUT_HANDLING_CHANGE_VS_V24": "OUT non-QB players are removed mass under V25's role rule (V24 left them to V23's gate whose renormalization also fed the QB)",
           "UNIT_TESTS_AT_FREEZE": tail, "V25_SIM_OUTPUT_EXISTED_AT_FREEZE": False}
    assert_market_free({k: v for k, v in rec.items() if k not in ("MIRROR_GATE_RULES", "CONFLICT_FINGERPRINTS")})
    PREREG_FILE.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("V25_HASH", "ROLE_POLICY_HASH", "REDISTRIBUTION_POLICY_HASH", "UNIT_TESTS_AT_FREEZE")}, indent=1))


# ---------------------------------------------------------------------------------------------- sim
def sim_worker_v25(args):
    gid, state, resolutions, n, outdir, snap = args
    base.MODEL_VERSION = cfg.MODEL_VERSION
    base.simulate_game = lambda st, nn, seed, mv: v25.simulate_game(st, nn, seed, mv, roster=snap)
    return base.sim_worker((gid, state, resolutions, n, outdir))


def cmd_sim(n: int, resume: bool = False) -> None:
    if not PREREG_FILE.exists():
        raise SystemExit("run `freeze` first: SPORTS_NOVA_V25_PREREG.json must exist before any V25 simulation")
    freeze = json.loads(PREREG_FILE.read_text())
    assert freeze["V25_HASH"] == v25_hash(), "V25 package changed after the freeze"
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
        repaired, audit = v25.prepare_state(inp.state, snap)
        names = ctx["names"]
        (OUT_DIR / "audit" / f"{gid}.json").write_text(json.dumps({
            "GAME_ID": gid, "CHANGED": audit.changed, "RAW_STATE_HASH": audit.raw_state_hash, "REPAIRED_STATE_HASH": audit.repaired_state_hash,
            "QB_INPUTS_UNCHANGED": audit.qb_inputs_unchanged,
            "EXCLUDED_BY_REASON": {k: [dict(v, NAME=names.get(v["PLAYER_ID"])) for v in vs] for k, vs in audit.excluded().items()},
            "SUMMARIES": audit.summaries, "ROWS": [dict(r, NAME=names.get(r["PLAYER_ID"])) for r in audit.rows]}, indent=1, sort_keys=True), encoding="utf-8")
        if resume and (OUT_DIR / "raw" / f"{gid}.npz").exists() and (OUT_DIR / "games" / f"{gid}.json").exists():
            continue
        jobs.append((gid, inp.state, res, n, OUT_DIR, snap))
    base.set_identity_resolutions(res)
    rec0 = next(r for r in ctx["games"] if r["GAME_ID"] == "2026_02_IND_KC")
    args0 = (ctx["inputs"]["2026_02_IND_KC"].state, 40, base.seed_for("2026_02_IND_KC"), cfg.MODEL_VERSION)
    x = v25.simulate_game(*args0, roster=snapshot_for(rec0, ctx))
    y = v25.simulate_game(*args0, roster=snapshot_for(rec0, ctx))
    det = all(np.array_equal(x.player_stats[s], y.player_stats[s]) for s in base.STAT_NAMES)
    print(f"determinism smoke (N=40 x2, IND_KC): {det}", flush=True)
    prior = json.loads((OUT_DIR / "sim_run_meta.json").read_text()) if resume and (OUT_DIR / "sim_run_meta.json").exists() else {}
    (OUT_DIR / "sim_run_meta.json").write_text(json.dumps({
        "RESUMED_AT": datetime.now(timezone.utc).isoformat() if resume else None, "FIRST_STARTED_AT": prior.get("FIRST_STARTED_AT") or prior.get("STARTED_AT"),
        "STARTED_AT": prior.get("STARTED_AT") or datetime.now(timezone.utc).isoformat(), "N_SIMS": n, "MODEL_VERSION": cfg.MODEL_VERSION, "V25_HASH": v25_hash(),
        "ENGINE_HASHES_BEFORE": before, "DETERMINISM_SMOKE": det, "PANEL_SHA256": ctx["raw_sha"], "SEED_POLICY": base.SEED_POLICY,
        "PREREG_SHA256": sha256_file(PREREG_FILE)}, indent=1), encoding="utf-8")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(sim_worker_v25, j): j[0] for j in jobs}
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
    if cmd == "freeze":
        cmd_freeze()
    elif cmd == "sim":
        cmd_sim(int(sys.argv[2]) if len(sys.argv) > 2 else N_DEFAULT, resume=len(sys.argv) > 3 and sys.argv[3] == "resume")
    elif cmd == "analyze":
        from sports_nova_m1_v25_role_aware_analysis import cmd_analyze
        cmd_analyze()
    elif cmd == "mirror":
        from sports_nova_v25_kalshi_mirror import cmd_mirror
        cmd_mirror(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
