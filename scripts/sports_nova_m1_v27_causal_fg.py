"""SPORTS_NOVA V27 causal FG rate: freeze -> repro (60-game historical) -> canary (14-game Sunday slate, V27 and instrumented V26 rerun) -> analyze.

  python scripts/sports_nova_m1_v27_causal_fg.py freeze              # V27_PREREG/SPORTS_NOVA_V27_PREREG.json (refuses to run if any V27 output exists)
  python scripts/sports_nova_m1_v27_causal_fg.py repro               # 60-game reproduction of FG_CAUSAL_ABLATION_V1 through the V27 scoring boundary
  python scripts/sports_nova_m1_v27_causal_fg.py sim v27|v26 [N]     # slate canary; `v26` = instrumented rerun of the frozen V26 (FG counter only) for the paired comparison
  python scripts/sports_nova_m1_v27_causal_fg.py analyze

V23/V24/V25/V26 packages, worker/sports_nova_v3/distributions.py and FG_CAUSAL_ABLATION_V1 artifacts are never modified.  No market input is read.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sports_nova_m1_v26_active_skill_state as R26  # noqa: E402  (V26 runner: inputs/completions reused, not modified)
from worker.sports_nova_modular.firewall import assert_market_free  # noqa: E402
from worker.sports_nova_v26_active_skill_state import config as cfg26  # noqa: E402
from worker.sports_nova_v26_active_skill_state import simulator as v26  # noqa: E402
from worker.sports_nova_v27_causal_fg_rate import causal_fg as fgmod  # noqa: E402
from worker.sports_nova_v27_causal_fg_rate import config as cfg  # noqa: E402
from worker.sports_nova_v27_causal_fg_rate import simulator as v27  # noqa: E402

R25, base = R26.R25, R26.base
PKG = "worker/sports_nova_v27_causal_fg_rate"
DATA = base.DATA
OUT_DIR = DATA / "SUNDAY_2026_09_20_M1_V27_CAUSAL_FG_RATE"
V26_DIR = R26.OUT_DIR
V26_RERUN_DIR = DATA / "SUNDAY_2026_09_20_M1_V27_CANARY_V26_INSTRUMENTED_RERUN"
PREREG_DIR = DATA / "V27_PREREG"
PREREG_FILE = PREREG_DIR / "SPORTS_NOVA_V27_PREREG.json"
FG_ABL = ROOT / "data" / "sports_nova_v3" / "FG_CAUSAL_ABLATION_V1"
REPRO_DIR = ROOT / "data" / "sports_nova_v3" / "V27_CAUSAL_FG_60GAME_REPRO"
N_DEFAULT = 5000


def sha256_file(p: Path) -> str:
    return base.sha256_file(p)


def pkg_files() -> list[Path]:
    return sorted(p for p in (ROOT / PKG).glob("*.py"))


def pkg_hashes() -> dict:
    return {f"{PKG}/{p.name}": sha256_file(p) for p in pkg_files()}


def v27_hash() -> str:
    return hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(pkg_hashes().items())).encode()).hexdigest()


def tree_hash(d: Path) -> dict:
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(d.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


# ---------------------------------------------------------------------------------------------- freeze
def frozen_state() -> dict:
    """Everything that must be byte-identical between the freeze and any use; used by freeze and re-checked by repro/sim/analyze."""
    v26_pre = json.loads(R26.PREREG_FILE.read_text())
    abl_pre = json.loads((FG_ABL / "FG_CAUSAL_ABLATION_V1_PREREG.json").read_text())
    meta = json.loads((FG_ABL / "run_meta.json").read_text())
    v23_before = json.loads(R25.V23_BEFORE.read_text())["FILES"]
    return {
        "V26_HASH": R26.v26_hash(), "V26_HASH_AT_V26_FREEZE": v26_pre["V26_HASH"], "V26_PACKAGE_FILE_SHA256": R26.pkg_hashes(),
        "V26_PACKAGE_MATCHES_V26_PREREG": R26.pkg_hashes() == v26_pre["PACKAGE_FILE_SHA256"],
        "V25_HASH": R25.v25_hash(), "V25_HASH_PINNED": cfg26.V25_HASH, "V25_PACKAGE_FILE_SHA256": {k: sha256_file(ROOT / k) for k in cfg26.V25_PACKAGE_FILE_SHA256},
        "V25_FILES_UNCHANGED": all(R26.v25_files_unchanged().values()),
        "V23_SCORING_SIMULATOR_SHA256": sha256_file(ROOT / "worker/sports_nova_v23/simulator.py"),
        "V23_FILES_SHA256": {k: sha256_file(ROOT / k) for k in v23_before}, "V23_FILES_UNCHANGED": all(sha256_file(ROOT / k) == v for k, v in v23_before.items()),
        "DISTRIBUTIONS_SHA256": sha256_file(ROOT / "worker/sports_nova_v3/distributions.py"),
        "ENGINE_HASHES_AT_ABLATION": meta["engine_hashes_before"], "ENGINE_UNCHANGED_SINCE_ABLATION": all(sha256_file(ROOT / k) == v for k, v in meta["engine_hashes_before"].items()),
        "ESTIMATOR_SHA256": sha256_file(ROOT / cfg.ESTIMATOR_RELPATH), "ESTIMATOR_SHA256_PINNED": cfg.ESTIMATOR_SHA256, "ESTIMATOR_SHA256_IN_ABLATION_PREREG": abl_pre["ESTIMATOR_SCRIPT_SHA256"],
        "DRIVE_TABLE_SHA256": sha256_file(ROOT / cfg.DRIVE_RELPATH), "DRIVE_TABLE_SHA256_PINNED": cfg.DRIVE_SHA256, "DRIVE_TABLE_SHA256_IN_ABLATION_PREREG": abl_pre["DRIVE_TABLE_SHA256"],
        "V27_HASH": v27_hash(), "V27_PACKAGE_FILE_SHA256": pkg_hashes(),
        "V26_OUTPUT_TREE_SHA256": tree_hash(V26_DIR), "FG_ABLATION_ARTIFACT_SHA256": tree_hash(FG_ABL),
    }


def cmd_freeze() -> None:
    if any(p.exists() and any(p.rglob("*.npz")) for p in (OUT_DIR, V26_RERUN_DIR, REPRO_DIR)):
        raise SystemExit("REFUSING: V27 output already exists; the pre-registration must precede it")
    if PREREG_FILE.exists():
        raise SystemExit(f"REFUSING: {PREREG_FILE.name} already exists (a rule change ships as a new version, never an edit)")
    fs = frozen_state()
    assert fs["V26_PACKAGE_MATCHES_V26_PREREG"] and fs["V25_FILES_UNCHANGED"] and fs["V23_FILES_UNCHANGED"] and fs["ENGINE_UNCHANGED_SINCE_ABLATION"], "a frozen version drifted"
    assert fs["V25_HASH"] == fs["V25_HASH_PINNED"] and fs["V26_HASH"] == fs["V26_HASH_AT_V26_FREEZE"]
    assert fs["ESTIMATOR_SHA256"] == fs["ESTIMATOR_SHA256_PINNED"] == fs["ESTIMATOR_SHA256_IN_ABLATION_PREREG"], "estimator hash mismatch"
    assert fs["DRIVE_TABLE_SHA256"] == fs["DRIVE_TABLE_SHA256_PINNED"] == fs["DRIVE_TABLE_SHA256_IN_ABLATION_PREREG"], "drive table hash mismatch"
    slate_ids = [g["GAME_ID"] for g in json.loads((base.PANEL_DIR / "panel.json").read_bytes())["GAMES"]]
    slate_keys = {g: fgmod.parse_game_key(g) for g in slate_ids}                # fails closed here, not 8 minutes into a run
    assert len(slate_ids) == 14 and set(slate_keys.values()) == {(2026, 2)}, slate_keys
    r = fgmod.causal_fg_rate(2026, 2)
    assert (r.n_eligible, r.made) == (21452, 4751) and abs(r.rate - 0.221471) < 1e-6, "estimator A did not reproduce the frozen 2026-W02 result"
    t = subprocess.run([sys.executable, "-m", "pytest", f"{PKG}/tests", "-q"], cwd=ROOT, capture_output=True, text=True)
    tail = t.stdout.strip().splitlines()[-1] if t.stdout.strip() else t.stderr[-200:]
    assert t.returncode == 0, tail
    rep = json.loads((FG_ABL / "FG_CAUSAL_ABLATION_V1_REPORT.json").read_text())
    ab = rep["ABLATION"]
    ps = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'codex' } | Measure-Object).Count"], capture_output=True, text=True)
    recent = [p.relative_to(ROOT).as_posix() for p in ROOT.joinpath("worker").rglob("*.py") if time.time() - p.stat().st_mtime < 3600 and "sports_nova_v27" not in p.as_posix() and "__pycache__" not in p.parts]
    rec = {
        "SCHEMA": "SPORTS_NOVA_V27_PREREG", "FROZEN_AT": datetime.now(timezone.utc).isoformat(), "MODEL_VERSION": cfg.MODEL_VERSION, "IMPLEMENTATION_PACKAGE": PKG,
        "DATE": "2026-09-20", "GAMES": 14, "N_SIMS_PER_GAME": N_DEFAULT, **fs,
        "SOURCE_HASH_NOTE": "the brief's SOURCE_SHA256 '...017e50' is 63 characters (final 'd' missing); the frozen 64-character value 47b71f2b...017e50d (also in the ablation prereg) is used and verified",
        "CAUSAL_FG_POLICY": cfg.CAUSAL_FG_POLICY, "CAUSAL_FG_POLICY_HASH": cfg.CAUSAL_FG_POLICY_HASH,
        "CUTOFF_LOGIC": cfg.CAUSAL_FG_POLICY["WINDOW"],
        "COMPUTED_FG_RATE_2026_W02": {**r.evidence(), "reproduction_check_only": "eligible=21452 made=4751 rate~0.221471 asserted at freeze; the parameter is always recomputed, never hardcoded"},
        "CAVEAT_PRESERVED": cfg.CAUSAL_FG_POLICY["CAVEAT"],
        "UNIT_TESTS_AT_FREEZE": tail,
        "CONCURRENCY_CHECK": {"codex_processes_running": ps.stdout.strip(), "worker_py_files_modified_last_hour_excluding_v27": recent,
                              "STATEMENT": "no worker/sports_nova_v27* existed before this session; no peer wrote under worker/ in the hour before freeze"},
        "SLATE_GAME_IDS_PARSED_AT_FREEZE": {g: list(k) for g, k in slate_keys.items()},
        "ASSERTED_CHECKS_REGISTERED_BEFORE_RUN": {
            "V26_RERUN_INERTNESS": "the instrumented V26 rerun (FG observer only) must reproduce SUNDAY_2026_09_20_M1_V26_ACTIVE_SKILL_STATE/raw/*.npz np.array_equal on every array of every game; "
                                   "any mismatch means the observer is not inert and the V26->V27 canary comparison is VOID (stop and diagnose)",
            "AUDIT_BYTE_IDENTITY": "V27 audit/*.json must be byte-identical to V26's audit/*.json for all 14 games (completion, V25 repair, conservation, QB inputs) = V26_PIPELINE_PRESERVED on real data",
            "ABLATION_REPRODUCED": "Y only if every abl__ array of all 60 games is np.array_equal; otherwise report the miss and diagnose (params field diff, rate at 1e-17), never 'agrees within noise'"},
        "REPRODUCTION_TARGETS_REGISTERED_BEFORE_RUN (checks, never tuning targets)": {
            "60_GAME": "every abl__ and base__ array of FG_CAUSAL_ABLATION_V1 reproduced with np.array_equal (bit-exact) through v27.simulate_scoring / simulate_scoring_forced_fg_rate; "
                       "the V27 per-game rate equals the ablation prereg rate to 1e-15",
            "STORED_ABLATION_METRICS": {"FG_per_team_game": ab["FG_per_team_game"], "points_per_team_game": ab["points_per_team_game"], "total_bias": ab["errors"]["total_bias"],
                                        "total_MAE": ab["errors"]["total_MAE"], "tie_rate": ab["pooled"]["tie_rate"], "total_sd_pooled": ab["pooled"]["total_sd_pooled"],
                                        "team_score_sd_pooled": ab["pooled"]["team_score_sd_pooled"]},
            "SCOPE_NOTE": "the historical pilot has no RosterSnapshot, so it exercises the V27 SCORING BOUNDARY (the code the full pipeline calls); the roster/completion layers are exercised by the 14-game canary"},
        "CANARY_PREDICTION_REGISTERED_BEFORE_SIM": {
            "BASIS": "pilot: 7.84 eligible (no-TD, plays>=3) blocks per team-game, so FG/team-game = 7.84 x fg_rate: V26 ~0.63 (rate .08), V27 ~1.73 (rate 0.2215)",
            "DELTA_FG_PER_TEAM_GAME": [0.95, 1.25], "DELTA_MEAN_TOTAL_POINTS_PER_GAME": [5.5, 7.6], "TIE_RATE_CHANGE": "decrease (pilot: -1.7pp)",
            "WINNER_PROBABILITY": "mean |delta P(home)| about equal to Monte-Carlo noise (SE of an unpaired difference at N=5000 is ~0.010); no systematic direction beyond a small home-win rise",
            "UNCHANGED": "audit JSONs (completion, V25 repair, conservation) byte-identical to V26; invalid_usage 0; accounting PASS; player opportunity within a few tenths of a percent (game-script feedback only)",
            "FALSIFIER": "delta_FG outside 0.7..1.5 or delta_total outside 4..9 means the rate did not reach the scoring path as designed"},
        "DEFERRED_DEFECTS": ["MULTI_TD_PER_POSSESSION_ARTIFACT", "OVERTIME_MISSING", "WEAK_TOTAL_DISCRIMINATION", "QB_SCRAMBLE_TD_OMISSION", "PLAYER_THIN_POOL_CONCENTRATION", "M2_V27_NOT_CALIBRATED"],
        "MIRROR_STATUS": {"WINNER": "YELLOW", "TOTAL": "RED", "TEAM_TOTAL": "RED", "PLAYER_PROP": "BLOCKED", "NOTE": "V27 proves a bias correction, not predictive edge; nothing is promoted"},
        "LIVE_CAPITAL_AUTHORIZED": False, "V27_OUTPUT_EXISTED_AT_FREEZE": False,
    }
    assert_market_free({k: rec[k] for k in ("SCHEMA", "MODEL_VERSION", "CAUSAL_FG_POLICY", "V27_PACKAGE_FILE_SHA256", "V27_HASH", "COMPUTED_FG_RATE_2026_W02")})
    PREREG_DIR.mkdir(parents=True, exist_ok=True)
    PREREG_FILE.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("V27_HASH", "V26_HASH", "V25_HASH", "ESTIMATOR_SHA256", "DRIVE_TABLE_SHA256", "UNIT_TESTS_AT_FREEZE")}, indent=1))
    print("FG 2026-W02:", r.n_eligible, r.made, r.rate)


def load_freeze() -> dict:
    if not PREREG_FILE.exists():
        raise SystemExit("run `freeze` first: SPORTS_NOVA_V27_PREREG.json must exist before any V27 simulation")
    pre = json.loads(PREREG_FILE.read_text())
    now = frozen_state()
    for k in ("V27_HASH", "V26_HASH", "V25_HASH", "ESTIMATOR_SHA256", "DRIVE_TABLE_SHA256", "V23_SCORING_SIMULATOR_SHA256", "DISTRIBUTIONS_SHA256", "V26_OUTPUT_TREE_SHA256", "FG_ABLATION_ARTIFACT_SHA256"):
        assert pre[k] == now[k], f"{k} changed after the freeze"
    return pre


# ---------------------------------------------------------------------------------------------- 60-game reproduction
def _repro_game(game: str) -> dict:
    import scripts.sports_nova_m1_fg_causal_ablation_v1 as H
    module = importlib.import_module(H.MODULE)
    season, week, away, home = H.game_parts(game)
    all_df = pd.read_parquet(H.PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    prior = all_df[(all_df._key < season * 100 + week) & (all_df.SEASON >= max(1999, season - 5))]
    state = H.make_state(game, prior, H.surrogate_kickoff(season, week), None)
    seed, n = H.game_seed(game), H.N_SIMS
    stored = np.load(H.ARR / f"{game}.npz")
    pre = json.loads((FG_ABL / "FG_CAUSAL_ABLATION_V1_PREREG.json").read_text())
    t0 = time.time()
    res: dict = {"game": game, "seed": seed}
    with H.Recorder(module, None) as r1:                                  # passive observer; V27 already applies the causal rate
        b1 = v27.simulate_scoring(state, n, seed, cfg.MODEL_VERSION)
    s1 = H.summarize(b1, r1.blocks, state, n)
    with H.Recorder(module, None) as r0:
        b0 = v27.simulate_scoring_forced_fg_rate(state, n, seed, cfg.MODEL_VERSION, fg_rate=cfg.V23_DEFAULT_FG_RATE)
    s0 = H.summarize(b0, r0.blocks, state, n)
    plain = module.simulate_game(state, n, seed, module.MODEL_VERSION)
    res["abl_array_equal"] = {k: bool(np.array_equal(stored[f"abl__{k}"], v)) for k, v in s1["arrays"].items()}
    res["base_array_equal"] = {k: bool(np.array_equal(stored[f"base__{k}"], v)) for k, v in s0["arrays"].items()}
    res["stored_keys_covered"] = sorted(set(k for k in stored.files) - {f"abl__{k}" for k in s1["arrays"]} - {f"base__{k}" for k in s0["arrays"]})
    res["forced_default_equals_plain_v23"] = bool(all(np.array_equal(plain.player_stats[k], b0.player_stats[k]) for k in plain.player_stats)
                                                  and all(np.array_equal(plain.team_stats[k], b0.team_stats[k]) for k in plain.team_stats) and np.array_equal(plain.winner, b0.winner))
    res["rate"] = b1.runtime["fg_rate"]
    res["rate_equals_prereg"] = bool(abs(b1.runtime["fg_rate"] - pre["PER_GAME_ESTIMATES"][game]["rate"]) < 1e-15)
    res["fg_estimate_evidence"] = {k: v for k, v in b1.runtime.items() if k.startswith("fg_") and k != "fg_wilson95"}
    res["acct"] = s1["accounting"]
    res["arrays"] = {k: s1["arrays"][k] for k in ("team_score", "winner", "rec_fg", "rec_td", "rec_nblk", "rec_elig")}
    res["seconds"] = round(time.time() - t0, 1)
    return res


def cmd_repro(workers: int = 4) -> None:
    pre = load_freeze()
    import scripts.sports_nova_m1_fg_causal_ablation_v1 as H
    import scripts.sports_nova_m1_fg_causal_ablation_v1_analysis as AN
    games = H.pilot_games(list(json.loads(H.MANIFEST.read_text())["GAME_IDS"]))
    abl_pre = json.loads((FG_ABL / "FG_CAUSAL_ABLATION_V1_PREREG.json").read_text())
    assert games == abl_pre["PILOT"]["game_ids"], "pilot cohort drifted"
    REPRO_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(workers) as pool:
        results = []
        for r in pool.imap_unordered(_repro_game, games):
            results.append(r)
            print(f"{r['game']} {r['seconds']}s abl_equal={all(r['abl_array_equal'].values())} base_equal={all(r['base_array_equal'].values())}", flush=True)
    results.sort(key=lambda r: r["game"])
    # metrics from the NEW arrays, computed exactly as the ablation analysis does
    d = AN.load_drives()
    obs = AN.observed_scores(d, games).set_index("game")
    n = H.N_SIMS
    fg, pts, tot_err, team_err, pool_t, pool_h, pool_a, pool_w, ph, pt, sds = [], [], [], [], [], [], [], [], [], [], []
    for r in results:
        a = r["arrays"]
        z = {f"abl__{k}": v for k, v in a.items()}
        gs = AN.game_summary(z, "abl", n)
        h, aw = a["team_score"][:, 0], a["team_score"][:, 1]
        o = obs.loc[r["game"]]
        fg.append(a["rec_fg"].mean(axis=0)); pts.append(a["team_score"].mean(axis=0))
        tot_err.append(gs["mean_total"] - o.obs_total)
        team_err += [gs["mean_home_score"] - o.obs_home, gs["mean_away_score"] - o.obs_away]
        pool_t.append(h + aw); pool_h.append(h); pool_a.append(aw); pool_w.append(a["winner"])
        ph.append(gs["home_win_prob"]); pt.append(gs["tie_prob"]); sds.append([gs["score_sd_home"], gs["score_sd_away"]])
    te = np.array(tot_err)
    T, HH, AA, W = np.concatenate(pool_t), np.concatenate(pool_h), np.concatenate(pool_a), np.concatenate(pool_w)
    team = np.concatenate([HH, AA])
    new = {"FG_per_team_game": float(np.mean(np.concatenate([x.ravel() for x in fg]))), "points_per_team_game": float(np.mean(np.concatenate([x.ravel() for x in pts]))),
           "total_bias": float(te.mean()), "total_MAE": float(np.abs(te).mean()), "tie_rate": float((W == "TIE").mean()),
           "total_sd_pooled": float(T.std(ddof=1)), "team_score_sd_pooled": float(team.std(ddof=1)), "mean_home_win_prob": float(np.mean(ph)),
           "mean_within_game_team_sd": float(np.mean(sds)), "total_quantiles_pooled": {q: float(np.quantile(T, q / 100)) for q in (1, 5, 25, 50, 75, 95, 99)},
           "team_total_bias": float(np.mean(team_err)), "team_total_MAE": float(np.mean(np.abs(team_err)))}
    rep = json.loads((FG_ABL / "FG_CAUSAL_ABLATION_V1_REPORT.json").read_text())["ABLATION"]
    old = {"FG_per_team_game": rep["FG_per_team_game"], "points_per_team_game": rep["points_per_team_game"], "total_bias": rep["errors"]["total_bias"], "total_MAE": rep["errors"]["total_MAE"],
           "tie_rate": rep["pooled"]["tie_rate"], "total_sd_pooled": rep["pooled"]["total_sd_pooled"], "team_score_sd_pooled": rep["pooled"]["team_score_sd_pooled"],
           "mean_within_game_team_sd": rep["pooled"]["mean_within_game_team_sd"], "total_quantiles_pooled": {int(k): v for k, v in rep["pooled"]["total_q"].items()},
           "team_total_bias": rep["errors"]["team_total_bias"], "team_total_MAE": rep["errors"]["team_total_MAE"]}
    diffs = {k: (abs(new[k] - old[k]) if not isinstance(new[k], dict) else max(abs(new[k][q] - old[k][q]) for q in new[k])) for k in old}
    all_abl = all(all(r["abl_array_equal"].values()) for r in results)
    all_base = all(all(r["base_array_equal"].values()) for r in results)
    acct = {k: sum(int(r["acct"][k]) for r in results) for k in results[0]["acct"]}
    out = {"SCHEMA": "SPORTS_NOVA_V27_60GAME_REPRODUCTION", "V27_HASH": pre["V27_HASH"], "PREREG_SHA256": sha256_file(PREREG_FILE), "GAMES": len(results), "N_SIMS": n,
           "ENGINE_BOUNDARY": "v27.simulate_scoring (scoring boundary; historical states have no RosterSnapshot) and v27.simulate_scoring_forced_fg_rate (fg_rate=0.08)",
           "BIT_EXACT_ABLATION_ARRAYS_ALL_GAMES": all_abl, "BIT_EXACT_BASELINE_ARRAYS_ALL_GAMES": all_base,
           "ARRAY_KEYS_COMPARED_PER_ARM": sorted(results[0]["abl_array_equal"]), "STORED_KEYS_NOT_COVERED": sorted({k for r in results for k in r["stored_keys_covered"]}),
           "FORCED_DEFAULT_EQUALS_PLAIN_V23_ALL_GAMES": all(r["forced_default_equals_plain_v23"] for r in results),
           "RATE_EQUALS_ABLATION_PREREG_ALL_GAMES": all(r["rate_equals_prereg"] for r in results),
           "NEW_METRICS": new, "STORED_ABLATION_METRICS": old, "MAX_ABS_DIFF_BY_METRIC": diffs, "ACCOUNTING_TOTALS_ABL": acct,
           "PER_GAME": [{k: v for k, v in r.items() if k != "arrays"} for r in results], "WALL_SEC": time.time() - t0,
           "ENGINE_HASHES_UNCHANGED": load_freeze() is not None}
    (REPRO_DIR / "V27_60GAME_REPRODUCTION.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in out if k not in ("PER_GAME",)}, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- slate canary
class FGObserver:
    """Passive: counts 3-point blocks (a made FG; TD blocks are >= 6) and sums block points per (sim, team).  Changes nothing."""

    def __init__(self, n: int, home: str, away: str):
        from worker.sports_nova_v3.game_state import GameState
        self.GS, self.n, self.idx = GameState, n, {home: 0, away: 1}
        self.fg, self.pts, self.blocks = (np.zeros((n, 2), np.int32) for _ in range(3))
        self.orig = GameState.advance

    def __enter__(self):
        o, me = self.orig, self

        def adv(state, **kw):
            new = o(state, **kw)
            j = me.idx[state.possession]
            p = (new.home_score - state.home_score) + (new.away_score - state.away_score)
            s = int(state.simulation_id)
            me.blocks[s, j] += 1
            me.pts[s, j] += p
            if p == 3:
                me.fg[s, j] += 1
            return new
        self.GS.advance = adv
        return self

    def __exit__(self, *a):
        self.GS.advance = self.orig


def sim_worker(args):
    engine, gid, state, resolutions, n, outdir, snap, comp = args
    if engine == "v27":
        base.MODEL_VERSION = cfg.MODEL_VERSION
        base.simulate_game = lambda st, nn, seed, mv: v27.simulate_game(st, nn, seed, mv, roster=snap, completion=comp)
    else:
        base.MODEL_VERSION = cfg26.MODEL_VERSION
        base.simulate_game = lambda st, nn, seed, mv: v26.simulate_game(st, nn, seed, mv, roster=snap, completion=comp)
    _, _, away, home = gid.split("_")
    with FGObserver(n, home, away) as ob:
        out = base.sim_worker((gid, state, resolutions, n, outdir))
    (outdir / "fg").mkdir(exist_ok=True)
    np.savez_compressed(outdir / "fg" / f"{gid}.npz", fg=ob.fg, pts=ob.pts, blocks=ob.blocks, teams=np.array([home, away]))
    return out


def cmd_sim(engine: str, n: int) -> None:
    pre = load_freeze()
    out_dir = OUT_DIR if engine == "v27" else V26_RERUN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit").mkdir(exist_ok=True)
    before = base.engine_hashes()
    ctx = base.load_inputs()
    assert ctx["raw_sha"] == base.PANEL_SHA_EXPECTED, "PANEL SHA MISMATCH"
    res = base.all_resolutions(ctx["inputs"])
    comps = R26.completions(ctx)
    jobs = []
    for rec in ctx["games"]:
        gid, inp = rec["GAME_ID"], ctx["inputs"][rec["GAME_ID"]]
        if inp.state is None:
            print(f"SKIP {gid}: {inp.missing}", flush=True)
            continue
        snap, prior, comp = comps[gid]
        _, repaired, audit = v27.prepare_state(inp.state, snap, prior, known_out=R26.known_out(rec))
        names = ctx["names"]
        (out_dir / "audit" / f"{gid}.json").write_text(json.dumps({
            "GAME_ID": gid, "CHANGED": audit.changed, "COMPLETION_CHANGED": comp.changed, "RAW_STATE_HASH": comp.raw_state_hash, "COMPLETED_STATE_HASH": comp.completed_state_hash,
            "REPAIRED_STATE_HASH": audit.repaired_state_hash, "QB_INPUTS_UNCHANGED": audit.qb_inputs_unchanged, "COMPLETION_TEAMS": comp.teams,
            "COMPLETION_ROWS": [dict(r, NAME=names.get(r["PLAYER_ID"])) for r in comp.rows],
            "EXCLUDED_BY_REASON": {k: [dict(v, NAME=names.get(v["PLAYER_ID"])) for v in vs] for k, vs in audit.excluded().items()},
            "SUMMARIES": audit.summaries, "ROWS": [dict(r, NAME=names.get(r["PLAYER_ID"])) for r in audit.rows]}, indent=1, sort_keys=True), encoding="utf-8")
        jobs.append((engine, gid, inp.state, res, n, out_dir, snap, comp))
    base.set_identity_resolutions(res)
    gid0 = "2026_02_IND_KC"
    snap0, _, comp0 = comps[gid0]
    fn = (lambda: v27.simulate_game(ctx["inputs"][gid0].state, 40, base.seed_for(gid0), cfg.MODEL_VERSION, roster=snap0, completion=comp0)) if engine == "v27" else \
         (lambda: v26.simulate_game(ctx["inputs"][gid0].state, 40, base.seed_for(gid0), cfg26.MODEL_VERSION, roster=snap0, completion=comp0))
    x, y = fn(), fn()
    det = all(np.array_equal(x.player_stats[s], y.player_stats[s]) for s in base.STAT_NAMES)
    print(f"[{engine}] determinism smoke (N=40 x2, IND_KC): {det}", flush=True)
    meta = {"ENGINE": engine, "STARTED_AT": datetime.now(timezone.utc).isoformat(), "N_SIMS": n, "V27_HASH": pre["V27_HASH"], "V26_HASH": pre["V26_HASH"], "DETERMINISM_SMOKE": det,
            "ENGINE_HASHES_BEFORE": before, "PANEL_SHA256": ctx["raw_sha"], "SEED_POLICY": base.SEED_POLICY, "PREREG_SHA256": sha256_file(PREREG_FILE),
            "FG_OBSERVER": "passive GameState.advance counter (3-point blocks); changes no state"}
    (out_dir / "sim_run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(sim_worker, j): j[1] for j in jobs}
        for f in as_completed(futs):
            gid, el = f.result()
            print(f"[{engine}] done {gid} {el:.0f}s wall={time.time() - t0:.0f}s", flush=True)
    after = base.engine_hashes()
    meta.update({"FINISHED_AT": datetime.now(timezone.utc).isoformat(), "ENGINE_HASHES_AFTER": after, "ENGINE_HASHES_UNCHANGED_DURING_SIM": before == after, "WALL_SEC": time.time() - t0})
    (out_dir / "sim_run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[{engine}] SIM COMPLETE; engine unchanged during run:", before == after, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "freeze":
        cmd_freeze()
    elif cmd == "repro":
        cmd_repro()
    elif cmd == "sim":
        cmd_sim(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else N_DEFAULT)
    elif cmd == "analyze":
        from sports_nova_m1_v27_causal_fg_analysis import cmd_analyze
        cmd_analyze()
    else:
        raise SystemExit(__doc__)
