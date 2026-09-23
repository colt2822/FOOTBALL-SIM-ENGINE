"""SPORTS_NOVA_SUNDAY_BASELINE_V1 (N3 / M1_VALIDATION): 14-game Sunday slate baseline + roster-contamination audit.

  python scripts/sports_nova_sunday_m1_baseline_v1.py sim [N]      # parallel sims, one raw file per game written as each finishes
  python scripts/sports_nova_sunday_m1_baseline_v1.py analyze      # contamination / QB / trust / anomaly audit + the 3 artifacts

M1 (worker.sports_nova_v23) is NEVER modified: this script only calls the sunday-panel adapter's build_m1_input(), set_identity_resolutions()
and simulate_game().  Inputs come ONLY from the immutable PANEL V2 directory (panel.json + its sources/ copies); no market data is read.
Terminal prepare() / validation_inputs_live are not touched.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worker.sports_nova_modular.firewall import assert_market_free  # noqa: E402
from worker.sports_nova_sunday_panel import adapter  # noqa: E402
from worker.sports_nova_v19.simulator import STAT_NAMES, _qb_shares, _state_hash  # noqa: E402
from worker.sports_nova_v23.simulator import MODEL_VERSION, set_identity_resolutions, simulate_game  # noqa: E402
from worker.sports_nova_v3.allocation import _shares  # noqa: E402  (read-only use: which players V23 actually draws from)

DATA = ROOT / "data" / "sports_nova_v3" / "sunday_2026_09_20"
PANEL_DIR = DATA / "SUNDAY_2026_09_20_PRE_GAME_PANEL_V2"
PANEL_SHA_EXPECTED = "dd16b820d779b12df93734bafa5b9dcadc00bc22029d4cc9eaf15e26b0f38840"
import os
OUT_DIR = Path(os.environ["SUNDAY_BASELINE_OUT"]) if os.environ.get("SUNDAY_BASELINE_OUT") else DATA / "SUNDAY_2026_09_20_M1_BASELINE_V1"  # env override = test runs only
N_SIMS_DEFAULT = 5000
SEED_POLICY = ("seed = int(sha256(game_id + '_SUNDAY_BASELINE_V1')[:8], 16); per-sim RNG = PCG64DXSM(SeedSequence([seed, sim_id])); "
               "same policy and N for every game including RED diagnostics; QB identity installed from panel via adapter (union of all 14 games, keyed by (game_id, team))")
DIAG_LABELS = {"2026_02_CIN_HOU": "DIAGNOSTIC_ONLY_QB_UNRESOLVED", "2026_02_SEA_ARI": "DIAGNOSTIC_ONLY_ROSTER_ALLOCATION_INVALID"}
PCTS = (10, 25, 75, 90)
CLASS_EDGES = [(0.02, "CLEAN"), (0.10, "LOW"), (0.25, "MATERIAL"), (0.50, "SEVERE")]  # > 0.50 -> CRITICAL
MEANINGFUL_SHARE = 0.01     # realized share of a team pool
MEANINGFUL_YARDS = 5.0      # realized mean rush+receiving+passing yards
PROD_STAT = {"QB": "pass_yards", "RB": "rush_yards", "WR": "receiving_yards", "TE": "receiving_yards"}


# ---------------------------------------------------------------------------------------------- shared helpers
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def engine_hashes() -> dict:
    """sha256 of every worker.sports_nova_v* module actually loaded (V23 + the v21/v19/v3 code it imports unchanged)."""
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and name.startswith("worker.sports_nova_v") and f.endswith(".py"):
            out[str(Path(f).resolve().relative_to(ROOT)).replace("\\", "/")] = sha256_file(Path(f))
    return out


def model_hash(hashes: dict) -> str:
    return hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(hashes.items())).encode()).hexdigest()


def summ(a) -> dict:
    a = np.asarray(a, dtype=float)
    q = np.percentile(a, PCTS)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "P10": float(q[0]), "P25": float(q[1]),
            "P75": float(q[2]), "P90": float(q[3])}


def seed_for(gid: str) -> int:
    return int(hashlib.sha256((gid + "_SUNDAY_BASELINE_V1").encode()).hexdigest()[:8], 16)


def load_inputs() -> dict:
    raw = (PANEL_DIR / "panel.json").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    panel = json.loads(raw)
    games = panel["GAMES"]
    inv = panel["HEADER"]["SOURCE_INVENTORY"]
    panel_df, panel_pq_sha = adapter.load_panel(PANEL_DIR / "sources" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.live.parquet")
    import pandas as pd
    names_df = panel_df.drop_duplicates("PLAYER_ID", keep="last")
    names = dict(zip(names_df.PLAYER_ID.astype(str), names_df.PLAYER_NAME))
    ros = pd.read_parquet(PANEL_DIR / "sources" / "roster_weekly_2026.parquet")
    max_week = max(int(g["WEEK"]) for g in games)
    ros = ros[ros.week == ros[ros.week <= max_week].week.max()]          # identical to freeze.py:296
    roster_idx = {str(r.gsis_id): (str(r.team), str(r.status)) for r in ros.itertuples() if r.gsis_id}
    inputs = {}
    for rec in games:
        inp = adapter.build_m1_input(rec, panel_df, panel_pq_sha, inv, roster_idx, names)
        inputs[rec["GAME_ID"]] = inp
    return {"raw_sha": sha, "panel": panel, "games": games, "inv": inv, "panel_df": panel_df, "panel_pq_sha": panel_pq_sha,
            "names": names, "roster_idx": roster_idx, "inputs": inputs, "roster_week": int(ros.week.iloc[0])}


def all_resolutions(inputs: dict) -> dict:
    res = {}
    for inp in inputs.values():
        res.update(inp.resolutions)
    return res


# ---------------------------------------------------------------------------------------------- sim worker (one game / process)
def sim_worker(args):
    gid, state, resolutions, n, outdir = args
    set_identity_resolutions(resolutions)                # global module state; installed BEFORE _qb_shares/simulate_game are read
    _, _, away, home = gid.split("_")[0], gid.split("_")[1], gid.split("_")[2], gid.split("_")[3]
    qb = {}
    for tid in (away, home):
        ids, shares = _qb_shares(state, tid)
        qb[tid] = {"CANDIDATES": [{"PLAYER_ID": i, "ENGINE_SHARE": float(s)} for i, s in zip(ids, shares)],
                   "PRIMARY_QB_ID": ids[int(np.argmax(shares))] if len(ids) else None}
    seed = seed_for(gid)
    t0 = time.time()
    b = simulate_game(state, n, seed, MODEL_VERSION)
    el = time.time() - t0
    hi, ai = b.team_ids.index(home), b.team_ids.index(away)
    hs, as_ = b.team_stats["score"][:, hi], b.team_stats["score"][:, ai]
    margin, total = hs - as_, hs + as_
    p_home, p_away, p_tie = float(np.mean(hs > as_)), float(np.mean(as_ > hs)), float(np.mean(hs == as_))
    team = {}
    for tid, j in ((away, ai), (home, hi)):
        py, ry = b.team_stats["pass_yards"][:, j], b.team_stats["rush_yards"][:, j]
        team[tid] = {"pass_attempts": summ(b.team_stats["pass_attempts"][:, j]), "rush_attempts": summ(b.team_stats["rush_attempts"][:, j]),
                     "pass_yards": summ(py), "rush_yards": summ(ry), "total_yards": summ(py + ry)}
    pos = {p.player_id: p.position for p in state.players}
    players, keep = [], []
    for j, pid in enumerate(b.player_ids):
        means = {s: float(b.player_stats[s][:, j].mean()) for s in STAT_NAMES}
        if not any(b.player_stats[s][:, j].any() for s in STAT_NAMES):
            continue
        keep.append(j)
        row = {"PLAYER_ID": pid, "TEAM": b.player_team[pid], "POSITION": pos.get(pid), "MEAN": means,
               "P_ANY_TD": float(np.mean(b.player_stats["scored_tds"][:, j] > 0))}
        ps = PROD_STAT.get(pos.get(pid))
        if ps:
            row["PRODUCTION_STAT"] = ps
            row["PRODUCTION_DIST"] = summ(b.player_stats[ps][:, j])
        players.append(row)
    for tid in (away, home):                              # realized pass-attempt share per QB (how identity really landed)
        tot = sum(p["MEAN"]["pass_attempts"] for p in players if p["TEAM"] == tid) or 1.0
        for p in players:
            if p["TEAM"] == tid and p["POSITION"] == "QB" and p["MEAN"]["pass_attempts"] > 0:
                qb[tid].setdefault("REALIZED_PASS_ATT_SHARE", {})[p["PLAYER_ID"]] = p["MEAN"]["pass_attempts"] / tot
    res = {"GAME_ID": gid, "AWAY": away, "HOME": home, "N_SIMS": n, "SEED": seed, "STATE_HASH": b.state_hash, "ENGINE_STATUS": b.status,
           "MODEL_VERSION": b.model_version, "RUNTIME_SEC": el, "SIMS_PER_SEC": n / el,
           "P_HOME_STRICT": p_home, "P_AWAY_STRICT": p_away, "P_TIE": p_tie,
           "HOME_WIN_PROB_TIE_SPLIT": p_home + 0.5 * p_tie, "AWAY_WIN_PROB_TIE_SPLIT": p_away + 0.5 * p_tie,
           "SCORE_AWAY": summ(as_), "SCORE_HOME": summ(hs), "MARGIN_HOME_MINUS_AWAY": summ(margin), "TOTAL_POINTS": summ(total),
           "TEAM": team, "QB": qb, "PLAYERS": players,
           "RAW_FILE": f"raw/{gid}.npz"}
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    (outdir / "games").mkdir(parents=True, exist_ok=True)
    arrs = {f"player_{s}": b.player_stats[s][:, keep].astype(np.int32) for s in STAT_NAMES}
    arrs.update({f"team_{s}": b.team_stats[s] for s in b.team_stats})
    np.savez_compressed(outdir / "raw" / f"{gid}.npz", player_ids=np.array([b.player_ids[j] for j in keep]), team_ids=np.array(b.team_ids),
                        winners=b.winner, **arrs)
    (outdir / "games" / f"{gid}.json").write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    return gid, el


def cmd_sim(n: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before = engine_hashes()
    ctx = load_inputs()
    assert ctx["raw_sha"] == PANEL_SHA_EXPECTED, "PANEL SHA MISMATCH"
    res = all_resolutions(ctx["inputs"])
    # determinism smoke: same state + seed twice -> identical arrays
    gid0 = "2026_02_IND_KC"
    set_identity_resolutions(res)
    a = simulate_game(ctx["inputs"][gid0].state, 40, seed_for(gid0), MODEL_VERSION)
    b = simulate_game(ctx["inputs"][gid0].state, 40, seed_for(gid0), MODEL_VERSION)
    det = all(np.array_equal(a.player_stats[s], b.player_stats[s]) for s in STAT_NAMES) and np.array_equal(a.team_stats["score"], b.team_stats["score"])
    print(f"determinism smoke (N=40 x2, {gid0}): {det}", flush=True)
    (OUT_DIR / "sim_run_meta.json").write_text(json.dumps({
        "STARTED_AT": datetime.now(timezone.utc).isoformat(), "N_SIMS": n, "ENGINE_HASHES_BEFORE": before, "DETERMINISM_SMOKE": det,
        "PANEL_SHA256": ctx["raw_sha"], "SEED_POLICY": SEED_POLICY}, indent=1), encoding="utf-8")
    jobs = []
    for gid, inp in ctx["inputs"].items():
        if inp.state is None:
            print(f"SKIP {gid}: no state: {inp.missing}", flush=True)
            continue
        jobs.append((gid, inp.state, res, n, OUT_DIR))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(sim_worker, j): j[0] for j in jobs}
        for f in as_completed(futs):
            gid, el = f.result()
            print(f"done {gid} {el:.0f}s ({n / el:.1f} sims/s)  wall={time.time() - t0:.0f}s", flush=True)
    after = engine_hashes()
    meta = json.loads((OUT_DIR / "sim_run_meta.json").read_text())
    meta.update({"FINISHED_AT": datetime.now(timezone.utc).isoformat(), "ENGINE_HASHES_AFTER": after, "ENGINE_HASHES_UNCHANGED_DURING_SIM": before == after,
                 "WALL_SEC": time.time() - t0})
    (OUT_DIR / "sim_run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("SIM COMPLETE; engine unchanged during run:", before == after, flush=True)


# ---------------------------------------------------------------------------------------------- analysis
def classify(x: float) -> str:
    for edge, name in CLASS_EDGES:
        if x <= edge:
            return name
    return "CRITICAL"


def rcat(pid, team, roster_idx) -> str:
    rt, rs = roster_idx.get(pid, (None, None))
    if rt == team and rs == "ACT":
        return "ACTIVE_THIS_TEAM"
    if rt == team:
        return rs
    if rt is not None:
        return "ON_OTHER_TEAM_ROSTER"
    return "NOT_ON_ANY_2026_ROSTER"


def analyze_game(rec, inp, g, ctx) -> dict:
    gid, away, home = rec["GAME_ID"], rec["AWAY_TEAM"], rec["HOME_TEAM"]
    ridx, names, state = ctx["roster_idx"], ctx["names"], inp.state
    out_ids = {p["PLAYER_ID"] for s in ("AWAY", "HOME") for p in rec["INJURY_STATE"][s]["VALUE"]["PLAYERS"] if p["M1_AVAILABILITY"] == "OUT"}
    doubtful = {p["PLAYER_ID"]: p for s in ("AWAY", "HOME") for p in rec["INJURY_STATE"][s]["VALUE"]["PLAYERS"] if p["GAME_STATUS"] == "DOUBTFUL"}
    prow = {p["PLAYER_ID"]: p for p in g["PLAYERS"]}
    pos_state = {p.player_id: p.position for p in state.players}
    teams = {}
    invalid = []
    for team in (away, home):
        ts = state.home if state.home.team_id == team else state.away
        raw_in = {"carry": dict(zip(ts.carry_shares.player_ids, ts.carry_shares.shares)),
                  "target": dict(zip(ts.target_shares.player_ids, ts.target_shares.shares))}
        eff = {}
        for kind in ("carry", "target"):
            ids, vals, _ = _shares(state, team, kind)
            tot = float(np.sum(vals)) or 1.0
            eff[kind] = {i: float(v) / tot for i, v in zip(ids, vals)}
        tp = [p for p in g["PLAYERS"] if p["TEAM"] == team]
        pool = {"carry": sum(p["MEAN"]["rush_attempts"] for p in tp), "target": sum(p["MEAN"]["targets"] for p in tp)}
        pids = set(raw_in["carry"]) | set(raw_in["target"]) | {p["PLAYER_ID"] for p in tp}
        pids = {p for p in pids if any(p == s.player_id and s.team_id == team for s in state.players)}
        rows, nonact = [], {"carry": 0.0, "target": 0.0}
        res_share = {"carry": 0.0, "target": 0.0}
        for pid in sorted(pids):
            cat = rcat(pid, team, ridx)
            is_out = pid in out_ids
            pr = prow.get(pid)
            m = pr["MEAN"] if pr else {s: 0.0 for s in STAT_NAMES}
            rc = m["rush_attempts"] / pool["carry"] if pool["carry"] else 0.0
            rt = m["targets"] / pool["target"] if pool["target"] else 0.0
            noneligible = cat != "ACTIVE_THIS_TEAM" or is_out
            usage = m["pass_attempts"] + m["targets"] + m["rush_attempts"]
            yards = m["pass_yards"] + m["rush_yards"] + m["receiving_yards"]
            meaningful = usage > 0 and (rc >= MEANINGFUL_SHARE or rt >= MEANINGFUL_SHARE or abs(yards) >= MEANINGFUL_YARDS)
            if noneligible:
                nonact["carry"] += rc
                nonact["target"] += rt
                if cat == "RES":
                    res_share["carry"] += rc
                    res_share["target"] += rt
            row = {"PLAYER_ID": pid, "NAME": names.get(pid), "POSITION": pos_state.get(pid), "ROSTER_CATEGORY": cat,
                   "PANEL_OUT": is_out, "PANEL_DOUBTFUL": pid in doubtful,
                   "INPUT_CARRY_SHARE_RAW": round(float(raw_in["carry"].get(pid, 0.0)), 5),
                   "INPUT_TARGET_SHARE_RAW": round(float(raw_in["target"].get(pid, 0.0)), 5),
                   "INPUT_CARRY_SHARE_EFFECTIVE": round(eff["carry"].get(pid, 0.0), 5),
                   "INPUT_TARGET_SHARE_EFFECTIVE": round(eff["target"].get(pid, 0.0), 5),
                   "SIM_CARRY_SHARE": round(rc, 5), "SIM_TARGET_SHARE": round(rt, 5),
                   "SIM_MEAN_PASS_ATT": round(m["pass_attempts"], 3), "SIM_MEAN_CARRIES": round(m["rush_attempts"], 3),
                   "SIM_MEAN_TARGETS": round(m["targets"], 3), "SIM_MEAN_PASS_YDS": round(m["pass_yards"], 3),
                   "SIM_MEAN_RUSH_YDS": round(m["rush_yards"], 3), "SIM_MEAN_REC_YDS": round(m["receiving_yards"], 3),
                   "SIM_P_ANY_TD": round(pr["P_ANY_TD"], 4) if pr else 0.0,
                   "NON_ELIGIBLE": noneligible, "ANY_SIM_USAGE": usage > 0, "MEANINGFUL_SIM_PRODUCTION": bool(noneligible and meaningful)}
            rows.append(row)
            if noneligible and usage > 0:
                invalid.append({"TEAM": team, **{k: row[k] for k in ("PLAYER_ID", "NAME", "POSITION", "ROSTER_CATEGORY", "PANEL_OUT", "SIM_CARRY_SHARE",
                                                                     "SIM_TARGET_SHARE", "SIM_MEAN_CARRIES", "SIM_MEAN_TARGETS", "SIM_MEAN_PASS_ATT",
                                                                     "SIM_MEAN_RUSH_YDS", "SIM_MEAN_REC_YDS", "SIM_MEAN_PASS_YDS", "MEANINGFUL_SIM_PRODUCTION")}})
        comb_pool = pool["carry"] + pool["target"]
        comb_non = nonact["carry"] * pool["carry"] + nonact["target"] * pool["target"]
        teams[team] = {
            "SIM_POOL_MEAN_CARRIES": round(pool["carry"], 3), "SIM_POOL_MEAN_TARGETS": round(pool["target"], 3),
            "NON_ACTIVE_CARRY_SHARE": round(nonact["carry"], 5), "ACTIVE_CARRY_SHARE": round(1 - nonact["carry"], 5),
            "NON_ACTIVE_TARGET_SHARE": round(nonact["target"], 5), "ACTIVE_TARGET_SHARE": round(1 - nonact["target"], 5),
            "NON_ACTIVE_ROSTER_SHARE": round(comb_non / comb_pool, 5), "ACTIVE_ROSTER_SHARE": round(1 - comb_non / comb_pool, 5),
            "RES_ONLY_CARRY_SHARE": round(res_share["carry"], 5), "RES_ONLY_TARGET_SHARE": round(res_share["target"], 5),
            "PANEL_INPUT_SIDE_NON_ACTIVE_RAW": {k: rec["M1_ADAPTER"]["ROSTER_ALIGNMENT"][f"{team}_{k}"]["SHARE_NOT_ACTIVE_THIS_TEAM"] for k in ("carry", "target")},
            "PLAYERS": rows}
    pools = [teams[t][k] for t in (away, home) for k in ("NON_ACTIVE_CARRY_SHARE", "NON_ACTIVE_TARGET_SHARE")]
    tc = sum(teams[t]["NON_ACTIVE_CARRY_SHARE"] * teams[t]["SIM_POOL_MEAN_CARRIES"] for t in (away, home)) / sum(teams[t]["SIM_POOL_MEAN_CARRIES"] for t in (away, home))
    tt = sum(teams[t]["NON_ACTIVE_TARGET_SHARE"] * teams[t]["SIM_POOL_MEAN_TARGETS"] for t in (away, home)) / sum(teams[t]["SIM_POOL_MEAN_TARGETS"] for t in (away, home))
    mx = max(pools)
    by_cat = {}
    for r in invalid:
        c = "PANEL_OUT_BUT_USED" if r["PANEL_OUT"] else r["ROSTER_CATEGORY"]
        by_cat[c] = by_cat.get(c, 0) + 1
    game = {"GAME_ID": gid, "MAX_TEAM_NONACTIVE_SHARE": round(mx, 5), "MAX_TEAM_POOL_DEFINITION": "max over the 4 (team x {carry,target}) pools of simulated non-eligible share",
            "TOTAL_NONACTIVE_CARRY_SHARE": round(tc, 5), "TOTAL_NONACTIVE_TARGET_SHARE": round(tt, 5),
            "CONTAMINATION_CLASS": classify(mx), "INVALID_PLAYER_USAGE_COUNT": len(invalid),
            "INVALID_PLAYER_USAGE_MEANINGFUL_COUNT": sum(1 for r in invalid if r["MEANINGFUL_SIM_PRODUCTION"]),
            "INVALID_PLAYER_USAGE_BY_CATEGORY": by_cat, "INVALID_PLAYERS": sorted(invalid, key=lambda r: -(r["SIM_CARRY_SHARE"] + r["SIM_TARGET_SHARE"])),
            "PANEL_OUT_PLAYERS_WITH_SIM_USAGE": [r for r in invalid if r["PANEL_OUT"]],
            "DOUBTFUL_PLAYERS_WITH_SIM_USAGE_INFO_ONLY": [
                {"PLAYER_ID": pid, "NAME": d["NAME"], "TEAM": d["TEAM"], "POSITION": d["POSITION"],
                 "SIM_MEAN_OPPS": round(sum(prow[pid]["MEAN"][s] for s in ("pass_attempts", "targets", "rush_attempts")), 3) if pid in prow else 0.0}
                for pid, d in doubtful.items() if pid in prow and sum(prow[pid]["MEAN"][s] for s in ("pass_attempts", "targets", "rush_attempts")) > 0],
            "TEAMS": teams}
    # QB integrity
    qbrows = []
    for side, team in (("AWAY", away), ("HOME", home)):
        q = rec["QB_STATE"][side]["VALUE"]
        exp = q["QB1_EXPECTED"]["PLAYER_ID"] if q["QB1_EXPECTED"] else None
        primary = g["QB"][team]["PRIMARY_QB_ID"]
        installed = (gid, team) in inp.resolutions
        realized = g["QB"][team].get("REALIZED_PASS_ATT_SHARE", {})
        in_state = exp in pos_state if exp else False
        qbrows.append({"TEAM": team, "SIDE": side, "PANEL_QB_EXPECTED_ID": exp, "PANEL_QB_EXPECTED_NAME": q["QB1_EXPECTED"]["NAME"] if exp else None,
                       "PANEL_STARTER_STATUS": q["STARTER_STATUS"], "PANEL_QB_CONFIDENCE": q["QB_CONFIDENCE"],
                       "ENGINE_IDENTITY_INSTALLED": installed, "EXPECTED_QB_IN_M1_STATE": in_state,
                       "ENGINE_PRIMARY_QB_ID": primary, "ENGINE_PRIMARY_QB_NAME": names.get(primary),
                       "MATCH": bool(exp and primary == exp),
                       "MATCH_BASIS": ("PANEL_IDENTITY_INSTALLED" if installed else "ENGINE_FALLBACK_NOT_CONFIRMED_STARTER") if (exp and primary == exp) else None,
                       "REALIZED_PASS_ATT_SHARE_BY_QB": {f"{names.get(k, k)} ({k})": round(v, 4) for k, v in sorted(realized.items(), key=lambda kv: -kv[1])},
                       "REALIZED_SHARE_OF_EXPECTED_QB": round(realized.get(exp, 0.0), 4) if exp else None,
                       "ENGINE_CANDIDATES": [{"NAME": names.get(c["PLAYER_ID"]), **c, "ROSTER_CATEGORY": rcat(c["PLAYER_ID"], team, ctx["roster_idx"])}
                                             for c in g["QB"][team]["CANDIDATES"]]})
    game["QB_INTEGRITY"] = qbrows
    return game


def sim_class(panel_class: str, cont: dict, qbrows: list, state_hash_ok: bool, invalid_out: int) -> tuple[str, list]:
    red, yel = [], []
    if cont["CONTAMINATION_CLASS"] == "CRITICAL":
        red.append(f"CRITICAL_ROSTER_MISUSE (max pool non-eligible share {cont['MAX_TEAM_NONACTIVE_SHARE']:.1%})")
    for q in qbrows:
        if not q["ENGINE_IDENTITY_INSTALLED"] and q["PANEL_QB_CONFIDENCE"] == "LOW":
            red.append(f"QB_UNRESOLVED:{q['TEAM']} (panel LOW confidence, no identity installed; engine fallback used)")
        elif not q["MATCH"]:
            red.append(f"QB_MISMATCH:{q['TEAM']}")
        elif q["PANEL_QB_CONFIDENCE"] == "MEDIUM":
            yel.append(f"QB_CONFIDENCE_MEDIUM:{q['TEAM']}")
        if q["MATCH"] and q["REALIZED_SHARE_OF_EXPECTED_QB"] is not None and q["REALIZED_SHARE_OF_EXPECTED_QB"] < 0.95 and q["ENGINE_IDENTITY_INSTALLED"]:
            yel.append(f"QB_SPLIT:{q['TEAM']} realized {q['REALIZED_SHARE_OF_EXPECTED_QB']:.2f}")
    if invalid_out:
        red.append(f"PANEL_OUT_PLAYERS_USED_BY_SIM:{invalid_out}")
    if not state_hash_ok:
        red.append("INPUT_HASH_DRIFT_VS_PANEL_SMOKE")
    if cont["CONTAMINATION_CLASS"] in ("MATERIAL", "SEVERE"):
        yel.append(f"CONTAMINATION_{cont['CONTAMINATION_CLASS']}")
    elif cont["CONTAMINATION_CLASS"] == "LOW":
        pass
    if red:
        return "SIM_RED", red + yel
    if cont["CONTAMINATION_CLASS"] in ("CLEAN", "LOW") and not yel:
        if panel_class == "RED":                 # spec: SIM_GREEN needs an acceptable panel; this is the ONLY place the panel class is read
            return "SIM_YELLOW", ["PANEL_RED_NOT_ACCEPTABLE_FOR_GREEN_GATE"]
        return "SIM_GREEN", []
    return "SIM_YELLOW", yel


def cmd_analyze() -> None:
    ctx = load_inputs()
    assert ctx["raw_sha"] == PANEL_SHA_EXPECTED, "PANEL SHA MISMATCH"
    meta = json.loads((OUT_DIR / "sim_run_meta.json").read_text())
    now_hashes = engine_hashes()
    man = json.loads((PANEL_DIR / "MANIFEST.json").read_text())
    v23_ok = {k: (now_hashes.get(k.replace("worker/", "worker/", 1)) == v) for k, v in man["M1_ADAPTER_CODE_SHA256"].items() if "sports_nova_v23" in k}
    m1_unchanged = bool(meta.get("ENGINE_HASHES_UNCHANGED_DURING_SIM")) and now_hashes == meta["ENGINE_HASHES_BEFORE"] and all(v23_ok.values())
    games_out, cont_out, board, anomalies = [], [], [], []
    per = {}
    for rec in ctx["games"]:
        gid = rec["GAME_ID"]
        gp = OUT_DIR / "games" / f"{gid}.json"
        if not gp.exists():
            print("MISSING sim for", gid)
            continue
        g = json.loads(gp.read_text())
        inp = ctx["inputs"][gid]
        per[gid] = (rec, inp, g)
    for gid, (rec, inp, g) in per.items():
        a = analyze_game(rec, inp, g, ctx)
        smoke_hash = rec["M1_ADAPTER"]["SMOKE"].get("M1_STATE_HASH")
        input_hash = _state_hash(inp.state)
        hash_ok = (input_hash == smoke_hash == g["STATE_HASH"])
        pc = rec["DATA_QUALITY"]["CLASS"]
        sc, reasons = sim_class(pc, a, a["QB_INTEGRITY"], hash_ok, len(a["PANEL_OUT_PLAYERS_WITH_SIM_USAGE"]))
        sc_if_panel_green = sim_class("GREEN", a, a["QB_INTEGRITY"], hash_ok, len(a["PANEL_OUT_PLAYERS_WITH_SIM_USAGE"]))[0]
        label = DIAG_LABELS.get(gid) or ("BASELINE_" + sc)
        a.update({"SIM_CLASS": sc, "SIM_CLASS_REASONS": reasons, "PUBLICATION_LABEL": label, "PANEL_CLASS": pc,
                  "SIM_CLASS_DEPENDS_ON_PANEL_CLASS": sc != sc_if_panel_green,
                  "INPUT_HASH": input_hash, "PANEL_SMOKE_STATE_HASH": smoke_hash, "INPUT_HASH_MATCHES_PANEL_AND_SIM": hash_ok})
        per[gid] = (rec, inp, g, a)
    # ------------------------------------------------------------------ slate stats for anomaly rules
    totals = [g["TOTAL_POINTS"]["median"] for _, _, g, _ in per.values()]
    tm, ts = float(np.mean(totals)), float(np.std(totals)) or 1.0
    scores = [g[k]["median"] for _, _, g, _ in per.values() for k in ("SCORE_AWAY", "SCORE_HOME")]
    sm, ss = float(np.mean(scores)), float(np.std(scores)) or 1.0
    for gid, (rec, inp, g, a) in per.items():
        away, home = g["AWAY"], g["HOME"]
        fl = []

        def flag(kind, detail, cause):
            fl.append({"GAME": f"{away}@{home}", "KIND": kind, "DETAIL": detail, "CAUSE_CANDIDATE": cause})

        ph = g["HOME_WIN_PROB_TIE_SPLIT"]
        if ph < 0.15 or ph > 0.85:
            flag("EXTREME_WIN_PROB", f"home win {ph:.1%}", "PLAYER_MODEL" if a["CONTAMINATION_CLASS"] in ("SEVERE", "CRITICAL") else "UNKNOWN")
        for side, tid in (("SCORE_AWAY", away), ("SCORE_HOME", home)):
            med = g[side]["median"]
            if med < 10 or med > 30 or abs(med - sm) > 2 * ss:
                flag("EXTREME_SCORE_MEDIAN", f"{tid} median {med:.0f} (slate mean of medians {sm:.1f} sd {ss:.1f})", "SCORE_MODEL")
        tmed = g["TOTAL_POINTS"]["median"]
        if tmed < 28 or tmed > 52 or abs(tmed - tm) > 2 * ts:
            flag("EXTREME_TOTAL", f"total median {tmed:.0f} (slate mean {tm:.1f} sd {ts:.1f})", "SCORE_MODEL")
        mm = g["MARGIN_HOME_MINUS_AWAY"]["median"]
        if abs(mm) > 14:
            flag("EXTREME_MARGIN", f"median margin (home-away) {mm:+.0f}", "SCORE_MODEL")
        for tid in (away, home):
            t = g["TEAM"][tid]
            pa, ra = t["pass_attempts"]["mean"], t["rush_attempts"]["mean"]
            ps = pa / (pa + ra)
            if ps < 0.45 or ps > 0.65 or pa < 20 or ra < 15:
                flag("STRANGE_PASS_RUSH_SPLIT", f"{tid} pass {pa:.1f} (net of sacks+scrambles) / rush {ra:.1f} (designed+scrambles) (pass share {ps:.0%}); possibly shares a root cause with the known low-total behaviour", "UNKNOWN")
            tp = a["TEAMS"][tid]["PLAYERS"]
            tc = max(tp, key=lambda r: r["SIM_CARRY_SHARE"])
            tt = max(tp, key=lambda r: r["SIM_TARGET_SHARE"])
            if tc["SIM_CARRY_SHARE"] > 0.60:
                flag("OPPORTUNITY_CONCENTRATION", f"{tid} {tc['NAME']} {tc['SIM_CARRY_SHARE']:.0%} of carries", "ROSTER_ALLOCATION")
            if tt["SIM_TARGET_SHARE"] > 0.30:
                flag("OPPORTUNITY_CONCENTRATION", f"{tid} {tt['NAME']} {tt['SIM_TARGET_SHARE']:.0%} of targets", "ROSTER_ALLOCATION")
            for r in tp:
                if r["MEANINGFUL_SIM_PRODUCTION"]:
                    flag("NON_ACTIVE_PLAYER_PRODUCTION",
                         f"{tid} {r['NAME']} [{r['ROSTER_CATEGORY']}{'/OUT' if r['PANEL_OUT'] else ''}] carries {r['SIM_MEAN_CARRIES']:.1f} targets {r['SIM_MEAN_TARGETS']:.1f} "
                         f"yds {r['SIM_MEAN_RUSH_YDS'] + r['SIM_MEAN_REC_YDS'] + r['SIM_MEAN_PASS_YDS']:.0f}", "ROSTER_ALLOCATION")
        for q in a["QB_INTEGRITY"]:
            if not q["ENGINE_IDENTITY_INSTALLED"]:
                flag("QB_IDENTITY", f"{q['TEAM']} identity not installed ({q['PANEL_STARTER_STATUS']}/{q['PANEL_QB_CONFIDENCE']}); engine primary "
                                    f"{q['ENGINE_PRIMARY_QB_NAME']} (fallback, realized split {q['REALIZED_PASS_ATT_SHARE_BY_QB']})", "QB_UNCERTAINTY")
            elif not q["MATCH"]:
                flag("QB_IDENTITY", f"{q['TEAM']} engine QB {q['ENGINE_PRIMARY_QB_NAME']} != panel {q['PANEL_QB_EXPECTED_NAME']}", "INPUT_DATA")
            elif q["REALIZED_SHARE_OF_EXPECTED_QB"] is not None and q["REALIZED_SHARE_OF_EXPECTED_QB"] < 0.95:
                flag("QB_IDENTITY", f"{q['TEAM']} expected QB only {q['REALIZED_SHARE_OF_EXPECTED_QB']:.0%} of realized pass attempts", "QB_UNCERTAINTY")
            elif not q["EXPECTED_QB_IN_M1_STATE"]:
                flag("QB_IDENTITY", f"{q['TEAM']} expected QB absent from M1 state", "INPUT_DATA")
        a["ANOMALIES"] = fl
        anomalies += fl
        per[gid] = (rec, inp, g, a)

    # ------------------------------------------------------------------ slate aggregates
    order = [r["GAME_ID"] for r in ctx["games"] if r["GAME_ID"] in per]
    team_comb = [a["TEAMS"][t]["NON_ACTIVE_ROSTER_SHARE"] for gid in order for t in (per[gid][2]["AWAY"], per[gid][2]["HOME"]) for a in [per[gid][3]]]
    pool_all = [a["TEAMS"][t][k] for gid in order for a in [per[gid][3]] for t in (per[gid][2]["AWAY"], per[gid][2]["HOME"])
                for k in ("NON_ACTIVE_CARRY_SHARE", "NON_ACTIVE_TARGET_SHARE")]
    worst = max(order, key=lambda gid: per[gid][3]["MAX_TEAM_NONACTIVE_SHARE"])
    qb_rows = [q for gid in order for q in per[gid][3]["QB_INTEGRITY"]]
    qm = sum(1 for q in qb_rows if q["MATCH"])
    qfb = sum(1 for q in qb_rows if q["MATCH"] and q["MATCH_BASIS"] != "PANEL_IDENTITY_INSTALLED")
    sc_counts = {k: sum(1 for gid in order if per[gid][3]["SIM_CLASS"] == k) for k in ("SIM_GREEN", "SIM_YELLOW", "SIM_RED")}
    cont_counts = {k: sum(1 for gid in order if per[gid][3]["CONTAMINATION_CLASS"] == k) for k in ("CLEAN", "LOW", "MATERIAL", "SEVERE", "CRITICAL")}
    invalid_total = sum(per[gid][3]["INVALID_PLAYER_USAGE_COUNT"] for gid in order)
    invalid_material = sum(per[gid][3]["INVALID_PLAYER_USAGE_MEANINGFUL_COUNT"] for gid in order)
    panel_out_bad = sum(len(per[gid][3]["PANEL_OUT_PLAYERS_WITH_SIM_USAGE"]) for gid in order)
    diag = [gid for gid in order if gid in DIAG_LABELS or per[gid][3]["SIM_CLASS"] == "SIM_RED"]
    published = [gid for gid in order if gid not in DIAG_LABELS]
    n_sims = {per[gid][2]["N_SIMS"] for gid in order}

    # SEA@ARI special audit
    sa = per.get("2026_02_SEA_ARI")
    special = {}
    if sa:
        _, _, g, a = sa
        by_name = {}
        for t in ("SEA", "ARI"):
            for r in a["TEAMS"][t]["PLAYERS"]:
                by_name[(t, r["NAME"])] = r
        for nm in ("James Conner", "Trey Benson"):
            hits = [dict(r, TEAM=t) for (t, n2), r in by_name.items() if n2 == nm]
            special[nm] = hits or "NOT_IN_M1_STATE_FOR_SEA_OR_ARI"
        special["ALL_NON_ACTIVE_PLAYERS_WITH_ALLOCATION"] = {
            t: sorted([{k: r[k] for k in ("NAME", "POSITION", "ROSTER_CATEGORY", "PANEL_OUT", "INPUT_CARRY_SHARE_RAW", "INPUT_TARGET_SHARE_RAW", "SIM_CARRY_SHARE",
                                          "SIM_TARGET_SHARE", "SIM_MEAN_CARRIES", "SIM_MEAN_TARGETS", "SIM_MEAN_RUSH_YDS", "SIM_MEAN_REC_YDS", "SIM_P_ANY_TD")}
                       for r in a["TEAMS"][t]["PLAYERS"] if r["NON_ELIGIBLE"] and r["ANY_SIM_USAGE"]],
                      key=lambda r: -(r["SIM_CARRY_SHARE"] + r["SIM_TARGET_SHARE"])) for t in ("SEA", "ARI")}
        special["TEAM_SHARES"] = {t: {k: v for k, v in a["TEAMS"][t].items() if k != "PLAYERS"} for t in ("SEA", "ARI")}
        special["RBS_CARRYING_WITH_ROSTER_CATEGORY"] = {
            t: sorted([{"NAME": r["NAME"], "SIM_CARRY_SHARE": r["SIM_CARRY_SHARE"], "SIM_MEAN_CARRIES": r["SIM_MEAN_CARRIES"], "ROSTER_CATEGORY": r["ROSTER_CATEGORY"]}
                       for r in a["TEAMS"][t]["PLAYERS"] if r["POSITION"] == "RB" and r["SIM_CARRY_SHARE"] > 0.02], key=lambda r: -r["SIM_CARRY_SHARE"])
            for t in ("SEA", "ARI")}

    now = datetime.now(timezone.utc).isoformat()
    common = {"CREATED_AT": now, "M1_VERSION": MODEL_VERSION, "M1_MODEL_HASH": model_hash(now_hashes), "M1_ENGINE_FILE_HASHES": now_hashes,
              "M1_UNCHANGED": m1_unchanged, "M1_V23_FILES_MATCH_PANEL_MANIFEST": v23_ok,
              "PANEL_PATH": str(PANEL_DIR / "panel.json"), "PANEL_SHA256": ctx["raw_sha"], "PANEL_SHA_VERIFIED": ctx["raw_sha"] == PANEL_SHA_EXPECTED,
              "PANEL_MARKET_FIELDS_PRESENT_PER_PANEL_MANIFEST": man["MARKET_FIELDS_PRESENT"],
              "SEED_POLICY": SEED_POLICY, "SIMULATION_COUNT_PER_GAME": sorted(n_sims), "INPUT_ADAPTER": "worker.sports_nova_sunday_panel.adapter.build_m1_input",
              "NOT_USED": "terminal prepare() / validation_inputs_live injury path / any market input"}

    # ------------------------------------------------------------------ artifact 1: baseline
    def game_block(gid):
        rec, inp, g, a = per[gid]
        return {"GAME_ID": gid, "GAME": f"{g['AWAY']}@{g['HOME']}", "PANEL_CLASS": a["PANEL_CLASS"], "SIM_CLASS": a["SIM_CLASS"],
                "PUBLICATION_LABEL": a["PUBLICATION_LABEL"], "IS_CLEAN_NOVA_FAIR_VALUE": a["SIM_CLASS"] == "SIM_GREEN" and gid not in DIAG_LABELS,
                "INPUT_HASH": a["INPUT_HASH"], "SEED": g["SEED"], "N_SIMS": g["N_SIMS"], "RAW_FILE": g["RAW_FILE"],
                "away_win_prob": g["AWAY_WIN_PROB_TIE_SPLIT"], "home_win_prob": g["HOME_WIN_PROB_TIE_SPLIT"],
                "p_home_strict": g["P_HOME_STRICT"], "p_away_strict": g["P_AWAY_STRICT"], "p_tie": g["P_TIE"],
                "away_score": g["SCORE_AWAY"], "home_score": g["SCORE_HOME"], "margin_home_minus_away": g["MARGIN_HOME_MINUS_AWAY"],
                "total_points": g["TOTAL_POINTS"], "team": g["TEAM"],
                "FAIR_SPREAD_RAW_HOME": -g["MARGIN_HOME_MINUS_AWAY"]["median"], "FAIR_TOTAL_RAW": g["TOTAL_POINTS"]["median"],
                "RAW_NOTE": "raw M1-derived research outputs; NOT calibrated, NOT actionable",
                "QB": {q["TEAM"]: {k: q[k] for k in ("PANEL_QB_EXPECTED_NAME", "PANEL_STARTER_STATUS", "PANEL_QB_CONFIDENCE", "ENGINE_IDENTITY_INSTALLED", "ENGINE_PRIMARY_QB_NAME",
                                                      "MATCH", "MATCH_BASIS", "REALIZED_PASS_ATT_SHARE_BY_QB")} for q in a["QB_INTEGRITY"]},
                "players": [dict(p, NAME=ctx["names"].get(p["PLAYER_ID"]), ROSTER_CATEGORY=rcat(p["PLAYER_ID"], p["TEAM"], ctx["roster_idx"]))
                            for p in sorted(g["PLAYERS"], key=lambda p: (p["TEAM"], -(p["MEAN"]["targets"] + p["MEAN"]["rush_attempts"] + p["MEAN"]["pass_attempts"])))
                            if "PRODUCTION_DIST" in p],
                "sim_class_reasons": a["SIM_CLASS_REASONS"], "runtime_sec": g["RUNTIME_SEC"]}

    baseline = {**common, "SCHEMA": "SUNDAY_2026_09_20_M1_BASELINE_V1", "GAMES_TOTAL": len(order),
                "DEFINITIONS": {
                    "away_win_prob/home_win_prob": "P(win) with ties split 50/50; strict P(home>away)/P(away>home)/P(tie) also given",
                    "margin": "home minus away", "FAIR_SPREAD_RAW_HOME": "-(median home-minus-away margin); raw, uncalibrated",
                    "FAIR_TOTAL_RAW": "median of home+away points; raw, uncalibrated",
                    "team.rush_attempts": "designed rushes + QB scrambles (play_selection.py: rush_attempts = designed_rushes + scrambles); equals the sum of player rush_attempts",
                    "team.pass_attempts": "dropbacks minus sacks minus scrambles (pass_attempts = non_sack - scrambles), so it is NOT dropbacks; pass share computed in the anomaly rule is pass_attempts/(pass_attempts+rush_attempts)",
                    "team.rush_yards": "includes QB scramble yards", "team.total_yards": "pass_yards + rush_yards per sim",
                    "players": "QB/RB/WR/TE players with nonzero simulated production; PRODUCTION_DIST = pass_yards (QB) / rush_yards (RB) / receiving_yards (WR,TE)"},
                "RED_HANDLING": DIAG_LABELS, "GAMES": [game_block(g) for g in order]}
    # ------------------------------------------------------------------ artifact 2: contamination audit
    audit = {**common, "SCHEMA": "SUNDAY_2026_09_20_ROSTER_CONTAMINATION_AUDIT", "ROSTER_SOURCE": f"panel V2 sources/roster_weekly_2026.parquet week {ctx['roster_week']} (status ACT on the same team = active)",
             "DEFINITIONS": {
                 "NON_ELIGIBLE": "player is not ACT on this team's current roster (RES / DEV / CUT / other team / on no 2026 roster) OR is panel-OUT",
                 "SIM_*_SHARE": "player's mean simulated carries (rush_attempts, incl. QB scrambles) or targets divided by the team's mean simulated pool -- measured on M1 OUTPUT",
                 "INPUT_*_SHARE_RAW": "share in the M1 PregameState before OUT flips (what N2's panel measures)",
                 "INPUT_*_SHARE_EFFECTIVE": "share V23 actually draws from after dropping availability==OUT and renormalizing",
                 "NON_ACTIVE_ROSTER_SHARE": "team combined (carries+targets) simulated non-eligible share",
                 "MAX_TEAM_NONACTIVE_SHARE": "game-level: max over the four team-pools (carry/target x 2 teams) of simulated non-eligible share -- feeds CONTAMINATION_CLASS and the SIM_GREEN<=10% gate",
                 "AVG_NONACTIVE_SHARE": "mean of the 28 team-slot combined shares", "MAX_NONACTIVE_SHARE": "max over all 56 team-pools",
                 "CLASS_EDGES": "CLEAN<=2%, LOW<=10%, MATERIAL<=25%, SEVERE<=50%, CRITICAL>50% (diagnostic only)",
                 "INVALID_PLAYER_USAGE_COUNT": "distinct non-eligible players with mean(pass_attempts+targets+rush_attempts)>0 in the sims; MEANINGFUL subset = share>=1% of pool or |mean yards|>=5"},
             "SLATE": {"AVG_NONACTIVE_SHARE": float(np.mean(team_comb)), "MAX_NONACTIVE_SHARE": float(max(pool_all)),
                       "MEDIAN_TEAM_COMBINED_NONACTIVE_SHARE": float(np.median(team_comb)), "WORST_GAME": worst,
                       "CONTAMINATION_CLASS_COUNTS": cont_counts, "INVALID_PLAYER_USAGE_COUNT_TOTAL": invalid_total,
                       "INVALID_PLAYER_USAGE_MEANINGFUL_TOTAL": invalid_material, "PANEL_OUT_PLAYERS_WITH_SIM_USAGE_TOTAL": panel_out_bad},
             "SEA_ARI_SPECIAL_AUDIT": special,
             "GAMES": [{"GAME_ID": gid, "GAME": f"{per[gid][2]['AWAY']}@{per[gid][2]['HOME']}", "PANEL_CLASS": per[gid][3]["PANEL_CLASS"],
                        **{k: v for k, v in per[gid][3].items() if k in ("MAX_TEAM_NONACTIVE_SHARE", "MAX_TEAM_POOL_DEFINITION", "TOTAL_NONACTIVE_CARRY_SHARE", "TOTAL_NONACTIVE_TARGET_SHARE",
                                                                         "CONTAMINATION_CLASS", "INVALID_PLAYER_USAGE_COUNT", "INVALID_PLAYER_USAGE_MEANINGFUL_COUNT",
                                                                         "INVALID_PLAYER_USAGE_BY_CATEGORY", "INVALID_PLAYERS", "PANEL_OUT_PLAYERS_WITH_SIM_USAGE",
                                                                         "DOUBTFUL_PLAYERS_WITH_SIM_USAGE_INFO_ONLY", "TEAMS", "QB_INTEGRITY")}} for gid in order],
             "QB_INTEGRITY_SUMMARY": {"QB_MATCH_COUNT": qm, "QB_MISMATCH_COUNT": len(qb_rows) - qm, "SLOTS": len(qb_rows), "MATCH_VIA_ENGINE_FALLBACK_NOT_CONFIRMED": qfb}}
    kinds, causes_ = {}, {}
    for f_ in anomalies:
        kinds[f_["KIND"]] = kinds.get(f_["KIND"], 0) + 1
        causes_[f_["CAUSE_CANDIDATE"]] = causes_.get(f_["CAUSE_CANDIDATE"], 0) + 1
    dbl = [dict(GAME=f"{per[gid][2]['AWAY']}@{per[gid][2]['HOME']}", **d) for gid in order for d in per[gid][3]["DOUBTFUL_PLAYERS_WITH_SIM_USAGE_INFO_ONLY"]]
    team_comb_gt50 = [f"{per[gid][2]['AWAY']}@{per[gid][2]['HOME']}:{t}" for gid in order for t in (per[gid][2]["AWAY"], per[gid][2]["HOME"])
                      if per[gid][3]["TEAMS"][t]["NON_ACTIVE_ROSTER_SHARE"] > 0.5]
    mandated = [gid for gid in order if gid in DIAG_LABELS]
    additional_red = [gid for gid in order if gid not in DIAG_LABELS and per[gid][3]["SIM_CLASS"] == "SIM_RED"]
    findings = {
        "MANDATED_DIAGNOSTIC_ONLY": {g_: DIAG_LABELS[g_] for g_ in mandated},
        "ADDITIONAL_SIM_RED_FROM_M1_OUTPUT_EVIDENCE": additional_red,
        "ADDITIONAL_SIM_RED_REASON": "max team-pool non-eligible share >50% (CRITICAL by the mission's own class edges); the panel did not measure non-ACT (non-RES) share, so it scored these games YELLOW/GREEN",
        "ADDITIONAL_SIM_RED_SENSITIVITY": f"if the gate used TEAM-COMBINED share instead of max pool, team slots >50% would be: {team_comb_gt50}",
        "AVAILABILITY_MECHANISM_CONTRAST": {"PANEL_OUT_PLAYERS_WITH_SIM_USAGE": panel_out_bad, "NON_ELIGIBLE_NON_OUT_PLAYERS_WITH_SIM_USAGE": invalid_total,
                                            "MEANINGFUL_SUBSET": invalid_material,
                                            "READING": "availability==OUT works exactly where wired (0 uses of injury-report Out players); the same mechanism is not applied to RES / off-roster / other-team players"},
        "QB_MATCH_DECOMPOSITION": {"MATCH_VIA_INSTALLED_PANEL_IDENTITY": qm - qfb, "MATCH_ONLY_VIA_ENGINE_FALLBACK_NOT_CONFIRMED": qfb, "MISMATCH": len(qb_rows) - qm,
                                   "FALLBACK_DETAIL": [f"{q['TEAM']}: engine picked {q['ENGINE_PRIMARY_QB_NAME']} with realized pass-attempt share {q['REALIZED_SHARE_OF_EXPECTED_QB']} while panel says "
                                                       f"{q['PANEL_STARTER_STATUS']}/{q['PANEL_QB_CONFIDENCE']}" for q in qb_rows if q["MATCH"] and q["MATCH_BASIS"] != "PANEL_IDENTITY_INSTALLED"]},
        "ANOMALY_DECOMPOSITION": {"ALL_ANOMALIES": len(anomalies), "BY_KIND": kinds, "BY_CAUSE_CANDIDATE": causes_,
                                  "NOTE": "NON_ACTIVE_PLAYER_PRODUCTION is one repeated finding (one row per non-eligible player with meaningful simulated production)"},
        "DOUBTFUL_PLAYERS_WITH_FULL_SIM_ALLOCATION": dbl,
        "DOUBTFUL_NOTE": "distinct, narrower defect: Doubtful != OUT under the OUT-only rule, so Doubtful skill players receive full allocation (QB_RULE_V1 already excludes Doubtful QBs)",
        "CARRY_POOL_BIAS_DIRECTION": "carry pool includes QB scrambles (team rush_attempts = designed + scrambles), which enlarges the denominator, so NON_ACTIVE_CARRY_SHARE is biased slightly LOW (conservative)",
        "PANEL_CLASS_INHERITANCE": {"GAMES_WHERE_SIM_CLASS_DEPENDS_ON_PANEL_CLASS": [gid for gid in order if per[gid][3]["SIM_CLASS_DEPENDS_ON_PANEL_CLASS"]],
                                    "RULE": "panel class is read in exactly one place: a RED panel cannot reach SIM_GREEN. Every other SIM_CLASS input is M1-output evidence."},
        "SIM_CLASS_MOVES_VS_PANEL": {f"{per[gid][2]['AWAY']}@{per[gid][2]['HOME']}": f"{per[gid][3]['PANEL_CLASS']}->{per[gid][3]['SIM_CLASS']}" for gid in order}}
    baseline["FINDINGS"] = findings
    audit["FINDINGS"] = findings
    baseline["SLATE"] = {"SIM_CLASS_COUNTS": sc_counts, "BASELINE_PUBLISHED_NON_RED_PANEL_GAMES": len(published), "CLEAN_BASELINES_SIM_GREEN": sc_counts["SIM_GREEN"],
                         "DIAGNOSTIC_ONLY_MANDATED": mandated, "ADDITIONAL_SIM_RED": additional_red, "QB_MATCH_COUNT": qm, "QB_MISMATCH_COUNT": len(qb_rows) - qm, "QB_MATCH_VIA_ENGINE_FALLBACK": qfb,
                         "ANOMALY_COUNT": len(anomalies), "SLATE_MEAN_OF_MEDIAN_TOTALS": tm, "SLATE_MEAN_OF_MEDIAN_TEAM_SCORES": sm}
    for name, obj in (("SUNDAY_2026_09_20_M1_BASELINE_V1.json", baseline), ("SUNDAY_2026_09_20_ROSTER_CONTAMINATION_AUDIT.json", audit)):
        assert_market_free(obj)                                                    # firewall over the artifact we are about to persist
        import re
        bad = re.compile(r"moneyline|sportsbook|kalshi|\bodds\b|betting|\bvegas\b|\bpnl\b|closing line", re.I)
        text = json.dumps(obj, default=str)
        hits = bad.findall(text)
        assert not hits, f"market vocabulary in {name}: {hits[:5]}"
        (OUT_DIR / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str, allow_nan=False), encoding="utf-8")

    # ------------------------------------------------------------------ artifact 3: scorecard.md
    L = []
    w = L.append
    w("# SUNDAY_2026_09_20 M1 BASELINE SCORECARD (V1)\n")
    w(f"- created_at: {now}\n- M1: `{MODEL_VERSION}`  model_hash `{common['M1_MODEL_HASH'][:16]}...`  M1_UNCHANGED={m1_unchanged}\n"
      f"- panel: V2 sha256 `{ctx['raw_sha']}` (verified={common['PANEL_SHA_VERIFIED']}, market fields={man['MARKET_FIELDS_PRESENT']})\n"
      f"- sims/game: {sorted(n_sims)}; seed policy: {SEED_POLICY}\n"
      "- **All FAIR_SPREAD_RAW / FAIR_TOTAL_RAW values are RAW M1-derived research outputs. They are not calibrated and not actionable.** "
      "V23 totals are a known low-running engine characteristic (see memory: terminal note).\n")
    w("## Anomaly board\n")
    w("| GAME | PANEL | SIM | HOME_WIN% | FAIR_SPREAD_RAW (home) | FAIR_TOTAL_RAW | NONACTIVE_SHARE (max pool) | CONTAM | QB_STATUS | WARNINGS |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for gid in order:
        rec, inp, g, a = per[gid]
        qbs = "; ".join(f"{q['TEAM']}:{q['PANEL_STARTER_STATUS']}/{q['PANEL_QB_CONFIDENCE']}{'' if q['ENGINE_IDENTITY_INSTALLED'] else '/FALLBACK'}{'' if q['MATCH'] else '/MISMATCH'}"
                        for q in a["QB_INTEGRITY"])
        wr = "; ".join([a["PUBLICATION_LABEL"]] * (gid in DIAG_LABELS) + [f"{len(a['ANOMALIES'])} anomalies", f"invalid-usage {a['INVALID_PLAYER_USAGE_COUNT']}"]
                       + a["SIM_CLASS_REASONS"][:2])
        w(f"| {g['AWAY']}@{g['HOME']} | {a['PANEL_CLASS']} | {a['SIM_CLASS']} | {g['HOME_WIN_PROB_TIE_SPLIT']:.1%} | {-g['MARGIN_HOME_MINUS_AWAY']['median']:+.1f} | "
          f"{g['TOTAL_POINTS']['median']:.0f} | {a['MAX_TEAM_NONACTIVE_SHARE']:.1%} | {a['CONTAMINATION_CLASS']} | {qbs} | {wr} |")
    w("\n## Per-team roster contamination (simulated share of pool on non-eligible players)\n")
    w("| TEAM | GAME | ACTIVE_ROSTER_SHARE | NON_ACTIVE_ROSTER_SHARE | NON_ACTIVE_CARRY | NON_ACTIVE_TARGET | RES-only carry / target | input-side (panel) carry / target |")
    w("|---|---|---|---|---|---|---|---|")
    for gid in order:
        _, _, g, a = per[gid]
        for t in (g["AWAY"], g["HOME"]):
            T = a["TEAMS"][t]
            w(f"| {t} | {g['AWAY']}@{g['HOME']} | {T['ACTIVE_ROSTER_SHARE']:.1%} | {T['NON_ACTIVE_ROSTER_SHARE']:.1%} | {T['NON_ACTIVE_CARRY_SHARE']:.1%} | {T['NON_ACTIVE_TARGET_SHARE']:.1%} | "
              f"{T['RES_ONLY_CARRY_SHARE']:.1%} / {T['RES_ONLY_TARGET_SHARE']:.1%} | {T['PANEL_INPUT_SIDE_NON_ACTIVE_RAW']['carry']:.1%} / {T['PANEL_INPUT_SIDE_NON_ACTIVE_RAW']['target']:.1%} |")
    w("\n## Per-game contamination\n")
    w("| GAME | MAX_TEAM_NONACTIVE_SHARE | TOTAL_NONACTIVE_CARRY | TOTAL_NONACTIVE_TARGET | CLASS | INVALID_PLAYER_USAGE (any / meaningful) | by category |")
    w("|---|---|---|---|---|---|---|")
    for gid in order:
        _, _, g, a = per[gid]
        w(f"| {g['AWAY']}@{g['HOME']} | {a['MAX_TEAM_NONACTIVE_SHARE']:.1%} | {a['TOTAL_NONACTIVE_CARRY_SHARE']:.1%} | {a['TOTAL_NONACTIVE_TARGET_SHARE']:.1%} | "
          f"{a['CONTAMINATION_CLASS']} | {a['INVALID_PLAYER_USAGE_COUNT']} / {a['INVALID_PLAYER_USAGE_MEANINGFUL_COUNT']} | {a['INVALID_PLAYER_USAGE_BY_CATEGORY']} |")
    w("\n## QB integrity (28 slots)\n")
    w("| TEAM | panel expected | status/conf | identity installed | engine primary | match | realized pass-att share |")
    w("|---|---|---|---|---|---|---|")
    for q in qb_rows:
        w(f"| {q['TEAM']} | {q['PANEL_QB_EXPECTED_NAME']} | {q['PANEL_STARTER_STATUS']}/{q['PANEL_QB_CONFIDENCE']} | {q['ENGINE_IDENTITY_INSTALLED']} | {q['ENGINE_PRIMARY_QB_NAME']} | "
          f"{q['MATCH']}{'' if q['MATCH_BASIS'] in (None, 'PANEL_IDENTITY_INSTALLED') else ' (FALLBACK, not confirmed starter)'} | {q['REALIZED_PASS_ATT_SHARE_BY_QB']} |")
    w(f"\nQB_MATCH_COUNT={qm} QB_MISMATCH_COUNT={len(qb_rows) - qm} (of which matched only via engine fallback: {qfb})\n")
    if special:
        w("## SEA@ARI special audit\n")
        for nm in ("James Conner", "Trey Benson"):
            v = special[nm]
            if isinstance(v, str):
                w(f"- {nm}: {v}")
            else:
                for r in v:
                    w(f"- {nm} ({r['TEAM']}, {r['ROSTER_CATEGORY']}, panel_out={r['PANEL_OUT']}): input carry share raw {r['INPUT_CARRY_SHARE_RAW']:.1%} / effective "
                      f"{r['INPUT_CARRY_SHARE_EFFECTIVE']:.1%}; SIM carry share {r['SIM_CARRY_SHARE']:.1%} ({r['SIM_MEAN_CARRIES']:.1f} carries, {r['SIM_MEAN_RUSH_YDS']:.1f} rush yds, "
                      f"P(TD)={r['SIM_P_ANY_TD']:.1%}); targets {r['SIM_MEAN_TARGETS']:.1f}")
        for t in ("SEA", "ARI"):
            w(f"\n{t} non-active players with allocation (sim): " + "; ".join(
                f"{r['NAME']} [{r['ROSTER_CATEGORY']}] c{r['SIM_CARRY_SHARE']:.1%}/t{r['SIM_TARGET_SHARE']:.1%}" for r in special["ALL_NON_ACTIVE_PLAYERS_WITH_ALLOCATION"][t][:12]))
            w(f"{t} RBs carrying (>2% share, with roster category): " + "; ".join(f"{r['NAME']} [{r['ROSTER_CATEGORY']}] {r['SIM_CARRY_SHARE']:.1%}" for r in special["RBS_CARRYING_WITH_ROSTER_CATEGORY"][t]))
    w("\n## Anomalies (not fixed; cause is a candidate only)\n")
    for f in anomalies:
        w(f"- **{f['GAME']}** {f['KIND']}: {f['DETAIL']}  -> cause candidate `{f['CAUSE_CANDIDATE']}`")
    w(chr(10) + "## Findings" + chr(10))
    for k_, v_ in findings.items():
        w(f"- **{k_}**: {json.dumps(v_, default=str)[:1500]}")
    w("\n## Slate summary\n")
    w(f"- SIM classes: {sc_counts}; contamination classes: {cont_counts}\n- AVG_NONACTIVE_SHARE {np.mean(team_comb):.1%} (28 team slots, combined); MAX pool {max(pool_all):.1%}; worst game {worst}\n"
      f"- INVALID_PLAYER_USAGE_COUNT total {invalid_total} (meaningful {invalid_material}); panel-OUT players used by sims: {panel_out_bad}\n"
      f"- input hash vs panel smoke hash match: {sum(1 for gid in order if per[gid][3]['INPUT_HASH_MATCHES_PANEL_AND_SIM'])}/{len(order)}")
    (OUT_DIR / "SUNDAY_2026_09_20_M1_BASELINE_SCORECARD.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ console summary
    summary = {"M1_UNCHANGED": m1_unchanged, "PANEL_SHA_VERIFIED": common["PANEL_SHA_VERIFIED"], "GAMES_TOTAL": len(order),
               "SIM_CLASS_COUNTS": sc_counts, "CONTAM": cont_counts, "MANDATED_DIAG": mandated, "ADDITIONAL_SIM_RED": additional_red, "QB_MATCH": qm, "QB_MISMATCH": len(qb_rows) - qm, "QB_MATCH_VIA_FALLBACK": qfb,
               "INVALID_USAGE": invalid_total, "INVALID_MEANINGFUL": invalid_material, "PANEL_OUT_USED": panel_out_bad,
               "AVG_NONACTIVE": float(np.mean(team_comb)), "MAX_NONACTIVE": float(max(pool_all)), "WORST": worst, "ANOMALIES": len(anomalies),
               "HASH_MATCH": sum(1 for gid in order if per[gid][3]["INPUT_HASH_MATCHES_PANEL_AND_SIM"])}
    print(json.dumps(summary, indent=1))


def cmd_scan() -> None:
    """Market-free proof over EVERY persisted file in the output dir (json/md/log/npz); written last, so it is not itself scanned."""
    import re
    from worker.sports_nova_sunday_panel.freeze import bundle_scan
    bad = re.compile(r"moneyline|sportsbook|kalshi|odds|betting|vegas|pnl|closing line|implied prob", re.I)
    per_file, vocab = {}, []
    for p in sorted(OUT_DIR.rglob("*")):
        if not p.is_file() or p.name == "MARKET_FREE_SCAN.json":
            continue
        rel = str(p.relative_to(OUT_DIR)).replace("\\", "/")
        if p.suffix == ".npz":
            names = list(np.load(p, allow_pickle=False).files)
            per_file[rel] = {"kind": "npz", "array_names": len(names), "bad_array_names": [n for n in names if bad.search(n)]}
            continue
        hits = bad.findall(p.read_text(encoding="utf-8", errors="replace"))
        per_file[rel] = {"kind": p.suffix, "vocab_hits": len(hits)}
        vocab += [f"{rel}:{h}" for h in hits[:3]]
    bs = bundle_scan(OUT_DIR)
    out = {"SCOPE": "every file under the baseline directory (json/md/log text-scanned for market vocabulary; npz array names checked)",
           "VOCAB_HITS_TOTAL": len(vocab), "VOCAB_HITS": vocab[:20], "PER_FILE": per_file,
           "N2_BUNDLE_SCAN_MARKET_FIELDS_PRESENT": bs["MARKET_FIELDS_PRESENT"], "N2_BUNDLE_SCAN_HITS": bs["HITS"],
           "N2_SCAN_NOTE": "N2 key-token scan flags the substring 'spread' in the mission-mandated key FAIR_SPREAD_RAW (a raw model-margin field, not a market field); any hit should be only that name",
           "INPUT_STATEMENT": "no market source is read anywhere in this pipeline: inputs are panel V2 only (bundle scanned 0 at N2 build)"}
    (OUT_DIR / "MARKET_FREE_SCAN.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("VOCAB_HITS_TOTAL", "N2_BUNDLE_SCAN_MARKET_FIELDS_PRESENT", "N2_BUNDLE_SCAN_HITS")}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "sim":
        cmd_sim(int(sys.argv[2]) if len(sys.argv) > 2 else N_SIMS_DEFAULT)
    elif cmd == "analyze":
        cmd_analyze()
    elif cmd == "scan":
        cmd_scan()
    else:
        raise SystemExit(__doc__)
