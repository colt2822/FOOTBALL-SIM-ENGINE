"""SPORTS_NOVA V28 14-game current-slate canary (2026-09-20 panel V2, N>=5000, same seeds as V26/V27) and V27-vs-V28 comparison.

  python scripts/sports_nova_m1_v28_slate_canary.py sim VARIANT [N]        # writes data/sports_nova_v3/sunday_2026_09_20/SUNDAY_2026_09_20_M1_V28_<VARIANT>/
  python scripts/sports_nova_m1_v28_slate_canary.py analyze VARIANT

Inputs come only from the immutable panel V2 (via the V23 baseline runner's load_inputs) and the V26 completion layer; V23-V27 and the V27 canary directory are only READ.
No market price is read; M1 output is sealed (hashed) before anything else can touch it.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sports_nova_m1_v26_active_skill_state as R26  # noqa: E402
import sports_nova_m1_v27_causal_fg as R27  # noqa: E402
import sports_nova_m1_v26_active_skill_state_analysis as R26A  # noqa: E402
import scripts.sports_nova_m1_v28_game_state_scoring as R28  # noqa: E402
from worker.sports_nova_modular.firewall import assert_market_free  # noqa: E402
from worker.sports_nova_v28_game_state_scoring import config as cfg  # noqa: E402
from worker.sports_nova_v28_game_state_scoring import simulator as v28  # noqa: E402

base, A, A25 = R26.base, R26A.A, R26A.A25
DATA = base.DATA
V27_DIR = R27.OUT_DIR
N_DEFAULT = 5000


def out_dir(variant: int) -> Path:
    return DATA / f"SUNDAY_2026_09_20_M1_V28_{variant}"


def ref_for(variant: int):
    if not cfg.VARIANTS[variant][1]:
        return None
    return R28.load_ref(variant, "V28_1")


def sim_worker(args):
    variant, gid, state, resolutions, n, outdir, snap, comp, ref = args
    base.MODEL_VERSION = cfg.VERSIONS[variant]
    base.simulate_game = lambda st, nn, seed, mv: v28.simulate_game(st, nn, seed, mv, roster=snap, completion=comp, ref_quantiles=ref)
    return base.sim_worker((gid, state, resolutions, n, outdir))


def cmd_sim(variant: int, n: int) -> None:
    out = out_dir(variant)
    (out / "audit").mkdir(parents=True, exist_ok=True)
    before = R28.guarded_hashes()
    ctx = base.load_inputs()
    assert ctx["raw_sha"] == base.PANEL_SHA_EXPECTED, "PANEL SHA MISMATCH"
    res = base.all_resolutions(ctx["inputs"])
    comps = R26.completions(ctx)
    ref = ref_for(variant)
    jobs = []
    for rec in ctx["games"]:
        gid, inp = rec["GAME_ID"], ctx["inputs"][rec["GAME_ID"]]
        if inp.state is None:
            print(f"SKIP {gid}: {inp.missing}", flush=True)
            continue
        snap, prior, comp = comps[gid]
        _, repaired, audit = v28.prepare_state(inp.state, snap, prior, known_out=R26.known_out(rec))
        names = ctx["names"]
        (out / "audit" / f"{gid}.json").write_text(json.dumps({
            "GAME_ID": gid, "CHANGED": audit.changed, "COMPLETION_CHANGED": comp.changed, "RAW_STATE_HASH": comp.raw_state_hash, "COMPLETED_STATE_HASH": comp.completed_state_hash,
            "REPAIRED_STATE_HASH": audit.repaired_state_hash, "QB_INPUTS_UNCHANGED": audit.qb_inputs_unchanged, "COMPLETION_TEAMS": comp.teams,
            "COMPLETION_ROWS": [dict(r, NAME=names.get(r["PLAYER_ID"])) for r in comp.rows],
            "EXCLUDED_BY_REASON": {k: [dict(v, NAME=names.get(v["PLAYER_ID"])) for v in vs] for k, vs in audit.excluded().items()},
            "SUMMARIES": audit.summaries, "ROWS": [dict(r, NAME=names.get(r["PLAYER_ID"])) for r in audit.rows]}, indent=1, sort_keys=True), encoding="utf-8")
        jobs.append((variant, gid, inp.state, res, n, out, snap, comp, ref))
    base.set_identity_resolutions(res)
    gid0 = "2026_02_IND_KC"
    snap0, _, comp0 = comps[gid0]
    fn = lambda: v28.simulate_game(ctx["inputs"][gid0].state, 40, base.seed_for(gid0), cfg.VERSIONS[variant], roster=snap0, completion=comp0, ref_quantiles=ref)
    x, y = fn(), fn()
    det = all(np.array_equal(x.player_stats[s], y.player_stats[s]) for s in base.STAT_NAMES) and np.array_equal(x.team_stats["score"], y.team_stats["score"])
    print(f"[v28.{variant}] determinism smoke (N=40 x2, IND_KC): {det}", flush=True)
    meta = {"ENGINE": cfg.VERSIONS[variant], "STARTED_AT": datetime.now(timezone.utc).isoformat(), "N_SIMS": n, "V28_HASH": R28.v28_hash(), "DETERMINISM_SMOKE": det,
            "GUARDED_BEFORE": before, "PANEL_SHA256": ctx["raw_sha"], "SEED_POLICY": base.SEED_POLICY, "REFERENCE": None if ref is None else "V28_YARD_REFERENCE.json (POST_HOC)"}
    (out / "sim_run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(sim_worker, j): j[1] for j in jobs}
        for f in as_completed(futs):
            gid, el = f.result()
            print(f"[v28.{variant}] done {gid} {el:.0f}s wall={time.time() - t0:.0f}s", flush=True)
    after = R28.guarded_hashes()
    meta.update({"FINISHED_AT": datetime.now(timezone.utc).isoformat(), "GUARDED_UNCHANGED_DURING_SIM": before == after, "WALL_SEC": time.time() - t0})
    (out / "sim_run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("SIM COMPLETE; guarded files unchanged during run:", before == after, flush=True)


def raw(d: Path, gid: str):
    return np.load(d / "raw" / f"{gid}.npz", allow_pickle=False)


def cmd_analyze(variant: int) -> dict:
    out = out_dir(variant)
    meta = json.loads((out / "sim_run_meta.json").read_text())
    ctx = base.load_inputs()
    comps = R26.completions(ctx)
    ctx26 = R26A.make_ctx26(ctx, comps)
    gids = [r["GAME_ID"] for r in ctx["games"]]
    per28, per27 = R26A.load_dir26(out, ctx, ctx26), R26A.load_dir26(V27_DIR, ctx, ctx26)
    ag28, ag27 = A.aggregate(per28), A.aggregate(per27)
    acc = A.accounting(out, ctx26)
    audits = A25.audits_of(out)
    bad = [s for au in audits.values() for s in au["SUMMARIES"] if abs(s["POST_REDISTRIBUTION_TOTAL"] - s["PRE_REMOVAL_TOTAL"]) > 1e-9 or abs(s["REMOVED_MASS"] - s["REDISTRIBUTED_MASS"]) > 1e-9
           or s["MASS_TO_HELD_ROLES"] > 1e-9 or abs(s["SUM_SHARE_PLUS_RESIDUAL_AFTER"] - 1.0) > 1e-6]
    audit_equal = {g: (out / "audit" / f"{g}.json").read_bytes() == (V27_DIR / "audit" / f"{g}.json").read_bytes() for g in gids}
    rows, score_acct_bad = [], 0
    for g in gids:
        z8, z7 = raw(out, g), raw(V27_DIR, g)
        n = z8["team_score"].shape[0]
        reg8 = z8["team_reg_score"]
        fin8 = z8["team_score"]
        pat = fin8 - 6 * z8["team_tds"] - 3 * z8["team_fgs"]
        score_acct_bad += int(((pat < 0) | (pat > z8["team_tds"])).sum())
        score_acct_bad += int((z8["team_tds"] != z8["team_td_pass"] + z8["team_td_rush"]).sum())
        score_acct_bad += int((z8["team_tds"] != z8["player_scored_tds"].sum(axis=1, keepdims=True)).sum() if False else 0)
        ids = [str(x) for x in z8["player_ids"]]
        team_of = {p.player_id: p.team_id for p in ctx26["inputs"][g].state.players}
        home = str(z8["team_ids"][0])
        for j, t in enumerate(z8["team_ids"]):
            idx = [i for i, pid in enumerate(ids) if team_of.get(pid) == str(t)]
            score_acct_bad += int((z8["player_scored_tds"][:, idx].sum(axis=1) != z8["team_tds"][:, j]).sum())
            score_acct_bad += int((z8["player_receiving_tds"][:, idx].sum(axis=1) != z8["team_td_pass"][:, j]).sum())
            score_acct_bad += int((z8["player_rush_tds"][:, idx].sum(axis=1) != z8["team_td_rush"][:, j]).sum())
        stat = lambda z, k: z[k]
        row = {"GAME_ID": g, "N": n}
        for tag, z in (("V27", z7), ("V28", z8)):
            fin = z["team_score"]
            h, a_ = fin[:, 0], fin[:, 1]
            ph, pa, pt = float((h > a_).mean()), float((a_ > h).mean()), float((h == a_).mean())
            row[tag] = {"P_HOME": ph, "P_AWAY": pa, "TIE": pt, "HOME_WIN_PROB_TIE_SPLIT": ph + .5 * pt, "TOTAL": float((h + a_).mean()), "HOME_SCORE": float(h.mean()), "AWAY_SCORE": float(a_.mean()),
                        "P_TOTAL_GE_55": float(((h + a_) >= 55).mean()), "P_TOTAL_LE_30": float(((h + a_) <= 30).mean()), "P_TEAM_GE_35": float((fin >= 35).mean()), "TOTAL_SD": float((h + a_).std(ddof=1))}
        row["V28"]["REG_TIE"] = float((reg8[:, 0] == reg8[:, 1]).mean())
        row["V28"]["OVERTIME_RATE"] = float(z8["team_went_ot"][:, 0].mean())
        row["V28"]["TD_PER_TEAM_GAME"] = float(z8["team_tds"].mean())
        row["V28"]["FG_PER_TEAM_GAME"] = float(z8["team_fgs"].mean())
        row["V28"]["QB_RUSH_TD_PER_TEAM_GAME"] = float(z8["team_td_qb_rush"].mean())
        row["V28"]["TD_UNATTRIBUTABLE"] = int(z8["team_td_unattributable"].sum())
        row["V28"]["TOTAL_REG"] = float(reg8.sum(axis=1).mean())
        row["V28"]["P_HOME_REG_TIE_SPLIT"] = float((reg8[:, 0] > reg8[:, 1]).mean() + .5 * (reg8[:, 0] == reg8[:, 1]).mean())
        p7, p8 = row["V27"]["HOME_WIN_PROB_TIE_SPLIT"], row["V28"]["HOME_WIN_PROB_TIE_SPLIT"]
        se = float(np.sqrt(p7 * (1 - p7) / n + p8 * (1 - p8) / n))
        t7, t8 = z7["team_score"].sum(axis=1), fin8.sum(axis=1)
        row["DELTA"] = {"HOME_WIN_PROB": p8 - p7, "HOME_WIN_PROB_UNPAIRED_SE": se, "TOTAL": float(t8.mean() - t7.mean()), "TOTAL_SE": float(np.sqrt(t7.var(ddof=1) / n + t8.var(ddof=1) / n)),
                        "HOME_SCORE": row["V28"]["HOME_SCORE"] - row["V27"]["HOME_SCORE"], "AWAY_SCORE": row["V28"]["AWAY_SCORE"] - row["V27"]["AWAY_SCORE"], "TIE": row["V28"]["TIE"] - row["V27"]["TIE"],
                        "FAVORITE_FLIP": bool((p7 - .5) * (p8 - .5) < 0)}
        rows.append(row)
    mean = lambda tag, k: float(np.mean([r[tag][k] for r in rows]))
    slate = {k: {"V27": mean("V27", k), "V28": mean("V28", k)} for k in ("TOTAL", "P_TOTAL_GE_55", "P_TOTAL_LE_30", "P_TEAM_GE_35", "TIE", "TOTAL_SD", "HOME_WIN_PROB_TIE_SPLIT")}
    slate["MEAN_POINTS_PER_TEAM"] = {"V27": float(np.mean([(r["V27"]["HOME_SCORE"] + r["V27"]["AWAY_SCORE"]) / 2 for r in rows])), "V28": float(np.mean([(r["V28"]["HOME_SCORE"] + r["V28"]["AWAY_SCORE"]) / 2 for r in rows]))}
    slate["V28_ONLY"] = {k: mean("V28", k) for k in ("OVERTIME_RATE", "REG_TIE", "TD_PER_TEAM_GAME", "FG_PER_TEAM_GAME", "QB_RUSH_TD_PER_TEAM_GAME", "TOTAL_REG")}
    dp = np.array([r["DELTA"]["HOME_WIN_PROB"] for r in rows])
    se_ = np.array([r["DELTA"]["HOME_WIN_PROB_UNPAIRED_SE"] for r in rows])
    winner = {"MEAN_ABS_DELTA_HOME_WIN_PROB": float(np.abs(dp).mean()), "EXPECTED_MEAN_ABS_FROM_UNPAIRED_MC_NOISE": float(np.mean(np.sqrt(2 / np.pi) * se_)), "MAX_ABS_DELTA": float(np.abs(dp).max()),
              "MEAN_DELTA": float(dp.mean()), "FAVORITE_FLIPS": int(sum(r["DELTA"]["FAVORITE_FLIP"] for r in rows)), "GAMES_ABS_Z_GT_3": int((np.abs(dp) / se_ > 3).sum())}
    team_totals = {r["GAME_ID"]: {"V27": [r["V27"]["HOME_SCORE"], r["V27"]["AWAY_SCORE"]], "V28": [r["V28"]["HOME_SCORE"], r["V28"]["AWAY_SCORE"]]} for r in rows}
    opp = []
    for g in gids:
        z7, z8 = raw(V27_DIR, g), raw(out, g)
        for key in ("team_pass_attempts", "team_rush_attempts"):
            for j in (0, 1):
                a0, a1 = z7[key][:, j].astype(float), z8[key][:, j].astype(float)
                s = float(np.sqrt(a0.var(ddof=1) / a0.size + a1.var(ddof=1) / a1.size))
                opp.append({"GAME": g, "STAT": key, "REL_CHANGE": float(a1.mean() / a0.mean() - 1), "Z": float((a1.mean() - a0.mean()) / s)})
    forb = ("sports_market_interface", "sports_research", "market_book", "comparator", "trading_lab", "kalshi", "sportsbook")
    market_ok = True
    try:
        assert_market_free(cfg.HAZARD_POLICY)
        assert_market_free(cfg.TD_ALLOCATION_POLICY)
        assert_market_free(cfg.OVERTIME_POLICY)
    except Exception:  # noqa: BLE001
        market_ok = False
    checks = {"DETERMINISM_SMOKE": meta["DETERMINISM_SMOKE"], "AUDITS_BYTE_IDENTICAL_TO_V27": all(audit_equal.values()), "INVALID_USAGE_ZERO": ag28["INVALID_PLAYER_USAGE"] == 0,
              "MEANINGFUL_INVALID_USAGE_ZERO": ag28["MEANINGFUL_INVALID_USAGE"] == 0, "ACCOUNTING_TEAM_PLAYER": acc["STATUS"] == "PASS", "TD_SCORE_ACCOUNTING_BAD": score_acct_bad,
              "CONSERVATION": not bad, "MARKET_ISOLATION": market_ok, "GUARDED_UNCHANGED": meta["GUARDED_UNCHANGED_DURING_SIM"], "QB_MISMATCH_ZERO": ag28["QB_MISMATCH"] == 0}
    report = {"SCHEMA": "SPORTS_NOVA_V28_SLATE_CANARY", "CREATED_AT": datetime.now(timezone.utc).isoformat(), "MODEL_VERSION": cfg.VERSIONS[variant], "V28_HASH": R28.v28_hash(), "N_SIMS": sorted({r["N"] for r in rows}),
              "SLATE": slate, "WINNER_MOVEMENT_VS_V27": winner, "PER_GAME": rows, "TEAM_TOTALS": team_totals,
              "OPPORTUNITY_TEAM_LEVEL": {"MAX_ABS_REL_CHANGE": float(max(abs(o["REL_CHANGE"]) for o in opp)), "MAX_ABS_Z": float(max(abs(o["Z"]) for o in opp)), "N": len(opp),
                                         "N_ABS_Z_GT_3": int(sum(abs(o["Z"]) > 3 for o in opp))},
              "AGGREGATE": {"V27": ag27, "V28": ag28}, "ACCOUNTING": {"STATUS": acc["STATUS"], "CHECKS": acc["CHECKS"], "FAILURES": acc["FAILURES"]}, "CHECKS": checks,
              "ALL_CHECKS_PASS": all(v is True or v == 0 for v in checks.values()), "STATUS_NOTE": "UNCALIBRATED_RESEARCH_ONLY; no market price read"}
    (out / "SPORTS_NOVA_V28_CANARY_REPORT.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8")
    keep = {k: report[k] for k in ("SLATE", "WINNER_MOVEMENT_VS_V27", "OPPORTUNITY_TEAM_LEVEL", "CHECKS", "ALL_CHECKS_PASS")}
    keep["INVALID"] = [ag28["INVALID_PLAYER_USAGE"], ag28["MEANINGFUL_INVALID_USAGE"]]
    print(json.dumps(keep, indent=1, default=str))
    return report


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "sim":
        cmd_sim(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else N_DEFAULT)
    elif cmd == "analyze":
        cmd_analyze(int(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
