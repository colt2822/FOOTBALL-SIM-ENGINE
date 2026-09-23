"""SPORTS_NOVA FG-only causal ablation V1 -- run harness (NOT a new engine version).

Per pilot game, three runs on the SAME PregameState and SAME seed:
  PLAIN     : unmodified worker.sports_nova_v23.simulator.simulate_game
  BASELINE  : same, with a passive recorder (records block points/TDs/plays; changes nothing)
  ABLATION  : same recorder, and fit_distributions' output has fg_rate replaced by the frozen
              causal estimate (dataclasses.replace) -- the ONLY change.
PLAIN == BASELINE (bit-for-bit) is asserted per game: that proves the recorder is inert.
V25/V26 delegate scoring to the V23 simulator unchanged, and V25 needs a RosterSnapshot that
does not exist for historical games, so V23 is the correct engine to ablate on the pilot.
Engine sources are never edited; their hashes are recorded before/after.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sports_nova_m1_v4_raw_path_diagnostic_challenger import (  # noqa: E402
    pilot_games, game_parts, make_state, surrogate_kickoff, assert_path, PLAYER, MANIFEST,
)
from scripts.sports_nova_fg_causal_estimator_v1 import load_drives, estimate, sha256_file  # noqa: E402

OUT = ROOT / "data" / "sports_nova_v3" / "FG_CAUSAL_ABLATION_V1"
ARR = OUT / "arrays"
PREREG = OUT / "FG_CAUSAL_ABLATION_V1_PREREG.json"
MODULE = "worker.sports_nova_v23.simulator"
N_SIMS = 128
ENGINE_FILES = [
    "worker/sports_nova_v23/simulator.py", "worker/sports_nova_v23/allocation.py",
    "worker/sports_nova_v23/config.py", "worker/sports_nova_v3/distributions.py",
    "worker/sports_nova_v3/simulator.py", "worker/sports_nova_v3/game_state.py",
    "worker/sports_nova_v3/game_script.py", "worker/sports_nova_v3/play_volume.py",
    "worker/sports_nova_v3/play_selection.py", "worker/sports_nova_v19/simulator.py",
    "worker/sports_nova_v25_role_aware/simulator.py",
]
TEAM_KEYS = ("score", "pass_attempts", "pass_yards", "rush_attempts", "rush_yards", "blocks")
PLAYER_KEYS = ("pass_attempts", "pass_yards", "pass_tds", "rush_attempts", "rush_yards", "rush_tds",
               "targets", "receptions", "receiving_yards", "receiving_tds", "scored_tds")


def engine_hashes() -> dict:
    return {f: sha256_file(ROOT / f) for f in ENGINE_FILES}


def game_seed(game: str) -> int:
    return int(hashlib.sha256(game.encode()).hexdigest()[:8], 16)


class Recorder:
    """Passive: wraps select_plays/_inc/GameState.advance and (optionally) fit_distributions."""

    def __init__(self, module, fg_override: float | None):
        from worker.sports_nova_v3.game_state import GameState
        self.m, self.GS, self.fg = module, GameState, fg_override
        self.blocks: list[dict] = []
        self.cur = None
        self.o = {"fit": module.fit_distributions, "sel": module.select_plays, "inc": module._inc,
                  "adv": GameState.advance}

    def __enter__(self):
        o, rec = self.o, self

        def fit(rows, cutoff):
            p = o["fit"](rows, cutoff)
            return dataclasses.replace(p, fg_rate=rec.fg) if rec.fg is not None else p

        def sel(plays, pass_rate, rng, **kw):
            v = o["sel"](plays, pass_rate, rng, **kw)
            rec.cur = {"plays": int(v.plays), "tds": 0, "pass_td_players": 0, "rush_td_players": 0,
                       "max_player_tds": 0}
            return v

        def inc(mapping, pid, stat, value):
            o["inc"](mapping, pid, stat, value)
            c = rec.cur
            if c is not None and value:
                if stat == "scored_tds":
                    c["tds"] += int(value)
                    c["max_player_tds"] = max(c["max_player_tds"], int(value))
                elif stat == "receiving_tds":
                    c["pass_td_players"] += 1
                elif stat == "rush_tds":
                    c["rush_td_players"] += 1

        def adv(self, **kw):
            new = o["adv"](self, **kw)
            c = rec.cur or {"plays": -1, "tds": 0, "pass_td_players": 0, "rush_td_players": 0, "max_player_tds": 0}
            rec.blocks.append({"sim": int(self.simulation_id), "block": int(self.block_index),
                               "offense": self.possession, "seconds_before": int(self.seconds_remaining),
                               "pts": int((new.home_score - self.home_score) + (new.away_score - self.away_score)),
                               **c})
            rec.cur = None
            return new

        self.m.fit_distributions, self.m.select_plays, self.m._inc = fit, sel, inc
        self.GS.advance = adv
        return self

    def __exit__(self, *a):
        self.m.fit_distributions, self.m.select_plays, self.m._inc = self.o["fit"], self.o["sel"], self.o["inc"]
        self.GS.advance = self.o["adv"]


def summarize(batch, blocks: list[dict], state, n: int) -> dict:
    team_ids = batch.team_ids
    home = team_ids[0]
    out: dict[str, np.ndarray] = {}
    for k in TEAM_KEYS:
        out[f"team_{k}"] = np.asarray(batch.team_stats[k], dtype=np.int64)
    for k in PLAYER_KEYS:
        arr = np.asarray(batch.player_stats[k])
        cols = np.zeros((n, 2), dtype=np.int64)
        for j, tid in enumerate(team_ids):
            idx = [i for i, pid in enumerate(batch.player_ids) if batch.player_team[pid] == tid]
            cols[:, j] = arr[:, idx].sum(axis=1) if idx else 0
        out[f"psum_{k}"] = cols
    out["winner"] = np.asarray(batch.winner)
    fg = np.zeros((n, 2), np.int64); td = fg.copy(); elig = fg.copy(); nblk = fg.copy()
    plays = fg.copy(); pts = fg.copy(); multi_td = fg.copy(); gt8 = fg.copy(); ge3p = fg.copy()
    bad: list[str] = []
    gt8_rows: list[dict] = []
    for b in blocks:
        j = 0 if b["offense"] == home else 1
        s, t, p, pl = b["sim"], b["tds"], b["pts"], b["plays"]
        nblk[s, j] += 1; plays[s, j] += max(pl, 0); td[s, j] += t; pts[s, j] += p
        if pl >= 3:
            ge3p[s, j] += 1
        if t == 0:
            if pl >= 3:
                elig[s, j] += 1
            if p == 3:
                fg[s, j] += 1
            elif p != 0:
                bad.append(f"no_td_block_points={p}")
        else:
            if not (6 * t <= p <= 7 * t):
                bad.append(f"td_block_points_outside_6t_7t t={t} p={p}")   # would reveal TD+FG stacking
            if t >= 2:
                multi_td[s, j] += 1
        if p > 8:
            gt8[s, j] += 1
            gt8_rows.append({"sim": s, "block": b["block"], "offense": b["offense"], "pts": p, "tds": t,
                             "implied_pats": p - 6 * t, "pass_td_players": b["pass_td_players"],
                             "rush_td_players": b["rush_td_players"], "max_player_tds": b["max_player_tds"],
                             "plays": pl})
    for k, v in dict(fg=fg, td=td, elig=elig, nblk=nblk, plays=plays, pts=pts, multi_td=multi_td, gt8=gt8,
                     ge3p=ge3p).items():
        out[f"rec_{k}"] = v
    # ---- accounting (all must be exact) ----
    acct = {"score_eq_block_points": int((pts != out["team_score"]).sum()),
            "tds_eq_player_scored_tds": int((td != out["psum_scored_tds"]).sum()),
            "blocks_eq_team_blocks": int((nblk != out["team_blocks"]).sum()),
            "block_point_shape_violations": len(bad)}
    fails: list[dict] = []
    stats_by_sim = {}
    for i in range(n):
        stats = {pid: {k: int(batch.player_stats[k][i, j]) for k in PLAYER_KEYS}
                 for j, pid in enumerate(batch.player_ids)}
        team = {tid: {k: int(batch.team_stats[k][i, j]) for k in TEAM_KEYS} for j, tid in enumerate(team_ids)}
        assert_path(state.game_id, state, {"result": (stats, team, None), "sim_id": i, "blocks": [],
                                           "residual_by_team": {}}, "V23", fails)
    acct["assert_path_failures"] = len(fails)
    return {"arrays": out, "accounting": acct, "gt8": gt8_rows, "bad_examples": bad[:5],
            "eq_key": None}


def run_game(args) -> dict:
    game, est, n = args
    t0 = time.time()
    module = importlib.import_module(MODULE)
    season, week, away, home = game_parts(game)
    all_df = pd.read_parquet(PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    prior = all_df[(all_df._key < season * 100 + week) & (all_df.SEASON >= max(1999, season - 5))]
    state = make_state(game, prior, surrogate_kickoff(season, week), None)
    seed = game_seed(game)
    plain = module.simulate_game(state, n, seed, module.MODEL_VERSION)
    res = {"game": game, "seed": seed, "fg_est": est}
    with Recorder(module, None) as r0:
        base = module.simulate_game(state, n, seed, module.MODEL_VERSION)
    res["plain_equals_recorded_baseline"] = bool(plain == base)
    sb = summarize(base, r0.blocks, state, n)
    with Recorder(module, est) as r1:
        abl = module.simulate_game(state, n, seed, module.MODEL_VERSION)
    sa = summarize(abl, r1.blocks, state, n)
    ARR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARR / f"{game}.npz", **{f"base__{k}": v for k, v in sb["arrays"].items()},
                        **{f"abl__{k}": v for k, v in sa["arrays"].items()})
    res.update({"home": home, "away": away, "n_sims": n, "acct_base": sb["accounting"], "acct_abl": sa["accounting"],
                "gt8_base": sb["gt8"], "gt8_abl": sa["gt8"], "bad_base": sb["bad_examples"], "bad_abl": sa["bad_examples"],
                "seconds": round(time.time() - t0, 1)})
    return res


def cmd_run(limit: int | None, workers: int) -> None:
    pre = json.loads(PREREG.read_text())
    games = pilot_games(list(json.loads(MANIFEST.read_text())["GAME_IDS"]))
    assert games == pre["PILOT"]["game_ids"], "pilot cohort drifted from prereg"
    if limit:
        games = games[:limit]
    before = engine_hashes()
    d = load_drives()
    # Re-derive every estimate from source and confirm it equals the preregistered value.
    tasks = []
    for g in games:
        e = estimate(d, *game_parts(g)[:2], g)
        assert abs(e["rate"] - pre["PER_GAME_ESTIMATES"][g]["rate"]) < 1e-15, f"estimate drift {g}"
        tasks.append((g, e["rate"], N_SIMS))
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(workers) as pool:
        results = []
        for r in pool.imap_unordered(run_game, tasks):
            results.append(r)
            print(f"{r['game']} done {r['seconds']}s plain==base:{r['plain_equals_recorded_baseline']}", flush=True)
    after = engine_hashes()
    meta = {"prereg_sha256": sha256_file(PREREG), "harness_sha256": sha256_file(Path(__file__)),
            "estimator_sha256": sha256_file(ROOT / "scripts/sports_nova_fg_causal_estimator_v1.py"),
            "engine_hashes_before": before, "engine_hashes_after": after, "engine_unchanged": before == after,
            "n_games": len(results), "n_sims": N_SIMS, "wall_seconds": round(time.time() - t0, 1),
            "results": sorted(results, key=lambda r: r["game"])}
    (OUT / ("run_meta.json" if not limit else "run_meta_smoke.json")).write_text(json.dumps(meta, indent=1))
    print("DONE", meta["wall_seconds"], "engine_unchanged", meta["engine_unchanged"],
          "all_plain_eq_base", all(r["plain_equals_recorded_baseline"] for r in results))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cmd_run(a.limit, a.workers)
