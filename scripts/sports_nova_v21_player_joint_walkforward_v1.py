"""Event-causal, player/joint-only V21 walk-forward runner.

Same-cohort comparator for scripts/sports_nova_v3_player_joint_walkforward_v1.py.
Reuses that script's make_state() verbatim (identical PregameState construction,
identical hash-gated inputs, identical 1,693-game manifest) and swaps only the
simulation engine: worker.sports_nova_v21.simulator.simulate_game in place of
worker.sports_nova_v3.simulator.simulate_game. This is the necessary same-cohort
V21 baseline for SPORTS_NOVA_M1_V3_PLAYER_JOINT_VALIDATION_BLUE_TEAM_INTEGRATION
STEP_2; it does not modify worker/sports_nova_v21/* or M1.
"""
from __future__ import annotations

import hashlib, json, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worker.sports_nova_v3.schemas import (Evidence, Feature, PlayerState,
    TeamState, OpportunityShares, PregameState, Uncertainty)
from worker.sports_nova_v21.simulator import simulate_game
from worker.sports_nova_v21.config import MODEL_VERSION
from worker.sports_nova_v3.pregame_state import build_pregame_state

DATA = ROOT / "data" / "sports_nova_v3"
INPUT = DATA / "validation_inputs"
PLAYER = INPUT / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
DRIVE = INPUT / "NFL_V3_DRIVE_BLOCK_CANONICAL_V1.parquet"
MANIFEST = DATA / "SPORTS_NOVA_V3_PHASE08_OOS_GAME_MANIFEST_V1.json"
OUT = DATA
N_SIMS = 16  # matches the V3 comparator run exactly, so resolution is comparable
RUN_ID = "V21_5SEASON"
PLAYER_SHA = "cb8e0ee8df6819961d154236dcbefe0d1399d05d50b1bf857a9624df9ab5388c"
DRIVE_SHA = "47b71f2b40866e0d1dcba2191ff0fd14a601777548b357076a1179e57017e50d"
MANIFEST_SHA = "03bdd94b5656b4e6fdde37fd8571d26700eb5139881bb90c33bc991400bc94fa"

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def game_parts(g):
    a = str(g).split("_")
    return int(a[0]), int(a[1]), a[-2], a[-1]  # away, home in canonical IDs

def surrogate_kickoff(season, week):
    return datetime(season, 1, 1, tzinfo=timezone.utc) + timedelta(days=week * 7)

def evidence(event_end, raw_sha):
    t = event_end if event_end.tzinfo else event_end.replace(tzinfo=timezone.utc)
    return Evidence(source_id="NFL_V3_EVENT_CAUSAL_ARCHIVE", raw_sha256=raw_sha,
        event_end=t, available_at=t, retrieved_at=t,
        availability_basis="VERIFIED_ARCHIVE")

def feat(name, value, ev):
    return Feature(name=name, value=None if value is None or not np.isfinite(value) else float(value),
                   unit="model_unit", evidence=ev)

def make_state(g, prior, current_time, positions):
    # Verbatim copy of scripts/sports_nova_v3_player_joint_walkforward_v1.py::make_state.
    # Do not diverge from that function -- same-cohort comparability depends on
    # both runners building the identical PregameState for a given game_id.
    season, week, away, home = game_parts(g)
    cutoff = current_time - timedelta(seconds=1)
    last = cutoff - timedelta(seconds=1)
    ev = evidence(last, PLAYER_SHA)
    players = []
    latest_team = (prior.sort_values(["SEASON", "WEEK"])
                   .drop_duplicates("PLAYER_ID", keep="last")
                   .set_index("PLAYER_ID")["TEAM"].to_dict())
    for tid in (home, away):
        q = prior[(prior.TEAM == tid) &
                  (prior.PLAYER_ID.map(latest_team).fillna("") == tid)]
        if q.empty: continue
        stats = q.groupby(["PLAYER_ID"], as_index=False).agg(
            n=("GAME_ID", "count"), targets=("TARGETS", "sum"), rec=("RECEPTIONS", "sum"),
            carries=("RUSH_ATTEMPTS", "sum"), pass_att=("PASS_ATTEMPTS", "sum"),
            pass_y=("PASS_YARDS", "sum"), rush_y=("RUSH_YARDS", "sum"), rec_y=("RECEIVING_YARDS", "sum"),
            POSITION=("POSITION", lambda x: x.mode().iat[0] if not x.mode().empty else "OTHER"))
        team_game_order = {tuple(x): i for i, x in enumerate(
            sorted(set(map(tuple, q[["SEASON", "WEEK"]].to_numpy()))))}
        n_team_games = len(team_game_order)
        last_idx = (q.assign(_idx=q[["SEASON", "WEEK"]].apply(
            lambda r: team_game_order[(r.SEASON, r.WEEK)], axis=1))
            .groupby("PLAYER_ID")["_idx"].max())
        qb_rows = stats[stats.POSITION == "QB"].copy()
        skill_rows = stats[stats.POSITION != "QB"].sort_values(
            ["targets", "carries", "pass_att"], ascending=False).head(40)
        stats = pd.concat([qb_rows, skill_rows], ignore_index=True)
        for r in stats.itertuples():
            pos = r.POSITION if r.POSITION in ("QB", "RB", "WR", "TE") else "OTHER"
            fs = [feat("catch_rate", (r.rec / r.targets) if r.targets else .64, ev),
                  feat("yards_per_reception", (r.rec_y / r.rec) if r.rec else 10., ev),
                  feat("yards_per_carry", (r.rush_y / r.carries) if r.carries else 4.2, ev)]
            recent_n = r.n
            if pos == "QB":
                recent = q[q.PLAYER_ID == r.PLAYER_ID].sort_values(["SEASON", "WEEK"]).tail(4)
                recent_n = max(1, len(recent))
                recent_rate = float(recent.PASS_ATTEMPTS.sum()) / recent_n
                fs.append(feat("pass_rate", recent_rate, ev))
                gap = n_team_games - 1 - int(last_idx[r.PLAYER_ID])
                fs.append(feat("games_since_last_team_game", float(gap), ev))
            players.append(PlayerState(player_id=str(r.PLAYER_ID), team_id=tid, position=pos,
                availability="UNKNOWN", identity_evidence=ev, features=tuple(fs),
                uncertainty=Uncertainty(effective_sample_size=float(recent_n), personnel_unknown=True)))
    _by_team_game = prior.groupby(["TEAM", "SEASON", "WEEK"], as_index=False).agg(
        pa=("PASS_ATTEMPTS", "sum"), ra=("RUSH_ATTEMPTS", "sum"), py=("PASS_YARDS", "sum"),
        pass_td=("PASS_TD", "sum"), rush_td=("RUSH_TD", "sum"), rec=("RECEPTIONS", "sum"))
    _recent_by_team = (_by_team_game.sort_values(["TEAM", "SEASON", "WEEK"])
                       .groupby("TEAM").tail(8))
    league_plays_pg = (float((_recent_by_team.pa + _recent_by_team.ra).mean())
                       if len(_recent_by_team) else 60.0)
    league_py_pg = (float(_recent_by_team.py.mean()) if len(_recent_by_team) else 220.0)
    _rec_sum = float(_recent_by_team.rec.sum())
    league_pass_td_rate = (float(_recent_by_team.pass_td.sum() / _rec_sum)
                            if _rec_sum > 0 else .045)
    _recent_carries = (prior.groupby(["TEAM", "SEASON", "WEEK"], as_index=False)
                       .agg(ca=("RUSH_ATTEMPTS", "sum"))
                       .sort_values(["TEAM", "SEASON", "WEEK"]).groupby("TEAM").tail(8))
    league_rush_td_rate = (float(_recent_by_team.rush_td.sum() / _recent_carries.ca.sum())
                            if len(_recent_carries) and _recent_carries.ca.sum() > 0 else .029)
    def team_state(tid):
        q = prior[prior.TEAM == tid]
        ids = [p.player_id for p in players if p.team_id == tid]
        def shares(col):
            vals = []
            for pid in ids:
                z = q[q.PLAYER_ID == pid][col].sum()
                vals.append(max(0., float(z)))
            current_roster_mask = q.PLAYER_ID.map(latest_team).fillna("") == tid
            team_total = max(0., float(q.loc[current_roster_mask, col].sum()))
            if team_total <= 0:
                return OpportunityShares(player_ids=(), shares=(), residual_share=1., evidence=ev)
            s = np.asarray(vals) / team_total
            residual = 1.0 - float(s.sum())
            if -1e-9 < residual < 0.0:
                residual = 0.0
            return OpportunityShares(player_ids=tuple(ids), shares=tuple(map(float, s)),
                residual_share=residual, evidence=ev)
        by_game = q.groupby(["SEASON", "WEEK"], as_index=False).agg(
            pa=("PASS_ATTEMPTS", "sum"), ra=("RUSH_ATTEMPTS", "sum"), py=("PASS_YARDS", "sum"),
            pass_td=("PASS_TD", "sum"), rush_td=("RUSH_TD", "sum"), rec=("RECEPTIONS", "sum"))
        recent_games = by_game.sort_values(["SEASON", "WEEK"]).tail(8)
        team_pass_sum = float(recent_games.pa.sum())
        team_rush_sum = float(recent_games.ra.sum())
        team_pass_fraction = (team_pass_sum / (team_pass_sum + team_rush_sum)
                               if (team_pass_sum + team_rush_sum) > 0 else .58)
        team_plays_pg = float((recent_games.pa + recent_games.ra).mean()) if len(recent_games) else None
        pace_seconds = 27.0 * (league_plays_pg / team_plays_pg) if team_plays_pg else 27.0
        team_py_pg = float(recent_games.py.mean()) if len(recent_games) else None
        recent_form = (team_py_pg / league_py_pg - 1.0) if team_py_pg else 0.0
        rec_sum = float(recent_games.rec.sum())
        carry_sum = float(recent_games.ra.sum())
        pass_td_rate = ((float(recent_games.pass_td.sum()) + 50.0 * league_pass_td_rate) /
                        (rec_sum + 50.0))
        rush_td_rate = ((float(recent_games.rush_td.sum()) + 50.0 * league_rush_td_rate) /
                        (carry_sum + 50.0))
        tf = [feat("pass_rate", team_pass_fraction, ev),
              feat("recent_form", recent_form, ev),
              feat("pace_seconds", float(np.clip(pace_seconds, 15.0, 45.0)), ev),
              feat("pass_td_rate", float(np.clip(pass_td_rate, .001, .2)), ev),
              feat("rush_td_rate", float(np.clip(rush_td_rate, .0005, .15)), ev),
              feat("sack_rate", .065, ev), feat("scramble_rate", .07, ev)]
        return TeamState(team_id=tid, identity_evidence=ev, features=tuple(tf),
                         target_shares=shares("TARGETS"), carry_shares=shares("RUSH_ATTEMPTS"))
    assembled = PregameState(game_id=g, kickoff=current_time, as_of=cutoff,
        ruleset_id="NFL_V3_RULES_RESEARCH_V1", identity_registry_sha256="0"*64,
        game_evidence=ev, home=team_state(home), away=team_state(away),
        players=tuple(players), environment=())
    payload = assembled.model_dump()
    return build_pregame_state(game_id=g, as_of=cutoff,
        feature_store={"game": payload}, identity_registry={"registry_sha256": "0"*64})

def main():
    for p, expected in ((PLAYER, PLAYER_SHA), (DRIVE, DRIVE_SHA), (MANIFEST, MANIFEST_SHA)):
        if not p.is_file() or sha(p) != expected: raise SystemExit(f"BLOCKED_HASH:{p}")
    games = json.loads(MANIFEST.read_text())["GAME_IDS"]
    df = pd.read_parquet(PLAYER)
    df = df[df.GAME_ID.isin(games)].copy()
    all_df = pd.read_parquet(PLAYER)
    ck = OUT / f"SPORTS_NOVA_V21_PLAYER_WALKFORWARD_CHECKPOINT_V1_{RUN_ID}.json"
    done = set()
    if ck.exists():
        old = json.loads(ck.read_text())
        if old.get("n_sims") == N_SIMS: done = set(old.get("completed_game_ids", []))
    pred_rows, joint_rows = [], []
    frag = OUT / f"_player_joint_fragments_{RUN_ID}"; frag.mkdir(exist_ok=True)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    game_keys = defaultdict(list)
    for g in games:
        s,w,_,_=game_parts(g); game_keys[s*100+w].append(g)
    for key in sorted(game_keys):
      season_now = key // 100
      prior_snapshot = all_df[(all_df._key < key) &
                              (all_df.SEASON >= max(1999, season_now - 5))]
      for g in game_keys[key]:
        if g in done:
            f = frag / f"{g}.json"
            if f.exists():
                x=json.loads(f.read_text()); pred_rows.extend(x["p"]); joint_rows.extend(x["j"])
            continue
        season, week, away, home = game_parts(g)
        t = surrogate_kickoff(season, week)
        state = make_state(g, prior_snapshot, t, None)
        sim = simulate_game(state, N_SIMS, int(hashlib.sha256(g.encode()).hexdigest()[:8],16), MODEL_VERSION)
        cur = df[df.GAME_ID == g]
        p_rows=[]
        for r in cur.itertuples():
            if str(r.PLAYER_ID) not in sim.player_team: continue
            target_map={"QB":"pass_yards","RB":"rush_yards","WR":"receiving_yards","TE":"receiving_yards"}
            stat=target_map.get(r.POSITION)
            if not stat: continue
            a=np.asarray(sim.player(str(r.PLAYER_ID),stat),dtype=float)
            if len(a)==0: continue
            p_rows.append({"GAME_ID":g,"PLAYER_ID":str(r.PLAYER_ID),"POSITION":r.POSITION,
                "TARGET":stat.upper(),"OBSERVED":float(getattr(r,stat.upper() if stat!='pass_yards' else 'PASS_YARDS')),
                "PRED_MEAN":float(a.mean()),"PRED_SD":float(a.std()),
                "P10":float(np.quantile(a,.1)),"P25":float(np.quantile(a,.25)),"P50":float(np.quantile(a,.5)),
                "P75":float(np.quantile(a,.75)),"P90":float(np.quantile(a,.9)),"P_OVER_P50":float(np.mean(a>np.median(a))),
                "N_SIMS":N_SIMS,"CAUSALITY_MODE":"EVENT_CAUSAL_ONLY"})
        j_rows=[]
        qbs=cur[cur.POSITION=="QB"]; rbs=cur[cur.POSITION=="RB"]
        for qr in qbs.itertuples():
            qid=str(qr.PLAYER_ID)
            if qid not in sim.player_team: continue
            qa=np.asarray(sim.player(qid,"pass_yards"),float); win=np.asarray(sim.winner==("HOME" if home==qr.TEAM else "AWAY"))
            qmed=float(np.median(qa)); qo=float(qr.PASS_YARDS)>qmed
            j_rows.append({"GAME_ID":g,"FAMILY":"QB_PASS_YARDS_PLUS_TEAM_WIN","PLAYER_ID":qid,"N":N_SIMS,
                "V21_JOINT_P":float(np.mean((qa>qmed)&win)),"A_MEDIAN":qmed,"REALIZED":int(qo and bool(cur[cur.TEAM==qr.TEAM]["PASS_YARDS"].iloc[0]>qmed))})
            for wr in cur[cur.POSITION.isin(["WR","TE"])].itertuples():
                wid=str(wr.PLAYER_ID)
                if wid not in sim.player_team: continue
                wa=np.asarray(sim.player(wid,"receiving_yards"),float); wm=float(np.median(wa))
                j_rows.append({"GAME_ID":g,"FAMILY":"QB_PASS_YARDS_PLUS_WR_RECEIVING_YARDS","PLAYER_ID":qid+"|"+wid,"N":N_SIMS,
                    "V21_JOINT_P":float(np.mean((qa>qmed)&(wa>wm))),"A_MEDIAN":qmed,"B_MEDIAN":wm,
                    "REALIZED":int(float(qr.PASS_YARDS)>qmed and float(wr.RECEIVING_YARDS)>wm)})
        for rr in rbs.itertuples():
            rid=str(rr.PLAYER_ID)
            if rid in sim.player_team:
                ra=np.asarray(sim.player(rid,"rush_yards"),float); rm=float(np.median(ra)); win=np.asarray(sim.winner==("HOME" if home==rr.TEAM else "AWAY"))
                j_rows.append({"GAME_ID":g,"FAMILY":"RB_RUSH_YARDS_PLUS_TEAM_WIN","PLAYER_ID":rid,"N":N_SIMS,
                    "V21_JOINT_P":float(np.mean((ra>rm)&win)),"A_MEDIAN":rm,"REALIZED":int(float(rr.RUSH_YARDS)>rm and bool(cur[cur.TEAM==rr.TEAM]["RUSH_YARDS"].sum()>=0))})
        pred_rows.extend(p_rows); joint_rows.extend(j_rows)
        (frag/f"{g}.json").write_text(json.dumps({"p":p_rows,"j":j_rows}, separators=(",",":")))
        done.add(g); ck.write_text(json.dumps({"completed_game_ids":sorted(done),"expected_games":len(games),"n_sims":N_SIMS,"status":"RUNNING"},indent=2))
    pd.DataFrame(pred_rows).to_parquet(OUT/"SPORTS_NOVA_V21_PLAYER_WALKFORWARD_PREDICTIONS_V1.parquet",index=False)
    pd.DataFrame(joint_rows).to_parquet(OUT/"SPORTS_NOVA_V21_JOINT_WALKFORWARD_PREDICTIONS_V1.parquet",index=False)
    ck.write_text(json.dumps({"completed_game_ids":sorted(done),"expected_games":len(games),"n_sims":N_SIMS,"status":"COMPLETE"},indent=2))
    summary={"status":"EVENT_CAUSAL_RESEARCH_ONLY","oos_games":len(games),"simulated_games":len(done),"player_prediction_rows":len(pred_rows),"joint_prediction_rows":len(joint_rows),"n_sims":N_SIMS,"model_version":MODEL_VERSION}
    (OUT/"SPORTS_NOVA_V21_PLAYER_MARGINAL_WALKFORWARD_STATUS_V1.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__ == "__main__": main()
