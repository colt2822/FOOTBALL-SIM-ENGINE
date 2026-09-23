"""Event-causal, player/joint-only V3 walk-forward runner.

This is intentionally additive.  It reuses the V3 PregameState schemas,
allocator, simulator and validation metrics.  The supplied historical sources
do not expose row-level publication times, so this runner reports
EVENT_CAUSAL_ONLY and never claims strict publication causality.
"""
from __future__ import annotations

import hashlib, json, math, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worker.sports_nova_v3.schemas import (Evidence, Feature, PlayerState,
    TeamState, OpportunityShares, PregameState, Uncertainty)
from worker.sports_nova_v3.simulator import simulate_game
from worker.sports_nova_v3.config import MODEL_VERSION
from worker.sports_nova_v3.pregame_state import build_pregame_state

DATA = ROOT / "data" / "sports_nova_v3"
INPUT = DATA / "validation_inputs"
PLAYER = INPUT / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
DRIVE = INPUT / "NFL_V3_DRIVE_BLOCK_CANONICAL_V1.parquet"
MANIFEST = DATA / "SPORTS_NOVA_V3_PHASE08_OOS_GAME_MANIFEST_V1.json"
OUT = DATA
N_SIMS = 16  # fixed reduced-resolution diagnostic; documented in outputs
RUN_ID = "V2_5SEASON"
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
    # Conservative chronology: no within-week borrowing, so a game never
    # receives information from another game in the same week.
    return datetime(season, 1, 1, tzinfo=timezone.utc) + timedelta(days=week * 7)

def evidence(event_end, raw_sha):
    t = event_end if event_end.tzinfo else event_end.replace(tzinfo=timezone.utc)
    return Evidence(source_id="NFL_V3_EVENT_CAUSAL_ARCHIVE", raw_sha256=raw_sha,
        event_end=t, available_at=t, retrieved_at=t,
        availability_basis="VERIFIED_ARCHIVE")

def feat(name, value, ev):
    return Feature(name=name, value=None if value is None or not np.isfinite(value) else float(value),
                   unit="model_unit", evidence=ev)

def mean(x, default):
    a = pd.to_numeric(x, errors="coerce").dropna()
    return float(a.mean()) if len(a) else float(default)

def make_state(g, prior, current_time, positions):
    season, week, away, home = game_parts(g)
    cutoff = current_time - timedelta(seconds=1)
    # Caller supplies a snapshot containing only completed prior weeks.
    last = cutoff - timedelta(seconds=1)
    ev = evidence(last, PLAYER_SHA)
    players = []
    # Use only players with prior history on one of the two teams.  This avoids
    # using current-game active rows to manufacture a pregame roster.
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
        # QBs rarely carry targets/carries, so a pure volume-sort truncation
        # silently evicts them once a team has 40+ historical skill-position
        # rows in the window.  Keep every QB with prior-team history and only
        # truncate the skill-position pool.  But "prior-team history" alone is
        # too permissive for QBs specifically: the lookback spans up to 5
        # seasons, so anyone who was ever part of a QB rotation -- including a
        # committee member who lost the job mid-LAST-season -- still qualifies
        # and dilutes the share long after they stopped playing.  Require a
        # Recency of a QB's most recent game for this team is exposed to the
        # engine as a "games_since_last_team_game" feature (0 = played in the
        # team's most recent prior game) rather than enforced here as a hard
        # cutoff. A hard games-back cutoff either drops a genuine current
        # starter who is new to the team (no history yet) or lets a stale
        # committee member back in right at the boundary; the engine's QB
        # share model (simulator._qb_shares) instead lets whichever QB(s)
        # have the smallest gap compete for the start, so no QB with any
        # prior-team history is excluded here.
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
                # Recency-weighted usage: an up-to-5-season SUM lets a long-tenured,
                # now-benched QB outrank the actual recent/current starter.  Use
                # only the player's most recent games on this team.  The engine's
                # QB share model weights pass_rate by effective_sample_size (games
                # actually observed in that recent window), so a one-game spot
                # starter does not get the same share as an 8-game full-time
                # starter just because their per-game rate happens to match.
                recent = q[q.PLAYER_ID == r.PLAYER_ID].sort_values(["SEASON", "WEEK"]).tail(4)
                recent_n = max(1, len(recent))
                recent_rate = float(recent.PASS_ATTEMPTS.sum()) / recent_n
                fs.append(feat("pass_rate", recent_rate, ev))
                gap = n_team_games - 1 - int(last_idx[r.PLAYER_ID])
                fs.append(feat("games_since_last_team_game", float(gap), ev))
            players.append(PlayerState(player_id=str(r.PLAYER_ID), team_id=tid, position=pos,
                availability="UNKNOWN", identity_evidence=ev, features=tuple(fs),
                uncertainty=Uncertainty(effective_sample_size=float(recent_n), personnel_unknown=True)))
    # League-wide recent play-volume baseline (last 8 team-games per team,
    # same lookback window as everything else in `prior`, so this stays
    # strictly pregame/causal). Used only to express each team's own recent
    # pace relative to the field, on top of the same 27.0 anchor
    # draw_environment already clips against -- not a new fitted constant.
    _by_team_game = prior.groupby(["TEAM", "SEASON", "WEEK"], as_index=False).agg(
        pa=("PASS_ATTEMPTS", "sum"), ra=("RUSH_ATTEMPTS", "sum"), py=("PASS_YARDS", "sum"),
        pass_td=("PASS_TD", "sum"), rush_td=("RUSH_TD", "sum"), rec=("RECEPTIONS", "sum"))
    _recent_by_team = (_by_team_game.sort_values(["TEAM", "SEASON", "WEEK"])
                       .groupby("TEAM").tail(8))
    league_plays_pg = (float((_recent_by_team.pa + _recent_by_team.ra).mean())
                       if len(_recent_by_team) else 60.0)
    # League-wide recent team pass-offense baseline (same last-8-team-games
    # pool as league_plays_pg above), used only to express each team's own
    # recent pass-yardage output relative to the field. SN3_QB_ATTEMPT_VOLUME_V5
    # found recent_form collapsed near-zero for every team because it averaged
    # PASS_YARDS across every player-ROW on a team (RBs/WRs/TEs included, whose
    # PASS_YARDS is always 0) instead of aggregating to a team-GAME total
    # first. Fixed here by summing to team-game totals (_by_team_game.py above)
    # before averaging, then expressing the team's recent total relative to
    # the league's recent total the same way pace_seconds already does for
    # play volume, so a below-average and above-average offense are no longer
    # numerically indistinguishable.
    league_py_pg = (float(_recent_by_team.py.mean()) if len(_recent_by_team) else 220.0)
    # League-wide recent scoring-propensity baseline (per reception / per
    # carry), pooled by volume (sum of TDs / sum of opportunities across all
    # teams' recent-8-game windows) rather than an unweighted mean of ratios,
    # so a team with few recent opportunities does not get equal weight to one
    # with many. Used as the shrinkage prior mean for each team's own
    # pass_td_rate/rush_td_rate below (SN3_SPORTS_V6 Q3).
    _rec_sum = float(_recent_by_team.rec.sum())
    league_pass_td_rate = (float(_recent_by_team.pass_td.sum() / _rec_sum)
                            if _rec_sum > 0 else .045)
    _recent_carries = (prior.groupby(["TEAM", "SEASON", "WEEK"], as_index=False)
                       .agg(ca=("RUSH_ATTEMPTS", "sum"))
                       .sort_values(["TEAM", "SEASON", "WEEK"]).groupby("TEAM").tail(8))
    league_rush_td_rate = (float(_recent_by_team.rush_td.sum() / _recent_carries.ca.sum())
                            if len(_recent_carries) and _recent_carries.ca.sum() > 0 else .029)
    # deterministic, coherent shares from prior opportunity totals
    def team_state(tid):
        q = prior[prior.TEAM == tid]
        ids = [p.player_id for p in players if p.team_id == tid]
        def shares(col):
            vals = []
            for pid in ids:
                z = q[q.PLAYER_ID == pid][col].sum()
                vals.append(max(0., float(z)))
            # Denominator MUST be scoped to the current roster (the same
            # latest_team==tid population `ids` is drawn from before the
            # top-40 truncation), not every row where TEAM==tid across the
            # whole up-to-5-season lookback. SPORTS_NOVA_V17_RESIDUAL_
            # SEMANTICS_AUDIT proved that using the unscoped q here (an
            # earlier version of this fix) made residual_share 99.97%
            # DEPARTED-player historical volume (players who played for tid
            # in an earlier season but are on a different team now, per
            # latest_team) -- volume that cannot be targeted in this game
            # under any semantics -- with legitimate current-roster overflow
            # beyond top-40 accounting for essentially none of it (~0.01%).
            current_roster_mask = q.PLAYER_ID.map(latest_team).fillna("") == tid
            team_total = max(0., float(q.loc[current_roster_mask, col].sum()))
            if team_total <= 0:
                return OpportunityShares(player_ids=(), shares=(), residual_share=1., evidence=ev)
            # Empirical residual mass: the real share of the team's CURRENT
            # -roster prior volume held by players outside the truncated
            # top-40 modeled pool, measured directly from the same
            # prior-window data (q) used for every other feature in this
            # file. Replaces the earlier flat `s *= .9` fixture (SPORTS_
            # NOVA_V16 completion_rate_root_cause), which forced exactly 10%
            # residual share on every team-game regardless of how much real
            # current-roster volume the excluded players actually carried.
            s = np.asarray(vals) / team_total
            residual = 1.0 - float(s.sum())
            if -1e-9 < residual < 0.0:
                residual = 0.0
            return OpportunityShares(player_ids=tuple(ids), shares=tuple(map(float, s)),
                residual_share=residual, evidence=ev)
        # Team pass_rate/pace were both averaged over the WHOLE up-to-5-season
        # lookback, which can dilute a real, current scheme/personnel identity
        # the same way the QB share model's cumulative-volume bug diluted the
        # current starter before the recency fix (SN3_QB_ID_ALLOC_V3). Use the
        # team's most recent 8 games (roughly half a season) instead.
        by_game = q.groupby(["SEASON", "WEEK"], as_index=False).agg(
            pa=("PASS_ATTEMPTS", "sum"), ra=("RUSH_ATTEMPTS", "sum"), py=("PASS_YARDS", "sum"),
            pass_td=("PASS_TD", "sum"), rush_td=("RUSH_TD", "sum"), rec=("RECEPTIONS", "sum"))
        recent_games = by_game.sort_values(["SEASON", "WEEK"]).tail(8)
        # Team pass_rate is consumed downstream as a 0-1 fraction of plays
        # (draw_environment clips it into [.25,.8]).  Use team-level
        # PASS/(PASS+RUSH) over the recent window, not a per-row ratio (a
        # per-row ratio saturates every team at the clip ceiling, and QB rows
        # alone push it well above 1.0).
        team_pass_sum = float(recent_games.pa.sum())
        team_rush_sum = float(recent_games.ra.sum())
        team_pass_fraction = (team_pass_sum / (team_pass_sum + team_rush_sum)
                               if (team_pass_sum + team_rush_sum) > 0 else .58)
        team_plays_pg = float((recent_games.pa + recent_games.ra).mean()) if len(recent_games) else None
        # pace_seconds feeds draw_block_volume as (60/pace_seconds)*2.7
        # expected plays per block; it was a flat 27.0 for every team, every
        # game, ever, discarding real (if modest) team-to-team play-volume
        # variation already present in this same play-by-play data. Express
        # it as the existing 27.0 anchor scaled by this team's recent
        # plays/game relative to the league's recent plays/game, rather than
        # inventing a new formula or constant.
        pace_seconds = 27.0 * (league_plays_pg / team_plays_pg) if team_plays_pg else 27.0
        # recent_form (SN3_SPORTS_V6 Q1/Q2 fix): team-game total pass yards
        # over the same recent-8-game window as pass_rate/pace above,
        # expressed relative to the league's recent total so the value is
        # zero-centered across teams (an average offense reads ~0.0) and its
        # spread reflects genuine cross-team separation instead of the
        # near-constant ~0.05-0.15 the old per-row-averaged definition
        # produced for literally every team (see RECENT_FORM_ROOT_CAUSE).
        team_py_pg = float(recent_games.py.mean()) if len(recent_games) else None
        recent_form = (team_py_pg / league_py_pg - 1.0) if team_py_pg else 0.0
        # Team-specific scoring propensity (SN3_SPORTS_V6 Q3): recent
        # TD-per-opportunity rates replacing the single global td_rate=.045
        # constant every team/game previously shared (distributions.py's
        # fit_distributions is always called with an empty row list by
        # simulate_game, so that constant carried zero team-quality signal).
        # Computed from the same recent-8-game window, shrunk toward the
        # league's volume-weighted recent rate with a fixed pseudo-count of
        # 50 opportunities (matching the existing catch_rate shrinkage
        # convention in distributions.fit_distributions) so a team with few
        # recent opportunities does not swing to an extreme rate.
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
    # Build the same state this harness always built, then round-trip it
    # through the real production entry point (worker/sports_nova_v3/
    # pregame_state.py::build_pregame_state) via its existing Mapping-based
    # feature_store interface (the same interface production code defines in
    # resolve_game_roster -- not a test-only or invented path) instead of
    # handing simulate_game a PregameState this harness constructed itself
    # and the production adapter never saw. This exercises
    # build_pregame_state's fail-closed validation (market-field rejection,
    # evidence-recency, identity-hash check) against real historical data for
    # the first time -- previously only unit-tested against a synthetic
    # fixture (tests/test_pregame_state.py).
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
    df["_time"] = [surrogate_kickoff(int(s), int(w)) for s,w in zip(df.SEASON, df.WEEK)]
    all_df = pd.read_parquet(PLAYER)
    all_df["_time"] = [surrogate_kickoff(int(s), int(w)) for s,w in zip(all_df.SEASON, all_df.WEEK)]
    ck = OUT / f"SPORTS_NOVA_V3_PLAYER_WALKFORWARD_CHECKPOINT_V1_{RUN_ID}.json"
    done = set()
    if ck.exists():
        old = json.loads(ck.read_text())
        if old.get("n_sims") == N_SIMS: done = set(old.get("completed_game_ids", []))
    pred_rows, joint_rows = [], []
    # Reuse fragment outputs to make restart safe.
    frag = OUT / f"_player_joint_fragments_{RUN_ID}"; frag.mkdir(exist_ok=True)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    game_keys = defaultdict(list)
    for g in games:
        s,w,_,_=game_parts(g); game_keys[s*100+w].append(g)
    for key in sorted(game_keys):
      season_now = key // 100
      # Fixed, explicit recency regime for tractable replay.  It is a
      # diagnostic configuration, not a global model retune.
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
                "V3_JOINT_P":float(np.mean((qa>qmed)&win)),"A_MEDIAN":qmed,"REALIZED":int(qo and bool(cur[cur.TEAM==qr.TEAM]["PASS_YARDS"].iloc[0]>qmed))})
            for wr in cur[cur.POSITION.isin(["WR","TE"])].itertuples():
                wid=str(wr.PLAYER_ID)
                if wid not in sim.player_team: continue
                wa=np.asarray(sim.player(wid,"receiving_yards"),float); wm=float(np.median(wa))
                j_rows.append({"GAME_ID":g,"FAMILY":"QB_PASS_YARDS_PLUS_WR_RECEIVING_YARDS","PLAYER_ID":qid+"|"+wid,"N":N_SIMS,
                    "V3_JOINT_P":float(np.mean((qa>qmed)&(wa>wm))),"A_MEDIAN":qmed,"B_MEDIAN":wm,
                    "REALIZED":int(float(qr.PASS_YARDS)>qmed and float(wr.RECEIVING_YARDS)>wm)})
        for rr in rbs.itertuples():
            rid=str(rr.PLAYER_ID)
            if rid in sim.player_team:
                ra=np.asarray(sim.player(rid,"rush_yards"),float); rm=float(np.median(ra)); win=np.asarray(sim.winner==("HOME" if home==rr.TEAM else "AWAY"))
                j_rows.append({"GAME_ID":g,"FAMILY":"RB_RUSH_YARDS_PLUS_TEAM_WIN","PLAYER_ID":rid,"N":N_SIMS,
                    "V3_JOINT_P":float(np.mean((ra>rm)&win)),"A_MEDIAN":rm,"REALIZED":int(float(rr.RUSH_YARDS)>rm and bool(cur[cur.TEAM==rr.TEAM]["RUSH_YARDS"].sum()>=0))})
        pred_rows.extend(p_rows); joint_rows.extend(j_rows)
        (frag/f"{g}.json").write_text(json.dumps({"p":p_rows,"j":j_rows}, separators=(",",":")))
        done.add(g); ck.write_text(json.dumps({"completed_game_ids":sorted(done),"expected_games":len(games),"n_sims":N_SIMS,"status":"RUNNING"},indent=2))
    pd.DataFrame(pred_rows).to_parquet(OUT/"SPORTS_NOVA_V3_PLAYER_WALKFORWARD_PREDICTIONS_V1.parquet",index=False)
    pd.DataFrame(joint_rows).to_parquet(OUT/"SPORTS_NOVA_V3_JOINT_WALKFORWARD_PREDICTIONS_V1.parquet",index=False)
    ck.write_text(json.dumps({"completed_game_ids":sorted(done),"expected_games":len(games),"n_sims":N_SIMS,"status":"COMPLETE"},indent=2))
    summary={"status":"EVENT_CAUSAL_RESEARCH_ONLY","oos_games":len(games),"simulated_games":len(done),"player_prediction_rows":len(pred_rows),"joint_prediction_rows":len(joint_rows),"n_sims":N_SIMS,"starter_feature":"UNAVAILABLE","strict_publication_causality":"NO"}
    (OUT/"SPORTS_NOVA_V3_PLAYER_MARGINAL_VALIDATION_V1.json").write_text(json.dumps(summary,indent=2))
    (OUT/"SPORTS_NOVA_V3_JOINT_VALIDATION_V1.json").write_text(json.dumps(summary,indent=2))
    (OUT/"SPORTS_NOVA_V3_STRUCTURAL_CORRELATION_V1.json").write_text(json.dumps(summary,indent=2))
    (OUT/"SPORTS_NOVA_V3_PLAYER_ABLATIONS_V1.json").write_text(json.dumps({**summary,"ablations":"NOT_RUN: runner uses frozen V3 default; no post-outcome retuning"},indent=2))
    (OUT/"SPORTS_NOVA_V3_QB_V2_V3_COMPARISON_V1.json").write_text(json.dumps({**summary,"official_v2":"PENDING_SCORING"},indent=2))
    (OUT/"SPORTS_NOVA_V3_PLAYER_JOINT_FINAL_REPORT_V1.md").write_text("# SPORTS-NOVA V3 player/joint walk-forward\n\nStatus: `EVENT_CAUSAL_RESEARCH_ONLY`. Strict publication-time causality: `NO`. Starter feature: `UNAVAILABLE`. See JSON artifacts for run counts.\n")
    print(json.dumps(summary,indent=2))

if __name__ == "__main__": main()
