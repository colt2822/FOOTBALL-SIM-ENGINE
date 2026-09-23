"""V6 of the post-repair QB allocation validation.

Identical to V5 (same 60-game/128-sim cohort, same metrics) except it runs
against the SN3_QB_ATTEMPT_VOLUME_V5-repaired engine: make_state's team
pass_rate/plays-per-game are now computed from each team's most recent 8
games instead of the whole up-to-5-season lookback, and pace_seconds is now
derived from that same recent plays/game relative to the league's recent
plays/game instead of a flat 27.0 for every team. V1-V5 outputs are left
untouched for audit.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from worker.sports_nova_v3.config import MODEL_VERSION
from worker.sports_nova_v3.simulator import simulate_game, _primary_qb
from sports_nova_v3_player_joint_walkforward_v1 import (PLAYER, MANIFEST, PLAYER_SHA,
    game_parts, surrogate_kickoff, make_state)
OUT = ROOT / 'data' / 'sports_nova_v3'; N = 128

def main():
    games = json.loads(MANIFEST.read_text())['GAME_IDS']
    by = {}
    for g in games:
        by.setdefault(game_parts(g)[0], []).append(g)
    selected = []
    for season in sorted(by):
        xs = by[season]
        selected.extend([xs[int(i)] for i in np.linspace(0, len(xs) - 1, 10, dtype=int)])
    all_df = pd.read_parquet(PLAYER); all_df['_key'] = all_df.SEASON * 100 + all_df.WEEK
    oos = all_df[all_df.GAME_ID.isin(selected)].copy()
    rows = []; stage = []; identity = []; conservation = []; qb_conservation = []; team_game_identity = []
    for g in selected:
        s, w, away, home = game_parts(g); key = s * 100 + w
        prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, s - 5))]
        state = make_state(g, prior, surrogate_kickoff(s, w), None)
        sim = simulate_game(state, N, int(hashlib.sha256(g.encode()).hexdigest()[:8], 16), MODEL_VERSION)
        cur = oos[oos.GAME_ID == g]
        team_plays = []
        for tid in sim.team_ids:
            pa = sim.team(tid, 'pass_attempts'); ra = sim.team(tid, 'rush_attempts')
            team_plays.append(float(np.mean(pa + ra)))
            qb_ids_on_team = [pid for pid in sim.player_ids if sim.player_team[pid] == tid]
            for i in range(N):
                rec_y = sum(sim.player(pid, 'receiving_yards')[i] for pid in sim.player_ids if sim.player_team[pid] == tid)
                rec_t = sum(sim.player(pid, 'targets')[i] for pid in sim.player_ids if sim.player_team[pid] == tid)
                conservation.append(bool(rec_t <= sim.team(tid, 'pass_attempts')[i] and rec_y <= sim.team(tid, 'pass_yards')[i] + 1e-9))
                qb_att_sum = sum(sim.player(pid, 'pass_attempts')[i] for pid in qb_ids_on_team)
                qb_conservation.append(bool(qb_att_sum <= sim.team(tid, 'pass_attempts')[i] + 1e-9))
        stage.append({'GAME_ID': g, 'SIM_GAME_PLAYS': float(sum(team_plays)), 'SIM_TEAM_PLAYS_MEAN': team_plays,
                      'SIM_PASS_ATTEMPTS': float(sum(np.mean(sim.team(t, 'pass_attempts')) for t in sim.team_ids)),
                      'SIM_RUSH_ATTEMPTS': float(sum(np.mean(sim.team(t, 'rush_attempts')) for t in sim.team_ids))})
        qcur = cur[cur.POSITION == 'QB']
        primary_by_team = {}
        for tid in sim.team_ids:
            pq = _primary_qb(state, tid)
            model_starter = pq.player_id if pq is not None else None
            primary_by_team[tid] = model_starter
            team_qb_rows = qcur[qcur.TEAM == tid]
            realized_starter = (str(team_qb_rows.sort_values('PASS_ATTEMPTS', ascending=False).iloc[0].PLAYER_ID)
                                 if len(team_qb_rows) else None)
            team_game_identity.append({'GAME_ID': g, 'TEAM': tid, 'model_starter': model_starter,
                'realized_starter': realized_starter, 'unknown': model_starter is None,
                'match': (model_starter is not None and model_starter == realized_starter)})
        for r in qcur.itertuples():
            pid = str(r.PLAYER_ID)
            if pid not in sim.player_team: continue
            a = np.asarray(sim.player(pid, 'pass_yards'), float); at = np.asarray(sim.player(pid, 'pass_attempts'), float)
            identity.append({'zero': bool(at.mean() == 0), 'obs_att': float(r.PASS_ATTEMPTS), 'sim_att': float(at.mean()),
                'obs_y': float(r.PASS_YARDS), 'sim_y': float(a.mean()), 'sim_q': np.quantile(a, [.1, .9]).tolist(),
                'matched': bool(pid == primary_by_team.get(r.TEAM))})
        for r in cur.itertuples():
            pid = str(r.PLAYER_ID); stat = {'QB': 'pass_yards', 'RB': 'rush_yards', 'WR': 'receiving_yards', 'TE': 'receiving_yards'}.get(r.POSITION)
            if not stat or pid not in sim.player_team: continue
            a = np.asarray(sim.player(pid, stat), float); y = float(getattr(r, stat.upper()))
            rows.append({'POSITION': r.POSITION, 'TARGET': stat, 'OBS': y, 'PRED': float(a.mean()),
                'LO': float(np.quantile(a, .1)), 'HI': float(np.quantile(a, .9))})
    p = pd.DataFrame(rows)
    result = {'games': len(selected), 'sims_per_game': N, 'stage': stage, 'player_metrics': {},
        'conservation_pass': all(conservation), 'conservation_checks': len(conservation),
        'qb_conservation_pass': all(qb_conservation), 'qb_conservation_checks': len(qb_conservation)}
    for (pos, target), gdf in p.groupby(['POSITION', 'TARGET']):
        y = gdf.OBS.to_numpy(); x = gdf.PRED.to_numpy()
        result['player_metrics'][pos + '_' + target] = {'N': len(gdf), 'MAE': float(np.mean(abs(y - x))),
            'RMSE': float(np.sqrt(np.mean((y - x) ** 2))), 'coverage90': float(np.mean((y >= gdf.LO) & (y <= gdf.HI)))}
    qb = pd.DataFrame(identity)
    tgi = pd.DataFrame(team_game_identity)
    obs_att = qb.obs_att.to_numpy(); sim_att = qb.sim_att.to_numpy()
    pct = lambda a, q: float(np.quantile(a, q)) if len(a) else None
    result['qb'] = {
        'evaluation_rows': len(qb),
        'identity_match_rate': float(tgi.match.mean()) if len(tgi) else None,
        'identity_team_games': len(tgi),
        'unknown_rate': float(tgi.unknown.mean()) if len(tgi) else None,
        'zero_rate_after': float(qb.zero.mean()) if len(qb) else None,
        'gt42_attempt_rate': float((sim_att[sim_att > 0] > 42).mean()) if (sim_att > 0).any() else None,
        'gt50_attempt_rate': float((sim_att[sim_att > 0] > 50).mean()) if (sim_att > 0).any() else None,
        'realized_attempts_mean': float(qb.obs_att.mean()) if len(qb) else None,
        'sim_attempts_mean': float(qb.sim_att.mean()) if len(qb) else None,
        'realized_yards_mean': float(qb.obs_y.mean()) if len(qb) else None,
        'sim_yards_mean': float(qb.sim_y.mean()) if len(qb) else None,
        'realized_ypa': float(qb.obs_y.sum() / qb.obs_att.sum()),
        'sim_ypa': float(qb.sim_y.sum() / qb.sim_att.sum()),
        'realized_attempts_percentiles': {q: pct(obs_att, q / 100) for q in (10, 25, 50, 75, 90)},
        'sim_attempts_percentiles': {q: pct(sim_att, q / 100) for q in (10, 25, 50, 75, 90)},
    }
    def attempt_stats(g):
        if len(g) < 2: return None
        o, s = g.obs_att.to_numpy(), g.sim_att.to_numpy()
        return {'N': len(g), 'corr': float(np.corrcoef(o, s)[0, 1]),
                'mae': float(np.mean(np.abs(s - o))), 'bias_sim_minus_real': float(np.mean(s - o)),
                'sd_real': float(np.std(o, ddof=1)), 'sd_sim': float(np.std(s, ddof=1))}
    result['qb']['attempts_all_evaluated'] = attempt_stats(qb)
    result['qb']['attempts_matched_only'] = attempt_stats(qb[qb.matched]) if len(qb) else None
    (OUT / 'SPORTS_NOVA_V3_ENGINE_REPAIR_VALIDATION_V6.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
if __name__ == '__main__': main()
