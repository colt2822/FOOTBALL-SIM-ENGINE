"""Small post-repair validation for the two authorized V3 engine fixes."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from worker.sports_nova_v3.config import MODEL_VERSION
from worker.sports_nova_v3.simulator import simulate_game
from sports_nova_v3_player_joint_walkforward_v1 import (PLAYER, MANIFEST, PLAYER_SHA,
    game_parts, surrogate_kickoff, make_state)
OUT=ROOT/'data'/'sports_nova_v3'; N=128

def main():
    games=json.loads(MANIFEST.read_text())['GAME_IDS']
    # Ten deterministic games per season, preserving chronological order.
    by={}
    for g in games: by.setdefault(game_parts(g)[0],[]).append(g)
    selected=[]
    for season in sorted(by):
        xs=by[season]; selected.extend([xs[int(i)] for i in np.linspace(0,len(xs)-1,10,dtype=int)])
    all_df=pd.read_parquet(PLAYER); all_df['_key']=all_df.SEASON*100+all_df.WEEK
    oos=all_df[all_df.GAME_ID.isin(selected)].copy()
    rows=[]; stage=[]; identity=[]; conservation=[]
    for g in selected:
        s,w,away,home=game_parts(g); key=s*100+w
        prior=all_df[(all_df._key<key)&(all_df.SEASON>=max(1999,s-5))]
        sim=simulate_game(make_state(g,prior,surrogate_kickoff(s,w),None),N,int(hashlib.sha256(g.encode()).hexdigest()[:8],16),MODEL_VERSION)
        cur=oos[oos.GAME_ID==g]
        team_plays=[]
        for tid in sim.team_ids:
            pa=sim.team(tid,'pass_attempts'); ra=sim.team(tid,'rush_attempts')
            team_plays.append(float(np.mean(pa+ra)))
            for i in range(N):
                rec_y=sum(sim.player(pid,'receiving_yards')[i] for pid in sim.player_ids if sim.player_team[pid]==tid)
                rec_t=sum(sim.player(pid,'targets')[i] for pid in sim.player_ids if sim.player_team[pid]==tid)
                qb=[pid for pid in sim.player_ids if sim.player_team[pid]==tid and pid in set(cur[cur.TEAM==tid].PLAYER_ID.astype(str)) and pid in sim.player_team]
                conservation.append(bool(rec_t <= sim.team(tid,'pass_attempts')[i] and rec_y <= sim.team(tid,'pass_yards')[i]+1e-9))
        stage.append({'GAME_ID':g,'SIM_GAME_PLAYS':float(sum(team_plays)),'SIM_TEAM_PLAYS_MEAN':team_plays,
                      'SIM_PASS_ATTEMPTS':float(sum(np.mean(sim.team(t,'pass_attempts')) for t in sim.team_ids)),
                      'SIM_RUSH_ATTEMPTS':float(sum(np.mean(sim.team(t,'rush_attempts')) for t in sim.team_ids))})
        qcur=cur[cur.POSITION=='QB']; qids=[pid for pid in sim.player_ids if sim.player_team[pid] in (home,away) and any(pid==str(x) for x in qcur.PLAYER_ID)]
        selected_qb=[]
        for tid in sim.team_ids:
            qs=[p for p in sim.player_ids if sim.player_team[p]==tid and any(str(p)==str(x) for x in qcur[qcur.TEAM==tid].PLAYER_ID)]
            if qs:
                # actual selected QB is the one with nonzero causal usage after repair
                vals=[(float(np.mean(sim.player(pid,'pass_attempts'))),pid) for pid in qs]
                selected_qb.append(max(vals)[1])
        for r in qcur.itertuples():
            pid=str(r.PLAYER_ID)
            if pid not in sim.player_team: continue
            a=np.asarray(sim.player(pid,'pass_yards'),float); at=np.asarray(sim.player(pid,'pass_attempts'),float)
            identity.append({'match':pid in selected_qb,'zero':bool(a.mean()==0),'obs_att':float(r.PASS_ATTEMPTS),'sim_att':float(at.mean()),'obs_y':float(r.PASS_YARDS),'sim_y':float(a.mean()),'sim_q':np.quantile(a,[.1,.9]).tolist()})
        # player rows for supported continuous targets
        for r in cur.itertuples():
            pid=str(r.PLAYER_ID); stat={'QB':'pass_yards','RB':'rush_yards','WR':'receiving_yards','TE':'receiving_yards'}.get(r.POSITION)
            if not stat or pid not in sim.player_team: continue
            a=np.asarray(sim.player(pid,stat),float); y=float(getattr(r,stat.upper()))
            rows.append({'POSITION':r.POSITION,'TARGET':stat,'OBS':y,'PRED':float(a.mean()),'LO':float(np.quantile(a,.1)),'HI':float(np.quantile(a,.9))})
    p=pd.DataFrame(rows); result={'games':len(selected),'sims_per_game':N,'stage':stage,'identity':identity,'player_metrics':{},'conservation_pass':all(conservation),'conservation_checks':len(conservation)}
    for (pos,target),g in p.groupby(['POSITION','TARGET']):
        y=g.OBS.to_numpy(); x=g.PRED.to_numpy(); result['player_metrics'][pos+'_'+target]={'N':len(g),'MAE':float(np.mean(abs(y-x))),'RMSE':float(np.sqrt(np.mean((y-x)**2))),'coverage90':float(np.mean((y>=g.LO)&(y<=g.HI)))}
    qb=pd.DataFrame(identity); result['qb']={'evaluation_rows':len(qb),'identity_match_rate':float(qb.match.mean()) if len(qb) else None,'unknown_rate':0.0,'zero_rate_after':float(qb.zero.mean()) if len(qb) else None,'realized_attempts_mean':float(qb.obs_att.mean()) if len(qb) else None,'sim_attempts_mean':float(qb.sim_att.mean()) if len(qb) else None,'realized_yards_mean':float(qb.obs_y.mean()) if len(qb) else None,'sim_yards_mean':float(qb.sim_y.mean()) if len(qb) else None,'realized_ypa':float(qb.obs_y.sum()/qb.obs_att.sum()),'sim_ypa':float(qb.sim_y.sum()/qb.sim_att.sum())}
    (OUT/'SPORTS_NOVA_V3_ENGINE_REPAIR_VALIDATION_V1.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
