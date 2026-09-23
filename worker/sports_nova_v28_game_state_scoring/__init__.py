"""SPORTS_NOVA_V28 (sports_nova_v28.game_state_scoring.K): possession-terminal scoring boundary.

Parents (never edited): V23 scoring executor, V25 role-aware repair, V26 active-skill completion, V27 causal FG estimator.  Package note: relative imports only.

V28 REPLACES `worker.sports_nova_v23.simulator._run_one` (the per-receiver / per-carrier independent TD binomials) with a fork, so the V27 trick of calling the
V23 `_run_one` unchanged does not apply here; that is the point of the version.  Everything upstream of the score decision (volume, play selection, opportunity
allocation, QB selection, completion / yardage draws) is the V23 code path, statement for statement, on the same main RNG stream.

What changes (one coherent change set: the three parts cannot be separated because "at most one TD" requires deciding who gets it):
  1. ONE possession terminates in at most ONE offensive scoring event {TD, FG, NO_SCORE}.  P(TD) is a causal historical hazard of the block's simulated production
     (yards, plays), score differential and clock -- not an independent per-player trial.
  2. The single TD is allocated to ONE scorer with probability proportional to the block's realized opportunity (receptions x pass TD rate, carries x rush TD rate).
     QB scrambles are carries of the QB, so the QB can be the rushing scorer (the V23 loop never gave scrambles a TD trial).
  3. FG: the V27 causal estimator-A rate, flat, applied exactly where V23 applied it (no TD AND plays >= 3).  Unchanged, fail-closed.

Version strings (each a separate, attributable candidate):
  .1  = 1-3, regulation only (regulation ties stay TIE, as V27)
  .2  = .1 + NFL-rule overtime
  .3  = .1 + POST_HOC yardage-marginal alignment of the hazard input (only if the raw hazard overshoots; see PREREG)
  .4  = .2 + .3
  .5  = .1 + POST_HOC efficiency-noise removal: the per-game efficiency draw's NOISE (sigma .35) is shrunk by kappa=0; recent_form (the team signal) is untouched.  Motivation is a
        measurement made after the V28.1 run: coupling scores to yards exposed a 2x-overdispersed team-game yardage marginal (sim SD 167 vs history 80); with the noise removed the
        simulated SD is 79 (history 80).  The target is an intermediate quantity from the drive table, never a game outcome or a market number.
  .6  = .5 + NFL-rule overtime
  .7  = .5 + team finishing multiplier (POST_HOC #2): the V23 team-specific pass_td_rate / rush_td_rate signal, dropped by the league-average hazard, returns as an odds multiplier
        (team intensity / pooled league intensity over the same causal window).  Reason: V28.6's winner Brier was significantly worse than V27's (paired bootstrap), i.e. V28 had thrown away
        team-specific scoring propensity that V23 carried.
  .8  = .7 + drive-clock scale (POST_HOC #2): a block's elapsed seconds x (V27 sim blocks/team-game / history drives/team-game = 1.0508), the block-vs-drive definition gap the brief names
  .9  = .8 + NFL-rule overtime          .10 = .7 + NFL-rule overtime
  .11 = .8 + recent_form shrink (POST_HOC #3): the efficiency multiplier's team signal is scaled by its causal empirical slope on next-game total yards (~0.25; history says ~1/4 of
        trailing-8 pass-yardage form persists, the simulator used 1.0).  .12 = .11 + NFL-rule overtime
"""
