"""SPORTS_NOVA_V23 challenger namespace -- SN3_V23_RESIDUAL_ALLOCATION_FIX.

Tests exactly ONE change against V21 (current structural champion):
`allocate_opportunities` (worker/sports_nova_v3/allocation.py) appends the
panel's explicit residual target/carry share as an EXTRA bucket in the same
dirichlet-multinomial draw used to split pass attempts / rush attempts across
named active players, then DISCARDS whatever count lands in that bucket
(`AllocationResult.untargeted` / `.residual_carries`) instead of ever
crediting it to a real player.

Evidence this is a real, silent accounting loss in V21 (not a modeling
choice): an isolated V21-vs-V21+redistribution-fix ablation
(scripts/sports_nova_m1_v4_v21_residual_ablation.py, pilot 60 games/128 sims)
found 1231 accounting failures in V21 baseline (585 RUSH_ATTEMPTS_CONSERVATION
+ 566 RUSH_YARDS_TEAM_RECONCILIATION -- sum of per-player rush attempts/yards
falling short of the team total by exactly the dropped residual count) vs 0
in the fix arm, with no material change to QB_PASS_YARD_MAE (77.05 -> 77.20,
paired sign test p=0.38, n=141). An independent full-cohort run (1,693 games/
16 sims, via the separate scripts/sports_nova_m1_v4_raw_path_diagnostic_challenger.py
harness applying the same redistribution idea to V3) confirmed
ALLOCATION_NORMALIZATION:target_conservation and :carry_conservation both at
exactly 1.0 across 54,176 team-blocks with 0 accounting failures.
See memory project_sports_nova_v21_residual_fix_ablation for the full record.

A second, more severe instance of the same defect was found while designing
this fix: worker/sports_nova_v21/simulator.py's `_run_one` credits orphaned
`residual_carries` at least *team-level* rush yards (a separate yards-only
draw, still with zero player attribution) but has NO equivalent handling for
`untargeted` (pass/target residual) at all -- that mass is dropped from BOTH
player and team pass_yards. This was invisible to the existing accounting
suite because two of its six checks are tautological by construction
(`RECEPTIONS_LE_COMPLETIONS`, `RECEIVING_YARDS_TEAM_RECONCILIATION` are each
computed from the same loop on both sides of the comparison) and a third
(`TARGETS_LE_PASS_ATTEMPTS`) is a `<=` check that cannot detect a shortfall.

Fix scope (deliberately narrow, one interpretable change, same pattern as
V21's own PAT fix): worker/sports_nova_v23/allocation.py forks ONLY
`allocate_opportunities` to drop the residual bucket from the probability
vector before the dirichlet-multinomial draw, letting the existing
renormalization (`shares / shares.sum()` in
worker/sports_nova_v3/distributions.py::sample_dirichlet_multinomial) spread
the full opportunity count across only the named active players,
proportional to their existing shares. This is the SAME per-call RNG
contract as before (one `rng.dirichlet` + one `rng.multinomial`) -- no extra
draw is introduced, unlike the ablation harness's own validation monkeypatch,
which redistributes *after* the original draw with an additional
`rng.multinomial` call. The two mechanisms are intentionally different (this
one is a single joint draw over a reshaped probability vector); the ablation
result validates the FIX'S EFFECT, not this exact draw sequence -- V23 was
re-validated directly (see SPORTS_NOVA_M1_V23_VS_V21_ABLATION_V1.json), not
assumed equivalent to the harness's monkeypatch arm.

Every other mechanism (PAT crediting, pace/block-volume, play selection,
efficiency, TD rates, catch rate, QB identity/roster construction) is
imported UNCHANGED from worker.sports_nova_v21 (and, through it, v19/v3) --
worker/sports_nova_v23/simulator.py's only diff from
worker/sports_nova_v21/simulator.py is the `allocate_opportunities` import
line and the addition of `simulate_game`'s copy (required because Python
functions close over their own module's globals, not because its logic
differs).

Naming note: V22 is not used here. worker/sports_nova_v22/ was created and
populated the same session by a process this session could not attribute to
any live/reachable agent (an unrelated pregame-roster/QB-identity fix,
sports_nova_m1_roster_state_repair_v22.py); it produced no output artifact
and is no longer running. V22 is left untouched; this fix is V23 to avoid a
naming collision.
"""
