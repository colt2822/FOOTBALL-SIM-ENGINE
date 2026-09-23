"""SPORTS_NOVA_V27 (sports_nova_v27.causal_fg_rate.1): the V26 pipeline with ONE scoring parameter replaced.

V26 (active-skill state completion) -> V25 (role-aware redistribution) -> V23 scoring execution are preserved exactly.  The only change:
`fg_rate` (the P(made FG | block has no TD AND plays >= 3) that worker/sports_nova_v23/simulator.py:114 consumes) is no longer the unfitted 0.08 default that
`fit_distributions([], as_of)` returns; it is the frozen historical estimate from scripts/sports_nova_fg_causal_estimator_v1.py (estimator A),
recomputed from the frozen drive-block table for the simulated season-week and substituted with dataclasses.replace.

Nothing is edited in V23/V24/V25/V26 or worker/sports_nova_v3/distributions.py (shared by frozen versions).  There is no fallback to 0.08: an unavailable,
hash-mismatched, empty or temporally invalid estimator raises CausalFGError and the simulation does not run.

Package note: relative imports only, so it is rename-safe.
"""
