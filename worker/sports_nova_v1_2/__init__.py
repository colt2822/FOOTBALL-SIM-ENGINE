"""SPORTS_NOVA_V1_2 challenger namespace.

Independent of worker/sports_nova_v1_1 (the hurry-up pace experiment,
REJECTED -- see SPORTS_NOVA_V1_1_SCORECARD.json EXPERIMENT_001). This
namespace tests exactly ONE change in isolation, per the mission's
one-change-per-experiment rule: empirical-Bayes shrinkage of the
`yards_per_reception` feature toward a training-derived league mean.

Only simulator.py is forked. play_volume.py, and every other mechanism,
is imported UNCHANGED from worker.sports_nova_v3 -- this experiment does
NOT include the pace multiplier, so its effect can be measured cleanly
against the V18 baseline without confounding with EXPERIMENT_001.

No identity-gate wiring: offline DEV/VALIDATION calibration only, same
caveat as worker/sports_nova_v1_1/__init__.py.
"""
