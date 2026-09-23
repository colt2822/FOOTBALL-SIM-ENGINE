"""SPORTS_NOVA_V19 challenger namespace -- SPORTS_V19_QB_IDENTITY_ONLY_CHALLENGER.

Tests exactly ONE change against frozen V18: `_qb_shares`' hard min-gap
recency filter is bypassed for a team-game where an independent, causal,
pre-kickoff QB-identity resolution is available (season-boundary depth-chart
snapshot; see scripts/sports_nova_v19_qb_identity_resolver.py). Every other
mechanism (pace/block-volume, allocation, efficiency, scoring, TD rates,
catch rate, volume-based QB share splitting once eligibility is decided) is
imported UNCHANGED from worker.sports_nova_v3 -- only simulator.py is
forked, same minimal-diff pattern as worker/sports_nova_v1_1/2/3.

No identity-gate wiring: this is an offline DEV/replay comparison against a
frozen historical week, not a live prospective capture path.
"""
