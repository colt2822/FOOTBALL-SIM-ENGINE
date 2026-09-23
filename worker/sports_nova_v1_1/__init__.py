"""SPORTS_NOVA_V1_1 challenger namespace.

Minimal-diff fork of worker/sports_nova_v3: reuses every V3 module UNCHANGED
(schemas, pregame_state, distributions, environment, game_script, game_state,
allocation, scoring, efficiency, calibration, play_selection) by importing
them directly from worker.sports_nova_v3, so PregameState/SimulationBatch
instances stay interchangeable between the two engines (same classes, not
structurally-similar duplicates). Only the two modules implementing the
V1.1 hypothesis are forked: play_volume.py (adds a data-derived hurry-up
pace multiplier) and simulator.py (passes the offense's live score
differential into draw_block_volume so it can apply that multiplier).

worker/sports_nova_v3 remains completely untouched by this package -- no
file under sports_nova_v3/ is imported for its side effects, none is
edited, and this package does not monkeypatch anything on it.

No identity-gate wiring yet: this namespace is for offline DEV/VALIDATION
calibration against historical (already-kickoff) games only, where
worker/sports_nova_v3/__init__.py's own gate already bypasses via its
`kickoff <= now` check. Before this namespace is ever used for a
PROSPECTIVE (future-kickoff) capture, it needs the same identity-gate
enforcement V3 has -- named here as a gap, not silently assumed covered.
"""
