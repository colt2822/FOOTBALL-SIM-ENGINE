"""SPORTS_NOVA_V25 (sports_nova_v25.role_aware_redistribution.1): role-aware redistribution of removed opportunity mass.

V24 (worker/sports_nova_v24_roster_eligibility, hash 491d536c...) zeroes ineligible players and lets the unchanged V23 allocation renormalize
pro-rata over EVERY survivor.  That pool contains the QB's historical carry share (scrambles included), so removed RB mass also lands on the QB
(KC Mahomes 7.7 -> 24.5 mean carries).  V25 changes ONLY that redistribution step: mass removed from a team/kind pool goes exclusively to eligible
RB/FB/WR/TE recipients; the QB (and any non-skill role) is HELD at the share it would have had if nobody were removed.
Play volume, play selection, QB selection, scoring, RNG contract, calibration = frozen V23, imported unchanged.  QB scrambling stays on V23's
separate scramble path.  This does NOT fix V23's inherited QB double count (designed-carry share + scrambles both credited to the QB).

Package note: relative imports only, so it is rename-safe.  It imports nothing from either V24 package.
"""
