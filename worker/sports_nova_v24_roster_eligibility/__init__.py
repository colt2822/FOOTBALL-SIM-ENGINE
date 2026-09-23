"""SPORTS_NOVA_V24 (sports_nova_v24.roster_eligibility_fix.1): game-eligibility repair of M1 opportunity shares.

V23 zeroes only injury-report OUT players; players on injured reserve, off the roster, on another team, exempt-listed or
practice-squad still received carries/targets from stale history.  V24 changes ONLY the frozen opportunity shares that
reach V23's allocation, using the pregame roster/injury snapshot.  Play volume, play selection, efficiency, scoring, QB selection,
RNG contract and calibration are the frozen V23 code, imported unchanged.  V23 (worker/sports_nova_v23) is not modified.

Package note: this directory uses relative imports so it can be renamed to worker/sports_nova_v24 without edits.
"""
