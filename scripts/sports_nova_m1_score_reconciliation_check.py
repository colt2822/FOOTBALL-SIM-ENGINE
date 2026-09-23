"""SPORTS_NOVA M1 Phase-2 gap check: team-points reconciliation.

worker/sports_nova_v21/simulator.py (and the V3/V4 engines that share the same
_run_one scoring block) never persist a PAT-made or field-goal count anywhere
-- `made_pats` and the FG `block_points = 3` branch are added straight into
`state.home_score`/`away_score` with no counter retained. That means no
downstream consumer, including the Phase-2 accounting suite in
scripts/sports_nova_m1_v4_raw_path_diagnostic_challenger.py, can check "team
points reconciles with PAT/FG" against real data -- the field simply doesn't
exist to check.

This script is the cheapest thing that can still test the invariant from the
outside, using the same mod-N/discreteness method that found the V21 missing-
PAT bug (see memory: project-sports-nova-v20-v21-findings): for team points
`S` and realized `T = pass_tds + rush_tds` (already persisted per team in the
raw ledger), the scoring model can only produce `S = 6*T + P + 3*F` for some
integer `0 <= P <= T` (made PATs, capped by TD count) and `F >= 0` (made FGs).
For any given T, that is feasible unless `S < 6*T` or no P in [0, T] leaves
`S - 6*T - P` a non-negative multiple of 3. This is necessary, not sufficient
-- passing this does not prove PAT/FG accounting is correct, only that it is
not OBVIOUSLY broken (e.g. a dropped PAT/FG bucket analogous to the residual-
target bug would very likely show up here, as it did for the original V21
missing-PAT defect via the simpler mod-3 form of the same check).

Read-only: consumes the already-completed
SPORTS_NOVA_M1_V4_RAW_PATH_LEDGER_V1.jsonl (1,693-game, 16-sims/game V4 run).
Does not run any new simulations, does not touch worker/ source.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "sports_nova_v3" / "SPORTS_NOVA_M1_V4_RAW_PATH_LEDGER_V1.jsonl"
OUT = ROOT / "data" / "sports_nova_v3" / "SPORTS_NOVA_M1_SCORE_RECONCILIATION_REPORT_V1.json"


def feasible(score: int, tds: int) -> bool:
    if score < 0 or tds < 0 or score < 6 * tds:
        return False
    for pats in range(tds + 1):
        rem = score - 6 * tds - pats
        if rem >= 0 and rem % 3 == 0:
            return True
    return False


def main() -> None:
    n_rows = 0
    n_teams = 0
    failures: list[dict] = []
    td_hist: Counter = Counter()
    with LEDGER.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            n_rows += 1
            for team in json.loads(row["team_offense_json"]):
                n_teams += 1
                tds = int(team["pass_tds"]) + int(team["rush_tds"])
                score = int(team["score"])
                td_hist[tds] += 1
                if not feasible(score, tds):
                    failures.append({
                        "game_id": row["game_id"], "sim_id": row["sim_id"],
                        "team_id": team["team_id"], "team_tds": tds, "team_score": score,
                    })
    out = {
        "rows_checked": n_rows,
        "team_slots_checked": n_teams,
        "SCORE_TD_FEASIBILITY_FAILURES": len(failures),
        "failure_rate": len(failures) / n_teams if n_teams else None,
        "sample_failures": failures[:25],
        "td_count_histogram": dict(sorted(td_hist.items())),
        "note": "Necessary-but-not-sufficient check (see module docstring). "
                "0 failures does not certify PAT/FG accounting; it only rules out "
                "an obvious dropped-scoring-bucket defect at the team-points level. "
                "PAT_MAKE_RATE/FG rate are UNVALIDATED_DEFAULT constants "
                "(worker/sports_nova_v21/config.py) not checked here.",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
