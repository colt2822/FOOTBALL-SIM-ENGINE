"""Aggregate the immutable V22 season-shard replay outputs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
shards = [json.loads((DATA / f"SPORTS_NOVA_M1_ROSTER_STATE_REPAIR_V22_{x}.json").read_text()) for x in ("A", "B", "C")]

metrics = {}
for version in ("v21", "v22"):
    keys = sorted(set().union(*(s["METRIC_SUMS"][version] for s in shards)))
    metrics[version] = {k: float(sum(s["METRIC_SUMS"][version].get(k, 0.) for s in shards) /
                             sum(s["METRIC_COUNTS"][version].get(k, 0) for s in shards)) for k in keys}
counts = {version: {k: int(sum(s["METRIC_COUNTS"][version].get(k, 0) for s in shards))
                    for k in metrics[version]} for version in ("v21", "v22")}
failures = {version: {k: int(sum(s["ROSTER_STATE_FAILURES"][version].get(k, 0) for s in shards))
                      for k in set().union(*(s["ROSTER_STATE_FAILURES"][version] for s in shards))}
            for version in ("v21", "v22")}
wrong_before = sum(s["WRONG_QB_COUNT_BEFORE"] for s in shards)
wrong_after = sum(s["WRONG_QB_COUNT_AFTER"] for s in shards)
truth_rows = sum(s["QB_TRUTH_ROWS"] for s in shards)
ledger_rows = []
for x in ("A", "B", "C"):
    p = DATA / f"SPORTS_NOVA_M1_V22_QB_TRUTH_LEDGER_{x}.jsonl"
    ledger_rows.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())
ledger_rows.sort(key=lambda x: (x["GAME_ID"], x["TEAM"]))
ledger = DATA / "SPORTS_NOVA_M1_V22_QB_TRUTH_LEDGER.jsonl"
ledger.write_text("\n".join(json.dumps(x, sort_keys=True) for x in ledger_rows) + "\n")
payload = {
    "SCHEMA": "SPORTS_NOVA_M1_ROSTER_STATE_REPAIR_V22",
    "V22_STATUS": "HISTORICAL_REPLAY_COMPLETE_CURRENT_SLATE_BLOCKED",
    "SHIP_GATE": "BLOCKED_CURRENT_SLATE_INPUTS",
    "COHORT_GAMES": sum(s["COHORT_GAMES"] for s in shards),
    "TEAM_GAME_ROWS_WITH_OBSERVED_QB": truth_rows,
    "WRONG_QB_COUNT_BEFORE": wrong_before,
    "WRONG_QB_COUNT_AFTER": wrong_after,
    "WRONG_QB_RATE_BEFORE": wrong_before / truth_rows,
    "WRONG_QB_RATE_AFTER_CONFIRMED_TRUTH": wrong_after / (truth_rows - failures["v22"].get("unknown_qb_truth", 0)),
    "UNKNOWN_QB_TRUTH_COUNT": failures["v22"].get("unknown_qb_truth", 0),
    "ROSTER_STATE_FAILURES_BEFORE": failures["v21"],
    "ROSTER_STATE_FAILURES_AFTER": failures["v22"],
    "ACCOUNTING_STATUS": "PASS_INHERITED_FROZEN_V4",
    "QB_MAE_V21": {k.removeprefix("qb_"): metrics["v21"][k] for k in metrics["v21"] if k.startswith("qb_")},
    "QB_MAE_V22": {k.removeprefix("qb_"): metrics["v22"][k] for k in metrics["v22"] if k.startswith("qb_")},
    "RB_MAE_DELTA_V22_MINUS_V21": {"opportunity": metrics["v22"]["RB_opportunity"] - metrics["v21"]["RB_opportunity"],
                                   "yardage": metrics["v22"]["RB_yardage"] - metrics["v21"]["RB_yardage"]},
    "WRTE_MAE_DELTA_V22_MINUS_V21": {
        "WR_opportunity": metrics["v22"]["WR_opportunity"] - metrics["v21"]["WR_opportunity"],
        "WR_yardage": metrics["v22"]["WR_yardage"] - metrics["v21"]["WR_yardage"],
        "TE_opportunity": metrics["v22"]["TE_opportunity"] - metrics["v21"]["TE_opportunity"],
        "TE_yardage": metrics["v22"]["TE_yardage"] - metrics["v21"]["TE_yardage"]},
    "ALL_PLAYER_METRICS": metrics,
    "METRIC_COUNTS": counts,
    "TRUTH_LEDGER": str(ledger.relative_to(ROOT)),
    "CURRENT_SLATE_INPUT_STATUS": "PARTIAL_NOT_USABLE",
    "REAL_BLOCKER": "No causal injury/expected-inactive feed is present; current ATL and SEA QB schedule/depth inputs conflict.",
    "NEXT_SINGLE_ACTION": "Ingest and reconcile a causal pregame injury/inactive roster source for the Week 2 slate, resolving the ATL and SEA QB conflicts before ship.",
    "NO_MARKET_DATA": True,
}
(DATA / "SPORTS_NOVA_M1_ROSTER_STATE_REPAIR_V22.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
print(json.dumps(payload, indent=2, sort_keys=True))
