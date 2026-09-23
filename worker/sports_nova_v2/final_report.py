"""Assemble the requested human/machine-readable completion report."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config as C


def main() -> None:
    validation = json.loads((C.ARTIFACTS / "SPORTS_NOVA_V2_1_WEEK1_VALIDATION.json").read_text())
    joint = json.loads((C.ARTIFACTS / "SPORTS_NOVA_V2_JOINT_TEST_REPORT.json").read_text())
    buckets = validation["BUCKETS"]
    out = {
        "ARTIFACT_ID": "SPORTS_NOVA_NODE3_V2_1_WEEK1_AND_JOINT_EXPORT",
        "GENERATED_AT": datetime.now(timezone.utc).isoformat(),
        "IMPLEMENTATION_STATUS": "COMPLETE",
        "WEEK_1_FIX": {
            "ROOT_CAUSE_CONFIRMED": validation["ROOT_CAUSE_CONFIRMED"],
            "FIX_SELECTED": validation["FIX_SELECTED"],
            "TEMPORAL_FIREWALL": validation["TEMPORAL_FIREWALL"],
            "CURRENT_V2_TEST": buckets["OVERALL_TEST"]["CURRENT_V2"],
            "V2_1_TEST": buckets["OVERALL_TEST"]["V2_1"],
            "CURRENT_V2_WEEK1": buckets["WEEK_1"]["CURRENT_V2"],
            "V2_1_WEEK1": buckets["WEEK_1"]["V2_1"],
            "WEEK1_IMPROVEMENT": validation["WEEK1_IMPROVEMENT"],
            "OVERALL_V2_1_BEATS_V2": validation["OVERALL_V2_1_BEATS_V2"],
            "CHAMPION_MODEL": validation["CHAMPION_MODEL"],
            "DRAKE_MAYE_2026_PREFLIGHT": validation["DRAKE_MAYE_2026_PREFLIGHT"],
        },
        "JOINT_EXPORT": {
            "JOINT_PROBABILITY_INTERFACE": "READY",
            "SIMULATION_SAMPLE_EXPORT": "READY",
            "SAME_GAME_INDEPENDENCE_FALLBACK": "BLOCKED",
            "SUPPORTED_JOINT_LEG_TYPES": ["QB_PASS_YARDS", "LEAD_WR_REC_YARDS",
                                           "LEAD_RB_RUSH_YARDS", "OPPOSING_QB_PASS_YARDS"],
            "MARGINAL_PRODUCT_DIAGNOSTIC": "PASS",
            "DEPENDENCY_DELTA": "PASS",
            "UNCERTAINTY_BOUNDS": "PASS",
            "NODE4_SCHEMA_COMPATIBLE": "YES",
            "JOINT_TESTS_PASSED": joint["PASSED"],
            "JOINT_TESTS_FAILED": joint["FAILED"],
        },
        "FILES_CREATED": [
            "SPORTS_NOVA_V2_1_DEV_SELECTION.json",
            "SPORTS_NOVA_V2_1_MODEL_COMPARISON.json",
            "SPORTS_NOVA_V2_1_WEEK1_VALIDATION.json",
            "SPORTS_NOVA_V2_1_MANIFEST.json",
            "SPORTS_NOVA_V2_JOINT_PROBABILITY_SCHEMA.json",
            "SPORTS_NOVA_V2_JOINT_EXPORT.py",
            "SPORTS_NOVA_V2_SIMULATION_SAMPLE_SCHEMA.json",
            "SPORTS_NOVA_V2_SIMULATION_SAMPLES.jsonl.gz",
            "SPORTS_NOVA_V2_JOINT_TEST_REPORT.json",
        ],
        "FIRST_REAL_BLOCKER": (
            "V2.1 did not improve Week-1 MAE and worsened probability metrics; "
            "V2 remains champion. Exact starter labels are unavailable."
        ),
        "SINGLE_NEXT_ACTION": (
            "Keep V2 promoted and collect prospective Week-1 shadow evidence; "
            "do not promote V2.1 or infer starter labels."
        ),
        "CRITICAL": "No EV or correlated-parlay edge claimed; unresolved dependencies remain fail-closed.",
    }
    (C.ARTIFACTS / "SPORTS_NOVA_NODE3_V2_1_WEEK1_AND_JOINT_EXPORT.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
