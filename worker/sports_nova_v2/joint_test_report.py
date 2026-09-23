"""Run and persist the focused joint-export test report."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config as C
from . import joint_export as J
from . import montecarlo as MC
from .tests import test_joint_export as T


def main() -> int:
    names = [name for name in sorted(dir(T)) if name.startswith("test_")]
    rows = []
    for name in names:
        try:
            getattr(T, name)()
            rows.append({"test": name, "status": "PASS"})
        except Exception as exc:  # pragma: no cover - report path
            rows.append({"test": name, "status": "FAIL", "error": repr(exc)})
    passed = sum(row["status"] == "PASS" for row in rows)
    report = {
        "ARTIFACT_ID": "SPORTS_NOVA_V2_JOINT_TEST_REPORT",
        "GENERATED_AT": datetime.now(timezone.utc).isoformat(),
        "TEST_COUNT": len(rows), "PASSED": passed,
        "FAILED": len(rows) - passed, "TESTS": rows,
        "NO_SPORTSBOOK_DATA": True,
        "SAME_GAME_INDEPENDENCE_FALLBACK": "BLOCKED",
    }
    # Emit a real 10,000-path compressed sample sidecar using the same
    # measured dependency inputs as the existing Node3 Monte Carlo foundation.
    spec = MC.GameSpec("NE", "SEA", 64, 63, .58, .56, [
        MC.PlayerSpec("qb", "QB", "home", 216.49, 71.98),
        MC.PlayerSpec("wr", "WR", "home", 75.0, 25.0, share=.26),
        MC.PlayerSpec("rb", "RB", "home", 62.0, 25.0, share=.55),
        MC.PlayerSpec("opp", "QB", "away", 230.0, 45.0),
    ])
    sim = MC.GameSimulator({"qb_vs_opposing_qb": .0516480520,
                            "qb_vs_own_lead_receiver": .4573309562,
                            "qb_pass_vs_own_lead_rb_rush": -.1309180879,
                            "receiver1_vs_receiver2": .0442474153}).simulate(
                                spec, n_paths=10_000, seed=C.SEED)
    sim["market_state_map"] = {"QB_PASS_YARDS": "qb", "LEAD_WR_REC_YARDS": "wr",
                                "LEAD_RB_RUSH_YARDS": "rb", "OPPOSING_QB_PASS_YARDS": "opp"}
    sim["marginal_model_status"] = {
        "QB_PASS_YARDS": "VALIDATED",
        "LEAD_WR_REC_YARDS": "UNVALIDATED_PLACEHOLDER",
        "LEAD_RB_RUSH_YARDS": "UNVALIDATED_PLACEHOLDER",
        "OPPOSING_QB_PASS_YARDS": "UNVALIDATED_PLACEHOLDER",
    }
    sample_path = J.export_simulation_samples(
        sim, C.ARTIFACTS / "SPORTS_NOVA_V2_SIMULATION_SAMPLES.jsonl.gz",
        model_id="SPORTS_NOVA_V2_FROZEN_MODEL", model_version="SPORTS_NOVA_V2",
        model_sha256=J.sha256_file(C.ARTIFACTS / "SPORTS_NOVA_V2_FROZEN_MODEL.json"),
        simulation_id="SPORTS_NOVA_V2_MC_10000_SEEDED", compressed=True,
        marginal_model_status=sim["marginal_model_status"])
    report["SAMPLE_EXPORT"] = {"path": str(sample_path), "paths": 10_000,
                                "compressed": True,
                                "MARGINAL_MODEL_STATUS": sim["marginal_model_status"]}
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (C.ARTIFACTS / "SPORTS_NOVA_V2_JOINT_TEST_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not report["FAILED"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
