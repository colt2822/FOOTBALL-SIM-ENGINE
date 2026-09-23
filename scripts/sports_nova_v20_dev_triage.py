"""SPORTS_NOVA_V20 -- DEV-set (2025 Week 1) roster-bucket triage.

Same bucket logic as scripts/sports_nova_v20_case_b_triage.py, run against
the DEV resolutions from scripts/sports_nova_v20_dev_resolver.py, over all
32 2025-Week-1 team-slots (not just the 6 known 2026 mismatches). This
tells us, BEFORE writing any injection code:
  - how many DEV team-slots are genuine injection candidates
    (ROWS_OTHER_TEAM_ONLY and RESOLUTION_CORRECT)
  - how many would be injected on a WRONG resolution (regression risk)
  - how many are PRESENT_IN_ROSTER already (injection must never touch these)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "sports_nova_v3"
LIVE_PANEL = DATA / "validation_inputs_live" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
WALKFORWARD_SCRIPT = ROOT / "scripts" / "sports_nova_v3_player_joint_walkforward_v1.py"
RESOLUTION_PATH = DATA / "SPORTS_NOVA_V20_DEV_2025_WEEK1_QB_RESOLUTION.json"
OUT_PATH = DATA / "SPORTS_NOVA_V20_DEV_TRIAGE.json"
SEASON, WEEK = 2025, 1


def main():
    spec = importlib.util.spec_from_file_location("wf_v20_dev_triage", WALKFORWARD_SCRIPT)
    wf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wf)

    panel = pd.read_parquet(LIVE_PANEL)
    panel["_event_time"] = pd.to_datetime(panel["EVENT_TIME"], utc=True, errors="coerce")

    payload = json.loads(RESOLUTION_PATH.read_text())
    resolutions = payload["RESOLUTIONS"]
    ground_truth = payload["GROUND_TRUTH"]

    report = []
    for key, gt in ground_truth.items():
        gid, team = key.split("__")
        res = resolutions[key]
        # Kickoff cutoff: strictly before this game's own 2025-Week1 rows.
        game_rows = panel[panel.GAME_ID == gid]
        kickoff = game_rows._event_time.min()
        prior = panel[(panel._event_time < kickoff) & (panel.SEASON >= SEASON - 5)].copy()
        prior_for_state = prior.drop(columns=["_event_time"])

        state = wf.make_state(gid, prior_for_state, kickoff, None)
        roster_ids = {p.player_id for p in state.players if p.team_id == team}
        actual_id = gt["ACTUAL_STARTER_ID"]
        present_in_roster = actual_id in roster_ids

        rows_any = prior_for_state[prior_for_state.PLAYER_ID == actual_id]
        rows_tid = rows_any[rows_any.TEAM == team]
        latest_team_map = (prior_for_state.sort_values(["SEASON", "WEEK"])
                            .drop_duplicates("PLAYER_ID", keep="last")
                            .set_index("PLAYER_ID")["TEAM"].to_dict())
        latest_team_for_player = latest_team_map.get(actual_id)

        if present_in_roster:
            bucket = "PRESENT_IN_ROSTER"
        elif len(rows_any) == 0:
            bucket = "NO_ROWS_ANYWHERE"
        elif len(rows_tid) == 0:
            bucket = "ROWS_OTHER_TEAM_ONLY"
        elif latest_team_for_player != team:
            bucket = "ROWS_FOR_TID_BUT_STALE_LATEST_TEAM"
        else:
            bucket = "UNEXPLAINED"

        # Would the resolver-driven injection rule fire here, and would it be correct?
        res_id = res.get("QB_ID")
        res_confirmed = res.get("STATUS") == "CONFIRMED"
        would_inject = res_confirmed and bucket in ("ROWS_OTHER_TEAM_ONLY", "NO_ROWS_ANYWHERE")
        injected_player_correct = (res_id == actual_id) if would_inject else None

        report.append({
            "GAME_ID": gid, "TEAM": team,
            "ACTUAL_STARTER_ID": actual_id, "ACTUAL_STARTER_NAME": gt["ACTUAL_STARTER_NAME"],
            "RESOLUTION_QB_ID": res_id, "RESOLUTION_QB_NAME": res.get("QB_NAME"),
            "RESOLUTION_CORRECT": res_id == actual_id,
            "BUCKET": bucket,
            "WOULD_INJECT": would_inject,
            "INJECTED_PLAYER_WOULD_BE_CORRECT": injected_player_correct,
            "N_ROWS_ANY_TEAM": int(len(rows_any)), "N_ROWS_FOR_TID": int(len(rows_tid)),
        })

    print(json.dumps(report, indent=2, default=str))
    summary = {
        "N_TEAM_SLOTS": len(report),
        "BUCKET_COUNTS": {b: sum(1 for r in report if r["BUCKET"] == b)
                           for b in set(r["BUCKET"] for r in report)},
        "N_WOULD_INJECT": sum(1 for r in report if r["WOULD_INJECT"]),
        "N_INJECTION_CORRECT": sum(1 for r in report if r["WOULD_INJECT"] and r["INJECTED_PLAYER_WOULD_BE_CORRECT"]),
        "N_INJECTION_WRONG": sum(1 for r in report if r["WOULD_INJECT"] and not r["INJECTED_PLAYER_WOULD_BE_CORRECT"]),
        "N_PRESENT_IN_ROSTER_UNTOUCHED": sum(1 for r in report if r["BUCKET"] == "PRESENT_IN_ROSTER"),
    }
    print(json.dumps(summary, indent=2))
    OUT_PATH.write_text(json.dumps({"SUMMARY": summary, "RESULTS": report}, indent=2, default=str))
    print("WROTE", OUT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
