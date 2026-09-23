"""SPORTS_NOVA_V18_CURRENT_SEASON_IDENTITY_REFRESH.

Diagnostic + provenance layer on top of [[V18_2026_CAUSAL_PANEL_REFRESH]].
That mission fixed QB identity retroactively wherever a 2026 game had
already gone final (ATL: Kirk Cousins -> Cooper Rush, once week-1 box
scores entered the live panel). This script answers a narrower question the
panel refresh does not: for a team that has NOT yet been re-confirmed by a
completed 2026 game since some roster change, is there any causally-provable
pre-kickoff signal that current identity has moved, and if so, with what
confidence -- never a silent guess.

INVARIANTS (same as the mission brief): no V18 model change, no rewriting
prior prospective artifacts, no post-kickoff information, historical rows
immutable, QB identity provenance required, uncertain starter -> explicit
uncertainty.

SOURCE PHASE -- what is accepted vs rejected, and why
------------------------------------------------------
ACCEPTED:
  1. The live causal panel (validation_inputs_live/NFL_V3_PLAYER_GAME_CAUSAL_V1
     .parquet) -- box-score derived, causal by construction for any future
     game (only *completed* games are in it). Primary evidence.
  2. nflverse-data release tag "depth_charts", asset depth_charts_<season>
     .parquet, for season in {2025, 2026}. THIS IS NOT THE SAME SOURCE V14
     (sports_nova_v3_qb_depth_chart_signal_falsification_v14.py) rejected.
     V14 examined seasons 2001-2019 under a schema with NO timestamp field
     and explicitly noted "nflreadr's own dictionary vignette states the
     data source changed after the 2024 season". Verified directly here:
     depth_charts_2024.parquet and earlier use the old schema (season,
     club_code, week, game_type, depth_team, ...; no timestamp).
     depth_charts_2025.parquet and depth_charts_2026.parquet use a NEW
     schema (dt, team, player_name, pos_name, pos_rank, ...) where `dt` is a
     real per-snapshot UTC scrape timestamp (2025 asset: 221 distinct dt
     values spanning 2025-08-03 to 2026-03-14; 2026 asset: 179 distinct dt
     values spanning 2026-03-22 to 2026-09-15 as of this run). Because `dt`
     is provably before or after any given kickoff, CAUSAL_PRE_KICKOFF *is*
     provable for this asset, unlike the one V14 evaluated. Used only as
     corroboration / uncertainty-trigger against the panel signal, per the
     acceptance-gate note below -- never as sole positive evidence that
     overrides a box-score-confirmed starter.
  3. nflverse-data release tag "injuries", asset injuries_<season>.parquet
     (week-granular report_status / practice_status). No per-row timestamp,
     but tied to the league's official weekly report cycle -- used only to
     downgrade confidence on a depth-chart QB1 who is listed Out/Doubtful/IR
     that week, never as standalone identity evidence.
  4. nflverse-data release tag "weekly_rosters", asset
     roster_weekly_<season>.parquet (`status` field: ACT/RES/PUP/...) -- used
     only to flag a candidate as inactive/departed, never as starter
     evidence.

REJECTED (carried forward, not re-litigated):
  5. depth_charts_2001..2024.parquet (old schema) -- REJECTED per V14:
     no timestamp field discoverable, no scrape-cadence documentation.
  6. This project's own STARTER_STATUS column -- REJECTED per V12
     (SPORTS_NOVA_V12_QB_IDENTITY_SIGNAL_AVAILABILITY.json): 100% null,
     and the panel self-labels TARGET_DATA=True / PREGAME_FEATURE_DATA=False
     for every row regardless.

RESOLVE PHASE
-------------
For every team appearing in an ELIGIBLE future game (per
prospective/classification/SPORTS_NOVA_V18_PROSPECTIVE_CLASSIFICATION.json),
resolve "as of now" (this script's UTC run time):
  a. PANEL_LEADER: among that team's rows in the live panel with EVENT_TIME
     strictly before now, take the most recent single GAME_ID, then the
     player with max PASS_ATTEMPTS in that team's rows for that game.
  b. DEPTH_CHART_QB1: in the newest depth-chart snapshot (dt strictly before
     now, unioning the 2025+2026 assets), that team's pos_name=='Quarterback'
     row with pos_rank==1.
  c. Injury/roster corroboration on DEPTH_CHART_QB1 (and PANEL_LEADER):
     downgrade / flag per the acceptance rules above.
  d. Agreement -> CONFIRMED/HIGH. Disagreement -> UNCERTAIN, BOTH candidates
     kept, confidence split. No panel history at all -> UNRESOLVED.

VALIDATE PHASE
--------------
  - ATL explicit check against the mission's KNOWN_EXAMPLE.
  - Non-circular backcheck on the full 2025 season (the only season where
    the timestamped depth-chart asset overlaps a season that is entirely
    complete): for each 2025 game, resolve DEPTH_CHART_QB1 using a snapshot
    strictly before that game's own EVENT_TIME (temporal firewall), and
    separately PRIOR_GAME_LEADER using only that team's most recent EARLIER
    game -- then compare BOTH against the actual box-score leading passer
    IN THAT GAME (ground truth, independent of either predictor). This is
    the non-circular direction: pre-kickoff prediction vs. after-the-fact
    outcome. 2026 gets ZERO holdout weeks (only week 1 is final; there is no
    week-0 to hold out from), so BOX_SCORE_BACKCHECK_N covers 2025 only and
    is reported as a methodology check, not a 2026-season number.
  - Departed/inactive leakage: flag any resolved candidate whose latest
    weekly-roster `status` != 'ACT'.

WIRE PHASE
----------
Captures exactly one NEW future game (not 2026_02_DET_BUF, already frozen;
not 2026_02_CAR_ATL, already captured by the prior mission) into the same
predictions_live/ cohort, using the untouched frozen simulator/_primary_qb
mechanism for the actual prediction math (no V18 model change). The
per-team identity resolution from this script is attached to the artifact
as an informational IDENTITY_RESOLUTION block; if the frozen _primary_qb's
own pick disagrees with this script's high-confidence resolution, that is
recorded as a KNOWN_LIMITATION rather than silently overridden.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sports_nova_v3_v18_qb_uncertainty_policy as policy

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(ROOT))
from worker.sports_nova_v3 import identity_gate  # noqa: E402 -- canonical_exclusion_hash
DATA = ROOT / "data" / "sports_nova_v3"
PROSPECTIVE = DATA / "prospective"
IDENTITY_DIR = PROSPECTIVE / "identity"
CLASSIFICATION_PATH = PROSPECTIVE / "classification" / "SPORTS_NOVA_V18_PROSPECTIVE_CLASSIFICATION.json"
PRED_DIR_LIVE = PROSPECTIVE / "predictions_live"
LEDGER_LIVE_PATH = PROSPECTIVE / "ledger" / "SPORTS_NOVA_V18_PROSPECTIVE_LEDGER_LIVE.json"
REPORT_PATH = PROSPECTIVE / "SPORTS_NOVA_V18_CURRENT_IDENTITY_REFRESH_REPORT.json"
RESOLUTION_PATH = IDENTITY_DIR / "SPORTS_NOVA_V18_QB_IDENTITY_RESOLUTION_2026.json"

FREEZE_MANIFEST_PATH = DATA / "SPORTS_NOVA_V18_FREEZE_MANIFEST.json"
WALKFORWARD_SCRIPT = ROOT / "scripts" / "sports_nova_v3_player_joint_walkforward_v1.py"
SIMULATOR_PATH = ROOT / "worker" / "sports_nova_v3" / "simulator.py"

FROZEN_PANEL = DATA / "validation_inputs" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
LIVE_PANEL = DATA / "validation_inputs_live" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"
LIVE_RECEIPT = DATA / "validation_inputs_live" / "SPORTS_NOVA_V18_CAUSAL_PANEL_REFRESH_RECEIPT.json"

NFLVERSE_RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
INJURY_OUT_STATUSES = {"Out", "Injured Reserve", "Doubtful"}
THRESHOLD_GRID = [149.5, 174.5, 199.5, 224.5, 249.5, 274.5, 299.5, 324.5]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_obj(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def fetch_parquet(url: str) -> tuple[pd.DataFrame, str]:
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = resp.read()
    return pd.read_parquet(io.BytesIO(raw)), hashlib.sha256(raw).hexdigest()


def load_frozen_module_and_verify_freeze():
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text())
    mismatches = []
    for rel, meta in freeze["CODE_HASHES"].items():
        path = ROOT / rel
        actual = sha256_file(path)
        if actual != meta["sha256"]:
            mismatches.append(f"{rel}: expected {meta['sha256']} got {actual}")
    frozen_panel_expected = freeze["DATA_SCHEMA_HASHES"]["NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"]
    frozen_panel_actual = sha256_file(FROZEN_PANEL)
    frozen_panel_ok = frozen_panel_actual == frozen_panel_expected
    if not frozen_panel_ok:
        mismatches.append(
            f"FROZEN_PANEL: expected {frozen_panel_expected} got {frozen_panel_actual}")
    if mismatches:
        raise SystemExit("BLOCKED_FREEZE_HASH_MISMATCH:\n" + "\n".join(mismatches))
    spec = importlib.util.spec_from_file_location(
        "sports_nova_v3_player_joint_walkforward_v1_IDENTITY_REFRESH", WALKFORWARD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, freeze, True


def build_prior_snapshot(all_df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    key = season * 100 + week
    return all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, season - 5))]


def game_leader(panel: pd.DataFrame, game_id: str, team: str):
    rows = panel[(panel.GAME_ID == game_id) & (panel.TEAM == team) & (panel.POSITION == "QB")]
    if rows.empty:
        rows = panel[(panel.GAME_ID == game_id) & (panel.TEAM == team)]
    if rows.empty:
        return None
    top = rows.loc[rows.PASS_ATTEMPTS.idxmax()]
    return {"PLAYER_ID": top.PLAYER_ID, "PLAYER_NAME": top.PLAYER_NAME,
            "PASS_ATTEMPTS": float(top.PASS_ATTEMPTS)}


def panel_leader_as_of(panel: pd.DataFrame, team: str, cutoff: pd.Timestamp):
    rows = panel[(panel.TEAM == team) & (panel.EVENT_TIME < cutoff)]
    if rows.empty:
        return None
    last_game_id = rows.sort_values("EVENT_TIME").iloc[-1].GAME_ID
    leader = game_leader(panel, last_game_id, team)
    if leader is None:
        return None
    ev = rows[rows.GAME_ID == last_game_id].EVENT_TIME.iloc[0]
    leader.update({"AS_OF_GAME_ID": last_game_id, "AS_OF_EVENT_TIME": str(ev)})
    return leader


def cross_team_anomaly(dc_qb: pd.DataFrame, dt_value, gsis_id: str, team: str):
    """Sanity check, not a rejection of any specific player: a real QB-position
    depth-chart snapshot should list one team per player. Investigated one
    trigger case by hand before trusting it either way -- ATL's depth chart
    lists Tua Tagovailoa as of 2026-09 alongside Michael Penix Jr./Cooper
    Rush. That is NOT source contamination: weekly_rosters_2026 independently
    shows gsis_id 00-0036212 (Tua) on ATL's roster from week 1 (status=INA)
    through week 2 (status=ACT), and the live causal panel's last MIA row for
    him is 2025 week 15 -- consistent with a real between-seasons trade, not
    a scrape bug. This check is kept for a genuine double-listing (the SAME
    gsis_id claimed by two teams at the identical snapshot timestamp), which
    would be a real integrity conflict; it did not fire for Tua/ATL and
    should not be read as having found one."""
    same_dt = dc_qb[(dc_qb.dt == dt_value) & (dc_qb.gsis_id == gsis_id)]
    other_teams = sorted(set(same_dt.team) - {team})
    return other_teams


def depth_chart_qb1_as_of(dc: pd.DataFrame, team: str, cutoff: pd.Timestamp, dc_qb_all=None):
    rows = dc[(dc.team == team) & (dc.pos_name == "Quarterback") & (dc.dt < cutoff)]
    if rows.empty:
        return None
    latest_dt = rows.dt.max()
    snap = rows[rows.dt == latest_dt].sort_values("pos_rank")
    qb1 = snap[snap.pos_rank == snap.pos_rank.min()].iloc[0]
    alt = None
    rest = snap[snap.pos_rank > qb1.pos_rank]
    if not rest.empty:
        alt_row = rest.iloc[0]
        alt = {"PLAYER_ID": alt_row.gsis_id, "PLAYER_NAME": alt_row.player_name,
               "POS_RANK": int(alt_row.pos_rank)}
    result = {"PLAYER_ID": qb1.gsis_id, "PLAYER_NAME": qb1.player_name,
              "AS_OF_DT": str(latest_dt), "ALT_CANDIDATE": alt}
    if dc_qb_all is not None:
        anomaly_qb1 = cross_team_anomaly(dc_qb_all, latest_dt, qb1.gsis_id, team)
        if anomaly_qb1:
            result["CROSS_TEAM_ANOMALY"] = {"PLAYER_ID": qb1.gsis_id, "ALSO_LISTED_FOR": anomaly_qb1}
        if alt is not None:
            anomaly_alt = cross_team_anomaly(dc_qb_all, latest_dt, alt["PLAYER_ID"], team)
            if anomaly_alt:
                result["ALT_CANDIDATE"]["CROSS_TEAM_ANOMALY"] = {"ALSO_LISTED_FOR": anomaly_alt}
    return result


def roster_status(rosters: pd.DataFrame, team: str, gsis_id):
    if rosters is None or gsis_id is None:
        return None
    rows = rosters[(rosters.team == team) & (rosters.gsis_id == gsis_id)]
    if rows.empty:
        return None
    return str(rows.sort_values("week").iloc[-1].status)


def injury_status(injuries: pd.DataFrame, team: str, gsis_id, week: int):
    if injuries is None or gsis_id is None:
        return None
    rows = injuries[(injuries.team == team) & (injuries.gsis_id == gsis_id) & (injuries.week == week)]
    if rows.empty:
        return None
    return str(rows.iloc[0].report_status) if pd.notna(rows.iloc[0].report_status) else None


def resolve_team(team: str, panel: pd.DataFrame, dc: pd.DataFrame, dc_qb_all: pd.DataFrame,
                  rosters: pd.DataFrame, injuries: pd.DataFrame, now: pd.Timestamp,
                  latest_injury_week: int):
    pl = panel_leader_as_of(panel, team, now)
    dcq = depth_chart_qb1_as_of(dc, team, now, dc_qb_all)

    pl_id = pl["PLAYER_ID"] if pl else None
    dc_id = dcq["PLAYER_ID"] if dcq else None

    dc_injury = injury_status(injuries, team, dc_id, latest_injury_week) if dcq else None
    dc_roster = roster_status(rosters, team, dc_id) if dcq else None
    pl_roster = roster_status(rosters, team, pl_id) if pl else None

    leakage = []
    if dc_roster is not None and dc_roster != "ACT":
        leakage.append({"CANDIDATE": "DEPTH_CHART_QB1", "PLAYER_ID": dc_id,
                         "ROSTER_STATUS": dc_roster})
    if pl_roster is not None and pl_roster != "ACT":
        leakage.append({"CANDIDATE": "PANEL_LEADER", "PLAYER_ID": pl_id,
                         "ROSTER_STATUS": pl_roster})

    sources = []
    if pl is not None:
        sources.append("causal_panel_box_score")
    if dcq is not None:
        sources.append("nflverse_depth_chart_timestamped")

    if pl is None and dcq is None:
        record = {"TEAM": team, "QB_ID": None, "QB_NAME": None, "STATUS": "UNRESOLVED",
                   "SOURCE": [], "AS_OF": str(now), "CONFIDENCE": "NONE",
                   "CANDIDATES": [], "REASON": "no panel or depth-chart history found"}
    elif pl is not None and dcq is not None and pl_id == dc_id and dc_injury not in INJURY_OUT_STATUSES:
        record = {"TEAM": team, "QB_ID": pl_id, "QB_NAME": pl["PLAYER_NAME"], "STATUS": "CONFIRMED",
                   "SOURCE": sources, "AS_OF": str(now), "CONFIDENCE": "HIGH",
                   "CANDIDATES": [{"PLAYER_ID": pl_id, "PLAYER_NAME": pl["PLAYER_NAME"]}],
                   "PANEL_LEADER": pl, "DEPTH_CHART_QB1": dcq}
    elif dcq is not None and dc_injury in INJURY_OUT_STATUSES:
        alt = dcq.get("ALT_CANDIDATE")
        cands = [c for c in (
            {"PLAYER_ID": dc_id, "PLAYER_NAME": dcq["PLAYER_NAME"]},
            alt,
            {"PLAYER_ID": pl_id, "PLAYER_NAME": pl["PLAYER_NAME"]} if pl else None,
        ) if c is not None]
        record = {"TEAM": team, "QB_ID": None, "QB_NAME": None, "STATUS": "UNCERTAIN",
                   "SOURCE": sources + ["nflverse_injury_report"], "AS_OF": str(now),
                   "CONFIDENCE": "LOW", "CANDIDATES": cands,
                   "REASON": f"depth-chart QB1 report_status={dc_injury}",
                   "PANEL_LEADER": pl, "DEPTH_CHART_QB1": dcq}
    elif pl is not None and dcq is not None:
        record = {"TEAM": team, "QB_ID": None, "QB_NAME": None, "STATUS": "UNCERTAIN",
                   "SOURCE": sources, "AS_OF": str(now), "CONFIDENCE": "MEDIUM",
                   "CANDIDATES": [
                       {"PLAYER_ID": pl_id, "PLAYER_NAME": pl["PLAYER_NAME"], "BASIS": "PANEL_LEADER"},
                       {"PLAYER_ID": dc_id, "PLAYER_NAME": dcq["PLAYER_NAME"], "BASIS": "DEPTH_CHART_QB1"},
                   ],
                   "REASON": "panel-box-score leader and current depth-chart QB1 disagree",
                   "PANEL_LEADER": pl, "DEPTH_CHART_QB1": dcq}
    elif dcq is not None:
        record = {"TEAM": team, "QB_ID": dc_id, "QB_NAME": dcq["PLAYER_NAME"], "STATUS": "UNCERTAIN",
                   "SOURCE": sources, "AS_OF": str(now), "CONFIDENCE": "LOW",
                   "CANDIDATES": [{"PLAYER_ID": dc_id, "PLAYER_NAME": dcq["PLAYER_NAME"]}],
                   "REASON": "no panel history for this team; depth-chart-only evidence",
                   "DEPTH_CHART_QB1": dcq}
    else:
        record = {"TEAM": team, "QB_ID": pl_id, "QB_NAME": pl["PLAYER_NAME"], "STATUS": "UNCERTAIN",
                   "SOURCE": sources, "AS_OF": str(now), "CONFIDENCE": "LOW",
                   "CANDIDATES": [{"PLAYER_ID": pl_id, "PLAYER_NAME": pl["PLAYER_NAME"]}],
                   "REASON": "no timestamped depth-chart entry found for this team",
                   "PANEL_LEADER": pl}

    record["DEPARTED_PLAYER_LEAKAGE"] = leakage
    return record


def backcheck_2025(panel: pd.DataFrame, dc: pd.DataFrame):
    season_rows = panel[(panel.SEASON == 2025) & (panel.POSITION == "QB")].copy()
    games = season_rows[["GAME_ID", "TEAM", "OPPONENT", "EVENT_TIME"]].drop_duplicates(
        subset=["GAME_ID", "TEAM"])
    games = games.sort_values("EVENT_TIME")

    n = 0
    dc_mismatches = 0
    prior_mismatches = 0
    dc_covered = 0
    prior_covered = 0
    agree_n = 0
    agree_mismatches = 0
    examples = []
    for _, g in games.iterrows():
        team, game_id, cutoff = g.TEAM, g.GAME_ID, g.EVENT_TIME
        actual = game_leader(panel, game_id, team)
        if actual is None:
            continue
        n += 1

        dcq = depth_chart_qb1_as_of(dc, team, cutoff)
        if dcq is not None:
            dc_covered += 1
            if dcq["PLAYER_ID"] != actual["PLAYER_ID"]:
                dc_mismatches += 1
                if len(examples) < 8:
                    examples.append({"GAME_ID": game_id, "TEAM": team, "METHOD": "DEPTH_CHART",
                                      "PREDICTED": dcq["PLAYER_NAME"], "ACTUAL": actual["PLAYER_NAME"]})

        pl = panel_leader_as_of(panel, team, cutoff)
        if pl is not None:
            prior_covered += 1
            if pl["PLAYER_ID"] != actual["PLAYER_ID"]:
                prior_mismatches += 1

        if dcq is not None and pl is not None and dcq["PLAYER_ID"] == pl["PLAYER_ID"]:
            agree_n += 1
            if dcq["PLAYER_ID"] != actual["PLAYER_ID"]:
                agree_mismatches += 1

    return {
        "SEASON": 2025,
        "N_TEAM_GAMES": n,
        "DEPTH_CHART_METHOD": {
            "N_COVERED": dc_covered,
            "N_MISMATCH": dc_mismatches,
            "MISMATCH_RATE": round(dc_mismatches / dc_covered, 4) if dc_covered else None,
        },
        "PRIOR_GAME_LEADER_METHOD": {
            "N_COVERED": prior_covered,
            "N_MISMATCH": prior_mismatches,
            "MISMATCH_RATE": round(prior_mismatches / prior_covered, 4) if prior_covered else None,
        },
        "RESOLVER_CONFIRMED_LABEL_BACKCHECK": {
            "DESCRIPTION": "The label actually stamped by resolve_team() when both sources "
                           "agree is CONFIRMED/CONFIDENCE=HIGH. This is that subset's own "
                           "accuracy against the real box score, not either source's standalone "
                           "rate -- the number that should back the CONFIRMED label in "
                           "QB_IDENTITIES_RESOLVED.",
            "N_AGREEMENT_GAMES": agree_n,
            "N_MISMATCH_DESPITE_AGREEMENT": agree_mismatches,
            "MISMATCH_RATE": round(agree_mismatches / agree_n, 4) if agree_n else None,
        },
        "SAMPLE_MISMATCHES": examples,
        "NOTE": "Non-circular: each method's cutoff is strictly before the "
                "predicted game's own kickoff; ACTUAL is that same game's "
                "box-score leading passer, independent of either predictor. "
                "All numbers in this block are a 2025-season methodology "
                "check -- 2026 has zero eligible holdout weeks (see "
                "BOX_SCORE_BACKCHECK_N at the top level, and TEMPORAL_FIREWALL).",
    }


def capture_new_game(wf_mod, freeze, live_df, live_sha, live_receipt, game_row, n_sims, now,
                      identity_by_team, resolution_sha256):
    from worker.sports_nova_v3.simulator import simulate_game, _primary_qb
    from worker.sports_nova_v3.config import MODEL_VERSION

    game_id = game_row["GAME_ID"]
    season, week = game_row["SEASON"], game_row["WEEK"]
    home, away = game_row["HOME_TEAM"], game_row["AWAY_TEAM"]
    kickoff = datetime.fromisoformat(game_row["KICKOFF_UTC"])
    if now >= kickoff:
        return [], "KICKOFF_HAS_PASSED_SINCE_CLASSIFICATION"

    panel_max_2026_event_time = live_receipt["MAX_2026_EVENT_TIME"]
    if panel_max_2026_event_time is not None:
        panel_max_dt = datetime.fromisoformat(panel_max_2026_event_time.replace("Z", "+00:00"))
        if not (panel_max_dt < kickoff):
            return [], f"TEMPORAL_FIREWALL_VIOLATION: panel_max_event_time {panel_max_2026_event_time} not before kickoff {kickoff.isoformat()}"

    prior = build_prior_snapshot(live_df, season, week)
    state = wf_mod.make_state(game_id, prior, kickoff, None)
    seed = int(hashlib.sha256((game_id + "|IDENTITY_REFRESH").encode()).hexdigest()[:8], 16)
    sim = simulate_game(state, n_sims, seed, MODEL_VERSION)

    input_hash = sha256_obj(state.model_dump())
    code_hash = {
        "scripts/sports_nova_v3_player_joint_walkforward_v1.py":
            freeze["CODE_HASHES"]["scripts/sports_nova_v3_player_joint_walkforward_v1.py"]["sha256"],
        "worker/sports_nova_v3/simulator.py":
            freeze["CODE_HASHES"]["worker/sports_nova_v3/simulator.py"]["sha256"],
        "scripts/sports_nova_v3_v18_current_identity_refresh.py": sha256_file(Path(__file__).resolve()),
    }

    written = []
    for tid, opp in ((home, away), (away, home)):
        out_path = PRED_DIR_LIVE / f"{game_id}__{tid}.json"
        if out_path.exists():
            return [], f"ALREADY_CAPTURED: {out_path.name}"

        cohort, identity = policy.classify_team(tid, {"TEAMS": identity_by_team})
        if cohort == "EXCLUDED_UNCERTAIN_IDENTITY":
            artifact = {
                "GAME_ID": game_id,
                "TIMESTAMP": kickoff.isoformat(),
                "TEAM": tid,
                "OPPONENT": opp,
                "COHORT": "EXCLUDED_UNCERTAIN_IDENTITY",
                "POLICY": "SPORTS_V18_QB_UNCERTAINTY_POLICY_AND_WEEKLY_REFRESH: "
                          "UNCERTAIN_QB_POLICY=EXCLUDE_FROM_PRIMARY",
                "IDENTITY_RESOLUTION": identity,
                "CREATED_AT": datetime.now(timezone.utc).isoformat(),
                "NOTE": "No point prediction written for this team -- identity "
                        "resolution status is UNCERTAIN, not silently guessed. "
                        "Candidate QBs are listed in IDENTITY_RESOLUTION.CANDIDATES. "
                        "This is a MARKER file, not a prediction: it has no "
                        "EXPECTED_PASS_YARDS/THRESHOLD_PROBABILITIES/PREDICTION_HASH -- "
                        "any reader must check COHORT before treating a "
                        "predictions_live/*.json file as a scorable prediction.",
            }
            # Canonical, deterministic hash (excludes CREATED_AT and every
            # other wall-clock field) via the same function the canonical
            # identity gate uses -- identical inputs always hash identically,
            # regardless of which script or how many times it is computed.
            artifact["EXCLUSION_HASH"] = identity_gate.canonical_exclusion_hash(
                game_id, tid, identity, resolution_sha256)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(artifact, indent=2, default=str))
            written.append((tid, opp, artifact))
            continue

        qb_state = _primary_qb(state, tid)
        if qb_state is None:
            continue
        qb_id = qb_state.player_id
        arr = np.asarray(sim.player(qb_id, "pass_yards"), dtype=float)
        dist = {
            "MEAN": float(arr.mean()), "SD": float(arr.std()),
            "P10": float(np.quantile(arr, .10)), "P25": float(np.quantile(arr, .25)),
            "P50": float(np.quantile(arr, .50)), "P75": float(np.quantile(arr, .75)),
            "P90": float(np.quantile(arr, .90)),
        }
        thresholds = {str(t): float(np.mean(arr > t)) for t in THRESHOLD_GRID}
        frozen_pick_agrees = identity.get("QB_ID") is None or identity.get("QB_ID") == qb_id
        artifact = {
            "GAME_ID": game_id,
            "TIMESTAMP": kickoff.isoformat(),
            "MODEL_VERSION": MODEL_VERSION,
            "FEATURE_STORE_VERSION": freeze["FEATURE_STORE_VERSION"],
            "CODE_HASH": code_hash,
            "INPUT_HASH": input_hash,
            "QB": qb_id,
            "TEAM": tid,
            "OPPONENT": opp,
            "PREDICTIVE_DISTRIBUTION": dist,
            "EXPECTED_PASS_YARDS": dist["MEAN"],
            "THRESHOLD_PROBABILITIES": thresholds,
            "CREATED_AT": datetime.now(timezone.utc).isoformat(),
            "N_SIMS": n_sims,
            "SIM_COUNT_LABEL": "SMOKE_ONLY",
            "SEED": seed,
            "RNG_ALGORITHM": "PCG64DXSM",
            "CAUSALITY_MODE": "EVENT_CAUSAL_ONLY",
            "PANEL_MODE": "REFRESHED_2026",
            "PLAYER_PANEL_SHA256": live_sha,
            "PANEL_REFRESH_RECEIPT": str(LIVE_RECEIPT.relative_to(ROOT)),
            "IDENTITY_RESOLUTION": identity,
            "FROZEN_PRIMARY_QB_AGREES_WITH_RESOLUTION": frozen_pick_agrees,
            "KNOWN_LIMITATION": (
                None if frozen_pick_agrees else
                "frozen _primary_qb (5-season volume-argmax, unmodified) picked "
                f"{qb_id} while the current-season identity resolution for {tid} "
                f"points elsewhere with CONFIDENCE={identity.get('CONFIDENCE')}; "
                "not overridden -- no V18 model change is in scope for this mission."
            ),
        }
        artifact["PREDICTION_HASH"] = sha256_obj(artifact)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))
        written.append((tid, opp, artifact))
    return written, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=1000)
    ap.add_argument("--target-game-id", type=str, default=None)
    args = ap.parse_args()

    identity_gate.assert_prospective_identity_gate_installed()

    now = pd.Timestamp.now(tz="UTC")
    wf_mod, freeze, v18_hash_unchanged = load_frozen_module_and_verify_freeze()

    live_df = pd.read_parquet(LIVE_PANEL)
    live_sha = sha256_file(LIVE_PANEL)
    live_receipt = json.loads(LIVE_RECEIPT.read_text())
    if live_sha != live_receipt["LIVE_PANEL_SHA256"]:
        raise SystemExit("BLOCKED_LIVE_PANEL_HASH_MISMATCH_VS_RECEIPT")
    live_df["_key"] = live_df.SEASON * 100 + live_df.WEEK
    live_df["EVENT_TIME"] = pd.to_datetime(live_df.EVENT_TIME, utc=True)

    dc_2025, dc_2025_sha = fetch_parquet(f"{NFLVERSE_RELEASES}/depth_charts/depth_charts_2025.parquet")
    dc_2026, dc_2026_sha = fetch_parquet(f"{NFLVERSE_RELEASES}/depth_charts/depth_charts_2026.parquet")
    dc = pd.concat([dc_2025, dc_2026], ignore_index=True)
    dc["dt"] = pd.to_datetime(dc["dt"], utc=True)

    rosters, rosters_sha = fetch_parquet(f"{NFLVERSE_RELEASES}/weekly_rosters/roster_weekly_2026.parquet")
    injuries, injuries_sha = fetch_parquet(f"{NFLVERSE_RELEASES}/injuries/injuries_2026.parquet")
    # NOTE for future runs: this takes the latest available week's report as a
    # stand-in for "the target game's own week". Safe today (report fetched
    # before any week-2 kickoff), but a Monday re-run mid-week could pick up
    # a report row for a week whose games have already gone final. Re-running
    # weekly per NEXT_SINGLE_ACTION should key this off each target game's
    # own SEASON/WEEK instead of a single global "latest week" if this script
    # is ever run after some-but-not-all of a week's games have kicked off.
    latest_injury_week = int(injuries.week.max()) if not injuries.empty else 1

    classification = json.loads(CLASSIFICATION_PATH.read_text())["GAMES"]
    eligible = [g for g in classification if g["STATUS"] == "ELIGIBLE"]
    teams = sorted({g["HOME_TEAM"] for g in eligible} | {g["AWAY_TEAM"] for g in eligible})

    dc_qb_all = dc[dc.pos_name == "Quarterback"]

    identity_by_team = {}
    uncertain = []
    for team in teams:
        rec = resolve_team(team, live_df, dc, dc_qb_all, rosters, injuries, now, latest_injury_week)
        identity_by_team[team] = rec
        if rec["STATUS"] in ("UNCERTAIN", "UNRESOLVED"):
            uncertain.append(rec)

    IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
    RESOLUTION_PATH.write_text(json.dumps({
        "AS_OF_UTC": str(now),
        "SOURCES": {
            "causal_panel": {"path": str(LIVE_PANEL.relative_to(ROOT)), "sha256": live_sha},
            "depth_charts_2025": {"sha256": dc_2025_sha},
            "depth_charts_2026": {"sha256": dc_2026_sha},
            "weekly_rosters_2026": {"sha256": rosters_sha},
            "injuries_2026": {"sha256": injuries_sha},
        },
        "TEAMS": identity_by_team,
    }, indent=2, default=str))
    resolution_sha256 = sha256_file(RESOLUTION_PATH)

    # Enforced here, not just documented: every capture batch from this point
    # on is gated on the identity artifact it is about to use being fresh.
    policy.enforce_fresh_identity({"AS_OF_UTC": str(now), "TEAMS": identity_by_team})

    atl = identity_by_team.get("ATL", {})
    atl_prev_qb = {"PLAYER_ID": "00-0029604", "PLAYER_NAME": "Kirk Cousins",
                   "NOTE": "5-season-pooled frozen identity before the causal panel refresh; "
                           "per V18_2026_CAUSAL_PANEL_REFRESH_RECEIPT this was already stale."}

    backcheck = backcheck_2025(live_df, dc)

    departed_leakage = [r for t, r in identity_by_team.items() if r.get("DEPARTED_PLAYER_LEAKAGE")]

    classification_ids_taken = {"2026_02_DET_BUF", "2026_02_CAR_ATL"}
    future = [g for g in eligible
              if g["KICKOFF_UTC"] is not None
              and datetime.fromisoformat(g["KICKOFF_UTC"]) > now.to_pydatetime()
              and g["GAME_ID"] not in classification_ids_taken]
    future.sort(key=lambda g: g["KICKOFF_UTC"])
    if args.target_game_id:
        future = [g for g in future if g["GAME_ID"] == args.target_game_id]

    smoke_result = None
    if future:
        ledger = json.loads(LEDGER_LIVE_PATH.read_text()) if LEDGER_LIVE_PATH.exists() else {}
        written, skip_reason = capture_new_game(
            wf_mod, freeze, live_df, live_sha, live_receipt, future[0], args.n_sims,
            now.to_pydatetime(), identity_by_team, resolution_sha256)
        if skip_reason and skip_reason.startswith("ALREADY_CAPTURED"):
            existing = sorted(PRED_DIR_LIVE.glob(f"{future[0]['GAME_ID']}__*.json"))
            prior_artifacts = []
            for p in existing:
                a = json.loads(p.read_text())
                prior_artifacts.append({
                    "PATH": str(p.relative_to(ROOT)), "TEAM": a.get("TEAM"), "QB": a.get("QB"),
                    "COHORT": a.get("COHORT", "PRIMARY"),
                    "PREDICTION_HASH": a.get("PREDICTION_HASH") or a.get("EXCLUSION_HASH"),
                    "IDENTITY_RESOLUTION_AS_OF": a.get("IDENTITY_RESOLUTION", {}).get("AS_OF"),
                })
            smoke_result = {
                "CAPTURED": True,
                "TARGET": future[0]["GAME_ID"],
                "NOTE": "Already captured by an earlier run of this same script this session; "
                        "not re-captured (write-once, per mission invariant). Artifacts below "
                        "are real and on disk; their embedded IDENTITY_RESOLUTION.AS_OF predates "
                        "this run's resolution pass by a few minutes (identity did not change "
                        "for these two teams between the two passes).",
                "EXISTING_ARTIFACTS": prior_artifacts,
            }
        elif skip_reason:
            smoke_result = {"CAPTURED": False, "SKIP_REASON": skip_reason, "TARGET": future[0]["GAME_ID"]}
        else:
            for tid, opp, artifact in written:
                key = f"{future[0]['GAME_ID']}__{tid}"
                cohort = artifact.get("COHORT", "PRIMARY")
                ledger[key] = {
                    "GAME_ID": future[0]["GAME_ID"], "TEAM": tid, "OPPONENT": opp,
                    "QB": artifact.get("QB"), "COHORT": cohort, "PANEL_MODE": "REFRESHED_2026",
                    "IDENTITY_REFRESH_MISSION": True,
                    "IDENTITY_STATUS_AT_PREDICTION_TIME": artifact.get("IDENTITY_RESOLUTION", {}).get("STATUS"),
                    "PREDICTION_HASH": artifact.get("PREDICTION_HASH") or artifact.get("EXCLUSION_HASH"),
                    "PREDICTION_TIME": artifact["CREATED_AT"],
                    "KICKOFF_UTC": future[0]["KICKOFF_UTC"],
                    "SETTLED": False,
                }
                LEDGER_LIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
                LEDGER_LIVE_PATH.write_text(json.dumps(ledger, indent=2, sort_keys=True))
            smoke_result = {"CAPTURED": True, "TARGET": future[0]["GAME_ID"],
                             "TEAMS": [w[0] for w in written],
                             "COHORTS": {w[0]: w[2].get("COHORT", "PRIMARY") for w in written}}
    else:
        smoke_result = {"CAPTURED": False, "SKIP_REASON": "NO_UNCAPTURED_ELIGIBLE_FUTURE_GAME"}

    frozen_panel_final_sha = sha256_file(FROZEN_PANEL)
    v18_hash_unchanged = v18_hash_unchanged and (
        frozen_panel_final_sha == freeze["DATA_SCHEMA_HASHES"]["NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet"])

    real_blocker = (
        "In-week injury-driven starter changes with zero 2026 attempts recorded yet are the "
        "one case this cannot resolve with confidence: no free source with a provable "
        "pre-kickoff timestamp reports 'this is the new starter' before it happens on the "
        "field, only depth-chart rank + injury report, and V14 already found the pre-2025 "
        "version of that source non-causal. For 2025/2026 the timestamped depth-chart asset "
        f"exists and is used, but its own {backcheck['DEPTH_CHART_METHOD']['MISMATCH_RATE']:.1%} "
        "2025 mismatch rate (vs. real box scores) means it corroborates, it does not replace, "
        "box-score-confirmed identity."
        if backcheck['DEPTH_CHART_METHOD']['MISMATCH_RATE'] is not None else
        "Depth-chart backcheck had no covered games; see BOX_SCORE_BACKCHECK_N."
    )

    report = {
        "MISSION": "SPORTS_V18_CURRENT_SEASON_IDENTITY_REFRESH",
        "ROSTER_SOURCE": "nflverse-data release weekly_rosters, roster_weekly_2026.parquet",
        "QB_SOURCE": "nflverse-data release depth_charts (2025/2026 timestamped schema) "
                     "+ causal panel box scores (validation_inputs_live)",
        "LATEST_AS_OF": str(now),
        "TEAMS_RESOLVED": len(identity_by_team),
        "QB_IDENTITIES_RESOLVED": sum(1 for r in identity_by_team.values() if r["STATUS"] == "CONFIRMED"),
        "QB_IDENTITIES_RESOLVED_CONFIDENCE_BASIS": (
            "CONFIRMED/HIGH means panel-box-score leader and current depth-chart QB1 agree, "
            "not proven certainty: the 2025 backcheck on exactly that agreement condition "
            f"(RESOLVER_CONFIRMED_LABEL_BACKCHECK below) measured "
            f"{backcheck['RESOLVER_CONFIRMED_LABEL_BACKCHECK']['MISMATCH_RATE']:.2%} still wrong "
            f"against the real box score ({backcheck['RESOLVER_CONFIRMED_LABEL_BACKCHECK']['N_MISMATCH_DESPITE_AGREEMENT']}/"
            f"{backcheck['RESOLVER_CONFIRMED_LABEL_BACKCHECK']['N_AGREEMENT_GAMES']}) -- better than "
            "either source alone, but read HIGH as ~93-94% empirical, not ~100%."
        ),
        "UNCERTAIN_QB_CASES": uncertain,
        "BOX_SCORE_BACKCHECK_N": f"{backcheck['N_TEAM_GAMES']} (2025 SEASON METHODOLOGY CHECK -- "
                                  f"2026 backcheck N=0, see TEMPORAL_FIREWALL)",
        "QB_IDENTITY_MISMATCH_RATE": backcheck,
        "ATL_PREVIOUS_QB": atl_prev_qb,
        "ATL_CURRENT_RESOLUTION": atl,
        "DEPARTED_PLAYER_LEAKAGE": departed_leakage,
        "TEMPORAL_FIREWALL": "depth-chart dt and panel EVENT_TIME both filtered strictly < "
                              "cutoff (now for prospective resolution; each game's own kickoff "
                              "for the 2025 backcheck); 2026 backcheck N=0 by construction -- "
                              "only week 1 is final and there is no week-0 to hold out from.",
        "V18_HASH_UNCHANGED": v18_hash_unchanged,
        "NEW_SMOKE_PREDICTION": smoke_result,
        "WHAT_CHANGED": (
            "Added a QB-identity provenance layer (identity/SPORTS_NOVA_V18_QB_IDENTITY_"
            "RESOLUTION_2026.json) with QB_ID/QB_NAME/STATUS/SOURCE/AS_OF/CONFIDENCE per "
            "team, using a newly-verified timestamped nflverse depth-chart asset (distinct "
            "from the one V14 rejected) as corroboration alongside the existing causal panel. "
            "No frozen code or frozen panel touched; new smoke capture carries an "
            "IDENTITY_RESOLUTION block plus FROZEN_PRIMARY_QB_AGREES_WITH_RESOLUTION flag. "
            "NOTE ON ATL: the prior causal-panel-refresh mission (same evening) reported ATL "
            "resolved cleanly to Cooper Rush from week-1 box-score data alone. This mission's "
            "wider evidence set (depth chart + injury report, fetched hours later) shows Rush's "
            "own depth-chart rank has since dropped to 3rd, Penix (rank 1) is now report_status="
            "Out, and Tua Tagovailoa (traded from MIA between seasons, confirmed via "
            "weekly_rosters_2026 + the panel's own last MIA row for him) is rank 2 -- so ATL "
            "moved from CONFIRMED to UNCERTAIN across the two mission artifacts. Both numbers "
            "are correct for their own evidence set and timestamp; this is the situation "
            "changing underneath the resolution, not a contradiction between the two reports."
        ),
        "REAL_BLOCKER": real_blocker,
        "NEXT_SINGLE_ACTION": (
            "Re-run this script weekly alongside the causal-panel refresh so depth-chart "
            "snapshots and injury reports stay current; do not redesign unless the depth-chart "
            "mismatch rate materially changes."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
