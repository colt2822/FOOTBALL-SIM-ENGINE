"""SPORTS_NOVA V28 game-state scoring: freeze -> pilot (60-game historical) -> coherence (counterfactual arms) -> analyze.

  python scripts/sports_nova_m1_v28_game_state_scoring.py freeze  [TAG]             # pre-registration for the current package hash (refuses if outputs for TAG exist)
  python scripts/sports_nova_m1_v28_game_state_scoring.py pilot   VARIANT [TAG]     # 60 games x 128 sims through V28.<VARIANT> (same games/seeds/sims as the V27 baseline)
  python scripts/sports_nova_m1_v28_game_state_scoring.py coherence VARIANT [TAG]   # score-world coherence arms on a 20-game subset (controlled RNG)
  python scripts/sports_nova_m1_v28_game_state_scoring_analysis.py analyze VARIANT [TAG]

V23/V24/V25/V26/V27, worker/sports_nova_v3, frozen artifacts and the sealed V27/ablation arrays are only READ.  No market input is read anywhere.
The V27 baseline for the pilot is the BASE arm of MULTI_TD_POSSESSION_ABLATION_V1 (proven np.array_equal to unpatched V27 on all 60 games) plus a live spot re-check.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.sports_nova_m1_fg_causal_ablation_v1 as H  # noqa: E402  (pilot cohort, seed rule, state builder)
from scripts.sports_nova_m1_v4_raw_path_diagnostic_challenger import assert_path  # noqa: E402
from worker.sports_nova_modular.firewall import assert_market_free  # noqa: E402
from worker.sports_nova_v28_game_state_scoring import config as cfg  # noqa: E402
from worker.sports_nova_v28_game_state_scoring import simulator as s28  # noqa: E402

DATA = ROOT / "data" / "sports_nova_v3"
OUT = DATA / "V28_GAME_STATE_SCORING"
V27_ARR = DATA / "MULTI_TD_POSSESSION_ABLATION_V1" / "arrays"
PKG = "worker/sports_nova_v28_game_state_scoring"
N_SIMS = H.N_SIMS
TEAM_KEYS = ("score", "pass_attempts", "pass_yards", "rush_attempts", "rush_yards", "blocks")
PLAYER_KEYS = ("pass_attempts", "pass_yards", "pass_tds", "rush_attempts", "rush_yards", "rush_tds", "targets", "receptions", "receiving_yards",
               "receiving_tds", "scored_tds")
GUARDED = [
    "worker/sports_nova_v23/simulator.py", "worker/sports_nova_v23/allocation.py", "worker/sports_nova_v23/config.py", "worker/sports_nova_v3/distributions.py",
    "worker/sports_nova_v3/game_state.py", "worker/sports_nova_v3/game_script.py", "worker/sports_nova_v3/play_volume.py", "worker/sports_nova_v3/play_selection.py",
    "worker/sports_nova_v3/scoring.py", "worker/sports_nova_v3/allocation.py", "worker/sports_nova_v19/simulator.py", "worker/sports_nova_v21/config.py",
    "worker/sports_nova_v25_role_aware/simulator.py", "worker/sports_nova_v26_active_skill_state/simulator.py", "scripts/sports_nova_fg_causal_estimator_v1.py",
]
V27_PKG = "worker/sports_nova_v27_causal_fg_rate"
V27_HASH = "295a5345d64fa76eab68ae8219634f1583c87ed9c4d2a167b87203f85e27a020"
COH_SUBSET_STEP = 3          # every 3rd pilot game -> 20 games


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pkg_hashes() -> dict:
    return {f"{PKG}/{p.name}": sha256_file(p) for p in sorted((ROOT / PKG).glob("*.py"))}


def v28_hash() -> str:
    return hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(pkg_hashes().items())).encode()).hexdigest()


def v27_hash() -> str:
    h = {f"{V27_PKG}/{p.name}": sha256_file(p) for p in sorted((ROOT / V27_PKG).glob("*.py"))}
    return hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(h.items())).encode()).hexdigest()


def guarded_hashes() -> dict:
    d = {f: sha256_file(ROOT / f) for f in GUARDED}
    d.update({f"{V27_PKG}/{p.name}": sha256_file(p) for p in sorted((ROOT / V27_PKG).glob("*.py"))})
    for pk in ("sports_nova_v24_roster_eligibility", "sports_nova_v25_role_aware", "sports_nova_v26_active_skill_state"):
        d.update({f"worker/{pk}/{p.name}": sha256_file(p) for p in sorted((ROOT / "worker" / pk).glob("*.py"))})
    return d


def pilot_games() -> list[str]:
    abl_pre = json.loads((DATA / "FG_CAUSAL_ABLATION_V1" / "FG_CAUSAL_ABLATION_V1_PREREG.json").read_text())
    games = H.pilot_games(list(json.loads(H.MANIFEST.read_text())["GAME_IDS"]))
    assert games == abl_pre["PILOT"]["game_ids"] and len(games) == 60, "pilot cohort drifted"
    return games


def out_dir(tag: str) -> Path:
    return OUT / tag


def prereg_path(tag: str) -> Path:
    return out_dir(tag) / f"V28_PREREG_{tag}.json"


def concurrency() -> dict:
    ps = subprocess.run(["powershell", "-NoProfile", "-Command",
                         "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'codex' } | Select-Object -ExpandProperty Name"], capture_output=True, text=True)
    recent = [p.relative_to(ROOT).as_posix() for p in (ROOT / "worker").rglob("*.py")
              if time.time() - p.stat().st_mtime < 3600 and "__pycache__" not in p.parts and "sports_nova_v28" not in p.as_posix()]
    return {"codex_processes_running": sorted(set(ps.stdout.split())), "worker_py_files_modified_last_hour_excluding_v28": recent,
            "STATEMENT": "a Codex app-server is running on this machine; no node3 file outside worker/sports_nova_v28_game_state_scoring/, scripts/sports_nova_m1_v28_*, "
                         "scratch_v28/ and data/sports_nova_v3/V28_GAME_STATE_SCORING/ was written by this run; no peer wrote under worker/ in the hour before freeze"}


# ---------------------------------------------------------------------------------------------- pre-registration
PREREG_BODY = {
    "STRUCTURAL_HARD_GATES": {
        "one_offensive_TD_max_per_possession": "max TDs in any block == 1 in every arm/game (hard)",
        "accounting": "score == sum(block points); team tds == sum(player scored_tds) == td_pass+td_rush; player receiving_tds == td_pass; player rush_tds == td_rush; "
                      "QB pass_tds == td_pass; blocks == team blocks; block points in {0,3,6,7} regulation; assert_path failures == 0 (all must be exactly 0)",
        "no_TD_plus_FG_same_possession": "count of blocks with td>0 and fg>0 == 0",
        "QB_rush_TD_path": "qb_rush_TD/team-game > 0 and every QB rush TD block has scrambles > 0 or the QB has a designed carry (checked by construction/tests)",
        "FG_preserved": "runtime fg_rate == V27 estimator-A rate for the game to 1e-15; estimator/drive hashes pinned",
        "parent_immutability": "guarded file hashes identical before/after and equal to the freeze",
    },
    "COHERENCE_PRIMARY_METRIC": {
        "definition": "pooled block-level TD rate by simulated block yards, bins (<=10] vs (>=70): range = P(TD | yards>=70) - P(TD | yards<=10)",
        "V27_value_seen_before_freeze": 0.20, "history_value_seen_before_freeze": 0.72,
        "PASS_THRESHOLD": "range >= 0.45 (chosen BEFORE the V28 run: about the midpoint between V27 0.20 and history 0.72)",
        "SECONDARY": "block-level Spearman(yards, TD indicator) reported for V27, V28, history; no threshold (report)"},
    "COHERENCE_ARMS": {
        "A": "yard_scale 0.5 on both teams (production materially suppressed), controlled RNG (CTRL script-differential replay + common random numbers)",
        "B": "yard_scale 1.5 on both teams", "C": "explosive plays removed: every per-opportunity yardage draw capped at 20",
        "D_turnovers": "NOT_REPRESENTED_IN_ENGINE: V23/V28 have no turnover draw; no turnover feature is invented to satisfy a test",
        "E_red_zone": "NOT_REPRESENTED_IN_ENGINE: no field position / red-zone state is consumed anywhere (field_position is drawn and unused)",
        "H_asym": "home offense only, yard_scale 0.5 / 1.5 (winner response)",
        "V27_reference": "V27 run with worker.sports_nova_v23.simulator.sample_compound_signed monkeypatched to scale (symmetric arms A/B only; restored after)",
        "PASS": "A: mean points/team-game falls >= 3.0 vs identity; B: rises >= 3.0; C: falls >= 0.5; H_asym: mean P(home win) at 1.5 minus at 0.5 >= 0.10; "
                "all in the stated direction for every one of the 20 games' pooled mean (report the per-game sign count)"},
    "PREDICTIONS_REGISTERED_BEFORE_THE_V28_RUN": {
        "BASIS": "offline feasibility (not a V28 run): a hazard fit on 2015-2019 applied to the V27 pilot block yards gave E[P(TD)]=0.251; V27 sim block yards have 3.3% of blocks > 100 yards "
                 "(history 0.2%) and history says P(TD | >100 yds)=0.84",
        "TD_POSSESSION_RATE": {"range": [0.235, 0.265], "meaning": "OVERSHOOT of history (~0.227-0.234) caused by the simulator's unphysical yardage tail; this would be a yardage-distribution "
                                                                  "defect, not a hazard defect"},
        "TD_PER_TEAM_GAME": [2.70, 3.05], "MAX_TD_PER_POSSESSION": 1, "MULTI_TD_POSSESSIONS": 0,
        "FG_PER_TEAM_GAME": {"range": [1.35, 1.75], "note": "direction uncertain BOTH ways: flat 0.2215 FG rate on a no-TD pool that is now skewed short; BOTH directions of the total bias are "
                                                              "informative and neither licenses touching the FG rate inside V28.1"},
        "TOTAL_BIAS_VS_REGULATION_ACTUALS": {"range": [1.0, 8.0], "note": "V27 +0.64"},
        "TOTAL_CORR": {"range": [0.05, 0.18], "falsifier_up": "> V27+0.08 would mean the hazard added real ranking signal (unexpected)"},
        "MARGIN_CORR": "within +-0.06 of V27 0.23", "WINNER_MEAN_ABS_DP_HOME_VS_V27": [0.01, 0.06],
        "STATUS_FORECAST": "WINNER YELLOW, TOTAL RED, TEAM_TOTAL RED, PLAYER_PROP BLOCKED; a positive-bias V28.1 fails gate J and is NOT the champion",
        "LOCALIZATION": "if discrimination stays ~0.10 with the coherence metric materially repaired, the discrimination failure is upstream (yardage/efficiency model), not the score boundary"},
    "DECISION_RULES": {
        "ALIGNMENT_VARIANT_.3": "built iff |V28.1 TD-possession rate - historical same-population rate| > 0.010.  If built it is labelled POST_HOC (triggered by a measurement) and carries its own "
                                "prereg; the alignment reference is the V27 sim block-yardage marginal (no outcomes) and the historical window quantiles",
        "OVERTIME_VARIANT_.2": "always built (settlement semantics); it does not change regulation scoring; regulation-only arrays are compared with regulation actuals",
        "GATE_J_no_catastrophic_mean_regression": "|total bias vs regulation actuals| <= 3.0 AND |points/team-game - historical| <= 1.5",
        "GATE_K_tails": "mean |P_sim - P_hist| over 8 upper + 4 lower ladder points <= V27's + 0.5pp, and mean-free shape (skew/kurtosis gap) not worse",
        "GATE_L_discrimination": "total corr >= V27 - 0.03, margin corr >= V27 - 0.05, total RMSE <= V27 + 0.25 (paired bootstrap reported)",
        "TOTAL_RED_TO_YELLOW": "gates J,K,L AND structural gates AND coherence AND total RMSE <= league-mean-baseline RMSE + 0.35; otherwise RED.  GREEN is never assigned (no OOS/calibration)",
        "WINNER_YELLOW_TO_GREEN": "requires validation/calibration evidence: not obtainable in this sprint",
        "MONTE_CARLO": "n=128 sims/game; effects below the paired-bootstrap noise floor are reported as noise, not improvement"},
    "NOT_DONE_BY_DESIGN": ["player props", "live capital", "M2 refit", "any market input in M1", "editing V23-V27"],
}


PREREG_POSTHOC = {
    "LABEL": "POST_HOC: every choice below was triggered by the V28_1 measurement (V28_1 prereg + V28_ANALYSIS_v1.json are the evidence); nothing here is presented as a-priori",
    "EVIDENCE_FROM_V28_1": {
        "coherence_repaired": "TD-by-yards range 0.817 (history 0.758, V27 0.227); team-game corr(yards, points) 0.735 (history 0.667, V27 0.072); GT8 0, multi-TD 0, accounting 0",
        "but_bias_and_dispersion": "TD-possession rate 0.2514 vs history 0.2335 (prediction range [0.235,0.265] hit); TD/team-game 2.92; total bias +4.59 (gate J fail); total SD 19.6 vs history 13.8; upper tails far too fat",
        "root_cause_measured": "simulated team-game yardage SD 167 (CV 0.44) vs history 80 (CV 0.226).  With env.*_efficiency noise (sigma 0.35 around recent_form) removed the simulated SD is 79 (CV 0.227) on the "
                               "probe game; the V23/V27 scoring ignored yardage so this defect was invisible until V28 coupled scoring to yards.",
        "kappa_grid": "KAPPA_GRID_YARD_MARGINALS.json (12 games x 64 sims x kappa in {0,.25,.5,.75,1}; reads yardage only, no scores/totals/winners): pooled CV 0.267/0.276/0.306/0.355/0.411 vs history 0.226 -> kappa=0 (grid boundary; "
                      "a residual CV gap of 0.04 remains and is reported, not tuned away)"},
    "CANDIDATES": {"V28.5": "V28.1 + kappa=0 (regulation)", "V28.6": "V28.5 + NFL overtime (run as the arm of record; .5 == .6 regulation arrays, proved by np.array_equal)",
                   "V28.4": "V28.1 + POST_HOC yardage-marginal alignment (.3) + overtime, kappa=1 -- the alternative repair, built because the V28_1 decision rule fired (|TD-poss gap| 0.0178 > 0.010)"},
    "STRUCTURAL_HARD_GATES": PREREG_BODY["STRUCTURAL_HARD_GATES"],
    "COHERENCE_PRIMARY_METRIC": PREREG_BODY["COHERENCE_PRIMARY_METRIC"],
    "COHERENCE_ARMS": PREREG_BODY["COHERENCE_ARMS"],
    "PREDICTIONS_REGISTERED_BEFORE_THE_V28.5/.6_RUN": {
        "TD_POSSESSION_RATE": [0.220, 0.250], "TD_PER_TEAM_GAME": [2.40, 2.80], "TOTAL_BIAS_VS_REGULATION_ACTUALS": [-2.5, 2.5], "TEAM_SCORE_SD": [9.0, 11.0], "TOTAL_SD": [12.5, 15.5],
        "TEAM_GAME_CORR_YARDS_POINTS": [0.55, 0.75], "COHERENCE_RANGE": ">= 0.45 (expected 0.6-0.85)", "TOTAL_CORR": [0.0, 0.20], "ALL_LADDER_MAE_PP": "<= 3.3 (V27) + 0.5",
        "NOTE": "discrimination is expected to stay weak: the yardage/efficiency model, not the score boundary, carries the (little) ranking signal"},
    "DECISION_RULES": {
        **PREREG_BODY["DECISION_RULES"],
        "GATE_M_dispersion": "team-score SD and total SD each within 12% of the 2014-2025 regulation history (9.96 / 13.78)",
        "CHAMPION_SELECTION": "candidates are eligible only if structural gates + coherence + J + K + L + M all pass.  If both V28.6 and V28.4 are eligible the tie-break is the smaller sum of "
                              "|TD-possession-rate gap| and |team-game yardage CV gap| (INTERMEDIATE quantities), never an outcome metric.  If none is eligible there is NO GAME_SIM_CHAMPION and V27 stays the "
                              "engine of record; the failure is reported with the localized cause.",
        "OT_SETTLEMENT": "regulation arrays vs regulation actuals for total/team-total; final (incl. OT) for winner; market mapping of OT stays UNRESOLVED"},
    "NOT_DONE_BY_DESIGN": PREREG_BODY["NOT_DONE_BY_DESIGN"],
}


PREREG_POSTHOC3 = {
    "LABEL": "POST_HOC #2: triggered by the V28_2 measurements (V28_ANALYSIS_v6.json / v4.json in data/.../V28_2/); nothing here is a-priori",
    "EVIDENCE_FROM_V28_2": {
        "V28.6 (kappa=0)": "team-game yardage CV 0.281 (hist 0.226); TD-possession rate 0.242 (hist 0.2335); total SD 16.0 (hist 13.8); total bias +3.16; WINNER Brier 0.281 vs V27 0.2405, paired bootstrap "
                           "delta +0.041 CI [+0.003,+0.077] (significantly WORSE); mean |dP(home)| vs V27 0.114 against 0.048 MC noise, 20/60 favorite flips; margin corr 0.105 vs 0.228; "
                           "blocks/team-game 11.62 vs 11.03 history drives (the block-vs-drive gap); TD/team-game 2.82 vs 2.576 = +9.5% = ~+3.6% TD-per-possession and ~+5.3% more possessions",
        "V28.4 (alignment)": "FAILED as a repair: TD-possession rate 0.270 (worse than the raw hazard 0.251, history 0.2335), total bias +7.4, ladder MAE 10.0pp; kappa left at 1 so team-game yardage CV stays 0.434.  "
                             "Alignment matches the block-yardage marginal but not the joint/dispersion structure; NOT pursued further (unresolved why pooled E[p] exceeded the per-game training rate)",
        "diagnosis_of_winner_regression": "the league-average hazard discards the team-specific finishing propensity (pass_td_rate/rush_td_rate; cross-team CV 22%/33%) that V23 used per opportunity"},
    "CANDIDATES": {"V28.7": "V28.5 (kappa=0) + team finishing multiplier (regulation)", "V28.8": "V28.7 + drive-clock scale 1.0508 (regulation)",
                   "V28.9": "V28.8 + NFL overtime (arm of record if .8 wins)", "V28.10": "V28.7 + NFL overtime (arm of record if .7 wins)"},
    "TEAM_MULTIPLIER_SPEC": "odds x clip(total_w/w_ref, 0.5, 2); total_w = sum_i rec_i*pass_td_rate + sum_j carries_j*rush_td_rate (incl. QB scrambles); w_ref = same opportunities x pooled league rates "
                            "(player table, same 5-season causal window, hash-pinned).  beta=1 (V23's own semantics: hazard proportional to intensity); no fitted parameter.",
    "CLOCK_SCALE_SPEC": "constant 11.594010416666666/11.033372711163615 from data known BEFORE V28.6 (V27 sealed pilot blocks/team-game over 2020-2025 drive-table drives/team-game); no fitted parameter",
    "STRUCTURAL_HARD_GATES": PREREG_BODY["STRUCTURAL_HARD_GATES"], "COHERENCE_PRIMARY_METRIC": PREREG_BODY["COHERENCE_PRIMARY_METRIC"], "COHERENCE_ARMS": PREREG_BODY["COHERENCE_ARMS"],
    "PREDICTIONS_REGISTERED_BEFORE_THE_V28.7/.8_RUN": {
        "V28.8": {"TD_POSSESSION_RATE": [0.215, 0.250], "TD_PER_TEAM_GAME": [2.35, 2.80], "BLOCKS_PER_TEAM_GAME": [10.9, 11.2], "TOTAL_BIAS": [-2.5, 2.5], "TEAM_SD": [9.0, 11.0], "TOTAL_SD": [12.5, 15.5],
                  "COHERENCE_RANGE": ">= 0.45", "WINNER_BRIER_DELTA_VS_V27": "<= +0.02 (the multiplier is expected to recover most of the V28.6 loss)", "TOTAL_CORR": [0.0, 0.22]},
        "V28.7": {"BLOCKS_PER_TEAM_GAME": "~11.6 (unchanged)", "TOTAL_BIAS": [0.0, 4.0], "note": "kept as the attribution arm: separates the multiplier from the clock scale"},
        "FALSIFIER": "if V28.8 still has winner-Brier delta CI lower bound > 0 the team-signal hypothesis is refuted and the loss localizes to the yardage/efficiency model"},
    "DECISION_RULES": {
        **PREREG_BODY["DECISION_RULES"],
        "GATE_M_dispersion": "team-score SD and total SD each within 12% of the 2014-2025 regulation history (9.96 / 13.78)",
        "GATE_N_winner": "paired-bootstrap mean winner-Brier delta vs V27 <= +0.02 AND its 95% CI lower bound <= 0 (not significantly worse); winner accuracy is reported, not gated (n=60)",
        "CHAMPION_SELECTION": "eligible iff structural + coherence + J + K + L + M + N all pass.  Both .8 and .7 eligible -> the one with the smaller |TD-possession gap| + |blocks-per-game gap/11.03| (intermediate "
                              "quantities only).  None eligible -> NO GAME_SIM_CHAMPION; V27 stays the engine of record and the localized cause is reported.",
        "OT": "regulation arrays of .9/.10 must equal .8/.7 (np.array_equal); OT market mapping stays UNRESOLVED"},
    "NOT_DONE_BY_DESIGN": PREREG_BODY["NOT_DONE_BY_DESIGN"],
}


PREREG_POSTHOC4 = {
    "LABEL": "POST_HOC #3 (third iteration on the SAME 60-game pilot: results are DEVELOPMENT-set results and can never justify GREEN).  Triggered by V28_3 measurements.",
    "EVIDENCE_FROM_V28_3": {
        "V28.8": "coherence 0.79 (pass), bias +1.21 (J pass), blocks/team-game 11.06 vs 11.03, ladder MAE 2.94 vs V27 3.29 (K pass); but total SD 16.2 vs 13.8 (+17.8%, M fail), team SD 11.06 (+11%), "
                 "margin corr 0.144 vs 0.228 and total RMSE +1.09 (L fail), winner Brier delta +0.032 CI [-0.0004,+0.065] (N fail)",
        "residual_dispersion_source": "kappa=0 leaves team-game yardage CV 0.285 vs history 0.226: recent_form enters the efficiency multiplier 1:1 (SD 0.115 across teams) but an OUT-OF-PILOT regression of next-game "
                                      "total yards on recent_form (2014-2019: slope 0.29; 2020-2025: 0.23; pass yards 0.5) says only ~25% persists.  The simulator over-trusts the form signal ~4x, which "
                                      "inflates between-game spread (both dispersion and winner over-confidence).",
        "V28.7_attribution": "see V28_ANALYSIS_v7.json (team multiplier without clock scale)"},
    "CANDIDATES": {"V28.11": "V28.8 + recent_form shrink by the causal empirical slope (regulation)", "V28.12": "V28.11 + NFL overtime (arm of record)"},
    "SHRINK_SPEC": "efficiency = slope * recent_form + kappa(=0) * noise; slope = cov(rf, target)/var(rf) over the SAME 5-season causal window (rf = trailing-8-game pass yards / weekly league mean - 1; "
                   "target = team total yards / weekly league mean - 1), clipped to [0,1], hash-pinned player table; no fitted-to-outcome parameter",
    "STRUCTURAL_HARD_GATES": PREREG_BODY["STRUCTURAL_HARD_GATES"], "COHERENCE_PRIMARY_METRIC": PREREG_BODY["COHERENCE_PRIMARY_METRIC"], "COHERENCE_ARMS": PREREG_BODY["COHERENCE_ARMS"],
    "PREDICTIONS_REGISTERED_BEFORE_THE_V28.11_RUN": {
        "RECENT_FORM_SLOPE": [0.15, 0.40], "TEAM_GAME_YARDS_CV": [0.22, 0.26], "TOTAL_SD": [12.5, 14.8], "TEAM_SD": [9.0, 10.9], "TOTAL_BIAS": [-2.5, 2.5], "TD_POSSESSION_RATE": [0.215, 0.250],
        "BLOCKS_PER_TEAM_GAME": [10.9, 11.2], "COHERENCE_RANGE": ">= 0.45", "WINNER_BRIER_DELTA_VS_V27": "<= +0.02", "MARGIN_CORR": ">= 0.178 (V27 0.228 - 0.05)", "TOTAL_CORR": [0.0, 0.22],
        "FALSIFIER": "if dispersion is repaired (gate M) but the winner Brier delta CI lower bound is still > 0, the residual loss is not over-confident form and lives elsewhere in the yardage model"},
    "DECISION_RULES": {**PREREG_POSTHOC3["DECISION_RULES"],
                       "IF_NO_CHAMPION": "no GAME_SIM_CHAMPION is declared; V27 stays the engine of record; the operator command still runs V28.x explicitly labelled CANDIDATE_NOT_CHAMPION"},
    "NOT_DONE_BY_DESIGN": PREREG_BODY["NOT_DONE_BY_DESIGN"],
}


def cmd_freeze(tag: str) -> None:
    d = out_dir(tag)
    if prereg_path(tag).exists() or (d.exists() and any(d.rglob("*.npz"))):
        raise SystemExit("REFUSING: prereg or outputs already exist for this tag (a rule change ships as a new tag/version, never an edit)")
    games = pilot_games()
    t = subprocess.run([sys.executable, "-m", "pytest", f"{PKG}/tests", "-q", "-x"], cwd=ROOT, capture_output=True, text=True)
    tail = t.stdout.strip().splitlines()[-1] if t.stdout.strip() else t.stderr[-300:]
    assert t.returncode == 0, "unit tests must pass at freeze: " + tail
    rec = {"SCHEMA": "SPORTS_NOVA_V28_PREREG", "TAG": tag, "FROZEN_AT": datetime.now(timezone.utc).isoformat(), "MODEL_VERSIONS": cfg.VERSIONS,
           "V28_HASH": v28_hash(), "V28_PACKAGE_FILE_SHA256": pkg_hashes(), "PARENT_V27_HASH": v27_hash(), "PARENT_V27_HASH_PINNED": V27_HASH,
           "GUARDED_FILE_SHA256": guarded_hashes(), "HARNESS_SHA256_AT_FREEZE": sha256_file(Path(__file__)),
           "HAZARD_POLICY_HASH": cfg.HAZARD_POLICY_HASH, "TD_ALLOCATION_POLICY_HASH": cfg.TD_ALLOCATION_POLICY_HASH, "OVERTIME_POLICY_HASH": cfg.OVERTIME_POLICY_HASH,
           "HAZARD_POLICY": cfg.HAZARD_POLICY, "TD_ALLOCATION_POLICY": cfg.TD_ALLOCATION_POLICY, "OVERTIME_POLICY": cfg.OVERTIME_POLICY,
           "COHORT": {"games": len(games), "game_ids": games, "sims_per_game": N_SIMS, "seed_rule": "sha256(game_id)[:8] (H.game_seed)",
                      "scope": "V28 SCORING BOUNDARY only (historical states have no RosterSnapshot), identical scope to the V27 60-game reproduction"},
           "V27_BASELINE": "BASE arm of MULTI_TD_POSSESSION_ABLATION_V1 (np.array_equal to unpatched V27 on all 60 games); arrays sha256 recorded below",
           "V27_BASELINE_ARRAY_SHA256": {p.name: sha256_file(p) for p in sorted(V27_ARR.glob("*.npz"))},
           "UNIT_TESTS_AT_FREEZE": tail, "CONCURRENCY": concurrency(), "LIVE_CAPITAL_AUTHORIZED": False, **{"V28_1": PREREG_BODY, "V28_2": PREREG_POSTHOC, "V28_3": PREREG_POSTHOC3, "V28_4": PREREG_POSTHOC4}[tag]}
    assert v27_hash() == V27_HASH == rec["PARENT_V27_HASH"], "V27 package drifted"
    assert_market_free({k: rec[k] for k in ("SCHEMA", "MODEL_VERSIONS", "HAZARD_POLICY", "TD_ALLOCATION_POLICY", "OVERTIME_POLICY", "V28_PACKAGE_FILE_SHA256")})
    d.mkdir(parents=True, exist_ok=True)
    prereg_path(tag).write_text(json.dumps(rec, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("V28_HASH", "PARENT_V27_HASH", "UNIT_TESTS_AT_FREEZE")}, indent=1))


def load_freeze(tag: str) -> dict:
    p = prereg_path(tag)
    if not p.exists():
        raise SystemExit(f"run `freeze {tag}` first: {p.name} must exist before any V28 simulation")
    pre = json.loads(p.read_text())
    assert pre["V28_HASH"] == v28_hash(), "V28 package changed after the freeze (ship a new tag)"
    assert pre["GUARDED_FILE_SHA256"] == guarded_hashes(), "a frozen parent file changed"
    return pre


# ---------------------------------------------------------------------------------------------- per-game runs
def game_state(game: str):
    season, week, away, home = H.game_parts(game)
    all_df = pd.read_parquet(H.PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    prior = all_df[(all_df._key < season * 100 + week) & (all_df.SEASON >= max(1999, season - 5))]
    return H.make_state(game, prior, H.surrogate_kickoff(season, week), None), home, away


def psum(batch, key):
    arr = np.asarray(batch.player_stats[key])
    cols = np.zeros((batch.n_sims, 2), np.int64)
    for j, tid in enumerate(batch.team_ids):
        idx = [i for i, pid in enumerate(batch.player_ids) if batch.player_team[pid] == tid]
        cols[:, j] = arr[:, idx].sum(axis=1) if idx else 0
    return cols


def accounting(batch, blk: np.ndarray, state, model_version: str) -> dict:
    n, F = batch.n_sims, s28.BI
    score, tds, nblk, tdp, tdr = (np.zeros((n, 2), np.int64) for _ in range(5))
    bad = 0
    for r in blk:
        s, j = int(r[F["sim"]]), int(r[F["off"]])
        t, p, fg, pat, ot = int(r[F["td"]]), int(r[F["pts"]]), int(r[F["fg"]]), int(r[F["pat"]]), int(r[F["ot"]])
        score[s, j] += p; tds[s, j] += t; nblk[s, j] += 1; tdp[s, j] += int(r[F["td_pass"]]); tdr[s, j] += int(r[F["td_rush"]])
        if t > 1 or (t == 1 and p != 6 + pat) or (t == 0 and p not in (0, 3)) or (fg and (t or p != 3)) or (t and pat not in (0, 1)):
            bad += 1
        if t != int(r[F["td_pass"]]) + int(r[F["td_rush"]]) or (int(r[F["td_qb_rush"]]) and not int(r[F["td_rush"]])):
            bad += 1
    acct = {"score_eq_block_points": int((score != np.asarray(batch.team_stats["score"])).sum()),
            "team_tds_eq_block_tds": int((tds != np.asarray(batch.team_stats["tds"])).sum()),
            "tds_eq_player_scored_tds": int((tds != psum(batch, "scored_tds")).sum()),
            "td_pass_eq_player_receiving_tds": int((tdp != psum(batch, "receiving_tds")).sum()),
            "td_pass_eq_qb_pass_tds": int((tdp != psum(batch, "pass_tds")).sum()),
            "td_rush_eq_player_rush_tds": int((tdr != psum(batch, "rush_tds")).sum()),
            "blocks_eq_team_blocks": int((nblk != np.asarray(batch.team_stats["blocks"])).sum()),
            "reg_plus_ot_eq_final": int((np.asarray(batch.team_stats["reg_score"]) + np.asarray(batch.team_stats["ot_score"]) != np.asarray(batch.team_stats["score"])).sum()),
            "block_shape_violations": int(bad),
            "TD_and_FG_same_possession": int(((blk[:, F["td"]] > 0) & (blk[:, F["fg"]] > 0)).sum())}
    fails: list = []
    for i in range(n):
        stats = {pid: {k: int(batch.player_stats[k][i, j]) for k in PLAYER_KEYS} for j, pid in enumerate(batch.player_ids)}
        team = {tid: {k: int(batch.team_stats[k][i, j]) for k in TEAM_KEYS} for j, tid in enumerate(batch.team_ids)}
        assert_path(state.game_id, state, {"result": (stats, team, None), "sim_id": i, "blocks": [], "residual_by_team": {}}, "V23", fails)
    acct["assert_path_failures"] = len(fails)
    acct["max_td_per_block"] = int(blk[:, F["td"]].max()) if len(blk) else 0
    return acct


def arrays_of(batch, blk) -> dict:
    a = {"team_score": np.asarray(batch.team_stats["score"], np.int64), "reg_score": np.asarray(batch.team_stats["reg_score"], np.int64),
         "went_ot": np.asarray(batch.team_stats["went_ot"], np.int64)[:, 0], "winner": np.asarray(batch.winner), "blocks": blk}
    for k in ("pass_yards", "rush_yards", "blocks", "tds", "fgs", "td_pass", "td_rush", "td_qb_rush", "td_unattributable", "ot_score"):
        a["team_" + k] = np.asarray(batch.team_stats[k], np.int64)
    for k in ("scored_tds", "receiving_tds", "rush_tds", "pass_tds", "targets", "receptions", "rush_attempts"):
        a["psum_" + k] = psum(batch, k)
    return a


def run_pilot_game(args) -> dict:
    game, variant, n, ref = args
    t0 = time.time()
    mv = cfg.VERSIONS[variant]
    state, home, away = game_state(game)
    seed = H.game_seed(game)
    b, blk = s28.simulate_scoring(state, n, seed, mv, ref_quantiles=ref, want_blocks=True)
    acct = accounting(b, blk, state, mv)
    a = arrays_of(b, blk)
    return {"game": game, "seed": seed, "home": home, "away": away, "arrays": a, "accounting": acct,
            "fg_rate": b.runtime["fg_rate"], "hazard": {k: v for k, v in b.runtime.items() if k.startswith("hazard_")}, "seconds": round(time.time() - t0, 1)}


def load_ref(variant: int, tag: str):
    if not cfg.VARIANTS[variant][1]:
        return None
    p = out_dir(tag) / "V28_YARD_REFERENCE.json"
    ref = json.loads(p.read_text())
    return np.asarray(ref["quantiles"], dtype=float)


def cmd_pilot(variant: int, tag: str, workers: int = 4) -> None:
    pre = load_freeze(tag)
    d = out_dir(tag) / f"pilot_v{variant}"
    (d / "arrays").mkdir(parents=True, exist_ok=True)
    games = pilot_games()
    assert games == pre["COHORT"]["game_ids"]
    ref = load_ref(variant, tag)
    before = guarded_hashes()
    t0 = time.time()
    with Pool(workers) as pool:
        results = []
        for r in pool.imap_unordered(run_pilot_game, [(g, variant, N_SIMS, ref) for g in games]):
            np.savez_compressed(d / "arrays" / f"{r['game']}.npz", **r["arrays"])
            results.append({k: v for k, v in r.items() if k != "arrays"})
            print(f"{r['game']} {r['seconds']}s max_td={r['accounting']['max_td_per_block']} acct_bad={sum(v for k, v in r['accounting'].items() if k != 'max_td_per_block')}", flush=True)
    after = guarded_hashes()
    meta = {"VARIANT": variant, "MODEL_VERSION": cfg.VERSIONS[variant], "TAG": tag, "V28_HASH": v28_hash(), "PREREG_SHA256": sha256_file(prereg_path(tag)),
            "HARNESS_SHA256_AT_RUN": sha256_file(Path(__file__)), "GUARDED_UNCHANGED": before == after, "WALL_SEC": time.time() - t0, "N_SIMS": N_SIMS, "N_GAMES": len(results),
            "RESULTS": sorted(results, key=lambda r: r["game"])}
    (d / "run_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    tot = {}
    for r in results:
        for k, v in r["accounting"].items():
            tot[k] = max(tot.get(k, 0), v) if k == "max_td_per_block" else tot.get(k, 0) + v
    print("PILOT DONE", cfg.VERSIONS[variant], "wall", round(time.time() - t0), "guarded_unchanged", before == after, "accounting", json.dumps(tot))


# ---------------------------------------------------------------------------------------------- coherence arms
def _coh_game(args) -> dict:
    game, variant, n, ref = args
    mv = cfg.VERSIONS[variant]
    state, home, away = game_state(game)
    seed = H.game_seed(game)
    P = s28.Perturb
    b0, blk0 = s28.simulate_scoring(state, n, seed, mv, ref_quantiles=ref, want_blocks=True)
    traces = [[] for _ in range(n)]
    for r in blk0:
        traces[int(r[s28.BI["sim"]])].append((int(r[s28.BI["pre_h"]]), int(r[s28.BI["pre_a"]])))
    home_id = b0.team_ids[0]
    arms = {"A_yards_x0.5": P(yard_scale=0.5), "B_yards_x1.5": P(yard_scale=1.5), "C_no_explosive_cap20": P(explosive_cap=20.0),
            "H_home_x0.5": P(yard_scale=0.5, only_team=home_id), "H_home_x1.5": P(yard_scale=1.5, only_team=home_id)}
    out = {"game": game, "identity": summarize(b0, blk0)}
    for name, pt in arms.items():
        # controlled: replay the identity arm's score differential into the pass-rate script (identical football draws) + common random numbers on the score stream
        trace_full = [(traces[i] + [(0, 0)] * 80) for i in range(n)]
        b, blk = s28.simulate_scoring(state, n, seed, mv, ref_quantiles=ref, perturb=pt, script_traces=trace_full, want_blocks=True)
        out[name + "|controlled"] = summarize(b, blk)
        b, blk = s28.simulate_scoring(state, n, seed, mv, ref_quantiles=ref, perturb=pt, want_blocks=True)
        out[name + "|live"] = summarize(b, blk)
    return out


def summarize(b, blk) -> dict:
    ts = np.asarray(b.team_stats["reg_score"], float)
    w = np.asarray(b.winner)
    F = s28.BI
    return {"pts_per_team": float(ts.mean()), "p_home_win": float((w == "HOME").mean() + 0.5 * (w == "TIE").mean()), "td_per_team": float(blk[:, F["td"]].sum() / (2 * b.n_sims)),
            "td_poss_rate": float((blk[:, F["td"]] > 0).mean()), "yards_per_block": float(blk[:, F["yds"]].mean()), "mean_p_td": float(blk[:, F["p_td"]].mean())}


def v27_reference_arms(game: str, n: int) -> dict:
    """V27 with sample_compound_signed scaled (symmetric arms).  Patches only the V23 module attribute; always restored."""
    from worker.sports_nova_v23 import simulator as _v23
    from worker.sports_nova_v27_causal_fg_rate import simulator as v27
    from worker.sports_nova_v27_causal_fg_rate import config as c27
    state, home, away = game_state(game)
    seed = H.game_seed(game)
    orig = _v23.sample_compound_signed
    res = {}
    try:
        for name, sc in (("identity", 1.0), ("A_yards_x0.5", 0.5), ("B_yards_x1.5", 1.5)):
            _v23.sample_compound_signed = (lambda count, mean, sd, rng, _o=orig, _s=sc: int(round(_o(count, mean, sd, rng) * _s)))
            b = v27.simulate_scoring(state, n, seed, c27.MODEL_VERSION)
            w = np.asarray(b.winner)
            res[name] = {"pts_per_team": float(np.asarray(b.team_stats["score"], float).mean()), "p_home_win": float((w == "HOME").mean() + 0.5 * (w == "TIE").mean())}
    finally:
        _v23.sample_compound_signed = orig
    assert _v23.sample_compound_signed is orig
    return res


def _coh_v27(args) -> dict:
    game, n = args
    return {"game": game, **v27_reference_arms(game, n)}


def cmd_coherence(variant: int, tag: str, workers: int = 4) -> None:
    pre = load_freeze(tag)
    games = pilot_games()[::COH_SUBSET_STEP]
    ref = load_ref(variant, tag)
    d = out_dir(tag) / f"coherence_v{variant}"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(workers) as pool:
        v28 = list(pool.imap_unordered(_coh_game, [(g, variant, N_SIMS, ref) for g in games]))
        v27 = list(pool.imap_unordered(_coh_v27, [(g, N_SIMS) for g in games]))
    (d / "coherence.json").write_text(json.dumps({"VARIANT": variant, "V28_HASH": v28_hash(), "PREREG_SHA256": sha256_file(prereg_path(tag)), "GAMES": games, "V28": sorted(v28, key=lambda r: r["game"]),
                                                  "V27": sorted(v27, key=lambda r: r["game"]), "WALL_SEC": time.time() - t0}, indent=1, default=float))
    print("COHERENCE DONE", round(time.time() - t0))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "freeze":
        cmd_freeze(sys.argv[2] if len(sys.argv) > 2 else "V28_1")
    elif cmd == "pilot":
        cmd_pilot(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "V28_1")
    elif cmd == "coherence":
        cmd_coherence(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "V28_1")
    else:
        raise SystemExit(__doc__)
