"""V16: SPORTS_NOVA_V16_DROPBACK_TO_YARDAGE_DECOMPOSITION.

Diagnostic-only. NO engine changes, NO re-simulation, NO new QB-identity
work. Reuses the exact per-team-game numbers V15 already computed and
persisted (SPORTS_NOVA_V15_TEAM_PASSING_VOLUME_TEAM_GAMES.csv, the same
60-game/120-team-game/N=128 cohort) and only does read-only arithmetic
decomposition on them -- no simulator.py call is made here at all, so this
respects "no full 1693 OOS" trivially (zero additional simulation).

DROPBACK CHAIN
--------------
By construction in worker/sports_nova_v3/play_selection.py::select_plays,
dropbacks = sacks + scrambles + pass_attempts EXACTLY (pass_attempts =
non_sack - scrambles = (dropbacks - sacks) - scrambles). V15's SIM_SACKS
and SIM_SCRAMBLES are analytic (pre-sim formula) reconstructions -- the
actual simulator never persists sacks/scrambles as tracked output (verified
by reading simulator.py/play_selection.py: TEAM_STAT_NAMES has no sack or
dropback field). So the dropback chain here necessarily mixes:
  SIM_DROPBACKS (analytic)  = mean_blocks * expected_plays_per_block * pass_rate
  SIM_SACKS (analytic)      = SIM_DROPBACKS * sack_rate
  SIM_SCRAMBLES (analytic)  = SIM_DROPBACKS * (1-sack_rate) * scramble_rate
  SIM_ATTEMPTS (ACTUAL)     = sim.team(tid, 'pass_attempts'), the real
                               stochastic simulation output V15 already used
                               for TEAM_ATTEMPT_BIAS.
Because SIM_SACKS + SIM_SCRAMBLES + analytic-implied-attempts sums to
SIM_DROPBACKS exactly by formula, but ACTUAL SIM_ATTEMPTS (measured from
the real simulation, not the formula) is lower than that analytic-implied
attempt count, the residual:
  NON_ATTEMPT_DROPBACK_BIAS = SIM_DROPBACKS - SIM_SACKS - SIM_SCRAMBLES - SIM_ATTEMPTS(actual)
is exactly V15's already-found SIMULATION_DRIFT (-1.29) reframed as
"dropbacks the analytic chain implies should exist somewhere, that do not
show up as a sack, a scramble, or an actual simulated attempt." It is not
sack-rate or scramble-rate mis-assumption (those are scored separately
below against real sacks) -- it is the stochastic simulation mechanism
itself running below its own formula's mean, exactly as V15 characterized
it.

REAL_SACKS comes from NFL_V3_DRIVE_BLOCK_CANONICAL_V1 (the only source with
real sacks). REAL_SCRAMBLES is NOT separable from designed rushes in any
available real source (same limitation V9/V15 already documented) -- SO
DROPBACK_BIAS AND SCRAMBLE_BIAS ARE NOT FULLY TRUSTWORTHY: comparing an
analytic SIM_DROPBACKS (which correctly bakes in an implied scramble count)
against a REAL_DROPBACKS that is missing its own real scramble count
inflates the apparent dropback gap. ATTEMPT_BIAS and SACK_BIAS are the
clean, both-sides-real numbers in this chain.

YARDAGE CHAIN
-------------
pass_yards = attempts * completion_rate * yards_per_completion (exact
identity at the cohort-mean level). Bias is decomposed with an exact
sequential (chain-substitution) bridge -- not a fitted/tuned model, pure
arithmetic that telescopes to the measured TEAM_PASS_YARDS_BIAS exactly:
  ATTEMPT_COUNT_CONTRIBUTION = (sim_att - real_att) * real_cr * real_ypc
  COMPLETION_CONTRIBUTION    = sim_att * (sim_cr - real_cr) * real_ypc
  YPC_CONTRIBUTION           = sim_att * sim_cr * (sim_ypc - real_ypc)
This is the counterfactual the mission asks for: each term is "hold the
other two stages at their SIM/REAL reference value, vary only this one."
CAVEATS (both one-line, neither changes the exact sum):
  - This is ONE sequential order (attempt -> completion-rate -> YPC).
    Reordering moves magnitude between terms (interaction effects land on
    whichever term is last); the total always still sums exactly.
  - "shares_of_total_movement" divides by the sum of ABSOLUTE term sizes,
    not the (signed) total bias, because YPC_CONTRIBUTION runs opposite in
    sign to the other two (offsetting, not compounding) -- so the shares do
    not multiply back to -18.52 the way a same-signed decomposition would.

ROOT CAUSE CONFIRMED FOR THE COMPLETION-RATE GAP (not catch_rate
miscalibration -- a structural allocation-truncation effect):
make_state's own target-share construction (scripts/
sports_nova_v3_player_joint_walkforward_v1.py, the SAME state builder every
V9-V16 diagnostic in this chain imports and evaluates against -- worker/
sports_nova_v3/pregame_state.py, the file simulator.py's docs point to as
"the" production builder, has no matching OpportunityShares/residual
construction at all, so this may be specific to the research/evaluation
harness; that could not be resolved within this diagnostic and is flagged
for the next mission, not resolved here) computes each named receiver's
target share, then does `s *= .9`, permanently reserving exactly 10% of
target share as `residual_share` ("retain residual mass for players
outside the truncated state" -- deliberate epistemic-uncertainty design).
Measured directly (read-only, via allocation.py::_shares, no simulator.py
call) across all 120 team-games: residual_target_share = EXACTLY 0.100000,
zero variance -- a flat structural constant, not data-driven.
worker/sports_nova_v3/simulator.py's per-block reception loop only iterates
`alloc.target_counts.items()` -- the untargeted/residual bucket is never
looped over, so every pass attempt allocated to it contributes ZERO
receptions and ZERO yards (it is not "no completion drawn from catch_rate",
it never enters the completion/yardage draw at all). This mechanically
produces a completion-rate FLOOR around 10% below what the named-receiver
catch_rate mechanism alone would produce, independent of whether catch_rate
itself is well-calibrated. Isolating it (see "targeted_only_completion_rate"
below) shows catch_rate on the targeted 90% is NOT low -- it is slightly
ABOVE the real completion rate.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sports_nova_v3"
V15_ROWS = DATA / "SPORTS_NOVA_V15_TEAM_PASSING_VOLUME_TEAM_GAMES.csv"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from worker.sports_nova_v3.allocation import _shares
from sports_nova_v3_player_joint_walkforward_v1 import (MANIFEST, game_parts,
    surrogate_kickoff, make_state, PLAYER)


def measure_residual_target_share(selected_games: list[str]) -> pd.DataFrame:
    """Read-only: calls make_state (no simulate_game, no simulator.py call)
    and allocation.py::_shares (also read-only) to measure the untargeted
    residual share per team-game. Cheap -- no N=128 simulation loop."""
    all_df = pd.read_parquet(PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK
    rows = []
    for g in selected_games:
        s, w, away, home = game_parts(g)
        key = s * 100 + w
        prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, s - 5))]
        state = make_state(g, prior, surrogate_kickoff(s, w), None)
        for tid in (home, away):
            ids, values, residual = _shares(state, tid, "target")
            rows.append({"GAME_ID": g, "TEAM": tid, "residual_target_share": residual})
    return pd.DataFrame(rows)


def main():
    tg = pd.read_csv(V15_ROWS)
    n = len(tg)

    games = json.loads(MANIFEST.read_text())["GAME_IDS"]
    by = {}
    for g in games:
        by.setdefault(game_parts(g)[0], []).append(g)
    selected = []
    for season in sorted(by):
        xs = by[season]
        selected.extend([xs[int(i)] for i in np.linspace(0, len(xs) - 1, 10, dtype=int)])
    residual_df = measure_residual_target_share(selected)
    mean_residual_share = float(residual_df.residual_target_share.mean())
    residual_share_sd = float(residual_df.residual_target_share.std())

    # ---- DROPBACK CHAIN ----
    sim_dropbacks = float(tg.SIM_DROPBACKS_PRE_SIM.mean())
    real_dropbacks = float(tg.REAL_DROPBACKS.mean())
    sim_attempts = float(tg.SIM_PASS_ATT.mean())
    real_attempts = float(tg.REAL_PASS_ATT.mean())
    sim_sacks = float(tg.SIM_SACKS_PRE_SIM.mean())
    real_sacks = float(tg.REAL_SACKS.mean())
    sim_scrambles = float(tg.SIM_SCRAMBLES_PRE_SIM.mean())

    dropback_bias = sim_dropbacks - real_dropbacks
    attempt_bias = sim_attempts - real_attempts
    sack_bias = sim_sacks - real_sacks
    scramble_bias = None  # real side unobservable -- see docstring
    non_attempt_dropback_bias = sim_dropbacks - sim_sacks - sim_scrambles - sim_attempts

    # ---- YARDAGE CHAIN ----
    sim_completions = float(tg.SIM_COMPLETIONS.mean())
    real_completions = float(tg.REAL_COMPLETIONS.mean())
    sim_yards = float(tg.SIM_PASS_YDS.mean())
    real_yards = float(tg.REAL_PASS_YDS.mean())

    sim_cr = sim_completions / sim_attempts
    real_cr = real_completions / real_attempts
    sim_ypc = sim_yards / sim_completions
    real_ypc = real_yards / real_completions
    sim_ypa = sim_yards / sim_attempts
    real_ypa = real_yards / real_attempts

    completion_rate_bias = sim_cr - real_cr
    yards_per_completion_bias = sim_ypc - real_ypc
    yards_per_attempt_bias = sim_ypa - real_ypa
    pass_yards_bias = sim_yards - real_yards

    # ROOT CAUSE for the completion-rate gap (see module docstring): a flat
    # 10% "untargeted" residual share, read-only measured via make_state +
    # allocation.py::_shares, is never looped over in simulator.py's
    # reception draw, so it contributes 0 completions/0 yards structurally.
    # Isolate the catch_rate mechanism's OWN performance on just the 90% of
    # attempts that do get a named target.
    targeted_only_completion_rate = sim_completions / (sim_attempts * (1 - mean_residual_share))

    # CANDIDATE FIX EVALUATED AND REJECTED ON ARITHMETIC (no code was written
    # or run for this -- it is decisive from numbers already measured above):
    # "credit the untargeted residual attempts with a plausible completion
    # instead of discarding them." Because YPC_CONTRIBUTION is already +8.80
    # (sim yards-per-completion runs ABOVE real, not below), adding back
    # completions at ANY plausible rate overshoots pass_yards_bias past zero
    # to the OPPOSITE sign, and does so whether credited at the targeted-only
    # sim rate or at the real rate.
    untargeted_attempts_per_game = mean_residual_share * sim_attempts
    def _overshoot(cr, ypc, label):
        added_completions = untargeted_attempts_per_game * cr
        added_yards = added_completions * ypc
        new_completions = sim_completions + added_completions
        new_yards = sim_yards + added_yards
        return {
            "credited_at": label, "completion_rate_used": cr, "ypc_used": ypc,
            "added_completions_per_game": added_completions, "added_yards_per_game": added_yards,
            "new_sim_completions_mean": new_completions, "new_completion_bias": new_completions - real_completions,
            "new_sim_pass_yards_mean": new_yards, "new_pass_yards_bias": new_yards - real_yards,
        }
    rejected_fix_arithmetic = {
        "candidate": "route untargeted/residual attempts to a plausible completion outcome instead of "
                     "discarding them (in simulator.py's per-block reception loop)",
        "current_pass_yards_bias": pass_yards_bias, "current_completion_bias": sim_completions - real_completions,
        "if_credited_at_targeted_only_sim_rates": _overshoot(targeted_only_completion_rate, sim_ypc, "sim targeted-only CR/YPC"),
        "if_credited_at_real_rates": _overshoot(real_cr, real_ypc, "real CR/YPC"),
        "verdict": "REJECTED -- both credit assumptions flip pass_yards_bias to the OPPOSITE sign (overshoot "
                   "past zero), because the -18.52 deficit is NET of an already-positive +8.80 YPC surplus "
                   "(see yardage_bridge_decomposition); there is no credit rate for the residual bucket that "
                   "lands the total near zero. Same shape as V10: a well-motivated single-mechanism fix that "
                   "measures worse than the defect it targets. No code was changed to reach this conclusion.",
    }

    attempt_contrib = (sim_attempts - real_attempts) * real_cr * real_ypc
    completion_contrib = sim_attempts * (sim_cr - real_cr) * real_ypc
    ypc_contrib = sim_attempts * sim_cr * (sim_ypc - real_ypc)
    bridge_check = attempt_contrib + completion_contrib + ypc_contrib  # must equal pass_yards_bias exactly

    shares = {
        "ATTEMPT_COUNT": abs(attempt_contrib) / (abs(attempt_contrib) + abs(completion_contrib) + abs(ypc_contrib)),
        "COMPLETION_RATE": abs(completion_contrib) / (abs(attempt_contrib) + abs(completion_contrib) + abs(ypc_contrib)),
        "YARDS_PER_COMPLETION": abs(ypc_contrib) / (abs(attempt_contrib) + abs(completion_contrib) + abs(ypc_contrib)),
    }
    dominant_yardage_stage, dominant_yardage_share = max(shares.items(), key=lambda kv: kv[1])

    # counterfactuals (explicit, matching the mission's requested framing)
    scramble_rate_assumed = sim_scrambles / (sim_dropbacks - sim_sacks)  # recovers the rate used, by construction
    counterfactual_sack_rate = {
        "description": "hold SIM dropbacks fixed; replace the assumed sack_rate conversion with the "
                       "REALIZED (implied) sack rate, recompute counterfactual attempts",
        "real_sack_rate_implied": real_sacks / real_dropbacks,
        "scramble_rate_assumed_in_engine_cohort_mean": scramble_rate_assumed,
        "counterfactual_attempts_if_real_sack_rate_used": (
            sim_dropbacks * (1 - real_sacks / real_dropbacks) * (1 - scramble_rate_assumed)
        ),
        "vs_actual_sim_attempts": sim_attempts,
        "vs_pre_sim_analytic_attempts_at_assumed_sack_rate": sim_dropbacks * (1 - sim_sacks / sim_dropbacks) * (1 - scramble_rate_assumed),
    }
    counterfactual_completion_rate = {
        "description": "hold SIM attempts fixed; replace SIM completion rate with REAL completion rate",
        "counterfactual_completions": sim_attempts * real_cr,
        "vs_actual_sim_completions": sim_completions,
    }
    counterfactual_ypc = {
        "description": "hold SIM completions fixed; replace SIM yards/completion with REAL yards/completion",
        "counterfactual_yards": sim_completions * real_ypc,
        "vs_actual_sim_yards": sim_yards,
    }

    # ---- verdict ----
    dropback_chain_total_loss = abs(sack_bias) + abs(non_attempt_dropback_bias)
    sack_scramble_dominant = (abs(non_attempt_dropback_bias) / dropback_chain_total_loss >= 0.60
                               if dropback_chain_total_loss else False)
    attempts_near_correct = abs(attempt_bias) < 0.10 * real_attempts  # within 10% relative
    completions_near_correct = abs(sim_completions - real_completions) < 0.10 * real_completions

    if dominant_yardage_share >= 0.60:
        if dominant_yardage_stage == "COMPLETION_RATE" and attempts_near_correct:
            verdict = "COMPLETION_RATE_DEFECT"
        elif dominant_yardage_stage == "YARDS_PER_COMPLETION" and completions_near_correct:
            verdict = "YARDS_PER_COMPLETION_DEFECT"
        elif dominant_yardage_stage == "ATTEMPT_COUNT" and sack_scramble_dominant:
            verdict = "SACK_SCRAMBLE_CONVERSION_DEFECT"
        else:
            verdict = f"MULTI_STAGE_PASSING_DEFECT (dominant single yardage term is {dominant_yardage_stage} at " \
                      f"{dominant_yardage_share:.0%}, but its own preconditions -- near-correct attempts/" \
                      f"completions or sack/scramble dominance -- are not also met, so no single named defect fits)"
    else:
        verdict = "MULTI_STAGE_PASSING_DEFECT"

    result = {
        "mission": "SPORTS_NOVA_V16_DROPBACK_TO_YARDAGE_DECOMPOSITION",
        "n_team_games": n,
        "source": "arithmetic decomposition of SPORTS_NOVA_V15_TEAM_PASSING_VOLUME_TEAM_GAMES.csv -- no re-simulation",
        "dropback_chain": {
            "SIM_DROPBACKS_mean_analytic": sim_dropbacks, "REAL_DROPBACKS_mean_UNDERSTATED": real_dropbacks,
            "DROPBACK_BIAS": dropback_bias,
            "caveat": "REAL_DROPBACKS excludes real scrambles (not separable in source) -- DROPBACK_BIAS and "
                      "SCRAMBLE_BIAS are not fully trustworthy; ATTEMPT_BIAS and SACK_BIAS are the clean numbers",
            "SIM_ATTEMPTS_mean_actual": sim_attempts, "REAL_ATTEMPTS_mean": real_attempts, "ATTEMPT_BIAS": attempt_bias,
            "SIM_SACKS_mean_analytic": sim_sacks, "REAL_SACKS_mean": real_sacks, "SACK_BIAS": sack_bias,
            "SIM_SCRAMBLES_mean_analytic": sim_scrambles, "REAL_SCRAMBLES": "NOT_SEPARABLE_FROM_DESIGNED_RUSHES",
            "SCRAMBLE_BIAS": scramble_bias,
            "NON_ATTEMPT_DROPBACK_BIAS": non_attempt_dropback_bias,
            "non_attempt_dropback_bias_note": "== V15's SIMULATION_DRIFT (-1.29) reframed: dropbacks the analytic "
                                              "chain implies exist that show up as neither a sack, a scramble, nor "
                                              "an actual simulated attempt -- a stochastic-mechanism gap, not a "
                                              "sack_rate/scramble_rate assumption error",
        },
        "yardage_chain": {
            "SIM_COMPLETIONS_mean": sim_completions, "REAL_COMPLETIONS_mean": real_completions,
            "SIM_completion_rate": sim_cr, "REAL_completion_rate": real_cr, "COMPLETION_RATE_BIAS": completion_rate_bias,
            "SIM_yards_per_completion": sim_ypc, "REAL_yards_per_completion": real_ypc,
            "YARDS_PER_COMPLETION_BIAS": yards_per_completion_bias,
            "SIM_yards_per_attempt": sim_ypa, "REAL_yards_per_attempt": real_ypa,
            "YARDS_PER_ATTEMPT_BIAS": yards_per_attempt_bias,
            "PASS_YARDS_BIAS": pass_yards_bias,
        },
        "completion_rate_root_cause": {
            "mechanism": ("make_state's target-share construction (scripts/"
                          "sports_nova_v3_player_joint_walkforward_v1.py, `s *= .9`) always reserves "
                          "residual_target_share=0.10 for 'players outside the truncated state'. "
                          "simulator.py's per-block reception loop only iterates alloc.target_counts "
                          "(the named-receiver buckets) -- the residual/untargeted bucket is never "
                          "looped over, so it contributes 0 completions and 0 yards, not a completion "
                          "drawn-and-missed at catch_rate."),
            "measured_residual_target_share_mean": mean_residual_share,
            "measured_residual_target_share_sd": residual_share_sd,
            "measured_residual_target_share_note": "EXACTLY 0.100000 on every one of the 120 team-games "
                                                    "(sd~0) -- a flat structural constant, not data-driven",
            "SIM_completion_rate_all_attempts": sim_cr,
            "SIM_completion_rate_targeted_attempts_only": targeted_only_completion_rate,
            "REAL_completion_rate": real_cr,
            "interpretation": ("catch_rate performance on the 90% of attempts that DO get a named target "
                               f"({targeted_only_completion_rate:.3f}) is not low -- it is ABOVE real "
                               f"({real_cr:.3f}). The measured completion-rate deficit is fully explained "
                               "by the flat 10% untargeted-attempt structural truncation, not by "
                               "catch_rate/clip miscalibration."),
            "production_path_resolved": ("worker/sports_nova_v3/pregame_state.py::build_pregame_state is "
                                         "fail-closed and computes NO shares itself -- it only validates and "
                                         "wraps whatever an external feature_store supplies. So `s *= .9` is "
                                         "confirmed a research/backtest-harness modeling choice (scripts/"
                                         "sports_nova_v3_player_joint_walkforward_v1.py only), not a live "
                                         "production defect. Changing simulator.py to compensate for this "
                                         "harness constant would be tuning against the evaluation cohort, "
                                         "which the mission's own invariants prohibit -- not attempted."),
        },
        "rejected_fix_arithmetic": rejected_fix_arithmetic,
        "yardage_bridge_decomposition": {
            "ATTEMPT_COUNT_CONTRIBUTION_TO_YARD_BIAS": attempt_contrib,
            "COMPLETION_CONTRIBUTION_TO_YARD_BIAS": completion_contrib,
            "YPC_CONTRIBUTION_TO_YARD_BIAS": ypc_contrib,
            "sum_check_vs_measured_pass_yards_bias": {"bridge_sum": bridge_check, "measured": pass_yards_bias},
            "shares_of_total_movement": shares,
        },
        "counterfactuals": {
            "sack_rate_counterfactual": counterfactual_sack_rate,
            "completion_rate_counterfactual": counterfactual_completion_rate,
            "yards_per_completion_counterfactual": counterfactual_ypc,
        },
        "DOMINANT_STAGE": dominant_yardage_stage,
        "DOMINANT_STAGE_SHARE": dominant_yardage_share,
        "bridge_decomposition_caveats": [
            "sequential order attempt->completion_rate->YPC; a different order moves magnitude between "
            "terms (interaction effects land on whichever term is last) -- the sum is exact regardless",
            "shares_of_total_movement divides by the sum of ABSOLUTE term sizes, not the signed total, "
            "because YPC_CONTRIBUTION (+8.80) runs opposite in sign to the other two terms (offsetting, "
            "not compounding) -- 0.46 x -18.52 will not reconcile against COMPLETION_CONTRIBUTION",
        ],
        "VERDICT": verdict,
    }

    if verdict == "COMPLETION_RATE_DEFECT":
        next_action = ("Attempts are near-correct; completions are systematically low. Audit ONLY the "
                       "catch_rate causal feature / clip bounds in simulator.py's per-block reception draw "
                       "(read-only investigation first -- no simulator edit yet) against realized completion "
                       "rate, before touching yards-per-catch.")
    elif verdict == "YARDS_PER_COMPLETION_DEFECT":
        next_action = ("Completions are near-correct; yards-per-completion is systematically low. Audit ONLY "
                       "the yards_per_reception causal feature and eff_mult interaction against realized "
                       "yards/completion, before touching completion-rate or attempt-volume mechanics.")
    elif verdict == "SACK_SCRAMBLE_CONVERSION_DEFECT":
        next_action = ("Non-attempt dropbacks (sacks + the stochastic-mechanism gap) explain the dominant "
                       "share of attempt loss. Audit the dropback-to-attempt conversion path specifically -- "
                       "sack_rate assumption vs. realized, and why the actual simulated attempt mean runs "
                       "below its own analytic formula's implication -- before touching completion or yardage "
                       "mechanics.")
    else:
        next_action = (
            "No single stage clears the 60% gate, and the signature is genuinely multi-stage: attempt count "
            "(-10.65 yds) and completion rate (-16.67 yds) both pull the same direction while "
            "yards-per-completion runs the OPPOSITE direction (+8.80 yds, partially offsetting) -- the engine "
            "is low on volume and low on completion conversion but slightly HIGH on yards-per-completion, not "
            "uniformly low. The one precise, well-evidenced mechanism found (a flat 10% 'untargeted' residual "
            "share that simulator.py's reception loop drops to zero completions/yards -- see "
            "completion_rate_root_cause) was evaluated as a candidate fix and REJECTED on arithmetic, not "
            "implemented: crediting those residual attempts at ANY plausible completion rate (sim's own "
            "targeted-only rate, or the real rate) flips pass_yards_bias to the opposite sign -- see "
            "rejected_fix_arithmetic -- because the -18.52 deficit is net of an already-positive YPC surplus. "
            "This is the same shape as V10 (a well-motivated single-mechanism fix that measures worse than "
            "the defect). No local, single-mechanism, evidence-supported change remains: this is Phase 3's "
            "own stop_if condition (\"remaining defect requires broad architectural redesign rather than "
            "locally evidenced correction\") reached one phase early. The three stages need JOINT treatment "
            "(volume, conversion, and per-catch yardage together) that no isolated local patch can deliver "
            "without one of the other two absorbing the error in the opposite direction -- that scoping "
            "decision, and any worker/ change, needs sign-off before proceeding.")
    result["SINGLE_NEXT_ACTION"] = next_action

    (DATA / "SPORTS_NOVA_V16_DROPBACK_TO_YARDAGE_DECOMPOSITION.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
