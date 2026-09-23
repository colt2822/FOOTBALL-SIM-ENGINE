"""SPORTS_NOVA_V18_CALIBRATION_PIT_COVERAGE.

P5 CALIBRATION / P3 UNCERTAINTY mission work. Diagnostic-only, no engine
changes: worker/sports_nova_v3/simulator.py is called exactly as V9-V18
already call it (same 60-game/120-team-game cohort, same seeds, same
make_state). The only difference from scripts/sports_nova_v3_team_
passing_volume_calibration_v15.py is that THIS script keeps each
team-game's FULL N=128-sim distribution instead of collapsing it to a
mean, to answer a question point-estimate MAE/bias cannot: is the
simulator's predictive DISTRIBUTION well-calibrated, not just its center?

Motivated directly by SPORTS_NOVA_V1_1_CHANGELOG.md's error-concentration
finding: 15 of 120 baseline team-games (12.5%) carry 28% of total
attempt-count absolute error, concentrated at both extremes of real
attempt volume, while the simulator's mean output stays clustered near a
narrow center -- a dispersion/tail signature, not a center-bias one.

Computes, per team-game, for TEAM_PASS_ATT and TEAM_PASS_YARDS:
  - PIT = fraction of this team-game's N=128 sims <= the real value
    (the real observation's empirical percentile inside its own
    predictive distribution -- if the model is calibrated, PIT values
    across many team-games should be ~Uniform(0,1))
  - whether the real value falls inside the sim's own [5,95], [10,90],
    [25,75] empirical-quantile intervals (nominal 90%/80%/50% coverage)

Aggregates: PIT histogram bucket counts (for a rough uniformity check),
Kolmogorov-Smirnov test of PIT vs Uniform(0,1), and empirical coverage
rates for each nominal interval (a calibrated model's realized coverage
should match the nominal rate).
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worker.sports_nova_v3.config import MODEL_VERSION
from worker.sports_nova_v3.simulator import simulate_game
from sports_nova_v3_player_joint_walkforward_v1 import (MANIFEST, game_parts,
    surrogate_kickoff, make_state)
from sports_nova_v3_player_joint_walkforward_v1 import PLAYER

OUT = ROOT / "data" / "sports_nova_v3"
DRIVE = OUT / "validation_inputs" / "NFL_V3_DRIVE_BLOCK_CANONICAL_V1.parquet"
N = 128
NOMINAL_INTERVALS = {"90%": (0.05, 0.95), "80%": (0.10, 0.90), "50%": (0.25, 0.75)}
# From SPORTS_NOVA_V15_TEAM_PASSING_VOLUME_CALIBRATION.json TEAM_ATTEMPT_BIAS
# (SIM_TEAM_ATTEMPTS_MEAN - REAL_TEAM_ATTEMPTS_MEAN), same reproduced baseline
# used throughout tonight's session.
TEAM_ATTEMPT_BIAS = -1.5303385416666666


def main():
    games = json.loads(MANIFEST.read_text())["GAME_IDS"]
    by = {}
    for g in games:
        by.setdefault(game_parts(g)[0], []).append(g)
    selected = []
    for season in sorted(by):
        xs = by[season]
        selected.extend([xs[int(i)] for i in np.linspace(0, len(xs) - 1, 10, dtype=int)])

    all_df = pd.read_parquet(PLAYER)
    all_df["_key"] = all_df.SEASON * 100 + all_df.WEEK

    panel_team = (all_df[all_df.GAME_ID.isin(selected)]
                  .groupby(["GAME_ID", "TEAM"], as_index=False)
                  .agg(REAL_PASS_ATT=("PASS_ATTEMPTS", "sum"), REAL_PASS_YDS=("PASS_YARDS", "sum")))

    rows = []
    for g in selected:
        s, w, away, home = game_parts(g)
        key = s * 100 + w
        prior = all_df[(all_df._key < key) & (all_df.SEASON >= max(1999, s - 5))]
        state = make_state(g, prior, surrogate_kickoff(s, w), None)
        sim = simulate_game(state, N, int(hashlib.sha256(g.encode()).hexdigest()[:8], 16), MODEL_VERSION)

        for tid in (home, away):
            real = panel_team[(panel_team.GAME_ID == g) & (panel_team.TEAM == tid)]
            if real.empty:
                continue
            r = real.iloc[0]
            att_dist = sim.team(tid, "pass_attempts").astype(float)
            yds_dist = sim.team(tid, "pass_yards").astype(float)

            row = {"GAME_ID": g, "TEAM": tid, "SEASON": s,
                   "REAL_PASS_ATT": float(r.REAL_PASS_ATT), "REAL_PASS_YDS": float(r.REAL_PASS_YDS)}
            for label, dist, real_val in (("ATT", att_dist, r.REAL_PASS_ATT), ("YDS", yds_dist, r.REAL_PASS_YDS)):
                row[f"PIT_{label}"] = float(np.mean(dist <= real_val))
                for interval_name, (lo_q, hi_q) in NOMINAL_INTERVALS.items():
                    lo, hi = np.quantile(dist, lo_q), np.quantile(dist, hi_q)
                    row[f"COVERED_{label}_{interval_name}"] = bool(lo <= real_val <= hi)
            # Bias-decomposition check for ATT only (per advisor review):
            # is coverage failure explained by the known mean bias alone,
            # or is there genuine additional dispersion miscalibration on
            # top of it? Shift the WHOLE simulated distribution by the
            # known aggregate TEAM_ATTEMPT_BIAS (SIM-REAL mean, from
            # SPORTS_NOVA_V15_TEAM_PASSING_VOLUME_CALIBRATION.json) so its
            # center matches real, then recompute PIT/coverage unchanged.
            shifted_att_dist = att_dist - TEAM_ATTEMPT_BIAS
            row["PIT_ATT_BIAS_SHIFTED"] = float(np.mean(shifted_att_dist <= r.REAL_PASS_ATT))
            for interval_name, (lo_q, hi_q) in NOMINAL_INTERVALS.items():
                lo, hi = np.quantile(shifted_att_dist, lo_q), np.quantile(shifted_att_dist, hi_q)
                row[f"COVERED_ATT_BIAS_SHIFTED_{interval_name}"] = bool(lo <= r.REAL_PASS_ATT <= hi)
            rows.append(row)

    tg = pd.DataFrame(rows)
    n = len(tg)

    result = {
        "mission": "SPORTS_NOVA_V18_CALIBRATION_PIT_COVERAGE",
        "MODEL_VERSION": MODEL_VERSION,
        "note": "Diagnostic-only against the FROZEN V18 simulator, same cohort/seeds as SPORTS_NOVA_V15_TEAM_PASSING_VOLUME_CALIBRATION.json. No engine changes.",
        "cohort": {"games": len(selected), "team_games": n, "sims_per_game": N},
        "TARGETS": {},
    }
    for label in ("ATT", "YDS"):
        pit = tg[f"PIT_{label}"].to_numpy()
        ks_stat, ks_p = scipy_stats.kstest(pit, "uniform")
        coverage = {name: float(tg[f"COVERED_{label}_{name}"].mean()) for name in NOMINAL_INTERVALS}
        result["TARGETS"][label] = {
            "PIT_mean": float(pit.mean()), "PIT_sd": float(pit.std()),
            "PIT_below_0.05_frac": float(np.mean(pit < 0.05)),
            "PIT_above_0.95_frac": float(np.mean(pit > 0.95)),
            "PIT_uniform_KS_stat": float(ks_stat), "PIT_uniform_KS_pvalue": float(ks_p),
            "empirical_coverage": coverage,
            "nominal_vs_empirical_gap": {name: coverage[name] - (hi - lo)
                                          for name, (lo, hi) in NOMINAL_INTERVALS.items()},
        }

    pit_shifted = tg["PIT_ATT_BIAS_SHIFTED"].to_numpy()
    ks_stat_shifted, ks_p_shifted = scipy_stats.kstest(pit_shifted, "uniform")
    coverage_shifted = {name: float(tg[f"COVERED_ATT_BIAS_SHIFTED_{name}"].mean()) for name in NOMINAL_INTERVALS}
    result["ATT_BIAS_DECOMPOSITION"] = {
        "shift_applied": TEAM_ATTEMPT_BIAS,
        "note": "Shifts each team-game's simulated attempt distribution by the known aggregate mean bias, then recomputes PIT/coverage. If tail fractions come back near nominal after this, the coverage failure IS the bias with no separate dispersion defect; if they stay skewed, there is genuine additional under-dispersion.",
        "PIT_mean_after_shift": float(pit_shifted.mean()),
        "PIT_below_0.05_frac_after_shift": float(np.mean(pit_shifted < 0.05)),
        "PIT_above_0.95_frac_after_shift": float(np.mean(pit_shifted > 0.95)),
        "PIT_uniform_KS_stat_after_shift": float(ks_stat_shifted),
        "PIT_uniform_KS_pvalue_after_shift": float(ks_p_shifted),
        "empirical_coverage_after_shift": coverage_shifted,
        "nominal_vs_empirical_gap_after_shift": {name: coverage_shifted[name] - (hi - lo)
                                                   for name, (lo, hi) in NOMINAL_INTERVALS.items()},
    }

    (OUT / "SPORTS_NOVA_V18_CALIBRATION_PIT_COVERAGE.json").write_text(json.dumps(result, indent=2))
    tg.to_csv(OUT / "SPORTS_NOVA_V18_CALIBRATION_PIT_COVERAGE_TEAM_GAMES.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
