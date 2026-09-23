"""Focused tests for the V2 engine.

Run: python -m worker.sports_nova_v2.tests.test_v2
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .. import config as C
from .. import distribution as D
from .. import features as F
from .. import firewall as FW
from .. import models as Mo
from .. import montecarlo as MC
from .. import panel as P
from ..walkforward import walk_forward

_PANEL = {}


def panel():
    if "el" not in _PANEL:
        el, led, full = F.build_v2_panel(return_full=True)
        _PANEL.update(el=el, led=led, full=full)
    return _PANEL["el"], _PANEL["led"], _PANEL["full"]


# ---------------------------------------------------------------------------
def test_shift1_synthetic():
    """A trailing mean must equal the mean of strictly prior games."""
    df = pd.DataFrame({
        "player_id": ["A"] * 6,
        "passing_yards": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
    })
    got = F.trailing(df, "player_id", "passing_yards", 3).values
    assert np.isnan(got[0])
    assert got[1] == 100.0
    assert got[2] == 150.0
    assert got[3] == 200.0
    assert got[4] == 300.0                     # mean(200,300,400)
    assert got[5] == 400.0                     # mean(300,400,500)


def test_shift1_never_sees_own_game():
    """Perturbing a game's outcome must not change that game's own feature."""
    df = pd.DataFrame({"player_id": ["A"] * 6,
                       "passing_yards": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]})
    base = F.trailing(df, "player_id", "passing_yards", 3).values
    df2 = df.copy()
    df2.loc[4, "passing_yards"] = 9999.0
    bumped = F.trailing(df2, "player_id", "passing_yards", 3).values
    assert bumped[4] == base[4], "feature moved when its own game changed"
    assert bumped[5] != base[5], "later game should react to the change"


def test_no_future_leakage_on_real_panel():
    el, _, full = panel()
    res = FW.run_all(el, full)
    assert res["TEMPORAL_FIREWALL"] == "PASS", json.dumps(res, indent=2, default=str)


def test_team_opponent_joins_are_causal():
    el, _, _ = panel()
    r = FW.check_team_join_causality(el)
    assert r["PASS"], r
    assert r["opponent_rows_checked"] > 10_000


def test_season_boundary_resets():
    _, _, full = panel()
    r = FW.check_season_boundary(full)
    assert r["PASS"], r


def test_rookie_prior_falls_back_to_league():
    """With no career history the shrunk value must equal the prior exactly."""
    raw = np.array([np.nan, 100.0])
    n = np.array([0.0, 4.0])
    prior = np.array([220.0, 220.0])
    out = F.shrink(raw, n, prior, k=10.0)
    assert out[0] == 220.0
    expected = (4.0 * 100.0 + 10.0 * 220.0) / 14.0
    assert abs(out[1] - expected) < 1e-9


def test_shrinkage_pulls_small_samples_harder():
    """A 2-game sample must sit closer to the prior than a 20-game sample."""
    raw = np.array([400.0, 400.0])
    n = np.array([2.0, 20.0])
    prior = np.array([220.0, 220.0])
    out = F.shrink(raw, n, prior, k=10.0)
    assert abs(out[0] - 220.0) < abs(out[1] - 220.0)


def test_missing_history_rows_are_kept_not_dropped():
    """V2 must score exactly V1's rows; imputation, never row removal."""
    el, _, _ = panel()
    qb1 = P.build_qb_panel()
    el1 = P.apply_v1_eligibility(qb1)
    assert len(el) == len(el1) == 14666
    assert P.row_key(el) == P.row_key(el1)


def test_imputer_handles_all_nan_column():
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [np.nan] * 3})
    prep = Mo.ImputeStandardize().fit(X)
    Z = prep.transform(X)
    assert np.isfinite(Z).all(), "imputation left non-finite values"


def test_probability_curve_is_monotonic():
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 80, 5000)
    curve = D.dense_curve(250.0, sample)
    keys = sorted(float(k) for k in curve)
    vals = [curve[str(round(k, 1))] for k in keys]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))
    assert vals[0] >= 0.99 and vals[-1] <= 0.01


def test_probability_curve_bounds():
    rng = np.random.default_rng(1)
    curve = D.dense_curve(200.0, rng.normal(0, 70, 2000))
    for v in curve.values():
        assert 0.0 <= v <= 1.0


def test_calibration_boundaries():
    """Calibrated probabilities stay in [0,1] and degrade to identity when
    there is not enough history to fit."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)
    for method in ("isotonic", "platt"):
        cal = D.CausalCalibrator(method).fit(p, y)
        out = cal.transform(np.array([0.0, 0.5, 1.0, -0.2, 1.4]))
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
    empty = D.CausalCalibrator("isotonic").fit(p[:10], y[:10])
    assert not empty.fitted_
    np.testing.assert_allclose(empty.transform(p[:5]), p[:5])


def test_calibration_never_fits_on_test_period():
    """The calibrator for season s must be blind to season s and later."""
    el, _, _ = panel()
    seen = []

    class Spy(D.CausalCalibrator):
        def fit(self, p, y, min_n=400):
            seen.append(len(p))
            return super().fit(p, y, min_n)

    orig = D.CausalCalibrator
    D.CausalCalibrator = Spy
    try:
        from .. import experiment as E
        oos, folds = E.run_candidate(el, Mo.make_ridge(C.V1_FEATURES),
                                     seasons=[2006, 2007, 2008])
    finally:
        D.CausalCalibrator = orig
    # First fold has no earlier OOS data, so no fit is attempted.
    assert len(seen) == 2, seen
    assert seen[0] < seen[1], "calibration pool must only grow with elapsed time"


def test_walkforward_splits_are_chronological():
    el, _, _ = panel()
    folds = walk_forward(el, Mo.make_ridge(C.V1_FEATURES))
    assert len(folds) == 20, len(folds)
    for f in folds:
        assert f.cutoff_ok, f.season
    seasons = [f.season for f in folds]
    assert seasons == sorted(seasons) == list(range(2006, 2026))


def test_walkforward_train_is_strictly_earlier():
    el, _, _ = panel()
    for s in (2010, 2018, 2025):
        train = el[el.season < s]
        test = el[el.season == s]
        assert train.kickoff_timestamp_utc.max() < test.kickoff_timestamp_utc.min()


def test_no_market_columns_anywhere():
    el, led, _ = panel()
    P._assert_no_market_columns(el)
    P._assert_no_market_columns(led)


def test_air_yards_masked_before_2006():
    _, _, full = panel()
    pre = full[full.season < 2006]
    assert pre.passing_air_yards.notna().sum() == 0, \
        "pre-2006 air yards must be NaN, not a zero sentinel"
    post = full[full.season >= 2006]
    assert post.passing_air_yards.notna().mean() > 0.9


def test_excluded_columns_absent_from_feature_list():
    for c in F.EXCLUDED_SEASON_VARYING:
        assert c not in F.V2_FEATURES


def test_deterministic_frozen_artifacts():
    """Two identical runs must produce bit-identical predictions."""
    el, _, _ = panel()
    fp = Mo.make_lgbm(F.V2_FEATURES)
    train = el[el.season < 2015]
    test = el[el.season == 2015]
    a, _, _ = fp(train, test)
    b, _, _ = fp(train, test)
    np.testing.assert_array_equal(a, b)


def test_simulator_is_deterministic_and_correlated():
    corr = {"qb_vs_opposing_qb": 0.05, "qb_vs_own_lead_receiver": 0.55,
            "receiver1_vs_receiver2": -0.05, "qb_pass_vs_own_lead_rb_rush": -0.10}
    spec = MC.GameSpec(
        home_team="NE", away_team="BUF",
        exp_plays_home=64, exp_plays_away=63,
        exp_pass_rate_home=0.58, exp_pass_rate_away=0.56,
        players=[
            MC.PlayerSpec("qb_h", "QB", "home", 245.0, 70.0),
            MC.PlayerSpec("wr_h", "WR", "home", 72.0, 34.0, share=0.26),
            MC.PlayerSpec("rb_h", "RB", "home", 61.0, 30.0, share=0.55),
            MC.PlayerSpec("qb_a", "QB", "away", 231.0, 68.0),
        ])
    sim = MC.GameSimulator(corr)
    a = sim.simulate(spec, n_paths=4000, seed=7)
    b = sim.simulate(spec, n_paths=4000, seed=7)
    np.testing.assert_array_equal(a["draws"]["qb_h"], b["draws"]["qb_h"])

    r = np.corrcoef(a["draws"]["qb_h"], a["draws"]["wr_h"])[0, 1]
    assert r > 0.25, f"QB and own receiver should co-move, got {r:.3f}"

    j = MC.joint_p_over(a, [("qb_h", 250.0), ("wr_h", 70.0)])
    assert j["JOINT_PROBABILITY"] > j["INDEPENDENCE_ASSUMPTION"], j
    assert 0.0 <= j["JOINT_PROBABILITY"] <= 1.0


def test_simulator_correlation_matrix_is_psd():
    R = MC._nearest_psd(np.array([[1.0, 0.9, -0.8],
                                  [0.9, 1.0, 0.95],
                                  [-0.8, 0.95, 1.0]]))
    vals = np.linalg.eigvalsh(R)
    assert (vals > 0).all(), vals
    np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-9)


def test_v1_reproduction_gate():
    from ..v1_baseline import run_v1
    r = run_v1()
    assert r["reproduction"]["GATE_PASS"], r["reproduction"]
    assert r["metrics"]["N"] == C.V1_PUBLISHED["OOS_N"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}", flush=True)
        except Exception as exc:
            failed.append((t.__name__, exc))
            print(f"FAIL  {t.__name__}: {exc}", flush=True)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
