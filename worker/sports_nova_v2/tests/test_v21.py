"""Tests for the V2.1 season-opener fix.

Run: python -m worker.sports_nova_v2.tests.test_v21
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .. import config as C
from .. import features_v21 as V21
from .. import firewall as FW
from .. import panel as P
from ..final_run_v21 import (check_offseason_state_separation,
                             check_prev_game_share_offset)

_P = {}


def panel():
    if "el" not in _P:
        el, led, full = V21.build_v21_panel(return_full=True)
        _P.update(el=el, led=led, full=full)
    return _P["el"], _P["led"], _P["full"]


def test_row_set_still_pinned_to_v1():
    el, _, _ = panel()
    el1 = P.apply_v1_eligibility(P.build_qb_panel())
    assert len(el) == len(el1) == 14666
    assert P.row_key(el) == P.row_key(el1)


def test_season_opener_and_in_season_absence_are_distinguishable():
    """The whole point of V2.1: the two states must not share a feature vector."""
    el, _, _ = panel()
    r = check_offseason_state_separation(el)
    assert r["PASS"], r
    assert r["season_opener_n"] > 500 and r["in_season_absence_n"] > 200


def test_offseason_gap_is_not_clipped_to_30():
    el, _, _ = panel()
    op = el[(el.week == 1) & (el.crossed_season_boundary == 1)]
    assert op.offseason_gap_days.min() > 100, "offseason gap collapsed to the 30d clip"
    assert op.offseason_gap_days.max() <= V21.OFFSEASON_GAP_CAP


def test_in_season_rest_is_nan_across_a_season_boundary():
    el, _, _ = panel()
    crossed = el[el.crossed_season_boundary == 1]
    assert crossed.in_season_days_rest.isna().all()
    same = el[(el.crossed_season_boundary == 0) & (el.games_played_prior > 0)]
    assert same.in_season_days_rest.notna().mean() > 0.95


def test_team_games_missed_is_zero_across_offseason():
    el, _, _ = panel()
    assert (el.loc[el.crossed_season_boundary == 1, "team_games_missed"] == 0).all()
    ins = el[(el.crossed_season_boundary == 0) & (el.days_rest >= 29.9)]
    assert ins.team_games_missed.mean() > 1.0, "in-season absence should miss games"


def test_prev_game_attempt_share_has_no_off_by_one():
    """Guards a leak the |r|>0.90 correlation scan is too blunt to catch."""
    _, _, full = panel()
    r = check_prev_game_share_offset(full)
    assert r["PASS"], r
    assert r["rows_checked"] > 5000


def test_prev_game_share_is_not_current_game_share():
    _, _, full = panel()
    df = full[full.own_share_this_game.notna() &
              full.prev_game_attempt_share.notna()]
    same = np.isclose(df.own_share_this_game, df.prev_game_attempt_share)
    # Some coincidental equality is fine (both 1.0 for a full-time starter);
    # near-total equality would mean the current game leaked in.
    assert same.mean() < 0.95, f"prev-game share tracks current game ({same.mean():.3f})"


def test_role_features_use_only_prior_seasons():
    """prior_season_* must describe season-1, never the current season."""
    el, _, _ = panel()
    sub = el[el.prior_season_games.notna()]
    assert len(sub) > 5000
    # A rookie season cannot have a prior-season record.
    rookies = el[el.games_played_prior <= 4]
    if len(rookies):
        assert rookies.prior_season_games.isna().mean() > 0.3


def test_firewall_still_passes_with_v21_features():
    el, _, full = panel()
    res = FW.run_all(el, full)
    assert res["TEMPORAL_FIREWALL"] == "PASS", json.dumps(res, indent=2, default=str)


def test_buckets_partition_the_ceiling_cohort():
    el, _, _ = panel()
    b = V21.buckets(el)
    ceil = (el.days_rest >= 29.9)
    total = int((b["SEASON_OPENER"] | b["LATE_SEASON_DEBUT"] |
                 b["LONG_IN_SEASON_REST"]).sum())
    assert total == int(ceil.sum()), (total, int(ceil.sum()))
    # and they are mutually exclusive
    assert int((b["SEASON_OPENER"] & b["LATE_SEASON_DEBUT"]).sum()) == 0
    assert int((b["LATE_SEASON_DEBUT"] & b["LONG_IN_SEASON_REST"]).sum()) == 0


def test_week1_is_in_distribution_not_a_backup_regime():
    """Regression test for the incorrect Node4 blocker."""
    el, _, _ = panel()
    w1 = el[(el.week == 1) & (el.season >= 2006)]
    assert len(w1) > 600
    assert w1.passing_yards.mean() > 200, w1.passing_yards.mean()
    ins = el[(el.days_rest >= 29.9) & (el.week > 1) &
             (el.crossed_season_boundary == 0)]
    # The two cohorts sharing days_rest==30 are genuinely different populations.
    assert w1.passing_yards.mean() - ins.passing_yards.mean() > 80


def test_frozen_v21_artifact_is_complete_and_hashed():
    """V2's artifact omitted the scale model's preprocessing constants, making
    sigma irreproducible. V2.1 must not repeat that."""
    import hashlib
    p = C.ARTIFACTS / "SPORTS_NOVA_V2_1_FROZEN_MODEL.json"
    assert p.is_file(), "V2.1 frozen model not written"
    fm = json.loads(p.read_text())
    d = fm["DISTRIBUTION"]
    for k in ("scale_impute_median", "scale_standardize_mu",
              "scale_standardize_sigma", "scale_indicator_columns",
              "scale_sigma_multiplier"):
        assert k in d, f"missing {k} - sigma would not be reproducible"
    for f, h in (("GBM_BOOSTER_FILE", "GBM_BOOSTER_SHA256"),):
        path = C.ARTIFACTS / fm[f]
        assert path.is_file()
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        assert got == fm[h], f"{f} hash mismatch"
    zp = C.ARTIFACTS / d["z_pool_file"]
    assert hashlib.sha256(zp.read_bytes()).hexdigest() == d["z_pool_sha256"]


def test_v2_artifacts_untouched_by_v21():
    """V2 must remain a valid, independently verifiable champion."""
    import hashlib
    man = json.loads((C.ARTIFACTS / "SPORTS_NOVA_V2_ENGINE_MANIFEST.json").read_text())
    bad = []
    for name, expected in man["ARTIFACT_SHA256"].items():
        p = C.ARTIFACTS / name
        if not p.is_file():
            bad.append((name, "missing"))
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(8192), b""):
                h.update(c)
        if h.hexdigest() != expected:
            bad.append((name, "hash changed"))
    assert not bad, bad


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
