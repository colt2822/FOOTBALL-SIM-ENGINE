"""Focused contract tests for Node4 joint exports."""
from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

import numpy as np

from .. import joint_export as J
from .. import montecarlo as MC


def _sim(seed=11):
    spec = MC.GameSpec("NE", "SEA", 64, 63, 0.58, 0.56, [
        MC.PlayerSpec("qb", "QB", "home", 250, 45),
        MC.PlayerSpec("wr", "WR", "home", 75, 25, share=.26),
        MC.PlayerSpec("rb", "RB", "home", 62, 25, share=.55),
        MC.PlayerSpec("opp", "QB", "away", 230, 45),
    ])
    return MC.GameSimulator({"qb_vs_opposing_qb": .05,
                             "qb_vs_own_lead_receiver": .55,
                             "qb_pass_vs_own_lead_rb_rush": -.15,
                             "receiver1_vs_receiver2": -.05}).simulate(spec, 5000, seed)


def test_same_game_qb_wr_joint_and_delta():
    sim = _sim()
    out = J.export_joint_probability({"game_id": "G", "legs": [
        {"player": "qb", "market": "QB_PASS_YARDS", "side": "over", "line": 250},
        {"player": "wr", "market": "LEAD_WR_REC_YARDS", "side": "over", "line": 70},
    ]}, sim, model_id="M", model_version="v", model_sha256="sha",
    simulation_id="S", simulation_seed=11)
    assert out["DEPENDENCY_STATUS"] == "SIMULATED_JOINT"
    assert out["JOINT_MODEL_PROBABILITY"] > out["MARGINAL_PRODUCT_PROBABILITY"]
    assert abs(out["DEPENDENCY_DELTA"] - (out["JOINT_MODEL_PROBABILITY"] -
                                           out["MARGINAL_PRODUCT_PROBABILITY"])) < 1e-12


def test_qb_rb_and_opposing_qb_cases_are_correlated():
    sim = _sim()
    for legs in [
        [{"player": "qb", "market": "QB_PASS_YARDS", "side": "over", "line": 250},
         {"player": "rb", "market": "LEAD_RB_RUSH_YARDS", "side": "over", "line": 60}],
        [{"player": "qb", "market": "QB_PASS_YARDS", "side": "over", "line": 250},
         {"player": "opp", "market": "OPPOSING_QB_PASS_YARDS", "side": "over", "line": 225}],
    ]:
        out = J.export_joint_probability({"game_id": "G", "legs": legs}, sim,
            model_id="M", model_version="v", model_sha256="sha",
            simulation_id="S", simulation_seed=11)
        assert out["DEPENDENCY_STATUS"] == "SIMULATED_JOINT"
        assert 0 <= out["JOINT_MODEL_PROBABILITY"] <= 1
        assert out["JOINT_P05"] <= out["JOINT_P50"] <= out["JOINT_P95"]


def test_seeded_simulation_and_model_sha_are_deterministic():
    a, b = _sim(17), _sim(17)
    np.testing.assert_array_equal(a["draws"]["qb"], b["draws"]["qb"])
    out = J.export_joint_probability({"game_id": "G", "legs": [
        {"player": "qb", "market": "QB_PASS_YARDS", "side": "over", "line": 250},
    ]}, a, model_id="M", model_version="v", model_sha256="MODEL_SHA",
    simulation_id="S", simulation_seed=17)
    assert out["MODEL_SHA256"] == "MODEL_SHA"


def test_no_independence_fallback_and_unresolved_leg():
    sim = _sim()
    out = J.export_joint_probability({"game_id": "G", "legs": [
        {"player": "qb", "market": "QB_PASS_YARDS", "side": "over", "line": 250},
        {"player": "missing", "market": "QB_PASS_YARDS", "side": "over", "line": 200},
    ]}, sim, model_id="M", model_version="v", model_sha256="sha",
    simulation_id="S", simulation_seed=11)
    assert out["DEPENDENCY_STATUS"] == "UNRESOLVED_DEPENDENCY"
    assert out["JOINT_MODEL_PROBABILITY"] is None
    assert "INDEPENDENT_PRODUCT" not in json.dumps(out)


def test_schema_sample_export_and_market_rejection():
    sim = _sim()
    sim["marginal_model_status"] = {"QB_PASS_YARDS": "VALIDATED"}
    sim["market_state_map"] = {"QB_PASS_YARDS": "qb", "LEAD_WR_REC_YARDS": "wr",
                                "LEAD_RB_RUSH_YARDS": "rb", "OPPOSING_QB_PASS_YARDS": "opp"}
    with tempfile.TemporaryDirectory() as td:
        path = J.export_simulation_samples(sim, Path(td) / "samples.jsonl.gz",
            model_id="M", model_version="v", model_sha256="sha", simulation_id="S")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rows = [json.loads(next(fh))]
            rows.append(json.loads(next(fh)))
        assert rows[0]["MODEL_SHA256"] == "sha"
        assert rows[0]["MARGINAL_MODEL_STATUS"]["QB_PASS_YARDS"] == "VALIDATED"
        assert set(("QB_PASS_YARDS", "LEAD_WR_REC_YARDS", "LEAD_RB_RUSH_YARDS",
                    "OPPOSING_QB_PASS_YARDS")).issubset(rows[1])
        assert rows[1]["path_index"] == 0
    try:
        J.export_joint_probability({"game_id": "G", "odds": 1.2, "legs": []}, sim,
            model_id="M", model_version="v", model_sha256="sha",
            simulation_id="S", simulation_seed=11)
    except ValueError:
        pass
    else:
        raise AssertionError("market data must be rejected")
