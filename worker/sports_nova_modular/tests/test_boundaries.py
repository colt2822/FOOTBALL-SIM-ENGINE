from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from worker.sports_nova_modular.audit import run_architecture_audit
from worker.sports_nova_modular.contracts import M1OutputSchema, M2OutputSchema
from worker.sports_nova_modular.firewall import M1MarketInputError, assert_market_free
from worker.sports_nova_modular.health import simulator_health_report
from worker.sports_nova_modular.market_book import normalize_market_observation
from worker.sports_nova_modular.nova_book import build_nova_book
from worker.sports_nova_modular.comparator import compare_nova_to_market


UTC = timezone.utc


def _m1() -> M1OutputSchema:
    return M1OutputSchema(
        game_id="g1",
        simulation_version="v21",
        simulation_count=2,
        team_distributions={},
        player_distributions={},
        joint_distributions={},
        metadata={},
        input_cutoff_ts=datetime(2026, 9, 17, tzinfo=UTC),
        hash="a" * 64,
    )


def test_m1_firewall_rejects_market_and_strategy_keys():
    with pytest.raises(M1MarketInputError):
        assert_market_free({"weather": {"temperature": 70}, "sportsbook_odds": -110})
    with pytest.raises(M1MarketInputError):
        assert_market_free({"strategy_id": "value"})


def test_m2_m3_m4_chain_is_explicit_and_one_way():
    simulation = _m1()
    nova = build_nova_book(
        simulation,
        nova_book_version="book-v1",
        fair_prices={"HOME_ML": 0.60},
        fair_lines={},
        probabilities={"HOME_ML": 0.60},
    )
    market = normalize_market_observation(
        game_id="g1", venue="test", market="HOME_ML",
        timestamp=datetime(2026, 9, 17, tzinfo=UTC), price=0.55,
    )
    comparison = compare_nova_to_market(nova, market)
    assert comparison.delta == pytest.approx(0.05)
    assert comparison.game_id == "g1"


def test_health_report_has_only_simulator_metrics():
    report = simulator_health_report(
        simulator_version="V21",
        structural_test_status="PASS",
        temporal_firewall="PASS",
        data_health={"status": "CHECKED"},
        game_metrics={"score_MAE": None},
        team_metrics={},
        player_metrics={},
        distribution_metrics={},
        top_failure_modes=[],
    )
    assert "GAME_METRICS" in report
    assert "PNL" not in report


def test_static_audit_has_no_m1_downstream_imports():
    repository_root = Path(__file__).resolve().parents[3]
    audit = run_architecture_audit(repository_root)
    assert audit["MODULAR_RESET_STATUS"] == "PASS"
    assert audit["M1_IMPORT_VIOLATIONS"] == []
    assert audit["V22_CLASSIFICATION"] == "NOVA_BOOK_CALIBRATION_COMPONENT"
