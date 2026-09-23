"""SPORTS-NOVA terminal tests (headless; no network).  Real V23 simulations at 100 / 1,000 sims."""
from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from worker.sports_nova_modular.firewall import M1MarketInputError, assert_market_free
from worker.sports_nova_terminal import cli, core
from worker.sports_nova_terminal.core import ManualBook, TerminalError

ROOT = core.ROOT
GAME = "2026_02_IND_KC"  # exists in the schedule snapshot (with QB designations), independent of today's date


@pytest.fixture(autouse=True)
def isolated_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "RUNS_DIR", tmp_path / "sports_nova_runs")


@pytest.fixture(scope="module")
def spec():
    sched = core.load_schedule()
    return core.spec_from_row(next(sched[sched.game_id == GAME].itertuples()))


def absurd_book() -> ManualBook:
    return ManualBook(book_name="ABSURD", home_ml=-100000.0, away_ml=+100000.0, spread_team="KC", spread_line=-99.5,
                      spread_price=-5000.0, spread_other_price=+4000.0, total_line=500.0, over_price=-9999.0,
                      under_price=+9999.0)


def assert_batches_identical(a, b):
    assert a.state_hash == b.state_hash and a.seed == b.seed and a.n_sims == b.n_sims
    assert a.player_ids == b.player_ids and a.team_ids == b.team_ids
    assert np.array_equal(a.winner, b.winner)
    for k in a.team_stats:
        assert np.array_equal(a.team_stats[k], b.team_stats[k]), k
    for k in a.player_stats:
        assert np.array_equal(a.player_stats[k], b.player_stats[k]), k


# 1. launch ------------------------------------------------------------------------------------------------------
def test_launch_and_exit():
    proc = subprocess.run([sys.executable, str(ROOT / "sports_nova.py")], input="3\n", capture_output=True, text=True,
                          cwd=ROOT, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "[1] Simulate Game" in proc.stdout and "[2] View Previous Run" in proc.stdout and "bye" in proc.stdout


def test_module_entrypoint_launches():
    proc = subprocess.run([sys.executable, "-m", "worker.sports_nova_terminal"], input="3\n", capture_output=True,
                          text=True, cwd=ROOT, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "SPORTS-NOVA" in proc.stdout


# 2. manual game entry --------------------------------------------------------------------------------------------
def test_manual_game_entry_offschedule_and_validation():
    sched = core.load_schedule()
    s = core.manual_spec(sched, "frozen", 2026, 7, "DET", "BUF", "2026-10-25 13:00")
    assert s.source == "MANUAL" and s.game_id == "2026_07_DET_BUF" and s.kickoff_utc.tzinfo is not None
    assert s.kickoff_utc.hour == 17  # 13:00 EDT
    with pytest.raises(TerminalError):
        core.manual_spec(sched, "frozen", 2026, 7, "XXX", "BUF", "2026-10-25 13:00")
    with pytest.raises(TerminalError):
        core.manual_spec(sched, "frozen", 2026, 7, "DET", "DET", "2026-10-25 13:00")
    with pytest.raises(TerminalError):
        core.manual_spec(sched, "frozen", 2026, 7, "DET", "BUF", "not a time")
    # a matchup that is on the schedule returns the scheduled record (with QB designations)
    sch = core.manual_spec(sched, "frozen", 2026, 2, "IND", "KC", "")
    assert sch.source == "SCHEDULE" and sch.away_qb_id and sch.home_qb_id


def test_manual_game_runs_without_qb_designation():
    sched = core.load_schedule()
    s = core.manual_spec(sched, "frozen", 2026, 7, "DET", "BUF", "2026-10-25 13:00")
    r = core.execute_run(s, 100)
    assert r.record["STATUS"] == "OK"
    assert all(v.startswith("NONE") for v in r.record["FOOTBALL_DATA"]["qb_identity"].values())


# 3. simulator invocation -----------------------------------------------------------------------------------------
def test_simulator_invocation_uses_shipped_v23(spec):
    prepared = core.prepare(spec, "frozen")
    batch = core.run_simulation(prepared, 20, core.default_seed(spec.game_id))
    assert batch.model_version == core.MODEL_VERSION == "sports_nova_v23.drive_block.residual_allocation_fix.1"
    assert set(np.unique(batch.winner)) <= {"HOME", "AWAY", "TIE"} and batch.n_sims == 20
    with pytest.raises(TerminalError):
        core.run_simulation(prepared, 0, 1)
    with pytest.raises(TerminalError):
        core.run_simulation(prepared, 10, -1)


# 4 / 5. smokes ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [100, 1000])
def test_sim_smoke(spec, n):
    r = core.execute_run(spec, n)
    s = r.record["M1_SUMMARY"]
    w = s["win_probability"]
    assert r.record["STATUS"] == "OK" and s["n_sims"] == n
    assert abs(w["AWAY"] + w["HOME"] + w["TIE"] - 1) < 1e-12
    assert s["score"]["home"]["P10"] <= s["score"]["home"]["P90"] and s["score"]["home"]["mean"] >= 0
    assert len(s["players"]) > 10
    assert r.artifact_path.is_file()
    assert r.m2.source_sim_hash == r.m1.hash


# 6 / 7. isolation ------------------------------------------------------------------------------------------------
def test_m1_has_no_market_parameter():
    assert "book" not in inspect.signature(core.run_simulation).parameters
    code = core.run_simulation.__code__  # names actually referenced by the function body (docstring excluded)
    assert not [n for n in code.co_names + code.co_varnames if any(w in n.lower() for w in ("book", "odds", "line", "price"))]
    assert set(inspect.signature(core.run_simulation).parameters) == {"prepared", "n_sims", "seed"}


def test_schedule_market_columns_never_loaded():
    sched = core.load_schedule()
    for c in ("spread_line", "total_line", "home_moneyline", "away_moneyline", "over_odds", "under_odds", "result", "home_score"):
        assert c not in sched.columns


def test_firewall_rejects_market_keys_in_state_payload(spec):
    prepared = core.prepare(spec, "frozen")
    assert_market_free(prepared.state.model_dump(mode="json"))  # real state is clean
    with pytest.raises(M1MarketInputError):
        assert_market_free({**prepared.state.model_dump(mode="json"), "spread_line": -3.5, "sportsbook_odds": -110})


def test_market_book_cannot_change_m1_output_same_seed(spec):
    """RUN A: no external book.  RUN B: absurd external book.  Raw draw arrays must be byte-identical."""
    a = core.execute_run(spec, 100, seed=12345, book=None)
    b = core.execute_run(spec, 100, seed=12345, book=absurd_book())
    assert a.record["M4_COMPARISON"] is None and b.record["M4_COMPARISON"] is not None
    assert_batches_identical(a.batch, b.batch)
    assert a.m1.hash == b.m1.hash
    assert a.record["M1_SUMMARY"] == b.record["M1_SUMMARY"]
    assert a.record["M2_NOVA_BOOK"] == b.record["M2_NOVA_BOOK"]
    assert a.record["M1_INPUT_HASH"] == b.record["M1_INPUT_HASH"]
    contamination = 0 if (a.m1.hash == b.m1.hash and np.array_equal(a.batch.winner, b.batch.winner)) else 1
    assert contamination == 0  # M1_MARKET_CONTAMINATION=0


def test_attaching_books_after_the_run_leaves_batch_untouched(spec):
    r = core.execute_run(spec, 100, seed=7)
    before = {k: v.copy() for k, v in r.batch.team_stats.items()}
    before_hash, before_m1 = r.m1.hash, json.dumps(r.record["M1_SUMMARY"], sort_keys=True)
    core.attach_book(r, absurd_book())
    core.attach_book(r, ManualBook(home_ml=-150.0))
    for k, v in before.items():
        assert np.array_equal(v, r.batch.team_stats[k])
    assert r.m1.hash == before_hash and json.dumps(r.record["M1_SUMMARY"], sort_keys=True) == before_m1
    assert len(r.record["M3_EXTERNAL_BOOKS"]) == 2


# 8. artifact -----------------------------------------------------------------------------------------------------
def test_run_artifact_serialization_and_replay(spec):
    r = core.execute_run(spec, 100, seed=99)
    book = ManualBook(book_name="TestBook", home_ml=-130.0, away_ml=110.0, spread_team="KC", spread_line=-3.5,
                      spread_price=-110.0, spread_other_price=-110.0, total_line=47.5, over_price=-110.0, under_price=-110.0)
    core.attach_book(r, book)
    path = core.save_artifact(r.record)
    assert path.parent.name == datetime.now().strftime("%Y%m%d") and path.name.startswith(GAME + "_")
    rec = json.loads(path.read_text(encoding="utf-8"))
    for k in ("GAME_INPUTS", "FOOTBALL_DATA", "M1_RUN", "M1_SUMMARY", "M2_NOVA_BOOK", "M3_EXTERNAL_BOOK", "M4_COMPARISON", "WARNINGS"):
        assert k in rec
    m1 = rec["M1_RUN"]
    assert m1["simulation_version"] == core.MODEL_VERSION and m1["random_seed"] == 99 and m1["simulation_count"] == 100
    assert len(m1["model_hash"]) == 64 and len(m1["input_hash"]) == 64 and m1["forbidden_m1_inputs_present"] is False
    assert rec["FOOTBALL_DATA"]["panel"]["sha256"] and rec["FOOTBALL_DATA"]["schedule_snapshot"]["market_columns_loaded"] is False
    # the football-input side of the artifact contains no market keys; the book lives only under M3/M4
    assert_market_free(rec["GAME_INPUTS"])
    assert_market_free(rec["FOOTBALL_DATA"])
    assert rec["M3_EXTERNAL_BOOK"]["home_ml"] == -130.0
    # previous-run view renders entirely from the saved file
    lines: list = []
    cli.render_run(rec, lines.append)
    text = "\n".join(lines)
    assert "NOVA GAME SIM" in text and "NOVA FAIR BOOK" in text and "NOVA VS MARKET" in text
    assert "EXTERNAL MARKET" in text and "NOVA FAIR VALUE" in text and "DIFFERENCE" in text
    assert "informational, not bets" in text and not re.search(r"LOCK|GUARANTEED", text.upper())


# comparator maths ------------------------------------------------------------------------------------------------
def test_odds_math_and_devig():
    assert core.implied_prob(-110) == pytest.approx(110 / 210)
    assert core.implied_prob(150) == pytest.approx(0.4)
    a, b, over = core.devig(core.implied_prob(-130), core.implied_prob(110))
    assert a + b == pytest.approx(1.0) and a == pytest.approx(0.5427, abs=1e-3) and over > 0
    assert core.prob_to_american(0.6) == pytest.approx(-150) and core.prob_to_american(0.4) == pytest.approx(150)
    for bad in ("abc", "50", "-99", "nan"):
        with pytest.raises(TerminalError):
            core.parse_american(bad)
    assert core.parse_american("") is None and core.parse_american("+150") == 150 and core.parse_american("even") == 100


def test_cover_and_over_probabilities_on_known_distribution():
    margin = np.array([-7, -3, -3, 0, 3, 3, 3, 10], dtype=float)  # home minus away
    c = core._cover(margin, "HOME", -3.0)   # home covers -3 when margin > 3 (only +10); push at margin==3 (3 sims)
    assert (c["win"], c["push"], c["lose"]) == (1 / 8, 3 / 8, 4 / 8)
    assert c["win_ex_push"] == pytest.approx(1 / 5)
    c2 = core._cover(margin, "AWAY", 3.5)   # away +3.5 covers when margin < 3.5 -> margin in {-7,-3,-3,0,3,3,3} = 7/8
    assert c2["win"] == 7 / 8
    o = core._over(np.array([40.0, 47.0, 47.0, 50.0]), 47.0)
    assert (o["over"], o["push"], o["under"]) == (0.25, 0.5, 0.25) and o["over_ex_push"] == 0.5


def test_partial_book_and_one_sided_prices_are_labelled_not_invented(spec):
    r = core.execute_run(spec, 100, seed=5)
    e = core.attach_book(r, ManualBook(home_ml=-150.0, total_line=44.5, over_price=-105.0))
    assert e["market_detail"]["moneyline"]["vig_adjusted"] is False
    assert e["market_detail"]["total"]["vig_adjusted"] is False
    assert {c["market"] for c in e["m4"]} == {"HOME_ML", "TOTAL_OVER"}  # nothing fabricated for missing sides
    assert "spread" not in e["market_detail"]
    lines: list = []
    cli.render_comparison(r.record, lines.append)
    assert "raw, includes vig" in "\n".join(lines)
    with pytest.raises(TerminalError):
        core.attach_book(r, ManualBook(spread_team="XXX", spread_line=-3.0, spread_price=-110.0))


# resilience ------------------------------------------------------------------------------------------------------
def test_simulator_exception_saves_failed_artifact(spec, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("engine exploded")
    monkeypatch.setattr(core, "simulate_game", boom)
    with pytest.raises(TerminalError, match="engine exploded"):
        core.execute_run(spec, 10)
    saved = core.list_artifacts()
    assert len(saved) == 1
    rec = json.loads(saved[0].read_text())
    assert rec["STATUS"] == "FAILED" and "engine exploded" in rec["ERROR"]
    lines: list = []
    cli.render_run(rec, lines.append)
    assert any("ERROR" in ln for ln in lines)


def test_missing_prior_data_is_blocked_not_invented():
    sched = core.load_schedule()
    s = core.manual_spec(sched, "frozen", 1995, 3, "DET", "BUF", "1995-09-24 13:00")
    with pytest.raises(TerminalError):
        core.prepare(s, "frozen")


def test_missing_injury_report_degrades_gracefully(spec, monkeypatch):
    monkeypatch.setattr(core, "INJURIES_PARQUET", Path("does_not_exist.parquet"))
    p = core.prepare(spec, "frozen")
    assert p.football_data["injury_report"]["status"] == "NO_DATA"
    assert any("injury" in w for w in p.warnings)
    assert core.execute_run(spec, 20, prepared=p, save=False).record["STATUS"] == "OK"


def test_manual_out_player_and_qb_override_are_applied_and_recorded(spec):
    base = core.prepare(spec, "frozen")
    rb = next(r for r in base.roster if r["team"] == "IND" and r["position"] == "RB" and r["availability"] == "UNKNOWN")
    ov = core.Overrides(out_players={rb["player_id"]: rb["name"]}, notes="operator note")
    p = core.prepare(spec, "frozen", ov)
    assert next(r for r in p.roster if r["player_id"] == rb["player_id"])["availability"] == "OUT"
    assert p.football_data["players_out_manual"] == [rb["player_id"]]
    r = core.execute_run(spec, 100, seed=3, prepared=p)
    assert r.record["GAME_INPUTS"]["operator_overrides"]["out_players"]
    idx = r.batch.player_ids.index(rb["player_id"])
    assert r.batch.player_stats["rush_attempts"][:, idx].sum() == 0  # OUT player gets no carries
    assert r.record["M1_HASH"] != core.execute_run(spec, 100, seed=3).record["M1_HASH"]  # football edit DOES change M1


def test_identity_resolution_matches_production_slate_script(spec):
    """Mirror check: terminal's scheduled-QB resolution == scripts/sports_nova_m1_v23_upcoming_slate.build_identity_resolutions."""
    from scripts import sports_nova_m1_v23_upcoming_slate as slate_script
    from worker.sports_nova_v23 import simulator as v23
    core.prepare(spec, "frozen")
    installed = dict(v23.__dict__["_IDENTITY_RESOLUTIONS"]) if "_IDENTITY_RESOLUTIONS" in v23.__dict__ else None
    if installed is None:
        from worker.sports_nova_v19 import simulator as v19
        installed = dict(v19._IDENTITY_RESOLUTIONS)
    sched = core.load_schedule()
    row = sched[sched.game_id == GAME]
    ts = json.loads(core.RECEIPT_JSON.read_text())["SOURCES"]["schedule"]["fetched_at_utc"]
    expected = slate_script.build_identity_resolutions(row, ts)
    assert installed == expected


# scripted end-to-end CLI -----------------------------------------------------------------------------------------
def scripted(answers):
    queue = list(answers)
    prompts: list = []

    def _inp(prompt=""):
        prompts.append(prompt)
        if not queue:
            raise EOFError
        return queue.pop(0)
    return _inp, prompts, queue


def test_cli_full_flow_manual_entry_book_and_previous_run(monkeypatch):
    monkeypatch.setattr(core, "upcoming_slate", lambda sched, now=None: sched.iloc[0:0])  # time-independent: manual path
    answers = ["1",                      # main menu: simulate
               "",                       # panel default (frozen)
               "M", "2026", "2", "IND", "KC", "",   # manual game entry (on schedule -> scheduled record)
               "n",                      # edit football inputs?
               "1",                      # 100 sims
               "",                       # default seed
               "",                       # run? default Y
               "y",                      # view players
               "y",                      # enter book
               "TestBook", "-130", "+110", "KC", "-3.5", "-110", "-110", "47.5", "-110", "-110", "",
               "n",                      # another book?
               "2", "1", "n",            # view previous run #1, no players
               "3"]
    inp, prompts, queue = scripted(answers)
    out: list = []
    rc = cli.App(inp, out.append).main()
    text = "\n".join(map(str, out))
    assert rc == 0 and not queue, f"unconsumed inputs: {queue}"
    for needle in ("NOVA GAME SIM", "WIN PROBABILITY", "NOVA FAIR BOOK", "NOVA_FAIR_SPREAD", "PLAYER PROJECTIONS",
                   "NOVA VS MARKET", "MONEYLINE", "SPREAD", "TOTAL", "Saved:", "vig-adjusted"):
        assert needle in text, needle
    assert len(core.list_artifacts()) == 1
    rec = json.loads(core.list_artifacts()[0].read_text())
    assert rec["M4_COMPARISON"] and rec["M1_RUN"]["simulation_count"] == 100


def test_cli_rejects_bad_sim_counts_and_bad_prices(monkeypatch):
    monkeypatch.setattr(core, "upcoming_slate", lambda sched, now=None: sched.iloc[0:0])
    inp, _, queue = scripted(["6", "abc", "6", "-5", "6", "0", "6", "250"])
    app = cli.App(inp, lambda *_: None)
    assert app.pick_sims() == 250
    inp2, _, _ = scripted(["50", "-110"])
    assert cli.App(inp2, lambda *_: None)._field("HOME_ML", core.parse_american) == -110.0
    inp3, _, _ = scripted(["banana"])
    app3 = cli.App(inp3, lambda *_: None)
    with pytest.raises(TerminalError):
        app3.pick_panel()
