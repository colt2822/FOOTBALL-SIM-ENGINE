"""Operator entry for the V28 game-state scoring engine (explicit, opt-in; the legacy V23 terminal default is untouched).

    python sports_nova.py --engine v28 --game 2026_02_IND_KC [--sims 5000] [--variant 6] [--panel frozen] [--seed N]

Sequence (enforced by types, not by convention):  simulate -> build NOVA book -> SEAL (canonical-JSON sha256) -> only then may a market book be joined.  `join_market`
takes a SealedBook and re-verifies its hash; nothing about a market quote can reach the simulator, and a book edited after sealing is refused.

State path note: the terminal prepares a V23-level PregameState from the panel.  The V25 role-aware repair and V26 completion layers need a RosterSnapshot (only the Sunday-panel
runner has one) and are NOT applied here; `state_layers` says so in every output.  Calibration status is UNCALIBRATED_RESEARCH_ONLY until an M2 exists for this exact engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.sports_nova_modular.firewall import assert_market_free  # noqa: E402
from worker.sports_nova_terminal import core  # noqa: E402
from worker.sports_nova_v28_game_state_scoring import config as cfg  # noqa: E402
from worker.sports_nova_v28_game_state_scoring import simulator as s28  # noqa: E402

PKG = ROOT / "worker" / "sports_nova_v28_game_state_scoring"
CHAMPION_FILE = ROOT / "data" / "sports_nova_v3" / "V28_GAME_STATE_SCORING" / "CHAMPION.json"
CALIBRATION_STATUS = "UNCALIBRATED_RESEARCH_ONLY"
TOTAL_LINES = [x + 0.5 for x in range(29, 70)]
TEAM_LINES = [x + 0.5 for x in range(9, 40)]
CALIBRATORS = {"order_of_selection": ["Brier", "LogLoss", "ECE"], "candidates": ["RAW", "PLATT", "BETA"], "comparison_only": ["ISOTONIC"]}


class EngineError(RuntimeError):
    pass


def package_hash() -> str:
    files = {f"worker/sports_nova_v28_game_state_scoring/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PKG.glob("*.py"))}
    return hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()


def champion_variant() -> int | None:
    if not CHAMPION_FILE.is_file():
        return None
    rec = json.loads(CHAMPION_FILE.read_text())
    return int(rec["VARIANT"]) if rec.get("CHAMPION") else None


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, default=float).encode("utf-8")


@dataclass(frozen=True)
class SealedBook:
    book: dict
    sha256: str

    def verify(self) -> None:
        if hashlib.sha256(_canon(self.book)).hexdigest() != self.sha256:
            raise EngineError("NOVA book changed after it was sealed")


def build_nova_book(batch, home: str, away: str, version: str) -> dict:
    """NOVA probabilities from simulation arrays and NOTHING else (no market field exists in this function's inputs)."""
    hi, ai = batch.team_ids.index(home), batch.team_ids.index(away)
    fin = np.asarray(batch.team_stats["score"])
    reg = np.asarray(batch.team_stats["reg_score"])
    book: dict = {"SCHEMA": "SPORTS_NOVA_V28_NOVA_BOOK", "ENGINE_VERSION": version, "ENGINE_HASH": package_hash(), "GAME_ID": batch.game_id, "SIMS": int(batch.n_sims), "SEED": int(batch.seed),
                  "STATE_HASH": batch.state_hash, "CALIBRATION_STATUS": CALIBRATION_STATUS, "OVERTIME_SIMULATED": bool(batch.runtime.get("overtime")),
                  "OT_MARKET_MAPPING": "UNRESOLVED (no local Kalshi NFL rules text): both 'including OT' and 'regulation only' books are exported; the operator must pick per the live contract text"}
    for name, sc in (("INCLUDING_OT", fin), ("REGULATION_ONLY", reg)):
        h, a = sc[:, hi], sc[:, ai]
        tot = h + a
        book[name] = {"WINNER": {"HOME": float((h > a).mean()), "AWAY": float((a > h).mean()), "TIE": float((h == a).mean()), "HOME_TIE_SPLIT": float((h > a).mean() + 0.5 * (h == a).mean()),
                                 "AWAY_TIE_SPLIT": float((a > h).mean() + 0.5 * (h == a).mean())},
                      "TOTAL_MEAN": float(tot.mean()), "TOTAL_OVER": {str(x): float((tot > x).mean()) for x in TOTAL_LINES},
                      "TEAM_TOTAL_MEAN": {home: float(h.mean()), away: float(a.mean())},
                      "TEAM_TOTAL_OVER": {home: {str(x): float((h > x).mean()) for x in TEAM_LINES}, away: {str(x): float((a > x).mean()) for x in TEAM_LINES}}}
    assert_market_free(json.loads(json.dumps(book)))
    return book


def seal(book: dict) -> SealedBook:
    assert_market_free(json.loads(json.dumps(book)))
    return SealedBook(json.loads(json.dumps(book)), hashlib.sha256(_canon(book)).hexdigest())


def join_market(sealed: SealedBook, quotes: list[dict]) -> dict:
    """Compare a SEALED NOVA book with externally supplied quotes [{market: WINNER_HOME|TOTAL_OVER|TEAM_TOTAL_OVER, line?, team?, price: 0..1, basis: INCLUDING_OT|REGULATION_ONLY}].
    Read-only on the book; returns rows.  There is no path from `quotes` back into any simulation."""
    if not isinstance(sealed, SealedBook):
        raise EngineError("a SealedBook is required: seal the NOVA book before any market price is joined")
    sealed.verify()
    rows = []
    for q in quotes:
        p = float(q["price"])
        if not 0.0 <= p <= 1.0:
            raise EngineError("price must be a probability in [0,1]")
        b = sealed.book[q.get("basis", "INCLUDING_OT")]
        if q["market"] == "WINNER_HOME":
            nova = b["WINNER"]["HOME_TIE_SPLIT"]
        elif q["market"] == "TOTAL_OVER":
            nova = b["TOTAL_OVER"][str(float(q["line"]))]
        elif q["market"] == "TEAM_TOTAL_OVER":
            nova = b["TEAM_TOTAL_OVER"][q["team"]][str(float(q["line"]))]
        else:
            raise EngineError(f"unsupported market {q['market']!r}")
        rows.append({**q, "nova_probability_raw": nova, "edge_raw": nova - p, "book_sha256": sealed.sha256, "STATUS": CALIBRATION_STATUS, "EXECUTION_AUTHORIZED": False})
    return {"book_sha256": sealed.sha256, "rows": rows, "LIVE_CAPITAL_AUTHORIZED": False}


def calibration_rows(book: dict, outcome: dict) -> list[dict]:
    """Adapter for a FUTURE M2 fit: (market, line, p_raw, y).  outcome = {'home_score':..,'away_score':..,'reg_home':..,'reg_away':..}.  No calibration is fit here."""
    home, away = list(book["INCLUDING_OT"]["TEAM_TOTAL_MEAN"])
    rows = []
    for basis, hs, as_ in (("INCLUDING_OT", outcome["home_score"], outcome["away_score"]), ("REGULATION_ONLY", outcome["reg_home"], outcome["reg_away"])):
        b = book[basis]
        rows.append({"basis": basis, "market": "WINNER_HOME", "line": None, "p_raw": b["WINNER"]["HOME_TIE_SPLIT"], "y": 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0})
        rows += [{"basis": basis, "market": "TOTAL_OVER", "line": float(x), "p_raw": p, "y": float(hs + as_ > float(x))} for x, p in b["TOTAL_OVER"].items()]
        rows += [{"basis": basis, "market": "TEAM_TOTAL_OVER", "team": t, "line": float(x), "p_raw": p, "y": float((hs if t == home else as_) > float(x))}
                 for t in (home, away) for x, p in b["TEAM_TOTAL_OVER"][t].items()]
    return rows


def run(game_id: str, sims: int = 5000, variant: int | None = None, panel: str = "frozen", seed: int | None = None) -> dict:
    champ = champion_variant()
    v = variant if variant is not None else champ
    if v is None:
        raise EngineError("no V28 GAME_SIM_CHAMPION is recorded; pass --variant K explicitly to run a CANDIDATE (output is labelled)")
    version = cfg.VERSIONS[v]
    ref = None
    if cfg.VARIANTS[v][1]:
        ref_path = ROOT / "data/sports_nova_v3/V28_GAME_STATE_SCORING/V28_2/V28_YARD_REFERENCE.json"
        ref = np.asarray(json.loads(ref_path.read_text())["quantiles"], dtype=float)
    sched = core.load_schedule()
    hit = sched[sched.game_id == game_id]
    if hit.empty:
        _, wk, away, home = game_id.split("_")[0], int(game_id.split("_")[1]), game_id.split("_")[2], game_id.split("_")[3]
        spec = core.manual_spec(sched, panel, int(game_id.split("_")[0]), wk, away, home, "2026-01-01 13:00")
    else:
        spec = core.spec_from_row(next(hit.itertuples()))
    prepared = core.prepare(spec, panel)
    core.assert_market_free(prepared.state.model_dump(mode="json"))
    sd = core.default_seed(game_id) if seed is None else int(seed)
    t0 = time.perf_counter()
    batch = s28.simulate_scoring(prepared.state, int(sims), sd, version, ref_quantiles=ref)
    sealed = seal(build_nova_book(batch, spec.home_team, spec.away_team, version))           # sealed BEFORE any market data can exist in this process
    b = sealed.book["INCLUDING_OT"]
    return {"engine_version": version, "engine_hash": package_hash(), "engine_role": "GAME_SIM_CHAMPION" if v == champ else "CANDIDATE_NOT_CHAMPION",
            "game": f"{spec.away_team} @ {spec.home_team} ({game_id})", "sims": int(sims), "seed": sd,
            "NOVA_home_win": b["WINNER"]["HOME"], "NOVA_away_win": b["WINNER"]["AWAY"], "NOVA_tie": b["WINNER"]["TIE"], "NOVA_home_win_tie_split": b["WINNER"]["HOME_TIE_SPLIT"],
            "NOVA_total_mean": b["TOTAL_MEAN"], "NOVA_team_totals": b["TEAM_TOTAL_MEAN"], "NOVA_total_mean_regulation_only": sealed.book["REGULATION_ONLY"]["TOTAL_MEAN"],
            "overtime_simulated": sealed.book["OVERTIME_SIMULATED"], "calibration_status": CALIBRATION_STATUS,
            "market_status": "NOT_JOINED: no market price was read; NOVA book sealed before any market join", "book_sha256": sealed.sha256,
            "state_layers": "terminal V23-level PregameState; V25 repair + V26 completion NOT applied in this path", "runtime_seconds": round(time.perf_counter() - t0, 2),
            "LIVE_CAPITAL_AUTHORIZED": False}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sports_nova.py --engine v28")
    ap.add_argument("--engine", required=True, choices=["v28"])
    ap.add_argument("--game", required=True)
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--variant", type=int, default=None)
    ap.add_argument("--panel", default="frozen", choices=["frozen", "live"])
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        out = run(a.game, a.sims, a.variant, a.panel, a.seed)
    except (EngineError, core.TerminalError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
