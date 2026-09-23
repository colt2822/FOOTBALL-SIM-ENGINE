"""SPORTS-NOVA terminal: headless pipeline (M1 -> M2 -> M3 -> M4 -> artifact).

Nothing here is football modelling.  M1 is the already-shipped V23 simulator,
invoked through the same pregame-state builder the production slate script uses
(scripts/sports_nova_m1_v23_upcoming_slate.py).  Module order is one-way:

    football inputs -> M1 (simulate) -> M2 (NOVA fair book)
                                              |
    manual external book -> M3 (normalise) -> M4 (compare)

`run_simulation()` has no book parameter; an external book can only be attached
to a finished run via `attach_book()`, so M1 cannot see it by construction.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.sports_nova_v3_player_joint_walkforward_v1 as wf  # state builder (make_state / game_parts)
from worker.sports_nova_modular.comparator import compare_nova_to_market
from worker.sports_nova_modular.contracts import M1OutputSchema, M2OutputSchema, M3OutputSchema, M4OutputSchema
from worker.sports_nova_modular.firewall import assert_market_free
from worker.sports_nova_modular.m1_game_simulator import build_m1_output
from worker.sports_nova_modular.market_book import normalize_market_observation
from worker.sports_nova_modular.nova_book import build_nova_book
from worker.sports_nova_v3.schemas import Evidence
from worker.sports_nova_v23.simulator import MODEL_VERSION, set_identity_resolutions, simulate_game

DATA = ROOT / "data" / "sports_nova_v3"
LIVE = DATA / "validation_inputs_live"
SCHEDULE_CSV = LIVE / "schedule_snapshot_2026.csv"
RECEIPT_JSON = LIVE / "SPORTS_NOVA_V18_CAUSAL_PANEL_REFRESH_RECEIPT.json"
INJURIES_PARQUET = LIVE / "injuries_2026.parquet"
INJURIES_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2026.parquet"
RUNS_DIR = ROOT / "sports_nova_runs"
ET = ZoneInfo("America/New_York")

PANELS = {
    "frozen": {"path": DATA / "validation_inputs" / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet",
               "label": "FROZEN (through 2025 W22; the panel the shipped V23 Week-2 slate run used)"},
    "live": {"path": LIVE / "NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet",
             "label": "LIVE (frozen + 2026 completed weeks; validation_inputs_live)"},
}
NOVA_BOOK_VERSION = "SPORTS_NOVA_TERMINAL_M2_MEDIAN_FAIR_BOOK_1"
SIMS_PER_SEC_ESTIMATE = 44.0  # measured single-threaded, see memory project_sports_nova_m1_saturday_ship

# Schedule columns the terminal is allowed to load.  The CSV also carries moneyline/spread/total/odds and final
# scores; they are excluded at read time so they can never reach M1 (or even sit in a DataFrame).
SCHEDULE_COLS = ["game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "home_team",
                 "location", "roof", "stadium", "away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name"]

STAT_NAMES = ("pass_attempts", "pass_yards", "pass_tds", "rush_attempts", "rush_yards", "rush_tds",
              "targets", "receptions", "receiving_yards", "receiving_tds", "scored_tds")


class TerminalError(Exception):
    """Operator-facing failure (bad input, missing data). Never swallowed into a fake value."""


# --------------------------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def default_seed(game_id: str) -> int:
    """Same policy as the production slate script, so terminal runs are comparable to shipped slate runs."""
    return int(hashlib.sha256((game_id + "_UPCOMING_SLATE").encode()).hexdigest()[:8], 16)


_MODEL_HASH_CACHE: dict = {}


def model_hash() -> tuple[str, dict]:
    """sha256 over every .py file in the V23 import closure (v3, v19, v21, v23)."""
    if not _MODEL_HASH_CACHE:
        files = {}
        for pkg in ("sports_nova_v3", "sports_nova_v19", "sports_nova_v21", "sports_nova_v23"):
            for p in sorted((ROOT / "worker" / pkg).glob("*.py")):
                files[str(p.relative_to(ROOT)).replace("\\", "/")] = sha256_file(p)
        _MODEL_HASH_CACHE["hash"] = _sha_obj(files)
        _MODEL_HASH_CACHE["v23_files"] = {k: v for k, v in files.items() if "/sports_nova_v23/" in k}
    return _MODEL_HASH_CACHE["hash"], _MODEL_HASH_CACHE["v23_files"]


def dist(arr) -> dict:
    a = np.asarray(arr, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "P10": float(np.percentile(a, 10)),
            "P25": float(np.percentile(a, 25)), "P75": float(np.percentile(a, 75)),
            "P90": float(np.percentile(a, 90))}


# --------------------------------------------------------------------------------------------- game spec
@dataclass(frozen=True)
class GameSpec:
    game_id: str
    season: int
    week: int
    away_team: str
    home_team: str
    kickoff_utc: datetime
    source: str  # "SCHEDULE" | "MANUAL"
    venue: str | None = None
    away_qb_id: str | None = None
    home_qb_id: str | None = None
    away_qb_name: str | None = None
    home_qb_name: str | None = None

    @property
    def label(self) -> str:
        return f"{self.away_team} @ {self.home_team}"

    def kickoff_et(self) -> str:
        return self.kickoff_utc.astimezone(ET).strftime("%a %Y-%m-%d %H:%M ET")


def load_schedule() -> pd.DataFrame:
    if not SCHEDULE_CSV.is_file():
        raise TerminalError(f"schedule snapshot missing: {SCHEDULE_CSV}")
    df = pd.read_csv(SCHEDULE_CSV, usecols=SCHEDULE_COLS)
    banned = ("odds", "spread", "moneyline", "total", "line", "result", "score")
    assert not [c for c in df.columns if any(b in c.lower() for b in banned)], "market/result column loaded"
    df = df[df.season == 2026].copy()

    def ko(row):
        if pd.isna(row.gameday) or pd.isna(row.gametime):
            return pd.NaT
        return datetime.strptime(f"{row.gameday} {row.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET).astimezone(timezone.utc)

    df["kickoff_utc"] = df.apply(ko, axis=1)
    return df


def _nn(v):
    return None if pd.isna(v) else str(v)


def spec_from_row(row) -> GameSpec:
    return GameSpec(game_id=row.game_id, season=int(row.season), week=int(row.week), away_team=row.away_team,
                    home_team=row.home_team, kickoff_utc=row.kickoff_utc.to_pydatetime(), source="SCHEDULE",
                    venue=_nn(row.stadium), away_qb_id=_nn(row.away_qb_id), home_qb_id=_nn(row.home_qb_id),
                    away_qb_name=_nn(row.away_qb_name), home_qb_name=_nn(row.home_qb_name))


def upcoming_slate(sched: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Lowest week that still has a game kicking off in the future; games already kicked off are dropped."""
    now = now or datetime.now(timezone.utc)
    future = sched[sched.kickoff_utc.notna() & (sched.kickoff_utc > now)]
    if future.empty:
        return future
    return future[future.week == future.week.min()].sort_values("kickoff_utc")


_PANEL_CACHE: dict = {}


def load_panel(key: str) -> dict:
    if key not in PANELS:
        raise TerminalError(f"unknown panel {key!r}")
    if key not in _PANEL_CACHE:
        path = PANELS[key]["path"]
        if not path.is_file():
            raise TerminalError(f"football panel missing: {path}")
        df = pd.read_parquet(path)
        df["_key"] = df.SEASON * 100 + df.WEEK
        latest = df[df.SEASON == df.SEASON.max()]
        last = df.drop_duplicates("PLAYER_ID", keep="last")
        _PANEL_CACHE[key] = {
            "df": df, "path": path, "sha256": sha256_file(path),
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            "max_season": int(df.SEASON.max()), "max_week_in_max_season": int(latest.WEEK.max()),
            "teams": set(map(str, latest.TEAM.dropna().unique())),
            "names": dict(zip(last.PLAYER_ID.astype(str), last.PLAYER_NAME)),
        }
    return _PANEL_CACHE[key]


def manual_spec(sched: pd.DataFrame, panel_key: str, season: int, week: int, away: str, home: str,
                kickoff_local_et: str) -> GameSpec:
    """Manual game entry.  If the matchup is on the schedule snapshot, the scheduled record wins."""
    away, home = away.strip().upper(), home.strip().upper()
    teams = load_panel(panel_key)["teams"]
    for t in (away, home):
        if t not in teams:
            raise TerminalError(f"unknown team code {t!r}; known: {', '.join(sorted(teams))}")
    if away == home:
        raise TerminalError("away and home team must differ")
    if not 1 <= week <= 22:
        raise TerminalError("week must be 1-22")
    gid = f"{season}_{week:02d}_{away}_{home}"
    hit = sched[sched.game_id == gid]
    if not hit.empty and pd.notna(hit.iloc[0].kickoff_utc):
        return spec_from_row(next(hit.itertuples()))
    try:
        ko = datetime.strptime(kickoff_local_et.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=ET).astimezone(timezone.utc)
    except ValueError:
        raise TerminalError("kickoff must look like 2026-09-20 13:00 (Eastern time)")
    return GameSpec(game_id=gid, season=season, week=week, away_team=away, home_team=home, kickoff_utc=ko, source="MANUAL")


# --------------------------------------------------------------------------------------------- football inputs
@dataclass
class Overrides:
    """Operator football-data edits.  Recorded verbatim in the artifact."""
    out_players: dict = field(default_factory=dict)   # player_id -> label
    qb_override: dict = field(default_factory=dict)   # team -> player_id
    notes: str = ""

    def as_dict(self) -> dict:
        return {"out_players": dict(self.out_players), "qb_override": dict(self.qb_override), "notes": self.notes}


@dataclass
class Prepared:
    spec: GameSpec
    panel_key: str
    state: object
    football_data: dict
    warnings: list
    overrides: Overrides
    roster: list  # [{player_id,name,team,position,availability}]


def load_out_ids(week: int, season: int, path: Path | None = None) -> tuple[set, dict, list]:
    path = path or INJURIES_PARQUET
    warnings: list = []
    if season != 2026:
        return set(), {}, [f"injury report covers season 2026 only; season {season} game has no injury input"]
    if not path.is_file():
        return set(), {}, ["injury report file missing: no players flagged OUT from data (injury input = NO_DATA)"]
    try:
        df = pd.read_parquet(path)
        wk = df[(df.week == week) & (df.report_status == "Out")]
    except Exception as exc:  # malformed / schema drift
        return set(), {}, [f"injury report unreadable ({exc.__class__.__name__}: {exc}); injury input = NO_DATA"]
    fetched = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    meta = {"raw_sha256": sha256_file(path), "fetched_at": fetched}
    return set(wk.gsis_id.astype(str)), meta, warnings


def refresh_injury_report() -> dict:
    """Download the current nflverse injury report; validate before replacing (old copy kept as a backup)."""
    tmp = INJURIES_PARQUET.with_suffix(".parquet.download")
    try:
        req = urllib.request.Request(INJURIES_URL, headers={"User-Agent": "sports-nova-terminal"})
        with urllib.request.urlopen(req, timeout=30) as resp, tmp.open("wb") as fh:
            fh.write(resp.read())
        df = pd.read_parquet(tmp)
        missing = {"week", "report_status", "gsis_id"} - set(df.columns)
        if missing or df.empty:
            raise TerminalError(f"downloaded injury file failed validation (missing {sorted(missing)}, rows={len(df)})")
        if INJURIES_PARQUET.is_file():
            backup = INJURIES_PARQUET.with_name(f"injuries_2026.parquet.bak_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
            os.replace(INJURIES_PARQUET, backup)
        os.replace(tmp, INJURIES_PARQUET)
        return {"rows": int(len(df)), "max_week": int(df.week.max()), "sha256": sha256_file(INJURIES_PARQUET)}
    except TerminalError:
        raise
    except Exception as exc:
        raise TerminalError(f"injury refresh failed ({exc.__class__.__name__}: {exc}); existing file left in place")
    finally:
        if tmp.exists():
            tmp.unlink()


def _evidence(source_id: str, raw_sha: str, when: datetime) -> Evidence:
    return Evidence(source_id=source_id, raw_sha256=raw_sha, event_end=when, available_at=when, retrieved_at=when,
                    availability_basis="LIVE_RECEIPT")


def prepare(spec: GameSpec, panel_key: str = "frozen", overrides: Overrides | None = None) -> Prepared:
    """Build the pregame state from football data only and install QB identity.  Raises TerminalError if blocked."""
    overrides = overrides or Overrides()
    warnings: list = []
    panel = load_panel(panel_key)
    df = panel["df"]
    key = spec.season * 100 + spec.week
    prior = df[(df._key < key) & (df.SEASON >= max(1999, spec.season - 5))]
    if spec.kickoff_utc <= datetime.now(timezone.utc):
        warnings.append("kickoff time has already passed: this is a retrospective run, not a pregame one")
    if panel["max_season"] == spec.season and panel["max_week_in_max_season"] < spec.week - 1:
        warnings.append(f"football panel is {spec.week - 1 - panel['max_week_in_max_season']} week(s) behind this game "
                        f"(panel ends {panel['max_season']} W{panel['max_week_in_max_season']})")
    elif panel["max_season"] < spec.season:
        warnings.append(f"football panel ends {panel['max_season']} W{panel['max_week_in_max_season']}: "
                        f"no {spec.season} games are in the prior data")

    wf.PLAYER_SHA = panel["sha256"]  # evidence hash inside make_state must name the panel actually used
    try:
        state = wf.make_state(spec.game_id, prior, spec.kickoff_utc, None)
    except Exception as exc:
        raise TerminalError(f"pregame state could not be built ({exc.__class__.__name__}: {exc})")
    for team in (spec.home_team, spec.away_team):
        n = sum(1 for p in state.players if p.team_id == team)
        if n == 0:
            raise TerminalError(f"no prior player data for {team}: cannot simulate (BLOCKED, nothing invented)")
        if n < 8:
            warnings.append(f"only {n} players with history for {team}: partial player outputs")

    # injuries from the nflverse report ('Out' only; Questionable/Doubtful stay UNKNOWN, same as the slate script)
    out_ids, meta, w = load_out_ids(spec.week, spec.season)
    warnings += w
    now = datetime.now(timezone.utc)
    flips: dict = {}
    if out_ids and meta:
        ev = _evidence("NFLVERSE_INJURY_REPORT_2026", meta["raw_sha256"], meta["fetched_at"])
        for pid in sorted(p.player_id for p in state.players if p.player_id in out_ids):
            flips[pid] = ev
    # manual OUT edits
    manual_ids = [pid for pid in overrides.out_players if pid in {p.player_id for p in state.players}]
    for pid in overrides.out_players:
        if pid not in manual_ids:
            warnings.append(f"manual OUT player {pid} is not on this game's roster: ignored")
    if manual_ids:
        ev_m = _evidence("OPERATOR_MANUAL_ENTRY", _sha_obj(overrides.as_dict()), now)
        for pid in manual_ids:
            flips[pid] = ev_m
    if flips:
        state = state.model_copy(update={"players": tuple(
            p.model_copy(update={"availability": "OUT", "availability_evidence": flips[p.player_id]})
            if p.player_id in flips else p for p in state.players)})

    # QB identity: schedule designation (several days out) unless the operator overrides
    resolutions: dict = {}
    receipt_ts = None
    if RECEIPT_JSON.is_file():
        try:
            receipt_ts = json.loads(RECEIPT_JSON.read_text())["SOURCES"]["schedule"]["fetched_at_utc"]
        except Exception:
            receipt_ts = None
    qb_source: dict = {}
    roster_ids = {p.player_id for p in state.players}
    for team, sched_qb in ((spec.away_team, spec.away_qb_id), (spec.home_team, spec.home_qb_id)):
        qb, src, ts, conf = None, None, None, None
        if team in overrides.qb_override:
            cand = overrides.qb_override[team]
            if cand in roster_ids:
                qb, src, ts, conf = cand, "operator_manual_entry", now.isoformat(), "HIGH"
            else:
                warnings.append(f"QB override {cand} for {team} is not on this game's roster: ignored")
        if qb is None and sched_qb and spec.source == "SCHEDULE":
            if receipt_ts:
                qb, src, ts, conf = sched_qb, "nflverse_schedule_games_csv_scheduled_starter", receipt_ts, "HIGH"
            else:
                warnings.append("schedule QB designation has no fetch timestamp: not installed")
        if qb is None:
            qb_source[team] = "NONE (engine's own recency-weighted QB share model)"
            continue
        resolutions[(spec.game_id, team)] = {
            "GAME_ID": spec.game_id, "TEAM": team, "QB_ID": qb, "STATUS": "CONFIRMED", "CONFIDENCE": conf,
            "SOURCE": src, "SOURCE_TIMESTAMP": ts, "KICKOFF_TIME": spec.kickoff_utc.isoformat()}
        qb_source[team] = f"{qb} via {src} @ {ts}"
    try:
        set_identity_resolutions(resolutions)
    except ValueError as exc:
        warnings.append(f"QB identity not installed ({exc}); engine falls back to its own QB share model")
        set_identity_resolutions({})
        qb_source = {t: "NONE (identity rejected)" for t in (spec.away_team, spec.home_team)}

    names = panel["names"]
    roster = [{"player_id": p.player_id, "name": names.get(p.player_id), "team": p.team_id, "position": p.position,
               "availability": p.availability} for p in state.players]
    football_data = {
        "panel": {"key": panel_key, "label": PANELS[panel_key]["label"], "path": str(panel["path"]),
                  "sha256": panel["sha256"], "file_mtime_utc": panel["mtime_utc"],
                  "max_season": panel["max_season"], "max_week_in_max_season": panel["max_week_in_max_season"]},
        "prior_window": f"seasons>={max(1999, spec.season - 5)} with SEASON*100+WEEK < {key}",
        "schedule_snapshot": {"path": str(SCHEDULE_CSV), "sha256": sha256_file(SCHEDULE_CSV) if SCHEDULE_CSV.is_file() else None,
                              "fetched_at_utc": receipt_ts, "market_columns_loaded": False},
        "injury_report": {"status": "PRESENT" if meta else "NO_DATA",
                          "fetched_at_utc": meta["fetched_at"].isoformat() if meta else None,
                          "sha256": meta.get("raw_sha256") if meta else None,
                          "rule": "report_status=='Out' -> OUT; Questionable/Doubtful stay UNKNOWN"},
        "players_out_from_report": sorted(pid for pid in flips if pid not in manual_ids),
        "players_out_manual": sorted(manual_ids),
        "qb_identity": qb_source,
        "weather": "NOT MODELLED (V23 has no weather input; nothing entered or invented)",
        "state_as_of": state.as_of.isoformat(),
        "n_players_in_state": len(state.players),
    }
    return Prepared(spec, panel_key, state, football_data, warnings, overrides, roster)


# --------------------------------------------------------------------------------------------- M1
def run_simulation(prepared: Prepared, n_sims: int, seed: int):
    """M1.  Inputs: football state, N, seed.  There is deliberately no book/market parameter."""
    if type(n_sims) is not int or n_sims <= 0:
        raise TerminalError("simulation count must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise TerminalError("seed must be a non-negative integer")
    assert_market_free(prepared.state.model_dump(mode="json"))  # fail-closed contamination gate
    return simulate_game(prepared.state, n_sims, seed, MODEL_VERSION)


def summarize(prepared: Prepared, batch) -> dict:
    home, away = prepared.spec.home_team, prepared.spec.away_team
    hi, ai = batch.team_ids.index(home), batch.team_ids.index(away)
    hs, as_ = batch.team_stats["score"][:, hi], batch.team_stats["score"][:, ai]
    team = {}
    for side, tid, j in (("away", away, ai), ("home", home, hi)):
        pa, ra = batch.team_stats["pass_attempts"][:, j], batch.team_stats["rush_attempts"][:, j]
        py, ry = batch.team_stats["pass_yards"][:, j], batch.team_stats["rush_yards"][:, j]
        team[side] = {"team": tid, "pass_attempts": dist(pa), "rush_attempts": dist(ra), "plays_pass_plus_rush": dist(pa + ra),
                      "pass_yards": dist(py), "rush_yards": dist(ry), "total_yards": dist(py + ry)}
    names = load_panel(prepared.panel_key)["names"]
    pos = {r["player_id"]: r["position"] for r in prepared.roster}
    players = []
    for j, pid in enumerate(batch.player_ids):
        stats = {s: batch.player_stats[s][:, j] for s in STAT_NAMES}
        if not any(v.any() for v in stats.values()):
            continue
        players.append({"player_id": pid, "name": names.get(pid), "team": batch.player_team[pid], "position": pos.get(pid),
                        "td_probability": float(np.mean(stats["scored_tds"] > 0)),
                        "stats": {s: dist(v) for s, v in stats.items()}})
    n = batch.n_sims
    return {
        "teams": {"away": away, "home": home},
        "n_sims": n,
        "win_probability": {"AWAY": float(np.mean(batch.winner == "AWAY")), "HOME": float(np.mean(batch.winner == "HOME")),
                            "TIE": float(np.mean(batch.winner == "TIE"))},
        "score": {"away": dist(as_), "home": dist(hs)},
        "margin_home_minus_away": dist(hs - as_),
        "total": dist(hs + as_),
        "team": team,
        "players": players,
        "unavailable_outputs": ["team touchdowns", "team turnovers", "QB completions", "QB interceptions",
                                "RB/WR/TE fumbles (engine does not emit these)"],
    }


# --------------------------------------------------------------------------------------------- odds math
def parse_american(text) -> float | None:
    if text is None or str(text).strip() == "":
        return None
    s = str(text).strip().lower().replace(",", "")
    if s in ("even", "ev", "pk"):
        return 100.0
    try:
        v = float(s)
    except ValueError:
        raise TerminalError(f"{text!r} is not an American price (examples: -110, +150)")
    if not math.isfinite(v) or abs(v) < 100:
        raise TerminalError(f"{text!r} is not a valid American price (must be <= -100 or >= +100)")
    return v


def parse_line(text) -> float | None:
    if text is None or str(text).strip() == "":
        return None
    try:
        v = float(str(text).strip().replace("+", ""))
    except ValueError:
        raise TerminalError(f"{text!r} is not a number")
    if not math.isfinite(v):
        raise TerminalError(f"{text!r} is not finite")
    return v


def implied_prob(american: float) -> float:
    return 100.0 / (american + 100.0) if american > 0 else -american / (-american + 100.0)


def prob_to_american(p: float) -> float | None:
    if not 0.0 < p < 1.0:
        return None
    return -100.0 * p / (1.0 - p) if p >= 0.5 else 100.0 * (1.0 - p) / p


def devig(pa: float, pb: float) -> tuple[float, float, float]:
    total = pa + pb
    return pa / total, pb / total, total - 1.0


@dataclass
class ManualBook:
    """External book as typed by the operator.  Blank fields stay None."""
    book_name: str = "Manual"
    home_ml: float | None = None
    away_ml: float | None = None
    spread_team: str | None = None      # team code
    spread_line: float | None = None    # line for spread_team (e.g. -3.5)
    spread_price: float | None = None
    spread_other_price: float | None = None  # optional: opposite side's price at the mirrored line
    total_line: float | None = None
    over_price: float | None = None
    under_price: float | None = None
    timestamp: datetime | None = None

    def is_empty(self) -> bool:
        return all(getattr(self, f) is None for f in ("home_ml", "away_ml", "spread_line", "spread_price", "total_line",
                                                      "over_price", "under_price"))

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "timestamp"}
        d["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return d


# --------------------------------------------------------------------------------------------- M2 / M3 / M4
def _margins(batch, spec: GameSpec):
    hi, ai = batch.team_ids.index(spec.home_team), batch.team_ids.index(spec.away_team)
    hs, as_ = batch.team_stats["score"][:, hi], batch.team_stats["score"][:, ai]
    return (hs - as_).astype(float), (hs + as_).astype(float)


def _cover(margin: np.ndarray, side: str, line: float) -> dict:
    adj = margin if side == "HOME" else -margin
    win, push, lose = float(np.mean(adj + line > 0)), float(np.mean(adj + line == 0)), float(np.mean(adj + line < 0))
    return {"win": win, "push": push, "lose": lose, "win_ex_push": win / (win + lose) if win + lose > 0 else float("nan")}


def _over(total: np.ndarray, line: float) -> dict:
    over, push, under = float(np.mean(total > line)), float(np.mean(total == line)), float(np.mean(total < line))
    return {"over": over, "push": push, "under": under,
            "over_ex_push": over / (over + under) if over + under > 0 else float("nan")}


def build_m2(spec: GameSpec, batch, m1: M1OutputSchema, book: ManualBook | None = None) -> M2OutputSchema:
    """NOVA fair book from M1 only.  `book` is used purely as *query points* (which line to evaluate the
    already-finished simulated distribution at); it never re-enters M1."""
    margin, total = _margins(batch, spec)
    pa = float(np.mean(batch.winner == "AWAY"))
    ph = float(np.mean(batch.winner == "HOME"))
    pt = float(np.mean(batch.winner == "TIE"))
    two_way = ph + pa
    probs = {"NOVA_ML_AWAY_PROB": pa, "NOVA_ML_HOME_PROB": ph, "NOVA_TIE_PROB": pt}
    prices = {"HOME_ML": ph / two_way, "AWAY_ML": pa / two_way}
    lines = {"NOVA_FAIR_SPREAD_HOME": -float(np.median(margin)), "NOVA_FAIR_TOTAL": float(np.median(total)),
             "NOVA_MEAN_MARGIN_HOME": float(margin.mean()), "NOVA_MEAN_TOTAL": float(total.mean())}
    if book is not None and book.spread_line is not None and book.spread_team:
        side = "HOME" if book.spread_team == spec.home_team else "AWAY"
        c = _cover(margin, side, book.spread_line)
        probs.update({"SPREAD_COVER_RAW": c["win"], "SPREAD_PUSH": c["push"], "SPREAD_COVER_EX_PUSH": c["win_ex_push"]})
        prices["SPREAD_COVER"] = c["win_ex_push"]
    if book is not None and book.total_line is not None:
        o = _over(total, book.total_line)
        probs.update({"TOTAL_OVER_RAW": o["over"], "TOTAL_UNDER_RAW": o["under"], "TOTAL_PUSH": o["push"]})
        prices["TOTAL_OVER"] = o["over_ex_push"]
        prices["TOTAL_UNDER"] = 1.0 - o["over_ex_push"]
    probs.update({k: v for k, v in prices.items()})
    return build_nova_book(m1, nova_book_version=NOVA_BOOK_VERSION, fair_prices=prices, fair_lines=lines,
                           probabilities={k: float(v) for k, v in probs.items()})


def build_market_and_comparison(spec: GameSpec, nova: M2OutputSchema, book: ManualBook, batch) -> dict:
    """M3 (normalise the typed book) then M4 (mechanical NOVA-minus-market deltas).  Read-only w.r.t. M1/M2."""
    ts = book.timestamp or datetime.now(timezone.utc)
    warnings: list = []
    obs: list[M3OutputSchema] = []
    detail: dict = {}

    def vig_or_none(v):
        if v < 0:
            warnings.append("negative overround from entered prices (arbitrage-looking); vig not recorded")
            return None
        return v

    def add(market, price, line=None, vig=None):
        obs.append(normalize_market_observation(game_id=spec.game_id, venue=book.book_name, market=market, timestamp=ts,
                                                price=float(price), line=None if line is None else float(line), vig=vig))

    # moneyline
    if book.home_ml is not None or book.away_ml is not None:
        ph = implied_prob(book.home_ml) if book.home_ml is not None else None
        pa = implied_prob(book.away_ml) if book.away_ml is not None else None
        if ph is not None and pa is not None:
            dh, da, over = devig(ph, pa)
            detail["moneyline"] = {"home_raw": ph, "away_raw": pa, "home_devig": dh, "away_devig": da, "overround": over,
                                   "vig_adjusted": True}
            add("HOME_ML", dh, vig=vig_or_none(over))
            add("AWAY_ML", da, vig=vig_or_none(over))
        else:
            detail["moneyline"] = {"home_raw": ph, "away_raw": pa, "vig_adjusted": False,
                                   "note": "only one side entered: raw implied probability, still contains vig"}
            if ph is not None:
                add("HOME_ML", ph)
            if pa is not None:
                add("AWAY_ML", pa)
    # spread
    if book.spread_line is not None and book.spread_price is not None and book.spread_team:
        p = implied_prob(book.spread_price)
        d = {"team": book.spread_team, "line": book.spread_line, "price": book.spread_price, "raw": p}
        if book.spread_other_price is not None:
            q = implied_prob(book.spread_other_price)
            dp, _, over = devig(p, q)
            d.update({"other_price": book.spread_other_price, "other_raw": q, "devig": dp, "overround": over, "vig_adjusted": True})
            add("SPREAD_COVER", dp, line=book.spread_line, vig=vig_or_none(over))
        else:
            d.update({"vig_adjusted": False, "note": "opposite-side price not entered: raw implied probability, contains vig"})
            add("SPREAD_COVER", p, line=book.spread_line)
        detail["spread"] = d
    elif book.spread_line is not None or book.spread_price is not None:
        warnings.append("spread needs team + line + price to be compared; incomplete spread ignored")
    # total
    if book.total_line is not None and (book.over_price is not None or book.under_price is not None):
        po = implied_prob(book.over_price) if book.over_price is not None else None
        pu = implied_prob(book.under_price) if book.under_price is not None else None
        d = {"line": book.total_line, "over_raw": po, "under_raw": pu}
        if po is not None and pu is not None:
            do, du, over = devig(po, pu)
            d.update({"over_devig": do, "under_devig": du, "overround": over, "vig_adjusted": True})
            add("TOTAL_OVER", do, line=book.total_line, vig=vig_or_none(over))
            add("TOTAL_UNDER", du, line=book.total_line, vig=vig_or_none(over))
        else:
            d.update({"vig_adjusted": False, "note": "only one side entered: raw implied probability, contains vig"})
            if po is not None:
                add("TOTAL_OVER", po, line=book.total_line)
            if pu is not None:
                add("TOTAL_UNDER", pu, line=book.total_line)
        detail["total"] = d
    elif book.total_line is not None or book.over_price is not None or book.under_price is not None:
        warnings.append("total needs a line and at least one price to be compared; incomplete total ignored")

    m4: list[M4OutputSchema] = [compare_nova_to_market(nova, o) for o in obs]
    line_diffs = {}
    if book.spread_line is not None and book.spread_team:
        mkt_home = book.spread_line if book.spread_team == spec.home_team else -book.spread_line
        line_diffs["spread_home"] = {"nova_fair": nova.fair_lines["NOVA_FAIR_SPREAD_HOME"], "market": mkt_home,
                                     "difference": nova.fair_lines["NOVA_FAIR_SPREAD_HOME"] - mkt_home}
    if book.total_line is not None:
        line_diffs["total"] = {"nova_fair": nova.fair_lines["NOVA_FAIR_TOTAL"], "market": book.total_line,
                               "difference": nova.fair_lines["NOVA_FAIR_TOTAL"] - book.total_line}
    return {"m3": [o.model_dump(mode="json") for o in obs], "m4": [c.model_dump(mode="json") for c in m4],
            "market_detail": detail, "line_differences": line_diffs, "warnings": warnings}


def attach_book(result: "RunResult", book: ManualBook) -> dict:
    """Attach an external book to a finished run.  Touches nothing that M1 produced."""
    spec = result.prepared.spec
    if book.spread_team and book.spread_team not in (spec.home_team, spec.away_team):
        raise TerminalError(f"SPREAD_TEAM must be {spec.away_team} or {spec.home_team}")
    nova = build_m2(spec, result.batch, result.m1, book)
    comp = build_market_and_comparison(spec, nova, book, result.batch)
    entry = {"entered_at_utc": datetime.now(timezone.utc).isoformat(), "external_book_input": book.as_dict(),
             "M2_NOVA_BOOK_AT_ENTERED_LINES": nova.model_dump(mode="json"), **comp}
    result.record.setdefault("M3_EXTERNAL_BOOKS", []).append(entry)
    result.record["M3_EXTERNAL_BOOK"] = entry["external_book_input"]
    result.record["M4_COMPARISON"] = entry
    return entry


# --------------------------------------------------------------------------------------------- run + artifact
@dataclass
class RunResult:
    prepared: Prepared
    batch: object
    m1: M1OutputSchema
    m2: M2OutputSchema
    record: dict
    artifact_path: Path | None = None


def execute_run(spec: GameSpec, n_sims: int, seed: int | None = None, overrides: Overrides | None = None,
                panel_key: str = "frozen", book: ManualBook | None = None, save: bool = True,
                prepared: Prepared | None = None) -> RunResult:
    """Whole pipeline.  `book` is applied strictly AFTER the simulation."""
    prepared = prepared or prepare(spec, panel_key, overrides)
    seed_policy = "DEFAULT sha256(game_id+'_UPCOMING_SLATE')[:8]; per-sim RNG SeedSequence([seed, sim_id])"
    if seed is None:
        seed = default_seed(spec.game_id)
    else:
        seed_policy = "OPERATOR_OVERRIDE; per-sim RNG SeedSequence([seed, sim_id])"
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc)
    try:
        batch = run_simulation(prepared, n_sims, seed)
    except Exception as exc:
        rec = base_record(prepared, n_sims, seed, seed_policy, started, time.perf_counter() - t0)
        rec["STATUS"] = "FAILED"
        rec["ERROR"] = f"{exc.__class__.__name__}: {exc}"
        if save:
            save_artifact(rec)
        raise TerminalError(f"simulator failed: {rec['ERROR']} (failed-run artifact saved)") from exc
    sim_seconds = time.perf_counter() - t0
    m1 = build_m1_output(batch, input_cutoff_ts=prepared.state.as_of)
    m2 = build_m2(spec, batch, m1)
    rec = base_record(prepared, n_sims, seed, seed_policy, started, sim_seconds)
    rec.update({"STATUS": "OK", "M1_HASH": m1.hash, "M1_INPUT_HASH": batch.state_hash,
                "M1_SUMMARY": summarize(prepared, batch), "M1_SCHEMA_OUTPUT": m1.model_dump(mode="json"),
                "M2_NOVA_BOOK": m2.model_dump(mode="json"), "M3_EXTERNAL_BOOK": None, "M4_COMPARISON": None})
    if batch.status != "RESEARCH_GRADE_DEFAULT_PARAMETERS":
        rec["WARNINGS"].append(f"batch status: {batch.status}")
    rec["WARNINGS"].append("distribution params are RESEARCH_GRADE_DEFAULT_PARAMETERS (engine status, not tuned/certified)")
    result = RunResult(prepared, batch, m1, m2, rec)
    if book is not None and not book.is_empty():
        attach_book(result, book)
    if save:
        result.artifact_path = save_artifact(rec)
    return result


def base_record(prepared: Prepared, n: int, seed: int, seed_policy: str, started: datetime, seconds: float) -> dict:
    mh, v23_files = model_hash()
    spec = prepared.spec
    return {
        "SCHEMA": "SPORTS_NOVA_TERMINAL_RUN_V1",
        "RUN_TIMESTAMP_UTC": started.isoformat(),
        "STATUS": "PENDING",
        "GAME_INPUTS": {
            "game_id": spec.game_id, "game_source": spec.source, "season": spec.season, "week": spec.week,
            "away_team": spec.away_team, "home_team": spec.home_team, "kickoff_utc": spec.kickoff_utc.isoformat(),
            "kickoff_et": spec.kickoff_et(), "venue": spec.venue,
            "scheduled_qbs": {spec.away_team: spec.away_qb_name, spec.home_team: spec.home_qb_name},
            "operator_overrides": prepared.overrides.as_dict(),
            "operator_notes_NOT_FED_TO_M1": prepared.overrides.notes,
        },
        "FOOTBALL_DATA": prepared.football_data,
        "M1_RUN": {"simulation_version": MODEL_VERSION, "model_hash": mh, "engine_v23_source_sha256": v23_files,
                   "input_hash": None, "simulation_count": n, "random_seed": seed, "seed_policy": seed_policy,
                   "forbidden_m1_inputs_present": False, "runtime_seconds": round(seconds, 3)},
        "WARNINGS": list(prepared.warnings),
        "NOTE": "External book values live only under M3_*/M4_* and are never inputs to M1. Fair values are simulator "
                "outputs, not bets, locks, or guaranteed edges.",
    }


def save_artifact(rec: dict) -> Path:
    """sports_nova_runs/YYYYMMDD/GAMEID_TIMESTAMP.json; re-saving the same run overwrites the same file."""
    started = datetime.fromisoformat(rec["RUN_TIMESTAMP_UTC"])
    day = started.astimezone().strftime("%Y%m%d")
    out = RUNS_DIR / day / f"{rec['GAME_INPUTS']['game_id']}_{started:%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if rec.get("M1_RUN") and rec.get("M1_INPUT_HASH"):
        rec["M1_RUN"]["input_hash"] = rec["M1_INPUT_HASH"]
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, out)
    rec["ARTIFACT_PATH"] = str(out)
    return out


def list_artifacts(limit: int = 20) -> list[Path]:
    if not RUNS_DIR.is_dir():
        return []
    return sorted(RUNS_DIR.glob("*/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
