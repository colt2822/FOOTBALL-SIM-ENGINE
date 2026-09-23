"""SPORTS_NOVA M1 Phase-5: per-TD-event ledger extraction and combo queries.

Additive, read-only post-processing of an already-instrumented simulation
path (scripts/sports_nova_m1_v4_raw_path_diagnostic_challenger.py's
`instrument(..., detail=True)` context, or any producer of the same
per-block shape: `block["allocation"]["realized_outcome"]` mapping
player_id -> {stat: value}, plus `block_index`/`offense_team`/
`seconds_remaining_before`). Does not touch any frozen engine module --
scoring itself is unchanged; this only extracts what already happened into a
queryable per-event row instead of the aggregate per-player counts the raw
ledger currently keeps.

A touchdown is any block-level realized_outcome entry with a nonzero
`rush_tds` or `receiving_tds` (mutually exclusive per current engines' `_inc`
call sites -- a player is credited one or the other, never both, in the same
block). `value` can exceed 1 if a player scores more than once in the same
block; each unit becomes its own row (TD combo probabilities are defined
per-scoring-event, not per-block).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping

SECONDS_PER_QUARTER = 900
GAME_SECONDS = 3600


def _quarter(seconds_remaining_before: int) -> int:
    elapsed = GAME_SECONDS - int(seconds_remaining_before)
    return max(1, min(4, elapsed // SECONDS_PER_QUARTER + 1))


def extract_td_events(game_id: str, sim_id: int, path_blocks: list[dict],
                       player_name: Mapping[str, str] | None = None) -> list[dict]:
    """One row per individual touchdown in a single simulated game path.

    `path_blocks` is `path["blocks"]` from an instrumented run (each block
    has `allocation.realized_outcome`, `offense_team`,
    `seconds_remaining_before`). `player_name` is an optional PLAYER_ID ->
    PLAYER_NAME lookup (e.g. from NFL_V3_PLAYER_GAME_CAUSAL_V1.parquet);
    scorer_name is omitted (not null-filled) when not provided.
    """
    player_name = player_name or {}
    rows: list[dict] = []
    for block in path_blocks:
        realized = block.get("allocation", {}).get("realized_outcome", {})
        team = block.get("offense_team")
        secs = block.get("seconds_remaining_before")
        for pid, stat_values in realized.items():
            for stat, td_type in (("rush_tds", "rushing"), ("receiving_tds", "receiving")):
                count = int(stat_values.get(stat, 0))
                for _ in range(count):
                    row = {
                        "game_id": game_id, "sim_id": int(sim_id), "team": team,
                        "scorer_id": pid, "TD_type": td_type,
                        "block_index": block.get("block_index"),
                        "seconds_remaining": secs,
                        "quarter": _quarter(secs) if secs is not None else None,
                    }
                    if pid in player_name:
                        row["scorer_name"] = player_name[pid]
                    rows.append(row)
    return rows


@dataclass(frozen=True)
class TDComboQuery:
    """Per-game-id TD combo probabilities estimated from a set of simulated
    paths' TD ledgers (as produced by `extract_td_events`, concatenated
    across sim_id for one game_id).
    """
    game_id: str
    n_sims: int
    scorers_by_sim: dict[int, set[str]]

    @classmethod
    def from_rows(cls, game_id: str, n_sims: int, rows: Iterable[dict]) -> "TDComboQuery":
        scorers_by_sim: dict[int, set[str]] = defaultdict(set)
        for r in rows:
            if r["game_id"] != game_id:
                continue
            scorers_by_sim[int(r["sim_id"])].add(r["scorer_id"])
        return cls(game_id=game_id, n_sims=n_sims, scorers_by_sim=dict(scorers_by_sim))

    def p_scores(self, player_id: str) -> float:
        return sum(1 for s in self.scorers_by_sim.values() if player_id in s) / self.n_sims

    def p_combo(self, player_ids: Iterable[str], mode: str = "all") -> float:
        """mode='all': every named player scores in the same sim (AND).
        mode='any': at least one of the named players scores (OR)."""
        ids = set(player_ids)
        if mode == "all":
            hits = sum(1 for s in self.scorers_by_sim.values() if ids <= s)
        elif mode == "any":
            hits = sum(1 for s in self.scorers_by_sim.values() if ids & s)
        else:
            raise ValueError("mode must be 'all' or 'any'")
        return hits / self.n_sims

    def scorer_correlation(self, player_a: str, player_b: str) -> float:
        """Pearson correlation of the two players' per-sim scored-TD indicators."""
        a = [1.0 if player_a in s else 0.0 for s in self.scorers_by_sim.values()]
        b = [1.0 if player_b in s else 0.0 for s in self.scorers_by_sim.values()]
        n = len(a)
        if n < 2:
            return float("nan")
        mean_a, mean_b = sum(a) / n, sum(b) / n
        cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / n
        var_a = sum((x - mean_a) ** 2 for x in a) / n
        var_b = sum((y - mean_b) ** 2 for y in b) / n
        if var_a <= 0 or var_b <= 0:
            return float("nan")
        return cov / (var_a * var_b) ** 0.5

    def top_combos(self, k: int = 2, min_count: int = 1) -> list[tuple[tuple[str, ...], float]]:
        """All observed size-k scorer combos in this game, by joint probability,
        restricted to combos that actually co-occurred >= min_count times."""
        counts: dict[tuple[str, ...], int] = defaultdict(int)
        for scorers in self.scorers_by_sim.values():
            if len(scorers) < k:
                continue
            for combo in combinations(sorted(scorers), k):
                counts[combo] += 1
        ranked = sorted(((c, n) for c, n in counts.items() if n >= min_count),
                        key=lambda kv: kv[1], reverse=True)
        return [(combo, n / self.n_sims) for combo, n in ranked]
