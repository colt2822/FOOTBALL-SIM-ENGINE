from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .distributions import sample_dirichlet_multinomial
from .schemas import OpportunityShares, PregameState

@dataclass(frozen=True)
class AllocationResult:
    player_ids: tuple[str, ...]
    target_counts: dict[str, int]
    carry_counts: dict[str, int]
    untargeted: int
    residual_carries: int

def _shares(state: PregameState, team_id: str, kind: str) -> tuple[tuple[str, ...], np.ndarray, float]:
    team = state.home if state.home.team_id == team_id else state.away
    source: OpportunityShares = team.target_shares if kind == "target" else team.carry_shares
    active = {p.player_id for p in state.players if p.team_id == team_id and p.availability != "OUT"}
    ids, values = [], []
    for pid, share in zip(source.player_ids, source.shares):
        if pid in active and share > 0:
            ids.append(pid); values.append(float(share))
    residual = float(source.residual_share) + max(0., 1. - sum(values) - float(source.residual_share))
    return tuple(ids), np.asarray(values, dtype=float), residual

def allocate_opportunities(state: PregameState, team_id: str, pass_attempts: int,
                           rush_attempts: int, rng: np.random.Generator,
                           *, concentration: float = 80.0) -> AllocationResult:
    target_ids, target_shares, target_residual = _shares(state, team_id, "target")
    carry_ids, carry_shares, carry_residual = _shares(state, team_id, "carry")
    if target_shares.size:
        target_probs = np.r_[target_shares, target_residual]
        target_counts = sample_dirichlet_multinomial(pass_attempts, target_probs, concentration, rng)
        target_map = {pid: int(target_counts[i]) for i, pid in enumerate(target_ids)}
        untargeted = int(target_counts[-1])
    else:
        target_map, untargeted = {}, int(pass_attempts)
    if carry_shares.size:
        carry_probs = np.r_[carry_shares, carry_residual]
        carry_counts = sample_dirichlet_multinomial(rush_attempts, carry_probs, concentration, rng)
        carry_map = {pid: int(carry_counts[i]) for i, pid in enumerate(carry_ids)}
        residual_carries = int(carry_counts[-1])
    else:
        carry_map, residual_carries = {}, int(rush_attempts)
    return AllocationResult(tuple(sorted(set(target_ids) | set(carry_ids))), target_map, carry_map,
                            untargeted, residual_carries)
