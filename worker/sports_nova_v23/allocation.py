from __future__ import annotations
from worker.sports_nova_v3.allocation import AllocationResult, _shares
from worker.sports_nova_v3.distributions import sample_dirichlet_multinomial

# SN3_V23_RESIDUAL_ALLOCATION_FIX: worker/sports_nova_v3/allocation.py's
# allocate_opportunities appends the explicit residual share as an EXTRA
# bucket in the same dirichlet-multinomial draw, then discards whatever
# count lands in that bucket (`result.untargeted` / `result.residual_carries`)
# instead of ever crediting it to a real player. See
# worker/sports_nova_v23/__init__.py for the evidence and ablation result.
#
# Fix: drop the residual bucket from the probability vector entirely and let
# sample_dirichlet_multinomial's existing renormalization (shares / shares.sum())
# spread the full opportunity count across only the named active players,
# proportional to their existing shares. This is the SAME draw shape as
# before minus one bucket -- no extra rng call, same RNG contract
# (one rng.dirichlet + one rng.multinomial per allocation call, exactly as
# worker/sports_nova_v3/allocation.py already does).
#
# `untargeted`/`residual_carries` are always 0 here except in the degenerate
# case where a team has NO listed active target/carry share players at all --
# that fallback (all opportunities go unattributed) is unchanged from V3/V21
# and is a separate, rarer roster-data gap, not this defect.


def allocate_opportunities(state, team_id: str, pass_attempts: int, rush_attempts: int,
                           rng, *, concentration: float = 80.0) -> AllocationResult:
    target_ids, target_shares, _target_residual = _shares(state, team_id, "target")
    carry_ids, carry_shares, _carry_residual = _shares(state, team_id, "carry")
    if target_shares.size:
        target_counts = sample_dirichlet_multinomial(pass_attempts, target_shares, concentration, rng)
        target_map = {pid: int(target_counts[i]) for i, pid in enumerate(target_ids)}
        untargeted = 0
    else:
        target_map, untargeted = {}, int(pass_attempts)
    if carry_shares.size:
        carry_counts = sample_dirichlet_multinomial(rush_attempts, carry_shares, concentration, rng)
        carry_map = {pid: int(carry_counts[i]) for i, pid in enumerate(carry_ids)}
        residual_carries = 0
    else:
        carry_map, residual_carries = {}, int(rush_attempts)
    return AllocationResult(tuple(sorted(set(target_ids) | set(carry_ids))), target_map, carry_map,
                            untargeted, residual_carries)
