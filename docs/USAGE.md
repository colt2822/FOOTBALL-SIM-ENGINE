# Usage

## Input contract

M1 takes a `PregameState` from `worker.sports_nova_v3.schemas`. Build it from validated football-only inputs with a defensible as-of cutoff. The schema, player identity policy, and validation rules are in `worker/sports_nova_v3/schemas.py`, `pregame_state.py`, and `identity_gate.py`.

Do not use current roster information to reconstruct historical inputs. When player or starter identity is unresolved, retain `UNKNOWN` or fail closed according to the applicable schema and identity gate. The M1 firewall rejects market-derived fields.

## Run the simulator

```python
from datetime import datetime, timezone
from worker.sports_nova_v3.schemas import PregameState
from worker.sports_nova_modular import build_m1_output, simulate_m1

state: PregameState = ...  # validated football-only pregame state
batch = simulate_m1(state, n_sims=10_000, seed=42)
result = build_m1_output(batch, input_cutoff_ts=datetime.now(timezone.utc))

# Stable, immutable output with a canonical content hash
payload = result.model_dump(mode="json")
```

Set the output cutoff to the timestamp that actually governed the input snapshot. The wall-clock example is only for a live run; it is not suitable for historical replay or causal evaluation.

## Reproducibility

Use an explicit positive simulation count and a fixed seed. Preserve the simulator version, input state hash, cutoff timestamp, and output hash with downstream artifacts. Changing the input state, simulator version, seed, or simulation count creates a different run.

## Validation

The package contains unit and contract tests, causal validation utilities, and artifact-freeze helpers. These tools verify software behavior and artifact integrity; they do not by themselves establish model accuracy or economic value. Historical data required for empirical validation is intentionally not included in this public source backup.
