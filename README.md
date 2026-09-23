# Football Sim Engine (M1)

A public snapshot of the football-only M1 simulation engine from the SPORTS-NOVA research project. M1 turns a validated pregame football state into reproducible distributions for game scores, player statistics, and joint outcomes. The M1 boundary rejects market, betting, trading, and execution inputs.

## What is included

- The modular M1 adapter and its immutable output contract and input firewall.
- The legacy V1.x/V2 implementation history, frozen V21 simulator, and V23 challenger/terminal layers, plus the V19/V3 implementation layers they import.
- Preserved M1 challenger implementations from V24 through V28, with their focused tests.
- The complete V3 schema, simulation, state, allocation, scoring, calibration, and validation package.
- The legacy probability engine and its shared validation dependency.
- M1/V3/V19/V20/V21 analysis and replay scripts, later M1 research scripts, focused tests, requirements, and selected technical notes.

The source snapshot was captured on 2026-09-22. `SNAPSHOT_MANIFEST.sha256` records hashes for every included file except the manifest itself. This repository does not include local datasets or frozen validation artifacts, live captures, secrets, environment files, generated run outputs, or unrelated Node3 research. Some challenger and replay workflows expect hash-identified local data files; those inputs are intentionally excluded and must be supplied separately. The simulator accepts a validated pregame state as input; see [`docs/USAGE.md`](docs/USAGE.md).

## Quick start

Use Python 3.11 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
```

The Python package layout uses `worker.*` imports, so run commands from the repository root with the root on `PYTHONPATH`.

Minimal API shape:

```python
from datetime import datetime, timezone
from worker.sports_nova_v3.schemas import PregameState
from worker.sports_nova_modular import build_m1_output, simulate_m1

# Construct PregameState from validated, point-in-time football inputs.
state: PregameState = ...
batch = simulate_m1(state, n_sims=10_000, seed=42)
output = build_m1_output(batch, input_cutoff_ts=datetime.now(timezone.utc))
print(output.model_dump(mode="json"))
```

For production research, `input_cutoff_ts` must reflect the actual input cutoff; do not use the illustrative current-time expression above as historical evidence.

## Model status and scope

The snapshot's documented champion is V21. The modular-reset note records the V21 football-only health results and architecture checks. Those historical results describe the recorded cohort and do not certify current-slate inputs, player/joint calibration, profitability, or future performance. M1 is a simulator; it does not establish betting or trading value.

Keep future work simulator-only at the M1 boundary. Do not feed odds, prices, lines, market probabilities, P&L, strategy, or execution data into M1. Preserve unknown player or starter identity as unknown and fail closed when required truth is unavailable.

## Tests

Focused tests are under `worker/sports_nova_v3/tests/`, `worker/sports_nova_modular/tests/`, and `tests/test_sports_probability_engine.py`. Run them from the repository root after installing dependencies:

```powershell
python -m pytest worker/sports_nova_v3/tests worker/sports_nova_modular/tests tests/test_sports_probability_engine.py
```

## Contributions

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Keep changes scoped, reproducible, and backed by tests or evidence. Do not add private datasets, credentials, personal records, live account data, or generated artifacts without checking that they are safe and needed for public distribution.

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).
