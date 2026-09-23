"""POST_HOC helper for V28.3/.4: freeze the simulator's OWN block-yardage marginal (V27 pilot BASE blocks: no outcomes, no market) as a 2001-point quantile grid.

  python scripts/sports_nova_m1_v28_yard_reference.py TAG
Written only after the V28.1 measurement showed the TD-possession rate off history by more than the preregistered 0.010 (decision rule ALIGNMENT_VARIANT_.3).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.sports_nova_m1_v28_game_state_scoring as R  # noqa: E402
from worker.sports_nova_v28_game_state_scoring import config as cfg  # noqa: E402

FIELDS = ["sim", "block", "off", "sec_before", "pre_h", "pre_a", "plays", "pass_att", "designed", "scrambles", "targets", "rec", "pass_yds", "rush_yds"]


def main(tag: str) -> None:
    pre = R.load_freeze(tag)
    out = R.out_dir(tag) / "V28_YARD_REFERENCE.json"
    if out.exists():
        raise SystemExit("REFUSING: reference already frozen")
    ys, src = [], {}
    for g in pre["COHORT"]["game_ids"]:
        p = R.V27_ARR / f"{g}.npz"
        b = np.load(p)["base__blocks"]
        ys.append(b[:, 12] + b[:, 13])
        src[p.name] = R.sha256_file(p)
    y = np.concatenate(ys).astype(float)
    grid = np.linspace(0.0, 1.0, cfg.ALIGN_GRID)
    q = np.quantile(y, grid)
    rec = {"SCHEMA": "SPORTS_NOVA_V28_YARD_REFERENCE", "LABEL": "POST_HOC", "SOURCE": "V27 BASE blocks (sealed MULTI_TD ablation arrays), pass_yds+rush_yds per block", "N_BLOCKS": int(len(y)),
           "SOURCE_ARRAY_SHA256": src, "GRID": cfg.ALIGN_GRID, "quantiles": q.tolist(), "MEAN": float(y.mean()), "NOTE": "sim marginal only; no historical outcome and no market data used"}
    out.write_text(json.dumps(rec))
    print("reference written", out, "blocks", len(y), "mean", y.mean(), "sha", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "V28_1")
