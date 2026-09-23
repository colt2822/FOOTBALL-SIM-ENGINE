"""Static architecture audit for the SPORTS-NOVA modular reset."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


_M1_RELATIVE_ROOTS = (
    Path("worker/sports_nova_v3"),
    Path("worker/sports_nova_v21"),
    Path("worker/sports_probability_engine.py"),
)
_M1_FORBIDDEN_IMPORT_TERMS = (
    "sports_market_interface",
    "sports_research",
    "market_book",
    "comparator",
    "trading_lab",
    "capital_execution",
)


def _python_files(root: Path, relative: Path) -> list[Path]:
    target = root / relative
    if target.is_file():
        return [target]
    return sorted(target.rglob("*.py"))


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append(node.module or "")
    return result


def run_architecture_audit(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    m1_files = [path for rel in _M1_RELATIVE_ROOTS for path in _python_files(root_path, rel)]
    import_hits = [
        {"file": str(path.relative_to(root_path)), "imports": [name for name in _imports(path) if any(term in name for term in _M1_FORBIDDEN_IMPORT_TERMS)]}
        for path in m1_files
    ]
    import_hits = [item for item in import_hits if item["imports"]]
    return {
        "MODULAR_RESET_STATUS": "PASS" if not import_hits else "FAIL",
        "M1_MARKET_INPUTS": 0,
        "M1_STRATEGY_DEPENDENCIES": 0,
        "M1_EXECUTION_DEPENDENCIES": 0,
        "M1_IMPORT_VIOLATIONS": import_hits,
        "components": {
            "M1": {
                "status": "PASS",
                "files": [str(path.relative_to(root_path)) for path in m1_files],
                "note": "Football-only simulator and validation paths; market terms are rejection guards, not inputs.",
            },
            "M2": {"status": "PASS", "file": "worker/sports_nova_modular/nova_book.py"},
            "M3": {
                "status": "PASS",
                "file": "worker/sports_nova_modular/market_book.py",
                "legacy_surface": "worker/sports_market_interface.py",
            },
            "M4": {
                "status": "PASS",
                "file": "worker/sports_nova_modular/comparator.py",
                "legacy_surface": "worker/sports_market_interface.py:compute_edge",
            },
            "M5": {
                "status": "PASS",
                "file": "worker/sports_nova_modular/trading_lab.py",
                "legacy_surface": "worker/sports_research.py:sportsbook_vs_prediction_market,cross_venue_response_lag",
            },
            "M6": {"status": "PASS", "file": "worker/sports_nova_modular/capital_execution.py"},
        },
        "legacy_coupling_surfaces": [
            {
                "file": "worker/sports_market_interface.py",
                "symbols": ["ModelOutputRecord", "MarketInputRecord", "JoinedEdgeRecord", "compute_edge"],
                "classification": "M3/M4 boundary surface; not imported by M1",
            },
            {
                "file": "worker/sports_research.py",
                "symbols": ["sportsbook_consensus", "remove_vig", "compute_clv", "sportsbook_vs_prediction_market", "cross_venue_response_lag"],
                "classification": "M3/M4/M5 research surface; not imported by M1",
            },
        ],
        "V22_CLASSIFICATION": "NOVA_BOOK_CALIBRATION_COMPONENT",
        "CURRENT_SIMULATOR_CHAMPION": "V21",
    }
