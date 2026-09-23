"""Layer 9: correlated game simulation.

Independence is not acceptable for same-game parlays, so the simulator is
hierarchical: one game-level environment draw feeds both teams, team volume
feeds player opportunity, and player efficiency is drawn through a Gaussian
copula whose correlation matrix is estimated from historical same-game
residual pairs on training seasons only.

The chain is:

    game pace/volume -> team plays -> pass/rush split -> player opportunity
    -> yards per opportunity -> player yardage

Receiver yardage is drawn as a share of the quarterback's realised passing
yards, so QB/WR correlation is structural rather than bolted on afterwards.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import panel as P

GAME_ID_COL = "m_canonical_game_id"


# ---------------------------------------------------------------------------
# Calibration from history (train seasons only)
# ---------------------------------------------------------------------------
def _causal_expectation(df: pd.DataFrame, col: str, window: int = 5) -> pd.Series:
    return df.groupby("player_id")[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=2).mean())


def estimate_correlations(max_season: int = 2019) -> dict:
    """Empirical same-game residual correlations.

    Residual is actual minus the player's own trailing-5 mean, which is the
    simplest causal expectation available; the correlation of those residuals
    is the part of a game two players genuinely share.
    """
    df = P.load_event_causal()
    df = df[df.season <= max_season].copy()
    df["kick"] = pd.to_datetime(df["m_kickoff_timestamp_utc"], utc=True)
    df = df.sort_values(["player_id", "kick"])

    df["exp_pass"] = _causal_expectation(df, "passing_yards")
    df["exp_rec"] = _causal_expectation(df, "receiving_yards")
    df["exp_rush"] = _causal_expectation(df, "rushing_yards")
    df["r_pass"] = df.passing_yards - df.exp_pass
    df["r_rec"] = df.receiving_yards - df.exp_rec
    df["r_rush"] = df.rushing_yards - df.exp_rush

    out = {"estimated_on_seasons_upto": max_season}

    qbs = df[(df.position == "QB") & (df.attempts >= 10) & df.r_pass.notna()]
    # One QB per team-game: the primary attempt-getter.
    qbs = qbs.sort_values("attempts", ascending=False).groupby(
        [GAME_ID_COL, "recent_team"], as_index=False).head(1)

    pairs = qbs.merge(qbs, on=GAME_ID_COL, suffixes=("_a", "_b"))
    pairs = pairs[pairs.recent_team_a < pairs.recent_team_b]
    if len(pairs) > 200:
        out["qb_vs_opposing_qb"] = float(
            np.corrcoef(pairs.r_pass_a, pairs.r_pass_b)[0, 1])
        out["qb_vs_opposing_qb_n"] = int(len(pairs))

    # QB and his own lead receiver.
    recs = df[(df.position.isin(["WR", "TE"])) & (df.targets >= 3) & df.r_rec.notna()]
    lead = recs.sort_values("targets", ascending=False).groupby(
        [GAME_ID_COL, "recent_team"], as_index=False).head(1)
    qw = qbs.merge(lead, on=[GAME_ID_COL, "recent_team"], suffixes=("_qb", "_wr"))
    if len(qw) > 200:
        out["qb_vs_own_lead_receiver"] = float(
            np.corrcoef(qw.r_pass_qb, qw.r_rec_wr)[0, 1])
        out["qb_vs_own_lead_receiver_n"] = int(len(qw))

    # QB and his own lead back's rushing - the game-script channel.
    rbs = df[(df.position == "RB") & (df.carries >= 5) & df.r_rush.notna()]
    lead_rb = rbs.sort_values("carries", ascending=False).groupby(
        [GAME_ID_COL, "recent_team"], as_index=False).head(1)
    qr = qbs.merge(lead_rb, on=[GAME_ID_COL, "recent_team"], suffixes=("_qb", "_rb"))
    if len(qr) > 200:
        out["qb_pass_vs_own_lead_rb_rush"] = float(
            np.corrcoef(qr.r_pass_qb, qr.r_rush_rb)[0, 1])
        out["qb_pass_vs_own_lead_rb_rush_n"] = int(len(qr))

    # Two receivers on the same team compete for the same targets.
    two = recs.sort_values("targets", ascending=False).groupby(
        [GAME_ID_COL, "recent_team"], as_index=False).head(2)
    two["rank"] = two.groupby([GAME_ID_COL, "recent_team"]).cumcount()
    w1 = two[two["rank"] == 0]
    w2 = two[two["rank"] == 1]
    ww = w1.merge(w2, on=[GAME_ID_COL, "recent_team"], suffixes=("_1", "_2"))
    if len(ww) > 200:
        out["receiver1_vs_receiver2"] = float(
            np.corrcoef(ww.r_rec_1, ww.r_rec_2)[0, 1])
        out["receiver1_vs_receiver2_n"] = int(len(ww))

    return out


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
@dataclass
class PlayerSpec:
    player_id: str
    position: str          # QB | WR | TE | RB
    team: str              # "home" or "away"
    exp_yards: float       # model point prediction
    sigma: float           # model residual scale
    share: float = 0.0     # target share (WR/TE) or carry share (RB)


@dataclass
class GameSpec:
    home_team: str
    away_team: str
    exp_plays_home: float
    exp_plays_away: float
    exp_pass_rate_home: float
    exp_pass_rate_away: float
    players: list[PlayerSpec] = field(default_factory=list)


class GameSimulator:
    """Correlated draws for every player in one game.

    `corr` comes from `estimate_correlations` on training seasons; the copula
    is built from those measured pairwise values rather than assumed ones.
    """

    def __init__(self, corr: dict, plays_sd: float = 6.5,
                 pass_rate_sd: float = 0.06, shared_pace_weight: float = 0.55):
        self.corr = corr
        self.plays_sd = plays_sd
        self.pass_rate_sd = pass_rate_sd
        self.shared_pace_weight = shared_pace_weight

    def _corr_matrix(self, players: list[PlayerSpec]) -> np.ndarray:
        n = len(players)
        R = np.eye(n)
        c_qq = self.corr.get("qb_vs_opposing_qb", 0.0)
        c_qw = self.corr.get("qb_vs_own_lead_receiver", 0.45)
        c_qr = self.corr.get("qb_pass_vs_own_lead_rb_rush", -0.05)
        c_ww = self.corr.get("receiver1_vs_receiver2", -0.05)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = players[i], players[j]
                same = a.team == b.team
                pa, pb = a.position, b.position
                if pa == "QB" and pb == "QB":
                    r = c_qq
                elif same and (pa == "QB" and pb in ("WR", "TE")):
                    r = c_qw
                elif same and (pb == "QB" and pa in ("WR", "TE")):
                    r = c_qw
                elif same and (pa == "QB" and pb == "RB"):
                    r = c_qr
                elif same and (pb == "QB" and pa == "RB"):
                    r = c_qr
                elif same and pa in ("WR", "TE") and pb in ("WR", "TE"):
                    r = c_ww
                elif not same:
                    # Opposing skill players share only the game environment.
                    r = 0.5 * c_qq
                else:
                    r = 0.0
                R[i, j] = R[j, i] = float(np.clip(r, -0.95, 0.95))
        return _nearest_psd(R)

    def simulate(self, spec: GameSpec, n_paths: int = 10_000,
                 seed: int = C.SEED) -> dict:
        rng = np.random.default_rng(seed)
        players = spec.players
        n = len(players)

        # Level 1: shared game environment.
        pace_z = rng.standard_normal(n_paths)
        idio_h = rng.standard_normal(n_paths)
        idio_a = rng.standard_normal(n_paths)
        w = self.shared_pace_weight
        zh = w * pace_z + np.sqrt(1 - w ** 2) * idio_h
        za = w * pace_z + np.sqrt(1 - w ** 2) * idio_a

        plays_h = np.clip(spec.exp_plays_home + self.plays_sd * zh, 30, 100)
        plays_a = np.clip(spec.exp_plays_away + self.plays_sd * za, 30, 100)

        # Level 2: pass/rush split, then team dropbacks.
        pr_h = np.clip(spec.exp_pass_rate_home +
                       self.pass_rate_sd * rng.standard_normal(n_paths), 0.2, 0.85)
        pr_a = np.clip(spec.exp_pass_rate_away +
                       self.pass_rate_sd * rng.standard_normal(n_paths), 0.2, 0.85)
        db = {"home": plays_h * pr_h, "away": plays_a * pr_a}
        carries = {"home": plays_h * (1 - pr_h), "away": plays_a * (1 - pr_a)}

        # Level 3: correlated efficiency shocks through the copula.
        R = self._corr_matrix(players)
        L = np.linalg.cholesky(R)
        Z = (L @ rng.standard_normal((n, n_paths)))

        draws = {}
        qb_yards = {}
        for i, p in enumerate(players):
            if p.position == "QB":
                vol = db[p.team]
                vol_ratio = vol / max(np.mean(vol), 1e-6)
                y = p.exp_yards * vol_ratio + p.sigma * Z[i]
                y = np.clip(y, 0, None)
                qb_yards[p.team] = y
                draws[p.player_id] = y

        for i, p in enumerate(players):
            if p.position in ("WR", "TE"):
                team_pass = qb_yards.get(p.team)
                if team_pass is None:
                    y = np.clip(p.exp_yards + p.sigma * Z[i], 0, None)
                else:
                    # Share of the QB's realised yards, with idiosyncratic noise.
                    base = np.mean(team_pass) if np.mean(team_pass) > 0 else 1.0
                    y = p.exp_yards * (team_pass / base) + p.sigma * Z[i]
                    y = np.clip(y, 0, None)
                draws[p.player_id] = y
            elif p.position == "RB":
                vol = carries[p.team]
                vol_ratio = vol / max(np.mean(vol), 1e-6)
                draws[p.player_id] = np.clip(
                    p.exp_yards * vol_ratio + p.sigma * Z[i], 0, None)

        return {
            "n_paths": n_paths,
            "seed": seed,
            "draws": draws,
            "team_plays": {"home": plays_h, "away": plays_a},
            "team_dropbacks": db,
            "correlation_matrix": R.tolist(),
            "player_order": [p.player_id for p in players],
        }


def _nearest_psd(R: np.ndarray) -> np.ndarray:
    """Project a correlation matrix onto the PSD cone by clipping eigenvalues,
    so an assembled matrix of pairwise estimates is always factorisable."""
    vals, vecs = np.linalg.eigh((R + R.T) / 2)
    vals = np.clip(vals, 1e-6, None)
    out = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(out))
    out = out / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return out


def marginal_p_over(sim: dict, player_id: str, line: float) -> float:
    return float((sim["draws"][player_id] > line).mean())


def joint_p_over(sim: dict, legs: list[tuple[str, float]]) -> dict:
    """P(all legs hit), with the independence product alongside it.

    The ratio of the two is the quantity that makes or loses money on a
    same-game parlay.
    """
    mask = np.ones(sim["n_paths"], dtype=bool)
    marginals = []
    for pid, line in legs:
        hit = sim["draws"][pid] > line
        marginals.append(float(hit.mean()))
        mask &= hit
    joint = float(mask.mean())
    indep = float(np.prod(marginals))
    return {
        "legs": [{"player_id": p, "line": l, "p_over": m}
                 for (p, l), m in zip(legs, marginals)],
        "JOINT_PROBABILITY": joint,
        "INDEPENDENCE_ASSUMPTION": indep,
        "CORRELATION_MULTIPLIER": round(joint / indep, 4) if indep > 0 else None,
    }


if __name__ == "__main__":
    corr = estimate_correlations()
    print(json.dumps(corr, indent=2))
