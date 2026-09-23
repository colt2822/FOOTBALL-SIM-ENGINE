"""Synthetic tests A-J from the V24 brief plus bit-identity, reason-code, fail-closed and QB-unchanged checks."""
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from worker.sports_nova_modular.firewall import assert_market_free
from worker.sports_nova_v3.allocation import _shares
from worker.sports_nova_v3.schemas import (
    Evidence, OpportunityShares, PlayerState, PregameState, TeamState, Uncertainty)
from worker.sports_nova_v23 import simulator as v23
from worker.sports_nova_v24_roster_eligibility import config as cfg
from worker.sports_nova_v24_roster_eligibility.eligibility import RosterSnapshot, classify_player
from worker.sports_nova_v24_roster_eligibility.redistribution import assert_repair_invariants, repair_pregame_state
from worker.sports_nova_v24_roster_eligibility.simulator import MODEL_VERSION, prepare_state, simulate_game

T0 = datetime(2026, 9, 19, 12, tzinfo=timezone.utc)
EV = Evidence(source_id="TEST_ONLY", raw_sha256="a" * 64, event_end=T0, available_at=T0, retrieved_at=T0,
              availability_basis="LIVE_RECEIPT")

HOME_CARRY = {"h-star": .50, "h-b1": .20, "h-b2": .10, "h-b3": .05, "h-qb": .05, "h-wr": .02}
HOME_TARGET = {"h-wr1": .30, "h-wr2": .20, "h-te": .15, "h-star": .10, "h-b1": .10, "h-b2": .05}
AWAY_CARRY = {"a-rb1": .60, "a-rb2": .25, "a-qb": .05}
AWAY_TARGET = {"a-wr1": .40, "a-wr2": .30, "a-te": .10}
POS = {"h-qb": "QB", "a-qb": "QB", "h-wr": "WR", "h-wr1": "WR", "h-wr2": "WR", "a-wr1": "WR", "a-wr2": "WR", "h-te": "TE", "a-te": "TE"}


def _player(pid, team, availability="UNKNOWN", sample=30.0):
    return PlayerState(player_id=pid, team_id=team, position=POS.get(pid, "RB"), availability=availability,
                       identity_evidence=EV, availability_evidence=EV if availability != "UNKNOWN" else None,
                       uncertainty=Uncertainty(effective_sample_size=sample, personnel_unknown=True))


def _shares_obj(d):
    total = sum(d.values())
    return OpportunityShares(player_ids=tuple(d), shares=tuple(d.values()), residual_share=1.0 - total, evidence=EV)


def make_state(home_carry=None, home_target=None, out=(), samples=None):
    home_carry, home_target = dict(home_carry or HOME_CARRY), dict(home_target or HOME_TARGET)
    ids = {"HOM": set(home_carry) | set(home_target) | {"h-qb"}, "AWY": set(AWAY_CARRY) | set(AWAY_TARGET) | {"a-qb"}}
    players = tuple(_player(p, t, "UNKNOWN", (samples or {}).get(p, 30.0)) for t, s in ids.items() for p in sorted(s))
    team = lambda tid, c, t: TeamState(team_id=tid, identity_evidence=EV, features=(), carry_shares=_shares_obj(c), target_shares=_shares_obj(t))
    state = PregameState(game_id="TEST_GAME", kickoff=T0 + timedelta(days=1), as_of=T0, ruleset_id="TEST", identity_registry_sha256="b" * 64,
                         game_evidence=EV, home=team("HOM", home_carry, home_target), away=team("AWY", AWAY_CARRY, AWAY_TARGET), players=players)
    if out:   # same post-construction flip the panel adapter uses (the schema validator forbids building OUT + positive share directly)
        state = state.model_copy(update={"players": tuple(_player(p.player_id, p.team_id, "OUT", p.uncertainty.effective_sample_size)
                                                          if p.player_id in out else p for p in state.players)})
    return state


def snapshot(state, overrides=None, game_status=None):
    """Everyone ACT on their state team unless overridden: overrides = {pid: (team, status) | None}."""
    roster = {p.player_id: (p.team_id, "ACT") for p in state.players}
    for pid, v in (overrides or {}).items():
        if v is None:
            roster.pop(pid, None)
        else:
            roster[pid] = v
    return RosterSnapshot(roster=roster, game_status=game_status or {}, roster_week=2)


def eff(state, team, kind):
    ids, vals, _ = _shares(state, team, kind)
    t = float(vals.sum())
    return {i: float(v) / t for i, v in zip(ids, vals)} if t else {}


class RosterEligibilityTests(unittest.TestCase):
    # A
    def test_A_historical_star_now_on_ir_gets_zero_allocation(self):
        s = make_state()
        rep, audit = prepare_state(s, snapshot(s, {"h-star": ("HOM", "RES")}))
        self.assertEqual(audit.eligibility["h-star"].reason, "IR")
        self.assertNotIn("h-star", eff(rep, "HOM", "carry"))
        self.assertNotIn("h-star", eff(rep, "HOM", "target"))
        b = simulate_game(s, 30, 7, MODEL_VERSION, roster=snapshot(s, {"h-star": ("HOM", "RES")}))
        j = b.player_ids.index("h-star")
        self.assertEqual(int(b.player_stats["rush_attempts"][:, j].sum() + b.player_stats["targets"][:, j].sum()), 0)
        v = v23.simulate_game(s, 30, 7, v23.MODEL_VERSION)     # control: the frozen V23 does give the IR player usage
        self.assertGreater(int(v.player_stats["rush_attempts"][:, v.player_ids.index("h-star")].sum()), 0)

    # B / G
    def test_B_G_player_now_on_another_team_or_wrong_team_gets_zero(self):
        s = make_state()
        snap = snapshot(s, {"h-star": ("AWY", "ACT"), "h-b1": ("HOM", "ACT"), "h-b2": None})
        rep, audit = prepare_state(s, snap)
        self.assertEqual(audit.eligibility["h-star"].reason, "OTHER_TEAM")
        self.assertEqual(audit.eligibility["h-b2"].reason, "OFF_ROSTER")
        for pid in ("h-star", "h-b2"):
            self.assertNotIn(pid, eff(rep, "HOM", "carry"))
        self.assertTrue(set(eff(rep, "HOM", "carry")) <= {p.player_id for p in rep.players if p.team_id == "HOM"})
        self.assertNotIn("h-star", eff(rep, "AWY", "carry"))          # never leaks to the new team's pool

    # C
    def test_C_official_out_existing_zero_behaviour_preserved(self):
        s = make_state(out=("h-star",))
        rep, audit = prepare_state(s, snapshot(s))
        self.assertIs(rep, s)                                          # OUT-only: state untouched, V23 gate does the work
        self.assertEqual(audit.eligibility["h-star"].reason, "OUT")
        self.assertNotIn("h-star", eff(rep, "HOM", "carry"))
        a = simulate_game(s, 25, 3, MODEL_VERSION, roster=snapshot(s))
        b = v23.simulate_game(s, 25, 3, v23.MODEL_VERSION)
        for k in a.player_stats:
            self.assertTrue(np.array_equal(a.player_stats[k], b.player_stats[k]))

    def test_C2_out_from_injury_report_without_flip_is_zeroed(self):
        s = make_state()
        rep, audit = prepare_state(s, snapshot(s, game_status={"h-star": "OUT"}))
        self.assertNotIn("h-star", eff(rep, "HOM", "carry"))

    # D / E
    def test_D_E_eligible_backups_receive_pro_rata_share(self):
        s = make_state()
        rep, audit = prepare_state(s, snapshot(s, {"h-star": ("HOM", "RES")}))
        pre, post = eff(s, "HOM", "carry"), eff(rep, "HOM", "carry")
        for pid in ("h-b1", "h-b2", "h-b3", "h-qb", "h-wr"):
            self.assertGreater(post[pid], pre[pid])
        keep = [p for p in pre if p != "h-star"]
        scale = post[keep[0]] / pre[keep[0]]
        for pid in keep:                                               # frozen weights: ratios among survivors unchanged
            self.assertAlmostEqual(post[pid] / pre[pid], scale, places=12)
        self.assertAlmostEqual(post["h-b1"] / post["h-b2"], HOME_CARRY["h-b1"] / HOME_CARRY["h-b2"], places=12)

    # F
    def test_F_all_leaders_unavailable_fallback_keeps_mass(self):
        s = make_state()
        snap = snapshot(s, {"h-star": ("HOM", "RES"), "h-b1": ("HOM", "RES"), "h-b2": ("HOM", "INA")})
        rep, audit = prepare_state(s, snap)
        post = eff(rep, "HOM", "carry")
        self.assertAlmostEqual(sum(post.values()), 1.0, places=12)
        self.assertEqual(set(post), {"h-b3", "h-qb", "h-wr"})
        # no survivor at all -> existing V3/V23 degenerate path, reported, opportunity counted at team level
        every = {p: ("HOM", "RES") for p in HOME_CARRY}
        rep2, audit2 = repair_pregame_state(s, snapshot(s, every))
        self.assertIn("HOM:carry", audit2.zero_survivor_fallbacks)
        from worker.sports_nova_v23.allocation import allocate_opportunities
        res = allocate_opportunities(rep2, "HOM", 30, 25, np.random.default_rng(1))
        self.assertEqual(res.residual_carries, 25)                     # unattributed, not deleted
        self.assertEqual(sum(res.carry_counts.values()) + res.residual_carries, 25)

    # H
    def test_H_doubtful_matches_frozen_policy(self):
        s = make_state()
        rep, audit = prepare_state(s, snapshot(s, game_status={"h-b1": "DOUBTFUL"}))
        pre, post = eff(s, "HOM", "carry"), eff(rep, "HOM", "carry")
        w = cfg.DOUBTFUL_AVAILABILITY_WEIGHT
        denom = sum(v for k, v in pre.items() if k != "h-b1") + pre["h-b1"] * w
        self.assertAlmostEqual(post["h-b1"], pre["h-b1"] * w / denom, places=12)
        self.assertLess(post["h-b1"], pre["h-b1"] * 0.05)
        self.assertEqual(audit.eligibility["h-b1"].availability_weight, w)
        # Doubtful AND ineligible: exclusion wins, no weight
        rep2, a2 = prepare_state(s, snapshot(s, {"h-b1": ("HOM", "RES")}, {"h-b1": "DOUBTFUL"}))
        self.assertNotIn("h-b1", eff(rep2, "HOM", "carry"))

    # I
    def test_I_questionable_existing_policy_preserved(self):
        s = make_state()
        rep, audit = prepare_state(s, snapshot(s, game_status={"h-b1": "QUESTIONABLE"}))
        self.assertIs(rep, s)
        self.assertEqual(audit.eligibility["h-b1"].availability_weight, 1.0)

    # J
    def test_J_team_player_accounting_and_mass_conserved(self):
        s = make_state()
        snap = snapshot(s, {"h-star": ("HOM", "RES"), "a-rb2": ("AWY", "DEV")}, {"h-b1": "DOUBTFUL"})
        rep, audit = prepare_state(s, snap)
        for summ in audit.summaries:
            self.assertAlmostEqual(summ["SUM_SHARE_PLUS_RESIDUAL_AFTER"], 1.0, places=10)
            self.assertAlmostEqual(summ["EFFECTIVE_SUM_AFTER"], 1.0, places=10)
            self.assertAlmostEqual(summ["REMOVED_MASS"], summ["REDISTRIBUTED_MASS"], places=10)
        b = simulate_game(s, 60, 11, MODEL_VERSION, roster=snap)
        for j, tid in enumerate(b.team_ids):
            mine = [i for i, pid in enumerate(b.player_ids) if b.player_team[pid] == tid]
            self.assertTrue(np.array_equal(sum(b.player_stats["rush_attempts"][:, i] for i in mine), b.team_stats["rush_attempts"][:, j]))
            self.assertTrue(np.array_equal(sum(b.player_stats["targets"][:, i] for i in mine), b.team_stats["pass_attempts"][:, j]))

    def test_unchanged_team_and_game_are_bit_identical_to_v23(self):
        s = make_state()
        snap = snapshot(s)
        rep, audit = repair_pregame_state(s, snap)
        self.assertIs(rep, s)
        self.assertFalse(audit.changed)
        a = simulate_game(s, 40, 5, MODEL_VERSION, roster=snap)
        b = v23.simulate_game(s, 40, 5, v23.MODEL_VERSION)
        for k in a.player_stats:
            self.assertTrue(np.array_equal(a.player_stats[k], b.player_stats[k]))
        for k in a.team_stats:
            self.assertTrue(np.array_equal(a.team_stats[k], b.team_stats[k]))
        # a repair on one team leaves the other team's OpportunityShares object untouched
        rep2, _ = repair_pregame_state(s, snapshot(s, {"h-star": ("HOM", "RES")}))
        self.assertIs(rep2.away, s.away)

    def test_reason_codes_and_fail_closed(self):
        s = make_state()
        p = next(x for x in s.players if x.player_id == "h-b1")
        cases = {("HOM", "RES"): "IR", ("HOM", "INA"): "INACTIVE", ("HOM", "DEV"): "INACTIVE", ("HOM", "CUT"): "OFF_ROSTER",
                 ("HOM", "RET"): "OFF_ROSTER", ("HOM", "EXE"): "OTHER_EXPLICIT_INELIGIBLE", ("HOM", "???"): "OTHER_EXPLICIT_INELIGIBLE",
                 ("AWY", "ACT"): "OTHER_TEAM", None: "OFF_ROSTER"}
        for roster_val, want in cases.items():
            e = classify_player(p, snapshot(s, {"h-b1": roster_val}))
            self.assertFalse(e.eligible)
            self.assertEqual(e.reason, want)
            self.assertIn(want, cfg.REASON_CODES)
        both = classify_player(p, snapshot(s, {"h-b1": ("HOM", "RES")}, {"h-b1": "OUT"}))
        self.assertEqual((both.reason, both.all_reasons), ("IR", ("IR", "OUT")))

    def test_qb_inputs_unchanged_and_no_qb_state_edit(self):
        s = make_state()
        rep, audit = prepare_state(s, snapshot(s, {"h-qb": ("HOM", "RES")}))       # QB loses only his carry share
        self.assertTrue(audit.qb_inputs_unchanged)
        self.assertNotIn("h-qb", eff(rep, "HOM", "carry"))
        self.assertEqual([(p.player_id, p.availability, p.features) for p in s.players], [(p.player_id, p.availability, p.features) for p in rep.players])

    def test_history_flags_and_no_cap(self):
        s = make_state(samples={"h-b3": 3.0})
        rep, audit = prepare_state(s, snapshot(s, {"h-star": ("HOM", "RES"), "h-b1": ("HOM", "RES"), "h-b2": ("HOM", "RES"), "h-qb": ("HOM", "RES")}))
        row = next(r for r in audit.rows if r["PLAYER_ID"] == "h-b3" and r["KIND"] == "carry")
        self.assertEqual(row["PLAYER_HISTORY_SAMPLE"], 3.0)
        self.assertGreater(row["POST_REPAIR_SHARE"], cfg.CONCENTRATION_THRESHOLD["carry"])         # not capped
        self.assertIn("CONCENTRATION_ABOVE_THRESHOLD", row["UNCERTAINTY_FLAG"])
        self.assertIn("THIN_HISTORY_ROLE_GROWTH", row["UNCERTAINTY_FLAG"])
        self.assertAlmostEqual(row["SHARE_DELTA"], row["POST_REPAIR_SHARE"] - row["PRE_REPAIR_SHARE"], places=15)

    def test_requires_snapshot_and_frozen_version(self):
        s = make_state()
        with self.assertRaises(TypeError):
            simulate_game(s, 5, 1, MODEL_VERSION, roster=None)
        with self.assertRaises(ValueError):
            simulate_game(s, 5, 1, "sports_nova_v23.drive_block.residual_allocation_fix.1", roster=snapshot(s))

    def test_policies_are_market_free_and_hashes_stable(self):
        for d in (cfg.ELIGIBILITY_POLICY, cfg.DOUBTFUL_POLICY, cfg.REDISTRIBUTION_POLICY):
            assert_market_free(d)
        self.assertEqual(cfg.policy_hash(cfg.DOUBTFUL_POLICY), cfg.DOUBTFUL_POLICY_HASH)
        self.assertEqual(len({cfg.ELIGIBILITY_POLICY_HASH, cfg.DOUBTFUL_POLICY_HASH, cfg.REDISTRIBUTION_POLICY_HASH}), 3)


if __name__ == "__main__":
    unittest.main()
