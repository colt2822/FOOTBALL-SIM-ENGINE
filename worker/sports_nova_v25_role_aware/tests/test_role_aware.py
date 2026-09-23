"""V25 tests T1-T7 from the brief, plus role resolution, OUT handling, Doubtful, bit-identity and accounting checks."""
import hashlib
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from worker.sports_nova_modular.firewall import assert_market_free
from worker.sports_nova_v3.allocation import _shares
from worker.sports_nova_v3.schemas import (
    Evidence, OpportunityShares, PlayerState, PregameState, TeamState, Uncertainty)
from worker.sports_nova_v23 import simulator as v23
from worker.sports_nova_v25_role_aware import config as cfg
from worker.sports_nova_v25_role_aware.eligibility import RosterSnapshot, classify_player
from worker.sports_nova_v25_role_aware.redistribution import NoEligibleRecipientsError, repair_pregame_state
from worker.sports_nova_v25_role_aware.simulator import MODEL_VERSION, prepare_state, simulate_game

ROOT = Path(__file__).resolve().parents[3]
T0 = datetime(2026, 9, 19, 12, tzinfo=timezone.utc)
EV = Evidence(source_id="TEST_ONLY", raw_sha256="a" * 64, event_end=T0, available_at=T0, retrieved_at=T0,
              availability_basis="LIVE_RECEIPT")

HOME_CARRY = {"h-star": .50, "h-b1": .20, "h-b2": .10, "h-b3": .05, "h-qb": .12, "h-wr": .02, "h-p": .01}
HOME_TARGET = {"h-wr1": .30, "h-wr2": .20, "h-te": .15, "h-star": .10, "h-b1": .10, "h-b2": .05, "h-qb": .01}
AWAY_CARRY = {"a-rb1": .60, "a-rb2": .25, "a-qb": .05}
AWAY_TARGET = {"a-wr1": .40, "a-wr2": .30, "a-te": .10}
POS = {"h-qb": "QB", "a-qb": "QB", "h-wr": "WR", "h-wr1": "WR", "h-wr2": "WR", "a-wr1": "WR", "a-wr2": "WR", "h-te": "TE", "a-te": "TE", "h-p": "OTHER"}


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
    if out:
        state = state.model_copy(update={"players": tuple(_player(p.player_id, p.team_id, "OUT", p.uncertainty.effective_sample_size)
                                                          if p.player_id in out else p for p in state.players)})
    return state


def snapshot(state, overrides=None, game_status=None, positions=None):
    roster = {p.player_id: (p.team_id, "ACT") for p in state.players}
    for pid, v in (overrides or {}).items():
        if v is None:
            roster.pop(pid, None)
        else:
            roster[pid] = v
    return RosterSnapshot(roster=roster, game_status=game_status or {}, roster_week=2, positions=positions or {})


def eff(state, team, kind):
    ids, vals, _ = _shares(state, team, kind)
    t = float(vals.sum())
    return {i: float(v) / t for i, v in zip(ids, vals)} if t else {}


def v24_style(state, team, kind, dead):
    """What V24's pro-rata-over-every-survivor gives (independent re-implementation, used as the control)."""
    pre = eff(state, team, kind)
    live = {k: v for k, v in pre.items() if k not in dead}
    t = sum(live.values())
    return {k: v / t for k, v in live.items()}


class RoleAwareTests(unittest.TestCase):
    # T1
    def test_T1_removed_rb_carry_mass_cannot_increase_qb_share(self):
        s = make_state()
        pre = eff(s, "HOM", "carry")
        control = v24_style(s, "HOM", "carry", {"h-star", "h-b1"})
        self.assertGreater(control["h-qb"], pre["h-qb"] * 2)                   # the V24 defect exists in the control
        rep, audit = prepare_state(s, snapshot(s, {"h-star": ("HOM", "RES"), "h-b1": ("HOM", "INA")}))
        post = eff(rep, "HOM", "carry")
        self.assertAlmostEqual(post["h-qb"], pre["h-qb"], places=12)           # QB held exactly at the no-removal share
        self.assertAlmostEqual(post["h-p"], pre["h-p"], places=12)             # non-skill role held too
        self.assertAlmostEqual(sum(post.values()), 1.0, places=12)
        for pid in ("h-b2", "h-b3", "h-wr"):
            self.assertGreater(post[pid], pre[pid])
        # every QB in the audit: no increase
        for r in audit.rows:
            if r["ROLE"] == "QB":
                self.assertLessEqual(r["V25_POST_SHARE"], r["PRE_FULL_POOL_SHARE"] + 1e-12)

    # T2
    def test_T2_qb_scramble_component_unchanged_by_non_qb_removal(self):
        s = make_state()
        snap = snapshot(s, {"h-star": ("HOM", "RES")})
        rep, audit = prepare_state(s, snap)
        self.assertTrue(audit.qb_inputs_unchanged)
        # scramble path draws only selection.scrambles into the QB rush_attempts; identical seed => identical scramble-driven QB pass_attempts stream
        # QB share in the designed-carry pool is equal, so QB mean rush attempts must not rise beyond MC noise vs the no-removal V23 baseline
        a = simulate_game(s, 600, 21, MODEL_VERSION, roster=snap)
        b = v23.simulate_game(s, 600, 21, v23.MODEL_VERSION)
        j = a.player_ids.index("h-qb")
        qa, qb = a.player_stats["rush_attempts"][:, j].mean(), b.player_stats["rush_attempts"][:, j].mean()
        se = a.player_stats["rush_attempts"][:, j].std() / np.sqrt(600) * 2.5
        self.assertLess(qa, qb + se)                                           # V23 with the RB present is the ceiling for QB carries
        # and it is clearly below the V24-style inflation (control state built by hand)
        ctrl = s.model_copy(update={"home": s.home.model_copy(update={"carry_shares": s.home.carry_shares.model_copy(update={
            "shares": tuple(0.0 if pid == "h-star" else sh for pid, sh in zip(s.home.carry_shares.player_ids, s.home.carry_shares.shares)), "residual_share": 0.0})})})
        c = v23.simulate_game(ctrl, 600, 21, v23.MODEL_VERSION)
        self.assertGreater(c.player_stats["rush_attempts"][:, j].mean(), qa + 0.5)

    # T3
    def test_T3_same_team_non_qb_eligible_recipients_conserve_removed_mass(self):
        s = make_state()
        rep, audit = prepare_state(s, snapshot(s, {"h-star": ("HOM", "RES"), "h-b3": ("AWY", "ACT")}))
        for summ in audit.summaries:
            self.assertAlmostEqual(summ["PRE_REMOVAL_TOTAL"], summ["POST_REDISTRIBUTION_TOTAL"], places=10)
            self.assertAlmostEqual(summ["REMOVED_MASS"], summ["REDISTRIBUTED_MASS"], places=10)
            self.assertLessEqual(summ["MASS_TO_HELD_ROLES"], 1e-12)
        pre, post = eff(s, "HOM", "carry"), eff(rep, "HOM", "carry")
        recips = ("h-b1", "h-b2", "h-wr")
        ratios = [post[k] / pre[k] for k in recips]
        for r in ratios:
            self.assertAlmostEqual(r, ratios[0], places=12)                    # pro-rata among eligible non-QB recipients
        self.assertTrue(set(post) <= {p.player_id for p in rep.players if p.team_id == "HOM"})
        self.assertNotIn("h-b3", post)                                         # other-team player never leaks
        self.assertNotIn("h-b3", eff(rep, "AWY", "carry"))

    # T4
    def test_T4_target_redistribution_stays_in_valid_receiving_roles(self):
        s = make_state()
        rep, audit = prepare_state(s, snapshot(s, {"h-wr1": ("HOM", "RES"), "h-te": ("HOM", "INA")}))
        pre, post = eff(s, "HOM", "target"), eff(rep, "HOM", "target")
        self.assertAlmostEqual(post["h-qb"], pre["h-qb"], places=12)           # QB target sliver held, receives nothing
        self.assertAlmostEqual(sum(post.values()), 1.0, places=12)
        got = {k for k in post if post[k] > pre[k] + 1e-12}
        self.assertTrue(got <= {"h-wr2", "h-star", "h-b1", "h-b2"})
        for r in audit.rows:
            if r["KIND"] == "target" and r["V25_POST_SHARE"] > r["PRE_FULL_POOL_SHARE"] + 1e-12:
                self.assertIn(r["ROLE"], cfg.RECIPIENT_ROLES)
        # no carry-role leakage: the target pool result is independent of what happens in the carry pool
        rep2, _ = prepare_state(s, snapshot(s, {"h-wr1": ("HOM", "RES"), "h-te": ("HOM", "INA"), "h-star": ("HOM", "RES")}))
        t1, t2 = eff(rep, "HOM", "target"), eff(rep2, "HOM", "target")
        self.assertEqual(set(t1) - {"h-star"}, set(t2) - {"h-star"})
        # h-star is also lost from the target pool in rep2 -> only that removal differs; without it the pools are identical
        rep3, _ = prepare_state(s, snapshot(s, {"h-wr1": ("HOM", "RES"), "h-te": ("HOM", "INA"), "h-b3": ("HOM", "RES")}))
        for k, v in t1.items():
            self.assertAlmostEqual(eff(rep3, "HOM", "target")[k], v, places=12)   # h-b3 has no target share: carry-only removal changes nothing in targets

    # T5
    def test_T5_zero_surviving_recipients_is_an_explicit_error(self):
        s = make_state()
        every_skill = {p: ("HOM", "RES") for p in HOME_CARRY if p not in ("h-qb", "h-p")}
        with self.assertRaises(NoEligibleRecipientsError):
            repair_pregame_state(s, snapshot(s, every_skill))                 # only the QB/punter survive: mass must NOT go to them
        with self.assertRaises(NoEligibleRecipientsError):
            prepare_state(s, snapshot(s, every_skill))
        # never silently allocated: the raise happens before any state is produced
        with self.assertRaises(NoEligibleRecipientsError):
            simulate_game(s, 5, 1, MODEL_VERSION, roster=snapshot(s, every_skill))

    # T6
    def test_T6_v23_artifacts_and_hashes_unchanged(self):
        frozen = json.loads((ROOT / "data/sports_nova_v3/sunday_2026_09_20/V24_PREREG/V23_HASHES_BEFORE.json").read_text())["FILES"]
        self.assertGreater(len(frozen), 10)
        for rel, want in frozen.items():
            got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            self.assertEqual(got, want, rel)
        pkg = Path(__file__).resolve().parents[1]
        for f in pkg.glob("*.py"):
            imports = [ln for ln in f.read_text().splitlines() if ln.startswith(("import ", "from "))]
            self.assertFalse([ln for ln in imports if "sports_nova_v24" in ln], f.name)      # imports nothing from either V24 package

    # T7
    def test_T7_market_contamination_zero(self):
        for d in (cfg.ELIGIBILITY_POLICY, cfg.DOUBTFUL_POLICY, cfg.ROLE_POLICY, cfg.REDISTRIBUTION_POLICY):
            assert_market_free(d)
        s = make_state()
        rep, audit = prepare_state(s, snapshot(s, {"h-star": ("HOM", "RES")}))
        assert_market_free(json.loads(rep.model_dump_json()))
        assert_market_free([r for r in audit.rows])
        assert_market_free(audit.summaries)
        src = (Path(__file__).resolve().parents[1] / "simulator.py").read_text() + (Path(__file__).resolve().parents[1] / "redistribution.py").read_text()
        for bad in ("kalshi", "sportsbook", "odds", "orderbook"):
            self.assertNotIn(bad, src.lower())

    # --- additional checks
    def test_no_removal_pool_is_bit_identical_to_v23(self):
        s = make_state()
        snap = snapshot(s)
        rep, audit = repair_pregame_state(s, snap)
        self.assertIs(rep, s)
        self.assertFalse(audit.changed)
        a = simulate_game(s, 40, 5, MODEL_VERSION, roster=snap)
        b = v23.simulate_game(s, 40, 5, v23.MODEL_VERSION)
        for k in a.player_stats:
            self.assertTrue(np.array_equal(a.player_stats[k], b.player_stats[k]))
        rep2, _ = repair_pregame_state(s, snapshot(s, {"h-star": ("HOM", "RES")}))
        self.assertIs(rep2.away, s.away)                                       # untouched team keeps the identical object

    def test_out_player_mass_also_bypasses_the_qb(self):
        s = make_state(out=("h-star",))
        pre_v23 = eff(s, "HOM", "carry")                                       # V23's own gate: OUT mass flows pro-rata incl. the QB
        rep, _ = prepare_state(s, snapshot(s))
        post = eff(rep, "HOM", "carry")
        self.assertNotIn("h-star", post)
        self.assertLess(post["h-qb"], pre_v23["h-qb"])
        self.assertAlmostEqual(post["h-qb"], HOME_CARRY["h-qb"] / sum(HOME_CARRY.values()), places=12)

    def test_role_resolution_uses_roster_position_then_depth_then_state(self):
        s = make_state()
        p = next(x for x in s.players if x.player_id == "h-b1")               # state position RB
        self.assertEqual(classify_player(p, snapshot(s)).role, "RB")
        self.assertEqual(classify_player(p, snapshot(s, positions={"h-b1": ("RB", "FB")})).role, "RB")
        self.assertEqual(classify_player(p, snapshot(s, positions={"h-b1": ("P", "P")})).role, "RB")   # state position is the last skill fallback
        pun = next(x for x in s.players if x.player_id == "h-p")               # state position OTHER
        self.assertEqual(classify_player(pun, snapshot(s)).role, cfg.NON_SKILL_ROLE)
        self.assertEqual(classify_player(pun, snapshot(s, positions={"h-p": ("RB", "FB")})).role, "RB")
        self.assertEqual(classify_player(pun, snapshot(s, positions={"h-p": ("RB", "FB")})).eligible, True)

    def test_roster_position_qb_is_held_even_if_state_says_rb(self):
        s = make_state()
        snap = snapshot(s, {"h-star": ("HOM", "RES")}, positions={"h-b1": ("QB", "QB")})
        rep, audit = prepare_state(s, snap)
        pre = eff(s, "HOM", "carry")
        self.assertAlmostEqual(eff(rep, "HOM", "carry")["h-b1"], pre["h-b1"], places=12)

    def test_doubtful_weight_mass_goes_to_recipients_not_qb(self):
        s = make_state()
        rep, audit = prepare_state(s, snapshot(s, game_status={"h-b1": "DOUBTFUL"}))
        pre, post = eff(s, "HOM", "carry"), eff(rep, "HOM", "carry")
        self.assertLess(post["h-b1"], pre["h-b1"] * 0.05)
        self.assertAlmostEqual(post["h-qb"], pre["h-qb"], places=12)
        self.assertGreater(post["h-star"], pre["h-star"])

    def test_team_player_accounting(self):
        s = make_state()
        snap = snapshot(s, {"h-star": ("HOM", "RES"), "a-rb2": ("AWY", "DEV")}, {"h-b1": "DOUBTFUL"})
        b = simulate_game(s, 60, 11, MODEL_VERSION, roster=snap)
        for j, tid in enumerate(b.team_ids):
            mine = [i for i, pid in enumerate(b.player_ids) if b.player_team[pid] == tid]
            self.assertTrue(np.array_equal(sum(b.player_stats["rush_attempts"][:, i] for i in mine), b.team_stats["rush_attempts"][:, j]))
            self.assertTrue(np.array_equal(sum(b.player_stats["targets"][:, i] for i in mine), b.team_stats["pass_attempts"][:, j]))

    def test_requires_snapshot_and_frozen_version(self):
        s = make_state()
        with self.assertRaises(TypeError):
            simulate_game(s, 5, 1, MODEL_VERSION, roster=None)
        with self.assertRaises(ValueError):
            simulate_game(s, 5, 1, "sports_nova_v23.drive_block.residual_allocation_fix.1", roster=snapshot(s))

    def test_policy_hashes_stable_and_distinct(self):
        self.assertEqual(cfg.policy_hash(cfg.ROLE_POLICY), cfg.ROLE_POLICY_HASH)
        self.assertEqual(len({cfg.ELIGIBILITY_POLICY_HASH, cfg.DOUBTFUL_POLICY_HASH, cfg.ROLE_POLICY_HASH, cfg.REDISTRIBUTION_POLICY_HASH}), 4)


if __name__ == "__main__":
    unittest.main()
