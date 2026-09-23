"""SPORTS-NOVA terminal UI.  Plain stdin/stdout, standard library only.

Flow: select game -> (optional) edit football inputs -> sim count -> run M1 -> distributions -> NOVA fair book ->
(optional) enter external book -> NOVA vs market -> artifact saved.  All numbers come from `core`.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from . import core
from .core import Overrides, ManualBook, TerminalError

BAR = "=" * 60
SIM_CHOICES = {"1": 100, "2": 1000, "3": 5000, "4": 10000, "5": 25000}
DEFAULT_SIMS = 5000


# ------------------------------------------------------------------------------------------- rendering
def pct(x) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.1f}%"


def pp(x) -> str:
    return f"{100 * x:+.1f}pp"


def num(x) -> str:
    return f"{x:.1f}"


def american(p) -> str:
    a = core.prob_to_american(p)
    return "n/a" if a is None else f"{a:+.0f}"


def dist_line(d: dict) -> str:
    return (f"mean {num(d['mean']):>6}  median {num(d['median']):>6}  P10 {num(d['P10']):>6}  "
            f"P25 {num(d['P25']):>6}  P75 {num(d['P75']):>6}  P90 {num(d['P90']):>6}")


def render_sim(rec: dict, out=print) -> None:
    s = rec["M1_SUMMARY"]
    away, home = s["teams"]["away"], s["teams"]["home"]
    out(BAR)
    out("NOVA GAME SIM  [SIMULATOR OUTPUT]")
    out(f"{away} @ {home}")
    out(f"N = {s['n_sims']:,}   {rec['M1_RUN']['simulation_version']}")
    out(BAR)
    w = s["win_probability"]
    out("\nWIN PROBABILITY")
    out(f"  {away} (AWAY): {pct(w['AWAY'])}")
    out(f"  {home} (HOME): {pct(w['HOME'])}")
    if w["TIE"] > 0:
        out(f"  TIE:          {pct(w['TIE'])}")
    out("\nSCORE")
    out(f"  {away:<4} {dist_line(s['score']['away'])}")
    out(f"  {home:<4} {dist_line(s['score']['home'])}")
    out(f"\nMARGIN ({home} minus {away})")
    out(f"       {dist_line(s['margin_home_minus_away'])}")
    out("\nTOTAL")
    out(f"       {dist_line(s['total'])}")
    out("\nTEAM VOLUME / YARDS")
    for side in ("away", "home"):
        t = s["team"][side]
        out(f"  {t['team']}:")
        for k, label in (("plays_pass_plus_rush", "plays (pass+rush att)"), ("pass_attempts", "pass attempts"),
                         ("rush_attempts", "rush attempts"), ("pass_yards", "pass yards"), ("rush_yards", "rush yards"),
                         ("total_yards", "total yards")):
            out(f"    {label:<22}{dist_line(t[k])}")
    out(f"\n  Not emitted by M1 (not shown, not invented): {', '.join(s['unavailable_outputs'])}")
    m1 = rec["M1_RUN"]
    out(f"\nseed {m1['random_seed']}  |  M1 hash {rec.get('M1_HASH', '')[:16]}...  |  runtime {m1['runtime_seconds']}s")


def render_players(rec: dict, out=print) -> None:
    s = rec["M1_SUMMARY"]
    plan = {"QB": (1, "pass_yards", ["pass_attempts", "pass_tds"]), "RB": (3, "rush_yards", ["rush_attempts"]),
            "WR": (4, "receiving_yards", ["targets", "receptions"]), "TE": (2, "receiving_yards", ["targets", "receptions"])}
    sort_by = {"QB": "pass_attempts", "RB": "rush_attempts", "WR": "targets", "TE": "targets"}
    out(BAR)
    out("PLAYER PROJECTIONS  [SIMULATOR OUTPUT - distributions only, not props]")
    out(BAR)
    for team in (s["teams"]["away"], s["teams"]["home"]):
        out(f"\n{team}")
        shown = 0
        for pos, (k, headline, extras) in plan.items():
            ps = [p for p in s["players"] if p["team"] == team and p["position"] == pos and p["stats"][sort_by[pos]]["mean"] > 0.05]
            ps.sort(key=lambda p: -p["stats"][sort_by[pos]]["mean"])
            for p in ps[:k]:
                st = p["stats"]
                ex = "  ".join(f"{e} {st[e]['mean']:.1f}" for e in extras)
                out(f"  {pos:<3} {(p['name'] or p['player_id'])[:22]:<22} {headline}: "
                    f"P10 {st[headline]['P10']:.0f}  P25 {st[headline]['P25']:.0f}  P50 {st[headline]['median']:.0f}  "
                    f"P75 {st[headline]['P75']:.0f}  P90 {st[headline]['P90']:.0f}   | {ex}  P(scores rush/rec TD) {pct(p['td_probability'])}")
                shown += 1
        if shown == 0:
            out("  (no tracked players)")


def render_nova_book(rec: dict, out=print) -> None:
    m2 = rec["M2_NOVA_BOOK"]
    p, fl, s = m2["probabilities"], m2["fair_lines"], rec["M1_SUMMARY"]
    away, home = s["teams"]["away"], s["teams"]["home"]
    out(BAR)
    out("NOVA FAIR BOOK  [NOVA FAIR VALUE - derived from M1 only]")
    out(BAR)
    out(f"  NOVA_ML_AWAY_PROB  {away}: {pct(p['NOVA_ML_AWAY_PROB'])}   fair odds {american(m2['fair_prices']['AWAY_ML'])}")
    out(f"  NOVA_ML_HOME_PROB  {home}: {pct(p['NOVA_ML_HOME_PROB'])}   fair odds {american(m2['fair_prices']['HOME_ML'])}")
    if p["NOVA_TIE_PROB"] > 0:
        out(f"  (tie {pct(p['NOVA_TIE_PROB'])}; fair odds above are two-way, ties excluded)")
    fs = fl["NOVA_FAIR_SPREAD_HOME"]
    out(f"  NOVA_FAIR_SPREAD   {home} {fs:+.1f}   (median simulated margin; mean {-fl['NOVA_MEAN_MARGIN_HOME']:+.1f})")
    out(f"  NOVA_FAIR_TOTAL    {fl['NOVA_FAIR_TOTAL']:.1f}   (median simulated total; mean {fl['NOVA_MEAN_TOTAL']:.1f})")


def render_comparison(rec: dict, out=print) -> None:
    entry = rec.get("M4_COMPARISON")
    if not entry:
        out("(no external book entered for this run)")
        return
    s = rec["M1_SUMMARY"]
    away, home = s["teams"]["away"], s["teams"]["home"]
    b = entry["external_book_input"]
    d = entry["market_detail"]
    nova = entry["M2_NOVA_BOOK_AT_ENTERED_LINES"]
    m4 = {c["market"]: c for c in entry["m4"]}
    out(BAR)
    out("NOVA VS MARKET")
    out("SIMULATOR OUTPUT (NOVA FAIR VALUE) vs EXTERNAL MARKET -> DIFFERENCE")
    out(f"Book: {b['book_name']}   entered {entry['entered_at_utc'][:19]}Z   (differences are informational, not bets)")
    out(BAR)
    if "moneyline" in d:
        ml = d["moneyline"]
        out("\nMONEYLINE")
        for side, team, key, raw_k, dv_k, price in (("HOME", home, "HOME_ML", "home_raw", "home_devig", b["home_ml"]),
                                                     ("AWAY", away, "AWAY_ML", "away_raw", "away_devig", b["away_ml"])):
            if ml.get(raw_k) is None:
                continue
            mkt = ml[dv_k] if ml.get("vig_adjusted") else ml[raw_k]
            tag = "vig-adjusted" if ml.get("vig_adjusted") else "raw, includes vig"
            out(f"  {team} ({side})  NOVA FAIR VALUE {pct(m4[key]['nova_value'])}   EXTERNAL MARKET {price:+.0f} -> "
                f"{pct(mkt)} ({tag}; raw {pct(ml[raw_k])})   DIFFERENCE {pp(m4[key]['delta'])}")
    if "spread" in d:
        sp = d["spread"]
        out("\nSPREAD")
        out(f"  EXTERNAL MARKET: {sp['team']} {sp['line']:+g} @ {sp['price']:+.0f}")
        pr = nova["probabilities"]
        c = m4["SPREAD_COVER"]
        tag = "vig-adjusted" if sp.get("vig_adjusted") else "raw, includes vig"
        out(f"  NOVA cover prob (pushes excluded): {pct(c['nova_value'])}   (raw {pct(pr['SPREAD_COVER_RAW'])}, push {pct(pr['SPREAD_PUSH'])})")
        out(f"  EXTERNAL MARKET implied ({tag}): {pct(c['external_value'])}   DIFFERENCE {pp(c['delta'])}")
        ld = entry["line_differences"]["spread_home"]
        out(f"  Line: NOVA fair {home} {ld['nova_fair']:+.1f} vs market {home} {ld['market']:+.1f}  (difference {ld['difference']:+.1f} pts)")
    if "total" in d:
        t = d["total"]
        out("\nTOTAL")
        out(f"  EXTERNAL MARKET: {t['line']:g}")
        pr = nova["probabilities"]
        tag = "vig-adjusted" if t.get("vig_adjusted") else "raw, includes vig"
        for key, lab, raw in (("TOTAL_OVER", "OVER", "TOTAL_OVER_RAW"), ("TOTAL_UNDER", "UNDER", "TOTAL_UNDER_RAW")):
            if key in m4:
                c = m4[key]
                out(f"  {lab:<5} NOVA FAIR VALUE {pct(c['nova_value'])} (raw {pct(pr[raw])})   EXTERNAL MARKET {pct(c['external_value'])} "
                    f"({tag})   DIFFERENCE {pp(c['delta'])}")
        if pr["TOTAL_PUSH"] > 0:
            out(f"  (push probability {pct(pr['TOTAL_PUSH'])}; NOVA over/under above exclude pushes)")
        ld = entry["line_differences"]["total"]
        out(f"  Line: NOVA fair {ld['nova_fair']:.1f} vs market {ld['market']:.1f}  (difference {ld['difference']:+.1f} pts)")
    for w in entry.get("warnings", []):
        out(f"  ! {w}")
    if not d:
        out("  (nothing comparable was entered)")


def render_run(rec: dict, out=print) -> None:
    gi = rec["GAME_INPUTS"]
    out(f"{gi['away_team']} @ {gi['home_team']}   {gi['kickoff_et']}   run {rec['RUN_TIMESTAMP_UTC'][:19]}Z   status {rec['STATUS']}")
    if rec["STATUS"] != "OK":
        out(f"  ERROR: {rec.get('ERROR')}")
        return
    render_sim(rec, out)
    render_nova_book(rec, out)
    render_comparison(rec, out)
    for w in rec.get("WARNINGS", []):
        out(f"  ! {w}")


# ------------------------------------------------------------------------------------------- app
class App:
    def __init__(self, inp=input, out=print):
        self.inp, self.out = inp, out

    def ask(self, prompt: str, default: str | None = None) -> str:
        suffix = f" [{default}]" if default else ""
        v = self.inp(f"{prompt}{suffix}: ").strip()
        return v if v else (default if default is not None else "")

    def yn(self, prompt: str, default: bool = False) -> bool:
        v = self.ask(f"{prompt} [{'Y/n' if default else 'y/N'}]", "").lower()
        return default if v == "" else v.startswith("y")

    # ---------------------------------------------------------------- main menu
    def main(self) -> int:
        self.out("\nN.O.V.A SPORTS\nFootball Simulation Engine   (SPORTS-NOVA terminal, M1 = " + core.MODEL_VERSION + ")")
        while True:
            self.out("\nSPORTS-NOVA\n\n[1] Simulate Game\n[2] View Previous Run\n[3] Exit")
            try:
                choice = self.ask("Select", "")
            except (EOFError, KeyboardInterrupt):
                self.out("\nbye")
                return 0
            try:
                if choice == "1":
                    self.simulate_flow()
                elif choice == "2":
                    self.view_previous()
                elif choice == "3":
                    self.out("bye")
                    return 0
                else:
                    self.out("Please enter 1, 2 or 3.")
            except (EOFError, KeyboardInterrupt):
                self.out("\n(cancelled - back to menu)")
            except TerminalError as exc:
                self.out(f"\nERROR: {exc}")

    # ---------------------------------------------------------------- simulate flow
    def pick_panel(self) -> str:
        self.out("\nFootball data panel:")
        for i, k in enumerate(core.PANELS, 1):
            self.out(f"  [{i}] {core.PANELS[k]['label']}")
        keys = list(core.PANELS)
        c = self.ask("Panel", "1")
        if not c.isdigit() or not 1 <= int(c) <= len(keys):
            raise TerminalError("not a valid panel number")
        return keys[int(c) - 1]

    def pick_game(self, panel_key: str) -> core.GameSpec:
        sched = core.load_schedule()
        slate = core.upcoming_slate(sched)
        self.out("\nSelect game")
        specs: list[core.GameSpec] = []
        if slate.empty:
            self.out("  (no upcoming games on the schedule snapshot; use manual entry)")
        else:
            self.out(f"  Upcoming slate: 2026 week {int(slate.week.iloc[0])}  ({len(slate)} games not yet kicked off)")
            for row in slate.itertuples():
                specs.append(core.spec_from_row(row))
            days = sorted({s.kickoff_utc.astimezone(core.ET).date() for s in specs})
            self.out("  Dates: " + ", ".join(f"{d:%a %m-%d}" for d in days) + "   (Enter = whole slate)")
            d = self.ask("Date (MM-DD)", "")
            if d:
                keep = [s for s in specs if s.kickoff_utc.astimezone(core.ET).strftime("%m-%d") == d]
                if not keep:
                    self.out(f"  no games on {d}; showing whole slate")
                else:
                    specs = keep
            for i, s in enumerate(specs, 1):
                self.out(f"  {i:>2}. {s.away_team} @ {s.home_team}   {s.kickoff_et()}")
        self.out("   M. manual game entry")
        c = self.ask("Game", "M" if not specs else "").upper()
        if c == "M" or not specs:
            season = int(self.ask("Season", "2026"))
            week = int(self.ask("Week"))
            away = self.ask("Away team code (e.g. DET)")
            home = self.ask("Home team code (e.g. BUF)")
            ko = self.ask("Kickoff, Eastern time (YYYY-MM-DD HH:MM)", "")
            return core.manual_spec(sched, panel_key, season, week, away, home, ko)
        if not c.isdigit() or not 1 <= int(c) <= len(specs):
            raise TerminalError("not a valid game number")
        return specs[int(c) - 1]

    def show_setup(self, p: core.Prepared) -> None:
        sp = p.spec
        fd = p.football_data
        self.out("\n" + BAR + "\nGAME SETUP")
        self.out(f"Game:        {sp.label}   ({sp.game_id}, source {sp.source})")
        self.out(f"Kickoff:     {sp.kickoff_et()}   Venue: {sp.venue or 'n/a'}")
        self.out(f"Panel:       {fd['panel']['key'].upper()}  ends {fd['panel']['max_season']} W{fd['panel']['max_week_in_max_season']}  "
                 f"sha256 {fd['panel']['sha256'][:16]}...")
        self.out(f"Injuries:    {fd['injury_report']['status']}  as of {fd['injury_report']['fetched_at_utc'] or 'n/a'}  "
                 f"({len(fd['players_out_from_report'])} OUT from report, {len(fd['players_out_manual'])} manual)")
        for t, q in fd["qb_identity"].items():
            self.out(f"QB {t}:      {q}")
        self.out(f"Weather:     {fd['weather']}")
        self.out(f"Football data: {'READY' if p.roster else 'BLOCKED'}   players in state: {fd['n_players_in_state']}")
        for w in p.warnings:
            self.out(f"  ! {w}")
        self.out(BAR)

    def edit_inputs(self, spec, overrides: Overrides, panel_key: str) -> core.Prepared:
        while True:
            p = core.prepare(spec, panel_key, overrides)
            self.show_setup(p)
            if not self.yn("Edit football inputs (injuries / QB / refresh report)?", False):
                return p
            self.out("  [O] mark a player OUT   [Q] set starting QB   [C] clear manual edits   "
                     "[R] refresh injury report (network)   [N] operator note   [D] done")
            c = self.ask("Edit", "D").upper()
            if c == "D":
                return p
            if c == "O":
                who = self.pick_player(p, None)
                if who:
                    overrides.out_players[who["player_id"]] = who["name"] or who["player_id"]
            elif c == "Q":
                team = self.ask(f"Team ({spec.away_team}/{spec.home_team})").upper()
                who = self.pick_player(p, "QB", team)
                if who:
                    overrides.qb_override[who["team"]] = who["player_id"]
            elif c == "C":
                overrides.out_players.clear()
                overrides.qb_override.clear()
            elif c == "R":
                r = core.refresh_injury_report()
                self.out(f"  refreshed injury report: {r['rows']} rows, latest week {r['max_week']}")
            elif c == "N":
                overrides.notes = self.ask("Note (recorded in artifact only, NOT fed to M1)")

    def pick_player(self, p: core.Prepared, position: str | None, team: str | None = None):
        term = self.ask("Search player name / id (part)").lower()
        hits = [r for r in p.roster if (position is None or r["position"] == position) and (team is None or r["team"] == team)
                and (term in (r["name"] or "").lower() or term in r["player_id"].lower())][:12]
        if not hits:
            self.out("  no match")
            return None
        for i, r in enumerate(hits, 1):
            self.out(f"  {i:>2}. {r['team']} {r['position']:<3} {r['name'] or '?':<24} {r['player_id']}  [{r['availability']}]")
        c = self.ask("Pick #", "")
        return hits[int(c) - 1] if c.isdigit() and 1 <= int(c) <= len(hits) else None

    def pick_sims(self) -> int:
        self.out("\nHow many simulations?\n  [1] 100\n  [2] 1,000\n  [3] 5,000\n  [4] 10,000\n  [5] 25,000\n  [6] Custom")
        while True:
            c = self.ask("Select", "3")
            if c in SIM_CHOICES:
                n = SIM_CHOICES[c]
            elif c == "6":
                raw = self.ask("Custom count").replace(",", "")
                if not raw.isdigit() or int(raw) <= 0:
                    self.out("  must be a positive integer")
                    continue
                n = int(raw)
            else:
                self.out("  choose 1-6")
                continue
            est = n / core.SIMS_PER_SEC_ESTIMATE
            eta = f"~{est:.0f} sec" if est < 90 else f"~{est / 60:.1f} min"
            self.out(f"  {n:,} simulations, estimated {eta} on CPU (~{core.SIMS_PER_SEC_ESTIMATE:.0f} sims/sec measured)")
            if n > 10000 and not self.yn("  That is a long run. Continue?", False):
                continue
            return n

    def run_with_spinner(self, spec, n, seed, prepared):
        box: dict = {}

        def work():
            try:
                box["r"] = core.execute_run(spec, n, seed, prepared=prepared, save=True)
            except BaseException as exc:  # noqa: BLE001 - reported to the operator, never swallowed silently
                box["e"] = exc

        th = threading.Thread(target=work, daemon=True)
        t0 = time.time()
        th.start()
        est = n / core.SIMS_PER_SEC_ESTIMATE
        interactive = sys.stdout.isatty()
        while th.is_alive():
            th.join(0.5)
            if interactive:
                el = time.time() - t0
                sys.stdout.write(f"\r  simulating... elapsed {el:5.0f}s / est ~{est:.0f}s (M1 has no progress hook)   ")
                sys.stdout.flush()
        if interactive:
            sys.stdout.write("\r" + " " * 78 + "\r")
        if "e" in box:
            raise box["e"]
        return box["r"]

    def simulate_flow(self) -> None:
        panel_key = self.pick_panel()
        spec = self.pick_game(panel_key)
        overrides = Overrides()
        prepared = self.edit_inputs(spec, overrides, panel_key)
        n = self.pick_sims()
        seed = core.default_seed(spec.game_id)
        s = self.ask(f"Seed (default {seed}, same policy as the slate run)", "")
        if s:
            if not s.isdigit():
                raise TerminalError("seed must be a non-negative integer")
            seed = int(s)
        self.out(f"\nGame: {spec.label}\nSimulations: [{n}]\nFootball data: READY")
        if not self.yn("Run simulation?", True):
            return
        result = self.run_with_spinner(spec, n, seed if s else None, prepared)
        self.out("")
        render_sim(result.record, self.out)
        self.out("")
        render_nova_book(result.record, self.out)
        self.out(f"\nSaved: {result.artifact_path}")
        for w in result.record["WARNINGS"]:
            self.out(f"  ! {w}")
        if self.yn("\nView player projections?", False):
            render_players(result.record, self.out)
        while self.yn("\nEnter current external book?", False):
            try:
                book = self.enter_book(spec)
                core.attach_book(result, book)
            except TerminalError as exc:
                self.out(f"  ERROR: {exc}")
                continue
            self.out("")
            render_comparison(result.record, self.out)
            core.save_artifact(result.record)
            self.out(f"\nSaved: {result.record['ARTIFACT_PATH']}")
            self.out("Enter another/updated book? (the simulation is NOT re-run; the book cannot change it)")

    # ---------------------------------------------------------------- manual book (M3)
    def _field(self, prompt: str, parser):
        while True:
            try:
                return parser(self.ask(prompt, ""))
            except TerminalError as exc:
                self.out(f"  {exc}")

    def enter_book(self, spec: core.GameSpec) -> ManualBook:
        self.out("\nEnter external book (blank = skip field; nothing is assumed).  American prices, e.g. -110 / +150.")
        b = ManualBook(book_name=self.ask("BOOK_NAME (DraftKings/FanDuel/BetMGM/Consensus/Manual)", "Manual"))
        b.home_ml = self._field(f"HOME_ML ({spec.home_team})", core.parse_american)
        b.away_ml = self._field(f"AWAY_ML ({spec.away_team})", core.parse_american)
        while True:
            t = self.ask(f"SPREAD_TEAM ({spec.away_team}/{spec.home_team})", "").upper()
            if t in ("", spec.away_team, spec.home_team):
                b.spread_team = t or None
                break
            self.out("  must be one of the two teams (or blank)")
        if b.spread_team:
            b.spread_line = self._field(f"SPREAD_LINE for {b.spread_team} (e.g. -3.5)", core.parse_line)
            b.spread_price = self._field("SPREAD_PRICE", core.parse_american)
            b.spread_other_price = self._field("Opposite-side spread price (optional, enables vig adjustment)", core.parse_american)
        b.total_line = self._field("TOTAL_LINE", core.parse_line)
        b.over_price = self._field("OVER_PRICE", core.parse_american)
        b.under_price = self._field("UNDER_PRICE", core.parse_american)
        ts = self.ask("Timestamp (ISO, blank = now)", "")
        if ts:
            try:
                b.timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if b.timestamp.tzinfo is None:
                    b.timestamp = b.timestamp.replace(tzinfo=timezone.utc)
            except ValueError:
                self.out("  timestamp not understood: using now")
        if b.is_empty():
            raise TerminalError("nothing entered")
        return b

    # ---------------------------------------------------------------- previous runs
    def view_previous(self) -> None:
        files = core.list_artifacts(15)
        if not files:
            self.out("\nNo saved runs yet.")
            return
        self.out("\nPrevious runs (newest first):")
        for i, f in enumerate(files, 1):
            self.out(f"  {i:>2}. {f.parent.name}/{f.name}")
        c = self.ask("Open #", "1")
        if not c.isdigit() or not 1 <= int(c) <= len(files):
            self.out("not a valid run number")
            return
        try:
            rec = json.loads(files[int(c) - 1].read_text(encoding="utf-8"))
            render_run(rec, self.out)
        except (OSError, ValueError, KeyError) as exc:
            self.out(f"could not read that artifact ({exc.__class__.__name__}: {exc})")
            return
        if rec.get("STATUS") == "OK" and self.yn("\nView player projections?", False):
            render_players(rec, self.out)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    return App().main()
