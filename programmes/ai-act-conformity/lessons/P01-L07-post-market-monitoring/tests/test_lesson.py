"""Autograder rubric for P01-L07. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson and are deliberate:

* Every expected value here is re-derived from `lesson.PROVIDER_FEED`,
  `lesson.DEPLOYER_FEED` and `lesson.INCIDENT_CANDIDATES` by code in THIS file, in a different
  shape from the lesson's own — plain Python lists, dicts and `statistics`, never the
  notebook's helpers. Nothing is a literal copied out of the notebook's output, and nothing
  calls the student's function to decide what the student's function should have returned.
* The fixtures are never mutated. A test that needs a different feed builds a small one of
  its own. A mutated fixture would leak into the tests that run after it and make a failure
  depend on rubric order.
* Several tests deliberately run a function on a HAND-BUILT feed rather than on the lesson's,
  because the lesson's feed has exactly one instance of each defect and an implementation can
  pass it by hard-coding. The "orders by seq not ts" test, the "min_run is a parameter" test,
  the "switch_off_after is a parameter" test, the "outside is strict" test, the median/mean
  separation test and every incident-branch test are all of that kind.
* The fixture is one-sided in three ways that hand-built cases have to cover, because a wrong
  implementation passes it otherwise. Every clock offset in the feed is POSITIVE, so a
  reversed subtraction is only caught by a case with a negative offset. Nothing in the feed
  ever breaches the LOWER control limit, so a monitor that only ever compares against the
  upper one scores full marks on it. And no window in the feed lands exactly ON a limit, so
  `>` and `>=` agree everywhere in it.
* `test_monitor_alerts_once_per_confirmed_run` is what stops an implementation that emits one
  alert per outside window, or that special-cases `min_run`, from scoring full marks: the run
  rule is exercised at min_run 1, 2 and 3 on a feed built for it.
* The incident tests never assert on `trigger`'s exact wording, only that it is non-empty,
  reasonably long and DIFFERENT between the three deadline branches — because a classifier
  whose explanation is the same string whatever it decided has not explained anything, and
  pinning the wording would grade prose rather than the decision.
"""
import importlib.util, math, os, statistics
from datetime import datetime, timedelta
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

FEED = lesson.PROVIDER_FEED
DEPLOYER = lesson.DEPLOYER_FEED
CANDIDATES = lesson.INCIDENT_CANDIDATES
WINDOW = lesson.WINDOW_SIZE
DEGRADED = tuple(lesson.DEGRADED_WINDOWS)


# --- independent re-derivations, in plain Python ----------------------------------------

def _dt(value: str) -> datetime:
    """Parse a timestamp here rather than through the lesson's helper."""
    return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)


def _stamp(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _limits(errors: int, n: int, window: int, k: float) -> dict:
    if n <= 0 or window <= 0:
        return {"centre": 0.0, "sigma": 0.0, "ucl": 1.0, "lcl": 0.0}
    centre = errors / n
    sigma = math.sqrt(centre * (1.0 - centre) / window)
    return {"centre": centre, "sigma": sigma,
            "ucl": min(1.0, centre + k * sigma), "lcl": max(0.0, centre - k * sigma)}


def _expected_windows(rows: list, window: int = WINDOW) -> list:
    """Re-derive the window table with a plain loop over a seq-sorted copy."""
    ordered = sorted(rows, key=lambda row: row["seq"])
    out = []
    for index in range(len(ordered) // window):
        chunk = ordered[index * window:(index + 1) * window]
        errors = len([row for row in chunk if row["wrong"]])
        out.append({"index": index, "errors": errors, "rate": errors / window,
                    "first_seq": chunk[0]["seq"], "last_seq": chunk[-1]["seq"],
                    "first_ts": chunk[0]["ts"], "last_ts": chunk[-1]["ts"]})
    return out


def _expected_alerts(window_rates: list, ucl: float, lcl: float, min_run: int) -> list:
    outside = [rate > ucl or rate < lcl for rate in window_rates]
    alerts, run = [], 0
    for index, flag in enumerate(outside):
        if not flag:
            run = 0
            continue
        run += 1
        if run == min_run:
            alerts.append(index - min_run + 1)
    return alerts


def _expected_costs(alerts: list, degraded, cost_false: int, cost_missed: int,
                    switch_off_after: int) -> dict:
    degraded = sorted(set(degraded))
    live, seen, muted = [], 0, None
    for alert in sorted(alerts):
        live.append(alert)
        if alert not in degraded:
            seen += 1
            if seen >= switch_off_after:
                muted = alert
                break
    false_alerts = len([a for a in live if a not in degraded])
    true_alerts = [a for a in live if a in degraded]
    first = min(true_alerts) if true_alerts else None
    missed = []
    for window in degraded:
        if first is None or window < first or (muted is not None and window > muted):
            missed.append(window)
    return {"false_alerts": false_alerts, "true_alerts": len(true_alerts),
            "switched_off_at": muted, "first_detection": first, "missed_windows": missed,
            "detection_delay_windows": None if first is None else first - degraded[0],
            "false_alert_cost": false_alerts * cost_false,
            "missed_cost": len(missed) * cost_missed,
            "total_cost": false_alerts * cost_false + len(missed) * cost_missed}


def _pair_deltas() -> list:
    held = {}
    for row in FEED:
        held[row["decision_id"]] = row["ts"]
    out = []
    for row in DEPLOYER:
        if row["decision_id"] in held:
            out.append((_dt(row["ts"]) - _dt(held[row["decision_id"]])).total_seconds())
    return out


def _expected_merge(offset: float) -> dict:
    held = {row["decision_id"] for row in FEED}
    rows = [{"ts": row["ts"], "decision_id": row["decision_id"], "wrong": bool(row["wrong"]),
             "source": "provider", "src_seq": row["seq"]} for row in FEED]
    overlap = added = 0
    for row in DEPLOYER:
        if row["decision_id"] in held:
            overlap += 1
        else:
            added += 1
            rows.append({"ts": _stamp(_dt(row["ts"]) - timedelta(seconds=offset)),
                         "decision_id": row["decision_id"], "wrong": bool(row["wrong"]),
                         "source": "deployer", "src_seq": None})
    rows = sorted(rows, key=lambda row: (row["ts"], row["decision_id"]))
    own = [row["src_seq"] for row in rows if row["source"] == "provider"]
    return {"rows": rows, "n_overlap": overlap, "n_added": added,
            "out_of_seq_rows": len([1 for got, want in zip(own, sorted(own)) if got != want]),
            "error_rate": len([r for r in rows if r["wrong"]]) / len(rows)}


def _expected_branch(candidate: dict) -> tuple:
    points = tuple(sorted(candidate["points"]))
    if not points:
        return points, None, None
    if candidate["widespread"] or "b" in points:
        return points, "73(3)", 2
    if candidate["death"]:
        return points, "73(4)", 10
    return points, "73(2)", 15


def _expected_log(candidates: list, as_of: datetime) -> dict:
    entries, not_serious = [], 0
    for candidate in candidates:
        points, paragraph, days = _expected_branch(candidate)
        if paragraph is None:
            not_serious += 1
            continue
        aware = _dt(candidate["aware_ts"])
        due = aware + timedelta(days=days)
        reported = candidate["reported_ts"]
        if reported is None:
            status = "open_in_time" if as_of <= due else "open_overdue"
        else:
            status = "reported_in_time" if _dt(reported) <= due else "reported_late"
        entries.append({
            "incident_id": candidate["incident_id"], "paragraph": paragraph,
            "deadline_days": days, "points": points, "due_ts": _stamp(due),
            "awareness_lag_hours": math.floor(
                (aware - _dt(candidate["occurred_ts"])).total_seconds() / 3600),
            "hours_remaining": math.floor((due - as_of).total_seconds() / 3600),
            "status": status})
    entries = sorted(entries, key=lambda entry: (entry["due_ts"], entry["incident_id"]))
    counts = {name: 0 for name in ("reported_in_time", "reported_late",
                                   "open_in_time", "open_overdue")}
    for entry in entries:
        counts[entry["status"]] += 1
    return {"entries": entries, "counts": counts, "n_not_serious": not_serious,
            "verdict": "fail" if counts["reported_late"] or counts["open_overdue"] else "pass"}


def _rows(spec: list, start_ts: str = "2026-01-01T00:00:00Z", step_h: int = 1) -> list:
    """Build a tiny feed: spec is a list of (seq, wrong) or (seq, wrong, ts)."""
    base = _dt(start_ts)
    out = []
    for position, item in enumerate(spec):
        seq, wrong = item[0], item[1]
        stamp = item[2] if len(item) > 2 else _stamp(base + timedelta(hours=step_h * position))
        out.append({"seq": seq, "decision_id": f"T-{seq:04d}", "ts": stamp,
                    "wrong": bool(wrong), "source": "provider"})
    return out


def _incident(iid, points, death, widespread, occurred_h, aware_h, reported_h=None):
    """A candidate built here, on hours rather than days, so branches can be separated."""
    base = lesson.AS_OF_DT
    return {"incident_id": iid, "points": points, "death": death, "widespread": widespread,
            "summary": "hand-built case",
            "occurred_ts": _stamp(base - timedelta(hours=occurred_h)),
            "aware_ts": _stamp(base - timedelta(hours=aware_h)),
            "reported_ts": None if reported_h is None
            else _stamp(base - timedelta(hours=reported_h))}


# --- exercise 1: control limits ----------------------------------------------------------

def test_control_limits_centre_and_sigma():
    for errors, n, window, k in ((48, 800, 50, 3.0), (7, 100, 20, 2.0), (250, 1000, 5, 1.0),
                                 (1, 3, 7, 2.5), (99, 100, 50, 3.0)):
        want = _limits(errors, n, window, k)
        got = lesson.control_limits(errors, n, window, k)
        assert abs(got["centre"] - want["centre"]) < 1e-12, (
            f"control_limits({errors}, {n}, {window}) centre is {got['centre']!r}, expected "
            f"{want['centre']!r} — it is baseline_errors / baseline_n"
        )
        assert abs(got["sigma"] - want["sigma"]) < 1e-12, (
            f"control_limits({errors}, {n}, {window}) sigma is {got['sigma']!r}, expected "
            f"{want['sigma']!r} — sqrt(centre * (1 - centre) / WINDOW SIZE)"
        )
    same_p = lesson.control_limits(48, 800, 800, 3.0)
    window_p = lesson.control_limits(48, 800, 50, 3.0)
    assert window_p["sigma"] > same_p["sigma"] * 3.9, (
        "the same baseline scored over windows of 50 must give a much wider sigma than over "
        "windows of 800; yours barely moves, so sigma is probably dividing by baseline_n"
    )


def test_control_limits_clamp_to_the_unit_interval():
    wide = lesson.control_limits(400, 800, 4, 3.0)
    assert wide["ucl"] == 1.0, (
        f"ucl is {wide['ucl']!r}; centre + k*sigma exceeds 1 here and a rate cannot. Cap at 1.0"
    )
    assert wide["lcl"] == 0.0, f"lcl is {wide['lcl']!r}; floor a negative lower limit at 0.0"
    inside = lesson.control_limits(500, 1000, 10000, 1.0)
    want = _limits(500, 1000, 10000, 1.0)
    assert abs(inside["ucl"] - want["ucl"]) < 1e-12 and abs(inside["lcl"] - want["lcl"]) < 1e-12, (
        "a limit that is already inside [0, 1] must NOT be clamped — clamping everything to "
        "0.0 and 1.0 makes the chart unable to alert at all"
    )
    assert inside["lcl"] > 0.0, "this lower limit is positive; it must survive the floor"


def test_control_limits_empty_baseline_is_permissive():
    for errors, n, window in ((0, 0, 50), (5, -1, 50), (5, 100, 0), (0, 0, 0)):
        got = lesson.control_limits(errors, n, window)
        assert (got["centre"], got["sigma"]) == (0.0, 0.0), (
            f"control_limits({errors}, {n}, {window}) gave centre/sigma "
            f"{(got['centre'], got['sigma'])}; with no baseline or no window there is nothing "
            "to compute, so both are 0.0 — and it must not raise ZeroDivisionError"
        )
        assert (got["ucl"], got["lcl"]) == (1.0, 0.0), (
            f"control_limits({errors}, {n}, {window}) gave limits "
            f"{(got['ucl'], got['lcl'])}; a baseline of nothing constrains nothing, so "
            "ucl 1.0 and lcl 0.0. Returning (0.0, 0.0) pages on every window ever plotted"
        )


def test_control_limits_k_is_a_parameter():
    previous = None
    for k in (0.5, 1.0, 2.0, 3.0, 4.0):
        got = lesson.control_limits(48, 800, 50, k)
        assert abs(got["k"] - k) < 1e-12, (
            f"the returned k is {got['k']!r} for an argument of {k} — report the k you used, "
            "not the default"
        )
        assert abs(got["ucl"] - _limits(48, 800, 50, k)["ucl"]) < 1e-12, (
            f"ucl at k={k} is {got['ucl']!r}; it is centre + k*sigma, so k must be USED and "
            "not only stored"
        )
        if previous is not None:
            assert got["ucl"] > previous, "a larger k must give a wider upper limit"
        previous = got["ucl"]
    assert lesson.control_limits(48, 800, 50)["window_size"] == 50, "echo the window size back"
    assert lesson.control_limits(48, 800, 50)["baseline_n"] == 800, "echo the baseline size"


# --- exercise 2: the windowed monitor ----------------------------------------------------

def test_monitor_windows_match_an_independent_derivation():
    limits = lesson.control_limits(lesson.BASELINE_ERRORS, lesson.BASELINE_N, WINDOW, 3.0)
    got = lesson.windowed_monitor(FEED, limits)
    want = _expected_windows(FEED)
    assert len(got["windows"]) == len(want), (
        f"{len(got['windows'])} windows, expected {len(want)}"
    )
    for mine, theirs in zip(want, got["windows"]):
        assert theirs["index"] == mine["index"], "windows are numbered from 0, in order"
        assert theirs["n"] == WINDOW, "every complete window holds window_size rows"
        assert theirs["errors"] == mine["errors"], (
            f"window {mine['index']} has {theirs['errors']} errors, expected {mine['errors']}"
            " — count rows whose 'wrong' is truthy, in SEQUENCE order"
        )
        assert abs(theirs["rate"] - mine["rate"]) < 1e-12, (
            f"window {mine['index']} rate is {theirs['rate']!r}, expected {mine['rate']!r}"
        )
        assert (theirs["first_seq"], theirs["last_seq"]) == (mine["first_seq"], mine["last_seq"]), (
            f"window {mine['index']} covers seq {mine['first_seq']}..{mine['last_seq']}; you "
            f"reported {theirs['first_seq']}..{theirs['last_seq']}"
        )
        assert (theirs["first_ts"], theirs["last_ts"]) == (mine["first_ts"], mine["last_ts"]), (
            f"window {mine['index']} timestamps are wrong — report the ts of the first and "
            "last row of the window as it stands in sequence order, so a breach can be dated"
        )


def test_monitor_orders_by_seq_not_timestamp():
    # Sequence order and timestamp order disagree: seq 2 carries the earliest clock reading,
    # exactly like a buffered batch or a clock that stepped back.
    rows = _rows([(0, False, "2026-01-01T05:00:00Z"), (1, False, "2026-01-01T06:00:00Z"),
                  (2, True, "2026-01-01T01:00:00Z"), (3, True, "2026-01-01T02:00:00Z")])
    got = lesson.windowed_monitor(rows, {"ucl": 0.4, "lcl": 0.0}, window_size=2, min_run=1)
    assert [w["errors"] for w in got["windows"]] == [0, 2], (
        f"window errors came out {[w['errors'] for w in got['windows']]}, expected [0, 2]. "
        "Sorting by 'ts' puts the two wrong decisions first and gives [2, 0]"
    )
    assert got["windows"][0]["first_seq"] == 0 and got["windows"][1]["first_seq"] == 2, (
        "the windows must be cut on seq, so window 1 starts at seq 2"
    )
    assert got["alerts"] == [1], (
        f"the breach is in the SECOND window; you reported alerts {got['alerts']}"
    )
    shuffled = [rows[2], rows[0], rows[3], rows[1]]
    assert lesson.windowed_monitor(shuffled, {"ucl": 0.4, "lcl": 0.0}, window_size=2,
                                   min_run=1)["alerts"] == [1], (
        "the answer must not depend on the order the rows arrive in — sort by seq first"
    )


def test_monitor_drops_the_incomplete_tail():
    limits = lesson.control_limits(lesson.BASELINE_ERRORS, lesson.BASELINE_N, WINDOW, 3.0)
    got = lesson.windowed_monitor(FEED, limits)
    assert got["n_windows"] == len(FEED) // WINDOW, (
        f"n_windows is {got['n_windows']}, expected {len(FEED) // WINDOW}"
    )
    assert got["dropped_rows"] == len(FEED) % WINDOW, (
        f"dropped_rows is {got['dropped_rows']}, expected {len(FEED) % WINDOW}"
    )
    assert all(w["n"] == WINDOW for w in got["windows"]), (
        "every reported window holds exactly window_size rows; a short final window is not a "
        "window and its rate is a statement about where you stopped counting"
    )
    tail = _rows([(0, False), (1, False), (2, True), (3, True), (4, True)])
    short = lesson.windowed_monitor(tail, {"ucl": 0.9, "lcl": 0.0}, window_size=2, min_run=1)
    assert short["n_windows"] == 2 and short["dropped_rows"] == 1, (
        f"five rows in windows of two is 2 windows and 1 dropped row; you reported "
        f"{short['n_windows']} and {short['dropped_rows']}"
    )
    empty = lesson.windowed_monitor([], {"ucl": 0.5, "lcl": 0.0}, window_size=2, min_run=1)
    assert empty["n_windows"] == 0 and empty["alerts"] == [] and empty["dropped_rows"] == 0, (
        "an empty feed is 0 windows, 0 dropped rows and no alerts, not a crash"
    )


def test_monitor_outside_is_strict_and_two_sided():
    exact = _rows([(0, True), (1, False), (2, True), (3, False)])
    on_limit = lesson.windowed_monitor(exact, {"ucl": 0.5, "lcl": 0.0},
                                       window_size=2, min_run=1)
    assert [w["outside"] for w in on_limit["windows"]] == [False, False], (
        "a window whose rate lands EXACTLY on the upper limit is inside it. Use `rate > ucl`, "
        "not `>=`, or a chart alerts on windows that are precisely where you expected them"
    )
    assert on_limit["alerts"] == [] and on_limit["verdict"] == "in_control", (
        "nothing is outside, so no alerts and the verdict is 'in_control'"
    )
    quiet = _rows([(0, False), (1, False), (2, False), (3, False)])
    low = lesson.windowed_monitor(quiet, {"ucl": 0.9, "lcl": 0.2}, window_size=2, min_run=1)
    assert [w["outside"] for w in low["windows"]] == [True, True], (
        "a rate BELOW the lower control limit is outside the limits too. A monitor that only "
        "compares against the upper one cannot see a metric that collapses — which on a "
        "quality feed usually means the metric stopped being computed at all"
    )
    assert low["alerts"] == [0] and low["verdict"] == "alert", (
        "a confirmed excursion below the lower limit is still an alert"
    )


def test_monitor_alerts_once_per_confirmed_run():
    limits = lesson.control_limits(lesson.BASELINE_ERRORS, lesson.BASELINE_N, WINDOW, 3.0)
    for k in (1.0, 2.0, 3.0, 5.0):
        limits = lesson.control_limits(lesson.BASELINE_ERRORS, lesson.BASELINE_N, WINDOW, k)
        got = lesson.windowed_monitor(FEED, limits)
        want = _expected_alerts([w["rate"] for w in _expected_windows(FEED)],
                                limits["ucl"], limits["lcl"], lesson.MIN_RUN)
        assert got["alerts"] == want, (
            f"at k={k} alerts came out {got['alerts']}, expected {want}"
        )
    # out, out, out, in, out, out  -> two runs of >= 2, and one of them is three long
    pattern = [(0, True), (1, True), (2, True), (3, True), (4, True), (5, True),
               (6, False), (7, False), (8, True), (9, True), (10, True), (11, True)]
    rows = _rows(pattern)
    for min_run, want in ((1, [0, 4]), (2, [0, 4]), (3, [0]), (4, [])):
        got = lesson.windowed_monitor(rows, {"ucl": 0.4, "lcl": 0.0},
                                      window_size=2, min_run=min_run)
        assert got["alerts"] == want, (
            f"with min_run={min_run} the alerts are {want}; you reported {got['alerts']}. One "
            "alert per maximal run, at the run's FIRST window, and min_run is an argument "
            "rather than a constant"
        )


# --- exercise 3: the two costs -----------------------------------------------------------

def test_costs_match_an_independent_derivation():
    for k in (1.0, 1.5, 2.0, 2.5, 3.0, 5.0):
        limits = lesson.control_limits(lesson.BASELINE_ERRORS, lesson.BASELINE_N, WINDOW, k)
        report = lesson.windowed_monitor(FEED, limits)
        got = lesson.monitor_costs(report, DEGRADED)
        want = _expected_costs(report["alerts"], DEGRADED, lesson.COST_PER_FALSE_ALERT,
                               lesson.COST_PER_MISSED_WINDOW, lesson.SWITCH_OFF_AFTER)
        for key in ("false_alerts", "true_alerts", "switched_off_at", "first_detection",
                    "missed_windows", "detection_delay_windows", "false_alert_cost",
                    "missed_cost", "total_cost"):
            assert got[key] == want[key], (
                f"at k={k}, {key} is {got[key]!r}, expected {want[key]!r} (alerts were "
                f"{report['alerts']}, degraded windows {list(DEGRADED)})"
            )


def test_costs_mute_after_the_threshold():
    degraded = {10, 11, 12}
    report = {"alerts": [1, 3, 5, 7, 10]}
    got = lesson.monitor_costs(report, degraded, switch_off_after=3)
    assert got["switched_off_at"] == 5, (
        f"switched_off_at is {got['switched_off_at']!r}; the third FALSE alert is at window 5"
    )
    assert got["false_alerts"] == 3, (
        f"false_alerts is {got['false_alerts']}; the alert at 7 comes after the monitor was "
        "muted and cannot have paged anyone, so it is not counted"
    )
    assert got["true_alerts"] == 0 and got["first_detection"] is None, (
        "the true alert at window 10 also comes after the mute, so there is no detection"
    )
    assert got["missed_windows"] == [10, 11, 12], (
        f"missed_windows is {got['missed_windows']}; a muted monitor misses everything that "
        "comes after it was muted, which is the whole point of the rule"
    )
    patient = lesson.monitor_costs(report, degraded, switch_off_after=99)
    assert patient["switched_off_at"] is None and patient["false_alerts"] == 4, (
        "switch_off_after is a PARAMETER: raise it and the monitor stays on, so all four "
        "false alerts count and the true one at 10 detects"
    )
    assert patient["first_detection"] == 10 and patient["missed_windows"] == [], (
        "with the monitor still on, window 10 detects and nothing is missed"
    )
    twitchy = lesson.monitor_costs(report, degraded, switch_off_after=1)
    assert twitchy["switched_off_at"] == 1 and twitchy["false_alerts"] == 1, (
        "with switch_off_after=1 the very first false page mutes it"
    )


def test_costs_missed_windows_before_detection_and_after_mute():
    got = lesson.monitor_costs({"alerts": [12]}, {10, 11, 12, 13}, switch_off_after=3)
    assert got["missed_windows"] == [10, 11], (
        f"missed_windows is {got['missed_windows']}; windows 10 and 11 degraded before the "
        "alert, and 13 is covered by it"
    )
    assert got["detection_delay_windows"] == 2, (
        f"the delay is 12 - 10 = 2 windows; you reported {got['detection_delay_windows']}"
    )
    assert got["false_alert_cost"] == 0 and got["missed_cost"] == 2 * lesson.COST_PER_MISSED_WINDOW, (
        "the two cost lines are reported separately as well as summed"
    )
    priced = lesson.monitor_costs({"alerts": [12]}, {10, 11, 12, 13},
                                  cost_false=7, cost_missed=11)
    assert priced["total_cost"] == 22 and priced["missed_cost"] == 22, (
        f"total_cost is {priced['total_cost']}; the two prices are PARAMETERS, so 2 missed "
        "windows at 11 each is 22"
    )
    late_mute = lesson.monitor_costs({"alerts": [2, 4, 5, 6]}, {2, 3, 8, 9},
                                     switch_off_after=3)
    assert late_mute["switched_off_at"] == 6 and late_mute["first_detection"] == 2, (
        f"switched_off_at/first_detection came out {late_mute['switched_off_at']!r}/"
        f"{late_mute['first_detection']!r}; the alert at 2 is true and the third false one, "
        "at 6, mutes the monitor"
    )
    assert late_mute["missed_windows"] == [8, 9], (
        f"missed_windows is {late_mute['missed_windows']}, expected [8, 9]. The monitor "
        "detected at window 2 and was then muted at window 6; windows 8 and 9 degraded after "
        "that and nobody saw them. A detection earlier in the quarter does not cover a window "
        "a muted monitor never scored"
    )
    unsorted = lesson.monitor_costs({"alerts": [13, 9]}, {10, 11, 12, 13}, switch_off_after=3)
    assert unsorted["first_detection"] == 13 and unsorted["missed_windows"] == [10, 11, 12], (
        "walk the alerts in ascending order whatever order they arrive in: the false one at 9 "
        "comes first, and detection is at 13"
    )


def test_costs_blind_and_perfect_monitors():
    blind = lesson.monitor_costs({"alerts": []}, {4, 5})
    assert blind["missed_windows"] == [4, 5] and blind["first_detection"] is None, (
        "a monitor that never alerts misses every degraded window"
    )
    assert blind["detection_delay_windows"] is None, (
        "no detection means no delay: None, not 0, and not the number of windows"
    )
    assert blind["false_alerts"] == 0 and blind["switched_off_at"] is None, (
        "it raised nothing, so it cannot have raised anything false or been muted"
    )
    assert blind["total_cost"] == 2 * lesson.COST_PER_MISSED_WINDOW, (
        "the bill for seeing nothing is every missed window; a cost model that prices only "
        "false alerts says a monitor that never fires is free"
    )
    perfect = lesson.monitor_costs({"alerts": [4]}, {4, 5})
    assert perfect["total_cost"] == 0 and perfect["detection_delay_windows"] == 0, (
        "an alert on the first degraded window costs nothing and has no delay; a verdict that "
        "can never come back clean is not a verdict"
    )
    assert perfect["true_alerts"] == 1 and perfect["missed_windows"] == []


# --- exercise 4: the deployer's clock ----------------------------------------------------

def test_offset_is_the_median_not_the_mean():
    got = lesson.estimate_clock_offset(FEED, DEPLOYER)
    deltas = _pair_deltas()
    assert got["n_pairs"] == len(deltas), (
        f"n_pairs is {got['n_pairs']}, expected {len(deltas)} — pair on decision_id"
    )
    assert abs(got["offset_seconds"] - statistics.median(deltas)) < 1e-9, (
        f"offset_seconds is {got['offset_seconds']!r}, expected the median "
        f"{statistics.median(deltas)!r}"
    )
    assert abs(got["mean_seconds"] - sum(deltas) / len(deltas)) < 1e-9, (
        f"mean_seconds is {got['mean_seconds']!r}, expected {sum(deltas) / len(deltas)!r} — "
        "report the mean as well, so the reader can see what it would have cost"
    )
    # A hand-built pair set where the mean and the median disagree, and the deployer's clock
    # is BEHIND the provider's, so a reversed subtraction changes the sign.
    provider = [{"decision_id": f"D-{i}", "ts": _stamp(_dt("2026-01-01T00:00:00Z")
                                                       + timedelta(hours=i))}
                for i in range(5)]
    deployer = [{"decision_id": f"D-{i}",
                 "ts": _stamp(_dt("2026-01-01T00:00:00Z") + timedelta(hours=i, seconds=-60))}
                for i in range(4)]
    deployer.append({"decision_id": "D-4",
                     "ts": _stamp(_dt("2026-01-01T04:00:00Z") + timedelta(seconds=100000))})
    small = lesson.estimate_clock_offset(provider, deployer)
    assert small["offset_seconds"] == -60.0, (
        f"offset_seconds is {small['offset_seconds']!r}, expected -60.0. The offset is "
        "deployer minus provider and can be NEGATIVE; the median ignores the one row that is "
        "a day out, and the mean does not"
    )
    assert small["mean_seconds"] > 19000, (
        f"mean_seconds is {small['mean_seconds']!r}; one outlier drags it above 19000 here, "
        "which is the whole reason the estimate is a median"
    )


def test_offset_spread_is_the_mad_not_the_sd():
    got = lesson.estimate_clock_offset(FEED, DEPLOYER)
    deltas = _pair_deltas()
    median = statistics.median(deltas)
    want = statistics.median([abs(delta - median) for delta in deltas])
    assert abs(got["spread_seconds"] - want) < 1e-9, (
        f"spread_seconds is {got['spread_seconds']!r}, expected {want!r} — the MEDIAN of the "
        "absolute deviations from the median"
    )
    sd = statistics.pstdev(deltas)
    assert abs(got["spread_seconds"] - sd) > 1.0, (
        f"spread_seconds equals the standard deviation ({sd:.1f}). The same handful of rows "
        "that broke the mean breaks the standard deviation; a robust offset deserves a robust "
        "spread beside it"
    )
    tight = [{"decision_id": "D-0", "ts": "2026-01-01T00:00:00Z"},
             {"decision_id": "D-1", "ts": "2026-01-01T01:00:00Z"},
             {"decision_id": "D-2", "ts": "2026-01-01T02:00:00Z"}]
    same = [{"decision_id": row["decision_id"],
             "ts": _stamp(_dt(row["ts"]) + timedelta(seconds=30))} for row in tight]
    perfect = lesson.estimate_clock_offset(tight, same)
    assert perfect["spread_seconds"] == 0.0 and perfect["offset_seconds"] == 30.0, (
        "three pairs all exactly 30s apart have an offset of 30 and a spread of 0"
    )


def test_offset_counts_outliers_against_the_tolerance_parameter():
    got = lesson.estimate_clock_offset(FEED, DEPLOYER)
    deltas = _pair_deltas()
    median = statistics.median(deltas)
    want = len([d for d in deltas if abs(d - median) > lesson.OUTLIER_TOLERANCE_SECONDS])
    assert got["outliers"] == want, (
        f"outliers is {got['outliers']}, expected {want} — pairs further than "
        f"{lesson.OUTLIER_TOLERANCE_SECONDS}s from the median"
    )
    loose = lesson.estimate_clock_offset(FEED, DEPLOYER, tolerance=10 ** 9)
    assert loose["outliers"] == 0, (
        "tolerance is a PARAMETER: widen it far enough and nothing is an outlier. Yours "
        "ignores the argument"
    )
    tight = lesson.estimate_clock_offset(FEED, DEPLOYER, tolerance=0.0)
    assert tight["outliers"] > want, (
        "and narrow it to zero and every pair that is not exactly on the median counts"
    )
    # The BOUNDARY. At the default tolerance no pair in this feed sits exactly on it, so
    # `>` and `>=` agree there and the loose/tight cases above are satisfied by both. Pick a
    # tolerance that IS one of the deviations the feed actually contains and they separate.
    edge_tolerance = 4.0
    strict = len([d for d in deltas if abs(d - median) > edge_tolerance])
    non_strict = len([d for d in deltas if abs(d - median) >= edge_tolerance])
    on_the_line = lesson.estimate_clock_offset(FEED, DEPLOYER, tolerance=edge_tolerance)
    assert on_the_line["outliers"] == strict, (
        f"with tolerance={edge_tolerance} outliers is {on_the_line['outliers']}, expected "
        f"{strict}. A pair sitting EXACTLY on the tolerance is within it, not outside it: "
        f"the documented test is `> tolerance`, not `>= tolerance`. On this feed `>=` calls "
        f"{non_strict} pairs suspect where {strict} are, because {non_strict - strict} pairs "
        f"are exactly {edge_tolerance:.0f} seconds from the median"
    )


def test_offset_handles_no_pairs():
    got = lesson.estimate_clock_offset(FEED, [{"decision_id": "NOPE", "ts": "2026-01-01T00:00:00Z"}])
    assert got["n_pairs"] == 0, "nothing matched, so n_pairs is 0"
    for key in ("offset_seconds", "mean_seconds", "spread_seconds"):
        assert got[key] == 0.0, (
            f"{key} is {got[key]!r} with no pairs at all; return 0.0 rather than raising on "
            "an empty median"
        )
    assert got["outliers"] == 0, "no pairs, no outliers"
    empty = lesson.estimate_clock_offset([], [])
    assert empty["n_pairs"] == 0 and empty["offset_seconds"] == 0.0, (
        "two empty feeds must not raise either"
    )


# --- exercise 5: the merge ---------------------------------------------------------------

def test_merge_drops_the_overlap_and_counts_it():
    want = _expected_merge(7620.0)
    got = lesson.merge_deployer_feed(FEED, DEPLOYER, 7620.0)
    assert got["n_overlap"] == want["n_overlap"], (
        f"n_overlap is {got['n_overlap']}, expected {want['n_overlap']}"
    )
    assert got["n_added"] == want["n_added"], (
        f"n_added is {got['n_added']}, expected {want['n_added']}"
    )
    assert len(got["merged"]) == len(want["rows"]), (
        f"the merged feed has {len(got['merged'])} rows, expected {len(want['rows'])}. "
        f"Appending the whole deployer feed adds {want['n_overlap']} rows you already held "
        "and dilutes every rate you compute from it"
    )
    assert got["n_provider"] == len(FEED) and got["n_deployer"] == len(DEPLOYER), (
        "report the two input sizes as given, so the arithmetic can be checked from the face "
        "of the artefact"
    )
    ids = [row["decision_id"] for row in got["merged"]]
    assert len(set(ids)) == len(ids), (
        "a decision_id appears once in the merged feed; a duplicate means an overlap row was "
        "appended rather than dropped"
    )
    sources = [row["source"] for row in got["merged"]]
    assert sources.count("deployer") == want["n_added"], (
        f"{sources.count('deployer')} rows are tagged 'deployer', expected {want['n_added']}"
    )


def test_merge_corrects_the_added_timestamps_by_subtracting():
    want = _expected_merge(7620.0)
    got = lesson.merge_deployer_feed(FEED, DEPLOYER, 7620.0)
    mine = {row["decision_id"]: row["ts"] for row in want["rows"]}
    for row in got["merged"]:
        assert row["ts"] == mine[row["decision_id"]], (
            f"{row['decision_id']} is at {row['ts']}, expected {mine[row['decision_id']]}. A "
            "provider row keeps its own timestamp; an added deployer row has the offset "
            "SUBTRACTED, because a clock that runs ahead must be wound back"
        )
    other = lesson.merge_deployer_feed(FEED, DEPLOYER, 0.0)
    theirs = _expected_merge(0.0)
    added_other = sorted(row["ts"] for row in other["merged"] if row["source"] == "deployer")
    added_want = sorted(row["ts"] for row in theirs["rows"] if row["source"] == "deployer")
    assert added_other == added_want, (
        "offset_seconds is a PARAMETER: pass 0.0 and the deployer rows keep their own "
        "timestamps. Yours ignores the argument, or hard-codes the fixture's offset"
    )
    assert abs(other["offset_seconds"] - 0.0) < 1e-12, "echo the offset that was used"


def test_merge_order_renumber_and_row_shape():
    want = _expected_merge(7620.0)
    got = lesson.merge_deployer_feed(FEED, DEPLOYER, 7620.0)
    assert [row["decision_id"] for row in got["merged"]] == \
           [row["decision_id"] for row in want["rows"]], (
        "the merged rows are sorted by (ts, decision_id) — the timestamp first, and the "
        "decision id only to break ties, so two feeds merged on the same second agree"
    )
    assert [row["seq"] for row in got["merged"]] == list(range(len(want["rows"]))), (
        "renumber seq from 0 across the merged feed; keeping the provider's own numbers "
        "leaves gaps and collisions where the deployer's rows were inserted"
    )
    for mine, theirs in zip(want["rows"], got["merged"]):
        assert set(theirs) == {"seq", "ts", "decision_id", "wrong", "source", "src_seq"}, (
            f"a merged row has keys {sorted(theirs)}; return exactly the six documented names"
        )
        assert theirs["source"] == mine["source"], (
            f"{mine['decision_id']} is tagged {theirs['source']!r}, expected "
            f"{mine['source']!r} — an artefact that cannot say which party a row came from "
            "cannot be argued with"
        )
        assert theirs["src_seq"] == mine["src_seq"], (
            f"{mine['decision_id']} has src_seq {theirs['src_seq']!r}, expected "
            f"{mine['src_seq']!r}: the provider's own number, or None for an added row"
        )
        assert bool(theirs["wrong"]) == mine["wrong"], "carry each row's outcome through"


def test_merge_out_of_sequence_and_error_rate():
    want = _expected_merge(7620.0)
    got = lesson.merge_deployer_feed(FEED, DEPLOYER, 7620.0)
    assert got["out_of_seq_rows"] == want["out_of_seq_rows"], (
        f"out_of_seq_rows is {got['out_of_seq_rows']}, expected {want['out_of_seq_rows']} — "
        "compare the provider rows in merged order against the same rows in src_seq order, "
        "position by position, and count the positions that disagree"
    )
    assert abs(got["merged_error_rate"] - want["error_rate"]) < 1e-12, (
        f"merged_error_rate is {got['merged_error_rate']!r}, expected {want['error_rate']!r} "
        "— over ALL merged rows, not over the added ones alone"
    )
    same_moment = "2026-03-01T00:00:00Z"
    provider_tie = [{"seq": 0, "decision_id": "D-9", "ts": same_moment, "wrong": False},
                    {"seq": 1, "decision_id": "D-1", "ts": same_moment, "wrong": False}]
    tied = lesson.merge_deployer_feed(
        provider_tie, [{"decision_id": "P-5", "ts": same_moment, "wrong": True}], 0.0)
    assert [row["decision_id"] for row in tied["merged"]] == ["D-1", "D-9", "P-5"], (
        f"three rows sharing one timestamp came out "
        f"{[row['decision_id'] for row in tied['merged']]}, expected ['D-1', 'D-9', 'P-5']. "
        "Sorting by ts alone leaves them in whatever order they arrived in, so two runs of "
        "the same merge can disagree — break the tie on decision_id"
    )
    tidy_provider = [{"seq": i, "decision_id": f"D-{i}", "wrong": i == 0,
                      "ts": _stamp(_dt("2026-01-01T00:00:00Z") + timedelta(hours=i))}
                     for i in range(4)]
    clean = lesson.merge_deployer_feed(tidy_provider, [], 0.0)
    assert clean["out_of_seq_rows"] == 0, (
        f"out_of_seq_rows is {clean['out_of_seq_rows']} on a feed whose timestamps already "
        "march with its sequence numbers; it must be 0. A count that is never zero is not a "
        "measurement"
    )
    assert clean["n_added"] == 0 and clean["n_overlap"] == 0 and len(clean["merged"]) == 4, (
        "an empty deployer feed adds nothing and overlaps nothing"
    )
    assert abs(clean["merged_error_rate"] - 0.25) < 1e-12, "one wrong row in four is 0.25"
    # The same tidy feed as a LATER SLICE of a continuing log — seq 500..503 rather than
    # 0..3, which is what a quarter's extract of the Article 12 log actually looks like.
    # Comparing each src_seq against its POSITION in the merged list agrees with the
    # documented comparison only while a feed happens to begin at seq 0, so every case
    # above passes it. This one does not.
    later_slice = [{"seq": 500 + i, "decision_id": f"D-{500 + i}", "wrong": False,
                    "ts": _stamp(_dt("2026-01-01T00:00:00Z") + timedelta(hours=i))}
                   for i in range(4)]
    sliced = lesson.merge_deployer_feed(later_slice, [], 0.0)
    assert sliced["out_of_seq_rows"] == 0, (
        f"out_of_seq_rows is {sliced['out_of_seq_rows']} on a quarter's slice of a continuing "
        "log whose sequence numbers start at 500 and whose clock already marches with them. "
        "Nothing moved, so it must be 0 — comparing src_seq against the row's position in "
        "the merged list reports every row out of sequence on a feed that is in perfect "
        "order. Compare the provider rows in merged order against the SAME rows sorted by "
        "src_seq, position by position"
    )
    shuffled_slice = [
        {"seq": 500, "decision_id": "D-500", "wrong": False, "ts": "2026-01-01T02:00:00Z"},
        {"seq": 501, "decision_id": "D-501", "wrong": False, "ts": "2026-01-01T01:00:00Z"},
        {"seq": 502, "decision_id": "D-502", "wrong": False, "ts": "2026-01-01T03:00:00Z"}]
    moved = lesson.merge_deployer_feed(shuffled_slice, [], 0.0)
    assert moved["out_of_seq_rows"] == 2, (
        f"out_of_seq_rows is {moved['out_of_seq_rows']}, expected 2. On the clock seq 501 "
        "arrives before seq 500 and seq 502 stays where it was, so two of the three "
        "positions disagree with sequence order and the third agrees"
    )


# --- exercise 6: the Article 73 classifier -----------------------------------------------

def test_classify_not_serious_has_no_clock():
    for points in ((), []):
        got = lesson.classify_incident({"incident_id": "X", "points": points,
                                        "death": False, "widespread": False})
        assert got["serious"] is False, "no Article 3(49) point means not a serious incident"
        assert got["paragraph"] is None and got["deadline_days"] is None, (
            f"a candidate that is not serious has no paragraph and no deadline; you returned "
            f"{got['paragraph']!r} and {got['deadline_days']!r}"
        )
    still_none = lesson.classify_incident({"incident_id": "X", "points": (),
                                           "death": True, "widespread": True})
    assert still_none["serious"] is False and still_none["deadline_days"] is None, (
        "the not-serious test comes FIRST: an outage that killed nobody and disrupted nothing "
        "is not made reportable by a flag on the record"
    )
    assert isinstance(still_none["trigger"], str) and len(still_none["trigger"]) > 10, (
        "say why it is not serious, in words a reviewer can disagree with"
    )


def test_classify_two_day_trigger_is_widespread_or_point_b():
    two_day = lesson.classify_incident({"incident_id": "B", "points": ("b",),
                                        "death": False, "widespread": False})
    assert (two_day["paragraph"], two_day["deadline_days"]) == ("73(3)", 2), (
        f"a point (b) critical-infrastructure disruption is 73(3) and two days; you returned "
        f"{two_day['paragraph']!r} and {two_day['deadline_days']!r}"
    )
    widespread = lesson.classify_incident({"incident_id": "W", "points": ("c",),
                                           "death": False, "widespread": True})
    assert (widespread["paragraph"], widespread["deadline_days"]) == ("73(3)", 2), (
        "Article 73(3) names a WIDESPREAD INFRINGEMENT separately from the Article 3(49) "
        "points, so a widespread point (c) case is still two days"
    )
    rights = lesson.classify_incident({"incident_id": "C", "points": ("c",),
                                       "death": False, "widespread": False})
    assert (rights["paragraph"], rights["deadline_days"]) == ("73(2)", 15), (
        f"a point (c) fundamental-rights infringement that is NOT widespread runs on 73(2)'s "
        f"fifteen days; you returned {rights['deadline_days']!r}. Giving (c) the two-day clock "
        "is the most common error in this area, and it comes with giving (b) a fortnight"
    )
    for points in (("d",), ("c", "d")):
        got = lesson.classify_incident({"incident_id": "D", "points": points,
                                        "death": False, "widespread": False})
        assert got["deadline_days"] == 15, (
            f"points {points} is serious and not urgent under 73(3) or 73(4): fifteen days"
        )


def test_classify_death_is_ten_days_and_only_a_death():
    death = lesson.classify_incident({"incident_id": "A", "points": ("a",),
                                      "death": True, "widespread": False})
    assert (death["paragraph"], death["deadline_days"]) == ("73(4)", 10), (
        f"a death with no critical-infrastructure element is 73(4) and ten days; you returned "
        f"{death['paragraph']!r} and {death['deadline_days']!r}"
    )
    health = lesson.classify_incident({"incident_id": "H", "points": ("a",),
                                       "death": False, "widespread": False})
    assert (health["paragraph"], health["deadline_days"]) == ("73(2)", 15), (
        f"Article 3(49)(a) covers death OR serious harm to health, and 73(4)'s ten days is "
        f"for the DEATH of a person. A non-fatal point (a) harm is fifteen days; you returned "
        f"{health['deadline_days']!r}"
    )
    triggers = {lesson.classify_incident({"incident_id": "X", "points": p, "death": d,
                                          "widespread": w})["trigger"]
                for p, d, w in ((("b",), False, False), (("a",), True, False),
                                (("c",), False, False))}
    assert len(triggers) == 3, (
        f"the three deadline branches gave {len(triggers)} distinct trigger strings. A "
        "classifier whose explanation is the same sentence whatever it decided has not "
        "explained anything"
    )
    assert all(len(t) > 10 for t in triggers), "each trigger names what decided the branch"


def test_classify_overlap_takes_the_shorter_deadline():
    both = lesson.classify_incident({"incident_id": "AB", "points": ("a", "b"),
                                     "death": True, "widespread": False})
    assert (both["paragraph"], both["deadline_days"]) == ("73(3)", 2), (
        f"an incident that is both a death and a critical-infrastructure disruption takes the "
        f"SHORTER deadline in this lesson: 73(3), two days. You returned "
        f"{both['paragraph']!r} and {both['deadline_days']!r}, which resolves the overlap by "
        "choosing the later date"
    )
    for candidate in CANDIDATES:
        want_points, want_paragraph, want_days = _expected_branch(candidate)
        got = lesson.classify_incident(candidate)
        assert got["points"] == want_points, (
            f"{candidate['incident_id']} points came back {got['points']!r}, expected "
            f"{want_points!r} — a sorted tuple of letters"
        )
        assert (got["paragraph"], got["deadline_days"]) == (want_paragraph, want_days), (
            f"{candidate['incident_id']} classified as {got['paragraph']!r}/"
            f"{got['deadline_days']!r}, expected {want_paragraph!r}/{want_days!r}"
        )
        assert got["incident_id"] == candidate["incident_id"], "carry the id through"
    unsorted = lesson.classify_incident({"incident_id": "S", "points": ("d", "c"),
                                         "death": False, "widespread": False})
    assert unsorted["points"] == ("c", "d"), (
        f"points came back {unsorted['points']!r}; sort them, so two records of the same "
        "incident compare equal"
    )


# --- exercise 7: the incident log --------------------------------------------------------

def test_log_clock_starts_at_awareness():
    want = _expected_log(CANDIDATES, lesson.AS_OF_DT)
    got = lesson.incident_log(CANDIDATES)
    mine = {entry["incident_id"]: entry for entry in want["entries"]}
    for entry in got["entries"]:
        expected = mine[entry["incident_id"]]
        assert entry["due_ts"] == expected["due_ts"], (
            f"{entry['incident_id']} is due at {entry['due_ts']}, expected "
            f"{expected['due_ts']} — awareness plus the deadline, not occurrence plus it"
        )
    # A case built so that the two clocks give OPPOSITE verdicts: aware late, reported
    # promptly after awareness but long after the incident itself.
    late_news = [_incident("L-1", ("c",), False, False, occurred_h=24 * 40,
                           aware_h=24 * 5, reported_h=24 * 2)]
    log = lesson.incident_log(late_news)
    assert log["entries"][0]["status"] == "reported_in_time", (
        f"L-1 was reported three days after awareness against a fifteen-day clock, so it is "
        f"in time; you returned {log['entries'][0]['status']!r}. From the OCCURRENCE it would "
        "look twenty-five days late, which is the mistake this case exists to catch"
    )
    assert log["entries"][0]["awareness_lag_hours"] == 24 * 35, (
        f"awareness_lag_hours is {log['entries'][0]['awareness_lag_hours']}, expected "
        f"{24 * 35} — whole hours from occurrence to awareness"
    )


def test_log_status_ladder():
    want = _expected_log(CANDIDATES, lesson.AS_OF_DT)
    got = lesson.incident_log(CANDIDATES)
    mine = {entry["incident_id"]: entry["status"] for entry in want["entries"]}
    for entry in got["entries"]:
        assert entry["status"] == mine[entry["incident_id"]], (
            f"{entry['incident_id']} is {entry['status']!r}, expected "
            f"{mine[entry['incident_id']]!r}"
        )
    # The boundary: reported exactly ON the deadline is in time, and open exactly on it is too.
    on_time = [_incident("E-1", ("c",), False, False, occurred_h=24 * 20, aware_h=24 * 20,
                         reported_h=24 * 5)]
    assert lesson.incident_log(on_time)["entries"][0]["status"] == "reported_in_time", (
        "a report filed at the very moment the deadline falls is in time, not late — the "
        "Regulation says 'not later than'"
    )
    open_on_the_line = [_incident("E-2", ("c",), False, False, occurred_h=24 * 20,
                                  aware_h=24 * 15)]
    assert lesson.incident_log(open_on_the_line)["entries"][0]["status"] == "open_in_time", (
        "an open incident whose deadline falls exactly now has not yet been missed"
    )
    just_over = [_incident("E-3", ("c",), False, False, occurred_h=24 * 20,
                           aware_h=24 * 15 + 1)]
    assert lesson.incident_log(just_over)["entries"][0]["status"] == "open_overdue", (
        "one hour past the deadline is overdue; the comparison is against as_of, which is a "
        "parameter of the log and not today's date"
    )


def test_log_order_and_tie_break():
    want = _expected_log(CANDIDATES, lesson.AS_OF_DT)
    got = lesson.incident_log(CANDIDATES)
    assert [e["incident_id"] for e in got["entries"]] == \
           [e["incident_id"] for e in want["entries"]], (
        f"entries came out {[e['incident_id'] for e in got['entries']]}, expected "
        f"{[e['incident_id'] for e in want['entries']]} — sorted by due_ts, soonest first. "
        "Sorting by occurrence, by awareness or by id puts the thing that is due tomorrow "
        "below the thing that happened first"
    )
    due = [e["due_ts"] for e in got["entries"]]
    assert due == sorted(due), "due_ts ascending, without exception"
    tie = [_incident("Z-9", ("c",), False, False, occurred_h=400, aware_h=48),
           _incident("A-1", ("c",), False, False, occurred_h=400, aware_h=48)]
    order = [e["incident_id"] for e in lesson.incident_log(tie)["entries"]]
    assert order == ["A-1", "Z-9"], (
        f"two incidents due at the same moment came out {order}; break the tie by "
        "incident_id, so the log is stable across runs"
    )


def test_log_hours_are_floored_and_signed():
    want = _expected_log(CANDIDATES, lesson.AS_OF_DT)
    got = lesson.incident_log(CANDIDATES)
    mine = {e["incident_id"]: e for e in want["entries"]}
    for entry in got["entries"]:
        expected = mine[entry["incident_id"]]
        assert entry["hours_remaining"] == expected["hours_remaining"], (
            f"{entry['incident_id']} hours_remaining is {entry['hours_remaining']}, expected "
            f"{expected['hours_remaining']}"
        )
        assert entry["awareness_lag_hours"] == expected["awareness_lag_hours"], (
            f"{entry['incident_id']} awareness_lag_hours is {entry['awareness_lag_hours']}, "
            f"expected {expected['awareness_lag_hours']}"
        )
    overdue = [e for e in got["entries"] if e["status"].endswith("overdue")]
    assert overdue and all(e["hours_remaining"] < 0 for e in overdue), (
        "an overdue entry has NEGATIVE hours remaining; reporting the absolute value hides "
        "which side of the deadline you are on"
    )
    ragged = [_incident("R-1", ("b",), False, False, occurred_h=100, aware_h=49)]
    entry = lesson.incident_log(ragged)["entries"][0]
    assert entry["hours_remaining"] == -1, (
        f"hours_remaining is {entry['hours_remaining']}, expected -1: the deadline fell "
        "exactly one hour ago"
    )
    part_hour = [{"incident_id": "H-1", "points": ("b",), "death": False,
                  "widespread": False, "summary": "hand-built case",
                  "occurred_ts": _stamp(lesson.AS_OF_DT - timedelta(hours=100)),
                  "aware_ts": _stamp(lesson.AS_OF_DT - timedelta(hours=48, minutes=30)),
                  "reported_ts": None}]
    part = lesson.incident_log(part_hour)["entries"][0]
    assert part["hours_remaining"] == -1, (
        f"hours_remaining is {part['hours_remaining']}, expected -1. This deadline fell "
        "THIRTY MINUTES ago: int() and round() both give 0, which an inspector reads as 'due "
        "now' rather than 'already missed'. Truncate towards minus infinity"
    )
    assert part["status"] == "open_overdue", (
        "half an hour past the deadline is overdue, whatever the rounding says"
    )
    lag_case = [{"incident_id": "H-2", "points": ("c",), "death": False,
                 "widespread": False, "summary": "hand-built case",
                 "occurred_ts": _stamp(lesson.AS_OF_DT - timedelta(hours=80, minutes=45)),
                 "aware_ts": _stamp(lesson.AS_OF_DT - timedelta(hours=30)),
                 "reported_ts": None}]
    lag = lesson.incident_log(lag_case)["entries"][0]
    assert lag["awareness_lag_hours"] == 50, (
        f"awareness_lag_hours is {lag['awareness_lag_hours']}, expected 50 — fifty hours and "
        "forty-five minutes, reported in whole hours"
    )


def test_log_counts_verdict_and_excluded_candidates():
    want = _expected_log(CANDIDATES, lesson.AS_OF_DT)
    got = lesson.incident_log(CANDIDATES)
    assert got["n_not_serious"] == want["n_not_serious"], (
        f"n_not_serious is {got['n_not_serious']}, expected {want['n_not_serious']}"
    )
    assert len(got["entries"]) == len(want["entries"]), (
        f"{len(got['entries'])} entries, expected {len(want['entries'])} — a candidate that "
        "is not a serious incident is counted, not logged with a null deadline"
    )
    assert got["counts"] == want["counts"], (
        f"counts came out {got['counts']}, expected {want['counts']}"
    )
    assert set(got["counts"]) == {"reported_in_time", "reported_late", "open_in_time",
                                  "open_overdue"}, (
        f"counts keys are {sorted(got['counts'])}; report all four statuses, with a 0 for any "
        "that does not occur, so a clean quarter and a missing branch look different"
    )
    assert got["verdict"] == want["verdict"] == "fail", (
        f"the verdict is {got['verdict']!r}; anything late or overdue fails"
    )
    clean = [_incident("G-1", ("c",), False, False, occurred_h=200, aware_h=100,
                       reported_h=90),
             _incident("G-2", (), False, False, occurred_h=50, aware_h=49)]
    ok = lesson.incident_log(clean)
    assert ok["verdict"] == "pass", (
        f"a log with one incident reported in time and one non-incident is a PASS; yours says "
        f"{ok['verdict']!r}. A verdict that can never come back clean carries no information"
    )
    assert ok["counts"] == {"reported_in_time": 1, "reported_late": 0,
                            "open_in_time": 0, "open_overdue": 0}, (
        f"counts on the clean log came out {ok['counts']}"
    )
    assert ok["as_of"].startswith("2026-09-16T12:00:00"), (
        "echo the as_of the log was cut at, so two logs can be compared"
    )


RUBRIC = [
    (test_control_limits_centre_and_sigma, 4,
     "centre = errors/n; sigma = sqrt(centre*(1-centre)/WINDOW SIZE), not /baseline_n"),
    (test_control_limits_clamp_to_the_unit_interval, 3,
     "cap the upper limit at 1.0 and floor the lower at 0.0, but only when they escape"),
    (test_control_limits_empty_baseline_is_permissive, 2,
     "no baseline or no window: centre 0.0, sigma 0.0, ucl 1.0, lcl 0.0, and no exception"),
    (test_control_limits_k_is_a_parameter, 2,
     "k is an argument to be used and echoed, not a constant"),
    (test_monitor_windows_match_an_independent_derivation, 5,
     "tumbling windows in seq order; report errors, rate and the seq and ts bounds"),
    (test_monitor_orders_by_seq_not_timestamp, 4,
     "sort by row['seq']; a buffered batch makes timestamp order a different feed"),
    (test_monitor_drops_the_incomplete_tail, 3,
     "len(rows) // window_size windows; the remainder is dropped_rows, not a short window"),
    (test_monitor_outside_is_strict_and_two_sided, 3,
     "rate > ucl or rate < lcl — strictly outside, and both limits"),
    (test_monitor_alerts_once_per_confirmed_run, 4,
     "one alert per maximal run of >= min_run outside windows, at the run's first index"),
    (test_costs_match_an_independent_derivation, 4,
     "false alerts, misses, delay and the two cost lines, summed"),
    (test_costs_mute_after_the_threshold, 5,
     "stop at the switch_off_after-th false alert; later alerts never fired"),
    (test_costs_missed_windows_before_detection_and_after_mute, 4,
     "a degraded window before the first live true alert, or after the mute, is missed"),
    (test_costs_blind_and_perfect_monitors, 3,
     "no alerts means everything missed and no delay; a clean run must be able to cost 0"),
    (test_offset_is_the_median_not_the_mean, 5,
     "median of (deployer ts - provider ts); the sign matters and so does the outlier"),
    (test_offset_spread_is_the_mad_not_the_sd, 4,
     "spread is the median absolute deviation from the median, not the standard deviation"),
    (test_offset_counts_outliers_against_the_tolerance_parameter, 2,
     "count pairs further than the tolerance ARGUMENT from the median"),
    (test_offset_handles_no_pairs, 2,
     "no matching decision_id: n_pairs 0 and every figure 0.0, without raising"),
    (test_merge_drops_the_overlap_and_counts_it, 5,
     "a decision the provider already holds is counted and dropped, never appended"),
    (test_merge_corrects_the_added_timestamps_by_subtracting, 4,
     "subtract offset_seconds from an added row's timestamp; the offset is an argument"),
    (test_merge_order_renumber_and_row_shape, 4,
     "sort by (ts, decision_id), renumber seq from 0, and carry source and src_seq"),
    (test_merge_out_of_sequence_and_error_rate, 3,
     "compare provider rows in merged order against src_seq order, position by position"),
    (test_classify_not_serious_has_no_clock, 3,
     "no Article 3(49) point: not serious, no paragraph, no deadline — checked first"),
    (test_classify_two_day_trigger_is_widespread_or_point_b, 5,
     "73(3) is widespread OR point (b); point (c) alone is 73(2)'s fifteen days"),
    (test_classify_death_is_ten_days_and_only_a_death, 4,
     "73(4) is a DEATH; a non-fatal point (a) harm is fifteen days"),
    (test_classify_overlap_takes_the_shorter_deadline, 3,
     "test the two-day branch before the death branch, and sort the points"),
    (test_log_clock_starts_at_awareness, 5,
     "due_ts is aware_ts + deadline_days, never occurred_ts + deadline_days"),
    (test_log_status_ladder, 4,
     "reported vs open, then against the deadline, with 'not later than' inclusive"),
    (test_log_order_and_tie_break, 4,
     "sorted by (due_ts, incident_id) — soonest deadline first"),
    (test_log_hours_are_floored_and_signed, 3,
     "whole hours towards minus infinity, signed, for both clocks"),
    (test_log_counts_verdict_and_excluded_candidates, 3,
     "all four status counts, a fail on anything late or overdue, and a pass that is reachable"),
]
