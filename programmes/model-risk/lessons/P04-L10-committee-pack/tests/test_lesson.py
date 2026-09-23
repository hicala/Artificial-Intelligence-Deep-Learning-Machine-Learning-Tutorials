"""Autograder rubric for P04-L10. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson, and are deliberate:

* every expected value is worked out HERE: dates by `datetime` arithmetic, figures by hand
  from the definition in the docstring, never lifted from the notebook's synthetic run. The
  one test that uses the lesson's own draft (`test_gate_end_to_end_on_the_draft`) asks only
  that the single typed figure, and nothing else, comes back untraceable.
* the exercises chain — the gate needs the extractor and the collector, the pack needs the
  gate — so each is graded in isolation. `trace_gate` takes `extract=` and `collect=`, and
  `render_committee_pack` takes `gate=`, exactly so this file can hand in its OWN reference
  implementations (`_ref_extract`, `_ref_collect`, `_ref_gate`, written a different way from
  the lesson's). A broken exercise 5 does not cost marks in exercise 7.
* the centre of the rubric is what a gate does to silence itself. `test_gate_half_unit_is_the_
  tolerance` puts results 1.005 to 2 half-units from their figures — one of them a scientific
  figure whose half unit is 5e-07, which an absolute epsilon of 1e-6 lets through — (a gate
  with a fixed, widened or epsilon-padded tolerance passes one) beside results Python's own
  formatting rounds onto the boundary (a gate with no slack fails them). `test_gate_no_class_of_figure_is_exempt`
  plants an untraceable integer, percentage, table cell, heading figure and negative;
  excluding any class wholesale loses those points. `test_gate_adds_no_sources` hands the gate
  findings whose STRINGS contain the typed figure: a gate that looks anywhere but the
  collector's values traces the typed number to the text that typed it.
* the extractor is graded on its definition directly — dates, section numbers, identifiers and
  rating codes must yield nothing; a bare year, a figure on a heading line and a numbered line
  that is not a heading must still yield figures — so an extractor cannot pass by being timid.
"""
import datetime as dt
import importlib.util
import math
import os
import re
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

F = lesson.Finding
E = lesson.RegisterEntry
A = lesson.AgedEntry

POL = {"owners": {s: f"owner of {s}" for s in lesson.SOURCES},
       "days_to_remediate": {"critical": 30, "high": 90, "medium": 180, "low": 365},
       "escalation": ((1, "director"), (31, "head of model risk"), (91, "committee"))}


def _plus(day: str, days: int) -> str:
    return (dt.date.fromisoformat(day) + dt.timedelta(days=days)).isoformat()


def _days(a: str, b: str) -> int:
    return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days


def _entry(fid, status="open", severity="high", raised="2026-03-31", due="2026-06-29",
           source="stability", rule="STABILITY", subject=None, detail="detail"):
    return E(fid, source, rule, fid if subject is None else subject, severity, detail,
             f"owner of {source}", raised, due, status)


def _raises(fn, *args, **kwargs) -> str:
    """Call fn and return the ValueError's message; fail if it does not raise ValueError."""
    try:
        fn(*args, **kwargs)
    except NotImplementedError:
        raise
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        raise AssertionError(f"expected ValueError, got {type(exc).__name__}: {exc}") from None
    raise AssertionError("expected ValueError, and nothing was raised")


# --- independent references for the gate, written a different way from the lesson ----------

_REF_NUM = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+(?![\d])(?!,\d)|\d+)?(?:\.\d+)?"
                      r"(?:[eE][+-]?\d+)?%?")


def _ref_extract(text):
    """Scan left to right, deciding each candidate's fate from its surroundings."""
    out = []
    for line_no, line in enumerate(text.split("\n"), start=1):
        blocked = set()
        for m in re.finditer(r"\d{4}-\d{2}-\d{2}", line):
            before = line[m.start() - 1] if m.start() else " "
            after = line[m.end()] if m.end() < len(line) else " "
            if not before.isdigit() and not after.isdigit():
                blocked.update(range(m.start(), m.end()))
        h = re.match(r"(#+)([ \t]+)(\S+)", line)
        if h and re.fullmatch(r"(?:[A-Za-z]\.)?\d+(?:\.\d+)*\.?", h.group(3)):
            blocked.update(range(h.start(3), h.end(3)))
        for m in re.finditer(r"[\w.+\-]+", line):
            w = m.group()
            if any(c.isalpha() for c in w) and any(c.isdigit() for c in w):
                core = w.rstrip(".")
                if not re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)[eE][+-]?\d+", core):
                    blocked.update(range(m.start(), m.end()))
        i = 0
        while i < len(line):
            ch = line[i]
            starts = ch.isdigit() or (ch == "." and i + 1 < len(line) and line[i + 1].isdigit())
            signed = ch in "+-" and i + 1 < len(line) and (
                line[i + 1].isdigit() or (line[i + 1] == "." and i + 2 < len(line)
                                          and line[i + 2].isdigit()))
            prev = line[i - 1] if i else " "
            if (starts or signed) and not (prev.isdigit() or prev == "."):
                m = _REF_NUM.match(line, i)
                tok = m.group()
                body = tok[1:] if tok[0] in "+-" else tok
                digits_at = i + (len(tok) - len(body))
                if body and body not in ("%",) and any(c.isdigit() for c in body):
                    if digits_at not in blocked:
                        pct = body.endswith("%")
                        num = body.rstrip("%")
                        mant, _, expo = num.replace("E", "e").partition("e")
                        dec = len(mant.split(".")[1]) if "." in mant else 0
                        e = int(expo) if expo else 0
                        val = float(num.replace(",", ""))
                        half = 0.5 * 10.0 ** (e - dec)
                        if tok[0] == "-":
                            val = -val
                        if pct:
                            val, half = val / 100.0, half / 100.0
                        out.append(lesson.Figure(tok, val, half, line_no))
                    i += len(tok)
                    continue
            i += 1
    return tuple(out)


def _ref_collect(findings):
    """An explicit stack instead of recursion; same order, same paths."""
    out, stack = [], [("", findings)]
    while stack:
        path, obj = stack.pop()
        if isinstance(obj, (bool, np.bool_, str, bytes)) or obj is None:
            continue
        if isinstance(obj, (int, float, np.integer, np.floating)):
            if math.isfinite(float(obj)):
                out.append((path, float(obj)))
            continue
        children = []
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind != "b":
                children = [(f"{path}[{','.join(map(str, idx))}]", obj[idx])
                            for idx in np.ndindex(obj.shape)]
        elif hasattr(obj, "items"):
            children = [(f"{path}.{k}" if path else str(k), v) for k, v in obj.items()]
        elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
            children = [(f"{path}.{n}" if path else n, getattr(obj, n)) for n in obj._fields]
        elif isinstance(obj, (list, tuple)):
            children = [(f"{path}[{i}]", v) for i, v in enumerate(obj)]
        stack.extend(reversed(children))
    return tuple(out)


def _ref_gate(text, findings):
    figures = _ref_extract(text)
    values = _ref_collect(findings)
    traced, untraced = [], []
    for f in figures:
        hit = [p for p, v in values if abs(v - f.value) <= f.half_unit * (1 + 1e-9)]
        if hit:
            traced.append((f, hit[0]))
        else:
            untraced.append(f)
    return lesson.GateResult(not untraced, len(figures), tuple(traced), tuple(untraced))


def _gate(text, findings):
    return lesson.trace_gate(text, findings, extract=_ref_extract, collect=_ref_collect)


def _texts(figs):
    return [f.text for f in figs]


# --- exercise 1: build_register() -----------------------------------------------------------

def test_register_new_entries():
    raised = {"monitoring": (F("MONITOR-BREACH", "m", "high", "d1"),),
              "reproduction": (F("NOT-REPRODUCED", "agg", "critical", "d2"),),
              "calibration": (F("CALIBRATION", "champion", "high", "d3"),)}
    reg = lesson.build_register(raised, (), POL, "2027-01-31")
    assert isinstance(reg, tuple) and all(isinstance(e, E) for e in reg), (
        "return a tuple of RegisterEntry")
    got = {e.rule: e for e in reg}
    assert {r: e.finding_id for r, e in got.items()} == {
        "NOT-REPRODUCED": "F-001", "CALIBRATION": "F-002", "MONITOR-BREACH": "F-003"}, (
        f"ids {[(e.finding_id, e.rule) for e in reg]} — severity first, then SOURCES order "
        "(calibration before monitoring), whatever order the dict arrives in")
    crit, cal = got["NOT-REPRODUCED"], got["CALIBRATION"]
    assert (crit.raised, crit.due) == ("2027-01-31", _plus("2027-01-31", 30)), (
        f"a new critical finding: raised {crit.raised}, due {crit.due} — raised on as_of and "
        "due 30 calendar days later")
    assert cal.due == _plus("2027-01-31", 90), f"a new high finding is due 90 days on: {cal.due}"
    assert (cal.owner, cal.status, cal.detail, cal.severity) == (
        "owner of calibration", "open", "d3", "high"), (
        f"owner {cal.owner!r}, status {cal.status!r} — owner from policy['owners'][source], "
        "status 'open'")


def test_register_id_order_ties():
    # At one severity, SOURCES order and alphabetical rule order DISAGREE here, so a sort that
    # drops either key numbers these differently.
    raised = {"monitoring": (F("MONITOR-BREACH", "m", "medium", "a"),),
              "soundness": (F("VAR-UNUSED", "b", "medium", "b"), F("MONO", "z", "medium", "c"),
                            F("MONO", "a", "medium", "d")),
              "discrimination": (F("AUC-FLOOR", "seg", "medium", "e"),),
              "stability": (F("CSI", "util", "medium", "f"),),
              "calibration": (F("CALIBRATION", "x", "medium", "g"),)}
    reg = lesson.build_register(raised, (), POL, "2026-09-30")
    order = [(e.finding_id, e.source, e.rule, e.subject) for e in reg]
    assert order == [("F-001", "calibration", "CALIBRATION", "x"),
                     ("F-002", "stability", "CSI", "util"),
                     ("F-003", "soundness", "MONO", "a"),
                     ("F-004", "soundness", "MONO", "z"),
                     ("F-005", "soundness", "VAR-UNUSED", "b"),
                     ("F-006", "discrimination", "AUC-FLOOR", "seg"),
                     ("F-007", "monitoring", "MONITOR-BREACH", "m")], (
        f"got {order} — at equal severity new ids go by SOURCES position, then rule, then "
        "subject; not by the order the dict arrives in, and not by rule name alone")


def test_register_recurring_keeps_history():
    history = (E("F-003", "calibration", "CALIBRATION", "champion", "high", "old", "old owner",
                 "2026-03-31", "2026-06-29", "open"),
               _entry("F-004", status="risk accepted", severity="medium"),
               _entry("F-005", status="reopened", source="tiering", rule="VALIDATION-OVERDUE",
                      subject="m1"))
    raised = {"calibration": (F("CALIBRATION", "champion", "critical", "new evidence"),),
              "stability": (F("STABILITY", "F-004", "high", "psi now"),),
              "tiering": (F("VALIDATION-OVERDUE", "m1", "high", "still overdue"),)}
    reg = {e.finding_id: e for e in lesson.build_register(raised, history, POL, "2026-09-30")}
    assert set(reg) == {"F-003", "F-004", "F-005"}, (
        f"ids {sorted(reg)} — every raised key already in the history keeps its identifier; "
        "none of these is new")
    cal = reg["F-003"]
    assert (cal.raised, cal.due, cal.owner) == ("2026-03-31", "2026-06-29", "old owner"), (
        f"raised {cal.raised}, due {cal.due}, owner {cal.owner!r} — a recurring finding keeps its "
        "FIRST raised date, its due date and its owner from the history")
    assert (cal.severity, cal.detail) == ("critical", "new evidence"), (
        "severity and detail are this cycle's evidence, not last quarter's")
    assert reg["F-004"].status == "risk accepted" and reg["F-005"].status == "reopened", (
        "risk accepted and reopened carry over unchanged when the finding recurs")


def test_register_closed_recurrence_reopens():
    history = (_entry("F-001", status="closed"),)
    reg = lesson.build_register({"stability": (F("STABILITY", "F-001", "high", "back"),)},
                                history, POL, "2026-09-30")
    assert len(reg) == 1 and reg[0].finding_id == "F-001" and reg[0].status == "reopened", (
        f"got {[(e.finding_id, e.status) for e in reg]} — a closed finding that comes back is "
        "REOPENED under its old identifier, not raised as new")


def test_register_carries_unraised_history():
    history = (_entry("F-001", status="open"), _entry("F-002", status="closed"),
               _entry("F-003", status="risk accepted", severity="medium"))
    reg = lesson.build_register({}, history, POL, "2026-09-30")
    assert set(reg) == set(history), (
        f"with nothing raised this cycle the register is the history, unchanged: got "
        f"{[(e.finding_id, e.status) for e in reg]} — not being raised is not being fixed")


def test_register_numbering_continues_from_max():
    history = (_entry("F-002"), _entry("F-017", source="tiering", rule="R"),
               _entry("F-005", source="soundness", rule="S"))
    raised = {"monitoring": (F("MONITOR-BREACH", "m", "high", "d"),),
              "calibration": (F("CALIBRATION", "c", "low", "d"),)}
    ids = {e.rule: e.finding_id for e in lesson.build_register(raised, history, POL,
                                                                "2026-09-30")}
    assert (ids["MONITOR-BREACH"], ids["CALIBRATION"]) == ("F-018", "F-019"), (
        f"new ids {ids['MONITOR-BREACH']}, {ids['CALIBRATION']} — continue from the HIGHEST "
        "history id (F-017), never from the count of entries, and never re-use a gap")


def test_register_sorted():
    history = (_entry("F-009", severity="low"), _entry("F-002", severity="high", rule="R2"),
               _entry("F-004", severity="critical", rule="R4"))
    reg = lesson.build_register({"calibration": (F("CALIBRATION", "c", "high", "d"),)},
                                history, POL, "2026-09-30")
    assert [e.finding_id for e in reg] == ["F-004", "F-002", "F-010", "F-009"], (
        f"order {[e.finding_id for e in reg]} — sort by severity rank, then by finding_id")


def test_register_rejects():
    ok = {"calibration": (F("CALIBRATION", "c", "high", "d"),)}
    msg = _raises(lesson.build_register, ok, (), {"owners": {}, "days_to_remediate":
                                                  POL["days_to_remediate"]}, "2026-09-30")
    assert "calibration" in msg, f"name the source with no owner; the message was {msg!r}"
    _raises(lesson.build_register, {"calibration": (F("A", "c", "high", "d"),
                                                    F("A", "c", "high", "e"))},
            (), POL, "2026-09-30")
    _raises(lesson.build_register, {"calibration": (F("A", "c", "severe", "d"),)}, (), POL,
            "2026-09-30")
    _raises(lesson.build_register, ok, (), {"owners": POL["owners"],
                                            "days_to_remediate": {"low": 365}}, "2026-09-30")
    # A source outside SOURCES is refused by name, even when the policy names an owner for it.
    msg = _raises(lesson.build_register, {"audit": (F("AUDIT", "x", "high", "d"),)}, (),
                  {"owners": dict(POL["owners"], audit="internal audit"),
                   "days_to_remediate": POL["days_to_remediate"]}, "2026-09-30")
    assert "audit" in msg, f"name the source that is not in SOURCES; the message was {msg!r}"


# --- exercise 2: age_register() -------------------------------------------------------------

def test_ageing_arithmetic():
    reg = (_entry("F-001", raised="2028-02-01", due="2028-03-01"),
           _entry("F-002", raised="2027-12-15", due="2028-01-31"))
    rows = {r.finding_id: r for r in lesson.age_register(reg, "2028-03-15", POL)}
    assert all(isinstance(r, A) for r in rows.values()), "return a tuple of AgedEntry"
    for e in reg:
        want = (_days(e.raised, "2028-03-15"), max(0, _days(e.due, "2028-03-15")))
        got = (rows[e.finding_id].age_days, rows[e.finding_id].days_overdue)
        assert got == want, (
            f"{e.finding_id}: got age and overdue {got}, want {want} — whole calendar days, a "
            "leap February included")
    r = rows["F-001"]
    assert (r.severity, r.owner) == ("high", "owner of stability"), "carry severity and owner"


def test_ageing_due_date_is_not_overdue():
    reg = (_entry("F-001", raised="2026-09-01", due="2026-09-30"),
           _entry("F-002", raised="2026-09-01", due="2026-09-29"))
    rows = {r.finding_id: r for r in lesson.age_register(reg, "2026-09-30", POL)}
    assert (rows["F-001"].days_overdue, rows["F-001"].escalation) == (0, "none"), (
        "due ON the as-of date is due, not late: 0 days overdue, escalation 'none'")
    assert (rows["F-002"].days_overdue, rows["F-002"].escalation) == (1, "director"), (
        "one day past due reaches the first level")
    # The other side of the same boundary: every NEW finding is raised ON the as-of date.
    try:
        fresh = lesson.age_register((_entry("F-003", raised="2026-09-30", due="2026-12-29"),),
                                    "2026-09-30", POL)
    except ValueError as exc:
        raise AssertionError(
            f"a finding raised ON the as-of date was refused ({exc}) — it is 0 days old; only a "
            "raised date AFTER as_of is a data-entry error") from None
    assert (fresh[0].age_days, fresh[0].days_overdue) == (0, 0), (
        f"raised on the as-of date: age {fresh[0].age_days}, overdue {fresh[0].days_overdue} — "
        "want 0 and 0")


def test_ageing_escalation_is_inclusive():
    pol = {"escalation": ((1, "one"), (31, "two"), (91, "three"), (181, "four"))}
    as_of = "2027-06-30"
    cases = {30: "one", 31: "two", 90: "two", 91: "three", 180: "three", 181: "four", 400: "four"}
    reg = tuple(_entry(f"F-{k:03d}", raised="2025-01-01", due=_plus(as_of, -k)) for k in cases)
    got = {r.days_overdue: r.escalation for r in lesson.age_register(reg, as_of, pol)}
    assert got == cases, (
        f"got {got} — the level is the LAST one whose threshold is <= days overdue: a "
        "threshold is reached on the day it is equalled")


def test_ageing_live_only():
    reg = tuple(_entry(f"F-00{i}", status=s) for i, s in
                enumerate(("open", "reopened", "risk accepted", "closed"), start=1))
    ids = sorted(r.finding_id for r in lesson.age_register(reg, "2026-09-30", POL))
    assert ids == ["F-001", "F-002"], (
        f"tracked {ids} — only open and reopened findings are live; a risk-accepted finding "
        "is a decision already taken, and a closed one is done")


def test_ageing_order():
    reg = (_entry("F-003", due="2026-09-01"), _entry("F-001", due="2026-09-20"),
           _entry("F-002", due="2026-09-01"), _entry("F-004", due="2026-12-31"))
    ids = [r.finding_id for r in lesson.age_register(reg, "2026-09-30", POL)]
    assert ids == ["F-002", "F-003", "F-001", "F-004"], (
        f"order {ids} — most overdue first, ties by finding_id")


def test_ageing_counts_from_as_of_not_the_clock():
    reg = (_entry("F-001", raised="2018-05-04", due="2018-08-02"),)
    for as_of in ("2019-01-01", "2031-07-15"):
        r = lesson.age_register(reg, as_of, POL)[0]
        assert r.age_days == _days("2018-05-04", as_of), (
            f"as of {as_of} the age is {_days('2018-05-04', as_of)} days, got {r.age_days} — "
            "count from the as_of argument, never from date.today()")


def test_ageing_rejects():
    msg = _raises(lesson.age_register, (_entry("F-042", raised="2026-10-01",
                                               due="2026-12-30"),), "2026-09-30", POL)
    assert "F-042" in msg, f"name the finding raised after as_of; the message was {msg!r}"
    _raises(lesson.age_register, (_entry("F-001"),), "2026-09-30",
            {"escalation": ((31, "a"), (31, "b"))})


# --- exercise 3: executive_summary() --------------------------------------------------------

def _summary(reg, tracker=(), policy=POL):
    return lesson.executive_summary(reg, tracker, policy)


def test_summary_opinion_rule():
    cases = [
        ((_entry("F-001", severity="critical"),), (), "NOT FIT FOR USE",
         "a live critical finding"),
        ((_entry("F-001", severity="critical", status="risk accepted"),
          _entry("F-002", severity="medium")), (), "FIT FOR USE",
         "a risk-accepted critical beside a live medium"),
        ((_entry("F-001", severity="high", status="reopened"),), (),
         "FIT FOR USE WITH CONDITIONS", "a reopened high"),
        ((_entry("F-001", severity="medium"),),
         (A("F-001", "medium", "o", 200, 120, "committee"),), "FIT FOR USE WITH CONDITIONS",
         "a live medium escalated to the top level"),
        ((_entry("F-001", severity="critical", status="closed"),), (), "FIT FOR USE",
         "a closed critical and nothing else"),
    ]
    for reg, tracker, want, what in cases:
        got = _summary(reg, tracker).opinion
        assert got == want, (
            f"{what}: opinion {got!r}, want {want!r} — live means open or reopened, and a "
            "finding at the TOP escalation level forces conditions")


def test_summary_counts():
    reg = (_entry("F-001", severity="critical", status="risk accepted"),
           _entry("F-002", severity="high", status="reopened"),
           _entry("F-003", severity="medium"), _entry("F-004", severity="medium"),
           _entry("F-005", severity="high", status="closed"),
           _entry("F-006", severity="low", status="closed"))
    tracker = (A("F-002", "high", "o", 183, 93, "board"), A("F-003", "medium", "o", 40, 12,
                                                             "committee"),
               A("F-004", "medium", "o", 3, 0, "none"))
    pol = {"escalation": ((1, "committee"), (91, "board"))}
    s = _summary(reg, tracker, pol)
    assert s.live == 3 and s.by_severity == {"critical": 0, "high": 1, "medium": 2, "low": 0}, (
        f"live {s.live}, by severity {s.by_severity} — count open and reopened entries only, "
        "every severity present as a key")
    assert list(s.by_severity) == list(lesson.SEVERITY_ORDER), (
        "by_severity keys in SEVERITY_ORDER order")
    assert (s.overdue, s.escalated) == (2, 1), (
        f"overdue {s.overdue}, escalated {s.escalated} — overdue counts tracker rows with any "
        "days overdue; escalated counts rows at the LAST level of the policy's ladder only")
    assert (s.accepted, s.closed) == (1, 2), f"accepted {s.accepted}, closed {s.closed}"


def test_summary_lines_exact():
    reg = (_entry("F-001", severity="high", status="reopened"),
           _entry("F-002", severity="medium"), _entry("F-003", severity="low",
                                                      status="risk accepted"))
    tracker = (A("F-001", "high", "o", 183, 93, "board"),)
    pol = {"escalation": ((1, "committee"), (91, "board"))}
    s = _summary(reg, tracker, pol)
    want = ("Opinion: FIT FOR USE WITH CONDITIONS.",
            "Live findings: 2 (0 critical, 1 high, 1 medium, 0 low).",
            "Overdue: 1. Escalated to the board: 1.",
            "Risk accepted: 1. Closed: 0.")
    assert tuple(s.lines[:4]) == want, f"the first four lines were {tuple(s.lines[:4])}"
    assert len(s.lines) == 5, f"one headline line per live critical or high entry: {s.lines}"


def test_summary_headline():
    reg = (_entry("F-009", severity="critical", source="reproduction", detail="gap 0.0068"),
           _entry("F-002", severity="high", detail="psi 0.5982"),
           _entry("F-001", severity="high", status="risk accepted"),
           _entry("F-004", severity="medium"),
           _entry("F-003", severity="high", status="reopened", source="soundness",
                  detail="mono 0.0123"))
    s = _summary(reg)
    assert tuple(s.headline) == ("F-009", "F-002", "F-003"), (
        f"headline {s.headline} — live critical and high entries in REGISTER order, not "
        "re-sorted, and nothing risk accepted")
    assert tuple(s.lines[4:]) == ("- F-009 (critical, reproduction): gap 0.0068",
                                  "- F-002 (high, stability): psi 0.5982",
                                  "- F-003 (high, soundness): mono 0.0123"), (
        f"headline lines {s.lines[4:]} — '- id (severity, source): detail'")


# --- exercise 4: limitations() --------------------------------------------------------------

def _table(counts):
    counts = np.asarray(counts)
    edges = np.linspace(0.0, 1.0, counts.size + 1)
    nan = np.full(counts.size, np.nan)
    return lesson.ReliabilityTable(edges[:-1], edges[1:], counts, nan, nan)


def _evidence(sub=None, tables=None, comps=None, floor=200, material=0.01):
    sub = sub or {"group": np.array(["A"]), "count": np.array([900]),
                  "reportable": np.array([True])}
    return {"calibration": {"subgroups": sub, "min_support": floor,
                            "reliability": tables or {}},
            "discrimination": {"comparisons": comps or {}, "material": material}}


def test_limitations_suppressed_groups():
    # C sits one record below the minimum and B, D exactly at it: both sides of the boundary.
    sub = {"group": np.array(["A", "B", "C", "D"]), "count": np.array([150, 250, 249, 250]),
           "reportable": np.array([False, True, False, True])}
    got = lesson.limitations(_evidence(sub=sub, floor=250))
    assert [(x.kind, x.subject) for x in got] == [("suppressed subgroup", "A"),
                                                  ("suppressed subgroup", "C")], (
        f"got {[(x.kind, x.subject) for x in got]} — one per group that is not reportable, in "
        "table order; a group exactly at the minimum is reportable, one below it is not")
    assert got[1].detail == ("249 records against a minimum support of 250; no metric is "
                             "reported for it"), f"detail was {got[1].detail!r}"


def test_limitations_empty_bin_runs():
    tables = {"first": _table([0, 0, 4, 0, 7, 7, 0, 0, 0, 1]), "full": _table([3, 3]),
              "empty": _table([0, 0, 0, 0])}
    got = lesson.limitations(_evidence(tables=tables))
    assert [(x.subject, x.detail) for x in got] == [
        ("first", "no observations between 0.00 and 0.20"),
        ("first", "no observations between 0.30 and 0.40"),
        ("first", "no observations between 0.60 and 0.90"),
        ("empty", "no observations between 0.00 and 1.00")], (
        f"got {[(x.subject, x.detail) for x in got]} — one limitation per MAXIMAL run of empty "
        "bins, from the run's first lower edge to its last upper edge, tables in dict order")
    assert all(x.kind == "empty bins" for x in got), "kind is 'empty bins'"


def test_limitations_not_measurable_only():
    M = lesson.Measurability
    comps = {"wide": M(0.08412, 13647, "NOT MEASURABLE", "r"),
             "adequate": M(0.0050, 900, "NOT SIGNIFICANT", "r"),
             "found": M(0.0050, 900, "SIGNIFICANT", "r"),
             "tiny": M(0.2, 1234567, "NOT MEASURABLE", "r")}
    got = lesson.limitations(_evidence(comps=comps, material=0.015))
    assert [x.subject for x in got] == ["wide", "tiny"], (
        f"got {[x.subject for x in got]} — only NOT MEASURABLE is a limitation; NOT SIGNIFICANT "
        "at adequate power is a result")
    assert got[0].detail == ("minimum detectable difference 0.0841 against a material "
                             "difference of 0.0150; 13,647 records would be needed"), (
        f"detail was {got[0].detail!r}")
    assert got[1].detail.endswith("1,234,567 records would be needed"), (
        "n_required prints with thousands separators")


def test_limitations_order_and_clean():
    sub = {"group": np.array(["S"]), "count": np.array([5]), "reportable": np.array([False])}
    M = lesson.Measurability
    ev = _evidence(sub=sub, tables={"t": _table([1, 0])},
                   comps={"c": M(0.05, 5000, "NOT MEASURABLE", "r")})
    assert [x.kind for x in lesson.limitations(ev)] == list(lesson.LIMITATION_KINDS), (
        "kinds in LIMITATION_KINDS order: suppressed subgroups, then empty bins, then not "
        "measurable")
    clean = lesson.limitations(_evidence(tables={"t": _table([1, 2])}))
    assert clean == (), f"nothing unmeasured returns (), got {clean}"


# --- exercise 5: extract_figures() ----------------------------------------------------------

def _vals(text):
    return [(f.text, round(f.value, 12), round(f.half_unit, 15)) for f in
            lesson.extract_figures(text)]


def test_figures_plain_values_and_half_units():
    got = _vals("ECE 0.0727 on 6000 records, AUC 0.695, PSI 0.5982.")
    assert got == [("0.0727", 0.0727, 5e-05), ("6000", 6000.0, 0.5), ("0.695", 0.695, 0.0005),
                   ("0.5982", 0.5982, 5e-05)], (
        f"got {got} — each figure with its value and HALF a unit in its last printed place")
    assert all(isinstance(f, lesson.Figure) for f in lesson.extract_figures("1")), (
        "return Figure records")


def test_figures_thousands_separators():
    got = _vals("n = 1,234 of 4,194,304 records, sum 6,000. Pairs 0.5,0.6 and 12,3456")
    assert got == [("1,234", 1234.0, 0.5), ("4,194,304", 4194304.0, 0.5),
                   ("6,000", 6000.0, 0.5), ("0.5", 0.5, 0.05), ("0.6", 0.6, 0.05),
                   ("12", 12.0, 0.5), ("3456", 3456.0, 0.5)], (
        f"got {got} — '4,194,304' is ONE figure; a comma separates thousands only between "
        "groups of EXACTLY three digits, so in '12,3456' the ',345' is not a group and no digit "
        "may be dropped")


def test_figures_percentages():
    got = _vals("rates 12.5% and -3% and 100%")
    assert got == [("12.5%", 0.125, 0.0005), ("-3%", -0.03, 0.005), ("100%", 1.0, 0.005)], (
        f"got {got} — a percent divides the value AND the half unit by 100")


def test_figures_signs_and_ranges():
    got = _vals("| 0.00-0.10 | x=-0.0588 y=+0.0269 (-2) 1-2")
    assert got == [("0.00", 0.0, 0.005), ("0.10", 0.1, 0.005), ("-0.0588", -0.0588, 5e-05),
                   ("+0.0269", 0.0269, 5e-05), ("-2", -2.0, 0.5), ("1", 1.0, 0.5),
                   ("2", 2.0, 0.5)], (
        f"got {got} — a dash straight after a digit is a range, not a minus sign")


def test_figures_dates_are_not_figures():
    assert _vals("as of 2026-09-30, raised 2026-03-31.") == [], (
        "a YYYY-MM-DD date is a name, not a figure")
    assert _vals("in 2026 we saw 12") == [("2026", 2026.0, 0.5), ("12", 12.0, 0.5)], (
        "only a full YYYY-MM-DD date is excluded — a bare year is still a figure")


def test_figures_section_numbers():
    doc = "## 3. Findings register\n### 3.2 Ageing\n## A.1 Data\n#### 10. Appendix"
    assert _vals(doc) == [], f"section numbers are names: got {_vals(doc)}"
    assert _vals("## 4. Limitations (5 found)") == [("5", 5.0, 0.5)], (
        "only the section number is a name; other numbers on a heading are figures")
    assert _vals("3. Not a heading") == [("3", 3.0, 0.5)], (
        "a numbered line that is not a markdown heading carries a figure")


def test_figures_identifiers():
    doc = "F-003 champion-v3 S1 MR-2026-014 A.1 x86_64 sha256 P04-L10"
    assert _vals(doc) == [], (
        f"got {_vals(doc)} — a word holding a letter and a digit is a name, every number in it")


def test_figures_scientific_notation():
    got = _vals("differs by 6.836e-03, bound 1.273e-03. Or 2.000e+05 and 3E+2")
    assert got == [("6.836e-03", 0.006836, 5e-07), ("1.273e-03", 0.001273, 5e-07),
                   ("2.000e+05", 200000.0, 50.0), ("3E+2", 300.0, 50.0)], (
        f"got {got} — scientific notation is a figure, and its half unit is scaled by the "
        "exponent")


def test_figures_lines_and_leading_point():
    got = [(f.text, f.line) for f in lesson.extract_figures("a 1\n\nb .5\nc 2.25")]
    assert got == [("1", 1), (".5", 3), ("2.25", 4)], (
        f"got {got} — lines are numbered from 1, blank lines included, and '.5' is a figure")


# --- exercise 6: computed_values() ----------------------------------------------------------

def test_values_walk_and_paths():
    st = lesson.StabilityResult(0.25, np.array([0.1, 0.15]), np.array([0.5, 0.5]),
                                np.array([0.4, 0.6]))
    got = lesson.computed_values({"a": {"b": [1, (2.5, {"c": 3})]}, "psi": st, "z": 9})
    want = (("a.b[0]", 1.0), ("a.b[1][0]", 2.5), ("a.b[1][1].c", 3.0), ("psi.psi", 0.25),
            ("psi.contributions[0]", 0.1), ("psi.contributions[1]", 0.15),
            ("psi.expected_pct[0]", 0.5), ("psi.expected_pct[1]", 0.5),
            ("psi.actual_pct[0]", 0.4), ("psi.actual_pct[1]", 0.6), ("z", 9.0))
    assert tuple(got) == want, (
        f"got {got} — walk in order; keys and fields join with '.', positions with brackets")


def test_values_numpy_scalars():
    got = dict(lesson.computed_values({"i": np.int64(7), "u": np.uint8(3),
                                       "f": np.float32(0.5), "g": np.float64(-1.25)}))
    assert got == {"i": 7.0, "u": 3.0, "f": 0.5, "g": -1.25}, (
        f"got {got} — numpy integers and floats are results too")
    assert all(type(v) is float for v in got.values()), "convert every value to a Python float"


def test_values_skip_booleans():
    got = lesson.computed_values({"flag": True, "np": np.bool_(False), "mask": np.array([True]),
                                  "mixed": [True, 2]})
    assert tuple(got) == (("mixed[1]", 2.0),), (
        f"got {got} — True is an int to Python and a flag to everyone else; a boolean is never "
        "a computed result, alone, as a numpy scalar, or as an array")


def test_values_skip_strings():
    got = lesson.computed_values({"note": "PSI 0.2874", "b": b"12", "none": None,
                                  "labels": np.array(["1", "2"]), "x": 0.5})
    assert tuple(got) == (("x", 0.5),), (
        f"got {got} — a string is not a result, however many digits it holds: the typed "
        "commentary lives in the findings as a string")


def test_values_skip_non_finite_and_keys():
    got = lesson.computed_values({"n": float("nan"), "i": float("inf"),
                                  "arr": np.array([np.nan, 1.0]), 0.25: "label", 3: 4.0})
    assert tuple(got) == (("arr[1]", 1.0), ("3", 4.0)), (
        f"got {got} — NaN and infinities are never printed figures, and dict keys are labels")


def test_values_multidimensional_paths():
    got = lesson.computed_values({"m": np.array([[1.0, 2.0], [3.0, 4.0]])})
    assert tuple(got) == (("m[0,0]", 1.0), ("m[0,1]", 2.0), ("m[1,0]", 3.0),
                          ("m[1,1]", 4.0)), f"got {got} — a 2-D element's path is 'm[i,j]'"


# --- exercise 7: trace_gate() ---------------------------------------------------------------

def test_gate_traces_formatted_figures():
    doc = "n = 6,000; rate 22.6%; ECE 0.0727; gap -0.0215; bound 1.273e-03"
    fnd = {"n": 6000, "rate": 0.22598, "ece": 0.072691, "gap": -0.021524,
           "bound": 0.0012734}
    r = _gate(doc, fnd)
    assert isinstance(r, lesson.GateResult), "return a GateResult"
    assert r.passed and [p for _, p in r.traced] == ["n", "rate", "ece", "gap", "bound"], (
        f"passed={r.passed}, untraced {_texts(r.untraced)} — every figure here is a formatting "
        "of a result: compare on the figure's own scale within its half unit")


def test_gate_half_unit_is_the_tolerance():
    # 1.5 to 2 half-units off; then a scientific figure 1.5 of ITS half-units (5e-07) off, which
    # an absolute epsilon such as 1e-6 lets through; then a figure only 1.005 half-units off.
    far = _gate("ECE 0.0214 and 6,001 and 22.7% and 1.274e-03 and 0.0331",
                {"a": 0.021325, "b": 6000, "c": 0.2260, "d": 0.00127325, "e": 0.03304975})
    assert _texts(far.untraced) == ["0.0214", "6,001", "22.7%", "1.274e-03", "0.0331"], (
        f"untraced {_texts(far.untraced)} — every figure here is more than its own half unit "
        "from its nearest result and must fail; a tolerance wider than the half unit, a fixed "
        "or absolute epsilon added to it, lets one through")
    near = _gate("ECE 0.0214 and 22.6%", {"a": 0.021351, "c": 0.22649})
    assert near.passed, f"inside the half unit must trace: untraced {_texts(near.untraced)}"
    boundary = [0.125, 0.375, 2.5, 0.0215, 0.02135, 1.0005]
    doc = " ".join([f"{0.125:.2f}", f"{0.375:.2f}", f"{2.5:.0f}", f"{0.0215:.3f}",
                    f"{0.02135:.4f}", f"{1.0005:.3f}"])
    r = _gate(doc, {f"v{i}": v for i, v in enumerate(boundary)})
    assert r.passed, (
        f"{doc!r} is Python's own formatting of {boundary}, rounding onto the half-unit "
        f"boundary, yet {_texts(r.untraced)} did not trace — use <= and TRACE_SLACK")


def test_gate_no_class_of_figure_is_exempt():
    doc = ("## 3. Register (9 open)\n"
           "| a | 42.0% |\n"
           "7 findings were raised\n"
           "a gap of -0.0123\n"
           "total 1,234,568\n")
    fnd = {"x": 0.5, "y": 0.0123, "n": 1234567, "k": 3}
    r = _gate(doc, fnd)
    assert _texts(r.untraced) == ["9", "42.0%", "7", "-0.0123", "1,234,568"], (
        f"untraced {_texts(r.untraced)} — a figure on a heading, in a table, a whole number, "
        "a negative and a large number with separators are all figures, and none traces here")
    assert r.n_figures == 5 and not r.passed, "count every figure; any untraced one fails"


def test_gate_sign_is_a_direction():
    assert not _gate("gap -0.0123", {"gap": 0.0123}).passed, (
        "-0.0123 does not trace to +0.0123; comparing magnitudes loses the direction")
    assert _gate("gap -0.0123", {"gap": -0.0123}).passed, "-0.0123 traces to -0.0123"


def test_gate_adds_no_sources():
    fnd = {"commentary": "calibration improves to 0.0140", "flag": True, "x": 0.25}
    r = _gate("calibration improves to 0.0140, with 1 flag", fnd)
    assert _texts(r.untraced) == ["0.0140", "1"], (
        f"untraced {_texts(r.untraced)} — trace only against collect(findings); searching the "
        "findings' strings or counting True as 1 lets a typed number trace to itself")


def test_gate_first_match_path():
    r = _gate("PSI 0.25", {"contribution": 0.2512, "threshold": 0.25})
    assert [p for _, p in r.traced] == ["contribution"], (
        f"traced to {[p for _, p in r.traced]} — the FIRST value within tolerance, in "
        "collect order, not the closest one")


def test_gate_result_shape():
    r = _gate("a 1 b 0.5 c 9", {"x": 1, "y": 0.5})
    assert (r.passed, r.n_figures, len(r.traced), _texts(r.untraced)) == (False, 3, 2, ["9"]), (
        f"got passed={r.passed}, n_figures={r.n_figures}, traced={len(r.traced)}, untraced="
        f"{_texts(r.untraced)}")
    assert [f.text for f, _ in r.traced] == ["1", "0.5"], "traced figures in document order"
    empty = _gate("", {})
    assert empty.passed and empty.n_figures == 0, "an empty document passes with no figures"


def test_gate_uses_the_helpers_it_is_given():
    fig = lesson.Figure("99", 99.0, 0.5, 1)
    r = lesson.trace_gate("no digits here", {"ignored": 1.0}, extract=lambda text: (fig,),
                          collect=lambda findings: (("given", 99.2),))
    assert r.passed and r.traced == ((fig, "given"),), (
        "trace_gate must use the `extract` and `collect` it is handed, looked up when called")


def test_gate_end_to_end_on_the_draft():
    r = lesson.trace_gate(lesson.DRAFT, lesson.VALIDATION)
    assert _texts(r.untraced) == ["0.0140"], (
        f"on the lab's draft pack your own extractor and collector left "
        f"{_texts(r.untraced)} untraced; exactly one figure in it was typed by hand")
    assert r.n_figures == len(_ref_extract(lesson.DRAFT)), (
        f"you found {r.n_figures} figures in the draft; the definition finds "
        f"{len(_ref_extract(lesson.DRAFT))}")


# --- exercise 8: render_committee_pack() ----------------------------------------------------

def _probe_validation():
    rel = lesson.ReliabilityTable(np.array([0.0, 0.5]), np.array([0.5, 1.0]), np.array([4, 0]),
                                  np.array([0.25, np.nan]), np.array([0.5, np.nan]))
    psi = lesson.StabilityResult(0.7777, np.array([0.1, 0.6777]), np.array([0.5, 0.5]),
                                 np.array([0.2, 0.8]))
    sub = {"group": np.array(["big"]), "count": np.array([900]), "event_rate": np.array([0.3]),
           "mean_pred": np.array([0.35]), "gap": np.array([0.05]), "ece": np.array([0.0505]),
           "reportable": np.array([True])}
    return {"model_id": "probe", "as_of": "2026-09-30", "data_note": "PROBE NOTE",
            "reliability": rel, "ece": 0.4242, "psi": psi, "psi_threshold": 0.25,
            "subgroups": sub, "min_support": 200, "champion": {"auc": 0.616, "ece": 0.5252},
            "challenger": {"auc": 0.818, "ece": 0.0303},
            "decision": lesson.Decision("hold", ("PASS probe rule",), {"auc_gain": 0.202}),
            "worst": 1}


def _probe_pack(commentary="Reviewed by the second line."):
    reg = (E("F-007", "reproduction", "NOT-REPRODUCED", "portfolio aggregate", "critical",
             "differs by 6.836e-03", "credit risk reporting", "2026-09-30", "2026-10-30",
             "open"),
           E("F-001", "stability", "STABILITY", "", "high", "PSI 0.7777 above 0.25",
             "retail credit modelling", "2026-03-31", "2026-06-29", "reopened"))
    tracker = (A("F-001", "high", "retail credit modelling", 183, 93, "committee"),
               A("F-007", "critical", "credit risk reporting", 0, 0, "none"))
    summary = lesson.Summary("NOT FIT FOR USE", 2, {"critical": 1, "high": 1, "medium": 0,
                                                    "low": 0}, 1, 1, 0, 0, ("F-007", "F-001"),
                             ("Opinion: NOT FIT FOR USE.", "Live findings: 2."))
    lims = (lesson.Limitation("empty bins", "probe table",
                              "no observations between 0.50 and 1.00"),)
    return {"model_id": "probe", "as_of": "2026-09-30", "validation": _probe_validation(),
            "register": reg, "tracker": tracker, "summary": summary, "limitations": lims,
            "commentary": commentary, "evidence": {"gap": 0.0068359375}}


def _passing(text, findings):
    return lesson.GateResult(True, 0, (), ())


def _render(**changes):
    return lesson.render_committee_pack(dict(_probe_pack(), **changes), gate=_passing)


def test_pack_title_and_sections():
    lines = _render().splitlines()
    assert lines[0] == "# Committee pack — probe", f"line 1 was {lines[0]!r}"
    assert lines[2] == ("As of 2026-09-30. Every figure in this pack was traced to a computed "
                        "result before it rendered."), f"line 3 was {lines[2]!r}"
    pos = [lines.index(h) if h in lines else -1 for h in lesson.PACK_SECTIONS]
    assert all(p >= 0 for p in pos) and pos == sorted(pos), (
        "the six PACK_SECTIONS headings, each on its own line, in order")


def test_pack_summary_verbatim():
    text = _render()
    block = text.split(lesson.PACK_SECTIONS[0])[1].split(lesson.PACK_SECTIONS[1])[0]
    assert [l for l in block.splitlines() if l] == ["Opinion: NOT FIT FOR USE.",
                                                    "Live findings: 2."], (
        f"section 1 held {block!r} — the summary's lines, verbatim, and nothing else")


def test_pack_register_table():
    lines = _render().splitlines()
    head = "| id | severity | source | rule | subject | owner | raised | due | status |"
    assert head in lines, "the register table's header, exactly"
    i = lines.index(head)
    assert lines[i + 1] == "| --- | --- | --- | --- | --- | --- | --- | --- | --- |", (
        "the separator row follows the header")
    assert lines[i + 2:i + 4] == [
        "| F-007 | S1 critical | reproduction | NOT-REPRODUCED | portfolio aggregate | "
        "credit risk reporting | 2026-09-30 | 2026-10-30 | open |",
        "| F-001 | S2 high | stability | STABILITY | — | retail credit modelling | 2026-03-31 | "
        "2026-06-29 | reopened |"], (
        f"rows were {lines[i + 2:i + 4]} — severity preceded by its code, '—' for an empty "
        "subject, register order")
    assert lines[i + 4] == "" and lines[i + 5:i + 7] == [
        "- F-007: differs by 6.836e-03", "- F-001: PSI 0.7777 above 0.25"], (
        "after a blank line, one '- id: detail' bullet per entry, in register order")


def test_pack_tracker_table():
    lines = _render().splitlines()
    head = "| id | severity | owner | age (days) | overdue (days) | escalation |"
    assert head in lines, "the tracker table's header, exactly"
    i = lines.index(head)
    assert lines[i + 1] == "| --- | --- | --- | --- | --- | --- |" and lines[i + 2:i + 4] == [
        "| F-001 | high | retail credit modelling | 183 | 93 | committee |",
        "| F-007 | critical | credit risk reporting | 0 | 0 | none |"], (
        f"tracker rows were {lines[i + 2:i + 4]}")
    empty = _render(tracker=()).splitlines()
    assert "No live findings." in empty and head not in empty, (
        "an empty tracker prints 'No live findings.' and no table")


def test_pack_limitations_and_commentary():
    lines = _render().splitlines()
    assert "- empty bins — probe table: no observations between 0.50 and 1.00" in lines, (
        "a limitation prints as '- kind — subject: detail'")
    assert "Reviewed by the second line." in lines, "the commentary is printed verbatim"
    bare = _render(limitations=(), commentary="").splitlines()
    assert "None: the suite measured everything it was asked to." in bare, (
        "no limitations prints the docstring's sentence")
    assert "No commentary." in bare, "an empty commentary prints 'No commentary.'"


def test_pack_embeds_validation_report():
    text = _render()
    tail = text.split(lesson.PACK_SECTIONS[5])[1]
    assert "### 2. Calibration" in tail and "Expected calibration error: 0.4242" in tail, (
        "module 1's report follows section 6, every heading one level down")
    assert "# Validation report — probe" not in text, "without its own title line"


def test_pack_gates_the_final_text():
    seen = []

    def spy(text, findings):
        seen.append((text, findings))
        return lesson.GateResult(True, 0, (), ())

    probe = _probe_pack(commentary="Typed after the fact: 0.0140.")
    text = lesson.render_committee_pack(probe, gate=spy)
    assert len(seen) == 1 and seen[0][0] == text and seen[0][1] is probe, (
        "call the gate exactly once, on the COMPLETE text you return — commentary included — "
        "with the findings you were given")


def test_pack_refuses_an_untraceable_figure():
    clean = lesson.render_committee_pack(_probe_pack(), gate=_ref_gate)
    assert clean.startswith("# Committee pack — probe"), (
        "a pack whose every figure traces must render")
    try:
        lesson.render_committee_pack(
            _probe_pack(commentary="The challenger improves calibration to 0.0140."),
            gate=_ref_gate)
    except lesson.UntraceableFigures as exc:
        assert "0.0140" in str(exc) and _texts(exc.figures) == ["0.0140"], (
            f"the refusal must name the untraceable figure; it said {exc}")
    except NotImplementedError:
        raise
    else:
        raise AssertionError("a typed figure in the commentary must stop the pack: raise "
                             "UntraceableFigures, never return the text or a warning")


def test_pack_missing_keys():
    for dropped in (("summary",), ("commentary",), ("tracker", "limitations")):
        msg = _raises(lesson.render_committee_pack,
                      {k: v for k, v in _probe_pack().items() if k not in dropped}, gate=_passing)
        assert all(k in msg for k in dropped), (
            f"name EVERY missing key {dropped}: the message was {msg!r}")


RUBRIC = [
    # exercise 1 — build_register (21)
    (test_register_new_entries, 4,
     "raised = as_of, due = as_of + policy days, owner from policy, ids by severity then SOURCES"),
    (test_register_id_order_ties, 3,
     "sort new findings by (severity rank, SOURCES position, rule, subject) before numbering"),
    (test_register_recurring_keeps_history, 4,
     "a recurring key keeps id, raised, due and owner; severity and detail are this cycle's"),
    (test_register_closed_recurrence_reopens, 3,
     "history status 'closed' + raised again -> 'reopened'"),
    (test_register_carries_unraised_history, 2,
     "carry every history entry this cycle did not raise, unchanged"),
    (test_register_numbering_continues_from_max, 2,
     "new numbers start after the HIGHEST existing F-number"),
    (test_register_sorted, 1, "sort the result by severity rank, then finding_id"),
    (test_register_rejects, 2,
     "ValueError, naming it, for a missing owner or period, an unknown severity or source, "
     "a duplicate key"),
    # exercise 2 — age_register (16)
    (test_ageing_arithmetic, 4, "(as_of - raised).days and max(0, (as_of - due).days)"),
    (test_ageing_due_date_is_not_overdue, 2,
     "due on the as-of date is 0 days overdue; raised on it is 0 days old, not an error"),
    (test_ageing_escalation_is_inclusive, 3,
     "keep the last level whose threshold <= days_overdue"),
    (test_ageing_live_only, 3, "track only statuses in LIVE"),
    (test_ageing_order, 2, "sort by -days_overdue, then finding_id"),
    (test_ageing_counts_from_as_of_not_the_clock, 1, "never call date.today()"),
    (test_ageing_rejects, 1, "ValueError for raised after as_of, and for thresholds that do not rise"),
    # exercise 3 — executive_summary (16)
    (test_summary_opinion_rule, 5,
     "live critical -> NOT FIT; live high or a top-level escalation -> WITH CONDITIONS"),
    (test_summary_counts, 4,
     "count live by severity; overdue and escalated from the tracker; top level = last name"),
    (test_summary_lines_exact, 4, "the four count lines exactly as the docstring spells them"),
    (test_summary_headline, 3, "live critical and high, in register order, with their details"),
    # exercise 4 — limitations (15)
    (test_limitations_suppressed_groups, 4, "one per non-reportable group, with its true count"),
    (test_limitations_empty_bin_runs, 5, "merge each maximal run of empty bins into one"),
    (test_limitations_not_measurable_only, 4, "NOT MEASURABLE only; n_required with ':,'"),
    (test_limitations_order_and_clean, 2, "kinds in LIMITATION_KINDS order; () when clean"),
    # exercise 5 — extract_figures (26)
    (test_figures_plain_values_and_half_units, 3, "half_unit = 0.5 * 10 ** -decimals"),
    (test_figures_thousands_separators, 3, "commas only between groups of exactly three digits"),
    (test_figures_percentages, 3, "divide value and half unit by 100 for a %"),
    (test_figures_signs_and_ranges, 3, "a sign counts only when no digit stands before it"),
    (test_figures_dates_are_not_figures, 3, "mask YYYY-MM-DD dates, and nothing more"),
    (test_figures_section_numbers, 3, "mask only the section number after a heading's #s"),
    (test_figures_identifiers, 3, "mask every word that holds both a letter and a digit"),
    (test_figures_scientific_notation, 3, "a word in scientific notation is a figure"),
    (test_figures_lines_and_leading_point, 2, "number lines from 1; accept '.5'"),
    # exercise 6 — computed_values (14)
    (test_values_walk_and_paths, 4, "mappings and fields join with '.', positions with []"),
    (test_values_numpy_scalars, 2, "float() every Python or numpy int and float"),
    (test_values_skip_booleans, 3, "test for bool and np.bool_ BEFORE int"),
    (test_values_skip_strings, 3, "strings are never results, whatever they hold"),
    (test_values_skip_non_finite_and_keys, 1, "skip NaN and infinities; keys are labels"),
    (test_values_multidimensional_paths, 1, "np.ndindex gives the '[i,j]' path"),
    # exercise 7 — trace_gate (28)
    (test_gate_traces_formatted_figures, 4, "compare figure.value with each value"),
    (test_gate_half_unit_is_the_tolerance, 5,
     "abs(value - figure.value) <= figure.half_unit * (1 + TRACE_SLACK), nothing wider"),
    (test_gate_no_class_of_figure_is_exempt, 3, "check every figure the extractor returns"),
    (test_gate_sign_is_a_direction, 2, "compare signed values, not magnitudes"),
    (test_gate_adds_no_sources, 2, "trace only against collect(findings)"),
    (test_gate_first_match_path, 2, "take the first value in collect order within tolerance"),
    (test_gate_result_shape, 2, "passed = no untraced; n_figures counts them all"),
    (test_gate_uses_the_helpers_it_is_given, 2, "use extract= and collect= when given"),
    (test_gate_end_to_end_on_the_draft, 3,
     "exercises 5 and 6 together must leave only the typed figure untraced"),
    # exercise 8 — render_committee_pack (25)
    (test_pack_title_and_sections, 3, "title, as-of line, then PACK_SECTIONS in order"),
    (test_pack_summary_verbatim, 2, "section 1 is the summary's lines"),
    (test_pack_register_table, 4, "code + severity, '—' for no subject, then detail bullets"),
    (test_pack_tracker_table, 3, "tracker rows, or 'No live findings.'"),
    (test_pack_limitations_and_commentary, 3, "bullets; the empty-case sentences; verbatim"),
    (test_pack_embeds_validation_report, 2, "module 1's report, title dropped, headings demoted"),
    (test_pack_gates_the_final_text, 3, "build all of it, THEN call gate(text, findings) once"),
    (test_pack_refuses_an_untraceable_figure, 4,
     "raise UntraceableFigures(result.untraced) when the gate fails"),
    (test_pack_missing_keys, 1, "name every missing key in one ValueError"),
]
