"""Autograder rubric for P04-L03. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson, and are deliberate:

* every expected value is worked out HERE, by hand or from a tiny closed form. No case
  depends on the notebook's synthetic extract, so a change to the generator cannot silently
  break grading, and no student is ever graded against their own output.
* the eight exercises chain in the notebook (the register consumes the other seven), so the
  register's cases build their evidence bundles from the lesson's NamedTuples directly. A
  broken prober must not cost a student the register's points.
* `test_observe_records_get` is the one that stops a recorder overriding only `__getitem__`.
  An implementation is as entitled to `row.get(name, 0.0)` as to `row[name]`, and the whole
  reconciliation downstream rests on the answer.
* `test_mono_skips_raising_points` hands the prober a function that raises on the low half of
  the domain and rises on the rest. Treating a raise as a score, or as a counterexample, both
  fail here and pass everything else. Its second half puts the unevaluable point in the MIDDLE
  of the grid, bracketed by the only wrong-way pair there is: a prober that increments
  `n_skipped` and then leaves the point in the comparison — substituting `0.0`, or keeping the
  `nan`, both of which pass the first half — reports the wrong `worst_violation` or no
  violation at all. Found by mutation, 2026-09-16; see meta.yaml.
* `test_reconcile_unknown_is_not_late` separates "the catalogue stages this after the
  decision" from "the catalogue has never heard of it". A `catalogue.get(n) != "application"`
  one-liner merges them and passes every other reconciliation case.
* `test_leak_flags_downward` uses a feature with an AUC of 0.0. Flagging on `auc >= threshold`
  misses perfect separation pointing the other way — which is what a mis-signed join produces.
* `test_baseline_negative_skill` is what stops a clamp at zero. "Worse than predicting the
  base rate" is the single most reportable result the check can produce.
* `test_findings_severity_from_policy` re-grades the same evidence under a second policy. A
  register with its own hard-coded severities passes every other register case and fails this.
* `test_findings_domain_collapses_per_variable` hands the register three raising points on one
  variable and expects ONE finding, reported at the first of them.
"""
import importlib.util, math, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

V = lesson.Variable


# --- exercise 1: observe_feature_use() -----------------------------------------------------

def test_observe_union_across_rows():
    def branchy(row):
        return row["a"] if row["flag"] else row["b"]

    rows = [{"a": 1.0, "b": 2.0, "flag": 1.0}, {"a": 1.0, "b": 2.0, "flag": 0.0}]
    first = lesson.observe_feature_use(branchy, [rows[0]])
    assert first == ("a", "flag"), (
        f"one row gave {first}; the else branch never ran, so 'b' was not read"
    )
    both = lesson.observe_feature_use(branchy, rows)
    assert both == ("a", "b", "flag"), f"the union over both rows should be ('a','b','flag'), got {both}"
    assert isinstance(both, tuple), f"return a tuple, got {type(both).__name__}"


def test_observe_records_get():
    def gets(row):
        return row["here"] + row.get("elsewhere", 0.0)

    seen = lesson.observe_feature_use(gets, [{"here": 1.0, "elsewhere": 2.0, "unread": 3.0}])
    assert seen == ("elsewhere", "here"), (
        f"got {seen} — 'elsewhere' is read through .get() and must be recorded; 'unread' is "
        "in the row and is never read, so it must not be"
    )


def test_observe_skips_raising_row():
    def crashes(row):
        row["first"]
        raise ZeroDivisionError("boom")

    def fine(row):
        return row["later"]

    partial = lesson.observe_feature_use(crashes, [{"first": 1.0, "second": 2.0}])
    assert partial == ("first",), (
        f"got {partial} — the row raised, but 'first' was read before it did"
    )
    mixed = lesson.observe_feature_use(
        lambda r: crashes(r) if r["which"] else fine(r),
        [{"which": 1.0, "first": 1.0, "later": 2.0}, {"which": 0.0, "first": 1.0, "later": 2.0}])
    assert mixed == ("first", "later", "which"), (
        f"got {mixed} — a raising row is skipped, not fatal; the second row still contributes"
    )


def test_observe_sorted_tuple_and_empty():
    seen = lesson.observe_feature_use(lambda r: r["zulu"] + r["alpha"], [{"zulu": 1.0, "alpha": 2.0}])
    assert seen == ("alpha", "zulu"), f"sort the names ascending; got {seen}"
    try:
        lesson.observe_feature_use(lambda r: 0.0, [])
    except ValueError:
        pass
    else:
        raise AssertionError("an empty rows iterable should raise ValueError, not return ()")


# --- exercise 2: reconcile_variables() -----------------------------------------------------

_VARS = (V("a", "increasing", 0.0, 1.0), V("b", "none", 0.0, 1.0), V("c", "decreasing", 0.0, 1.0))
_CAT = {"a": "application", "b": "application", "c": "post_decision", "z": "post_decision"}


def test_reconcile_four_lists():
    r = lesson.reconcile_variables(_VARS, ["a", "c", "z"], _CAT)
    assert r.documented_and_used == ("a", "c"), f"documented_and_used {r.documented_and_used}"
    assert r.documented_not_used == ("b",), f"documented_not_used {r.documented_not_used}"
    assert r.used_not_documented == ("z",), f"used_not_documented {r.used_not_documented}"
    for field in ("documented_and_used", "documented_not_used", "used_not_documented",
                  "used_after_decision", "used_not_in_catalogue"):
        value = getattr(r, field)
        assert isinstance(value, tuple), f"{field} is a {type(value).__name__}, not a tuple"
        assert list(value) == sorted(value), f"{field} is not sorted ascending: {value}"


def test_reconcile_availability_on_used_only():
    r = lesson.reconcile_variables(_VARS, ["a", "c", "z"], _CAT)
    assert r.used_after_decision == ("c", "z"), (
        f"used_after_decision {r.used_after_decision} — both 'c' and 'z' are read and staged "
        "post_decision"
    )
    quiet = lesson.reconcile_variables(_VARS, ["a"], _CAT)
    assert quiet.used_after_decision == (), (
        f"used_after_decision {quiet.used_after_decision} — 'c' and 'z' are staged "
        "post_decision and are NOT READ here; filter on the used names, not on the catalogue"
    )


def test_reconcile_unknown_is_not_late():
    r = lesson.reconcile_variables(_VARS, ["a", "mystery"], {"a": "application"})
    assert r.used_not_in_catalogue == ("mystery",), (
        f"used_not_in_catalogue {r.used_not_in_catalogue}"
    )
    assert r.used_after_decision == (), (
        f"used_after_decision {r.used_after_decision} — 'mystery' has no catalogue entry, so "
        "its stage is UNKNOWN; recording unknown as late is a different finding with a "
        "different owner"
    )
    assert r.used_not_documented == ("mystery",), f"used_not_documented {r.used_not_documented}"


def test_reconcile_rejects_bad_input():
    for bad, why in (
        (((), ["a"], {}), "an empty documented variable list"),
        (((_VARS[0], _VARS[0]), ["a"], {}), "the same variable documented twice"),
        ((_VARS, ["a"], {"a": "later_maybe"}), "a catalogue stage outside AVAILABILITY_STAGES"),
    ):
        try:
            lesson.reconcile_variables(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{why} should raise ValueError")


# --- exercise 3: probe_monotonicity() ------------------------------------------------------

_MVARS = (V("x", "increasing", 0.0, 1.0), V("q", "none", 0.0, 1.0), V("d", "decreasing", 0.0, 1.0))
_MBASE = {"x": 0.0, "q": 0.0, "d": 0.0}


def test_mono_direction_none_not_probed():
    out = lesson.probe_monotonicity(lambda r: 0.5, _MVARS, _MBASE, 3)
    assert set(out) == {"x", "d"}, (
        f"probed {sorted(out)} — a variable documented with no direction has no direction to "
        "contradict and must not appear in the result"
    )


def test_mono_increasing_counterexample():
    # Sweep 0.0, 0.5, 1.0 over min(x, 1-x): scores 0.0, 0.5, 0.0. The second pair falls by 0.5.
    out = lesson.probe_monotonicity(lambda r: min(r["x"], 1.0 - r["x"]), _MVARS, _MBASE, 3)
    p = out["x"]
    assert p.violated is True, "the sweep falls from 0.5 to 0.0 and is documented as increasing"
    assert abs(p.worst_violation - 0.5) < 1e-12, f"worst_violation {p.worst_violation}, expected 0.5"
    assert p.counterexample == (0.5, 1.0, 0.5, 0.0), (
        f"counterexample {p.counterexample}, expected (x_lo, x_hi, score_lo, score_hi) = "
        "(0.5, 1.0, 0.5, 0.0)"
    )
    assert p.direction == "increasing" and p.variable == "x", "carry the name and the direction"
    clean = lesson.probe_monotonicity(lambda r: r["x"], _MVARS, _MBASE, 5)["x"]
    assert clean.violated is False and clean.worst_violation == 0.0 and clean.counterexample == (), (
        f"a strictly increasing sweep reported violated={clean.violated}, "
        f"worst={clean.worst_violation}, counterexample={clean.counterexample}"
    )
    flat = lesson.probe_monotonicity(lambda r: 0.25, _MVARS, _MBASE, 5)["x"]
    assert flat.violated is False, "flat is not a wrong-way move"


def test_mono_decreasing_sign():
    rising = lesson.probe_monotonicity(lambda r: r["d"], _MVARS, _MBASE, 4)
    assert rising["d"].violated is True, (
        "a sweep that RISES contradicts a documented decreasing direction"
    )
    assert abs(rising["d"].worst_violation - 1.0 / 3.0) < 1e-12, (
        f"worst_violation {rising['d'].worst_violation} — the grid steps by 1/3, so the worst "
        "wrong-way move for a decreasing variable is score_hi - score_lo = 1/3"
    )
    assert rising["x"].violated is False, "the same sweep rises, which is what 'x' documents"


def test_mono_skips_raising_points():
    def raises_low(row):
        if row["x"] < 0.4:
            raise ZeroDivisionError("boom")
        return row["x"]

    p = lesson.probe_monotonicity(raises_low, _MVARS, _MBASE, 5)["x"]
    assert p.n_skipped == 2, (
        f"n_skipped {p.n_skipped} — the grid is 0.0, 0.25, 0.5, 0.75, 1.0 and the first two raise"
    )
    assert p.violated is False, (
        "a point that raised is the domain prober's finding; it is not a monotonicity "
        "counterexample, and the surviving points rise"
    )

    def nan_high(row):
        return float("nan") if row["x"] > 0.6 else row["x"]

    q = lesson.probe_monotonicity(nan_high, _MVARS, _MBASE, 5)["x"]
    assert q.n_skipped == 2 and q.violated is False, (
        f"n_skipped {q.n_skipped}, violated {q.violated} — a non-finite return is skipped too; "
        "comparing against nan silently reports no violation anywhere"
    )

    # Counting a point in n_skipped is not the same as dropping it from the sequence. Here the
    # unevaluable point sits in the MIDDLE of the grid and the pair that brackets it is the only
    # wrong-way move there is, so a probe that leaves the point in the comparison — as a
    # substituted 0.0, or as the nan itself — reports a different answer from one that drops it.
    _bracket = {0.0: 0.0, 0.25: 0.9, 0.75: 0.1, 1.0: 0.2}

    def raises_middle(row):
        x = round(row["x"], 6)
        if x == 0.5:
            raise ZeroDivisionError("boom")
        return _bracket[x]

    def nan_middle(row):
        x = round(row["x"], 6)
        return float("nan") if x == 0.5 else _bracket[x]

    for fn, how in ((raises_middle, "raised"), (nan_middle, "returned nan")):
        r = lesson.probe_monotonicity(fn, _MVARS, _MBASE, 5)["x"]
        assert r.n_skipped == 1, (
            f"n_skipped {r.n_skipped} — exactly one grid point, 0.5, {how}"
        )
        assert r.violated is True and abs(r.worst_violation - 0.8) < 1e-9, (
            f"the point at 0.5 {how}; the probe reported violated={r.violated}, "
            f"worst_violation={r.worst_violation} — DROP that point from the sequence and "
            "compare 0.25 to 0.75 directly: 0.9 falling to 0.1 is a wrong-way move of 0.8 on a "
            "variable documented increasing. Counting the point in n_skipped and then leaving "
            "it in the comparison hides that pair"
        )
        assert r.counterexample == (0.25, 0.75, 0.9, 0.1), (
            f"counterexample {r.counterexample} — the worst pair is the one that BRACKETS the "
            "skipped point, (x_lo, x_hi, score_lo, score_hi) = (0.25, 0.75, 0.9, 0.1); a pair "
            "naming 0.5, or a score of 0.0 or nan, means the skipped point is still in there"
        )


def test_mono_does_not_mutate_base_row():
    base = {"x": 0.0, "q": 0.0, "d": 0.0}
    lesson.probe_monotonicity(lambda r: r["x"], _MVARS, base, 4)
    assert base == {"x": 0.0, "q": 0.0, "d": 0.0}, (
        f"base_row came back as {base} — build a copy per grid point"
    )


def test_mono_rejects_bad_input():
    for bad, why in (
        ((lambda r: 0.5, _MVARS, _MBASE, 1), "n_grid of 1"),
        ((lambda r: 0.5, _MVARS, {"q": 0.0}, 3), "a probed variable missing from base_row"),
        ((lambda r: 0.5, (V("x", "increasing", 1.0, 1.0),), _MBASE, 3), "lo == hi"),
    ):
        try:
            lesson.probe_monotonicity(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{why} should raise ValueError")


# --- exercise 4: probe_domain() ------------------------------------------------------------

_DVARS = (V("x", "none", 0.0, 1.0),)


def test_domain_raised_vs_impossible():
    out = lesson.probe_domain(lambda r: 1.0 / r["x"], _DVARS, {"x": 0.5}, 3)
    assert len(out.failures) == 2, (
        f"{len(out.failures)} failures — 1/0 raises, 1/0.5 = 2.0 is not a probability, "
        "1/1.0 = 1.0 is"
    )
    first, second = out.failures
    assert first.kind == "raised" and first.exception == "ZeroDivisionError", (
        f"got kind={first.kind!r} exception={first.exception!r} — record the exception class name"
    )
    assert math.isnan(first.returned), "a call that raised has no return value; report nan"
    assert second.kind == "impossible" and second.exception == "", (
        f"got kind={second.kind!r} exception={second.exception!r}"
    )
    assert abs(second.returned - 2.0) < 1e-12, f"returned {second.returned}, expected 2.0"
    assert abs(second.value - 0.5) < 1e-12, f"value {second.value}, expected the input 0.5"
    nans = lesson.probe_domain(lambda r: float("nan"), _DVARS, {"x": 0.5}, 2)
    assert [f.kind for f in nans.failures] == ["impossible", "impossible"], (
        "nan is not a finite number in [0, 1]"
    )


def test_domain_counts_every_call():
    two = (V("x", "none", 0.0, 1.0), V("y", "increasing", 0.0, 1.0))
    ok = lesson.probe_domain(lambda r: 0.5, two, {"x": 0.5, "y": 0.5}, 7)
    assert ok.n_tested == 14, (
        f"n_tested {ok.n_tested} — two variables at 7 grid points is 14 calls, whether they "
        "failed or not"
    )
    assert ok.failures == (), f"a constant 0.5 is a probability everywhere; got {ok.failures}"
    bad = lesson.probe_domain(lambda r: 2.0, two, {"x": 0.5, "y": 0.5}, 7)
    assert bad.n_tested == 14, f"n_tested {bad.n_tested} — count the calls, not the successes"


def test_domain_bounds_inclusive():
    out = lesson.probe_domain(lambda r: r["x"], _DVARS, {"x": 0.5}, 3)
    assert out.failures == (), (
        f"got {out.failures} — 0.0 and 1.0 ARE probabilities; the bounds are inclusive"
    )
    below = lesson.probe_domain(lambda r: r["x"] - 1e-9, _DVARS, {"x": 0.5}, 3)
    assert len(below.failures) == 1 and abs(below.failures[0].value) < 1e-12, (
        f"got {below.failures} — a hair below zero is not a probability, and only the first "
        "grid point is"
    )


def test_domain_probe_order():
    two = (V("x", "none", 0.0, 1.0), V("w", "increasing", 0.0, 1.0))
    out = lesson.probe_domain(lambda r: -1.0, two, {"x": 0.5, "w": 0.5}, 2)
    assert [f.variable for f in out.failures] == ["x", "x", "w", "w"], (
        f"order {[f.variable for f in out.failures]} — variables in the order given, not sorted"
    )
    assert [f.value for f in out.failures[:2]] == [0.0, 1.0], (
        f"values {[f.value for f in out.failures[:2]]} — ascending within a sweep"
    )


def test_domain_probes_all_directions_and_rejects():
    # 'q' has no documented direction. A domain is documented whether a shape is or not.
    vs = (V("q", "none", 0.0, 1.0),)
    out = lesson.probe_domain(lambda r: 3.0, vs, {"q": 0.5}, 4)
    assert out.n_tested == 4 and len(out.failures) == 4, (
        f"n_tested {out.n_tested}, failures {len(out.failures)} — sweep every documented "
        "variable, direction or no direction"
    )
    base = {"q": 0.5}
    lesson.probe_domain(lambda r: 0.5, vs, base, 3)
    assert base == {"q": 0.5}, f"base_row came back as {base} — copy it per grid point"
    for bad, why in (((lambda r: 0.5, vs, {"q": 0.5}, 1), "n_grid of 1"),
                     ((lambda r: 0.5, vs, {}, 3), "a variable missing from base_row")):
        try:
            lesson.probe_domain(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{why} should raise ValueError")


# --- exercise 5: scan_leakage() ------------------------------------------------------------

def test_leak_auc_values():
    y = np.array([0, 0, 1, 1])
    s = lesson.scan_leakage({"tell": np.array([0.0, 0.0, 1.0, 1.0]),
                             "noise": np.array([1.0, 0.0, 1.0, 0.0])}, y, 0.95)
    assert abs(s.auc["tell"] - 1.0) < 1e-12, f"the perfect column should score 1.0, got {s.auc['tell']}"
    assert abs(s.auc["noise"] - 0.5) < 1e-12, f"the noise column should score 0.5, got {s.auc['noise']}"
    assert type(s.auc["tell"]) is float, (
        f"auc values should be plain floats, got {type(s.auc['tell']).__name__} — np.float64 "
        "passes isinstance(x, float) and still stops json.dumps"
    )
    assert s.flagged == ("tell",), f"flagged {s.flagged}"


def test_leak_flags_downward():
    y = np.array([0, 0, 1, 1])
    s = lesson.scan_leakage({"upside_down": np.array([1.0, 1.0, 0.0, 0.0])}, y, 0.95)
    assert abs(s.auc["upside_down"]) < 1e-12, (
        f"report the AUC in the feature's own direction, which is 0.0 here, not "
        f"{s.auc['upside_down']}"
    )
    assert s.flagged == ("upside_down",), (
        f"flagged {s.flagged} — an AUC of 0.0 is perfect separation pointing downwards; flag "
        "on max(auc, 1 - auc), not on auc"
    )


def test_leak_threshold_inclusive_and_order():
    # Worked by hand: neg values 0,1,2,3 and pos values 2.5,2.5,4,5 give 14 wins of 16 = 0.875.
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    edge = np.array([0.0, 1.0, 2.0, 3.0, 2.5, 2.5, 4.0, 5.0])
    s = lesson.scan_leakage({"edge": edge}, y, 0.875)
    assert abs(s.auc["edge"] - 0.875) < 1e-12, f"the hand-worked AUC is 0.875, got {s.auc['edge']}"
    assert s.flagged == ("edge",), (
        f"flagged {s.flagged} — a feature landing exactly on the threshold is flagged; the "
        "rule is a minimum, not a strict inequality"
    )
    # 'zulu' and 'alpha' separate perfectly (strength 1.0); 'weaker' scores 0.875 by hand.
    ordered = lesson.scan_leakage({"zulu": np.array([0.0, 0.0, 1.0, 1.0]),
                                  "alpha": np.array([0.0, 0.0, 1.0, 1.0]),
                                  "weaker": np.array([0.0, 1.0, 1.0, 2.0])},
                                 np.array([0, 0, 1, 1]), 0.6)
    assert ordered.flagged[:2] == ("alpha", "zulu"), (
        f"flagged {ordered.flagged} — strongest first, and two features of equal strength are "
        "ordered alphabetically so the register reads the same way twice"
    )
    assert ordered.flagged[2] == "weaker", f"flagged {ordered.flagged} — the weaker one comes last"


def test_leak_rejects_bad_input():
    y = np.array([0, 0, 1, 1])
    for bad, why in ((({}, y, 0.95), "no features at all"),
                     (({"a": np.array([1.0, 2.0])}, y, 0.95), "a column of the wrong length"),
                     (({"a": np.zeros(4)}, np.array([0, 0, 0, 0]), 0.95), "a single class"),
                     (({"a": np.zeros(4)}, np.array([0, 0, 1, 2]), 0.95), "a label of 2"),
                     (({"a": np.zeros(4)}, y, 0.4), "a threshold of 0.4"),
                     (({"a": np.zeros(4)}, y, 1.5), "a threshold above 1.0")):
        try:
            lesson.scan_leakage(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{why} should raise ValueError")


# --- exercise 6: duplicate_contamination() -------------------------------------------------

def test_contam_counts_and_indices():
    tr = np.array([[1.0, 2.0], [3.0, 4.0]])
    te = np.array([[3.0, 4.0], [5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])
    c = lesson.duplicate_contamination(tr, te)
    assert c.n_test == 4, f"n_test {c.n_test}"
    assert c.n_in_train == 3, f"n_in_train {c.n_in_train} — rows 0, 2 and 3 are in the training set"
    assert c.indices == (0, 2, 3), f"indices {c.indices} — every index, ascending"
    assert abs(c.share - 0.75) < 1e-12 and type(c.share) is float, f"share {c.share!r}, expected 0.75"


def test_contam_exact_match_only():
    near = lesson.duplicate_contamination(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0000001]]))
    assert near.n_in_train == 0 and near.indices == (), (
        "matching is exact — deciding how close counts as the same record is not this "
        "function's job, and a tolerance turns a count into an opinion"
    )
    partial = lesson.duplicate_contamination(np.array([[1.0, 2.0]]), np.array([[1.0, 9.0]]))
    assert partial.n_in_train == 0, "a row matching on one column of two is not a duplicate row"


def test_contam_repeats_within_test():
    thrice = lesson.duplicate_contamination(np.array([[9.0]]), np.array([[7.0], [7.0], [7.0]]))
    assert thrice.repeats_within_test == 2, (
        f"repeats_within_test {thrice.repeats_within_test} — three identical rows are two "
        "repeats of the first, not three"
    )
    assert thrice.n_in_train == 0, "none of them is in the training extract"
    none = lesson.duplicate_contamination(np.array([[9.0]]), np.array([[7.0], [8.0]]))
    assert none.repeats_within_test == 0, f"got {none.repeats_within_test} on distinct rows"


def test_contam_rejects_bad_input():
    tr = np.array([[1.0, 2.0]])
    te = np.array([[1.0, 2.0]])
    for a, b, why in ((np.array([1.0, 2.0]), te, "a 1-D training extract"),
                      (tr, np.array([1.0, 2.0]), "a 1-D test extract"),
                      (tr, np.array([[1.0]]), "mismatched column counts"),
                      (tr, np.empty((0, 2)), "an empty test extract")):
        try:
            lesson.duplicate_contamination(a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{why} should raise ValueError")


# --- exercise 7: baseline_margin_check() ---------------------------------------------------

def test_baseline_brier_and_skill():
    # Base rate 0.5; baseline Brier 0.25. Model predicts 0.25 and 0.75: Brier 0.0625.
    c = lesson.baseline_margin_check(np.array([0, 1]), np.array([0.25, 0.75]), 0.5)
    assert abs(c.brier_baseline - 0.25) < 1e-12, f"brier_baseline {c.brier_baseline}, expected 0.25"
    assert abs(c.brier_model - 0.0625) < 1e-12, f"brier_model {c.brier_model}, expected 0.0625"
    assert abs(c.skill - 0.75) < 1e-12, f"skill {c.skill}, expected 1 - 0.0625/0.25 = 0.75"
    assert c.passed is True, "0.75 clears a margin of 0.5"
    flat = lesson.baseline_margin_check(np.array([0, 1]), np.array([0.5, 0.5]), 0.1)
    assert abs(flat.skill) < 1e-12 and flat.passed is False, (
        f"skill {flat.skill} — predicting the base rate IS the baseline, so the skill is 0.0"
    )


def test_baseline_negative_skill():
    c = lesson.baseline_margin_check(np.array([0, 0, 1, 1]), np.array([1.0, 1.0, 0.0, 0.0]), 0.1)
    assert abs(c.brier_baseline - 0.25) < 1e-12, f"brier_baseline {c.brier_baseline}, expected 0.25"
    assert abs(c.skill - (-3.0)) < 1e-12, (
        f"skill {c.skill} — a Brier of 1.0 against a baseline of 0.25 is a skill of -3.0; "
        "clamping a negative skill at zero hides the most reportable result this check has"
    )
    assert c.passed is False, "a model worse than the base rate does not pass any margin"


def test_baseline_margin_inclusive_and_types():
    c = lesson.baseline_margin_check(np.array([0, 1]), np.array([0.0, 1.0]), 1.0)
    assert c.passed is True, (
        f"skill {c.skill} against a margin of 1.0 — landing exactly on the margin passes it"
    )
    assert abs(c.min_margin - 1.0) < 1e-12, f"carry the margin back: min_margin {c.min_margin}"
    for field in ("brier_model", "brier_baseline", "skill", "min_margin"):
        assert type(getattr(c, field)) is float, (
            f"{field} is a {type(getattr(c, field)).__name__}; return plain Python floats"
        )


def test_baseline_rejects_bad_input():
    for bad, why in (
        ((np.array([0, 1]), np.array([0.5]), 0.1), "mismatched lengths"),
        ((np.array([0, 2]), np.array([0.5, 0.5]), 0.1), "a label of 2"),
        ((np.array([1, 1]), np.array([0.5, 0.5]), 0.1), "a single class"),
        ((np.array([0, 1]), np.array([0.5, 1.5]), 0.1), "a prediction above 1"),
        ((np.array([0, 1]), np.array([0.5, 0.5]), -0.1), "a negative margin"),
    ):
        try:
            lesson.baseline_margin_check(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{why} should raise ValueError")


# --- exercise 8: soundness_findings() ------------------------------------------------------

def _bundle():
    """An evidence bundle built here, from the lesson's types, with unmistakable figures."""
    rec = lesson.Reconciliation(documented_and_used=("kept",), documented_not_used=("ghost",),
                                used_not_documented=("stowaway",),
                                used_after_decision=("stowaway",),
                                used_not_in_catalogue=("mystery",))
    mono = {"straight": lesson.MonotonicityProbe("straight", "increasing", False, 0.0, (), 0),
            "bent": lesson.MonotonicityProbe("bent", "decreasing", True, 0.345678,
                                             (1.0, 2.0, 0.1, 0.445678), 0)}
    dom = lesson.DomainProbe(n_tested=9, failures=(
        lesson.DomainFailure("blows_up", 12.5, "raised", "ZeroDivisionError", float("nan")),
        lesson.DomainFailure("blows_up", 13.5, "raised", "OverflowError", float("nan")),
        lesson.DomainFailure("blows_up", 14.5, "raised", "ZeroDivisionError", float("nan")),
        lesson.DomainFailure("silly", 7.25, "impossible", "", 1.0625),
        lesson.DomainFailure("silly", 7.75, "impossible", "", 9.5),
    ))
    leak = lesson.LeakageScan(auc={"tell": 0.9876, "quiet": 0.51}, flagged=("tell",))
    contam = lesson.Contamination(n_test=1000, n_in_train=37, share=0.037,
                                  indices=tuple(range(37)), repeats_within_test=0)
    base = lesson.BaselineCheck(brier_model=0.2, brier_baseline=0.21, skill=0.0476,
                                min_margin=0.1000, passed=False)
    return {"reconciliation": rec, "monotonicity": mono, "domain": dom, "leakage": leak,
            "contamination": contam, "baseline": base}


_POLICY = {"max_contamination_share": 0.01,
           "severity": {"VAR-UNDOCUMENTED": "high", "VAR-UNUSED": "medium", "AVAIL": "critical",
                        "AVAIL-UNKNOWN": "medium", "MONO": "high", "DOMAIN-RAISED": "critical",
                        "DOMAIN-IMPOSSIBLE": "critical", "LEAK": "critical", "CONTAM": "high",
                        "BASELINE": "high"}}


def test_findings_one_per_rule():
    reg = lesson.soundness_findings(_bundle(), _POLICY)
    assert isinstance(reg, tuple), f"return a tuple, got {type(reg).__name__}"
    rules = sorted(f.rule for f in reg)
    assert rules == sorted(["VAR-UNDOCUMENTED", "VAR-UNUSED", "AVAIL", "AVAIL-UNKNOWN", "MONO",
                            "DOMAIN-RAISED", "DOMAIN-IMPOSSIBLE", "LEAK", "CONTAM", "BASELINE"]), (
        f"rules raised: {rules} — each family in the table fires exactly once on this bundle"
    )
    by_rule = {f.rule: f for f in reg}
    assert by_rule["MONO"].subject == "bent", (
        f"MONO subject {by_rule['MONO'].subject!r} — 'straight' did not violate anything and "
        "must not appear"
    )
    assert by_rule["LEAK"].subject == "tell", (
        f"LEAK subject {by_rule['LEAK'].subject!r} — only the FLAGGED features are findings; "
        "'quiet' has an AUC of 0.51 and is not one"
    )
    assert by_rule["CONTAM"].subject == "" and by_rule["BASELINE"].subject == "", (
        "a register-level finding has an empty subject"
    )
    for f in reg:
        assert isinstance(f.detail, str) and f.detail.strip(), f"{f.rule} has an empty detail"


def test_findings_domain_collapses_per_variable():
    reg = lesson.soundness_findings(_bundle(), _POLICY)
    raised = [f for f in reg if f.rule == "DOMAIN-RAISED"]
    impossible = [f for f in reg if f.rule == "DOMAIN-IMPOSSIBLE"]
    assert len(raised) == 1 and raised[0].subject == "blows_up", (
        f"{len(raised)} DOMAIN-RAISED findings — three raising points on one variable are ONE "
        "finding; a register with a row per grid point is unreadable"
    )
    assert len(impossible) == 1 and impossible[0].subject == "silly", (
        f"{len(impossible)} DOMAIN-IMPOSSIBLE findings, expected one for 'silly'"
    )
    assert "12.5" in raised[0].detail and "ZeroDivisionError" in raised[0].detail, (
        f"detail {raised[0].detail!r} — report the FIRST failing point, 12.5, and what it raised"
    )
    assert "13.5" not in raised[0].detail and "14.5" not in raised[0].detail, (
        f"detail {raised[0].detail!r} — the first failing point, not all of them"
    )
    assert "1.0625" in impossible[0].detail and "7.25" in impossible[0].detail, (
        f"detail {impossible[0].detail!r} — the first impossible return and the input that "
        "produced it"
    )


def test_findings_severity_from_policy():
    base = {f.rule: f.severity for f in lesson.soundness_findings(_bundle(), _POLICY)}
    assert base["AVAIL"] == "critical" and base["VAR-UNUSED"] == "medium", (
        f"severities came back {base} — read them from policy['severity']"
    )
    moved = dict(_POLICY, severity=dict(_POLICY["severity"], MONO="low", AVAIL="medium",
                                        LEAK="high"))
    second = {f.rule: f.severity for f in lesson.soundness_findings(_bundle(), moved)}
    assert (second["MONO"], second["AVAIL"], second["LEAK"]) == ("low", "medium", "high"), (
        f"under a different policy the register still said {second} — every severity comes "
        "from policy['severity'], never from a constant in your own code"
    )
    stripped = dict(_POLICY, severity={k: v for k, v in _POLICY["severity"].items()
                                       if k != "MONO"})
    try:
        lesson.soundness_findings(_bundle(), stripped)
    except ValueError:
        pass
    else:
        raise AssertionError("a rule that fired with no severity in policy should raise "
                             "ValueError, not fall back to a default severity")


def test_findings_sort_order():
    reg = lesson.soundness_findings(_bundle(), _POLICY)
    keys = [(lesson.SEVERITY_ORDER.index(f.severity), f.rule, f.subject) for f in reg]
    assert keys == sorted(keys), (
        f"the register came out in this order: {[(f.severity, f.rule, f.subject) for f in reg]} "
        "— sort by severity (SEVERITY_ORDER, most severe first), then rule, then subject"
    )
    assert lesson.SEVERITY_ORDER.index(reg[0].severity) == 0, (
        f"the register opens with a {reg[0].severity} finding; a critical one exists"
    )
    # Sorting the severity STRINGS puts critical, high, medium in the right order by luck of
    # the alphabet. 'low' is where luck runs out: it ranks last, and it sorts third.
    graded = dict(_POLICY, severity=dict(_POLICY["severity"], MONO="low", VAR_UNUSED_IGNORED=""))
    graded["severity"]["VAR-UNUSED"] = "medium"
    ordered = [f for f in lesson.soundness_findings(_bundle(), graded)
               if f.rule in ("MONO", "VAR-UNUSED")]
    assert [f.rule for f in ordered] == ["VAR-UNUSED", "MONO"], (
        f"a 'medium' and a 'low' finding came out as {[(f.rule, f.severity) for f in ordered]} "
        "— 'low' ranks BELOW 'medium' in SEVERITY_ORDER and above it in the alphabet; rank by "
        "SEVERITY_ORDER.index(severity), never by the string"
    )


def test_findings_details_carry_figures():
    by_rule = {f.rule: f for f in lesson.soundness_findings(_bundle(), _POLICY)}
    assert "0.345678" in by_rule["MONO"].detail, (
        f"MONO detail {by_rule['MONO'].detail!r} — carry the worst violation to six decimals"
    )
    assert "decreasing" in by_rule["MONO"].detail, (
        f"MONO detail {by_rule['MONO'].detail!r} — say which direction was documented"
    )
    assert "0.9876" in by_rule["LEAK"].detail, (
        f"LEAK detail {by_rule['LEAK'].detail!r} — carry the AUC to four decimals"
    )
    assert "0.0370" in by_rule["CONTAM"].detail and "0.0100" in by_rule["CONTAM"].detail, (
        f"CONTAM detail {by_rule['CONTAM'].detail!r} — carry the share and the tolerance, "
        "both to four decimals"
    )
    assert "0.0476" in by_rule["BASELINE"].detail and "0.1000" in by_rule["BASELINE"].detail, (
        f"BASELINE detail {by_rule['BASELINE'].detail!r} — carry the skill and the margin"
    )
    assert "stowaway" in by_rule["AVAIL"].detail, (
        f"AVAIL detail {by_rule['AVAIL'].detail!r} — name the feature"
    )


def test_findings_missing_keys_and_clean():
    clean = {"reconciliation": lesson.Reconciliation((), (), (), (), ()), "monotonicity": {},
             "domain": lesson.DomainProbe(0, ()), "leakage": lesson.LeakageScan({}, ()),
             "contamination": lesson.Contamination(10, 0, 0.0, (), 0),
             "baseline": lesson.BaselineCheck(0.1, 0.2, 0.5, 0.1, True)}
    assert lesson.soundness_findings(clean, _POLICY) == (), (
        "a review that found nothing returns an empty register; an empty register is a result"
    )
    # Exactly on the tolerance is not a breach: the rule is strictly greater.
    on_limit = dict(clean, contamination=lesson.Contamination(1000, 10, 0.01, tuple(range(10)), 0))
    assert lesson.soundness_findings(on_limit, _POLICY) == (), (
        "a contamination share exactly equal to the tolerance is within it; the rule is "
        "share > tolerance"
    )
    try:
        lesson.soundness_findings({"reconciliation": lesson.Reconciliation((), (), (), (), ())},
                                  _POLICY)
    except ValueError as exc:
        for key in ("monotonicity", "domain", "leakage", "contamination", "baseline"):
            assert key in str(exc), (
                f"the error said {str(exc)!r} — name EVERY missing key in one message, so a "
                "caller fixes the bundle once rather than five times"
            )
    else:
        raise AssertionError("an incomplete evidence bundle should raise ValueError")


RUBRIC = [
    (test_observe_union_across_rows, 4,
     "run fn on every row and union the keys; a branch not taken reads nothing"),
    (test_observe_records_get, 3,
     "override .get() as well as __getitem__ — a .get(name, default) is a read"),
    (test_observe_skips_raising_row, 3,
     "wrap the call in try/except and carry on; the keys read before the raise are still read"),
    (test_observe_sorted_tuple_and_empty, 2,
     "tuple(sorted(seen)), and ValueError when there is nothing to run on"),
    (test_reconcile_four_lists, 4,
     "set operations on the two name sets, each result sorted into a tuple"),
    (test_reconcile_availability_on_used_only, 4,
     "filter the USED names by their catalogue stage, not the catalogue by anything"),
    (test_reconcile_unknown_is_not_late, 3,
     "`n not in catalogue` is its own bucket; do not let catalogue.get(n) != 'application' "
     "swallow it"),
    (test_reconcile_rejects_bad_input, 3,
     "check for an empty list, a repeated name, and a stage outside AVAILABILITY_STAGES"),
    (test_mono_direction_none_not_probed, 3,
     "skip any variable whose direction is not 'increasing' or 'decreasing'"),
    (test_mono_increasing_counterexample, 4,
     "compare adjacent points: for 'increasing' the violation is score_lo - score_hi"),
    (test_mono_decreasing_sign, 3,
     "for 'decreasing' the violation is score_hi - score_lo — the other way round"),
    (test_mono_skips_raising_points, 4,
     "try/except around fn and math.isfinite on the result; count and drop those points"),
    (test_mono_does_not_mutate_base_row, 2,
     "row = dict(base_row) inside the loop, never base_row[v.name] = x"),
    (test_mono_rejects_bad_input, 2,
     "n_grid < 2, a name missing from base_row, and lo >= hi are all ValueError"),
    (test_domain_raised_vs_impossible, 4,
     "two kinds: 'raised' carries type(exc).__name__, 'impossible' carries the value"),
    (test_domain_counts_every_call, 3,
     "n_tested counts calls made — len(variables) * n_grid — not failures found"),
    (test_domain_bounds_inclusive, 3,
     "the test is score < 0.0 or score > 1.0; 0.0 and 1.0 are probabilities"),
    (test_domain_probe_order, 3,
     "iterate variables in the order given and the grid ascending; do not sort afterwards"),
    (test_domain_probes_all_directions_and_rejects, 3,
     "probe_domain has no direction filter; copy base_row; raise on n_grid < 2 and a missing key"),
    (test_leak_auc_values, 4,
     "auc_by_ranks per column, wrapped in float()"),
    (test_leak_flags_downward, 4,
     "flag on max(auc, 1 - auc) >= threshold; report auc itself unchanged"),
    (test_leak_threshold_inclusive_and_order, 4,
     "sort by (-strength, name); >= not > on the threshold"),
    (test_leak_rejects_bad_input, 2,
     "empty features, a wrong-length column, a single class, a bad label, a threshold "
     "outside (0.5, 1.0]"),
    (test_contam_counts_and_indices, 4,
     "a set of row tuples from train, then one pass over test collecting indices"),
    (test_contam_exact_match_only, 3,
     "tuple equality on the whole row — no np.isclose, no per-column tolerance"),
    (test_contam_repeats_within_test, 3,
     "a second set, grown as you go: a row repeats only if it was seen EARLIER in test"),
    (test_contam_rejects_bad_input, 3,
     "check ndim == 2 on both, equal column counts, and a non-empty test extract"),
    (test_baseline_brier_and_skill, 4,
     "brier = mean((p - y) ** 2); the baseline predicts y.mean() for everyone"),
    (test_baseline_negative_skill, 4,
     "skill = 1 - brier_model / brier_baseline, unclamped"),
    (test_baseline_margin_inclusive_and_types, 3,
     "passed is skill >= min_margin; wrap every number in float()"),
    (test_baseline_rejects_bad_input, 2,
     "lengths, labels, a single class, probabilities outside [0, 1], a negative margin"),
    (test_findings_one_per_rule, 5,
     "walk the six evidence objects and raise a finding for each failure they report"),
    (test_findings_domain_collapses_per_variable, 4,
     "group the domain failures by (variable, kind) and keep the first of each"),
    (test_findings_severity_from_policy, 4,
     "severity = policy['severity'][rule], with a ValueError when the key is absent"),
    (test_findings_sort_order, 3,
     "sort key: (SEVERITY_ORDER.index(severity), rule, subject)"),
    (test_findings_details_carry_figures, 4,
     "format the figures exactly as the docstring's table says; the rest of the wording is yours"),
    (test_findings_missing_keys_and_clean, 2,
     "collect the missing keys in one list; CONTAM fires on share > tolerance, strictly"),
]
