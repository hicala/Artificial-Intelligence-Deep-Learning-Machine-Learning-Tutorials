"""Autograder rubric for P04-L01. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson, and are deliberate:

* every expected value is computed HERE, by an independent route — a pure-Python reference
  (`_ece_reference`, `_psi_reference`) or a closed form worked out by hand. The rubric never
  grades a student against their own code, and it never hard-codes a figure produced by the
  notebook's synthetic sample, which would turn a data tweak into a grading failure.
* the exercises chain (ECE builds on the reliability table, the subgroup table builds on ECE),
  so each exercise's own properties are graded by cases that do NOT depend on the exercise
  before it. A broken exercise 2 must not cost a student the four exercise-4 cases that are
  about support rules and signs.
* `test_ece_ignores_empty_bins` is what stops `sum(weights * gaps)` taken over ALL bins from
  scoring full marks: an empty bin's gap is `nan`, and `0 * nan` is `nan`, not 0.
* `test_psi_floor_is_two_sided` injects a bin that empties out AND a bin that appears from
  nothing. Flooring only one side leaves the other producing an infinity, and an infinite PSI
  in a monitoring pack reads as a broken tool rather than as the finding it is.
* `test_decision_reject_precedence` hands the challenger a spectacular AUC gain alongside a
  blown ECE ceiling. Any rule engine that scores improvement before absolute gates passes
  every other decision case and fails this one.
* `test_report_uses_findings_only` renders a findings dict whose figures are unmistakable
  (0.4242, 0.7777, 0.6161) and asserts they reach the page. A `render_validation_report` that
  recomputes anything from module-level data cannot produce them.
* `test_report_psi_verdict_boundary` pins the inclusive boundary: a PSI exactly on the policy
  threshold is WITHIN it. `>=` there would fail a model that is precisely at its limit, which
  is the one case a validator will be asked to defend.
"""
import importlib.util, math, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)


# --- independent references, so nothing is graded against the student's own answer ---------

def _ece_reference(y, p, n_bins):
    """Pure-Python ECE. Deliberately written a different way from the numpy exercise."""
    buckets = {}
    for label, prob in zip(y, p):
        m = min(int(math.ceil(prob * n_bins)) - 1, n_bins - 1)
        m = max(m, 0)
        buckets.setdefault(m, []).append((float(label), float(prob)))
    total = len(list(y))
    out = 0.0
    for rows in buckets.values():
        obs = sum(r[0] for r in rows) / len(rows)
        pred = sum(r[1] for r in rows) / len(rows)
        out += (len(rows) / total) * abs(obs - pred)
    return out


def _psi_reference(expected, actual, edges, floor=1e-6):
    """Pure-Python PSI. Same definition, none of the same code."""
    edges = [float(e) for e in edges]
    n_bins = len(edges) - 1

    def shares(values):
        counts = [0] * n_bins
        for v in values:
            slot = n_bins - 1
            for i in range(n_bins):
                if edges[i] <= v < edges[i + 1]:
                    slot = i
                    break
            else:
                slot = 0 if v < edges[0] else n_bins - 1
            counts[slot] += 1
        return [c / len(values) for c in counts]

    e, a = shares(expected), shares(actual)
    return sum((max(ai, floor) - max(ei, floor)) * math.log(max(ai, floor) / max(ei, floor))
               for ei, ai in zip(e, a))


POLICY = {"min_auc": 0.65, "max_ece": 0.04, "min_auc_gain": 0.02}


def _probe():
    """A findings dict whose every figure is unmistakable, built here and not in the lesson."""
    rel = lesson.ReliabilityTable(
        lo=np.array([0.0, 0.5]), hi=np.array([0.5, 1.0]), count=np.array([7, 0]),
        mean_pred=np.array([0.1313, np.nan]), obs_rate=np.array([0.2424, np.nan]))
    psi = lesson.StabilityResult(psi=0.7777, contributions=np.array([0.1000, 0.6777]),
                                 expected_pct=np.array([0.5, 0.5]),
                                 actual_pct=np.array([0.2, 0.8]))
    sub = {"group": np.array(["alpha", "omega"]), "count": np.array([900, 11]),
           "event_rate": np.array([0.3131, np.nan]), "mean_pred": np.array([0.3535, np.nan]),
           "gap": np.array([0.0404, np.nan]), "ece": np.array([0.0505, 0.9191]),
           "reportable": np.array([True, False])}
    dec = lesson.Decision(verdict="hold",
                          reasons=("PASS probe rule zero", "PASS probe rule one",
                                   "FAIL probe rule two", "PASS probe rule three"),
                          margins={"auc_gain": 0.0202})
    return {"model_id": "probe-model", "as_of": "1970-01-01",
            "data_note": "PROBE DATA NOTE, SYNTHETIC", "reliability": rel, "ece": 0.4242,
            "psi": psi, "psi_threshold": 0.25, "subgroups": sub, "min_support": 200,
            "champion": {"auc": 0.6161, "ece": 0.5252},
            "challenger": {"auc": 0.8181, "ece": 0.0303}, "decision": dec}


# --- exercise 1: reliability_table() -------------------------------------------------------

def test_reliability_shape_and_edges():
    t = lesson.reliability_table(np.array([0, 1, 1, 0]), np.array([0.05, 0.15, 0.95, 0.45]), 10)
    assert isinstance(t, lesson.ReliabilityTable), f"expected a ReliabilityTable, got {type(t).__name__}"
    for field in ("lo", "hi", "count", "mean_pred", "obs_rate"):
        assert len(getattr(t, field)) == 10, (
            f"{field} has {len(getattr(t, field))} entries for n_bins=10; every bin gets a row"
        )
    assert np.allclose(t.lo, np.linspace(0, 1, 11)[:-1]), f"lower edges wrong: {t.lo}"
    assert np.allclose(t.hi, np.linspace(0, 1, 11)[1:]), f"upper edges wrong: {t.hi}"
    assert t.count.tolist() == [1, 1, 0, 0, 1, 0, 0, 0, 0, 1], f"counts wrong: {t.count.tolist()}"
    assert int(t.count.sum()) == 4, "every record lands in exactly one bin"


def test_reliability_empty_bins_are_nan():
    t = lesson.reliability_table(np.array([1, 1]), np.array([0.05, 0.06]), 10)
    assert int(t.count[0]) == 2 and int(t.count[4]) == 0
    assert np.isnan(t.obs_rate[4]), (
        f"an empty bin reported an observed rate of {t.obs_rate[4]} — a rate of 0.0 is a "
        "finding, and you observed nothing up there; report nan"
    )
    assert np.isnan(t.mean_pred[4]), f"an empty bin's mean prediction is nan, got {t.mean_pred[4]}"
    assert np.isfinite(t.obs_rate[0]) and np.isfinite(t.mean_pred[0]), (
        "a bin that HAS records must report real numbers"
    )


def test_reliability_edge_membership():
    ten = lesson.reliability_table(np.array([1]), np.array([0.1]), 10)
    assert int(ten.count[0]) == 1 and int(ten.count[1]) == 0, (
        f"0.1 landed in bin {int(np.argmax(ten.count))}; bins are ((m-1)/M, m/M], so a value "
        "exactly on an interior edge belongs to the LOWER bin"
    )
    zero = lesson.reliability_table(np.array([0]), np.array([0.0]), 4)
    assert int(zero.count[0]) == 1, "0.0 belongs to the first bin — clip the bin index at 0"
    one = lesson.reliability_table(np.array([1]), np.array([1.0]), 4)
    assert int(one.count[3]) == 1, "1.0 belongs to the last bin — clip the bin index at n_bins-1"
    single = lesson.reliability_table(np.array([0, 1]), np.array([0.2, 0.8]), 1)
    assert int(single.count[0]) == 2, "n_bins=1 puts everything in one bin"


def test_reliability_rates_are_means():
    y = np.array([1, 0, 0, 0, 1, 1])
    p = np.array([0.11, 0.13, 0.15, 0.17, 0.81, 0.89])
    t = lesson.reliability_table(y, p, 10)
    assert abs(float(t.mean_pred[1]) - 0.14) < 1e-12, (
        f"bin 1 holds 0.11, 0.13, 0.15, 0.17 and its mean prediction is 0.14, got {t.mean_pred[1]}"
    )
    assert abs(float(t.obs_rate[1]) - 0.25) < 1e-12, (
        f"bin 1 holds one event in four records: 0.25, got {t.obs_rate[1]}"
    )
    assert abs(float(t.obs_rate[8]) - 1.0) < 1e-12, f"bin 8 holds two events in two records"
    assert abs(float(t.mean_pred[8]) - 0.85) < 1e-12, f"bin 8 mean prediction is 0.85"


def test_reliability_rejects_bad_input():
    cases = (
        ((np.array([0, 1]), np.array([0.5, 1.5]), 10), "a probability above 1.0"),
        ((np.array([0, 1]), np.array([-0.1, 0.5]), 10), "a probability below 0.0"),
        ((np.array([0, 1]), np.array([0.5]), 10), "arrays of different lengths"),
        ((np.array([0, 7]), np.array([0.5, 0.6]), 10), "a label that is neither 0 nor 1"),
        ((np.array([0, 1]), np.array([0.5, 0.6]), 0), "n_bins of 0"),
    )
    for args, why in cases:
        try:
            lesson.reliability_table(*args)
        except ValueError:
            continue
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(f"{why} raised {type(exc).__name__}, not ValueError") from None
        raise AssertionError(
            f"{why} was accepted silently. A validation suite that does not reject a malformed "
            "extract is how a bad number reaches a committee."
        )


# --- exercise 2: expected_calibration_error() ---------------------------------------------

def test_ece_weighted_by_support():
    y = np.array([0] * 100 + [0])
    p = np.array([0.05] * 100 + [0.95])
    got = lesson.expected_calibration_error(y, p, 10)
    want = _ece_reference(y, p, 10)
    assert abs(got - want) < 1e-9, (
        f"got {got:.6f}, expected {want:.6f}. One hundred records sit in the first bin with a "
        "gap of 0.05 and a single record sits in the last with a gap of 0.95; weighting the "
        "two bins equally gives 0.5 and is the mistake this case exists to catch"
    )


def test_ece_absolute_gap():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.4, 0.4, 0.6, 0.6])
    got = lesson.expected_calibration_error(y, p, 2)
    assert abs(got - 0.4) < 1e-9, (
        f"got {got:.6f}, expected 0.4. One bin over-predicts by 0.4 and the other under-"
        "predicts by 0.4; signed errors cancel to zero, and two findings that add up to "
        "nothing are still two findings"
    )


def test_ece_ignores_empty_bins():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.02, 0.04, 0.97, 0.99, 0.95])
    got = lesson.expected_calibration_error(y, p, 20)
    assert np.isfinite(got), (
        f"got {got} with 20 bins and only two of them occupied. An empty bin's gap is nan and "
        "0 * nan is nan, not 0 — reduce over the occupied bins only"
    )
    assert abs(got - _ece_reference(y, p, 20)) < 1e-9, f"got {got:.6f}, expected {_ece_reference(y, p, 20):.6f}"


def test_ece_perfect_and_type():
    perfect = lesson.expected_calibration_error(np.array([0, 1]), np.array([0.0, 1.0]), 10)
    assert abs(perfect) < 1e-12, f"a perfectly calibrated sample scores 0.0, got {perfect}"
    assert type(perfect) is float, (
        f"return a plain Python float, got {type(perfect).__name__}. np.float64 is a SUBCLASS "
        "of float, so isinstance() will not save you here — and json.dumps refuses it, which "
        "is how this escapes into a report pipeline and stops it at the last step. Wrap the "
        "reduction in float()"
    )
    worst = lesson.expected_calibration_error(np.array([0, 0]), np.array([1.0, 1.0]), 10)
    assert abs(worst - 1.0) < 1e-12, (
        f"two records predicted at certainty that never happened scores 1.0, got {worst}"
    )


# --- exercise 3: population_stability_index() ----------------------------------------------

def test_psi_zero_on_identical_shape():
    same = np.array([0.1, 0.2, 0.8, 0.9])
    edges = np.array([0.0, 0.5, 1.0])
    flat = lesson.population_stability_index(same, same, edges)
    assert isinstance(flat, lesson.StabilityResult), f"expected a StabilityResult, got {type(flat).__name__}"
    assert abs(flat.psi) < 1e-12, f"identical samples must score 0.0, got {flat.psi:.6f}"
    doubled = lesson.population_stability_index(np.array([0.1, 0.9]),
                                                np.array([0.1, 0.1, 0.9, 0.9]), edges)
    assert abs(doubled.psi) < 1e-12, (
        f"the same shape at twice the size scored {doubled.psi:.6f} — PSI compares shares, so "
        "divide each side by its own total; on raw counts every growing portfolio looks unstable"
    )


def test_psi_closed_form():
    expected = np.array([0.1, 0.1, 0.1, 0.9])   # shares 0.75 / 0.25
    actual = np.array([0.1, 0.9, 0.9, 0.9])     # shares 0.25 / 0.75
    edges = np.array([0.0, 0.5, 1.0])
    got = lesson.population_stability_index(expected, actual, edges).psi
    want = math.log(3.0)                        # 2 * 0.5 * ln(3), worked out by hand
    assert abs(got - want) < 1e-9, (
        f"got {got:.6f}, expected {want:.6f} = ln(3). The term is (a - e) * ln(a / e) per bin, "
        "summed — check you are not using |a - e|, a natural log of the ratio the other way "
        "round, or log base 10"
    )


def test_psi_floor_is_two_sided():
    edges = np.array([0.0, 0.5, 1.0])
    emptied = lesson.population_stability_index(np.array([0.1, 0.9]), np.array([0.9, 0.9]), edges)
    assert np.isfinite(emptied.psi), (
        f"a bin that emptied out gave {emptied.psi} — floor the ACTUAL share before the log"
    )
    appeared = lesson.population_stability_index(np.array([0.9, 0.9]), np.array([0.1, 0.9]), edges)
    assert np.isfinite(appeared.psi), (
        f"a bin that appeared from nothing gave {appeared.psi} — floor the EXPECTED share too; "
        "flooring one side leaves the other dividing by zero"
    )
    assert emptied.psi > 1.0 and appeared.psi > 1.0, (
        "a bin appearing or disappearing is the loudest signal PSI can give; both must be large"
    )
    assert abs(emptied.psi - _psi_reference([0.1, 0.9], [0.9, 0.9], [0.0, 0.5, 1.0])) < 1e-6


def test_psi_contributions():
    expected = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.3])
    actual = np.array([0.1, 0.9, 0.9, 0.9, 0.2, 0.2])
    edges = np.array([0.0, 0.25, 0.5, 1.0])
    out = lesson.population_stability_index(expected, actual, edges)
    assert len(out.contributions) == 3, (
        f"three edges gaps means three bins; got {len(out.contributions)} contributions"
    )
    assert abs(float(out.contributions.sum()) - out.psi) < 1e-12, (
        f"contributions sum to {out.contributions.sum():.6f} but psi is {out.psi:.6f}; a breach "
        "nobody can trace back to a bin is not a finding, it is a rumour"
    )
    assert bool(np.all(out.contributions >= -1e-12)), (
        f"every per-bin term of PSI is non-negative by construction, got {out.contributions}"
    )
    assert abs(float(out.expected_pct.sum()) - 1.0) < 1e-12, "expected_pct is shares, summing to 1"
    assert abs(float(out.actual_pct.sum()) - 1.0) < 1e-12, "actual_pct is shares, summing to 1"
    assert abs(out.psi - _psi_reference(expected, actual, edges)) < 1e-9, (
        f"got {out.psi:.6f}, an independent implementation gets "
        f"{_psi_reference(expected, actual, edges):.6f}"
    )


def test_psi_outside_edges_and_bad_edges():
    # Values below the first edge and above the last must fall into the end bins, not be
    # dropped: dropping them silently renormalises both samples and hides the tails, which is
    # where drift shows up first.
    inside = lesson.population_stability_index(np.array([0.3, 0.6]), np.array([0.0, 0.95]),
                                               np.array([0.2, 0.5, 0.8]))
    assert abs(inside.psi) < 1e-12, (
        f"got {inside.psi:.6f}: 0.0 belongs to the first bin and 0.95 to the last, which makes "
        "the two samples identically shaped — are you dropping out-of-range values?"
    )
    for bad, why in ((np.array([0.5]), "fewer than two edges"),
                     (np.array([0.0, 0.5, 0.5, 1.0]), "a repeated edge"),
                     (np.array([0.0, 1.0, 0.5]), "edges out of order")):
        try:
            lesson.population_stability_index(np.array([0.1]), np.array([0.2]), bad)
        except ValueError:
            continue
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(f"{why} raised {type(exc).__name__}, not ValueError") from None
        raise AssertionError(f"{why} should raise ValueError")


# --- exercise 4: subgroup_table() ----------------------------------------------------------

def test_subgroup_rows_and_order():
    y = np.array([0, 1, 0, 1, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.4, 0.5])
    g = np.array(["north", "north", "south", "south", "south", "tiny", "north"])
    t = lesson.subgroup_table(y, p, g, min_support=3)
    assert list(t["group"]) == ["north", "south", "tiny"], (
        f"groups {list(t['group'])} — one row per distinct value, ascending, and EVERY value "
        "gets a row; a group below support is suppressed, never deleted"
    )
    assert t["count"].tolist() == [3, 3, 1], (
        f"counts {t['count'].tolist()} should be [3, 3, 1]; the count is reported for every "
        "group, including the ones whose metrics are not"
    )
    for key in ("group", "count", "event_rate", "mean_pred", "gap", "ece", "reportable"):
        assert key in t, f"the table is missing the {key!r} column"
        assert len(t[key]) == 3, f"{key!r} has {len(t[key])} entries for 3 groups"


def test_subgroup_min_support_boundary():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.4])
    g = np.array(["at", "at", "at", "at", "below", "below"])
    t = lesson.subgroup_table(y, p, g, min_support=4)
    assert t["reportable"].tolist() == [True, False], (
        f"reportable {t['reportable'].tolist()} — 'at' has exactly 4 records against a minimum "
        "of 4 and IS reportable; a minimum is >=, not >"
    )
    for field in ("event_rate", "mean_pred", "gap", "ece"):
        assert np.isnan(t[field][1]), (
            f"{field} for the below-support group is {t[field][1]}, not nan. You declared the "
            "group cannot support a metric; computing one anyway is the finding"
        )
        assert np.isfinite(t[field][0]), f"{field} for the reportable group must be a real number"
    assert int(t["count"][1]) == 2, "the suppressed group's COUNT is still reported"


def test_subgroup_gap_sign_and_values():
    over = lesson.subgroup_table(np.array([0, 1]), np.array([0.9, 0.9]),
                                 np.array(["x", "x"]), min_support=2)
    assert float(over["gap"][0]) > 0, (
        f"gap {over['gap'][0]}: predicted 0.9 against an observed 0.5 is OVER-prediction and "
        "must be positive — gap is mean_pred - event_rate, not the reverse"
    )
    assert abs(float(over["gap"][0]) - 0.4) < 1e-12, f"gap should be 0.9 - 0.5 = 0.4"
    under = lesson.subgroup_table(np.array([1, 1]), np.array([0.2, 0.2]),
                                  np.array(["x", "x"]), min_support=2)
    assert abs(float(under["gap"][0]) + 0.8) < 1e-12, (
        f"gap {under['gap'][0]}: predicted 0.2 against an observed 1.0 is -0.8"
    )
    assert abs(float(under["event_rate"][0]) - 1.0) < 1e-12
    assert abs(float(under["mean_pred"][0]) - 0.2) < 1e-12


def test_subgroup_metrics_are_computed_per_group():
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    p = np.array([0.15, 0.15, 0.15, 0.15, 0.85, 0.85, 0.85, 0.85])
    g = np.array(["low", "low", "low", "low", "high", "high", "high", "high"])
    t = lesson.subgroup_table(y, p, g, min_support=4, n_bins=10)
    order = list(t["group"])
    hi, lo = order.index("high"), order.index("low")
    assert abs(float(t["ece"][lo]) - _ece_reference(y[:4], p[:4], 10)) < 1e-9, (
        f"the 'low' group's ECE is {t['ece'][lo]:.6f}; computed on that group's records alone "
        f"it is {_ece_reference(y[:4], p[:4], 10):.6f} — slice first, then measure"
    )
    assert abs(float(t["ece"][hi]) - _ece_reference(y[4:], p[4:], 10)) < 1e-9, (
        f"the 'high' group's ECE is {t['ece'][hi]:.6f}, expected "
        f"{_ece_reference(y[4:], p[4:], 10):.6f}"
    )
    assert t["ece"][hi] != t["ece"][lo], (
        "the two groups have different calibration; identical ECEs mean the whole sample is "
        "being measured for every row, which is the thing a subgroup table exists to avoid"
    )


def test_subgroup_rejects_bad_input():
    y, p, g = np.array([0, 1]), np.array([0.2, 0.8]), np.array(["a", "b"])
    for args, why in (((y, p, g, 0), "a min_support of 0"),
                      ((y, np.array([0.2]), g, 1), "arrays of different lengths")):
        try:
            lesson.subgroup_table(*args)
        except ValueError:
            continue
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(f"{why} raised {type(exc).__name__}, not ValueError") from None
        raise AssertionError(f"{why} should raise ValueError")


# --- exercise 5: challenger_decision() -----------------------------------------------------

def test_decision_verdicts():
    champ = {"auc": 0.70, "ece": 0.05}
    cases = (
        ({"auc": 0.74, "ece": 0.02}, "promote", "clears both gates, gains 0.04, better calibrated"),
        ({"auc": 0.705, "ece": 0.02}, "hold", "gains only 0.005 against a required 0.02"),
        ({"auc": 0.74, "ece": 0.055}, "reject", "ECE 0.055 blows the 0.04 ceiling"),
        ({"auc": 0.60, "ece": 0.01}, "reject", "AUC 0.60 is under the 0.65 floor"),
    )
    for challenger, want, why in cases:
        got = lesson.challenger_decision(champ, challenger, POLICY)
        assert got.verdict == want, (
            f"{challenger} against {champ}: expected {want!r}, got {got.verdict!r} ({why})"
        )
    regressed = lesson.challenger_decision({"auc": 0.70, "ece": 0.01},
                                           {"auc": 0.74, "ece": 0.03}, POLICY)
    assert regressed.verdict == "hold", (
        f"got {regressed.verdict!r}: the challenger gains 0.04 of AUC but its calibration is "
        "worse than the champion's, so rule 3 fails and it is held, not promoted"
    )


def test_decision_reject_precedence():
    wild = lesson.challenger_decision({"auc": 0.60, "ece": 0.50},
                                      {"auc": 0.95, "ece": 0.42}, POLICY)
    assert wild.verdict == "reject", (
        f"got {wild.verdict!r}: an AUC gain of 0.35 and a calibration improvement, but an ECE "
        "of 0.42 against a ceiling of 0.04. Absolute gates are evaluated first and a large "
        "improvement cannot buy past them"
    )
    assert wild.reasons[1].startswith("FAIL "), "rule 1 is the ECE ceiling and it failed here"
    assert wild.reasons[2].startswith("PASS "), (
        "rules 2 and 3 are still evaluated and still reported on a reject — the committee needs "
        "to see what was true as well as what was disqualifying"
    )


def test_decision_boundaries_are_inclusive():
    exact = lesson.challenger_decision({"auc": 0.70, "ece": 0.04},
                                       {"auc": 0.72, "ece": 0.04}, POLICY)
    assert exact.verdict == "promote", (
        f"got {exact.verdict!r}: AUC gain exactly 0.02, ECE exactly on the 0.04 ceiling, "
        "calibration exactly unchanged. Every comparison in this rule set is inclusive, so all "
        "four rules pass"
    )
    floor = lesson.challenger_decision({"auc": 0.60, "ece": 0.02},
                                       {"auc": 0.65, "ece": 0.02}, POLICY)
    assert floor.verdict == "promote", "an AUC exactly on the floor clears the floor"
    just_under = lesson.challenger_decision({"auc": 0.70, "ece": 0.04},
                                            {"auc": 0.72, "ece": 0.0400001}, POLICY)
    assert just_under.verdict == "reject", (
        f"got {just_under.verdict!r}: 0.0400001 is over the ceiling, inclusive or not"
    )


def test_decision_reasons():
    got = lesson.challenger_decision({"auc": 0.70, "ece": 0.05},
                                     {"auc": 0.705, "ece": 0.02}, POLICY)
    assert isinstance(got.reasons, tuple), f"reasons is a tuple, got {type(got.reasons).__name__}"
    assert len(got.reasons) == 4, (
        f"{len(got.reasons)} reasons — all four rules are reported every time, in order, "
        "including the ones that passed. A decision an auditor cannot re-derive from the "
        "document is not a decision"
    )
    assert [r.split()[0] for r in got.reasons] == ["PASS", "PASS", "FAIL", "PASS"], (
        f"expected PASS, PASS, FAIL, PASS in rule order, got {[r.split()[0] for r in got.reasons]}"
    )
    assert "0.6500" in got.reasons[0], f"rule 0 must name the floor it judged against: {got.reasons[0]!r}"
    assert "0.0400" in got.reasons[1], f"rule 1 must name the ceiling: {got.reasons[1]!r}"
    assert "0.0200" in got.reasons[2], f"rule 2 must name the required gain: {got.reasons[2]!r}"
    assert all(isinstance(r, str) and r.strip() for r in got.reasons), "reasons are non-empty strings"


def test_decision_margins():
    got = lesson.challenger_decision({"auc": 0.70, "ece": 0.05},
                                     {"auc": 0.74, "ece": 0.02}, POLICY)
    for key in ("auc_gain", "ece_change", "auc_headroom", "ece_headroom"):
        assert key in got.margins, f"margins is missing {key!r}"
    assert abs(got.margins["auc_gain"] - 0.04) < 1e-12, f"auc_gain is 0.74 - 0.70 = 0.04"
    assert abs(got.margins["ece_change"] + 0.03) < 1e-12, (
        f"ece_change is {got.margins['ece_change']}: challenger minus champion, so a better "
        "calibrated challenger gives a NEGATIVE number. A positive one means the subtraction "
        "is the wrong way round and the sign in the report will read as a regression"
    )
    assert abs(got.margins["auc_headroom"] - 0.09) < 1e-12, "auc_headroom is 0.74 - 0.65"
    assert abs(got.margins["ece_headroom"] - 0.02) < 1e-12, "ece_headroom is 0.04 - 0.02"


def test_decision_missing_keys():
    for bad, why in ((({"auc": 0.7}, {"auc": 0.7, "ece": 0.0}, POLICY), "champion.ece"),
                     (({"auc": 0.7, "ece": 0.0}, {"ece": 0.0}, POLICY), "challenger.auc"),
                     (({"auc": 0.7, "ece": 0.0}, {"auc": 0.7, "ece": 0.0},
                       {"min_auc": 0.6, "max_ece": 0.1}), "policy.min_auc_gain")):
        try:
            lesson.challenger_decision(*bad)
        except ValueError as exc:
            assert why.split(".")[1] in str(exc), (
                f"the error should name the missing key {why!r}, got {exc}"
            )
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(f"missing {why} raised {type(exc).__name__}, not ValueError") from None
        else:
            raise AssertionError(f"a missing {why} should raise ValueError")


# --- exercise 6: render_validation_report() ------------------------------------------------

def test_report_structure():
    text = lesson.render_validation_report(_probe())
    assert isinstance(text, str), f"return a string, got {type(text).__name__}"
    first = text.splitlines()[0]
    assert first == "# Validation report — probe-model", (
        f"first line was {first!r}; it must be '# Validation report — <model_id>' with the id "
        "taken from findings"
    )
    positions = [text.find(h) for h in lesson.REPORT_SECTIONS]
    missing = [h for h, pos in zip(lesson.REPORT_SECTIONS, positions) if pos < 0]
    assert not missing, f"missing sections: {missing}"
    assert positions == sorted(positions), (
        "the six sections must appear in the order given by REPORT_SECTIONS"
    )
    assert "PROBE DATA NOTE, SYNTHETIC" in text, (
        "the data note goes in section 1, verbatim. A report that cannot say where its data "
        "came from is not evidence"
    )


def test_report_uses_findings_only():
    text = lesson.render_validation_report(_probe())
    for needle, why in (("0.4242", "the ECE from findings, at 4 decimal places"),
                        ("0.7777", "the PSI from findings, at 4 decimal places"),
                        ("0.616", "the champion AUC from findings, at 3 decimal places"),
                        ("0.818", "the challenger AUC from findings"),
                        ("0.5252", "the champion ECE from findings"),
                        ("0.0303", "the challenger ECE from findings"),
                        ("0.25", "the PSI threshold from findings, at 2 decimal places")):
        assert needle in text, (
            f"{needle} is missing: {why}. This function reports what it was handed — it does "
            "not recompute, re-round or reach for module-level data"
        )
    assert "1970-01-01" in text, "the as-of date comes from findings"


def test_report_empty_bin_is_not_a_rate():
    text = lesson.render_validation_report(_probe())
    calibration = text.split(lesson.REPORT_SECTIONS[1])[1].split(lesson.REPORT_SECTIONS[2])[0]
    rows = [r for r in calibration.splitlines() if r.startswith("| 0.")]
    assert len(rows) == 2, f"one row per bin, empty bins included; got {len(rows)}: {rows}"
    empty = [r for r in rows if "| 0" in r and r.rstrip().endswith("0 |")]
    assert empty, f"the empty bin must still have a row: {rows}"
    assert "n/a" in empty[0], (
        f"the empty bin row is {empty[0]!r} — it must print n/a, not nan and certainly not a "
        "rate; the bin is evidence that the model was never exercised up there"
    )
    assert "0.1313" in calibration and "0.2424" in calibration, (
        "the occupied bin's mean prediction and observed rate come from findings"
    )


def test_report_suppresses_without_hiding():
    text = lesson.render_validation_report(_probe())
    assert "suppressed (n=11 < minimum support 200)" in text, (
        "a below-support group is listed with its count and the threshold it missed, marked "
        "suppressed. Dropping the row is how a model gets signed off as fair on the segments "
        "that happened to be big enough to measure"
    )
    assert "omega" in text, "the suppressed group is named"
    assert "0.9191" not in text, (
        "the suppressed group's ECE reached the page. You declared you cannot support that "
        "number; printing it anyway is worse than not having the rule"
    )
    assert "0.0505" in text and "alpha" in text, "the reportable group's metrics DO go in"


def test_report_psi_verdict_boundary():
    over = _probe()
    assert "Verdict: BREACH" in lesson.render_validation_report(over), (
        "a PSI of 0.7777 against a threshold of 0.25 is a breach"
    )
    at = _probe()
    at["psi"] = lesson.StabilityResult(psi=0.25, contributions=np.array([0.1, 0.15]),
                                       expected_pct=np.array([0.5, 0.5]),
                                       actual_pct=np.array([0.4, 0.6]))
    text = lesson.render_validation_report(at)
    assert "Verdict: WITHIN THRESHOLD" in text and "Verdict: BREACH" not in text, (
        "a PSI sitting exactly ON the policy threshold is within it — the comparison is "
        "psi > threshold. This is the one case a validator will be asked to defend"
    )
    under = _probe()
    under["psi"] = lesson.StabilityResult(psi=0.0101, contributions=np.array([0.0051, 0.005]),
                                          expected_pct=np.array([0.5, 0.5]),
                                          actual_pct=np.array([0.49, 0.51]))
    assert "Verdict: WITHIN THRESHOLD" in lesson.render_validation_report(under)


def test_report_decision_block():
    text = lesson.render_validation_report(_probe())
    assert "Verdict: HOLD" in text, (
        "the decision verdict is printed in UPPER CASE; findings carried verdict='hold'"
    )
    for reason in _probe()["decision"].reasons:
        assert f"- {reason}" in text, (
            f"every reason appears verbatim, one per line, prefixed '- ': missing {reason!r}"
        )
    tail = text.split(lesson.REPORT_SECTIONS[5])[1]
    assert tail.count("PASS probe rule zero") == 1, "each reason appears once"
    assert "FAIL probe rule two" in tail, "the failing rule is reported alongside the passing ones"


def test_report_missing_keys():
    for dropped in ("psi", "subgroups", "decision"):
        probe = {k: v for k, v in _probe().items() if k != dropped}
        try:
            lesson.render_validation_report(probe)
        except ValueError as exc:
            assert dropped in str(exc), f"the error should name the missing key: got {exc}"
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(
                f"incomplete findings raised {type(exc).__name__}, not ValueError"
            ) from None
        else:
            raise AssertionError(f"findings without {dropped!r} should raise ValueError")


RUBRIC = [
    (test_reliability_shape_and_edges, 3,
     "edges are np.linspace(0, 1, n_bins + 1); keep one row per bin, empty ones included"),
    (test_reliability_empty_bins_are_nan, 4,
     "divide by np.where(count > 0, count, np.nan) so an empty bin reports nan, not 0.0"),
    (test_reliability_edge_membership, 3,
     "np.searchsorted(edges, p, side='left') - 1, clipped to [0, n_bins - 1]"),
    (test_reliability_rates_are_means, 3,
     "np.bincount with weights gives the per-bin sums; divide by the per-bin counts"),
    (test_reliability_rejects_bad_input, 3,
     "check n_bins, matching shapes, probabilities inside [0, 1] and labels in {0, 1}"),
    (test_ece_weighted_by_support, 5,
     "each bin's weight is count / total records, never 1 / n_bins"),
    (test_ece_absolute_gap, 3,
     "np.abs(obs_rate - mean_pred) inside each bin, before any averaging"),
    (test_ece_ignores_empty_bins, 3,
     "select the bins with count > 0 first: 0 * nan is nan, so an empty bin poisons the sum"),
    (test_ece_perfect_and_type, 2,
     "return float(...) — a numpy scalar is not a Python float"),
    (test_psi_zero_on_identical_shape, 4,
     "divide each side's bin counts by its OWN total before comparing"),
    (test_psi_closed_form, 5,
     "(a - e) * np.log(a / e) per bin, summed — signed difference, natural log"),
    (test_psi_floor_is_two_sided, 4,
     "np.maximum(share, floor) on BOTH sides before the ratio and the log"),
    (test_psi_contributions, 3,
     "return the per-bin array as well as its sum, plus both share vectors"),
    (test_psi_outside_edges_and_bad_edges, 3,
     "np.clip the bin index into range so the tails land in the end bins; reject bad edges"),
    (test_subgroup_rows_and_order, 4,
     "np.unique(groups) gives every distinct value, already sorted"),
    (test_subgroup_min_support_boundary, 4,
     "reportable is count >= min_support; a suppressed row keeps its count and nothing else"),
    (test_subgroup_gap_sign_and_values, 3,
     "gap = mean_pred - event_rate, so over-prediction is positive"),
    (test_subgroup_metrics_are_computed_per_group, 4,
     "mask the arrays to the group first, then measure that slice"),
    (test_subgroup_rejects_bad_input, 2,
     "min_support below 1, and arrays of different lengths, are both ValueError"),
    (test_decision_verdicts, 5,
     "absolute gates first, then the gain and regression rules; 'better' alone is a hold"),
    (test_decision_reject_precedence, 3,
     "a failed absolute gate rejects whatever rules 2 and 3 say — and they are still reported"),
    (test_decision_boundaries_are_inclusive, 4,
     ">= for the floor and the gain, <= for the ceiling: landing on a limit passes it"),
    (test_decision_reasons, 4,
     "four reasons in rule order, each 'PASS '/'FAIL ' plus the figure and the limit"),
    (test_decision_margins, 3,
     "ece_change is challenger minus champion, so an improvement is negative"),
    (test_decision_missing_keys, 2,
     "collect every missing key and name them all in one ValueError"),
    (test_report_structure, 4,
     "build the page from REPORT_SECTIONS in order, with the model id on the first line"),
    (test_report_uses_findings_only, 5,
     "format every figure out of `findings`; recompute nothing here"),
    (test_report_empty_bin_is_not_a_rate, 3,
     "branch on count == 0 and print n/a for both rates"),
    (test_report_suppresses_without_hiding, 4,
     "print the row, the count and the suppression note; print none of the metrics"),
    (test_report_psi_verdict_boundary, 3,
     "BREACH when psi > threshold, WITHIN THRESHOLD otherwise — strictly greater"),
    (test_report_decision_block, 3,
     "verdict.upper(), then one '- <reason>' line per reason, verbatim"),
    (test_report_missing_keys, 2,
     "check for every required key up front and name the ones that are absent"),
]
