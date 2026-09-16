"""Autograder rubric for P03-L01. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson and are deliberate:

* the rubric grades observable behaviour, never one particular call. A causal rolling median
  written as a Python loop and one written with stride tricks are both correct; what is
  graded is that position t never depends on a sample after t. `test_causal_median_is_causal`
  proves that by mutating the tail of the input and re-reading the head.
* `test_health_index_baseline_is_raw` exists because the docstring states a requirement —
  baseline from the RAW rms — that every plausible implementation would otherwise satisfy by
  accident on smooth data. It is graded on a step input where the two routes differ by 5x.
* `test_classify_outcomes_lead_time` is the case that separates this from ordinary binary
  classification. An implementation that ignores `lead_hours` passes every other outcome
  test, because on the lesson's own fleet most caught failures are caught early.
* `test_pr_points_precision_convention` grades the stated convention (precision 1.0 when
  nothing is flagged) rather than a NaN, because a NaN there silently poisons every mean and
  every area computed from the curve downstream.
* `test_best_operating_point_rejects_mismatch` grades a ValueError that no student would
  reach by accident. A sweep of 61 counts indexed by a grid of 60 thresholds returns the
  wrong threshold for the right confusion matrix, and nothing about the output looks wrong.
* `test_the_gap_is_real` is an integration test on the lesson's own fleet. It asserts the
  DIRECTION and a conservative magnitude of the result — cost-optimal strictly below
  accuracy-optimal, at least twice the money, strictly more failures accepted — not the exact
  thresholds, so a student who reasonably rounds the health index elsewhere is not failed for
  landing one grid step away.
* memory and timing are not graded here; tools/execute.py owns those.
"""
import importlib.util, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

C = lesson.Counts


def _rms(values) -> float:
    """Independent reference, so the rubric never grades the student against their own code."""
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(sum(v * v for v in arr.ravel()) / arr.size))


# --- exercise 1: burst_rms -------------------------------------------------------------

def test_burst_rms():
    got = lesson.burst_rms(np.array([3.0, 4.0]))
    assert np.isclose(got, _rms([3.0, 4.0])), f"burst_rms([3, 4]) = {got!r}, expected 3.5355"
    constant = lesson.burst_rms(np.array([10.0, 10.0, 10.0]))
    assert np.isclose(constant, 10.0), (
        f"a constant burst has RMS equal to that constant; got {constant!r}. "
        "np.std would give 0.0 here — RMS does not subtract the mean"
    )
    biased = lesson.burst_rms(np.array([1.0, 3.0]))
    assert np.isclose(biased, _rms([1.0, 3.0])), (
        f"burst_rms([1, 3]) = {biased!r}, expected sqrt(5) = 2.2360 (np.std gives 1.0)"
    )
    assert lesson.burst_rms(np.zeros((2, 3, 7))).shape == (2, 3), "reduce the last axis only"
    rng = np.random.default_rng(7)
    block = rng.normal(5.0, 2.0, (3, 4, 16))
    expected = np.sqrt((block ** 2).mean(axis=-1))
    assert np.allclose(lesson.burst_rms(block), expected), (
        "on a (3, 4, 16) block with a non-zero mean the answer must be the elementwise RMS "
        "along the last axis"
    )


# --- exercise 2: causal_rolling_median -------------------------------------------------

def test_causal_rolling_median_values():
    spike = lesson.causal_rolling_median(np.array([0.0, 10.0, 0.0, 0.0, 0.0]), 3)
    assert np.allclose(spike, [0.0, 5.0, 0.0, 0.0, 0.0]), (
        f"got {np.asarray(spike).tolist()}, expected [0, 5, 0, 0, 0]"
    )
    grown = lesson.causal_rolling_median(np.array([4.0, 0.0, 0.0, 9.0]), 3)
    assert np.allclose(grown, [4.0, 2.0, 0.0, 0.0]), (
        f"got {np.asarray(grown).tolist()}, expected [4, 2, 0, 0] — the first positions use a "
        "GROWING window over the history that exists, not a window padded with x[0]"
    )
    assert np.allclose(lesson.causal_rolling_median(np.array([1.0, 2.0, 3.0]), 1),
                       [1.0, 2.0, 3.0]), "window=1 returns the series unchanged"
    long_window = lesson.causal_rolling_median(np.array([1.0, 2.0, 3.0]), 99)
    assert np.allclose(long_window, [1.0, 1.5, 2.0]), (
        f"a window longer than the series is not an error; got "
        f"{np.asarray(long_window).tolist()}, expected the running median [1, 1.5, 2]"
    )
    two_d = lesson.causal_rolling_median(np.array([[0.0, 8.0, 0.0], [5.0, 5.0, 5.0]]), 3)
    assert np.asarray(two_d).shape == (2, 3), f"shape changed to {np.asarray(two_d).shape}"
    assert np.allclose(two_d, [[0.0, 4.0, 0.0], [5.0, 5.0, 5.0]]), (
        f"smoothing must run along the LAST axis, per row; got {np.asarray(two_d).tolist()}"
    )


def test_causal_median_is_causal():
    # Same head, two different tails. A causal filter cannot tell them apart in the head.
    head = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    calm = lesson.causal_rolling_median(np.concatenate([head, np.zeros(5)]), 5)
    wild = lesson.causal_rolling_median(np.concatenate([head, np.full(5, 1000.0)]), 5)
    assert np.allclose(calm[:5], wild[:5]), (
        f"changing hours 5-9 changed hours 0-4: {np.asarray(calm)[:5].tolist()} became "
        f"{np.asarray(wild)[:5].tolist()}. The window is reading the future — it must span "
        "[t - window + 1, t], never past t"
    )
    assert np.allclose(calm[:5], [1.0, 1.5, 2.0, 2.5, 3.0]), (
        f"the growing-window head should be [1, 1.5, 2, 2.5, 3], got "
        f"{np.asarray(calm)[:5].tolist()}"
    )


def test_causal_rolling_median_validates_window():
    for bad in (0, -1, -7):
        try:
            lesson.causal_rolling_median(np.array([1.0, 2.0]), bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"window={bad} must raise ValueError")


# --- exercise 3: health_index ----------------------------------------------------------

def test_health_index_values():
    doubling = np.array([[[1.0], [1.0], [2.0], [2.0]]])
    got = lesson.health_index(doubling, baseline_hours=2, window=1)
    assert np.asarray(got).shape == (1, 4), f"expected (1, 4), got {np.asarray(got).shape}"
    assert np.allclose(got, [[1.0, 1.0, 2.0, 2.0]]), (
        f"got {np.asarray(got).round(4).tolist()}, expected [[1, 1, 2, 2]]"
    )
    # Two machines of very different size behaving identically must give identical indices.
    small = np.full((1, 30, 8), 0.5)
    big = np.full((1, 30, 8), 40.0)
    pair = lesson.health_index(np.concatenate([small, big]), baseline_hours=10, window=5)
    assert np.allclose(pair, 1.0), (
        "two machines with different absolute amplitudes and no change over time must both "
        f"sit at 1.0; got a range of {np.asarray(pair).min():.3f} to "
        f"{np.asarray(pair).max():.3f}. Each machine is normalised by ITS OWN baseline"
    )
    ramp = np.concatenate([np.full((1, 20, 4), 3.0), np.full((1, 20, 4), 6.0)], axis=1)
    doubled = lesson.health_index(ramp, baseline_hours=20, window=1)
    assert np.isclose(doubled[0, -1], 2.0), (
        f"an amplitude that doubles after the baseline window gives a health index of 2.0; "
        f"got {doubled[0, -1]:.4f}"
    )


def test_health_index_baseline_is_raw():
    step = np.array([[[1.0]] * 4 + [[9.0]] * 4])
    got = lesson.health_index(step, baseline_hours=8, window=3)
    assert np.isclose(got[0, 0], 0.2), (
        f"health_index[0, 0] = {got[0, 0]:.4f}; the RAW rms over the baseline window is "
        "[1,1,1,1,9,9,9,9] with median 5.0, so the first value is 1/5 = 0.2. A 1.0 means the "
        "baseline came from the SMOOTHED series, whose median over the same window is 1.0"
    )
    assert np.isclose(got[0, -1], 1.8), (
        f"health_index[0, -1] = {got[0, -1]:.4f}, expected 9/5 = 1.8"
    )


def test_health_index_validates_baseline_hours():
    doubling = np.array([[[1.0], [1.0], [2.0], [2.0]]])
    for bad in (0, -5):
        try:
            lesson.health_index(doubling, baseline_hours=bad, window=1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"baseline_hours={bad} must raise ValueError")


# --- exercise 4: alarm_times -----------------------------------------------------------

def test_alarm_times():
    health = np.array([[1.0, 1.5, 2.0],
                       [1.0, 1.0, 1.0],
                       [3.0, 1.0, 1.0]])
    got = np.asarray(lesson.alarm_times(health, 1.5))
    assert got.tolist() == [1, -1, 0], (
        f"got {got.tolist()}, expected [1, -1, 0]. A 2 in the first slot means you used > "
        "instead of >=; a 0 in the second means argmax's answer for a row with no crossing "
        "at all was returned unchecked"
    )
    assert np.issubdtype(got.dtype, np.integer), (
        f"alarm_times returns hour indices, so an integer dtype; got {got.dtype}"
    )
    assert np.asarray(lesson.alarm_times(health, 0.5)).tolist() == [0, 0, 0], (
        "a threshold below every value alarms in hour 0 for every machine"
    )
    assert np.asarray(lesson.alarm_times(health, 99.0)).tolist() == [-1, -1, -1], (
        "a threshold above every value gives -1 everywhere, not the number of hours"
    )
    # Exactly at the threshold counts, and only the FIRST crossing is reported.
    repeated = np.array([[1.0, 2.0, 1.0, 2.0]])
    assert np.asarray(lesson.alarm_times(repeated, 2.0)).tolist() == [1], (
        "report the first crossing, and treat equality as a crossing"
    )


# --- exercise 5: classify_outcomes -----------------------------------------------------

def test_classify_outcomes_buckets():
    got = lesson.classify_outcomes(np.array([5, 95, -1, 20, -1]),
                                   np.array([100, 100, 100, -1, -1]), 10)
    assert isinstance(got, C), f"expected a Counts, got {type(got).__name__}"
    quad = (int(got.tp), int(got.fp), int(got.fn), int(got.tn))
    assert quad == (1, 1, 2, 1), (
        f"expected (tp, fp, fn, tn) = (1, 1, 2, 1), got {quad}. Machine 1 alarmed at hour 95 "
        "for a failure at 100 with 10 hours required, so it is a MISSED failure; machine 2 "
        "never alarmed and also failed; machine 3 alarmed and never failed"
    )
    assert sum(quad) == 5, f"every machine is counted exactly once; the four add to {sum(quad)}"
    nothing = lesson.classify_outcomes(np.array([-1, -1, -1]), np.array([-1, -1, -1]), 10)
    assert (int(nothing.tp), int(nothing.fp), int(nothing.fn), int(nothing.tn)) == (0, 0, 0, 3)
    everything = lesson.classify_outcomes(np.array([0, 0]), np.array([-1, -1]), 10)
    assert (int(everything.fp), int(everything.tn)) == (2, 0), (
        "two healthy machines that both alarm are two false positives"
    )


def test_classify_outcomes_lead_time():
    fail = np.array([100, 100, 100, 100])
    alarm = np.array([49, 50, 51, 100])
    got = lesson.classify_outcomes(alarm, fail, 50)
    assert (int(got.tp), int(got.fn)) == (2, 2), (
        f"with 50 hours of lead required and alarms at 49, 50, 51 and 100 for a failure at "
        f"100, exactly two are in time (51 and 50 hours of warning); got tp={got.tp}, "
        f"fn={got.fn}. An implementation that ignores lead_hours entirely reports tp=4"
    )
    boundary = lesson.classify_outcomes(np.array([50]), np.array([100]), 50)
    assert int(boundary.tp) == 1, (
        "exactly lead_hours of warning is in time: the comparison is >=, not >"
    )
    strict = lesson.classify_outcomes(np.array([51]), np.array([100]), 50)
    assert int(strict.fn) == 1, "49 hours of warning when 50 are needed is a missed failure"
    # lead_hours = 0 turns this back into ordinary binary classification.
    plain = lesson.classify_outcomes(np.array([99]), np.array([100]), 0)
    assert int(plain.tp) == 1, "with lead_hours=0 any alarm before the failure counts"


def test_classify_outcomes_validates_lead():
    for bad in (-1, -48):
        try:
            lesson.classify_outcomes(np.array([1]), np.array([10]), bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"lead_hours={bad} must raise ValueError")


# --- exercise 6: sweep_thresholds ------------------------------------------------------

def test_sweep_thresholds():
    health = np.array([[1.0, 3.0], [1.0, 1.0], [1.0, 2.0]])
    fail = np.array([1, -1, -1])
    swept = lesson.sweep_thresholds(health, fail, np.array([1.5, 2.5, 4.0]), lead_hours=0)
    assert isinstance(swept, C), f"expected a Counts, got {type(swept).__name__}"
    for name in ("tp", "fp", "fn", "tn"):
        field = np.asarray(getattr(swept, name))
        assert field.shape == (3,), (
            f"counts.{name} should be one entry per threshold, shape (3,), got {field.shape}"
        )
    assert np.asarray(swept.tp).tolist() == [1, 1, 0], f"tp = {np.asarray(swept.tp).tolist()}"
    assert np.asarray(swept.fp).tolist() == [1, 0, 0], (
        f"fp = {np.asarray(swept.fp).tolist()}, expected [1, 0, 0] — machine 2 reaches 2.0, "
        "so it alarms at a threshold of 1.5 and not at 2.5"
    )
    assert np.asarray(swept.fn).tolist() == [0, 0, 1]
    assert np.asarray(swept.tn).tolist() == [1, 2, 2]
    totals = sum(np.asarray(getattr(swept, n)) for n in ("tp", "fp", "fn", "tn"))
    assert np.all(totals == 3), f"each threshold must account for all 3 machines: {totals}"
    # A rising threshold can never raise more alarms.
    assert np.all(np.diff(np.asarray(swept.fp)) <= 0), "fp must be non-increasing in threshold"


# --- exercises 7 and 8: the curves -----------------------------------------------------

def test_roc_points():
    demo = C(tp=np.array([3, 0]), fp=np.array([1, 0]), fn=np.array([1, 4]),
             tn=np.array([9, 10]))
    fpr, tpr = lesson.roc_points(demo)
    assert np.allclose(fpr, [0.1, 0.0]), (
        f"fpr = {np.asarray(fpr).tolist()}, expected [0.1, 0.0]. roc_points returns "
        "(fpr, tpr) in that order — a first element of 0.75 means they are swapped"
    )
    assert np.allclose(tpr, [0.75, 0.0]), f"tpr = {np.asarray(tpr).tolist()}"


def test_roc_points_zero_denominators():
    empty = C(tp=np.array([0]), fp=np.array([0]), fn=np.array([0]), tn=np.array([0]))
    fpr, tpr = lesson.roc_points(empty)
    assert np.all(np.isfinite(fpr)) and np.all(np.isfinite(tpr)), (
        f"an all-zero Counts gave {np.asarray(fpr).tolist()} and "
        f"{np.asarray(tpr).tolist()}; both rates are defined as 0.0 there, not NaN"
    )
    assert np.allclose(fpr, 0.0) and np.allclose(tpr, 0.0)
    no_failures = C(tp=np.array([0]), fp=np.array([2]), fn=np.array([0]), tn=np.array([8]))
    fpr2, tpr2 = lesson.roc_points(no_failures)
    assert np.allclose(tpr2, 0.0) and np.allclose(fpr2, 0.2), (
        "with no failures in the fleet the recall is 0.0 by definition while the false "
        f"positive rate is still 2/10; got tpr={np.asarray(tpr2).tolist()}, "
        f"fpr={np.asarray(fpr2).tolist()}"
    )


def test_pr_points():
    demo = C(tp=np.array([3, 2]), fp=np.array([1, 6]), fn=np.array([1, 2]),
             tn=np.array([9, 4]))
    rec, prec = lesson.pr_points(demo)
    assert np.allclose(rec, [0.75, 0.5]), (
        f"recall = {np.asarray(rec).tolist()}, expected [0.75, 0.5]. pr_points returns "
        "(recall, precision) in that order"
    )
    assert np.allclose(prec, [0.75, 0.25]), f"precision = {np.asarray(prec).tolist()}"


def test_pr_points_precision_convention():
    silent = C(tp=np.array([0]), fp=np.array([0]), fn=np.array([4]), tn=np.array([96]))
    rec, prec = lesson.pr_points(silent)
    assert np.all(np.isfinite(prec)), (
        f"a threshold that flags nothing gave precision {np.asarray(prec).tolist()}; a NaN "
        "here poisons every mean and every area computed from the curve"
    )
    assert np.allclose(prec, 1.0), (
        f"flagging nothing means raising no false alarms, so precision is 1.0 by the stated "
        f"convention; got {np.asarray(prec).tolist()}"
    )
    assert np.allclose(rec, 0.0), "flagging nothing means recall 0.0"


# --- exercises 9-11: the economics -----------------------------------------------------

def test_expected_cost():
    cheap = lesson.CostModel(planned=10.0, unplanned=100.0, false_alarm=1.0)
    one = lesson.expected_cost(C(tp=2, fp=3, fn=1, tn=94), cheap)
    assert np.isclose(one, 123.0), (
        f"expected 2*10 + 1*100 + 3*1 = 123.0, got {one!r}. 217.0 means the 94 true "
        "negatives were charged for; 2000-something means fn and fp are swapped"
    )
    free = lesson.expected_cost(C(tp=0, fp=0, fn=0, tn=1000), cheap)
    assert np.isclose(free, 0.0), f"a fleet where nothing happens costs 0.0, got {free!r}"
    swept = C(tp=np.array([2, 1]), fp=np.array([8, 0]), fn=np.array([0, 1]),
              tn=np.array([90, 98]))
    over = lesson.expected_cost(swept, cheap)
    assert np.allclose(over, [28.0, 110.0]), (
        f"the same function must work elementwise on a swept Counts: expected [28, 110], got "
        f"{np.asarray(over).tolist()}"
    )
    # Each price must be used for its own cell and nothing else.
    only_fn = lesson.expected_cost(C(tp=0, fp=0, fn=7, tn=0), cheap)
    assert np.isclose(only_fn, 700.0), f"7 missed failures at 100 each is 700.0, got {only_fn!r}"
    only_fp = lesson.expected_cost(C(tp=0, fp=7, fn=0, tn=0), cheap)
    assert np.isclose(only_fp, 7.0), f"7 false alarms at 1 each is 7.0, got {only_fp!r}"


def test_accuracy():
    got = lesson.accuracy(C(tp=2, fp=3, fn=1, tn=94))
    assert np.isclose(got, 0.96), f"(2 + 94)/100 = 0.96, got {got!r}"
    swept = lesson.accuracy(C(tp=np.array([2, 0]), fp=np.array([3, 0]),
                              fn=np.array([1, 3]), tn=np.array([94, 97])))
    assert np.allclose(swept, [0.96, 0.97]), (
        f"expected [0.96, 0.97], got {np.asarray(swept).tolist()} — note that the detector "
        "which catches nothing at all is the more accurate of the two"
    )
    empty = lesson.accuracy(C(tp=0, fp=0, fn=0, tn=0))
    assert np.isfinite(empty) and np.isclose(empty, 0.0), (
        f"an empty fleet has accuracy 0.0 by the stated convention, not NaN; got {empty!r}"
    )


def test_best_operating_point_picks():
    swept = C(tp=np.array([2, 1]), fp=np.array([8, 0]), fn=np.array([0, 1]),
              tn=np.array([90, 98]))
    model = lesson.CostModel(planned=10.0, unplanned=1000.0, false_alarm=1.0)
    grid = np.array([1.0, 2.0])
    by_cost = lesson.best_operating_point(grid, swept, model, "cost")
    by_acc = lesson.best_operating_point(grid, swept, model, "accuracy")
    assert isinstance(by_cost, lesson.OperatingPoint), "return an OperatingPoint"
    assert np.isclose(by_cost.threshold, 1.0) and by_cost.index == 0, (
        f"cost is 28 at threshold 1.0 and 1010 at 2.0, so the minimum is 1.0; got "
        f"{by_cost.threshold} — are you maximising?"
    )
    assert np.isclose(by_acc.threshold, 2.0) and by_acc.index == 1, (
        f"accuracy is 0.92 at 1.0 and 0.99 at 2.0, so the maximum is 2.0; got "
        f"{by_acc.threshold}"
    )
    assert np.isclose(by_cost.cost, 28.0) and np.isclose(by_acc.cost, 1010.0), (
        f"OperatingPoint.cost is the cost AT the chosen threshold; got {by_cost.cost} and "
        f"{by_acc.cost}"
    )
    assert np.isclose(by_cost.accuracy, 0.92), (
        f"OperatingPoint.accuracy is the accuracy at the chosen threshold, got "
        f"{by_cost.accuracy}"
    )
    assert isinstance(by_cost.counts, C), "OperatingPoint.counts must be a Counts"
    assert (int(by_cost.counts.tp), int(by_cost.counts.fp)) == (2, 8), (
        f"counts must be the confusion matrix at the chosen threshold, sliced out of the "
        f"swept arrays; got tp={by_cost.counts.tp}, fp={by_cost.counts.fp}"
    )
    assert (int(by_acc.counts.tp), int(by_acc.counts.fp)) == (1, 0)


def test_best_operating_point_ties_and_objectives():
    tie = C(tp=np.array([1, 1, 1]), fp=np.array([1, 1, 1]), fn=np.array([1, 1, 1]),
            tn=np.array([1, 1, 1]))
    model = lesson.CostModel(planned=1.0, unplanned=1.0, false_alarm=1.0)
    grid = np.array([1.0, 2.0, 3.0])
    for objective in ("cost", "accuracy"):
        chosen = lesson.best_operating_point(grid, tie, model, objective)
        assert chosen.index == 0, (
            f"three thresholds scoring identically on {objective}: take the lowest index, "
            f"got {chosen.index}"
        )
    for bad in ("recall", "COST", "", "min"):
        try:
            lesson.best_operating_point(grid, tie, model, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"objective={bad!r} must raise ValueError")


def test_best_operating_point_rejects_mismatch():
    swept = C(tp=np.array([2, 1]), fp=np.array([8, 0]), fn=np.array([0, 1]),
              tn=np.array([90, 98]))
    model = lesson.CostModel(planned=10.0, unplanned=1000.0, false_alarm=1.0)
    for grid in (np.array([1.0]), np.array([1.0, 2.0, 3.0])):
        try:
            lesson.best_operating_point(grid, swept, model, "cost")
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"{grid.size} thresholds against 2 counts must raise ValueError — indexing a "
                "sweep with the wrong grid returns a plausible threshold for the wrong "
                "confusion matrix, and nothing about the output looks wrong"
            )


# --- the result, end to end ------------------------------------------------------------

def test_the_gap_is_real():
    bursts, fail_hour = lesson.generate_fleet()
    health = lesson.health_index(bursts)
    counts = lesson.sweep_thresholds(health, fail_hour, lesson.THRESHOLDS)
    by_cost = lesson.best_operating_point(lesson.THRESHOLDS, counts, lesson.SCENARIO, "cost")
    by_acc = lesson.best_operating_point(lesson.THRESHOLDS, counts, lesson.SCENARIO,
                                         "accuracy")
    assert by_cost.threshold < by_acc.threshold, (
        f"on this fleet and this cost model the cost-optimal threshold ({by_cost.threshold}) "
        f"must sit BELOW the accuracy-optimal one ({by_acc.threshold}): a missed failure "
        "costs twenty planned interventions, so cost buys alarms that accuracy will not"
    )
    assert by_acc.cost >= 2.0 * by_cost.cost, (
        f"the accuracy-optimal point costs {by_acc.cost:,.0f} against {by_cost.cost:,.0f} — "
        "at least twice as much is expected here; if the two are close, check that "
        "expected_cost charges the unplanned price to fn and not to fp"
    )
    assert int(by_acc.counts.fn) > int(by_cost.counts.fn), (
        "the accuracy-optimal threshold must accept strictly more unplanned failures than "
        f"the cost-optimal one; got {by_acc.counts.fn} against {by_cost.counts.fn}"
    )
    assert by_acc.accuracy > by_cost.accuracy, (
        "by construction the accuracy-optimal point is the more accurate one; if it is not, "
        "best_operating_point is not maximising accuracy"
    )
    # The point of the lesson: a detector that is good by the usual measure.
    fpr, tpr = lesson.roc_points(counts)
    order = np.argsort(np.asarray(fpr))
    auc = float(np.trapezoid(np.asarray(tpr)[order], np.asarray(fpr)[order]))
    assert auc > 0.85, (
        f"the ROC curve built from these counts has AUC {auc:.3f}; the lesson's argument "
        "depends on the detector being good, so an AUC this low means the health index or "
        "the sweep is wrong, not that the economics argument failed"
    )


RUBRIC = [
    (test_burst_rms, 4,
     "np.sqrt(np.mean(x**2, axis=-1)) — np.std subtracts the mean first and RMS does not"),
    (test_causal_rolling_median_values, 5,
     "y[t] = median(x[max(0, t-window+1) : t+1]); the first positions use a growing window"),
    (test_causal_median_is_causal, 4,
     "the slice must END at t. A centred window reads samples that have not happened yet"),
    (test_causal_rolling_median_validates_window, 2,
     "a window of zero or less is a ValueError, not an empty slice"),
    (test_health_index_values, 5,
     "smoothed rms divided by each machine's own baseline, keeping shape (machines, 1)"),
    (test_health_index_baseline_is_raw, 3,
     "the baseline is the median of the RAW rms over the first baseline_hours, not of the "
     "smoothed series"),
    (test_health_index_validates_baseline_hours, 2,
     "a non-positive baseline_hours is a ValueError"),
    (test_alarm_times, 5,
     "first index where health >= threshold, and -1 where there is no crossing at all — "
     "argmax alone returns 0 for a row that never crosses"),
    (test_classify_outcomes_buckets, 4,
     "a survivor that alarms is a false positive; a failure that does not alarm in time is a "
     "false negative; the four counts add to the fleet size"),
    (test_classify_outcomes_lead_time, 5,
     "fail_hour - alarm_hour >= lead_hours is the catch condition; exactly lead_hours counts"),
    (test_classify_outcomes_validates_lead, 2,
     "a negative lead_hours is a ValueError"),
    (test_sweep_thresholds, 4,
     "loop the thresholds, call alarm_times then classify_outcomes, stack the four counts "
     "into arrays"),
    (test_roc_points, 3,
     "return (fpr, tpr): fp/(fp+tn) first, tp/(tp+fn) second"),
    (test_roc_points_zero_denominators, 3,
     "a rate with a zero denominator is 0.0, never NaN — safe_ratio is given to you"),
    (test_pr_points, 3,
     "return (recall, precision): tp/(tp+fn) first, tp/(tp+fp) second"),
    (test_pr_points_precision_convention, 3,
     "precision is 1.0 when nothing is flagged: safe_ratio(tp, tp + fp, 1.0)"),
    (test_expected_cost, 4,
     "tp*planned + fn*unplanned + fp*false_alarm, and nothing for the true negatives"),
    (test_accuracy, 3,
     "(tp + tn) over the fleet size, 0.0 when the fleet is empty"),
    (test_best_operating_point_picks, 5,
     "argmin of the cost array, argmax of the accuracy array, and slice the chosen index out "
     "of every counts field"),
    (test_best_operating_point_ties_and_objectives, 3,
     "argmin and argmax already return the lowest tied index; any objective other than 'cost' "
     "and 'accuracy' is a ValueError"),
    (test_best_operating_point_rejects_mismatch, 2,
     "compare len(thresholds) against each counts field before indexing anything"),
    (test_the_gap_is_real, 5,
     "if the two operating points coincide, check that fn is charged the unplanned price and "
     "that accuracy counts the true negatives"),
]
