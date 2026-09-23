"""Autograder rubric for P03-L09, the capstone. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These rules matter in this particular lesson and are deliberate:

* the capstone's PASS CONDITION is graded by two cases that no amount of model quality can
  satisfy: `test_checklist_refuses_a_threshold_without_its_prices` and
  `test_checklist_refuses_a_threshold_without_its_give_ups`. A `check_specification` that
  waves either through scores well everywhere else and is exactly the defect this module
  exists to prevent, so both are worth more than a shape test.
* `test_causal_health_fits_on_the_commissioning_block_only` is the only case that can tell a
  causal feature from a good one. Every other feature case passes under a whole-window fit.
* `test_causality_margin_perturbs_all_three_channels` grades the probe against features that
  leak through duty and through ambient, not only through raw. A probe that only perturbs
  `raw` passes every other case and certifies a feature that reads the future duty point.
* `test_given_up_lead_boundary_is_inclusive` and `test_given_up_ignores_censored_runs` pin
  the two halves of the definition separately, because an implementation that gets one right
  and the other wrong produces a plausible-looking list of the wrong pumps.
* `test_price_uses_the_mean_not_the_median` uses a right-skewed class where the two differ.
  On a symmetric sample they agree and the bug is invisible — and the outcome whose tail is
  the reason anyone monitors is the one a median under-prices.
* `test_trigger_reads_which_price` exists because `which` is easy to accept and ignore; an
  implementation that always scales `unplanned` passes every other trigger case.
* `test_deployment_wire_threshold_rounds_to_nearest` pins the direction of the rounding.
  Rounding the threshold down makes the gateway alarm earlier than the published number, and
  no other case in this rubric would notice.
* the rubric grades observable behaviour, never one particular call. `check_specification`
  is graded on which ITEM fails, not on the wording of its `detail`, beyond the requirement
  that every detail is a non-empty string a person can act on.
* every validation case names an input EXACTLY ONE STEP outside its bound, and, where the
  bound is itself a legal value, a second case one step inside it. A guard loosened by one
  is silent: `label_runs` writes an order for a pump the fleet does not have into a dict key
  nobody reads and the order vanishes, `censoring_report` returns a NEGATIVE count of
  unobserved unit-hours, and `failures_given_up` reports a missing unit as an IndexError
  from inside `alarm_hours` instead of as the ValueError its docstring promises. A case that
  names a unit id of 9 against a fleet of 3 catches none of those.
* `test_checklist_refuses_a_leak_and_a_programme_that_will_not_fit` is the only place item
  2's second rule is reachable: the lead time has to be greater than zero as well as equal
  to the evidence's, and every other lead-time case breaks the equality first, so the
  document and the evidence have to be moved to zero together.
* `test_deployment_counts_readings_and_moved_alarms` carries one pump with three disagreeing
  readings. It is the only case where a count of readings and a count of pumps are different
  numbers, and without it `disagreeing_readings` can be a count of units.
* the three integration cases at the end measure the lesson's claim on the generated fleet
  rather than asserting it: the reference specification passes its own checklist, the chosen
  threshold really does give up named failures, and the accuracy-optimal threshold really is
  dearer in currency.
"""
import importlib.util, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

Counts = lesson.Counts
Prices = lesson.Prices
Run = lesson.Run
WorkOrder = lesson.WorkOrder
AuditEntry = lesson.AuditEntry
GivenUp = lesson.GivenUp
GatewayLimits = lesson.GatewayLimits
Specification = lesson.Specification
Evidence = lesson.Evidence
FAILURE, CENSORED = lesson.FAILURE, lesson.CENSORED
AUDITED, GUESSED = lesson.AUDITED, lesson.GUESSED
NEVER_ALARMED, TOO_LATE = lesson.NEVER_ALARMED, lesson.TOO_LATE
OUTCOMES = lesson.OUTCOMES
CHECKLIST_ITEMS = lesson.CHECKLIST_ITEMS

# A toy plant whose duty and ambient are not collinear, so the three-parameter fit on the
# first four hours is exact and every expected value below is arithmetic rather than a
# regression output.
D8 = np.array([[0.2, 0.8, 0.2, 0.8, 0.5, 0.3, 0.9, 0.6]])
A8 = np.array([[10.0, 10.0, 30.0, 30.0, 20.0, 25.0, 15.0, 22.0]])
R8 = 1.0 + 2.0 * D8 + 0.1 * A8


def toy_runs(n, end, kind=FAILURE):
    return tuple(Run(i, end, kind) for i in range(n))


# --- exercise 1: causal_health -----------------------------------------------------------

def test_causal_health_is_one_at_commissioning_condition():
    got = lesson.causal_health(R8, D8, A8, baseline_hours=4, window=1)
    assert np.asarray(got).shape == R8.shape, (
        f"the feature has one value per reading: expected {R8.shape}, got "
        f"{np.asarray(got).shape}"
    )
    assert np.allclose(got, 1.0), (
        f"a pump whose reading is exactly a + b*duty + c*ambient must read 1.0 at every "
        f"hour; got {np.asarray(got).round(4).tolist()}"
    )
    doubled = R8.copy()
    doubled[:, 4:] *= 2.0
    grew = lesson.causal_health(doubled, D8, A8, baseline_hours=4, window=1)
    assert np.allclose(grew[:, :4], 1.0) and np.allclose(grew[:, 4:], 2.0), (
        f"the index is a RATIO to the commissioning condition, so a doubled amplitude reads "
        f"2.0; got {np.asarray(grew).round(3).tolist()}"
    )


def test_causal_health_removes_pump_size_and_duty():
    two = np.concatenate([R8, 3.0 * R8])
    got = lesson.causal_health(two, np.concatenate([D8, D8]), np.concatenate([A8, A8]),
                               baseline_hours=4, window=1)
    assert np.allclose(got, 1.0), (
        f"a pump three times the size of its neighbour is not three times as sick; got "
        f"{np.asarray(got).round(3).tolist()}"
    )
    swing = np.array([[0.2, 0.9, 0.2, 0.9, 0.9, 0.2, 0.9, 0.2]])
    amb = np.array([[10.0, 10.0, 30.0, 30.0, 12.0, 28.0, 14.0, 26.0]])
    raw = 1.0 + 2.0 * swing + 0.1 * amb
    flat = lesson.causal_health(raw, swing, amb, baseline_hours=4, window=1)
    assert np.allclose(flat, 1.0), (
        f"the duty and ambient dependence has to be divided out before anything is called "
        f"health; this pump's duty swings between 0.2 and 0.9 and it is not degrading, yet "
        f"the feature reads {np.asarray(flat).round(3).tolist()}"
    )


def test_causal_health_fits_on_the_commissioning_block_only():
    future = R8.copy()
    future[:, -1] = 500.0
    head = lesson.causal_health(future, D8, A8, baseline_hours=4, window=1)[:, :-1]
    assert np.allclose(head, 1.0), (
        f"changing hour 7 moved hours 0-6 to {np.asarray(head).round(3).tolist()}. Both the "
        "duty model and each pump's own level are fitted on the commissioning block; a fit "
        "over the whole window reads the future"
    )
    bent = R8.copy()
    bent[:, 4:] = bent[:, 4:] * 4.0 + 20.0
    tail = lesson.causal_health(bent, D8, A8, baseline_hours=4, window=1)
    assert np.allclose(tail[:, :4], 1.0), (
        f"the first four hours are the commissioning block and cannot move when later hours "
        f"change; got {np.asarray(tail[:, :4]).round(3).tolist()}"
    )
    assert np.all(np.asarray(tail)[:, 4:] > 1.5), (
        "a fourfold rise after the commissioning block has to show up in the feature"
    )


def test_causal_health_smooths_causally_and_validates():
    spiked = R8.copy()
    spiked[0, 5] = 900.0
    smoothed = lesson.causal_health(spiked, D8, A8, baseline_hours=4, window=3)
    assert smoothed[0, 5] < 5.0, (
        f"a single shock at hour 5 left the feature at {smoothed[0, 5]:.1f}; a rolling MEDIAN "
        "shrugs one outlier off and a rolling mean does not"
    )
    assert smoothed[0, 4] < 1.2 and smoothed[0, 6] < 5.0, (
        "the shock must not raise the hour before it at all"
    )
    lead = R8.copy()
    lead[0, 6] = 900.0
    ahead = lesson.causal_health(lead, D8, A8, baseline_hours=4, window=3)
    assert np.isclose(ahead[0, 5], 1.0, atol=1e-6), (
        f"hour 5 came back {ahead[0, 5]:.3f} after hour 6 was changed — the smoothing window "
        "is centred and reads ahead. It must end at the hour being scored"
    )
    # One shock INSIDE the commissioning block, on one of two otherwise identical pumps.
    twin = np.tile(D8, 2)
    twin_a = np.tile(A8, 2)
    clean = 1.0 + 2.0 * twin + 0.1 * twin_a
    pair = np.concatenate([clean, clean])
    pair[1, 9] *= 3.0
    levels = lesson.causal_health(pair, np.concatenate([twin, twin]), np.concatenate([twin_a, twin_a]),
                                       baseline_hours=16, window=1)
    quiet = [h for h in range(16) if h != 9]
    assert np.allclose(levels[0, quiet], levels[1, quiet], rtol=0.03), (
        f"two identical pumps, one of which took a single shock during commissioning, now "
        f"read {levels[0, 0]:.4f} and {levels[1, 0]:.4f}. The commissioning level is a "
        "MEDIAN for the same reason the smoothing is: one shock in the baseline window must "
        "not make a pump look healthier for the rest of its life"
    )
    # A step inside the commissioning block, where the two possible orderings of "normalise"
    # and "smooth" give different answers.
    stepped = np.tile(D8, 2)
    step_a = np.tile(A8, 2)
    level_raw = (1.0 + 2.0 * stepped + 0.1 * step_a) * np.where(np.arange(16) < 8, 1.0, 1.6)
    ordered = lesson.causal_health(level_raw, stepped, step_a, baseline_hours=16, window=5)
    assert np.isclose(ordered[0, 0], 1.0 / 1.3) and np.isclose(ordered[0, -1], 1.6 / 1.3), (
        f"this pump's commissioning ratio steps from 1.0 to 1.6, so its commissioning MEDIAN "
        f"is 1.3 and the two halves must read {1.0 / 1.3:.6f} and {1.6 / 1.3:.6f}; got "
        f"{ordered[0, 0]:.6f} and {ordered[0, -1]:.6f}. Reading 1.0 and 1.6 means you "
        "smoothed BEFORE taking the commissioning level: the level comes from the unsmoothed "
        "ratio and the smoothing is the last step"
    )
    for bad in ({"baseline_hours": 0}, {"baseline_hours": 9}, {"baseline_hours": -2},
                {"window": 0}, {"window": -1}):
        try:
            lesson.causal_health(R8, D8, A8, **{"baseline_hours": 4, "window": 1, **bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"causal_health with {bad} must raise ValueError")
    for args in ((R8, D8, A8[:, :4]), (R8, D8[:, :4], A8), (R8.ravel(), D8, A8)):
        try:
            lesson.causal_health(*args, baseline_hours=4, window=1)
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched or non-2-D inputs must raise ValueError")
    try:
        lesson.causal_health(np.array([[1.0, 1.0, -3.0, 5.0]]), np.array([[0.0, 1.0, 2.0, 3.0]]),
                             np.array([[0.0, 0.0, 1.0, 1.0]]), baseline_hours=4, window=1)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "a fitted duty-and-ambient level that is not strictly positive everywhere must "
            "raise: dividing by it silently flips the sign of the feature")


# --- exercise 2: causality_margin --------------------------------------------------------

def test_causality_margin_is_zero_for_a_causal_feature():
    ones = np.ones((2, 6))
    margin = lesson.causality_margin(lambda r, d, a: r * 1.0, ones * 3.0, ones, ones,
                                     tail_hours=2)
    assert margin == 0.0, (
        f"a feature that reads only the current hour has a margin of exactly 0.0, got "
        f"{margin!r}. Anything non-zero means you compared the perturbed columns too"
    )
    assert isinstance(margin, float), f"the margin is a float, got {type(margin).__name__}"
    trailing = lesson.causality_margin(
        lambda r, d, a: lesson.causal_rolling_median(r, 3), ones, ones, ones, tail_hours=2)
    assert trailing == 0.0, "a trailing median is causal and must score exactly 0.0"


def test_causality_margin_perturbs_all_three_channels():
    ones = np.ones((2, 6))

    def leaks(channel):
        def feature(r, d, a):
            src = {"raw": r, "duty": d, "ambient": a}[channel]
            return np.broadcast_to(src.max(axis=1, keepdims=True), src.shape).copy()
        return feature

    for channel in ("raw", "duty", "ambient"):
        got = lesson.causality_margin(leaks(channel), ones, ones * 0.1, ones, tail_hours=2)
        assert got > 0.0, (
            f"a feature that reads the whole {channel.upper()} row scored {got}; the probe "
            f"has to perturb all three channels, not just raw — a feature that reads next "
            f"week's duty point is exactly as undeployable as one that reads next week's "
            f"vibration"
        )


def test_causality_margin_reads_the_head_and_validates():
    ones = np.ones((1, 6))
    big = lesson.causality_margin(
        lambda r, d, a: np.broadcast_to(r.mean(axis=1, keepdims=True), r.shape).copy(),
        ones, ones, ones, tail_hours=3)
    assert big > 0.0, "a whole-row mean leaks into every hour and must score above zero"
    half = lesson.causality_margin(
        lambda r, d, a: np.broadcast_to(r.mean(axis=1, keepdims=True), r.shape).copy(),
        ones, ones, ones, tail_hours=1)
    assert big > half, (
        f"perturbing three hours moves a whole-row mean further than perturbing one "
        f"({big} against {half}); `tail_hours` is a COUNT of hours at the end, not a start "
        "index"
    )
    # A leak that touches ONE hour of a long head: the worst case is 6.0 and the average is
    # a fifteenth of that, which is the difference between a probe that fails a feature and
    # a probe that reports 0.4 and lets it through.
    def reads_one_hour_ahead(r, d, a):
        return np.concatenate([r[:, 1:], r[:, -1:]], axis=1)

    long_run = np.ones((1, 20))
    single = lesson.causality_margin(reads_one_hour_ahead, long_run, long_run, long_run,
                                     tail_hours=5)
    assert np.isclose(single, 6.0), (
        f"a feature that reads one hour ahead moves exactly one head reading, from 1.0 to "
        f"7.0, so the margin is 6.0; got {single}. A value near 0.4 means you averaged the "
        "change over the head instead of taking the WORST of it — and a leak that touches "
        "one reading in fifteen would then be reported as a rounding error"
    )
    for bad in (0, -1, 6, 11):
        try:
            lesson.causality_margin(lambda r, d, a: r, ones, ones, ones, tail_hours=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"tail_hours={bad} on a 6-hour history must raise ValueError: the probe has "
                "to leave at least one hour of head to read")


# --- exercise 3: label_runs --------------------------------------------------------------

def test_label_runs_one_per_unit_including_the_unmentioned():
    got = lesson.label_runs((), n_units=4, horizon_end=10)
    assert tuple(got) == tuple(Run(u, 10, CENSORED) for u in range(4)), (
        f"an empty log still produces one run per unit, censored at the horizon; got {got}"
    )
    log = (WorkOrder(2, 5, 40, "BM", "mechanical seal failed, unit off line"),)
    runs = lesson.label_runs(log, n_units=4, horizon_end=99)
    assert len(runs) == 4 and [r.unit for r in runs] == [0, 1, 2, 3], (
        f"exactly one run per unit, sorted by unit; got {[r.unit for r in runs]}"
    )
    assert runs[2] == Run(2, 5, FAILURE), f"unit 2 failed at hour 5; got {runs[2]}"
    assert all(r.kind == CENSORED and r.end_hour == 99 for r in (runs[0], runs[1], runs[3])), (
        "a unit the log never mentions is censored at the horizon, not dropped and not a "
        "survivor with a known end"
    )


def test_label_runs_uses_the_raised_hour():
    log = (WorkOrder(0, 5, 88, "BM", "thrust bearing burnt, casing opened"),)
    got = lesson.label_runs(log, n_units=1, horizon_end=99)
    assert got == (Run(0, 5, FAILURE),), (
        f"this policy books the event at the RAISED hour; expected end_hour 5, got {got}. "
        "The closed hour is when somebody typed, not when the pump stopped"
    )


def test_label_runs_censors_suspensions_and_ignores_call_outs():
    susp = lesson.label_runs(
        (WorkOrder(0, 12, 50, "PM", "planned overhaul, unit out of service"),),
        n_units=1, horizon_end=99)
    assert susp == (Run(0, 12, CENSORED),), (
        f"a planned removal is a suspension and is right-censored at that hour; expected "
        f"Run(0, 12, '{CENSORED}'), got {susp}. Calling it a failure invents one, and "
        "calling it a survivor teaches the detector that a pump about to be pulled looks fine"
    )
    for text in ("attended, no fault found, unit restarted", "false trip on low suction",
                 "nff - operator error on start permissive"):
        quiet = lesson.label_runs((WorkOrder(0, 12, 50, "BM", text),), n_units=1,
                                  horizon_end=99)
        assert quiet == (Run(0, 99, CENSORED),), (
            f"a call-out that found nothing ended nothing: {text!r} must leave the unit "
            f"censored at the horizon, got {quiet}"
        )
    both = lesson.label_runs((WorkOrder(0, 12, 50, "BM", "false trip on low suction"),
                              WorkOrder(0, 40, 44, "BM", "drive end bearing seized")),
                             n_units=1, horizon_end=99)
    assert both == (Run(0, 40, FAILURE),), (
        f"the ignored call-out must not block the real failure that follows it; got {both}"
    )


def test_label_runs_earliest_wins_with_failure_on_a_tie():
    first = lesson.label_runs(
        (WorkOrder(0, 60, 61, "BM", "thrust bearing burnt, casing opened"),
         WorkOrder(0, 20, 90, "MOD", "removed for line reconfiguration")),
        n_units=1, horizon_end=99)
    assert first == (Run(0, 20, CENSORED),), (
        f"the EARLIEST surviving order ends the run, whatever order the log arrives in; got "
        f"{first}"
    )
    tie = lesson.label_runs(
        (WorkOrder(0, 30, 90, "PM", "campaign end, pump taken off duty"),
         WorkOrder(0, 30, 33, "BM", "drive end bearing seized, shaft sheared")),
        n_units=1, horizon_end=99)
    assert tie == (Run(0, 30, FAILURE),), (
        f"on a tie in raised hour a FAILURE beats a SUSPENSION — a defect found during "
        f"planned work is still a defect; got {tie}"
    )
    reversed_tie = lesson.label_runs(
        (WorkOrder(0, 30, 33, "BM", "drive end bearing seized, shaft sheared"),
         WorkOrder(0, 30, 90, "PM", "campaign end, pump taken off duty")),
        n_units=1, horizon_end=99)
    assert reversed_tie == (Run(0, 30, FAILURE),), (
        "the tie rule cannot depend on the order the rows happen to arrive in"
    )


def test_label_runs_validates():
    log = (WorkOrder(0, 5, 40, "BM", "mechanical seal failed, unit off line"),)
    for orders, n_units, horizon in ((log, 0, 99), (log, -1, 99), (log, 1, -1),
                                     ((WorkOrder(9, 1, 2, "BM", "seized"),), 3, 99),
                                     ((WorkOrder(-1, 1, 2, "BM", "seized"),), 3, 99),
                                     ((WorkOrder(0, 400, 401, "BM", "seized"),), 3, 99),
                                     ((WorkOrder(0, -5, 2, "BM", "seized"),), 3, 99),
                                     # Each of the next four sits exactly ONE past a bound.
                                     # An off-by-one in the guard is silent here and nowhere
                                     # else: with `<=` for the unit the order is written into
                                     # a dict key nobody reads and vanishes, and with an empty
                                     # log the n_units and horizon_end guards are the only
                                     # things standing between a typo and a run table.
                                     ((WorkOrder(3, 1, 2, "BM", "seized"),), 3, 99),
                                     ((WorkOrder(0, 100, 101, "BM", "seized"),), 3, 99),
                                     ((), 0, 99),
                                     ((), 2, -1)):
        try:
            lesson.label_runs(orders, n_units=n_units, horizon_end=horizon)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"label_runs({orders!r}, n_units={n_units}, horizon_end={horizon}) must "
                "raise ValueError. A unit id equal to n_units, a raised hour one past "
                "horizon_end, an empty fleet and a negative horizon are each exactly one "
                "step outside the bound, which is where an off-by-one lives: compare the "
                "unit with `< n_units` and the hour with `<= horizon_end`, and reject "
                "n_units < 1 and horizon_end < 0 before looking at the log")
    # ...and the bound itself is INSIDE: an order raised in the horizon's own last hour is a
    # real event, and a guard tightened to `< horizon_end` throws it away.
    edge = lesson.label_runs((WorkOrder(0, 99, 99, "BM", "drive end bearing seized"),),
                             n_units=1, horizon_end=99)
    assert edge == (Run(0, 99, FAILURE),), (
        f"a failure raised at hour 99 of a 0..99 horizon is inside the window and must be "
        f"booked, not rejected and not censored; got {edge}"
    )


# --- exercise 4: censoring_report --------------------------------------------------------

def test_censoring_report_counts_and_units():
    got = lesson.censoring_report((Run(0, 99, CENSORED), Run(1, 40, FAILURE),
                                   Run(2, 60, CENSORED)), n_hours=100)
    assert (got.n_failures, got.n_censored) == (1, 2), (
        f"one failure and two censored runs; got {got.n_failures} and {got.n_censored}"
    )
    assert tuple(got.censored_units) == (0, 2), (
        f"censored_units is a sorted tuple of unit ids, got {got.censored_units!r}"
    )
    assert np.isclose(got.censored_fraction, 2 / 3), (
        f"censored_fraction is the share of RUNS, 2/3 here; got {got.censored_fraction}"
    )
    unsorted = lesson.censoring_report((Run(5, 10, CENSORED), Run(1, 10, CENSORED)),
                                       n_hours=20)
    assert tuple(unsorted.censored_units) == (1, 5), (
        f"the report sorts the units whatever order the runs arrive in; got "
        f"{unsorted.censored_units!r}"
    )


def test_censoring_report_unobserved_hours():
    got = lesson.censoring_report((Run(0, 99, CENSORED), Run(1, 40, FAILURE),
                                   Run(2, 60, CENSORED)), n_hours=100)
    assert got.unobserved_unit_hours == 39, (
        f"expected 39: unit 0 ran to the last hour and hid nothing, unit 2 stopped at hour "
        f"60 of 0..99 and hid 39. Got {got.unobserved_unit_hours}. A total of 98 means you "
        "counted the FAILURE run's remaining hours as unobserved — a pump that broke did not "
        "hide its failure from you; 40 means you counted the end hour itself twice"
    )
    clean = lesson.censoring_report((Run(0, 10, FAILURE),), n_hours=11)
    assert tuple(clean.censored_units) == () and clean.unobserved_unit_hours == 0, (
        f"a fleet with no censored runs reports an empty tuple and zero hours, got {clean}"
    )
    early = lesson.censoring_report((Run(0, 0, CENSORED),), n_hours=50)
    assert early.unobserved_unit_hours == 49, (
        f"a unit censored in its first hour of a 50-hour window hid 49; got "
        f"{early.unobserved_unit_hours}"
    )


def test_censoring_report_validates():
    for runs, n_hours in (((), 10), ((Run(0, 5, FAILURE),), 0), ((Run(0, 5, FAILURE),), -3),
                          ((Run(0, 5, "suspension"),), 10), ((Run(0, 5, "unknown"),), 10),
                          ((Run(0, 50, FAILURE),), 10), ((Run(0, -1, FAILURE),), 10),
                          # Exactly one hour past the end of a 0..n_hours-1 window. A guard
                          # written `<= n_hours` lets this through and the report comes back
                          # with a NEGATIVE unobserved_unit_hours, which no other case sees.
                          ((Run(0, 10, CENSORED),), 10),
                          ((Run(0, 10, FAILURE),), 10)):
        try:
            lesson.censoring_report(runs, n_hours=n_hours)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"censoring_report({runs!r}, n_hours={n_hours}) must raise ValueError. The "
                f"last hour of an {n_hours}-hour window is hour {n_hours - 1}: an end hour "
                f"of {n_hours} is off the end, and accepting it reports a negative number of "
                "unobserved unit-hours")
    # The bound itself is a real run: hour n_hours - 1 is the last hour the historian holds.
    last = lesson.censoring_report((Run(0, 9, CENSORED),), n_hours=10)
    assert last.unobserved_unit_hours == 0 and last.n_censored == 1, (
        f"a run censored in the window's LAST hour is valid and hid nothing; got {last}"
    )


# --- exercise 5: price_the_outcomes ------------------------------------------------------

def test_price_averages_what_it_has():
    rows = (AuditEntry(1, 0, "planned", 10.0), AuditEntry(2, 1, "planned", 20.0))
    ev = lesson.price_the_outcomes(rows, Prices(1.0, 2.0, 3.0), min_observations=2)
    assert ev.prices == Prices(15.0, 2.0, 3.0), (
        f"the mean of the two planned invoices and the fallback for the rest; got {ev.prices}"
    )
    assert tuple(ev.counts) == (2, 0, 0), (
        f"counts report what the AUDIT holds, in OUTCOMES order, not what was used; got "
        f"{ev.counts!r}"
    )
    assert tuple(ev.provenance) == (AUDITED, GUESSED, GUESSED), (
        f"one provenance label per price, in OUTCOMES order; got {ev.provenance!r}"
    )
    full = tuple(AuditEntry(i, i, name, 100.0 * (k + 1))
                 for k, name in enumerate(OUTCOMES) for i in range(3))
    both = lesson.price_the_outcomes(full, Prices(1.0, 2.0, 3.0), min_observations=3)
    assert tuple(both.provenance) == (AUDITED, AUDITED, AUDITED), (
        f"all three classes clear the minimum here; got {both.provenance!r}"
    )
    assert both.prices == Prices(100.0, 200.0, 300.0), (
        f"the three prices come back in OUTCOMES order; got {both.prices}. A shuffled order "
        "prices a missed failure as a false alarm and is invisible in every other case"
    )


def test_price_keeps_the_guess_for_a_thin_class():
    rows = (AuditEntry(1, 0, "planned", 10.0), AuditEntry(2, 1, "planned", 20.0))
    short = lesson.price_the_outcomes(rows, Prices(1.0, 2.0, 3.0), min_observations=3)
    assert short.prices.planned == 1.0, (
        f"two invoices against a minimum of three is not an estimate; the price stays at the "
        f"fallback 1.0, got {short.prices.planned}"
    )
    assert short.provenance[0] == GUESSED, (
        f"a fallen-back price is reported as {GUESSED!r}, got {short.provenance[0]!r}"
    )
    assert tuple(short.counts) == (2, 0, 0), (
        f"counts still report the two jobs the audit holds; got {short.counts!r}"
    )
    empty = lesson.price_the_outcomes((), Prices(7.0, 8.0, 9.0), min_observations=1)
    assert empty.prices == Prices(7.0, 8.0, 9.0), (
        f"an empty audit keeps all three guesses; got {empty.prices}. np.mean of an empty "
        "list is nan and a 0.0 prices a missed failure at nothing, which makes 'never alarm' "
        "the cost-optimal policy"
    )
    assert tuple(empty.provenance) == (GUESSED, GUESSED, GUESSED), (
        f"and says so on every line; got {empty.provenance!r}"
    )
    assert all(np.isfinite(v) for v in empty.prices), "no price may come back nan"


def test_price_uses_the_mean_not_the_median():
    skew = tuple(AuditEntry(i, i, "unplanned", c) for i, c in
                 enumerate([10.0, 10.0, 10.0, 10.0, 60.0]))
    got = lesson.price_the_outcomes(skew, Prices(1.0, 2.0, 3.0), min_observations=5)
    assert np.isclose(got.prices.unplanned, 20.0), (
        f"the price is the arithmetic MEAN of the invoices, 20.0 on this right-skewed class; "
        f"got {got.prices.unplanned}. A median gives 10.0 and systematically under-prices "
        "the outcome whose tail is the reason anyone is monitoring"
    )


def test_price_validates():
    rows = (AuditEntry(1, 0, "planned", 10.0),)
    for audit, kwargs in (((AuditEntry(1, 0, "planned_repair", 10.0),), {}),
                          ((AuditEntry(1, 0, "", 10.0),), {}),
                          ((AuditEntry(1, 0, "planned", -5.0),), {}),
                          # Just below zero, where a guard written against a round negative
                          # rather than against zero lets an invoice through and drags the
                          # mean of the class down with it.
                          ((AuditEntry(1, 0, "planned", -0.5),), {}),
                          ((AuditEntry(1, 0, "planned", np.nan),), {}),
                          ((AuditEntry(1, 0, "planned", np.inf),), {}),
                          (rows, {"min_observations": 0}),
                          (rows, {"min_observations": -2}),
                          (rows, {"fallback": Prices(1.0, -2.0, 3.0)}),
                          (rows, {"fallback": Prices(1.0, -0.5, 3.0)}),
                          (rows, {"fallback": Prices(1.0, np.nan, 3.0)})):
        try:
            lesson.price_the_outcomes(audit, **{"fallback": Prices(1.0, 2.0, 3.0),
                                                "min_observations": 1, **kwargs})
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"price_the_outcomes({audit!r}, {kwargs!r}) must raise ValueError")


# --- exercise 6: choose_operating_point --------------------------------------------------

def test_choose_minimises_cost():
    counts = Counts(tp=np.array([2, 1]), fp=np.array([8, 0]), fn=np.array([0, 1]),
                    tn=np.array([90, 98]))
    dear = Prices(planned=10.0, unplanned=1000.0, false_alarm=1.0)
    point = lesson.choose_operating_point(np.array([1.0, 2.0]), counts, dear)
    assert (point.index, point.threshold) == (0, 1.0), (
        f"with a missed failure at 1000 and a false alarm at 1 the cheap threshold is 1.00 "
        f"at index 0; got index {point.index}, threshold {point.threshold}. A threshold of "
        "2.00 means you maximised something — accuracy is 0.99 there and 0.92 at the right "
        "answer"
    )
    assert np.isclose(point.cost, 28.0), f"cost at the chosen index is 28.0, got {point.cost}"
    assert np.isclose(point.accuracy, 0.92), (
        f"accuracy at the chosen index is 0.92, got {point.accuracy}. Reporting the BEST "
        "accuracy over the sweep instead of the accuracy at the chosen threshold hides the "
        "whole argument"
    )


def test_choose_reads_the_prices():
    counts = Counts(tp=np.array([2, 1]), fp=np.array([8, 0]), fn=np.array([0, 1]),
                    tn=np.array([90, 98]))
    cheap = Prices(planned=10.0, unplanned=12.0, false_alarm=1000.0)
    assert lesson.choose_operating_point(np.array([1.0, 2.0]), counts, cheap).threshold == 2.0, (
        "re-price a false alarm at 1000 and the same counts choose the other threshold; a "
        "chooser that ignores `prices` passes the first case and fails the plant"
    )
    middling = Prices(planned=1.0, unplanned=100.0, false_alarm=10.0)
    three = Counts(tp=np.array([3, 2, 1]), fp=np.array([20, 5, 0]), fn=np.array([0, 1, 2]),
                   tn=np.array([77, 92, 97]))
    got = lesson.choose_operating_point(np.array([1.0, 2.0, 3.0]), three, middling)
    assert got.threshold == 2.0 and got.index == 1, (
        f"costs here are 203, 152 and 201, so the middle threshold wins; got "
        f"{got.threshold} at index {got.index}"
    )


def test_choose_breaks_ties_low_and_returns_ints():
    flat = Counts(tp=np.array([1, 1, 1]), fp=np.array([1, 1, 1]), fn=np.array([1, 1, 1]),
                  tn=np.array([1, 1, 1]))
    dear = Prices(planned=10.0, unplanned=1000.0, false_alarm=1.0)
    tie = lesson.choose_operating_point(np.array([1.0, 2.0, 3.0]), flat, dear)
    assert tie.threshold == 1.0 and tie.index == 0, (
        f"on a flat cost curve take the LOWEST threshold — of two equally priced choices the "
        f"one that alarms earlier is the defensible one; got {tie.threshold}"
    )
    counts = Counts(tp=np.array([2, 1]), fp=np.array([8, 0]), fn=np.array([0, 1]),
                    tn=np.array([90, 98]))
    point = lesson.choose_operating_point(np.array([1.0, 2.0]), counts, dear)
    assert tuple(point.counts) == (2, 8, 0, 90), (
        f"the returned counts are the four numbers at the CHOSEN index, got {point.counts!r}"
    )
    assert all(isinstance(v, int) for v in point.counts), (
        f"they are plain ints, not numpy scalars or slices of the swept arrays; got "
        f"{[type(v).__name__ for v in point.counts]}"
    )
    assert isinstance(point.threshold, float) and isinstance(point.cost, float), (
        "threshold and cost come back as plain floats"
    )


def test_choose_validates():
    counts = Counts(tp=np.array([2, 1]), fp=np.array([8, 0]), fn=np.array([0, 1]),
                    tn=np.array([90, 98]))
    dear = Prices(10.0, 1000.0, 1.0)
    for grid in (np.array([]), np.array([1.0, 2.0, 3.0]), np.array([1.0])):
        try:
            lesson.choose_operating_point(grid, counts, dear)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"a grid of {grid.size} thresholds against counts of length 2 must raise "
                "ValueError: a silent mismatch reports the wrong threshold for the right "
                "confusion matrix")


# --- exercise 7: failures_given_up -------------------------------------------------------

def test_given_up_names_reason_and_warning():
    health = np.array([[1.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
    runs = (Run(0, 2, FAILURE), Run(1, 2, FAILURE))
    got = lesson.failures_given_up(health, runs, 1.5, lead_hours=2)
    assert got == (GivenUp(0, 2, 1, 1, TOO_LATE), GivenUp(1, 2, -1, -1, NEVER_ALARMED)), (
        f"expected pump 0 late with one hour of warning and pump 1 never alarmed; got {got}"
    )
    assert got[1].alarm_hour == -1 and got[1].warning_hours == -1, (
        f"a pump that never alarmed has no alarm hour and no warning: both are -1, not 0 and "
        f"not the end hour; got {got[1]}"
    )


def test_given_up_lead_boundary_is_inclusive():
    health = np.array([[1.0, 2.0, 2.0]])
    exactly = lesson.failures_given_up(health, (Run(0, 2, FAILURE),), 1.5, lead_hours=1)
    assert exactly == (), (
        f"exactly lead_hours of warning counts as CAUGHT — compare with >=, not > — so this "
        f"pump must not appear; got {exactly}"
    )
    one_more = lesson.failures_given_up(health, (Run(0, 2, FAILURE),), 1.5, lead_hours=2)
    assert len(one_more) == 1 and one_more[0].reason == TOO_LATE, (
        f"one hour of warning against a two-hour requirement is a give-up; got {one_more}"
    )
    caught = lesson.failures_given_up(health, (Run(0, 2, FAILURE),), 1.5, lead_hours=0)
    assert caught == (), "with no lead requirement an alarm at hour 1 is a catch"


def test_given_up_ignores_censored_runs():
    health = np.array([[1.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
    censored = lesson.failures_given_up(health, (Run(0, 2, CENSORED), Run(1, 2, CENSORED)),
                                        1.5, lead_hours=2)
    assert censored == (), (
        f"a censored run is not a failure and can never be a failure you gave up; got "
        f"{censored}. A suspension that alarmed is a false alarm and classify_runs counts it "
        "as one"
    )
    mixed = lesson.failures_given_up(health, (Run(0, 2, CENSORED), Run(1, 2, FAILURE)),
                                     1.5, lead_hours=2)
    assert [g.unit for g in mixed] == [1], (
        f"only the FAILURE run can appear; got {[g.unit for g in mixed]}"
    )
    after = lesson.failures_given_up(np.array([[1.0, 1.0, 9.0]]), (Run(0, 1, FAILURE),), 1.5,
                                     lead_hours=0)
    assert after == (GivenUp(0, 1, -1, -1, NEVER_ALARMED),), (
        f"the crossing at hour 2 happened after the pump had already stopped at hour 1, so "
        f"it is not an alarm; got {after}"
    )


def test_given_up_sorted_and_validates():
    order = lesson.failures_given_up(np.ones((3, 2)),
                                     (Run(2, 1, FAILURE), Run(0, 1, FAILURE)), 5.0,
                                     lead_hours=1)
    assert [g.unit for g in order] == [0, 2], (
        f"the report comes back sorted by unit whatever order the runs arrive in; got "
        f"{[g.unit for g in order]}"
    )
    health = np.array([[1.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
    runs = (Run(0, 2, FAILURE), Run(1, 2, FAILURE))
    for bad in ({"threshold": np.nan}, {"threshold": np.inf}, {"threshold": -np.inf},
                {"lead_hours": -1}):
        try:
            lesson.failures_given_up(health, runs, **{"threshold": 1.5, "lead_hours": 2,
                                                      **bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"failures_given_up with {bad} must raise ValueError")
    try:
        lesson.failures_given_up(health, (Run(7, 2, FAILURE),), 1.5, lead_hours=2)
    except ValueError:
        pass
    else:
        raise AssertionError("a run naming a unit with no row in health must raise ValueError")
    # health has two rows, so unit 2 is the FIRST unit that does not exist — one past the
    # bound. A guard written `> h.shape[0]` waves it through and the pump is reported on by
    # an IndexError from deep inside alarm_hours instead of by this function.
    try:
        lesson.failures_given_up(health, (Run(2, 2, FAILURE),), 1.5, lead_hours=2)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "health has 2 rows, so unit 2 has no row: compare with `>= h.shape[0]`, not "
            "`> h.shape[0]`, and raise ValueError here rather than letting the indexing "
            "fail later with a message about an axis")


# --- exercise 8: deployment_report -------------------------------------------------------

def test_deployment_bytes_and_fit():
    lim = GatewayLimits(arena_bytes=40, sample_bytes=4, constant_bytes=12, wire_frac=8)
    got = lesson.deployment_report(np.zeros((2, 2)), 1.5, window=4, limits=lim)
    assert got.bytes_per_unit == 28, (
        f"4 hours x 4 bytes + 12 bytes of constants = 28 per pump; got {got.bytes_per_unit}. "
        "Forgetting the constants understates every gateway budget by a fixed amount"
    )
    assert got.total_bytes == 56, (
        f"two pumps at 28 bytes is 56; the row count comes from `health`, got "
        f"{got.total_bytes}"
    )
    assert got.fits is False, "56 bytes do not fit a 40-byte arena"
    exact = lesson.deployment_report(np.zeros((2, 2)), 1.5, window=4,
                                     limits=lim._replace(arena_bytes=56))
    assert exact.fits is True, "exactly filling the arena fits: compare with <=, not <"
    assert isinstance(exact.fits, bool), "`fits` is a bool a specification can print"


def test_deployment_wire_threshold_rounds_to_nearest():
    lim = GatewayLimits(arena_bytes=10 ** 6, sample_bytes=4, constant_bytes=12, wire_frac=8)
    exact = lesson.deployment_report(np.zeros((1, 2)), 1.5, window=1, limits=lim)
    assert np.isclose(exact.wire_threshold, 1.5), (
        f"1.5 is exactly representable in Q8.8 and comes back unchanged; got "
        f"{exact.wire_threshold}"
    )
    coarse = lim._replace(wire_frac=1)
    up = lesson.deployment_report(np.zeros((1, 2)), 1.30, window=1, limits=coarse)
    assert np.isclose(up.wire_threshold, 1.5), (
        f"with one fractional bit the grid is 0.5 wide and 1.30 rounds to the NEAREST "
        f"multiple, 1.5; got {up.wire_threshold}. Rounding down would make the gateway alarm "
        "earlier than the number the specification published"
    )
    down = lesson.deployment_report(np.zeros((1, 2)), 1.20, window=1, limits=coarse)
    assert np.isclose(down.wire_threshold, 1.0), (
        f"1.20 rounds down to 1.0 on the same grid; got {down.wire_threshold}"
    )
    fine = lesson.deployment_report(np.zeros((1, 2)), 1.15, window=1, limits=lim)
    assert np.isclose(fine.wire_threshold, round(1.15 * 256) / 256), (
        f"1.15 is not representable in Q8.8 and becomes {round(1.15 * 256) / 256}; got "
        f"{fine.wire_threshold}"
    )


def test_deployment_counts_readings_and_moved_alarms():
    coarse = GatewayLimits(arena_bytes=10 ** 6, sample_bytes=4, constant_bytes=12,
                           wire_frac=1)
    quiet = lesson.deployment_report(np.array([[1.30, 1.26]]), 1.30, window=1, limits=coarse)
    assert quiet.disagreeing_readings == 1, (
        f"on the wire both readings round to 1.5 and clear the 1.5 threshold; in float only "
        f"the first clears 1.30, so exactly one reading changes side. Got "
        f"{quiet.disagreeing_readings} — rounding the health index but comparing it against "
        "the unrounded threshold, or the reverse, misses this"
    )
    assert quiet.units_whose_alarm_moves == 0, (
        f"both crossings still start at hour 0, so no pump's first alarm moved; got "
        f"{quiet.units_whose_alarm_moves}. A reading that changes side in the middle of a "
        "quiet week is not an alarm that moved"
    )
    moved = lesson.deployment_report(np.array([[1.26, 1.30]]), 1.30, window=1, limits=coarse)
    assert moved.units_whose_alarm_moves == 1, (
        f"in float this pump first crosses at hour 1 and on the wire at hour 0, so its first "
        f"alarm moved; got {moved.units_whose_alarm_moves}"
    )
    # The threshold rounds DOWN on this grid and a reading rounds UP to meet it: on the wire
    # the gateway alarms on a pump the published threshold would have left alone.
    both_ways = lesson.deployment_report(np.array([[1.10]]), 1.20, window=1, limits=coarse)
    assert both_ways.disagreeing_readings == 1, (
        f"1.20 becomes 1.0 on this grid and 1.10 becomes 1.0 too, so the wire alarms where "
        f"the float does not: one reading disagrees, got {both_ways.disagreeing_readings}. "
        "A zero here means you compared the rounded health index against the UNROUNDED "
        "threshold, which is not what the gateway does"
    )
    assert both_ways.units_whose_alarm_moves == 1, (
        f"and that pump never alarms in float and alarms at hour 0 on the wire, so its first "
        f"alarm moved; got {both_ways.units_whose_alarm_moves}"
    )
    never = lesson.deployment_report(np.array([[1.0, 1.0]]), 9.0, window=1, limits=coarse)
    assert never.units_whose_alarm_moves == 0 and never.disagreeing_readings == 0, (
        f"a pump that never crosses in either arithmetic has not moved; got {never}"
    )
    appears = lesson.deployment_report(np.array([[1.26, 1.26]]), 1.30, window=1,
                                       limits=coarse)
    assert appears.units_whose_alarm_moves == 1, (
        f"in float this pump never crosses and on the wire it crosses at hour 0; 'never "
        f"crosses' is a value of its own and that pump HAS moved. Got "
        f"{appears.units_whose_alarm_moves}"
    )
    # One pump, three hours, all three disagreeing: the only case in this rubric where a
    # count of READINGS and a count of PUMPS are different numbers.
    many = lesson.deployment_report(np.array([[1.26, 1.26, 1.26]]), 1.30, window=1,
                                    limits=coarse)
    assert many.disagreeing_readings == 3, (
        f"all three readings round up to 1.5 and clear the rounded threshold while none of "
        f"them clears 1.30 in float, so THREE readings disagree; got "
        f"{many.disagreeing_readings}. A 1 here means you counted the pumps that hold a "
        "disagreement rather than the readings, and the specification publishes a reading "
        "count"
    )
    assert many.units_whose_alarm_moves == 1, (
        f"and all three belong to one pump, so exactly one alarm moved; got "
        f"{many.units_whose_alarm_moves}. These two numbers answer different questions and "
        "must not be the same count"
    )


def test_deployment_validates():
    lim = GatewayLimits(arena_bytes=40, sample_bytes=4, constant_bytes=12, wire_frac=8)
    for health, kwargs in ((np.zeros(4), {}), (np.zeros((2, 2, 2)), {}),
                           (np.zeros((2, 2)), {"threshold": np.nan}),
                           (np.zeros((2, 2)), {"threshold": np.inf}),
                           (np.zeros((2, 2)), {"window": 0}),
                           (np.zeros((2, 2)), {"window": -4}),
                           (np.zeros((2, 2)), {"limits": lim._replace(arena_bytes=0)}),
                           (np.zeros((2, 2)), {"limits": lim._replace(sample_bytes=0)}),
                           (np.zeros((2, 2)), {"limits": lim._replace(constant_bytes=-1)}),
                           (np.zeros((2, 2)), {"limits": lim._replace(wire_frac=0)}),
                           (np.zeros((2, 2)), {"limits": lim._replace(wire_frac=31)})):
        try:
            lesson.deployment_report(health, **{"threshold": 1.5, "window": 4,
                                                "limits": lim, **kwargs})
        except ValueError:
            pass
        else:
            raise AssertionError(f"deployment_report with {kwargs!r} must raise ValueError")


# --- exercise 9: rederivation_trigger ----------------------------------------------------

_CLOSE = Counts(tp=np.array([2, 1]), fp=np.array([20, 2]), fn=np.array([1, 2]),
                tn=np.array([77, 95]))
_GRID2 = np.array([1.0, 2.0])
_WIDE = Counts(tp=np.array([2, 1]), fp=np.array([8, 0]), fn=np.array([0, 1]),
               tn=np.array([90, 98]))


def test_trigger_reports_the_smallest_move():
    got = lesson.rederivation_trigger(_GRID2, _CLOSE, Prices(10.0, 30.0, 1.0),
                                      np.array([1.0, 0.5, 0.95, 0.9]), 0.033)
    assert got.threshold == 1.0, (
        f"the threshold in force at the unscaled prices is 1.00; got {got.threshold}"
    )
    assert np.isclose(got.price_move_fraction, 0.1), (
        f"0.50 and 0.90 both move this threshold and 0.90 is the nearer, so the answer is "
        f"abs(0.90 - 1) = 0.10; got {got.price_move_fraction}. Reporting the first mover met "
        "in the grid gives 0.50, and reporting the multiplier itself gives 0.90"
    )
    assert got.moved_threshold == 2.0, (
        f"moved_threshold is what the threshold becomes at the SMALLEST moving multiplier; "
        f"got {got.moved_threshold}"
    )
    assert got.drift_alert == 0.033, (
        f"drift_alert is carried through unchanged; got {got.drift_alert}"
    )
    rising = lesson.rederivation_trigger(_GRID2, _CLOSE, Prices(10.0, 27.0, 1.0),
                                         np.array([1.0, 1.5, 1.1]), 0.0)
    assert np.isclose(rising.price_move_fraction, 0.1), (
        f"a rise of 10% moves this one; abs(m - 1) treats a rise and a fall as the same size "
        f"of move; got {rising.price_move_fraction}"
    )


def test_trigger_reports_inf_when_nothing_moves():
    stuck = lesson.rederivation_trigger(_GRID2, _WIDE, Prices(10.0, 1000.0, 1.0),
                                        np.array([1.0, 1.05, 0.95]), 0.0)
    assert stuck.price_move_fraction == float("inf"), (
        f"no multiplier in that grid moves the threshold, so the honest report is inf — "
        f"'not within the range we looked at', never 'never'. Got "
        f"{stuck.price_move_fraction}; 0.0 would say the threshold is about to move"
    )
    assert stuck.moved_threshold == stuck.threshold == 1.0, (
        f"and moved_threshold is the unchanged threshold; got {stuck.moved_threshold}"
    )


def test_trigger_reads_which_price():
    unplanned = lesson.rederivation_trigger(_GRID2, _WIDE, Prices(10.0, 1000.0, 1.0),
                                            np.array([1.0, 500.0]), 0.0, which="false_alarm")
    assert unplanned.moved_threshold == 2.0, (
        f"scaling the FALSE ALARM price by 500 moves this threshold to 2.00; got "
        f"{unplanned.moved_threshold}. An implementation that always scales `unplanned` "
        "passes every other trigger case"
    )
    planned = lesson.rederivation_trigger(_GRID2, _WIDE, Prices(10.0, 1000.0, 1.0),
                                          np.array([1.0, 500.0]), 0.0, which="planned")
    assert planned.moved_threshold == 2.0 and np.isclose(planned.price_move_fraction, 499.0), (
        f"and scaling the PLANNED price by 500 moves it too; got "
        f"{planned.moved_threshold} at {planned.price_move_fraction}"
    )


def test_trigger_validates():
    prices = Prices(10.0, 1000.0, 1.0)
    for kwargs in ({"which": "unplanned_repair"}, {"which": "Unplanned"}, {"which": ""},
                   {"multipliers": np.array([])},
                   {"multipliers": np.array([1.0, 0.0])},
                   {"multipliers": np.array([1.0, -1.0])},
                   {"multipliers": np.array([1.0, np.nan])},
                   {"drift_alert": -0.1}, {"drift_alert": np.nan}):
        try:
            lesson.rederivation_trigger(_GRID2, _WIDE, prices,
                                        **{"multipliers": _GRID2, "drift_alert": 0.0,
                                           **kwargs})
        except ValueError:
            pass
        else:
            raise AssertionError(f"rederivation_trigger with {kwargs!r} must raise ValueError")


# --- exercise 10: check_specification ----------------------------------------------------

def _pair():
    return lesson.toy_pair()


def test_checklist_shape_and_order():
    spec, ev = _pair()
    results = lesson.check_specification(spec, ev)
    assert len(results) == len(CHECKLIST_ITEMS), (
        f"one result per checklist item: expected {len(CHECKLIST_ITEMS)}, got {len(results)}"
    )
    assert tuple(r.item for r in results) == CHECKLIST_ITEMS, (
        f"the results come back in CHECKLIST_ITEMS order; got {[r.item for r in results]}"
    )
    assert all(isinstance(r.detail, str) and r.detail.strip() for r in results), (
        "every result carries a non-empty detail, including the passing ones — a checklist "
        "is read by a person who has to act on it"
    )
    assert all(isinstance(r.ok, (bool, np.bool_)) for r in results), (
        "`ok` is a boolean, not a score or a string"
    )


def test_checklist_accepts_a_consistent_specification():
    spec, ev = _pair()
    results = lesson.check_specification(spec, ev)
    assert all(r.ok for r in results), (
        "this specification agrees with its evidence on every line and must pass all eight: "
        f"{[(r.item, r.detail) for r in results if not r.ok]}"
    )
    assert lesson.specification_passes(results) is True, (
        "specification_passes must agree with eight ok results"
    )


def test_checklist_each_item_fails_on_its_own_line():
    spec, ev = _pair()

    def only(field, value, index):
        broken = lesson.check_specification(spec._replace(**{field: value}), ev)
        bad = [i for i, r in enumerate(broken) if not r.ok]
        assert bad == [index], (
            f"setting {field}={value!r} must fail item {index} ({CHECKLIST_ITEMS[index]!r}) "
            f"and nothing else; the failing items were "
            f"{[CHECKLIST_ITEMS[i] for i in bad]}"
        )
        assert not lesson.specification_passes(broken), (
            f"specification_passes must be False when item {index} fails"
        )

    only("feature", "", 0)
    only("feature", "   ", 0)
    only("causal_margin", 0.5, 0)
    only("labelling_policy", "", 1)
    only("censors_units", (1, 4), 1)
    only("censors_units", (4, 1, 7), 1)
    only("lead_hours", 12, 2)
    only("threshold", 2.0, 4)
    only("deployment_constraint", "  ", 6)
    only("memory_bytes", 561, 6)
    only("rederive_when", ("only one",), 7)
    only("rederive_when", (), 7)
    only("rederive_when", ("a", ""), 7)
    only("price_move_fraction", 0.25, 7)
    only("drift_alert", 0.04, 7)


def test_checklist_refuses_a_threshold_without_its_prices():
    spec, ev = _pair()
    for field, value in (("prices", Prices(10.0, 90.0, 5.0)),
                         ("prices", Prices(11.0, 100.0, 5.0)),
                         ("prices", Prices(10.0, 100.0, 0.0)),
                         # the same three numbers in the wrong order: a document that prices
                         # a missed failure as a planned repair is worse than one that leaves
                         # the prices out, because it looks complete
                         ("prices", Prices(100.0, 10.0, 5.0)),
                         ("prices", Prices(5.0, 100.0, 10.0)),
                         ("price_provenance", (AUDITED, AUDITED, AUDITED)),
                         ("price_provenance", (GUESSED, GUESSED, GUESSED)),
                         ("price_provenance", (AUDITED, GUESSED, AUDITED))):
        broken = lesson.check_specification(spec._replace(**{field: value}), ev)
        assert not broken[3].ok, (
            f"a specification whose {field} is {value!r} does not match the evidence and "
            f"must FAIL item 3, 'three prices carry their provenance'. This is the capstone's "
            "pass condition: a threshold reported without the prices it came from, or with "
            "prices whose evidence is not stated, is refused however good the detector is"
        )
        assert not lesson.specification_passes(broken), (
            "and the submission is rejected, not warned about"
        )
    good = lesson.check_specification(spec, ev)
    assert good[3].ok, "the matching specification must still pass item 3"


def test_checklist_refuses_a_threshold_without_its_give_ups():
    spec, ev = _pair()
    measured = tuple(g.unit for g in ev.given_up)
    assert measured, "the toy evidence must name at least one given-up failure"
    for value in ((), measured[:-1], measured + (11,), tuple(reversed(measured))):
        if value == measured:
            continue
        broken = lesson.check_specification(spec._replace(gives_up=value), ev)
        assert not broken[5].ok, (
            f"a specification naming {value!r} as given up against evidence of {measured!r} "
            "must FAIL item 5, 'failures given up are named'. An empty list in particular is "
            "the specification almost everybody writes, and it is the one this whole "
            "programme exists to refuse"
        )
        assert not lesson.specification_passes(broken), (
            "and the submission is rejected, not warned about"
        )
    assert lesson.check_specification(spec, ev)[5].ok, (
        "naming exactly the units the evidence gives up must pass"
    )


def test_checklist_refuses_a_leak_and_a_programme_that_will_not_fit():
    spec, ev = _pair()
    leaky = lesson.check_specification(spec._replace(causal_margin=0.5),
                                       ev._replace(causal_margin=0.5))
    assert not leaky[0].ok, (
        "a document that honestly reports a non-zero causality margin still fails item 0: "
        "the feature leaks, and reporting the leak accurately does not make it deployable"
    )
    wont_fit = lesson.check_specification(
        spec, ev._replace(deployment=ev.deployment._replace(fits=False)))
    assert not wont_fit[6].ok, (
        "a programme that does not fit the gateway fails item 6 even when the document's "
        "byte count is right — the arena is a fact about the plant, not a preference"
    )
    assert wont_fit[4].ok and wont_fit[5].ok, (
        "and it fails only that item; the threshold and the give-ups are still correct"
    )
    inf_ev = ev._replace(trigger=ev.trigger._replace(price_move_fraction=float("inf")))
    assert lesson.check_specification(spec._replace(price_move_fraction=float("inf")),
                                      inf_ev)[7].ok, (
        "when no price move in the grid shifts the threshold both sides are inf and item 7 "
        "passes; abs(inf - inf) is nan, so a plain tolerance test calls two identical "
        "answers different"
    )
    assert not lesson.check_specification(spec, inf_ev)[7].ok, (
        "and a finite claim against an infinite measurement is still a mismatch"
    )
    # A document and an evidence bundle that AGREE on a lead time of zero. Every other
    # lead-time case breaks the agreement, so this is the only one that reaches the second
    # half of item 2's rule, and a lead time of zero is the one number that makes the alarm
    # raised as the pump stops count as a catch.
    zero_lead = lesson.check_specification(spec._replace(lead_hours=0),
                                           ev._replace(lead_hours=0))
    assert not zero_lead[2].ok, (
        "the document and the evidence agree on a lead time of ZERO and item 2 still has to "
        "fail: with no lead requirement the alarm raised as the pump stops is scored as a "
        "catch, so the counts behind the threshold mean nothing. Item 2 checks that the two "
        "agree AND that the number is greater than zero"
    )
    assert not lesson.specification_passes(zero_lead), (
        "and a submission whose alarm rule carries no lead time is rejected"
    )
    assert [i for i, r in enumerate(zero_lead) if not r.ok] == [2], (
        f"and it fails only item 2; the failing items were "
        f"{[CHECKLIST_ITEMS[i] for i, r in enumerate(zero_lead) if not r.ok]}"
    )


def test_checklist_validates():
    spec, ev = _pair()
    for args in ((ev, ev), (spec, spec), ("spec", ev), (spec, None)):
        try:
            lesson.check_specification(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"check_specification{tuple(type(a).__name__ for a in args)} must raise "
                "ValueError")
    for tol in (-1e-9, -1.0, np.nan, np.inf):
        try:
            lesson.check_specification(spec, ev, tolerance=tol)
        except ValueError:
            pass
        else:
            raise AssertionError(f"tolerance={tol} must raise ValueError")


# --- the programme, measured on the fleet ------------------------------------------------

def test_the_specification_passes_its_own_checklist():
    bundle = lesson.evidence()
    spec = lesson.my_specification(bundle)
    assert isinstance(spec, Specification), (
        f"my_specification returns a Specification, got {type(spec).__name__}"
    )
    results = lesson.check_specification(spec, bundle)
    assert lesson.specification_passes(results), (
        "your own specification must pass your own checklist before you submit it. Failing "
        f"items: {[(r.item, r.detail) for r in results if not r.ok]}"
    )
    assert spec.threshold == bundle.point.threshold, (
        "every number in the document comes from the evidence bundle, not from the prose"
    )
    stripped = spec._replace(gives_up=(), prices=lesson.KICKOFF,
                             price_provenance=tuple(GUESSED for _ in OUTCOMES))
    failed = [r.item for r in lesson.check_specification(stripped, bundle) if not r.ok]
    assert CHECKLIST_ITEMS[3] in failed and CHECKLIST_ITEMS[5] in failed, (
        f"your own specification with the kick-off prices and no give-ups must fail both "
        f"pass-condition items; it failed {failed}"
    )


def test_the_programme_gives_up_failures_and_names_them():
    bundle = lesson.evidence()
    runs = lesson.label_runs(lesson.ORDERS)
    by_unit = {r.unit: r for r in runs}
    assert bundle.given_up, (
        "on this fleet the cost-optimal threshold does not catch everything, so the "
        "give-up list cannot be empty — an empty list means the threshold, the run table or "
        "failures_given_up is wrong"
    )
    assert bundle.point.counts.fn == len(bundle.given_up), (
        f"the swept counts say {bundle.point.counts.fn} failures are missed at this "
        f"threshold and the report names {len(bundle.given_up)}. The two come from the same "
        "definition and must agree"
    )
    for g in bundle.given_up:
        assert by_unit[g.unit].kind == FAILURE, (
            f"pump {g.unit} is named as a failure given up and its run is "
            f"{by_unit[g.unit].kind!r}"
        )
        assert g.reason in (NEVER_ALARMED, TOO_LATE), (
            f"pump {g.unit} has reason {g.reason!r}, which is neither of the two"
        )
        if g.reason == TOO_LATE:
            assert 0 <= g.warning_hours < bundle.lead_hours, (
                f"pump {g.unit} is listed as late with {g.warning_hours} h of warning "
                f"against a {bundle.lead_hours} h requirement"
            )
        else:
            assert g.alarm_hour == -1 and g.warning_hours == -1, (
                f"pump {g.unit} never alarmed, so both its alarm hour and its warning are -1"
            )
    caught = lesson.alarm_counts(lesson.causal_health(lesson.RAW, lesson.DUTY,
                                                      lesson.AMBIENT),
                                runs, bundle.point.threshold)
    assert caught.tp > 0, "the chosen threshold must still catch most of the fleet's failures"


def test_the_cost_optimal_point_is_cheaper_than_the_accuracy_optimal_one():
    bundle = lesson.evidence()
    health = lesson.causal_health(lesson.RAW, lesson.DUTY, lesson.AMBIENT)
    runs = lesson.label_runs(lesson.ORDERS)
    swept = lesson.sweep_counts(health, runs, lesson.THRESHOLDS)
    prices = bundle.prices.prices
    accs = np.asarray(lesson.accuracy(swept), dtype=float)
    index = int(np.argmax(accs))
    acc_counts = Counts(*(int(np.asarray(f)[index]) for f in swept))
    acc_cost = float(lesson.expected_cost(acc_counts, prices))
    assert bundle.point.cost < acc_cost, (
        f"the cost-optimal threshold must be cheaper in currency than the accuracy-optimal "
        f"one: {bundle.point.cost:,.0f} against {acc_cost:,.0f}. If they are equal, "
        "choose_operating_point is maximising accuracy"
    )
    assert acc_counts.fn > bundle.point.counts.fn, (
        f"and the accuracy-optimal threshold gives up MORE failures: {acc_counts.fn} against "
        f"{bundle.point.counts.fn}. That gap is what module 1 opened this programme with and "
        "what the specification has to state out loud"
    )
    assert lesson.THRESHOLDS[index] != bundle.point.threshold, (
        "on this fleet the two objectives choose different thresholds; if they agree, the "
        "sweep or the prices are not being read"
    )


RUBRIC = [
    (test_causal_health_is_one_at_commissioning_condition, 4,
     "divide raw by expected_level, then divide each pump's ratio by its own commissioning "
     "median; a pump at its commissioning condition reads 1.0"),
    (test_causal_health_removes_pump_size_and_duty, 4,
     "the duty-and-ambient model removes the process dependence and the per-pump median "
     "removes pump size; both are divisions, not subtractions"),
    (test_causal_health_fits_on_the_commissioning_block_only, 5,
     "fit_duty_model and the per-pump level both see raw[:, :baseline_hours] only; a fit "
     "over the whole window reads hours the plant will not have"),
    (test_causal_health_smooths_causally_and_validates, 4,
     "causal_rolling_median, and ValueError for a baseline_hours outside 1..n_hours, a "
     "window below 1, mismatched shapes and a non-positive fitted level"),
    (test_causality_margin_is_zero_for_a_causal_feature, 4,
     "compare only the first n_hours - tail_hours columns; the perturbed columns are meant "
     "to differ"),
    (test_causality_margin_perturbs_all_three_channels, 4,
     "perturb raw, duty AND ambient — a feature that reads next week's duty point is as "
     "undeployable as one that reads next week's vibration"),
    (test_causality_margin_reads_the_head_and_validates, 4,
     "tail_hours is a COUNT of hours at the end, and it must leave at least one hour of head; "
     "0, a negative and anything >= n_hours raise ValueError"),
    (test_label_runs_one_per_unit_including_the_unmentioned, 4,
     "start every unit as Run(unit, horizon_end, CENSORED) and let the log overwrite it; "
     "return exactly n_units runs sorted by unit"),
    (test_label_runs_uses_the_raised_hour, 4,
     "the event hour is o.raised_hour; o.closed_hour is when the paperwork was closed"),
    (test_label_runs_censors_suspensions_and_ignores_call_outs, 5,
     "classify_work_order gives FAILURE, SUSPENSION or IGNORE; a SUSPENSION becomes a "
     "CENSORED run and an IGNORE ends nothing at all"),
    (test_label_runs_earliest_wins_with_failure_on_a_tie, 4,
     "sort the surviving orders by (raised_hour, FAILURE before SUSPENSION) and let the "
     "first one per unit win"),
    (test_label_runs_validates, 2,
     "n_units < 1, horizon_end < 0, and any order outside 0..n_units-1 or 0..horizon_end "
     "all raise ValueError — check the bounds themselves: unit == n_units and raised_hour "
     "== horizon_end + 1 are rejected, raised_hour == horizon_end is kept, and n_units and "
     "horizon_end are checked before the log is looked at"),
    (test_censoring_report_counts_and_units, 4,
     "count the kinds, collect the CENSORED units in sorted order, and divide by the number "
     "of runs"),
    (test_censoring_report_unobserved_hours, 5,
     "sum n_hours - 1 - end_hour over the CENSORED runs only; a run that reached the last "
     "hour contributes zero and a failure contributes nothing at all"),
    (test_censoring_report_validates, 2,
     "an empty run table, n_hours < 1, an unknown kind and an end hour outside "
     "0..n_hours-1 all raise ValueError; the last hour of an n_hours window is n_hours - 1, "
     "so end_hour == n_hours is off the end and end_hour == n_hours - 1 is not"),
    (test_price_averages_what_it_has, 4,
     "group the audit by outcome and take the arithmetic mean, in OUTCOMES order"),
    (test_price_keeps_the_guess_for_a_thin_class, 5,
     "a class below min_observations keeps the FALLBACK price and is reported as GUESSED; "
     "np.mean of an empty list is nan and a 0.0 prices a missed failure at nothing"),
    (test_price_uses_the_mean_not_the_median, 4,
     "the mean, not the median: the outcome whose tail is the reason anyone monitors is the "
     "one a median under-prices"),
    (test_price_validates, 2,
     "an unknown outcome, a negative or non-finite cost, min_observations < 1 and a bad "
     "fallback price all raise ValueError; the bound is zero, not some round negative, so "
     "a cost of -0.5 is rejected too"),
    (test_choose_minimises_cost, 4,
     "expected_cost over the whole sweep, then argmin; report the accuracy AT that index, "
     "not the best accuracy anywhere"),
    (test_choose_reads_the_prices, 4,
     "the same counts choose a different threshold under different prices — that is the "
     "whole point of taking `prices`"),
    (test_choose_breaks_ties_low_and_returns_ints, 4,
     "argmin already breaks ties towards the lowest index; the returned counts are plain "
     "ints at the chosen index"),
    (test_choose_validates, 2,
     "an empty grid and a counts length that does not match the grid both raise ValueError"),
    (test_given_up_names_reason_and_warning, 5,
     "TOO_LATE with warning_hours = end_hour - alarm_hour, or NEVER_ALARMED with both "
     "alarm_hour and warning_hours at -1"),
    (test_given_up_lead_boundary_is_inclusive, 5,
     "exactly lead_hours of warning is a CATCH: compare with >=, not >"),
    (test_given_up_ignores_censored_runs, 5,
     "walk the FAILURE runs only, and take the alarm hour from alarm_hours, which masks each "
     "unit to its own run"),
    (test_given_up_sorted_and_validates, 3,
     "sort by unit; a non-finite threshold, a negative lead time and a run naming a unit "
     "with no row in health all raise ValueError — compare the unit with `>= h.shape[0]`, "
     "so the first missing unit is caught here and not by an IndexError later"),
    (test_deployment_bytes_and_fit, 4,
     "window * sample_bytes + constant_bytes per pump, times the rows of health, and fits "
     "compares with <="),
    (test_deployment_wire_threshold_rounds_to_nearest, 5,
     "np.rint(threshold * 2**wire_frac) / 2**wire_frac — to the NEAREST representable "
     "value, in both directions"),
    (test_deployment_counts_readings_and_moved_alarms, 5,
     "round the health index AND the threshold, compare the two verdicts per reading, and "
     "compare first-crossing indices per row with 'never crosses' as its own value; "
     "disagreeing_readings counts READINGS and units_whose_alarm_moves counts PUMPS, and "
     "one pump can hold several disagreeing readings"),
    (test_deployment_validates, 2,
     "a non-2-D health, a non-finite threshold, a window below 1, a non-positive byte budget "
     "and a wire_frac outside 1..30 all raise ValueError"),
    (test_trigger_reports_the_smallest_move, 5,
     "keep the multiplier with the smallest abs(m - 1) among those that CHANGE the "
     "threshold, and the threshold it moves to there"),
    (test_trigger_reports_inf_when_nothing_moves, 4,
     "float('inf') and the unchanged threshold — 'not within the range we looked at', never "
     "'never' and never 0.0"),
    (test_trigger_reads_which_price, 4,
     "prices._replace(**{which: ...}); an implementation that always scales `unplanned` "
     "passes every other trigger case"),
    (test_trigger_validates, 2,
     "a `which` outside OUTCOMES, an empty grid, a non-positive or non-finite multiplier and "
     "a negative or non-finite drift_alert all raise ValueError"),
    (test_checklist_shape_and_order, 4,
     "one CheckResult per CHECKLIST_ITEMS entry, in that order, each with a non-empty detail"),
    (test_checklist_accepts_a_consistent_specification, 4,
     "a specification that agrees with its evidence on every line passes all eight items"),
    (test_checklist_each_item_fails_on_its_own_line, 6,
     "each item reads only its own fields, so breaking one line fails one item and leaves "
     "the other seven alone"),
    (test_checklist_refuses_a_threshold_without_its_prices, 5,
     "item 3 compares all three prices AND the provenance tuple with the evidence; this is "
     "the pass condition, and a checker that waves it through is the defect this module "
     "exists to prevent"),
    (test_checklist_refuses_a_threshold_without_its_give_ups, 5,
     "item 5 compares gives_up with the units of evidence.given_up, exactly and in order; an "
     "empty list must fail"),
    (test_checklist_refuses_a_leak_and_a_programme_that_will_not_fit, 4,
     "item 0 also requires the evidence's margin to be 0.0, item 2 also requires the lead "
     "time to be greater than zero, item 6 also requires deployment.fits, and item 7 "
     "compares infinities with same_number"),
    (test_checklist_validates, 2,
     "a spec or evidence of the wrong type, and a negative or non-finite tolerance, all "
     "raise ValueError"),
    (test_the_specification_passes_its_own_checklist, 6,
     "fill every numeric field of Specification from the evidence bundle; if it does not "
     "pass your own checklist, the checklist is telling you which line is missing"),
    (test_the_programme_gives_up_failures_and_names_them, 5,
     "the report's length must equal the swept fn at the same threshold, and every named "
     "unit must be a FAILURE run with a reason that matches its alarm hour"),
    (test_the_cost_optimal_point_is_cheaper_than_the_accuracy_optimal_one, 5,
     "if these two agree, choose_operating_point is maximising accuracy instead of "
     "minimising cost, or the audited prices are not reaching it"),
]
