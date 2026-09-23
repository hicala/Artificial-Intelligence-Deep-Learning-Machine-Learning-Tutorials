"""Autograder rubric for P03-L04. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These rules matter in this particular lesson and are deliberate:

* `build_runs` composes `classify_work_order` and `event_hour`, so a student who has not
  finished exercises 2 and 3 loses those cases too. That is the honest shape of the
  dependency and it is why the earlier exercises carry their own points: the rubric still
  pays for every piece that works.
* `test_horizon_labels_censoring_is_unknown` is the only case that separates a labeller
  emitting 0 from one emitting -1 on a censored unit's final window. Every other case a
  zero-filling implementation passes, and that implementation is the single most expensive
  bug this module exists to prevent, so it is graded on its own.
* `test_cost_optimal_stops_at_the_run_end` puts a value no healthy machine ever shows at an
  hour AFTER the run ended. An implementation that searches the whole row scores full marks
  on every other case and fails only here — which is the leak, isolated.
* `test_rank_auc_rejects_unknown_labels` is what stops an AUC that silently treats -1 as a
  negative. Nothing else in the rubric hands it an unknown row.
* `test_parse_log_hour_day_first` includes a day greater than 12, so an implementation that
  reads the slash format month-first raises rather than returning a plausible wrong date. It
  is graded separately from the format coverage because the two failures are different bugs.
* `test_build_runs_takes_the_earliest_terminating_order` puts the tied pair in both row
  orders. Independent review, 2026-09-16: with one order only, `hour <= current[0]` — keep
  whichever tied order the export listed last — scored 113/113, because the failure happened
  to be second. The reversed pair is what grades the tie RULE rather than the file's order.
* `test_split_by_unit_is_stratified` includes two strata whose size does not divide by the
  test fraction (6 and 3 failures at 0.25). Same review: every earlier case divided exactly,
  so `int(n * test_fraction)` truncating instead of rounding scored 113/113; on 3 failures it
  returns a test fold with none in it.
* the rubric grades observable behaviour, never one particular call. `split_by_unit` is
  checked for the properties the docstring promises — a partition, stratified, seeded — and
  not for one particular permutation, so a differently-shaped correct answer still scores.
"""
import importlib.util, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

WO = lesson.WorkOrder
Run = lesson.Run
FAILURE, SUSPENSION, CENSORED, IGNORE = (lesson.FAILURE, lesson.SUSPENSION,
                                         lesson.CENSORED, lesson.IGNORE)


def _wo(unit=0, job_type="BM", text="bearing seized", raised="2026-03-02 00:00",
        closed="2026-03-09 00:00", trip_hour=30):
    return WO(unit, job_type, text, raised, closed, trip_hour)


# --- exercise 1: parse_log_hour ---------------------------------------------------------

def test_parse_log_hour_formats():
    assert lesson.parse_log_hour("2026-03-01 00:00") == 0, "the epoch itself is hour 0"
    assert lesson.parse_log_hour("2026-03-02 12:00") == 36, (
        f"ISO format: expected 36, got {lesson.parse_log_hour('2026-03-02 12:00')}"
    )
    assert lesson.parse_log_hour("02 Mar 2026 12:00") == 36, (
        "the '%d %b %Y %H:%M' format must parse too — three formats in one column is the "
        "whole exercise"
    )
    assert lesson.parse_log_hour("2026-04-01 00:00") == 31 * 24, (
        "March has 31 days; use datetime arithmetic rather than counting days yourself"
    )
    assert isinstance(lesson.parse_log_hour("2026-03-02 12:00"), int), (
        "return a whole number of hours as an int, not a float or a timedelta"
    )


def test_parse_log_hour_day_first():
    got = lesson.parse_log_hour("04/03/2026 06:00")
    assert got == 78, (
        f"'04/03/2026 06:00' is 4 March 06:00, hour 78; got {got}. 774 means you parsed it "
        "month-first as 3 April — this export is day-first and LOG_FORMATS says so"
    )
    # A day past 12 cannot be a month, so a month-first implementation raises here instead of
    # returning a plausible wrong answer. Both failures are bugs; only one of them is visible.
    assert lesson.parse_log_hour("13/03/2026 00:00") == 288, (
        "'13/03/2026' is the thirteenth of March, hour 288. If this raised ValueError you are "
        "trying %m/%d/%Y, which happens to work on the first twelve days of every month"
    )


def test_parse_log_hour_edges():
    assert lesson.parse_log_hour("  2026-03-02 00:00  ") == 24, (
        "the export pads its columns; strip the stamp before parsing"
    )
    assert lesson.parse_log_hour("2026-02-27 00:00") == -48, (
        "a stamp before LOG_EPOCH is a negative hour. Do not clamp and do not raise — "
        "build_runs owns that decision"
    )
    for bad in ("", "   ", "last tuesday", "2026-13-40 99:99", "03-04-2026 06:00"):
        try:
            lesson.parse_log_hour(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} matches no format in LOG_FORMATS and must raise "
                             "ValueError")


# --- exercise 2: classify_work_order ----------------------------------------------------

def test_classify_no_fault_first():
    for text in ("attended, no fault found, unit restarted",
                 "NFF - reset and returned to service",
                 "operator error, unit restarted without work",
                 "false trip - reset, no work carried out"):
        got = lesson.classify_work_order("BM", text)
        assert got == IGNORE, (
            f"classify_work_order('BM', {text!r}) came back {got!r}. Rule 1 runs before rule "
            "2: a breakdown call-out that found nothing ended nothing, and counting it as a "
            "failure invents one failure per interrupted shift"
        )
    # And the same phrases still win when a failure word is sitting right next to them.
    mixed = lesson.classify_work_order("BM", "reported as seized - attended, no fault found")
    assert mixed == IGNORE, (
        f"got {mixed!r}: the no-fault phrase outranks a failure word in the same sentence, "
        "because the fitter's finding outranks the caller's guess"
    )


def test_classify_job_type_and_keyword_fallback():
    assert lesson.classify_work_order("BM", "unit stopped in the night") == FAILURE, (
        "a BM job type with no keyword in the text is still a failure by rule 2"
    )
    assert lesson.classify_work_order("", "gearbox failed, oil black") == FAILURE, (
        "a blank job type falls back to the text, and 'failed' is in FAILURE_WORDS"
    )
    assert lesson.classify_work_order("PM", "strip and rebuild per plan") == SUSPENSION, (
        "a PM job type with no keyword is a suspension by rule 3"
    )
    for job in ("MOD", "DECOM"):
        assert lesson.classify_work_order(job, "routine work") == SUSPENSION, (
            f"{job} is in SUSPENSION_JOB_TYPES"
        )
    assert lesson.classify_work_order("", "removed for line reconfiguration") == SUSPENSION, (
        "a blank job type with a suspension phrase is a suspension"
    )
    assert lesson.classify_work_order("INSP", "routine walkdown, nothing to report") == IGNORE, (
        "an inspection with no keyword anywhere matches no rule and ends nothing"
    )
    assert lesson.classify_work_order("", "") == IGNORE, "an empty order ends nothing"


def test_classify_failure_outranks_suspension():
    got = lesson.classify_work_order("PM", "found seized during planned overhaul")
    assert got == FAILURE, (
        f"got {got!r}: rule 2 is checked before rule 3, so a defect found during planned work "
        "is still a defect. Checking the job type first inverts this"
    )
    assert lesson.classify_work_order("bm", "UNIT STOPPED") == FAILURE, (
        "normalise the job type's case; the column is not reliably upper-case"
    )
    assert lesson.classify_work_order("BM", "Bearing SEIZED") == FAILURE, (
        "match the free text case-insensitively"
    )


# --- exercise 3: event_hour -------------------------------------------------------------

def test_event_hour_policies():
    wo = _wo(raised="2026-03-02 00:00", closed="2026-03-20 00:00", trip_hour=30)
    assert lesson.event_hour(wo, "raised") == 24, "the raised policy reads order.raised"
    assert lesson.event_hour(wo, "closed") == 456, "the closed policy reads order.closed"
    assert lesson.event_hour(wo, "trip") == 30, (
        "the trip policy prefers the historian's trip_hour when it has one"
    )
    swapped = _wo(raised="2026-03-20 00:00", closed="2026-03-02 00:00", trip_hour=-1)
    assert lesson.event_hour(swapped, "closed") == 24 and lesson.event_hour(swapped, "raised") == 456, (
        "read the columns the policy names, whatever order the dates happen to be in"
    )


def test_event_hour_trip_fallback():
    wo = _wo(raised="2026-03-02 00:00", closed="2026-03-20 00:00", trip_hour=-1)
    assert lesson.event_hour(wo, "trip") == 24, (
        f"with trip_hour == -1 the historian has nothing, so 'trip' falls back to the raised "
        f"hour 24; got {lesson.event_hour(wo, 'trip')}. Falling back to the CLOSED date puts "
        "the fallback on the worst of the three policies"
    )
    zero = _wo(raised="2026-03-02 00:00", trip_hour=0)
    assert lesson.event_hour(zero, "trip") == 0, (
        f"hour 0 is a real trip hour and got {lesson.event_hour(zero, 'trip')}. Test `>= 0`, "
        "not truthiness — `if order.trip_hour:` throws away every trip in the first hour"
    )
    assert lesson.event_hour(_wo(trip_hour=-5), "trip") == 24, (
        "any negative trip_hour means 'no historian record', not a negative hour"
    )


def test_event_hour_rejects_unknown_policy():
    wo = _wo()
    for bad in ("close", "TRIP", "Raised", "", None, 0):
        try:
            lesson.event_hour(wo, bad)
        except ValueError:
            continue
        raise AssertionError(
            f"policy {bad!r} must raise ValueError. Falling through to a default is how a "
            "typo in a config file becomes a six-week labelling error")


# --- exercise 4: build_runs -------------------------------------------------------------

def test_build_runs_censors_unmentioned_units():
    log = [_wo(unit=0, trip_hour=48)]
    runs = lesson.build_runs(log, 4, 500, "trip")
    assert [r.unit for r in runs] == [0, 1, 2, 3], (
        f"one Run per unit, ordered by unit id; got {[r.unit for r in runs]}"
    )
    assert runs[0] == Run(0, 48, FAILURE), f"unit 0 failed at 48; got {runs[0]}"
    for u in (1, 2, 3):
        assert runs[u] == Run(u, 500, CENSORED), (
            f"unit {u} has no work order, so it ran to horizon_end and is censored there; got "
            f"{runs[u]}. Dropping unmentioned units throws away the entire healthy population "
            "and with it the denominator of every false-alarm rate you will ever quote"
        )
    assert lesson.build_runs([], 3, 200, "trip") == [Run(u, 200, CENSORED) for u in range(3)], (
        "an empty log means every unit is censored at horizon_end, not that there are no runs"
    )


def test_build_runs_takes_the_earliest_terminating_order():
    unsorted = [_wo(unit=0, text="gearbox failed", trip_hour=456),
                _wo(unit=0, text="coupling broken", trip_hour=72)]
    got = lesson.build_runs(unsorted, 1, 900, "trip")[0]
    assert got.end_hour == 72, (
        f"two terminating orders on one unit: the run ends at the EARLIEST, 72, got "
        f"{got.end_hour}. The export is not sorted, so orders[0] is not the first event"
    )
    nff = [_wo(unit=0, text="attended, no fault found, unit restarted", trip_hour=-1,
               raised="2026-03-02 00:00"),
           _wo(unit=0, text="bearing seized", trip_hour=216)]
    assert lesson.build_runs(nff, 1, 900, "trip")[0] == Run(0, 216, FAILURE), (
        "a no-fault-found order is earlier and is coded BM, and it must not end the run"
    )
    # The tie is put in BOTH orders on purpose. `hour <= current[0]` — replace whenever the
    # next order is no later — happens to answer FAILURE when the failure is listed second,
    # and SUSPENSION when it is listed first. One order alone grades the export's row order
    # rather than the tie rule.
    tie = [_wo(unit=0, job_type="PM", text="planned overhaul", trip_hour=100),
           _wo(unit=0, job_type="BM", text="bearing seized", trip_hour=100)]
    assert lesson.build_runs(tie, 1, 900, "trip")[0].kind == FAILURE, (
        "on a tie in event hour the failure wins — the docstring says so, and the alternative "
        "silently converts a failure into censored data"
    )
    tie_reversed = [tie[1], tie[0]]
    got_tie = lesson.build_runs(tie_reversed, 1, 900, "trip")[0].kind
    assert got_tie == FAILURE, (
        f"the same tie with the failure listed FIRST came back {got_tie!r}. The tie rule is "
        "about the kinds, not about which row the export happened to put first: `hour <= "
        "current[0]` keeps whichever order is last in the file, which is FAILURE half the "
        "time by luck"
    )
    mixed = [_wo(unit=0, job_type="PM", text="planned overhaul", trip_hour=60),
             _wo(unit=0, job_type="BM", text="bearing seized", trip_hour=300)]
    first = lesson.build_runs(mixed, 1, 900, "trip")[0]
    assert first == Run(0, 60, SUSPENSION), (
        f"the suspension at 60 comes first, so the run is censored there and the later "
        f"failure was never observed; got {first}. Preferring failures over suspensions "
        "regardless of time is a different bug from the tie rule"
    )


def test_build_runs_clamps_and_validates():
    late = [_wo(unit=0, closed="2026-06-01 00:00", trip_hour=-1, raised="2026-05-01 00:00")]
    assert lesson.build_runs(late, 1, 300, "closed")[0].end_hour == 300, (
        "a close date past the end of the sensor history clamps to horizon_end"
    )
    early = [_wo(unit=0, raised="2026-02-20 00:00", trip_hour=-1)]
    assert lesson.build_runs(early, 1, 300, "trip")[0].end_hour == 0, (
        "an order raised before LOG_EPOCH gives a negative hour; clamp it to 0"
    )
    for n_units, horizon_end in ((0, 300), (-2, 300), (1, 0), (1, -5)):
        try:
            lesson.build_runs([], n_units, horizon_end, "trip")
        except ValueError:
            continue
        raise AssertionError(f"n_units={n_units}, horizon_end={horizon_end} must raise")
    try:
        lesson.build_runs([_wo(unit=9)], 3, 300, "trip")
    except ValueError:
        pass
    else:
        raise AssertionError("an order naming a unit outside range(n_units) must raise "
                             "ValueError rather than being silently dropped")


# --- exercise 5: horizon_labels ---------------------------------------------------------

def test_horizon_labels_failure_window():
    got = lesson.horizon_labels(8, 6, FAILURE, 2)
    assert got.tolist() == [0, 0, 0, 0, 1, 1, -1, -1], (
        f"expected [0, 0, 0, 0, 1, 1, -1, -1], got {np.asarray(got).tolist()}. A 1 at index 6 "
        "labels the failure hour itself, which is the event and not a prediction of it"
    )
    assert np.asarray(got).dtype == np.int8, (
        f"return an int8 array; got dtype {np.asarray(got).dtype}"
    )
    assert np.asarray(got).shape == (8,), "the result has one entry per hour"
    long_run = lesson.horizon_labels(10, 10, FAILURE, 3)
    assert long_run.tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1], (
        f"a failure at the very last hour still leaves 3 positive hours; got "
        f"{np.asarray(long_run).tolist()}"
    )


def test_horizon_labels_censoring_is_unknown():
    cens = lesson.horizon_labels(8, 6, CENSORED, 2)
    assert cens.tolist() == [0, 0, 0, 0, -1, -1, -1, -1], (
        f"expected [0, 0, 0, 0, -1, -1, -1, -1], got {np.asarray(cens).tolist()}. Zeros at "
        "indices 4 and 5 assert that this unit survived a 2-hour window nobody watched — "
        "that is the censoring bug, and on a real fleet it is most of the rows you get wrong"
    )
    susp = lesson.horizon_labels(8, 6, SUSPENSION, 2)
    assert susp.tolist() == cens.tolist(), (
        "a suspension censors exactly like an administrative cut-off; the only difference is "
        "why you stopped watching, and neither is a negative example"
    )
    assert np.asarray(lesson.horizon_labels(8, 6, SUSPENSION, 2)).max() == 0, (
        "a suspended unit has no positive rows at all — it did not fail"
    )


def test_horizon_labels_boundaries():
    edge = lesson.horizon_labels(6, 4, FAILURE, 4)
    assert edge.tolist() == [1, 1, 1, 1, -1, -1], (
        f"with horizon == end_hour every observed hour is positive; got "
        f"{np.asarray(edge).tolist()}. `end_hour - t <= horizon` is inclusive"
    )
    wide = lesson.horizon_labels(6, 4, FAILURE, 99)
    assert wide.tolist() == [1, 1, 1, 1, -1, -1], "a horizon longer than the run is not an error"
    assert lesson.horizon_labels(4, 0, FAILURE, 2).tolist() == [-1, -1, -1, -1], (
        "end_hour 0 means no observed rows at all"
    )
    assert lesson.horizon_labels(5, 5, CENSORED, 2).tolist() == [0, 0, 0, -1, -1], (
        "a unit censored at the edge of the window still contributes negatives"
    )


def test_horizon_labels_validates():
    for horizon in (0, -1):
        try:
            lesson.horizon_labels(8, 6, FAILURE, horizon)
        except ValueError:
            pass
        else:
            raise AssertionError(f"horizon={horizon} must raise ValueError")
    for kind in ("Failure", "broken", "", None):
        try:
            lesson.horizon_labels(8, 6, kind, 2)
        except ValueError:
            pass
        else:
            raise AssertionError(f"kind={kind!r} must raise ValueError")
    for end_hour in (-1, 9):
        try:
            lesson.horizon_labels(8, end_hour, FAILURE, 2)
        except ValueError:
            pass
        else:
            raise AssertionError(f"end_hour={end_hour} is outside [0, 8] and must raise")


# --- exercise 6: split_by_unit ----------------------------------------------------------

def _toy_runs(n_fail=8, n_other=12):
    return ([Run(i, 10, FAILURE) for i in range(n_fail)]
            + [Run(n_fail + i, 10, CENSORED if i % 2 else SUSPENSION) for i in range(n_other)])


def test_split_by_unit_partitions_every_unit():
    runs = _toy_runs()
    train, test = lesson.split_by_unit(runs, 0.25, 7)
    both = sorted(np.asarray(train).tolist() + np.asarray(test).tolist())
    assert both == list(range(20)), (
        f"every unit lands on exactly one side; got {both}"
    )
    assert set(np.asarray(train).tolist()).isdisjoint(np.asarray(test).tolist()), (
        "train and test must not share a unit — that is the leak this exercise prevents"
    )
    assert np.all(np.diff(np.asarray(train)) > 0) and np.all(np.diff(np.asarray(test)) > 0), (
        "return both sides sorted ascending"
    )
    assert np.asarray(train).dtype.kind == "i" and np.asarray(test).dtype.kind == "i", (
        "return integer arrays, not floats or lists of Run"
    )


def test_split_by_unit_is_stratified():
    runs = _toy_runs(n_fail=8, n_other=12)
    train, test = lesson.split_by_unit(runs, 0.25, 7)
    test_ids = set(np.asarray(test).tolist())
    n_test_fail = sum(1 for u in test_ids if u < 8)
    assert len(test_ids) == 5, f"a quarter of 20 units is 5 test units; got {len(test_ids)}"
    assert n_test_fail == 2, (
        f"a quarter of 8 failures is 2, and a quarter of 12 non-failures is 3; got "
        f"{n_test_fail} test failures. An unstratified shuffle can hand you a fold with no "
        "failures in it at all, and every metric on that fold is undefined"
    )
    halves = lesson.split_by_unit(_toy_runs(4, 4), 0.5, 3)
    assert sum(1 for u in halves[1].tolist() if u < 4) == 2, (
        "both strata are drawn at the same fraction, independently"
    )
    # Every fraction above divides exactly, so int() and round() agree on all of them. These
    # two do not divide: 6 * 0.25 is 1.5 and 3 * 0.25 is 0.75, and truncation rounds the
    # failures DOWN both times — the second one all the way to an empty stratum.
    odd = lesson.split_by_unit(_toy_runs(n_fail=6, n_other=10), 0.25, 5)
    n_odd_fail = sum(1 for u in odd[1].tolist() if u < 6)
    assert n_odd_fail == 2, (
        f"6 failures at a quarter is round(1.5) = 2 test failures; got {n_odd_fail}. A 1 means "
        "int(n * test_fraction) truncating — the docstring says round, and truncation shrinks "
        "the rarer stratum every time the product is not whole"
    )
    scarce = lesson.split_by_unit(_toy_runs(n_fail=3, n_other=17), 0.25, 5)
    n_scarce_fail = sum(1 for u in scarce[1].tolist() if u < 3)
    assert n_scarce_fail == 1, (
        f"3 failures at a quarter is round(0.75) = 1 test failure; got {n_scarce_fail}. A 0 is "
        "truncation handing you a test fold with no failures in it at all, which is the exact "
        "outcome this exercise says stratifying prevents"
    )


def test_split_by_unit_uses_the_seed():
    runs = _toy_runs()
    a = lesson.split_by_unit(runs, 0.25, 7)
    b = lesson.split_by_unit(runs, 0.25, 7)
    assert np.array_equal(a[1], b[1]), (
        "the same seed must give the same split; build the generator with "
        "np.random.default_rng(seed) inside the function"
    )
    seeds = {tuple(lesson.split_by_unit(runs, 0.25, s)[1].tolist()) for s in range(12)}
    assert len(seeds) > 1, (
        "twelve seeds gave one split: an implementation that slices the first k unit ids is "
        "not a split, it is a sort, and it puts every low-numbered unit in the same fold"
    )


def test_split_by_unit_validates():
    runs = _toy_runs()
    for bad in (0.0, 1.0, -0.2, 1.5, 2):
        try:
            lesson.split_by_unit(runs, bad, 1)
        except ValueError:
            continue
        raise AssertionError(f"test_fraction={bad} must raise ValueError")
    try:
        lesson.split_by_unit(runs + [Run(0, 10, CENSORED)], 0.25, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("a duplicate unit id must raise ValueError; splitting one unit "
                             "twice is exactly the leak this function exists to stop")


# --- exercise 7: rank_auc ---------------------------------------------------------------

def test_rank_auc_values_and_ties():
    got = lesson.rank_auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1]))
    assert np.isclose(got, 0.75), (
        f"expected 0.75 on the worked example, got {got!r}; 0.25 means the classes are the "
        "wrong way round"
    )
    assert np.isclose(lesson.rank_auc(np.array([5.0, 4.0, 1.0, 0.5]), np.array([1, 1, 0, 0])),
                      1.0), "a perfectly separated score is AUC 1.0"
    assert np.isclose(lesson.rank_auc(np.array([0.5, 1.0, 4.0, 5.0]), np.array([1, 1, 0, 0])),
                      0.0), "a perfectly inverted score is AUC 0.0, not 1.0"
    tie = lesson.rank_auc(np.array([1.0, 1.0]), np.array([0, 1]))
    assert np.isclose(tie, 0.5), (
        f"one positive tied with one negative is 0.5, got {tie!r}. A 0.0 or 1.0 means the tied "
        "values are taking their sort position instead of their shared average rank"
    )
    many = lesson.rank_auc(np.array([2.0, 2.0, 2.0, 1.0]), np.array([1, 1, 0, 0]))
    assert np.isclose(many, 0.75), (
        f"expected 0.75 with a three-way tie, got {many!r}"
    )
    all_tied = lesson.rank_auc(np.array([3.0, 3.0, 3.0, 3.0]), np.array([1, 1, 0, 0]))
    assert np.isclose(all_tied, 0.5), "a constant score cannot rank anything: AUC 0.5"
    # Order of presentation must not matter, and a bigger sample must still be exact.
    rng = np.random.default_rng(4)
    s = np.round(rng.normal(size=400), 2)
    y = (rng.random(400) < 0.3).astype(int)
    shuffle = rng.permutation(400)
    assert np.isclose(lesson.rank_auc(s, y), lesson.rank_auc(s[shuffle], y[shuffle])), (
        "the AUC is a property of the ranking, not of the row order"
    )
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]])
    assert np.isclose(lesson.rank_auc(s, y), brute), (
        f"your AUC is {lesson.rank_auc(s, y):.6f} and the brute-force probability that a "
        f"random positive outranks a random negative is {brute:.6f}; with ~100 tied pairs in "
        "this sample the gap is the tie handling"
    )


def test_rank_auc_rejects_unknown_labels():
    try:
        lesson.rank_auc(np.array([1.0, 2.0, 3.0]), np.array([1, 0, -1]))
    except ValueError:
        pass
    else:
        raise AssertionError(
            "a -1 in labels means an unknown row reached the scorer, where it would be graded "
            "as a negative — the exact censoring bug exercise 5 exists to prevent. Raise "
            "ValueError instead of scoring it")
    for scores, labels in ((np.array([1.0, 2.0]), np.array([1, 2])),
                            (np.array([1.0, 2.0]), np.array([1, 1])),
                            (np.array([1.0, 2.0]), np.array([0, 0])),
                            (np.array([1.0, 2.0, 3.0]), np.array([0, 1]))):
        try:
            lesson.rank_auc(scores, labels)
        except ValueError:
            continue
        raise AssertionError(
            f"rank_auc({scores.tolist()}, {labels.tolist()}) must raise ValueError: a label "
            "outside {0, 1}, a missing class or a length mismatch are all undefined")


# --- exercise 8: leakage_report ---------------------------------------------------------

def test_leakage_report_counts_post_event_rows():
    ref = [Run(0, 4, FAILURE), Run(1, 10, CENSORED)]
    rep = lesson.leakage_report(np.array([0, 0, 1]), np.array([3, 4, 2]), np.array([1, 0, 0]),
                                np.array([2.0, 5.0, 1.0]), ref, 1.5)
    assert rep.post_event_rows == 1, (
        f"unit 0's run ends at hour 4, so the row at hour 4 is post-event and the row at "
        f"hour 3 is not; expected 1, got {rep.post_event_rows}. The comparison is `>=`: the "
        "end hour itself is already the event"
    )
    assert isinstance(rep, lesson.LeakageReport), "return a LeakageReport"
    clean = lesson.leakage_report(np.array([0, 1]), np.array([3, 2]), np.array([1, 0]),
                                  np.array([2.0, 1.0]), ref, 1.5)
    assert clean.post_event_rows == 0 and clean.suspect is False, (
        f"nothing post-event and an AUC below the ceiling is not suspect; got {clean}"
    )
    assert np.isclose(clean.auc, lesson.rank_auc(np.array([2.0, 1.0]), np.array([1, 0]))), (
        "the report's auc field is rank_auc of the scores and labels you were handed"
    )


def test_leakage_report_ceiling_and_validation():
    ref = [Run(0, 9, FAILURE), Run(1, 9, CENSORED)]
    units, hours = np.array([0, 1]), np.array([3, 2])
    labels, scores = np.array([1, 0]), np.array([2.0, 1.0])
    assert lesson.leakage_report(units, hours, labels, scores, ref, 1.0).suspect is True, (
        "an AUC of 1.0 against a ceiling of 1.0 is suspect: the comparison is `>=`, not `>`"
    )
    assert lesson.leakage_report(units, hours, labels, scores, ref, 1.01).suspect is False, (
        "below the ceiling with no post-event rows, nothing is suspect"
    )
    post = lesson.leakage_report(np.array([0, 1]), np.array([9, 2]), labels, scores, ref, 1.01)
    assert post.suspect is True, (
        "a post-event row makes the table suspect whatever the AUC says — the two detectors "
        "are an OR, and they catch different diseases"
    )
    try:
        lesson.leakage_report(np.array([0, 5]), hours, labels, scores, ref, 0.9)
    except ValueError:
        pass
    else:
        raise AssertionError("a unit with no run in the reference table must raise ValueError, "
                             "not be skipped")
    try:
        lesson.leakage_report(np.array([0, 1, 1]), hours, labels, scores, ref, 0.9)
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned arrays must raise ValueError")


# --- exercise 9: cost_optimal_threshold -------------------------------------------------

_P = None


def _prices():
    global _P
    if _P is None:
        _P = lesson.Prices(1.0, 100.0, 5.0)
    return _P


def test_cost_optimal_counts_the_matrix():
    h = np.array([[1.0, 1.0, 3.0, 3.0], [1.0, 1.0, 1.0, 1.0], [4.0, 4.0, 4.0, 4.0],
                  [1.0, 1.0, 1.0, 1.0]])
    runs = [Run(0, 4, FAILURE), Run(1, 4, FAILURE), Run(2, 4, CENSORED), Run(3, 4, SUSPENSION)]
    got = lesson.cost_optimal_threshold(h, runs, np.array([2.0]), _prices(), 2)
    assert (got.tp, got.fn, got.fp, got.tn) == (1, 1, 1, 1), (
        f"expected tp=1 fn=1 fp=1 tn=1, got tp={got.tp} fn={got.fn} fp={got.fp} tn={got.tn}. "
        "A suspension that stays quiet is a true negative, exactly like an administratively "
        "censored unit"
    )
    assert np.isclose(got.cost, 1 * 1.0 + 1 * 100.0 + 1 * 5.0), (
        f"cost is tp*planned + fn*unplanned + fp*false_alarm = 106.0, got {got.cost}. True "
        f"negatives are free"
    )
    assert isinstance(got, lesson.OperatingPoint), "return an OperatingPoint"
    assert got.tp + got.fn + got.fp + got.tn == len(runs), (
        "every run lands in exactly one cell of the matrix"
    )


def test_cost_optimal_lead_time():
    h = np.array([[1.0, 1.0, 3.0, 3.0], [1.0, 1.0, 1.0, 1.0]])
    runs = [Run(0, 4, FAILURE), Run(1, 4, CENSORED)]
    exact = lesson.cost_optimal_threshold(h, runs, np.array([2.0]), _prices(), 2)
    assert exact.tp == 1, (
        "the alarm is at hour 2 and the run ends at 4, so exactly lead_hours of warning — the "
        "test is `end_hour - alarm >= lead_hours`, and exactly enough counts as caught"
    )
    late = lesson.cost_optimal_threshold(h, runs, np.array([2.0]), _prices(), 3)
    assert (late.tp, late.fn) == (0, 1), (
        f"with lead_hours=3 an alarm 2 hours out prevented nothing, so it is a MISSED failure, "
        f"not a hit and not a false alarm; got tp={late.tp} fn={late.fn}"
    )
    assert np.isclose(late.cost, 100.0), "a missed failure is charged the unplanned price"
    none = lesson.cost_optimal_threshold(h, runs, np.array([9.0]), _prices(), 1)
    assert (none.tp, none.fn, none.fp, none.tn) == (0, 1, 0, 1), (
        "a failing unit that never alarms is a missed failure"
    )
    zero_lead = lesson.cost_optimal_threshold(h, runs, np.array([2.0]), _prices(), 0)
    assert zero_lead.tp == 1, "lead_hours=0 is legal and means any alarm at all counts"


def test_cost_optimal_stops_at_the_run_end():
    # Hour 2 is AFTER the run ended, and it carries a value no healthy machine ever shows.
    # An implementation that searches the whole row alarms on it and reports a perfect catch.
    h = np.array([[1.0, 1.0, 9.0, 9.0], [1.0, 1.0, 1.0, 1.0]])
    runs = [Run(0, 2, FAILURE), Run(1, 4, CENSORED)]
    got = lesson.cost_optimal_threshold(h, runs, np.array([2.0]), _prices(), 1)
    assert (got.tp, got.fn) == (0, 1), (
        f"unit 0's run ends at hour 2, so hours 2 and 3 are post-event and must not be "
        f"searched; got tp={got.tp} fn={got.fn}. Slice `health[unit, :end_hour]` — the whole "
        "difference between a detector and a leak is that colon"
    )
    quiet = np.array([[1.0, 1.0, 9.0, 9.0]])
    censored = lesson.cost_optimal_threshold(quiet, [Run(0, 2, CENSORED)], np.array([2.0]),
                                             _prices(), 1)
    assert (censored.fp, censored.tn) == (0, 1), (
        "the same slice applies to a censored unit: post-event hours cannot raise a false "
        "alarm because nobody was watching them"
    )


def test_cost_optimal_ties_and_validates():
    h = np.array([[1.0, 1.0, 3.0, 3.0], [1.0, 1.0, 1.0, 1.0]])
    runs = [Run(0, 4, FAILURE), Run(1, 4, CENSORED)]
    swept = lesson.cost_optimal_threshold(h, runs, np.array([2.0, 2.5, 9.0]), _prices(), 2)
    assert np.isclose(swept.threshold, 2.0), (
        f"2.0 and 2.5 both cost 1.0 while 9.0 costs 100.0, so the lowest tied threshold wins; "
        f"got {swept.threshold}"
    )
    reversed_grid = lesson.cost_optimal_threshold(h, runs, np.array([9.0, 2.5, 2.0]),
                                                  _prices(), 2)
    assert np.isclose(reversed_grid.threshold, 2.5), (
        "'lowest tied index' means first in the grid you were given, not numerically smallest"
    )
    for thresholds, lead in ((np.array([]), 2), (np.array([2.0]), -1)):
        try:
            lesson.cost_optimal_threshold(h, runs, thresholds, _prices(), lead)
        except ValueError:
            continue
        raise AssertionError(f"thresholds={thresholds.tolist()}, lead_hours={lead} must raise")


# --- the result the module exists to show -----------------------------------------------

def test_the_policy_gap_is_real():
    """End to end, on the notebook's own plant: a date column moves the money."""
    grid = lesson.THRESHOLDS[::4]
    truth = lesson.cost_optimal_threshold(lesson.HEALTH, lesson.TRUTH_RUNS, grid,
                                          lesson.SCENARIO, lesson.LEAD_HOURS)
    closed_runs = lesson.build_runs(lesson.ORDERS, lesson.N_UNITS, lesson.N_HOURS, "closed")
    trip_runs = lesson.build_runs(lesson.ORDERS, lesson.N_UNITS, lesson.N_HOURS, "trip")
    chosen = lesson.cost_optimal_threshold(lesson.HEALTH, closed_runs, grid, lesson.SCENARIO,
                                           lesson.LEAD_HOURS)
    honest = lesson.cost_optimal_threshold(lesson.HEALTH, lesson.TRUTH_RUNS,
                                           np.array([chosen.threshold]), lesson.SCENARIO,
                                           lesson.LEAD_HOURS)
    assert chosen.threshold > truth.threshold, (
        f"the closed policy should choose a HIGHER threshold ({chosen.threshold}) than the "
        f"truth ({truth.threshold}): its alarm window runs weeks past the failure and "
        "includes the stopped machine's own readings, so even an absurd threshold looks like "
        "a catch"
    )
    assert honest.cost > 2 * truth.cost, (
        f"the closed policy's threshold costs {honest.cost:,.0f} when scored against the "
        f"truth, against the best available {truth.cost:,.0f}. Less than 2x means the run "
        "tables are not actually differing — check that build_runs uses the policy it is "
        "given and that cost_optimal_threshold stops at end_hour"
    )
    trip_chosen = lesson.cost_optimal_threshold(lesson.HEALTH, trip_runs, grid,
                                                lesson.SCENARIO, lesson.LEAD_HOURS)
    trip_honest = lesson.cost_optimal_threshold(lesson.HEALTH, lesson.TRUTH_RUNS,
                                                np.array([trip_chosen.threshold]),
                                                lesson.SCENARIO, lesson.LEAD_HOURS)
    assert trip_honest.cost < honest.cost / 2, (
        f"the trip policy's threshold should score far better than the closed policy's when "
        f"both are judged against the truth; got {trip_honest.cost:,.0f} against "
        f"{honest.cost:,.0f}"
    )
    assert truth.fn > 0, (
        "the cost-optimal point on this fleet deliberately gives up some failures; if yours "
        "catches every one of them, check that expected cost charges the false alarms"
    )


RUBRIC = [
    (test_parse_log_hour_formats, 4,
     "datetime.strptime against each entry of LOG_FORMATS in turn, then whole hours since "
     "LOG_EPOCH"),
    (test_parse_log_hour_day_first, 4,
     "the slash format is %d/%m/%Y. Month-first parses the first twelve days of every month "
     "into the wrong month and raises on the rest"),
    (test_parse_log_hour_edges, 3,
     "strip the stamp, allow a negative hour, and raise ValueError when no format matches"),
    (test_classify_no_fault_first, 5,
     "check NO_FAULT_PHRASES before anything else; a BM call-out that found nothing ended "
     "nothing"),
    (test_classify_job_type_and_keyword_fallback, 4,
     "job type first within each rule, then the free text as the fallback for the orders "
     "nobody coded"),
    (test_classify_failure_outranks_suspension, 3,
     "rule 2 before rule 3, and normalise both the job type's case and the text's"),
    (test_event_hour_policies, 4,
     "closed -> order.closed, raised -> order.raised, trip -> the historian"),
    (test_event_hour_trip_fallback, 4,
     "`order.trip_hour >= 0`, and the fallback is the RAISED date — hour 0 is a real hour and "
     "`if order.trip_hour:` throws it away"),
    (test_event_hour_rejects_unknown_policy, 2,
     "anything outside POLICIES is a ValueError, not a silent default"),
    (test_build_runs_censors_unmentioned_units, 5,
     "a unit with no terminating order is Run(unit, horizon_end, CENSORED) — it is not "
     "dropped, and it is not a failure"),
    (test_build_runs_takes_the_earliest_terminating_order, 5,
     "min over event_hour of the FAILURE and SUSPENSION orders, ties to FAILURE; the export "
     "is not sorted"),
    (test_build_runs_clamps_and_validates, 3,
     "max(0, min(hour, horizon_end)), and validate n_units, horizon_end and every unit id"),
    (test_horizon_labels_failure_window, 5,
     "1 where `0 < end_hour - t <= horizon`, 0 where the gap is larger, -1 from end_hour on"),
    (test_horizon_labels_censoring_is_unknown, 5,
     "on a censored or suspended unit the final `horizon` observed hours are -1, not 0: "
     "nobody watched what happened next"),
    (test_horizon_labels_boundaries, 3,
     "the horizon comparison is `<=`, and end_hour == 0 or == n_hours are both legal"),
    (test_horizon_labels_validates, 2,
     "a non-positive horizon, an unknown kind and an end_hour outside [0, n_hours] all raise"),
    (test_split_by_unit_partitions_every_unit, 4,
     "split unit ids, not rows; return two sorted integer arrays whose union is every unit"),
    (test_split_by_unit_is_stratified, 4,
     "draw round(n * test_fraction) from the failures and from the non-failures separately"),
    (test_split_by_unit_uses_the_seed, 3,
     "np.random.default_rng(seed).permutation, so the split is reproducible and is not just "
     "the first k unit ids"),
    (test_split_by_unit_validates, 2,
     "test_fraction strictly inside (0, 1), and a duplicate unit id is a ValueError"),
    (test_rank_auc_values_and_ties, 5,
     "(rank_sum_of_positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg), with tied scores sharing "
     "the average of the ranks their group spans"),
    (test_rank_auc_rejects_unknown_labels, 4,
     "reject anything but 0 and 1, reject a missing class, reject a length mismatch — a -1 "
     "scored as a negative is the censoring bug arriving at the metric"),
    (test_leakage_report_counts_post_event_rows, 4,
     "count rows whose hour is `>=` that unit's end_hour in the reference table"),
    (test_leakage_report_ceiling_and_validation, 4,
     "suspect = post_event_rows > 0 or auc >= ceiling, and validate the alignment and the "
     "unit ids"),
    (test_cost_optimal_counts_the_matrix, 5,
     "failures give tp/fn, everything else gives fp/tn, and the true negatives are free"),
    (test_cost_optimal_lead_time, 4,
     "`end_hour - alarm >= lead_hours` is the catch condition; a late alarm is a false "
     "NEGATIVE, because nothing was prevented"),
    (test_cost_optimal_stops_at_the_run_end, 5,
     "search `health[unit, :end_hour]`. Reading past end_hour finds the stopped machine's own "
     "readings and calls them a prediction"),
    (test_cost_optimal_ties_and_validates, 3,
     "keep the first strictly-cheaper point as you sweep, and validate thresholds and "
     "lead_hours"),
    (test_the_policy_gap_is_real, 5,
     "if the policies do not diverge, check that build_runs passes `policy` through to "
     "event_hour and that the alarm window stops at end_hour"),
]
