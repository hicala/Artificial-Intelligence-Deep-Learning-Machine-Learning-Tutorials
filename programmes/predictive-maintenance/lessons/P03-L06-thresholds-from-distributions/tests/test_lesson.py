"""Autograded rubric for P03-L06. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These rules matter in this particular lesson and are deliberate:

* `test_hazard_denominator` is the only case that separates `pmf[k] / survival[k]` from
  `pmf[k] / (survival[k] + pmf[k])`. The wrong denominator is off by one bin everywhere and
  infinite in the last one, and the hazard is what the whole module's answer is derived from,
  so it is graded on its own.
* `test_expected_cost_failure_at_t_is_a_failure` is the only case where the `<` and `<=`
  conventions differ by an amount a check can see. An implementation that calls a failure at
  exactly the booked hour a catch scores full marks on every other cost case.
* `test_realised_cost_matches_expected_cost` averages the per-outcome cost over a pmf and
  demands the expected cost back. Two functions that each use a defensible convention but not
  the SAME one pass their own cases and fail here, which is the point: that disagreement is
  invisible in any single-function test.
* `test_greedy_is_first_come_first_served` includes a matrix where processing the machines in
  row order and processing them cheapest-first give different answers. Without it, a greedy
  that sorts the machines is indistinguishable from the one the docstring specifies, and the
  whole of section 12's measured gap would move under the student.
* `test_forecast_summary_quantile_float_slack` is graded on its own because nothing else in
  the rubric hands the quantile rule a cdf whose float64 value lands BELOW an exact boundary.
  Mutation testing found `np.flatnonzero(cdf >= q)[0]` — a bare, slackless comparison, and an
  entirely reasonable-looking implementation — scoring 100% without it; over 99,515 random
  lookups it disagrees with the reference 794 times, always by one bin and always with the
  reference on the correct side, the gap never exceeding 3e-16.
* `test_slot_costs_uses_made_at` uses a forecast with `made_at` well away from 0. Every
  demonstration in the notebook has `made_at == 0`, so an implementation that passes the
  slot's absolute hour straight into `expected_cost_at` is otherwise invisible.
* `test_best_schedule_capacity_greater_than_one` needs the cap to bind on two different slots
  at once, so an implementation that counts assignments globally rather than per slot fails.
* `test_survival_values` ends on nine equal bins, where `np.cumsum` overshoots 1 by
  2.2e-16 and an unclipped `1 - cumsum(pmf)` returns a negative probability. Nothing else in
  the rubric hands survival a pmf whose cumsum lands above 1, and the clip is in the
  docstring, so without this the clip is ungraded.
* `test_forecast_summary_validates` carries TWO negative-bin pmfs. The first, [1.4, -0.4],
  has a negative variance, so an implementation with no sign check still raises — out of
  `math.sqrt`, for the wrong reason. The second, [0.6, -0.2, 0.6], sums to 1 and has variance
  1.2, so only an explicit sign check rejects it.
* `test_fleet_cost_matrix_validates` also builds a matrix whose second row merely CONTAINS an
  inf and must be ACCEPTED. The docstring raises only on a row that is entirely non-finite,
  and nothing else here distinguishes that from raising on any inf at all.
* `test_greedy_respects_capacity` ends with a machine whose every free slot is infinite. The
  docstring still promises it a slot; an implementation that only ever picks a strictly
  cheaper slot finds none and crashes, and no other case in the rubric starves a machine.
* the rubric grades observable behaviour, never one particular call. `best_schedule` is
  checked for the properties the docstring promises — feasible, minimal, lexicographic on a
  tie — and not for one search order, so a differently-shaped correct answer still scores.
"""
import importlib.util, math, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

F = lesson.RULForecast
P = lesson.Prices
CHEAP = P(planned=10.0, unplanned=100.0, life_per_hour=1.0)
TOY = F(0, 0, np.array([0, 1, 2, 3]), np.array([0.1, 0.4, 0.4, 0.1]))


# --- exercise 1: forecast_summary -------------------------------------------------------

def test_forecast_summary_moments():
    got = lesson.forecast_summary(TOY)
    assert abs(got.mean - 1.5) < 1e-9, (
        f"mean of [0.1, 0.4, 0.4, 0.1] over hours 0..3 is 1.5, got {got.mean!r}. A value of "
        "2.0 means the hours were averaged without their probabilities"
    )
    assert abs(got.sd - math.sqrt(0.65)) < 1e-9, (
        f"sd is sqrt(sum(pmf * (hours - mean)**2)) = {math.sqrt(0.65):.6f}, got {got.sd!r}. "
        "A pmf is already a weighting; dividing by n as well is the usual cause"
    )
    single = lesson.forecast_summary(F(0, 0, np.array([7]), np.array([1.0])))
    assert single.sd == 0.0 and single.mean == 7.0, (
        f"all the mass on hour 7 gives mean 7.0 and sd 0.0, got {single.mean!r}/{single.sd!r}"
    )
    wide = lesson.forecast_summary(F(0, 0, np.array([0, 10]), np.array([0.5, 0.5])))
    assert abs(wide.mean - 5.0) < 1e-9 and abs(wide.sd - 5.0) < 1e-9, (
        f"two equal bins at 0 and 10 give mean 5 and sd 5; got {wide.mean!r}/{wide.sd!r}"
    )


def test_forecast_summary_quantile_rule():
    got = lesson.forecast_summary(TOY)
    assert (got.p10, got.p50, got.p90) == (0, 1, 2), (
        f"got {(got.p10, got.p50, got.p90)}, expected (0, 1, 2). The cdf is "
        "[0.1, 0.5, 0.9, 1.0] and a quantile is the smallest hour whose cdf has REACHED q. "
        "(1, 2, 3) means the comparison is a strict `>`, which quietly widens every interval "
        "you will ever quote"
    )
    inner = lesson.forecast_summary(F(0, 0, np.array([4, 5, 6]), np.array([0.05, 0.5, 0.45])))
    assert (inner.p10, inner.p50, inner.p90) == (5, 5, 6), (
        f"cdf [0.05, 0.55, 1.0] over hours 4, 5, 6 gives (5, 5, 6); got "
        f"{(inner.p10, inner.p50, inner.p90)}. The percentiles are values from `hours`, not "
        "positions in the array"
    )
    real = lesson.forecast_summary(lesson.FLEET[0].forecast)
    assert real.p10 < real.p50 < real.p90, (
        f"percentiles out of order on a real forecast: {real.p10}, {real.p50}, {real.p90}"
    )
    assert real.p10 < real.mean < real.p90, (
        f"the mean {real.mean:.1f} must sit inside the P10-P90 interval "
        f"[{real.p10}, {real.p90}]"
    )


def test_forecast_summary_quantile_float_slack():
    """A cdf that lands on q in exact arithmetic and a hair below it in float64."""
    pmf = np.array([1, 1, 3, 1, 6], dtype=float) / 12.0
    got = lesson.forecast_summary(F(0, 0, np.arange(5), pmf))
    assert got.p50 == 3, (
        f"the median of [1, 1, 3, 1, 6]/12 is hour 3: the first four bins are exactly half "
        f"the mass. Got {got.p50}. np.cumsum puts that boundary at 0.5 - 5.6e-17, so a bare "
        "`cdf >= q` reads it as short and hands back hour 4 — which is why the docstring asks "
        "for a little slack in the comparison"
    )


def test_forecast_summary_validates():
    for bad, why in (
            (F(0, 0, np.array([0, 1]), np.array([0.5, 0.4])), "sums to 0.9"),
            (F(0, 0, np.array([0, 1]), np.array([0.5, 0.6])), "sums to 1.1"),
            (F(0, 0, np.array([0, 1]), np.array([1.4, -0.4])), "has a negative bin"),
            (F(0, 0, np.array([0, 1, 2]), np.array([0.6, -0.2, 0.6])),
             "has a negative bin, sums to 1 and still has a positive variance, so nothing "
             "else in the function trips over it"),
            (F(0, 0, np.array([0, 1, 2]), np.array([0.5, 0.5])), "is shorter than hours")):
        try:
            lesson.forecast_summary(bad)
        except ValueError:
            continue
        raise AssertionError(
            f"a pmf that {why} must raise ValueError. Renormalising or truncating it here "
            "hides a bug in whatever produced the forecast, which is the one place it can "
            "still be fixed")


# --- exercise 2: survival_and_hazard ----------------------------------------------------

def test_survival_values():
    surv, _ = lesson.survival_and_hazard(F(0, 0, np.array([0, 1, 2]),
                                           np.array([0.2, 0.3, 0.5])))
    assert np.allclose(surv, [0.8, 0.5, 0.0]), (
        f"survival came back {np.round(np.asarray(surv), 4).tolist()}, expected "
        "[0.8, 0.5, 0.0]. S[k] = P(RUL > hours[k]) = 1 - cumsum(pmf)[k]. [1.0, 0.8, 0.5] is "
        "P(RUL >= hours[k]), which is shifted one bin late"
    )
    surv2, _ = lesson.survival_and_hazard(lesson.FLEET[1].forecast)
    surv2 = np.asarray(surv2)
    assert np.all(np.diff(surv2) <= 1e-12), "survival must be non-increasing"
    assert surv2.min() >= 0.0 and surv2.max() <= 1.0, (
        f"survival must stay inside [0, 1]; got [{surv2.min():.3e}, {surv2.max():.3e}]. "
        "A tiny negative at the tail is floating-point crumbs from cumsum — clip it"
    )
    p50 = lesson.forecast_summary(lesson.FLEET[1].forecast).p50
    assert abs(surv2[p50] - 0.5) < 0.02, (
        f"survival at the median hour {p50} should be about 0.5, got {surv2[p50]:.4f}"
    )
    ninths, _ = lesson.survival_and_hazard(F(0, 0, np.arange(9), np.full(9, 1.0 / 9.0)))
    ninths = np.asarray(ninths)
    assert float(ninths.min()) >= 0.0, (
        f"nine equal bins: np.cumsum overshoots 1 by 2.2e-16, so a bare `1 - cumsum(pmf)` "
        f"ends at {float(ninths.min()):.3e}. Got {float(ninths.min())!r}. A survival "
        "probability is never negative — clip into [0, 1] instead of handing the tail's "
        "floating-point crumbs to every hazard and cost function downstream"
    )


def test_hazard_denominator():
    _, haz = lesson.survival_and_hazard(F(0, 0, np.array([0, 1, 2]),
                                          np.array([0.2, 0.3, 0.5])))
    haz = np.asarray(haz)
    assert np.all(np.isfinite(haz)), (
        f"hazard came back {haz.tolist()} — an inf or nan in the last bin means you divided "
        "by survival[k], which is 0 once every machine has failed. The population at risk at "
        "the START of hour k is survival[k] + pmf[k]"
    )
    assert np.allclose(haz, [0.2, 0.375, 1.0]), (
        f"hazard came back {np.round(haz, 4).tolist()}, expected [0.2, 0.375, 1.0]. "
        "[0.25, 0.6, inf] is pmf[k]/survival[k]; [0.2, 0.3, 0.5] is the pmf itself, which is "
        "not a conditional probability at all"
    )
    _, haz2 = lesson.survival_and_hazard(F(0, 0, np.array([0, 1]), np.array([0.5, 0.5])))
    assert np.allclose(np.asarray(haz2), [0.5, 1.0]), (
        f"two equal bins give hazard [0.5, 1.0]; got {np.round(np.asarray(haz2), 4).tolist()}"
    )


def test_hazard_zero_tail():
    _, haz = lesson.survival_and_hazard(F(0, 0, np.array([0, 1, 2]),
                                          np.array([1.0, 0.0, 0.0])))
    haz = np.asarray(haz)
    assert np.all(np.isfinite(haz)), (
        f"once the mass is exhausted the at-risk population is 0 and the hazard must be 0.0, "
        f"not nan; got {haz.tolist()}"
    )
    assert np.allclose(haz, [1.0, 0.0, 0.0]), (
        f"expected [1.0, 0.0, 0.0], got {np.round(haz, 4).tolist()}"
    )
    surv, haz3 = lesson.survival_and_hazard(lesson.FLEET[0].forecast)
    assert np.all(np.isfinite(np.asarray(haz3))), (
        "the real forecast's tail bins have no mass left; the hazard there must still be "
        "finite"
    )
    assert len(np.asarray(haz3)) == len(np.asarray(surv)) == \
        lesson.FLEET[0].forecast.pmf.size, "both curves are one value per pmf bin"


# --- exercise 3: expected_cost_at -------------------------------------------------------

def test_expected_cost_value():
    got = lesson.expected_cost_at(TOY, 1, CHEAP)
    assert abs(got - 55.6) < 1e-9, (
        f"expected 55.6, got {got!r}. Hours 0 and 1 (0.5 of the mass) pay 100; hours 2 and 3 "
        "pay 10 plus 1 per hour of life left, which is 0.4*1 + 0.1*2 = 0.6. A value of 55.0 "
        "means the wasted life is missing; 60.1 means it is being charged to the machines "
        "that already failed too"
    )
    at_zero = lesson.expected_cost_at(TOY, 0, CHEAP)
    assert abs(at_zero - 20.5) < 1e-9, (
        f"expected 20.5 at hours_ahead=0, got {at_zero!r}. Hour 0 is a real hour: 0.1 of the "
        "mass has already failed by then. A value of 19.5 means `hours < 0` instead of "
        "`hours <= 0`"
    )
    dear = P(planned=10.0, unplanned=100.0, life_per_hour=0.0)
    free_life = lesson.expected_cost_at(TOY, 1, dear)
    assert abs(free_life - 55.0) < 1e-9, (
        f"with life_per_hour=0 the cost is just the two prices: 0.5*100 + 0.5*10 = 55.0, got "
        f"{free_life!r}"
    )


def test_expected_cost_failure_at_t_is_a_failure():
    got = lesson.expected_cost_at(TOY, 2, CHEAP)
    assert abs(got - 91.1) < 1e-9, (
        f"expected 91.1 at hours_ahead=2, got {got!r}. 0.9 of the mass fails at or before "
        "hour 2 and pays 100; only the 0.1 at hour 3 is caught, paying 10 + 1. A value of "
        "51.5 or 55.6 means a machine that fails during the hour you booked is being counted "
        "as a catch — it is not, and realised_cost has to agree with whatever you decide here"
    )
    last = lesson.expected_cost_at(TOY, 3, CHEAP)
    assert abs(last - 100.0) < 1e-9, (
        f"at hours_ahead=3 every machine in this forecast has already failed, so the cost is "
        f"exactly the unplanned price; got {last!r}"
    )


def test_expected_cost_edges_and_validation():
    far = lesson.expected_cost_at(TOY, 99, CHEAP)
    assert abs(far - 100.0) < 1e-9, (
        f"past the end of the grid nothing survives and the cost is the unplanned price; got "
        f"{far!r}. A nan or a crash means the survivor branch was not written to cope with an "
        "empty selection"
    )
    try:
        lesson.expected_cost_at(TOY, -1, CHEAP)
    except ValueError:
        pass
    else:
        raise AssertionError("a negative hours_ahead must raise ValueError rather than be "
                             "clamped to 0 — you cannot book a slot in the past")
    real = lesson.FLEET[2].forecast
    now = lesson.expected_cost_at(real, 0, lesson.SCENARIO)
    never = lesson.expected_cost_at(real, lesson.GRID_HOURS, lesson.SCENARIO)
    assert abs(never - lesson.SCENARIO.unplanned) < 1e-6, (
        f"never intervening costs exactly the unplanned price, got {never:,.4f}"
    )
    mean = lesson.forecast_summary(real).mean
    assert now < never, (
        f"pulling the machine now ({now:,.0f}) must be cheaper than never pulling it "
        f"({never:,.0f}) at these prices"
    )
    assert lesson.expected_cost_at(real, int(mean / 2), lesson.SCENARIO) < now, (
        "waiting half the mean RUL must beat pulling it immediately on this forecast; if not, "
        "the wasted-life term is not being subtracted as the plan moves later"
    )


# --- exercise 4: best_intervention_time -------------------------------------------------

def test_best_intervention_finds_the_minimum():
    plan = lesson.best_intervention_time(TOY, [0, 1, 2, 3], CHEAP)
    assert (plan.hours_ahead, round(float(plan.cost), 6)) == (0, 20.5), (
        f"expected Plan(0, 20.5) on the toy forecast, got {plan}"
    )
    real = lesson.FLEET[3].forecast
    grid = np.arange(0, 601, 7)
    plan = lesson.best_intervention_time(real, grid, lesson.SCENARIO)
    sweep = np.array([lesson.expected_cost_at(real, int(t), lesson.SCENARIO) for t in grid])
    assert abs(float(plan.cost) - float(sweep.min())) < 1e-9, (
        f"your plan costs {float(plan.cost):,.4f}; the cheapest point on the same candidate "
        f"grid is {float(sweep.min()):,.4f}. A gap means the sweep is maximising, or is "
        "returning the cost of a different candidate from the one it reports"
    )
    assert int(plan.hours_ahead) == int(grid[int(np.argmin(sweep))]), (
        f"the reported hour {plan.hours_ahead} is not the cheapest candidate "
        f"{int(grid[int(np.argmin(sweep))])}"
    )
    assert 0 < int(plan.hours_ahead) < 601, (
        f"unit 3's cheapest candidate came back {plan.hours_ahead}, which is outside the "
        "grid you were given. hours_ahead must be one of the candidates, reported as an int"
    )


def test_best_intervention_tie_goes_early():
    flat = F(0, 0, np.array([0, 1, 2]), np.array([0.0, 0.0, 1.0]))
    free = P(planned=10.0, unplanned=10.0, life_per_hour=0.0)
    tie = lesson.best_intervention_time(flat, [0, 1, 2], free)
    assert int(tie.hours_ahead) == 0, (
        f"every candidate costs 10.0 here, so the tie goes to the EARLIEST; got "
        f"{tie.hours_ahead}. `<=` in the sweep keeps overwriting the best with the next equal "
        "candidate and lands on the latest, which is the dangerous side of a tie"
    )
    reversed_order = lesson.best_intervention_time(flat, [2, 1, 0], free)
    assert int(reversed_order.hours_ahead) == 2, (
        f"the rule is 'first strictly-cheaper as you sweep in the order given', so on the "
        f"candidates [2, 1, 0] the answer is 2; got {reversed_order.hours_ahead}. Sorting the "
        "candidates yourself changes the documented behaviour"
    )


def test_best_intervention_validates():
    try:
        lesson.best_intervention_time(TOY, [], CHEAP)
    except ValueError:
        pass
    else:
        raise AssertionError("an empty candidate list must raise ValueError, not return a "
                             "plan with an infinite cost or None")


# --- exercise 5: realised_cost ----------------------------------------------------------

def test_realised_cost_values():
    assert lesson.realised_cost(5, 12, CHEAP) == 17.0, (
        f"12 hours of life, pulled at hour 5: 10 + 1*7 = 17.0, got "
        f"{lesson.realised_cost(5, 12, CHEAP)!r}"
    )
    assert lesson.realised_cost(5, 5, CHEAP) == 100.0, (
        f"a machine that fails at exactly the hour you booked has failed; expected 100.0, got "
        f"{lesson.realised_cost(5, 5, CHEAP)!r}. This is the same `<=` convention "
        "expected_cost_at uses, and the two must not disagree"
    )
    assert lesson.realised_cost(5, 2, CHEAP) == 100.0, "a machine gone at hour 2 is gone"
    assert lesson.realised_cost(0, 1, CHEAP) == 11.0, (
        f"pulled at hour 0 with 1 hour of life left: 10 + 1 = 11.0, got "
        f"{lesson.realised_cost(0, 1, CHEAP)!r}"
    )
    for bad in ((-1, 5), (5, -2)):
        try:
            lesson.realised_cost(bad[0], bad[1], CHEAP)
        except ValueError:
            continue
        raise AssertionError(f"realised_cost{bad} must raise ValueError: neither an "
                             "intervention nor a remaining life can be negative")


def test_realised_cost_matches_expected_cost():
    """Average the per-outcome cost over a forecast and the expected cost must come back."""
    for forecast in (TOY, F(1, 0, np.array([0, 1, 2, 3, 4]),
                            np.array([0.3, 0.1, 0.2, 0.15, 0.25]))):
        for t in (0, 1, 2, 4, 9):
            by_hand = sum(float(p) * lesson.realised_cost(t, int(h), CHEAP)
                          for h, p in zip(forecast.hours, forecast.pmf))
            direct = lesson.expected_cost_at(forecast, t, CHEAP)
            assert abs(by_hand - float(direct)) < 1e-9, (
                f"at t={t} the pmf-weighted realised cost is {by_hand:.6f} but "
                f"expected_cost_at says {float(direct):.6f}. The two functions disagree about "
                "what happens when the machine fails at exactly t: one is using `<` and the "
                "other `<=`, and the difference is pmf[t] times the gap between the prices")


# --- exercise 6: slot_costs -------------------------------------------------------------

def test_slot_costs_uses_made_at():
    late = F(0, 10, np.array([0, 1, 2, 3]), np.array([0.1, 0.4, 0.4, 0.1]))
    got = np.asarray(lesson.slot_costs(late, [10, 11, 12], CHEAP))
    assert got.shape == (3,), f"one cost per slot; got shape {got.shape}"
    assert np.allclose(got, [20.5, 55.6, 91.1]), (
        f"slots 10, 11 and 12 are 0, 1 and 2 hours ahead of made_at=10, costing "
        f"[20.5, 55.6, 91.1]; got {np.round(got, 3).tolist()}. All three equal to the "
        "unplanned price means the slot's ABSOLUTE hour is being passed to expected_cost_at — "
        "the forecast's clock starts at made_at, not at midnight"
    )
    zero_based = np.asarray(lesson.slot_costs(TOY, [0, 1, 2], CHEAP))
    assert np.allclose(zero_based, [20.5, 55.6, 91.1]), (
        f"with made_at=0 the slot hour and the hours-ahead coincide; got "
        f"{np.round(zero_based, 3).tolist()}"
    )


def test_slot_costs_past_slots_are_infinite():
    late = F(0, 10, np.array([0, 1, 2, 3]), np.array([0.1, 0.4, 0.4, 0.1]))
    got = np.asarray(lesson.slot_costs(late, [0, 8, 9, 10], CHEAP))
    assert np.all(np.isinf(got[:3])), (
        f"slots 0, 8 and 9 are before made_at=10 and are not choices: expected inf, got "
        f"{np.round(got[:3], 3).tolist()}. Zero makes the past the cheapest option in every "
        "`min` downstream; clamping hours_ahead to 0 makes it merely the second cheapest"
    )
    assert abs(float(got[3]) - 20.5) < 1e-9, (
        f"the slot at exactly made_at IS reachable — 0 hours ahead — and costs 20.5; got "
        f"{float(got[3])!r}"
    )
    real = np.asarray(lesson.slot_costs(lesson.FLEET[0].forecast, lesson.SLOTS,
                                        lesson.SCENARIO))
    assert np.all(np.isfinite(real)) and real.size == lesson.SLOTS.size, (
        "every shortlisted slot is in the future of hour 0, so none of them is infinite"
    )


# --- exercise 7: fleet_cost_matrix ------------------------------------------------------

def test_fleet_cost_matrix_shape_and_rows():
    other = F(1, 0, np.array([0, 1, 2, 3]), np.array([0.4, 0.3, 0.2, 0.1]))
    got = np.asarray(lesson.fleet_cost_matrix([TOY, other], [0, 2], CHEAP))
    assert got.shape == (2, 2), (
        f"expected shape (2, 2) — machines down, slots across — got {got.shape}"
    )
    assert np.allclose(got[0], [20.5, 91.1]), (
        f"row 0 must be the first forecast's slot costs [20.5, 91.1]; got "
        f"{np.round(got[0], 3).tolist()}"
    )
    assert np.allclose(got[1], np.asarray(lesson.slot_costs(other, [0, 2], CHEAP))), (
        f"row 1 does not match slot_costs for the second forecast; got "
        f"{np.round(got[1], 3).tolist()}. A transposed matrix passes the shape check here by "
        "accident and schedules slots onto machines for the rest of the module"
    )
    tall = np.asarray(lesson.fleet_cost_matrix([TOY, other, TOY], [0, 1, 2], CHEAP))
    assert tall.shape == (3, 3) and np.allclose(tall[0], tall[2]), (
        f"three machines by three slots, and the repeated forecast must give an identical "
        f"row; got shape {tall.shape}"
    )


def test_fleet_cost_matrix_validates():
    unreachable = F(1, 500, np.array([0, 1]), np.array([0.5, 0.5]))
    try:
        lesson.fleet_cost_matrix([TOY, unreachable], [0, 2], CHEAP)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "a machine whose every slot is before its own made_at has no feasible plan. That "
            "must raise ValueError: dropping the row silently renumbers every machine below "
            "it, and zeroing it makes the impossible look free")
    later = F(2, 100, np.array([0, 1, 2, 3]), np.array([0.1, 0.4, 0.4, 0.1]))
    mixed = np.asarray(lesson.fleet_cost_matrix([TOY, later], [0, 100, 102], CHEAP))
    assert mixed.shape == (2, 3), f"expected shape (2, 3), got {mixed.shape}"
    assert math.isinf(float(mixed[1, 0])) and np.all(np.isfinite(mixed[1, 1:])), (
        f"row 1 was made at hour 100, so slot 0 is in its past and slots 100 and 102 are not: "
        f"expected [inf, 20.5, 91.1], got {np.round(mixed[1], 3).tolist()}"
    )
    assert np.all(np.isfinite(mixed[0])), (
        f"row 0 was made at hour 0 and can reach every slot; got "
        f"{np.round(mixed[0], 3).tolist()}"
    )
    for forecasts, slots, why in (([], [0, 2], "an empty fleet"),
                                  ([TOY], [], "an empty calendar")):
        try:
            lesson.fleet_cost_matrix(forecasts, slots, CHEAP)
        except ValueError:
            continue
        raise AssertionError(f"{why} must raise ValueError rather than return a (0, n) array "
                             "that every later `min` will choke on")


# --- exercise 8: greedy_schedule --------------------------------------------------------

def test_greedy_is_first_come_first_served():
    two = np.array([[1.0, 2.0], [1.0, 5.0]])
    got = lesson.greedy_schedule(two, capacity=1)
    assert tuple(got.assignment) == (0, 1) and abs(float(got.total) - 6.0) < 1e-9, (
        f"machine 0 goes first and takes slot 0; machine 1 is left with slot 1, for 6.0. Got "
        f"{got}. The assignment (1, 0) totalling 3.0 is the OPTIMUM, which is exercise 9 — "
        "this function is deliberately the worse one"
    )
    order = np.array([[5.0, 6.0], [1.0, 9.0]])
    got = lesson.greedy_schedule(order, capacity=1)
    assert tuple(got.assignment) == (0, 1) and abs(float(got.total) - 14.0) < 1e-9, (
        f"row order, not urgency order: machine 0 is served first even though machine 1 has "
        f"far more to lose. Expected ((0, 1), 14.0), got {got}. An answer of ((1, 0), 7.0) "
        "means the machines are being sorted before they are served, which is a different "
        "heuristic and changes what section 12 measures"
    )


def test_greedy_respects_capacity():
    two = np.array([[1.0, 2.0], [1.0, 5.0]])
    got = lesson.greedy_schedule(two, capacity=1)
    assert len(set(tuple(got.assignment))) == 2, (
        f"at capacity 1 no slot may be used twice; got {tuple(got.assignment)}. Both machines "
        "in slot 0 means the capacity counter is never decremented"
    )
    roomy = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    wide = lesson.greedy_schedule(roomy, capacity=2)
    assert tuple(wide.assignment) == (0, 0, 1) and abs(float(wide.total) - 10.0) < 1e-9, (
        f"with capacity 2 the first two machines share slot 0 and the third is pushed to slot "
        f"1; expected ((0, 0, 1), 10.0), got {wide}"
    )
    starved = lesson.greedy_schedule(np.array([[1.0, math.inf], [math.inf, math.inf]]),
                                     capacity=1)
    assert tuple(starved.assignment) == (0, 1) and math.isinf(float(starved.total)), (
        f"machine 0 takes the only finite slot, and machine 1 must still be given the one "
        f"slot left even though it costs inf — a tie between infinities goes to the lowest "
        f"free index. Expected ((0, 1), inf), got {starved}. A crash here means the search "
        "for a cheapest slot never considered that every remaining slot might be infinite"
    )
    matrix = lesson.fleet_cost_matrix([fu.forecast for fu in lesson.FLEET], lesson.SLOTS,
                                      lesson.SCENARIO)
    plan = lesson.greedy_schedule(matrix, lesson.CAPACITY)
    assert len(set(tuple(plan.assignment))) == len(lesson.FLEET), (
        f"on the lesson's own fleet at capacity 1 every machine needs its own slot; got "
        f"{tuple(plan.assignment)}"
    )
    assert abs(float(plan.total) -
               sum(float(matrix[i, plan.assignment[i]])
                   for i in range(len(lesson.FLEET)))) < 1e-6, (
        "the reported total must be the sum of the entries actually chosen")


def test_greedy_breaks_ties_low():
    ties = np.array([[3.0, 3.0, 9.0], [3.0, 9.0, 9.0]])
    got = lesson.greedy_schedule(ties, capacity=1)
    assert tuple(got.assignment) == (0, 1), (
        f"machine 0 is indifferent between slots 0 and 1 and must take the LOWEST index, "
        f"leaving machine 1 the one it can actually use; got {tuple(got.assignment)}. Taking "
        "the highest index costs machine 1 six units here and costs nothing to avoid"
    )


def test_greedy_validates():
    two = np.array([[1.0, 2.0], [1.0, 5.0]])
    for cap in (0, -1):
        try:
            lesson.greedy_schedule(two, capacity=cap)
        except ValueError:
            continue
        raise AssertionError(f"capacity={cap} must raise ValueError")
    try:
        lesson.greedy_schedule(np.ones((3, 2)), capacity=1)
    except ValueError:
        pass
    else:
        raise AssertionError("3 machines into 2 slots at capacity 1 does not fit; say so "
                             "rather than leaving a machine unplanned or crashing later")


# --- exercise 9: best_schedule ----------------------------------------------------------

def test_best_schedule_finds_the_optimum():
    two = np.array([[1.0, 2.0], [1.0, 5.0]])
    got = lesson.best_schedule(two, capacity=1)
    assert tuple(got.assignment) == (1, 0) and abs(float(got.total) - 3.0) < 1e-9, (
        f"expected ((1, 0), 3.0), got {got}. Machine 0 gives up its favourite slot because "
        "machine 1 has nowhere else cheap to go. ((0, 1), 6.0) is the greedy answer and "
        "((0, 0), 2.0) ignores the capacity limit altogether"
    )
    three = np.array([[4.0, 1.0, 8.0],
                      [9.0, 2.0, 9.0],
                      [7.0, 3.0, 1.0]])
    got = lesson.best_schedule(three, capacity=1)
    assert abs(float(got.total) - 7.0) < 1e-9 and tuple(got.assignment) == (0, 1, 2), (
        f"the cheapest feasible schedule here is ((0, 1, 2), 7.0); got {got}. Machines 0 and "
        "1 both want slot 1 and the optimum gives it to the one with the dearer fallback. "
        "((1, 0, 2), 11.0) is what first come, first served produces here"
    )
    infeasible = np.array([[math.inf, 4.0], [math.inf, 1.0]])
    forced = lesson.best_schedule(infeasible, capacity=1)
    assert math.isinf(float(forced.total)), (
        f"when only one slot can serve either machine and the capacity is 1, the best total "
        f"is infinite and the caller must be told; got {forced}"
    )


def test_best_schedule_capacity_greater_than_one():
    costs = np.array([[1.0, 9.0, 9.0],
                      [1.0, 9.0, 9.0],
                      [1.0, 2.0, 9.0],
                      [1.0, 9.0, 3.0]])
    got = lesson.best_schedule(costs, capacity=2)
    assert abs(float(got.total) - 7.0) < 1e-9, (
        f"two machines may share slot 0, and the other two go to their own second choices: "
        f"1 + 1 + 2 + 3 = 7.0. Got {float(got.total)!r}. A total of 4.0 ignores the capacity; "
        "a total of 15.0 applies it per machine instead of per slot"
    )
    assert tuple(got.assignment) == (0, 0, 1, 2), (
        f"expected (0, 0, 1, 2), got {tuple(got.assignment)}"
    )
    used = {}
    for slot in got.assignment:
        used[slot] = used.get(slot, 0) + 1
    assert max(used.values()) <= 2, f"a slot is used {max(used.values())} times at capacity 2"


def test_best_schedule_lexicographic_tie():
    flat = np.array([[1.0, 1.0], [1.0, 1.0]])
    got = lesson.best_schedule(flat, capacity=1)
    assert tuple(got.assignment) == (0, 1), (
        f"both feasible schedules cost 2.0, so return the lexicographically smallest "
        f"assignment (0, 1); got {tuple(got.assignment)}. Returning whichever the search "
        "happened to reach last makes the schedule depend on the iteration order, which is "
        "not something a planner can be asked to reproduce"
    )
    flat3 = np.ones((3, 3))
    got3 = lesson.best_schedule(flat3, capacity=1)
    assert tuple(got3.assignment) == (0, 1, 2), (
        f"nine equal costs, capacity 1: the answer is (0, 1, 2); got "
        f"{tuple(got3.assignment)}"
    )


def test_best_schedule_validates():
    two = np.array([[1.0, 2.0], [1.0, 5.0]])
    for cap in (0, -3):
        try:
            lesson.best_schedule(two, capacity=cap)
        except ValueError:
            continue
        raise AssertionError(f"capacity={cap} must raise ValueError")
    try:
        lesson.best_schedule(np.ones((5, 2)), capacity=2)
    except ValueError:
        pass
    else:
        raise AssertionError("5 machines into 2 slots at capacity 2 does not fit and must "
                             "raise ValueError")


# --- the results the module exists to show ----------------------------------------------

def test_the_plan_is_not_the_mean():
    """On the notebook's own fleet: the optimum is neither the mean nor near it."""
    worst = 1.0
    for fu in lesson.FLEET:
        f = fu.forecast
        summary = lesson.forecast_summary(f)
        plan = lesson.best_intervention_time(f, lesson.CANDIDATES, lesson.SCENARIO)
        at_mean = lesson.expected_cost_at(f, int(round(summary.mean)), lesson.SCENARIO)
        assert int(plan.hours_ahead) < summary.mean, (
            f"unit {f.unit}: the optimal intervention {plan.hours_ahead} h is not earlier "
            f"than the mean RUL {summary.mean:.0f} h. At these prices an unplanned failure "
            "costs 20x a planned one, so the plan must sit well inside the distribution"
        )
        assert at_mean > float(plan.cost), (
            f"unit {f.unit}: planning at the mean ({at_mean:,.0f}) must cost more than the "
            f"optimum ({float(plan.cost):,.0f})"
        )
        cross = lesson.hazard_crossing(f, lesson.SCENARIO)
        assert abs(cross - int(plan.hours_ahead)) <= 2, (
            f"unit {f.unit}: the cost sweep says {plan.hours_ahead} h and the hazard crossing "
            f"says {cross} h. These are two independent routes to the same number and they "
            "must agree within a bin or two; a large gap means the hazard's denominator or "
            "the cost function's `<=` convention is wrong"
        )
        worst = max(worst, at_mean / float(plan.cost))
    assert worst > 2.0, (
        f"the worst overspend from planning at the mean came out at {worst:.2f}x. On this "
        "fleet and these prices it is over 3x; a small number means the wasted-life term is "
        "dominating and the two branches of the cost function are not both live"
    )


def test_the_plan_moves_with_the_width():
    """Mean held fixed, width changed: the plan must move, and in the right direction."""
    times = []
    for sd in (30.0, 60.0, 90.0, 120.0, 150.0):
        f = lesson.make_forecast(99, 300.0, sd)
        times.append(int(lesson.best_intervention_time(f, lesson.CANDIDATES,
                                                       lesson.SCENARIO).hours_ahead))
    assert times == sorted(times, reverse=True), (
        f"widening a forecast without moving its mean must pull the intervention EARLIER "
        f"every time; got {times}. A flat list means the plan is being derived from the mean "
        "alone, which is the belief this module exists to break"
    )
    assert times[0] - times[-1] > 50, (
        f"the tightest and widest forecasts here differ by only {times[0] - times[-1]} hours. "
        "The same mean should give plans more than 50 hours apart at these prices"
    )
    centres = [int(lesson.best_intervention_time(lesson.make_forecast(98, m, 90.0),
                                                 lesson.CANDIDATES,
                                                 lesson.SCENARIO).hours_ahead)
               for m in (200.0, 300.0, 400.0)]
    assert centres == sorted(centres) and centres[0] < centres[-1], (
        f"moving the centre later must move the plan later too; got {centres}"
    )


def test_capacity_and_the_planner_both_cost_money():
    """The three prices of section 12, separated: calendar, capacity, and method."""
    forecasts = [fu.forecast for fu in lesson.FLEET]
    matrix = lesson.fleet_cost_matrix(forecasts, lesson.SLOTS, lesson.SCENARIO)
    free = sum(float(lesson.best_intervention_time(f, lesson.CANDIDATES,
                                                   lesson.SCENARIO).cost) for f in forecasts)
    uncapped = float(np.asarray(matrix).min(axis=1).sum())
    greedy = lesson.greedy_schedule(matrix, lesson.CAPACITY)
    best = lesson.best_schedule(matrix, lesson.CAPACITY)
    assert uncapped > free, (
        f"a calendar of fixed slots cannot beat a free clock: {uncapped:,.0f} against "
        f"{free:,.0f}"
    )
    assert float(best.total) > uncapped, (
        f"the capacity limit must bite on this fleet — several machines want the same slot — "
        f"so the feasible optimum {float(best.total):,.0f} must exceed the infeasible "
        f"{uncapped:,.0f}"
    )
    assert float(greedy.total) > float(best.total), (
        f"first come, first served must be measurably worse than the optimum here: "
        f"{float(greedy.total):,.0f} against {float(best.total):,.0f}. Equality means "
        "best_schedule is returning the greedy answer"
    )
    assert float(greedy.total) - float(best.total) > 1_000, (
        f"the planner's error on this fleet is only "
        f"{float(greedy.total) - float(best.total):,.0f}; it should be several thousand"
    )
    assert len(set(tuple(best.assignment))) == len(lesson.FLEET), (
        f"the optimal schedule must still be feasible: {tuple(best.assignment)}"
    )


RUBRIC = [
    (test_forecast_summary_moments, 5,
     "mean = sum(hours*pmf); sd = sqrt(sum(pmf*(hours-mean)**2)). The pmf is the weighting — "
     "do not divide by n as well"),
    (test_forecast_summary_quantile_rule, 4,
     "a quantile is the smallest hours[k] whose cumulative probability is `>= q`, and the "
     "value comes from `hours`, not from the index"),
    (test_forecast_summary_quantile_float_slack, 2,
     "compare against `q - 1e-12`, not against q: np.cumsum lands a hair below an exact "
     "boundary and a bare `>=` then steps one bin past the right answer"),
    (test_forecast_summary_validates, 3,
     "length mismatch, a negative bin and a pmf that does not sum to 1 are all ValueError; "
     "renormalising here hides the bug where it can still be fixed. Test the sign of every "
     "bin explicitly — a negative bin can still sum to 1 and still have a positive variance"),
    (test_survival_values, 4,
     "S[k] = 1 - cumsum(pmf)[k], CLIPPED into [0, 1]. That is P(RUL > hours[k]), strictly "
     "greater, so the last entry is 0 — and on nine equal bins cumsum overshoots 1, so "
     "without the clip the last entry is -2.2e-16 instead"),
    (test_hazard_denominator, 5,
     "hazard[k] = pmf[k] / (survival[k] + pmf[k]): the machines still at risk at the START of "
     "hour k include the ones that fail during it"),
    (test_hazard_zero_tail, 3,
     "where the at-risk population is 0 the hazard is 0.0, not nan — mask the division "
     "instead of dividing and repairing afterwards"),
    (test_expected_cost_value, 5,
     "unplanned on the mass at or before t; planned plus life_per_hour*(RUL - t) on the rest. "
     "The wasted life is charged only to the machines you actually caught"),
    (test_expected_cost_failure_at_t_is_a_failure, 5,
     "`hours <= hours_ahead` is a failure. A machine that goes down during the hour you were "
     "going to collect it did not get collected"),
    (test_expected_cost_edges_and_validation, 4,
     "an hours_ahead past the grid costs exactly the unplanned price; a negative one is a "
     "ValueError, not a clamp"),
    (test_best_intervention_finds_the_minimum, 5,
     "sweep the candidates in the order given, keeping the first STRICTLY cheaper one, and "
     "return the hour together with its own cost"),
    (test_best_intervention_tie_goes_early, 4,
     "`<` not `<=`, and do not sort the candidates: the documented rule is 'first strictly "
     "cheaper in the order given', which puts a tie on the early, safe side"),
    (test_best_intervention_validates, 2,
     "an empty candidate list is a ValueError, not a Plan with an infinite cost"),
    (test_realised_cost_values, 4,
     "unplanned when true_rul <= hours_ahead, otherwise planned + life_per_hour*(true_rul - "
     "hours_ahead); both arguments must be non-negative"),
    (test_realised_cost_matches_expected_cost, 5,
     "expected_cost_at is the pmf-weighted average of realised_cost. If they disagree, one of "
     "them is using `<` where the other uses `<=`"),
    (test_slot_costs_uses_made_at, 5,
     "slot s is `s - forecast.made_at` hours ahead; the forecast's clock starts when it was "
     "made, not at midnight"),
    (test_slot_costs_past_slots_are_infinite, 4,
     "a slot before made_at is np.inf. Zero makes the past the cheapest option in every `min` "
     "downstream, and clamping to 0 hours ahead makes it the second cheapest"),
    (test_fleet_cost_matrix_shape_and_rows, 4,
     "one row per machine, one column per slot, row i = slot_costs(forecasts[i], ...). A "
     "transposed matrix passes a square shape check and schedules slots onto machines"),
    (test_fleet_cost_matrix_validates, 3,
     "an empty fleet, an empty calendar and a machine with no finite slot are all ValueError; "
     "dropping the row renumbers every machine below it. A row that merely CONTAINS an inf is "
     "a normal row — that machine simply cannot use that one slot"),
    (test_greedy_is_first_come_first_served, 5,
     "walk the rows in order and give each machine the cheapest slot with room left. Sorting "
     "the machines first is a different, better heuristic and not the one specified"),
    (test_greedy_respects_capacity, 4,
     "decrement the chosen slot's remaining capacity, and report the total of the entries you "
     "actually chose. When every slot still free is infinite for this machine, it still gets "
     "one — the lowest free index — rather than nothing at all"),
    (test_greedy_breaks_ties_low, 3,
     "a tie between slots goes to the lowest index — it costs nothing here and it leaves the "
     "later machines something they can use"),
    (test_greedy_validates, 2,
     "capacity below 1, and more machines than capacity*n_slots, are both ValueError"),
    (test_best_schedule_finds_the_optimum, 6,
     "minimise the TOTAL over feasible assignments, not each row in turn. Recursion over the "
     "machines with a capacity counter, or itertools.product with a filter, both work here"),
    (test_best_schedule_capacity_greater_than_one, 5,
     "the cap is per slot, not per machine and not on the schedule as a whole: two machines "
     "may share one slot when capacity is 2"),
    (test_best_schedule_lexicographic_tie, 3,
     "among equal-cost schedules return the lexicographically smallest assignment, so the "
     "answer does not depend on the order your search happened to visit"),
    (test_best_schedule_validates, 3,
     "the same two validations as greedy_schedule, so the two functions can be swapped in a "
     "comparison without one of them crashing"),
    (test_the_plan_is_not_the_mean, 5,
     "if the optimum lands on the mean, or the hazard crossing disagrees with the sweep, the "
     "cost function's two branches are not both live"),
    (test_the_plan_moves_with_the_width, 5,
     "a wider forecast at the same mean must be pulled earlier: the balancing condition is on "
     "the hazard, and a wider distribution reaches any hazard sooner"),
    (test_capacity_and_the_planner_both_cost_money, 5,
     "if the greedy and the optimal schedules cost the same, best_schedule is optimising each "
     "row rather than the total"),
]
