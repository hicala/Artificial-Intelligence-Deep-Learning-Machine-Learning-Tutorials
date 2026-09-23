"""Autograder rubric for P03-L05. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These rules matter in this particular lesson and are deliberate:

* `rul_track` composes `fit_posterior` and `rul_samples`, so a student who has not finished
  exercises 2 and 3 loses those cases too. That is the honest shape of the dependency, and it
  is why the earlier exercises carry their own points: the rubric still pays for every piece
  that works.
* `test_rul_track_uses_only_the_past` is the leak case and carries the largest single block
  of points in the rubric. It replaces every reading AFTER the checkpoint with nonsense and
  requires the answer not to move. An implementation that fits the whole row scores full
  marks on every other case in the file and fails only here — which is module 4's post-event
  leak, one layer up, and the single most expensive bug in this module.
* `test_phm_score_penalises_late_more` is the only case that separates the two time
  constants. An implementation that uses one constant on both branches, or swaps them, passes
  every shape and validation case and inverts the entire lesson. It is graded on its own.
* `test_cone_breakdown_scales_with_true_rul` is what stops a fixed tolerance in hours
  masquerading as an alpha cone. Every other cone case uses a single true RUL of 100, where
  the two are indistinguishable.
* `test_warning_lead_takes_the_first_crossing` gives the track a dip, a rise and a second dip.
  With a monotone track, "first crossing", "last crossing" and "argmin" all agree.
* `test_best_quantile_breaks_ties_towards_caution` matters because the sweep in section 11 has
  ties in it, and a tie resolved towards the later quantile is a tie resolved towards the
  dangerous model.
* the rubric grades observable behaviour, never one particular call. `rul_samples` is checked
  for the properties the docstring promises — seeded, shaped, plug-in under a degenerate
  posterior, capped on a non-degrading draw, clipped when overdue — and not for one way of
  drawing from a bivariate Gaussian, so `rng.multivariate_normal` and a Cholesky factor both
  score full marks.
"""
import importlib.util, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

VAGUE = np.diag([1e6, 1e6])
TINY = np.diag([1e-12, 1e-18])


# --- exercise 1: log_signal -------------------------------------------------------------

def test_log_signal_baseline_and_floor():
    one = float(lesson.log_signal(lesson.HEALTHY + np.e))
    assert abs(one - 1.0) < 1e-12, (
        f"log_signal(HEALTHY + e) must be 1.0, got {one:.6f}. "
        f"{np.log(lesson.HEALTHY + np.e):.6f} means the healthy baseline was never subtracted"
    )
    two = float(lesson.log_signal(lesson.HEALTHY + np.e ** 2))
    assert abs(two - 2.0) < 1e-12, f"log_signal(HEALTHY + e**2) must be 2.0, got {two:.6f}"
    floored = lesson.log_signal(np.array([0.2, 1.0, 1.0 - 1e-9]))
    assert np.allclose(floored, np.log(lesson.LOG_FLOOR)), (
        f"every value at or below the healthy baseline floors at log(LOG_FLOOR) = "
        f"{np.log(lesson.LOG_FLOOR):.4f}; got {np.round(floored, 4)}. A nan here propagates "
        "silently into every posterior mean downstream"
    )
    assert np.isfinite(lesson.log_signal(lesson.HEALTH)).all(), (
        "the whole fleet must come back finite — real health indices dip below 1.0"
    )


def test_log_signal_shape_and_dtype():
    out = lesson.log_signal(np.array([[1.0 + np.e, 1.0 + np.e], [2.0, 3.0]]))
    assert out.shape == (2, 2), f"shape must survive, got {out.shape}"
    assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
    ints = lesson.log_signal(np.array([[4, 9]]))
    assert ints.dtype == np.float64 and np.allclose(ints, np.log([[3.0, 8.0]])), (
        "an integer health array must come back as float; np.log on an int array without an "
        "asarray(..., float) first is the usual cause"
    )


# --- exercise 2: fit_posterior ----------------------------------------------------------

def test_fit_posterior_recovers_the_line():
    t = np.arange(0.0, 60.0)
    post = lesson.fit_posterior(t, -2.0 + 0.01 * t, np.zeros(2), VAGUE, 0.2)
    assert np.allclose(post.mean, [-2.0, 0.01], atol=1e-6), (
        f"with a vague prior and noiseless data the posterior mean is the least-squares fit "
        f"[-2.0, 0.01]; got {np.round(post.mean, 6)}. Swapped values mean a design matrix of "
        "[t, 1] rather than [1, t]"
    )
    assert post.cov.shape == (2, 2), f"cov must be 2x2, got {post.cov.shape}"
    assert np.allclose(post.cov, post.cov.T, atol=1e-12), "cov must be symmetric"
    steep = lesson.fit_posterior(t, 1.0 + 0.05 * t, np.zeros(2), VAGUE, 0.2)
    assert np.allclose(steep.mean, [1.0, 0.05], atol=1e-6), (
        f"a second line must come back too: expected [1.0, 0.05], got {np.round(steep.mean, 6)}"
    )


def test_fit_posterior_with_no_data_is_the_prior():
    prior_mean, prior_cov = np.array([-2.5, 0.008]), np.diag([0.16, 9e-6])
    post = lesson.fit_posterior(np.array([]), np.array([]), prior_mean, prior_cov, 0.2)
    assert np.allclose(post.mean, prior_mean), (
        f"with no readings the posterior mean IS the prior mean {prior_mean}; got "
        f"{np.round(post.mean, 6)}. If this raised, something divides by len(t)"
    )
    assert np.allclose(post.cov, prior_cov), (
        f"and the covariance is the prior covariance; got {post.cov}"
    )


def test_fit_posterior_respects_the_prior():
    strong = np.diag([1e-8, 1e-12])
    pinned = lesson.fit_posterior(np.array([0.0, 1.0]), np.array([5.0, 5.0]),
                                  np.array([-2.5, 0.008]), strong, 0.2)
    assert abs(pinned.mean[0] + 2.5) < 1e-3 and abs(pinned.mean[1] - 0.008) < 1e-5, (
        f"two readings against a very tight prior must barely move it; got "
        f"{np.round(pinned.mean, 4)}. Returning the plain least-squares fit and ignoring "
        "prior_cov passes the vague-prior case and fails here"
    )
    loose = lesson.fit_posterior(np.arange(0.0, 40.0), -1.0 + 0.02 * np.arange(0.0, 40.0),
                                 np.array([-2.5, 0.008]), np.diag([1.0, 1e-3]), 0.2)
    assert abs(loose.mean[1] - 0.02) < 1e-3, (
        f"a loose prior must let forty clean readings win: expected a slope near 0.02, got "
        f"{loose.mean[1]:.5f}"
    )


def test_fit_posterior_tightens_and_validates():
    t = np.arange(0.0, 80.0, 8.0)
    y = -2.0 + 0.01 * t
    few = lesson.fit_posterior(t[:3], y[:3], np.zeros(2), VAGUE, 0.2)
    many = lesson.fit_posterior(t, y, np.zeros(2), VAGUE, 0.2)
    assert np.trace(many.cov) < np.trace(few.cov), (
        f"the posterior must TIGHTEN as readings arrive: trace went {np.trace(few.cov):.3e} "
        f"-> {np.trace(many.cov):.3e}. If it grew, the data precision is being subtracted "
        "from the prior precision rather than added to it"
    )
    quiet = lesson.fit_posterior(t, y, np.zeros(2), VAGUE, 0.05)
    assert np.trace(quiet.cov) < np.trace(many.cov), (
        "a smaller noise_sd means the same readings say more, so the posterior must be "
        "tighter. Yours did not move, which means noise_sd is not in the update at all"
    )
    # noise_sd is a standard DEVIATION and the update divides by its square. One reading of
    # 5.0 at t = 0, against a unit prior centred on 0 with noise_sd 0.5, has an exactly
    # computable answer: precision 1 + 1/0.25 = 5, so cov[0, 0] = 0.2 and the intercept is
    # 0.2 * (5.0/0.25) = 4.0. Dividing by noise_sd instead gives 0.333 and 3.333, and every
    # interval width in section 6 is then wrong by a factor of noise_sd.
    one = lesson.fit_posterior(np.array([0.0]), np.array([5.0]), np.zeros(2),
                               np.diag([1.0, 1.0]), 0.5)
    assert abs(float(one.mean[0]) - 4.0) < 1e-9 and abs(float(one.cov[0, 0]) - 0.2) < 1e-9, (
        f"one reading of 5.0 at t=0 against a unit prior with noise_sd 0.5: the conjugate "
        f"update gives an intercept of exactly 4.0 and cov[0, 0] of exactly 0.2; got "
        f"{float(one.mean[0]):.6f} and {float(one.cov[0, 0]):.6f}. 3.333333 and 0.333333 "
        "mean you divided by noise_sd where the update needs noise_sd ** 2 — it is a "
        "standard deviation, and the precision is built from the variance"
    )
    for bad_t, bad_y, bad_sd in ((np.arange(3.0), np.arange(4.0), 0.2),
                                 (np.arange(3.0), np.arange(3.0), 0.0),
                                 (np.arange(3.0), np.arange(3.0), -0.5)):
        try:
            lesson.fit_posterior(bad_t, bad_y, np.zeros(2), VAGUE, bad_sd)
        except ValueError:
            continue
        raise AssertionError(f"t of {bad_t.size}, y of {bad_y.size}, noise_sd {bad_sd} must "
                             "raise ValueError, not return a confident answer")


# --- exercise 3: rul_samples ------------------------------------------------------------

def test_rul_samples_plug_in_value():
    draws = lesson.rul_samples(np.array([0.0, 0.01]), TINY, 50.0, 2.0, 400, seed=1)
    assert draws.shape == (400,), f"expected 400 draws, got shape {draws.shape}"
    assert draws.dtype == np.float64, f"expected float64, got {draws.dtype}"
    assert abs(float(draws.mean()) - 150.0) < 1e-3, (
        f"crossing at (2.0 - 0.0)/0.01 = hour 200, t_now 50, so every draw is 150; got "
        f"{draws.mean():.3f}. 200 means t_now was never subtracted, -150 means it was "
        "subtracted the wrong way round"
    )
    later = lesson.rul_samples(np.array([0.0, 0.01]), TINY, 120.0, 2.0, 50, seed=1)
    assert abs(float(later.mean()) - 80.0) < 1e-3, (
        f"the same posterior seventy hours later has 80 hours left, got {later.mean():.3f}"
    )
    other_fail = lesson.rul_samples(np.array([0.0, 0.01]), TINY, 0.0, 1.0, 50, seed=1)
    assert abs(float(other_fail.mean()) - 100.0) < 1e-3, (
        f"y_fail must be read from the argument, not from the module's Y_FAIL: a failure "
        f"level of 1.0 crosses at hour 100, got {other_fail.mean():.3f}"
    )


def test_rul_samples_is_seeded():
    a = lesson.rul_samples(np.array([-1.0, 0.01]), np.diag([0.04, 4e-7]), 100.0, 2.0, 300,
                           seed=7)
    b = lesson.rul_samples(np.array([-1.0, 0.01]), np.diag([0.04, 4e-7]), 100.0, 2.0, 300,
                           seed=7)
    assert np.array_equal(a, b), (
        "same seed, same draws. Seed a fresh np.random.default_rng(seed) inside the function; "
        "np.random's global state makes this notebook irreproducible"
    )
    c = lesson.rul_samples(np.array([-1.0, 0.01]), np.diag([0.04, 4e-7]), 100.0, 2.0, 300,
                           seed=8)
    assert not np.array_equal(a, c), "a different seed must give different draws"
    assert float(np.std(a)) > 1.0, (
        f"a posterior with real width must give draws with real spread; yours have a standard "
        f"deviation of {np.std(a):.4f}, so cov is being ignored and you are returning the "
        "plug-in answer with no error bars at all"
    )


def test_rul_samples_caps_non_degrading_draws():
    flat = lesson.rul_samples(np.array([0.0, -0.01]), TINY, 0.0, 2.0, 60, seed=1)
    assert np.allclose(flat, lesson.RUL_CAP), (
        f"a draw whose slope is <= 0 never reaches the failure level and must report "
        f"RUL_CAP ({lesson.RUL_CAP:.0f}); got {flat.min():.1f} .. {flat.max():.1f}. A "
        "negative crossing time here drags the lower quantile below zero"
    )
    stalled = lesson.rul_samples(np.array([0.0, 0.0]), TINY, 0.0, 2.0, 60, seed=1)
    assert np.allclose(stalled, lesson.RUL_CAP), (
        f"a posterior centred on a slope of zero says the unit is not degrading at all, so "
        f"every draw must report RUL_CAP; got {np.round(stalled[:3], 1)}. A bare division "
        "gives inf here, which poisons np.quantile downstream"
    )
    # Every case above sits at t_now = 0.0, where RUL_CAP and RUL_CAP - t_now agree. The
    # fleet re-predicts from hour 96 to hour 1100, so the cap has to hold away from zero
    # too: it is an amount of remaining life, not a crossing time waiting to be shifted.
    late_flat = lesson.rul_samples(np.array([0.0, -0.01]), TINY, 400.0, 2.0, 60, seed=1)
    assert np.allclose(late_flat, lesson.RUL_CAP), (
        f"the same non-degrading posterior read at t_now=400 must still report RUL_CAP "
        f"({lesson.RUL_CAP:.0f}); got {late_flat.min():.1f}. "
        f"{lesson.RUL_CAP - 400.0:.0f} means t_now was subtracted from the cap as well as "
        "from the crossing time — RUL_CAP is already a remaining life, so it is capped, "
        "not shifted"
    )
    mixed = lesson.rul_samples(np.array([0.0, 0.0005]), np.diag([1e-12, 1e-6]), 0.0, 2.0,
                               2000, seed=3)
    assert np.isfinite(mixed).all(), "no draw may come back inf or nan"
    assert float(mixed.max()) <= lesson.RUL_CAP + 1e-9, (
        f"nothing may exceed RUL_CAP, got {mixed.max():.1f}"
    )
    assert float(mixed.max()) >= lesson.RUL_CAP - 1e-9, (
        "roughly a sixth of these draws have a slope at or below zero, so RUL_CAP must "
        "appear among them; it does not, so the non-degrading branch is unreachable"
    )


def test_rul_samples_clips_overdue_at_zero():
    overdue = lesson.rul_samples(np.array([0.0, 0.01]), TINY, 900.0, 2.0, 60, seed=1)
    assert np.allclose(overdue, 0.0), (
        f"the crossing time is hour 200 and t_now is 900, so every draw is overdue and clips "
        f"to 0.0; got {overdue.min():.1f}. Left negative, this puts probability mass below "
        "zero in a quantity that cannot be negative"
    )
    spread = lesson.rul_samples(np.array([-0.2, 0.01]), np.diag([0.09, 9e-7]), 150.0, 2.0,
                                3000, seed=5)
    assert float(spread.min()) >= 0.0, (
        f"no draw may be negative after clipping, got {spread.min():.2f}"
    )
    assert float(np.quantile(spread, 0.05)) < float(np.quantile(spread, 0.95)), (
        "a real posterior must give an ordered interval"
    )


# --- exercise 4: rul_track --------------------------------------------------------------

def _clean_ramp(n=420):
    return -2.0 + 0.01 * np.arange(float(n))


def test_rul_track_shape_and_ordering():
    qs = np.array([0.05, 0.5, 0.95])
    track = lesson.rul_track(_clean_ramp(), np.array([200, 260, 320]), np.zeros(2), VAGUE,
                             0.05, y_fail=2.0, quantiles=qs)
    assert track.shape == (3, 3), (
        f"one row per checkpoint, one column per quantile: expected (3, 3), got {track.shape}. "
        "A transposed result breaks every metric in section 11"
    )
    assert np.all(np.diff(track, axis=1) >= -1e-9), (
        f"each row must be non-decreasing across the quantiles; got {np.round(track, 2)}. "
        "np.quantile guarantees that, so an out-of-order row means the draws are being "
        "re-generated per quantile"
    )
    single = lesson.rul_track(_clean_ramp(), np.array([200]), np.zeros(2), VAGUE, 0.05,
                              y_fail=2.0, quantiles=np.array([0.5]))
    assert single.shape == (1, 1), f"one checkpoint, one quantile: expected (1, 1), got "\
                                   f"{single.shape}"


def test_rul_track_uses_only_the_past():
    qs = np.array([0.05, 0.5, 0.95])
    clean = _clean_ramp()
    dirty = clean.copy()
    dirty[201:] = 50.0                      # the overhauled unit, or a sensor swap
    ref = lesson.rul_track(clean, np.array([200]), np.zeros(2), VAGUE, 0.05, y_fail=2.0,
                           quantiles=qs)
    leaked = lesson.rul_track(dirty, np.array([200]), np.zeros(2), VAGUE, 0.05, y_fail=2.0,
                              quantiles=qs)
    assert np.allclose(ref, leaked), (
        f"the prediction at hour 200 changed when hours 201 onward changed: "
        f"{np.round(ref[0], 1)} -> {np.round(leaked[0], 1)}. You are fitting the whole row. "
        "Everything after the checkpoint is the future, and on this fleet it is a different "
        "machine — this is module 4's post-event leak, one layer up"
    )
    zeroed = clean.copy()
    zeroed[151:] = -9.0
    early = lesson.rul_track(clean, np.array([150]), np.zeros(2), VAGUE, 0.05, y_fail=2.0,
                             quantiles=qs)
    early_leaked = lesson.rul_track(zeroed, np.array([150]), np.zeros(2), VAGUE, 0.05,
                                    y_fail=2.0, quantiles=qs)
    assert np.allclose(early, early_leaked), (
        "same leak at an earlier checkpoint: hours after 150 must not reach the fit. Slicing "
        "with `[:t_k]` and then appending the whole row is the usual way this survives the "
        "first case and fails this one"
    )
    # sample_every == t_k is the case that separates arange(0, t_k + 1, ...) from
    # arange(0, t_k, ...): the inclusive rule reads hours 0 and 200 and fits this ramp
    # exactly, so the median lands on the true 200 hours of remaining life. The exclusive
    # rule reads hour 0 alone, leaves the slope at a vague prior, and answers nonsense.
    inclusive = lesson.rul_track(clean, np.array([200]), np.zeros(2), VAGUE, 0.05,
                                 y_fail=2.0, quantiles=qs, sample_every=200)
    assert np.isfinite(inclusive).all(), "no quantile may come back inf or nan"
    median = float(inclusive[0, 1])
    assert abs(median - 200.0) < 25.0, (
        f"with sample_every=200 the readings are hours 0 AND 200, which pin this ramp to a "
        f"crossing at hour 400 and 200 hours of remaining life; got a median of "
        f"{median:.0f}. The checkpoint hour itself is included — np.arange(0, t_k + 1, "
        "sample_every). Stopping at t_k drops the most informative reading you have"
    )


def test_rul_track_advances_with_time():
    qs = np.array([0.5])
    track = lesson.rul_track(_clean_ramp(), np.array([150, 250, 350]), np.zeros(2), VAGUE,
                             0.05, y_fail=2.0, quantiles=qs)
    assert track[0, 0] > track[1, 0] > track[2, 0], (
        f"on a steady ramp there is less life left at every later checkpoint; got "
        f"{np.round(track[:, 0], 1)}. If the numbers are flat, t_now is not being passed "
        "through to rul_samples"
    )


def test_rul_track_seeds_each_checkpoint():
    qs = np.array([0.05, 0.5, 0.95])
    args = (np.zeros(420), np.array([100, 100]), np.zeros(2), np.diag([1.0, 1e-4]), 0.3)
    track = lesson.rul_track(*args, y_fail=2.0, quantiles=qs)
    assert not np.allclose(track[0], track[1]), (
        "two checkpoints at the SAME hour with the same data must still draw different "
        "samples: pass `seed + k` so each checkpoint has its own stream. A single fixed seed "
        "makes every checkpoint's Monte-Carlo error identical and correlated"
    )
    repeat = lesson.rul_track(*args, y_fail=2.0, quantiles=qs)
    assert np.array_equal(track, repeat), "the same inputs must give the same track"


# --- exercise 5: rul_rmse ---------------------------------------------------------------

def test_rul_rmse_value():
    got = lesson.rul_rmse(np.array([90.0, 110.0]), np.array([100.0, 100.0]))
    assert abs(got - 10.0) < 1e-9, (
        f"one row 10 hours early and one 10 hours late is an RMSE of 10.0, got {got}. 0.0 "
        "means the signed errors were averaged before squaring; 100.0 means the square root "
        "is missing"
    )
    assert isinstance(got, float), f"return a Python float, got {type(got).__name__}"
    assert abs(lesson.rul_rmse(np.array([100.0]), np.array([100.0]))) < 1e-12, (
        "an exactly right prediction has an RMSE of 0"
    )
    mixed = lesson.rul_rmse(np.array([0.0, 0.0, 30.0]), np.array([0.0, 40.0, 0.0]))
    assert abs(mixed - np.sqrt((0 + 1600 + 900) / 3)) < 1e-9, (
        f"expected sqrt((0 + 1600 + 900)/3) = {np.sqrt(2500/3):.6f}, got {mixed:.6f}. "
        "Dividing by n-1 rather than n is the usual cause"
    )


def test_rul_rmse_is_symmetric_and_validates():
    early = lesson.rul_rmse(np.array([60.0]), np.array([100.0]))
    late = lesson.rul_rmse(np.array([140.0]), np.array([100.0]))
    assert abs(early - late) < 1e-9, (
        f"RMSE is symmetric by construction: 40 early scored {early} and 40 late scored "
        f"{late}. Do not fix the lesson's problem inside the lesson's villain — section 8 is "
        "where asymmetry belongs"
    )
    for bad in ((np.array([1.0, 2.0]), np.array([1.0])), (np.array([]), np.array([]))):
        try:
            lesson.rul_rmse(bad[0], bad[1])
        except ValueError:
            continue
        raise AssertionError("a length mismatch and an empty table must both raise "
                             "ValueError; an RMSE of nan wins every argmin it is fed to")


# --- exercise 6: phm_score --------------------------------------------------------------

def test_phm_score_exact_values():
    zero = lesson.phm_score(np.array([100.0]), np.array([100.0]))
    assert zero.shape == (1,), (
        f"return one score per row, not a total: expected shape (1,), got {zero.shape}"
    )
    assert abs(float(zero[0])) < 1e-12, f"an exact prediction scores 0, got {float(zero[0])}"
    late = float(lesson.phm_score(np.array([110.0]), np.array([100.0]))[0])
    assert abs(late - (np.exp(10.0 / lesson.LATE_TAU) - 1.0)) < 1e-9, (
        f"ten hours late is exp(10/LATE_TAU) - 1 = "
        f"{np.exp(10.0 / lesson.LATE_TAU) - 1.0:.6f}, got {late:.6f}"
    )
    early = float(lesson.phm_score(np.array([90.0]), np.array([100.0]))[0])
    assert abs(early - (np.exp(10.0 / lesson.EARLY_TAU) - 1.0)) < 1e-9, (
        f"ten hours early is exp(10/EARLY_TAU) - 1 = "
        f"{np.exp(10.0 / lesson.EARLY_TAU) - 1.0:.6f}, got {early:.6f}. A missing minus sign "
        "on the early branch gives a score below zero"
    )
    row = lesson.phm_score(np.array([40.0, 100.0, 260.0]), np.array([100.0] * 3))
    assert (row >= 0).all(), (
        f"every score is >= 0 because exp(x) >= 1 on both branches; got {np.round(row, 3)}. "
        "A negative value means the -1 landed outside the exponential"
    )


def test_phm_score_penalises_late_more():
    late = float(lesson.phm_score(np.array([110.0]), np.array([100.0]))[0])
    early = float(lesson.phm_score(np.array([90.0]), np.array([100.0]))[0])
    assert late > early + 1e-9, (
        f"ten hours late ({late:.4f}) must cost MORE than ten hours early ({early:.4f}). "
        "Equal means one time constant on both branches, which throws away the whole point "
        "of this metric; smaller means LATE_TAU and EARLY_TAU are swapped, which inverts "
        "every conclusion in section 11"
    )
    big_late = float(lesson.phm_score(np.array([200.0]), np.array([100.0]))[0])
    big_early = float(lesson.phm_score(np.array([0.0]), np.array([100.0]))[0])
    assert big_late / big_early > 5.0, (
        f"the gap must WIDEN with the error: a hundred hours late is {big_late:.1f} and a "
        f"hundred early {big_early:.1f}, a ratio of {big_late / big_early:.2f}. A linear "
        "asymmetry (a fixed weight on the late side) gives a constant ratio instead"
    )
    assert float(lesson.phm_score(np.array([100.0 + 1e-12]), np.array([100.0]))[0]) >= 0.0, (
        "d exactly at or just above zero belongs on the late branch: the docstring says "
        "`d >= 0`"
    )


def test_phm_score_shape_and_validation():
    many = lesson.phm_score(np.linspace(0.0, 200.0, 17), np.full(17, 100.0))
    assert many.shape == (17,), f"one score per row: expected (17,), got {many.shape}"
    assert np.isfinite(many).all(), "no score may be inf or nan on ordinary inputs"
    assert float(np.mean(many)) == lesson.mean_phm_score(np.linspace(0.0, 200.0, 17),
                                                         np.full(17, 100.0)), (
        "mean_phm_score is given and simply averages your per-row scores; if these disagree "
        "you are returning an already-aggregated number from phm_score"
    )
    try:
        lesson.phm_score(np.array([1.0, 2.0]), np.array([1.0]))
    except ValueError:
        return
    raise AssertionError("a length mismatch must raise ValueError rather than broadcast one "
                         "truth across every prediction")


# --- exercise 7: cone_breakdown ---------------------------------------------------------

def test_cone_breakdown_bounds_are_inclusive():
    on_bound = lesson.cone_breakdown(np.array([80.0, 120.0]), np.array([100.0, 100.0]), 0.2)
    assert on_bound == (0.0, 1.0, 0.0), (
        f"predictions exactly on the bounds are INSIDE the cone; got {on_bound}. A strict "
        "comparison at both ends drops them and quietly lowers every accuracy in section 11"
    )
    just_out = lesson.cone_breakdown(np.array([79.999, 120.001]),
                                     np.array([100.0, 100.0]), 0.2)
    assert just_out == (0.5, 0.0, 0.5), (
        f"a hair outside is outside; got {just_out}"
    )
    rates = lesson.cone_breakdown(np.array([10.0, 100.0, 300.0]), np.full(3, 100.0), 0.2)
    assert abs(sum(rates) - 1.0) < 1e-12, (
        f"the three rates must sum to 1.0, got {sum(rates)}. Counting rows twice, or "
        "returning counts rather than rates, breaks this"
    )
    assert all(isinstance(r, float) for r in rates), "return three Python floats"


def test_cone_breakdown_direction():
    early_only = lesson.cone_breakdown(np.array([10.0]), np.array([100.0]), 0.2)
    assert early_only == (1.0, 0.0, 0.0), (
        f"a prognosis of 10 hours when 100 remain understates the life left, which is EARLY; "
        f"got {early_only}. If this came back as late you have the two ends swapped, which "
        "reverses every conclusion in section 11"
    )
    late_only = lesson.cone_breakdown(np.array([300.0]), np.array([100.0]), 0.2)
    assert late_only == (0.0, 0.0, 1.0), (
        f"a prognosis of 300 hours when 100 remain is LATE; got {late_only}"
    )
    both = lesson.cone_breakdown(np.array([10.0, 100.0, 300.0, 300.0]), np.full(4, 100.0), 0.2)
    assert both == (0.25, 0.25, 0.5), (
        f"one early, one inside and two late out of four: expected (0.25, 0.25, 0.5), got "
        f"{both}"
    )


def test_cone_breakdown_scales_with_true_rul():
    # Five rows chosen so that NO fixed tolerance in hours can reproduce the answer: the
    # cone has to be 20% of 100, of 50, of 400, of 200 and of 1000 all at once.
    scaled = lesson.cone_breakdown(np.array([90.0, 90.0, 90.0, 170.0, 1100.0]),
                                   np.array([100.0, 50.0, 400.0, 200.0, 1000.0]), 0.2)
    assert np.allclose(scaled, (0.2, 0.6, 0.2)), (
        f"the cone is a FRACTION of the true RUL, not a fixed number of hours. At alpha 0.2: "
        f"90 is inside for a true RUL of 100, LATE for 50 and EARLY for 400; 170 is inside "
        f"for 200; 1100 is inside for 1000. Expected (0.2, 0.6, 0.2), got "
        f"{tuple(round(v, 3) for v in scaled)} — a fixed +/- 20 hour tolerance gives "
        "(0.4, 0.2, 0.4) here"
    )
    wider = lesson.cone_breakdown(np.array([70.0]), np.array([100.0]), 0.4)
    assert wider == (0.0, 1.0, 0.0), (
        f"alpha must be read from the argument, not from the module's ALPHA: at alpha 0.4 the "
        f"cone is 60 to 140 and 70 is inside it. Got {wider}"
    )


def test_cone_breakdown_validates():
    for bad_alpha in (0.0, 1.0, -0.2, 1.5):
        try:
            lesson.cone_breakdown(np.array([1.0]), np.array([1.0]), bad_alpha)
        except ValueError:
            continue
        raise AssertionError(f"alpha={bad_alpha} is not a fraction strictly between 0 and 1 "
                             "and must raise ValueError")
    for bad in ((np.array([1.0, 2.0]), np.array([1.0])), (np.array([]), np.array([]))):
        try:
            lesson.cone_breakdown(bad[0], bad[1], 0.2)
        except ValueError:
            continue
        raise AssertionError("a length mismatch and an empty table must both raise ValueError")


# --- exercise 8: warning_lead -----------------------------------------------------------

def test_warning_lead_takes_the_first_crossing():
    got = lesson.warning_lead(np.array([100, 124, 148]), np.array([200.0, 80.0, 40.0]),
                              200, 72.0)
    assert got == 52.0, (
        f"the first checkpoint at or below 72 is hour 148, so the warning came 52 hours "
        f"before the failure at hour 200; got {got}"
    )
    noisy = lesson.warning_lead(np.array([100, 124, 148]), np.array([60.0, 200.0, 30.0]),
                                200, 72.0)
    assert noisy == 100.0, (
        f"the job is raised at the FIRST dip below the action level, hour 100, whatever the "
        f"track does afterwards; got {noisy}. 52.0 means you took the last crossing, which "
        "reports a warning the plant never acted on"
    )
    assert isinstance(got, float), f"return a Python float, got {type(got).__name__}"


def test_warning_lead_reports_no_warning():
    none = lesson.warning_lead(np.array([100, 124]), np.array([200.0, 180.0]), 200, 72.0)
    assert none == -1.0, (
        f"this track never reaches the action level, so no warning was given: return -1.0, "
        f"got {none}. 0.0 makes 'no warning at all' indistinguishable from 'a warning with "
        "no notice', and they are different failures"
    )
    late_only = lesson.warning_lead(np.array([100, 124]), np.array([200.0, 10.0]), 130, 72.0)
    assert late_only == 6.0, (
        f"a warning six hours before the end is still a warning, just a useless one: expected "
        f"6.0, got {late_only}. Clamping a short lead to -1 hides exactly the units section "
        "11 counts"
    )


def test_warning_lead_is_inclusive_and_validates():
    on_bound = lesson.warning_lead(np.array([100, 124]), np.array([72.0, 10.0]), 200, 72.0)
    assert on_bound == 100.0, (
        f"exactly at the action level counts — the test is `<=`, not `<`; got {on_bound}"
    )
    other_level = lesson.warning_lead(np.array([100, 124]), np.array([90.0, 40.0]), 200, 95.0)
    assert other_level == 100.0, (
        f"action_rul must be read from the argument, not from the module's ACTION_RUL: at an "
        f"action level of 95 the first checkpoint already qualifies. Got {other_level}"
    )
    for bad in ((np.array([1, 2]), np.array([1.0])), (np.array([]), np.array([]))):
        try:
            lesson.warning_lead(bad[0], bad[1], 100, 72.0)
        except ValueError:
            continue
        raise AssertionError("mismatched lengths and an empty track must raise ValueError")


# --- exercise 9: best_quantile ----------------------------------------------------------

_Q3 = np.array([0.1, 0.5, 0.9])
_M3 = np.array([[70.0, 100.0, 130.0], [70.0, 100.0, 130.0]])
_T3 = np.array([100.0, 100.0])


def test_best_quantile_min_and_max():
    q, value = lesson.best_quantile(_Q3, _M3, _T3, lesson.rul_rmse, "min")
    assert (q, round(value, 9)) == (0.5, 0.0), (
        f"the middle column is exactly right, so 'min' on RMSE returns (0.5, 0.0); got "
        f"({q}, {value}). Returning the column index rather than the quantile is the usual "
        "slip, and it makes every printout in section 11 wrong by a factor of twenty"
    )
    q_max, acc = lesson.best_quantile(_Q3, _M3, _T3, lesson.cone_accuracy, "max")
    assert (q_max, acc) == (0.5, 1.0), (
        f"only the middle column is inside the cone, so 'max' on cone accuracy returns "
        f"(0.5, 1.0); got ({q_max}, {acc}). 0.1 means 'max' is being minimised"
    )


def test_best_quantile_prefers_the_right_column():
    # Column 0 is perfect on its FIRST row and terrible on the other three; column 1 is the
    # other way round. Anything that judges a column by one of its rows picks column 0.
    lopsided = np.array([[100.0, 140.0, 130.0],
                         [200.0, 100.0, 130.0],
                         [200.0, 100.0, 130.0],
                         [200.0, 100.0, 130.0]])
    truth4 = np.full(4, 100.0)
    q_col, value = lesson.best_quantile(_Q3, lopsided, truth4, lesson.rul_rmse, "min")
    assert q_col == 0.5 and abs(value - 20.0) < 1e-9, (
        f"column 1 has an RMSE of 20.0 and column 0 an RMSE of 86.6, so 'min' returns "
        f"(0.5, 20.0); got ({q_col}, {value:.2f}). 0.1 means the column is being judged by "
        "one of its rows rather than by metric(matrix[:, i], true_rul) over all of them"
    )
    q_acc, acc = lesson.best_quantile(_Q3, lopsided, truth4, lesson.cone_accuracy, "max")
    assert q_acc == 0.5 and abs(acc - 0.75) < 1e-12, (
        f"three of column 1's four rows are inside the cone against one of column 0's, so "
        f"'max' returns (0.5, 0.75); got ({q_acc}, {acc})"
    )
    shifted = np.array([[66.0, 130.0, 200.0], [66.0, 130.0, 200.0]])
    q_rmse2, _ = lesson.best_quantile(_Q3, shifted, _T3, lesson.rul_rmse, "min")
    q_score2, _ = lesson.best_quantile(_Q3, shifted, _T3, lesson.mean_phm_score, "min")
    assert q_rmse2 == 0.5 and q_score2 == 0.1, (
        f"34 hours early against 30 hours late: RMSE prefers the LATE column (0.5) because "
        f"30 < 34, and the asymmetric score prefers the EARLY one (0.1) because being late "
        f"costs more per hour. Got {q_rmse2} and {q_score2}. This is section 11 in "
        "miniature, and if it does not reproduce here the fault is in whichever of the two "
        "metrics disagrees"
    )


def test_best_quantile_breaks_ties_towards_caution():
    tied = np.array([[90.0, 110.0, 130.0]])
    q, _ = lesson.best_quantile(_Q3, tied, np.array([100.0]), lesson.rul_rmse, "min")
    assert q == 0.1, (
        f"columns 0 and 1 are both 10 hours out, so the tie goes to the earlier, more "
        f"cautious quantile 0.1; got {q}. A tie resolved towards the later quantile is a tie "
        "resolved towards the dangerous model, which is the whole argument of this lesson"
    )


def test_best_quantile_validates():
    for bad_mode in ("MIN", "lowest", "", None, 0):
        try:
            lesson.best_quantile(_Q3, _M3, _T3, lesson.rul_rmse, bad_mode)
        except ValueError:
            continue
        raise AssertionError(f"mode={bad_mode!r} must raise ValueError rather than silently "
                             "minimising a metric that should be maximised")
    try:
        lesson.best_quantile(np.array([0.1, 0.5]), _M3, _T3, lesson.rul_rmse, "min")
    except ValueError:
        pass
    else:
        raise AssertionError("a matrix with more columns than quantiles must raise ValueError")
    try:
        lesson.best_quantile(_Q3, _M3, np.array([100.0]), lesson.rul_rmse, "min")
    except ValueError:
        pass
    else:
        raise AssertionError("a matrix with more rows than truths must raise ValueError")


# --- the lesson's own claim, graded -----------------------------------------------------

def test_rmse_prefers_the_later_of_two_tied_models():
    """The argument of section 11, on a fixed table, with no fleet and no Monte Carlo.

    Two models, one 30 hours early on every row and one 34 hours late on every row. RMSE
    ranks the LATE one better. The asymmetric score and the cone's late rate both rank it
    worse. A student whose metrics are all correct reproduces this; a student who has
    'fixed' RMSE, or swapped the score's time constants, does not.
    """
    truth = np.full(40, 120.0)
    cautious = truth - 34.0
    late = truth + 30.0
    r_cautious = lesson.rul_rmse(cautious, truth)
    r_late = lesson.rul_rmse(late, truth)
    assert r_late < r_cautious, (
        f"RMSE must prefer the model that is 30 hours late ({r_late:.2f}) over the one that "
        f"is 34 hours early ({r_cautious:.2f}), because it only sees the magnitude. Yours "
        "does not, so rul_rmse is not a plain root mean square of the signed difference"
    )
    s_cautious = lesson.mean_phm_score(cautious, truth)
    s_late = lesson.mean_phm_score(late, truth)
    assert s_late > s_cautious, (
        f"the asymmetric score must rank them the other way: late {s_late:.4g} against "
        f"cautious {s_cautious:.4g}. If it agrees with RMSE, LATE_TAU and EARLY_TAU are "
        "swapped or the branch test is the wrong way round"
    )
    assert lesson.cone_breakdown(late, truth, 0.2)[2] == 1.0, (
        "30 hours late on a true RUL of 120 is outside a 20% cone, so the late rate is 1.0"
    )
    assert lesson.cone_breakdown(cautious, truth, 0.2)[2] == 0.0, (
        "and the cautious model's late rate is 0.0 — it is never late, only early"
    )


RUBRIC = [
    (test_log_signal_baseline_and_floor, 5,
     "subtract HEALTHY before the log and clamp at LOG_FLOOR; a nan here never raises"),
    (test_log_signal_shape_and_dtype, 3,
     "np.asarray(health, float) first, then numpy's own log — shape and dtype must survive"),

    (test_fit_posterior_recovers_the_line, 5,
     "design matrix [1, t]; with a vague prior the answer is the least-squares fit"),
    (test_fit_posterior_with_no_data_is_the_prior, 4,
     "no readings means the posterior IS the prior — do not divide by len(t)"),
    (test_fit_posterior_respects_the_prior, 4,
     "inv(prior_cov) belongs in the precision and prior_mean in the mean update; a plain "
     "least-squares fit ignores both"),
    (test_fit_posterior_tightens_and_validates, 3,
     "add X.T@X/noise_sd**2 to the prior precision, and raise ValueError on a length "
     "mismatch or a non-positive noise_sd"),

    (test_rul_samples_plug_in_value, 4,
     "rul = (y_fail - a)/b - t_now, read from the arguments — not from Y_FAIL"),
    (test_rul_samples_is_seeded, 3,
     "np.random.default_rng(seed) inside the function, and the draws must use cov"),
    (test_rul_samples_caps_non_degrading_draws, 4,
     "a slope <= 0 never crosses: report RUL_CAP, and note that zero counts as <= 0"),
    (test_rul_samples_clips_overdue_at_zero, 3,
     "clip into [0, RUL_CAP]; a predictive distribution cannot put mass below zero"),

    (test_rul_track_shape_and_ordering, 4,
     "return (len(checkpoints), len(quantiles)) — rows are checkpoints, columns quantiles"),
    (test_rul_track_uses_only_the_past, 6,
     "fit on np.arange(0, t_k + 1, sample_every) and nothing after t_k; the hours after the "
     "checkpoint are a different machine"),
    (test_rul_track_advances_with_time, 2,
     "pass the checkpoint hour to rul_samples as t_now, or the track never moves"),
    (test_rul_track_seeds_each_checkpoint, 2,
     "seed + k, so each checkpoint draws its own reproducible sample"),

    (test_rul_rmse_value, 4,
     "square first, then mean over n, then the square root"),
    (test_rul_rmse_is_symmetric_and_validates, 4,
     "keep RMSE symmetric; raise ValueError on a mismatch or an empty table"),

    (test_phm_score_exact_values, 5,
     "exp(-d/EARLY_TAU) - 1 for d < 0 and exp(d/LATE_TAU) - 1 for d >= 0, one value per row"),
    (test_phm_score_penalises_late_more, 5,
     "LATE_TAU is the smaller constant, so the late branch grows faster — one constant on "
     "both branches, or the two swapped, inverts this lesson"),
    (test_phm_score_shape_and_validation, 4,
     "return the per-row array, not the total, and raise ValueError on a length mismatch"),

    (test_cone_breakdown_bounds_are_inclusive, 4,
     "the bounds belong to the cone: compare with <= and >=, and return rates that sum to 1"),
    (test_cone_breakdown_direction, 4,
     "below the cone is early (it understated the life left); above it is late"),
    (test_cone_breakdown_scales_with_true_rul, 4,
     "the cone is (1 +/- alpha) * true_rul, not a fixed tolerance in hours, and alpha comes "
     "from the argument"),
    (test_cone_breakdown_validates, 2,
     "0 < alpha < 1, equal lengths, at least one row — otherwise ValueError"),

    (test_warning_lead_takes_the_first_crossing, 5,
     "the FIRST checkpoint at or below action_rul is when the job was raised"),
    (test_warning_lead_reports_no_warning, 4,
     "-1.0 when the track never reaches the action level; a real but short lead is not -1"),
    (test_warning_lead_is_inclusive_and_validates, 3,
     "the test is <=, action_rul comes from the argument, and empty or mismatched inputs "
     "raise ValueError"),

    (test_best_quantile_min_and_max, 5,
     "return the quantile and its value, and honour mode: argmin for 'min', argmax for 'max'"),
    (test_best_quantile_prefers_the_right_column, 4,
     "evaluate metric(matrix[:, i], true_rul) column by column — one column is one model"),
    (test_best_quantile_breaks_ties_towards_caution, 4,
     "np.argmin and np.argmax already return the first winner; keep it that way"),
    (test_best_quantile_validates, 3,
     "mode must be exactly 'min' or 'max', and the matrix must have a column per quantile "
     "and a row per truth"),

    (test_rmse_prefers_the_later_of_two_tied_models, 6,
     "this is section 11 on a fixed table: if it does not reproduce, the fault is in "
     "rul_rmse, phm_score or cone_breakdown, not in the argument"),
]
