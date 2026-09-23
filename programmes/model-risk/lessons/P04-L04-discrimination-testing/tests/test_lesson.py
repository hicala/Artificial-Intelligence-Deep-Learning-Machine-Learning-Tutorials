"""Autograder rubric for P04-L04. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson, and are deliberate:

* every expected value is computed HERE, by an independent route — a pure-Python pairwise
  reference (`_auc_reference`, `_ks_reference`, `_delong_var_reference`,
  `_delong_pair_var_reference`) or a closed form worked out by hand. Nothing is graded against
  the student's own code, and no expected figure is lifted from the notebook's synthetic
  sample, so a change to the generator cannot silently break grading.
* the exercises chain hard — everything downstream of `auc_by_ranks` calls it — so each
  exercise's own properties are graded by cases that lean on the previous exercises as little
  as possible, and where a chain is unavoidable the expected value still comes from the
  reference above rather than from the student's earlier answer.
* `test_bootstrap_preserves_record_pairing` is what stops the single most common bootstrap
  bug: resampling labels and scores with two separate index draws. It scores a perfectly
  separating model, where the honest interval sits at 1.0 and the broken one at 0.5.
* `test_bootstrap_stratifies_thin_class` hands a 3-event sample to 500 replicates. An
  unstratified draw produces a replicate with no events at all with probability about 4.6%
  per replicate, so over 500 replicates it is a certainty, and AUC is undefined there.
* `test_paired_identical_models` hands the same score array in twice. Every replicate
  difference is then exactly zero, so `se_paired` is zero — while `se_independent` is not.
  An implementation that draws separate resamples for the two models passes everything else
  in this rubric and fails this one case.
* `test_delong_covariance_term` checks the cross term `- 2 * S[0,1]` specifically, against a
  pure-Python variance of the DIFFERENCED structural components. Dropping it leaves a test
  that is right in every other respect and systematically underpowered.
* `test_measurability_precedence` pins the order of the rules: a difference 15 standard
  errors out is SIGNIFICANT whatever the minimum detectable difference says. A rule engine
  that checks measurability first passes every other decision case and fails this one.
* `test_coverage_inclusive_endpoints` pins the boundary: an interval whose endpoint IS the
  truth covers it. Exclusive comparisons there would fail exactly the intervals a validator
  is asked to defend.
"""
import importlib.util
import math
import os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)


# --- independent references, so nothing is graded against the student's own answer ---------

def _split(y, p):
    pos = [float(s) for s, l in zip(p, y) if int(l) == 1]
    neg = [float(s) for s, l in zip(p, y) if int(l) == 0]
    return pos, neg


def _psi(a, b):
    return 1.0 if a > b else (0.5 if a == b else 0.0)


def _auc_reference(y, p):
    """AUC straight from the definition, in pure Python. O(n_pos * n_neg) on purpose."""
    pos, neg = _split(y, p)
    return sum(_psi(a, b) for a in pos for b in neg) / (len(pos) * len(neg))


def _ks_reference(y, p):
    """KS and the score that attains it, in pure Python, evaluated at the observed scores."""
    pos, neg = _split(y, p)
    best_gap, best_at = -1.0, None
    for t in sorted({float(s) for s in p}):
        f_pos = sum(1 for v in pos if v <= t) / len(pos)
        f_neg = sum(1 for v in neg if v <= t) / len(neg)
        gap = abs(f_neg - f_pos)
        if gap > best_gap:
            best_gap, best_at = gap, t
    return best_gap, best_at


def _components_reference(y, p):
    """DeLong structural components, in pure Python, from the definition."""
    pos, neg = _split(y, p)
    v10 = [sum(_psi(a, b) for b in neg) / len(neg) for a in pos]
    v01 = [sum(_psi(a, b) for a in pos) / len(pos) for b in neg]
    return v10, v01


def _sample_var(v):
    m = sum(v) / len(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1)


def _delong_var_reference(y, p):
    v10, v01 = _components_reference(y, p)
    return _sample_var(v10) / len(v10) + _sample_var(v01) / len(v01)


def _delong_pair_var_reference(y, a, b):
    """Variance of the DIFFERENCE, by differencing the components rather than by a covariance
    matrix. Algebraically the same quantity as S[0,0] + S[1,1] - 2 * S[0,1], written another
    way, so it cannot agree with a student who dropped the cross term."""
    a10, a01 = _components_reference(y, a)
    b10, b01 = _components_reference(y, b)
    d10 = [x - z for x, z in zip(a10, b10)]
    d01 = [x - z for x, z in zip(a01, b01)]
    return _sample_var(d10) / len(d10) + _sample_var(d01) / len(d01)


def _normal_cdf(z):
    return 0.5 * math.erfc(-float(z) / math.sqrt(2.0))


def _z(q):
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _normal_cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


Z_975 = _z(0.975)
Z_80 = _z(0.80)


def _tie_rich(seed=17, n=180):
    """A sample with plenty of ties: scores rounded to one decimal place, as a scorecard."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    y[0], y[1] = 1, 0                      # both classes present, whatever the draw did
    p = (rng.random(n) * 0.9 + 0.05).round(1)
    return y, p


def _separable(n_pos=90, n_neg=210):
    y = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])
    p = np.concatenate([np.full(n_pos, 0.90), np.full(n_neg, 0.10)])
    return y, p


def _two_models(seed=23, n_pos=140, n_neg=360):
    """Two correlated scores on one set of records: a shared view plus private noise."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])
    latent = np.concatenate([rng.normal(1.1, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)])
    a = latent + rng.normal(0.0, 0.35, n_pos + n_neg)
    b = latent + rng.normal(0.0, 0.80, n_pos + n_neg)
    return y, a, b


# --- exercise 1: average_ranks() and auc_by_ranks() -----------------------------------------

def test_average_ranks_basic():
    r = lesson.average_ranks(np.array([0.1, 0.5, 0.5, 0.9]))
    assert np.allclose(r, [1.0, 2.5, 2.5, 4.0]), (
        f"expected [1.0, 2.5, 2.5, 4.0], got {np.asarray(r).tolist()}"
    )
    r2 = lesson.average_ranks(np.array([0.9, 0.1, 0.5]))
    assert np.allclose(r2, [3.0, 1.0, 2.0]), (
        f"ranks come back in INPUT order, not sorted order: got {np.asarray(r2).tolist()}"
    )


def test_average_ranks_tied_blocks():
    r = lesson.average_ranks(np.array([2.0, 2.0, 2.0, 5.0]))
    assert np.allclose(r, [2.0, 2.0, 2.0, 4.0]), (
        f"a block of three tied values shares rank (1+2+3)/3 = 2.0, got "
        f"{np.asarray(r).tolist()}"
    )
    flat = lesson.average_ranks(np.array([7.0, 7.0, 7.0, 7.0, 7.0]))
    assert np.allclose(flat, 3.0), f"all five tied share rank 3.0, got {np.asarray(flat).tolist()}"
    assert abs(float(np.sum(lesson.average_ranks(np.array([3.0, 1.0, 1.0, 9.0])))) - 10.0) < 1e-12, (
        "averaging inside a block preserves the total 1 + 2 + ... + n"
    )


def test_auc_hand_worked():
    got = lesson.auc_by_ranks(np.array([1, 0, 1, 0]), np.array([0.9, 0.9, 0.2, 0.1]))
    assert abs(got - 0.625) < 1e-12, (
        f"expected 0.625, got {got} — 0.75 means the tied pair was counted as a whole win"
    )
    got2 = lesson.auc_by_ranks(np.array([0, 1, 0, 1, 1]), np.array([0.4, 0.4, 0.1, 0.7, 0.2]))
    assert abs(got2 - _auc_reference(np.array([0, 1, 0, 1, 1]),
                                     np.array([0.4, 0.4, 0.1, 0.7, 0.2]))) < 1e-12, (
        f"got {got2}, the pairwise definition gives "
        f"{_auc_reference(np.array([0, 1, 0, 1, 1]), np.array([0.4, 0.4, 0.1, 0.7, 0.2]))}"
    )


def test_auc_matches_pairwise_with_ties():
    y, p = _tie_rich()
    got, want = lesson.auc_by_ranks(y, p), _auc_reference(y, p)
    assert abs(got - want) < 1e-12, (
        f"on a sample of scores rounded to one decimal place your AUC is {got:.8f} and the "
        f"pairwise definition gives {want:.8f}; a tie is worth half a win, so every member "
        "of a tied block takes the average of the ranks that block spans"
    )
    y2, p2 = _tie_rich(seed=91, n=140)
    assert abs(lesson.auc_by_ranks(y2, p2) - _auc_reference(y2, p2)) < 1e-12, (
        "a second tie-rich sample disagrees too"
    )


def test_auc_extremes_and_type():
    y, p = _separable()
    assert abs(lesson.auc_by_ranks(y, p) - 1.0) < 1e-12, "a perfectly separating score is 1.0"
    assert abs(lesson.auc_by_ranks(y, -p) - 0.0) < 1e-12, "reverse it and the AUC is 0.0"
    flat = lesson.auc_by_ranks(np.array([1, 1, 0, 0]), np.array([0.3, 0.3, 0.3, 0.3]))
    assert abs(flat - 0.5) < 1e-12, f"one score for everybody is exactly 0.5, got {flat}"
    out = lesson.auc_by_ranks(np.array([0, 1]), np.array([0.1, 0.9]))
    assert type(out) is float, (
        f"return float(...): got {type(out).__name__}, and json.dumps refuses np.float64"
    )


def test_auc_rejects_bad_input():
    cases = [
        ((np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])), "no non-events in the sample"),
        ((np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])), "no events in the sample"),
        ((np.array([1, 0, 1]), np.array([0.1, 0.2])), "labels and scores of different length"),
        ((np.array([1, 0, 2]), np.array([0.1, 0.2, 0.3])), "a label outside {0, 1}"),
        ((np.array([], dtype=np.int64), np.array([])), "an empty sample"),
    ]
    for args, why in cases:
        try:
            lesson.auc_by_ranks(*args)
        except ValueError:
            continue
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(f"{why} raised {type(exc).__name__}, not ValueError") from None
        raise AssertionError(f"{why} returned a number instead of raising ValueError")


# --- exercise 2: ks_and_gini() --------------------------------------------------------------

def test_ks_hand_worked():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.8, 0.6, 0.6, 0.3, 0.1])
    d = lesson.ks_and_gini(y, p)
    assert abs(d.ks - 2 / 3) < 1e-12, (
        f"the worked example's KS is 0.666667, got {d.ks} — compare the two classes' shares "
        "at each observed score, using 'at or below'"
    )
    assert abs(d.ks_threshold - 0.3) < 1e-12, (
        f"the widest gap sits at the score 0.3, got {d.ks_threshold}"
    )
    assert abs(d.auc - _auc_reference(y, p)) < 1e-12, (
        f"auc should be {_auc_reference(y, p):.6f}, got {d.auc}"
    )


def test_gini_is_two_auc_minus_one():
    y, p = _tie_rich(seed=41)
    d = lesson.ks_and_gini(y, p)
    want = 2.0 * _auc_reference(y, p) - 1.0
    assert abs(d.gini - want) < 1e-12, (
        f"gini is 2 * auc - 1 = {want:.6f}, got {d.gini:.6f}"
    )
    y2, p2 = _separable()
    assert abs(lesson.ks_and_gini(y2, p2).gini - 1.0) < 1e-12, (
        "a perfectly separating score has Gini 1.0"
    )


def test_ks_uses_class_shares():
    # 40 events against 460 non-events: a KS built from raw counts measures the imbalance.
    rng = np.random.default_rng(55)
    y = np.concatenate([np.ones(40, dtype=np.int64), np.zeros(460, dtype=np.int64)])
    p = np.concatenate([rng.normal(1.2, 1.0, 40), rng.normal(0.0, 1.0, 460)]).round(2)
    d = lesson.ks_and_gini(y, p)
    want_ks, want_at = _ks_reference(y, p)
    assert abs(d.ks - want_ks) < 1e-12, (
        f"on a 40-against-460 sample your KS is {d.ks:.6f} and the definition gives "
        f"{want_ks:.6f}; divide each class's cumulative count by ITS OWN size"
    )
    assert abs(d.ks_threshold - want_at) < 1e-12, (
        f"the attaining score should be {want_at}, got {d.ks_threshold}"
    )


def test_ks_threshold_is_an_observed_score():
    y, p = _tie_rich(seed=63, n=200)
    d = lesson.ks_and_gini(y, p)
    assert float(d.ks_threshold) in {float(v) for v in p}, (
        f"ks_threshold {d.ks_threshold} is not one of the observed scores — evaluate the gap "
        "at the data, not on an evenly spaced grid, or you will walk past the maximum"
    )
    want_ks, want_at = _ks_reference(y, p)
    assert abs(d.ks - want_ks) < 1e-12 and abs(d.ks_threshold - want_at) < 1e-12, (
        f"expected KS {want_ks:.6f} at {want_at}, got {d.ks:.6f} at {d.ks_threshold}; on a "
        "tie in the gap take the LOWEST attaining score"
    )


def test_ks_extremes():
    y, p = _separable()
    assert abs(lesson.ks_and_gini(y, p).ks - 1.0) < 1e-12, "perfect separation gives KS 1.0"
    flat = lesson.ks_and_gini(np.array([1, 1, 0, 0]), np.array([0.4, 0.4, 0.4, 0.4]))
    assert abs(flat.ks) < 1e-12, f"one score for everybody separates nothing; KS 0, got {flat.ks}"
    assert abs(flat.gini) < 1e-12, f"and Gini 0, got {flat.gini}"


# --- exercise 3: bootstrap_auc_ci() ---------------------------------------------------------

def test_bootstrap_point_is_sample_estimate():
    y, a, _ = _two_models(seed=31)
    ci = lesson.bootstrap_auc_ci(y, a, n_boot=2, alpha=0.05, rng=np.random.default_rng(1))
    assert abs(ci.point - _auc_reference(y, a)) < 1e-12, (
        f"with two replicates `point` came back {ci.point:.6f} against a sample AUC of "
        f"{_auc_reference(y, a):.6f} — report the sample's estimate, not the bootstrap mean"
    )
    big = lesson.bootstrap_auc_ci(y, a, n_boot=200, alpha=0.05, rng=np.random.default_rng(1))
    assert abs(big.point - ci.point) < 1e-12, "the point estimate cannot depend on n_boot"
    assert big.lo < big.point < big.hi, f"[{big.lo:.4f}, {big.hi:.4f}] misses its own point"


def test_bootstrap_preserves_record_pairing():
    y, p = _separable(n_pos=90, n_neg=210)
    ci = lesson.bootstrap_auc_ci(y, p, n_boot=300, alpha=0.05, rng=np.random.default_rng(2))
    assert ci.lo > 0.99, (
        f"a perfectly separating model's interval came back [{ci.lo:.4f}, {ci.hi:.4f}]; a "
        "lower bound near 0.5 means labels and scores were drawn with two separate index "
        "sets, which destroys the model instead of resampling it"
    )
    assert ci.hi <= 1.0 + 1e-12, f"an AUC above 1.0 is not possible; got hi {ci.hi}"


def test_bootstrap_percentile_tails():
    y, a, _ = _two_models(seed=37, n_pos=200, n_neg=500)
    ci = lesson.bootstrap_auc_ci(y, a, n_boot=1200, alpha=0.05, rng=np.random.default_rng(3))
    width_in_se = (ci.hi - ci.lo) / ci.se
    assert 3.55 <= width_in_se <= 4.35, (
        f"a 95% percentile interval spans about 2 * 1.96 = 3.92 standard errors; yours spans "
        f"{width_in_se:.2f}. Near 3.3 means the alpha and 1-alpha quantiles instead of "
        "alpha/2 and 1-alpha/2"
    )
    tight = lesson.bootstrap_auc_ci(y, a, n_boot=1200, alpha=0.50,
                                    rng=np.random.default_rng(3))
    assert tight.hi - tight.lo < ci.hi - ci.lo, (
        "a 50% interval must be narrower than a 95% one; alpha has to reach the quantiles"
    )


def test_bootstrap_stratifies_thin_class():
    rng = np.random.default_rng(4)
    y = np.concatenate([np.ones(3, dtype=np.int64), np.zeros(60, dtype=np.int64)])
    p = np.concatenate([rng.normal(1.0, 1.0, 3), rng.normal(0.0, 1.0, 60)])
    ci = lesson.bootstrap_auc_ci(y, p, n_boot=500, alpha=0.05, rng=np.random.default_rng(5))
    assert math.isfinite(ci.lo) and math.isfinite(ci.hi) and math.isfinite(ci.se), (
        f"got [{ci.lo}, {ci.hi}] se {ci.se} on a 3-event sample"
    )
    assert ci.lo < ci.hi, "500 replicates on 3 events should still produce a real interval"
    # The same thin sample, but perfectly separating. EVERY stratified replicate holds 3
    # events and 60 non-events, so every replicate AUC is exactly 1.0 and the interval is a
    # point. An unstratified draw produces an eventless replicate about 4.6% of the time; a
    # student who rescues that undefined AUC with a 0.5 default rather than by stratifying
    # gets a finite, ordered, entirely plausible interval — and it says a model that ranks
    # perfectly might be a coin. Finiteness alone cannot tell the two apart. This can.
    sep_y = np.concatenate([np.ones(3, dtype=np.int64), np.zeros(60, dtype=np.int64)])
    sep_p = np.concatenate([np.full(3, 0.90), np.full(60, 0.10)])
    sep = lesson.bootstrap_auc_ci(sep_y, sep_p, n_boot=500, alpha=0.05,
                                  rng=np.random.default_rng(16))
    assert sep.lo > 0.999 and sep.se < 1e-12, (
        f"on a perfectly separating 3-event sample every stratified replicate is exactly "
        f"1.0, so the interval is [1.0, 1.0] with se 0; yours is [{sep.lo:.4f}, "
        f"{sep.hi:.4f}] with se {sep.se:.4f}. A lower bound near 0.5 means the draw was NOT "
        "stratified and the replicates that came back with no events at all were filled in "
        "with a default instead — resample the events among the events and the non-events "
        "among the non-events, so the question never arises"
    )


def test_bootstrap_se_and_reproducibility():
    y, a, _ = _two_models(seed=43)
    first = lesson.bootstrap_auc_ci(y, a, n_boot=250, alpha=0.05, rng=np.random.default_rng(6))
    again = lesson.bootstrap_auc_ci(y, a, n_boot=250, alpha=0.05, rng=np.random.default_rng(6))
    assert (first.lo, first.hi, first.se) == (again.lo, again.hi, again.se), (
        "the same seed must reproduce the interval exactly; take every random number from "
        "the `rng` you were handed, never from np.random directly"
    )
    assert first.n_boot == 250, f"n_boot must echo back 250, got {first.n_boot}"
    assert 0.0 < first.se < 0.2, f"se {first.se} is not a plausible spread of replicate AUCs"
    # And with NO generator handed in at all. Seeding inside the call keeps two runs
    # identical; a `rng=np.random.default_rng(...)` DEFAULT in the signature is evaluated
    # once when the function is defined and then shared by every call, so the second run
    # silently disagrees with the first and a pack cannot be regenerated.
    d1 = lesson.bootstrap_auc_ci(y, a, n_boot=30, alpha=0.05)
    d2 = lesson.bootstrap_auc_ci(y, a, n_boot=30, alpha=0.05)
    assert (d1.lo, d1.hi, d1.se) == (d2.lo, d2.hi, d2.se), (
        f"two calls with no `rng` gave [{d1.lo:.6f}, {d1.hi:.6f}] and [{d2.lo:.6f}, "
        f"{d2.hi:.6f}]. Seed a FRESH generator inside the call when `rng` is None; a "
        "generator built in the default argument is created once and carries its state "
        "between calls"
    )


def test_bootstrap_rejects_bad_input():
    y, a, _ = _two_models(seed=47)
    for kwargs, why in (({"n_boot": 1}, "n_boot below 2"),
                        ({"n_boot": 0}, "n_boot of zero"),
                        ({"alpha": 0.0}, "alpha of zero"),
                        ({"alpha": 1.0}, "alpha of one")):
        try:
            lesson.bootstrap_auc_ci(y, a, rng=np.random.default_rng(7),
                                    **{"n_boot": 50, "alpha": 0.05, **kwargs})
        except ValueError:
            continue
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(f"{why} raised {type(exc).__name__}, not ValueError") from None
        raise AssertionError(f"{why} returned an interval instead of raising ValueError")


# --- exercise 4: paired_bootstrap_difference() ----------------------------------------------

def test_paired_identical_models():
    y, a, _ = _two_models(seed=53)
    out = lesson.paired_bootstrap_difference(y, a, a, n_boot=150, alpha=0.05,
                                             rng=np.random.default_rng(8))
    assert abs(out.diff) < 1e-12, f"a model against itself differs by 0, got {out.diff}"
    assert out.se_paired < 1e-12, (
        f"a model against itself has NO paired uncertainty; se_paired came back "
        f"{out.se_paired:.6f}, which means the two models were given different resamples"
    )
    assert out.se_independent > 1e-4, (
        f"se_independent is sqrt(se_a**2 + se_b**2) and neither is zero; got "
        f"{out.se_independent}"
    )
    assert abs(out.corr - 1.0) < 1e-6, (
        f"two identical models correlate at 1.0 across replicates, got {out.corr}"
    )
    assert abs(out.lo) < 1e-12 and abs(out.hi) < 1e-12, (
        f"every replicate difference is exactly zero, so the interval is [0, 0]; got "
        f"[{out.lo}, {out.hi}]"
    )


def test_paired_se_below_independent():
    y, a, b = _two_models(seed=59)
    out = lesson.paired_bootstrap_difference(y, a, b, n_boot=400, alpha=0.05,
                                             rng=np.random.default_rng(9))
    assert out.se_paired < out.se_independent, (
        f"pairing must shrink the standard error, but se_paired {out.se_paired:.5f} is not "
        f"below se_independent {out.se_independent:.5f} — one index draw has to feed BOTH "
        "models within a replicate"
    )
    assert 0.2 < out.corr < 1.0, (
        f"two models built on the same latent driver correlate strongly across replicates; "
        f"got {out.corr:.3f}"
    )
    want = math.sqrt(_delong_pair_var_reference(y, a, b))
    assert abs(out.se_paired - want) < 0.5 * want, (
        f"se_paired {out.se_paired:.5f} is nowhere near the analytic standard error of the "
        f"difference, {want:.5f}"
    )


def test_paired_diff_is_on_the_original_sample():
    y, a, b = _two_models(seed=61)
    out = lesson.paired_bootstrap_difference(y, a, b, n_boot=60, alpha=0.05,
                                             rng=np.random.default_rng(10))
    want = _auc_reference(y, a) - _auc_reference(y, b)
    assert abs(out.diff - want) < 1e-12, (
        f"diff is auc(a) - auc(b) on the original sample, {want:+.6f}; got {out.diff:+.6f}"
    )
    assert out.n_boot == 60, f"n_boot must echo back 60, got {out.n_boot}"


def test_paired_interval_brackets_the_difference():
    y, a, b = _two_models(seed=67)
    out = lesson.paired_bootstrap_difference(y, a, b, n_boot=400, alpha=0.05,
                                             rng=np.random.default_rng(11))
    assert out.lo < out.diff < out.hi, (
        f"the percentile interval [{out.lo:+.5f}, {out.hi:+.5f}] does not contain the "
        f"measured difference {out.diff:+.5f}"
    )
    wide = lesson.paired_bootstrap_difference(y, a, b, n_boot=400, alpha=0.30,
                                              rng=np.random.default_rng(11))
    assert wide.hi - wide.lo < out.hi - out.lo, (
        "a 70% interval must be narrower than a 95% one; alpha has to reach the quantiles"
    )


def test_paired_reproducible_and_validates():
    y, a, b = _two_models(seed=71)
    one = lesson.paired_bootstrap_difference(y, a, b, n_boot=120, alpha=0.05,
                                             rng=np.random.default_rng(12))
    two = lesson.paired_bootstrap_difference(y, a, b, n_boot=120, alpha=0.05,
                                             rng=np.random.default_rng(12))
    assert (one.lo, one.hi, one.se_paired) == (two.lo, two.hi, two.se_paired), (
        "the same seed must reproduce the interval exactly"
    )
    d1 = lesson.paired_bootstrap_difference(y, a, b, n_boot=30, alpha=0.05)
    d2 = lesson.paired_bootstrap_difference(y, a, b, n_boot=30, alpha=0.05)
    assert (d1.lo, d1.hi, d1.se_paired) == (d2.lo, d2.hi, d2.se_paired), (
        "two calls with no `rng` disagreed; seed a FRESH generator inside the call when "
        "`rng` is None, never in the default argument, which is evaluated once at "
        "definition time and then shared by every call"
    )
    try:
        lesson.paired_bootstrap_difference(y, a, b[:-1], n_boot=20,
                                           rng=np.random.default_rng(13))
    except ValueError:
        return
    except NotImplementedError:
        raise
    except Exception as exc:
        raise AssertionError(f"mismatched lengths raised {type(exc).__name__}") from None
    raise AssertionError("three arrays of different lengths must raise ValueError")


# --- exercise 5: delong_components(), delong_auc_ci(), delong_test() ------------------------

def test_delong_components_hand_worked():
    y = np.array([1, 1, 0, 0])
    v10, v01 = lesson.delong_components(y, np.array([0.9, 0.2, 0.5, 0.1]))
    assert np.allclose(np.asarray(v10, dtype=float), [1.0, 0.5]), (
        f"V10 should be [1.0, 0.5], got {np.asarray(v10).tolist()}"
    )
    assert np.allclose(np.asarray(v01, dtype=float), [0.5, 1.0]), (
        f"V01 should be [0.5, 1.0], got {np.asarray(v01).tolist()} — V01[j] counts the "
        "events ABOVE non-event j, with a tie worth half"
    )
    tied10, tied01 = lesson.delong_components(np.array([1, 0]), np.array([0.4, 0.4]))
    assert np.allclose(np.asarray(tied10, dtype=float), [0.5]) and \
        np.allclose(np.asarray(tied01, dtype=float), [0.5]), (
        "an event tied with the only non-event scores 0.5 on both sides, got "
        f"{np.asarray(tied10).tolist()} and {np.asarray(tied01).tolist()}"
    )


def test_delong_components_mean_is_the_auc():
    y, p = _tie_rich(seed=73, n=160)
    v10, v01 = lesson.delong_components(y, p)
    want = _auc_reference(y, p)
    assert abs(float(np.mean(v10)) - want) < 1e-12, (
        f"mean(V10) is the AUC {want:.8f}, got {float(np.mean(v10)):.8f}"
    )
    assert abs(float(np.mean(v01)) - want) < 1e-12, (
        f"mean(V01) is the same AUC {want:.8f}, got {float(np.mean(v01)):.8f}"
    )
    assert len(v10) == int((np.asarray(y) == 1).sum()), "one V10 entry per event"
    assert len(v01) == int((np.asarray(y) == 0).sum()), "one V01 entry per non-event"


def test_delong_variance_matches_reference():
    y, p = _tie_rich(seed=79, n=150)
    ci = lesson.delong_auc_ci(y, p, 0.05)
    want = math.sqrt(_delong_var_reference(y, p))
    assert abs(ci.se - want) < 1e-12, (
        f"se {ci.se:.10f} against the reference {want:.10f}; the variance is "
        "S10 / n_pos + S01 / n_neg with both variances taken at ddof=1"
    )
    y2, a2, _ = _two_models(seed=83, n_pos=60, n_neg=90)
    want2 = math.sqrt(_delong_var_reference(y2, a2))
    assert abs(lesson.delong_auc_ci(y2, a2, 0.05).se - want2) < 1e-12, (
        "a second, tie-free sample disagrees too; check you are not dividing by n instead of "
        "n - 1, or by the wrong class count"
    )


def test_delong_ci_shape():
    y, a, _ = _two_models(seed=89, n_pos=80, n_neg=200)
    ci = lesson.delong_auc_ci(y, a, 0.05)
    assert ci.n_boot == 0, (
        f"this interval resampled nothing, so n_boot is 0, got {ci.n_boot}"
    )
    assert abs(ci.point - _auc_reference(y, a)) < 1e-12, (
        f"point should be the sample AUC {_auc_reference(y, a):.8f}, got {ci.point:.8f}"
    )
    assert abs((ci.hi - ci.lo) - 2 * Z_975 * ci.se) < 1e-9, (
        f"a 95% normal interval is 2 * 1.959964 * se wide; yours is {ci.hi - ci.lo:.8f} "
        f"against {2 * Z_975 * ci.se:.8f}"
    )
    wide = lesson.delong_auc_ci(y, a, 0.20)
    assert abs((wide.hi - wide.lo) - 2 * _z(0.90) * ci.se) < 1e-9, (
        "alpha must reach the quantile: an 80% interval is 2 * 1.281552 * se wide"
    )


def test_delong_identical_models():
    y, a, _ = _two_models(seed=97)
    out = lesson.delong_test(y, a, a)
    assert out.se == 0.0 and out.diff == 0.0, (
        f"a model against itself has diff 0 and se 0; got diff {out.diff}, se {out.se}"
    )
    assert out.z == 0.0, f"z is 0 when se is 0, got {out.z}"
    assert abs(out.p_value - 1.0) < 1e-12, (
        f"z = 0 gives a two-sided p-value of exactly 1.0, got {out.p_value} — a value near "
        "0.5 is a one-sided tail"
    )


def test_delong_covariance_term():
    y, a, b = _two_models(seed=101, n_pos=70, n_neg=130)
    out = lesson.delong_test(y, a, b)
    want = math.sqrt(_delong_pair_var_reference(y, a, b))
    assert abs(out.se - want) < 1e-12, (
        f"se of the difference is {out.se:.10f} against the reference {want:.10f}; you need "
        "S[0,0] + S[1,1] - 2 * S[0,1], and the cross term is where the pairing lives"
    )
    independent = math.hypot(math.sqrt(_delong_var_reference(y, a)),
                             math.sqrt(_delong_var_reference(y, b)))
    assert out.se < independent, (
        f"the paired standard error {out.se:.5f} is not below the independent "
        f"{independent:.5f}; the covariance term was dropped"
    )


def test_delong_test_fields():
    y, a, b = _two_models(seed=103, n_pos=90, n_neg=160)
    out = lesson.delong_test(y, a, b)
    assert abs(out.auc_a - _auc_reference(y, a)) < 1e-12, "auc_a is model a's AUC"
    assert abs(out.auc_b - _auc_reference(y, b)) < 1e-12, "auc_b is model b's AUC"
    assert abs(out.diff - (_auc_reference(y, a) - _auc_reference(y, b))) < 1e-12, (
        f"diff is auc_a - auc_b, got {out.diff:+.6f}"
    )
    want_z = out.diff / math.sqrt(_delong_pair_var_reference(y, a, b))
    assert abs(out.z - want_z) < 1e-9, f"z should be {want_z:.6f}, got {out.z:.6f}"
    want_p = 2.0 * (1.0 - _normal_cdf(abs(want_z)))
    assert abs(out.p_value - want_p) < 1e-12, (
        f"the two-sided p-value is 2 * (1 - Phi(|z|)) = {want_p:.8f}, got {out.p_value:.8f}"
    )


# --- exercise 6: measurability() -------------------------------------------------------------

def test_mdd_formula():
    m = lesson.measurability(0.004, 0.006, 1400, 0.010, 0.05, 0.80)
    want = (Z_975 + Z_80) * 0.006
    assert abs(m.mdd - want) < 1e-9, (
        f"mdd should be (1.959964 + 0.841621) * 0.006 = {want:.8f}, got {m.mdd:.8f}; a value "
        f"near {Z_975 * 0.006:.6f} means you used the critical value alone and asked for no "
        "power at all"
    )
    m90 = lesson.measurability(0.004, 0.006, 1400, 0.010, 0.05, 0.90)
    assert m90.mdd > m.mdd, "asking for 90% power raises the minimum detectable difference"
    m01 = lesson.measurability(0.004, 0.006, 1400, 0.010, 0.01, 0.80)
    assert m01.mdd > m.mdd, "a stricter alpha raises it too"


def test_measurability_significance_boundary():
    se = 0.004
    on = lesson.measurability(Z_975 * se, se, 900, 0.010, 0.05, 0.80)
    assert on.verdict == "SIGNIFICANT", (
        f"a difference landing EXACTLY on the critical value is significant; got "
        f"{on.verdict}. The comparison is >=, and this is the case a validator is asked to "
        "defend"
    )
    just_under = lesson.measurability(Z_975 * se * (1 - 1e-9), se, 900, 0.010, 0.05, 0.80)
    assert just_under.verdict != "SIGNIFICANT", (
        "a hair below the critical value is not significant"
    )
    neg = lesson.measurability(-Z_975 * se * 1.5, se, 900, 0.010, 0.05, 0.80)
    assert neg.verdict == "SIGNIFICANT", (
        "the test is two-sided: a large NEGATIVE difference is significant too, so compare "
        "the absolute value"
    )


def test_measurability_precedence():
    loud = lesson.measurability(0.30, 0.02, 500, 0.001, 0.05, 0.80)
    assert loud.mdd > 0.001, "this case is set up so the mdd is above the material threshold"
    assert loud.verdict == "SIGNIFICANT", (
        f"got {loud.verdict}: a difference 15 standard errors out is a finding whatever the "
        "minimum detectable difference says. Significance is decided FIRST"
    )


def test_measurability_not_significant_versus_not_measurable():
    powered = lesson.measurability(0.0005, 0.0010, 9000, 0.010, 0.05, 0.80)
    assert powered.verdict == "NOT SIGNIFICANT", (
        f"mdd {powered.mdd:.5f} is inside the material threshold 0.010, so this sample DID "
        f"have the power and found nothing; got {powered.verdict}"
    )
    starved = lesson.measurability(0.0005, 0.0100, 200, 0.010, 0.05, 0.80)
    assert starved.verdict == "NOT MEASURABLE", (
        f"mdd {starved.mdd:.5f} is above the material threshold 0.010, so this sample could "
        f"not have found a difference worth acting on; got {starved.verdict}"
    )


def test_n_required_scaling_and_ceiling():
    m = lesson.measurability(0.004, 0.006, 1400, 0.010, 0.05, 0.80)
    want = math.ceil(1400 * (((Z_975 + Z_80) * 0.006) / 0.010) ** 2)
    assert m.n_required == want, (
        f"n_required should be ceil(1400 * (mdd / 0.010)**2) = {want}, got {m.n_required}"
    )
    assert isinstance(m.n_required, int) and not isinstance(m.n_required, bool), (
        f"n_required is a Python int, got {type(m.n_required).__name__}"
    )
    doubled = lesson.measurability(0.004, 0.006, 1400, 0.020, 0.05, 0.80)
    assert abs(doubled.n_required - want / 4) <= 1, (
        f"doubling the material threshold quarters the records needed: {want} -> "
        f"{doubled.n_required}"
    )
    exact = lesson.measurability(0.0, 1.0, 1, 1000.0, 0.05, 0.80)
    assert exact.n_required >= 1, "n_required is rounded UP and is never zero"


def test_measurability_reason_and_validation():
    m = lesson.measurability(0.004, 0.006, 1400, 0.010, 0.05, 0.80)
    head = f"mdd={m.mdd:.4f} material={0.010:.4f}"
    assert m.reason.startswith(head), (
        f"reason must open with {head!r} so a downstream report can parse it; yours opens "
        f"{m.reason[:40]!r}"
    )
    assert str(m.n_required) in m.reason, (
        "a NOT MEASURABLE verdict has to say how many records would have been enough"
    )
    for kwargs, why in (({"se": 0.0}, "a standard error of zero"),
                        ({"se": -0.1}, "a negative standard error"),
                        ({"material": 0.0}, "a material threshold of zero"),
                        ({"n": 0}, "a sample size of zero"),
                        ({"power": 1.0}, "power of one")):
        args = {"observed_diff": 0.01, "se": 0.01, "n": 100, "material": 0.01,
                "alpha": 0.05, "power": 0.8, **kwargs}
        try:
            lesson.measurability(**args)
        except ValueError:
            continue
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(f"{why} raised {type(exc).__name__}, not ValueError") from None
        raise AssertionError(f"{why} returned a verdict instead of raising ValueError")


# --- exercise 7: interval_coverage() ---------------------------------------------------------

def test_coverage_hand_worked():
    cov = lesson.interval_coverage(np.array([0.1, 0.4, 0.7]), np.array([0.3, 0.6, 0.9]), 0.5)
    assert abs(cov.coverage - 1 / 3) < 1e-12, f"coverage should be 1/3, got {cov.coverage}"
    assert cov.n == 3, f"n should be 3, got {cov.n}"
    assert cov.miss_low == 1, (
        f"one interval sits entirely ABOVE the truth, so the truth fell below it: miss_low "
        f"is 1, got {cov.miss_low}"
    )
    assert cov.miss_high == 1, (
        f"one interval sits entirely BELOW the truth: miss_high is 1, got {cov.miss_high}"
    )


def test_coverage_inclusive_endpoints():
    cov = lesson.interval_coverage(np.array([0.5, 0.2]), np.array([0.8, 0.5]), 0.5)
    assert cov.coverage == 1.0, (
        f"both intervals touch the truth at an endpoint and therefore cover it; got "
        f"{cov.coverage} — the comparisons are lo <= truth <= hi"
    )
    assert cov.miss_low == 0 and cov.miss_high == 0, "a touching endpoint is not a miss"
    assert cov.se == 0.0, f"perfect coverage has zero binomial se, got {cov.se}"


def test_coverage_miss_sides_are_not_swapped():
    los = np.array([0.90, 0.10, 0.10, 0.95])
    his = np.array([0.99, 0.20, 0.80, 0.99])
    cov = lesson.interval_coverage(los, his, 0.50)
    assert cov.miss_low == 2, (
        f"two intervals start above 0.50, so the truth fell below them: expected miss_low 2, "
        f"got {cov.miss_low} (miss_high {cov.miss_high}) — the two counters are swapped"
    )
    assert cov.miss_high == 1, f"one interval ends below 0.50; expected miss_high 1, got {cov.miss_high}"
    assert abs(cov.coverage - 0.25) < 1e-12, f"one of four covers; got {cov.coverage}"
    assert cov.miss_low + cov.miss_high + int(round(cov.coverage * cov.n)) == cov.n, (
        "every interval is a hit, a low miss or a high miss, and nothing else"
    )


def test_coverage_standard_error():
    rng = np.random.default_rng(15)
    n = 400
    hit = rng.random(n) < 0.93
    los = np.where(hit, 0.4, 0.6)
    his = np.where(hit, 0.6, 0.8)
    cov = lesson.interval_coverage(los, his, 0.5)
    want = math.sqrt(cov.coverage * (1 - cov.coverage) / n)
    assert abs(cov.se - want) < 2e-2 * want, (
        f"se is sqrt(p * (1 - p) / n) = {want:.6f}, got {cov.se:.6f}; a figure about "
        f"{math.sqrt(cov.coverage / n):.6f} means the (1 - p) factor is missing, and one "
        f"about {math.sqrt(cov.coverage * (1 - cov.coverage)):.6f} means the division by n is"
    )


def test_coverage_rejects_bad_input():
    for args, why in (((np.array([0.1]), np.array([0.2, 0.3]), 0.5), "mismatched lengths"),
                      ((np.array([]), np.array([]), 0.5), "an empty family of intervals")):
        try:
            lesson.interval_coverage(*args)
        except ValueError:
            continue
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(f"{why} raised {type(exc).__name__}, not ValueError") from None
        raise AssertionError(f"{why} returned a result instead of raising ValueError")


RUBRIC = [
    (test_average_ranks_basic, 3,
     "np.argsort(kind='mergesort'), then scatter the averaged ranks back to input order"),
    (test_average_ranks_tied_blocks, 3,
     "np.unique(sorted_values, return_counts=True) gives every tied block's size in one call"),
    (test_auc_hand_worked, 4,
     "(sum of event ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)"),
    (test_auc_matches_pairwise_with_ties, 5,
     "every member of a tied block takes the mean of the ranks the block spans"),
    (test_auc_extremes_and_type, 4,
     "return float(...) — a numpy scalar is not a Python float"),
    (test_auc_rejects_bad_input, 4,
     "check shapes, emptiness, labels in {0, 1} and both classes present, before ranking"),
    (test_ks_hand_worked, 5,
     "np.searchsorted(sorted_class, thresholds, side='right') / class_size, for both classes"),
    (test_gini_is_two_auc_minus_one, 3,
     "gini = 2 * auc - 1, computed from the AUC you already have"),
    (test_ks_uses_class_shares, 4,
     "divide each class's cumulative count by ITS OWN size, never by the total"),
    (test_ks_threshold_is_an_observed_score, 3,
     "thresholds are np.unique(scores); np.argmax returns the FIRST maximum, which is the "
     "lowest attaining score"),
    (test_ks_extremes, 3,
     "perfect separation is KS 1.0 and Gini 1.0; a constant score is 0.0 for both"),
    (test_bootstrap_point_is_sample_estimate, 4,
     "point = auc_by_ranks(y, p) on the original arrays, never reps.mean()"),
    (test_bootstrap_preserves_record_pairing, 5,
     "draw index positions once and index BOTH y and the scores with them"),
    (test_bootstrap_percentile_tails, 4,
     "np.quantile(reps, [alpha / 2, 1 - alpha / 2])"),
    (test_bootstrap_stratifies_thin_class, 4,
     "resample the events among the events and the non-events among the non-events"),
    (test_bootstrap_se_and_reproducibility, 3,
     "take every draw from the `rng` argument; se is reps.std(ddof=1); echo n_boot back"),
    (test_bootstrap_rejects_bad_input, 3,
     "n_boot < 2 and alpha outside the open interval (0, 1) are both ValueError"),
    (test_paired_identical_models, 6,
     "ONE stratified index draw per replicate, then score both models on those same records"),
    (test_paired_se_below_independent, 4,
     "se_paired is the sd of (reps_a - reps_b); se_independent is hypot of the two sds"),
    (test_paired_diff_is_on_the_original_sample, 3,
     "diff = auc(y, a) - auc(y, b) measured outside the replicate loop"),
    (test_paired_interval_brackets_the_difference, 3,
     "percentiles of the replicate DIFFERENCES, at alpha / 2 and 1 - alpha / 2"),
    (test_paired_reproducible_and_validates, 3,
     "draw from `rng` only, and reject arrays of differing length"),
    (test_delong_components_hand_worked, 4,
     "V10[i] = (#non-events below i + 0.5 * #tied) / n_neg; V01[j] counts events ABOVE j"),
    (test_delong_components_mean_is_the_auc, 4,
     "both component vectors average to the AUC; if they do not, a side or a tie is wrong"),
    (test_delong_variance_matches_reference, 6,
     "Var = S10 / n_pos + S01 / n_neg, both sample variances at ddof=1"),
    (test_delong_ci_shape, 4,
     "half-width = normal_quantile(1 - alpha / 2) * se; point = mean(V10); n_boot = 0"),
    (test_delong_identical_models, 5,
     "se 0 gives z 0, and 2 * (1 - normal_cdf(0)) is exactly 1.0"),
    (test_delong_covariance_term, 5,
     "Var(diff) = S[0,0] + S[1,1] - 2 * S[0,1]; np.cov on the stacked component vectors"),
    (test_delong_test_fields, 4,
     "z = diff / se and p_value = 2 * (1 - normal_cdf(abs(z)))"),
    (test_mdd_formula, 5,
     "mdd = (normal_quantile(1 - alpha / 2) + normal_quantile(power)) * se"),
    (test_measurability_significance_boundary, 4,
     "abs(observed_diff) >= z_crit * se, inclusive and two-sided"),
    (test_measurability_precedence, 4,
     "test significance FIRST, then fall through to the measurability rule"),
    (test_measurability_not_significant_versus_not_measurable, 3,
     "NOT MEASURABLE when mdd > material; NOT SIGNIFICANT when it is not"),
    (test_n_required_scaling_and_ceiling, 5,
     "n_required = int(math.ceil(n * (mdd / material) ** 2))"),
    (test_measurability_reason_and_validation, 3,
     "open the reason with the f-string head from the docstring, then the sentence"),
    (test_coverage_hand_worked, 4,
     "miss_low counts truth < lo; miss_high counts truth > hi"),
    (test_coverage_inclusive_endpoints, 3,
     "lo <= truth <= hi, so an endpoint exactly on the truth is a hit"),
    (test_coverage_miss_sides_are_not_swapped, 4,
     "an interval that sits ABOVE the truth is a miss_low: the TRUTH fell low, not the "
     "interval"),
    (test_coverage_standard_error, 3,
     "se = math.sqrt(coverage * (1 - coverage) / n)"),
    (test_coverage_rejects_bad_input, 2,
     "mismatched lengths and an empty family are both ValueError"),
]
