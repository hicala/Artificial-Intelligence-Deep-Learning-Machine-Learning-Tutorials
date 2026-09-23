"""Autograder rubric for P04-L05. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson, and are deliberate:

* every expected value is computed HERE by an independent route: bin edges from a scan of
  cumulative counts over the distinct scores (the grouping the Stata manual documents) rather
  than from sorted positions; ECE, PSI and the Hosmer-Lemeshow statistic from plain loops; the
  chi-square tail from the regularised incomplete gamma by series and continued fraction, not
  the two-step recurrence the notebook teaches, and against the NIST/SEMATECH published table.
  Nothing is graded against the notebook's own synthetic sample.
* the exercises chain, so each exercise is graded on cases that do NOT depend on the one before
  it. For the length of a case, `_Swap` puts this file's reference in place of the earlier
  exercise the function calls — `equal_frequency_edges` and `ece_on_edges` under exercise 3,
  `equal_frequency_edges` and `chi2_sf` under exercise 5, `characteristic_edges` under
  exercises 6 and 7 — and restores the student's own afterwards. A broken exercise 1 costs
  exercise 1's points, not exercise 5's as well. An answer that skips the helper and computes
  the same thing inline is graded on what it returns, so it still passes.
* `test_hl_dof_independent_sample` is the wrong-dof case: the textbook g - 2 applied to a
  sample the model was not fitted on. `test_hl_dof_follows_groups_formed` is the other one:
  the degrees of freedom taken from the groups REQUESTED rather than formed.
* `test_null_two_sample_scale` is what stops a null with a frozen baseline (or one drawn
  without replacement at full size, which is the same thing) from scoring: its mean is about
  half the two-sample figure when the two sizes are equal.
* `test_null_current_size_and_edges` puts twenty records against two thousand. The honest
  null is dominated by bins the month leaves empty; edges cut on the month, or a month drawn
  at the baseline's size, both come out an order of magnitude too small.
* `test_null_edges_cut_on_the_baseline` turns that round: twenty development records against
  a month of two thousand. Edges cut on the month, or on both samples pooled, leave the
  twenty-record baseline with empty bins, and the floor makes the null several times too
  large. Pooled edges passed every other case in the mutation run; this one exists for it.
* `test_csi_cut_on_baseline` and `test_null_two_sample_scale` each repeat their case at
  n_bins=4. Independent review found that a function which ignored its own n_bins argument
  (calling characteristic_edges at its default) scored 141/141, because every other case used
  the default of ten.
* `test_threshold_float_tolerance` uses 0.29 x 100, which is 28.999999999999996 in floating
  point: a bare floor allows one alarm too few.
"""
import importlib.util, inspect, math, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)


# --- independent references -----------------------------------------------------------------

def _distinct_with_counts(values):
    counts = {}
    for v in values:
        counts[float(v)] = counts.get(float(v), 0) + 1
    return sorted(counts.items())


def _ef_reference(p, n_bins):
    """Equal-frequency edges by scanning cumulative counts: a distinct score is an edge when the
    running count first reaches k * n / n_bins for some k (the Stata grouping)."""
    n = len(p)
    ends, cum = [], 0
    for value, count in _distinct_with_counts(p):
        prev, cum = cum, cum + count
        if any(prev * n_bins < k * n <= cum * n_bins for k in range(1, n_bins + 1)):
            ends.append(value)
    return np.array(sorted(set([0.0] + ends[:-1] + [1.0])))


def _right_closed_bin(v, edges):
    for i in range(len(edges) - 1):
        if edges[i] < v <= edges[i + 1]:
            return i
    return 0  # only v == edges[0] == 0.0 falls through


def _ece_reference(y, p, edges):
    k = len(edges) - 1
    cnt, obs, pred = [0] * k, [0.0] * k, [0.0] * k
    for label, prob in zip(y, p):
        b = _right_closed_bin(float(prob), edges)
        cnt[b] += 1
        obs[b] += float(label)
        pred[b] += float(prob)
    n = len(p)
    return sum((cnt[b] / n) * abs(obs[b] / cnt[b] - pred[b] / cnt[b]) for b in range(k) if cnt[b])


def _char_edges_reference(values, n_bins):
    """Left-closed stability edges by scanning cumulative counts: the value found at zero-based
    position min(ceil(k n / n_bins), n - 1), for k = 1 .. n_bins - 1."""
    n = len(values)
    pairs = _distinct_with_counts(values)
    wanted = [min((k * n + n_bins - 1) // n_bins, n - 1) for k in range(1, n_bins)]
    inner, cum = set(), 0
    for value, count in pairs:
        prev, cum = cum, cum + count
        if any(prev <= j < cum for j in wanted):
            inner.add(value)
    smallest = pairs[0][0]
    return np.array([-np.inf] + sorted(v for v in inner if v > smallest) + [np.inf])


def _psi_reference(expected, actual, edges, floor=1e-6):
    edges = [float(e) for e in edges]
    k = len(edges) - 1

    def shares(values):
        counts = [0] * k
        for v in values:
            slot = k - 1
            for i in range(k):
                if edges[i] <= v < edges[i + 1]:
                    slot = i
                    break
            counts[slot] += 1
        return [c / len(values) for c in counts]

    e, a = shares(expected), shares(actual)
    return sum((max(ai, floor) - max(ei, floor)) * math.log(max(ai, floor) / max(ei, floor))
               for ei, ai in zip(e, a))


def _gamma_q(a, x):
    """Regularised upper incomplete gamma Q(a, x): series for P below a + 1, Lentz continued
    fraction for Q above (the Numerical Recipes route, not the notebook's recurrence)."""
    if x <= 0:
        return 1.0
    if x < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(100000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-17:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-17:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _chi2_reference(x, k):
    return _gamma_q(k / 2.0, x / 2.0)


def _hl_reference(y, p, edges):
    k = len(edges) - 1
    cnt, obs, exp_ = [0] * k, [0.0] * k, [0.0] * k
    for label, prob in zip(y, p):
        b = _right_closed_bin(float(prob), edges)
        cnt[b] += 1
        obs[b] += float(label)
        exp_[b] += float(prob)
    return sum((obs[b] - exp_[b]) ** 2 / (exp_[b] * (1 - exp_[b] / cnt[b])) for b in range(k))


def _raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    except NotImplementedError:
        raise
    except Exception:
        return False
    return False


class _Swap:
    """Put reference implementations in place of earlier exercises for one case, then restore."""

    def __init__(self, **replacements):
        self._replacements = replacements

    def __enter__(self):
        self._saved = {name: getattr(lesson, name) for name in self._replacements}
        for name, fn in self._replacements.items():
            setattr(lesson, name, fn)
        return self

    def __exit__(self, *exc):
        for name, fn in self._saved.items():
            setattr(lesson, name, fn)
        return False


def _ReferenceEdges():
    return _Swap(characteristic_edges=_char_edges_reference)


def _ReferenceBins():
    return _Swap(equal_frequency_edges=_ef_reference,
                 ece_on_edges=lambda y, p, edges: float(_ece_reference(y, p, edges)))


def _ReferenceGroups():
    return _Swap(equal_frequency_edges=_ef_reference,
                 chi2_sf=lambda x, dof: float(_chi2_reference(float(x), int(dof))))


# --- exercise 1: equal_frequency_edges() -----------------------------------------------------

def test_ef_untied_rank_rule():
    for n, m in ((37, 2), (37, 5), (40, 4), (40, 10), (12, 3), (7, 3), (11, 4)):
        p = np.linspace(0.013, 0.987, n)
        got = lesson.equal_frequency_edges(p, m)
        want = _ef_reference(p, m)
        assert np.array_equal(np.asarray(got, dtype=float), want), (
            f"{n} untied scores in {m} bins: got {np.round(got, 4).tolist()}, expected "
            f"{np.round(want, 4).tolist()}. Each edge is the score of the record at 1-based rank "
            "ceil(k * n / n_bins) — not an interpolated quantile, and not position "
            "k * n // n_bins"
        )


def test_ef_ties_merge_not_split():
    p = np.array([0.02] * 30 + [0.05] * 45 + [0.1] * 15 + [0.3] * 10)
    got = np.asarray(lesson.equal_frequency_edges(p, 10), dtype=float)
    want = _ef_reference(p, 10)
    assert np.all(np.diff(got) > 0), (
        f"edges {got.tolist()} are not strictly increasing: coincident edges must merge"
    )
    assert np.array_equal(got, want), (
        f"four tied scores in ten requested bins gave {got.tolist()}; expected {want.tolist()} — "
        "four bins, because several shares close on the same score"
    )
    interior = set(got[1:-1].tolist())
    assert interior <= set(p.tolist()), (
        f"interior edges {sorted(interior)} include a score nobody was given — linear "
        "quantiles interpolate between grades"
    )


def test_ef_never_empty_or_split():
    rng = np.random.default_rng(11)
    grades = np.array([0.004, 0.011, 0.023, 0.051, 0.12, 0.26, 0.55])
    p = grades[rng.choice(grades.size, 500, p=[0.05, 0.1, 0.3, 0.25, 0.15, 0.1, 0.05])]
    got = np.asarray(lesson.equal_frequency_edges(p, 20), dtype=float)
    assert np.array_equal(got, _ef_reference(p, 20)), (
        f"a seven-grade score in 20 requested bins gave {got.tolist()}"
    )
    counts = [0] * (got.size - 1)
    for v in p:
        counts[_right_closed_bin(float(v), got)] += 1
    assert min(counts) > 0, (
        f"bin counts {counts} include an empty bin on the very sample the edges were cut from"
    )
    assert got.size - 1 < 20, "twenty bins cannot be formed from seven grades without splitting"


def test_ef_extremes():
    flat = np.asarray(lesson.equal_frequency_edges(np.full(12, 0.3), 5), dtype=float)
    assert flat.tolist() == [0.0, 1.0], (
        f"a constant score has one bin, [0, 1]; got {flat.tolist()}"
    )
    top = np.asarray(lesson.equal_frequency_edges(np.array([0.2, 0.4] + [1.0] * 8), 4), dtype=float)
    assert top.tolist() == _ef_reference(np.array([0.2, 0.4] + [1.0] * 8), 4).tolist() and \
        np.all(np.diff(top) > 0), f"a score tied at 1.0 gave {top.tolist()}"
    zeros = np.asarray(lesson.equal_frequency_edges(np.array([0.0] * 6 + [0.5, 0.7]), 4),
                       dtype=float)
    assert zeros[0] == 0.0 and np.all(np.diff(zeros) > 0), (
        f"scores of exactly 0.0 gave {zeros.tolist()}: merge again after putting 0.0 in front"
    )
    one = np.asarray(lesson.equal_frequency_edges(np.array([0.1, 0.9]), 1), dtype=float)
    assert one.tolist() == [0.0, 1.0], "n_bins=1 is the single bin [0, 1]"


def test_ef_rejects_bad_input():
    for args, why in (((np.array([0.2, 0.4]), 0), "n_bins of 0"),
                      ((np.array([]), 3), "an empty sample"),
                      ((np.array([0.2, 1.2]), 2), "a probability above 1"),
                      ((np.array([-0.1, 0.5]), 2), "a probability below 0"),
                      ((np.array([0.2, np.nan]), 2), "a nan")):
        assert _raises(ValueError, lesson.equal_frequency_edges, *args), (
            f"{why} must raise ValueError"
        )


# --- exercise 2: ece_on_edges() --------------------------------------------------------------

def test_ece_edges_matches_module1():
    rng = np.random.default_rng(3)
    p = rng.random(400)
    y = (rng.random(400) < p ** 1.3).astype(int)
    for m in (1, 3, 10, 25):
        got = lesson.ece_on_edges(y, p, np.linspace(0.0, 1.0, m + 1))
        want = _ece_reference(y, p, np.linspace(0.0, 1.0, m + 1))
        assert abs(got - want) < 1e-12, (
            f"with {m} equal-width edges: got {got:.8f}, module 1's definition gives {want:.8f}"
        )


def test_ece_edges_right_closed_membership():
    edges = np.array([0.0, 0.25, 0.5, 1.0])
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([0.25, 0.5, 0.0, 0.26, 1.0])
    got = lesson.ece_on_edges(y, p, edges)
    want = _ece_reference(y, p, edges)
    assert abs(got - want) < 1e-12, (
        f"got {got:.6f}, expected {want:.6f}: a score on an interior edge joins the LOWER bin "
        "and 0.0 joins the first — bins are closed on the right, as in module 1"
    )


def test_ece_edges_arbitrary_edges():
    rng = np.random.default_rng(4)
    p = rng.beta(1.2, 6.0, 300)
    y = (rng.random(300) < p).astype(int)
    for edges in (np.array([0.0, 0.03, 0.05, 0.4, 0.95, 1.0]),
                  np.array([0.0, 0.9, 0.91, 0.92, 1.0])):
        got = lesson.ece_on_edges(y, p, edges)
        want = _ece_reference(y, p, edges)
        assert abs(got - want) < 1e-12, (
            f"on edges {edges.tolist()} got {got:.8f}, expected {want:.8f} — empty bins must "
            "contribute nothing, and each occupied bin weighs count / total"
        )


def test_ece_edges_type_and_perfect():
    got = lesson.ece_on_edges(np.array([0, 1, 0, 1]), np.array([0.0, 1.0, 0.5, 0.5]),
                              np.array([0.0, 0.4, 0.6, 1.0]))
    assert type(got) is float, f"return a plain Python float, got {type(got).__name__}"
    assert abs(got) < 1e-15, f"perfectly calibrated bins score 0.0, got {got}"


def test_ece_edges_rejects_bad_input():
    y, p = np.array([0, 1]), np.array([0.2, 0.8])
    for args, why in (((y, p, np.array([0.1, 0.5, 1.0])), "edges starting above 0.0"),
                      ((y, p, np.array([0.0, 0.5, 0.9])), "edges ending below 1.0"),
                      ((y, p, np.array([0.0, 0.5, 0.5, 1.0])), "a repeated edge"),
                      ((y, p, np.array([0.0])), "a single edge"),
                      ((np.array([0, 2]), p, np.array([0.0, 1.0])), "a label of 2"),
                      ((y, np.array([0.2, 1.3]), np.array([0.0, 1.0])), "a probability of 1.3"),
                      ((y, np.array([0.2]), np.array([0.0, 1.0])), "mismatched lengths")):
        assert _raises(ValueError, lesson.ece_on_edges, *args), f"{why} must raise ValueError"


# --- exercise 3: ece_sensitivity() -----------------------------------------------------------

_SENS_P = np.array([0.03] * 8 + [0.07] * 8 + [0.12] * 3 + [0.33, 0.35, 0.61, 0.62, 0.81])
_SENS_Y = np.array([0, 0, 0, 1, 0, 0, 0, 0,  0, 1, 0, 0, 0, 0, 0, 0,  0, 1, 0,
                    0, 1, 1, 0, 1])


def test_sensitivity_order_and_types():
    with _ReferenceBins():
        rows = lesson.ece_sensitivity(_SENS_Y, _SENS_P, (3, 6, 12))
        assert len(rows) == 6, f"two schemes times three counts is six rows, got {len(rows)}"
        assert [(r.scheme, r.requested) for r in rows] == [
            ("equal-width", 3), ("equal-width", 6), ("equal-width", 12),
            ("equal-frequency", 3), ("equal-frequency", 6), ("equal-frequency", 12)], (
            "every equal-width row first, then every equal-frequency row, counts in the order given"
        )
        for r in rows:
            assert type(r.ece) is float, "ece must be a plain float"
            ints = (r.requested, r.bins_used, r.occupied, r.min_support)
            assert all(type(v) is int for v in ints), (
                "requested, bins_used, occupied and min_support must be plain ints"
            )


def test_sensitivity_values():
    with _ReferenceBins():
        rows = lesson.ece_sensitivity(_SENS_Y, _SENS_P, (3, 6, 12))
        for r in rows:
            edges = (np.linspace(0.0, 1.0, r.requested + 1) if r.scheme == "equal-width"
                     else _ef_reference(_SENS_P, r.requested))
            want = _ece_reference(_SENS_Y, _SENS_P, edges)
            assert abs(r.ece - want) < 1e-12, (
                f"{r.scheme} with {r.requested} bins: ece {r.ece:.8f}, expected {want:.8f}"
            )


def test_sensitivity_bins_used_and_occupied():
    with _ReferenceBins():
        rows = {(r.scheme, r.requested): r
                for r in lesson.ece_sensitivity(_SENS_Y, _SENS_P, (3, 6, 12))}
        for m in (3, 6, 12):
            ew = rows[("equal-width", m)]
            assert ew.bins_used == m, f"equal-width with {m} bins uses {m} bins, got {ew.bins_used}"
            edges = np.linspace(0.0, 1.0, m + 1)
            counts = [0] * m
            for v in _SENS_P:
                counts[_right_closed_bin(float(v), edges)] += 1
            assert ew.occupied == sum(c > 0 for c in counts), (
                f"equal-width {m}: {ew.occupied} occupied, expected {sum(c > 0 for c in counts)}"
            )
            ef = rows[("equal-frequency", m)]
            assert ef.bins_used == _ef_reference(_SENS_P, m).size - 1, (
                f"equal-frequency {m}: bins_used {ef.bins_used}, but the tied score supports only "
                f"{_ef_reference(_SENS_P, m).size - 1} — report the bins USED"
            )


def test_sensitivity_min_support_and_validation():
    with _ReferenceBins():
        rows = lesson.ece_sensitivity(_SENS_Y, _SENS_P, (6,))
        edges = np.linspace(0.0, 1.0, 7)
        counts = [0] * 6
        for v in _SENS_P:
            counts[_right_closed_bin(float(v), edges)] += 1
        assert rows[0].min_support == min(c for c in counts if c), (
            f"min_support {rows[0].min_support}: the smallest OCCUPIED bin holds "
            f"{min(c for c in counts if c)}"
        )
        assert _raises(ValueError, lesson.ece_sensitivity, _SENS_Y, _SENS_P, ()), "() is ValueError"
        assert _raises(ValueError, lesson.ece_sensitivity, _SENS_Y, _SENS_P, (5, 0)), (
            "a bin count of 0 is ValueError"
        )


# --- exercise 4: chi2_sf() -------------------------------------------------------------------

# NIST/SEMATECH e-Handbook 1.3.6.7.4, upper-tail critical values: (dof, P(X < value), value).
_NIST_UPPER = ((1, 0.90, 2.706), (1, 0.95, 3.841), (1, 0.99, 6.635), (1, 0.999, 10.828),
               (2, 0.95, 5.991), (2, 0.999, 13.816), (5, 0.95, 11.070), (8, 0.95, 15.507),
               (8, 0.99, 20.090), (9, 0.95, 16.919), (10, 0.95, 18.307), (10, 0.999, 29.588),
               (20, 0.95, 31.410), (30, 0.95, 43.773), (100, 0.95, 124.342),
               (100, 0.999, 149.449))
# The same page's lower-tail table: (dof, P(X < value), value).
_NIST_LOWER = ((2, 0.05, 0.103), (4, 0.05, 0.711), (8, 0.05, 2.733), (10, 0.05, 3.940),
               (10, 0.10, 4.865))


def _brackets(dof, upper_tail, value):
    """The table rounds to three decimals, so the true critical value lies within 0.0005."""
    hi = lesson.chi2_sf(max(value - 0.0005, 0.0), dof)
    lo = lesson.chi2_sf(value + 0.0005, dof)
    return lo <= upper_tail <= hi, lo, hi


def test_chi2_nist_upper_table():
    for dof, below, value in _NIST_UPPER:
        ok, lo, hi = _brackets(dof, 1.0 - below, value)
        assert ok, (
            f"the table puts {1 - below:.3f} of a chi-square on {dof} dof above {value}; your "
            f"tail over [{value - 0.0005:.4f}, {value + 0.0005:.4f}] runs {hi:.6f} to {lo:.6f}"
        )


def test_chi2_nist_lower_table():
    for dof, below, value in _NIST_LOWER:
        ok, lo, hi = _brackets(dof, 1.0 - below, value)
        assert ok, (
            f"the lower-tail table puts {below:.2f} of a chi-square on {dof} dof below {value}, "
            f"so {1 - below:.2f} above it; your tail brackets {lo:.6f} to {hi:.6f}"
        )


def test_chi2_matches_incomplete_gamma():
    for dof in (1, 2, 3, 4, 7, 12, 25, 60):
        for x in (0.01, 0.5, 1.0, 3.3, 9.0, 20.0, 55.0, 130.0):
            got = lesson.chi2_sf(x, dof)
            want = _chi2_reference(x, dof)
            assert abs(got - want) <= 1e-10 + 1e-8 * want, (
                f"chi2_sf({x}, {dof}) = {got!r}, the incomplete gamma gives {want!r}"
            )


def test_chi2_large_dof():
    for dof, x in ((301, 290.0), (400, 400.0), (500, 560.0), (800, 700.0)):
        got = lesson.chi2_sf(x, dof)
        want = _chi2_reference(x, dof)
        assert abs(got - want) <= 1e-9 + 1e-7 * want, (
            f"chi2_sf({x}, {dof}) = {got!r}, expected {want!r} — build each term through "
            "logarithms; powers and gammas overflow here"
        )


def test_chi2_edges_and_type():
    assert lesson.chi2_sf(0.0, 4) == 1.0 and lesson.chi2_sf(-3.0, 1) == 1.0, "x <= 0 gives 1.0"
    assert lesson.chi2_sf(float("inf"), 3) == 0.0, "x = inf gives 0.0"
    got = lesson.chi2_sf(4.0, 5)
    assert type(got) is float, f"return a plain Python float, got {type(got).__name__}"
    tails = [lesson.chi2_sf(x, 6) for x in (1.0, 2.0, 5.0, 9.0, 15.0)]
    assert all(a > b for a, b in zip(tails, tails[1:])), "an upper tail falls as x grows"


def test_chi2_rejects_bad_dof():
    for bad in (0, -2, 2.5, True):
        assert _raises(ValueError, lesson.chi2_sf, 1.0, bad), f"dof={bad!r} must raise ValueError"
    assert _raises(ValueError, lesson.chi2_sf, float("nan"), 3), "x = nan must raise ValueError"


# --- exercise 5: hosmer_lemeshow() -----------------------------------------------------------

def _continuous_sample(seed, n):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(1.5, 7.0, n), 0.001, 0.999)
    y = (rng.random(n) < np.clip(p * 1.15, 0, 1)).astype(int)
    return y, p


def test_hl_statistic_hand_worked():
    with _ReferenceGroups():
        r = lesson.hosmer_lemeshow(np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0]),
                                   np.array([0.1] * 5 + [0.5] * 5), 2, fitted_here=False)
        assert abs(r.statistic - (0.25 / 0.45 + 0.25 / 1.25)) < 1e-12, (
            f"statistic {r.statistic:.6f}; each group adds (O - E)^2 / (E * (1 - mean prediction))"
        )
        y, p = _continuous_sample(21, 300)
        got = lesson.hosmer_lemeshow(y, p, 6, fitted_here=False)
        want = _hl_reference(y, p, _ef_reference(p, 6))
        assert abs(got.statistic - want) < 1e-9, (
            f"on 300 records in 6 groups the statistic is {want:.6f}, got {got.statistic:.6f}"
        )
        assert type(got.statistic) is float and type(got.p_value) is float, (
            "statistic and p_value must be plain floats"
        )


def test_hl_dof_independent_sample():
    with _ReferenceGroups():
        y, p = _continuous_sample(22, 500)
        r = lesson.hosmer_lemeshow(y, p, 10, fitted_here=False)
        assert r.groups == 10, f"500 untied scores form all 10 groups, got {r.groups}"
        assert r.dof == 10, (
            f"a model NOT fitted on this sample is judged on g = 10 dof, got {r.dof}. g - 2 is the "
            "fitting-sample rule: on a holdout it rejects a correct model far too often"
        )
        assert abs(r.p_value - _chi2_reference(r.statistic, 10)) < 1e-9, (
            "p_value is the chi-square tail at the statistic, on the dof you reported"
        )


def test_hl_dof_fitting_sample():
    with _ReferenceGroups():
        y, p = _continuous_sample(23, 500)
        r = lesson.hosmer_lemeshow(y, p, 10, fitted_here=True)
        assert r.dof == 8, f"on the fitting sample, 10 groups leave 10 - 2 = 8 dof, got {r.dof}"
        assert r.fitted_here is True, "record fitted_here on the result"
        assert abs(r.p_value - _chi2_reference(r.statistic, 8)) < 1e-9, "p_value on 8 dof"


def test_hl_dof_follows_groups_formed():
    with _ReferenceGroups():
        p = np.array([0.05] * 40 + [0.1] * 25 + [0.2] * 20 + [0.4] * 15)
        y = np.array(([0] * 38 + [1] * 2) + ([0] * 22 + [1] * 3) + ([0] * 15 + [1] * 5)
                     + ([0] * 9 + [1] * 6))
        indep = lesson.hosmer_lemeshow(y, p, 10, fitted_here=False)
        formed = _ef_reference(p, 10).size - 1
        assert indep.groups == formed, (
            f"ten groups requested over four distinct scores form {formed}; got {indep.groups}"
        )
        assert indep.dof == formed, (
            f"degrees of freedom follow the groups FORMED: {formed}, got {indep.dof}"
        )
        fitted = lesson.hosmer_lemeshow(y, p, 10, fitted_here=True)
        assert fitted.dof == formed - 2, f"fitting sample: {formed} - 2, got {fitted.dof}"


def test_hl_uses_equal_frequency_groups():
    with _ReferenceGroups():
        y, p = _continuous_sample(24, 240)
        r = lesson.hosmer_lemeshow(y, p, 8, fitted_here=False)
        edges = _ef_reference(p, 8)
        counts = [0] * (edges.size - 1)
        for v in p:
            counts[_right_closed_bin(float(v), edges)] += 1
        assert np.asarray(r.count).tolist() == counts, (
            f"group sizes {np.asarray(r.count).tolist()}, expected {counts}: the groups are "
            "exercise 1's equal-frequency bins, not equal-width ones"
        )
        assert abs(float(np.sum(r.expected)) - float(p.sum())) < 1e-9, (
            "expected per group is the sum of the predictions in it"
        )


def test_hl_signature_and_validation():
    with _ReferenceGroups():
        sig = inspect.signature(lesson.hosmer_lemeshow)
        param = sig.parameters.get("fitted_here")
        assert param is not None and param.default is inspect.Parameter.empty, (
            "fitted_here must have NO default: the caller has to say where the model was fitted"
        )
        assert _raises(ValueError, lesson.hosmer_lemeshow, np.array([0, 1, 0, 1]),
                       np.array([0.2, 0.2, 0.7, 0.7]), 2, fitted_here=True), (
            "two groups on the fitting sample leave no degrees of freedom: ValueError"
        )
        assert _raises(ValueError, lesson.hosmer_lemeshow, np.array([0, 1, 1, 1]),
                       np.array([0.2, 0.2, 1.0, 1.0]), 2, fitted_here=False), (
            "a group whose mean prediction is 1 has no variance: ValueError, not a division by "
            "zero"
        )
        assert _raises(ValueError, lesson.hosmer_lemeshow, np.array([0, 0, 0]),
                       np.array([0.0, 0.0, 0.0]), 3, fitted_here=False), (
            "a group whose mean prediction is 0 has no variance: ValueError"
        )


# --- exercise 6: characteristic_edges() and characteristic_stability() -----------------------

def test_char_edges_untied():
    rng = np.random.default_rng(31)
    v = rng.normal(size=203)
    for m in (2, 5, 10, 13):
        got = np.asarray(lesson.characteristic_edges(v, m), dtype=float)
        want = _char_edges_reference(v, m)
        assert np.array_equal(got, want), (
            f"203 untied values in {m} bins: got {np.round(got, 4).tolist()[:4]}..., expected "
            f"{np.round(want, 4).tolist()[:4]}... — each bin opens on the value at zero-based "
            "position ceil(k * n / n_bins)"
        )


def test_char_edges_binary_two_bins():
    for share in (0.25, 0.3, 0.5, 0.9):
        n1 = int(share * 200)
        v = np.array([0.0] * (200 - n1) + [1.0] * n1)
        got = np.asarray(lesson.characteristic_edges(v, 10), dtype=float)
        assert got.tolist() == [-np.inf, 1.0, np.inf], (
            f"a yes/no input with {share:.0%} ones gave {got.tolist()}; it needs exactly two "
            "bins, [-inf, 1, inf]. Module 1's quantile edges give one bin here and a CSI of zero"
        )


def test_char_edges_ties_no_empty_bin():
    rng = np.random.default_rng(32)
    v = np.minimum(rng.poisson(0.6, 700), 7).astype(float)
    got = np.asarray(lesson.characteristic_edges(v, 10), dtype=float)
    want = _char_edges_reference(v, 10)
    assert np.array_equal(got, want), (
        f"a tied count input gave {got.tolist()}, expected {want.tolist()}"
    )
    k = got.size - 1
    counts = [0] * k
    for x in v:
        for i in range(k):
            if got[i] <= x < got[i + 1]:
                counts[i] += 1
                break
    assert min(counts) > 0, f"bin counts {counts} include an empty bin on its own baseline"
    const = np.asarray(lesson.characteristic_edges(np.full(30, 4.0), 10), dtype=float)
    assert const.tolist() == [-np.inf, np.inf], "a constant input has one bin"


def test_char_edges_validation():
    assert _raises(ValueError, lesson.characteristic_edges, np.array([1.0, 2.0]), 0), "n_bins 0"
    assert _raises(ValueError, lesson.characteristic_edges, np.array([]), 5), "empty input"
    assert _raises(ValueError, lesson.characteristic_edges, np.array([1.0, np.inf]), 5), (
        "a non-finite value"
    )


def _stability_case():
    rng = np.random.default_rng(33)
    base = {"zeta": rng.normal(0.0, 1.0, 600), "alpha": (rng.random(600) < 0.3).astype(float),
            "mid": np.minimum(rng.poisson(1.0, 600), 5).astype(float)}
    cur = {"zeta": rng.normal(0.8, 1.0, 300), "alpha": (rng.random(300) < 0.6).astype(float),
           "mid": np.minimum(rng.poisson(1.0, 300), 5).astype(float)}
    coef = {"intercept": -2.0, "zeta": 0.1, "alpha": 0.05, "mid": 0.7}
    return base, cur, coef


def test_csi_cut_on_baseline():
    base, cur, coef = _stability_case()
    with _ReferenceEdges():
        rows = lesson.characteristic_stability(base, cur, coef)
    by_name = {r.driver: r for r in rows}
    for name in base:
        want = _psi_reference(base[name], cur[name], _char_edges_reference(base[name], 10))
        got = by_name[name].csi
        assert abs(got - want) < 1e-10, (
            f"CSI of {name}: got {got:.6f}, expected {want:.6f} — cut the edges once on the "
            "BASELINE values of that driver, and floor both sides as module 1 does"
        )
        assert by_name[name].bins == _char_edges_reference(base[name], 10).size - 1, (
            f"bins for {name} is the number of bins its edges define"
        )
    with _ReferenceEdges():
        coarse = {r.driver: r for r in lesson.characteristic_stability(base, cur, coef, n_bins=4)}
    for name in base:
        edges = _char_edges_reference(base[name], 4)
        want = _psi_reference(base[name], cur[name], edges)
        got = coarse[name]
        assert got.bins == edges.size - 1 and abs(got.csi - want) < 1e-10, (
            f"asked for n_bins=4, CSI of {name} came back {got.csi:.6f} over {got.bins} bins; "
            f"expected {want:.6f} over {edges.size - 1}. Pass n_bins on to characteristic_edges "
            "instead of leaving it at its default"
        )


def test_csi_log_odds_shift_and_identity():
    base, cur, coef = _stability_case()
    with _ReferenceEdges():
        rows = lesson.characteristic_stability(base, cur, coef)
    for r in rows:
        want = coef[r.driver] * (float(np.mean(cur[r.driver])) - float(np.mean(base[r.driver])))
        assert abs(r.log_odds_shift - want) < 1e-12, (
            f"{r.driver}: shift {r.log_odds_shift:+.6f}, expected {want:+.6f} — coefficient "
            "times (current mean - baseline mean)"
        )
    total = sum(r.log_odds_shift for r in rows)
    moved = sum(coef[k] * (np.mean(cur[k]) - np.mean(base[k])) for k in base)
    assert abs(total - moved) < 1e-12, "the shifts sum to the change in the mean log-odds"


def test_csi_order_fields_types():
    base, cur, coef = _stability_case()
    base = dict(base, twin_b=np.arange(50.0), twin_a=np.arange(50.0))
    cur = dict(cur, twin_b=np.arange(50.0), twin_a=np.arange(50.0))
    coef = dict(coef, twin_a=1.0, twin_b=1.0)
    with _ReferenceEdges():
        rows = lesson.characteristic_stability(base, cur, coef)
    csis = [r.csi for r in rows]
    assert csis == sorted(csis, reverse=True), "rows are ordered by CSI, largest first"
    names = [r.driver for r in rows]
    assert names.index("twin_a") < names.index("twin_b"), (
        "two drivers with equal CSI are ordered by name"
    )
    r = rows[0]
    assert type(r.csi) is float and type(r.log_odds_shift) is float and type(r.bins) is int, (
        "csi and log_odds_shift are plain floats, bins a plain int"
    )
    assert abs(float(np.sum(r.stability.contributions)) - r.csi) < 1e-12, (
        "stability is the full StabilityResult whose contributions sum to csi"
    )


def test_csi_rejects_mismatch():
    base, cur, coef = _stability_case()
    with _ReferenceEdges():
        short = {k: v for k, v in cur.items() if k != "mid"}
        assert _raises(ValueError, lesson.characteristic_stability, base, short, coef), (
            "a driver missing from the current sample must raise ValueError"
        )
        nocoef = {k: v for k, v in coef.items() if k != "zeta"}
        assert _raises(ValueError, lesson.characteristic_stability, base, cur, nocoef), (
            "a driver with no coefficient must raise ValueError"
        )


# --- exercise 7: psi_null_bootstrap() --------------------------------------------------------

def test_null_shape_and_reproducibility():
    pop = np.arange(800.0)
    with _ReferenceEdges():
        a = lesson.psi_null_bootstrap(pop, 300, 150, np.random.default_rng(41), n_boot=60)
        b = lesson.psi_null_bootstrap(pop, 300, 150, np.random.default_rng(41), n_boot=60)
        c = lesson.psi_null_bootstrap(pop, 300, 150, np.random.default_rng(42), n_boot=60)
    assert isinstance(a, np.ndarray) and a.shape == (60,), (
        f"expected shape (60,), got {np.shape(a)}"
    )
    assert np.all(np.isfinite(a)) and a.min() >= 0.0, "replicates are finite and non-negative"
    assert np.array_equal(a, b), "the same seed must give the same replicates"
    assert not np.array_equal(a, c), "a different seed must give different replicates"


def test_null_two_sample_scale():
    with _ReferenceEdges():
        z = lesson.psi_null_bootstrap(np.arange(1000.0), 400, 400, np.random.default_rng(43),
                                      n_boot=300)
    expect = 9 * (1 / 400 + 1 / 400)
    assert abs(z.mean() - expect) < 0.2 * expect, (
        f"mean null PSI {z.mean():.4f}; with 10 bins and 400 against 400 it should sit near "
        f"{expect:.4f}. About half that means the baseline was frozen, or drawn without "
        "replacement at full size; zero means one index draw served both samples"
    )
    with _ReferenceEdges():
        z4 = lesson.psi_null_bootstrap(np.arange(1000.0), 400, 400, np.random.default_rng(49),
                                       n_bins=4, n_boot=300)
    expect4 = 3 * (1 / 400 + 1 / 400)
    assert abs(z4.mean() - expect4) < 0.2 * expect4, (
        f"with n_bins=4 the mean null PSI was {z4.mean():.4f}; four bins put it near "
        f"{expect4:.4f}, because the chi-square has B - 1 degrees of freedom. About three times "
        "that means the edges were cut at the default ten bins: pass n_bins on to "
        "characteristic_edges"
    )


def test_null_current_size_and_edges():
    with _ReferenceEdges():
        z = lesson.psi_null_bootstrap(np.arange(5000.0), 2000, 20, np.random.default_rng(44),
                                      n_boot=200)
    assert z.mean() > 1.0, (
        f"twenty records against two thousand, mean null PSI {z.mean():.4f}. At this size most "
        "replicates leave bins empty and the floor makes PSI large; a small figure means the "
        "month was drawn at the baseline's size, or the edges were cut on the month"
    )
    assert np.all(np.isfinite(z)), "floored on both sides: every replicate is finite"


def test_null_edges_cut_on_the_baseline():
    with _ReferenceEdges():
        z = lesson.psi_null_bootstrap(np.arange(5000.0), 20, 2000, np.random.default_rng(48),
                                      n_boot=150)
    expect = 9 * (1 / 20 + 1 / 2000)
    assert 0.3 < z.mean() < 0.7, (
        f"a development sample of twenty against a month of two thousand gave mean null PSI "
        f"{z.mean():.4f}; cut on the twenty, it sits near {expect:.4f}. Several times that means "
        "the edges were cut on the month, or on the two samples pooled, which leaves the "
        "twenty-record baseline with empty bins; a figure near zero means the baseline was frozen"
    )

def test_null_binary_population():
    pop = np.array([0.0] * 700 + [1.0] * 300)
    with _ReferenceEdges():
        z = lesson.psi_null_bootstrap(pop, 400, 100, np.random.default_rng(45), n_boot=100)
    assert z.max() > 0.0 and abs(z.mean() - (1 / 400 + 1 / 100)) < 0.4 * (1 / 400 + 1 / 100), (
        f"on a yes/no population the mean null PSI was {z.mean():.5f}, expected near "
        f"{1 / 400 + 1 / 100:.5f}. All zeros means the edges collapsed to one bin: cut them "
        "with characteristic_edges"
    )


def test_null_validation():
    pop = np.arange(10.0)
    rng = np.random.default_rng(46)
    for args, kwargs, why in (((pop, 0, 5, rng), {}, "n_baseline of 0"),
                              ((pop, 5, 0, rng), {}, "n_current of 0"),
                              ((pop, 5, 5, rng), {"n_boot": 0}, "n_boot of 0"),
                              ((np.array([]), 5, 5, rng), {}, "an empty population"),
                              ((pop, 5, 5, None), {}, "no generator")):
        assert _raises(ValueError, lesson.psi_null_bootstrap, *args, **kwargs), (
            f"{why} must raise ValueError"
        )


# --- exercise 8: false_alarm_rate() and threshold_for_rate() ---------------------------------

def test_far_strict():
    null = np.array([0.05, 0.1, 0.1, 0.1, 0.2, 0.25, 0.3, 0.02])
    r = lesson.false_alarm_rate(null, 0.1)
    assert abs(r.rate - 3 / 8) < 1e-12, (
        f"three of eight lie strictly above 0.1: rate 0.375, got {r.rate}. A value exactly on "
        "the threshold is not an alarm — module 1's breach rule is strictly greater"
    )
    assert lesson.false_alarm_rate(null, 0.3).rate == 0.0, "nothing lies above the largest value"
    assert lesson.false_alarm_rate(null, 0.0).rate == 1.0, "everything lies above zero here"


def test_far_se_and_types():
    null = np.linspace(0.0, 1.0, 40)
    r = lesson.false_alarm_rate(null, 0.5)
    rate = float(np.sum(null > 0.5)) / 40
    assert abs(r.se - math.sqrt(rate * (1 - rate) / 40)) < 1e-12, "se = sqrt(rate(1-rate)/n)"
    assert r.n_boot == 40 and type(r.n_boot) is int, "n_boot is a plain int count"
    assert type(r.rate) is float and type(r.se) is float and r.threshold == 0.5, (
        "rate and se are plain floats; threshold is echoed back"
    )
    assert _raises(ValueError, lesson.false_alarm_rate, np.array([]), 0.1), "empty null"


def _threshold_brute(null, rate):
    n = len(null)
    for t in sorted(set(float(v) for v in null)):
        if sum(v > t for v in null) <= rate * n + 1e-9:
            return t
    raise AssertionError("unreachable")


def test_threshold_for_rate_brute_force():
    rng = np.random.default_rng(47)
    for trial in range(6):
        null = np.round(rng.gamma(2.0, 0.02, int(rng.integers(20, 400))), 3)
        for rate in (0.0, 0.01, 0.05, 0.1, 0.5):
            got = lesson.threshold_for_rate(null, rate)
            want = _threshold_brute(null.tolist(), rate)
            assert got == want, (
                f"{null.size} replicates at rate {rate}: threshold {got}, expected {want} — the "
                "smallest replicate value that at most rate x n replicates exceed"
            )
            assert type(got) is float, "return a plain float"


def test_threshold_float_tolerance():
    null = np.arange(100.0)
    got = lesson.threshold_for_rate(null, 0.29)
    assert got == 70.0, (
        f"at 29% of 100 replicates, 29 may lie above: threshold 70.0, got {got}. 0.29 * 100 is "
        "28.999999999999996 in floating point — floor it with a small tolerance"
    )


def test_threshold_validation():
    for rate in (1.0, -0.1, 1.5):
        assert _raises(ValueError, lesson.threshold_for_rate, np.array([0.1, 0.2]), rate), (
            f"rate {rate} must raise ValueError"
        )
    assert _raises(ValueError, lesson.threshold_for_rate, np.array([]), 0.05), "empty null"


# --- exercise 9: render_appendix() -----------------------------------------------------------

def _findings(p_value=0.0612, fitted_here=False, psi=0.2468, two_months=True):
    rows = (lesson.SensitivityRow("equal-width", 4, 4, 2, 13, 0.0123),
            lesson.SensitivityRow("equal-width", 40, 40, 9, 3, 0.0456),
            lesson.SensitivityRow("equal-frequency", 40, 11, 11, 17, 0.0789))
    stab = lesson.StabilityResult(psi=0.3131, contributions=np.array([0.3131]),
                                  expected_pct=np.array([1.0]), actual_pct=np.array([1.0]))
    s1 = lesson.CharacteristicShift("gamma_driver", 0.3131, 3, -0.0421, stab)
    s2 = lesson.CharacteristicShift("beta_driver", 0.1, 2, 0.5, stab)
    hl = lesson.HosmerLemeshow(statistic=17.1717, dof=6 if fitted_here else 8, p_value=p_value,
                               groups=8, fitted_here=fitted_here, count=np.array([1]),
                               observed=np.array([0.0]), expected=np.array([0.5]))
    months = [{"name": "first month", "n_current": 432, "score_psi": psi, "score_bins": 7,
               "score_far": {0.15: 0.0913, 0.3: 0.0047}, "shifts": (s1, s2),
               "csi_far": {"gamma_driver": {0.15: 0.1234, 0.3: 0.0101},
                           "beta_driver": {0.15: 0.2222, 0.3: 0.0333}}}]
    if two_months:
        months.append({"name": "second month", "n_current": 99, "score_psi": 0.3, "score_bins": 5,
                       "score_far": {0.15: 0.4, 0.3: 0.2}, "shifts": (s2,),
                       "csi_far": {"beta_driver": {0.15: 0.5, 0.3: 0.25}}})
    return {"model_id": "rubric-model-9", "data_note": "RUBRIC DATA NOTE", "n_calibration": 2468,
            "n_baseline": 13579, "ece_rows": rows, "ece_reported": rows[2],
            "ece_ceiling": 0.05, "ece_far": 0.0321, "hl": hl, "hl_requested": 12,
            "hl_alpha": 0.0612, "hl_far": 0.0654, "thresholds": (0.15, 0.3), "months": months}


def test_appendix_title_and_sections():
    text = lesson.render_appendix(_findings())
    assert text.splitlines()[0] == "# Calibration and stability appendix — rubric-model-9", (
        f"first line {text.splitlines()[0]!r}"
    )
    pos = [text.find(h) for h in lesson.APPENDIX_SECTIONS]
    assert all(p >= 0 for p in pos) and pos == sorted(pos), "the four headings, in order"


def test_appendix_data_and_table():
    text = lesson.render_appendix(_findings())
    assert "RUBRIC DATA NOTE" in text, "the data note is printed in A.1"
    assert "Calibration sample: 2468 records. Stability baseline: 13579 records." in text, (
        "A.1 states both sample sizes in the documented sentence"
    )
    assert "| scheme | bins requested | bins used | smallest bin | ECE |" in text, (
        "A.2's table opens with the header '| scheme | bins requested | bins used | smallest bin "
        "| ECE |', column names exactly as the docstring gives them"
    )
    for row in ("| equal-width | 4 | 4 | 13 | 0.0123 |", "| equal-width | 40 | 40 | 3 | 0.0456 |",
                "| equal-frequency | 40 | 11 | 17 | 0.0789 |"):
        assert row in text, f"sensitivity row missing or misformatted: {row!r}"
    assert "ECE across these 3 schemes runs from 0.0123 to 0.0789." in text, (
        "the range sentence reads the smallest and largest ECE from the rows"
    )


def test_appendix_ece_line():
    text = lesson.render_appendix(_findings())
    want = ("- reported ECE 0.0789 (equal-frequency, 40 bins requested, 11 used) against 0.05: "
            "BREACH (false-alarm rate 3.2% at this size)")
    assert want in text, f"the reported-ECE threshold line should read {want!r}"


def test_appendix_hl_line():
    text = lesson.render_appendix(_findings())
    assert ("Hosmer-Lemeshow statistic 17.1717 over 8 groups (12 requested), 8 degrees of "
            "freedom (independent of the fit), p-value 0.0612.") in text, (
        "the HL sentence states groups formed and requested, the dof and its basis"
    )
    assert "- p-value 0.0612 against 0.06: NOT REJECTED (false-alarm rate 6.5% at this size)" \
        in text, "a p-value exactly equal to alpha is NOT REJECTED: reject only when p < alpha"
    fitted = lesson.render_appendix(_findings(p_value=0.01, fitted_here=True))
    assert "6 degrees of freedom (fitted on this sample)" in fitted, "the basis follows fitted_here"
    assert "- p-value 0.0100 against 0.06: REJECTED" in fitted, "p < alpha is REJECTED"


def test_appendix_every_threshold_priced():
    f = _findings()
    text = lesson.render_appendix(f)
    n_thresholds = 2 + sum((1 + len(m["shifts"])) * len(f["thresholds"]) for m in f["months"])
    assert text.count("(false-alarm rate ") == n_thresholds, (
        f"{text.count('(false-alarm rate ')} lines carry a false-alarm rate; these findings "
        f"hold {n_thresholds} thresholds, and every one must"
    )
    bullets = [line for line in text.splitlines() if line.startswith("- ")]
    assert len(bullets) == n_thresholds and all(b.endswith("at this size)") for b in bullets), (
        "every threshold is one '- ' line ending '(false-alarm rate X% at this size)'"
    )
    assert "(false-alarm rate 12.3% at this size)" in text and \
           "(false-alarm rate 0.5% at this size)" in text, (
        "each rate comes from its own entry in score_far / csi_far, at one decimal of a percent"
    )


def test_appendix_breach_boundary():
    text = lesson.render_appendix(_findings())
    assert ("- CSI beta_driver 0.1000 over 2 bins, log-odds shift +0.5000, against 0.15: WITHIN "
            "(false-alarm rate 22.2% at this size)") in text, (
        "the CSI line format, with a WITHIN verdict below the threshold"
    )
    assert "- score PSI 0.3000 over 5 bins against 0.30: WITHIN (false-alarm rate 20.0% at this " \
           "size)" in text, "exactly on the threshold is WITHIN: BREACH means strictly above"
    assert ("- CSI gamma_driver 0.3131 over 3 bins, log-odds shift -0.0421, against 0.30: BREACH "
            "(false-alarm rate 1.0% at this size)") in text, "strictly above is BREACH"


def test_appendix_months_and_order():
    text = lesson.render_appendix(_findings())
    a = text.find("### first month: 432 records against 13579")
    b = text.find("### second month: 99 records against 13579")
    assert 0 <= a < b, "one sub-heading per month, in the order given"
    first = text[a:b]
    assert first.find("gamma_driver") < first.find("beta_driver"), (
        "drivers are printed in the order the shifts are given, not re-sorted"
    )
    assert first.find("score PSI 0.2468") < first.find("CSI gamma_driver"), (
        "each month prints its score lines before its driver lines"
    )


def test_appendix_missing_keys():
    f = _findings()
    broken = {k: v for k, v in f.items() if k not in ("hl_far", "ece_rows")}
    try:
        lesson.render_appendix(broken)
    except ValueError as exc:
        assert "hl_far" in str(exc) and "ece_rows" in str(exc), (
            f"name every missing key in one ValueError, got: {exc}"
        )
    else:
        raise AssertionError("incomplete findings must raise ValueError")
    f2 = _findings(two_months=False)
    f2["months"][0] = {k: v for k, v in f2["months"][0].items() if k != "csi_far"}
    assert _raises(ValueError, lesson.render_appendix, f2), (
        "a month missing csi_far must raise ValueError, not a KeyError halfway through"
    )


RUBRIC = [
    (test_ef_untied_rank_rule, 4,
     "the k-th edge is the score of the record at 1-based rank ceil(k * n / n_bins)"),
    (test_ef_ties_merge_not_split, 5,
     "merge coincident edges; never split a tie and never interpolate between grades"),
    (test_ef_never_empty_or_split, 3,
     "edges that sit on observed scores, merged, leave no bin empty"),
    (test_ef_extremes, 2,
     "swap the last candidate for 1.0, put 0.0 in front, then de-duplicate once more"),
    (test_ef_rejects_bad_input, 2,
     "validate n_bins, emptiness and the probability range before sorting"),
    (test_ece_edges_matches_module1, 4,
     "equal-width edges must reproduce module 1's ECE exactly"),
    (test_ece_edges_right_closed_membership, 3,
     "np.searchsorted(edges, p, side='left') - 1, clipped into range"),
    (test_ece_edges_arbitrary_edges, 3,
     "sum |events - predictions| over occupied bins, divided by the number of records"),
    (test_ece_edges_type_and_perfect, 2, "return float(...)"),
    (test_ece_edges_rejects_bad_input, 2,
     "edges strictly increasing from 0.0 to 1.0; labels 0/1; probabilities in [0, 1]"),
    (test_sensitivity_order_and_types, 3,
     "loop SCHEMES outside, bin_counts inside; cast the integer fields with int()"),
    (test_sensitivity_values, 4, "each row's ece is ece_on_edges over that row's edges"),
    (test_sensitivity_bins_used_and_occupied, 3,
     "bins_used = len(edges) - 1; occupied counts bins with at least one record"),
    (test_sensitivity_min_support_and_validation, 2,
     "min_support is the smallest non-zero count; reject empty or sub-1 bin counts"),
    (test_chi2_nist_upper_table, 4,
     "Q(x;1) = erfc(sqrt(x/2)), Q(x;2) = exp(-x/2), then add a term per two dof"),
    (test_chi2_nist_lower_table, 2, "an upper tail near 1 needs the same recurrence, no shortcut"),
    (test_chi2_matches_incomplete_gamma, 4,
     "each term is (x/2)^e * exp(-x/2) / gamma(e + 1), e starting at 1/2 or 1"),
    (test_chi2_large_dof, 3, "form each term as exp(e*log(x/2) - x/2 - lgamma(e + 1))"),
    (test_chi2_edges_and_type, 2, "x <= 0 -> 1.0, inf -> 0.0, and return a Python float"),
    (test_chi2_rejects_bad_dof, 1, "dof must be a whole number >= 1, and not a bool"),
    (test_hl_statistic_hand_worked, 4, "sum (O - E)^2 / (E * (1 - E / n)) over the groups"),
    (test_hl_dof_independent_sample, 4,
     "dof = groups when the model was fitted elsewhere; g - 2 only on the fitting sample"),
    (test_hl_dof_fitting_sample, 3, "dof = groups - 2 when fitted_here is True"),
    (test_hl_dof_follows_groups_formed, 3,
     "count the groups equal_frequency_edges actually produced, not n_groups"),
    (test_hl_uses_equal_frequency_groups, 2,
     "group with equal_frequency_edges and module 1's right-closed membership"),
    (test_hl_signature_and_validation, 2,
     "`*, fitted_here: bool` with no default; refuse dof < 1 and a group mean of 0 or 1"),
    (test_char_edges_untied, 3,
     "value at zero-based position min(ceil(k * n / n_bins), n - 1), k = 1 .. n_bins - 1"),
    (test_char_edges_binary_two_bins, 3, "drop candidates not strictly above the minimum"),
    (test_char_edges_ties_no_empty_bin, 3,
     "de-duplicate the candidates, drop the minimum, then add -inf and +inf"),
    (test_char_edges_validation, 1, "reject n_bins < 1, an empty input and non-finite values"),
    (test_csi_cut_on_baseline, 3,
     "population_stability_index(base, cur, characteristic_edges(base, n_bins))"),
    (test_csi_log_odds_shift_and_identity, 3,
     "coefficient * (mean(current) - mean(baseline)) for each driver"),
    (test_csi_order_fields_types, 2, "sorted(rows, key=lambda r: (-r.csi, r.driver))"),
    (test_csi_rejects_mismatch, 1, "compare the driver sets and the coefficient keys first"),
    (test_null_shape_and_reproducibility, 3, "take every draw from the rng argument"),
    (test_null_two_sample_scale, 4,
     "draw a fresh baseline AND a fresh month, with replacement, in every replicate, and cut "
     "the edges at the n_bins you were given"),
    (test_null_current_size_and_edges, 3,
     "the month has n_current records, and the edges are cut on the replicate's baseline"),
    (test_null_edges_cut_on_the_baseline, 3,
     "cut the edges on the replicate's baseline alone — not on the month, not on both pooled"),
    (test_null_binary_population, 2, "cut the replicate's edges with characteristic_edges"),
    (test_null_validation, 1, "reject sizes below 1, an empty population and a missing rng"),
    (test_far_strict, 3, "np.count_nonzero(null > threshold) / len(null)"),
    (test_far_se_and_types, 2, "se = math.sqrt(rate * (1 - rate) / n)"),
    (test_threshold_for_rate_brute_force, 3,
     "sorted null at index n - 1 - floor(rate * n + 1e-9)"),
    (test_threshold_float_tolerance, 1, "math.floor(rate * n + 1e-9), not math.floor(rate * n)"),
    (test_threshold_validation, 1, "rate must lie in [0, 1); the null must be non-empty"),
    (test_appendix_title_and_sections, 2, "title line, then APPENDIX_SECTIONS in order"),
    (test_appendix_data_and_table, 3, "one table row per SensitivityRow, ECE to 4 places"),
    (test_appendix_ece_line, 2, "the reported-ECE line, with its verdict and ece_far"),
    (test_appendix_hl_line, 3, "REJECTED only when p_value < hl_alpha"),
    (test_appendix_every_threshold_priced, 4,
     "every '- ' line ends with the false-alarm rate from findings, formatted {:.1%}"),
    (test_appendix_breach_boundary, 2, "BREACH when value > threshold, WITHIN otherwise"),
    (test_appendix_months_and_order, 2, "one '### name: n records against n_baseline' per month"),
    (test_appendix_missing_keys, 2, "collect every missing key, top level and per month, first"),
]
