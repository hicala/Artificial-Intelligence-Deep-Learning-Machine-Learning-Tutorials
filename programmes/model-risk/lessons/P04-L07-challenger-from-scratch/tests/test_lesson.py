"""Autograder rubric for P04-L07. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson, and are deliberate:

* every expected value is computed HERE, by an independent route: closed forms counted from a
  2x2 table, a pure-Python score vector, a Newton fit solved by least squares on the
  square-root-weighted system rather than on the normal equations, a pure-Python weight of
  evidence, a pooling reference that merges the LEFTMOST violating pair and starts again
  (the student's may use a stack — both reach the same unique answer), and a dictionary-based
  comparability audit. No expected figure is lifted from the notebook's synthetic book; the
  one test that uses the lesson's own frames derives its expected counts from those frames.
* the exercises chain — the scorecard needs the weights, the comparison needs the audit — so
  each exercise is graded as far as possible in isolation. Where a student function calls an
  earlier one (woe_table and enforce_monotonic call woe_from_counts, challenger_comparison
  calls comparability_audit), the test swaps in this file's reference for the earlier
  function while it runs, so a mistake in exercise 4 or 6 does not cost marks in 5 or 7.
* `test_irls_never_inverts` replaces np.linalg.inv and np.linalg.pinv with functions that
  fail while a fit runs. Solving the system is the requirement; this is how it is graded.
* `test_irls_quasi_complete_separation` is what a fitter with a deviance-based stopping rule
  fails: on a zero-cell table the deviance settles while one coefficient gains about one unit
  an update, so such a fitter reports "converged" with a coefficient of about twenty.
* `test_comparison_refuses_non_comparable` is the centre of the rubric: a challenger that
  would be promoted on its numbers, planted with each of the three asymmetries in turn. A
  comparison that crowns it — or that computes and returns its metrics anyway — fails.
"""
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


# --- independent references -----------------------------------------------------------------

def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ref_fit(X, y, iters=80):
    """Newton by least squares on the square-root-weighted system: a different route."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b
        p = _sig(eta)
        w = p * (1.0 - p)
        sw = np.sqrt(w)
        z = eta + (y - p) / w
        new = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)[0]
        done = np.max(np.abs(new - b)) < 1e-14
        b = new
        if done:
            break
    return b


def _ref_score(X, y, b):
    """X'(y - p) in pure Python."""
    rows = [[float(v) for v in r] for r in np.asarray(X)]
    out = [0.0] * len(rows[0])
    for r, label in zip(rows, y):
        eta = sum(a * c for a, c in zip(r, b))
        resid = float(label) - 1.0 / (1.0 + math.exp(-eta))
        for j, v in enumerate(r):
            out[j] += v * resid
    return out


def _ref_woe(events, non_events, a=0.5):
    occ = [e + g > 0 for e, g in zip(events, non_events)]
    m = sum(occ)
    big_e, big_g = sum(events), sum(non_events)
    out = []
    for e, g, o in zip(events, non_events, occ):
        if not o:
            out.append(float("nan"))
        else:
            out.append(math.log(((g + a) / (big_g + a * m)) / ((e + a) / (big_e + a * m))))
    return out


def _ref_pool(events, non_events, direction):
    """Merge the LEFTMOST violating adjacent pair, then start again from the left."""
    blocks = [[e, g, [i]] for i, (e, g) in enumerate(zip(events, non_events))]
    sign = 1.0 if direction == "increasing" else -1.0
    while True:
        for i in range(len(blocks) - 1):
            r1 = blocks[i][0] / (blocks[i][0] + blocks[i][1])
            r2 = blocks[i + 1][0] / (blocks[i + 1][0] + blocks[i + 1][1])
            if sign * (r2 - r1) < 0:
                a, b = blocks[i], blocks[i + 1]
                blocks[i:i + 2] = [[a[0] + b[0], a[1] + b[1], a[2] + b[2]]]
                break
        else:
            return blocks


def _ref_audit(reference, other):
    """A dictionary-based audit, used to isolate exercise 7 from exercise 6."""
    def index(frame):
        ids = [int(v) for v in frame.record_id]
        return ids, {rid: i for i, rid in reversed(list(enumerate(ids)))}
    r_ids, r_at = index(reference)
    o_ids, o_at = index(other)
    common = sorted(set(r_at) & set(o_at))
    rec = (len(r_ids) == len(r_at) and len(o_ids) == len(o_at) and set(r_at) == set(o_at)
           and len(common) > 0
           and all(int(reference.y[r_at[k]]) == int(other.y[o_at[k]]) for k in common))
    per = len(common) > 0 and all(str(reference.period[r_at[k]]) == str(other.period[o_at[k]])
                                  for k in common)
    mis = len(common) > 0
    for name, col in reference.inputs.items():
        if name not in other.inputs:
            mis = False
            continue
        for k in common:
            if math.isnan(float(col[r_at[k]])) != math.isnan(float(other.inputs[name][o_at[k]])):
                mis = False
    checks = (lesson.Check("records", rec, "reference audit"),
              lesson.Check("period", per, "reference audit"),
              lesson.Check("missing values", mis, "reference audit"))
    return lesson.Audit(all(c.held for c in checks), checks)


class _Swap:
    """Temporarily replace one of the lesson's functions with this file's reference."""

    def __init__(self, name, replacement):
        self.name, self.replacement = name, replacement

    def __enter__(self):
        self.saved = getattr(lesson, self.name)
        setattr(lesson, self.name, self.replacement)

    def __exit__(self, *exc):
        setattr(lesson, self.name, self.saved)
        return False


def _ref_woe_array(events, non_events, smoothing=0.5):
    return np.array(_ref_woe([float(v) for v in events], [float(v) for v in non_events],
                             smoothing))


def _dm(*cols):
    """A DesignMatrix built here, so exercise 2 is graded without exercise 1."""
    n = len(cols[0]) if cols else 0
    X = np.column_stack([np.ones(n)] + [np.asarray(c, dtype=float) for c in cols])
    names = ("intercept",) + tuple(f"x{i + 1}" for i in range(len(cols)))
    return lesson.DesignMatrix(X, names)


def _has_number(text, k):
    """True if the integer k appears in text as a number of its own, not inside another."""
    return re.search(rf"(?<![\d.]){k}(?![\d.])", text) is not None


def _frame(ids, period, inputs, y, score, model="f"):
    return lesson.EvalFrame(model, np.asarray(ids), np.asarray(period),
                            {k: np.asarray(v, dtype=float) for k, v in inputs.items()},
                            np.asarray(y), np.asarray(score, dtype=float))


def _raises_value_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


# --- exercise 1: design_matrix ----------------------------------------------------------------

def test_design_intercept_first():
    d = lesson.design_matrix({"b": np.array([3, 1, 2]), "a": np.array([0.5, 0.25, 0.75])})
    assert isinstance(d, lesson.DesignMatrix), "return a DesignMatrix(X=..., names=...)"
    assert tuple(d.names) == ("intercept", "b", "a"), (
        f"names {d.names}: intercept first, then the columns in the mapping's order (not sorted)")
    assert d.X.dtype == np.float64 and d.X.shape == (3, 3), f"X {d.X.shape} {d.X.dtype}"
    assert np.array_equal(d.X[:, 0], np.ones(3)), "column 0 must be all ones"
    assert np.array_equal(d.X[:, 1:], [[3, 0.5], [1, 0.25], [2, 0.75]]), "columns misplaced"


def test_design_rejects_missing_naming_column():
    for bad in (np.nan, np.inf, -np.inf):
        col = np.array([1.0, bad, 3.0])
        try:
            lesson.design_matrix({"ok": np.array([1.0, 2.0, 4.0]), "income": col})
        except ValueError as exc:
            assert "income" in str(exc), f"the error must name the column 'income': {exc}"
        else:
            raise AssertionError(f"a column holding {bad} must raise ValueError")


def test_design_rejects_constant_column():
    try:
        lesson.design_matrix({"x": np.array([1.0, 2.0, 3.0]), "flag": np.array([1, 1, 1])})
    except ValueError as exc:
        assert "flag" in str(exc), f"the error must name the constant column 'flag': {exc}"
    else:
        raise AssertionError("a constant column duplicates the intercept and must raise")
    ok = lesson.design_matrix({"flag": np.array([0, 0, 1])})
    assert ok.X.shape == (3, 2), "a column that varies, however little, is allowed"


def test_design_rejects_bad_shapes():
    cases = (({"a": np.array([1.0, 2.0]), "b": np.array([1.0, 2.0, 3.0])}, "lengths differ"),
             ({"a": np.array([[1.0, 2.0], [3.0, 4.0]])}, "a 2-D column"),
             ({}, "no columns at all"),
             ({"intercept": np.array([1.0, 2.0])}, "a column called 'intercept'"))
    for bad, why in cases:
        assert _raises_value_error(lesson.design_matrix, bad), f"{why} must raise ValueError"


# --- exercise 2: fit_logistic_irls ------------------------------------------------------------

def test_irls_intercept_only_closed_form():
    y = np.array([0] * 13 + [1] * 7)
    fit = lesson.fit_logistic_irls(lesson.DesignMatrix(np.ones((20, 1)), ("intercept",)), y)
    assert fit.status == "converged" and fit.coef is not None, f"status {fit.status!r}"
    assert abs(float(fit.coef[0]) - math.log(7 / 13)) < 1e-9, (
        f"intercept-only coef should be ln(7/13) = {math.log(7 / 13):.8f}, got {fit.coef[0]}")
    assert np.asarray(fit.path).shape == (fit.n_iter, 1), (
        f"path shape {np.asarray(fit.path).shape}, expected ({fit.n_iter}, 1)")
    assert np.allclose(np.asarray(fit.path)[-1], fit.coef), "path's last row must be coef"
    assert tuple(fit.names) == ("intercept",), "names come from the design matrix"


def test_irls_matches_two_by_two():
    # n00 = 40, n01 = 10, n10 = 15, n11 = 35 -> intercept ln(10/40), slope ln(35*40/(15*10))
    x = np.array([0] * 50 + [1] * 50)
    y = np.array([0] * 40 + [1] * 10 + [0] * 15 + [1] * 35)
    fit = lesson.fit_logistic_irls(_dm(x), y)
    assert fit.status == "converged", f"status {fit.status!r}: {fit.finding}"
    want = (math.log(10 / 40), math.log(35 * 40 / (15 * 10)))
    assert np.allclose(fit.coef, want, atol=1e-8, rtol=0), (
        f"coef {np.round(fit.coef, 8)} should reproduce the table: intercept {want[0]:.8f}, "
        f"slope {want[1]:.8f}")


def test_irls_score_equations_zero():
    rng = np.random.default_rng(31)
    n = 500
    a, b = rng.normal(0, 1, n), rng.uniform(0, 3, n)
    y = (rng.random(n) < _sig(-0.7 + 0.9 * a - 0.5 * b)).astype(np.int64)
    d = _dm(a, b)
    fit = lesson.fit_logistic_irls(d, y)
    assert fit.status == "converged", f"status {fit.status!r}: {fit.finding}"
    grad = _ref_score(d.X, y, [float(v) for v in fit.coef])
    assert max(abs(g) for g in grad) < 1e-7, (
        f"at the fitted coefficients the score X'(y - p) should vanish, got "
        f"{[f'{g:.2e}' for g in grad]} — the fit stopped short of the maximum")
    assert np.allclose(fit.coef, _ref_fit(d.X, y), atol=1e-7, rtol=0), (
        "coefficients disagree with an independent Newton fit")


def test_irls_never_inverts():
    rng = np.random.default_rng(32)
    a = rng.normal(0, 1, 300)
    y = (rng.random(300) < _sig(0.3 + a)).astype(np.int64)
    d = _dm(a)

    def refuse(*args, **kwargs):
        raise AssertionError("np.linalg.inv/pinv was called — solve the weighted normal "
                             "equations with np.linalg.solve instead of inverting X'WX")
    saved = (np.linalg.inv, np.linalg.pinv)
    np.linalg.inv, np.linalg.pinv = refuse, refuse
    try:
        fit = lesson.fit_logistic_irls(d, y)
    finally:
        np.linalg.inv, np.linalg.pinv = saved
    assert fit.status == "converged", f"status {fit.status!r}"


def test_irls_complete_separation():
    rng = np.random.default_rng(33)
    x = np.concatenate([rng.uniform(-3, -0.2, 30), rng.uniform(0.2, 3, 30)])
    noise = rng.normal(0, 1, 60)
    y = np.array([0] * 30 + [1] * 30)
    fit = lesson.fit_logistic_irls(_dm(noise, x), y)
    assert fit.status == "separation", (
        f"x1 separates the classes; status {fit.status!r} — after each update, test whether "
        "the new linear predictor puts every event above every non-event")
    assert fit.coef is None, "no finite estimate exists: coef must be None, not the last iterate"
    assert fit.n_iter < lesson.MAX_ITER, (
        f"complete separation was only declared after {fit.n_iter} updates, the whole budget — "
        "rule 1 is certain the moment the linear predictor orders the classes, so stop there "
        "rather than leaning on the quasi-complete fallback")
    assert fit.finding.startswith("SEPARATION"), f"finding must open 'SEPARATION': {fit.finding!r}"
    assert "x2" in fit.finding and "x1" not in fit.finding, (
        f"the finding must name the column that runs away (x2), not the noise: {fit.finding!r}")


def test_irls_quasi_complete_separation():
    # x = 1 -> always an event (a zero cell); x = 0 -> both classes. The slope has no finite MLE.
    rng = np.random.default_rng(34)
    x = np.array([1] * 25 + [0] * 75)
    y = np.array([1] * 25 + [0] * 50 + [1] * 25)
    noise = rng.normal(0, 1, 100)
    fit = lesson.fit_logistic_irls(_dm(noise, x), y)
    assert fit.status == "separation", (
        f"a zero cell is quasi-complete separation; status {fit.status!r}. A deviance-based "
        "stopping rule calls this converged while the slope is still running away")
    assert fit.coef is None, "coef must be None under separation"
    assert fit.n_iter == lesson.MAX_ITER, (
        f"quasi-complete separation is only declared once the iteration budget is spent; "
        f"n_iter {fit.n_iter} vs MAX_ITER {lesson.MAX_ITER}")
    assert "x2" in fit.finding, f"the finding must name x2 as the runaway: {fit.finding!r}"


def test_irls_runaway_named_by_standardised_coefficient():
    rng = np.random.default_rng(35)
    big = np.concatenate([rng.uniform(-3000, -500, 40), rng.uniform(500, 3000, 40)])
    tiny = rng.normal(0, 0.002, 80)
    y = np.array([0] * 40 + [1] * 40)
    fit = lesson.fit_logistic_irls(_dm(tiny, big), y)
    assert fit.status == "separation", f"status {fit.status!r}"
    assert "x2" in fit.finding and "x1" not in fit.finding, (
        f"x2 separates the classes but is measured in thousands, so its raw coefficient is "
        f"small; rank |b_j| * std(X[:, j]), not |b_j|. Finding: {fit.finding!r}")


def test_irls_not_converged():
    rng = np.random.default_rng(36)
    a = rng.normal(0, 1, 200)
    y = (rng.random(200) < _sig(-0.5 + 1.2 * a)).astype(np.int64)
    fit = lesson.fit_logistic_irls(_dm(a), y, max_iter=2)
    assert fit.status == "not converged", f"two updates cannot meet 1e-8; got {fit.status!r}"
    assert fit.coef is None, "a fit that did not converge has no estimate to report"
    assert fit.n_iter == 2 and np.asarray(fit.path).shape == (2, 2), (
        f"n_iter {fit.n_iter}, path {np.asarray(fit.path).shape}: one row per update performed")
    assert fit.finding.startswith("NOT CONVERGED"), f"finding: {fit.finding!r}"
    # A strong predictor on classes that still overlap (the full fit converges): after two
    # updates some records are predicted with probability above 0.999, but none within
    # SEPARATION_EPS of certainty. Confident is not certain.
    s = rng.normal(0, 2.5, 200)
    ys = (rng.random(200) < _sig(-0.5 + 2.0 * s)).astype(np.int64)
    strong = lesson.fit_logistic_irls(_dm(s), ys, max_iter=2)
    assert strong.status == "not converged" and strong.coef is None, (
        f"a strong but overlapping predictor stopped after two updates is 'not converged'; got "
        f"{strong.status!r}. Rule 3 needs a probability of the record's own outcome of at least "
        "1 - SEPARATION_EPS, not merely a confident one")


def test_irls_converged_finding_and_path():
    rng = np.random.default_rng(37)
    a = rng.normal(0, 1, 400)
    y = (rng.random(400) < _sig(-1.0 + 0.8 * a)).astype(np.int64)
    fit = lesson.fit_logistic_irls(_dm(a), y)
    assert fit.status == "converged" and fit.n_iter <= 15, (
        f"a well-posed fit converges in a handful of updates; {fit.status!r} after {fit.n_iter}")
    assert fit.finding.startswith("CONVERGED"), f"finding: {fit.finding!r}"
    path = np.asarray(fit.path)
    assert path.shape == (fit.n_iter, 2) and np.array_equal(path[-1], fit.coef), (
        "path holds the coefficients after each update and ends at coef")
    assert np.max(np.abs(path[-1] - path[-2])) <= lesson.TOL, (
        "the last update moved a coefficient by more than TOL, yet the fit says converged")


def test_irls_honours_tol():
    rng = np.random.default_rng(38)
    a = rng.normal(0, 1, 300)
    y = (rng.random(300) < _sig(-0.4 + 0.9 * a)).astype(np.int64)
    fit = lesson.fit_logistic_irls(_dm(a), y, tol=1e-3)
    assert fit.status == "converged", f"status {fit.status!r}: {fit.finding}"
    path = np.vstack([np.zeros(2), np.asarray(fit.path)])
    steps = np.max(np.abs(np.diff(path, axis=0)), axis=1)
    assert steps[-1] <= 1e-3 and np.all(steps[:-1] > 1e-3), (
        f"with tol=1e-3 the fit must stop at the FIRST update that moves no coefficient by more "
        f"than 1e-3; the largest step of each update was {[f'{s:.1e}' for s in steps]} — "
        "compare against the `tol` argument, not the module's TOL")


def test_irls_rejects_bad_labels():
    d = _dm(np.array([0.1, 0.5, 0.9, 0.3]))
    for y, why in ((np.array([0, 1, 2, 0]), "a label of 2"),
                   (np.array([1, 1, 1, 1]), "a single class"),
                   (np.array([0, 1, 1]), "a label vector shorter than X")):
        assert _raises_value_error(lesson.fit_logistic_irls, d, y), f"{why} must raise ValueError"


# --- exercise 3: score_equations and closed_form_2x2 -----------------------------------------

def test_score_equations_hand_worked():
    d = lesson.DesignMatrix(np.array([[1.0, 0.0], [1.0, 1.0]]), ("intercept", "x"))
    s = np.asarray(lesson.score_equations(d, np.array([0, 1]), np.array([0.0, 0.0])))
    assert s.shape == (2,) and np.allclose(s, [0.0, 0.5]), f"got {s}, expected [0.0, 0.5]"


def test_score_equations_match_reference():
    rng = np.random.default_rng(41)
    X = np.column_stack([np.ones(30), rng.normal(0, 1, 30), rng.uniform(0, 2, 30)])
    y = rng.integers(0, 2, 30)
    b = np.array([0.3, -0.7, 0.4])
    s = np.asarray(lesson.score_equations(lesson.DesignMatrix(X, ("intercept", "a", "b")), y, b))
    assert np.allclose(s, _ref_score(X, y, b), atol=1e-12), (
        f"X'(y - p) disagrees with the reference: {s} vs {np.round(_ref_score(X, y, b), 6)} "
        "— is the sign (y - p), and p the sigmoid of X @ coef?")


def test_score_equations_validate():
    d = lesson.DesignMatrix(np.ones((3, 2)), ("intercept", "x"))
    assert _raises_value_error(lesson.score_equations, d, np.array([0, 1, 0]), np.zeros(3)), (
        "a coef with the wrong number of entries must raise ValueError")
    assert _raises_value_error(lesson.score_equations, d, np.array([0, 1]), np.zeros(2)), (
        "a y with the wrong number of entries must raise ValueError")


def test_closed_form_hand_worked():
    x = np.array([0] * 7 + [1] * 9)
    y = np.array([0] * 5 + [1] * 2 + [0] * 3 + [1] * 6)       # n00 5, n01 2, n10 3, n11 6
    t = lesson.closed_form_2x2(x, y)
    assert t is not None, "no cell is empty, so the closed form exists"
    assert abs(t.intercept - math.log(2 / 5)) < 1e-12, (
        f"intercept should be ln(n01/n00) = {math.log(2 / 5):.6f}, got {t.intercept}")
    assert abs(t.slope - math.log(6 * 5 / (3 * 2))) < 1e-12, (
        f"slope should be the log odds ratio {math.log(5):.6f}, got {t.slope}")
    assert type(t.intercept) is float and type(t.slope) is float, "return plain Python floats"


def test_closed_form_zero_cell_is_none():
    base_x = np.array([0, 0, 0, 1, 1, 1])
    for y, cell in ((np.array([1, 1, 1, 0, 1, 1]), "n00"), (np.array([0, 0, 0, 0, 1, 1]), "n01"),
                    (np.array([0, 1, 1, 1, 1, 1]), "n10"), (np.array([0, 1, 1, 0, 0, 0]), "n11")):
        got = lesson.closed_form_2x2(base_x, y)
        assert got is None, (
            f"with {cell} = 0 there is no finite slope: return None, not a corrected or infinite "
            f"number — got {got}")
    # n00 2, n01 1, n10 1, n11 2: a cell holding ONE record is thin, not empty.
    thin = lesson.closed_form_2x2(base_x, np.array([0, 0, 1, 0, 1, 1]))
    assert thin is not None and abs(thin.slope - math.log(4)) < 1e-12, (
        f"a cell holding one record is not an empty cell: the slope is finite, ln 4 here — "
        f"return None only when a count is zero. Got {thin}")


def test_closed_form_validation():
    for x, y, why in ((np.array([0, 1, 2]), np.array([0, 1, 1]), "an x of 2"),
                      (np.array([1, 1, 1]), np.array([0, 1, 1]), "a constant x"),
                      (np.array([0, 1, 1]), np.array([0, 1]), "mismatched lengths"),
                      (np.array([0, 1, 1]), np.array([1, 1, 1]), "a single class of y")):
        assert _raises_value_error(lesson.closed_form_2x2, x, y), f"{why} must raise ValueError"


# --- exercise 4: woe_from_counts and woe_table ------------------------------------------------

def test_woe_from_counts_matches_reference():
    got = np.asarray(lesson.woe_from_counts(np.array([0, 1, 2]), np.array([2, 1, 0])))
    assert np.allclose(got, [math.log(5), 0.0, -math.log(5)]), f"worked example: got {got}"
    ev, ne = np.array([4, 11, 30, 2]), np.array([80, 120, 95, 40])   # unequal class totals
    got = np.asarray(lesson.woe_from_counts(ev, ne))
    want = _ref_woe_array(ev, ne)
    assert np.allclose(got, want, atol=1e-12), (
        f"got {np.round(got, 5)}, expected {np.round(want, 5)} — each share is "
        "(count + a) / (class total + a * m), with a*m in the denominator too")
    got = np.asarray(lesson.woe_from_counts(ev, ne, smoothing=0.25))
    assert np.allclose(got, _ref_woe_array(ev, ne, 0.25), atol=1e-12), "honour `smoothing`"


def test_woe_empty_bin_nan_not_counted():
    ev, ne = np.array([3, 0, 9, 1]), np.array([30, 0, 20, 12])
    got = np.asarray(lesson.woe_from_counts(ev, ne))
    want = _ref_woe_array(ev, ne)
    assert np.isnan(got[1]), f"an empty bin reports nan, got {got[1]}"
    assert np.allclose(got[[0, 2, 3]], want[[0, 2, 3]], atol=1e-12), (
        f"got {np.round(got, 5)}, expected {np.round(want, 5)} — an empty bin is not counted "
        "in m, the number of occupied bins")


def test_woe_from_counts_validation():
    for args, why in (((np.array([1, -1]), np.array([2, 2])), "a negative count"),
                      ((np.array([0, 0]), np.array([2, 2])), "no events at all"),
                      ((np.array([1, 2]), np.array([2])), "mismatched lengths")):
        assert _raises_value_error(lesson.woe_from_counts, *args), f"{why} must raise ValueError"
    assert _raises_value_error(lesson.woe_from_counts, np.array([1, 2]), np.array([2, 1]),
                               smoothing=0.0), "smoothing of 0 must raise ValueError"


def test_woe_table_layout_and_missing_bin():
    x = np.array([0.5, 1.5, 2.5, 3.5, np.nan, 0.2, np.nan, 3.9])
    y = np.array([0, 0, 1, 1, 1, 0, 0, 1])
    edges = np.array([-np.inf, 1.0, 3.0, np.inf])
    with _Swap("woe_from_counts", _ref_woe_array):
        t = lesson.woe_table(x, y, edges)
        no_missing = lesson.woe_table(np.array([0.5, 2.0, 3.5, 0.1]), np.array([0, 1, 1, 0]),
                                      edges)
    assert t.events.tolist() == [0, 1, 2, 1] and t.non_events.tolist() == [2, 1, 0, 1], (
        f"events {t.events.tolist()}, non_events {t.non_events.tolist()} — three value bins, "
        "then the missing bin last")
    assert np.allclose(t.woe, _ref_woe_array(t.events, t.non_events), equal_nan=True), (
        "woe must come from woe_from_counts on the table's counts")
    assert len(no_missing.events) == 4 and no_missing.events[-1] == 0 \
        and no_missing.non_events[-1] == 0 and np.isnan(no_missing.woe[-1]), (
            "the missing bin is always present — counts 0 and woe nan when nothing is missing")


def test_woe_table_edge_convention():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([1.0, 2.0, -5.0, 9.0, 0.0, 3.0])
    y = np.array([1, 1, 0, 1, 0, 0])
    with _Swap("woe_from_counts", _ref_woe_array):
        t = lesson.woe_table(x, y, edges)
    counts = (t.events + t.non_events).tolist()
    assert counts == [2, 1, 3, 0], (
        f"counts per bin {counts}: 1.0 and 2.0 sit on edges and go UP ([lo, hi)); -5 falls "
        "into the first bin and 9 and 3.0 into the last — out-of-range values are kept")


def test_woe_table_validation():
    x, y = np.array([0.1, 0.2]), np.array([0, 1])
    for edges, why in ((np.array([0.5]), "fewer than two edges"),
                       (np.array([0.0, 1.0, 1.0]), "a repeated edge"),
                       (np.array([1.0, 0.0]), "decreasing edges")):
        assert _raises_value_error(lesson.woe_table, x, y, edges), f"{why} must raise ValueError"
    assert _raises_value_error(lesson.woe_table, x, np.array([0, 3]), np.array([0.0, 1.0])), (
        "a label of 3 must raise ValueError")
    with _Swap("woe_from_counts", _ref_woe_array):
        one_bin = lesson.woe_table(np.array([0.1, 0.2, 0.4, 0.3]), np.array([0, 1, 0, 1]),
                                   np.array([-np.inf, np.inf]))
    assert one_bin.events.tolist() == [2, 0] and one_bin.non_events.tolist() == [2, 0], (
        f"two edges are enough — one value bin, then the missing bin — got events "
        f"{one_bin.events.tolist()}, non_events {one_bin.non_events.tolist()}")


def test_woe_table_honours_smoothing():
    x = np.array([0.5, 1.5, 2.5, 3.5, np.nan, 0.2, np.nan, 3.9])
    y = np.array([0, 0, 1, 1, 1, 0, 0, 1])
    with _Swap("woe_from_counts", _ref_woe_array):
        t = lesson.woe_table(x, y, np.array([-np.inf, 1.0, 3.0, np.inf]), smoothing=0.25)
    want = _ref_woe_array(t.events, t.non_events, 0.25)
    assert np.allclose(t.woe, want, equal_nan=True), (
        f"woe {np.round(t.woe, 5)} should be {np.round(want, 5)} with smoothing=0.25 — pass "
        "the table's `smoothing` on to woe_from_counts")


# --- exercise 5: enforce_monotonic ------------------------------------------------------------

def _table(edges, ev, ne):
    ev, ne = np.asarray(ev), np.asarray(ne)
    return lesson.WoeTable(np.asarray(edges, dtype=float), ev, ne, _ref_woe_array(ev, ne))


def test_monotonic_pools_repeatedly():
    # rates 0.05, 0.25, 0.30, 0.09: pooling the last two gives 0.20, which now falls below
    # the 0.25 BEFORE it — so the merged block must be compared backwards and pooled again.
    t = _table([-np.inf, 1, 2, 3, np.inf], [5, 20, 33, 9, 4], [95, 60, 77, 91, 6])
    with _Swap("woe_from_counts", _ref_woe_array):
        out = lesson.enforce_monotonic(t, "increasing")
    assert out.events.tolist() == [5, 62, 4] and out.non_events.tolist() == [95, 228, 6], (
        f"events {out.events.tolist()}: pooling 0.30 with 0.09 gives 0.20, which is now below "
        "the 0.25 before it — a merged block must be checked against its neighbour again")


def test_monotonic_matches_reference_on_random_tables():
    rng = np.random.default_rng(51)
    for trial in range(25):
        k = int(rng.integers(3, 9))
        ev = rng.integers(1, 40, k)
        ne = rng.integers(1, 60, k)
        direction = "increasing" if trial % 2 == 0 else "decreasing"
        edges = np.concatenate(([-np.inf], np.arange(1.0, k), [np.inf]))
        t = _table(edges, np.append(ev, 3), np.append(ne, 7))
        with _Swap("woe_from_counts", _ref_woe_array):
            out = lesson.enforce_monotonic(t, direction)
        blocks = _ref_pool(ev.tolist(), ne.tolist(), direction)
        assert out.events[:-1].tolist() == [b[0] for b in blocks] \
            and out.non_events[:-1].tolist() == [b[1] for b in blocks], (
                f"{direction} pooling of events {ev.tolist()} / non-events {ne.tolist()} should "
                f"give events {[b[0] for b in blocks]}, got {out.events[:-1].tolist()}")
        want_edges = [-np.inf] + [edges[b[2][-1] + 1] for b in blocks]
        assert np.array_equal(out.edges, want_edges), (
            f"edges {out.edges.tolist()} should be {want_edges}: keep the outer edges and the "
            "edges between pooled blocks only")
    # A bin in which every record is an event is occupied, with a rate of exactly 1 (and no
    # finite odds): rates 0.09, 1.00, 0.17, 0.38 pool the middle two to 0.33.
    pure = _table([-np.inf, 1, 2, 3, np.inf], [4, 9, 6, 12, 2], [40, 0, 30, 20, 2])
    with _Swap("woe_from_counts", _ref_woe_array):
        out = lesson.enforce_monotonic(pure, "increasing")
    assert out.events.tolist() == [4, 15, 12, 2] and out.non_events.tolist() == [40, 30, 20, 2], (
        f"a bin whose records are all events has rate 1.0 and must be pooled like any other; "
        f"got events {out.events.tolist()}, non_events {out.non_events.tolist()} — compare "
        "event RATES, events / (events + non-events)")


def test_monotonic_missing_bin_untouched_and_woe_recomputed():
    t = _table([-np.inf, 1, 2, np.inf], [30, 10, 20, 9], [70, 90, 80, 11])
    with _Swap("woe_from_counts", _ref_woe_array):
        out = lesson.enforce_monotonic(t, "decreasing")
    assert out.events[-1] == 9 and out.non_events[-1] == 11, (
        "the missing bin is not on the axis: it is never pooled and keeps its counts")
    want = _ref_woe_array(out.events, out.non_events)
    assert np.allclose(out.woe, want, atol=1e-12), (
        f"woe {np.round(out.woe, 5)} should be {np.round(want, 5)} — recompute EVERY bin, the "
        "missing bin included: pooling changed the number of occupied bins")


def test_monotonic_honours_smoothing():
    t = _table([-np.inf, 1, 2, np.inf], [30, 10, 20, 9], [70, 90, 80, 11])
    with _Swap("woe_from_counts", _ref_woe_array):
        out = lesson.enforce_monotonic(t, "decreasing", smoothing=0.25)
    want = _ref_woe_array(out.events, out.non_events, 0.25)
    assert np.allclose(out.woe, want, atol=1e-12), (
        f"woe {np.round(out.woe, 5)} should be {np.round(want, 5)} with smoothing=0.25 — "
        "recompute the weights with the `smoothing` you were given")


def test_monotonic_ties_are_not_violations():
    t = _table([-np.inf, 1, 2, np.inf], [10, 20, 20, 0], [90, 80, 80, 0])
    with _Swap("woe_from_counts", _ref_woe_array):
        out = lesson.enforce_monotonic(t, "increasing")
    assert out.events.tolist() == [10, 20, 20, 0] and np.array_equal(out.edges, t.edges), (
        f"rates 0.10, 0.20, 0.20 never fall, so nothing is pooled: equal rates are not a "
        f"violation. Got events {out.events.tolist()}")


def test_monotonic_validation():
    t = _table([-np.inf, 1, np.inf], [5, 6, 1], [5, 4, 1])
    assert _raises_value_error(lesson.enforce_monotonic, t, "upwards"), (
        "a direction other than 'increasing' or 'decreasing' must raise ValueError")
    empty = _table([-np.inf, 1, 2, np.inf], [5, 0, 6, 1], [5, 0, 4, 1])
    with _Swap("woe_from_counts", _ref_woe_array):
        assert _raises_value_error(lesson.enforce_monotonic, empty, "increasing"), (
            "an empty value bin has no rate: raise ValueError rather than invent one")


# --- exercise 6: comparability_audit ----------------------------------------------------------

def _base(n=40, seed=61):
    rng = np.random.default_rng(seed)
    ids = np.arange(1001, 1001 + n)
    per = np.array([f"2025-0{1 + i % 6}" for i in range(n)])
    inc = np.where(rng.random(n) < 0.3, np.nan, rng.uniform(1e4, 9e4, n).round())
    util = rng.random(n)
    y = (rng.random(n) < 0.3).astype(np.int64)
    score = rng.random(n)
    return ids, per, {"income": inc, "utilisation": util}, y, score


def _names(audit):
    return [c.name for c in audit.checks]


def test_audit_clean_shuffled_pair():
    ids, per, inputs, y, score = _base()
    ref = _frame(ids, per, inputs, y, score, "ref")
    order = np.random.default_rng(62).permutation(ids.size)
    other = _frame(ids[order], per[order], {k: v[order] for k, v in inputs.items()}, y[order],
                   score[order] * 0.9, "other")
    a = lesson.comparability_audit(ref, other)
    assert _names(a) == ["records", "period", "missing values"], (
        f"checks {_names(a)}: records, period, missing values, in that order")
    assert a.comparable and all(c.held for c in a.checks), (
        "the same records in another order, with gaps in the same places, are comparable: "
        + " | ".join(f"{c.name}: {c.evidence}" for c in a.checks if not c.held)
        + " — match by id, and compare missingness masks (NaN != NaN)")
    assert _has_number(a.checks[0].evidence, 40), (
        f"the records evidence must state how many records were compared (40): "
        f"{a.checks[0].evidence!r}")


def test_audit_records_dropped_and_extra():
    ids, per, inputs, y, score = _base()
    ref = _frame(ids, per, inputs, y, score)
    keep = np.arange(ids.size) >= 7                           # 7 dropped from the other
    ids2 = np.append(ids[keep], [9001, 9002, 9003])           # and 3 it has that ref lacks
    extra = {k: np.append(v[keep], [1.0, 2.0, 3.0]) for k, v in inputs.items()}
    other = _frame(ids2, np.append(per[keep], ["2025-01"] * 3), extra,
                   np.append(y[keep], [0, 1, 0]), np.append(score[keep], [0.5] * 3))
    a = lesson.comparability_audit(ref, other)
    assert not a.checks[0].held and a.checks[1].held and a.checks[2].held, (
        "records differ but the shared records agree on period and gaps: only 'records' fails; "
        f"held = {[c.held for c in a.checks]}")
    ev = a.checks[0].evidence
    assert _has_number(ev, 7) and _has_number(ev, 3) and _has_number(ev, 33), (
        f"the evidence must count the 7 ids only in the reference, the 3 only in the other "
        f"and the 33 in common: {ev!r}")


def test_audit_duplicate_id_same_length():
    ids, per, inputs, y, score = _base()
    dup = ids.copy()
    dup[-1] = dup[0]                                          # same length, one id twice
    other = _frame(dup, per, inputs, np.append(y[:-1], y[0]), score)
    a = lesson.comparability_audit(_frame(ids, per, inputs, y, score), other)
    assert not a.checks[0].held and not a.comparable, (
        "the same number of records is not the same records: one id appears twice and another "
        "is missing — compare the sets of ids and count duplicates")
    # Every id present, and record 1001 delivered twice (a join that fanned out): the SETS of
    # ids agree, so only the duplicate count can catch it.
    extra = np.append(np.arange(ids.size), 0)
    fanned = _frame(ids[extra], per[extra], {k: v[extra] for k, v in inputs.items()}, y[extra],
                    score[extra])
    b = lesson.comparability_audit(_frame(ids, per, inputs, y, score), fanned)
    assert not b.checks[0].held and not b.comparable, (
        "the other frame holds every id, and one of them twice: 'records' needs each id "
        "exactly once in each frame, so equal sets of ids are not enough — count duplicates")


def test_audit_outcome_disagreement():
    ids, per, inputs, y, score = _base()
    flipped = y.copy()
    flipped[[4, 9, 21, 30, 35]] = 1 - flipped[[4, 9, 21, 30, 35]]
    a = lesson.comparability_audit(_frame(ids, per, inputs, y, score),
                                   _frame(ids, per, inputs, flipped, score))
    assert not a.checks[0].held, "five outcomes disagree: the records are not the same records"
    assert _has_number(a.checks[0].evidence, 5), (
        f"the evidence must count the 5 outcomes that disagree: {a.checks[0].evidence!r}")


def test_audit_period_record_by_record():
    ids, per, inputs, y, score = _base()
    swapped = per.copy()
    swapped[[0, 1]] = swapped[[1, 0]]                          # same range, same multiset
    swapped[[10, 13]] = swapped[[13, 10]]
    swapped[[20, 27]] = swapped[[27, 20]]
    a = lesson.comparability_audit(_frame(ids, per, inputs, y, score),
                                   _frame(ids, swapped, inputs, y, score))
    assert a.checks[0].held and not a.checks[1].held and a.checks[2].held, (
        "six records carry a different month although the date RANGE is unchanged: 'period' "
        f"must fail record by record; held = {[c.held for c in a.checks]}")
    assert _has_number(a.checks[1].evidence, 6), (
        f"the period evidence must count the 6 records whose period differs: "
        f"{a.checks[1].evidence!r}")


def test_audit_missing_filled():
    ids, per, inputs, y, score = _base()
    gaps = np.isnan(inputs["income"])
    filled = dict(inputs, income=np.where(gaps, 31415.0, inputs["income"]))
    a = lesson.comparability_audit(_frame(ids, per, inputs, y, score),
                                   _frame(ids, per, filled, y, score))
    m = a.checks[2]
    assert a.checks[0].held and a.checks[1].held and not m.held, (
        f"the other frame filled every missing income: only 'missing values' fails; held = "
        f"{[c.held for c in a.checks]}")
    assert _has_number(m.evidence, int(gaps.sum())) and "31415" in m.evidence, (
        f"the evidence must count the {int(gaps.sum())} filled records and show the value "
        f"they were filled with (31415): {m.evidence!r}")
    back = lesson.comparability_audit(_frame(ids, per, filled, y, score),
                                      _frame(ids, per, inputs, y, score))
    assert not back.checks[2].held, (
        "the same asymmetry seen from the other side — gaps in the other frame where the "
        "reference has values — must fail too")


def test_audit_input_absent():
    ids, per, inputs, y, score = _base()
    a = lesson.comparability_audit(_frame(ids, per, inputs, y, score),
                                   _frame(ids, per, {"utilisation": inputs["utilisation"]}, y,
                                          score))
    assert not a.checks[2].held, (
        "the other frame does not carry 'income' at all, so its missing values cannot be "
        "shown to match: the check fails")


def test_audit_no_common_records():
    ids, per, inputs, y, score = _base()
    a = lesson.comparability_audit(_frame(ids, per, inputs, y, score),
                                   _frame(ids + 50000, per, inputs, y, score))
    assert not a.comparable and not any(c.held for c in a.checks), (
        f"no shared records: nothing held — got {[c.held for c in a.checks]}")


def test_audit_finds_the_planted_asymmetry():
    ref, vendor = lesson.CHAMPION_EXTRACT, lesson.VENDOR_FRAME
    a = lesson.comparability_audit(ref, vendor)
    assert a.checks[0].held and a.checks[1].held, (
        "the vendor file holds the same records and periods as the first-line extract, only "
        "sorted differently — match by id")
    assert not a.checks[2].held and not a.comparable, (
        "the first-line extract filled every missing income before scoring; the vendor was "
        "given the gaps. 'missing values' must fail")
    n_gaps = int(np.isnan(vendor.inputs["income"]).sum())
    fill = f"{lesson.CHAMPION_INCOME_FILL:g}"
    assert _has_number(a.checks[2].evidence, n_gaps) and fill in a.checks[2].evidence, (
        f"the evidence must count the {n_gaps} records that differ and show the value the "
        f"extract holds there ({fill}): {a.checks[2].evidence!r}")


# --- exercise 7: challenger_comparison --------------------------------------------------------

def _book(n, seed, champ_noise=1.2, champ_shift=0.0, chal_noise=0.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)
    y = (rng.random(n) < _sig(-1.0 + 1.5 * z)).astype(np.int64)
    ids = np.arange(5001, 5001 + n)
    per = np.array(["2025-02"] * n)
    inc = np.where(rng.random(n) < 0.15, np.nan, 1.0)
    champ = _sig(-1.0 + champ_shift + 1.5 * (z + rng.normal(0, champ_noise, n)))
    chal = _sig(-1.0 + 1.5 * (z + rng.normal(0, chal_noise, n))) if chal_noise else \
        _sig(-1.0 + 1.5 * z)
    return (_frame(ids, per, {"income": inc}, y, champ, "champion"),
            _frame(ids, per, {"income": inc}, y, chal, "challenger"))


def _compare(champ, chals, **kw):
    kw.setdefault("n_boot", 200)
    with _Swap("comparability_audit", _ref_audit):
        return lesson.challenger_comparison(champ, chals, lesson.POLICY, **kw)


def test_comparison_refuses_non_comparable():
    champ, good = _book(800, 71)
    n = good.record_id.size
    later = good._replace(model="scored later", period=np.array(["2025-05"] * n))
    filled = good._replace(model="filled", inputs={"income": np.ones(n)})
    fewer = good._replace(model="fewer records", record_id=good.record_id[:-40],
                          period=good.period[:-40], inputs={"income": good.inputs["income"][:-40]},
                          y=good.y[:-40], score=good.score[:-40])
    rows = _compare(champ, [good, later, filled, fewer])
    assert len(rows) == 4, f"one row per challenger, got {len(rows)}"
    assert rows[0].verdict == "promote", (
        f"control: the comparable challenger promotes, got {rows[0].verdict!r}")
    for row in rows[1:]:
        assert row.verdict == "not comparable", (
            f"{row.model!r} would promote on its numbers but is not comparable; verdict "
            f"{row.verdict!r} crowns it")
        assert row.decision is None and row.paired is None and row.challenger_metrics is None \
            and row.champion_metrics is None, (
                f"{row.model!r}: compute NOTHING on non-comparable data — metrics, decision "
                "and paired difference must all be None")


def test_comparison_promotes_clear_winner():
    champ, good = _book(800, 72)
    row = _compare(champ, [good])[0]
    assert row.verdict == "promote", f"a clear, calibrated winner promotes; got {row.verdict!r}"
    assert row.paired is not None and row.paired.diff > 0 and row.paired.lo > 0, (
        "the paired difference is challenger MINUS champion, so a better challenger is positive")


def test_comparison_holds_when_interval_includes_zero():
    # A small book: the challenger clears the policy's margin, but the paired interval cannot
    # tell its gain from zero.
    for seed in range(200, 400):
        champ, chal = _book(90, seed, champ_noise=0.9, champ_shift=0.9, chal_noise=0.75)
        y = champ.y
        m_c = {"auc": lesson.auc_by_ranks(y, champ.score),
               "ece": lesson.expected_calibration_error(y, champ.score)}
        m_x = {"auc": lesson.auc_by_ranks(y, chal.score),
               "ece": lesson.expected_calibration_error(y, chal.score)}
        policy = dict(lesson.POLICY, max_ece=1.0)
        if lesson.challenger_decision(m_c, m_x, policy).verdict != "promote":
            continue
        pd = lesson.paired_bootstrap_difference(y, chal.score, champ.score, 200, lesson.ALPHA,
                                                rng=np.random.default_rng(lesson.SEED))
        if pd.lo <= 0.0:
            break
    else:
        raise RuntimeError("rubric construction failed: no suitable book found")
    with _Swap("comparability_audit", _ref_audit):
        row = lesson.challenger_comparison(champ, [chal], policy, n_boot=200)[0]
    assert row.decision is not None and row.decision.verdict == "promote", (
        "module 1's rule promotes this challenger on its point estimates")
    assert row.verdict == "hold", (
        f"the paired interval [{pd.lo:+.4f}, {pd.hi:+.4f}] includes zero, so the gain is not "
        f"yet distinguishable from noise: 'hold', got {row.verdict!r}")


def test_comparison_holds_when_interval_touches_zero():
    # On a twenty-record book the AUC moves in coarse steps, so many replicate differences are
    # exactly zero and the interval's lower end can BE zero. An interval that starts at zero
    # has not left it: the rule is paired.lo > 0, strictly.
    policy = dict(lesson.POLICY, max_ece=1.0)
    for seed in range(400):
        champ, chal = _book(20, seed, champ_noise=1.2, chal_noise=0.3)
        y = champ.y
        if not 2 <= int(y.sum()) <= y.size - 2:
            continue
        m_c = {"auc": lesson.auc_by_ranks(y, champ.score),
               "ece": lesson.expected_calibration_error(y, champ.score)}
        m_x = {"auc": lesson.auc_by_ranks(y, chal.score),
               "ece": lesson.expected_calibration_error(y, chal.score)}
        if lesson.challenger_decision(m_c, m_x, policy).verdict != "promote":
            continue
        pd = lesson.paired_bootstrap_difference(y, chal.score, champ.score, 200, lesson.ALPHA,
                                                rng=np.random.default_rng(lesson.SEED))
        if pd.lo == 0.0:
            break
    else:
        raise RuntimeError("rubric construction failed: no book with a lower end of exactly 0")
    with _Swap("comparability_audit", _ref_audit):
        row = lesson.challenger_comparison(champ, [chal], policy, n_boot=200)[0]
    assert row.verdict == "hold", (
        f"the paired interval [{pd.lo:+.4f}, {pd.hi:+.4f}] starts at exactly zero, so it has not "
        f"left zero: 'hold', got {row.verdict!r} — promotion needs paired.lo > 0, strictly")


def test_comparison_rejects_on_gate():
    champ, good = _book(800, 73)
    wild = good._replace(model="over-confident", score=_sig(4.0 * np.log(good.score /
                                                                      (1 - good.score))))
    row = _compare(champ, [wild])[0]
    assert row.verdict == "reject" and row.decision is not None \
        and row.decision.verdict == "reject", (
            f"an ECE far above the ceiling fails module 1's absolute gate: 'reject', got "
            f"{row.verdict!r}")


def test_comparison_matches_carried_instruments():
    champ, chal = _book(500, 74, chal_noise=0.6)
    order = np.random.default_rng(75).permutation(chal.record_id.size)
    shuffled = chal._replace(record_id=chal.record_id[order], period=chal.period[order],
                             inputs={"income": chal.inputs["income"][order]},
                             y=chal.y[order], score=chal.score[order])
    row = _compare(champ, [shuffled], n_boot=150, alpha=0.1, seed=4242)[0]
    y = champ.y
    want_m = {"auc": lesson.auc_by_ranks(y, chal.score),
              "ece": lesson.expected_calibration_error(y, chal.score)}
    want_c = {"auc": lesson.auc_by_ranks(y, champ.score),
              "ece": lesson.expected_calibration_error(y, champ.score)}
    for got, want, who in ((row.challenger_metrics, want_m, "challenger"),
                           (row.champion_metrics, want_c, "champion")):
        assert got is not None and all(abs(got[k] - want[k]) < 1e-12 for k in ("auc", "ece")), (
            f"{who} metrics {got} should be {want}: auc_by_ranks and "
            "expected_calibration_error on the shared records")
    want_d = lesson.challenger_decision(want_c, want_m, lesson.POLICY)
    assert row.decision is not None and row.decision.verdict == want_d.verdict \
        and tuple(row.decision.reasons) == tuple(want_d.reasons), (
            "decision must be module 1's challenger_decision(champion_metrics, "
            "challenger_metrics, policy)")
    want_p = lesson.paired_bootstrap_difference(y, chal.score, champ.score, 150, 0.1,
                                                rng=np.random.default_rng(4242))
    got_p = row.paired
    assert got_p is not None and all(abs(getattr(got_p, f) - getattr(want_p, f)) < 1e-12
                                     for f in ("diff", "lo", "hi", "se_paired")), (
        f"paired {got_p} should equal paired_bootstrap_difference(champion.y, challenger "
        "scores in the champion's order, champion.score, n_boot, alpha, "
        "rng=np.random.default_rng(seed)) — challenger first, n_boot, alpha and seed passed "
        "through")


def test_comparison_aligns_by_id():
    champ, chal = _book(600, 76, chal_noise=0.5)
    back = np.arange(chal.record_id.size)[::-1]
    rev = chal._replace(record_id=chal.record_id[back], period=chal.period[back],
                        inputs={"income": chal.inputs["income"][back]}, y=chal.y[back],
                        score=chal.score[back])
    a, b = _compare(champ, [chal])[0], _compare(champ, [rev])[0]
    assert b.paired is not None and abs(a.paired.diff - b.paired.diff) < 1e-12 \
        and abs(a.paired.lo - b.paired.lo) < 1e-12, (
            f"the same challenger in reverse record order gave a different result "
            f"({a.paired.diff:+.4f} vs {b.paired.diff:+.4f}) — pair scores BY ID, in the "
            "champion's order, never by position")


def test_comparison_order_invariant():
    champ, one = _book(500, 77, chal_noise=0.4)
    _, two = _book(500, 77, chal_noise=0.9)
    two = two._replace(model="second")
    forward = _compare(champ, [one, two])
    reverse = _compare(champ, [two, one])
    assert [r.model for r in reverse] == ["second", "challenger"], "rows follow the order given"
    for a, b in ((forward[0], reverse[1]), (forward[1], reverse[0])):
        assert abs(a.paired.lo - b.paired.lo) < 1e-12 and abs(a.paired.hi - b.paired.hi) < 1e-12, (
            f"{a.model!r}'s interval changed when the challengers were listed in another order "
            "— seed a fresh generator for every challenger")


RUBRIC = [
    (test_design_intercept_first, 4,
     "np.column_stack([ones, *columns]) with the names ('intercept', *mapping order)"),
    (test_design_rejects_missing_naming_column, 4,
     "np.isfinite on each column; put the column's name in the ValueError"),
    (test_design_rejects_constant_column, 3,
     "a column equal to its own first value everywhere duplicates the intercept"),
    (test_design_rejects_bad_shapes, 3,
     "check lengths, dimensionality, an empty mapping and a column named 'intercept'"),
    (test_irls_intercept_only_closed_form, 4,
     "start from zeros; step = solve(X'WX, X'(y - p)); record every iterate in path"),
    (test_irls_matches_two_by_two, 5,
     "iterate until no coefficient moves by more than tol; the MLE reproduces the 2x2 table"),
    (test_irls_score_equations_zero, 4,
     "converge fully: at the MLE, X'(y - p) is zero in every entry"),
    (test_irls_never_inverts, 4,
     "np.linalg.solve(XtWX, rhs), never np.linalg.inv(XtWX) @ rhs"),
    (test_irls_complete_separation, 5,
     "after each update: if min(eta[events]) > max(eta[non-events]), status 'separation'"),
    (test_irls_quasi_complete_separation, 5,
     "when max_iter is spent, look for a probability of the record's own outcome >= 1 - eps"),
    (test_irls_runaway_named_by_standardised_coefficient, 3,
     "rank the non-intercept columns by |b_j| * X[:, j].std(), not by |b_j|"),
    (test_irls_not_converged, 3,
     "running out of updates without separation is 'not converged', with coef None"),
    (test_irls_converged_finding_and_path, 2,
     "finding opens 'CONVERGED'; path's last row is coef"),
    (test_irls_honours_tol, 2,
     "the step test compares against the `tol` argument, so a looser tol stops sooner"),
    (test_irls_rejects_bad_labels, 2,
     "labels in {0, 1}, both classes present, one per row of X"),
    (test_score_equations_hand_worked, 2, "X.T @ (y - sigmoid(X @ coef))"),
    (test_score_equations_match_reference, 3, "the residual is y - p, not p - y"),
    (test_score_equations_validate, 2, "coef needs one entry per column, y one per row"),
    (test_closed_form_hand_worked, 4,
     "intercept = ln(n01 / n00); slope = ln(n11 * n00 / (n10 * n01)); float(...)"),
    (test_closed_form_zero_cell_is_none, 4,
     "any empty cell: return None — no correction constant, no infinity"),
    (test_closed_form_validation, 2,
     "binary x and y of equal length, x not constant, both classes of y"),
    (test_woe_from_counts_matches_reference, 5,
     "share = (count + a) / (class total + a * m) for each class; WoE = ln(non-event share / "
     "event share)"),
    (test_woe_empty_bin_nan_not_counted, 3,
     "m counts OCCUPIED bins only; an empty bin stays nan"),
    (test_woe_from_counts_validation, 2,
     "negative counts, a missing class, unequal lengths and smoothing <= 0 are ValueErrors"),
    (test_woe_table_layout_and_missing_bin, 4,
     "value bins in order, then the missing bin — always present, even empty"),
    (test_woe_table_edge_convention, 3,
     "np.searchsorted(edges, x, side='right') - 1, clipped into range"),
    (test_woe_table_validation, 2,
     "edges: at least two, strictly increasing; labels 0 or 1"),
    (test_woe_table_honours_smoothing, 2,
     "woe_from_counts(events, non_events, smoothing), with the table's own smoothing"),
    (test_monotonic_pools_repeatedly, 5,
     "after merging a pair, compare the merged block with the block before it again"),
    (test_monotonic_matches_reference_on_random_tables, 5,
     "pool adjacent violators until none remain; the answer does not depend on the order"),
    (test_monotonic_missing_bin_untouched_and_woe_recomputed, 3,
     "leave the last (missing) bin out of pooling; recompute every WoE afterwards"),
    (test_monotonic_honours_smoothing, 2,
     "recompute the weights with woe_from_counts(..., smoothing), the smoothing passed in"),
    (test_monotonic_ties_are_not_violations, 2,
     "only a strict move against the direction is a violation"),
    (test_monotonic_validation, 2,
     "unknown direction or an empty value bin: ValueError"),
    (test_audit_clean_shuffled_pair, 5,
     "align by record id; compare np.isnan masks rather than raw values"),
    (test_audit_records_dropped_and_extra, 3,
     "np.setdiff1d both ways and np.intersect1d; put the counts in the evidence"),
    (test_audit_duplicate_id_same_length, 3,
     "count duplicates: len(ids) - len(np.unique(ids))"),
    (test_audit_outcome_disagreement, 2,
     "compare y for every shared id, and count the disagreements"),
    (test_audit_period_record_by_record, 4,
     "compare the period of each shared record, not the range of the column"),
    (test_audit_missing_filled, 4,
     "count records missing on one side only, and show the values the other side holds"),
    (test_audit_input_absent, 2,
     "an input the other frame lacks fails 'missing values'"),
    (test_audit_no_common_records, 2,
     "with nothing shared, period and missing values cannot hold"),
    (test_audit_finds_the_planted_asymmetry, 3,
     "the first-line extract holds one repeated income where the vendor's file has gaps"),
    (test_comparison_refuses_non_comparable, 7,
     "audit first; a non-comparable challenger gets verdict 'not comparable' and None for "
     "every measurement"),
    (test_comparison_promotes_clear_winner, 3,
     "paired_bootstrap_difference(y, challenger, champion, ...): challenger first"),
    (test_comparison_holds_when_interval_includes_zero, 4,
     "promote only when the decision promotes AND paired.lo > 0"),
    (test_comparison_holds_when_interval_touches_zero, 2,
     "paired.lo > 0 is strict: a lower end of exactly zero is a hold"),
    (test_comparison_rejects_on_gate, 2,
     "a decision of 'reject' is the verdict, whatever the interval says"),
    (test_comparison_matches_carried_instruments, 4,
     "use auc_by_ranks, expected_calibration_error, challenger_decision and "
     "paired_bootstrap_difference exactly as given, with n_boot, alpha and seed passed through"),
    (test_comparison_aligns_by_id, 3,
     "reorder the challenger's scores into the champion's record order by id"),
    (test_comparison_order_invariant, 3,
     "np.random.default_rng(seed) inside the loop, once per challenger"),
]
