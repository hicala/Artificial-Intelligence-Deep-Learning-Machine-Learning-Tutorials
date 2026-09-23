"""Autograder rubric for P04-L08. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson, and are deliberate:

* nothing is graded against the lesson's own synthetic book, and nothing against the student's
  own random stream. Permutation importance is graded through a SPY model that records every
  matrix it is shown: the rubric reads back which permutations the student actually drew and
  recomputes the drops, the mean and the spread from them with its own metric. Any correct use
  of the generator passes; any wrong arithmetic on the permutations drawn fails.
* every other expected value comes from an independent route: partial dependence and the
  Mahalanobis shares from explicit loops; Shapley values from the average over all p!
  orderings (the definition), never from the subset formula the student implements; and a
  linear value function whose Shapley values are known in closed form.
* the exercises chain (exercise 9 regenerates what exercises 1-8 build), so each exercise is
  graded on cases that do NOT depend on the others. Exercise 9 is handed tiny builders defined
  in this file and a serialiser defined in this file, so a broken exercise 8 cannot cost a
  student the exercise-9 marks, and a correct exercise 9 cannot hide behind exercise 8.
* the planted defects in exercise 9 build their sets from a tuple AT RUN TIME. A set literal
  of three or more strings compiles to a frozenset constant whose element order is baked in by
  the process that compiled it; building the set from a tuple makes each fresh process's order
  depend on its own hash seed alone, which is what the lesson claims and what these cases need
  to be deterministic. Under PYTHONHASHSEED 1, 2 and 3 the orders were checked distinct on
  Python 3.11 and 3.12.
* `test_repro_first_difference_exact` recomputes the first difference itself — the pack in
  hand in this process, each fresh run through the lesson's own helper — so a line number that
  is off by one, or a hash seed that is not the FIRST mismatching one, fails it.
* `test_repro_compares_with_pack_in_hand` uses a builder whose fresh processes all agree with
  each other and all disagree with this process. A check that compares the fresh runs only
  among themselves certifies it.
* every rejection rule is graded on BOTH sides of its boundary: the smallest legal input (two
  repeats, a one-value grid, a one-row background, two distinct hash seeds) must be accepted,
  so a guard that is off by one step in the refusing direction loses the row it belongs to.
"""
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)


# --- instruments and independent references ------------------------------------------------

class _Recorder:
    """A model that records a copy of every matrix it is shown."""

    def __init__(self, fn):
        self.fn, self.seen = fn, []

    def __call__(self, X):
        self.seen.append(np.array(X, dtype=float, copy=True))
        return self.fn(np.asarray(X, dtype=float))


def _neg_mse(y, s):
    return -float(np.mean((np.asarray(y, dtype=float) - np.asarray(s, dtype=float)) ** 2))


def _pi_model(A):
    return np.tanh(A[:, 0]) + 0.5 * A[:, 1] * A[:, 2] - 0.3 * A[:, 3]


def _pi_data(seed=101, n=24):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, 4))
    y = _pi_model(X) + rng.normal(0.0, 0.1, n)
    return X, y


def _changed_columns(A, X):
    return [j for j in range(X.shape[1]) if not np.array_equal(A[:, j], X[:, j])]


def _calls_by_column(seen, X):
    """For a single-feature run: column -> list of shuffled matrices, in call order."""
    out = {}
    for A in seen:
        cols = _changed_columns(A, X)
        assert len(cols) <= 1, (
            f"one model call changed columns {cols} at once — only the column being scored may "
            "differ from the caller's data; shuffle a fresh COPY for every repeat"
        )
        if cols:
            out.setdefault(cols[0], []).append(A)
    return out


def _pd_reference(model, X, j, grid):
    out = []
    for g in grid:
        rows = [list(r) for r in X]
        for r in rows:
            r[j] = g
        out.append(float(np.mean(model(np.array(rows)))))
    return out


def _mahalanobis_reference(X, j, k, grid, level, partner_values=None):
    """Pure-Python squared distances with the 2x2 inverse written out by hand."""
    a = [float(v) for v in X[:, j]]
    b = [float(v) for v in X[:, k]]
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    saa = sum((u - ma) ** 2 for u in a) / (n - 1)
    sbb = sum((v - mb) ** 2 for v in b) / (n - 1)
    sab = sum((u - ma) * (v - mb) for u, v in zip(a, b)) / (n - 1)
    det = saa * sbb - sab * sab
    inv = (sbb / det, -sab / det, saa / det)
    c = -2.0 * math.log(1.0 - level)

    def d2(u, v):
        du, dv = u - ma, v - mb
        return inv[0] * du * du + 2 * inv[1] * du * dv + inv[2] * dv * dv

    pv = b if partner_values is None else [float(v) for v in partner_values]
    data = sum(d2(u, v) > c for u, v in zip(a, b)) / n
    per = [sum(d2(g, v) > c for v in pv) / len(pv) for g in grid]
    return c, data, per


def _shapley_by_orderings(value, p):
    """The definition: average marginal contribution over all p! orderings."""
    phi = [0.0] * p
    orders = list(itertools.permutations(range(p)))
    for order in orders:
        members = []
        before = value(())
        for i in order:
            members.append(i)
            after = value(tuple(sorted(members)))
            phi[i] += after - before
            before = after
    return [x / len(orders) for x in phi]


def _random_game(p, seed):
    rng = np.random.default_rng(seed)
    table = {}
    for m in range(1 << p):
        table[tuple(i for i in range(p) if m >> i & 1)] = float(rng.normal())
    return lambda s: table[tuple(s)]


def _plain(v):
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def _ref_serialise(pack):
    """The rubric's own canonical form, with a prefix no student serialiser would produce."""
    text = json.dumps(_plain(pack), sort_keys=True, indent=2, allow_nan=False)
    return b"#rubric\n" + text.encode("utf-8") + b"\n"


# Builders handed to reproducibility_check. Module-level, so a fresh process can rebuild them.
# Each set is built from a tuple at run time; see the module docstring for why.

def _rubric_clean_build(seed):
    rng = np.random.default_rng(seed)
    return {"alpha": float(rng.random()), "beta": [float(v) for v in rng.random(3)],
            "seed": seed}


def _rubric_set_order_build(seed):
    rng = np.random.default_rng(seed)
    out = {}
    for name in set(("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")):
        out[name] = float(rng.random())
    return out


def _rubric_last_bit_build(seed):
    parts = {"big": 1.0, "tiny_a": 1e-16, "tiny_b": 1e-16, "tiny_c": 1e-16}
    total = 0.0
    for name in set(("big", "tiny_a", "tiny_b", "tiny_c")):
        total += parts[name]
    return {"seed": seed, "total": total}


def _rubric_env_build(seed):
    import os as _os
    return {"fresh_process": "PYTHONHASHSEED" in _os.environ, "seed": seed}


def _rubric_crash_build(seed):
    import os as _os
    if "PYTHONHASHSEED" in _os.environ:
        raise RuntimeError("this builder cannot be regenerated")
    return {"seed": seed}


def _rubric_insertion_build(seed):
    return {name: len(name) for name in set(("alpha", "bravo", "charlie", "delta", "echo"))}


class _EnvWithout:
    """Temporarily remove PYTHONHASHSEED from this process's environment (or pin it)."""

    def __init__(self, value=None):
        self.value, self.saved = value, None

    def __enter__(self):
        self.saved = os.environ.pop("PYTHONHASHSEED", None)
        if self.value is not None:
            os.environ["PYTHONHASHSEED"] = self.value

    def __exit__(self, *exc):
        os.environ.pop("PYTHONHASHSEED", None)
        if self.saved is not None:
            os.environ["PYTHONHASHSEED"] = self.saved


def _accepted(fn, what):
    """Call fn on input the docstring allows; a ValueError there is a guard one step too wide."""
    try:
        return fn()
    except NotImplementedError:
        raise
    except ValueError as exc:
        raise AssertionError(f"{what} is legal input and raised ValueError ({exc}) — only the "
                             "cases the docstring lists are errors") from None


def _raises_value_error(fn, what):
    try:
        fn()
    except ValueError:
        return
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"{what} raised {type(exc).__name__}, not ValueError") from None
    raise AssertionError(f"{what} should raise ValueError")


# --- exercise 1: permutation_importance() ---------------------------------------------------

def test_pi_shape_and_baseline():
    X, y = _pi_data()
    res = lesson.permutation_importance(_pi_model, _neg_mse, X, y, 3, np.random.default_rng(4))
    assert isinstance(res, dict), f"return a dict, got {type(res).__name__}"
    for key in ("baseline", "columns", "mean", "std", "drops"):
        assert key in res, f"the result has no {key!r}"
    assert np.asarray(res["drops"]).shape == (4, 3), (
        f"drops has shape {np.asarray(res['drops']).shape}; expected (columns, repeats) = (4, 3)"
    )
    assert np.asarray(res["mean"]).shape == (4,) and np.asarray(res["std"]).shape == (4,)
    assert tuple(res["columns"]) == (0, 1, 2, 3), (
        f"columns defaults to every column, left to right; got {tuple(res['columns'])}"
    )
    assert isinstance(res["baseline"], float), "baseline must be a Python float"
    assert abs(res["baseline"] - _neg_mse(y, _pi_model(X))) < 1e-12, (
        "baseline is metric(y, model(X)) on the untouched data"
    )


def test_pi_one_column_at_a_time():
    X, y = _pi_data()
    before = X.copy()
    spy = _Recorder(_pi_model)
    lesson.permutation_importance(spy, _neg_mse, X, y, 3, np.random.default_rng(5))
    assert np.array_equal(X, before), (
        "the caller's X was changed — every shuffle must happen on a copy"
    )
    groups = _calls_by_column(spy.seen, X)
    for j, mats in groups.items():
        for A in mats:
            assert np.array_equal(np.sort(A[:, j]), np.sort(X[:, j])), (
                f"column {j} was replaced by values that are not a reordering of its own"
            )
    assert any(not _changed_columns(A, X) for A in spy.seen), (
        "the model was never shown the untouched data — where did the baseline come from?"
    )


def test_pi_repeats_and_order():
    X, y = _pi_data()
    spy = _Recorder(_pi_model)
    res = lesson.permutation_importance(spy, _neg_mse, X, y, 4, np.random.default_rng(6),
                                        columns=(2, 0))
    assert tuple(res["columns"]) == (2, 0), (
        f"columns=(2, 0) must be reported back in that order, got {tuple(res['columns'])}"
    )
    order = [c[0] for c in (_changed_columns(A, X) for A in spy.seen) if c]
    assert order == [2] * 4 + [0] * 4, (
        f"columns were shuffled in the order {order}; score each column in `columns` in turn, "
        "all its repeats before the next one, and nothing else"
    )
    groups = _calls_by_column(spy.seen, X)
    for j in (2, 0):
        distinct = {tuple(A[:, j]) for A in groups[j]}
        assert len(distinct) > 1, (
            f"all four repeats of column {j} used the same shuffle — draw a fresh permutation "
            "for every repeat, or the spread you report is the spread of one number"
        )


def test_pi_values_recomputed():
    X, y = _pi_data(seed=202)
    spy = _Recorder(_pi_model)
    res = lesson.permutation_importance(spy, _neg_mse, X, y, 5, np.random.default_rng(7))
    base = _neg_mse(y, _pi_model(X))
    groups = _calls_by_column(spy.seen, X)
    drops = np.asarray(res["drops"], dtype=float)
    for a, j in enumerate(res["columns"]):
        want = [base - _neg_mse(y, _pi_model(A)) for A in groups[j]]
        assert np.allclose(drops[a], want, atol=1e-12), (
            f"column {j}: drops {np.round(drops[a], 6).tolist()} do not match the shuffles the "
            f"model was shown ({np.round(want, 6).tolist()}) — each drop is baseline minus the "
            "shuffled score"
        )
        assert abs(float(res["mean"][a]) - float(np.mean(want))) < 1e-12, "mean over repeats"
        assert abs(float(res["std"][a]) - float(np.std(want, ddof=1))) < 1e-12, (
            f"column {j}: std {float(res['std'][a]):.6f}, expected the SAMPLE standard deviation "
            f"across repeats {float(np.std(want, ddof=1)):.6f} (ddof=1)"
        )


def test_pi_seeded():
    X, y = _pi_data()
    one = lesson.permutation_importance(_pi_model, _neg_mse, X, y, 3, np.random.default_rng(8))
    two = lesson.permutation_importance(_pi_model, _neg_mse, X, y, 3, np.random.default_rng(8))
    other = lesson.permutation_importance(_pi_model, _neg_mse, X, y, 3, np.random.default_rng(9))
    assert np.array_equal(np.asarray(one["drops"]), np.asarray(two["drops"])), (
        "the same seed gave different results — every permutation must come from the rng "
        "argument, not from np.random or a generator you created yourself"
    )
    assert not np.array_equal(np.asarray(one["drops"]), np.asarray(other["drops"])), (
        "two different seeds gave identical results — the rng argument is being ignored"
    )


def test_pi_rejects():
    X, y = _pi_data()
    g = np.random.default_rng(1)
    _raises_value_error(lambda: lesson.permutation_importance(_pi_model, _neg_mse, X, y, 1, g),
                        "n_repeats=1")
    _raises_value_error(lambda: lesson.permutation_importance(_pi_model, _neg_mse, X, y[:-1], 3,
                                                              g), "a y shorter than X")
    _raises_value_error(lambda: lesson.permutation_importance(_pi_model, _neg_mse, X, y, 3, g,
                                                              columns=(4,)),
                        "a column index past the last column")
    two = _accepted(lambda: lesson.permutation_importance(_pi_model, _neg_mse, X, y, 2, g),
                    "n_repeats=2")
    assert np.asarray(two["drops"]).shape == (4, 2), (
        "two repeats is the smallest count with a spread: drops must have shape (4, 2)"
    )


# --- exercise 2: grouped_permutation_importance() -------------------------------------------

def test_grouped_joint_permutation():
    X, y = _pi_data(seed=303)
    before = X.copy()
    spy = _Recorder(_pi_model)
    lesson.grouped_permutation_importance(spy, _neg_mse, X, y, [(0, 2), (1,)], 3,
                                          np.random.default_rng(10))
    assert np.array_equal(X, before), "the caller's X was changed — shuffle a copy"
    pair_rows = sorted(map(tuple, X[:, [0, 2]].tolist()))
    pair_calls = single_calls = 0
    for A in spy.seen:
        cols = set(_changed_columns(A, X))
        if not cols:
            continue
        assert cols <= {0, 2} or cols <= {1}, (
            f"a call changed columns {sorted(cols)}, which is not inside any one group"
        )
        if cols <= {1}:
            single_calls += 1
            continue
        pair_calls += 1
        assert np.array_equal(A[:, [1, 3]], X[:, [1, 3]]), "columns outside the group moved"
        assert sorted(map(tuple, A[:, [0, 2]].tolist())) == pair_rows, (
            "after shuffling the group (0, 2), some rows hold a combination of its two columns "
            "that no real row had — the columns got different permutations. One permutation "
            "per group per repeat, applied to every column of the group at once."
        )
    assert pair_calls == 3 and single_calls == 3, (
        f"expected 3 shuffles of each group, saw {pair_calls} of the pair and {single_calls} of "
        "the single column"
    )


def test_grouped_values():
    X, y = _pi_data(seed=404)
    spy = _Recorder(_pi_model)
    res = lesson.grouped_permutation_importance(spy, _neg_mse, X, y, [(1, 2), (3,)], 4,
                                                np.random.default_rng(11))
    base = _neg_mse(y, _pi_model(X))
    assert tuple(tuple(g) for g in res["groups"]) == ((1, 2), (3,)), (
        f"groups must be reported back as given, got {res['groups']}"
    )
    by_group = {(1, 2): [], (3,): []}
    for A in spy.seen:
        cols = tuple(sorted(_changed_columns(A, X)))
        if cols in ((1, 2), (1,), (2,)):
            by_group[(1, 2)].append(A)
        elif cols == (3,):
            by_group[(3,)].append(A)
    drops = np.asarray(res["drops"], dtype=float)
    assert drops.shape == (2, 4), f"drops has shape {drops.shape}, expected (2, 4)"
    for a, g in enumerate(((1, 2), (3,))):
        want = [base - _neg_mse(y, _pi_model(A)) for A in by_group[g]]
        assert np.allclose(drops[a], want, atol=1e-12), (
            f"group {g}: the drops do not match the shuffles the model was shown"
        )
        assert abs(float(res["std"][a]) - float(np.std(want, ddof=1))) < 1e-12, (
            "std is the sample standard deviation across repeats (ddof=1)"
        )
        assert abs(float(res["mean"][a]) - float(np.mean(want))) < 1e-12, "mean over repeats"


def test_grouped_seeded():
    X, y = _pi_data()
    runs = [lesson.grouped_permutation_importance(_pi_model, _neg_mse, X, y, [(0, 1)], 3,
                                                  np.random.default_rng(s)) for s in (12, 12, 13)]
    assert np.array_equal(np.asarray(runs[0]["drops"]), np.asarray(runs[1]["drops"])), (
        "the same seed gave different results — use the rng you were handed"
    )
    assert not np.array_equal(np.asarray(runs[0]["drops"]), np.asarray(runs[2]["drops"])), (
        "different seeds gave identical results — the rng argument is being ignored"
    )


def test_grouped_rejects():
    X, y = _pi_data()
    g = np.random.default_rng(1)
    _raises_value_error(lambda: lesson.grouped_permutation_importance(
        _pi_model, _neg_mse, X, y, [()], 3, g), "an empty group")
    _raises_value_error(lambda: lesson.grouped_permutation_importance(
        _pi_model, _neg_mse, X, y, [(0, 9)], 3, g), "a column index out of range")
    _raises_value_error(lambda: lesson.grouped_permutation_importance(
        _pi_model, _neg_mse, X, y, [(0,)], 1, g), "n_repeats=1")
    two = _accepted(lambda: lesson.grouped_permutation_importance(
        _pi_model, _neg_mse, X, y, [(0, 1)], 2, g), "n_repeats=2")
    assert np.asarray(two["drops"]).shape == (1, 2), (
        "two repeats is the smallest count with a spread: drops must have shape (1, 2)"
    )


# --- exercise 3: partial_dependence() -------------------------------------------------------

def _pd_model(A):
    return 1.0 / (1.0 + np.exp(-(A[:, 0] ** 2 * A[:, 1] - A[:, 2] + 0.5)))


def _pd_data():
    return np.random.default_rng(505).normal(0.0, 1.0, (15, 3))


def test_pd_matches_reference():
    X = _pd_data()
    grid = (-1.5, -0.25, 0.0, 0.8, 2.0)
    got = np.asarray(lesson.partial_dependence(_pd_model, X, 0, grid), dtype=float)
    want = _pd_reference(_pd_model, X, 0, grid)
    assert got.shape == (5,), f"one value per grid point, got shape {got.shape}"
    assert np.allclose(got, want, atol=1e-12), (
        f"got {np.round(got, 6).tolist()}, expected {np.round(want, 6).tolist()}"
    )
    other = np.asarray(lesson.partial_dependence(_pd_model, X, 2, grid[:2]), dtype=float)
    assert np.allclose(other, _pd_reference(_pd_model, X, 2, grid[:2]), atol=1e-12), (
        "partial dependence on column 2 is wrong — is the feature index being honoured?"
    )


def test_pd_average_of_predictions():
    X = _pd_data()
    grid = (1.0, 2.0)
    got = np.asarray(lesson.partial_dependence(_pd_model, X, 1, grid), dtype=float)
    at_mean = []
    for g in grid:
        row = X.mean(axis=0)
        row[1] = g
        at_mean.append(float(_pd_model(row[None, :])[0]))
    assert not np.allclose(got, at_mean, atol=1e-6), (
        "these are the model's predictions at the AVERAGE applicant; partial dependence is the "
        "average of every applicant's prediction with the feature set to the grid value"
    )
    assert np.allclose(got, _pd_reference(_pd_model, X, 1, grid), atol=1e-12)


def test_pd_untouched_input():
    X = _pd_data()
    before = X.copy()
    lesson.partial_dependence(_pd_model, X, 1, [0.0, 3.0])
    assert np.array_equal(X, before), "partial_dependence changed the caller's X"


def test_pd_rejects():
    X = _pd_data()
    _raises_value_error(lambda: lesson.partial_dependence(_pd_model, X, 3, [0.0]),
                        "a feature index past the last column")
    _raises_value_error(lambda: lesson.partial_dependence(_pd_model, X, 0, []), "an empty grid")
    one = _accepted(lambda: lesson.partial_dependence(_pd_model, X, 0, [0.5]), "a one-value grid")
    assert np.asarray(one).shape == (1,), "a one-value grid gives one partial-dependence value"


# --- exercise 4: partial_dependence_2d() ----------------------------------------------------

def _pd2_model(A):
    return A[:, 0] * A[:, 1] + np.sin(A[:, 2]) * A[:, 0]


def test_pd2d_orientation():
    X = np.random.default_rng(606).normal(0.0, 1.0, (9, 3))
    g1, g2 = (0.5, 2.0), (-1.0, 0.0, 1.5)
    got = np.asarray(lesson.partial_dependence_2d(_pd2_model, X, (0, 1), (g1, g2)), dtype=float)
    assert got.shape == (2, 3), (
        f"shape {got.shape}: the first grid runs down the rows, so a 2-by-3 grid gives (2, 3)"
    )
    want = np.array([[u * v + u * float(np.mean(np.sin(X[:, 2]))) for v in g2] for u in g1])
    assert np.allclose(got, want, atol=1e-12), (
        f"got {np.round(got, 4).tolist()}, expected {np.round(want, 4).tolist()}"
    )


def test_pd2d_interaction():
    X = np.random.default_rng(707).normal(0.0, 1.0, (12, 3))
    g1, g2 = (-1.0, 1.0, 2.0), (-2.0, 0.5)
    got = np.asarray(lesson.partial_dependence_2d(_pd2_model, X, (2, 0), (g1, g2)), dtype=float)
    want = np.empty((3, 2))
    for a, u in enumerate(g1):
        for b, v in enumerate(g2):
            rows = X.copy()
            rows[:, 2] = u
            rows[:, 0] = v
            want[a, b] = float(np.mean(_pd2_model(rows)))
    additive = np.array([[_pd_reference(_pd2_model, X, 2, [u])[0]
                          + _pd_reference(_pd2_model, X, 0, [v])[0]
                          - float(np.mean(_pd2_model(X))) for v in g2] for u in g1])
    assert not np.allclose(got, additive, atol=1e-6), (
        "this is two one-dimensional curves added together — set BOTH features in the same copy "
        "before predicting, or the interaction disappears"
    )
    assert np.allclose(got, want, atol=1e-12), (
        f"features (2, 0): got {np.round(got, 4).tolist()}, expected {np.round(want, 4).tolist()}"
    )


def test_pd2d_rejects_and_untouched():
    X = np.random.default_rng(808).normal(0.0, 1.0, (6, 3))
    before = X.copy()
    lesson.partial_dependence_2d(_pd2_model, X, (0, 2), ((1.0,), (2.0,)))
    assert np.array_equal(X, before), "partial_dependence_2d changed the caller's X"
    _raises_value_error(lambda: lesson.partial_dependence_2d(_pd2_model, X, (1, 1),
                                                             ((0.0,), (1.0,))),
                        "the same feature twice")
    _raises_value_error(lambda: lesson.partial_dependence_2d(_pd2_model, X, (0, 5),
                                                             ((0.0,), (1.0,))),
                        "a feature index out of range")


# --- exercise 5: off_manifold_share() -------------------------------------------------------

def _manifold_data(n=300, rho=0.9, seed=909):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, n)
    b = rho * a + math.sqrt(1 - rho * rho) * rng.normal(0.0, 1.0, n)
    return np.column_stack([rng.normal(0.0, 1.0, n), 2.0 * a + 1.0, b - 0.5])


def test_manifold_threshold():
    X = _manifold_data()
    for level in (0.5, 0.9, 0.99):
        r = lesson.off_manifold_share(X, 1, 2, [0.0], level=level)
        assert abs(r["threshold"] - (-2.0 * math.log(1.0 - level))) < 1e-12, (
            f"level {level}: threshold {r['threshold']:.6f}, expected -2 ln(1 - level) = "
            f"{-2.0 * math.log(1.0 - level):.6f} — the squared radius of a TWO-dimensional "
            "normal's ellipse, not a one-dimensional quantile"
        )


def test_manifold_per_grid_reference():
    X = _manifold_data()
    grid = (-3.0, -1.0, 0.0, 1.0, 2.5, 4.0, 6.0)
    r = lesson.off_manifold_share(X, 1, 2, grid, level=0.95)
    _, _, per = _mahalanobis_reference(X, 1, 2, grid, 0.95)
    got = np.asarray(r["per_grid"], dtype=float)
    assert got.shape == (7,), f"one share per grid value, got shape {got.shape}"
    assert np.all(np.abs(got - np.array(per)) <= 1.0 / X.shape[0] + 1e-12), (
        f"per_grid {np.round(got, 3).tolist()}, expected {np.round(per, 3).tolist()} — the "
        "distance must use the INVERSE of the pair's sample covariance, with the column means "
        "as the centre; Euclidean distance ignores the correlation that is the whole point"
    )


def test_manifold_data_share_and_overall():
    X = _manifold_data(seed=1010)
    grid = (0.0, 3.0, 5.0)
    r = lesson.off_manifold_share(X, 1, 2, grid, level=0.8)
    _, data, per = _mahalanobis_reference(X, 1, 2, grid, 0.8)
    assert abs(r["data_share"] - data) <= 1.0 / X.shape[0] + 1e-12, (
        f"data_share {r['data_share']:.4f}, expected {data:.4f}: the share of the REAL rows, "
        "each at its own two values, outside the ellipse"
    )
    assert abs(r["overall"] - float(np.mean(np.asarray(r["per_grid"], dtype=float)))) < 1e-12, (
        "overall is the mean of per_grid"
    )
    assert abs(r["overall"] - float(np.mean(per))) <= 1.0 / X.shape[0] + 1e-12


def test_manifold_partner_values():
    X = _manifold_data(seed=1111)
    pv = (-3.0, -0.5, 0.0, 0.5, 2.0)
    r = lesson.off_manifold_share(X, 1, 2, (1.0, 4.0), level=0.9, partner_values=pv)
    _, _, per = _mahalanobis_reference(X, 1, 2, (1.0, 4.0), 0.9, partner_values=pv)
    assert np.allclose(np.asarray(r["per_grid"], dtype=float), per, atol=1e-12), (
        "with partner_values given, per_grid is the share over THOSE values, not over the rows"
    )


def test_manifold_sample_covariance():
    # Six rows, so the divisor matters: n / (n - 1) = 1.2. The grid point is placed, with the
    # partner at its mean, at a squared distance of 0.95 x threshold under the n - 1 divisor —
    # inside the ellipse. A population covariance (divisor n) shrinks every variance by 5/6,
    # lifts the distance to 1.14 x threshold and calls the point off the manifold.
    X = np.array([[0.0, 0.1], [1.0, 1.3], [2.0, 1.8], [3.0, 3.2], [4.0, 3.9], [5.0, 5.3]])
    a, b = X[:, 0], X[:, 1]
    n = a.size
    ma, mb = float(a.mean()), float(b.mean())
    saa = float(((a - ma) ** 2).sum()) / (n - 1)
    sbb = float(((b - mb) ** 2).sum()) / (n - 1)
    sab = float(((a - ma) * (b - mb)).sum()) / (n - 1)
    conditional = saa - sab * sab / sbb
    c = -2.0 * math.log(1.0 - 0.9)
    inside = ma + math.sqrt(0.95 * c * conditional)
    r = lesson.off_manifold_share(X, 0, 1, [inside], level=0.9, partner_values=[mb])
    assert float(np.asarray(r["per_grid"], dtype=float)[0]) == 0.0, (
        "a point just inside the ellipse was called off the manifold — the ellipse uses the "
        "SAMPLE covariance, divisor n - 1; with six rows the divisor n shrinks it visibly"
    )


def test_manifold_rejects():
    X = _manifold_data()
    _raises_value_error(lambda: lesson.off_manifold_share(X, 1, 2, [0.0], level=1.0), "level=1")
    _raises_value_error(lambda: lesson.off_manifold_share(X, 1, 2, [0.0], level=0.0), "level=0")
    _raises_value_error(lambda: lesson.off_manifold_share(X, 1, 1, [0.0]), "feature == partner")
    _raises_value_error(lambda: lesson.off_manifold_share(X, 1, 2, []), "an empty grid")


# --- exercise 6: coalition_value() ----------------------------------------------------------

def _cv_model(A):
    return 1.0 / (1.0 + np.exp(-(A[:, 0] * A[:, 1] - A[:, 2] ** 2 + 0.3 * A[:, 3])))


def _cv_case():
    rng = np.random.default_rng(1212)
    return rng.normal(0.0, 1.0, (7, 4)), rng.normal(0.0, 1.5, 4)


def test_value_empty_and_full():
    bg, x = _cv_case()
    v = lesson.coalition_value(_cv_model, x, bg)
    assert abs(v(()) - float(np.mean(_cv_model(bg)))) < 1e-12, (
        f"value(()) = {v(()):.6f}, expected the MEAN prediction over the background rows "
        f"{float(np.mean(_cv_model(bg))):.6f} — not the prediction at the average row"
    )
    assert abs(v((0, 1, 2, 3)) - float(_cv_model(x[None, :])[0])) < 1e-12, (
        "value(all features) must be the model's prediction for x"
    )
    one = _accepted(lambda: lesson.coalition_value(_cv_model, x, bg[:1]), "a one-row background")
    assert abs(one(()) - float(_cv_model(bg[:1])[0])) < 1e-12, (
        "with a one-row background, value(()) is that row's own prediction"
    )


def test_value_mixed_coalitions():
    bg, x = _cv_case()
    v = lesson.coalition_value(_cv_model, x, bg)
    for coalition in ((1,), (0, 2), (1, 2, 3)):
        rows = bg.copy()
        rows[:, list(coalition)] = x[list(coalition)]
        want = float(np.mean(_cv_model(rows)))
        assert abs(v(coalition) - want) < 1e-12, (
            f"value({coalition}) = {v(coalition):.6f}, expected {want:.6f} — overwrite the "
            "coalition's columns with x in every background row and keep each row's own values "
            "elsewhere (not zeros, not the column means)"
        )
    _raises_value_error(lambda: lesson.coalition_value(_cv_model, x[:3], bg),
                        "an x with the wrong number of features")


def test_value_no_hidden_state():
    bg, x = _cv_case()
    bg_before, x_before = bg.copy(), x.copy()
    v = lesson.coalition_value(_cv_model, x, bg)
    first = v(())
    v((0, 1))
    v((2,))
    again = v(())
    assert abs(first - again) < 1e-12, (
        "value(()) changed after other coalitions were evaluated — a call wrote into the "
        "background; copy it before overwriting columns"
    )
    assert np.array_equal(bg, bg_before) and np.array_equal(x, x_before), (
        "the caller's background or x was changed"
    )
    assert isinstance(first, float), f"return a float, got {type(first).__name__}"


# --- exercise 7: shapley_values() -----------------------------------------------------------

def test_shapley_matches_definition():
    for p, seed in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5)):
        game = _random_game(p, seed)
        got = np.asarray(lesson.shapley_values(game, p), dtype=float)
        want = _shapley_by_orderings(game, p)
        assert got.shape == (p,), f"p={p}: expected {p} values, got shape {got.shape}"
        assert np.allclose(got, want, atol=1e-10), (
            f"p={p}: got {np.round(got, 6).tolist()}, the average over all orderings gives "
            f"{np.round(want, 6).tolist()} — check the weight |S|!(p-|S|-1)!/p!"
        )


def test_shapley_axioms():
    # players 0 and 1 are interchangeable, player 3 contributes nothing to any coalition
    def game(s):
        s = set(s)
        base = 2.0 * (0 in s) + 2.0 * (1 in s) + 5.0 * (2 in s)
        return base + 3.0 * (0 in s and 2 in s) + 3.0 * (1 in s and 2 in s)

    phi = np.asarray(lesson.shapley_values(game, 4), dtype=float)
    assert abs(phi[0] - phi[1]) < 1e-12, "interchangeable players must get equal attributions"
    assert abs(phi[3]) < 1e-12, "a player who adds nothing to any coalition must get exactly 0"
    assert abs(phi.sum() - (game((0, 1, 2, 3)) - game(()))) < 1e-12, (
        "the attributions must add up to v(all) - v(()) — efficiency"
    )
    assert np.allclose(phi, [3.5, 3.5, 8.0, 0.0]), f"got {phi.tolist()}, expected [3.5, 3.5, 8, 0]"


def test_shapley_one_call_per_coalition():
    game = _random_game(5, 55)
    calls = []

    def spy(s):
        calls.append(s)
        return game(s)

    lesson.shapley_values(spy, 5)
    assert all(isinstance(c, tuple) for c in calls), "pass every coalition as a tuple"
    assert all(list(c) == sorted(c) for c in calls), "pass every coalition in ascending order"
    assert len(set(calls)) == 32, f"{len(set(calls))} distinct coalitions seen; there are 32"
    assert len(calls) == 32, (
        f"the value function was called {len(calls)} times for 32 coalitions — each call is a "
        "batch of model predictions; evaluate each coalition once and look it up"
    )


def test_shapley_linear_closed_form():
    w = np.array([0.7, -1.2, 0.4, 2.0])
    x = np.array([1.0, 0.5, -2.0, 0.25])
    b = np.array([0.2, -0.3, 0.1, 1.0])

    def value(s):
        known = np.zeros(4, dtype=bool)
        known[list(s)] = True
        return float(np.sum(w * np.where(known, x, b)))

    got = np.asarray(lesson.shapley_values(value, 4), dtype=float)
    assert np.allclose(got, w * (x - b), atol=1e-12), (
        f"for a linear value function each attribution is w_i (x_i - b_i) = "
        f"{np.round(w * (x - b), 6).tolist()}; got {np.round(got, 6).tolist()}"
    )


def test_shapley_rejects():
    _raises_value_error(lambda: lesson.shapley_values(lambda s: 0.0, 0), "n_features=0")


# --- exercise 8: canonical_bytes() ----------------------------------------------------------

_AWKWARD = [1 / 3, 0.1 + 0.2, 1e-300, 5e-324, 123456789.12345679, -0.5, 2.0 ** 60 + 1.0,
            math.nextafter(1.0, 2.0)]


def test_canonical_roundtrip():
    pack = {"floats": _AWKWARD, "nested": {"arr": np.array([[0.1, 0.2], [0.3, 0.4]]),
                                           "i": np.int64(7), "f": np.float64(1 / 7),
                                           "b": np.bool_(True), "t": (1, "x")},
            "s": "text", "n": None}
    out = lesson.canonical_bytes(pack)
    assert isinstance(out, bytes), f"return bytes, got {type(out).__name__}"
    back = json.loads(out)
    assert back == _plain(pack), (
        "json.loads of your output does not give the pack back — numpy arrays become lists, "
        "numpy scalars become Python numbers, tuples become lists"
    )
    for got, want in zip(back["floats"], _AWKWARD):
        assert float(got).hex() == float(want).hex(), (
            f"{want!r} came back as {got!r} — floats must be written exactly: no rounding, no "
            "float32"
        )


def test_canonical_key_order():
    one = {"b": {"y": 1.5, "x": [1, 2]}, "a": 0.25, "c": {"q": {"z": 1, "k": 2}}}
    two = {"c": {"q": {"k": 2, "z": 1}}, "a": 0.25, "b": {"x": [1, 2], "y": 1.5}}
    assert lesson.canonical_bytes(one) == lesson.canonical_bytes(two), (
        "the same content assembled in another order gave different bytes — sort the keys at "
        "every level of nesting"
    )


def test_canonical_exact_floats():
    ser = lesson.canonical_bytes
    assert ser({"v": 1.0}) != ser({"v": math.nextafter(1.0, 2.0)}), (
        "1.0 and the next float above it gave the same bytes — the serialisation is rounding"
    )
    assert ser({"v": 0.1 + 0.2}) != ser({"v": 0.3}), "0.1 + 0.2 and 0.3 must differ"
    assert ser({"v": np.float64(0.1)}) == ser({"v": 0.1}), (
        "np.float64(0.1) and 0.1 are the same number and must give the same bytes"
    )


def test_canonical_rejects_nonfinite():
    for bad in (float("nan"), float("inf"), -float("inf")):
        _raises_value_error(lambda: lesson.canonical_bytes({"v": bad}), f"a pack holding {bad}")
    _raises_value_error(lambda: lesson.canonical_bytes({"v": np.array([1.0, np.nan])}),
                        "an array holding NaN")


def test_canonical_line_per_leaf():
    pack = {f"key_{i}": float(i) / 7 for i in range(8)}
    base = lesson.canonical_bytes(pack).decode("utf-8").splitlines()
    assert len(base) >= 8, (
        f"{len(base)} lines for 8 scalars — each scalar needs its own line so a divergence can "
        "be located; dump with an indent"
    )
    moved = dict(pack)
    moved["key_5"] = 99.5
    after = lesson.canonical_bytes(moved).decode("utf-8").splitlines()
    diff = [i for i, (a, b) in enumerate(zip(base, after)) if a != b]
    assert len(base) == len(after) and len(diff) == 1 and "key_5" in after[diff[0]], (
        "changing one scalar should change exactly one line, and that line should name its key"
    )


# --- exercise 9: reproducibility_check() ----------------------------------------------------

def test_repro_clean_certificate():
    with _EnvWithout():
        cert = lesson.reproducibility_check(_rubric_clean_build, 21, hash_seeds=(4, 5, 6),
                                            serialise=_ref_serialise)
    ref = _ref_serialise(_rubric_clean_build(21))
    assert cert["reproduced"] is True, (
        "a builder with nothing to hide came back not reproduced — compare each fresh run's "
        "serialised bytes with the pack in hand"
    )
    assert cert["sha256"] == hashlib.sha256(ref).hexdigest(), (
        "sha256 must be the SHA-256 of serialise(build(seed)) — use the serialiser you were "
        "handed, and hash its bytes"
    )
    assert cert["n_bytes"] == len(ref), "n_bytes is the length of the pack in hand's bytes"
    assert cert["seed"] == 21 and list(cert["hash_seeds"]) == [4, 5, 6], (
        "record the seed and the hash seeds you were given, in order"
    )
    runs = cert["runs"]
    assert [r["hash_seed"] for r in runs] == [4, 5, 6] and all(r["match"] is True for r in runs), (
        "one run per hash seed, in the order given, each with its hash seed and match flag"
    )
    assert all(r["sha256"] == cert["sha256"] for r in runs), "each run records its own SHA-256"
    assert cert["first_difference"] is None, "a reproduced pack has no first difference"
    env = cert["environment"]
    assert set(env) >= {"python", "numpy"} and env["numpy"] == np.__version__, (
        "the certificate records the environment: python and numpy versions"
    )


def test_repro_detects_set_order():
    # PYTHONHASHSEED pinned in THIS process's environment, as some CI systems do: a check that
    # lets its fresh processes inherit it runs the same experiment three times.
    with _EnvWithout("0"):
        cert = lesson.reproducibility_check(_rubric_set_order_build, 21, hash_seeds=(1, 2, 3),
                                            serialise=_ref_serialise)
    assert cert["reproduced"] is False, (
        "this builder visits a set of strings, whose order depends on the process's hash seed, "
        "and it was certified — regenerate in FRESH processes, one per hash seed you were "
        "given, not twice in this one"
    )


def test_repro_detects_last_bit():
    with _EnvWithout():
        cert = lesson.reproducibility_check(_rubric_last_bit_build, 21, hash_seeds=(1, 2, 3),
                                            serialise=_ref_serialise)
    assert cert["reproduced"] is False, (
        "the runs differ only in the last bit of one float (1.0 against 1.0000000000000002), "
        "and the check let it through — compare bytes exactly, never within a tolerance"
    )


def test_repro_first_difference_exact():
    seeds = (1, 2, 3)
    with _EnvWithout():
        cert = lesson.reproducibility_check(_rubric_set_order_build, 22, hash_seeds=seeds,
                                            serialise=_ref_serialise)
        ref = _ref_serialise(_rubric_set_order_build(22))
        want = None
        for hs in seeds:
            other = _ref_serialise(lesson.regenerate_in_fresh_process(_rubric_set_order_build,
                                                                      22, hs))
            if other != ref:
                a = ref.decode().split("\n")
                b = other.decode().split("\n")
                k = next(i for i in range(max(len(a), len(b)))
                         if (a[i] if i < len(a) else "") != (b[i] if i < len(b) else ""))
                want = {"hash_seed": hs, "line": k + 1, "expected": a[k], "got": b[k]}
                break
    fd = cert["first_difference"]
    assert want is not None and fd is not None, "the packs differ, so first_difference is set"
    assert fd["hash_seed"] == want["hash_seed"], (
        f"first_difference names hash seed {fd['hash_seed']}; the FIRST run, in the order given, "
        f"that disagreed with the pack in hand was {want['hash_seed']}"
    )
    assert fd["line"] == want["line"], (
        f"first_difference says line {fd['line']}; the first differing line is {want['line']} "
        "counting from ONE"
    )
    assert fd["expected"] == want["expected"] and fd["got"] == want["got"], (
        "expected is that line in the pack in hand, got is the same line in the fresh run"
    )


def test_repro_compares_with_pack_in_hand():
    with _EnvWithout():
        cert = lesson.reproducibility_check(_rubric_env_build, 21, hash_seeds=(1, 2, 3),
                                            serialise=_ref_serialise)
    assert cert["reproduced"] is False and not any(r["match"] for r in cert["runs"]), (
        "every fresh process agreed with every other and none agreed with the pack in hand — "
        "and the check certified it. Compare each fresh run with the pack you are certifying, "
        "not only with each other."
    )


def test_repro_failed_regeneration():
    try:
        with _EnvWithout():
            cert = lesson.reproducibility_check(_rubric_crash_build, 21, hash_seeds=(1, 2, 3),
                                                serialise=_ref_serialise)
    except NotImplementedError:
        raise
    except Exception:  # noqa: BLE001 — refusing to certify by raising is correct
        return
    assert cert.get("reproduced") is False, (
        "every fresh regeneration crashed and the pack was certified anyway — a regeneration "
        "that failed has reproduced nothing; let the error propagate"
    )


def test_repro_no_false_alarm():
    with _EnvWithout():
        cert = lesson.reproducibility_check(_rubric_insertion_build, 21, hash_seeds=(1, 2, 3),
                                            serialise=_ref_serialise)
    assert cert["reproduced"] is True, (
        "the packs hold identical content assembled in different orders, and the check called "
        "them different — compare the SERIALISED bytes from the serialiser you were given, not "
        "repr(), pickle or insertion-ordered JSON"
    )


def test_repro_rejects_one_seed():
    _raises_value_error(lambda: lesson.reproducibility_check(
        _rubric_clean_build, 21, hash_seeds=(5, 5), serialise=_ref_serialise),
        "two identical hash seeds")
    _raises_value_error(lambda: lesson.reproducibility_check(
        _rubric_clean_build, 21, hash_seeds=(5,), serialise=_ref_serialise), "one hash seed")
    with _EnvWithout():
        cert = _accepted(lambda: lesson.reproducibility_check(
            _rubric_clean_build, 21, hash_seeds=(5, 6), serialise=_ref_serialise),
            "two distinct hash seeds")
    assert len(cert["runs"]) == 2, "two distinct hash seeds are enough: one fresh run each"


RUBRIC = [
    (test_pi_shape_and_baseline, 2,
     "return baseline, columns, mean, std and drops; drops is (columns, repeats)"),
    (test_pi_one_column_at_a_time, 4,
     "copy X, overwrite ONE column with its values in permuted row order, leave the rest"),
    (test_pi_repeats_and_order, 3,
     "walk `columns` in order, all repeats of one before the next, a fresh permutation each"),
    (test_pi_values_recomputed, 4,
     "drop = baseline - shuffled score; mean across repeats; std with ddof=1"),
    (test_pi_seeded, 3,
     "draw every permutation from the rng argument: same seed, same answer"),
    (test_pi_rejects, 2,
     "n_repeats below 2, a mismatched y and an out-of-range column are ValueErrors; 2 is legal"),
    (test_grouped_joint_permutation, 5,
     "ONE permutation per group per repeat, applied to every column of the group"),
    (test_grouped_values, 4,
     "same arithmetic as exercise 1, one row of drops per group"),
    (test_grouped_seeded, 2,
     "use the rng you were handed and nothing else"),
    (test_grouped_rejects, 3,
     "an empty group, an out-of-range index and n_repeats below 2 are ValueErrors; 2 is legal"),
    (test_pd_matches_reference, 4,
     "for each grid value: copy, set the column in every row, predict, average"),
    (test_pd_average_of_predictions, 2,
     "average the predictions; do not predict at the average row"),
    (test_pd_untouched_input, 2,
     "work on a copy of X"),
    (test_pd_rejects, 2,
     "an out-of-range feature and an empty grid are ValueErrors; one grid value is legal"),
    (test_pd2d_orientation, 3,
     "entry [a, b] belongs to grids[0][a] and grids[1][b]; shape (len(g1), len(g2))"),
    (test_pd2d_interaction, 4,
     "set both features in the same copy before predicting"),
    (test_pd2d_rejects_and_untouched, 3,
     "refuse the same feature twice and out-of-range indices; never change X"),
    (test_manifold_threshold, 3,
     "threshold = -2 ln(1 - level), the 2-D normal's squared radius"),
    (test_manifold_per_grid_reference, 5,
     "Mahalanobis distance with the inverse of the pair's sample covariance"),
    (test_manifold_data_share_and_overall, 3,
     "data_share over the real rows; overall is the mean of per_grid"),
    (test_manifold_partner_values, 1,
     "use partner_values when they are given"),
    (test_manifold_sample_covariance, 2,
     "variances and covariance with the n - 1 divisor, as the docstring says"),
    (test_manifold_rejects, 2,
     "level outside (0, 1), feature == partner and an empty grid are ValueErrors"),
    (test_value_empty_and_full, 4,
     "value(()) is the mean prediction over the background rows (one row is legal); "
     "value(all) is model(x)"),
    (test_value_mixed_coalitions, 3,
     "overwrite the coalition's columns with x in every background row"),
    (test_value_no_hidden_state, 3,
     "copy the background inside every call; return a float"),
    (test_shapley_matches_definition, 6,
     "weight |S|!(p-|S|-1)!/p! on v(S + {i}) - v(S), over every S without i"),
    (test_shapley_axioms, 4,
     "symmetry, the null player and efficiency all follow from the right weights"),
    (test_shapley_one_call_per_coalition, 4,
     "evaluate each of the 2^p coalitions once, as an ascending tuple, and look it up"),
    (test_shapley_linear_closed_form, 2,
     "a linear value function has Shapley values w_i (x_i - b_i)"),
    (test_shapley_rejects, 2,
     "n_features below 1 is a ValueError"),
    (test_canonical_roundtrip, 3,
     "convert numpy to plain Python, then JSON; floats by their exact repr"),
    (test_canonical_key_order, 3,
     "sort_keys at every level"),
    (test_canonical_exact_floats, 3,
     "never round and never cast to float32"),
    (test_canonical_rejects_nonfinite, 2,
     "NaN and infinity are ValueErrors (allow_nan=False)"),
    (test_canonical_line_per_leaf, 3,
     "an indent puts every scalar on its own line"),
    (test_repro_clean_certificate, 4,
     "certificate: reproduced, seed, sha256 and n_bytes of the pack in hand, hash_seeds, runs, "
     "first_difference, environment"),
    (test_repro_detects_set_order, 5,
     "regenerate with regenerate_in_fresh_process, once per hash seed you were given"),
    (test_repro_detects_last_bit, 3,
     "compare bytes with ==, never within a tolerance"),
    (test_repro_first_difference_exact, 4,
     "the first mismatching run in order; the first differing line, counted from 1"),
    (test_repro_compares_with_pack_in_hand, 3,
     "compare every fresh run against serialise(build(seed)) built in this process"),
    (test_repro_failed_regeneration, 2,
     "never catch a failed regeneration and carry on"),
    (test_repro_no_false_alarm, 2,
     "compare the serialiser's bytes, not repr() or pickle"),
    (test_repro_rejects_one_seed, 1,
     "fewer than two distinct hash seeds is a ValueError; exactly two is enough"),
]
