"""Autograder rubric for P04-L06. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure). Partial credit is the point — a
student who has written naive_sum but not the compensated one should see exactly that.

Every expected value in this file comes from a reference implemented HERE, in Python, from
the stated definition — never from the lesson's own helpers and never from a number typed by
the author. `_ref_naive`, `_ref_pairwise` and `math.fsum` are the three references; the first
two are deliberately written against a plain Python list so that a student who redefines
`python_naive_sum` or `generate_portfolio` in lesson.py cannot move them. The array itself is
regenerated here from the same seed and the same exact operations, and one rubric case exists
solely to prove that the C array and this one are the same bits.

A binary that exits 2 has an unfinished stub; lesson.run_c turns that into
NotImplementedError, which the grader prints as TODO rather than as an error.
"""
import importlib.util
import math
import os
import struct
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

N = 2_000_000
BLOCK = 128
SEED = 20260916

_CACHE = {}


def _cached(key, fn):
    """Run the expensive commands once per grading session, not once per test."""
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


# --------------------------------------------------------------------------------------
# Independent references
# --------------------------------------------------------------------------------------

def _reference_portfolio(n=N):
    """The same array, generated here from the same definition, in numpy.

    Every operation is exact: a 53-bit integer into a double, and a scaling by a power of
    two. If this and the C generator ever disagree, one of them departed from the definition.
    """
    k = np.arange(3 * n, dtype=np.uint64)
    z = np.uint64(SEED) + (k + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    u0, u1, u2 = z[0::3].copy(), z[1::3].copy(), z[2::3].copy()
    mant = (u0 >> np.uint64(11)).astype(np.float64)
    e = (u1 % np.uint64(41)).astype(np.int64) - 20
    sign = np.where((u1 >> np.uint64(63)) != np.uint64(0), -1.0, 1.0)
    return sign * np.ldexp(mant, (e - 53).astype(np.int32)), \
        np.ldexp((u2 >> np.uint64(11)).astype(np.float64), np.int32(-53))


def _lists():
    def build():
        ex, pd = _reference_portfolio(N)
        return ex, pd, ex.tolist(), np.abs(ex).tolist()
    return _cached("arrays", build)


def _ref_naive(xs, lo=0, hi=None):
    hi = len(xs) if hi is None else hi
    s = 0.0
    for i in range(lo, hi):
        s += xs[i]
    return s


def _ref_pairwise(xs, lo=0, hi=None):
    """Blocks of 128, split at n // 2 — the convention the exercise states."""
    hi = len(xs) if hi is None else hi
    n = hi - lo
    if n <= 0:
        return 0.0
    if n <= BLOCK:
        return _ref_naive(xs, lo, hi)
    half = n // 2
    return _ref_pairwise(xs, lo, lo + half) + _ref_pairwise(xs, lo + half, hi)


def _key(x):
    (u,) = struct.unpack("<Q", struct.pack("<d", x))
    return -(u & 0x7FFFFFFFFFFFFFFF) if u >> 63 else u


def _ulps(a, b):
    """The rubric's own ULP distance. Never the student's."""
    if a == b:
        return 0
    return abs(_key(a) - _key(b))


def _sums():
    return _cached("sums", lambda: lesson.metrics(lesson.run_c("sums", "--n", str(N))))


def _orders():
    def build():
        text = lesson.run_c("order", "--n", str(N))
        return {int(c): (nv, kh) for c, nv, kh in lesson.rows(text, "order")}
    return _cached("orders", build)


def _reproduce(a, b, n, sum_abs):
    text = lesson.run_c("reproduce", "--a", repr(a), "--b", repr(b), "--n", str(n),
                        "--sum-abs", repr(sum_abs))
    m = lesson.metrics(text)
    m["name"] = next(l.split()[1] for l in text.splitlines()
                     if l.startswith("verdict_name "))
    return m


# --------------------------------------------------------------------------------------
# The rubric
# --------------------------------------------------------------------------------------

def test_c_builds():
    ok, out = lesson.build(verbose=False)
    assert ok, f"the C source did not compile:\n{out}"


def test_c_selftest_passes():
    code, out = _cached("selftest", lesson.make_test)
    if code == 2:
        raise NotImplementedError("at least one C exercise is still a stub")
    assert code == 0, f"the binary's self-test failed with exit {code}:\n{out}"


def test_generated_array_is_identical_in_c_and_python():
    """Nothing else in this rubric is evidence unless both sides hold the same bits."""
    ex, pd, _xs, _mags = _lists()
    block = _cached("dump", lambda: lesson.rows(lesson.run_c("dump", "--dump", "4000"), "x"))
    assert len(block) == 4000, f"the dump command printed {len(block)} rows, expected 4000"
    for i, c_ex, c_pd in block:
        i = int(i)
        assert c_ex == float(ex[i]), (
            f"record {i}: C generated {c_ex!r}, the reference generator {float(ex[i])!r}. "
            "The exposure is sign * ldexp(u0 >> 11, e - 53) with e = (u1 % 41) - 20 and the "
            "sign taken from bit 63 of u1.")
        assert c_pd == float(pd[i]), (
            f"record {i}: C generated pd {c_pd!r}, expected {float(pd[i])!r} = "
            "ldexp(u2 >> 11, -53)")


def test_naive_sum_is_left_to_right_bit_for_bit():
    ex, _pd, xs, mags = _lists()
    m = _sums()
    want = _ref_naive(xs)
    assert m["naive"] == want, (
        f"naive_sum returned {m['naive']!r}; a left-to-right loop over the same array returns "
        f"{want!r}, which is {_ulps(m['naive'], want)} representable doubles away. These must "
        "agree exactly: one accumulator starting at 0.0, one addition per element, in index "
        "order, with no compensation and no partial sums.")
    # The gross total as well: all positive, so nothing cancels and a different bug shows.
    want_gross = _ref_naive(mags)
    assert m["gross_naive"] == want_gross, (
        f"the naive sum of the absolute values returned {m['gross_naive']!r} against a "
        f"left-to-right reference of {want_gross!r} ({_ulps(m['gross_naive'], want_gross)} "
        "doubles away)")
    # And it must NOT be the exactly-rounded answer: a "naive" sum that is exact is not naive.
    exact = math.fsum(xs)
    assert m["naive"] != exact, (
        "naive_sum returned the exactly-rounded total. Left-to-right accumulation over two "
        "million values whose scales span 2^-20 to 2^20 does not land there; you have "
        "compensated, "
        "sorted or accumulated in something wider than a double.")


def test_pairwise_sum_follows_the_stated_convention():
    _ex, _pd, xs, _mags = _lists()
    m = _sums()
    want = _ref_pairwise(xs)
    assert m["pairwise"] == want, (
        f"pairwise_sum returned {m['pairwise']!r}; the convention in the stub — blocks of "
        f"{BLOCK} summed left to right, otherwise split at half = n / 2 — gives {want!r}, "
        f"which is {_ulps(m['pairwise'], want)} representable doubles away. A different "
        "block size or a split at a power of two is a perfectly good pairwise sum and is "
        "still the wrong answer here: the lesson is that two implementations which do not "
        "agree on the convention do not reproduce each other.")
    assert m["pairwise"] != m["naive"], (
        "pairwise_sum returned the same double as naive_sum; over two million elements the "
        "recursion should be around fourteen levels deep, so they should differ")


def test_pairwise_beats_naive_on_the_hard_array():
    _ex, _pd, xs, _mags = _lists()
    m = _sums()
    exact = math.fsum(xs)
    d_pw, d_nv = _ulps(m["pairwise"], exact), _ulps(m["naive"], exact)
    assert d_pw < d_nv, (
        f"pairwise landed {d_pw} representable doubles from the exactly-rounded total and "
        f"naive landed {d_nv}. A tree of depth log2(n) should beat a chain of length n; "
        "check that the recursion actually splits rather than falling through to the base "
        "case.")


def test_kahan_sum_recovers_what_the_others_lose():
    _ex, _pd, xs, mags = _lists()
    m = _sums()
    exact = math.fsum(xs)
    d = _ulps(m["kahan"], exact)
    assert d <= 2, (
        f"kahan_sum returned {m['kahan']!r}, {d} representable doubles from the "
        f"exactly-rounded total {exact!r}. Two steps is the algorithm; more means the "
        "compensation is overwritten rather than accumulated, or you returned s rather than "
        "s + c.")
    # The gross total is the harder case: 5e10 with nothing cancelling, where naive is
    # thousands of ULPs out. A compensated sum must still be on the money.
    exact_gross = math.fsum(mags)
    dg = _ulps(m["gross_kahan"], exact_gross)
    assert dg <= 2, (
        f"the compensated sum of the absolute values is {dg} doubles from {exact_gross!r}; "
        f"the naive one is {_ulps(m['gross_naive'], exact_gross)} away, so the compensation "
        "is not doing its job")
    assert m["kahan"] != m["naive"], (
        "kahan_sum and naive_sum returned the same double; on this portfolio they cannot")


def test_kahan_handles_an_addend_larger_than_the_running_sum():
    """The case the unbranched 1965 form gets wrong, graded through the binary's self-test."""
    code, out = _cached("selftest", lesson.make_test)
    if code == 2:
        raise NotImplementedError("at least one C exercise is still a stub")
    assert "FAIL  exercise 3" not in out, (
        "the self-test's kahan_sum cases failed. kahan_sum({1, 1e16, -1e16}) must return "
        "exactly 1.0: here the ADDEND is larger than the running sum, which is the case the "
        "original unbranched compensation drops. Branch on fabs(s) >= fabs(x) and compensate "
        f"with (x - t) + s in the other direction.\n{out}")


def test_naive_summation_depends_on_the_order():
    o = _orders()
    assert len(o) == 4, f"the order command printed {len(o)} rows, expected 4"
    naive_vals = {c: v[0] for c, v in o.items()}
    assert len(set(naive_vals.values())) >= 3, (
        f"the four orderings produced only {len(set(naive_vals.values()))} distinct naive "
        f"totals ({sorted(set(naive_vals.values()))}). On two million signed values whose "
        "scales span 2^-20 to 2^20, left-to-right summation is order-dependent and all four "
        "differ.")
    _ex, _pd, xs, _mags = _lists()
    assert naive_vals[0] == _ref_naive(xs), (
        f"the as-generated ordering must reproduce the left-to-right reference {_ref_naive(xs)!r}"
        f"; it returned {naive_vals[0]!r}")
    assert naive_vals[1] == _ref_naive(xs[::-1]), (
        f"the reversed ordering must reproduce a left-to-right sum of the reversed array "
        f"{_ref_naive(xs[::-1])!r}; it returned {naive_vals[1]!r}")


def test_compensated_summation_does_not_depend_on_the_order():
    o = _orders()
    kahan_vals = {c: v[1] for c, v in o.items()}
    assert len(set(kahan_vals.values())) == 1, (
        f"compensated summation gave {len(set(kahan_vals.values()))} different answers over "
        f"the four orderings ({sorted(set(kahan_vals.values()))}); it should give one")
    _ex, _pd, xs, _mags = _lists()
    exact = math.fsum(xs)
    only = next(iter(set(kahan_vals.values())))
    assert _ulps(only, exact) <= 2, (
        f"the compensated total {only!r} is {_ulps(only, exact)} doubles from the "
        f"exactly-rounded {exact!r}; stable and wrong is not the goal")


def test_weighted_mean_inherits_the_summation():
    """Module 1's support-weighted shape, rebuilt on each of the three sums."""
    ex, pd, _xs, mags = _lists()
    block = _cached("wmean",
                    lambda: lesson.rows(lesson.run_c("wmean", "--n", str(N)), "wmean"))
    vals = {int(c): v for c, v in block}
    assert set(vals) == {0, 1, 2}, f"wmean printed codes {sorted(vals)}, expected 0, 1 and 2"
    exact = math.fsum((np.abs(ex) * pd).tolist()) / math.fsum(mags)
    assert _ulps(vals[2], exact) <= 2, (
        f"the compensated weighted mean is {_ulps(vals[2], exact)} doubles from the reference "
        f"{exact!r}; the numerator and the denominator are both sums and both must be right")
    assert _ulps(vals[0], exact) > _ulps(vals[2], exact), (
        f"the naive weighted mean ({vals[0]!r}) is no further from the reference than the "
        f"compensated one ({vals[2]!r}); on this portfolio it should be clearly worse")


def test_reproduce_returns_the_three_verdicts():
    same = _reproduce(1.25, 1.25, 1000, 1.0)
    assert same["verdict"] == 0 and same["gap"] == 0.0, (
        f"two identical figures are REPRODUCED (0) with gap 0.0; you returned "
        f"{same['verdict']:.0f} ({same['name']}) with gap {same['gap']!r}")
    near = _reproduce(1.0, math.nextafter(1.0, 2.0), 1000, 1.0)
    assert near["verdict"] == 1, (
        f"a one-ulp gap ({math.nextafter(1.0, 2.0) - 1.0!r}) against a bound of "
        f"{999 * 2.0 ** -53!r} is EXPLAINED (1); you returned {near['name']}")
    far = _reproduce(1.0, 1.5, 1000, 1.0)
    assert far["verdict"] == 2, (
        f"a gap of 0.5 against a bound of {999 * 2.0 ** -53!r} is a FINDING (2); you returned "
        f"{far['name']}")
    # REPRODUCED must beat EXPLAINED: equal figures, but a bound wide enough to swallow
    # anything. A verdict computed as "gap <= bound first" would report EXPLAINED here.
    both = _reproduce(3.5, 3.5, N, 1e12)
    assert both["verdict"] == 0, (
        f"equal figures are REPRODUCED even when the bound is enormous; you returned "
        f"{both['name']}. Test a == b before you test the bound — the two verdicts say very "
        "different things in a report.")
    nan = _reproduce(math.nan, math.nan, 1000, 1.0)
    assert nan["verdict"] == 2, (
        f"a NaN never reproduces anything — NaN == NaN is false and no comparison against the "
        f"bound succeeds — so the verdict is FINDING (2); you returned {nan['name']}")


def test_reproduce_bound_is_the_worst_case_and_the_boundary_is_inclusive():
    near = _reproduce(1.0, math.nextafter(1.0, 2.0), 1000, 1.0)
    want = 999 * 2.0 ** -53
    assert near["bound"] == want, (
        f"the bound came back {near['bound']!r}; (n - 1) * 2**-53 * sum_abs with n = 1000 and "
        f"sum_abs = 1.0 is {want!r}. Compute it in that order — (double)(n - 1), then the "
        "unit roundoff, then the absolute sum — so that the bound is itself reproducible.")
    one = _reproduce(1.0, math.nextafter(1.0, 2.0), 1, 1e300)
    assert one["bound"] == 0.0 and one["verdict"] == 2, (
        f"with n = 1 there are no additions, so the bound is 0.0 and any difference is a "
        f"finding; you returned bound {one['bound']!r} and verdict {one['name']}")
    zero = _reproduce(1.0, 2.0, 1000, 0.0)
    assert zero["bound"] == 0.0 and zero["verdict"] == 2, (
        f"a portfolio whose absolute values sum to zero cannot explain any gap; expected "
        f"bound 0.0 and FINDING, got {zero['bound']!r} and {zero['name']}")
    # (3 - 1) * 2**-53 * 2**53 is exactly 2.0, with no rounding anywhere.
    edge = _reproduce(0.0, 2.0, 3, 2.0 ** 53)
    assert edge["bound"] == 2.0, f"bound was {edge['bound']!r}, expected exactly 2.0"
    assert edge["verdict"] == 1, (
        f"a gap sitting exactly ON the bound is inside it — the comparison is <=, not < — so "
        f"the verdict is EXPLAINED; you returned {edge['name']}")
    over = _reproduce(0.0, math.nextafter(2.0, 3.0), 3, 2.0 ** 53)
    assert over["verdict"] == 2, (
        f"one representable step past the bound is a FINDING; you returned {over['name']}")
    # sum_abs is a magnitude. Hand the function the SIGNED net total by mistake — the figure
    # sitting right next to it in every reconciliation pack — and a bound without fabs goes
    # negative, `gap <= bound` is false for every possible gap, and every reconciliation ever
    # put to it comes back a finding while looking like it works.
    neg = _reproduce(1.0, math.nextafter(1.0, 2.0), 1000, -1.0)
    assert neg["bound"] == want, (
        f"a sum_abs of -1.0 must give the same bound as +1.0 ({want!r}); you returned "
        f"{neg['bound']!r}. Take fabs(sum_abs) — a bound is a size, and a negative one makes "
        "gap <= bound false for every gap.")
    assert neg["verdict"] == 1, (
        f"with a negative sum_abs the verdict must be unchanged (EXPLAINED); you returned "
        f"{neg['name']}")
    # The bound must scale with sum_abs, not be a constant dressed up as one.
    big = _reproduce(0.0, 4.0, 3, 2.0 ** 54)
    assert big["bound"] == 4.0 and big["verdict"] == 1, (
        f"doubling sum_abs must double the bound: expected 4.0 and EXPLAINED, got "
        f"{big['bound']!r} and {big['name']}")


def test_ulps_between():
    f = lesson.ulps_between
    assert f(1.0, 1.0) == 0, f"equal numbers are zero steps apart; you returned {f(1.0, 1.0)!r}"
    assert f(-0.0, 0.0) == 0, (
        f"-0.0 and +0.0 compare equal, so they are zero steps apart; you returned "
        f"{f(-0.0, 0.0)!r}. Check a == b before you touch the bits.")
    nxt = math.nextafter(1.0, 2.0)
    assert f(1.0, nxt) == 1 and f(nxt, 1.0) == 1, (
        f"1.0 and the next double above it are one step apart, in either order; you returned "
        f"{f(1.0, nxt)!r} and {f(nxt, 1.0)!r}")
    assert f(-0.0, 5e-324) == 1, (
        f"the two doubles either side of zero are one step apart; you returned "
        f"{f(-0.0, 5e-324)!r}. Subtracting the raw bit patterns gives 2**63 here — the key "
        "has to map the sign bit onto a negative number so the line joins up at zero.")
    assert f(-1.0, math.nextafter(-1.0, 0.0)) == 1, (
        f"adjacent NEGATIVE doubles are also one step apart; you returned "
        f"{f(-1.0, math.nextafter(-1.0, 0.0))!r}")
    assert f(-5e-324, 5e-324) == 2, (
        f"the smallest negative subnormal and the smallest positive one have both zeros "
        f"between them, so they are two steps apart; you returned {f(-5e-324, 5e-324)!r}")
    assert isinstance(f(1.0, nxt), int) and not isinstance(f(1.0, nxt), bool), (
        f"the distance must be a plain int; you returned a {type(f(1.0, nxt)).__name__}")
    for args in ((math.nan, 1.0), (1.0, math.nan), (math.nan, math.nan)):
        try:
            f(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ulps_between{args} must raise ValueError, not return "
                                 f"{f(*args)!r}")
    # And it must agree with the rubric's own instrument on a real, large gap.
    _ex, _pd, xs, _mags = _lists()
    m = _sums()
    exact = math.fsum(xs)
    assert f(m["naive"], exact) == _ulps(m["naive"], exact), (
        f"on this lesson's own totals you returned {f(m['naive'], exact)!r} where an "
        f"independent implementation gives {_ulps(m['naive'], exact)} — a key of about "
        "4.6e18 needs 63 bits, so subtract the keys as integers and convert afterwards")


def test_agreeing_significant_digits():
    f = lesson.agreeing_significant_digits
    assert f(1.0, 1.0) == 17, f"identical numbers agree on 17 digits; you returned {f(1.0, 1.0)!r}"
    assert f(-0.0, 0.0) == 17, (
        f"the two zeros are equal, so they agree on everything; you returned {f(-0.0, 0.0)!r}")
    assert f(1.2345678901234567, 1.2345699999999999) == 6, (
        f"the docstring's worked example must return 6; you returned "
        f"{f(1.2345678901234567, 1.2345699999999999)!r}")
    assert f(-1.0, 1.0) == 0, f"opposite signs agree on nothing; you returned {f(-1.0, 1.0)!r}"
    assert f(-1.5, -1.5000000000000002) == 16, (
        f"two adjacent NEGATIVE doubles agree on 16 of their 17 digits; you returned "
        f"{f(-1.5, -1.5000000000000002)!r} — strip the minus sign before counting, but only "
        "after checking that both signs match")
    assert f(9.999, 10.001) == 0, (
        f"different decimal exponents return 0; you returned {f(9.999, 10.001)!r}")
    # 1.0 and 10.0 have IDENTICAL 17-digit mantissas. Skip the exponent test and this pair
    # comes back as a perfect match, which would let a tool report that a figure and the same
    # figure off by a factor of ten agree to every digit it can print.
    assert f(1.0, 10.0) == 0, (
        f"1.0 and 10.0 agree on nothing; you returned {f(1.0, 10.0)!r}. Their mantissa digit "
        "strings are identical — only the exponent differs — so the exponents have to be "
        "compared before the digits are.")
    assert f(1.5, 0.15) == 0, (
        f"1.5 and 0.15 agree on nothing; you returned {f(1.5, 0.15)!r}")
    assert f(1.0, math.nextafter(1.0, 2.0)) == 16, (
        f"1.0 and the next double above it share 16 of 17 digits; you returned "
        f"{f(1.0, math.nextafter(1.0, 2.0))!r}")
    assert f(math.inf, 1e308) == 0, (
        f"an infinity agrees with nothing finite; you returned {f(math.inf, 1e308)!r}")
    assert f(math.inf, math.inf) == 17, (
        f"two infinities are equal, so 17; you returned {f(math.inf, math.inf)!r}")
    for args in ((math.nan, 1.0), (1.0, math.nan)):
        try:
            f(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"agreeing_significant_digits{args} must raise ValueError")
    # The real pair: the same portfolio summed forwards and backwards.
    o = _orders()
    fwd, rev = o[0][0], o[1][0]
    ma, mb = f"{fwd:.16e}".split("e")[0], f"{rev:.16e}".split("e")[0]
    want = 0
    for ca, cb in zip(ma.replace(".", ""), mb.replace(".", "")):
        if ca != cb:
            break
        want += 1
    assert f(fwd, rev) == want, (
        f"on the lesson's own two totals ({fwd!r} and {rev!r}) you returned {f(fwd, rev)!r} "
        f"where counting the leading matching digits of the two 17-digit mantissas gives "
        f"{want}")


RUBRIC = [
    (test_c_builds, 3,
     "run `make` in the lesson directory and read the first compiler error"),
    (test_c_selftest_passes, 5,
     "run `make test` — it names the unfinished exercise and what it expects"),
    (test_generated_array_is_identical_in_c_and_python, 4,
     "the generator is given; if this fails, lesson.c's generate() has been edited"),
    (test_naive_sum_is_left_to_right_bit_for_bit, 7,
     "one accumulator from 0.0, one addition per element, index order, no compensation"),
    (test_pairwise_sum_follows_the_stated_convention, 8,
     "blocks of 128 summed left to right; otherwise half = n / 2 and recurse on both sides"),
    (test_pairwise_beats_naive_on_the_hard_array, 3,
     "if the recursion never splits, pairwise IS naive — check the n <= PAIRWISE_BLOCK test"),
    (test_kahan_sum_recovers_what_the_others_lose, 6,
     "accumulate the compensation with c += ..., and return s + c rather than s"),
    (test_kahan_handles_an_addend_larger_than_the_running_sum, 4,
     "branch on fabs(s) >= fabs(x); the other branch compensates with (x - t) + s"),
    (test_naive_summation_depends_on_the_order, 4,
     "the four rows come from your naive_sum over four permutations of one array"),
    (test_compensated_summation_does_not_depend_on_the_order, 4,
     "a compensated sum that moves with the order is dropping its correction"),
    (test_weighted_mean_inherits_the_summation, 4,
     "the weighted mean is given; it fails only when the sums under it are wrong"),
    (test_reproduce_returns_the_three_verdicts, 7,
     "test a == b first, then gap <= bound, then fall through to a finding"),
    (test_reproduce_bound_is_the_worst_case_and_the_boundary_is_inclusive, 5,
     "(double)(n - 1) * 0x1p-53 * fabs(sum_abs), 0.0 when n <= 1, compared with <="),
    (test_ulps_between, 6,
     "map the sign bit onto a negative key, return 0 when a == b, raise ValueError on NaN"),
    (test_agreeing_significant_digits, 5,
     "equality first, then infinities, then sign, then exponent, then the 17 mantissa digits"),
]
