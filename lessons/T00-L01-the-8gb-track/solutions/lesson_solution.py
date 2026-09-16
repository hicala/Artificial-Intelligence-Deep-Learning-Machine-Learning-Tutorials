# %% [markdown]
# # T00-L01 · The 8 GB track: measure before you believe — REFERENCE SOLUTION
#
# Every stub filled in. Runs top to bottom in a fresh interpreter and prints real measured
# numbers. Self-check answers are in the comment block at the foot of this file.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import mmap
import resource
import subprocess
import sys
import time
import tracemalloc
from typing import Any, Callable, Mapping, NamedTuple

import numpy as np

MIB = 1024 * 1024
_LESSON_T0 = time.perf_counter()
print("python", sys.version.split()[0], "· numpy", np.__version__, "· platform", sys.platform)

# The two clocks this lesson is careful to keep apart, described by the interpreter itself
# rather than by me. One of them cannot go backwards; the other one can.
_PERF, _WALL = time.get_clock_info("perf_counter"), time.get_clock_info("time")
print(f"perf_counter: monotonic={_PERF.monotonic}, resolution {_PERF.resolution:.0e}s"
      f"  ·  time(): monotonic={_WALL.monotonic}")

# `ru_maxrss` is not in the same unit everywhere: the Linux man page documents kibibytes and
# the macOS and FreeBSD man pages say kilobytes, while a Darwin kernel is widely reported to
# hand back bytes. Guessing is a good way to be wrong by a factor of 1024, so the next section
# takes nobody's word for it — mine included. It calibrates this divisor against an allocation
# of known size, and says out loud whether the calibration held.
RU_MAXRSS_DIVISOR = MIB if sys.platform == "darwin" else 1024


def rss_hwm_mib() -> float:
    """Peak resident set size of THIS process so far, in MiB.

    This is a high-water mark: it only ever goes up. Freeing memory does not lower it.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RU_MAXRSS_DIVISOR


def touch_mmap(mib: int) -> int:
    """Map `mib` MiB of anonymous memory and write to every page, so it is really resident.

    The pages come from the operating system, never from Python's allocator, which is the
    whole point: `tracemalloc` cannot see them and the OS charges you for all of them.
    """
    page = b"\xff" * MIB
    buffer = mmap.mmap(-1, mib * MIB)
    try:
        for _ in range(mib):
            buffer.write(page)
        return len(buffer)
    finally:
        buffer.close()


_FAILED_CHECKS: list[str] = []


def _try(label: str, check: Callable[[], None]) -> None:
    """Run a check, or a demo that depends on your code, without derailing the notebook.

    A stub you have not filled in yet simply says so. A wrong answer prints the check's own
    message — which names the likely mistake — and the notebook carries on to the next cell,
    so one broken exercise never hides the feedback on the other three.

    Nothing is swallowed: every failure is recorded, and the `__main__` block at the foot of
    this file exits non-zero if the list is not empty. That is what stops tools/execute.py
    from ever reporting PASS on a reference implementation that is quietly wrong.
    """
    try:
        check()
    except NotImplementedError:
        print(f"{label}: not implemented yet — fill in the stub above, then re-run this cell.")
    except AssertionError as exc:
        _FAILED_CHECKS.append(label)
        print(f"{label}: FAILED — {exc}")
    except Exception as exc:  # a half-finished implementation raising something else
        _FAILED_CHECKS.append(label)
        print(f"{label}: raised {type(exc).__name__}: {exc}")


print(f"this process has already peaked at {rss_hwm_mib():.1f} MiB just by starting up")

# %% [markdown]
# ## 1. The phenomenon: two honest tools that disagree

# %%
tracemalloc.start()
tracemalloc.reset_peak()
_rss_before = rss_hwm_mib()
_mapped = touch_mmap(256)
_py_peak = tracemalloc.get_traced_memory()[1] / MIB
_rss_after = rss_hwm_mib()
tracemalloc.stop()

print(f"asked the OS for        {_mapped / MIB:.1f} MiB, and wrote to every page")
print(f"tracemalloc saw         {_py_peak:.2f} MiB")
print(f"the OS high-water mark  rose by {_rss_after - _rss_before:.1f} MiB")

_calibration_error = abs((_rss_after - _rss_before) - _mapped / MIB)
print(f"\ncalibration: divisor {RU_MAXRSS_DIVISOR} reproduces a known {_mapped / MIB:.0f} MiB "
      f"allocation to within {_calibration_error:.1f} MiB")
print("  → the unit is right on this machine" if _calibration_error < 32 else
      "  → the unit is WRONG on this machine: every MiB figure below is off by a constant\n"
      "    factor, so fix RU_MAXRSS_DIVISOR before believing anything this notebook prints")

# %% [markdown]
# ## 2. Exercise 1 — `measure()` (solved)

# %%
class Measurement(NamedTuple):
    """What one run of one callable cost."""

    result: Any           # whatever fn returned, so measuring is never in the way
    wall_s: float         # elapsed seconds on a monotonic clock
    py_peak_mib: float    # peak memory Python's own allocator held during the call
    rss_hwm_mib: float    # the whole process's high-water RSS after the call, absolute
    rss_delta_mib: float  # how much that high-water mark rose during the call, never negative


def measure(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Measurement:
    """Call `fn(*args, **kwargs)` and report what it cost.

    Reference implementation. Three deliberate choices:

    * `time.perf_counter()` rather than `time.time()`: it is monotonic, so a clock adjustment
      mid-call cannot produce a negative duration.
    * `reset_peak()` before the call and `stop()` after it. Either one alone would pass the
      rubric — `start()` on a stopped tracer already begins from zero — but together they make
      the guarantee unconditional: this is this call's peak, whatever was traced earlier and
      whether or not tracing happened to be running already when it was called.
    * the RSS reading is taken before *and* after, and the rise is clamped at 0.0, because a
      high-water mark cannot fall and a call that sets no new record legitimately shows 0.
      `rss_hwm_mib` is the reading taken AFTER; only `rss_delta_mib` is a difference.
    * `try`/`finally`, never `try`/`except`: whatever `fn` raises comes straight back out, so
      a measured call can stand in for the plain one, and the clean-up still runs on the way.
    """
    tracemalloc.start()
    tracemalloc.reset_peak()
    rss_before = rss_hwm_mib()
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        wall = time.perf_counter() - t0
        py_peak_bytes = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    rss_after = rss_hwm_mib()
    return Measurement(
        result=result,
        wall_s=wall,
        py_peak_mib=py_peak_bytes / MIB,
        rss_hwm_mib=rss_after,
        rss_delta_mib=max(0.0, rss_after - rss_before),
    )


# Public checks — run these as often as you like.
def _check_measure() -> None:
    passthrough = measure(sum, [1, 2, 3])
    assert isinstance(passthrough, Measurement), (
        "measure must return a Measurement, not a tuple, dict or bare number — "
        "build it with Measurement(result=..., wall_s=..., ...)"
    )
    assert passthrough.result == 6, (
        "measure lost the return value: call fn(*args, **kwargs), keep what it returns, "
        "and put it in Measurement.result"
    )

    slow = measure(time.sleep, 0.25)
    assert 0.2 <= slow.wall_s <= 1.5, (
        f"measure(time.sleep, 0.25) reported {slow.wall_s:.3f}s — near zero means you are "
        "timing the wrong thing; ~250 means you are reporting milliseconds, not seconds"
    )

    big = measure(bytearray, 48 * MIB)
    assert 40.0 <= big.py_peak_mib <= 80.0, (
        f"a 48 MiB bytearray showed as {big.py_peak_mib:.1f} MiB — are you dividing bytes by "
        "1024*1024, and reading the PEAK (second element) of get_traced_memory()?"
    )

    small = measure(bytearray, 1024)
    assert small.py_peak_mib < 5.0, (
        f"a 1 KiB allocation reported {small.py_peak_mib:.1f} MiB, which is the previous "
        "call's peak — reset it with tracemalloc.reset_peak() once tracing is on, and call "
        "tracemalloc.stop() when the call returns"
    )

    assert big.rss_hwm_mib > 5.0, (
        "rss_hwm_mib is the whole process's peak, so it is tens of MiB even for a tiny call; "
        "a near-zero value means you subtracted a baseline that belongs in rss_delta_mib"
    )
    assert small.rss_delta_mib >= 0.0, "rss_delta_mib must be clamped at 0.0, never negative"

    class _Boom(RuntimeError):
        pass

    def _explode() -> None:
        raise _Boom("this call was meant to fail")

    try:
        measure(_explode)
    except _Boom:
        pass
    else:
        raise AssertionError(
            "measure() swallowed the exception _explode raised and returned a Measurement "
            "anyway — that reports a cost for work that never finished. Catch nothing; put "
            "the clock and the tracemalloc clean-up in a finally block instead"
        )
    print("exercise 1 looks right — measure() reports the clock, Python's peak and the OS peak")


# %%
_try("exercise 1", _check_measure)

# %% [markdown]
# ### What `measure()` can and cannot tell you

# %%
def _show_high_water_trap() -> None:
    asked = 512
    first = measure(touch_mmap, asked)
    second = measure(touch_mmap, asked)
    print(f"first  touch_mmap({asked}): rose {first.rss_delta_mib:6.1f} MiB "
          f"(process peak now {first.rss_hwm_mib:.0f} MiB)")
    print(f"second touch_mmap({asked}): rose {second.rss_delta_mib:6.1f} MiB "
          f"(process peak now {second.rss_hwm_mib:.0f} MiB)")
    print(f"\nsame work, two answers, and neither of them is {asked} MiB:")
    print("  the first understates it, because the mark did not start from the floor;")
    print("  the second reports nothing at all, because it set no new record.")
    print("a high-water mark measures records, not usage — which is why the repository's")
    print("gate runs every lesson in a FRESH process, where the record starts at the floor.")


_try("high-water demo", _show_high_water_trap)

# %%
def _show_blind_spot() -> None:
    m = measure(touch_mmap, 256)
    print(f"asked the OS for      {m.result / MIB:.0f} MiB, and wrote to every page")
    print(f"tracemalloc saw       {m.py_peak_mib:.2f} MiB of it")
    print(f"process high-water is {m.rss_hwm_mib:.0f} MiB")
    print(f"the call took         {m.wall_s * 1000:.0f} ms")
    print("\nwhat tracemalloc did see is the reusable page buffer inside touch_mmap, which is")
    print("a genuine Python bytes object. The mapping itself never reached Python's allocator,")
    print("so tracemalloc cannot report it — and the gate charges you for it regardless.")


_try("blind-spot demo", _show_blind_spot)

# %% [markdown]
# ## 3. The floor you never asked for
#
# Measured where the repository's own gate measures it: in a fresh child interpreter.
# `tools/execute.py` reads the child's own `RUSAGE_SELF` from inside it; from out here the
# same peak arrives through `RUSAGE_CHILDREN`.

# %%
def measure_subprocess(snippet: str) -> tuple[float, float]:
    """Run `snippet` in a fresh interpreter; return (wall seconds, peak child RSS in MiB).

    The figure returned is the absolute high-water mark over all children, not the rise
    during this one. A difference would be the honest number if the mark could fall, and it
    cannot: that is the trap this whole section is about, one level up. Running the snippets
    in ascending order of cost is what keeps each reading its own.
    """
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
    wall = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if proc.returncode != 0:
        raise RuntimeError(f"child failed: {proc.stderr.strip().splitlines()[-1:]}")
    return wall, after / RU_MAXRSS_DIVISOR


for _label, _snippet in (("a bare interpreter", "pass"),
                         ("+ import numpy", "import numpy"),
                         ("+ import numpy, json, urllib.request",
                          "import numpy, json, urllib.request")):
    _wall, _peak = measure_subprocess(_snippet)
    print(f"{_label:38s} {_peak:7.1f} MiB   {_wall * 1000:5.0f} ms to start")

# %% [markdown]
# ## 4. Exercises 2 and 3 — profile one task, four ways (solved)

# %%
N_DEMO = 4_000_000


def exact_sum_squares(n: int) -> int:
    """Closed form, for checking the others: the sum of i*i for i in range(n)."""
    return (n - 1) * n * (2 * n - 1) // 6


def sum_squares_list(n: int) -> int:
    """Materialise every square in a list, then add them up."""
    return sum([i * i for i in range(n)])


def sum_squares_generator(n: int) -> int:
    """Same arithmetic, one square alive at a time."""
    return sum(i * i for i in range(n))


def sum_squares_numpy_whole(n: int) -> int:
    """One big int64 array, squared and summed by numpy."""
    values = np.arange(n, dtype=np.int64)
    return int((values * values).sum())


_truth = exact_sum_squares(N_DEMO)
print(f"int64 holds up to   {np.iinfo(np.int64).max}")
print(f"the exact answer is {_truth}\n")
print(f"{'implementation':22s} {'seconds':>8s} {'returns':>21s} {'exact?':>7s}")
for _name, _impl in (("list of squares", sum_squares_list),
                     ("generator", sum_squares_generator),
                     ("numpy, whole array", sum_squares_numpy_whole)):
    _t0 = time.perf_counter()
    _got = _impl(N_DEMO)
    _dt = time.perf_counter() - _t0
    print(f"{_name:22s} {_dt:8.3f} {_got:>21} {str(_got == _truth):>7s}")

# %%
def chunk_bounds(n: int, chunk: int) -> list[tuple[int, int]]:
    """Split range(n) into consecutive half-open [start, stop) slices of at most `chunk`.

    Reference implementation. `min(start + chunk, n)` is what makes the final slice short
    rather than running past the end, and `max(n, 0)` keeps a negative n from producing
    slices at all.
    """
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    return [(start, min(start + chunk, n)) for start in range(0, max(n, 0), chunk)]


def sum_squares_chunked(n: int, chunk: int = 500_000) -> int:
    """Exact sum of i*i for i in range(n), never holding more than `chunk` elements.

    Reference implementation. `total` is a Python int and each slice total is converted with
    `int(...)` before it is added, so the running total is never held in an int64 that would
    wrap. Only one slice is alive at a time, so memory is bounded by `chunk`, not by `n`.
    """
    total = 0
    for start, stop in chunk_bounds(n, chunk):
        values = np.arange(start, stop, dtype=np.int64)
        total += int((values * values).sum())
    return total


def profile(impls: Mapping[str, Callable[[int], int]], n: int) -> dict[str, Measurement]:
    """Measure every implementation in `impls` on the same input `n`.

    Reference implementation. `measure(fn, n)` already forwards the argument and keeps the
    return value, so the whole job is one dict comprehension.
    """
    return {name: measure(fn, n) for name, fn in impls.items()}


def _check_chunking() -> None:
    assert chunk_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)], (
        "chunk_bounds(10, 4) must be [(0, 4), (4, 8), (8, 10)] — the last slice is short, "
        "and stop is exclusive"
    )
    assert chunk_bounds(0, 4) == [], "chunk_bounds(0, 4) is an empty list, not [(0, 0)]"
    assert chunk_bounds(8, 4) == [(0, 4), (4, 8)], (
        "when chunk divides n exactly there is no extra empty slice at the end"
    )
    try:
        chunk_bounds(10, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("chunk_bounds(10, 0) must raise ValueError, not loop forever")

    assert sum_squares_chunked(7, chunk=3) == 91, (
        "sum_squares_chunked(7, chunk=3) should be 91 — do your slices cover range(n) "
        "exactly once?"
    )
    assert sum_squares_chunked(0) == 0, "an empty range sums to 0, not an error"
    big = sum_squares_chunked(N_DEMO)
    assert big == exact_sum_squares(N_DEMO), (
        f"off by {big - exact_sum_squares(N_DEMO)} at n={N_DEMO}: each slice's sum is fine, "
        "but the running total must be a Python int — add int(slice_total) to an int, do not "
        "accumulate into a numpy int64"
    )
    print("exercise 2 looks right — the chunked sum is exact at a size where numpy overflows")


def _check_profile() -> None:
    table = profile({"exact": exact_sum_squares, "generator": sum_squares_generator}, 7)
    assert set(table) == {"exact", "generator"}, (
        "profile must return one entry per implementation, keyed by the same names"
    )
    assert all(isinstance(m, Measurement) for m in table.values()), (
        "each value must be the Measurement returned by measure(), not just a duration"
    )
    assert table["exact"].result == 91, (
        "profile must call each implementation with n and keep its result — "
        "measure(fn, n) already does that for you"
    )
    print("exercise 3 looks right — profile() returns a Measurement per implementation")


# %%
_try("exercise 2", _check_chunking)
_try("exercise 3", _check_profile)

# %%
def _show_tradeoff() -> None:
    impls = {
        "list of squares": sum_squares_list,
        "generator": sum_squares_generator,
        "numpy, whole array": sum_squares_numpy_whole,
        "numpy, chunked": sum_squares_chunked,
    }
    table = profile(impls, N_DEMO)
    truth = exact_sum_squares(N_DEMO)
    print(f"{'implementation':22s} {'seconds':>8s} {'py peak MiB':>12s} {'exact?':>7s}")
    for name, m in table.items():
        print(f"{name:22s} {m.wall_s:8.3f} {m.py_peak_mib:12.1f} {str(m.result == truth):>7s}")
    fastest = min((m.wall_s, name) for name, m in table.items() if m.result == truth)
    leanest = min((m.py_peak_mib, name) for name, m in table.items() if m.result == truth)
    print(f"\nfastest correct: {fastest[1]} at {fastest[0]:.3f}s")
    print(f"leanest correct: {leanest[1]} at {leanest[0]:.1f} MiB")
    whole, chunked = table["numpy, whole array"], table["numpy, chunked"]
    print(f"whole array vs chunked: {whole.py_peak_mib / max(chunked.py_peak_mib, 1e-9):.1f}x "
          f"the memory, {whole.wall_s / max(chunked.wall_s, 1e-9):.1f}x the time, and "
          f"exact={whole.result == truth} against exact={chunked.result == truth}")


_try("trade-off table", _show_tradeoff)

# %% [markdown]
# ### The profiler is not free

# %%
def _show_observer_effect() -> None:
    t0 = time.perf_counter()
    sum_squares_list(N_DEMO)
    untraced = max(time.perf_counter() - t0, 1e-9)
    traced = measure(sum_squares_list, N_DEMO).wall_s
    print(f"sum_squares_list, tracemalloc off: {untraced:.3f}s")
    print(f"sum_squares_list, tracemalloc on:  {traced:.3f}s")
    print(f"tracing cost a factor of {traced / untraced:.1f} on this workload")
    print("\nthe ranking in the table above survives this; the absolute seconds do not.")
    print("tools/execute.py times a lesson with tracing OFF, in a fresh process, which is why")
    print("its wall-clock number — not this one — is what the budget is written against.")


_try("observer effect", _show_observer_effect)

# %% [markdown]
# ## 5. Exercise 4 — `tier_check()`, the gate itself (solved)

# %%
TIER_MIB = {
    "phone": 2048,
    "browser": 2048,
    "cpu8": 8192,
    "free-gpu": 16384,
    "gpu24": 24576,
    "gpu80": 81920,
    "multi-gpu": 163840,
    "api": 8192,
}


class Verdict(NamedTuple):
    """The gate's answer: did it pass, and if not, every reason why not."""

    passed: bool
    reasons: tuple[str, ...]


def tier_check(measured: Mapping[str, float], tier: str, budget_seconds: float) -> Verdict:
    """Decide whether a measured run fits its declared tier and budget.

    Reference implementation. Two details carry the whole exercise:

    * the unknown-tier case returns immediately, because without a ceiling there is nothing
      to compare memory against and a second reason would be invented rather than measured;
    * every other rule appends to `reasons` rather than returning, so an author who is over
      on both axes is told about both instead of fixing one and re-running to find the other.

    The comparisons are `>`, so a run that lands exactly on a limit passes.
    """
    ceiling = TIER_MIB.get(tier)
    if ceiling is None:
        return Verdict(False, (f"unknown tier {tier!r}; declare one of {sorted(TIER_MIB)}",))

    wall = float(measured["wall_s"])
    peak = float(measured["peak_mib"])
    reasons: list[str] = []
    if budget_seconds <= 0:
        reasons.append(f"budget_seconds must be positive, got {budget_seconds}")
    elif wall > budget_seconds:
        # `:.3g`, not `:.1f`: this gate is also run against sub-second budgets, where a fixed
        # one-decimal format reports "0.2s exceeds 0.1s" for 0.172s against 0.05s. Bare `:g`
        # fixes that but spends six significant figures on it — "0.172034s exceeds 0.05s".
        reasons.append(f"wall {wall:.3g}s exceeds budget {budget_seconds:.3g}s")
    if peak > ceiling:
        reasons.append(f"peak {peak:.0f}MiB exceeds {tier} ceiling {ceiling}MiB")
    return Verdict(not reasons, tuple(reasons))


def _check_tier_check() -> None:
    ok = tier_check({"wall_s": 12.0, "peak_mib": 900.0}, "cpu8", 600)
    assert ok.passed and ok.reasons == (), (
        "a run well inside its tier must pass with an EMPTY reasons tuple"
    )
    edge = tier_check({"wall_s": 600.0, "peak_mib": 8192.0}, "cpu8", 600)
    assert edge.passed, (
        "exactly at the limit passes: compare with > (over), not >= (at or over)"
    )
    unknown = tier_check({"wall_s": 1.0, "peak_mib": 1.0}, "cpu9000", 600)
    assert not unknown.passed and len(unknown.reasons) == 1, (
        "an unknown tier fails with exactly one reason and stops there — there is no ceiling "
        "to compare memory against"
    )
    both = tier_check({"wall_s": 900.0, "peak_mib": 9000.0}, "cpu8", 600)
    assert not both.passed and len(both.reasons) == 2, (
        f"over on both axes should give 2 reasons, got {len(both.reasons)}: append one reason "
        "per broken rule instead of returning at the first one"
    )
    assert "8192" in both.reasons[1], (
        "the memory reason must name the ceiling it broke, so the author can see the gap"
    )
    print("exercise 4 looks right — tier_check reproduces the gate, boundary included")


# %%
_try("exercise 4", _check_tier_check)

# %%
def _gate_this_lesson() -> None:
    elapsed = time.perf_counter() - _LESSON_T0
    peak = rss_hwm_mib()
    verdict = tier_check({"wall_s": elapsed, "peak_mib": peak}, "cpu8", 90)
    print(f"this notebook so far: {elapsed:.1f}s and {peak:.0f} MiB peak")
    print(f"tier_check says: passed={verdict.passed} reasons={verdict.reasons}")


_try("gate on this lesson", _gate_this_lesson)

# %% [markdown]
# ## 6. Why the budget does not move

# %%
def _show_why_the_budget_does_not_move() -> None:
    n = N_DEMO
    budget = 0.05
    print(f"n={n}, judged against the phone tier ({TIER_MIB['phone']} MiB) "
          f"and a {budget}s budget.")
    print("timed with tracing OFF, the way tools/execute.py times a lesson; the memory figure")
    print("comes from a separate traced run, because there is no per-call RSS number to take —")
    print("a high-water mark belongs to the process, not to the call.\n")
    answers = []
    for name, fn in (("as written (list)", sum_squares_list),
                     ("rewritten (chunked)", sum_squares_chunked)):
        t0 = time.perf_counter()
        answers.append(fn(n))
        untraced = time.perf_counter() - t0
        py_peak = measure(fn, n).py_peak_mib
        verdict = tier_check({"wall_s": untraced, "peak_mib": py_peak}, "phone", budget)
        print(f"  {name:22s} {untraced:6.3f}s {py_peak:7.1f} MiB  passed={verdict.passed}")
        for reason in verdict.reasons:
            print(f"  {'':22s}   {reason}")
    print(f"\nboth return the same answer: {answers[0] == answers[1]}")
    print("the rewrite passes a gate the original fails, and costs the student nothing.")
    print("a bigger budget would have hidden that difference instead of finding it.")


_try("budget versus rewrite", _show_why_the_budget_does_not_move)

# %%
def _show_current_versus_peak() -> None:
    tracemalloc.start()
    tracemalloc.reset_peak()
    blob = bytearray(32 * MIB)
    alive = tracemalloc.get_traced_memory()
    del blob
    freed = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"while the buffer is alive: current={alive[0] / MIB:6.1f} MiB  peak={alive[1] / MIB:6.1f} MiB")
    print(f"after it is thrown away:   current={freed[0] / MIB:6.1f} MiB  peak={freed[1] / MIB:6.1f} MiB")
    print("\n[0] is `current` and forgets; [1] is `peak` and remembers. A profiler that reads")
    print("[0] after the call has returned will report that this function cost nothing at all.")


_show_current_versus_peak()

# %% [markdown]
# ## Self-check answers
#
# 1. **(b)** — `ru_maxrss` is a high-water mark for the whole process. A 300 MiB allocation
#    that does not beat an earlier peak moves nothing, and reports a rise of 0.0. It does not
#    mean the memory was free; it means the record was already higher. (a) is wrong because a
#    high-water mark survives the memory being freed — that is what makes it a high-water
#    mark. (c) is wrong because `ru_maxrss` counts every resident page regardless of who
#    allocated it; it is `tracemalloc` that is selective, not the OS.
#
# 2. **(c)** — `tracemalloc` traces memory blocks allocated by Python. Pages mapped straight
#    from the OS never pass through those allocators, so they are invisible to it while the
#    kernel still charges every one of them to the process. (b) is refuted by the measurement
#    itself: the RSS high-water mark rose by the full amount, which only happens for resident
#    pages. Worth knowing: numpy's own array buffers *are* visible to `tracemalloc`, because
#    numpy routes its allocations into a tracemalloc domain of its own — so "C library, so
#    invisible" is not the rule either. Measure, do not assume.
#
# 3. **(a)** — a profiler answers "what did that cost", never "was that right". Both the time
#    and the memory figures for the whole-array version are perfectly accurate; the answer it
#    returns is simply wrong, because the int64 accumulator wrapped. Only comparing against an
#    independent reference catches it, which is why the lesson computes the closed form.
#
# 4. **(b)** — shrink the problem and re-measure. (a) moves the author's problem onto every
#    student whose machine cannot grow to match, and the gate goes green while the lesson gets
#    less teachable. (c) is (a) with a disclaimer. The budget is a promise, and the rewrite
#    usually improves the lesson anyway — as the chunked implementation did here, by being
#    faster, leaner and more correct than the version it replaced.
#
# 5. **(b)** — the instrument perturbs what it measures. `tracemalloc` records every block
#    Python allocates, so a workload that allocates four million integers pays the tax four
#    million times, while numpy's handful of large buffers barely notice it. The "profiler is
#    not free" cell measures the factor on this machine. The lesson generalises past this
#    notebook: a measurement is only comparable to another taken the same way, which is why
#    the repository's gate times every lesson identically — tracing off, fresh process.
#    (c) is wrong on its own terms: `sum_squares_list` is pure Python and never touches numpy.

# %%
if __name__ == "__main__":
    for _name, _check in (("exercise 1", _check_measure),
                          ("exercise 2", _check_chunking),
                          ("exercise 3", _check_profile),
                          ("exercise 4", _check_tier_check)):
        _try(_name, _check)
    # tools/execute.py runs this file and reads the exit code. Every check and every demo above
    # went through _try, so a reference implementation that is quietly wrong lands here rather
    # than in a PASS line.
    if _FAILED_CHECKS:
        raise SystemExit("checks failed: " + ", ".join(dict.fromkeys(_FAILED_CHECKS)))
