"""Autograder rubric for T00-L01. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Three rules matter in this particular lesson and are deliberate:

* `test_measure_rss_high_water` must allocate more than anything else that has run in the
  grading process, or a correct implementation would report a rise of 0.0 — the very trap the
  lesson teaches. It therefore reads the current high-water mark and sizes its own allocation
  against it, instead of trusting a hard-coded figure to stay the largest one.
* memory tolerances are ranges, not equalities. Interpreter overhead is real and varies.
* `test_measure_uses_a_monotonic_clock` swaps the lesson module's `time` for a namespace whose
  `time()` is frozen. `perf_counter` and `monotonic` still work, so only an implementation that
  timed with the wall clock — the trap the docstring warns about — measures zero and fails.
* the rubric grades observable behaviour, never one particular call. `reset_peak()` and a
  `stop()` after every call are two routes to the same guarantee — this call's peak is not the
  last call's — and `test_measure_peak_resets_between_calls` accepts either. Failing a correct
  implementation for choosing the other route would be a bug in the grader, not in the answer.
* `test_measure_delta_is_clamped` injects a falling reading. A real high-water mark never
  falls, so the clamp the exercise asks for would otherwise be ungradeable — stated in the
  docstring, enforced nowhere. An implementation that reads the OS directly rather than
  through the lesson's helper is untouched by the injection and still passes.
"""
import importlib.util, os, time, types
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

MIB = 1024 * 1024


def _exact(n: int) -> int:
    """Independent closed form, so the rubric never grades the student against their own code."""
    return (n - 1) * n * (2 * n - 1) // 6


# --- exercise 1: measure() -------------------------------------------------------------

def test_measure_shape():
    m = lesson.measure(sum, [1, 2, 3])
    assert isinstance(m, lesson.Measurement), f"expected a Measurement, got {type(m).__name__}"
    assert m.result == 6, f"measure dropped fn's return value: result={m.result!r}"
    kwargs_call = lesson.measure(lambda a, b=0: a + b, 4, b=5)
    assert kwargs_call.result == 9, "measure must forward both *args and **kwargs to fn"
    for field in ("wall_s", "py_peak_mib", "rss_hwm_mib", "rss_delta_mib"):
        value = getattr(m, field)
        assert isinstance(value, float), f"{field} should be a float, got {type(value).__name__}"


def test_measure_wall_time():
    slow = lesson.measure(time.sleep, 0.25)
    assert 0.2 <= slow.wall_s <= 1.5, f"sleeping 0.25s measured as {slow.wall_s:.3f}s"
    fast = lesson.measure(lambda: None)
    assert fast.wall_s < 0.2, f"an empty callable measured as {fast.wall_s:.3f}s"
    assert slow.wall_s > fast.wall_s, "the sleeping call must measure longer than the empty one"


def test_measure_python_peak():
    m = lesson.measure(bytearray, 48 * MIB)
    assert 40.0 <= m.py_peak_mib <= 80.0, (
        f"a 48 MiB bytearray reported {m.py_peak_mib:.2f} MiB as the Python-allocator peak"
    )
    # The buffer is gone before the call returns, so `current` is ~0 and only `peak` remembers
    # it. An implementation reporting get_traced_memory()[0] passes the case above (the result
    # keeps the object alive) and fails here, which is the whole difference between the two.
    transient = lesson.measure(lambda: len(bytearray(48 * MIB)))
    assert transient.result == 48 * MIB, "measure lost the return value of the transient call"
    assert 40.0 <= transient.py_peak_mib <= 80.0, (
        f"a 48 MiB buffer allocated and freed inside the call reported "
        f"{transient.py_peak_mib:.2f} MiB — that is get_traced_memory()[0], the CURRENT size "
        "after the buffer was released; the peak you want is [1]"
    )


def test_measure_peak_resets_between_calls():
    big = lesson.measure(bytearray, 48 * MIB)
    small = lesson.measure(bytearray, 1024)
    assert big.py_peak_mib >= 40.0, f"the 48 MiB call reported {big.py_peak_mib:.2f} MiB"
    assert small.py_peak_mib < 5.0, (
        f"a 1 KiB allocation reported {small.py_peak_mib:.2f} MiB — that is the previous "
        "call's peak leaking into this one"
    )


def test_measure_rss_high_water():
    baseline = lesson.measure(lambda: None)
    assert baseline.rss_hwm_mib > 8.0, (
        f"rss_hwm_mib is the whole process's peak and reported {baseline.rss_hwm_mib:.2f} MiB; "
        "a live interpreter is never that small, so this looks like a delta"
    )
    assert baseline.rss_delta_mib >= 0.0, "rss_delta_mib must never be negative"
    # Beat the existing record by a clear margin, whatever it happens to be: importing the
    # lesson has already mapped and touched hundreds of MiB, and a high-water mark only moves
    # on a new record. Sizing this against the live reading keeps the test honest if the
    # notebook's own demonstrations ever change size.
    record_to_beat = lesson.rss_hwm_mib()
    m = lesson.measure(lesson.touch_mmap, int(record_to_beat) + 400)
    assert m.rss_delta_mib >= 200.0, (
        f"touching {int(record_to_beat) + 400} MiB of fresh pages, against a previous record "
        f"of {record_to_beat:.0f} MiB, moved the high-water mark by only "
        f"{m.rss_delta_mib:.1f} MiB — rss_delta_mib should be the rise in the process peak "
        "across the call, not a constant"
    )
    assert m.py_peak_mib < 5.0, (
        f"mmap pages never reach Python's allocator, so tracemalloc should see almost "
        f"nothing, not {m.py_peak_mib:.1f} MiB — are both numbers coming from the same source?"
    )
    assert m.rss_hwm_mib >= baseline.rss_hwm_mib, "a high-water mark can never fall"


def test_measure_delta_is_clamped():
    real = lesson.rss_hwm_mib
    calls = []

    def falling():
        calls.append(None)
        return 500.0 if len(calls) == 1 else 400.0

    lesson.rss_hwm_mib = falling
    try:
        m = lesson.measure(lambda: None)
    finally:
        lesson.rss_hwm_mib = real
    assert m.rss_delta_mib >= 0.0, (
        f"with the reading falling from 500.0 to 400.0 MiB, rss_delta_mib came back "
        f"{m.rss_delta_mib:.1f} — a high-water mark that appears to drop is a reading "
        "artefact, not a negative cost; clamp the subtraction with max(0.0, ...)"
    )
    if len(calls) >= 2:
        assert m.rss_delta_mib == 0.0, (
            f"a fall of 100 MiB between the two readings should clamp to exactly 0.0, "
            f"got {m.rss_delta_mib:.1f}"
        )


def test_measure_uses_a_monotonic_clock():
    real = lesson.time
    frozen = types.SimpleNamespace(
        perf_counter=real.perf_counter, perf_counter_ns=real.perf_counter_ns,
        monotonic=real.monotonic, monotonic_ns=real.monotonic_ns,
        process_time=real.process_time, sleep=real.sleep,
        time=lambda: 1_600_000_000.0, time_ns=lambda: 1_600_000_000_000_000_000,
    )
    lesson.time = frozen
    try:
        m = lesson.measure(real.sleep, 0.25)
    finally:
        lesson.time = real
    assert m.wall_s >= 0.2, (
        f"with the wall clock frozen, a 0.25s sleep measured {m.wall_s:.3f}s — you are timing "
        "with time.time(), which follows the system clock and can even run backwards; "
        "time.perf_counter() is monotonic and is what this exercise asks for"
    )


# --- exercise 2: chunking, the exact sum, and profile() --------------------------------

def test_chunk_bounds():
    assert lesson.chunk_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert lesson.chunk_bounds(8, 4) == [(0, 4), (4, 8)], "no empty slice when chunk divides n"
    assert lesson.chunk_bounds(3, 10) == [(0, 3)], "a chunk larger than n gives one short slice"
    assert lesson.chunk_bounds(0, 4) == [], "range(0) needs no slices at all"
    assert lesson.chunk_bounds(1, 1) == [(0, 1)]
    for n, chunk in ((37, 5), (100, 7), (9, 9), (1000, 999)):
        bounds = lesson.chunk_bounds(n, chunk)
        covered = [i for start, stop in bounds for i in range(start, stop)]
        assert covered == list(range(n)), (
            f"chunk_bounds({n}, {chunk}) does not cover range({n}) exactly once in order"
        )
        assert all(stop - start <= chunk for start, stop in bounds), (
            f"chunk_bounds({n}, {chunk}) produced a slice longer than chunk"
        )
    try:
        lesson.chunk_bounds(10, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("chunk_bounds(10, 0) must raise ValueError")


def test_sum_squares_chunked_exact():
    for n in (0, 1, 2, 7, 100):
        assert lesson.sum_squares_chunked(n, chunk=3) == _exact(n), f"wrong at n={n}"
    assert lesson.sum_squares_chunked(99_991, chunk=1000) == _exact(99_991)
    big = lesson.sum_squares_chunked(4_000_000)
    assert big == _exact(4_000_000), (
        f"off by {big - _exact(4_000_000)} at n=4_000_000 — the total overflowed int64, so it "
        "is being accumulated in numpy rather than in a Python int"
    )
    assert lesson.sum_squares_chunked(4_000_000, chunk=250_000) == _exact(4_000_000), (
        "the answer must not depend on the chunk size"
    )


def test_sum_squares_chunked_stays_within_chunk():
    real_arange = lesson.np.arange
    sizes = []

    def spy(*args, **kwargs):
        out = real_arange(*args, **kwargs)
        sizes.append(out.size)
        return out

    lesson.np.arange = spy
    try:
        total = lesson.sum_squares_chunked(1_000_000, chunk=50_000)
    finally:
        lesson.np.arange = real_arange
    assert total == _exact(1_000_000), "wrong answer with chunk=50_000"
    assert sizes, "sum_squares_chunked never called np.arange — build each slice with it"
    assert max(sizes) <= 50_000, (
        f"the largest array built held {max(sizes)} elements with chunk=50_000 — the whole "
        "range is being materialised, which is the memory cost this exercise removes"
    )


def test_profile_returns_measurements():
    table = lesson.profile({"exact": lesson.exact_sum_squares, "gen": lesson.sum_squares_generator}, 7)
    assert isinstance(table, dict), f"profile must return a dict, got {type(table).__name__}"
    assert set(table) == {"exact", "gen"}, f"keys should be preserved, got {sorted(table)}"
    for name, m in table.items():
        assert isinstance(m, lesson.Measurement), f"{name} mapped to {type(m).__name__}"
        assert m.result == 91, f"{name} was not called with n=7 (result {m.result!r})"
        assert m.wall_s >= 0.0, f"{name} reported a negative duration"
    # A second, different n: the n must be the one profile was handed, not a remembered one.
    other = lesson.profile({"exact": lesson.exact_sum_squares}, 100)
    assert other["exact"].result == _exact(100), (
        f"profile(..., 100) gave {other['exact'].result!r}, not {_exact(100)} — an n is "
        "hard-coded somewhere; pass profile's own n through to each implementation"
    )


# --- exercise 3: tier_check() ----------------------------------------------------------

def test_tier_check_table():
    cases = [
        (({"wall_s": 12.0, "peak_mib": 900.0}, "cpu8", 600), True, "comfortably inside"),
        (({"wall_s": 600.0, "peak_mib": 8192.0}, "cpu8", 600), True, "exactly at both limits"),
        (({"wall_s": 600.1, "peak_mib": 10.0}, "cpu8", 600), False, "a tenth of a second over"),
        (({"wall_s": 1.0, "peak_mib": 8192.1}, "cpu8", 600), False, "a tenth of a MiB over"),
        (({"wall_s": 1.0, "peak_mib": 2048.0}, "phone", 600), True, "at the phone ceiling"),
        (({"wall_s": 1.0, "peak_mib": 3000.0}, "phone", 600), False, "over the phone ceiling"),
        (({"wall_s": 1.0, "peak_mib": 81920.0}, "gpu80", 600), True, "at the gpu80 ceiling"),
        (({"wall_s": 1.0, "peak_mib": 100000.0}, "gpu80", 600), False, "over the gpu80 ceiling"),
        (({"wall_s": 1.0, "peak_mib": 1.0}, "browser", 600), True, "tiny on browser"),
        (({"wall_s": 1.0, "peak_mib": 1.0}, "cpu9000", 600), False, "tier does not exist"),
        (({"wall_s": 1.0, "peak_mib": 1.0}, "cpu8", 0), False, "no budget declared"),
        (({"wall_s": 0.0, "peak_mib": 1.0}, "cpu8", 0), False,
         "zero budget and zero wall: an undeclared budget is not an infinite one, and it must "
         "fail on its own rather than only because some wall time happened to exceed it"),
    ]
    for (measured, tier, budget), expected, why in cases:
        got = lesson.tier_check(measured, tier, budget)
        assert got.passed is expected, (
            f"{why}: tier_check({measured}, {tier!r}, {budget}) returned passed={got.passed}, "
            f"expected {expected}"
        )


def test_tier_check_reasons():
    ok = lesson.tier_check({"wall_s": 1.0, "peak_mib": 1.0}, "cpu8", 600)
    assert ok.reasons == (), f"a passing verdict carries no reasons, got {ok.reasons!r}"
    unknown = lesson.tier_check({"wall_s": 1.0, "peak_mib": 1.0}, "nope", 600)
    assert len(unknown.reasons) == 1, f"unknown tier gives exactly one reason, got {unknown.reasons!r}"
    both = lesson.tier_check({"wall_s": 900.0, "peak_mib": 9000.0}, "cpu8", 600)
    assert len(both.reasons) == 2, f"over on both axes gives two reasons, got {both.reasons!r}"
    assert all(isinstance(r, str) and r.strip() for r in both.reasons), "reasons must be strings"
    assert "600" in both.reasons[0], f"the time reason must name the budget: {both.reasons[0]!r}"
    assert "8192" in both.reasons[1], f"the memory reason must name the ceiling: {both.reasons[1]!r}"


RUBRIC = [
    (test_measure_shape, 3,
     "return a Measurement built from fn's own return value, forwarding *args and **kwargs"),
    (test_measure_wall_time, 4,
     "time.perf_counter() around the call only — not time.time(), and not in milliseconds"),
    (test_measure_python_peak, 4,
     "get_traced_memory() returns (current, peak); you want [1], divided by 1024*1024"),
    (test_measure_peak_resets_between_calls, 4,
     "tracemalloc.reset_peak() before the call and tracemalloc.stop() after it"),
    (test_measure_rss_high_water, 4,
     "rss_hwm_mib is the absolute process peak; rss_delta_mib is its rise, clamped at 0.0"),
    (test_measure_delta_is_clamped, 2,
     "rss_delta_mib is max(0.0, after - before) — never a negative cost"),
    (test_measure_uses_a_monotonic_clock, 3,
     "time.perf_counter(), not time.time(): the wall clock is not a stopwatch"),
    (test_chunk_bounds, 4,
     "min(start + chunk, n) makes the last slice short; a non-positive chunk is a ValueError"),
    (test_sum_squares_chunked_exact, 5,
     "add int(slice_total) into a Python int; an int64 accumulator wraps around silently"),
    (test_sum_squares_chunked_stays_within_chunk, 3,
     "build one slice at a time with np.arange(start, stop), never the whole range"),
    (test_profile_returns_measurements, 3,
     "one dict entry per implementation: {name: measure(fn, n)}"),
    (test_tier_check_table, 5,
     "compare with > so that landing exactly on a limit passes"),
    (test_tier_check_reasons, 3,
     "append a reason per broken rule, naming the budget and the ceiling, and return early "
     "only for an unknown tier"),
]
