"""Autograder rubric for P04-L09. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure). Partial credit is the point: a student
who has the binning and the PSI right but not the reservoir should see exactly that.

Every expected value in this file comes from a reference implemented HERE, from the stated
definition — never from lesson.py's helpers and never from a number typed by the author. The
month and the development sample are regenerated here from the same seed and the same exact
operations; the bin rule is numpy's searchsorted, the PSI is a plain Python loop over
math.log, and the reservoir is a Python loop over Algorithm R. The memory ceiling is this
file's own constant and this file's own unit conversion, so a student who edits the
notebook's headroom or their own maxrss_mib() cannot move the gate that grades them.

The monitor is driven through lesson.run_worker, which feeds it through a PIPE on stdin. A
binary that exits 2 has an unfinished stub; the wrapper turns that into NotImplementedError,
which the grader prints as TODO rather than as an error. Everything generated at run time goes
to the system temporary directory and is deleted when grading ends.
"""
import atexit
import importlib.util
import math
import os
import platform
import shutil
import tempfile
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

# The rubric's own constants, deliberately not read from lesson.py.
N_BINS = 10
FLOOR = 1e-6
THRESHOLD = 0.25
CHECKPOINT = 262144
RECORD = 16
SEED = 20260909
N_DEV = 200_000
N_MONTH = 1 << 22
CHANNEL_FROM = 1_500_000
CHANNEL_Q32 = round(0.6 * 2 ** 32)
ACCOUNT_BASE = 100_000_000_000
TAIL_BIN = 8
K = 1000
RES_SEED = 4242424242
HEADROOM_MIB = 4.0
GAMMA = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1
REC = np.dtype([("account_id", "<u8"), ("score", "<f8")])

_TMP = Path(tempfile.mkdtemp(prefix="p04l09-grade-"))
atexit.register(shutil.rmtree, _TMP, True)
_CACHE = {}


def _cached(key, fn):
    """Run the expensive things once per grading session, not once per test."""
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


# --------------------------------------------------------------------------------------
# Independent references
# --------------------------------------------------------------------------------------

def _splitmix_counter(seed, index):
    """Counter-based splitmix64 over a uint64 index array: mix(seed + (index + 1) * GAMMA)."""
    z = np.uint64(seed) + (index + np.uint64(1)) * np.uint64(GAMMA)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _uniform(z):
    return (z >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


def _dev():
    def make():
        idx = np.arange(2 * N_DEV, dtype=np.uint64)
        z = _splitmix_counter(SEED + 1, idx)
        return _uniform(z[0::2]) * _uniform(z[1::2])
    return _cached("dev", make)


def _edges():
    def make():
        e = np.unique(np.quantile(_dev(), np.linspace(0.0, 1.0, N_BINS + 1)))
        e[0], e[-1] = -np.inf, np.inf
        return e
    return _cached("edges", make)


def _bins(x, edges):
    return np.clip(np.searchsorted(edges, np.asarray(x, dtype=float), side="right") - 1,
                   0, len(edges) - 2)


def _dev_counts():
    return _cached("dev_counts", lambda: np.bincount(_bins(_dev(), _edges()), minlength=N_BINS))


def _month():
    """The month, regenerated here from its definition, in one piece."""
    def make():
        n = N_MONTH
        z = _splitmix_counter(SEED, np.arange(3 * n, dtype=np.uint64))
        ua, ub, c = _uniform(z[0::3]), _uniform(z[1::3]), z[2::3]
        base = ua * ub
        pos = np.arange(n)
        channel = (pos >= CHANNEL_FROM) & ((c >> np.uint64(32)) < np.uint64(CHANNEL_Q32))
        score = base.copy()
        score[channel] = np.sqrt(np.sqrt(base[channel]))
        d = c & np.uint64(0xFFFF)
        score[d == 1] = -(score[d == 1] + 0.5)
        score[d == 0] = np.nan
        score[d == 2] = 1.0 + 2.0 ** -52
        score[d == 3] = 1.0
        score[d == 4] = 0.0
        score[d == 5] = -0.0
        rec = np.empty(n, dtype=REC)
        rec["account_id"] = np.uint64(ACCOUNT_BASE) + pos.astype(np.uint64)
        rec["score"] = score
        return rec
    return _cached("month", make)


def _month_file():
    def make():
        path = _TMP / "month.bin"
        _month().tofile(path)
        return path
    return _cached("month_file", make)


def _write_profile(name, edges, counts, tail_bin):
    path = _TMP / name
    path.write_text("p04l09-profile 1\n"
                    f"n_bins {len(counts)}\n"
                    f"tail_bin {tail_bin}\n"
                    "edges " + " ".join(repr(float(e)) for e in edges) + "\n"
                    "counts " + " ".join(str(int(c)) for c in counts) + "\n")
    return path


def _profile():
    return _cached("profile", lambda: _write_profile("profile.txt", _edges(), _dev_counts(),
                                                     TAIL_BIN))


def _ref_psi(expected, actual, floor=FLOOR):
    """Module 1's arithmetic as a plain Python loop over math.log. Never numpy's log."""
    ne, na = int(sum(expected)), int(sum(actual))
    if ne <= 0 or na <= 0:
        return math.nan, [math.nan] * len(expected)
    contrib = []
    for e, a in zip(expected, actual):
        es, as_ = max(int(e) / ne, floor), max(int(a) / na, floor)
        contrib.append((as_ - es) * math.log(as_ / es))
    return sum(contrib), contrib


def _tol(contrib):
    """64 units in the last place per bin, scaled by the size of the figures."""
    size = sum(abs(c) for c in contrib if not math.isnan(c))
    return 64 * len(contrib) * 2.0 ** -53 * max(1.0, size)


def _ref_breach(psi, contrib):
    if not psi > THRESHOLD:
        return -1
    best = 0
    for i, c in enumerate(contrib):
        if c > contrib[best]:
            best = i
    return best


def _splitmix_next(state):
    state = (state + GAMMA) & MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return state, z ^ (z >> 31)


def _ref_reservoir_loop(items, k, seed):
    """Algorithm R, one offer at a time, in plain Python. For the small cases."""
    ids, scores, state = [], [], seed
    for t, (i, s) in enumerate(items):
        if t < k:
            ids.append(i)
            scores.append(s)
        else:
            state, z = _splitmix_next(state)
            j = z % (t + 1)
            if j < k:
                ids[j], scores[j] = i, s
    return ids, scores, len(items), state


def _ref_reservoir_month():
    """Algorithm R over the month's tail: the draws in bulk, the replacements in order."""
    def make():
        m = _month()
        s = m["score"]
        valid = (s >= 0.0) & (s <= 1.0)
        tail = valid & (_bins(np.where(valid, s, 0.0), _edges()) >= TAIL_BIN)
        ids, scores = m["account_id"][tail], s[tail]
        t = ids.size
        kept_ids, kept_scores = [int(v) for v in ids[:K]], [float(v) for v in scores[:K]]
        # draw number d (from 1) is the output after the state has advanced d times
        z = _splitmix_counter(RES_SEED, np.arange(t - K, dtype=np.uint64))
        j = z % (np.arange(K, t, dtype=np.uint64) + np.uint64(1))
        for off in np.nonzero(j < np.uint64(K))[0]:
            kept_ids[int(j[off])] = int(ids[K + off])
            kept_scores[int(j[off])] = float(scores[K + off])
        return kept_ids, kept_scores, t
    return _cached("month_reservoir", make)


def _ref_month():
    def make():
        s = _month()["score"]
        valid = (s >= 0.0) & (s <= 1.0)
        idx = _bins(np.where(valid, s, 0.0), _edges())
        counts = np.bincount(idx[valid], minlength=N_BINS)
        rows, first = [], (-1, -1)
        for cp in range(CHECKPOINT, s.size + 1, CHECKPOINT):
            v = valid[:cp]
            a = np.bincount(idx[:cp][v], minlength=N_BINS)
            psi, contrib = _ref_psi(_dev_counts(), a)
            b = _ref_breach(psi, contrib)
            rows.append((cp, int(v.sum()), psi, b))
            if b >= 0 and first[0] < 0:
                first = (cp, b)
        psi, contrib = _ref_psi(_dev_counts(), counts)
        final = _ref_breach(psi, contrib)
        if final >= 0 and first[0] < 0:
            first = (s.size, final)
        return {"records_read": s.size, "valid": int(valid.sum()),
                "rejected": int((~valid).sum()), "counts": counts, "psi": psi,
                "contrib": contrib, "rows": rows, "first": first, "breach": final}
    return _cached("ref_month", make)


def _mib(raw):
    """The rubric's own ru_maxrss conversion: bytes on macOS, KiB on Linux."""
    system = platform.system().lower()
    if system == "darwin":
        return raw / 2 ** 20
    if system == "linux":
        return raw / 2 ** 10
    raise RuntimeError(f"the memory gate has no known ru_maxrss unit for {platform.system()!r}")


# --------------------------------------------------------------------------------------
# Driving the binary
# --------------------------------------------------------------------------------------

def _bins_c(edges, values):
    path = _write_profile("probe.txt", edges, [1] * (len(edges) - 1), len(edges) - 1)
    rows = lesson.run_c("bins", "--profile", path, *[repr(float(v)) for v in values])["bins"]
    return [b for _v, b in rows]


def _psi_c(expected, actual, floor=FLOOR):
    rep = lesson.run_c("psi", "--expected", ",".join(map(str, expected)),
                       "--actual", ",".join(map(str, actual)), "--floor", repr(float(floor)))
    return rep["psi"], [rep["contrib"][i] for i in range(len(expected))]


def _breach_c(psi, contrib, threshold=THRESHOLD):
    return lesson.run_c("breach", "--psi", repr(float(psi)),
                        "--contrib", ",".join(repr(float(c)) for c in contrib),
                        "--threshold", repr(float(threshold)))["bin"]


def _counts_only(feed, write_size=1 << 20, pause=0.0):
    return lesson.run_worker("monitor", "--profile", _profile(), "--k", K, "--seed", RES_SEED,
                             "--counts-only", feed=feed, write_size=write_size, pause=pause)


def _month_run(mode):
    """The month through the student's monitor, in one of the harness's modes: --counts-only
    needs exercises 2 and 6, --no-psi adds 5, --no-tail adds 3 and 4. Each row below uses the
    narrowest mode that can grade it, so an unfinished exercise costs only its own rows."""
    def make():
        return lesson.run_worker("monitor", "--profile", _profile(), "--k", K, "--seed",
                                 RES_SEED, mode, feed=_month_file())
    return _cached(("month_run", mode), make)


def _as_bytes(scores, first_id=7):
    rec = np.empty(len(scores), dtype=REC)
    rec["account_id"] = np.arange(first_id, first_id + len(scores), dtype=np.uint64)
    rec["score"] = np.asarray(scores, dtype=np.float64)
    return rec.tobytes()


def _expect_counts(rep, scores, what):
    s = np.asarray(scores, dtype=np.float64)
    valid = (s >= 0.0) & (s <= 1.0)
    counts = np.bincount(_bins(s[valid], _edges()), minlength=N_BINS)
    assert rep["records_read"] == s.size, (
        f"{what}: records_read is {rep['records_read']}, the stream held {s.size} whole records. "
        "A read from a pipe can return part of a record; fread(g_buffer, 1, ...) keeps reading "
        "until the buffer is full, read(2) does not")
    n_nan = int(np.isnan(s).sum())
    assert rep["rejected"] == int((~valid).sum()), (
        f"{what}: rejected is {rep['rejected']}, expected {int((~valid).sum())}"
        + (f" — short by exactly the {n_nan} NaN scores, so the validity test lets NaN through"
           if rep["rejected"] == int((~valid).sum()) - n_nan and n_nan else ""))
    assert rep["valid"] == int(valid.sum()), (
        f"{what}: valid is {rep['valid']}, expected {int(valid.sum())}")
    got = [rep["counts"][i] for i in range(N_BINS)]
    assert got == counts.tolist(), (
        f"{what}: bin counts {got}, module 1's rule gives {counts.tolist()}")


# --------------------------------------------------------------------------------------
# The rubric
# --------------------------------------------------------------------------------------

def test_c_selftest_passes():
    code, out = lesson.make_test()
    if code == 2:
        raise NotImplementedError("at least one C exercise is still a stub")
    assert code == 0, f"the binary's self-test failed with exit {code}:\n{out}"


def test_maxrss_mib():
    f = lesson.maxrss_mib
    assert f(1048576, "Darwin") == 1.0 and f(34603008, "Darwin") == 33.0, (
        f"macOS reports ru_maxrss in bytes: 34603008 is 33.0 MiB; you returned "
        f"{f(34603008, 'Darwin')!r}")
    assert f(33792, "Linux") == 33.0 and f(1024, "Linux") == 1.0, (
        f"Linux reports ru_maxrss in KiB: 33792 is 33.0 MiB; you returned {f(33792, 'Linux')!r}")
    assert f(1572864, "Darwin") == 1.5 and f(1536, "Linux") == 1.5, (
        f"1572864 bytes and 1536 KiB are both 1.5 MiB; you returned {f(1572864, 'Darwin')!r} and "
        f"{f(1536, 'Linux')!r} — use true division, so a fraction of a MiB survives")
    assert f(1536, "linux") == 1.5 and f(1572864, "DARWIN") == 1.5, (
        "compare the system name without regard to case")
    assert f(0, "Linux") == 0.0, "a reading of 0 is 0.0 MiB"
    assert f(1, "Darwin") == 2.0 ** -20, "one byte is 2**-20 MiB exactly"
    for args in ((1024, "Linux"), (1048576, "Darwin")):
        assert type(f(*args)) is float, (
            f"maxrss_mib{args} returned a {type(f(*args)).__name__}; return a plain float")
    for bad in ("Windows", "", "SunOS"):
        try:
            f(1024, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"maxrss_mib(1024, {bad!r}) must raise ValueError, not guess")
    try:
        f(-5, "Darwin")
    except ValueError:
        pass
    else:
        raise AssertionError("a negative reading must raise ValueError")


def test_bin_index_follows_module_1():
    rng = np.random.default_rng(9)
    wide = np.concatenate(([-np.inf], np.linspace(0.01, 0.99, 63), [np.inf]))  # 64 bins
    cases = [
        ([-np.inf, 0.1, 0.2, 0.3, np.inf],
         [0.0, -0.0, 1.0] + [v for e in (0.1, 0.2, 0.3)
                             for v in (e, math.nextafter(e, 0.0), math.nextafter(e, 1.0))]),
        ([0.05, 0.5, 0.95], [-1.0, 0.0, 0.05, 0.5, 0.95, 0.96, 1e300, -math.inf, math.inf]),
        (list(wide), list(rng.random(300)) + [float(e) for e in wide[1:-1]]),
        ([-np.inf, np.inf], [0.0, 0.5, 1.0]),
        (list(_edges()), [float(e) for e in _edges()[1:-1]] + [0.0, 1.0]),
    ]
    for edges, values in cases:
        got = _bins_c(edges, values)
        want = _bins(values, np.asarray(edges, dtype=float)).tolist()
        bad = [(v, g, w) for v, g, w in zip(values, got, want) if g != w]
        assert not bad, (
            f"with {len(edges) - 1} bins, bin_index({bad[0][0]!r}) returned {bad[0][1]}, "
            f"module 1's rule gives {bad[0][2]} ({len(bad)} of {len(values)} probes wrong). A "
            "score on an interior edge belongs to the bin above it; below the first edge is "
            "bin 0; at or above the last edge is the last bin")


def test_psi_from_counts_matches_module_1():
    rng = np.random.default_rng(11)
    cases = [([1, 2, 3, 4], [4, 3, 2, 1]), ([1000, 0, 0], [0, 0, 1000]),
             ([0, 5, 5], [0, 5, 5]), ([3, 3, 3], [1, 1, 1]), ([7], [9]),
             ([123456789, 1], [1, 123456789]), ([40, 60], [41, 59]),
             (rng.integers(0, 50, 64).tolist(), rng.integers(0, 50, 64).tolist())]
    for expected, actual in cases:
        got, contrib = _psi_c(expected, actual)
        want, want_c = _ref_psi(expected, actual)
        tol = _tol(want_c)
        assert math.isfinite(got) and abs(got - want) <= tol, (
            f"expected counts {expected[:6]}{'...' if len(expected) > 6 else ''} against "
            f"actual {actual[:6]}{'...' if len(actual) > 6 else ''}: psi {got!r}, module 1 "
            f"gives {want!r} (tolerance {tol:.1e}). Shares are doubles divided by their own "
            "side's total; the floor goes on both sides before a natural log")
        worst = max(range(len(contrib)), key=lambda i: abs(contrib[i] - want_c[i]))
        assert abs(contrib[worst] - want_c[worst]) <= tol, (
            f"expected {expected[:6]} against actual {actual[:6]}: contribution {worst} is "
            f"{contrib[worst]!r}, module 1 gives {want_c[worst]!r}")
    for expected, actual in (([50, 50], [0, 0]), ([0, 0], [5, 5])):
        got, contrib = _psi_c(expected, actual)
        assert math.isnan(got) and all(math.isnan(c) for c in contrib), (
            f"expected {expected} against actual {actual}: one side has no records, so there "
            f"are no shares — psi and every contribution are NaN; you returned {got!r}")
    # The floor is an ARGUMENT. The notebook's section 8 calls the function with floor 0.0 to
    # show what an unfloored index does; a body that floors at the module's own constant
    # instead passes every case above and silently falsifies that demonstration.
    for expected, actual, floor in (([90, 5, 5], [98, 1, 1], 0.05),
                                    ([50, 30, 20], [70, 30, 0], 1e-3)):
        got, contrib = _psi_c(expected, actual, floor)
        want, want_c = _ref_psi(expected, actual, floor)
        assert abs(got - want) <= _tol(want_c), (
            f"expected {expected} against actual {actual} with floor {floor}: psi {got!r}, "
            f"module 1 gives {want!r}. Floor at the `floor` argument you are given, not at the "
            "module's own constant — the harness passes other floors")


def test_breach_names_the_largest_bin():
    cases = [(0.30, [0.05, 0.12, 0.12, 0.01], THRESHOLD, 1),
             (0.25, [0.2, 0.05], THRESHOLD, -1),
             (math.nextafter(0.25, 1.0), [0.0, 0.25], THRESHOLD, 1),
             (math.nan, [0.5, 0.5], THRESHOLD, -1),
             (0.5, [0.5, 0.0, 0.0], 0.4, 0),
             (0.6, [0.1, 0.2, 0.3, 0.3, 0.3], THRESHOLD, 2),
             (0.05, [0.05, 0.0], 0.01, 0),
             (0.05, [0.0, 0.05], 0.1, -1),
             (1.2, [0.0] * 63 + [1.2], THRESHOLD, 63)]
    for psi, contrib, threshold, want in cases:
        got = _breach_c(psi, contrib, threshold)
        assert got == want, (
            f"breach_bin(psi={psi!r}, threshold={threshold}) with contributions "
            f"{contrib if len(contrib) < 8 else '63 zeros then 1.2'} returned {got}, expected "
            f"{want}: strictly greater than the threshold, NaN never a breach, the largest "
            "contribution, the lowest bin of a tie")


def test_reservoir_is_reproducible_bit_for_bit():
    for n, k, seed in ((1, 1, 7), (3, 5, 0), (50, 50, MASK), (1000, 1, 99),
                       (20000, 64, 123456789), (250000, 4096, 5)):
        rep = lesson.run_c("reservoir", "--n", n, "--k", k, "--seed", seed)
        items = [(1000 + i, i / n) for i in range(n)]
        ids, scores, seen, state = _ref_reservoir_loop(items, k, seed)
        assert rep["seen"] == seen, (
            f"n = {n}, k = {k}: the reservoir counted {rep['seen']} offers, not {seen}")
        assert rep["state"] == state, (
            f"n = {n}, k = {k}, seed = {seed}: the generator was not left where {max(0, n - k)} "
            "draws leave it — the first k offers draw nothing, every later offer draws once")
        assert rep["outside_writes"] == 0, (
            f"n = {n}, k = {k}: the reservoir wrote to {rep['outside_writes']} slot(s) beyond "
            "the first k. Replace only when the draw reduces to strictly less than k")
        got_ids = [sid for _j, sid, _s in rep["samples"]]
        got_scores = [sc for _j, _sid, sc in rep["samples"]]
        assert got_ids == ids, (
            f"n = {n}, k = {k}, seed = {seed}: the sample differs from Algorithm R at slot "
            f"{next((j for j, (a, b) in enumerate(zip(got_ids, ids)) if a != b), len(ids))}. "
            "The modulus is the number of offers INCLUDING this one; replace only when the draw "
            "reduces to strictly less than k")
        assert got_scores == scores, f"n = {n}, k = {k}: a kept score does not match its account"


def test_monitor_small_streams_are_exact():
    head = _month()[:3000]
    specials = [math.nan, -math.nan, -1e-308, 1.0000000000000002, -0.0, 0.0, 1.0, math.inf,
                -math.inf, 5e-324, math.nextafter(1.0, 0.0), math.nan]
    scores = head["score"].copy()
    scores[100:100 + len(specials)] = specials
    # Misaligned pieces with a pause after each, so the reader sees the stream end mid-record
    # several times: 4099 is 3 more than a multiple of 16, and no sum of fewer than 16 of them
    # is a whole number of records.
    rep = _counts_only(_as_bytes(scores) + b"\x01\x02\x03\x04\x05", write_size=4099,
                       pause=0.02)
    _expect_counts(rep, scores, "3000 records delivered in 4099-byte pieces")
    assert rep["truncated_bytes"] == 5, (
        f"five stray bytes followed the last whole record; truncated_bytes is "
        f"{rep['truncated_bytes']}. Read bytes, not records: fread with a 16-byte element "
        "swallows a partial record")
    one = _counts_only(_as_bytes([0.5]) + b"\x00" * 15)
    assert (one["records_read"], one["truncated_bytes"]) == (1, 15), (
        f"one record and 15 stray bytes: you reported {one['records_read']} record(s) and "
        f"{one['truncated_bytes']} truncated byte(s)")
    bad = _counts_only(_as_bytes([math.nan] * 100))
    assert (bad["valid"], bad["rejected"]) == (0, 100), (
        f"100 NaN scores must all be rejected; you counted {bad['valid']} valid — every "
        "ordered comparison with NaN is false, so `s < 0 || s > 1` lets them through")
    two = _month()[: 2 * CHECKPOINT]
    rep = _counts_only(two.tobytes())
    _expect_counts(rep, two["score"], f"{2 * CHECKPOINT} records")
    rows = [(at, valid) for at, valid, _p, _b in rep["checkpoint_rows"]]
    s = two["score"]
    want = [(cp, int(((s[:cp] >= 0.0) & (s[:cp] <= 1.0)).sum())) for cp in
            (CHECKPOINT, 2 * CHECKPOINT)]
    assert rep["checkpoints"] == 2 and rows == want, (
        f"{2 * CHECKPOINT} records cross two checkpoints, after records {CHECKPOINT} and "
        f"{2 * CHECKPOINT}, each called once the record is counted; you reported {rows}")


def test_monitor_month_counts_are_exact():
    rep, ref = _month_run("--counts-only"), _ref_month()
    _expect_counts(rep, _month()["score"], "the month")
    assert rep["truncated_bytes"] == 0, "the month is whole records; nothing is truncated"
    assert rep["checkpoints"] == len(ref["rows"]), (
        f"{rep['checkpoints']} checkpoints over the month, expected {len(ref['rows'])}")


def test_monitor_month_running_psi_and_breach():
    rep, ref = _month_run("--no-tail"), _ref_month()
    got = [rep["counts"][i] for i in range(N_BINS)]
    assert got == ref["counts"].tolist(), "the month's counts differ from the reference"
    tol = _tol(ref["contrib"])
    assert abs(rep["psi"] - ref["psi"]) <= tol, (
        f"the month's PSI is {rep['psi']!r}, module 1's arithmetic gives {ref['psi']!r} "
        f"(tolerance {tol:.1e})")
    rows = rep["checkpoint_rows"]
    assert len(rows) == len(ref["rows"]), (
        f"{len(rows)} checkpoint lines, expected {len(ref['rows'])}")
    for (at, valid, psi, b), (r_at, r_valid, r_psi, r_b) in zip(rows, ref["rows"]):
        assert (at, valid, b) == (r_at, r_valid, r_b), (
            f"checkpoint at record {at}: valid {valid}, breach bin {b}; the reference has "
            f"record {r_at}, valid {r_valid}, bin {r_b}")
        assert abs(psi - r_psi) <= 64 * N_BINS * 2.0 ** -53 * max(1.0, r_psi), (
            f"the running PSI at record {at} is {psi!r}, the reference {r_psi!r}")
    assert (rep["first_breach_at"], rep["first_breach_bin"]) == ref["first"], (
        f"the first breach is reported at record {rep['first_breach_at']}, bin "
        f"{rep['first_breach_bin']}; the reference has {ref['first']}")
    assert rep["breach_bin"] == ref["breach"], (
        f"the month's breach names bin {rep['breach_bin']}, the reference bin {ref['breach']}")


def test_monitor_month_reservoir():
    rep = _month_run("--no-psi")
    ids, scores, seen = _ref_reservoir_month()
    assert rep["tail_seen"] == seen, (
        f"the reservoir was offered {rep['tail_seen']} records; the month has {seen} valid "
        f"records in bins {TAIL_BIN} and above, in stream order")
    got_ids = [sid for _j, sid, _s in rep["samples"]]
    got_scores = [sc for _j, _sid, sc in rep["samples"]]
    assert got_ids == ids and got_scores == scores, (
        "the month's tail sample differs from Algorithm R over the tail: offer every valid "
        "record whose bin is at or above tail_bin, once, in stream order, and nothing else")


def _baseline_mib():
    return _cached("baseline", lambda: min(
        _mib(lesson.run_worker("baseline", "--profile", _profile(), "--k", K, "--seed",
                               RES_SEED)["worker_maxrss_raw"]) for _ in range(2)))


def _gate(rep, what, tail=False):
    # The gate's precondition is that the monitor did the WORK being measured — read every
    # record of the month, and, for the tail row, pushed the tail through its reservoir — not
    # that every count is right. A wrong validity test, bin rule or counter already costs the
    # rows that grade counts; it should not also cost the rows that grade memory.
    n = N_MONTH
    assert rep["records_read"] == n, (
        f"the memory gate only means something for a monitor that read the whole month: "
        f"records_read is {rep['records_read']}, the month has {n} records. Fix the reading "
        "loop first")
    if tail:
        seen = _ref_reservoir_month()[2]
        # At half the month's tail, a hoarded copy would still be several times the headroom.
        assert 2 * rep["tail_seen"] >= seen, (
            f"your monitor offered {rep['tail_seen']} records to the reservoir; the month has "
            f"{seen} tail records. The tail gate can only see a hoarded tail that went through "
            "your monitor — offer every valid record whose bin is at or above tail_bin")
    base, peak = _baseline_mib(), _mib(rep["worker_maxrss_raw"])
    assert peak <= base + HEADROOM_MIB, (
        f"{what} peaked at {peak:.2f} MiB; the worker's own baseline is {base:.2f} MiB and the "
        f"ceiling {base + HEADROOM_MIB:.2f} MiB. It keeps something that grows with the "
        "stream — an array of scores, a copy of the tail, a buffer that doubles")


def test_monitor_loop_stays_under_the_memory_ceiling():
    _gate(_month_run("--counts-only"), "your monitor, counting the month,")


def test_monitor_with_its_tail_stays_under_the_memory_ceiling():
    _gate(_month_run("--no-psi"), "your monitor, keeping its tail sample,", tail=True)


RUBRIC = [
    (test_c_selftest_passes, 2,
     "run `make test` — it names the unfinished exercise and what it expects"),
    (test_maxrss_mib, 12,
     "bytes on macOS, KiB on Linux, case-insensitive, ValueError for anything else"),
    (test_bin_index_follows_module_1, 12,
     "count the edges <= x, subtract one, clip into [0, n_bins - 1]"),
    (test_psi_from_counts_matches_module_1, 12,
     "shares as doubles, the floor on both sides before log, NaN when a side is empty"),
    (test_breach_names_the_largest_bin, 10,
     "strictly over the threshold; the largest contribution; the lowest bin of a tie"),
    (test_reservoir_is_reproducible_bit_for_bit, 12,
     "no draws for the first k; then j = draw % (seen + 1), replace when j < k"),
    (test_monitor_small_streams_are_exact, 4,
     "reject what is not 0 <= s <= 1 (NaN included); read bytes; count the leftover"),
    (test_monitor_month_counts_are_exact, 4,
     "every complete record counted once, in its module-1 bin, in a single pass"),
    (test_monitor_loop_stays_under_the_memory_ceiling, 6,
     "keep counts and g_buffer — nothing that grows with the stream"),
    (test_monitor_month_running_psi_and_breach, 4,
     "call monitor_checkpoint when records_read is a multiple of CHECKPOINT_EVERY"),
    (test_monitor_month_reservoir, 3,
     "offer each valid record whose bin >= tail_bin to the reservoir, in order"),
    (test_monitor_with_its_tail_stays_under_the_memory_ceiling, 5,
     "the tail goes straight into the reservoir; never into a list of its own"),
]
