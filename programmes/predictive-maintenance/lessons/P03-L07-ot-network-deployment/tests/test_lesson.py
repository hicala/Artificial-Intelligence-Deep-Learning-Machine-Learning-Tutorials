"""Autograder rubric for P03-L07. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure). Partial credit is the point — a
student who has written the allocator and the decoder but not the ring buffer should see
exactly that.

Every expected value in this file comes from a reference implemented HERE, in Python, from
the stated definition — never from the lesson's own helpers and never from a number typed by
the author. The capture is regenerated here from the same seed; `_ref_q_health` is the
fixed-point path written in Python integers, which are exact; `_ref_ring` is the
store-and-forward buffer written from the three functions' stated contracts; `_ref_forward`
replays the link script against it. A student who redefines `generate_counts` or
`float_health` in lesson.py cannot move any of them.

The scripted probes the binary runs for this rubric (`alloc`, `decode`, `qprobe`, `sfprobe`)
deliberately use different numbers from the binary's own self-test, so passing `make test` by
special-casing its cases does not pass this.

A binary that exits 2 has an unfinished stub; lesson.run_c turns that into
NotImplementedError, which the grader prints as TODO rather than as an error.
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

SEED = 20260922
SEED2 = SEED ^ 0xA5A5A5A5A5A5A5A5
GAMMA = 0x9E3779B97F4A7C15
N_MACHINES, N_HOURS, BURST_LEN = 80, 168, 32
N_FRAMES = N_MACHINES * N_HOURS
FRAME_BYTES = 3 + 2 * BURST_LEN + 2
FUNC_READ_INPUT = 0x04
CORRUPT_STRIDE, CORRUPT_PHASE = 997, 13
N_HEALTHY, N_MARGINAL = 40, 24
SAMPLE_SHIFT, Q_FRAC, WIRE_FRAC = 8, 16, 8
WIRE_SCALE = 1 << WIRE_FRAC
GW_ARENA_BYTES, RING_CAPACITY, BATCH_MAX = 40960, 3072, 64
THRESHOLD = 1.4732
INT32_MAX = 2 ** 31 - 1

_CACHE = {}


def _cached(key, fn):
    """Run the expensive commands once per grading session, not once per test."""
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def _require_build():
    """Every row that touches C starts here, so a compile error is reported once per row
    with the compiler's own words rather than as a traceback."""
    ok, out = lesson.build(verbose=False)
    assert ok, (
        "the C source did not compile. The first lines the compiler printed:\n"
        + "\n".join(out.strip().splitlines()[:8])
        + "\nA LINK error naming gw_no_malloc_in_the_gateway means one of the exercises "
          "called malloc, calloc, realloc or free. The gateway has no heap beyond g_arena; "
          "that is what the #define above EXERCISE 1 is for.")


def _probe(command, *args):
    _require_build()
    return _cached(("probe", command) + args,
                   lambda: lesson.metrics(lesson.run_c(command, *args)))


def _health_text():
    _require_build()
    return _cached("health_text", lambda: lesson.run_c("health"))


# --------------------------------------------------------------------------------------
# Independent references
# --------------------------------------------------------------------------------------

def _mix(seed, k):
    z = np.uint64(seed) + (k + np.uint64(1)) * np.uint64(GAMMA)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _ref_baseline(m):
    return 8192 + 64 * np.asarray(m, dtype=np.int64)


def _ref_counts():
    """The raw counts behind the capture, generated here from the same definition."""
    def build():
        idx = np.arange(N_FRAMES, dtype=np.int64)
        h, m = idx // N_MACHINES, idx % N_MACHINES
        u = _mix(SEED, idx.astype(np.uint64))
        ppm = np.zeros(N_FRAMES, dtype=np.int64)
        a = m < N_HEALTHY
        b = (m >= N_HEALTHY) & (m < N_HEALTHY + N_MARGINAL)
        c = m >= N_HEALTHY + N_MARGINAL
        ppm[a] = 1_000_000 + (u[a] % np.uint64(120001)).astype(np.int64) - 60_000
        ppm[b] = 1_473_200 + (u[b] % np.uint64(80001)).astype(np.int64) - 40_000
        d = m[c] - (N_HEALTHY + N_MARGINAL)
        ramp = np.where(h[c] > 24 + 6 * d, (h[c] - 24 - 6 * d) * (8_000 + 400 * d), 0)
        ppm[c] = 1_000_000 + ramp + (u[c] % np.uint64(40001)).astype(np.int64) - 20_000
        level = _ref_baseline(m) * ppm // 1_000_000
        j = idx[:, None] * BURST_LEN + np.arange(BURST_LEN, dtype=np.int64)[None, :]
        jitter = (_mix(SEED2, j.astype(np.uint64)) % np.uint64(129)).astype(np.int64) - 64
        return np.clip(level[:, None] + jitter, 0, 65535).astype(np.uint16)
    return _cached("counts", build)


def _ref_crc16(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= int(byte)
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if (crc & 1) else (crc >> 1)
    return crc


def _ref_capture():
    """The capture bytes, built here from the frame layout in the exercise's own comment.

    The CRC table is derived from the bit-by-bit definition in `_ref_crc16` above and then
    applied a byte position at a time across all frames at once, because a bit-by-bit pass
    over nine hundred thousand bytes in Python is slower than the lesson itself.
    """
    def build():
        table = np.array([_crc_seed(b) for b in range(256)], dtype=np.uint16)
        counts = _ref_counts()
        out = np.zeros((N_FRAMES, FRAME_BYTES), dtype=np.uint8)
        out[:, 0] = (np.arange(N_FRAMES) % N_MACHINES + 1).astype(np.uint8)
        out[:, 1] = FUNC_READ_INPUT
        out[:, 2] = 2 * BURST_LEN
        out[:, 3:3 + 2 * BURST_LEN:2] = (counts >> np.uint16(8)).astype(np.uint8)
        out[:, 4:4 + 2 * BURST_LEN:2] = (counts & np.uint16(0xFF)).astype(np.uint8)
        crc = np.full(N_FRAMES, 0xFFFF, dtype=np.uint16)
        for j in range(FRAME_BYTES - 2):
            crc = (crc >> np.uint16(8)) ^ table[(crc ^ out[:, j].astype(np.uint16))
                                                & np.uint16(0xFF)]
        out[:, FRAME_BYTES - 2] = (crc & np.uint16(0xFF)).astype(np.uint8)
        out[:, FRAME_BYTES - 1] = (crc >> np.uint16(8)).astype(np.uint8)
        bad = (np.arange(N_FRAMES) % CORRUPT_STRIDE) == CORRUPT_PHASE
        out[bad, 7] ^= np.uint8(0x40)
        # Spot-check the table against the bit-by-bit definition on a real frame.
        probe = out[1].copy()
        want = _ref_crc16(probe[:FRAME_BYTES - 2])
        assert int(probe[FRAME_BYTES - 2]) | (int(probe[FRAME_BYTES - 1]) << 8) == want
        return out
    return _cached("capture", build)


def _crc_seed(b):
    """One table entry: the CRC register after absorbing byte b from a zero register."""
    crc = b
    for _ in range(8):
        crc = (crc >> 1) ^ 0xA001 if (crc & 1) else (crc >> 1)
    return crc


def _corrupt_indices():
    return {i for i in range(N_FRAMES) if i % CORRUPT_STRIDE == CORRUPT_PHASE}


def _ref_q_health(x_q, n, baseline_q):
    """The fixed-point inference path in exact Python integers, from the stated recipe."""
    if n <= 0 or baseline_q <= 0:
        return -1
    acc = 0
    for i in range(n):
        acc += int(x_q[i]) * int(x_q[i])
    mean = acc // n                      # both operands non-negative, so // == C's /
    rms = math.isqrt(mean)
    health = (rms << Q_FRAC) // baseline_q
    return INT32_MAX if health > INT32_MAX else health


def _ref_health_stream():
    """Every reading the gateway should publish: (index, machine, hour, q, wire)."""
    def build():
        counts = _ref_counts()
        bad = _corrupt_indices()
        out = []
        for i in range(N_FRAMES):
            if i in bad:
                continue
            m = i % N_MACHINES
            x_q = (counts[i].astype(np.int64) << SAMPLE_SHIFT).tolist()
            q = _ref_q_health(x_q, BURST_LEN, int(_ref_baseline(m)) << SAMPLE_SHIFT)
            out.append((i, m, i // N_MACHINES, q, max(q, 0) >> (Q_FRAC - WIRE_FRAC)))
        return out
    return _cached("health_ref", build)


class _RefRing:
    """The store-and-forward buffer written from the three functions' stated contracts."""

    def __init__(self, cap):
        self.slot = [None] * cap
        self.cap, self.head, self.count = cap, 0, 0
        self.next_seq, self.refused, self.peak = 1, 0, 0

    def push(self, machine, hour, wire):
        if self.count >= self.cap:
            self.refused += 1
            return 0
        self.slot[(self.head + self.count) % self.cap] = (self.next_seq, hour, wire, machine)
        self.next_seq += 1
        self.count += 1
        self.peak = max(self.peak, self.count)
        return self.next_seq - 1

    def take(self, max_n):
        n = min(self.count, max_n)
        return [self.slot[(self.head + i) % self.cap] for i in range(n)]

    def ack(self, seq):
        dropped = 0
        while self.count > 0 and self.slot[self.head][0] <= seq:
            self.head = (self.head + 1) % self.cap
            self.count -= 1
            dropped += 1
        return dropped


def _ref_sfprobe():
    """The same script the binary's `sfprobe` command runs, against the reference ring."""
    r = _RefRing(5)
    out = {"push_a": r.push(0, 0, 900), "push_b": r.push(1, 0, 901),
           "push_c": r.push(2, 0, 902)}
    got = r.take(2)
    out["take2_n"] = len(got)
    out["take2_first"], out["take2_last"] = got[0][0], got[-1][0]
    out["count_after_take"] = r.count
    out["retake_first"] = r.take(2)[0][0]
    out["ack2_dropped"] = r.ack(2)
    out["count_after_ack"] = r.count
    out["ack2_again"] = r.ack(2)
    out["push_d"] = r.push(3, 1, 903)
    out["push_e"] = r.push(4, 1, 904)
    out["push_f"] = r.push(0, 1, 905)
    out["push_g"] = r.push(1, 1, 906)
    out["count_when_full"] = r.count
    out["push_h_refused"] = r.push(2, 1, 907)
    out["next_seq_after_refusal"] = r.next_seq
    out["count_after_refusal"] = r.count
    out["refused_total"] = r.refused
    drained = r.take(9)
    out["drain_n"] = len(drained)
    out["drain_first"], out["drain_last"] = drained[0][0], drained[-1][0]
    out["drain_machine_first"] = drained[0][3]
    out["drain_wire_first"] = drained[0][2]
    out["drain_hour_last"] = drained[-1][1]
    out["ack_low_dropped"] = r.ack(4)
    out["count_after_partial_ack"] = r.count
    out["ack_all_dropped"] = r.ack(99)
    out["count_at_end"] = r.count
    out["take_on_empty"] = len(r.take(9))
    out["peak"] = r.peak
    return out


def _link_up(scenario, hour):
    if scenario == "outage":
        return not (72 <= hour < 108)
    if scenario == "overflow":
        return not (40 <= hour < 140)
    return True


def _ack_lost(scenario, hour):
    return scenario == "lostack" and 40 <= hour < 44


def _ref_forward(scenario):
    """Replay the link script against the reference ring and an idempotent historian."""
    def build():
        bad = _corrupt_indices()
        r = _RefRing(RING_CAPACITY)
        last_seq = accepted = duplicates = sum_seq = 0
        for h in range(N_HOURS):
            for m in range(N_MACHINES):
                if h * N_MACHINES + m in bad:
                    continue
                r.push(m, h, 0)
            if not _link_up(scenario, h):
                continue
            while True:
                batch = r.take(BATCH_MAX)
                if not batch:
                    break
                for seq, _hour, _wire, _machine in batch:
                    if seq <= last_seq:
                        duplicates += 1
                    else:
                        last_seq, accepted, sum_seq = seq, accepted + 1, sum_seq + seq
                if _ack_lost(scenario, h):
                    break
                r.ack(batch[-1][0])
        return {"pushed": r.next_seq - 1, "refused": r.refused, "accepted": accepted,
                "duplicates": duplicates, "sum_seq": sum_seq, "max_seq": last_seq,
                "held_at_end": r.count, "peak_backlog": r.peak}
    return _cached(("forward_ref", scenario), build)


def _forward(scenario):
    return _probe("forward", "--scenario", scenario)


def _ref_float_health(counts, baseline):
    x = np.asarray(counts, dtype=np.float64) / float(1 << SAMPLE_SHIFT)
    return np.sqrt(np.mean(x * x, axis=-1)) / np.asarray(baseline, dtype=np.float64)


def _ref_psi(reference, current, n_bins=10, floor=1e-6):
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cur = np.asarray(current, dtype=np.float64).ravel()
    edges = np.linspace(float(ref.min()), float(ref.max()), n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.maximum(np.histogram(ref, bins=edges)[0] / ref.size, floor)
    q = np.maximum(np.histogram(cur, bins=edges)[0] / cur.size, floor)
    return float(np.sum((q - p) * np.log(q / p)))


# --------------------------------------------------------------------------------------
# The rubric
# --------------------------------------------------------------------------------------

def _require_identical_capture():
    """The two sides must hold the same bytes, or a comparison between them is theatre.

    Not a rubric row of its own: the generator is GIVEN, so a student who has written
    nothing would earn points for it. It is asserted inside the rows that depend on it.
    """
    def check():
        cap = _ref_capture()
        text = lesson.run_c("capture", "--dump", "64")
        m = lesson.metrics(text)
        flat = cap.reshape(-1).astype(np.uint64)
        want_sum = int(flat.sum())
        want_wsum = int((np.arange(1, flat.size + 1, dtype=np.uint64) * flat).sum())
        assert int(m["capture_bytes"]) == cap.size, (
            f"lesson.c built {int(m['capture_bytes'])} bytes of capture and the reference "
            f"generator {cap.size}; capture_build() or the frame layout has been edited")
        assert (int(m["capture_sum"]), int(m["capture_wsum"])) == (want_sum, want_wsum), (
            f"the capture is the right length but not the right bytes: C reports "
            f"({int(m['capture_sum'])}, {int(m['capture_wsum'])}) and the reference "
            f"computes ({want_sum}, {want_wsum}). capture_build() is given — if this fails "
            "it has been edited, and every comparison in this lesson is then between two "
            "different inputs.")
        dumped = [l.split()[2] for l in text.splitlines() if l.startswith("raw ")]
        for i, line in enumerate(dumped):
            assert line == cap[i].tobytes().hex(), (
                f"capture frame {i} does not match the reference byte for byte")
        return True
    return _cached("capture_identical", check)


def test_c_selftest_passes():
    _require_build()
    code, out = _cached("selftest", lesson.make_test)
    if code == 2:
        raise NotImplementedError("at least one C exercise is still a stub")
    assert code == 0, f"the binary's self-test failed with exit {code}:\n{out}"


def test_gw_alloc_aligns_and_honours_the_budget():
    a = _probe("alloc")
    assert a["zero"] == -1, (
        f"gw_alloc(0) must return NULL; you returned offset {a['zero']:.0f}")
    assert a["first"] == 0, (
        f"the first allocation out of an empty arena is at offset 0, you returned "
        f"{a['first']:.0f}")
    assert a["second"] == 8, (
        f"two 1-byte allocations must be 8 bytes apart, you returned offset "
        f"{a['second']:.0f} for the second. Align the cursor UP to 8 before carving.")
    assert a["used_after_two"] == 9, (
        f"after gw_alloc(1) twice the cursor sits at 9 (8 for the aligned base plus the 1 "
        f"byte handed out); yours sits at {a['used_after_two']:.0f}")
    assert a["whole"] == 0 and a["used_after_whole"] == GW_ARENA_BYTES, (
        f"a request for the whole {GW_ARENA_BYTES}-byte arena must succeed at offset 0 and "
        f"leave the cursor at the end; you returned offset {a['whole']:.0f} with the cursor "
        f"at {a['used_after_whole']:.0f}")
    assert a["one_past"] == -1 and a["refused_after_one_past"] == 1, (
        "one byte past a full arena must return NULL and be counted in g_alloc_refused; you "
        f"returned offset {a['one_past']:.0f} with {a['refused_after_one_past']:.0f} "
        "refusals counted")
    assert a["too_big_for_the_hole"] == -1, (
        "16 bytes will not fit in the 8 that were left, so gw_alloc must refuse")
    assert a["used_after_refusal"] == GW_ARENA_BYTES - 8, (
        f"a REFUSED allocation must leave the cursor where it was: it should still be at "
        f"{GW_ARENA_BYTES - 8} and it is at {a['used_after_refusal']:.0f}. Otherwise the "
        "budget that is left depends on the requests that failed.")
    assert a["fits_the_hole"] == GW_ARENA_BYTES - 8, (
        f"the 8 bytes left must still be available after a refusal, at offset "
        f"{GW_ARENA_BYTES - 8}; you returned {a['fits_the_hole']:.0f}")
    assert a["odd_too_big"] == -1, (
        "16 bytes will not fit in the 9 that were left either, once the cursor has been "
        "aligned up to 8")
    assert a["used_after_unaligned_refusal"] == GW_ARENA_BYTES - 9, (
        f"after a refusal from an UNALIGNED cursor it must still read {GW_ARENA_BYTES - 9} "
        f"and it reads {a['used_after_unaligned_refusal']:.0f}. Writing the aligned base "
        "back before deciding eats the padding of a request that was then refused — and the "
        "earlier case cannot see it, because the cursor was already on a boundary.")
    assert a["odd_fits"] == GW_ARENA_BYTES - 8, (
        f"the last 8 bytes are still there at offset {GW_ARENA_BYTES - 8}; you returned "
        f"{a['odd_fits']:.0f}")


def test_gw_alloc_closes_the_door_at_gw_freeze():
    a = _probe("alloc")
    assert a["before_freeze"] == 0, (
        "an 8-byte request on an empty arena, before gw_freeze(), must succeed at offset 0")
    assert a["after_freeze"] == -1, (
        "after gw_freeze() every allocation must return NULL: a gateway that allocates "
        "while it is running is a gateway that runs out of memory at 3 a.m.")
    assert a["after_freeze_count"] == 1, (
        f"the attempt after gw_freeze() must be counted in g_alloc_after_freeze, which "
        f"reads {a['after_freeze_count']:.0f}. The device has to be able to say that it "
        "happened.")
    assert a["refused_total"] == 1 and a["granted_total"] == 1, (
        f"in that last block exactly one allocation is granted and one refused; you counted "
        f"{a['granted_total']:.0f} granted and {a['refused_total']:.0f} refused")
    arena = _probe("arena")
    assert arena["started"] == 1 and arena["alloc_calls"] == 4, (
        f"start-up makes four allocations and all four must succeed; "
        f"{arena['alloc_calls']:.0f} did")
    assert arena["arena_used"] <= GW_ARENA_BYTES, "start-up cannot use more than the arena"
    assert arena["arena_peak"] == arena["arena_used"] > 0, (
        f"nothing is ever freed, so after start-up the high-water mark equals the cursor: "
        f"{arena['arena_used']:.0f}. Yours reads {arena['arena_peak']:.0f}. The peak is how "
        "the device reports its own headroom, so it has to be maintained on every "
        "successful allocation.")


def test_frame_decode_reads_the_register_map():
    d = _probe("decode")
    _require_identical_capture()
    assert d["rc_ok"] == 0, (
        f"a well-formed one-register frame must return FD_OK (0); you returned "
        f"{d['rc_ok']:.0f}")
    assert d["ok_nreg"] == 1 and d["ok_addr"] == 5, (
        f"that frame carries one register at address 5; you reported nreg "
        f"{d['ok_nreg']:.0f} at address {d['ok_addr']:.0f}")
    assert d["ok_reg0"] == 0x1234, (
        f"the payload bytes 0x12 0x34 are one BIG-endian register, 0x1234 = 4660; you "
        f"returned {d['ok_reg0']:.0f}. 13330 means you assembled it little-endian.")
    cap0 = _ref_counts()[0]
    assert d["rc_capture0"] == 0, "the first frame of the capture is intact and must decode"
    assert d["capture0_reg0"] == int(cap0[0]) and d["capture0_reg31"] == int(cap0[-1]), (
        f"the first and last registers of capture frame 0 are {int(cap0[0])} and "
        f"{int(cap0[-1])}; you returned {d['capture0_reg0']:.0f} and "
        f"{d['capture0_reg31']:.0f}")


def test_frame_decode_rejects_the_right_frames_for_the_right_reasons():
    d = _probe("decode")
    for key, want, why in (
            ("rc_short", -1, "four bytes is shorter than a header plus a checksum, so the "
                             "length is checked first and the answer is FD_SHORT"),
            ("rc_func", -2, "function code 0x03 is not this register map's, so FD_FUNC"),
            ("rc_bytecount", -3, "a byte count of 4 in a 7-byte frame does not agree with "
                                 "the length, so FD_BYTECOUNT"),
            ("rc_bytecount_small", -3, "a byte count of 0 in a 7-byte frame does not agree "
                                       "with the length either, and this frame's checksum is "
                                       "VALID. The rule is buf[2] != len - 5, not "
                                       "buf[2] > len - 5: a check that only refuses a byte "
                                       "count that is too BIG accepts a device that "
                                       "under-declares its payload and publishes a register "
                                       "it never sent"),
            ("rc_capacity", -5, "a frame claiming more registers than the Frame holds must "
                                "be refused BEFORE its checksum is looked at, because "
                                "writing them is what would corrupt memory"),
            ("rc_crc", -4, "one flipped payload bit must fail the checksum, so FD_CRC"),
            ("rc_capture_corrupt", -4, "the capture's own corrupt frame fails its checksum")):
        assert d[key] == want, f"{key}: expected {want} — {why} — and you returned {d[key]:.0f}"
    assert d["bytecount_small_left_out_alone"] == 1, (
        "a frame refused on its byte count must leave *out exactly as it found it, the same "
        "way a checksum failure does")
    assert d["crc_left_out_alone"] == 1, (
        "a frame that failed its checksum must leave *out exactly as it found it. Yours "
        "wrote into it. A half-filled frame is worse than no frame: the caller cannot tell "
        "which half is real.")
    rejected = {int(r[0]) for r in lesson.rows(_health_text(), "bad")}
    assert rejected == _corrupt_indices(), (
        f"your decoder rejected {len(rejected)} frames of the capture and exactly "
        f"{len(_corrupt_indices())} carry a flipped bit. Extra rejections usually mean the "
        "checksum is being read big-endian instead of low byte first.")


def test_q_health_matches_the_integer_reference_bit_for_bit():
    text = _health_text()
    _require_identical_capture()
    got = [(int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]))
           for r in lesson.rows(text, "h")]
    want = _ref_health_stream()
    assert len(got) == len(want), (
        f"your gateway published {len(got)} readings and the reference publishes "
        f"{len(want)}; the corrupt frames are the only ones that should be missing")
    for (gi, gm, gh, gq, gw), (wi, wm, wh, wq, ww) in zip(got, want):
        assert (gi, gm, gh) == (wi, wm, wh), (
            f"reading {gi} is attributed to machine {gm} hour {gh}; it belongs to machine "
            f"{wm} hour {wh}. The machine index is the frame's address byte minus one.")
        assert gq == wq, (
            f"on frame {gi} your q_health returned {gq} and the integer reference returns "
            f"{wq} ({(gq - wq) / 65536.0:+.6f} in health units). Sum the squares in "
            "int64_t, divide by n, isqrt64 that, then shift left by 16 in int64_t and "
            "divide by baseline_q — every step truncating towards zero.")
        assert gw == ww, (
            f"on frame {gi} the published register is {gw} and it should be {ww}: the Q8.8 "
            "register is the Q16.16 value shifted right by 8, truncated, not rounded")


def test_q_health_guards_its_inputs_and_saturates():
    q = _probe("qprobe")
    base = 2048 << SAMPLE_SHIFT
    cases = {
        "q_on_baseline": _ref_q_health([2048 << SAMPLE_SHIFT] * BURST_LEN, BURST_LEN, base),
        "q_n_zero": -1,
        "q_n_negative": -1,
        "q_baseline_zero": -1,
        "q_baseline_negative": -1,
        "q_2049": _ref_q_health([2049 << SAMPLE_SHIFT] * BURST_LEN, BURST_LEN, base),
        "q_full_scale": _ref_q_health([65535 << SAMPLE_SHIFT] * BURST_LEN, BURST_LEN,
                                      65535 << SAMPLE_SHIFT),
        "q_full_over_small": _ref_q_health([65535 << SAMPLE_SHIFT] * BURST_LEN, BURST_LEN,
                                           1 << SAMPLE_SHIFT),
        "q_ramp": _ref_q_health([(i + 1) << SAMPLE_SHIFT for i in range(BURST_LEN)],
                                BURST_LEN, 1 << SAMPLE_SHIFT),
        "q_ramp_half": _ref_q_health([(i + 1) << SAMPLE_SHIFT for i in range(BURST_LEN)],
                                     BURST_LEN // 2, 1 << SAMPLE_SHIFT),
        "q_truncates_the_mean": _ref_q_health([7, 0], 2, 65536),
        "q_truncates_the_divide": _ref_q_health([3, 3, 3, 3], 4, 7),
    }
    hints = {
        "q_n_zero": "n <= 0 has no mean to take, so the answer is -1",
        "q_n_negative": "a negative n is the same refusal as zero",
        "q_baseline_zero": "a health index divides by the baseline, so 0 must return -1",
        "q_baseline_negative": "a negative baseline must return -1, not a negative index",
        "q_full_scale": "32 squares of 16776960 sum to 8.99e15; an int32_t accumulator wraps "
                        "silently here",
        "q_full_over_small": f"this quotient does not fit in an int32_t, so it saturates at "
                             f"{INT32_MAX}; a wrapped value would look plausible and be wrong",
        "q_ramp_half": "only the first n samples are in the burst; reading all of them is an "
                       "off-by-one that the capture never exposes",
        "q_truncates_the_mean": "49 squares over 2 samples is 24 and a half; truncating "
                                "gives 24, whose integer square root is 4, and rounding "
                                "gives 25, whose root is 5. Truncate.",
        "q_truncates_the_divide": "(3 << 16) / 7 is 28086.857; the last divide truncates too",
    }
    for key, expected in cases.items():
        assert int(q[key]) == expected, (
            f"{key}: expected {expected}, you returned {int(q[key])}"
            + (f" — {hints[key]}" if key in hints else ""))


def test_store_and_forward_follows_the_three_contracts():
    got, want = _probe("sfprobe"), _ref_sfprobe()
    hints = {
        "count_after_take": "sf_take is a peek, not a pop: it must not change the ring, "
                            "because the link is allowed to lose the acknowledgement",
        "retake_first": "taking twice without an acknowledgement must hand back the SAME "
                        "records, starting at head",
        "ack2_again": "a repeated acknowledgement drops nothing and is not an error",
        "ack_low_dropped": "sf_ack(4) drops only the records at or below sequence 4, not a "
                           "fixed count and not the whole ring",
        "drain_first": "after the wrap the oldest held record is still the oldest by "
                       "sequence number — check the modulo on head + i",
        "drain_last": "the newest held record is the last one pushed",
        "drain_machine_first": "the record's fields must survive the ring: machine, hour and "
                               "wire are stored with the sequence number",
        "take_on_empty": "an empty ring hands back nothing",
    }
    for key, expected in want.items():
        assert int(got[key]) == expected, (
            f"{key}: expected {expected}, you returned {int(got[key])}"
            + (f" — {hints[key]}" if key in hints else ""))


def test_store_and_forward_loses_nothing_across_an_outage():
    for scenario in ("clean", "outage", "lostack"):
        got, want = _forward(scenario), _ref_forward(scenario)
        assert int(got["pushed"]) == want["pushed"], (
            f"[{scenario}] {int(got['pushed'])} readings entered the buffer and "
            f"{want['pushed']} should have; the {len(_corrupt_indices())} corrupt frames are "
            "the only ones that never get pushed, and this outage fits in the buffer")
        assert int(got["accepted"]) == want["accepted"], (
            f"[{scenario}] {int(got['accepted'])} readings reached the historian and "
            f"{want['accepted']} entered the buffer. A reading accepted at the door has to "
            "arrive: check that sf_ack drops only what was acknowledged and that sf_take "
            "starts at head.")
        assert int(got["sum_seq"]) == want["sum_seq"] and int(got["max_seq"]) == want["max_seq"], (
            f"[{scenario}] the delivered sequence numbers are not 1..{want['max_seq']}: they "
            f"sum to {int(got['sum_seq'])} instead of {want['sum_seq']}. A gap means a "
            "reading was lost in transit.")
        assert int(got["held_at_end"]) == 0, (
            f"[{scenario}] {int(got['held_at_end'])} records are still held at the end of "
            "the week with the link up and everything acknowledged")
    outage = _forward("outage")
    assert int(outage["peak_backlog"]) == _ref_forward("outage")["peak_backlog"], (
        f"the 36-hour outage should peak at {_ref_forward('outage')['peak_backlog']} records "
        f"and yours peaked at {int(outage['peak_backlog'])}")
    assert int(outage["refused"]) == 0, (
        "the outage is shorter than the buffer holds, so nothing should be refused")


def test_store_and_forward_duplicates_nothing_when_an_ack_is_lost():
    got, want = _forward("lostack"), _ref_forward("lostack")
    assert int(got["duplicates"]) == want["duplicates"], (
        f"the historian rejected {int(got['duplicates'])} already-stored records and should "
        f"have rejected {want['duplicates']}. Zero means sf_take is REMOVING the records it "
        "hands out, so a lost acknowledgement silently loses the batch; more than that "
        "means the ring is re-offering records it should have dropped.")
    assert int(got["accepted"]) == want["accepted"], (
        "every reading must still arrive exactly once even though four hours of "
        "acknowledgements were lost")
    for quiet in ("clean", "outage"):
        assert int(_forward(quiet)["duplicates"]) == 0, (
            f"[{quiet}] no acknowledgement is lost here, so the historian should never see "
            "a record twice")


def test_store_and_forward_refuses_rather_than_overwriting():
    got, want = _forward("overflow"), _ref_forward("overflow")
    assert int(got["refused"]) == want["refused"], (
        f"a 100-hour outage does not fit in a {RING_CAPACITY}-record buffer: "
        f"{want['refused']} readings must be refused at the door and you refused "
        f"{int(got['refused'])}. Zero refusals means the ring is overwriting the oldest "
        "unacknowledged record, which turns a countable loss into a silent one.")
    assert int(got["pushed"]) == want["pushed"], (
        f"{want['pushed']} readings should have been accepted into the buffer, "
        f"{int(got['pushed'])} were")
    assert int(got["accepted"]) == want["accepted"] and int(got["sum_seq"]) == want["sum_seq"], (
        f"the {want['accepted']} readings that DID get in must still arrive exactly once "
        f"each, as sequence numbers 1..{want['max_seq']} summing to {want['sum_seq']}; "
        f"you delivered {int(got['accepted'])} summing to {int(got['sum_seq'])}. A refused "
        "push that consumed a sequence number leaves a gap that looks exactly like a loss "
        "in transit.")
    assert int(got["peak_backlog"]) == RING_CAPACITY, (
        f"the buffer should have filled to exactly {RING_CAPACITY} records and yours peaked "
        f"at {int(got['peak_backlog'])}")
    for scenario in ("clean", "outage", "lostack", "overflow"):
        assert _forward(scenario)["alloc_after_freeze"] == 0, (
            f"[{scenario}] the gateway allocated memory after start-up, which it must never "
            "do — every buffer it uses is taken before gw_freeze()")


def test_float_health():
    f = lesson.float_health
    one = f(np.full((1, BURST_LEN), 2048), np.array([8.0]))
    assert np.shape(one) == (1,), (
        f"the burst axis is reduced away, so the result has shape (1,); yours has "
        f"{np.shape(one)}")
    assert abs(float(one[0]) - 1.0) < 1e-15, (
        f"the docstring's worked example must give 1.0; you returned {float(one[0])!r}")
    ramp = np.arange(1, BURST_LEN + 1, dtype=np.int64)[None, :]
    want = _ref_float_health(ramp, np.array([1.0]))
    assert abs(float(f(ramp, np.array([1.0]))[0]) - float(want[0])) < 1e-12, (
        f"on a ramp you returned {float(f(ramp, np.array([1.0]))[0])!r} where the root MEAN "
        f"square over the last axis gives {float(want[0])!r} — check you are not squaring "
        "the mean, or summing without dividing by n")
    stacked = np.arange(2 * 3 * BURST_LEN, dtype=np.int64).reshape(2, 3, BURST_LEN) % 4096
    base = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    got = f(stacked, base)
    assert np.shape(got) == (2, 3), (
        f"with a (2, 3, {BURST_LEN}) input the answer has shape (2, 3); yours has "
        f"{np.shape(got)} — reduce over the LAST axis only")
    assert np.allclose(np.asarray(got), _ref_float_health(stacked, base), rtol=0, atol=1e-12)
    assert np.asarray(f(np.full((1, BURST_LEN), 2048, dtype=np.int16),
                        np.array([8.0]))).dtype == np.float64, (
        "compute in float64 whatever the caller's array is typed as; an int16 input must not "
        "make the answer an int16")
    for bad in (np.array([0.0]), np.array([-1.0])):
        try:
            f(np.full((1, BURST_LEN), 2048), bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"a baseline of {float(bad[0])} must raise ValueError")
    try:
        f(np.zeros((2, 0), dtype=np.int64), np.array([1.0, 1.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("an empty burst axis must raise ValueError, not return nan")


def test_agreement_report():
    f = lesson.agreement_report
    toy = f(np.array([1.50, 1.60]), np.array([1.25, 1.60]), 1.55)
    assert set(toy) == {"max_abs_error", "mean_abs_error", "worst_index",
                        "effective_threshold", "n_float_only", "n_wire_only",
                        "n_disagreements"}, (
        f"the report must carry exactly the seven documented keys; yours has {sorted(toy)}")
    assert abs(toy["max_abs_error"] - 0.25) < 1e-15, (
        f"the worked example's largest absolute difference is 0.25; you returned "
        f"{toy['max_abs_error']!r}")
    assert abs(toy["mean_abs_error"] - 0.125) < 1e-15, (
        f"the two differences are 0.25 and 0.0, so the mean is 0.125; you returned "
        f"{toy['mean_abs_error']!r} — this is a mean of the ABSOLUTE differences, not of "
        "the signed ones")
    assert toy["worst_index"] == 0, (
        f"the worst reading is at flat index 0; you returned {toy['worst_index']!r}")
    assert toy["effective_threshold"] == 397 / 256, (
        f"the smallest Q8.8 value at or above 1.55 is ceil(1.55 * 256) / 256 = 397/256 = "
        f"{397 / 256!r}; you returned {toy['effective_threshold']!r}. Rounding to nearest, "
        "or taking the value below, describes a threshold the plant is not using.")
    below_half = f(np.array([1.0]), np.array([1.0]), THRESHOLD)
    assert below_half["effective_threshold"] == 378 / 256, (
        f"this lesson's own threshold, {THRESHOLD}, is 377.139 steps up the Q8.8 grid, so "
        f"the smallest representable value at or above it is 378/256 = {378 / 256!r}. You "
        f"returned {below_half['effective_threshold']!r} — 377/256 is the value BELOW it, "
        "which no wire value has to clear, and rounding to nearest gives the same wrong "
        "answer whenever the fraction is under a half.")
    negative_worst = f(np.array([1.0, 2.0]), np.array([3.0, 2.5]), 5.0)
    assert negative_worst["worst_index"] == 0, (
        "the largest gap here is row 0, where h_wire is 2.0 ABOVE h_float; you returned "
        f"{negative_worst['worst_index']!r}. worst_index is the argmax of the ABSOLUTE "
        "difference — taking it from the signed difference happens to agree on this "
        "lesson's own week, where the deployed path only ever truncates downwards, and "
        "disagrees the first time a reading comes back high.")
    assert abs(negative_worst["max_abs_error"] - 2.0) < 1e-15, (
        f"the largest absolute gap is 2.0; you returned {negative_worst['max_abs_error']!r}")
    mixed = f(np.array([1.0, 2.0]), np.array([1.5, 1.5]), 5.0)
    assert abs(mixed["mean_abs_error"] - 0.5) < 1e-15, (
        f"the two differences are -0.5 and +0.5, so the mean ABSOLUTE error is 0.5; you "
        f"returned {mixed['mean_abs_error']!r}. A signed mean cancels them to zero and "
        "reports a deployment that agrees perfectly.")
    exact = f(np.array([1.5]), np.array([1.5]), 1.5)
    assert exact["effective_threshold"] == 1.5, (
        "1.5 is exactly representable in Q8.8 (384/256), so the effective threshold is 1.5 "
        f"itself; you returned {exact['effective_threshold']!r}")
    assert isinstance(toy["max_abs_error"], float) and type(toy["n_float_only"]) is int, (
        "return plain Python floats and ints; a numpy scalar does not survive being written "
        "to a report")
    try:
        f(np.zeros(3), np.zeros(4), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("two differently shaped arrays must raise ValueError")
    try:
        f(np.zeros(4), np.zeros(1), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "(4,) against (1,) must raise ValueError too. numpy BROADCASTS this pair rather "
            "than refusing it, so leaning on the subtraction to raise for you reports four "
            "readings compared against one and calls it a clean deployment.")
    ref = _ref_health_stream()
    counts = _ref_counts()
    idx = np.array([r[0] for r in ref])
    mach = np.array([r[1] for r in ref])
    hf = _ref_float_health(counts[idx], _ref_baseline(mach) / float(1 << SAMPLE_SHIFT))
    wire = np.array([r[4] for r in ref], dtype=np.float64) / WIRE_SCALE
    internal = np.array([r[3] for r in ref], dtype=np.float64) / (1 << Q_FRAC)
    got = f(hf, wire, THRESHOLD)
    diff = np.abs(hf - wire)
    assert abs(got["max_abs_error"] - float(diff.max())) < 1e-15, (
        f"on the lesson's own week you returned a largest gap of {got['max_abs_error']!r} "
        f"where the reference measures {float(diff.max())!r}")
    assert abs(got["mean_abs_error"] - float(diff.mean())) < 1e-15
    assert got["worst_index"] == int(np.argmax(diff)), (
        "worst_index is the FLAT index of the largest absolute difference")
    arith = f(hf, internal, THRESHOLD)
    assert arith["max_abs_error"] < got["max_abs_error"], (
        "the arithmetic gap must come out smaller than the wire gap; Q8.8 throws away eight "
        "of the sixteen fractional bits the gateway computed with")


def test_agreement_report_counts_the_two_directions_separately():
    f = lesson.agreement_report
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 1.0, 3.0, 4.0])
    r = f(a, b, 1.5)
    assert r["n_float_only"] == 1, (
        f"row 1 alarms in float64 (2.0 >= 1.5) and not on the wire (1.0 < 1.5), so "
        f"n_float_only is 1; you returned {r['n_float_only']!r}")
    assert r["n_wire_only"] == 1, (
        f"row 0 alarms on the wire (2.0 >= 1.5) and not in float64 (1.0 < 1.5), so "
        f"n_wire_only is 1; you returned {r['n_wire_only']!r}. Counting only the total "
        "hides which way a deployment fails, and the two directions have different owners.")
    assert r["n_disagreements"] == 2, (
        f"n_disagreements is the sum of the two directions; you returned "
        f"{r['n_disagreements']!r}")
    on_the_line = f(np.array([1.5]), np.array([1.25]), 1.5)
    assert on_the_line["n_float_only"] == 1, (
        f"a float64 value sitting exactly ON the threshold alarms, because the plant's rule "
        f"is `>= threshold`; the wire value below it does not, so n_float_only is 1 and you "
        f"returned {on_the_line['n_float_only']!r}. A strict `>` silently narrows the "
        "operating point you were given by one representable step.")
    edge = f(np.array([1.5]), np.array([1.5]), 1.5)
    assert edge["n_disagreements"] == 0, (
        "the comparison is >=, so a value sitting exactly on the threshold alarms on both "
        "sides and there is no disagreement")
    ref = _ref_health_stream()
    counts = _ref_counts()
    idx = np.array([r[0] for r in ref])
    mach = np.array([r[1] for r in ref])
    hf = _ref_float_health(counts[idx], _ref_baseline(mach) / float(1 << SAMPLE_SHIFT))
    wire = np.array([r[4] for r in ref], dtype=np.float64) / WIRE_SCALE
    got = f(hf, wire, THRESHOLD)
    want_f = int(np.count_nonzero((hf >= THRESHOLD) & ~(wire >= THRESHOLD)))
    want_w = int(np.count_nonzero((wire >= THRESHOLD) & ~(hf >= THRESHOLD)))
    assert (got["n_float_only"], got["n_wire_only"]) == (want_f, want_w), (
        f"on the lesson's own week the counts are {want_f} float-only and {want_w} "
        f"wire-only; you returned {got['n_float_only']} and {got['n_wire_only']}")


def test_psi():
    f = lesson.psi
    rng = np.random.default_rng(11)
    a = rng.normal(0.0, 1.0, 5000)
    b = rng.normal(0.4, 1.3, 5000)
    assert f(a, a) == 0.0, (
        f"a window against itself must be exactly 0.0; you returned {f(a, a)!r}")
    got, want = f(a, b), _ref_psi(a, b)
    assert abs(got - want) < 1e-12, (
        f"you returned {got!r} where the stated recipe — {10} equal-width bins over the "
        f"REFERENCE range with infinite outer edges, proportions floored at 1e-6, and "
        f"sum((q - p) * ln(q / p)) — gives {want!r}")
    for n_bins in (2, 5, 20):
        assert abs(f(a, b, n_bins=n_bins) - _ref_psi(a, b, n_bins=n_bins)) < 1e-12, (
            f"with n_bins={n_bins} your answer does not match the reference; the edges are "
            f"linspace(reference.min(), reference.max(), n_bins + 1) with the two outer "
            "edges replaced by -inf and +inf")
    assert abs(f(b, a) - _ref_psi(b, a)) < 1e-12 and abs(f(a, b) - f(b, a)) > 1e-9, (
        "the bins come from the REFERENCE, so swapping the two windows must change the "
        "answer. Getting the same number both ways means the edges are being built from "
        "the current window, or from the two windows together — and a statistic whose bins "
        "move with the data cannot see the data move.")
    gapped = np.concatenate([rng.normal(0.0, 0.2, 800), rng.normal(10.0, 0.2, 800)])
    middle = rng.normal(5.0, 0.2, 800)
    hole = f(gapped, middle)
    assert math.isfinite(hole), (
        f"a reference with a gap in the middle has EMPTY bins; a current window that lands "
        f"in them divides by a proportion of zero and you returned {hole!r}. The floor "
        "applies to BOTH proportions — p as well as q — which is what makes an empty "
        "reference bin a large finite term instead of an infinite one.")
    assert abs(hole - _ref_psi(gapped, middle)) < 1e-9, (
        f"on a reference with empty bins you returned {hole!r} where the stated recipe "
        f"gives {_ref_psi(gapped, middle)!r}")
    far = f(a, a + 40.0)
    assert math.isfinite(far) and far > 0, (
        f"a window entirely outside the reference range must give a finite positive answer, "
        f"not {far!r}: the outer edges are infinite and the empty bins are floored")
    assert abs(far - _ref_psi(a, a + 40.0)) < 1e-9
    custom = f(a, b, n_bins=10, floor=1e-3)
    assert abs(custom - _ref_psi(a, b, n_bins=10, floor=1e-3)) < 1e-12, (
        "the floor is a parameter; apply the one you were given to BOTH proportions")
    for args, kwargs, why in ((([], a), {}, "an empty reference"),
                              ((a, []), {}, "an empty current window"),
                              ((np.ones(10), a), {}, "a reference with no spread"),
                              ((a, b), {"n_bins": 1}, "n_bins below 2")):
        try:
            f(*args, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{why} must raise ValueError")
    assert type(f(a, b)) is float, (
        f"return a plain Python float; you returned a {type(f(a, b)).__name__}")


def test_psi_sees_a_recalibration_with_no_labels_at_all():
    """The diode's whole argument, graded: the signal that survives it must work."""
    f = lesson.psi
    ref = _ref_health_stream()
    counts = _ref_counts()
    idx = np.array([r[0] for r in ref])
    mach = np.array([r[1] for r in ref])
    hour = np.array([r[2] for r in ref])
    wire = np.array([r[4] for r in ref], dtype=np.float64) / WIRE_SCALE
    healthy = mach < N_HEALTHY
    base = wire[healthy & (hour < 84)]
    same = wire[healthy & (hour >= 84)]
    recal = _ref_float_health((counts[idx].astype(np.int64) * 101) // 100,
                              _ref_baseline(mach) / float(1 << SAMPLE_SHIFT))
    recal = np.floor(recal * WIRE_SCALE) / WIRE_SCALE
    recal = recal[healthy & (hour >= 84)]
    quiet, drifted = f(base, same), f(base, recal)
    assert abs(quiet - _ref_psi(base, same)) < 1e-12
    assert abs(drifted - _ref_psi(base, recal)) < 1e-12
    assert drifted > quiet, (
        f"a 1% recalibration of the raw counts must move the published feature further than "
        f"the second half of an undisturbed week does; you measured {drifted!r} against "
        f"{quiet!r}. This is the only drift signal that crosses a one-way diode, so it has "
        "to work without a single label coming back.")


RUBRIC = [
    (test_c_selftest_passes, 6,
     "run `make test` — it names the unfinished exercise and what it expects"),
    (test_gw_alloc_aligns_and_honours_the_budget, 8,
     "align the cursor up to 8, refuse when the request will not fit, and leave the cursor "
     "untouched on a refusal"),
    (test_gw_alloc_closes_the_door_at_gw_freeze, 5,
     "after gw_freeze() return NULL and count the attempt in both refusal counters"),
    (test_frame_decode_reads_the_register_map, 9,
     "registers are BIG-endian: (buf[3 + 2*i] << 8) | buf[4 + 2*i]"),
    (test_frame_decode_rejects_the_right_frames_for_the_right_reasons, 7,
     "length, then function code, then byte count, then capacity, then checksum — and write "
     "nothing into *out unless all five pass"),
    (test_q_health_matches_the_integer_reference_bit_for_bit, 10,
     "int64_t for the accumulator AND for the shift into the numerator; truncate at every "
     "step and never round"),
    (test_q_health_guards_its_inputs_and_saturates, 5,
     "-1 for n <= 0 or baseline_q <= 0, INT32_MAX when the quotient will not fit"),
    (test_store_and_forward_follows_the_three_contracts, 9,
     "sf_take is a peek; sf_ack drops by sequence number, not by count; sf_push refuses "
     "without consuming a sequence number"),
    (test_store_and_forward_loses_nothing_across_an_outage, 7,
     "every reading pushed must arrive, and the delivered sequence numbers must be "
     "contiguous from 1"),
    (test_store_and_forward_duplicates_nothing_when_an_ack_is_lost, 7,
     "sf_take must not remove what it hands out, so a lost acknowledgement re-offers the "
     "batch and the historian's last_seq is what makes delivery exactly-once"),
    (test_store_and_forward_refuses_rather_than_overwriting, 6,
     "a full ring refuses and counts; it never overwrites the oldest unacknowledged record"),
    (test_float_health, 7,
     "float64 throughout, reduce over the last axis only, ValueError on a baseline <= 0 or "
     "an empty burst"),
    (test_agreement_report, 10,
     "absolute differences, the flat index of the worst, and ceil(threshold * 256) / 256 for "
     "the effective threshold"),
    (test_agreement_report_counts_the_two_directions_separately, 5,
     "count float-only and wire-only alarms separately; the direction is the finding"),
    (test_psi, 9,
     "bins from the REFERENCE window, infinite outer edges, both proportions floored, "
     "natural logs"),
    (test_psi_sees_a_recalibration_with_no_labels_at_all, 4,
     "if psi is right this follows; if it fails alone, check the floor and the outer edges"),
]
