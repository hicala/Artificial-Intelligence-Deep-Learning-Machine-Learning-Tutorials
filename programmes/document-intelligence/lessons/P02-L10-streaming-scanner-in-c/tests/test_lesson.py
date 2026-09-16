"""Autograder rubric for P02-L10-streaming-scanner-in-c. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure). Partial credit is the point — a
student whose `field_push` works scores for it while `scan_chunk` is still a TODO.

The cases below are deliberately not the ones in the notebook's public `_check_*` functions:
passing what you can see is not the same as being right. Every expected value is either
derived from the Python scanner in the notebook (the published specification) or written out
by hand from the grammar in lesson.c — never read back from the C being graded.
"""
import importlib.util
import os
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
# Each module points at the lesson.c beside it, so the reference and the student binary are
# built side by side and never overwrite each other.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

CAP = lesson.FIELD_CAP
_HIDDEN_BYTES = 137_003          # not a round number, and not a multiple of the record block
_SMALL_STREAM = 2 << 20
# 192 MiB, not 48: a scanner that keeps only a few BYTES per record grows about 4 MiB across a
# 2 -> 48 MiB span and slips under the margin below, and at 192 MiB it grows about 10 MiB and
# does not. The reference grows 0.00 MiB at every one of these sizes, so widening the span
# costs the margin nothing and costs the grader about half a second.
_BIG_STREAM = 192 << 20


def _scan_both(text: str, chunk: int = 0) -> tuple:
    """The C answer and the published Python answer for the same bytes."""
    return lesson.c_scan(text, chunk=chunk), lesson.py_scan(text.encode())


def _agree(text: str, keys=("records", "fields", "amount_cents", "eur_records",
                            "max_field_len", "field_overflow", "embedded_newlines",
                            "bad_arity", "truncated_tail")) -> None:
    got, want = _scan_both(text)
    for key in keys:
        assert got[key] == want[key], (
            f"on {text!r} your {key} is {got[key]} and the specification says {want[key]}")


def test_c_compiles():
    """Whatever the student did to lesson.c, it is still a valid C11 translation unit."""
    binary = lesson.build_c(force=True)
    assert Path(binary).exists(), f"make reported success but produced no binary at {binary}"


def test_field_push_capacity_and_content():
    """The bounds, on both sides of them, and the bytes that were kept."""
    for n in (0, 1, CAP - 1, CAP):
        got = lesson.c_push("q" * n)
        assert got["field_len"] == n, f"{n} bytes should all fit, got field_len {got['field_len']}"
        assert got["seen_len"] == n, f"{n} bytes pushed, seen_len says {got['seen_len']}"
        assert got["field_overflow"] == 0, f"{n} bytes fit inside {CAP}; nothing overflowed"
    over = lesson.c_push("q" * (CAP + 1))
    assert over["field_len"] == CAP, "one byte past the cap must not lengthen the buffer"
    assert over["seen_len"] == CAP + 1, "seen_len counts the byte that did not fit"
    assert over["field_overflow"] == 1, "one byte past the cap is one overflowed field"
    # The kept prefix is the FIRST kFieldCap bytes, in order — not the last, and not shuffled.
    mixed = "abcdefghij" * ((CAP // 10) + 2)
    assert lesson.c_push(mixed)["field"] == mixed[:CAP], (
        "the bytes that fit must be the first ones, in the order they arrived")


def test_field_push_overflow_is_per_field():
    """A field seven times the cap is still one overflowed field."""
    assert lesson.c_push("z" * (CAP * 7))["field_overflow"] == 1
    assert lesson.c_push("z" * (CAP * 7))["seen_len"] == CAP * 7
    # And the boundary in the other direction: exactly the cap is not an overflow.
    assert lesson.c_push("z" * CAP)["field_overflow"] == 0


def test_field_complete_columns_and_totals():
    """Column-specific rules must be column-specific."""
    # Money-shaped and EUR-shaped values in every column BUT the ones they belong to: only
    # column 3 may move amount_cents and only column 4 may move eur_records.
    decoy = lesson.c_field(["9.99", "EUR", "8.88", "2.50", "EUR", "7.77"])
    assert decoy["amount_cents"] == 250, (
        f"amount_cents is {decoy['amount_cents']}; only column {lesson.COL_AMOUNT} is money, "
        "so 9.99, 8.88 and 7.77 must contribute nothing")
    assert decoy["eur_records"] == 1, (
        f"eur_records is {decoy['eur_records']}; only column {lesson.COL_CURRENCY} is the "
        "currency, so the EUR sitting in column 1 is just a supplier name")
    assert decoy["fields"] == 6 and decoy["col"] == lesson.COLUMNS
    # max_field_len is the longest of the six, and it is measured before truncation.
    wide = lesson.c_field(["a", "b" * (CAP + 61), "c", "1.00", "EUR", "d"])
    assert wide["max_field_len"] == CAP + 61, (
        f"max_field_len is {wide['max_field_len']}; a truncated field still knows how long it "
        "was, because seen_len counted every byte")
    assert wide["field_overflow"] == 1
    # Empty fields are fields.
    empties = lesson.c_field(["", "", "", "", "", ""])
    assert empties["fields"] == 6 and empties["max_field_len"] == 0
    assert empties["amount_cents"] == 0 and empties["eur_records"] == 0
    # Closing the record is record_complete's job, and it is given: six fields, right arity.
    closed = lesson.c_field(["a", "b", "c", "1.00", "EUR", "f"], close=True)
    assert closed["records"] == 1 and closed["bad_arity"] == 0
    assert lesson.c_field(["a", "b"], close=True)["bad_arity"] == 1


def test_field_complete_money_refusals():
    """parse_cents returns a bool for a reason: a refusal adds nothing, not zero."""
    cases = {"1234.56": 123456, "7": 700, "-40.00": -4000, "0.01": 1,
             "1,234.56": 0, "12.5": 0, "12.345": 0, "12.00usd": 0, "": 0, "-": 0, ".50": 0}
    for value, cents in cases.items():
        got = lesson.c_field(["d", "s", "i", value, "EUR", "n"])
        assert got["amount_cents"] == cents, (
            f"the amount field {value!r} contributed {got['amount_cents']} cents, expected "
            f"{cents} — add the out-parameter only when parse_cents returns true")


def test_scan_chunk_quoting_hidden():
    """Quoting shapes the notebook's public cases do not contain.

    Two of these cases exist because of a hole found by mutation testing: a scanner that
    pushes BOTH halves of an escaped "" produces a field one byte too long, and nothing else
    changes. That is invisible unless the over-long field is the longest field in the record,
    or unless the extra byte is the one that tips it past kFieldCap. Both shapes are here.

    The last three exist because of the mirror-image hole, found by the review pass: a scanner
    that drops a bare CR *inside* a quoted field — the same leniency the grammar grants
    OUTSIDE one — produces a field one byte too SHORT, and again nothing else changes. Inside
    quotes every byte is data, CR included.
    """
    for text in (
        'a,"",c,1.00,EUR,f\n',                       # an empty quoted field
        '"a",b,c,1.00,EUR,f\n',                      # the record opens with a quote
        'a,"b""",c,1.00,EUR,f\n',                    # an escaped quote at the very end
        'a,"""b",c,1.00,EUR,f\n',                    # an escaped quote at the very start
        'a,"""",c,1.00,EUR,f\n',                     # a field that is one literal quote
        'a,"x,y,z",c,1.00,EUR,f\n',                  # several commas inside one field
        'a,"p\nq\nr",c,1.00,EUR,f\n',                # several newlines inside one field
        'a,b,c,1.00,EUR,f\r\n',                      # CRLF
        'a,b,c,1.00,EUR,f\n\n',                      # a blank line between records
        'a,b,c,1.00,EUR,f\na,b,c,2.00,USD,g\n',      # two records, one of them not EUR
        'a,b,c,1.00,EUR,f,extra\n',                  # seven fields: bad arity, still a record
        'a,b,c,1.00,EUR\n',                          # five fields: bad arity, still a record
        'a,"' + "u" * 20 + '""' + "v" * 20 + '",c,1.00,EUR,f\n',   # the escape is in the longest field
        'a,"' + "w" * (CAP - 1) + '""' + '",c,1.00,EUR,f\n',       # the escape sits on the cap
        'a,"x\r\ny",c,1.00,EUR,f\n',                               # a CRLF inside a quoted field
        'a,"' + "r" * 20 + "\r\n" + "s" * 20 + '",c,1.00,EUR,f\n',  # that CR is in the longest field
        'a,"' + "t" * (CAP - 1) + "\r" + '",c,1.00,EUR,f\n',        # that CR sits on the cap
    ):
        _agree(text)
    # Pin the two above explicitly, so a failure names the defect rather than a counter.
    longest = lesson.c_scan('a,"' + "u" * 20 + '""' + "v" * 20 + '",c,1.00,EUR,f\n')
    assert longest["max_field_len"] == 41, (
        f"the quoted field holds 20 + 1 + 20 = 41 bytes of data once the escaped \"\" is "
        f"resolved to a single quote; max_field_len says {longest['max_field_len']} — 42 "
        "means you pushed both halves of the escape")
    on_cap = lesson.c_scan('a,"' + "w" * (CAP - 1) + '""' + '",c,1.00,EUR,f\n')
    assert (on_cap["max_field_len"], on_cap["field_overflow"]) == (CAP, 0), (
        f"that field is exactly kFieldCap bytes once the escape is resolved, so nothing "
        f"overflowed; got max_field_len {on_cap['max_field_len']} and "
        f"{on_cap['field_overflow']} overflow(s)")
    # And the same two shapes for a CR inside quotes, which is data and not a line ending.
    cr_longest = lesson.c_scan('a,"' + "r" * 20 + "\r\n" + "s" * 20 + '",c,1.00,EUR,f\n')
    assert (cr_longest["max_field_len"], cr_longest["embedded_newlines"]) == (42, 1), (
        f"that quoted field holds 20 + CR + LF + 20 = 42 bytes, one of which is an embedded "
        f"newline; max_field_len says {cr_longest['max_field_len']} and embedded_newlines says "
        f"{cr_longest['embedded_newlines']} — 41 means you dropped the CR, and inside a quoted "
        "field a CR is data like any other byte")
    cr_on_cap = lesson.c_scan('a,"' + "t" * (CAP - 1) + "\r" + '",c,1.00,EUR,f\n')
    assert (cr_on_cap["max_field_len"], cr_on_cap["field_overflow"]) == (CAP, 0), (
        f"that field is exactly kFieldCap bytes with the CR counted, so nothing overflowed; got "
        f"max_field_len {cr_on_cap['max_field_len']} and {cr_on_cap['field_overflow']} "
        "overflow(s) — the CR-is-dropped leniency applies OUTSIDE quotes only")


def test_scan_chunk_boundary_invariance():
    """The answer may not depend on where the reader split the stream."""
    # Every interesting boundary is exercised by chunk size 1: between the two quotes of an
    # escaped "", between CR and LF, between a field's last byte and its comma.
    text = ('DOC-1,"Vantor, S.A.",INV-1,12.50,EUR,"note ""x""\nsecond line"\r\n'
            'DOC-2,"",INV-2,7.25,USD,' + "n" * (CAP + 9) + '\n'
            'DOC-3,Plain,INV-3,0.01,EUR,tail')
    reference = lesson.py_scan(text.encode())
    keys = ("records", "fields", "amount_cents", "eur_records", "max_field_len",
            "field_overflow", "embedded_newlines", "bad_arity", "truncated_tail", "bytes")
    for chunk in (1, 2, 3, 5, 8, 13, 64, 0):
        got = lesson.c_scan(text, chunk=chunk)
        for key in keys:
            assert got[key] == reference[key], (
                f"at chunk size {chunk} your {key} is {got[key]} and the specification says "
                f"{reference[key]} — state that has to cross a chunk boundary must live in "
                "the Scanner struct")


def test_scan_chunk_matches_oracle_on_export():
    """A real export, at a size that is not a round number of anything."""
    path = lesson.generate_export(_HIDDEN_BYTES, "hidden")
    got = lesson.c_scan_file(path, chunk=4096)
    want = lesson.py_scan(path.read_bytes())
    for key in ("records", "fields", "bytes", "amount_cents", "eur_records", "max_field_len",
                "field_overflow", "embedded_newlines", "bad_arity", "truncated_tail"):
        assert got[key] == want[key], (
            f"on {_HIDDEN_BYTES} bytes of export your {key} is {got[key]} and the "
            f"specification says {want[key]}")
    assert got["records"] > 1000, "an export this size holds more than a thousand records"


def test_scan_finish_tail_shapes():
    """Every way a stream can stop, including the ones that are not truncation."""
    shapes = {
        "a,b,c,1.00,EUR,f": (1, 0),          # no newline, full arity: a record
        "a,b,c,1.00,EUR,f\n": (1, 0),        # newline: a record
        # A trailing comma leaves an EMPTY sixth field pending, and completing it finishes a
        # whole record — so the col + 1 == kColumns rule makes this a record with an empty
        # note, exactly like the missing-newline case above. Truncation is about arity, not
        # about whether the last field had any bytes in it.
        "a,b,c,1.00,EUR,": (1, 0),
        "a,b,c,1.00,EUR": (0, 1),            # five fields: short
        "a": (0, 1),                         # one field: short
        "a,b,c,1.00,EUR,\"f": (0, 1),        # unterminated quoted field
        'a,b,c,1.00,EUR,"f"': (1, 0),        # closing quote, full arity: a record
        'a,b,c,1.00,EUR,"f""': (0, 1),       # trailing "" leaves the field open again
        "": (0, 0),                          # nothing at all
        "\n": (1, 0),                        # an empty line IS a terminated record
    }
    for text, (records, truncated) in shapes.items():
        got = lesson.c_scan(text)
        assert (got["records"], got["truncated_tail"]) == (records, truncated), (
            f"{text!r} should end with records={records} truncated_tail={truncated}, got "
            f"records={got['records']} truncated_tail={got['truncated_tail']}")
        _agree(text)


def test_scan_finish_no_phantom_record():
    """A clean end must not invent a field, a record or a truncation."""
    two = lesson.c_scan("a,b,c,1.00,EUR,f\na,b,c,2.00,EUR,g\n")
    assert two["records"] == 2 and two["fields"] == 12 and two["truncated_tail"] == 0
    assert two["bad_arity"] == 0 and two["amount_cents"] == 300
    # The same two records with the final newline removed: identical totals, still no
    # truncation. This is the RFC 4180 rule, and it is the one that costs a record when wrong.
    trimmed = lesson.c_scan("a,b,c,1.00,EUR,f\na,b,c,2.00,EUR,g")
    for key in ("records", "fields", "amount_cents", "bad_arity", "truncated_tail"):
        assert trimmed[key] == two[key], (
            f"dropping the final newline changed {key} from {two[key]} to {trimmed[key]} — "
            "RFC 4180 says the last record may or may not have an ending line break")
    empty = lesson.c_scan("")
    assert empty["records"] == 0 and empty["fields"] == 0 and empty["truncated_tail"] == 0


def test_constant_memory_under_growth():
    """Peak RSS must not track the size of the export.

    The threshold is deliberately loose — this is a property test, not a benchmark, and
    nobody should lose points because their machine was busy. A scanner that accumulates
    whole records misses it by hundreds of times this margin; the span rather than the
    threshold is what catches a scanner that keeps only a handful of bytes per record.
    """
    scale = lesson.rss_scale()
    small = lesson.c_bench(_SMALL_STREAM)["peak_rss_raw"] * scale / (1 << 20)
    big = lesson.c_bench(_BIG_STREAM)["peak_rss_raw"] * scale / (1 << 20)
    data_grew = (_BIG_STREAM - _SMALL_STREAM) / (1 << 20)
    assert big - small < 8.0, (
        f"peak RSS rose {big - small:.1f} MiB while the export grew {data_grew:.0f} MiB — the "
        "scanner is keeping the data instead of folding it into counters")


def test_scanner_matches_oracle_end_to_end():
    """Correctness backstop: the C scanner must agree with the Python one on the full export.

    Throughput and peak RSS are REPORTED in the notebook, never graded beyond the property
    above. Wall-clock time is a property of the machine a student happens to be on, and
    nobody should lose points because their laptop was busy. Correctness is the thing a
    rubric can fairly assert.
    """
    report = lesson.throughput_report()
    assert report["identical"], (
        "the C and Python scanners produced different counters on the full export; fix "
        "correctness before speed")
    assert report["records"] > 0 and report["naive_lines"] > report["records"], (
        "the naive split-on-newline should over-count, because a quoted newline looks like a "
        "record boundary to it — if it does not, the export or the scanner is wrong")


RUBRIC = [
    (test_c_compiles, 2, "read the compiler's FIRST error; the rest are usually its echo"),
    (test_field_push_capacity_and_content, 3,
     "exercise 1: keep at most kFieldCap bytes, count every byte in seen_len, and keep the "
     "FIRST bytes rather than the last"),
    (test_field_push_overflow_is_per_field, 2,
     "exercise 1: fire field_overflow once, at seen_len == kFieldCap + 1, not on every byte"),
    (test_field_complete_columns_and_totals, 4,
     "exercise 2: the money rule is column kColAmount only and the currency rule is column "
     "kColCurrency only; max_field_len comes from seen_len"),
    (test_field_complete_money_refusals, 3,
     "exercise 2: add the cents only when parse_cents returned true"),
    (test_scan_chunk_quoting_hidden, 4,
     "exercise 3: an opening or closing quote is not data; only a doubled \"\" inside a "
     "quoted field pushes one quote"),
    (test_scan_chunk_boundary_invariance, 4,
     "exercise 3: everything that must survive a chunk boundary belongs in the Scanner "
     "struct — no local holding the previous byte, no peek at buf[i+1]"),
    (test_scan_chunk_matches_oracle_on_export, 4,
     "exercise 3: the cases the toy strings do not contain are long fields, a quoted field "
     "landing on a chunk edge, and the very last byte of the stream"),
    (test_scan_finish_tail_shapes, 3,
     "exercise 4: kQuoted is truncation, kQuoteSeen is not; a short record is truncation and "
     "a missing final newline is not"),
    (test_scan_finish_no_phantom_record, 3,
     "exercise 4: when nothing is pending, do nothing — do not complete an empty field"),
    (test_constant_memory_under_growth, 3,
     "the scanner must hold one chunk and a fixed set of counters, and nothing else"),
    (test_scanner_matches_oracle_end_to_end, 4,
     "every counter must match the Python scanner on the full export before any timing means "
     "anything"),
]
