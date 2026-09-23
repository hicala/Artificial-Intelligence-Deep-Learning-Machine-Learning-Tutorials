"""Autograder rubric for P02-L01. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These rules matter in this particular lesson and are deliberate:

* Every case grades OBSERVABLE BEHAVIOUR, never one particular implementation. The scorer may
  loop or comprehend, the normaliser may use `re` or not; nothing here inspects source.
* `test_score_field_wrong_value_counts_twice` is what stops the most attractive wrong answer:
  counting a mismatched prediction as a false positive only, or a false negative only. Every
  other scoring case still passes with that bug, because they contain no wrong values.
* `test_match_fuzzy_is_confined_to_text` exists because fuzzy matching applied to money is the
  defect this lesson was written to prevent. `"1284.50"` scores 0.857 against `"1234.50"`,
  above the lesson's 0.85 threshold, so a matcher that fuzzes every field type reports a
  50-unit payment error as correct and scores full marks on every other case.
* `test_apply_reviews_leaves_the_input_alone` grades the aliasing bug rather than the arithmetic
  one. `apply_reviews` that edits the records in place still returns the right answer for one
  budget; it is only the SECOND budget in the quality/cost sweep that silently inherits the
  first one's reviews, and the curve then bends upwards for free.
* `test_macro_f1_is_an_unweighted_mean` separates macro from micro averaging on a corpus where
  the two differ by a factor of three. Pooling the counts is a different, defensible metric —
  and it is not the one the docstring asks for, because it lets the field on every page bury
  the rare field a regulator asks about.
* `test_modes_are_ordered_on_the_real_corpus` and the last block of
  `test_confusion_by_field_type_is_complete` are the only cases that touch the lesson's own
  180-document corpus. Both are deterministic (fixed seeds, no clock, no I/O). The first catches
  a scorer that ignores its `mode` argument entirely, which the hand-built cases above would let
  through, since each of them is checked under a single mode.
* `test_score_field_counts_and_ratios` is deliberately ASYMMETRIC (fp != fn). With fp == fn,
  precision equals recall, so a scorer with the two formulas swapped, or with F1 taken as the
  arithmetic rather than the harmonic mean, scores exactly what the right one does.
* Several cases pin a rule the corpus never exercises but the docstring states outright: the
  lone-comma rule in money, the FIRST run of digits in an integer, popping every trailing legal
  form and only trailing ones, the inclusive fuzzy threshold, the full (doc_id, field)
  tie-break. Each was a measured survivor: a mutant that broke it scored full marks.
* Tolerances on the corpus-level case are wide bands, not equalities. They are there to pin an
  ORDERING that no correct implementation can violate, not to freeze a particular number.
"""
import importlib.util, os
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

FIELDS = tuple(lesson.SCHEMA)


def _row(doc_id: str, gold: dict, pred: dict, conf: dict | None = None) -> dict:
    """A full record: every field in the schema present on all three sides, empty by default."""
    g = {f: "" for f in FIELDS} | gold
    p = {f: "" for f in FIELDS} | pred
    c = {f: 0.5 for f in FIELDS} | (conf or {})
    return {"doc_id": doc_id, "gold": g, "pred": p, "conf": c}


# --- exercise 1: normalise_value -------------------------------------------------------

def test_normalise_money():
    cases = {
        "€1,234.50": "1234.50",       # symbol and a thousands separator
        "$1,234.50": "1234.50",
        "1234.5": "1234.50",          # one decimal digit must become two
        "1234": "1234.00",            # no decimal point at all
        "EUR 1,234.50": "1234.50",    # currency code in front
        "1234.50 USD": "1234.50",     # currency code behind
        "1.234,50": "1234.50",        # decimal comma: the last separator wins
        "  1,234.50  ": "1234.50",    # surrounding whitespace
    }
    for raw, want in cases.items():
        got = lesson.normalise_value(raw, "money")
        assert got == want, f"normalise_value({raw!r}, 'money') = {got!r}, expected {want!r}"


def test_normalise_money_lone_comma():
    cases = {
        "1234,50": "1234.50",          # lone comma, two digits after it: decimal
        "1234,5": "1234.50",           # lone comma, ONE digit after it: still decimal --
                                       # a thousands group always has three digits
        "€1,234": "1234.00",           # lone comma, three digits after it: thousands
        "EUR 12,345,678": "12345678.00",  # several commas, none decimal
    }
    for raw, want in cases.items():
        got = lesson.normalise_value(raw, "money")
        assert got == want, (
            f"normalise_value({raw!r}, 'money') = {got!r}, expected {want!r}. When a comma is "
            "the ONLY separator it is a decimal comma if one or two digits follow it and end the "
            "number, and a thousands separator otherwise. Treating every lone comma the same way, "
            "accepting only exactly two digits, or not anchoring to the end gets one of these wrong.")


def test_normalise_date():
    cases = {
        "2026-03-04": "2026-03-04",
        "04/03/2026": "2026-03-04",   # day first: this corpus is European
        "4 March 2026": "2026-03-04",
        "04 March 2026": "2026-03-04",
        "28/12/2025": "2025-12-28",
        "2025-1-9": "2025-01-09",     # unpadded ISO must still pad
    }
    for raw, want in cases.items():
        got = lesson.normalise_value(raw, "date")
        assert got == want, (f"normalise_value({raw!r}, 'date') = {got!r}, expected {want!r} — "
                             "day comes before month in a slashed date here")
    for raw, want in {"4/3/2026": "2026-03-04", "28/2/2025": "2025-02-28"}.items():
        got = lesson.normalise_value(raw, "date")
        assert got == want, (
            f"normalise_value({raw!r}, 'date') = {got!r}, expected {want!r}. This is the "
            "extractor's own house style: a slashed date with an unpadded day or month. Accept "
            "one or two digits for each, still reading the day first.")
    swapped = lesson.normalise_value("11/02/2026", "date")
    assert swapped == "2026-02-11", (
        f"normalise_value('11/02/2026', 'date') = {swapped!r}: 11 February, not 2 November. "
        "Reading a slashed date month-first silently shifts a third of this corpus.")


def test_normalise_id_and_integer():
    assert lesson.normalise_value("INV-2026-0042", "id") == "INV20260042"
    assert lesson.normalise_value("INV/2026/0042", "id") == "INV20260042"
    assert lesson.normalise_value("Inv 2026 0042", "id") == "INV20260042", "ids upper case"
    assert lesson.normalise_value("eur", "id") == "EUR"
    assert lesson.normalise_value("30", "integer") == "30"
    assert lesson.normalise_value("net 30 days", "integer") == "30", "first run of digits"
    assert lesson.normalise_value("007", "integer") == "7", "as an int, so no leading zeros"
    got = lesson.normalise_value("net 30, or 45 for exports", "integer")
    assert got == "30", (
        f"normalise_value('net 30, or 45 for exports', 'integer') = {got!r}: the FIRST run of "
        "digits only. '3045' means every digit was concatenated; '45' means the last run won.")


def test_normalise_text_drops_legal_form():
    cases = {
        "Nordwind Logistik GmbH": "nordwind logistik",
        "Vantor Marine B.V.": "vantor marine",
        "Helix Pharma Limited": "helix pharma",
        "  Kestrel   Freight  Inc ": "kestrel freight",
        "Hollandse Kaasunie N.V.": "hollandse kaasunie",
        "Sable & Finch LLP": "sable finch llp",   # LLP is not in COMPANY_SUFFIXES: leave it
        "Nordwind Logistik GmbH & Co. KG": "nordwind logistik",  # three trailing forms, all go
        "Co-Op Haulage Ltd": "co op haulage",     # a leading "co" is part of the name: keep it
    }
    for raw, want in cases.items():
        got = lesson.normalise_value(raw, "text")
        assert got == want, (f"normalise_value({raw!r}, 'text') = {got!r}, expected {want!r} — "
                             "lower case, punctuation out, whitespace collapsed, then keep "
                             "popping the LAST token while it is in COMPANY_SUFFIXES — popping "
                             "only one misses 'GmbH & Co. KG', and filtering suffix words "
                             "anywhere in the name eats a leading 'Co'")


def test_normalise_empty_and_unparseable():
    for field_type in set(lesson.SCHEMA.values()):
        assert lesson.normalise_value("", field_type) == "", f"empty in, empty out ({field_type})"
        assert lesson.normalise_value("   ", field_type) == "", f"whitespace only ({field_type})"
    assert lesson.normalise_value("on receipt", "integer") == "ON RECEIPT", (
        "an unparseable value falls back to the raw string upper cased — returning '' would "
        "turn a value the extractor really did produce into a silent abstention")
    assert lesson.normalise_value("not a number", "money") == "NOT A NUMBER"
    assert lesson.normalise_value("undated", "date") == "UNDATED"
    try:
        lesson.normalise_value("x", "telephone")
        raise AssertionError("an unknown field_type must raise ValueError, not guess a rule")
    except ValueError:
        pass


# --- exercise 2: match_value -----------------------------------------------------------

def test_match_exact_and_normalised():
    assert lesson.match_value("1234.50", "1234.50", "money", "exact")
    assert not lesson.match_value("€1,234.50", "1234.50", "money", "exact"), \
        "exact mode is RAW string equality; it must not normalise anything"
    assert not lesson.match_value(" EUR ", "EUR", "id", "exact"), "exact mode keeps whitespace"
    assert lesson.match_value("€1,234.50", "1234.50", "money", "normalised")
    assert lesson.match_value("2026-03-04", "04/03/2026", "date", "normalised")
    assert lesson.match_value("Nordwind Logistik", "Nordwind Logistik GmbH", "text", "normalised")
    assert not lesson.match_value("1234.50", "1284.50", "money", "normalised"), \
        "normalisation must not make two different amounts equal"


def test_match_fuzzy_is_confined_to_text():
    assert lesson.match_value("Nordwind Logistlk", "Nordwind Logistik GmbH", "text", "fuzzy"), \
        "one substituted character in a supplier name is exactly what fuzzy mode is for"
    assert not lesson.match_value("Nordwind Logistik", "Kaneko Precision", "text", "fuzzy"), \
        "fuzzy mode still has a threshold; two different companies are not a match"
    danger = {
        ("1284.50", "1234.50", "money"),      # similarity 0.857, 50 units apart
        ("INV20260041", "INV20260042", "id"),  # similarity 0.909, a different invoice
        ("2026-03-14", "2026-03-04", "date"),  # similarity 0.9, ten days apart
        ("31", "30", "integer"),               # a different payment term
    }
    for pred, gold, field_type in danger:
        assert not lesson.match_value(pred, gold, field_type, "fuzzy"), (
            f"fuzzy matching accepted {pred!r} for {gold!r} on a {field_type} field. Those two "
            f"strings are {lesson.similarity(pred, gold):.3f} similar and mean different "
            "things. Gate the fuzzy branch on field_type == 'text'.")


def test_match_fuzzy_threshold_is_inclusive():
    gold = "Harbourline Shipping Ltd"        # normalises to 20 characters
    at_threshold = "Harbeurlina Shippimg"    # three substitutions: similarity exactly 0.85
    below = "Harbeurlina Shyppimg"           # four substitutions: similarity 0.80
    assert lesson.similarity(lesson.normalise_value(at_threshold, "text"),
                             lesson.normalise_value(gold, "text")) == lesson.FUZZY_THRESHOLD
    assert lesson.match_value(at_threshold, gold, "text", "fuzzy"), (
        "a similarity EXACTLY at FUZZY_THRESHOLD was rejected. The docstring says 'at or "
        "above': compare with >=, not >.")
    assert not lesson.match_value(below, gold, "text", "fuzzy"), \
        "a similarity below FUZZY_THRESHOLD must not match"


def test_match_rejects_empty_and_unknown_modes():
    for mode in lesson.MATCH_MODES:
        assert not lesson.match_value("", "EUR", "id", mode), "an empty prediction never matches"
        assert not lesson.match_value("EUR", "", "id", mode), "an empty gold never matches"
        assert not lesson.match_value("", "", "id", mode), (
            "two empty sides are not a match — a cell where nothing was expected and nothing "
            "was produced is a true negative, and it is counted somewhere else entirely")
        assert not lesson.match_value("   ", "EUR", "id", mode), "whitespace is not a value"
    for bad in ("approximate", "EXACT", "", "normalized"):
        try:
            lesson.match_value("a", "a", "text", bad)
            raise AssertionError(f"mode {bad!r} must raise ValueError, not return a verdict")
        except ValueError:
            pass


# --- exercise 3: score_field and macro_f1 ----------------------------------------------

def test_score_field_counts_and_ratios():
    rows = [
        _row("D1", {"currency": "EUR"}, {"currency": "eur"}),   # tp (normalised)
        _row("D2", {"currency": "USD"}, {}),                    # fn: missed
        _row("D3", {}, {"currency": "GBP"}),                    # fp: spurious
        _row("D4", {}, {}),                                     # neither
        _row("D5", {"currency": "GBP"}, {"currency": "GBP"}),   # tp
        _row("D6", {"currency": "EUR"}, {}),                    # fn: missed
    ]
    s = lesson.score_field(rows, "currency", "normalised")
    assert (s.tp, s.fp, s.fn) == (2, 1, 2), f"expected tp=2 fp=1 fn=2, got {s.tp} {s.fp} {s.fn}"
    assert abs(s.precision - 2 / 3) < 1e-9, (
        f"precision is tp/(tp+fp) = 2/3 here, got {s.precision:.4f}"
        + (" — that is tp/(tp+fn): precision and recall are the wrong way round"
           if abs(s.precision - 1 / 2) < 1e-9 else ""))
    assert abs(s.recall - 1 / 2) < 1e-9, f"recall is tp/(tp+fn) = 1/2 here, got {s.recall:.4f}"
    assert abs(s.f1 - 4 / 7) < 1e-9, (
        f"f1 is the harmonic mean 2PR/(P+R) = 4/7 here, got {s.f1:.4f}"
        + (" — that is the arithmetic mean (P+R)/2" if abs(s.f1 - 7 / 12) < 1e-9 else ""))
    assert s.field == "currency" and s.mode == "normalised", \
        "a FieldScore carries the field and mode it was computed under"


def test_score_field_wrong_value_counts_twice():
    rows = [_row("D1", {"currency": "EUR"}, {"currency": "USD"})]
    s = lesson.score_field(rows, "currency", "normalised")
    assert (s.tp, s.fp, s.fn) == (0, 1, 1), (
        f"one wrong value scored tp={s.tp} fp={s.fp} fn={s.fn}. It is BOTH: you claimed "
        "something untrue (a false positive) and you failed to produce the true value (a "
        "false negative). Counting it once flatters whichever of precision or recall you "
        "are reporting.")
    assert s.precision == 0.0 and s.recall == 0.0 and s.f1 == 0.0
    mixed = [_row("D1", {"currency": "EUR"}, {"currency": "EUR"}),
             _row("D2", {"currency": "EUR"}, {"currency": "USD"}),
             _row("D3", {"currency": "EUR"}, {"currency": "EUR"})]
    m = lesson.score_field(mixed, "currency", "normalised")
    assert (m.tp, m.fp, m.fn) == (2, 1, 1), \
        f"two right and one wrong is tp=2 fp=1 fn=1, got {m.tp} {m.fp} {m.fn}"


def test_score_field_zero_denominators():
    quiet = [_row("D1", {}, {}), _row("D2", {}, {})]
    s = lesson.score_field(quiet, "currency", "exact")
    assert (s.tp, s.fp, s.fn) == (0, 0, 0)
    assert s.precision == 0.0 and s.recall == 0.0 and s.f1 == 0.0, \
        "no predictions and no gold values: return 0.0, do not divide by zero"
    only_fp = [_row("D1", {}, {"currency": "EUR"})]
    s2 = lesson.score_field(only_fp, "currency", "exact")
    assert s2.precision == 0.0 and s2.recall == 0.0 and s2.f1 == 0.0, (
        "all false positives and no gold: precision is 0.0, recall has a zero denominator "
        "and is 0.0, and F1 must not raise ZeroDivisionError on p + r == 0")


def test_macro_f1_is_an_unweighted_mean():
    rows = [_row(f"D{i}", {"currency": "EUR", "counterparty": "Kaneko Precision KK"},
                 {"currency": "EUR", "counterparty": "Torrent Robotics Inc"})
            for i in range(10)]
    got = lesson.macro_f1(rows, "normalised")
    assert abs(got - 1 / len(FIELDS)) < 1e-9, (
        f"macro_f1 returned {got:.4f}; expected {1 / len(FIELDS):.4f}. One field of "
        f"{len(FIELDS)} is perfect, one is wrong on every row and the rest are empty on both "
        "sides, so the unweighted mean of the per-field F1 scores is 1/"
        f"{len(FIELDS)}. Pooling the counts instead (micro averaging) gives 0.5, and averaging "
        "only over the fields that have data gives 0.5 as well.")
    perfect = [_row("D1", {f: "EUR" for f in FIELDS}, {f: "EUR" for f in FIELDS})]
    assert abs(lesson.macro_f1(perfect, "exact") - 1.0) < 1e-9, \
        "every field right on every document is a macro F1 of exactly 1.0"


def test_modes_are_ordered_on_the_real_corpus():
    records = lesson.RECORDS
    exact = lesson.macro_f1(records, "exact")
    normalised = lesson.macro_f1(records, "normalised")
    fuzzy = lesson.macro_f1(records, "fuzzy")
    assert 0.2 < exact < 0.7, f"exact-mode macro F1 on the lesson corpus was {exact:.3f}"
    assert normalised > exact + 0.2, (
        f"normalised ({normalised:.3f}) must beat exact ({exact:.3f}) by a wide margin on this "
        "corpus — if it does not, score_field is probably ignoring its mode argument")
    assert fuzzy > normalised, (
        f"fuzzy ({fuzzy:.3f}) must beat normalised ({normalised:.3f}): the text field carries "
        "OCR-style single-character damage that only fuzzy matching forgives")
    assert fuzzy - normalised < 0.1, (
        f"fuzzy beat normalised by {fuzzy - normalised:.3f}. Fuzzy matching should only move "
        "the ONE text field in this schema; a bigger jump means it is being applied to the "
        "money, date, id or integer fields as well.")


# --- exercise 4: the confusion analysis -------------------------------------------------

def test_classify_cell_labels():
    cases = [
        (("", ""), "true_negative"), (("   ", ""), "true_negative"),
        (("", "30"), "miss"), (("  ", "30"), "miss"),
        (("30", ""), "spurious"),
        (("net 30 days", "30"), "correct"),
        (("45", "30"), "wrong_value"),
    ]
    for (pred, gold), want in cases:
        got = lesson.classify_cell(pred, gold, "integer", "normalised")
        assert got == want, f"classify_cell({pred!r}, {gold!r}) = {got!r}, expected {want!r}"
    assert set(dict.fromkeys(w for _, w in cases)) <= set(lesson.ERROR_LABELS), \
        "every label must come from ERROR_LABELS"
    assert lesson.classify_cell("€1,234.50", "1234.50", "money", "exact") == "wrong_value", \
        "classify_cell must honour the mode it is given, not pick its own"
    assert lesson.classify_cell("€1,234.50", "1234.50", "money", "normalised") == "correct"


def test_confusion_by_field_type_is_complete():
    empty = lesson.confusion_by_field_type([], "normalised")
    assert set(empty) == set(lesson.SCHEMA.values()), (
        f"one row per field TYPE, not per field name: expected {sorted(set(lesson.SCHEMA.values()))}, "
        f"got {sorted(empty)}")
    for field_type, row in empty.items():
        assert set(row) == set(lesson.ERROR_LABELS), (
            f"row {field_type!r} is missing labels {sorted(set(lesson.ERROR_LABELS) - set(row))}"
            " — a table with holes in it cannot be read across rows")
        assert all(v == 0 for v in row.values()), "no records means every count is 0"

    rows = [_row("D1", {"invoice_id": "INV-1", "currency": "EUR"},
                 {"invoice_id": "INV1", "currency": ""})]
    table = lesson.confusion_by_field_type(rows, "normalised")
    assert table["id"]["correct"] == 1 and table["id"]["miss"] == 1, (
        "invoice_id and currency are both 'id' fields, so they add into the same row: "
        f"got correct={table['id']['correct']} miss={table['id']['miss']}")
    assert sum(sum(r.values()) for r in table.values()) == len(FIELDS), \
        "every cell of every record lands in exactly one label, so the table totals to n*fields"

    real = lesson.confusion_by_field_type(lesson.RECORDS, "normalised")
    total = sum(sum(r.values()) for r in real.values())
    assert total == len(lesson.RECORDS) * len(FIELDS), (
        f"on the lesson corpus the table totals {total}, but there are "
        f"{len(lesson.RECORDS) * len(FIELDS)} cells")
    assert real["integer"]["spurious"] > 0 and real["integer"]["true_negative"] > 0, (
        "payment_terms_days is absent from many documents and the extractor answers on some "
        "of them, so the integer row must show both spurious extractions and true negatives")


def test_confusion_honours_mode():
    # One non-empty cell, and totals taken over the whole table, so this case grades the mode
    # and nothing else: a table with holes or the wrong keys loses its points in the case above.
    rows = [_row("D1", {"invoice_id": "INV-1"}, {"invoice_id": "INV1"})]
    for mode, want in (("exact", (0, 1)), ("normalised", (1, 0))):
        t = lesson.confusion_by_field_type(rows, mode)
        got = tuple(sum(row.get(lab, 0) for row in t.values()) for lab in ("correct", "wrong_value"))
        assert got == want, (
            f"'INV1' against 'INV-1' under mode {mode!r} counted correct={got[0]} "
            f"wrong_value={got[1]}; expected correct={want[0]} wrong_value={want[1]}. "
            "confusion_by_field_type must pass its OWN mode argument to classify_cell, not a "
            "mode of its choosing.")


# --- exercise 5: the review queue -------------------------------------------------------

def test_review_queue_orders_by_confidence():
    rows = [_row("D1", {}, {}, {"currency": 0.9, "counterparty": 0.1}),
            _row("D2", {}, {}, {"currency": 0.2, "counterparty": 0.8})]
    queue = lesson.review_queue(rows, 3)
    assert queue[0] == ("D1", "counterparty"), f"lowest confidence first, got {queue[0]}"
    assert queue[1] == ("D2", "currency"), f"then the next lowest, got {queue[1]}"
    assert isinstance(queue, tuple) and all(isinstance(c, tuple) and len(c) == 2 for c in queue), \
        "return a tuple of (doc_id, field) pairs"
    confs = {(r["doc_id"], f): r["conf"][f] for r in rows for f in FIELDS}
    ordered = [confs[c] for c in lesson.review_queue(rows, 12)]
    assert ordered == sorted(ordered), f"the whole queue must be sorted ascending: {ordered}"


def test_review_queue_tie_break_is_deterministic():
    rows = [_row("D2", {}, {}), _row("D1", {}, {})]   # every confidence 0.5
    first = lesson.review_queue(rows, 1)
    assert first == (("D1", "counterparty"),), (
        f"got {first}. With every confidence equal the tie-break decides: (doc_id, field) "
        "ascending puts D1 before D2, and 'counterparty' first alphabetically. Without a "
        "deterministic tie-break two runs of the same policy queue different work.")
    assert lesson.review_queue(rows, 4) == lesson.review_queue(rows, 4), "and it is stable"
    fields = sorted(FIELDS)
    want = tuple(("D1", f) for f in fields) + (("D2", fields[0]), ("D2", fields[1]))
    got = lesson.review_queue(rows, len(fields) + 2)
    assert got == want, (
        f"got {got}. Ties break on doc_id FIRST, then field: every cell of D1 comes before any "
        "cell of D2. Sorting on (field, doc_id) alternates the two documents instead.")
    assert lesson.review_queue(rows, 4)[:1] == first, "a bigger budget extends the same queue"


def test_review_queue_budget_boundaries():
    rows = [_row("D1", {}, {}), _row("D2", {}, {})]
    n_cells = len(rows) * len(FIELDS)
    assert lesson.review_queue(rows, 0) == (), "a budget of 0 routes nothing at all"
    assert len(lesson.review_queue(rows, 1)) == 1
    assert len(lesson.review_queue(rows, n_cells)) == n_cells
    assert len(lesson.review_queue(rows, n_cells * 10)) == n_cells, \
        "a budget larger than the corpus routes every cell and no more — do not pad the queue"
    assert len(set(lesson.review_queue(rows, n_cells))) == n_cells, "no cell twice"
    try:
        lesson.review_queue(rows, -1)
        raise AssertionError("a negative budget is a caller bug: raise ValueError")
    except ValueError:
        pass


def test_review_queue_can_route_a_silent_cell():
    rows = [_row("D1", {"currency": "EUR"}, {}, {"currency": 0.0}),
            _row("D2", {"currency": "USD"}, {"currency": "USD"}, {f: 0.99 for f in FIELDS})]
    queue = lesson.review_queue(rows, 1)
    assert queue == (("D1", "currency"),), (
        f"got {queue}. A cell where the extractor produced nothing is still a candidate for "
        "review — a confident silence and a missed field look identical from the outside, and "
        "filtering the queue down to non-empty predictions makes every miss unreviewable.")


def test_apply_reviews_corrects_routed_cells():
    rows = [_row("D1", {"currency": "EUR", "counterparty": "Kaneko Precision KK"},
                 {"currency": "USD", "counterparty": "Kaneko Precision KK"},
                 {"currency": 0.3, "counterparty": 0.9})]
    after = lesson.apply_reviews(rows, [("D1", "currency")])
    assert isinstance(after, list) and len(after) == len(rows), "one record out per record in"
    assert after[0]["pred"]["currency"] == "EUR", "a reviewed cell takes the gold value"
    assert after[0]["conf"]["currency"] == 1.0, "and the reviewer is certain afterwards"
    assert after[0]["pred"]["counterparty"] == "Kaneko Precision KK", "untouched elsewhere"
    assert after[0]["conf"]["counterparty"] == 0.9, "and its confidence is untouched too"
    assert after[0]["gold"] == rows[0]["gold"], "gold labels never change"
    assert after[0]["doc_id"] == "D1", "doc_id survives"
    full = lesson.apply_reviews(rows, lesson.review_queue(rows, 10_000))
    assert abs(lesson.macro_f1(full, "exact") - lesson.macro_f1(full, "normalised")) < 1e-9, \
        "after reviewing every cell, prediction and gold are identical strings"


def test_apply_reviews_leaves_the_input_alone():
    rows = [_row("D1", {"currency": "EUR"}, {"currency": ""}, {"currency": 0.0})]
    snapshot = ({f: rows[0]["pred"][f] for f in FIELDS},
                {f: rows[0]["conf"][f] for f in FIELDS})
    lesson.apply_reviews(rows, [("D1", "currency")])
    assert rows[0]["pred"] == snapshot[0] and rows[0]["conf"] == snapshot[1], (
        "apply_reviews mutated its input. Copy the dicts. The bug is invisible at one budget "
        "and fatal across a sweep: every later budget inherits the reviews of the earlier "
        "one, so the quality/cost curve rises for free.")
    twice_a = lesson.macro_f1(lesson.apply_reviews(rows, lesson.review_queue(rows, 1)), "exact")
    twice_b = lesson.macro_f1(lesson.apply_reviews(rows, lesson.review_queue(rows, 1)), "exact")
    assert twice_a == twice_b, "scoring the same budget twice must give the same answer"


RUBRIC = [
    (test_normalise_money, 5,
     "strip the symbol and the code, decide which separator is decimal, then format to 2dp"),
    (test_normalise_money_lone_comma, 2,
     "a lone comma is decimal when one or two digits follow it and end the number"),
    (test_normalise_date, 5,
     "three surface forms, one ISO output; a slashed date in this corpus is day-first, and "
     "its day and month may have one digit or two"),
    (test_normalise_id_and_integer, 3,
     "ids: upper case, alphanumerics only. integers: the first run of digits, as an int"),
    (test_normalise_text_drops_legal_form, 4,
     "lower case, punctuation to spaces, collapse whitespace, then pop trailing tokens that "
     "are in COMPANY_SUFFIXES — and only those"),
    (test_normalise_empty_and_unparseable, 3,
     "empty in, empty out; anything that will not parse falls back to the raw string upper "
     "cased, never to the empty string"),
    (test_match_exact_and_normalised, 4,
     "exact compares the raw strings; normalised compares normalise_value of both sides"),
    (test_match_fuzzy_is_confined_to_text, 5,
     "the fuzzy branch runs only when field_type == 'text'; on every other type, fall back to "
     "normalised equality"),
    (test_match_fuzzy_threshold_is_inclusive, 2,
     "'at or above FUZZY_THRESHOLD' is >=, not >"),
    (test_match_rejects_empty_and_unknown_modes, 3,
     "an empty side is never a match, and an unrecognised mode raises ValueError"),
    (test_score_field_counts_and_ratios, 5,
     "count per cell, then divide once at the end, guarding both denominators; precision "
     "divides by predictions, recall by gold values, and F1 is their harmonic mean"),
    (test_score_field_wrong_value_counts_twice, 5,
     "a prediction that is present and does not match is a false positive AND a false negative"),
    (test_score_field_zero_denominators, 3,
     "return 0.0 for precision, recall and F1 rather than raising ZeroDivisionError"),
    (test_macro_f1_is_an_unweighted_mean, 4,
     "average the per-field F1 scores over every field in SCHEMA, including the ones with no "
     "data — do not pool the counts and do not skip empty fields"),
    (test_modes_are_ordered_on_the_real_corpus, 4,
     "score_field must actually use its mode argument, and fuzzy must move only the text field"),
    (test_classify_cell_labels, 4,
     "both empty is true_negative; gold only is miss; prediction only is spurious; otherwise "
     "ask match_value"),
    (test_confusion_by_field_type_is_complete, 5,
     "pre-fill every field type with every label at 0, then add one per cell; two fields of "
     "the same type share a row"),
    (test_confusion_honours_mode, 2,
     "pass the mode you were given through to classify_cell"),
    (test_review_queue_orders_by_confidence, 4,
     "sort every cell on (confidence, doc_id, field) ascending and take the first `budget`"),
    (test_review_queue_tie_break_is_deterministic, 3,
     "ties break on (doc_id, field) so the same policy always queues the same work"),
    (test_review_queue_budget_boundaries, 4,
     "0 routes nothing, a budget past the end routes everything once, a negative budget raises"),
    (test_review_queue_can_route_a_silent_cell, 3,
     "candidates are all cells, not just the ones where the extractor said something"),
    (test_apply_reviews_corrects_routed_cells, 4,
     "a reviewed cell takes the gold value and confidence 1.0; nothing else changes"),
    (test_apply_reviews_leaves_the_input_alone, 4,
     "copy each record's pred and conf dicts before you change them, instead of writing into "
     "the ones you were given"),
]
