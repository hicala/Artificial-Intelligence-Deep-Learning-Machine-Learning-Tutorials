"""Autograder rubric for P02-L03-sequence-labelling. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These rules matter in this particular lesson and are deliberate:

* Every case grades OBSERVABLE BEHAVIOUR, never one particular implementation. `word_shape`
  may use a regex or a loop; `viterbi_decode` may keep its backpointers in a list or an array;
  nothing here inspects source.
* `test_viterbi_matches_brute_force` is the only honest way to grade an exact decoder: it
  enumerates all 3**5 tag sequences on a small random problem and checks that the returned
  path scores as well as the best of them. It catches greedy decoding, an argmax on the wrong
  axis, and a decoder that forgets the START row — three bugs that all return a well-shaped
  answer and would pass every hand-written case.
* `test_perceptron_update_transition_deltas` grades the half of the update people skip. A
  model that only fixes emissions still trains, still improves, and still scores well on every
  emission case above; it simply never learns a tag ORDER. The case is built so that one wrong
  tag breaks two bigrams, and checks both.
* `test_decode_spans_opens_on_a_stray_i` exists because Viterbi is free to emit `O I-x`.
  Dropping those tokens is the single most common span-reader bug and it shows up as a model
  regression rather than as a reader bug, which is how it survives.
* `test_decode_spans_absent_and_first_span_wins` grades the mirror image: `O` must CLOSE the
  open span, not merely be skipped. A reader that `continue`s on `O` buffers exactly the same
  tokens for every span in this lesson's corpus — 0 of 200 documents decode differently — and
  passes every other span case. It diverges only when an `I-` of the same field follows the
  `O`, which is why the case is written that way rather than left to the corpus to catch.
* `test_fields_improved_orders_and_thresholds` separates "sorted by baseline F1 ascending"
  from the attractive wrong answer "sorted by delta descending". The two agree on most inputs;
  the case here is one where they do not, because ordering by delta buries the field that was
  worst to begin with — which is the field a reader is looking for.
* `test_tagger_beats_baseline_on_held_out` is the only case that trains on the lesson's own
  corpus. It is deterministic (fixed seeds, no clock, no I/O) and it catches an implementation
  that passes every unit case but does not actually learn — a feature template that ignores
  its context, an update that moves the wrong sign. Its tolerances are wide bands, not
  equalities: they pin an ORDERING no correct implementation can violate, and leave room for a
  decoder that breaks ties differently from the reference.
"""
import importlib.util, itertools, os
from pathlib import Path

import numpy as np

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

FIELDS = tuple(lesson.SCHEMA)
Token = lesson.Token


def _line(texts, head=None, line=0, start_pos=0):
    """A line of Tokens sharing one line index and line_head."""
    head = head if head is not None else texts[0].lower()
    return [Token(t, line, start_pos + k, 1, head) for k, t in enumerate(texts)]


# --- exercise 1: word_shape ------------------------------------------------------------

def test_word_shape_collapses_runs():
    cases = {
        "INV-2026-0042": "X-d-d",
        "Invoice": "Xx",
        "1,234.50": "d,d.d",
        "B.V.": "X.X.",
        "REMITTANCE": "X",
        "2026-03-04": "d-d-d",
        "PO-2026-1188": "X-d-d",
        "accounts": "x",
    }
    for raw, want in cases.items():
        got = lesson.word_shape(raw)
        assert got == want, (f"word_shape({raw!r}) = {got!r}, expected {want!r} — map every "
                             "character to X/x/d or itself, THEN collapse runs of the same "
                             "result; 'INV' is one X, not three")


def test_word_shape_edges():
    assert lesson.word_shape("") == "", "empty in, empty out"
    assert lesson.word_shape("..") == ".", "punctuation runs collapse too, not just letters"
    assert lesson.word_shape("&") == "&", "a character with no class maps to itself"
    assert lesson.word_shape("Xx") == "Xx", "the shape of a shape-looking token is not magic"
    assert lesson.word_shape("aB1.") == "xXd.", "no collapsing when the classes differ"


# --- exercise 2: token_features --------------------------------------------------------

def test_token_features_core():
    toks = _line(["Amount", "due:", "EUR", "1,234.50"], head="amount")
    got = lesson.token_features(toks, 3)
    for feat in ("bias", "w=1,234.50", "shape=d,d.d", "pre3=1,2", "suf3=.50"):
        assert feat in got, (f"{feat!r} missing from token_features — the token's own word, "
                             "shape and three-character affixes are features 1 to 5")
    assert len(got) == 12, f"token_features returned {len(got)} features, expected exactly 12"
    assert all(isinstance(f, str) for f in got), "every feature is a string"
    # A token that actually has letters in it, so a template that forgets to lower-case the
    # word cannot slip past on a token made of digits and punctuation.
    upper = lesson.token_features(toks, 2)
    for feat in ("w=eur", "pre3=eur", "suf3=eur", "shape=X"):
        assert feat in upper, (f"{feat!r} missing for the token 'EUR' — the word, the prefix "
                               "and the suffix are all taken from the LOWER-CASED text; only "
                               f"the shape keeps the case (you gave {upper})")
    mixed = _line(["Remit", "to:", "Nordwind"], head="remit")
    assert "w=nordwind" in lesson.token_features(mixed, 2), "lower-case the word feature"


def test_token_features_context():
    toks = _line(["Amount", "due:", "EUR", "1,234.50"], head="amount")
    last = lesson.token_features(toks, 3)
    for feat in ("w-1=eur", "w+1=<EOS>", "shape-1=X", "shape+1=<EOS>", "head=amount",
                 "pos=3", "head|shape=amount|d,d.d"):
        assert feat in last, (f"{feat!r} missing — features 6 to 12 are the neighbours, their "
                              "shapes, the line head, the position in the line and the "
                              "head|shape conjunction")
    first = lesson.token_features(toks, 0)
    assert "w-1=<BOS>" in first and "shape-1=<BOS>" in first, \
        "at position 0 the previous token is the literal <BOS>, for both the word and the shape"
    assert "w+1=due:" in first, "the next token is lower-cased, punctuation kept"


def test_token_features_is_position_sensitive():
    # The line starts at tok.pos == 2, so a template that uses the LIST index i in place of
    # tok.pos gets the position feature wrong on every real document.
    toks = _line(["Vantor", "Marine", "B.V."], head="remit", start_pos=2)
    got = lesson.token_features(toks, 0)
    assert "pos=2" in got, ("pos comes from tok.pos — the position within the LINE — not from "
                            f"the index into the list you were handed (you gave {got})")
    long_line = _line(["a", "b", "c", "d", "e", "f", "g", "h"], head="page")
    assert "pos=4" in lesson.token_features(long_line, 7), \
        "pos is capped at 4; token 7 of a line is still pos=4"
    a = lesson.token_features(toks, 0)
    b = lesson.token_features(toks, 1)
    assert a != b, "two different tokens of the same line must not produce identical features"


# --- exercise 3: decode_spans ----------------------------------------------------------

def test_decode_spans_joins_a_span():
    toks = _line(["Remit", "to:", "Vantor", "Marine", "B.V."], head="remit")
    got = lesson.decode_spans(toks, ["O", "O", "B-counterparty", "I-counterparty",
                                     "I-counterparty"])
    assert got["counterparty"] == "Vantor Marine B.V.", \
        f"got {got['counterparty']!r} — join the span's token texts with single spaces"
    assert set(got) == set(FIELDS), "one key per field in SCHEMA, always"
    mixed = _line(["Total:", "EUR", "1,234.50"], head="total")
    out = lesson.decode_spans(mixed, ["O", "B-currency", "B-total_amount"])
    assert out["currency"] == "EUR" and out["total_amount"] == "1,234.50", \
        "two adjacent one-token spans of different fields are two spans, not one"


def test_decode_spans_opens_on_a_stray_i():
    toks = _line(["Remit", "to:", "Vantor", "Marine"], head="remit")
    got = lesson.decode_spans(toks, ["O", "O", "I-counterparty", "I-counterparty"])
    assert got["counterparty"] == "Vantor Marine", (
        "an I- tag with no B- before it must OPEN a span. Nothing in Viterbi forbids 'O I-x'; "
        f"dropping those tokens loses the span (you returned {got['counterparty']!r})")
    switch = lesson.decode_spans(toks, ["B-currency", "I-invoice_id", "O", "O"])
    assert switch["currency"] == "Remit" and switch["invoice_id"] == "to:", \
        "I- for a DIFFERENT field than the open span closes that span and opens a new one"


def test_decode_spans_absent_and_first_span_wins():
    toks = _line(["a", "b", "c", "d"], head="a")
    got = lesson.decode_spans(toks, ["B-currency", "O", "B-currency", "O"])
    assert got["currency"] == "a", \
        f"only the FIRST span counts; you returned {got['currency']!r}"
    assert got["invoice_id"] == "", "a field with no span is the empty string, not missing"
    # `O` is a BOUNDARY, not a token to skip. Without this case a reader that simply `continue`s
    # on O passes every other span test: the tokens it buffers are the same ones, only the
    # moment it closes moves. It diverges exactly here, where an I- of the same field turns up
    # after the O and is allowed to rejoin a span that should already have been closed.
    closed = lesson.decode_spans(toks, ["B-currency", "I-currency", "O", "I-currency"])
    assert closed["currency"] == "a b", (
        f"got {closed['currency']!r}, expected 'a b' — `O` CLOSES the open span. Skipping O "
        "instead of closing on it lets the later I-currency rejoin the first span, which on a "
        "real page staples a value from the footer onto a value from the header.")
    assert lesson.decode_spans([], []) == {f: "" for f in FIELDS}, "empty document, empty values"


# --- exercise 4: sequence_scores -------------------------------------------------------

def test_sequence_scores_sums_active_features():
    m = lesson.Tagger(3, 4)
    m.w[0] = [1.0, 2.0, 4.0, 8.0]
    m.w[1] = [-1.0, 0.0, 0.5, 100.0]
    m.w[2] = [0.0, 0.0, 0.0, 0.0]
    got = lesson.sequence_scores(m, [np.array([0, 3]), np.array([1, 2, 3])])
    assert np.allclose(got[0], [9.0, 99.0, 0.0]), (
        f"row 0 = {got[0]}, expected [9. 99. 0.] — sum model.w[tag, f] over the ACTIVE ids f")
    assert np.allclose(got[1], [14.0, 100.5, 0.0]), f"row 1 = {got[1]}, expected [14. 100.5 0.]"


def test_sequence_scores_empty_and_shape():
    m = lesson.Tagger(5, 3)
    m.w[2] = [7.0, 0.0, 0.0]
    got = lesson.sequence_scores(m, [np.array([], dtype=np.intp), np.array([0])])
    assert got.shape == (2, 5), (f"shape {got.shape}, expected (2, 5) — rows are TOKENS and "
                                 "columns are TAGS, not the transpose")
    assert np.allclose(got[0], 0.0), "a token with no known features scores zero for every tag"
    assert got[1, 2] == 7.0 and got[1, 0] == 0.0, "row 1 must read column 0 of every tag row"
    assert lesson.sequence_scores(m, []).shape == (0, 5), "an empty sequence gives 0 rows"


# --- exercise 5: viterbi_decode --------------------------------------------------------

def _path_score(path, emissions, transitions):
    n_tags = emissions.shape[1]
    total = transitions[n_tags, path[0]] + emissions[0, path[0]]
    for i in range(1, len(path)):
        total += transitions[path[i - 1], path[i]] + emissions[i, path[i]]
    return total


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(20260916)
    emissions = rng.normal(size=(5, 3))
    transitions = rng.normal(size=(4, 3))
    got = lesson.viterbi_decode(emissions, transitions)
    assert len(got) == 5, f"expected one tag per token, got {len(got)}"
    best = max(itertools.product(range(3), repeat=5),
               key=lambda p: _path_score(p, emissions, transitions))
    want, mine = _path_score(best, emissions, transitions), _path_score(got, emissions, transitions)
    assert mine >= want - 1e-9, (
        f"your path scores {mine:.4f}; the best sequence scores {want:.4f}. A greedy argmax "
        "per position, an argmax on the wrong axis, or a dp[0] that forgets the START row all "
        "return a well-shaped answer that is not the maximum.")


def test_viterbi_uses_transitions():
    emissions = np.array([[0.0, 1.0], [0.0, 1.0]])
    transitions = np.zeros((3, 2))
    transitions[1, 1] = -5.0
    got = lesson.viterbi_decode(emissions, transitions)
    assert got == [1, 0], (f"got {got}, expected [1, 0] — tag 1 wins at both positions on its "
                           "emissions alone, so a greedy decoder answers [1, 1]. Only a "
                           "decoder that reads transitions[1, 1] flips the second position.")
    reward = np.zeros((3, 2))
    reward[0, 1] = 4.0
    assert lesson.viterbi_decode(np.array([[1.0, 0.0], [0.0, 0.0]]), reward) == [0, 1], \
        "a transition can also make a WORSE local tag the right answer at the next position"


def test_viterbi_start_row_and_edges():
    # Flat emissions, so the START row is the ONLY thing that can pick the first tag. Tag 1 is
    # the answer; a decoder that drops the START row falls back to argmax of a row of zeros,
    # which is tag 0.
    start = np.zeros((3, 2))
    start[2, 1] = 9.0
    assert lesson.viterbi_decode(np.zeros((1, 2)), start) == [1], (
        "the LAST row of transitions (index n_tags) scores the tag at position 0; ignoring it "
        "makes the first token's tag a coin toss decided by argmax on a tie")
    start3 = np.zeros((4, 3))
    start3[3, 2] = 5.0
    assert lesson.viterbi_decode(np.zeros((3, 3)), start3)[0] == 2, \
        "the START row still decides position 0 when the sequence is longer than one token"
    assert lesson.viterbi_decode(np.zeros((0, 2)), np.zeros((3, 2))) == [], "empty in, empty out"
    got = lesson.viterbi_decode(np.tile([0.0, 0.1, -0.2], (9, 1)), np.zeros((4, 3)))
    assert got == [1] * 9, f"with flat transitions the answer is the per-token argmax; got {got}"


# --- exercise 6: perceptron_update -----------------------------------------------------

def test_perceptron_update_noop_when_correct():
    m = lesson.Tagger(3, 4)
    n = lesson.perceptron_update(m, [np.array([0, 1]), np.array([2])], [1, 2], [1, 2])
    assert n == 0, f"a correct decode has 0 mismatched tokens, you returned {n}"
    assert not m.w.any(), "a correct decode must not move a single emission weight"
    assert not m.t.any(), ("a correct decode must not move a single transition weight either — "
                          "the bigrams agree by construction when the sequences agree")


def test_perceptron_update_emission_deltas():
    m = lesson.Tagger(3, 5)
    feats = [np.array([0, 1]), np.array([2]), np.array([3, 4])]
    n = lesson.perceptron_update(m, feats, [1, 2, 0], [0, 2, 0])
    assert n == 1, f"one position differs, you returned {n}"
    assert np.allclose(m.w[1, [0, 1]], [1.0, 1.0]), \
        "the GOLD tag's row goes up by 1 at the wrong token's feature ids"
    assert np.allclose(m.w[0, [0, 1]], [-1.0, -1.0]), \
        "the PREDICTED tag's row goes down by 1 at the same ids"
    assert not m.w[:, 2].any() and not m.w[:, 3].any() and not m.w[:, 4].any(), (
        "tokens 1 and 2 were tagged correctly, so their features must not move at all — an "
        "update that touches every position is not the perceptron rule")
    assert m.w.sum() == 0.0, "every +1 is matched by a -1; the emission update is zero-sum"


def test_perceptron_update_transition_deltas():
    m = lesson.Tagger(3, 3)
    feats = [np.array([0]), np.array([1]), np.array([2])]
    lesson.perceptron_update(m, feats, [1, 2, 0], [0, 2, 0])
    start = 3
    assert m.t[start, 1] == 1.0 and m.t[start, 0] == -1.0, (
        "position 0's predecessor is the START row — the last row of model.t, index "
        "model.w.shape[0]; hard-coding this lesson's 13 breaks on any other model")
    assert m.t[1, 2] == 1.0 and m.t[0, 2] == -1.0, (
        "the bigram at position 1 is wrong too: gold (1,2) against predicted (0,2), even "
        "though the TAG at position 1 was right. One wrong tag breaks two bigrams, so the "
        "transition pass runs over every position, not only the mismatched ones.")
    assert m.t[2, 0] == 0.0, (
        "the bigram at position 2 is (2,0) in both sequences, so it must not move — an update "
        "that rewards every gold bigram unconditionally learns the tag prior, not the errors")


def test_perceptron_update_returns_mismatch_count():
    m = lesson.Tagger(4, 3)
    feats = [np.array([0]), np.array([1]), np.array([2])]
    assert lesson.perceptron_update(m, feats, [1, 2, 3], [0, 0, 0]) == 3, \
        "three positions differ; return the COUNT of them, not a bool and not the sequence"
    m2 = lesson.Tagger(4, 3)
    assert lesson.perceptron_update(m2, feats, [1, 2, 3], [1, 0, 3]) == 1, "one position differs"


# --- exercise 7: improvement_table / fields_improved ------------------------------------

def _records(pred_by_field, gold_by_field):
    return [{"doc_id": "D1", "gold": {f: "" for f in FIELDS} | gold_by_field,
             "pred": {f: "" for f in FIELDS} | pred_by_field}]


def test_improvement_table_covers_every_field():
    gold = {"currency": "EUR", "counterparty": "Vantor Marine B.V."}
    base = _records({"counterparty": "Vantor Marine B.V."}, gold)
    tagged = _records({"currency": "EUR"}, gold)
    table = lesson.improvement_table(base, tagged)
    assert list(table) == list(FIELDS), (
        "one entry per field in SCHEMA, in SCHEMA order, including the four fields neither "
        "extractor found — a field nobody extracts is a finding, not an absence")
    assert table["currency"].baseline_f1 == 0.0 and table["currency"].tagger_f1 == 1.0
    assert abs(table["currency"].delta - 1.0) < 1e-12, "delta is tagger_f1 minus baseline_f1"
    assert abs(table["counterparty"].delta + 1.0) < 1e-12, \
        "a field the tagger lost must carry a NEGATIVE delta, not an absolute difference"
    assert table["counterparty"].field_type == "text" and table["currency"].field_type == "id", \
        "carry the field TYPE from SCHEMA as well as the name"
    assert table["invoice_id"].delta == 0.0, "both scored nothing, so the delta is exactly 0.0"
    # `mode` must reach score_field on BOTH sides. Each prediction below matches its gold under
    # normalisation and not byte-for-byte, so the two modes give genuinely different F1 — which
    # a table that hard-codes one mode, or forwards `mode` to only one of the two extractors,
    # cannot reproduce. Checking that the keys survive the argument checks nothing.
    gold2 = {"total_amount": "1,234.50", "counterparty": "Nordwind Logistik GmbH"}
    base2 = _records({"counterparty": "Nordwind Logistik"}, gold2)
    tagged2 = _records({"total_amount": "EUR 1,234.50"}, gold2)
    norm = lesson.improvement_table(base2, tagged2)
    assert norm["counterparty"].baseline_f1 == 1.0 and norm["total_amount"].tagger_f1 == 1.0, (
        "the default mode is 'normalised', under which 'Nordwind Logistik' matches "
        "'Nordwind Logistik GmbH' and 'EUR 1,234.50' matches '1,234.50'")
    exact = lesson.improvement_table(base2, tagged2, mode="exact")
    assert list(exact) == list(FIELDS), "the mode argument must not change the set of fields"
    assert exact["counterparty"].baseline_f1 == 0.0, (
        "`mode` is not reaching score_field for the BASELINE records: under 'exact' "
        "'Nordwind Logistik' does not match 'Nordwind Logistik GmbH', so the baseline F1 must "
        f"be 0.0, not {exact['counterparty'].baseline_f1}")
    assert exact["total_amount"].tagger_f1 == 0.0, (
        "`mode` is not reaching score_field for the TAGGER records: under 'exact' "
        "'EUR 1,234.50' does not match '1,234.50', so the tagger F1 must be 0.0, not "
        f"{exact['total_amount'].tagger_f1}")


def test_fields_improved_orders_and_thresholds():
    Imp = lesson.Improvement
    table = {
        "x": Imp("x", "id", 0.10, 0.15, 0.05),     # worst baseline, small gain
        "y": Imp("y", "id", 0.80, 1.00, 0.20),     # good baseline, bigger gain
        "z": Imp("z", "id", 0.50, 0.50, 0.00),     # unchanged
        "w": Imp("w", "id", 0.90, 0.40, -0.50),    # regression
    }
    got = lesson.fields_improved(table)
    assert got == ["x", "y"], (
        f"fields_improved gave {got}, expected ['x', 'y'] — sort ASCENDING on baseline F1 so "
        "the weakest field the tagger rescued is named first. Sorting by delta descending "
        "gives ['y', 'x'] and buries the field that was worst to begin with.")
    assert lesson.fields_improved(table, min_delta=0.10) == ["y"], \
        "min_delta filters on the delta: 0.05 is below 0.10, 0.20 is above it"
    assert lesson.fields_improved(table, min_delta=0.05) == ["x", "y"], (
        "the threshold is inclusive — a field whose delta is exactly min_delta counts. "
        "A strict `>` silently drops the boundary case, which is the one a caller passing a "
        "hand-picked threshold is aiming at.")
    assert lesson.fields_improved(table, min_delta=9.0) == [], "nothing clears 9.0"
    assert lesson.fields_improved(table, min_delta=0.0) == ["x", "z", "y"], \
        "at a threshold of 0.0 the unchanged field qualifies too, and sorts by baseline F1"
    tie = {"b": Imp("b", "id", 0.5, 0.9, 0.4), "a": Imp("a", "id", 0.5, 0.7, 0.2)}
    assert lesson.fields_improved(tie) == ["a", "b"], \
        "equal baselines break on the field NAME, so the same table always gives one order"


# --- the corpus-level case -------------------------------------------------------------

def test_tagger_beats_baseline_on_held_out():
    if getattr(lesson, "TAGGER_RECORDS", None) is None:
        raise NotImplementedError("the tagger has not trained yet")
    base = lesson.macro_f1(lesson.BASELINE_RECORDS, "normalised")
    tagged = lesson.macro_f1(lesson.TAGGER_RECORDS, "normalised")
    assert tagged > 0.85, (
        f"held-out macro F1 is {tagged:.4f}. A tagger with a working feature template, decoder "
        "and update clears 0.85 on this corpus comfortably; a low score here with the unit "
        "cases passing usually means the feature template ignores its context.")
    assert tagged > base + 0.03, (
        f"the tagger scores {tagged:.4f} against the rule baseline's {base:.4f}. It is "
        "supposed to win, and by a margin — check the sign of your perceptron update.")
    table = lesson.improvement_table(lesson.BASELINE_RECORDS, lesson.TAGGER_RECORDS)
    assert len(lesson.fields_improved(table)) >= 3, (
        "at least three of the six fields must improve; the rule baseline has no label phrase "
        "for several of them and no anchor at all for the amount")


RUBRIC = [
    (test_word_shape_collapses_runs, 5,
     "map each character to X/x/d or itself, then collapse runs of the same result"),
    (test_word_shape_edges, 3,
     "empty in empty out; punctuation runs collapse too; an unclassed character maps to itself"),
    (test_token_features_core, 5,
     "features 1-5 are bias, the lower-cased word, its shape, and its first and last three "
     "characters — twelve features in total, all strings"),
    (test_token_features_context, 5,
     "features 6-12 are the neighbouring words and shapes (with the literal <BOS>/<EOS> at the "
     "edges), the line head, the capped line position, and the head|shape conjunction"),
    (test_token_features_is_position_sensitive, 3,
     "the position feature reads tok.pos, the position within the LINE, capped at 4 — not the "
     "index into the token list"),
    (test_decode_spans_joins_a_span, 5,
     "walk the tags, buffer the span's token texts, join with single spaces; every field in "
     "SCHEMA gets a key"),
    (test_decode_spans_opens_on_a_stray_i, 5,
     "I- with no open span of that field OPENS one; I- for a different field closes the open "
     "span and opens a new one"),
    (test_decode_spans_absent_and_first_span_wins, 3,
     "keep the first span per field, close the open span on `O`, return '' for a field with "
     "none, and survive an empty document"),
    (test_sequence_scores_sums_active_features, 5,
     "out[i, t] = sum of model.w[t, f] over token i's active feature ids f"),
    (test_sequence_scores_empty_and_shape, 3,
     "shape is (n_tokens, n_tags); a token with no known features is a row of zeros, not a "
     "crash"),
    (test_viterbi_matches_brute_force, 6,
     "dp[i, t] = max over a of (dp[i-1, a] + transitions[a, t]) then add emissions[i, t]; keep "
     "the argmax over a (axis 0) as a backpointer and walk it back from the best final tag"),
    (test_viterbi_uses_transitions, 6,
     "the transition term is what makes the best whole-sequence path differ from the per-token "
     "argmax; if your answer never differs from greedy, transitions are not being read"),
    (test_viterbi_start_row_and_edges, 4,
     "dp[0] = transitions[n_tags] + emissions[0]; an empty sequence returns []"),
    (test_perceptron_update_noop_when_correct, 4,
     "compare the two sequences first and return 0 without touching either weight array"),
    (test_perceptron_update_emission_deltas, 6,
     "only at positions where gold and pred differ: +1 on the gold tag's row, -1 on the "
     "predicted tag's row, at that token's feature ids"),
    (test_perceptron_update_transition_deltas, 5,
     "run the transition pass over EVERY position, comparing the gold bigram with the "
     "predicted one, with the START row (index model.w.shape[0]) as position 0's predecessor"),
    (test_perceptron_update_returns_mismatch_count, 3,
     "return the number of positions where gold and pred differ"),
    (test_improvement_table_covers_every_field, 5,
     "score BOTH sides with score_field under the caller's `mode`, one entry per SCHEMA field "
     "in SCHEMA order, carrying the field type, with delta = tagger - baseline and its sign "
     "intact"),
    (test_fields_improved_orders_and_thresholds, 4,
     "filter on delta >= min_delta, then sort ascending on baseline F1 with the field name as "
     "the tie-break"),
    (test_tagger_beats_baseline_on_held_out, 5,
     "train on the training split only, decode the held-out split with Viterbi, read the spans "
     "and hand them to module 1's harness unchanged"),
]
