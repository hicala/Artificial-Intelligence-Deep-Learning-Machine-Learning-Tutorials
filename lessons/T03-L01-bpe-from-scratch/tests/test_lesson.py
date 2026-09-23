"""Autograder rubric for T03-L01. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — a student whose `merge` works scores for it while `train_bpe`
is still a TODO.

Three rules matter in this lesson and are deliberate:

* The cases below are NOT the notebook's public `_check_*` cases. Passing what you can see is
  not the same as being right, so every test here uses different inputs.
* `train_bpe` and `encode` are graded against an INDEPENDENT reference implementation
  written at the foot of this file, never against the student's own other functions. A
  student whose `get_stats` is subtly wrong should lose points on `get_stats`, not have that
  error silently define what "correct" means for `train_bpe`. `token_tax` is the exception,
  and deliberately so: its job is to assemble rows from whatever `encode` returns, so its
  `tokens` column is checked against the student's own `encode` — grading it against the
  reference would charge one broken `encode` twice. Two order-independent anchors keep that
  from being circular: an untrained script must sit exactly on the byte floor, and a trained
  one must sit strictly below it.
* The tie-break is graded by permuting the insertion order of one tied set. Any single
  ordering can be satisfied by luck of the dict; only invariance across permutations
  distinguishes a real comparison from "the first maximum I met".
"""
import importlib.util
import os
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

BOUNDARY = 256
FIRST_MERGE_ID = 257
MIN_COUNT = 2
_WHITESPACE = frozenset(b" \t\n\r\f\v")

# A hidden corpus, unrelated to the notebook's: short, repetitive, and multi-script, so a
# trainer that only ever saw ASCII in testing cannot coast.
_HIDDEN_CORPUS = (
    "banana bandana banner banana bandana\n"
    "the theme of the theatre is the theme of the theatre\n"
    "мама мыла раму мама мыла раму\n"
) * 6
_HIDDEN_VOCAB = 320


# --- independent reference implementation (never the student's code) --------------------

def _ref_pretokenise(blob: bytes) -> list:
    seq, in_word = [], False
    for byte in blob:
        if byte in _WHITESPACE:
            if in_word:
                seq.append(BOUNDARY)
                in_word = False
        else:
            seq.append(byte)
            in_word = True
    if in_word:
        seq.append(BOUNDARY)
    return seq


def _ref_stats(seq: list) -> dict:
    counts = {}
    for left, right in zip(seq, seq[1:]):
        if left != BOUNDARY and right != BOUNDARY:
            counts[(left, right)] = counts.get((left, right), 0) + 1
    return counts


def _ref_best(stats: dict):
    if not stats:
        return None
    top = max(stats.values())
    winner = min(pair for pair, n in stats.items() if n == top)
    return (winner[0], winner[1], top)


def _ref_merge(seq: list, pair: tuple, new_id: int) -> list:
    a, b = pair
    out, i, n = [], 0, len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _ref_train(text: str, vocab_size: int):
    """Returns (merges dict, order list). Raises ValueError below FIRST_MERGE_ID."""
    if vocab_size < FIRST_MERGE_ID:
        raise ValueError("vocab_size too small")
    seq = _ref_pretokenise(text.encode("utf-8"))
    merges, order = {}, []
    for k in range(vocab_size - FIRST_MERGE_ID):
        win = _ref_best(_ref_stats(seq))
        if win is None or win[2] < MIN_COUNT:
            break
        a, b, count = win
        new_id = FIRST_MERGE_ID + k
        merges[(a, b)] = new_id
        order.append((a, b, new_id, count))
        seq = _ref_merge(seq, (a, b), new_id)
    return merges, order


def _ref_encode(text: str, merges: dict) -> list:
    def one_word(word):
        ids = list(word)
        for pair, new_id in merges.items():
            if len(ids) < 2:
                break
            ids = _ref_merge(ids, pair, new_id)
        return ids

    out, word = [], []
    for byte in text.encode("utf-8"):
        if byte in _WHITESPACE:
            if word:
                out.extend(one_word(word))
                word = []
            out.append(byte)
        else:
            word.append(byte)
    if word:
        out.extend(one_word(word))
    return out


def _student_model(text: str, vocab_size: int):
    return lesson.train_bpe(text, vocab_size)


# --- exercise 1: get_stats --------------------------------------------------------------

def test_get_stats_counts_and_overlap():
    # A run of four: three overlapping pairs, not two and not one.
    assert lesson.get_stats([7, 7, 7, 7]) == {(7, 7): 3}
    # Mixed counts in one sequence, including a pair that repeats non-adjacently.
    assert lesson.get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    # Ids above 255 are ordinary symbols.
    assert lesson.get_stats([300, 301, 300, 301]) == {(300, 301): 2, (301, 300): 1}
    # Real text, against the independent reference.
    seq = _ref_pretokenise(_HIDDEN_CORPUS.encode("utf-8"))
    assert lesson.get_stats(seq) == _ref_stats(seq)


def test_get_stats_boundary_and_edges():
    # Boundaries at both ends and doubled in the middle: no pair may touch one.
    assert lesson.get_stats(
        [BOUNDARY, 4, 5, BOUNDARY, BOUNDARY, 5, 4, BOUNDARY]) == {(4, 5): 1, (5, 4): 1}
    assert lesson.get_stats([BOUNDARY]) == {}
    assert lesson.get_stats([BOUNDARY, BOUNDARY]) == {}
    assert lesson.get_stats([]) == {}
    assert lesson.get_stats([42]) == {}
    # A single word with no boundary at all still counts normally.
    assert lesson.get_stats([1, 1]) == {(1, 1): 1}


# --- exercise 2: merge ------------------------------------------------------------------

def test_merge_overlap_and_tail():
    # An odd run, a match ending exactly at the last position, and one at the very start.
    assert lesson.merge([7, 7, 7, 7, 7], (7, 7), 400) == [400, 400, 7]
    assert lesson.merge([1, 9, 9], (9, 9), 400) == [1, 400]
    assert lesson.merge([9, 9, 1], (9, 9), 400) == [400, 1]
    assert lesson.merge([9, 9], (9, 9), 400) == [400]
    # Ids above 255 must survive: symbols are ints, not bytes.
    assert lesson.merge([300, 301, 300, 301], (300, 301), 500) == [500, 500]


def test_merge_leaves_others_alone():
    assert lesson.merge([1, 2, 3], (2, 1), 400) == [1, 2, 3]
    assert lesson.merge([], (1, 2), 400) == []
    assert lesson.merge([5], (5, 5), 400) == [5]
    # The reversed pair must not match, and a partial prefix must not either.
    assert lesson.merge([1, 2, 1], (2, 2), 400) == [1, 2, 1]
    # merge must BUILD a new list, not edit the one it was handed. An in-place merge returns
    # the right answer to its own caller and is invisible to every value-equality check above,
    # but train_bpe, encode and the section-13 comparison all pass the same list to merge
    # twice and expect the original back the second time.
    given = [7, 7, 7, 7, 1, 2, 7, 7]
    snapshot = list(given)
    first = lesson.merge(given, (7, 7), 400)
    assert given == snapshot, (
        f"merge modified its argument: the caller's list came back as {given}, not "
        f"{snapshot} — build and return a NEW list instead of editing `seq` in place"
    )
    assert lesson.merge(given, (7, 7), 400) == first, (
        "merging the same list twice must give the same answer twice"
    )
    # Against the reference on real text, using the reference's own winning pair.
    seq = _ref_pretokenise(_HIDDEN_CORPUS.encode("utf-8"))
    a, b, _ = _ref_best(_ref_stats(seq))
    seq_snapshot = list(seq)
    assert lesson.merge(seq, (a, b), 998) == _ref_merge(seq_snapshot, (a, b), 998)
    assert seq == seq_snapshot, "merge modified the sequence it was given on real text too"


# --- exercise 3: best_pair --------------------------------------------------------------

def test_best_pair_tie_break():
    # A three-way tie offered in an order that rewards keeping the first one seen.
    assert lesson.best_pair({(300, 1): 6, (12, 99): 6, (12, 5): 6}) == (12, 5, 6)
    # Tied on the left symbol, so the right symbol decides — both ways round, because any one
    # ordering can be satisfied by luck rather than by a real comparison.
    assert lesson.best_pair({(8, 40): 2, (8, 9): 2}) == (8, 9, 2)
    assert lesson.best_pair({(8, 9): 2, (8, 40): 2}) == (8, 9, 2)
    # The property a lucky iteration order cannot fake: the winner may not depend on the order
    # the counts arrived in.
    tied = [(5, 6), (19, 20), (44, 45), (91, 92), (160, 161), (262, 263), (303, 304)]
    for order in (tied, list(reversed(tied)), tied[2:] + tied[:2], tied[1::2] + tied[0::2]):
        assert lesson.best_pair({pair: 11 for pair in order}) == (5, 6, 11)
    # A clear winner must not be displaced by a later equal-count pair.
    assert lesson.best_pair({(1, 1): 9, (2, 2): 9, (3, 3): 4}) == (1, 1, 9)


def test_best_pair_empty_and_singleton():
    assert lesson.best_pair({}) is None
    assert lesson.best_pair({(262, 263): 1}) == (262, 263, 1)
    assert lesson.best_pair({(0, 0): 1}) == (0, 0, 1)


# --- exercise 4: train_bpe --------------------------------------------------------------

def test_train_bpe_matches_reference_merges():
    ref_merges, ref_order = _ref_train(_HIDDEN_CORPUS, _HIDDEN_VOCAB)
    model = _student_model(_HIDDEN_CORPUS, _HIDDEN_VOCAB)
    assert list(model.order) == ref_order, (
        "the merge list diverges from the reference trainer; the first differing entry is "
        "where the bug is"
    )
    assert dict(model.merges) == ref_merges
    # vocab must expand ids back to the bytes they stand for.
    for (a, b), new_id in ref_merges.items():
        assert model.vocab[new_id] == model.vocab[a] + model.vocab[b]
    assert model.vocab[65] == b"A"


def test_train_bpe_vocab_size_and_stop_rule():
    # The cap bites: a budget well below what the corpus could supply.
    assert len(_student_model(_HIDDEN_CORPUS, 270).order) == 270 - FIRST_MERGE_ID
    # Exactly FIRST_MERGE_ID is legal and buys nothing.
    assert _student_model(_HIDDEN_CORPUS, FIRST_MERGE_ID).order == []
    # No pair occurs twice, so no merge is worth making however many are asked for.
    assert _student_model("uvwxyz", 400).order == []
    # The MIN_COUNT boundary itself, which every case above steps over. A pair seen EXACTLY
    # MIN_COUNT times IS worth merging: the rule is "stop when count < MIN_COUNT", and the
    # off-by-one "<= MIN_COUNT" silently throws away the last useful merge of every corpus.
    assert _student_model("zz zz", 258).order == [(122, 122, 257, 2)], (
        "(z, z) occurs exactly MIN_COUNT times here, so it must be merged — the stop rule is "
        "strictly BELOW MIN_COUNT, not at it"
    )
    # ...and one occurrence short of it is not.
    assert _student_model("zq zz", 300).order == [], (
        "every pair here occurs once, which is below MIN_COUNT, so no merge may be recorded"
    )
    # merges and order must describe the same merge list. A trainer that leaks state between
    # calls — a mutable default argument, a module-level dict — returns a correct `order` and
    # a merges dict that has been growing since the first call, and nothing else here sees it.
    leaky = _student_model(_HIDDEN_CORPUS, 270)
    assert len(leaky.merges) == len(leaky.order) == 270 - FIRST_MERGE_ID, (
        f"merges has {len(leaky.merges)} entries and order has {len(leaky.order)}; both must "
        f"be {270 - FIRST_MERGE_ID} — build them fresh inside train_bpe on every call"
    )
    assert [(a, b) for a, b, _, _ in leaky.order] == list(leaky.merges), (
        "merges and order must hold the same pairs, in the same training order"
    )
    # Two merges available, both pinned exactly, ids included — a train_bpe that returns an
    # empty list, or numbers ids from anywhere but FIRST_MERGE_ID, cannot collect these points.
    assert [row[:3] for row in _student_model("xyxyxy xyxyxy", 259).order] == [
        (120, 121, 257), (257, 257, 258)]
    for bad in (256, 0, -1):
        try:
            _student_model("anything", bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"vocab_size={bad} must raise ValueError")


# --- exercise 5: encode / decode --------------------------------------------------------

def test_encode_decode_roundtrip():
    model = _student_model(_HIDDEN_CORPUS, _HIDDEN_VOCAB)
    # Every sample the notebook ships, plus cases it does not: whitespace at both ends, runs
    # of spaces, newlines and tabs, an emoji outside the BMP, and the empty string.
    cases = list(lesson.SAMPLES.values()) + [
        "", " ", "   leading and trailing   ", "tabs\tand\nnewlines\n",
        "banana bandana", "мама мыла раму", "🙂 mixed 漢字 with ASCII",
        "punctuation!?;:'\"()[]{}", "1234567890",
    ]
    for text in cases:
        ids = lesson.encode(text, model)
        assert lesson.decode(ids, model) == text, f"round trip failed for {text!r}"
        assert all(isinstance(i, int) for i in ids), "encode must return ints"
        assert BOUNDARY not in ids, f"BOUNDARY leaked into encode output for {text!r}"
    # Every case above hands decode a complete, valid UTF-8 sequence, so a strict decode
    # passes them all. The case that separates them is a token list that ends mid-character —
    # exactly what a streaming decoder or a truncated generation hands you. It must come back
    # as U+FFFD, neither raising nor being swallowed into an empty string.
    lead = lesson.decode([0xD0], model)     # the lead byte of a two-byte Cyrillic character
    assert lead == "�", (
        f"decode([0xD0]) gave {lead!r}, expected '\\ufffd' — a lone lead byte is not valid "
        "UTF-8, so decode must use errors='replace'. Letting it raise, or catching the error "
        "and returning '', both lose the bytes that did decode."
    )
    partial = lesson.decode(list("ab".encode("utf-8")) + [0xE4, 0xB8], model)
    assert partial == "ab�", (
        f"decode on 'ab' plus a truncated three-byte character gave {partial!r}, expected "
        "'ab\\ufffd' — the text that did decode must survive, with the broken tail replaced."
    )


def test_encode_applies_merges_and_matches_reference():
    ref_merges, _ = _ref_train(_HIDDEN_CORPUS, _HIDDEN_VOCAB)
    model = _student_model(_HIDDEN_CORPUS, _HIDDEN_VOCAB)
    for text in ("banana bandana banner", "the theme of the theatre", "мама мыла раму"):
        assert lesson.encode(text, model) == _ref_encode(text, ref_merges), (
            f"encode disagrees with the reference on {text!r} — are merges applied in "
            "training order, and whitespace emitted as its own byte?"
        )
        # It must actually compress: a trained word is shorter than its byte spelling.
        assert len(lesson.encode(text, model)) < len(text.encode("utf-8"))
    # A model with no merges at all must fall back to exactly the bytes.
    empty = _student_model(_HIDDEN_CORPUS, FIRST_MERGE_ID)
    assert lesson.encode("banana", empty) == list("banana".encode("utf-8"))


# --- exercise 6: token_tax --------------------------------------------------------------

def test_token_tax_rows():
    model = _student_model(_HIDDEN_CORPUS, _HIDDEN_VOCAB)
    # The keys are deliberately NOT in alphabetical order: "latin", "han", "cyrillic" sorts to
    # "cyrillic", "han", "latin". A token_tax that iterates sorted(samples) instead of the
    # mapping it was given answers a different question, and alphabetical keys would hide it.
    samples = {"latin": "banana bandana", "han": "知识就是力量。", "cyrillic": "мама мыла раму"}
    rows = lesson.token_tax(model, samples)
    assert [r["script"] for r in rows] == ["latin", "han", "cyrillic"], (
        f"rows came back as {[r['script'] for r in rows]} — the order must follow the mapping "
        "you were given, not the sorted order of its keys"
    )
    assert len(rows) == 3
    # Exactly these keys: no fewer, and no extra working columns left behind.
    assert set(rows[0]) == {"script", "text", "chars", "utf8_bytes", "tokens",
                            "tokens_per_char", "bytes_per_char"}, (
        f"row keys are {sorted(rows[0])} — return exactly the seven keys the docstring lists"
    )
    for row, (name, text) in zip(rows, samples.items()):
        assert row["text"] == text
        assert row["chars"] == len(text)
        assert row["utf8_bytes"] == len(text.encode("utf-8"))
        assert row["tokens"] == len(lesson.encode(text, model))
        assert abs(row["tokens_per_char"] - row["tokens"] / row["chars"]) < 1e-9
        assert abs(row["bytes_per_char"] - row["utf8_bytes"] / row["chars"]) < 1e-9
    # The Han sample was never in the hidden corpus, so no merge can fire: it must sit exactly
    # on the byte floor. This is the measurement the lesson exists to make.
    assert rows[1]["tokens"] == rows[1]["utf8_bytes"], (
        "the Han sample is absent from the training corpus, so no merge can fire and its "
        "token count must equal its UTF-8 byte count exactly"
    )
    # And the trained Latin sample must be strictly below its own floor.
    assert rows[0]["tokens"] < rows[0]["utf8_bytes"], (
        "the Latin sample was trained on, so its merges must take it below the byte floor"
    )
    # An empty samples mapping is an empty table, not a crash.
    assert lesson.token_tax(model, {}) == []


RUBRIC = [
    (test_get_stats_counts_and_overlap, 3,
     "exercise 1: one scan, counting overlaps — a run of four holds three pairs"),
    (test_get_stats_boundary_and_edges, 3,
     "exercise 1: guard len(seq) < 2, and skip any pair touching BOUNDARY entirely"),
    (test_merge_overlap_and_tail, 3,
     "exercise 2: advance the read cursor by two after a match, and never read past the end"),
    (test_merge_leaves_others_alone, 2,
     "exercise 2: an absent pair must leave the sequence and its length untouched"),
    (test_best_pair_tie_break, 4,
     "exercise 3: ties go to the smaller (a, b) — compare the tuples, not iteration order"),
    (test_best_pair_empty_and_singleton, 2,
     "exercise 3: an empty dict returns None; a lone entry is its own winner"),
    (test_train_bpe_matches_reference_merges, 5,
     "exercise 4: number merges FIRST_MERGE_ID + k, and apply each one before counting again"),
    (test_train_bpe_vocab_size_and_stop_rule, 3,
     "exercise 4: vocab_size - 257 merges, stop below MIN_COUNT, ValueError under 257"),
    (test_encode_decode_roundtrip, 5,
     "exercise 5: emit whitespace bytes as themselves and never emit BOUNDARY, so decode "
     "can rebuild the original exactly"),
    (test_encode_applies_merges_and_matches_reference, 3,
     "exercise 5: apply every merge in training order to each word, reassigning the result"),
    (test_token_tax_rows, 4,
     "exercise 6: tokens_per_char divides by CHARACTERS, and rows follow the input order"),
]
