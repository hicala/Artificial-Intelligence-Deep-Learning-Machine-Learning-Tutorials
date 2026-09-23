"""Autograder rubric for P02-L02. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These rules matter in this particular lesson and are deliberate:

* Every case grades OBSERVABLE BEHAVIOUR, never one particular implementation. `xy_cut` may
  recurse or use an explicit stack; `order_within_block` may sort once or bucket first;
  nothing here inspects source.
* `test_find_gaps_needs_a_running_maximum` is what stops the most attractive wrong answer:
  comparing each box only with its immediate predecessor. Every other gap case passes with
  that bug, because none of them contains an overlapping or nested box — and a real text
  layer is full of both.
* `test_xy_cut_cuts_bands_before_columns` exists because the axis order is the whole
  algorithm. Cutting x first glues a full-width heading to one column and reads it in the
  middle of that column's text; on a single-column page nothing at all goes wrong, so the
  bug survives every simpler case.
* `test_order_within_block_ignores_input_order` grades determinism rather than arithmetic.
  A block is a bag of boxes; an implementation that leans on the order it was handed gives a
  different reading order depending on how the page happened to be stored, and the lesson's
  own corpus shuffles the tokens precisely so that this shows up.
* `test_header_footer_masks_digits` is the trap the exercise exists for. A rule keyed on raw
  text never sees a page number twice, so it leaves one token per page in the body stream.
* `test_boundary_f1_scores_both_sides_along_the_order` separates the metric that was asked
  for from the tempting one. Comparing the two partitions directly (are these the same
  blocks?) is a different, defensible measure, and it is not the one the docstring specifies:
  boundaries are read along ONE shared order, which is what makes a predicted segmentation
  comparable with a gold one it does not resemble.
* `test_kendall_tau_and_boundary_f1_disagree` is the lesson's thesis as a graded case: at
  `min_gap=8` the page is shredded and tau does not move. A rubric that only ever checked tau
  would accept a segmenter that emits one block per token.
* The two corpus-level cases are the only ones that touch the lesson's own 48 documents. They
  are deterministic (fixed seed, no clock, no I/O) and they are bands, not equalities, except
  where an exact reconstruction of the gold order is the stated contract.
"""
import importlib.util, os
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)


def _box(tid: int, x: float, y: float, w: float = 10.0, h: float = 10.0, text: str = "x",
         page: int = 0) -> dict:
    return {"tid": tid, "text": text, "x": float(x), "y": float(y),
            "w": float(w), "h": float(h), "page": page}


# --- exercise 1: find_gaps -------------------------------------------------------------

def test_find_gaps_basic():
    a, b = _box(1, 0, 0), _box(2, 60, 0)
    assert lesson.find_gaps([a, b], "x", 20.0) == [(10.0, 60.0)], \
        "one interior gap, reported as (low edge, high edge) of the empty interval"
    assert lesson.find_gaps([a, b], "x", 50.0) == [(10.0, 60.0)], \
        "a gap of exactly min_gap counts: the comparison is >=, not >"
    assert lesson.find_gaps([a, b], "x", 50.1) == [], "a gap narrower than min_gap is not a gap"
    c, d = _box(3, 0, 0), _box(4, 0, 90)
    assert lesson.find_gaps([c, d], "y", 20.0) == [(10.0, 90.0)], \
        "the same sweep has to work along y, using the top and bottom edges"


def test_find_gaps_needs_a_running_maximum():
    a, b = _box(1, 0, 0), _box(2, 60, 0)
    wide = _box(3, 0, 0, w=100.0)
    assert lesson.find_gaps([a, wide, b], "x", 20.0) == [], (
        "a box spanning the empty interval closes it — carry a running MAXIMUM of the high "
        "edge seen so far instead of comparing each box with the one before it")
    nested = [_box(1, 0, 0, w=100.0), _box(2, 20, 0, w=10.0), _box(3, 200, 0)]
    assert lesson.find_gaps(nested, "x", 20.0) == [(100.0, 200.0)], (
        "a box nested inside another must not close the real gap that follows, nor open a "
        "spurious one behind it")
    overlapping = [_box(1, 0, 0, w=30.0), _box(2, 20, 0, w=30.0), _box(3, 120, 0)]
    assert lesson.find_gaps(overlapping, "x", 20.0) == [(50.0, 120.0)], \
        "overlapping boxes are one run; the gap starts at the furthest edge either reaches"


def test_find_gaps_multiple_and_ordered():
    boxes = [_box(3, 100, 0), _box(1, 0, 0), _box(2, 50, 0)]
    assert lesson.find_gaps(boxes, "x", 20.0) == [(10.0, 50.0), (60.0, 100.0)], (
        "two gaps, in increasing order along the axis, whatever order the boxes arrived in — "
        "sort by the LOW edge before sweeping")


def test_find_gaps_degenerate_inputs():
    a = _box(1, 0, 0)
    assert lesson.find_gaps([a], "x", 5.0) == [], "one box encloses no interior gap"
    assert lesson.find_gaps([], "y", 5.0) == [], "no boxes, no gaps — not a crash"
    for bad in (0.0, -1.0):
        try:
            lesson.find_gaps([a, _box(2, 60, 0)], "x", bad)
            raise AssertionError(
                f"min_gap={bad} must raise ValueError: a non-positive threshold makes every "
                "pair of touching boxes a gap and cuts the page to shreds")
        except ValueError:
            pass


# --- exercise 2: xy_cut ----------------------------------------------------------------

def test_xy_cut_splits_bands_and_columns():
    top, bottom = _box(1, 0, 0), _box(2, 0, 100)
    assert lesson.xy_cut([top, bottom], 30.0) == [[1], [2]], \
        "a horizontal gap makes two bands, the upper one first"
    assert lesson.xy_cut([top, bottom], 200.0) == [[1, 2]], \
        "no gap wide enough in either direction: the region is one block"
    left, right = _box(3, 0, 0), _box(4, 100, 0)
    assert lesson.xy_cut([left, right], 30.0) == [[3], [4]], \
        "a vertical gap makes two columns, the left one first"
    assert lesson.xy_cut([], 30.0) == [], "no boxes, no blocks"
    assert lesson.xy_cut([_box(7, 5, 5)], 30.0) == [[7]], "one box is one block"


def test_xy_cut_cuts_bands_before_columns():
    head = _box(1, 0, 0, w=120.0)
    left, right = _box(2, 0, 100), _box(3, 100, 100)
    assert lesson.xy_cut([head, left, right], 30.0) == [[1], [2], [3]], (
        "try the HORIZONTAL cut first: a heading above two columns must be its own band")
    # The heading here sits over the RIGHT column only. Cut on y first and it is still its own
    # band, read before both columns. Cut on x first and it is swallowed by the right column,
    # so the page reads left column, heading, right column — a plausible-looking order that
    # puts the title of the page a third of the way into it.
    offset_head = _box(1, 100, 0, w=120.0)
    narrow_left = _box(2, 0, 100, w=40.0)
    narrow_right = _box(3, 100, 100, w=40.0)
    got = lesson.xy_cut([offset_head, narrow_left, narrow_right], 30.0)
    assert got == [[1], [2], [3]], (
        f"expected [[1], [2], [3]] and got {got}: look for a HORIZONTAL cut before a vertical "
        "one. Cutting on x first puts the heading inside the column it happens to overlap")


def test_xy_cut_recurses_into_the_regions_it_makes():
    # No horizontal gap crosses the whole page (the right column's rows fill the left
    # column's hole), so the first cut must be vertical — and the LEFT column must then be
    # cut horizontally, which only happens if the recursion is real.
    boxes = [_box(1, 0, 0), _box(2, 0, 100), _box(3, 200, 40), _box(4, 200, 60)]
    got = lesson.xy_cut(boxes, 35.0)
    assert got == [[1], [2], [3, 4]], (
        f"expected [[1], [2], [3, 4]] and got {got}: after the vertical cut, each column has "
        "to go back through the same procedure, not be emitted as a block")


def test_xy_cut_is_a_partition_of_the_page():
    doc = next(d for d in lesson.CORPUS if d["family"] == "two_column")
    body = [t for t in doc["pages"][0] if t["tid"] not in doc["furniture"]]
    want = sorted(t["tid"] for t in body)
    for min_gap in (8.0, 24.0, lesson.DEFAULT_MIN_GAP, 60.0):
        blocks = lesson.xy_cut(body, min_gap)
        got = sorted(tid for block in blocks for tid in block)
        assert got == want, (
            f"at min_gap={min_gap} the blocks hold {len(got)} token ids and the page has "
            f"{len(want)}: every token must land in exactly one block, none lost, none "
            "duplicated by a band boundary that both sides claim")
    two = lesson.xy_cut(body, lesson.DEFAULT_MIN_GAP)
    assert len(two) == 2, (
        f"a two-column page at the default gap is two blocks, you produced {len(two)}: one "
        "means the vertical cut never ran, many means you are cutting on line spacing")


# --- exercise 3: order_within_block -----------------------------------------------------

def test_order_within_block_lines_then_x():
    boxes = [_box(1, 50, 0), _box(2, 0, 0), _box(3, 0, 40), _box(4, 25, 40)]
    assert lesson.order_within_block(boxes) == [2, 1, 3, 4], \
        "left to right inside a line, then down to the next line"
    assert lesson.order_within_block([]) == [], "no boxes, no order — not a crash"
    assert lesson.order_within_block([_box(9, 3, 3)]) == [9], "one box is one line"


def test_order_within_block_line_tolerance():
    tol = lesson.LINE_TOL
    same = [_box(1, 50, 0), _box(2, 0, tol - 0.5)]
    assert lesson.order_within_block(same) == [2, 1], (
        f"centres less than LINE_TOL={tol} apart are ONE line, so they sort by x — a strict "
        "sort on y alone would put the right-hand token first")
    # Exactly LINE_TOL apart. The docstring says a new line starts when the centre is MORE
    # than LINE_TOL from the line's first box, so this pair is still one line. Same boundary
    # question as `>= min_gap` in find_gaps, and the same answer.
    exact = [_box(1, 50, 0), _box(2, 0, tol)]
    assert lesson.order_within_block(exact) == [2, 1], (
        f"centres EXACTLY LINE_TOL={tol} apart are still one line: a new line starts when the "
        "distance is MORE than the tolerance, so the comparison is <=, not <")
    apart = [_box(1, 50, 0), _box(2, 0, 4 * tol)]
    assert lesson.order_within_block(apart) == [1, 2], \
        "centres far apart are two lines, so the higher one comes first whatever its x"
    # Four boxes drifting downwards, each within LINE_TOL of the one before it but not of the
    # first: two lines, not one. This is why the comparison is against the line's FIRST box.
    drift = [_box(1, 30, 0), _box(2, 20, tol - 1), _box(3, 10, 2 * tol - 2),
             _box(4, 0, 3 * tol - 3)]
    assert lesson.order_within_block(drift) == [2, 1, 4, 3], (
        "comparing each box with the PREVIOUS one lets a slow vertical drift swallow several "
        "lines into one; anchor the comparison on the first box of the current line")


def test_order_within_block_ignores_input_order():
    boxes = [_box(1, 0, 0), _box(2, 30, 0), _box(3, 60, 0), _box(4, 0, 40), _box(5, 30, 40)]
    want = [1, 2, 3, 4, 5]
    for rotation in range(len(boxes)):
        shuffled = boxes[rotation:] + boxes[:rotation]
        got = lesson.order_within_block(shuffled)
        assert got == want, (
            f"handed the same block in a different order you returned {got}, not {want}: sort "
            "the boxes yourself, the order you were given carries no information")
    # Two boxes at the same x, on one line, whose CENTRES differ: a per-line sort on x alone
    # is stable, so it falls back to whatever order the boxes reached the line in — here the
    # lower token id arrives second. Only an explicit tie-break on token id gives [5, 8].
    tied = [_box(8, 10, 0), _box(5, 10, 2)]
    assert lesson.order_within_block(tied) == [5, 8], (
        "two boxes at the same x must break the tie on TOKEN ID, not on the y they happen to "
        "sit at within the line's tolerance: sort each line by (x, tid)")
    assert lesson.order_within_block([_box(8, 10, 0), _box(5, 10, 0)]) == [5, 8], \
        "the same tie-break with identical boxes, so the result never depends on input order"


# --- exercise 4: detect_header_footer ---------------------------------------------------

def test_header_footer_masks_digits():
    pages = [[_box(1, 0, 760, text="Page 1"), _box(2, 0, 400, text="body")],
             [_box(3, 0, 760, text="Page 2"), _box(4, 0, 400, text="body")],
             [_box(5, 0, 760, text="Page 3"), _box(6, 0, 400, text="body")]]
    assert sorted(lesson.detect_header_footer(pages)) == [1, 3, 5], (
        "'Page 1', 'Page 2' and 'Page 3' are the same furniture once the digits are masked. "
        "Keyed on raw text each is seen on one page only, falls below the threshold, and "
        "every page number lands in the body stream")


def test_header_footer_needs_the_band_and_the_repetition():
    mid = [[_box(1, 0, 400, text="same")], [_box(2, 0, 400, text="same")]]
    assert lesson.detect_header_footer(mid) == set(), \
        "outside the top and bottom HF_BAND, repetition proves nothing"
    rare = [[_box(1, 0, 24, text="draft")], [_box(2, 0, 24, text="final")],
            [_box(3, 0, 24, text="draft")], [_box(4, 0, 24, text="spare")]]
    assert lesson.detect_header_footer(rare) == set(), (
        "'draft' appears on 2 of 4 pages, below HF_MIN_PAGES_FRACTION"
        f"={lesson.HF_MIN_PAGES_FRACTION}, so it is a banner and not furniture")
    moved = [[_box(1, 0, 24, text="acme")], [_box(2, 0, 24, text="acme")],
             [_box(3, 0, 770, text="acme")]]
    assert sorted(lesson.detect_header_footer(moved)) == [1, 2], (
        "the key carries the row the token sits on: the same word in the footer is not the "
        "same furniture as the one in the header, and on its own it is on 1 page in 3")
    # A centre exactly on the band edge (75 + h/2 = 80.0 = HF_BAND * PAGE_H) is body, however
    # often it repeats: the band is the top tenth, not the top tenth and its boundary.
    edge = [[_box(1, 0, 75, text="edge")], [_box(2, 0, 75, text="edge")],
            [_box(3, 0, 75, text="edge")]]
    assert lesson.detect_header_footer(edge) == set(), (
        f"a vertical centre of exactly HF_BAND * PAGE_H = {lesson.HF_BAND * lesson.PAGE_H} is "
        "outside the band: the test is y < top or y > bottom, strictly")
    # Exactly at the threshold: 3 pages out of 5 is 0.6 of them, and the rule says AT LEAST
    # HF_MIN_PAGES_FRACTION. A five-page document whose header stops after page 3 is the
    # ordinary case that separates >= from >.
    exact = [[_box(1, 0, 24, text="acme")], [_box(2, 0, 24, text="acme")],
             [_box(3, 0, 24, text="acme")], [_box(4, 0, 24, text="other")],
             [_box(5, 0, 24, text="other")]]
    assert sorted(lesson.detect_header_footer(exact)) == [1, 2, 3], (
        f"'acme' is on 3 of 5 pages, which IS HF_MIN_PAGES_FRACTION"
        f"={lesson.HF_MIN_PAGES_FRACTION} of them: the comparison is >=, not >. 'other' is on "
        "2 of 5 and must stay out")
    assert lesson.detect_header_footer([]) == set(), "no pages, no furniture"


def test_header_footer_on_the_corpus():
    got = lesson.detect_header_footer(lesson.ALL_PAGES)
    gold = {tid for doc in lesson.CORPUS for tid in doc["furniture"]}
    assert got == gold, (
        f"on the lesson's own corpus you flagged {len(got)} tokens where the gold answer is "
        f"{len(gold)}; {len(gold - got)} were missed and {len(got - gold)} body tokens were "
        "flagged by mistake")


# --- exercise 5: kendall_tau -------------------------------------------------------------

def test_kendall_tau_extremes():
    assert lesson.kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0, "identical orderings score +1.0"
    assert lesson.kendall_tau([3, 2, 1], [1, 2, 3]) == -1.0, "a reversal scores exactly -1.0"
    assert lesson.kendall_tau([], []) == 1.0, "an empty ordering is perfect, not an error"
    assert lesson.kendall_tau([4], [4]) == 1.0, "a single item is trivially in order"


def test_kendall_tau_known_values():
    assert abs(lesson.kendall_tau([2, 1, 3], [1, 2, 3]) - 1 / 3) < 1e-9, (
        "one discordant pair out of three: (2 - 1) / 3. Getting 2/3 here means you divided "
        "by the number of items rather than the number of PAIRS")
    assert abs(lesson.kendall_tau([1, 3, 2, 4], [1, 2, 3, 4]) - 4 / 6) < 1e-9, \
        "five concordant and one discordant pair out of six"
    assert abs(lesson.kendall_tau([4, 3, 2, 1], [1, 2, 3, 4]) + 1.0) < 1e-9, "a longer reversal"
    a, b = [3, 1, 4, 2], [1, 2, 3, 4]
    assert abs(lesson.kendall_tau(a, b) - lesson.kendall_tau(b, a)) < 1e-9, \
        "tau is symmetric: swapping the two arguments cannot change the answer"


def test_kendall_tau_validates_its_inputs():
    # The third case has matching LENGTHS and matching SETS and is still not a pair of
    # permutations, so a check built only from len() and set() lets it through and scores a
    # comparison in which one token was counted twice and another never.
    for pred, gold, why in (([1, 2], [1, 3], "different id sets"),
                            ([1, 2], [1, 2, 3], "different lengths"),
                            ([1, 1, 2], [1, 2, 2], "a repeated id on both sides")):
        try:
            lesson.kendall_tau(pred, gold)
            raise AssertionError(f"{why} must raise ValueError rather than return a number")
        except ValueError:
            pass


# --- exercise 6: boundary_f1 -------------------------------------------------------------

def test_boundary_f1_perfect_and_oversegmented():
    order = [1, 2, 3, 4]
    assert lesson.boundary_f1([[1, 2], [3, 4]], [[1, 2], [3, 4]], order) == (1.0, 1.0, 1.0), \
        "the same segmentation on both sides scores 1.0 across the board"
    p, r, f = lesson.boundary_f1([[1], [2], [3], [4]], [[1, 2], [3, 4]], order)
    assert abs(p - 1 / 3) < 1e-9 and r == 1.0 and abs(f - 0.5) < 1e-9, (
        f"one block per token proposes 3 boundaries and the gold has 1: precision 1/3, "
        f"recall 1.0, F1 0.5 — you returned ({p:.3f}, {r:.3f}, {f:.3f})")
    p, r, f = lesson.boundary_f1([[1, 2, 3, 4]], [[1, 2], [3, 4]], order)
    assert (p, r, f) == (0.0, 0.0, 0.0), \
        "proposing no boundary where the gold has one scores zero, it does not divide by zero"


def test_boundary_f1_scores_both_sides_along_the_order():
    # The two partitions do not resemble each other; the boundaries are still read along one
    # shared order, which is the only way the two become comparable.
    p, r, f = lesson.boundary_f1([[1, 2], [3, 4]], [[1, 3], [2, 4]], [1, 2, 3, 4])
    assert abs(p - 1.0) < 1e-9 and abs(r - 1 / 3) < 1e-9, (
        f"walking [1, 2, 3, 4] the gold partition changes block at every step (3 boundaries) "
        f"and yours changes once, correctly: precision 1.0, recall 1/3 — you returned "
        f"({p:.3f}, {r:.3f})")
    # Same partitions, different reference order: the answer must move.
    q = lesson.boundary_f1([[1, 2], [3, 4]], [[1, 3], [2, 4]], [1, 3, 2, 4])
    assert q != (p, r, f), \
        "the reference order is an argument, not decoration: changing it changes the score"
    # A boundary has to be in the RIGHT PLACE, not merely be one of the right number of them.
    # Walking [1, 2, 3, 4]: the prediction breaks after token 2, the gold breaks after 1 and
    # after 3, and not one predicted position is a gold position. Counting boundaries instead
    # of intersecting their positions scores this 1.0 / 0.5 and calls it half right.
    p2, r2, f2 = lesson.boundary_f1([[1, 2], [3, 4]], [[1], [2, 3], [4]], [1, 2, 3, 4])
    assert (p2, r2, f2) == (0.0, 0.0, 0.0), (
        f"you returned ({p2:.3f}, {r2:.3f}, {f2:.3f}) where every score is 0.0: the one "
        "boundary you proposed (after token 2) matches neither gold boundary (after 1, after "
        "3). Intersect the two SETS OF POSITIONS; do not compare how many each side has")


def test_boundary_f1_edges():
    assert lesson.boundary_f1([[1, 2]], [[1, 2]], [1, 2]) == (1.0, 1.0, 1.0), (
        "no boundary on either side means one block that both sides agree on: that is a "
        "perfect segmentation, so return 1.0, not 0.0 from a guarded division")
    try:
        lesson.boundary_f1([[1, 2]], [[1]], [1, 2])
        raise AssertionError("a token in `order` that no gold block holds must raise ValueError")
    except ValueError:
        pass


# --- the corpus: the pipeline end to end ------------------------------------------------

def test_pipeline_recovers_the_gold_order_at_the_default_gap():
    wrong = []
    for doc in lesson.CORPUS:
        got, _ = lesson.pipeline_order(doc, lesson.DEFAULT_MIN_GAP)
        if got != doc["gold_order"]:
            wrong.append(doc["doc_id"] + f" ({doc['family']})")
    assert not wrong, (
        f"{len(wrong)} of {len(lesson.CORPUS)} documents came back in the wrong order at "
        f"min_gap={lesson.DEFAULT_MIN_GAP}, starting with {wrong[0]}. At this threshold the "
        "gold order is exactly recoverable from the geometry; print your stream and the gold "
        "stream side by side and find the first token that differs")


def test_kendall_tau_and_boundary_f1_disagree():
    """The lesson's thesis, graded: a shredded page keeps a perfect token order."""
    def measure(min_gap):
        taus, bfs = [], []
        for doc in lesson.CORPUS:
            order, per_page = lesson.pipeline_order(doc, min_gap)
            taus.append(lesson.kendall_tau(order, doc["gold_order"]))
            for page_index, blocks in enumerate(per_page):
                ids = {t["tid"] for t in lesson.body_pages(doc)[page_index]}
                page_order = [tid for tid in doc["gold_order"] if tid in ids]
                bfs.append(lesson.boundary_f1(blocks, doc["gold_blocks"][page_index],
                                              page_order)[2])
        return sum(taus) / len(taus), sum(bfs) / len(bfs)

    tau_tight, bf_tight = measure(8.0)
    tau_good, bf_good = measure(lesson.DEFAULT_MIN_GAP)
    tau_loose, bf_loose = measure(60.0)
    assert tau_tight > 0.99, (
        f"at min_gap=8 the token ORDER is still right (tau {tau_tight:.3f}): the page is "
        "over-segmented, and tau cannot see blocks at all")
    assert bf_tight < 0.5 < bf_good, (
        f"boundary F1 must collapse where tau does not: got {bf_tight:.3f} at min_gap=8 "
        f"against {bf_good:.3f} at the default")
    assert tau_loose < tau_good, (
        f"at min_gap=60 the two-column pages are read straight across, so tau must fall: got "
        f"{tau_loose:.3f} against {tau_good:.3f}")


def test_reading_order_moves_downstream_field_f1():
    gold = lesson.macro_f1(lesson.records_for(lambda d: d["gold_order"]))
    mine = lesson.macro_f1(lesson.records_for(
        lambda d: lesson.pipeline_order(d, lesson.DEFAULT_MIN_GAP)[0]))
    raster = lesson.macro_f1(lesson.records_for(lesson.raster_order))
    assert gold > 0.99, f"the gold order is the ceiling and should score ~1.0, got {gold:.3f}"
    assert mine > 0.99, (
        f"your pipeline scores {mine:.3f} where the gold order scores {gold:.3f}: at the "
        "default gap the geometry is recoverable, so any gap here is a bug in the pipeline")
    assert raster < gold - 0.1, (
        f"the naive raster baseline must cost real field F1 on this corpus: got "
        f"{raster:.3f} against a ceiling of {gold:.3f}")


RUBRIC = [
    (test_find_gaps_basic, 4,
     "sweep the boxes by low edge and open a gap when the next one starts at least min_gap "
     "past the furthest edge reached so far; >= min_gap, not >"),
    (test_find_gaps_needs_a_running_maximum, 5,
     "carry a running MAXIMUM of the high edge; comparing each box with its predecessor lets "
     "a wide or nested box open gaps that are not there"),
    (test_find_gaps_multiple_and_ordered, 3,
     "sort by the low edge first, then report every gap, in increasing order along the axis"),
    (test_find_gaps_degenerate_inputs, 2,
     "fewer than two boxes: []. A min_gap that is not positive: ValueError"),
    (test_xy_cut_splits_bands_and_columns, 4,
     "y gaps make bands top to bottom, x gaps make columns left to right, no gap at all makes "
     "one block"),
    (test_xy_cut_cuts_bands_before_columns, 5,
     "look for a horizontal cut BEFORE a vertical one, so a full-width heading is its own "
     "band rather than part of a column"),
    (test_xy_cut_recurses_into_the_regions_it_makes, 4,
     "each band and each column goes back through the whole procedure; a region is a block "
     "only when neither axis yields a cut"),
    (test_xy_cut_is_a_partition_of_the_page, 4,
     "assign each box to a band by its LOW edge and count how many cut points lie at or below "
     "it, so no token is lost and none is claimed twice"),
    (test_order_within_block_lines_then_x, 4,
     "group boxes into lines by vertical centre, sort each line by left edge, concatenate"),
    (test_order_within_block_line_tolerance, 4,
     "two boxes are on one line when their vertical CENTRES are within LINE_TOL — within, so "
     "exactly LINE_TOL is one line; compare against the first box of the line, not the "
     "previous one"),
    (test_order_within_block_ignores_input_order, 4,
     "sort the boxes yourself and break every tie deterministically — on x, then on token id"),
    (test_header_footer_masks_digits, 5,
     "key on mask_digits(text.lower()) with the row the token sits on, or every page number "
     "looks unique and survives into the body"),
    (test_header_footer_needs_the_band_and_the_repetition, 4,
     "both conditions: inside the top or bottom HF_BAND of the page, AND on at least "
     "HF_MIN_PAGES_FRACTION of the pages, counting each page once"),
    (test_header_footer_on_the_corpus, 4,
     "run the same rule over every page of the corpus at once; the threshold is a fraction of "
     "all the pages you were given, not of one document's pages"),
    (test_kendall_tau_extremes, 4,
     "+1.0 for an identical ordering, -1.0 for a reversed one, and 1.0 for fewer than two "
     "items rather than a ZeroDivisionError"),
    (test_kendall_tau_known_values, 5,
     "divide the concordant-minus-discordant count by the number of PAIRS, n*(n-1)/2, not by "
     "the number of items"),
    (test_kendall_tau_validates_its_inputs, 3,
     "two orderings of the same ids, each seen once; anything else is a caller bug, so raise "
     "ValueError instead of scoring a comparison that means nothing"),
    (test_boundary_f1_perfect_and_oversegmented, 5,
     "a boundary is a position i where order[i] and order[i+1] sit in different blocks; score "
     "the predicted positions against the gold ones"),
    (test_boundary_f1_scores_both_sides_along_the_order, 4,
     "derive BOTH boundary sets by walking the reference order, rather than comparing the two "
     "partitions against each other directly, and score the INTERSECTION of the two sets of "
     "positions — how many boundaries each side proposed is not the measurement"),
    (test_boundary_f1_edges, 4,
     "no boundary on either side is a perfect segmentation (1.0, 1.0, 1.0); a token that "
     "neither partition holds is a ValueError"),
    (test_pipeline_recovers_the_gold_order_at_the_default_gap, 5,
     "find_gaps, xy_cut and order_within_block have to be right together: at the default gap "
     "the gold order is exactly recoverable from the geometry"),
    (test_kendall_tau_and_boundary_f1_disagree, 4,
     "over-segmentation leaves tau untouched and destroys boundary F1; if both move together, "
     "one of the two scores is measuring the other one"),
    (test_reading_order_moves_downstream_field_f1, 4,
     "the gold order and your order should both reach the ceiling at the default gap, and the "
     "naive raster order should not"),
]
