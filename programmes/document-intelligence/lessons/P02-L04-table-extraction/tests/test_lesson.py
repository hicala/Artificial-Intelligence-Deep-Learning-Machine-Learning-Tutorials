"""Autograder rubric for P02-L04-table-extraction. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

These rules matter in this particular lesson and are deliberate:

* Every case grades OBSERVABLE BEHAVIOUR, never one particular implementation. `ink_runs` may
  sort and sweep or sort and reduce; `teds_like` may use numpy or a list of lists; nothing
  here inspects source.
* `test_cells_in_band_clusters_before_assigning` is the case that stops the most attractive
  wrong answer: assigning each BOX to its columns and grouping by `(col, colspan)` afterwards.
  Every other cell case passes with that bug, because in them one cell is one box. It only
  shows up on a spanning header, where the first word covers one column and the last covers
  three, so the words disagree about the colspan and a single cell becomes several.
* `test_ink_runs_running_max_and_edges` grades the running maximum separately from the cut
  rule. Comparing against the PREVIOUS interval's high edge is correct on every ascending,
  non-overlapping input — which is every input a student is likely to try by hand.
* `test_merge_continuations_folds_the_tail` ends on the (col, colspan) pair because every
  other merge case has one cell per column, where keying the merge on `col` alone and sorting
  the merged row on `col` alone both give the right answer. A continuation whose colspan-1 cell
  starts at the same column as a colspan-3 cell above it is where the two diverge: the first
  bug joins the texts and rewrites the colspan, the second only misorders the row.
* `test_merge_continuations_leaves_input_alone` grades the aliasing bug rather than the
  arithmetic one. An in-place merge returns the right answer the first time and then poisons
  section 10, where seven variants are built from the same rows and each would measure the
  leftovers of the one before it.
* `test_teds_counts_the_row_node` pins the normalisation. Deleting a row of one cell costs
  1 (row) + 1 (cell) = 2 out of a five-node tree, so the score is 0.6. Forgetting the row node
  gives 0.8; dividing by the SMALLER tree gives 0.333. Both are plausible and both are wrong.
* `test_teds_identical_empty_and_clamped` covers the one case where this lesson's restricted
  distance is not bounded by the larger tree — one wide row against ten narrow ones — and
  requires the score to stay in [0, 1] rather than going negative.
* `test_pipeline_rebuilds_every_gold_table` and `test_the_two_scores_disagree_on_the_corpus`
  are the only cases that touch the lesson's own 24-table corpus. Both are deterministic
  (fixed seeds, no clock, no I/O). The first is the blast-radius case: a geometry bug that
  every hand-built case above happens to miss still cannot rebuild 24 tables exactly. The
  second pins the ORDERING the module exists to teach — content F1 blind to a shattered grid,
  TEDS-Struct blind to corrupted text — with wide bands rather than frozen numbers, because
  the point is that the two scores disagree, not that they disagree by a particular amount.
"""
import importlib.util, os
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

Box, Cell = lesson.Box, lesson.Cell
H = lesson.BOX_H


def _b(text: str, x: float, y: float = 0.0, w: float | None = None, h: float = H, page: int = 1) -> "Box":
    """A word box, width derived from the text unless you override it."""
    return Box(text, x, y, lesson.CHAR_W * len(text) if w is None else w, h, page)


# --- exercise 1: ink_runs ---------------------------------------------------------------

def test_ink_runs_merges_and_cuts():
    got = lesson.ink_runs([(0, 10), (12, 20)], 5)
    assert got == [(0.0, 20.0)], f"a 2-point gap is narrower than 5, so one run; got {got}"
    got = lesson.ink_runs([(0, 10), (20, 30)], 5)
    assert got == [(0.0, 10.0), (20.0, 30.0)], f"a 10-point gap cuts; got {got}"
    got = lesson.ink_runs([(0, 10), (15, 25)], 5)
    assert got == [(0.0, 25.0)], (
        f"ink_runs([(0,10),(15,25)], 5) = {got}: a gap of EXACTLY min_gap does not cut. The "
        "rule is strictly more than min_gap, and the section 6 sweep depends on it")
    got = lesson.ink_runs([(0, 10), (30, 40), (60, 70)], 5)
    assert got == [(0.0, 10.0), (30.0, 40.0), (60.0, 70.0)], f"three runs expected; got {got}"
    got = lesson.ink_runs([(0, 10), (16, 20), (21, 30)], 5)
    assert got == [(0.0, 10.0), (16.0, 30.0)], (
        f"ink_runs = {got}: cut once, then keep merging after the cut")
    got = lesson.ink_runs([(10, 20), (0, 100)], 5)
    assert got == [(0.0, 100.0)], (
        f"ink_runs = {got}: this needs BOTH halves of the rule — sort by low edge, and then "
        "let the running maximum swallow the interval nested inside the wide one")


def test_ink_runs_running_max_and_edges():
    got = lesson.ink_runs([(0, 100), (10, 20), (40, 50)], 5)
    assert got == [(0.0, 100.0)], (
        f"ink_runs = {got}: compare against the RUNNING MAXIMUM high edge, not the previous "
        "interval's high edge. One wide box otherwise invents a boundary just after itself")
    got = lesson.ink_runs([(0, 100), (10, 20), (140, 150)], 5)
    assert got == [(0.0, 100.0), (140.0, 150.0)], (
        f"ink_runs = {got}: the running maximum must survive a nested interval and still cut "
        "at the real gap afterwards")
    got = lesson.ink_runs([(40, 50), (0, 10)], 5)
    assert got == [(0.0, 10.0), (40.0, 50.0)], f"sort by low edge before sweeping; got {got}"
    assert lesson.ink_runs([], 5) == [], "empty in, empty out"
    assert lesson.ink_runs([(3, 9)], 5) == [(3.0, 9.0)], "a single interval is a single run"


# --- exercise 2: row_bands --------------------------------------------------------------

def test_row_bands_groups_and_sorts():
    got = [[z.text for z in band] for band in lesson.row_bands([_b("a", 0, 0), _b("b", 40, 3)], 12)]
    assert got == [["a", "b"]], (
        f"row_bands = {got}: boxes 3 points apart are one row. Cells are jittered off the "
        "baseline and the band has to tolerate that")
    got = [[z.text for z in band] for band in lesson.row_bands([_b("a", 0, 0), _b("b", 40, 40)], 12)]
    assert got == [["a"], ["b"]], f"row_bands = {got}: 40 points apart is two rows"
    got = [[z.text for z in band] for band in lesson.row_bands([_b("b", 40, 40), _b("a", 0, 0)], 12)]
    assert got == [["a"], ["b"]], (
        f"row_bands = {got}: bands come back top to bottom whatever order the boxes arrived in")
    got = [[z.text for z in band] for band in lesson.row_bands([_b("z", 90, 0), _b("a", 10, 2)], 12)]
    assert got == [["a", "z"]], f"boxes inside a band sort by x, not by y; got {got}"
    got = [[z.text for z in band] for band in lesson.row_bands([_b("m", 5, 0), _b("d", 5, 1)], 12)]
    assert got == [["d", "m"]], (
        f"row_bands = {got}: two boxes at the same x break the tie on text, so the order is "
        "the same on every machine and every run")
    assert lesson.row_bands([], 12) == [], "empty in, empty out"


def test_row_bands_running_max_and_boundary():
    got = [[z.text for z in band]
           for band in lesson.row_bands([_b("t", 0, 0, h=60.0), _b("b", 40, 40)], 12)]
    assert got == [["t", "b"]], (
        f"row_bands = {got}: the tall box reaches y=60, so the box at y=40 is inside it")
    three = [_b("t", 0, 0, h=60.0), _b("s", 20, 5), _b("x", 40, 40)]
    got = [[z.text for z in band] for band in lesson.row_bands(three, 12)]
    assert got == [["t", "s", "x"]], (
        f"row_bands = {got}: the tall box still reaches y=60 after the SHORT box at y=5 has "
        "been added to the band. Track the running maximum bottom edge; carrying the previous "
        "box's bottom edge forward drops the reach to 15 and splits the band")
    got = [[z.text for z in band] for band in lesson.row_bands([_b("a", 0, 0), _b("b", 40, 22)], 12)]
    assert got == [["a", "b"]], (
        f"row_bands = {got}: the box bottoms out at 10 and the next starts at 22, a gap of "
        "EXACTLY 12 — which does not cut, same rule as ink_runs")
    got = [[z.text for z in band] for band in lesson.row_bands([_b("a", 0, 0), _b("b", 40, 23)], 12)]
    assert got == [["a"], ["b"]], f"a gap of 13 is more than 12 and does cut; got {got}"


# --- exercise 3: cells_in_band ----------------------------------------------------------

def test_cells_in_band_clusters_before_assigning():
    cols = [(0.0, 50.0), (100.0, 150.0), (200.0, 250.0)]
    band = [_b("A", 10.0, w=40.0), _b("B", 56.0, w=60.0), _b("C", 122.0, w=100.0)]
    got = lesson.cells_in_band(band, cols, 16.0)
    assert got == [Cell("A B C", 0, 3)], (
        f"cells_in_band = {got}: cluster the band's ink into CELLS first, then ask which "
        "columns each cell covers. Assigning box by box gives 'A' colspan 1, 'B' colspan 2 "
        "and 'C' colspan 2, so one spanning header comes back as three cells")
    band = [_b("L", 10.0, w=30.0), _b("R", 210.0, w=30.0)]
    got = lesson.cells_in_band(band, cols, 16.0)
    assert got == [Cell("L", 0, 1), Cell("R", 2, 1)], (
        f"cells_in_band = {got}: a wide gap inside the band is a cell boundary, and the second "
        "cell belongs to column 2, not column 1")


def test_cells_in_band_colspan_and_join():
    cols = [(0.0, 50.0), (100.0, 150.0)]
    got = lesson.cells_in_band([_b("wide", 10.0, w=120.0)], cols, 16.0)
    assert got == [Cell("wide", 0, 2)], (
        f"cells_in_band = {got}: a cell overlapping two column runs has colspan 2")
    got = lesson.cells_in_band([_b("b", 13.0, w=7.0), _b("a", 0.0, w=7.0)], cols, 16.0)
    assert got == [Cell("a b", 0, 1)], (
        f"cells_in_band = {got}: join the member texts with ONE space, in ascending x order")
    got = lesson.cells_in_band([_b("r", 120.0, w=10.0), _b("l", 10.0, w=10.0)], cols, 16.0)
    assert got == [Cell("l", 0, 1), Cell("r", 1, 1)], (
        f"cells_in_band = {got}: cells come back sorted by (col, colspan)")


def test_cells_in_band_fallback_and_touching_edge():
    cols = [(0.0, 50.0), (100.0, 150.0)]
    got = lesson.cells_in_band([_b("t", 50.0, w=20.0)], cols, 16.0)
    assert got == [Cell("t", 0, 1)], (
        f"cells_in_band = {got}: this cell runs 50..70. It TOUCHES column 0's right edge and "
        "shares no ink with either column, so overlap is strictly positive and the cell falls "
        "back to the nearest column by midpoint — column 0, at distance 35 rather than 65")
    wide = [(0.0, 50.0), (100.0, 160.0)]
    got = lesson.cells_in_band([_b("t", 50.0, w=50.0)], wide, 16.0)
    assert got == [Cell("t", 0, 1)], (
        f"cells_in_band = {got}: this cell runs 50..100 and touches BOTH columns without "
        "sharing ink with either, so strict overlap finds nothing and the fallback gives it "
        "colspan 1. Inclusive bounds would hand it colspan 2 and invent a spanning cell")
    got = lesson.cells_in_band([_b("f", 400.0, w=10.0)], cols, 16.0)
    assert got == [Cell("f", 1, 1)], (
        f"cells_in_band = {got}: a cell far to the right of every column falls back to the "
        "LAST column, and is never dropped")
    assert lesson.cells_in_band([], cols, 16.0) == [], "empty band, empty row"


# --- exercise 4: merge_continuations -----------------------------------------------------

def test_merge_continuations_folds_the_tail():
    head = [Cell("R-1", 0, 1), Cell("Nordwind Logistik", 1, 1)]
    tail = [Cell("GmbH", 1, 1)]
    got = lesson.merge_continuations([head, tail], 0)
    assert got == [[Cell("R-1", 0, 1), Cell("Nordwind Logistik GmbH", 1, 1)]], (
        f"merge_continuations = {got}: a body row with nothing in the key column is the tail "
        "of the row above, and the two texts join with a single space")
    got = lesson.merge_continuations([[Cell("R-1", 0, 1)], [Cell("x", 2, 1)]], 0)
    assert got == [[Cell("R-1", 0, 1), Cell("x", 2, 1)]], (
        f"merge_continuations = {got}: a column the row above does not have yet is INSERTED, "
        "not discarded")
    got = lesson.merge_continuations([[Cell("R-1", 0, 1), Cell("z", 4, 1)], [Cell("m", 2, 1)]], 0)
    assert got == [[Cell("R-1", 0, 1), Cell("m", 2, 1), Cell("z", 4, 1)]], (
        f"merge_continuations = {got}: the merged row stays sorted by (col, colspan)")
    got = lesson.merge_continuations([[Cell("R-1", 0, 1)], [Cell("  ", 0, 1), Cell("t", 1, 1)]], 0)
    assert got == [[Cell("R-1", 0, 1), Cell("t", 1, 1)]], (
        f"merge_continuations = {got}: a key-column cell holding only whitespace is not a key. "
        "A blank reference is exactly what a wrapped line prints")
    got = lesson.merge_continuations([[Cell("R-1", 0, 1), Cell("Amounts", 2, 3)],
                                      [Cell("Net", 2, 1)]], 0)
    assert got == [[Cell("R-1", 0, 1), Cell("Net", 2, 1), Cell("Amounts", 2, 3)]], (
        f"merge_continuations = {got}: a colspan-1 cell and a colspan-3 cell that both start "
        "at column 2 are two different cells. Match on the PAIR (col, colspan): keying on col "
        "alone joins their texts and silently rewrites the spanning cell's colspan to 1, and "
        "sorting the merged row on col alone leaves the two in the wrong order")


def test_merge_continuations_headers_are_exempt():
    span = [Cell("Amounts", 2, 3)]
    sub = [Cell("Net", 2, 1), Cell("VAT", 3, 1)]
    body = [Cell("R-1", 0, 1)]
    got = lesson.merge_continuations([span, sub, body], 2)
    assert got == [span, sub, body], (
        f"merge_continuations = {got}: the sub-header row has nothing in column 0 either. "
        "Rows before n_header_rows are copied through untouched and nothing merges into them")
    tail = [Cell("GmbH", 1, 1)]
    got = lesson.merge_continuations([tail, [Cell("R-1", 0, 1)]], 0)
    assert got == [tail, [Cell("R-1", 0, 1)]], (
        f"merge_continuations = {got}: a continuation with no body row above it is kept as its "
        "own row — never dropped, and never merged forwards into the row below")
    got = lesson.merge_continuations([span, sub, tail], 2)
    assert got == [span, sub, tail], (
        f"merge_continuations = {got}: the first body row is a continuation and there is no "
        "body row above it, so it stands alone rather than merging into a header")


def test_merge_continuations_leaves_input_alone():
    head = [Cell("R-1", 0, 1), Cell("Nordwind Logistik", 1, 1)]
    tail = [Cell("GmbH", 1, 1)]
    rows = [head, tail]
    first = lesson.merge_continuations(rows, 0)
    assert head == [Cell("R-1", 0, 1), Cell("Nordwind Logistik", 1, 1)] and rows == [head, tail], (
        "merge_continuations mutated the rows it was given. Build new lists and new Cells. The "
        "bug is invisible once and fatal in section 10, where seven variants are built from "
        "the same rows and each would measure the leftovers of the one before it")
    second = lesson.merge_continuations(rows, 0)
    assert first == second, "merging the same rows twice must give the same answer"


# --- exercise 5: content_f1 --------------------------------------------------------------

def test_content_f1_is_a_multiset():
    got = lesson.content_f1([[Cell("30.00", 0, 1)]],
                            [[Cell("30.00", 0, 1), Cell("30.00", 1, 1)]])
    assert abs(got[0] - 1.0) < 1e-9 and abs(got[1] - 0.5) < 1e-9, (
        f"content_f1 = {got}: compare MULTISETS. Gold holds '30.00' twice and pred holds it "
        "once, so precision is 1.0 and recall 0.5 — a set would call this perfect")
    got = lesson.content_f1([[Cell("a", 0, 1), Cell("a", 1, 1)]], [[Cell("a", 0, 1)]])
    assert abs(got[0] - 0.5) < 1e-9 and abs(got[1] - 1.0) < 1e-9, (
        f"content_f1 = {got}: the extra copy is a false positive, so precision falls to 0.5")
    got = lesson.content_f1([[Cell("a", 0, 1), Cell("b", 1, 1)]], [[Cell("a", 0, 1)]])
    assert abs(got[2] - 2.0 / 3.0) < 1e-9, (
        f"content_f1 F1 = {got[2]}: precision 0.5 and recall 1.0 give F1 = 2/3. F1 is the "
        "HARMONIC mean of the two — the arithmetic mean would report 0.75 here, and would "
        "agree with the right answer on every case where precision and recall are equal")


def test_content_f1_ignores_position_and_blanks():
    got = lesson.content_f1([[Cell("net", 4, 3)]], [[Cell("Net", 0, 1)]])
    assert got == (1.0, 1.0, 1.0), (
        f"content_f1 = {got}: this score ignores col and colspan entirely and normalises the "
        "text with norm_cell, which lower-cases. That blindness is what section 10 measures")
    got = lesson.content_f1([[Cell("a", 0, 1)], [Cell("b", 0, 1)]],
                            [[Cell("b", 0, 1), Cell("a", 0, 1)]])
    assert got == (1.0, 1.0, 1.0), (
        f"content_f1 = {got}: flatten both tables. Which row a cell sits in is not this "
        "score's business")
    got = lesson.content_f1([[Cell("a", 0, 1), Cell("   ", 1, 1)]], [[Cell("a", 0, 1)]])
    assert got == (1.0, 1.0, 1.0), (
        f"content_f1 = {got}: cells that normalise to the empty string are dropped from the "
        "bag. Counting them makes a blank cell look like a prediction, and here it would pull "
        "precision to 0.5 for finding nothing")
    got = lesson.content_f1([[Cell("a", 0, 1)]], [[Cell("a", 0, 1), Cell("", 1, 1)]])
    assert got == (1.0, 1.0, 1.0), (
        f"content_f1 = {got}: the same on the gold side — a blank gold cell is not something "
        "an extractor can be asked to find")


def test_content_f1_zero_denominators():
    assert lesson.content_f1([[]], [[Cell("Net", 0, 1)]]) == (0.0, 0.0, 0.0), (
        "an empty prediction scores zero rather than raising ZeroDivisionError")
    assert lesson.content_f1([[Cell("Net", 0, 1)]], []) == (0.0, 0.0, 0.0), (
        "an empty gold scores zero rather than raising ZeroDivisionError")
    assert lesson.content_f1([], []) == (0.0, 0.0, 0.0), "two empty tables: guard both sides"


# --- exercise 6: row_distance ------------------------------------------------------------

def test_row_distance_substitution_cost():
    got = lesson.row_distance([Cell("x", 0, 1)], [Cell("x", 0, 3)])
    assert abs(got - 1.0) < 1e-9, (
        f"row_distance = {got}: identical text with a different colspan is a FULL-cost "
        "substitution. That rule is what makes a spanning header worth getting right")
    got = lesson.row_distance([Cell("abcd", 0, 1)], [Cell("abce", 0, 1)])
    assert abs(got - 0.25) < 1e-9, (
        f"row_distance = {got}: one character of four differs, so the substitution costs "
        "1 - 0.75 = 0.25. TEDS grades cell text by similarity, not by equality")
    got = lesson.row_distance([Cell("abcd", 0, 1)], [Cell("ABCD", 0, 1)])
    assert abs(got) < 1e-9, (
        f"row_distance = {got}: cell_sub_cost normalises both sides before comparing them")
    r = [Cell("Net", 0, 1), Cell("VAT", 1, 1)]
    assert lesson.row_distance(r, r) == 0.0, "a row against itself costs nothing"


def test_row_distance_insert_and_delete():
    r = [Cell("Net", 0, 1), Cell("VAT", 1, 1)]
    assert lesson.row_distance(r, r[:1]) == 1.0, (
        f"row_distance = {lesson.row_distance(r, r[:1])}: deleting one cell costs exactly 1.0")
    assert lesson.row_distance([], r) == 2.0, "inserting two cells costs 2.0"
    assert lesson.row_distance([], []) == 0.0, "two empty rows are distance 0.0"
    got = lesson.row_distance([Cell("a", 0, 1), Cell("b", 1, 1), Cell("c", 2, 1)],
                              [Cell("b", 0, 1)])
    assert abs(got - 2.0) < 1e-9, (
        f"row_distance = {got}: the cheapest path keeps 'b' and deletes the other two, for "
        "2.0. Aligning position by position instead would pay three substitutions")


def test_row_distance_structure_only():
    got = lesson.row_distance([Cell("abcd", 0, 1)], [Cell("zzzz", 0, 1)], structure_only=True)
    assert got == 0.0, f"row_distance(structure_only=True) = {got}: text must be ignored"
    got = lesson.row_distance([Cell("x", 0, 1)], [Cell("x", 0, 3)], structure_only=True)
    assert abs(got - 1.0) < 1e-9, (
        f"row_distance(structure_only=True) = {got}: a colspan disagreement still costs 1.0. "
        "structure_only drops the TEXT, not the shape")
    got = lesson.row_distance([Cell("a", 0, 1), Cell("b", 1, 1)], [Cell("q", 0, 1)],
                              structure_only=True)
    assert abs(got - 1.0) < 1e-9, (
        f"row_distance(structure_only=True) = {got}: one free substitution and one deletion")


# --- exercise 7: teds_like ----------------------------------------------------------------

def test_teds_counts_the_row_node():
    t = [[Cell("a", 0, 1)], [Cell("b", 0, 1)]]
    got = lesson.teds_like(t, t[:1])
    assert abs(got - 0.6) < 1e-9, (
        f"teds_like = {got}: deleting a row of one cell costs 1 for the ROW node plus 1 for "
        "the cell = 2, and the larger tree has 1 + 2 + 2 = 5 nodes, so the score is 0.6. "
        "Forgetting the row node gives 0.8; dividing by the smaller tree gives 0.333")
    got = lesson.teds_like(t[:1], t)
    assert abs(got - 0.6) < 1e-9, (
        f"teds_like = {got}: inserting costs the same as deleting, so the score is symmetric")
    wide = [[Cell("a", 0, 1), Cell("b", 1, 1), Cell("c", 2, 1)]]
    got = lesson.teds_like(wide, wide + [[Cell("d", 0, 1)]])
    assert abs(got - (1.0 - 2.0 / 7.0)) < 1e-9, (
        f"teds_like = {got}: one extra row of one cell costs 2, and the larger tree has "
        "1 root + 2 rows + 4 cells = 7 nodes, so the score is 1 - 2/7. A normaliser that "
        "counts only cells (6) or only the smaller tree (5) gives a different answer")


def test_teds_identical_empty_and_clamped():
    t = [[Cell("a", 0, 1)], [Cell("b", 0, 1)]]
    assert lesson.teds_like(t, t) == 1.0, "a table against itself scores 1.0"
    assert lesson.teds_like([], []) == 1.0, "two empty tables are identical, not undefined"
    assert 0.0 <= lesson.teds_like(t, []) <= 1.0, "a table against nothing still scores in [0,1]"
    wide = [[Cell(f"c{i}", i, 1) for i in range(10)]]
    tall = [[Cell(f"d{i}", 0, 1)] for i in range(10)]
    got = lesson.teds_like(wide, tall)
    assert 0.0 <= got <= 1.0, (
        f"teds_like = {got}: one wide row against ten narrow ones costs more than the larger "
        "tree has nodes, because this is a restricted tree edit distance rather than a full "
        "one. Clamp the score at 0 instead of reporting a negative similarity")


def test_teds_structure_only_threads_through():
    t = [[Cell("a", 0, 1)], [Cell("b", 0, 1)]]
    shifted = [[Cell("a", 0, 1)], [Cell("zzzz", 0, 1)]]
    assert lesson.teds_like(shifted, t, structure_only=True) == 1.0, (
        "structure_only has to reach cell_sub_cost: these tables have identical shape and "
        "differ only in text")
    assert lesson.teds_like(shifted, t) < 1.0, "with content on, a changed cell must cost"
    spanned = [[Cell("a", 0, 3)], [Cell("b", 0, 1)]]
    assert lesson.teds_like(spanned, t, structure_only=True) < 1.0, (
        "structure_only must still fail on a colspan disagreement — otherwise it is not a "
        "structure score at all")


# --- the corpus ---------------------------------------------------------------------------

def test_pipeline_rebuilds_every_gold_table():
    bad = [t["table_id"] for t in lesson.CORPUS
           if lesson.build_table(t["boxes"]) != t["gold"]]
    assert not bad, (
        f"{len(bad)} of {len(lesson.CORPUS)} tables did not rebuild exactly, first {bad[:3]}. "
        "At the default thresholds the pipeline must reproduce the gold grid cell for cell, "
        "including the colspan-3 header and the row whose wrap crosses the page break")


def test_the_two_scores_disagree_on_the_corpus():
    sample = lesson.CORPUS[:6]
    for t in sample:
        gold = t["gold"]
        built = lesson.build_table(t["boxes"])

        shattered = lesson.shatter_rows(built)
        assert lesson.content_f1(shattered, gold)[2] == 1.0, (
            f"{t['table_id']}: splitting every body row in two moves cells without changing "
            "one character, so the multiset of cell texts is untouched and content F1 must "
            "still be exactly 1.0")
        assert lesson.teds_like(shattered, gold, structure_only=True) < 0.75, (
            f"{t['table_id']}: TEDS-Struct must fall hard on a table with nearly twice the "
            "rows it should have — that is the whole claim of this module")

        noisy = lesson.ocr_noise(built)
        assert lesson.teds_like(noisy, gold, structure_only=True) == 1.0, (
            f"{t['table_id']}: corrupting cell text leaves every row, cell and colspan alone, "
            "so TEDS-Struct must read exactly 1.0. If it does not, structure_only is not "
            "reaching cell_sub_cost")
        assert lesson.content_f1(noisy, gold)[2] < 0.95, (
            f"{t['table_id']}: content F1 must notice corrupted cell text")

        reversed_body = lesson.reverse_body(built)
        assert lesson.content_f1(reversed_body, gold)[2] == 1.0, (
            f"{t['table_id']}: reading the body bottom-up keeps every cell, so content F1 is "
            "blind to it")
        assert lesson.teds_like(reversed_body, gold) < 0.9, (
            f"{t['table_id']}: TEDS aligns rows in order, so a reversed body must cost. A "
            "score that is order-blind is not measuring structure")


RUBRIC = [
    (test_ink_runs_merges_and_cuts, 5,
     "sort by low edge, sweep left to right, and cut only where the gap is STRICTLY wider "
     "than min_gap"),
    (test_ink_runs_running_max_and_edges, 4,
     "hold the running maximum high edge; a nested or very wide interval must not create a "
     "boundary behind itself"),
    (test_row_bands_groups_and_sorts, 5,
     "same rule on the y axis, but return the boxes: bands top to bottom, boxes inside a band "
     "sorted by (x, text)"),
    (test_row_bands_running_max_and_boundary, 4,
     "running maximum BOTTOM edge, and a gap of exactly min_gap does not start a new band"),
    (test_cells_in_band_clusters_before_assigning, 6,
     "ink_runs over the band first — each run is one cell — and only then ask which column "
     "runs that cell overlaps"),
    (test_cells_in_band_colspan_and_join, 4,
     "colspan is how many column runs the cell overlaps; the text is the member boxes joined "
     "by one space in ascending x order; cells come back sorted by (col, colspan)"),
    (test_cells_in_band_fallback_and_touching_edge, 4,
     "overlap is strictly positive, and a cell that overlaps nothing goes to the nearest "
     "column by midpoint rather than being dropped"),
    (test_merge_continuations_folds_the_tail, 5,
     "a body row with no non-blank cell at key_col appends into the row above, column by "
     "column, inserting any column that row lacks and keeping it sorted"),
    (test_merge_continuations_headers_are_exempt, 4,
     "rows before n_header_rows pass through untouched, and a continuation with no body row "
     "above it stands alone"),
    (test_merge_continuations_leaves_input_alone, 4,
     "build new rows and new Cells — never write into the lists you were handed"),
    (test_content_f1_is_a_multiset, 5,
     "count occurrences on both sides; a true positive is min(pred count, gold count); F1 is "
     "the harmonic mean"),
    (test_content_f1_ignores_position_and_blanks, 4,
     "flatten both tables, normalise with norm_cell, drop cells that normalise to empty, and "
     "ignore col, colspan and row entirely"),
    (test_content_f1_zero_denominators, 3,
     "return 0.0 for precision, recall and F1 rather than raising ZeroDivisionError"),
    (test_row_distance_substitution_cost, 5,
     "cell_sub_cost already encodes the rule: full cost when colspan differs, otherwise "
     "1 minus the normalised similarity of the texts"),
    (test_row_distance_insert_and_delete, 4,
     "a Levenshtein alignment over cells with insert and delete both costing 1.0, so the DP "
     "can skip a cell instead of paying for a bad substitution"),
    (test_row_distance_structure_only, 4,
     "pass structure_only down to cell_sub_cost; it drops the text comparison and keeps the "
     "colspan one"),
    (test_teds_counts_the_row_node, 4,
     "deleting or inserting a row costs 1 for the row node plus 1 per cell, and the "
     "normaliser is max(table_size(pred), table_size(gold))"),
    (test_teds_identical_empty_and_clamped, 4,
     "two empty tables score 1.0, and the score is clamped into [0, 1] because this "
     "restricted distance is not bounded by the larger tree"),
    (test_teds_structure_only_threads_through, 4,
     "structure_only has to reach row_distance and then cell_sub_cost, or it silently grades "
     "content while claiming not to"),
    (test_pipeline_rebuilds_every_gold_table, 6,
     "at the default thresholds, profile then band then cluster then stitch must reproduce "
     "every gold grid exactly — colspan-3 header and page-break wrap included"),
    (test_the_two_scores_disagree_on_the_corpus, 6,
     "content F1 must be exactly 1.0 on a shattered grid and on a reversed body; TEDS-Struct "
     "must be exactly 1.0 on corrupted text; and each must fail where the other is blind"),
]
