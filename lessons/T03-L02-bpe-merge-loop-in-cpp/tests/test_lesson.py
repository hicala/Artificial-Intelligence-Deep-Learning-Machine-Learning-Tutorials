"""Autograder rubric for T03-L02. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure). Partial credit is the point — a
student whose `apply_merge` works scores for it while `train` is still a TODO.

The cases below are deliberately not the ones in the notebook's public `_check_*` functions:
passing what you can see is not the same as being right.
"""
import importlib.util
import os
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
# Each module points at the lesson.cpp beside it, so the reference and the student binary are
# built side by side and never overwrite each other.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

# One slice, used by every test that needs real text: small enough that the Python oracle
# costs a fraction of a second, big enough that a lazy implementation cannot fake it.
_HIDDEN_BYTES = 24_000
_HIDDEN_MERGES = 25


def test_cpp_compiles():
    """Whatever the student did to lesson.cpp, it is still a valid C++17 translation unit."""
    binary = lesson.build_cpp(force=True)
    assert Path(binary).exists(), f"make reported success but produced no binary at {binary}"


def test_count_pairs_hidden_text():
    seq = lesson.corpus_symbols(limit=_HIDDEN_BYTES)
    assert lesson.cpp_count_pairs(seq) == lesson.count_pairs_py(seq)


def test_count_pairs_boundary_and_edges():
    boundary = lesson.BOUNDARY
    # Boundaries at both ends, a doubled boundary in the middle, and sequences too short to
    # hold a pair at all.
    assert lesson.cpp_count_pairs(
        [boundary, 4, 5, boundary, boundary, 5, 4, boundary]) == {(4, 5): 1, (5, 4): 1}
    assert lesson.cpp_count_pairs([boundary]) == {}
    assert lesson.cpp_count_pairs([3]) == {}
    assert lesson.cpp_count_pairs([]) == {}
    # Overlapping runs are counted in full: four 7s hold three (7,7) pairs.
    assert lesson.cpp_count_pairs([7, 7, 7, 7]) == {(7, 7): 3}


def test_best_pair_tie_break():
    # A three-way tie, offered in an order that rewards anyone who kept the first one seen.
    assert lesson.cpp_best_pair({(300, 1): 6, (12, 99): 6, (12, 5): 6}) == (12, 5, 6)
    # Tied on the left symbol, so the right symbol decides — both ways round, because any one
    # ordering can be satisfied by luck of the container rather than by a real tie-break.
    assert lesson.cpp_best_pair({(8, 40): 2, (8, 9): 2}) == (8, 9, 2)
    assert lesson.cpp_best_pair({(8, 9): 2, (8, 40): 2}) == (8, 9, 2)
    # The property that actually matters, and the only one a lucky iteration order cannot fake:
    # the winner may not depend on the order the counts arrived in. std::unordered_map promises
    # no ordering, so "the first maximum I met" answers differently for different permutations
    # of one tied set. Several permutations, so this bites on any standard library, not just
    # the one that happens to be installed here.
    tied = [(3, 4), (17, 18), (41, 42), (88, 89), (150, 151), (260, 261), (301, 302)]
    for order in (tied, list(reversed(tied)), tied[3:] + tied[:3], tied[1::2] + tied[0::2]):
        assert lesson.cpp_best_pair({pair: 9 for pair in order}) == (3, 4, 9)


def test_best_pair_empty_and_singleton():
    assert lesson.cpp_best_pair({}) is None
    assert lesson.cpp_best_pair({(260, 261): 1}) == (260, 261, 1)


def test_apply_merge_overlap_and_tail():
    # An odd run, a match ending exactly at the last position, and a match at the very start.
    assert lesson.cpp_apply_merge([7, 7, 7, 7, 7], 7, 7, 400) == [400, 400, 7]
    assert lesson.cpp_apply_merge([1, 9, 9], 9, 9, 400) == [1, 400]
    assert lesson.cpp_apply_merge([9, 9, 1], 9, 9, 400) == [400, 1]
    # Ids above 255 must survive: symbols are ints, not bytes.
    assert lesson.cpp_apply_merge([300, 301, 300, 301], 300, 301, 500) == [500, 500]


def test_apply_merge_leaves_others_alone():
    assert lesson.cpp_apply_merge([1, 2, 3], 2, 1, 400) == [1, 2, 3]
    assert lesson.cpp_apply_merge([], 1, 2, 400) == []
    assert lesson.cpp_apply_merge([5], 5, 5, 400) == [5]
    seq = lesson.corpus_symbols(limit=_HIDDEN_BYTES)
    a, b, _ = lesson.best_pair_py(lesson.count_pairs_py(seq))
    assert lesson.cpp_apply_merge(seq, a, b, 998) == lesson.apply_merge_py(seq, a, b, 998)


def test_train_matches_python_oracle():
    path = lesson.oracle_corpus(_HIDDEN_BYTES)
    cpp = lesson.cpp_train(path, _HIDDEN_MERGES)
    ref = lesson.train_py(lesson.pretokenise(path.read_bytes()), _HIDDEN_MERGES)
    assert list(cpp["merges"]) == list(ref["merges"])
    assert cpp["final_symbols"] == ref["final_symbols"]


def test_train_stop_rule():
    """Stop below kMinCount, stop at n_merges — and do neither by returning nothing."""
    path = Path(lesson.LESSON_ROOT) / "build" / f"rubric_stop_{lesson.CPP_BIN.name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    # No pair occurs twice here, so no merge is worth making however many are asked for.
    path.write_bytes(b"abcdef ghijkl")
    assert lesson.cpp_train(path, 50)["merges"] == []
    # Four merges are available and two are asked for, so the cap has to bite. The two it does
    # make are pinned exactly, ids included: a `train` that always returns an empty list, or
    # one that numbers ids from anywhere but kFirstMergeId, cannot collect these points by
    # doing nothing. Derived from the Python trainer in the notebook, not from the C++.
    path.write_bytes(b"abcabcabc abcabcabc abcabcabc")
    assert lesson.cpp_train(path, 2)["merges"] == [(97, 98, 257, 9), (257, 99, 258, 9)]


def test_cpp_matches_python_oracle():
    """Correctness backstop: the C++ trainer must produce the SAME merges as the Python one.

    The speed-up is REPORTED in the notebook, never graded. Wall-clock time is a property of
    the machine a student happens to be on, and nobody should lose points because their laptop
    was busy. Correctness is the thing a rubric can fairly assert.
    """
    report = lesson.speed_report(lesson.HEADLINE_MERGES)
    assert report["identical_merges"], (
        "the C++ and Python trainers produced different merges; fix correctness before speed")


RUBRIC = [
    (test_cpp_compiles, 2, "read the compiler's FIRST error; the rest are usually its echo"),
    (test_count_pairs_hidden_text, 3, "exercise 1: one scan, skipping pairs that touch kBoundary"),
    (test_count_pairs_boundary_and_edges, 3,
     "exercise 1: guard seq.size() < 2, and a doubled boundary must yield no pair"),
    (test_best_pair_tie_break, 3,
     "exercise 2: ties go to the smaller (a,b) — compare packed keys, not iteration order"),
    (test_best_pair_empty_and_singleton, 2,
     "exercise 2: an empty map returns false and writes nothing through the out-parameters"),
    (test_apply_merge_overlap_and_tail, 3,
     "exercise 3: advance the read cursor by two after a match, and never read past the end"),
    (test_apply_merge_leaves_others_alone, 2,
     "exercise 3: resize() to the WRITE cursor, so an absent pair changes nothing"),
    (test_train_matches_python_oracle, 5,
     "exercise 4: number merges kFirstMergeId + k, and apply each one before the next count"),
    (test_train_stop_rule, 2,
     "exercise 4: stop when the best count is below kMinCount, before recording that merge"),
    (test_cpp_matches_python_oracle, 3,
     "build with -O2 and keep allocations out of the loop; the Makefile already does both"),
]
