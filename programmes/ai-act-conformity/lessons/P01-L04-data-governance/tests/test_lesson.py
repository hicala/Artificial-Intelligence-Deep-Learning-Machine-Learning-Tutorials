"""Autograder rubric for P01-L04. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson and are deliberate:

* Every expected value here is re-derived from `lesson.TABLE` by code in THIS file, in a
  different shape from the lesson's own (plain Python loops and dicts rather than numpy
  masks). Nothing is a literal copied out of the notebook's output, and nothing calls the
  student's function to decide what the student's function should have returned.
* `lesson.TABLE` is never mutated. Any test that needs a different table builds a small one
  of its own, or copies every column first. A mutated fixture would leak into the tests that
  run after it and make a failure depend on rubric order.
* Several tests deliberately run a function on a HAND-BUILT table rather than on the lesson's,
  because the lesson's table has exactly one instance of each defect and an implementation can
  pass it by hard-coding. The precedence test, the tie-break test, the "min_support is a
  parameter, not a constant" test and the "category present in only one split" test are all of
  that kind.
* Four cases exist purely because the fixture is one-sided and mutation testing found it:
  the fixture's predictive identifier runs WITH the target, so `abs()` was invisible; every
  declared share lands on a whole number of rows, so `round` and `ceil` agreed; every numeric
  feature that drifted, drifted upwards, so comparing a signed statistic passed; and every
  subgroup gap sits either well inside or well outside its floor, so a fixed five-point
  threshold happened to produce the same findings. Each of those tests now also runs on a
  hand-built table that separates the right answer from the lucky one.
* `test_bias_probe_floor_is_the_parameter` is what stops an implementation that special-cases
  north/18-29, or that filters small subgroups out of `subgroups` entirely, from scoring full
  marks. Lower the floor and the same subgroup must become a finding.
* The power tests never assert a rounded literal. They recompute the closed form here. What is
  graded is that the value is the right closed form of the right quantity — including, for the
  bias probe, that it is the maximum over the SUPPORTED subgroups only.

Added by the independent review, 2026-09-16. A second mutation wave found seven more
implementations that scored a full 101/101 against the rubric as it stood, and every one of
them is a shape the fixture alone cannot separate from the right answer:

  * VERDICTS THAT COULD NOT COME BACK CLEAN. The fixture fails all five checks, so
    `coverage_report`, `drift_report` and `provenance_report` were never handed a split with
    nothing wrong with it. A verdict hard-wired to "fail" scored full marks, as did a coverage
    verdict that fired on "empty" and "undeclared" but shrugged at "thin" and "over". Each of
    those three tests now also runs on a hand-built clean table, and the coverage one on a
    split whose ONLY defect is over-representation and another whose only defect is thinness.
  * THE OVER-REPRESENTATION BOUNDARY. No stratum in the fixture sits on the line, so `>` and
    `>=` agreed everywhere. `test_coverage_status_precedence_and_thresholds` now includes a
    stratum at exactly declared * factor — 75 of 200 rows against a declared 0.25, which is
    0.375 on both sides in binary floating point, not merely close to it.
  * THE LEAKED ROW IDS. Only the LENGTH of `overlap_row_ids`, its sortedness and its
    membership of the test split were graded, so a list of twenty of the wrong test rows
    passed. `_expected_overlaps` now re-derives the ids themselves and they are compared.
  * THE ORDER OF THE BIAS FINDINGS. The fixture holds exactly one finding, so sorting by
    |gap| ascending was indistinguishable from descending. A three-subgroup table with two
    findings 35 and 25 points from its baseline now pins the order.
  * FINDINGS AS PLACEHOLDERS. `findings` was graded on count and non-emptiness alone, so
    ["finding", ...] scored full marks. They must now be pairwise distinct, at least 20
    characters, and carry the quantity they report — which the docstring in the notebook
    states, so it is a requirement a student can read rather than a hidden trap.
  * A SPURIOUS ZERO-COUNT FINDING. Raising the near-duplicate finding whenever `near_ids` is
    non-empty, rather than when it exceeds the exact count, prints "0 further test rows
    match" on a table whose duplicates are all exact. There is now such a table.

One mutant SURVIVES and is left alone because it is genuinely equivalent, not missed:
appending `exact_key(...)` instead of `row_id` to the local `exact_ids` list. That list is
only ever consumed by `len()` and a truthiness test, so the returned report is identical for
every input; it was measured on the fixture, on a table with repeated keys inside one split,
and on a table where every row is identical, and the two reports compared equal each time.
"""
import importlib.util, math, os
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

TABLE = lesson.TABLE
N_ROWS = len(TABLE["row_id"])
Z_SUM = lesson.Z_ALPHA_95 + lesson.Z_BETA_80


# --- independent re-derivations, in plain Python ----------------------------------------

def _col(name: str) -> list:
    """One column as a plain Python list, so nothing here shares the lesson's numpy path."""
    return [v.item() if hasattr(v, "item") else v for v in TABLE[name]]


def _rows_of(split: str) -> list:
    return [i for i, s in enumerate(_col("split")) if str(s) == split]


def _stratum_counts(split: str) -> dict:
    region, age = _col("region"), _col("age_band")
    out: dict = {}
    for i in _rows_of(split):
        key = (str(region[i]), str(age[i]))
        out[key] = out.get(key, 0) + 1
    return out


def _expected_coverage_status(split: str = "train") -> dict:
    """Re-derive each stratum's status without touching the lesson's coverage_report."""
    counts = _stratum_counts(split)
    n = len(_rows_of(split))
    declared = lesson.DECLARED_POPULATION
    out = {}
    for key in set(declared) | set(counts):
        count = counts.get(key, 0)
        if key not in declared:
            out[key] = "undeclared"
        elif count == 0:
            out[key] = "empty"
        elif count < lesson.MIN_SUPPORT:
            out[key] = "thin"
        elif count / n > declared[key] * lesson.OVER_REPRESENTATION_FACTOR:
            out[key] = "over"
        else:
            out[key] = "ok"
    return out


def _expected_overlaps() -> tuple:
    """Re-derive (exact, near, near_row_ids) test-vs-train overlap with lists, not key sets.

    The row_ids matter as much as the counts: a report that returns the right NUMBER of
    ids while naming the wrong rows sends a remediation team to twenty innocent files.
    """
    region, age, chan = _col("region"), _col("age_band"), _col("channel")
    income, dti = _col("income"), _col("dti")
    row_id = _col("row_id")

    def exact(i):
        return (str(region[i]), str(age[i]), str(chan[i]), float(income[i]), float(dti[i]))

    def near(i):
        return (str(region[i]), str(age[i]), str(chan[i]),
                int(round(float(income[i]))), round(float(dti[i]), lesson.NEAR_DTI_DP))

    train = _rows_of("train")
    train_exact = [exact(i) for i in train]
    train_near = [near(i) for i in train]
    n_exact = sum(1 for i in _rows_of("test") if exact(i) in train_exact)
    near_ids = sorted(int(row_id[i]) for i in _rows_of("test") if near(i) in train_near)
    return n_exact, len(near_ids), near_ids


def _expected_subgroups() -> tuple:
    """Re-derive the test split's subgroup sizes, error counts and baseline."""
    region, age = _col("region"), _col("age_band")
    label, score = _col("label"), _col("score")
    sizes: dict = {}
    errors: dict = {}
    total = 0
    rows = _rows_of("test")
    for i in rows:
        key = (str(region[i]), str(age[i]))
        predicted = 1 if float(score[i]) >= lesson.DECISION_THRESHOLD else 0
        bad = int(predicted != int(label[i]))
        sizes[key] = sizes.get(key, 0) + 1
        errors[key] = errors.get(key, 0) + bad
        total += bad
    return sizes, errors, total / len(rows)


def _mde(n_a: int, n_b: int, p: float) -> float:
    if n_a <= 0 or n_b <= 0:
        return 1.0
    return min(1.0, Z_SUM * math.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b)))


def _tiny_table(rows: list) -> dict:
    """Build a small column store from a list of dicts. Never touches lesson.TABLE."""
    import numpy as np
    keys = rows[0].keys()
    out = {k: np.array([r[k] for r in rows]) for k in keys}
    out["row_id"] = np.arange(len(rows), dtype=int)
    return out


# --- exercise 1: the power tools ---------------------------------------------------------

def test_min_detectable_share_is_the_exact_form():
    f = lesson.min_detectable_share
    for n in (1, 7, 40, 1000, 1500, 25000):
        want = 1.0 - 0.05 ** (1.0 / n)
        got = f(n)
        assert abs(got - want) < 1e-12, (
            f"min_detectable_share({n}) = {got!r}, expected {want!r} — solve "
            "(1 - p)**n = 1 - confidence, do not use the 3/n approximation"
        )
    assert abs(f(20) - (1 - 0.05 ** (1 / 20))) < 1e-12 and abs(f(20) - 3 / 20) > 1e-3, (
        "at n = 20 the exact form and the rule of three differ by more than a thousandth; "
        "yours matches the approximation"
    )


def test_min_detectable_share_honours_confidence_and_edges():
    f = lesson.min_detectable_share
    assert abs(f(100, 0.99) - (1 - 0.01 ** (1 / 100))) < 1e-12, (
        "the `confidence` argument must be used: 1 - (1 - confidence)**(1/n), not a hard-coded "
        "0.05"
    )
    assert f(100, 0.99) > f(100, 0.95), "a higher confidence demands a larger share"
    assert f(0) == 1.0 and f(-5) == 1.0, (
        "n <= 0 must return 1.0 — a sample of nothing rules nothing out, and returning 0.0 "
        "inverts the meaning of the number"
    )


def test_min_detectable_difference_is_the_two_proportion_formula():
    f = lesson.min_detectable_difference
    for n_a, n_b, p in ((60, 540, 0.1383333), (30, 570, 0.1383333), (300, 300, 0.25),
                        (1500, 600, 0.5), (12, 900, 0.02)):
        want = _mde(n_a, n_b, p)
        got = f(n_a, n_b, p)
        assert abs(got - want) < 1e-12, (
            f"min_detectable_difference({n_a}, {n_b}, {p}) = {got!r}, expected {want!r} — "
            "(z_alpha + z_beta) * sqrt(p*(1-p)*(1/n_a + 1/n_b)), capped at 1.0"
        )


def test_min_detectable_difference_uses_power_and_both_groups():
    f = lesson.min_detectable_difference
    alpha_only = lesson.Z_ALPHA_95 * math.sqrt(0.25 * (1 / 200 + 1 / 200))
    assert abs(f(200, 200, 0.5) - alpha_only) > 1e-4, (
        "dropping z_beta reports a floor about 30% smaller than the truth — equivalently, the\n"
        "honest floor is about 43% larger than the one you returned. Both quantiles are in\n"
        "the sum"
    )
    assert abs(f(50, 500, 0.3) - f(500, 50, 0.3)) < 1e-12, (
        "the formula is symmetric in the two group sizes; yours is not, so only one n is "
        "reaching the arithmetic"
    )
    assert f(3, 3, 0.5) == 1.0 and f(1, 10000, 0.5) == 1.0, (
        "cap the result at 1.0: a difference between two rates cannot exceed 1"
    )
    assert f(0, 10, 0.2) == 1.0 and f(10, 0, 0.2) == 1.0, "an empty group detects nothing"
    assert f(50, 50, 0.5) > f(50, 50, 0.1), (
        "variance peaks at p = 0.5, so the detectable difference must be largest there"
    )


# --- exercise 2: leakage -----------------------------------------------------------------

def test_near_key_rounds_to_recorded_precision():
    import numpy as np
    t = _tiny_table([
        {"region": "north", "age_band": "30-49", "channel": "branch",
         "income": 50000.0, "dti": 0.372},
        {"region": "north", "age_band": "30-49", "channel": "branch",
         "income": 50000.37, "dti": 0.3722},
        {"region": "north", "age_band": "30-49", "channel": "online",
         "income": 50000.0, "dti": 0.372},
        {"region": "north", "age_band": "30-49", "channel": "branch",
         "income": 50004.0, "dti": 0.372},
    ])
    k = lesson.near_key
    assert isinstance(k(t, 0), tuple) and len(k(t, 0)) == 5, "near_key returns a 5-tuple"
    assert isinstance(k(t, 0)[3], int), (
        f"the income element must be an int, not {type(k(t, 0)[3]).__name__} — round(x) "
        "returns a float and 50000.0 is a different dict key from 50000"
    )
    assert k(t, 0) == k(t, 1), "cent-level noise must not produce a second key"
    assert k(t, 0) != k(t, 2), "the channel is part of the key"
    assert k(t, 0) != k(t, 3), (
        "two applicants four units apart on income are two applicants; a key that merges "
        "them is measuring its own tolerance"
    )
    assert k(t, 1, dti_dp=1) != k(t, 1, dti_dp=4), (
        "dti_dp must be used; yours ignores it and hard-codes the precision"
    )


def test_near_key_is_hashable_and_pure():
    before = [float(v) for v in TABLE["income"][:5]]
    keys = [lesson.near_key(TABLE, i) for i in range(5)]
    assert len({type(k) for k in keys}) == 1 and all(isinstance(k, tuple) for k in keys)
    assert len(set(keys)) >= 1, "keys must be hashable"
    assert [float(v) for v in TABLE["income"][:5]] == before, (
        "near_key must not modify the table it reads"
    )


def test_leakage_overlap_counts():
    want_exact, want_near, want_ids = _expected_overlaps()
    rep = lesson.leakage_report(TABLE)
    assert rep["exact_overlap"] == want_exact, (
        f"exact_overlap is {rep['exact_overlap']}, expected {want_exact} — count TEST rows "
        "whose exact key appears in train"
    )
    assert rep["near_overlap"] == want_near, (
        f"near_overlap is {rep['near_overlap']}, expected {want_near}. The near set CONTAINS "
        "the exact matches; reporting only the extra ones understates the overlap"
    )
    assert rep["near_overlap"] > rep["exact_overlap"], (
        "some test rows were re-entered rather than copied; a report where near equals exact "
        "is matching on raw floats"
    )
    assert rep["overlap_row_ids"] == sorted(rep["overlap_row_ids"]), "and it is sorted"
    assert all(str(TABLE["split"][i]) == "test" for i in rep["overlap_row_ids"]), (
        "the ids reported are the TEST rows that leaked, not their training twins"
    )
    # Counting the leaked rows is half the job; naming them is the other half. The right
    # NUMBER of ids taken from the wrong rows sends a remediation team to clean files.
    assert rep["overlap_row_ids"] == want_ids, (
        f"overlap_row_ids must be the row_id of every near-overlapping TEST row. You "
        f"reported {len(rep['overlap_row_ids'])} ids and {len(set(rep['overlap_row_ids']) - set(want_ids))} "
        f"of them belong to rows that did not leak; {len(set(want_ids) - set(rep['overlap_row_ids']))} "
        "rows that did leak are missing. Append table['row_id'][i] for the matching test row i"
    )


def test_leakage_identifier_and_verdict():
    rep = lesson.leakage_report(TABLE)
    want = abs(lesson.pearson_r(TABLE["applicant_ref"], TABLE["label"]))
    assert abs(rep["id_target_r"] - want) < 1e-12, (
        f"id_target_r is {rep['id_target_r']!r}, expected the absolute correlation "
        f"{want!r} between applicant_ref and label over the whole table"
    )
    # The fixture's identifier happens to run WITH the target, so `abs` is invisible on it.
    # This one runs against it: the references were issued in descending order of outcome,
    # which is exactly as much leakage and exactly as useless to a model in production.
    rows = [{"split": "train", "region": "north", "age_band": "30-49", "channel": "branch",
             "income": 40000.0 + k * 1000, "dti": round(0.1 + k * 0.05, 3),
             "label": (0, 0, 1, 1)[k], "applicant_ref": (4, 3, 2, 1)[k]} for k in range(4)]
    negative = lesson.leakage_report(_tiny_table(rows))
    assert negative["id_target_r"] > 0, (
        f"id_target_r came back {negative['id_target_r']!r} for an identifier that runs "
        "AGAINST the target. Report the ABSOLUTE correlation: a minus sign is not a defence, "
        "and a signed value silently passes the threshold test"
    )
    assert abs(negative["id_target_r"] - 2 / math.sqrt(5)) < 1e-9, (
        f"expected |r| = {2 / math.sqrt(5)!r} on that table, got {negative['id_target_r']!r}"
    )
    assert negative["verdict"] == "fail", "a negatively predictive identifier is still leakage"
    assert rep["verdict"] == "fail", "this table leaks; the verdict is 'fail'"
    assert isinstance(rep["findings"], list) and len(rep["findings"]) == 3, (
        f"three distinct defects are present — exact duplicates, further near duplicates and "
        f"a predictive identifier. You reported {len(rep['findings'])}"
    )
    assert all(isinstance(f, str) and f.strip() for f in rep["findings"]), (
        "each finding is a non-empty string a human can read"
    )
    assert len(set(rep["findings"])) == 3, (
        f"three DISTINCT defects means three distinct sentences; yours repeat: {rep['findings']}"
    )
    assert all(len(f.strip()) >= 20 and any(c.isdigit() for c in f) for f in rep["findings"]), (
        f"each finding must name the quantity it reports — a count of rows, or the |r| it "
        f"measured — in at least 20 characters. A placeholder like 'leak' is a finding an "
        f"inspector cannot act on. Yours: {rep['findings']}"
    )


def test_leakage_verdict_passes_a_clean_table():
    """A report that always fails is not a check. Hand it a table with nothing wrong."""
    # The reference column is deliberately uncorrelated with the label: within each split
    # the labels run 0, 1, 1, 0 against references 1, 2, 3, 4, so the covariance is exactly
    # zero and the only thing this table can fail on is a duplicate that is not there.
    labels = (0, 1, 1, 0)
    rows = []
    for split, base in (("train", 40000.0), ("test", 61000.0)):
        for k in range(4):
            rows.append({"split": split, "region": "north", "age_band": "30-49",
                         "channel": "branch", "income": base + k * 1000,
                         "dti": round(0.10 + k * 0.05, 3),
                         "label": labels[k], "applicant_ref": 1 + k})
    clean = _tiny_table(rows)
    assert abs(lesson.pearson_r(clean["applicant_ref"], clean["label"])) < 1e-12
    rep = lesson.leakage_report(clean)
    assert rep["exact_overlap"] == 0 and rep["near_overlap"] == 0, (
        "no row of this table appears in both splits, so both overlaps are 0"
    )
    assert rep["verdict"] == "pass" and rep["findings"] == [], (
        f"a clean table must pass; yours reported {rep['findings']}"
    )
    # And the other half of "no spurious findings": a table whose duplicates are ALL exact.
    # The near set then equals the exact set, and there is no second duplicate defect to
    # report. `if near_ids:` announces "0 further test rows match", which is not a finding.
    dup = []
    for split in ("train", "test"):
        for k in range(4):
            dup.append({"split": split, "region": "north", "age_band": "30-49",
                        "channel": "branch", "income": 40000.0 + k * 1000,
                        "dti": round(0.10 + k * 0.05, 3),
                        "label": labels[k], "applicant_ref": 1 + k})
    exact_only = lesson.leakage_report(_tiny_table(dup))
    assert exact_only["exact_overlap"] == 4 and exact_only["near_overlap"] == 4, (
        "every test row of this table is byte-identical to a training row, so exact and near "
        f"are both 4; you reported {exact_only['exact_overlap']}/{exact_only['near_overlap']}"
    )
    assert len(exact_only["findings"]) == 1, (
        f"there is ONE duplicate defect here, not two: the near set adds nothing to the exact "
        f"set. Raising the near finding whenever near_ids is non-empty prints '0 further test "
        f"rows match' and pads the report. You reported {exact_only['findings']}"
    )


def test_leakage_power_is_the_correlation_floor():
    rep = lesson.leakage_report(TABLE)
    p = rep["power"]
    assert set(p) == {"statistic", "value", "statement"}, f"power() shape, got {sorted(p)}"
    assert p["statistic"] == "min_detectable_r", (
        f"the leakage power statistic is named 'min_detectable_r', yours is {p['statistic']!r}"
    )
    want = math.tanh(lesson.Z_ALPHA_95 / math.sqrt(N_ROWS - 3))
    assert abs(p["value"] - want) < 1e-12, (
        f"power value is {p['value']!r}, expected tanh(Z_ALPHA_95 / sqrt(n - 3)) = {want!r}"
    )
    assert isinstance(p["statement"], str) and len(p["statement"]) >= 40, (
        "the statement is prose an inspector reads; a blank one is a missing power statement"
    )


# --- exercise 3: coverage and the gap report ---------------------------------------------

def test_coverage_reports_the_union_of_strata():
    cov = lesson.coverage_report(TABLE)
    want = set(lesson.DECLARED_POPULATION) | set(_stratum_counts("train"))
    assert set(cov["strata"]) == want, (
        f"the report must cover the declared strata UNION the observed ones ({len(want)} of "
        f"them); yours covers {len(cov['strata'])}. Looping over DECLARED_POPULATION alone "
        "silently drops the stratum the data has and the declaration does not"
    )
    assert cov["n"] == len(_rows_of("train")), f"n is the split's row count, not the table's"
    counts = _stratum_counts("train")
    for key, cell in cov["strata"].items():
        assert cell["count"] == counts.get(key, 0), f"{key} count is {cell['count']}"
        assert abs(cell["observed_share"] - cell["count"] / cov["n"]) < 1e-12, (
            f"{key}: observed_share is the count over the SPLIT's n"
        )
        assert cell["declared_share"] == lesson.DECLARED_POPULATION.get(key, 0.0), (
            f"{key}: an undeclared stratum's declared_share is 0.0"
        )


def test_coverage_statuses_match_an_independent_derivation():
    cov = lesson.coverage_report(TABLE)
    want = _expected_coverage_status("train")
    wrong = {k: (cov["strata"][k]["status"], want[k])
             for k in want if cov["strata"][k]["status"] != want[k]}
    assert not wrong, f"status disagrees on {wrong} (yours, expected)"
    tally = {name: sum(1 for v in want.values() if v == name)
             for name in ("ok", "thin", "empty", "over", "undeclared")}
    assert cov["counts"] == tally, f"counts came out {cov['counts']}, expected {tally}"
    assert cov["verdict"] == "fail", "a split with a non-ok stratum does not pass"


def test_coverage_status_precedence_and_thresholds():
    """Hand-built table: every status boundary, including two that overlap."""
    rows = []
    plan = {("a", "x"): 40, ("a", "y"): 30, ("b", "x"): 29, ("b", "y"): 1, ("z", "z"): 5}
    for (region, age), n in plan.items():
        for _ in range(n):
            rows.append({"split": "train", "region": region, "age_band": age})
    tiny = _tiny_table(rows)
    declared = {("a", "x"): 0.40, ("a", "y"): 0.30, ("b", "x"): 0.29, ("b", "y"): 0.005,
                ("c", "x"): 0.005}
    cov = lesson.coverage_report(tiny, declared=declared, split="train")
    s = {k: v["status"] for k, v in cov["strata"].items()}
    assert s[("a", "x")] == "ok", "40 of 105 against a declared 0.40 is within the factor"
    assert s[("b", "x")] == "thin", "29 rows is under MIN_SUPPORT, so thin"
    assert s[("b", "y")] == "thin", (
        "1 row of a stratum declared at 0.005 is BOTH thin and over-represented. Thin wins: "
        "a stratum that cannot support a claim is not evidence of over-representation either"
    )
    assert s[("c", "x")] == "empty", "declared, zero rows"
    assert s[("z", "z")] == "undeclared", (
        "undeclared outranks thin — 5 rows of a stratum nobody declared is a finding about "
        "the declaration, not about the sample size"
    )
    assert cov["strata"][("z", "z")]["declared_share"] == 0.0
    # The over-representation boundary itself. 75 of 200 rows is 0.375, and a declared 0.25
    # times a factor of 1.5 is 0.375 EXACTLY in binary floating point, so this table sits on
    # the line rather than near it. The rule is a strict inequality: landing exactly on your
    # declared allowance is honouring it, not exceeding it.
    edge_rows = []
    for (region, age), n in {("a", "x"): 75, ("b", "x"): 125}.items():
        for _ in range(n):
            edge_rows.append({"split": "train", "region": region, "age_band": age})
    edge = lesson.coverage_report(_tiny_table(edge_rows), split="train",
                                  declared={("a", "x"): 0.25, ("b", "x"): 0.625})
    assert edge["strata"][("a", "x")]["observed_share"] == 0.25 * 1.5, (
        "this fixture is built so the observed share equals declared * factor exactly"
    )
    assert edge["strata"][("a", "x")]["status"] == "ok", (
        "a stratum sitting exactly on declared_share * OVER_REPRESENTATION_FACTOR has not "
        "exceeded it. The comparison is `>`, not `>=`; with `>=` a split that hits its "
        "allowance to the row is reported as over-represented"
    )
    assert edge["verdict"] == "pass"


def test_coverage_passes_a_representative_split():
    rows = []
    for (region, age), n in {("a", "x"): 50, ("a", "y"): 50}.items():
        for _ in range(n):
            rows.append({"split": "train", "region": region, "age_band": age})
    tiny = _tiny_table(rows)
    cov = lesson.coverage_report(tiny, declared={("a", "x"): 0.5, ("a", "y"): 0.5},
                                 split="train")
    assert cov["verdict"] == "pass", (
        f"this split matches its declaration exactly; yours reported {cov['counts']}"
    )
    assert cov["counts"]["ok"] == 2
    # ...and the verdict has to be sensitive to EVERY status, not only the dramatic two.
    # A verdict that fires on empty and undeclared alone passes both tables below, which is
    # how an over-represented or unexaminable split reaches an inspector marked green.
    def _split_of(plan, declared):
        rows = []
        for (region, age), n in plan.items():
            for _ in range(n):
                rows.append({"split": "train", "region": region, "age_band": age})
        return lesson.coverage_report(_tiny_table(rows), declared=declared, split="train")

    over_only = _split_of({("a", "x"): 60, ("b", "x"): 40},
                          {("a", "x"): 0.30, ("b", "x"): 0.40})
    assert over_only["counts"] == {"ok": 1, "thin": 0, "empty": 0, "over": 1, "undeclared": 0}
    assert over_only["verdict"] == "fail", (
        "no stratum here is empty, thin or undeclared — a/x is simply twice the share it was "
        "declared at. The verdict passes only when EVERY stratum is 'ok', so this one fails"
    )
    thin_only = _split_of({("a", "x"): 20, ("b", "x"): 80},
                          {("a", "x"): 0.20, ("b", "x"): 0.80})
    assert thin_only["counts"]["thin"] == 1 and thin_only["counts"]["ok"] == 1
    assert thin_only["verdict"] == "fail", (
        "a/x holds 20 rows against a floor of 30: the split cannot support a claim about it, "
        "and a report that calls that 'pass' is the green pack this module exists to stop"
    )


def test_coverage_power_is_the_share_floor():
    cov = lesson.coverage_report(TABLE)
    p = cov["power"]
    assert p["statistic"] == "min_detectable_share", f"named wrongly: {p['statistic']!r}"
    want = 1.0 - 0.05 ** (1.0 / len(_rows_of("train")))
    assert abs(p["value"] - want) < 1e-12, (
        f"power value is {p['value']!r}, expected {want!r} — the floor for THIS split's n, "
        "not for the whole table"
    )
    assert isinstance(p["statement"], str) and len(p["statement"]) >= 40


def test_gap_report_lists_every_defect_with_an_action():
    cov = lesson.coverage_report(TABLE)
    gaps = lesson.gap_report(cov)
    want = {k for k, v in _expected_coverage_status("train").items() if v != "ok"}
    assert {g["stratum"] for g in gaps} == want, (
        f"the gap report covers every non-ok stratum ({sorted(want)}); yours covered "
        f"{sorted(g['stratum'] for g in gaps)}. 'over' and 'undeclared' are gaps too"
    )
    for g in gaps:
        assert set(g) == {"stratum", "status", "count", "declared_share", "shortfall_rows",
                          "action"}, f"row keys are wrong: {sorted(g)}"
        assert isinstance(g["action"], str) and g["action"].strip(), (
            "Article 10(2)(h) asks for the gaps AND how they can be addressed"
        )
        assert g["action"] == lesson.REMEDIES[g["status"]], (
            f"the action for {g['status']!r} comes from REMEDIES"
        )


def test_gap_report_order_and_shortfall():
    cov = lesson.coverage_report(TABLE)
    gaps = lesson.gap_report(cov)
    keys = [(-g["declared_share"], g["stratum"]) for g in gaps]
    assert keys == sorted(keys), (
        f"gaps are ordered by declared share descending, ties by stratum; you produced "
        f"{[g['stratum'] for g in gaps]}"
    )
    n = cov["n"]
    for g in gaps:
        if g["status"] in ("empty", "thin"):
            want = math.ceil(g["declared_share"] * n) - g["count"]
            assert g["shortfall_rows"] == want, (
                f"{g['stratum']}: shortfall is ceil(declared_share * n) - count = {want}, "
                f"you gave {g['shortfall_rows']}"
            )
        else:
            assert g["shortfall_rows"] == 0, (
                f"{g['stratum']} is {g['status']}; it is not short of rows, so 0 — and never "
                "a negative number"
            )
    assert all(g["shortfall_rows"] >= 0 for g in gaps), "a shortfall is never negative"
    # Every declared share in the fixture lands on a whole number of rows, so ceil and round
    # agree there. This one does not: 0.041 of 100 rows is 4.1 applicants, and you cannot
    # close a gap with four-tenths of a person.
    rows = [{"split": "train", "region": "a", "age_band": "x"} for _ in range(100)]
    tiny = lesson.coverage_report(_tiny_table(rows), split="train",
                                  declared={("a", "x"): 0.959, ("c", "x"): 0.041})
    short = {g["stratum"]: g["shortfall_rows"] for g in lesson.gap_report(tiny)}
    assert short[("c", "x")] == 5, (
        f"an empty stratum declared at 0.041 of 100 rows is ceil(4.1) = 5 rows short, not "
        f"{short[('c', 'x')]}. round() gives 4 and leaves the stratum below its declaration"
    )


def test_gap_report_breaks_ties_by_stratum():
    rows = []
    for (region, age), n in {("a", "x"): 1, ("b", "x"): 1, ("c", "x"): 60}.items():
        for _ in range(n):
            rows.append({"split": "train", "region": region, "age_band": age})
    tiny = _tiny_table(rows)
    declared = {("a", "x"): 0.1, ("b", "x"): 0.1, ("c", "x"): 0.8}
    gaps = lesson.gap_report(lesson.coverage_report(tiny, declared=declared, split="train"))
    assert [g["stratum"] for g in gaps][:2] == [("a", "x"), ("b", "x")], (
        f"two gaps declared at the same share are ordered by the stratum tuple; you produced "
        f"{[g['stratum'] for g in gaps]}"
    )


# --- exercise 4: drift -------------------------------------------------------------------

def test_drift_statistics_match_an_independent_derivation():
    rep = lesson.drift_report(TABLE)
    tr, te = _rows_of("train"), _rows_of("test")
    assert rep["n_train"] == len(tr) and rep["n_test"] == len(te)
    assert set(rep["features"]) == set(lesson.FEATURE_KINDS), "report every declared feature"
    for name, kind in lesson.FEATURE_KINDS.items():
        col = _col(name)
        if kind == "categorical":
            cats = {str(col[i]) for i in tr} | {str(col[i]) for i in te}
            want = 0.5 * sum(
                abs(sum(1 for i in tr if str(col[i]) == c) / len(tr)
                    - sum(1 for i in te if str(col[i]) == c) / len(te)) for c in cats)
        else:
            a = [float(col[i]) for i in tr]
            b = [float(col[i]) for i in te]
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
            vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
            pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
            want = (mb - ma) / pooled
        got = rep["features"][name]["statistic"]
        assert abs(got - want) < 1e-9, (
            f"{name} ({kind}): statistic is {got!r}, expected {want!r}. For a categorical "
            "feature that is HALF the summed absolute share difference; for a numeric one it "
            "is the mean difference over the POOLED sd with ddof=1"
        )
        assert rep["features"][name]["kind"] == kind


def test_drift_uses_the_union_of_categories():
    """A category that exists only in training contributes its whole share to the distance."""
    rows = [{"split": "train", "region": "a", "income": 1.0} for _ in range(90)]
    rows += [{"split": "train", "region": "b", "income": 1.0} for _ in range(10)]
    rows += [{"split": "test", "region": "a", "income": 1.0} for _ in range(100)]
    tiny = _tiny_table(rows)
    rep = lesson.drift_report(tiny, kinds={"region": "categorical"})
    assert abs(rep["features"]["region"]["statistic"] - 0.10) < 1e-12, (
        f"region's TVD is {rep['features']['region']['statistic']!r}, expected 0.10: the test "
        "split has none of category 'b', and walking only the test split's categories scores "
        "that zero"
    )


def test_drift_flags_only_what_moved_and_reports_power():
    rep = lesson.drift_report(TABLE)
    want = sorted(name for name, f in rep["features"].items() if f["drifted"])
    assert rep["drifted"] == want, "the drifted list is the flagged features, sorted"
    for name, f in rep["features"].items():
        stat = f["statistic"]
        expect = (stat >= lesson.TVD_THRESHOLD if f["kind"] == "categorical"
                  else abs(stat) >= lesson.SMD_THRESHOLD)
        assert f["drifted"] is expect, (
            f"{name}: statistic {stat:.4f} against its threshold says drifted={expect}, you "
            "said the opposite. A numeric feature is compared on the ABSOLUTE value; a "
            "categorical one is already non-negative"
        )
    assert rep["verdict"] == ("fail" if rep["drifted"] else "pass")
    # Every numeric feature in the fixture that moved, moved UP. A feature that collapses is
    # the same defect with a minus sign, and `stat >= SMD_THRESHOLD` sails straight past it.
    rows = [{"split": "train", "region": "a", "income": float(x)} for x in range(100)]
    rows += [{"split": "test", "region": "a", "income": float(x) - 40} for x in range(100)]
    down = lesson.drift_report(_tiny_table(rows), kinds={"income": "numeric"})
    assert down["features"]["income"]["statistic"] < -lesson.SMD_THRESHOLD
    assert down["features"]["income"]["drifted"] is True, (
        "the test split's income sits far BELOW the training split's and this report says it "
        "has not drifted — compare the ABSOLUTE standardised mean difference"
    )
    assert down["drifted"] == ["income"] and down["verdict"] == "fail"
    # Both tables this test has seen so far drift, so "fail" has never had to be earned. A
    # check that cannot return pass is not a check; here the two splits are the same draw.
    steady = [{"split": "train", "region": "ab"[x % 2], "income": float(x)} for x in range(100)]
    steady += [{"split": "test", "region": "ab"[x % 2], "income": float(x)} for x in range(100)]
    calm = lesson.drift_report(_tiny_table(steady),
                               kinds={"region": "categorical", "income": "numeric"})
    assert calm["features"]["region"]["statistic"] == 0.0
    assert abs(calm["features"]["income"]["statistic"]) < 1e-12
    assert calm["drifted"] == [] and calm["verdict"] == "pass", (
        f"the two splits of this table are identical draws — nothing moved — so nothing is "
        f"flagged and the verdict is 'pass'. Yours said {calm['verdict']!r} with "
        f"{calm['drifted']}. A drift report hard-wired to 'fail' is indistinguishable from "
        "no drift report at all"
    )
    p = rep["power"]
    assert p["statistic"] == "min_detectable_share_difference", f"named {p['statistic']!r}"
    want_v = _mde(rep["n_train"], rep["n_test"], 0.5)
    assert abs(p["value"] - want_v) < 1e-12, (
        f"power value is {p['value']!r}, expected min_detectable_difference(n_train, n_test, "
        f"0.5) = {want_v!r}; p = 0.5 is the worst case, so it bounds every category"
    )
    assert isinstance(p["statement"], str) and len(p["statement"]) >= 40


# --- exercise 5: provenance ---------------------------------------------------------------

def test_provenance_tests_licence_membership_not_truthiness():
    rep = lesson.provenance_report(TABLE)
    assert set(rep["splits"]) == set(str(s) for s in set(_col("split")))
    for name, cell in rep["splits"].items():
        want = lesson.SOURCES[name]["licence"] in lesson.ALLOWED_LICENCES
        assert cell["licence_ok"] is want, (
            f"{name}: licence {cell['licence']!r} — licence_ok must be membership of "
            "ALLOWED_LICENCES. `if licence:` passes 'proprietary-internal' and 'TBD' alike"
        )
    assert rep["splits"]["val"]["licence_ok"] is False, (
        "val's licence is a non-empty string that is not on the allow-list"
    )
    assert rep["splits"]["val"]["status"] == "fail"
    assert rep["verdict"] == "fail"


def test_provenance_counts_labels_per_split():
    rep = lesson.provenance_report(TABLE)
    sources = _col("label_source")
    for name, cell in rep["splits"].items():
        rows = _rows_of(name)
        undoc = sum(1 for i in rows if not str(sources[i]).strip())
        unapp = sum(1 for i in rows
                    if lesson.LABEL_METHODS.get(str(sources[i])) == "unapproved")
        assert cell["n"] == len(rows), f"{name}: n is that split's rows"
        assert cell["undocumented_labels"] == undoc, (
            f"{name}: {cell['undocumented_labels']} undocumented labels, expected {undoc}. "
            "An empty or whitespace label_source is undocumented; 'repayment_proxy' is not"
        )
        assert cell["unapproved_labels"] == unapp, (
            f"{name}: {cell['unapproved_labels']} unapproved labels, expected {unapp} — the "
            "ones LABEL_METHODS maps to 'unapproved', which is a different defect from a "
            "missing method"
        )
        assert abs(cell["undocumented_share"] - undoc / len(rows)) < 1e-12, (
            f"{name}: the share is over that split's own n, not the whole table's"
        )
        want_fail = (not cell["licence_ok"]) or unapp > 0 or \
            undoc / len(rows) > lesson.UNDOCUMENTED_LABEL_THRESHOLD
        assert (cell["status"] == "fail") is want_fail, f"{name}: status {cell['status']!r}"


def test_provenance_separates_its_reasons_and_handles_a_missing_source():
    rep = lesson.provenance_report(TABLE)
    train, test = rep["splits"]["train"], rep["splits"]["test"]
    assert train["status"] == "ok" and train["reasons"] == [], (
        f"train is clean — allowed licence, 1% undocumented, no unapproved method — so it "
        f"must pass. Yours failed for {train['reasons']}"
    )
    assert len(test["reasons"]) == 2, (
        f"test fails on the undocumented share AND on the unapproved labels: two reasons, "
        f"not one. You gave {test['reasons']}"
    )
    copy = {k: v.copy() for k, v in TABLE.items()}
    orphan = lesson.provenance_report(copy, sources={"train": lesson.SOURCES["train"]})
    assert orphan["splits"]["test"]["status"] == "fail", (
        "a split with no source record at all must fail, not raise KeyError"
    )
    assert orphan["splits"]["test"]["reasons"], "and it must say why"
    assert orphan["splits"]["test"]["licence"] == "", (
        "a missing source record yields an empty licence string, not None"
    )
    # Every split this rubric has handed the check so far fails on something, so "pass" has
    # never been reached. A provenance report that cannot come back clean says nothing.
    clean = _tiny_table([{"split": "train", "label_source": "adjudicated"} for _ in range(40)])
    good = lesson.provenance_report(clean, sources={"train": {"source_id": "s",
                                                             "licence": "CC-BY-4.0"}})
    assert good["splits"]["train"]["reasons"] == [] and good["splits"]["train"]["status"] == "ok"
    assert good["verdict"] == "pass", (
        f"this split has an allowed licence, a source record and a documented method on every "
        f"one of its 40 labels. There is nothing left to fail on, yet you reported "
        f"{good['verdict']!r} — a verdict that is always 'fail' carries no information"
    )


def test_provenance_power_is_a_census_not_a_sample():
    rep = lesson.provenance_report(TABLE)
    p = rep["power"]
    assert p["statistic"] == "smallest_detectable_share", f"named {p['statistic']!r}"
    assert abs(p["value"] - 1.0 / N_ROWS) < 1e-15, (
        f"power value is {p['value']!r}, expected 1/{N_ROWS} = {1.0 / N_ROWS!r}. This check "
        "reads every row, so its detection limit is one row — not a sampling floor"
    )
    assert isinstance(p["statement"], str) and len(p["statement"]) >= 40


# --- exercise 6: the bias probe ------------------------------------------------------------

def test_bias_probe_rates_match_an_independent_derivation():
    sizes, errors, baseline = _expected_subgroups()
    rep = lesson.bias_probe(TABLE)
    assert rep["n"] == sum(sizes.values())
    assert abs(rep["baseline_error_rate"] - baseline) < 1e-12, (
        f"baseline is {rep['baseline_error_rate']!r}, expected {baseline!r} — a prediction is "
        "score >= DECISION_THRESHOLD and an error is a prediction that differs from the label"
    )
    assert set(rep["subgroups"]) == set(sizes), "one entry per observed (region, age_band)"
    for key, cell in rep["subgroups"].items():
        assert cell["n"] == sizes[key] and cell["errors"] == errors[key], f"{key}"
        assert abs(cell["error_rate"] - errors[key] / sizes[key]) < 1e-12, f"{key} rate"
        assert abs(cell["gap"] - (errors[key] / sizes[key] - baseline)) < 1e-12, (
            f"{key}: gap is the SIGNED difference from the baseline, not its absolute value"
        )
        want_mde = _mde(sizes[key], rep["n"] - sizes[key], baseline)
        assert abs(cell["min_detectable_gap"] - want_mde) < 1e-12, (
            f"{key}: min_detectable_gap compares this subgroup against THE REST of the split "
            f"({rep['n'] - sizes[key]} rows), at the baseline rate; expected {want_mde!r}"
        )


def test_bias_probe_support_floor_comes_first():
    sizes, errors, baseline = _expected_subgroups()
    rep = lesson.bias_probe(TABLE)
    tiny = min(sizes, key=lambda k: sizes[k])
    assert sizes[tiny] < lesson.MIN_SUPPORT
    assert rep["subgroups"][tiny]["status"] == "insufficient_support", (
        f"{tiny} has {sizes[tiny]} rows and the worst error rate in the table. The support "
        "floor is checked BEFORE the gap comparison, so its status is insufficient_support"
    )
    assert rep["subgroups"][tiny]["supported"] is False
    assert tiny not in rep["findings"], (
        "an unsupported subgroup is never a finding, however loud its rate"
    )
    want_insufficient = sorted(k for k in sizes if sizes[k] < lesson.MIN_SUPPORT)
    assert rep["insufficient"] == want_insufficient, (
        f"insufficient lists every subgroup under the floor, sorted: {want_insufficient}. "
        "Dropping them from the report entirely is the other half of the same mistake — a "
        "group too small to examine is itself a coverage finding"
    )
    boundary = [k for k in sizes if sizes[k] == lesson.MIN_SUPPORT]
    assert boundary, "the fixture has a subgroup of exactly MIN_SUPPORT rows"
    for k in boundary:
        assert rep["subgroups"][k]["supported"] is True, (
            f"{k} has exactly MIN_SUPPORT rows. The floor is inclusive: n >= MIN_SUPPORT"
        )


def test_bias_probe_floor_is_the_parameter_not_a_special_case():
    sizes, errors, baseline = _expected_subgroups()
    tiny = min(sizes, key=lambda k: sizes[k])
    rep = lesson.bias_probe(TABLE, min_support=1)
    assert rep["subgroups"][tiny]["status"] != "insufficient_support", (
        f"with min_support=1 nothing is under the floor, yet {tiny} is still reported as "
        "insufficient — the floor must come from the argument, not from a hard-coded size"
    )
    assert rep["insufficient"] == [], "no subgroup is under a floor of 1"
    assert tiny in rep["findings"], (
        f"{tiny} errs on every row; once it clears the floor its gap exceeds what its own "
        "sample could detect, so it becomes a finding. An implementation that filters small "
        "subgroups out of the report altogether never gets here"
    )


def test_bias_probe_findings_are_detectability_based():
    rep = lesson.bias_probe(TABLE)
    sizes, errors, baseline = _expected_subgroups()
    want = sorted((k for k in sizes
                   if sizes[k] >= lesson.MIN_SUPPORT
                   and abs(errors[k] / sizes[k] - baseline)
                   >= _mde(sizes[k], sum(sizes.values()) - sizes[k], baseline)),
                  key=lambda k: (-abs(errors[k] / sizes[k] - baseline), k))
    assert rep["findings"] == want, (
        f"findings are the supported subgroups whose |gap| reaches their OWN "
        f"min_detectable_gap, sorted by |gap| descending: {want}. You reported "
        f"{rep['findings']}. A fixed threshold flags noise in small subgroups and misses "
        "real harm in large ones"
    )
    assert len(want) == 1, "the fixture holds exactly one defensible finding"
    assert rep["verdict"] == "fail"
    for key, cell in rep["subgroups"].items():
        if cell["status"] == "ok":
            assert abs(cell["gap"]) < cell["min_detectable_gap"], (
                f"{key} is called ok while its gap exceeds its detectable floor"
            )
    # Every gap in the fixture is either well inside its floor or well outside it, so a fixed
    # threshold happens to agree with the per-subgroup one there. Here it does not: two
    # supported subgroups sit 7.5 points either side of the baseline, which is more than five
    # points and a long way less than what 40 rows against 40 could actually detect.
    rows = []
    for region, errors in (("a", 11), ("b", 5)):
        for k in range(40):
            rows.append({"split": "test", "region": region, "age_band": "x", "label": 0,
                         "score": 0.9 if k < errors else 0.1})
    marginal = lesson.bias_probe(_tiny_table(rows))
    assert abs(marginal["baseline_error_rate"] - 0.20) < 1e-12
    for key in (("a", "x"), ("b", "x")):
        cell = marginal["subgroups"][key]
        assert cell["supported"] is True and abs(abs(cell["gap"]) - 0.075) < 1e-12
        assert cell["status"] == "ok", (
            f"{key} is {abs(cell['gap']):.3f} from the baseline and this sample could only "
            f"have detected {cell['min_detectable_gap']:.3f}. Calling it a finding publishes "
            "a number the data does not support; a fixed five-point threshold does exactly "
            "that"
        )
    assert marginal["findings"] == [] and marginal["verdict"] == "pass"
    # The fixture holds exactly ONE finding, so its sort order is unobservable there. This
    # table holds two of different size: a/x errs 35 points above a 0.35 baseline, b/x 25
    # points below it. Both clear their own floor; the larger disparity is read first.
    pair = []
    for region, bad in (("a", 70), ("b", 10), ("c", 25)):
        for k in range(100):
            pair.append({"split": "test", "region": region, "age_band": "x", "label": 0,
                         "score": 0.9 if k < bad else 0.1})
    two = lesson.bias_probe(_tiny_table(pair))
    assert abs(two["baseline_error_rate"] - 0.35) < 1e-12
    assert abs(abs(two["subgroups"][("a", "x")]["gap"]) - 0.35) < 1e-12
    assert abs(abs(two["subgroups"][("b", "x")]["gap"]) - 0.25) < 1e-12
    assert two["subgroups"][("c", "x")]["status"] == "ok", "c/x is 0.10 from a floor of 0.16"
    assert two["findings"] == [("a", "x"), ("b", "x")], (
        f"two subgroups clear their own detectable floor here, and findings are ordered by "
        f"ABSOLUTE gap descending — the 35-point disparity before the 25-point one. You "
        f"reported {two['findings']}. Sorting the other way puts the smallest harm at the "
        "top of the report an inspector reads first, and a signed sort buries every "
        "subgroup the system treats too generously"
    )


def test_bias_probe_power_is_the_least_sensitive_comparison_made():
    rep = lesson.bias_probe(TABLE)
    p = rep["power"]
    assert p["statistic"] == "largest_min_detectable_gap", f"named {p['statistic']!r}"
    supported = [c["min_detectable_gap"] for c in rep["subgroups"].values() if c["supported"]]
    unsupported = [c["min_detectable_gap"] for c in rep["subgroups"].values()
                   if not c["supported"]]
    assert abs(p["value"] - max(supported)) < 1e-12, (
        f"power value is {p['value']!r}, expected {max(supported)!r} — the LARGEST detectable "
        "gap among the SUPPORTED subgroups, which is the least sensitive comparison this "
        "probe actually made"
    )
    assert p["value"] < max(unsupported), (
        "taking the maximum over all subgroups reports the sensitivity of a comparison you "
        "declined to make, which overstates how blind the probe was"
    )
    assert isinstance(p["statement"], str) and len(p["statement"]) >= 40


RUBRIC = [
    (test_min_detectable_share_is_the_exact_form, 3,
     "solve (1 - p)**n = 1 - confidence for p; 3/n is the approximation to it"),
    (test_min_detectable_share_honours_confidence_and_edges, 2,
     "use the confidence argument, and return 1.0 for n <= 0"),
    (test_min_detectable_difference_is_the_two_proportion_formula, 4,
     "(z_alpha + z_beta) * sqrt(p*(1-p)*(1/n_a + 1/n_b)), capped at 1.0"),
    (test_min_detectable_difference_uses_power_and_both_groups, 3,
     "both quantiles, both group sizes, and the cap"),
    (test_near_key_rounds_to_recorded_precision, 3,
     "round income to a whole unit as an int, dti to dti_dp places"),
    (test_near_key_is_hashable_and_pure, 2,
     "return a plain tuple and do not write to the table"),
    (test_leakage_overlap_counts, 4,
     "count TEST rows; the near set contains the exact matches"),
    (test_leakage_identifier_and_verdict, 4,
     "absolute correlation between applicant_ref and label, over the whole table"),
    (test_leakage_verdict_passes_a_clean_table, 3,
     "no findings means pass; a check that always fails is not a check"),
    (test_leakage_power_is_the_correlation_floor, 2,
     "tanh(Z_ALPHA_95 / sqrt(n - 3)), and a statement that says what the census cannot see"),
    (test_coverage_reports_the_union_of_strata, 4,
     "walk declared UNION observed, or the undeclared stratum disappears"),
    (test_coverage_statuses_match_an_independent_derivation, 4,
     "undeclared, empty, thin, over, ok — in that order of precedence"),
    (test_coverage_status_precedence_and_thresholds, 4,
     "thin beats over; undeclared beats everything; MIN_SUPPORT is exclusive below"),
    (test_coverage_passes_a_representative_split, 2,
     "a split that matches its declaration must come back pass"),
    (test_coverage_power_is_the_share_floor, 2,
     "min_detectable_share(n) for this split's own n"),
    (test_gap_report_lists_every_defect_with_an_action, 4,
     "every non-ok stratum, with the REMEDIES action for its status"),
    (test_gap_report_order_and_shortfall, 4,
     "ceil(declared_share * n) - count, never negative; sorted by declared share descending"),
    (test_gap_report_breaks_ties_by_stratum, 2,
     "ties in declared share are broken by the stratum tuple"),
    (test_drift_statistics_match_an_independent_derivation, 4,
     "half the summed absolute share difference; pooled sd with ddof=1"),
    (test_drift_uses_the_union_of_categories, 3,
     "a category present in only one split contributes its whole share"),
    (test_drift_flags_only_what_moved_and_reports_power, 4,
     "compare the numeric statistic on its absolute value; power at p = 0.5"),
    (test_provenance_tests_licence_membership_not_truthiness, 4,
     "licence_ok is membership of ALLOWED_LICENCES, not a truthiness test"),
    (test_provenance_counts_labels_per_split, 4,
     "empty label_source is undocumented; an unapproved method is a separate defect"),
    (test_provenance_separates_its_reasons_and_handles_a_missing_source, 3,
     "one reason per defect, and a missing source record must not raise"),
    (test_provenance_power_is_a_census_not_a_sample, 2,
     "1/n over the whole table, with a statement about the real blind spot"),
    (test_bias_probe_rates_match_an_independent_derivation, 4,
     "signed gap; the subgroup against the REST of the split at the baseline rate"),
    (test_bias_probe_support_floor_comes_first, 5,
     "status is decided by support before any gap comparison; the floor is inclusive"),
    (test_bias_probe_floor_is_the_parameter_not_a_special_case, 4,
     "min_support is an argument; never drop a small subgroup from the report"),
    (test_bias_probe_findings_are_detectability_based, 5,
     "flag a gap only when it reaches that subgroup's own detectable floor"),
    (test_bias_probe_power_is_the_least_sensitive_comparison_made, 3,
     "the maximum over SUPPORTED subgroups only"),
]
