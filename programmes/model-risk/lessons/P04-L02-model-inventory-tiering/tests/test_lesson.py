"""Autograder rubric for P04-L02. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson, and are deliberate:

* every record graded here is built HERE. The rubric never reads the notebook's synthetic
  inventory, so a change to the generator cannot silently break grading, and a student cannot
  pass by special-casing `MDL-003`.
* every expected tier is worked out by hand from `POLICY` in the comment above the case, so
  the rubric is not grading the student's arithmetic against a second copy of the student's
  arithmetic.
* the exercises chain (`validate_inventory` calls `validate_record`), so each exercise's own
  properties are graded by cases that do NOT depend on the exercise before it. The one
  exception is `test_inventory_reports_every_failure`, which necessarily needs a working
  exercise 1, and it is the only place that dependence is priced.
* `test_tier_single_attribute_flip` is the centre of this rubric. One base record sits at a
  score of 4 — tier 3 — and each of the five attributes, moved ONE step, takes it to tier 2.
  A tiering that drops an attribute, mis-weights one, or scores an exposure band with `>`
  instead of `>=` passes most other cases and fails this one.
* `test_tier_escalation_is_a_floor` builds the 13-point record that a floor applied as an
  assignment (`decision_tier = 2`) demotes from tier 1 to tier 2 — the mistake that makes an
  aggregate tier distribution improve while nothing about the book has changed.
* `test_validate_bool_is_not_an_int` is what stops `isinstance(v, int)` from scoring full
  marks. `True` IS an int in Python, so an exposure of `True` is one pound.
* `test_coverage_due_boundary` pins the inclusive boundary: a model exactly on its validation
  frequency is CURRENT. `>=` there reports every model as late on the day it is due, which is
  how a coverage report stops being read.
* `test_report_empty_tier_row` requires a row for a tier holding no models, for the same
  reason module 1 keeps its empty calibration bins: an absent row and an empty row are
  different claims.
"""
import importlib.util, os
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)


# --- fixtures built here, so nothing is graded against the notebook's own sample ------------

def _rec(**over) -> dict:
    """A well-formed record, built here so no case depends on the notebook's own sample."""
    base = {
        "model_id": "R-1", "name": "rubric model", "owner": "r.owner", "status": "in_use",
        "purpose": "rubric purpose", "exposure_gbp": 0, "customers": 0,
        "materiality": "low", "complexity": "low", "reversibility": "reversible",
        "human_override": True, "last_validated": "2026-01-16",
        "validation_frequency_months": 12,
    }
    base.update(over)
    return base


def _scorable(**over) -> dict:
    """Only the five fields `tier()` reads, so a tier case cannot lean on exercise 1."""
    base = {"materiality": "low", "complexity": "low", "exposure_gbp": 0,
            "reversibility": "reversible", "human_override": True}
    base.update(over)
    return base


def _expect_value_error(fn, *args, why=""):
    try:
        fn(*args)
    except ValueError:
        return
    except NotImplementedError:
        raise
    except Exception as exc:
        raise AssertionError(f"{why} raised {type(exc).__name__}, not ValueError") from None
    raise AssertionError(f"{why} should raise ValueError")


def _probe() -> dict:
    """A findings dict whose every figure is unmistakable, built here and not in the lesson."""
    return {
        "as_of": "1970-01-01",
        "data_note": "PROBE DATA NOTE, SYNTHETIC",
        "policy_name": "probe policy v9",
        "n_records": 4242,
        "schema_failures": {"BAD-1": ("missing field: owner", "missing field: purpose"),
                            "<record 3>": ("missing field: model_id",)},
        "tiers": {"T1": lesson.TierDecision(1, 13, ("score 13 -> tier 1",)),
                  "T3": lesson.TierDecision(3, 3, ("score 3 -> tier 3",)),
                  "T3b": lesson.TierDecision(3, 2, ("score 2 -> tier 3",))},
        # both and inventory_only are DIFFERENT lengths on purpose: with two equal-length
        # lists, a report that prints them the wrong way round scores full marks.
        "reconciliation": lesson.Reconciliation(
            both=("T1", "T3"), inventory_only=("GONE-7", "GONE-8", "GONE-9"),
            runtime_only=("SHADOW-9",), decommissioned_running=("T3",)),
        "coverage": {"T1": lesson.CoverageRow("T1", None, 12, "never_validated"),
                     "T3": lesson.CoverageRow("T3", 31, 24, "overdue"),
                     "T3b": lesson.CoverageRow("T3b", 2, 36, "current")},
    }


# --- exercise 1: validate_record() ---------------------------------------------------------

def test_validate_accepts_clean_record():
    out = lesson.validate_record(_rec())
    assert out == (), f"a well-formed record has no defects, you reported {out}"
    assert isinstance(out, tuple), f"return a tuple, got {type(out).__name__}"
    assert lesson.validate_record(_rec(last_validated=None)) == (), (
        "last_validated=None is valid: never validated is a coverage finding, not a schema one"
    )
    assert lesson.validate_record(_rec(extra_column="inventories grow columns")) == (), (
        "an unrecognised extra field is not a defect"
    )


def test_validate_reports_every_missing_field():
    out = lesson.validate_record({"model_id": "R-1"})
    n = len(lesson.REQUIRED_FIELDS) - 1
    assert len(out) == n, (
        f"{n} fields are absent and you reported {len(out)} message(s) — collect them all, "
        "do not return at the first"
    )
    assert all(e.startswith("missing field: ") for e in out), (
        f"every message should be 'missing field: <name>', got {out}"
    )
    assert list(out) == sorted(out), "sort the messages before returning them"
    assert set(out) == {f"missing field: {f}" for f in lesson.REQUIRED_FIELDS
                        if f != "model_id"}, f"the named fields are wrong: {out}"


def test_validate_enum_and_text():
    bad = lesson.validate_record(_rec(materiality="critical"))
    assert len(bad) == 1, f"one bad enumeration value, one message; got {bad}"
    assert "materiality" in bad[0] and "critical" in bad[0], (
        f"name the field and the offending value: {bad[0]!r}"
    )
    for field in ("status", "complexity", "reversibility"):
        out = lesson.validate_record(_rec(**{field: "nope"}))
        assert len(out) == 1 and field in out[0], f"{field} is not being checked: {out}"
    for value in ("", "   ", 7, None):
        out = lesson.validate_record(_rec(owner=value))
        assert any("owner" in e for e in out), (
            f"owner={value!r} is not a non-empty string and must be reported; got {out}"
        )
    assert any("name" in e for e in lesson.validate_record(_rec(name=""))), (
        "every one of model_id, name, owner and purpose is a non-empty string"
    )


def test_validate_bool_is_not_an_int():
    for field in ("exposure_gbp", "customers"):
        out = lesson.validate_record(_rec(**{field: True}))
        assert any(field in e for e in out), (
            f"{field}=True passed: isinstance(True, int) is True in Python, so an int check "
            "alone accepts a boolean — exclude bool explicitly"
        )
    for value in (1, 0, "yes", None):
        out = lesson.validate_record(_rec(human_override=value))
        assert any("human_override" in e for e in out), (
            f"human_override={value!r} is not a bool and must be reported; got {out}"
        )
    assert lesson.validate_record(_rec(human_override=False)) == (), (
        "False is a perfectly good human_override"
    )
    assert any("validation_frequency_months" in e
               for e in lesson.validate_record(_rec(validation_frequency_months=True))), (
        "a frequency of True is a boolean, not twelve months"
    )


def test_validate_numeric_and_date():
    for field, value in (("exposure_gbp", -1), ("exposure_gbp", 1.5e8),
                         ("customers", -3), ("customers", 2.0)):
        out = lesson.validate_record(_rec(**{field: value}))
        assert any(field in e for e in out), f"{field}={value!r} must be reported; got {out}"
    assert lesson.validate_record(_rec(exposure_gbp=0, customers=0)) == (), (
        "zero is a valid exposure and a valid customer count; the rule is non-negative"
    )
    for freq in (0, -1):
        assert any("validation_frequency_months" in e
                   for e in lesson.validate_record(_rec(validation_frequency_months=freq))), (
            f"a validation frequency of {freq} is not a frequency; the minimum is 1"
        )
    for date in ("16-01-2026", "2026/01/16", "2026-13-01", 20260116):
        out = lesson.validate_record(_rec(last_validated=date))
        assert any("last_validated" in e for e in out), (
            f"last_validated={date!r} is not a YYYY-MM-DD date; got {out}"
        )


def test_validate_missing_field_not_double_reported():
    r = _rec()
    del r["materiality"]
    del r["exposure_gbp"]
    out = lesson.validate_record(r)
    assert len(out) == 2, (
        f"two absent fields, two messages — do not also type-check a field that is not "
        f"there; got {out}"
    )
    assert set(out) == {"missing field: materiality", "missing field: exposure_gbp"}, out
    # ... and an absent field does not excuse the rest of the record: a validator that
    # returns as soon as it has collected the missing fields hides every other defect.
    mixed = _rec(materiality="critical", exposure_gbp=-1, owner="")
    del mixed["status"]
    got = lesson.validate_record(mixed)
    assert len(got) == 4, (
        f"one absent field and three bad values is four messages, you reported {len(got)}: "
        f"{got} — finding the missing fields is not a reason to stop; keep every check "
        "running and sort once at the end"
    )
    assert any(e == "missing field: status" for e in got) and all(
        f in " ".join(got) for f in ("materiality", "exposure_gbp", "owner")), (
        f"the absent field and each bad value must each be named; got {got}"
    )


# --- exercise 2: validate_inventory() ------------------------------------------------------

def test_inventory_reports_every_failure():
    records = [_rec(model_id="A"), _rec(model_id="B", owner=""),
               _rec(model_id="C", materiality="critical"), _rec(model_id="D")]
    del records[3]["status"]
    out = lesson.validate_inventory(records)
    assert isinstance(out, dict), f"return a dict, got {type(out).__name__}"
    assert set(out) == {"B", "C", "D"}, (
        f"three of four records fail and A passes; you reported {sorted(out)} — the loop must "
        "keep going after the first failure, and clean records must not appear"
    )
    assert out["D"] == ("missing field: status",), f"got {out['D']} for record D"
    assert lesson.validate_inventory([_rec()]) == {}, "a clean extract returns an empty dict"


def test_inventory_duplicate_ids():
    out = lesson.validate_inventory([_rec(model_id="X"), _rec(model_id="Y"),
                                     _rec(model_id="X")])
    assert set(out) == {"X"}, (
        f"only X is duplicated; Y is clean and must not appear. You reported {sorted(out)}"
    )
    assert out["X"] == ("duplicate model_id: X",), (
        f"both X records are otherwise perfect, so the only message is the duplicate; got "
        f"{out['X']}"
    )
    mixed = lesson.validate_inventory([_rec(model_id="X", owner=""), _rec(model_id="X")])
    assert len(mixed["X"]) >= 1 and any("duplicate" in e for e in mixed["X"]), (
        "the duplicate message attaches to every record carrying the id"
    )
    assert list(mixed["X"]) == sorted(mixed["X"]), (
        "sort each record's messages AFTER appending the duplicate message"
    )


def test_inventory_keys_and_order():
    nameless = _rec()
    del nameless["model_id"]
    out = lesson.validate_inventory([_rec(model_id="Z"), nameless, _rec(model_id="", owner="")])
    assert "<record 1>" in out, (
        f"a record with no usable model_id is keyed '<record 1>' by its position; got "
        f"{list(out)}"
    )
    assert "<record 2>" in out, "an empty-string model_id is not usable as a key either"
    assert list(out) == ["<record 1>", "<record 2>"], (
        f"Z is clean and absent, and the keys follow the records; got {list(out)}"
    )


def test_inventory_rejects_non_sequence():
    for shape in ({"A": _rec()}, (r for r in [_rec()])):
        try:
            lesson.validate_inventory(shape)
        except TypeError:
            continue
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(
                f"{type(shape).__name__} raised {type(exc).__name__}, not TypeError"
            ) from None
        raise AssertionError(
            f"a {type(shape).__name__} of records should raise TypeError — iterating a dict "
            "walks its keys and reports a clean bill of health it has not earned"
        )
    assert isinstance(lesson.validate_inventory(tuple([_rec(owner="")])), dict), (
        "a tuple of records is a perfectly good sequence and must be accepted"
    )


# --- exercise 3: tier() --------------------------------------------------------------------

def test_tier_points_table():
    # low/low/0/reversible/override -> 0+0+0+0+0 = 0 -> below every cut -> tier 4
    zero = lesson.tier(_scorable(), lesson.POLICY)
    assert isinstance(zero, lesson.TierDecision), (
        f"return a TierDecision, got {type(zero).__name__}"
    )
    assert zero.score == 0 and zero.tier == 4, (
        f"the harmless record scores 0 and falls to the default tier 4; got score "
        f"{zero.score}, tier {zero.tier}"
    )
    # high/high/2e9/irreversible/no override -> 4+2+3+3+1 = 13
    worst = lesson.tier(_scorable(materiality="high", complexity="high",
                                  exposure_gbp=2_000_000_000, reversibility="irreversible",
                                  human_override=False), lesson.POLICY)
    assert worst.score == 13, f"4+2+3+3+1 = 13, you got {worst.score}"
    assert worst.tier == 1, f"13 clears the cut of 8; got tier {worst.tier}"
    # each attribute alone, against the points table
    for over, expected in ((dict(materiality="medium"), 2), (dict(materiality="high"), 4),
                           (dict(complexity="medium"), 1), (dict(complexity="high"), 2),
                           (dict(reversibility="recoverable"), 1),
                           (dict(reversibility="irreversible"), 3),
                           (dict(human_override=False), 1)):
        got = lesson.tier(_scorable(**over), lesson.POLICY).score
        assert got == expected, (
            f"{over} alone is worth {expected} point(s) under POLICY, you scored {got}"
        )


def test_tier_exposure_band_boundary():
    for exposure, expected in ((0, 0), (9_999_999, 0), (10_000_000, 1), (99_999_999, 1),
                               (100_000_000, 2), (999_999_999, 2), (1_000_000_000, 3),
                               (4_100_000_000, 3)):
        got = lesson.tier(_scorable(exposure_gbp=exposure), lesson.POLICY).score
        assert got == expected, (
            f"an exposure of {exposure:,} scores {expected} under POLICY's bands, you scored "
            f"{got} — a model REACHES a band by landing exactly on its threshold, so the "
            "comparison is >=, not >"
        )


def test_tier_cut_boundaries():
    # (score built from the points table, the tier POLICY's cuts give it)
    cases = (
        (_scorable(materiality="high", complexity="high", exposure_gbp=100_000_000), 8, 1),
        (_scorable(materiality="high", complexity="medium", exposure_gbp=100_000_000), 7, 2),
        (_scorable(materiality="high"), 4, 3),
        (_scorable(materiality="medium", complexity="medium",
                   exposure_gbp=100_000_000), 5, 2),
        (_scorable(complexity="medium", exposure_gbp=10_000_000), 2, 3),
        (_scorable(complexity="medium"), 1, 4),
    )
    for record, score, want in cases:
        got = lesson.tier(record, lesson.POLICY)
        assert got.score == score, f"{record} scores {score}, you got {got.score}"
        assert got.tier == want, (
            f"a score of {score} earns tier {want} under POLICY's cuts {lesson.POLICY['tier_cuts']} "
            f"— landing exactly ON a cut passes it; you gave tier {got.tier}"
        )


def test_tier_single_attribute_flip():
    # medium/medium/10,000,000/reversible/override -> 2+1+1+0+0 = 4 -> tier 3
    base = _scorable(materiality="medium", complexity="medium", exposure_gbp=10_000_000)
    start = lesson.tier(base, lesson.POLICY)
    assert start.score == 4 and start.tier == 3, (
        f"the boundary record scores 2+1+1+0+0 = 4 and sits in tier 3; you got score "
        f"{start.score}, tier {start.tier}"
    )
    # (field, one step up, the score POLICY gives it) — each step clears the cut of 5
    one_step = (
        ("materiality", "high", 6), ("complexity", "high", 5),
        ("exposure_gbp", 100_000_000, 5), ("reversibility", "recoverable", 5),
        ("human_override", False, 5),
    )
    for field, value, score in one_step:
        moved_record = dict(base)
        moved_record[field] = value
        moved = lesson.tier(moved_record, lesson.POLICY)
        assert moved.score == score, (
            f"moving {field} one step to {value!r} takes the score from 4 to {score} under "
            f"POLICY; you scored {moved.score}"
        )
        assert moved.tier == 2, (
            f"and 5 is the cut for tier 2, so {field}={value!r} ALONE moves this model from "
            f"tier 3 to tier 2; you gave tier {moved.tier}"
        )
    # and one step the other way, on the one attribute that can go down, stays in tier 3
    down = dict(base)
    down["complexity"] = "low"
    assert lesson.tier(down, lesson.POLICY).tier == 3, (
        "3 is short of the cut of 5 and the record stays in tier 3"
    )


def test_tier_escalation_is_a_floor():
    # low/low/0/irreversible/no override -> 0+0+0+3+1 = 4 -> tier 3, floored to 2
    quiet = lesson.tier(_scorable(reversibility="irreversible", human_override=False),
                        lesson.POLICY)
    assert quiet.score == 4, f"0+0+0+3+1 = 4, you got {quiet.score}"
    assert quiet.tier == 2, (
        f"an irreversible decision no human can override is floored at tier "
        f"{lesson.POLICY['escalation_tier_at_most']} whatever it scored; you gave "
        f"{quiet.tier}"
    )
    worst = lesson.tier(_scorable(materiality="high", complexity="high",
                                  exposure_gbp=2_000_000_000, reversibility="irreversible",
                                  human_override=False), lesson.POLICY)
    assert worst.tier == 1, (
        "the escalation is a FLOOR, applied with min() — a tier 1 model must not be demoted "
        f"to tier {lesson.POLICY['escalation_tier_at_most']} by it, which would make the "
        "aggregate tier distribution improve while nothing changed. You gave tier "
        f"{worst.tier}"
    )
    for over in (dict(reversibility="irreversible", human_override=True),
                 dict(reversibility="recoverable", human_override=False)):
        got = lesson.tier(_scorable(**over), lesson.POLICY)
        assert got.tier != 2 or got.score >= 5, (
            f"{over} meets only half the escalation condition and must not trigger it; you "
            f"gave tier {got.tier} for score {got.score}"
        )


def test_tier_audit_trail():
    plain = lesson.tier(_scorable(materiality="high", complexity="medium",
                                  exposure_gbp=250_000_000,
                                  reversibility="recoverable"), lesson.POLICY)
    assert plain.reasons == (
        "materiality=high +4",
        "complexity=medium +1",
        "exposure=250000000 +2",
        "reversibility=recoverable +1",
        "human_override=True +0",
        "score 8 -> tier 1",
    ), f"the audit trail is wrong:\n  {plain.reasons}"
    escalated = lesson.tier(_scorable(reversibility="irreversible", human_override=False),
                            lesson.POLICY)
    assert len(escalated.reasons) == 7, (
        f"the escalation adds a seventh line; you produced {len(escalated.reasons)}"
    )
    assert escalated.reasons[5] == "score 4 -> tier 3", (
        f"the cut line records the tier BEFORE the escalation, so the trail shows the "
        f"override happening; got {escalated.reasons[5]!r}"
    )
    assert escalated.reasons[6] == (
        "escalation: irreversible with no human override -> tier 2"
    ), f"got {escalated.reasons[6]!r}"
    unmoved = lesson.tier(_scorable(materiality="high", complexity="high",
                                    exposure_gbp=2_000_000_000,
                                    reversibility="irreversible", human_override=False),
                          lesson.POLICY)
    assert unmoved.reasons[6] == (
        "escalation: irreversible with no human override -> tier 1"
    ), (
        "the escalation line is recorded whenever the condition holds, even when it moves "
        f"nothing — an audit trail records the rules that ran. Got {unmoved.reasons[6]!r}"
    )


def test_tier_rejects_unknown_values():
    for over, why in ((dict(materiality="critical"), "an unrecognised materiality"),
                      (dict(complexity="extreme"), "an unrecognised complexity"),
                      (dict(reversibility="maybe"), "an unrecognised reversibility"),
                      (dict(human_override=1), "human_override as an int"),
                      (dict(exposure_gbp=-1), "a negative exposure")):
        _expect_value_error(lesson.tier, _scorable(**over), lesson.POLICY, why=why)


# --- exercise 4: reconcile() ---------------------------------------------------------------

def test_reconcile_four_lists():
    records = [{"model_id": "A", "status": "in_use"},
               {"model_id": "B", "status": "decommissioned"},
               {"model_id": "C", "status": "in_development"}]
    r = lesson.reconcile(records, ["A", "B", "D"])
    assert isinstance(r, lesson.Reconciliation), (
        f"return a Reconciliation, got {type(r).__name__}"
    )
    assert r.both == ("A", "B"), f"A and B are known and running; got {r.both}"
    assert r.inventory_only == ("C",), f"C is known and not running; got {r.inventory_only}"
    assert r.runtime_only == ("D",), (
        f"D is running and in no record — this is the list the exercise exists for; got "
        f"{r.runtime_only}"
    )
    assert r.decommissioned_running == ("B",), (
        f"B is recorded as decommissioned and is still running; got {r.decommissioned_running}"
    )
    assert set(r.decommissioned_running) <= set(r.both), (
        "decommissioned_running is a subset of both, never of runtime_only"
    )


def test_reconcile_dedup_and_sort():
    records = [{"model_id": "zeta", "status": "in_use"},
               {"model_id": "alpha", "status": "in_use"},
               {"model_id": "mid", "status": "in_use"}]
    r = lesson.reconcile(records, ["zeta", "zeta", "omega", "omega", "beta"])
    assert r.both == ("zeta",), f"a repeated runtime id appears once; got {r.both}"
    assert r.runtime_only == ("beta", "omega"), (
        f"sorted and de-duplicated; got {r.runtime_only}"
    )
    assert r.inventory_only == ("alpha", "mid"), (
        f"sorted ascending, not in record order; got {r.inventory_only}"
    )
    for field in r._fields:
        values = getattr(r, field)
        assert isinstance(values, tuple), f"{field} is a tuple, got {type(values).__name__}"
        assert len(set(values)) == len(values), f"{field} repeats an identifier: {values}"


def test_reconcile_first_record_status_wins():
    r = lesson.reconcile([{"model_id": "A", "status": "in_use"},
                          {"model_id": "A", "status": "decommissioned"}], ["A"])
    assert r.both == ("A",), f"A appears once however many records carry the id; got {r.both}"
    assert r.decommissioned_running == (), (
        "the FIRST record's status wins — duplicates are exercise 2's finding, and raising "
        f"here would stop the reconciliation its owner needs. Got {r.decommissioned_running}"
    )
    ok = lesson.reconcile([{"model_id": "A"}], ["A"])
    assert ok.both == ("A",) and ok.decommissioned_running == (), (
        "a record with no status reconciles fine; its absence is exercise 1's finding"
    )


def test_reconcile_rejects_nameless_record():
    for records, why in (([{"model_id": "A"}, {"name": "no id"}], "a record with no model_id"),
                         ([{"model_id": ""}], "a record with an empty model_id"),
                         ([{"model_id": 7}], "a record whose model_id is not a string")):
        _expect_value_error(lesson.reconcile, records, [], why=why)


# --- exercise 5: validation_coverage() -----------------------------------------------------

def test_coverage_only_in_use():
    records = [_rec(model_id="U", status="in_use"),
               _rec(model_id="D", status="decommissioned", last_validated=None),
               _rec(model_id="N", status="in_development", last_validated=None)]
    rows = lesson.validation_coverage(records, "2026-09-16")
    assert set(rows) == {"U"}, (
        f"only in-use models are a coverage gap; you reported {sorted(rows)} — a "
        "decommissioned model with no validation on file is not a finding"
    )
    assert isinstance(rows["U"], lesson.CoverageRow), (
        f"values are CoverageRow, got {type(rows['U']).__name__}"
    )
    assert rows["U"].model_id == "U" and rows["U"].due_months == 12, (
        f"the row carries the model's own id and frequency; got {rows['U']}"
    )
    assert list(lesson.validation_coverage(
        [_rec(model_id="z"), _rec(model_id="a")], "2026-09-16")) == ["z", "a"], (
        "insertion order follows the records; do not sort"
    )


def test_coverage_months_arithmetic():
    cases = (("2026-03-16", "2026-09-16", 6), ("2026-03-16", "2026-09-15", 5),
             ("2026-03-16", "2026-09-17", 6), ("2025-12-31", "2026-09-16", 8),
             ("2023-09-16", "2026-09-16", 36), ("2026-09-16", "2026-09-16", 0),
             ("2026-01-01", "2026-09-16", 8))
    for validated, as_of, expected in cases:
        row = lesson.validation_coverage(
            [_rec(last_validated=validated, validation_frequency_months=120)], as_of)["R-1"]
        assert row.months_since == expected, (
            f"{validated} to {as_of} is {expected} whole months, you computed "
            f"{row.months_since} — subtract one when the day of the month has not come round"
        )


def test_coverage_due_boundary():
    for months_gap, due, want in ((6, 6, "current"), (7, 6, "overdue"), (5, 6, "current"),
                                  (12, 12, "current"), (13, 12, "overdue"),
                                  (0, 1, "current"), (2, 1, "overdue")):
        # validated exactly `months_gap` whole months before 2027-01-16
        validated = lesson.shift_months(lesson.parse_date("2027-01-16"), -months_gap)
        row = lesson.validation_coverage(
            [_rec(last_validated=validated.isoformat(),
                  validation_frequency_months=due)], "2027-01-16")["R-1"]
        assert row.months_since == months_gap, (
            f"the fixture is {months_gap} months old, you computed {row.months_since}"
        )
        assert row.verdict == want, (
            f"{months_gap} months against a {due}-month frequency is {want!r}, you said "
            f"{row.verdict!r} — landing exactly ON the frequency is CURRENT: the validation "
            "is due, not late"
        )


def test_coverage_never_validated():
    row = lesson.validation_coverage([_rec(last_validated=None)], "2026-09-16")["R-1"]
    assert row.verdict == "never_validated", (
        f"a model with no validation date reports 'never_validated', got {row.verdict!r}"
    )
    assert row.months_since is None, (
        f"months_since is None, not 0 and not a count from some epoch; got {row.months_since}"
    )
    assert row.due_months == 12, "the frequency is still reported for a never-validated model"


def test_coverage_future_date():
    try:
        lesson.validation_coverage([_rec(model_id="FUT", last_validated="2027-01-01")],
                                   "2026-09-16")
    except ValueError as exc:
        assert "FUT" in str(exc), f"name the model in the message, got: {exc}"
        return
    except NotImplementedError:
        raise
    except Exception as exc:
        raise AssertionError(
            f"a future validation date raised {type(exc).__name__}, not ValueError"
        ) from None
    raise AssertionError("a last_validated after the as-of date should raise ValueError")


# --- exercise 6: render_inventory_report() -------------------------------------------------

def test_report_structure():
    page = lesson.render_inventory_report(_probe())
    assert isinstance(page, str), f"return one string, got {type(page).__name__}"
    lines = page.splitlines()
    assert lines[0] == "# MODEL INVENTORY AND TIERING REPORT — 1970-01-01", (
        f"the first line is the title and the as-of date; got {lines[0]!r}"
    )
    at = []
    for heading in lesson.REPORT_SECTIONS:
        matches = [i for i, l in enumerate(lines) if l == f"## {heading}"]
        assert matches, f"the heading '## {heading}' is missing from the page"
        at.append(matches[0])
    assert at == sorted(at), (
        f"the six sections must appear in REPORT_SECTIONS order; you produced {at}"
    )


def test_report_uses_findings_only():
    lines = lesson.render_inventory_report(_probe()).splitlines()
    assert "  records: 4242" in lines, (
        "section 1 reports n_records straight from findings — recompute nothing"
    )
    assert "  policy: probe policy v9" in lines, "name the policy the tiers were cut under"
    assert "PROBE DATA NOTE, SYNTHETIC" in lines, (
        "the data note goes into the page verbatim, on its own line: a reader of the artefact "
        "must be able to see the figures are synthetic without opening the notebook"
    )
    assert "  records failing the schema: 2" in lines, (
        "section 2 counts the schema_failures dict it was handed"
    )
    assert "  - BAD-1: 2 defect(s)" in lines, (
        "one line per failing record, carrying that record's defect count"
    )
    assert "  - <record 3>: 1 defect(s)" in lines, (
        "a record keyed by position is reported under that key, unchanged"
    )
    assert "  in inventory and running: 2" in lines, "section 4 counts the reconciliation"
    assert "  in inventory, not running: 3" in lines, (
        "both counts come from findings, and they are not interchangeable: `both` is 2 and "
        "`inventory_only` is 3 in this probe"
    )
    # section 5: one line per coverage row, carrying the months and the frequency. A report
    # that prints the verdict alone has dropped the two figures the verdict is derived from.
    assert "  T1: never_validated (never validated, due every 12 months)" in lines, (
        "section 5 spells out a model with no validation date, and still reports the "
        "frequency it is supposed to be validated at"
    )
    assert "  T3: overdue (last validated 31 months ago, due every 24 months)" in lines, (
        "section 5 carries the age and the frequency for a model that has been validated — "
        "'overdue' on its own is a verdict nobody can check"
    )
    assert "  T3b: current (last validated 2 months ago, due every 36 months)" in lines, (
        "every coverage row gets a line, including the models that are current"
    )


def test_report_empty_tier_row():
    lines = lesson.render_inventory_report(_probe()).splitlines()
    for t, count in ((1, 1), (2, 0), (3, 2), (4, 0)):
        assert f"  tier {t}: {count} models" in lines, (
            f"tier {t} should read '{count} models' — every tier gets a row, including the "
            "empty ones. An absent row and an empty row are different claims, and only one "
            "of them is a measurement"
        )
    order = [next(i for i, l in enumerate(lines) if l.startswith(f"  tier {t}:"))
             for t in (1, 2, 3, 4)]
    assert order == sorted(order), "the tier rows run 1, 2, 3, 4 in that order"


def test_report_shadow_findings():
    lines = lesson.render_inventory_report(_probe()).splitlines()
    shadow = [l for l in lines if "FINDING" in l and "SHADOW-9" in l]
    assert shadow, (
        "a runtime-only identifier is a FINDING and must be named in section 4 — it is the "
        "one list on the page that describes a model nothing has ever governed"
    )
    dead = [l for l in lines if "FINDING" in l and "T3" in l and "still running" in l]
    assert dead, "a decommissioned identifier still running is the second FINDING line"
    empty = _probe()
    empty["reconciliation"] = lesson.Reconciliation(("T1",), (), (), ())
    quiet = lesson.render_inventory_report(empty).splitlines()
    assert "  none running outside the inventory" in quiet, (
        "when the dangerous list is empty, say so on its own line — a missing line reads as "
        "an omission and the reader cannot tell the difference"
    )
    assert "  none decommissioned and still running" in quiet, (
        "the same for the decommissioned-and-running list"
    )
    assert not [l for l in quiet if "FINDING" in l], (
        "with both lists empty there is no FINDING line to print"
    )


def test_report_findings_block():
    lines = lesson.render_inventory_report(_probe()).splitlines()
    for text in ("schema: BAD-1 has 2 defect(s)",
                 "coverage: T1 has never been validated",
                 "coverage: T3 is overdue by 7 month(s)",
                 "shadow: SHADOW-9 is running and is not in the inventory",
                 "shadow: T3 is decommissioned and is still running"):
        assert f"  - {text}" in lines, f"section 6 is missing the finding '{text}'"
    assert not any("T3b" in l and l.startswith("  - ") for l in lines), (
        "T3b is current and is not a finding — section 6 lists findings, not models"
    )
    assert "  total findings: 6" in lines, (
        "two schema failures, one never-validated, one overdue, one shadow and one "
        "decommissioned-but-running is 6; count the list you built, do not type a number"
    )
    where = {text: lines.index(f"  - {text}") for text in (
        "schema: BAD-1 has 2 defect(s)",
        "coverage: T1 has never been validated",
        "coverage: T3 is overdue by 7 month(s)",
        "shadow: SHADOW-9 is running and is not in the inventory",
        "shadow: T3 is decommissioned and is still running")}
    assert list(where.values()) == sorted(where.values()), (
        "the findings run in the documented order — schema, then never-validated, then "
        f"overdue, then shadow, then decommissioned-and-running; you produced {where}"
    )


def test_report_missing_keys():
    for dropped in ("tiers", "coverage", "reconciliation", "data_note", "n_records"):
        findings = _probe()
        del findings[dropped]
        try:
            lesson.render_inventory_report(findings)
        except ValueError as exc:
            assert dropped in str(exc), f"the error should name the missing key: got {exc}"
        except NotImplementedError:
            raise
        except Exception as exc:
            raise AssertionError(
                f"incomplete findings raised {type(exc).__name__}, not ValueError"
            ) from None
        else:
            raise AssertionError(f"findings without {dropped!r} should raise ValueError")
    findings = _probe()
    del findings["tiers"]
    del findings["coverage"]
    try:
        lesson.render_inventory_report(findings)
    except ValueError as exc:
        assert "tiers" in str(exc) and "coverage" in str(exc), (
            f"name EVERY missing key in one message, not the first one you met: {exc}"
        )
    else:
        raise AssertionError("findings missing two keys should still raise ValueError")


RUBRIC = [
    (test_validate_accepts_clean_record, 3,
     "a well-formed record returns an empty tuple; None is a valid last_validated"),
    (test_validate_reports_every_missing_field, 4,
     "loop over REQUIRED_FIELDS, append one message per absent field, sort at the end"),
    (test_validate_enum_and_text, 4,
     "check membership of ENUMS[field]; a text field must be a str and non-empty when stripped"),
    (test_validate_bool_is_not_an_int, 4,
     "isinstance(True, int) is True — exclude bool explicitly, and type-check human_override"),
    (test_validate_numeric_and_date, 3,
     "non-negative ints only, no floats; frequency >= 1; last_validated None or YYYY-MM-DD"),
    (test_validate_missing_field_not_double_reported, 2,
     "guard every type check with `if field in record`, so an absent field gives one message"),
    (test_inventory_reports_every_failure, 5,
     "call validate_record for each record, keep only the non-empty results, never break"),
    (test_inventory_duplicate_ids, 5,
     "count the ids first, then append the duplicate message to every record carrying one"),
    (test_inventory_keys_and_order, 3,
     "key by model_id when it is a non-empty string, else f'<record {i}>' using enumerate"),
    (test_inventory_rejects_non_sequence, 3,
     "isinstance(records, (list, tuple)) up front, or a dict silently reports nothing"),
    (test_tier_points_table, 5,
     "sum policy['points'][field][value] for materiality, complexity and reversibility"),
    (test_tier_exposure_band_boundary, 4,
     "walk exposure_bands and keep the points of the last threshold with exposure >= it"),
    (test_tier_cut_boundaries, 5,
     "the first (cut, tier) with score >= cut wins; nothing matched is policy['default_tier']"),
    (test_tier_single_attribute_flip, 4,
     "every attribute must be scored, and each one at the policy's own weight"),
    (test_tier_escalation_is_a_floor, 5,
     "min(tier, policy['escalation_tier_at_most']), and only when BOTH halves hold"),
    (test_tier_audit_trail, 5,
     "six lines in policy order, then the cut line, then the escalation line when it applies"),
    (test_tier_rejects_unknown_values, 3,
     "raise ValueError on a value outside the points table; dict.get(v, 0) hides the worst model"),
    (test_reconcile_four_lists, 5,
     "set intersection and both differences; decommissioned_running is filtered out of `both`"),
    (test_reconcile_dedup_and_sort, 4,
     "build sets from both sides, then tuple(sorted(...)) every field"),
    (test_reconcile_first_record_status_wins, 3,
     "`if mid not in seen` before recording the status; read status with .get"),
    (test_reconcile_rejects_nameless_record, 2,
     "raise ValueError naming the position when model_id is not a non-empty string"),
    (test_coverage_only_in_use, 4,
     "skip any record whose status is not 'in_use'; keep the records' own order"),
    (test_coverage_months_arithmetic, 5,
     "(a.year - v.year) * 12 + (a.month - v.month) - (1 if a.day < v.day else 0)"),
    (test_coverage_due_boundary, 5,
     "overdue is months_since > due_months — strictly greater, so on the day it is current"),
    (test_coverage_never_validated, 2,
     "months_since is None for a model with no date; the verdict is 'never_validated'"),
    (test_coverage_future_date, 2,
     "a negative month count is a data-entry error: raise ValueError naming the model"),
    (test_report_structure, 4,
     "title line, blank line, then one '## <heading>' per REPORT_SECTIONS entry, in order"),
    (test_report_uses_findings_only, 5,
     "format every figure out of `findings`; recompute nothing here"),
    (test_report_empty_tier_row, 4,
     "iterate (1, 2, 3, 4) and count the tiers dict, rather than iterating the tiers present"),
    (test_report_shadow_findings, 4,
     "branch on whether each dangerous list is empty and print an explicit line either way"),
    (test_report_findings_block, 3,
     "build one list of finding texts in the documented order, then print it and its length"),
    (test_report_missing_keys, 2,
     "collect every absent key up front and name them all in one ValueError"),
]
