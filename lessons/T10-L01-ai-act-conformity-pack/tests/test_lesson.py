"""Autograder rubric for T10-L01. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Two rules this file follows:

1. The dates asserted below are written out literally, from claims.yaml, rather than read
   from the student's OBLIGATIONS table. A rubric that grades a table against itself grades
   nothing — a student who edits a date to match their code would still score full marks.
2. The cases deliberately differ from the notebook's public `_check_*` functions: other
   systems, boundary dates, and structural probes (does the evidence set actually follow the
   obligation set?) that a plausible-looking wrong implementation fails.
"""
import importlib.util, os
from datetime import date
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

AS_OF = date(2026, 9, 16)
# A second vantage point, chosen because obligations fall EXACTLY on it: "already binds"
# and "still ahead" are both decided at this date, so < and <= stop being interchangeable.
ON_THE_DAY = date(2026, 8, 2)

# The law, written out here so the rubric is independent of the student's table.
ART4 = "2025-02-02"
ART5 = "2025-02-02"
GPAI = "2025-08-02"
ART50 = "2026-08-02"
ART50_LEGACY = "2026-12-02"
ANNEX_III = "2027-12-02"
ANNEX_I = "2028-08-02"


def _system(**overrides) -> dict:
    """A minimal-risk system, with whatever the caller wants changed."""
    base = {
        "id": "s", "name": "a system", "role": "provider",
        "prohibited_practice": None, "annex_iii_area": None,
        "embedded_in_regulated_product": False, "product_legislation": None,
        "is_gpai_model": False, "interacts_with_humans": False,
        "generates_synthetic_content": False, "placed_on_market": "2026-01-15",
        "evidence": {},
    }
    base.update(overrides)
    return base


def _present(document="DOC-1", issued="2025-06-01", **extra) -> dict:
    record = {"status": "present", "document": document, "date": issued}
    record.update(extra)
    return record


def _fully_evidenced(system: dict) -> dict:
    """Give a system a valid record for every item its obligations require."""
    required = set()
    for obligation in lesson.classify(system)["obligations"]:
        required.update(lesson.EVIDENCE_REQUIRED.get(obligation, ()))
    return dict(system, evidence={item: _present() for item in sorted(required)})


# --- exercise 1: classify -----------------------------------------------------------------

def test_classify_tiers_and_precedence():
    """Every tier, and the two precedence rules that cost two years when guessed."""
    cases = [
        (_system(), "minimal"),
        (_system(interacts_with_humans=True), "transparency_only"),
        (_system(generates_synthetic_content=True), "transparency_only"),
        (_system(is_gpai_model=True), "gpai_model"),
        (_system(annex_iii_area="education"), "high_risk_annex_iii"),
        (_system(embedded_in_regulated_product=True,
                 product_legislation="Regulation (EU) 2017/745"), "high_risk_annex_i"),
        # an Annex III area AND an embedding: the embedding decides, and it is the later date
        (_system(annex_iii_area="critical_infrastructure",
                 embedded_in_regulated_product=True), "high_risk_annex_i"),
        # a prohibited practice outranks every other flag, including an embedding
        (_system(prohibited_practice="untargeted_facial_scraping",
                 embedded_in_regulated_product=True, annex_iii_area="law_enforcement",
                 is_gpai_model=True), "prohibited"),
        # a GPAI model that is also an Annex III system is tiered by the heavier route
        (_system(is_gpai_model=True, annex_iii_area="employment"), "high_risk_annex_iii"),
    ]
    for system, expected in cases:
        got = lesson.classify(system)["risk_tier"]
        assert got == expected, (
            f"{ {k: v for k, v in system.items() if v not in (None, False, 'provider')} } "
            f"classified as {got!r}, expected {expected!r}")


def test_classify_obligations_cumulate():
    """The tier names the heaviest route; it never replaces the set."""
    minimal = lesson.classify(_system())["obligations"]
    assert minimal == ["art4_ai_literacy"], (
        f"a minimal-risk system owes Article 4 and nothing else, got {minimal}")

    both = lesson.classify(_system(annex_iii_area="employment", interacts_with_humans=True,
                                   generates_synthetic_content=True, is_gpai_model=True))
    assert set(both["obligations"]) == {
        "art4_ai_literacy", "ch3_high_risk_annex_iii", "art50_inform_interaction",
        "art50_mark_synthetic", "art53_gpai_model"}, (
        f"got {both['obligations']} — each flag adds its own obligation; high risk does not "
        "absorb the Article 50 or GPAI duties")
    assert both["obligations"] == sorted(both["obligations"]), "obligations must be sorted"

    embedded = lesson.classify(_system(embedded_in_regulated_product=True,
                                       interacts_with_humans=True))
    assert set(embedded["obligations"]) == {
        "art4_ai_literacy", "ch3_high_risk_annex_i", "art50_inform_interaction"}, (
        f"got {embedded['obligations']} — an embedded system that talks to people owes the "
        "Annex I Chapter III duties and Article 50(1)")
    assert "ch3_high_risk_annex_iii" not in embedded["obligations"], (
        "an embedded system takes the Annex I route only; it must not also carry Annex III")

    banned = lesson.classify(_system(prohibited_practice="social_scoring",
                                     annex_iii_area="employment", is_gpai_model=True,
                                     interacts_with_humans=True))
    assert banned["obligations"] == ["art4_ai_literacy", "art5_prohibited_practice"], (
        f"got {banned['obligations']} — once prohibited, stop; a system that must not be on "
        "the market does not also owe Chapter III, Article 50 or Article 53 duties")


def test_classify_deadlines_match_the_law():
    """Each obligation's date, and applies_from as the earliest of them."""
    everything = lesson.classify(_system(annex_iii_area="employment",
                                         interacts_with_humans=True, is_gpai_model=True))
    assert set(everything["deadlines"]) == set(everything["obligations"]), (
        "deadlines must have exactly one entry per obligation, no more and no fewer")
    for obligation, expected in (("art4_ai_literacy", ART4),
                                 ("art53_gpai_model", GPAI),
                                 ("art50_inform_interaction", ART50),
                                 ("ch3_high_risk_annex_iii", ANNEX_III)):
        got = everything["deadlines"][obligation]
        assert got == expected, f"{obligation} dated {got}, the law says {expected}"
    assert everything["applies_from"] == ART4, (
        f"applies_from is the earliest date in deadlines; got {everything['applies_from']}")

    embedded = lesson.classify(_system(embedded_in_regulated_product=True))
    assert embedded["deadlines"]["ch3_high_risk_annex_i"] == ANNEX_I, (
        f"the Annex I route is dated {embedded['deadlines']['ch3_high_risk_annex_i']}, the "
        f"law says {ANNEX_I}")

    banned = lesson.classify(_system(prohibited_practice="social_scoring"))
    assert banned["deadlines"]["art5_prohibited_practice"] == ART5, (
        f"Article 5 dated {banned['deadlines']['art5_prohibited_practice']}, not {ART5}")

    # A system whose only duties are far off must report the far-off date, not a constant.
    quiet = lesson.classify(_system(embedded_in_regulated_product=True))
    assert quiet["applies_from"] == ART4, (
        "even an Annex I system carries Article 4, so its earliest date is 2025-02-02")


def test_classify_article_50_transitional_rule():
    """The marking date is per system, and the boundary is exclusive."""
    cases = [
        ("2025-05-20", ART50_LEGACY),   # long before general application
        ("2026-08-01", ART50_LEGACY),   # the day before: still transitional
        ("2026-08-02", ART50),          # the day itself: no relief
        ("2026-08-30", ART50),          # after
        ("2027-04-04", ART50),          # placed later still
    ]
    for placed, expected in cases:
        got = lesson.classify(_system(generates_synthetic_content=True,
                                      placed_on_market=placed))
        assert got["deadlines"]["art50_mark_synthetic"] == expected, (
            f"placed on {placed}: marking dated "
            f"{got['deadlines']['art50_mark_synthetic']}, expected {expected} — the relief "
            "is for systems placed on the market STRICTLY before 2026-08-02")

    # The transitional rule touches Article 50(2) only.
    legacy = lesson.classify(_system(interacts_with_humans=True,
                                     generates_synthetic_content=True,
                                     placed_on_market="2025-05-20"))
    assert legacy["deadlines"]["art50_inform_interaction"] == ART50, (
        f"Article 50(1) dated {legacy['deadlines']['art50_inform_interaction']} — the "
        "transitional arrangement covers the marking obligation only, not the disclosure one")


# --- exercise 2: missing_evidence ----------------------------------------------------------

def test_missing_evidence_status_and_document():
    """A promise is not a document, and a reference to nothing is not a reference."""
    literacy = "ai_literacy_training_record"
    assert lesson.missing_evidence(_system(), AS_OF) == [literacy], (
        "an empty evidence map leaves the Article 4 record missing")

    for label, record in (
        ("status planned", {"status": "planned", "document": "D", "date": "2025-06-01"}),
        ("status missing", {"status": "missing", "document": "", "date": None}),
        ("status absent", {"document": "D", "date": "2025-06-01"}),
        ("empty document", _present(document="")),
        ("whitespace document", _present(document="  \t ")),
        ("null document", _present(document=None)),
    ):
        system = _system(evidence={literacy: record})
        assert lesson.missing_evidence(system, AS_OF) == [literacy], (
            f"{label}: this record must not count as evidence")

    assert lesson.missing_evidence(_system(evidence={literacy: _present()}), AS_OF) == [], (
        "a present, referenced, in-date record must count")


def test_missing_evidence_dates_and_review():
    """Undated, future-dated and lapsed records do not count; a future review is fine."""
    literacy = "ai_literacy_training_record"
    for label, record in (
        ("no date key", {"status": "present", "document": "D"}),
        ("null date", _present(issued=None)),
        ("dated tomorrow", _present(issued="2026-09-17")),
        ("dated next year", _present(issued="2027-01-01")),
        ("review lapsed yesterday", _present(review_due="2026-09-15")),
        ("review lapsed long ago", _present(review_due="2025-12-31")),
    ):
        system = _system(evidence={literacy: record})
        assert lesson.missing_evidence(system, AS_OF) == [literacy], (
            f"{label}: this record must not count as evidence")

    for label, record in (
        ("issued today", _present(issued="2026-09-16")),
        ("review due today", _present(review_due="2026-09-16")),
        ("review due next year", _present(review_due="2027-06-30")),
        ("no review_due key at all", _present()),
    ):
        system = _system(evidence={literacy: record})
        assert lesson.missing_evidence(system, AS_OF) == [], (
            f"{label}: this record is valid and must count — the boundaries are inclusive "
            "for the issue date and for review_due")


def test_missing_evidence_uses_the_obligation_set():
    """What a system owes follows from what applies to it, not from a fixed list."""
    heavy = _system(annex_iii_area="employment")
    required_missing = lesson.missing_evidence(heavy, AS_OF)
    # Sorted, and stated as one exact list: the required items come out of a set, and a set
    # has no order, so a report built on an unsorted return diffs differently on every run.
    assert required_missing == sorted(required_missing), (
        f"missing_evidence returned {required_missing} — return the ids SORTED. They come "
        "from a set, so whatever order you got is an accident and will not repeat")
    assert required_missing == [
        "accuracy_robustness_cybersecurity_report", "ai_literacy_training_record",
        "automatic_logging_design", "ce_marking_record", "conformity_assessment_record",
        "data_governance_record", "eu_database_registration", "eu_declaration_of_conformity",
        "human_oversight_plan", "instructions_for_use", "post_market_monitoring_plan",
        "quality_management_system", "risk_management_system", "serious_incident_procedure",
        "technical_documentation"], (
        f"an unevidenced Annex III system is missing exactly the 14 Chapter III items plus "
        f"the Article 4 record, sorted; got {required_missing}")
    assert "eu_database_registration" in required_missing, (
        "an Annex III system owes the EU database registration")
    assert "integrated_product_conformity_assessment" not in required_missing, (
        "an Annex III system does not owe the embedded-product conformity link")

    embedded = _system(embedded_in_regulated_product=True)
    embedded_missing = lesson.missing_evidence(embedded, AS_OF)
    assert "integrated_product_conformity_assessment" in embedded_missing, (
        "an Annex I system owes the integrated product conformity assessment")
    assert "eu_database_registration" not in embedded_missing, (
        "an Annex I system does not owe the Annex III database registration")

    # A minimal system holding a full high-risk pack still owes only its one record.
    stuffed = _system(evidence={item: _present() for item in
                                lesson.EVIDENCE_REQUIRED["ch3_high_risk_annex_iii"]})
    assert lesson.missing_evidence(stuffed, AS_OF) == ["ai_literacy_training_record"], (
        "a minimal-risk system requires the Article 4 record and nothing else; evidence it "
        "does not owe cannot substitute for evidence it does")

    for system in (heavy, embedded, _system(is_gpai_model=True),
                   _system(prohibited_practice="social_scoring")):
        assert lesson.missing_evidence(_fully_evidenced(system), AS_OF) == [], (
            "a system with a valid record for every required item has nothing missing")


# --- exercise 3: conformity_report ---------------------------------------------------------

def test_report_rows_and_shape():
    """Row structure, input order, and the arithmetic of days_to_deadline."""
    # The ids are deliberately NOT in alphabetical order: a report that sorts its rows instead
    # of preserving the registry order fails here, where ids like a/b/c would have hidden it.
    systems = [_system(id="zeta-hire", name="hiring", annex_iii_area="employment"),
               _system(id="alpha-note", name="notes"),
               _system(id="mid-score", name="scoring",
                       prohibited_practice="social_scoring")]
    report = lesson.conformity_report(systems, AS_OF)
    assert set(report) == {"as_of", "rows", "summary", "blocking"}, (
        f"top-level keys were {sorted(report)}")
    assert report["as_of"] == "2026-09-16", f"as_of was {report['as_of']!r}, want an ISO string"
    assert [r["id"] for r in report["rows"]] == ["zeta-hire", "alpha-note", "mid-score"], (
        f"rows came back as {[r['id'] for r in report['rows']]} — keep the input order; these "
        "ids are not alphabetical, so sorting them is visible here")

    for row in report["rows"]:
        assert {"id", "name", "risk_tier", "deadlines", "applies_from", "days_to_deadline",
                "missing", "complete", "overdue"} <= set(row), (
            f"row {row['id']} is missing fields: {sorted(row)}")
        expected_days = (date.fromisoformat(row["applies_from"]) - AS_OF).days
        assert row["days_to_deadline"] == expected_days, (
            f"row {row['id']}: days_to_deadline {row['days_to_deadline']} != "
            f"{expected_days}; it is (applies_from - as_of).days, negative once it binds")
        assert row["complete"] == (not row["missing"]), (
            f"row {row['id']}: complete must mean nothing missing")
    assert [r["name"] for r in report["rows"]] == ["hiring", "notes", "scoring"], (
        f"row names came back as {[r['name'] for r in report['rows']]} — copy the system's "
        "'name' across; a pack that labels every row with its id is not readable by the "
        "people who have to sign it")

    assert report["rows"][0]["days_to_deadline"] < 0, (
        "Article 4 bound in February 2025, so days_to_deadline is negative here")

    # A fully evidenced registry is complete and blocks nothing.
    clean = lesson.conformity_report([_fully_evidenced(_system(id="x"))], AS_OF)
    assert clean["rows"][0]["complete"] and not clean["rows"][0]["overdue"], (
        "a system with all its evidence is complete and not overdue")
    assert clean["blocking"] == [], f"nothing should block, got {clean['blocking']}"


def test_report_summary_arithmetic_and_next_deadline():
    """The summary a reviewer reads first, over the shipped registry."""
    report = lesson.conformity_report(lesson.REGISTRY, AS_OF)
    summary = report["summary"]
    assert summary["total"] == 8, f"the registry holds 8 systems, summary says {summary['total']}"
    assert summary["complete"] == 1, (
        f"exactly one system in the registry is fully evidenced, summary says "
        f"{summary['complete']}")
    assert summary["incomplete"] == 7, f"summary says {summary['incomplete']} incomplete"
    assert sum(summary["by_tier"].values()) == 8, "by_tier must account for every system"
    assert summary["by_tier"]["high_risk_annex_i"] == 2, (
        f"two systems take the Annex I route, by_tier says "
        f"{summary['by_tier'].get('high_risk_annex_i')}")
    assert summary["by_tier"]["high_risk_annex_iii"] == 2, (
        f"two systems take the Annex III route, by_tier says "
        f"{summary['by_tier'].get('high_risk_annex_iii')}")
    assert "minimal" in summary["by_tier"], "the registry contains a minimal-risk system"
    # The shipped registry happens to use all six tiers, so it cannot tell a tally from a
    # pre-seeded table of zeroes. This two-system registry can.
    narrow = lesson.conformity_report(
        [_system(id="one"), _system(id="two", is_gpai_model=True)], AS_OF)["summary"]
    assert narrow["by_tier"] == {"minimal": 1, "gpai_model": 1}, (
        f"by_tier was {narrow['by_tier']} — count the tiers that actually occur, and only "
        "those. A tier with no systems in it is absent from the table, not a row of zeroes")
    assert summary["next_deadline"] == "2026-12-02", (
        f"next_deadline was {summary['next_deadline']!r}; the earliest date still ahead of "
        "2026-09-16 among incomplete systems is the transitional marking date 2026-12-02")

    # With no gaps anywhere there is no next deadline to report.
    clean = lesson.conformity_report([_fully_evidenced(_system(id="x"))], AS_OF)
    assert clean["summary"]["next_deadline"] is None, (
        f"next_deadline should be None when nothing is incomplete, got "
        f"{clean['summary']['next_deadline']!r}")

    # next_deadline looks only at rows that still have gaps. Here the COMPLETE system carries
    # the earlier upcoming date (2026-12-02) and the incomplete one the later (2027-12-02):
    # a report that scans every row's deadlines returns the wrong one.
    mixed = [
        _fully_evidenced(_system(id="done-early", generates_synthetic_content=True,
                                 placed_on_market="2025-05-20")),
        _system(id="still-open", annex_iii_area="employment"),
    ]
    got = lesson.conformity_report(mixed, AS_OF)["summary"]["next_deadline"]
    assert got == ANNEX_III, (
        f"next_deadline was {got!r}, expected {ANNEX_III} — the finished system's "
        f"{ART50_LEGACY} marking date must not be reported; a date a fully evidenced system "
        "already meets is not the next thing anyone has to do")

    # Graded on the day itself: this system's Article 50(1) duty is dated exactly ON_THE_DAY.
    # "Strictly after as_of" means today is not the next deadline — today is now.
    edge = _system(id="edge", annex_iii_area="employment", interacts_with_humans=True,
                   evidence={"ai_literacy_training_record": _present()})
    on_the_day = lesson.conformity_report([edge], ON_THE_DAY)["summary"]["next_deadline"]
    assert on_the_day == ANNEX_III, (
        f"graded on {ON_THE_DAY.isoformat()}, next_deadline came back {on_the_day!r}. The "
        f"Article 50(1) date IS {ON_THE_DAY.isoformat()}, so it is not still ahead: "
        f"next_deadline is the earliest deadline STRICTLY after as_of, here {ANNEX_III}")


def test_report_blocking_and_overdue():
    """Overdue is a gap against a date that has already passed — not merely a gap."""
    report = lesson.conformity_report(lesson.REGISTRY, AS_OF)
    rows = {r["id"]: r for r in report["rows"]}

    assert rows["cv-screener"]["missing"] and not rows["cv-screener"]["overdue"], (
        "cv-screener has six gaps, but all of them sit under the Annex III obligation dated "
        "2027-12-02, so it is behind schedule, not in breach")
    assert not rows["cobot-guard"]["overdue"], (
        "cobot-guard's gaps belong to the 2028 Annex I route")
    assert rows["atlas-7b"]["overdue"], (
        "atlas-7b is missing its training-data summary, and the GPAI duties bound in "
        "August 2025")
    assert rows["loan-copilot"]["overdue"], (
        "loan-copilot was placed on the market after general application, so its marking "
        "obligation bound on 2026-08-02 and its two marking gaps are already overdue")
    assert not rows["support-copilot"]["overdue"], (
        "support-copilot has the same marking gaps but was placed on the market before "
        "2026-08-02, so the transitional date 2026-12-02 has not arrived yet — this is the "
        "whole point of the per-system date")

    assert report["blocking"] == ["atlas-7b", "citizen-score", "loan-copilot"], (
        f"blocking was {report['blocking']}; it is the sorted ids that are prohibited or "
        "overdue — note that cv-screener, which has the second-longest gap list in the "
        "registry, is not among them, because every one of its gaps is dated 2027")
    assert report["summary"]["overdue"] == 3, (
        f"summary['overdue'] was {report['summary']['overdue']}, expected 3")

    # A prohibited system with every record it owes is complete and not overdue — and still
    # blocks. "blocking" is prohibited OR overdue, never overdue alone.
    withdrawn = _fully_evidenced(_system(id="banned", prohibited_practice="social_scoring"))
    solo = lesson.conformity_report([withdrawn], AS_OF)
    assert solo["rows"][0]["complete"] and not solo["rows"][0]["overdue"], (
        "this system holds a valid record for every item its obligations require, so it has "
        "nothing missing and nothing overdue")
    assert solo["blocking"] == ["banned"], (
        f"blocking was {solo['blocking']} — a prohibited practice blocks even when the "
        "paperwork is complete; you cannot evidence your way out of Article 5")

    # The day an obligation binds, it binds. This system holds its Article 4 record, so its
    # only gaps sit under Article 50(1) — dated exactly ON_THE_DAY — and under the 2027
    # Annex III route. Grade it on 2026-08-02 and the Article 50 gap is a breach that day,
    # not the day after.
    edge = _system(id="edge", annex_iii_area="employment", interacts_with_humans=True,
                   evidence={"ai_literacy_training_record": _present()})
    day_of = lesson.conformity_report([edge], ON_THE_DAY)
    assert "ai_literacy_training_record" not in day_of["rows"][0]["missing"], (
        "the Article 4 record here is valid; only the Article 50 and Annex III items are gaps")
    assert day_of["rows"][0]["overdue"], (
        f"graded on {ON_THE_DAY.isoformat()} the row came back overdue="
        f"{day_of['rows'][0]['overdue']}. Article 50(1) is dated {ON_THE_DAY.isoformat()} "
        "and the disclosure is missing: an obligation binds ON its date, so compare with "
        "<= as_of, not < as_of")
    assert day_of["blocking"] == ["edge"], (
        f"blocking was {day_of['blocking']} — a row overdue on the day itself blocks")

    # One day earlier the same gap is still a plan, not a breach.
    day_before = lesson.conformity_report([edge], date(2026, 8, 1))
    assert not day_before["rows"][0]["overdue"], (
        "on 2026-08-01 nothing this system owes has bound yet, so it is not overdue — if "
        "this fails you are treating every gap as a breach")


def test_render_report_is_a_readable_artefact():
    """The rendered pack names every system, dates itself, and carries its own caveat."""
    report = lesson.conformity_report(lesson.REGISTRY, AS_OF)
    text = lesson.render_report(report)
    assert isinstance(text, str), f"render_report returned {type(text).__name__}, want a str"
    for system in lesson.REGISTRY:
        assert system["id"] in text, (
            f"render_report omitted {system['id']} — every row in the report must appear in "
            "the rendered table, including the ones with nothing outstanding")
    assert report["as_of"] in text, (
        "render_report must state the date the pack was assembled; an undated compliance "
        "artefact cannot be reviewed")
    assert "not legal advice" in text.lower(), (
        "the rendered pack must carry its own caveat — the words 'not legal advice' have to "
        "survive into the artefact, not just sit in the notebook prose")


RUBRIC = [
    (test_classify_tiers_and_precedence, 4,
     "test the flags in order: prohibited, then the embedding, then the Annex III area"),
    (test_classify_obligations_cumulate, 4,
     "build a set and add to it; the tier is a label, not a switch that picks one obligation"),
    (test_classify_deadlines_match_the_law, 3,
     "one deadline per obligation, read from OBLIGATIONS, and applies_from is their minimum"),
    (test_classify_article_50_transitional_rule, 3,
     "placed_on_market strictly before 2026-08-02 earns the 2026-12-02 marking date"),
    (test_missing_evidence_status_and_document, 3,
     "status must be exactly 'present' AND document must be non-empty after stripping"),
    (test_missing_evidence_dates_and_review, 3,
     "date > as_of fails, review_due < as_of fails, and a missing review_due is fine"),
    (test_missing_evidence_uses_the_obligation_set, 2,
     "union EVIDENCE_REQUIRED over the obligations classify returns; do not hard-code a list"),
    (test_report_rows_and_shape, 2,
     "one row per system in input order, with days_to_deadline = (applies_from - as_of).days"),
    (test_report_summary_arithmetic_and_next_deadline, 2,
     "next_deadline is the earliest date after as_of among INCOMPLETE rows only, else None"),
    (test_report_blocking_and_overdue, 2,
     "a row is overdue when a missing item belongs to an obligation whose date has passed; "
     "blocking is prohibited OR overdue"),
    (test_render_report_is_a_readable_artefact, 2,
     "return the table as a string: every id, the as_of date, and the caveat line"),
]
