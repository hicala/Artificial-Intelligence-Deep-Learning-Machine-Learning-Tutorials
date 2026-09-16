"""Autograder rubric for P01-L02. Hidden cases live here, not in the notebook.

Each RUBRIC entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.

Rules that matter in this particular lesson and are deliberate:

* Fixtures are built HERE, with `dict(..., **override)` over the lesson's own systems, so the
  rubric can reach states the shipped nine do not cover — a profiling system whose paperwork
  is also broken, a condition-order case where two conditions hold at once, a date exactly on
  a boundary. A rubric that can only see the shipped fixtures grades the fixtures.
* Nothing here asserts a literal digest. What is graded is the behaviour a seal must have:
  recomputable by a third party, and moved by a change to any covered field.
* `test_obligations_binding_now_and_next_deadline` grades both boundary comparisons at once by
  running as_of ON an obligation's own date. `<` instead of `<=` in binding_now, and `>=`
  instead of `>` in next_deadline, are the two most common ways to be one day wrong, and on
  the shipped AS_OF neither is visible.
* `test_derogation_profiling_overrides_everything` gives the profiling system BROKEN paperwork
  as well. An implementation that checks documentation before profiling returns a plausible
  refusal for the wrong reason, and a refusal for the wrong reason sends a provider off to fix
  the wrong thing.
* `test_disagreement_pivot_is_earliest` uses a panel that splits at TWO questions. An
  implementation that scans the questions in any order but QUESTION_IDS order, or that takes
  the last split rather than the first, passes a single-split panel by accident.
* `test_ask_tier_minimality_and_order` asserts an exact trail length per system rather than
  "a decisive answer exists". Minimality is the property that makes the trail an explanation;
  a classifier that answers all six questions and then picks is not wrong about the tier, it
  is wrong about what it knows.
"""
import copy
import hashlib
import importlib.util
import json
import os
from datetime import date
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("ATLAS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)

SYSTEMS = lesson.SYSTEMS
AS_OF = lesson.AS_OF
QUESTION_IDS = lesson.QUESTION_IDS


def sys_(name: str, **override) -> dict:
    """A private copy of a shipped system, with fields overridden."""
    return dict(copy.deepcopy(SYSTEMS[name]), **override)


def _seal(record: dict) -> str:
    """Re-derive a record's digest here, independently of the student's implementation."""
    bare = {k: v for k, v in record.items() if k != "record_hash"}
    return hashlib.sha256(lesson.canonical_bytes(bare)).hexdigest()


# --- exercise 1: the Article 6(3) derogation -------------------------------------------

def test_derogation_shape_and_not_claimed():
    got = lesson.derogation_assessment(sys_("credit-scorer"))
    assert set(got) == {"claimed", "granted", "condition", "reason", "documentation_gaps"}, (
        f"keys were {sorted(got)}; expected exactly claimed, granted, condition, reason, "
        "documentation_gaps"
    )
    assert got["claimed"] is False and got["granted"] is False, (
        f"credit-scorer sets derogation_claimed False, so claimed={got['claimed']!r} and "
        f"granted={got['granted']!r} should both be False — and they must be booleans, not "
        "the raw field value"
    )
    assert got["reason"] == lesson.REASON_NOT_CLAIMED, (
        f"reason was {got['reason']!r}; a derogation nobody claimed fails as "
        f"{lesson.REASON_NOT_CLAIMED!r}, not as a failure of its merits"
    )
    assert got["documentation_gaps"] == [], (
        f"documentation_gaps was {got['documentation_gaps']} — the Article 6(4) duty is "
        "created BY the claim. Reporting gaps against a provider who never claimed the "
        "derogation invents a finding"
    )
    # A system with no flags at all must not crash on the missing keys.
    bare = lesson.derogation_assessment({"id": "empty"})
    assert bare["granted"] is False and bare["condition"] is None, (
        "an empty description claims nothing and meets nothing: use .get(), not ['...']"
    )

    # `claimed` is the field that separates "nobody claimed it" from "claimed and refused",
    # and the whole Article 6(4) duty hangs off that difference. cv-ranker DID claim it.
    made_the_claim = lesson.derogation_assessment(sys_("cv-ranker"))
    assert made_the_claim["claimed"] is True, (
        f"cv-ranker sets derogation_claimed True but claimed came back "
        f"{made_the_claim['claimed']!r}. `claimed` mirrors the provider's own claim; it is "
        "not a synonym for `granted`, and a record that always says False cannot tell a "
        "refused claim from a claim nobody made — which is exactly the distinction that "
        "decides whether the Article 6(4) documentation duty arose at all"
    )

    # Not claimed, but it WOULD have met 6(3)(b). Report the condition anyway: "you could
    # have claimed this and did not" is a finding, and the shipped credit-scorer sets no
    # condition flag at all, so a `condition` hard-wired to None hides here.
    could_have = lesson.derogation_assessment(sys_("credit-scorer", improves_human_activity=True))
    assert could_have["condition"] == "improves_human_activity", (
        f"condition was {could_have['condition']!r}; this system meets Article 6(3)(b) and "
        "simply never claimed the derogation. Compute `condition` from the flags in every "
        "branch, including the not-claimed one — it is what the provider could have argued"
    )
    assert could_have["claimed"] is False and could_have["documentation_gaps"] == [], (
        "naming the condition must not turn an unmade claim into a claim: still "
        "REASON_NOT_CLAIMED with no gaps"
    )


def test_derogation_profiling_overrides_everything():
    got = lesson.derogation_assessment(sys_("cv-ranker"))
    assert got["reason"] == lesson.REASON_PROFILING_OVERRIDE and got["granted"] is False, (
        f"cv-ranker got {got['reason']!r}: it meets Article 6(3)(b), poses no significant "
        "risk on its own assessment, and documented and registered everything — and it "
        "profiles natural persons, which Article 6(3) says always makes it high-risk"
    )
    assert got["condition"] == "improves_human_activity", (
        f"condition was {got['condition']!r}; still report the condition the provider met. A "
        "refusal that hides the argument it refused cannot be reviewed"
    )

    # Profiling AND broken paperwork AND a stated significant risk. Only the override may win.
    worst = sys_("cv-ranker", significant_risk_of_harm=True, assessment_documented=False,
                 justification="", registered_in_eu_database=False, documented_on=None)
    got = lesson.derogation_assessment(worst)
    assert got["reason"] == lesson.REASON_PROFILING_OVERRIDE, (
        f"got {got['reason']!r} for a system that profiles AND has no documentation AND "
        "declares a significant risk. The profiling sentence is unconditional, so it is "
        "tested FIRST — a refusal for the wrong reason sends the provider to fix the wrong "
        "thing"
    )
    assert got["documentation_gaps"], (
        "the gaps are still reported alongside the override: the claim was made, so the "
        "Article 6(4) duty arose, and a reviewer wants both findings"
    )

    # And profiling beats a perfect record with every single condition met.
    perfect = sys_("shift-note-tidier", performs_profiling=True, improves_human_activity=True,
                   detects_decision_patterns=True, preparatory_task=True)
    assert lesson.derogation_assessment(perfect)["granted"] is False, (
        "meeting all four conditions does not outvote the profiling rule; the conditions are "
        "necessary, never sufficient"
    )


def test_derogation_condition_order_and_none():
    # (b) and (d) both hold; the article lists (b) first, so (b) is cited.
    two = sys_("shift-note-tidier", narrow_procedural_task=False,
               improves_human_activity=True, preparatory_task=True)
    got = lesson.derogation_assessment(two)
    assert got["condition"] == "improves_human_activity", (
        f"condition was {got['condition']!r}; two conditions hold and DEROGATION_CONDITIONS "
        "lists improves_human_activity first. Walk the tuple in order and take the first "
        "match, so two reviewers cite the same paragraph"
    )
    assert got["granted"] is True, "it still qualifies; the citation is what was under test"

    # (c) alone, to prove the walk is not hard-coded to the first two entries.
    third = sys_("shift-note-tidier", narrow_procedural_task=False,
                 detects_decision_patterns=True)
    assert lesson.derogation_assessment(third)["condition"] == "detects_decision_patterns", (
        "condition (c) on its own must be found too"
    )

    none_met = sys_("shift-note-tidier", narrow_procedural_task=False)
    got = lesson.derogation_assessment(none_met)
    assert got["condition"] is None, f"condition was {got['condition']!r}; none is truthy here"
    assert got["reason"] == lesson.REASON_NO_CONDITION_MET, (
        f"got {got['reason']!r}: this system's paperwork is perfect, so it fails on the "
        f"merits — {lesson.REASON_NO_CONDITION_MET!r} — and the merits are tested before the "
        "paperwork"
    )

    risky = sys_("shift-note-tidier", significant_risk_of_harm=True)
    assert lesson.derogation_assessment(risky)["reason"] == lesson.REASON_SIGNIFICANT_RISK, (
        "a declared significant risk of harm refuses the derogation before any condition is "
        "consulted: Article 6(3) requires BOTH limbs"
    )


def test_derogation_documentation_gaps():
    got = lesson.derogation_assessment(sys_("exam-prep-sorter"))
    assert got["reason"] == lesson.REASON_DOCUMENTATION_INCOMPLETE, (
        f"exam-prep-sorter got {got['reason']!r}; it qualifies under 6(3)(d) and has "
        "documented nothing, so the Article 6(4) duty is what refuses it"
    )
    assert got["documentation_gaps"] == sorted(lesson.DOCUMENTATION_DUTIES), (
        f"got {got['documentation_gaps']}; all four duties fail here and the list is sorted"
    )

    # Each duty on its own, against an otherwise clean record.
    base = "shift-note-tidier"
    for override, want in (
        ({"assessment_documented": False}, "assessment_documented"),
        ({"justification": "   "}, "justification"),
        ({"justification": None}, "justification"),
        ({"registered_in_eu_database": False}, "registered_in_eu_database"),
        ({"documented_on": None}, "documented_before_placing"),
        ({"placed_on_market": None}, "documented_before_placing"),
        ({"documented_on": "2026-03-16"}, "documented_before_placing"),
    ):
        got = lesson.derogation_assessment(sys_(base, **override))
        assert got["documentation_gaps"] == [want], (
            f"with {override} the gaps were {got['documentation_gaps']}, expected [{want!r}]. "
            "A justification of whitespace is not a justification (strip it), and a missing "
            "date is a gap rather than a crash"
        )
        assert got["granted"] is False, f"with {override} the derogation cannot be granted"

    # The boundary: documenting ON the day of placing is not documenting BEFORE it.
    same_day = sys_(base, documented_on=SYSTEMS[base]["placed_on_market"])
    assert lesson.derogation_assessment(same_day)["documentation_gaps"] == \
        ["documented_before_placing"], (
        "Article 6(4) says the assessment is documented before the system is placed on the "
        "market; the comparison is strict, so documented_on == placed_on_market is a gap"
    )
    day_before = sys_(base, documented_on="2026-03-14", placed_on_market="2026-03-15")
    assert lesson.derogation_assessment(day_before)["documentation_gaps"] == [], (
        "one day before is before: do not make the comparison so strict that nothing passes"
    )


# --- exercise 2: the interview ----------------------------------------------------------

def test_ask_tier_minimality_and_order():
    expected = {
        "mood-meter": ("prohibited", 1),
        "lift-door-sensor": ("high_risk_annex_i", 2),
        "credit-scorer": ("high_risk_annex_iii", 4),
        "cv-ranker": ("high_risk_annex_iii", 4),
        "shift-note-tidier": ("annex_iii_derogated", 4),
        "exam-prep-sorter": ("high_risk_annex_iii", 4),
        "house-lm": ("gpai_model", 4),
        "policy-drafter": ("transparency_only", 5),
        "demand-forecaster": ("minimal", 5),
    }
    for sid, (want_tier, want_len) in expected.items():
        interview = lesson.ask_tier(sys_(sid))
        asked = [step.question for step in interview.trail]
        assert interview.tier == want_tier, f"{sid}: got {interview.tier!r}, want {want_tier!r}"
        assert len(asked) == want_len, (
            f"{sid}: asked {asked} ({len(asked)} questions), expected {want_len}. Return as "
            "soon as a question decides; evaluating all six and then picking a winner gets "
            "the tier right and destroys the explanation"
        )
        assert asked == sorted(asked, key=QUESTION_IDS.index), (
            f"{sid}: {asked} is out of order — ask them as QUESTIONS lists them"
        )
        assert len(set(asked)) == len(asked), f"{sid}: {asked} asks a question twice"
        assert set(asked) <= set(QUESTION_IDS), (
            f"{sid}: {asked} contains a question id that is not in QUESTIONS"
        )

    # A GPAI model that is also in an Annex III area never reaches q5: the areas decide first.
    both = sys_("house-lm", annex_iii_area="employment (Annex III point 4(a))")
    asked = [step.question for step in lesson.ask_tier(both).trail]
    assert "q5_gpai_model" not in asked, (
        f"asked {asked}: a system in an Annex III area is settled at q4, so q5 is never "
        "reached. The interview must return, not merely stop recording"
    )


def test_ask_tier_exactly_one_decisive_and_it_is_last():
    for sid in SYSTEMS:
        interview = lesson.ask_tier(sys_(sid))
        decisive = [step for step in interview.trail if step.decisive]
        assert len(decisive) == 1, (
            f"{sid}: {len(decisive)} decisive answers in "
            f"{[(s.question, s.decisive) for s in interview.trail]}. Exactly one question "
            "ends an interview"
        )
        assert decisive[0].question == interview.trail[-1].question, (
            f"{sid}: the decisive answer is {decisive[0].question!r} but the trail ends on "
            f"{interview.trail[-1].question!r} — nothing is asked after the tier is settled"
        )
        assert isinstance(decisive[0].decisive, bool), "decisive is a bool, not a truthy value"

        # The yes/no questions record a BOOLEAN, not the raw field that answered them. q3 is
        # the documented exception: it records the area string itself.
        for step in interview.trail:
            if step.question == "q3_annex_iii_area":
                continue
            if step.question == "q4_article_6_3":
                assert isinstance(step.answer, str), (
                    f"{sid}: q4 records the derogation reason id, got {step.answer!r}"
                )
                continue
            assert isinstance(step.answer, bool), (
                f"{sid}: {step.question} recorded {step.answer!r} "
                f"({type(step.answer).__name__}). The yes/no questions record "
                "bool(system.get(...)), not the raw field. Storing the free-text flag puts "
                "prose inside a sealed record, and then rewording it counts as a changed "
                "answer in the next diff"
            )


def test_ask_tier_q3_is_never_decisive_and_q4_records_the_reason():
    for sid in ("credit-scorer", "cv-ranker", "shift-note-tidier", "exam-prep-sorter"):
        trail = {step.question: step for step in lesson.ask_tier(sys_(sid)).trail}
        assert trail["q3_annex_iii_area"].decisive is False, (
            f"{sid}: q3 was marked decisive. Naming an Annex III area opens Article 6(3); it "
            "does not settle the tier, and treating it as decisive is exactly the "
            "'Annex III means high-risk, full stop' error"
        )
        assert trail["q3_annex_iii_area"].answer == SYSTEMS[sid]["annex_iii_area"], (
            f"{sid}: q3's recorded answer is the area itself, not a boolean — the trail has "
            "to say WHICH area was named"
        )
        want = lesson.derogation_assessment(sys_(sid))["reason"]
        assert trail["q4_article_6_3"].answer == want, (
            f"{sid}: q4 recorded {trail['q4_article_6_3'].answer!r}, expected {want!r}. The "
            "reason id is what makes the trail an explanation rather than a verdict"
        )

    no_area = lesson.ask_tier(sys_("demand-forecaster"))
    assert no_area.trail[2].answer is None, (
        "with no area named, q3's answer is None rather than a falsy string — use "
        "`system.get('annex_iii_area') or None`"
    )


def test_ask_tier_q6_is_always_decisive():
    for sid, want in (("demand-forecaster", False), ("policy-drafter", True)):
        last = lesson.ask_tier(sys_(sid)).trail[-1]
        assert last.question == "q6_transparency" and last.decisive is True, (
            f"{sid}: the trail ends on {last.question!r} decisive={last.decisive!r}. q6 is "
            "ALWAYS decisive — answering 'no' to the last question is what produces "
            "'minimal', so every trail ends on the question that produced its tier"
        )
        assert last.answer is want, (
            f"{sid}: q6 answered {last.answer!r}, expected {want!r} — it is the boolean OR "
            "of interacts_with_humans and generates_synthetic_content"
        )
    only_synthetic = sys_("demand-forecaster", generates_synthetic_content=True)
    assert lesson.ask_tier(only_synthetic).tier == "transparency_only", (
        "either flag on its own is enough for transparency_only"
    )


# --- exercise 3: obligations and dates --------------------------------------------------

def test_obligations_shape_and_cumulation():
    got = lesson.obligations_for("minimal", sys_("demand-forecaster"))
    assert set(got) == {"obligations", "deadlines", "binding_now", "next_deadline"}, (
        f"keys were {sorted(got)}; expected obligations, deadlines, binding_now, next_deadline"
    )
    assert got["obligations"] == ["art4_ai_literacy"], (
        f"got {got['obligations']}; minimal risk is one duty, not none"
    )

    chatty = sys_("credit-scorer", interacts_with_humans=True, generates_synthetic_content=True)
    got = lesson.obligations_for("high_risk_annex_iii", chatty)
    assert set(got["obligations"]) == {"art4_ai_literacy", "ch3_high_risk_annex_iii",
                                       "art50_inform_interaction", "art50_mark_synthetic"}, (
        f"got {got['obligations']}; obligations cumulate and the Article 50 duties are read "
        "off the SYSTEM's own flags. Being high-risk adds duties, it never absorbs them"
    )
    assert got["obligations"] == sorted(got["obligations"]), "return the ids sorted"
    assert set(got["deadlines"]) == set(got["obligations"]), (
        f"deadlines covers {sorted(got['deadlines'])} but obligations is "
        f"{got['obligations']} — one date per obligation, no more and no fewer"
    )

    # The same flags on an Annex I system: the transparency duties still attach.
    embedded = sys_("lift-door-sensor", interacts_with_humans=True)
    assert "art50_inform_interaction" in \
        lesson.obligations_for("high_risk_annex_i", embedded)["obligations"]


def test_obligations_derogated_is_not_minimal():
    got = lesson.obligations_for("annex_iii_derogated", sys_("shift-note-tidier"))
    assert set(got["obligations"]) == {"art4_ai_literacy", "art6_4_derogation_documentation",
                                       "art49_2_registration"}, (
        f"got {got['obligations']}; a derogated system is NOT a minimal one. Concluding under "
        "Article 6(3) that the system is not high-risk creates the Article 6(4) duty to "
        "document that assessment and the Article 49(2) duty to register it"
    )
    assert got["deadlines"]["art49_2_registration"] == \
        lesson.OBLIGATIONS["art49_2_registration"]["applies_from"], (
        "take every date from the OBLIGATIONS table rather than typing one"
    )
    derogated_and_chatty = sys_("shift-note-tidier", interacts_with_humans=True)
    assert "art50_inform_interaction" in \
        lesson.obligations_for("annex_iii_derogated", derogated_and_chatty)["obligations"], (
        "the derogation exits the high-risk route, not the Regulation: Article 50 is untouched"
    )


def test_obligations_prohibited_stops():
    got = lesson.obligations_for("prohibited", sys_("mood-meter", interacts_with_humans=True))
    assert got["obligations"] == ["art4_ai_literacy", "art5_prohibited_practice"], (
        f"got {got['obligations']}; once prohibited, stop. A system that must not be placed "
        "on the market does not also collect the duties of one that is — including the "
        "Article 50 duties its flags would otherwise trigger"
    )


def test_obligations_legacy_marking_boundary():
    early = sys_("policy-drafter", placed_on_market="2026-08-01")
    got = lesson.obligations_for("transparency_only", early)
    assert got["deadlines"]["art50_mark_synthetic"] == lesson.ART50_MARK_LEGACY, (
        f"placed 2026-08-01, one day before general application, so marking is due "
        f"{lesson.ART50_MARK_LEGACY}, not {got['deadlines']['art50_mark_synthetic']}"
    )
    on_the_day = sys_("policy-drafter", placed_on_market=lesson.GENERAL_APPLICATION)
    got = lesson.obligations_for("transparency_only", on_the_day)
    assert got["deadlines"]["art50_mark_synthetic"] == lesson.GENERAL_APPLICATION, (
        f"placed ON {lesson.GENERAL_APPLICATION} is not placed BEFORE it; the transitional "
        f"date is for systems already on the market, so this one gets "
        f"{lesson.GENERAL_APPLICATION} and you got "
        f"{got['deadlines']['art50_mark_synthetic']}"
    )
    assert lesson.obligations_for("transparency_only", sys_(
        "policy-drafter", placed_on_market=None))["deadlines"]["art50_mark_synthetic"] == \
        lesson.GENERAL_APPLICATION, (
        "an unknown placing date is not evidence of a legacy system: fall back to the "
        "standard date instead of crashing"
    )
    # The transitional rule touches marking and nothing else.
    assert got["deadlines"]["art50_inform_interaction"] == lesson.GENERAL_APPLICATION, (
        "only Article 50(2) marking has a transitional date; 50(1) does not"
    )


def test_obligations_binding_now_and_next_deadline():
    system = sys_("policy-drafter")               # dates: 2025-02-02, 2026-08-02, 2026-12-02
    got = lesson.obligations_for("transparency_only", system, AS_OF)
    assert got["binding_now"] == ["art4_ai_literacy", "art50_inform_interaction"], (
        f"got {got['binding_now']} as of {AS_OF.isoformat()}; binding_now is every obligation "
        "whose date is on or before as_of, sorted"
    )
    assert got["next_deadline"] == lesson.ART50_MARK_LEGACY, (
        f"got {got['next_deadline']!r}; the next deadline is the earliest date still ahead"
    )

    # as_of ON an obligation's own date: it binds today, and today is not still ahead.
    boundary = lesson.obligations_for("transparency_only", system,
                                      date.fromisoformat(lesson.GENERAL_APPLICATION))
    assert "art50_inform_interaction" in boundary["binding_now"], (
        "an obligation whose date IS as_of binds now — the comparison is <=, not <"
    )
    assert boundary["next_deadline"] == lesson.ART50_MARK_LEGACY, (
        f"got {boundary['next_deadline']!r}; a date equal to as_of is not still ahead, so "
        "next_deadline is the earliest date STRICTLY after as_of"
    )

    # TWO dates still ahead, so 'earliest' and 'latest' stop coinciding. On the transparency
    # system above exactly one date is ahead, and min() and max() agree on a one-item list.
    two_ahead = lesson.obligations_for(
        "high_risk_annex_iii", sys_("credit-scorer", generates_synthetic_content=True), AS_OF)
    assert sorted(d for d in two_ahead["deadlines"].values() if d > AS_OF.isoformat()) == \
        [lesson.ART50_MARK_LEGACY, "2027-12-02"], "fixture check: two dates must be ahead here"
    assert two_ahead["next_deadline"] == lesson.ART50_MARK_LEGACY, (
        f"got {two_ahead['next_deadline']!r}; this system owes marking on "
        f"{lesson.ART50_MARK_LEGACY} and Chapter III on 2027-12-02, so the NEXT deadline is "
        "the earlier of the two. next_deadline is the minimum of the dates ahead — reporting "
        "the last one tells a release manager the deadline that is furthest away"
    )

    past_everything = lesson.obligations_for("transparency_only", system, date(2029, 1, 1))
    assert past_everything["next_deadline"] is None, (
        f"got {past_everything['next_deadline']!r}; with nothing ahead, return None"
    )
    assert len(past_everything["binding_now"]) == len(past_everything["obligations"])


# --- exercise 4: the sealed record ------------------------------------------------------

def test_record_shape_and_json_round_trip():
    record = lesson.classification_record(sys_("cv-ranker"))
    assert set(record) == {"system_id", "as_of", "risk_tier", "question_trail", "derogation",
                           "obligations", "deadlines", "record_hash"}, (
        f"keys were {sorted(record)}; expected exactly the eight documented names"
    )
    assert record["system_id"] == "cv-ranker" and record["risk_tier"] == "high_risk_annex_iii"
    assert record["as_of"] == AS_OF.isoformat(), (
        f"as_of was {record['as_of']!r}; store the ISO string, not the date object"
    )
    assert json.loads(json.dumps(record)) == record, (
        "the record must survive a JSON round trip so it can be appended to the P01-L01 log "
        "— a NamedTuple, a set or a date object in it will not"
    )
    trail = record["question_trail"]
    assert [step["question"] for step in trail] == \
        [step.question for step in lesson.ask_tier(sys_("cv-ranker")).trail], (
        "the stored trail is the interview's trail, in the same order"
    )
    assert all(set(step) == {"question", "answer", "decisive"} for step in trail), (
        f"a trail step had keys {sorted(trail[0])}; each is a plain dict with exactly "
        "question, answer and decisive"
    )
    assert all(isinstance(step, dict) for step in trail), (
        "serialise the Answer tuples; a NamedTuple survives json.dumps as a LIST and then "
        "never compares equal to the dict anyone reads back"
    )


def test_record_seal_is_recomputable_and_covers_every_field():
    record = lesson.classification_record(sys_("shift-note-tidier"))
    digest = record["record_hash"]
    assert isinstance(digest, str) and len(digest) == 64 and \
        set(digest) <= set("0123456789abcdef"), (
        f"record_hash was {digest!r}; a SHA-256 hexdigest is 64 lowercase hex characters"
    )
    assert _seal(record) == digest, (
        "a third party must be able to recompute the seal: hash canonical_bytes of the "
        "record WITHOUT record_hash in it, then add the field. A digest computed over a "
        "record that already contains a digest can never be checked by anyone"
    )

    bare = {k: v for k, v in record.items() if k != "record_hash"}
    for field, mutate in (
        ("system_id", lambda r: r.update(system_id="someone-else")),
        ("as_of", lambda r: r.update(as_of="2027-01-01")),
        ("risk_tier", lambda r: r.update(risk_tier="minimal")),
        ("obligations", lambda r: r.update(obligations=["art4_ai_literacy"])),
        ("deadlines", lambda r: r["deadlines"].update(art4_ai_literacy="2099-01-01")),
        ("derogation", lambda r: r["derogation"].update(granted=False)),
        ("question_trail", lambda r: r["question_trail"][-1].update(
            answer=lesson.REASON_NOT_CLAIMED)),
    ):
        tampered = copy.deepcopy(bare)
        mutate(tampered)
        assert hashlib.sha256(lesson.canonical_bytes(tampered)).hexdigest() != digest, (
            f"editing {field!r} did not move the digest, so the seal does not cover it. Hash "
            "the whole record: a seal over the tier alone lets someone re-explain a "
            "classification after the fact and still present a valid-looking record"
        )

    # Two runs of the same input agree to the last character, or the record is not evidence.
    assert lesson.classification_record(sys_("shift-note-tidier"))["record_hash"] == digest
    assert lesson.classification_record(sys_("shift-note-tidier"),
                                        date(2027, 1, 1))["record_hash"] != digest, (
        "as_of is part of the record, so a different as_of is a different record"
    )


def test_record_derogation_present_only_when_asked():
    for sid in ("credit-scorer", "cv-ranker", "shift-note-tidier", "exam-prep-sorter"):
        got = lesson.classification_record(sys_(sid))["derogation"]
        want = lesson.derogation_assessment(sys_(sid))
        assert got == want, (
            f"{sid}: the record's derogation block is {got!r}; store the whole "
            "derogation_assessment dict, so the condition and the gaps travel with the record"
        )
    for sid in ("mood-meter", "lift-door-sensor", "house-lm", "policy-drafter",
                "demand-forecaster"):
        assert lesson.classification_record(sys_(sid))["derogation"] is None, (
            f"{sid}: no Annex III area was named, so q4 was never asked and there is no "
            "derogation to record. Running the assessment anyway records an opinion about a "
            "question nobody put"
        )


# --- exercise 5: the diff ---------------------------------------------------------------

def test_diff_reports_added_and_removed_with_their_dates():
    before = lesson.classification_record(sys_("shift-note-tidier"))
    after = lesson.classification_record(sys_("shift-note-tidier", performs_profiling=True,
                                              narrow_procedural_task=False))
    got = lesson.diff_records(before, after)
    assert set(got) == {"system_id", "tier_before", "tier_after", "tier_changed", "added",
                        "removed", "changed_answers", "requires_reassessment"}, (
        f"keys were {sorted(got)}; expected exactly the eight documented names"
    )
    assert (got["tier_before"], got["tier_after"]) == \
        ("annex_iii_derogated", "high_risk_annex_iii")
    assert got["tier_changed"] is True and got["requires_reassessment"] is True
    assert set(got["added"]) == {"ch3_high_risk_annex_iii"}, f"added {sorted(got['added'])}"
    assert set(got["removed"]) == {"art6_4_derogation_documentation", "art49_2_registration"}, (
        f"removed {sorted(got['removed'])}; obligations vanish as well as appear, and a diff "
        "that only reports additions hides the duty the pack just stopped owing"
    )
    assert got["added"]["ch3_high_risk_annex_iii"] == after["deadlines"]["ch3_high_risk_annex_iii"], (
        "an added obligation's value is its date in the AFTER record"
    )
    assert got["removed"]["art49_2_registration"] == before["deadlines"]["art49_2_registration"], (
        "a removed obligation's value is its date in the BEFORE record — it does not exist in "
        "`after` to be read from"
    )
    assert got["changed_answers"] == ["q4_article_6_3"], (
        f"got {got['changed_answers']}; q1, q2 and q3 answered identically here"
    )


def test_diff_catches_a_change_the_tier_hides():
    before = lesson.classification_record(sys_("policy-drafter"))
    after = lesson.classification_record(sys_("policy-drafter",
                                              generates_synthetic_content=False))
    got = lesson.diff_records(before, after)
    assert got["tier_changed"] is False, "both are transparency_only"
    assert set(got["removed"]) == {"art50_mark_synthetic"} and got["added"] == {}, (
        f"added {sorted(got['added'])}, removed {sorted(got['removed'])}"
    )
    assert got["requires_reassessment"] is True, (
        "an obligation vanished while the tier held. requires_reassessment is True on ANY "
        "obligation appearing or vanishing, not only on a tier change — a tier-only diff "
        "reports 'no change' here and the pack quietly loses a duty"
    )
    assert got["changed_answers"] == [], (
        f"got {got['changed_answers']}; this is the sharpest case in the lesson. q6 answers "
        "the OR of two flags, and the other one is still True, so the trail is byte for byte "
        "IDENTICAL while an obligation vanished. Report what the trails actually say — "
        "reverse-engineering changed_answers from the obligation diff would fake a "
        "disagreement the interview never recorded"
    )

    # The mirror of the case above, and the reason requires_reassessment is not just "did
    # anything move?": cv-ranker stops profiling and declares a significant risk instead. The
    # derogation is refused for a DIFFERENT reason, so q4's answer changes — and the tier,
    # the obligations and every date stay exactly where they were.
    was = lesson.classification_record(sys_("cv-ranker"))
    now = lesson.classification_record(sys_("cv-ranker", performs_profiling=False,
                                            significant_risk_of_harm=True))
    got = lesson.diff_records(was, now)
    assert got["changed_answers"] == ["q4_article_6_3"], (
        f"got {got['changed_answers']}; the derogation reason moved from "
        f"{lesson.REASON_PROFILING_OVERRIDE!r} to {lesson.REASON_SIGNIFICANT_RISK!r}"
    )
    assert got["added"] == {} and got["removed"] == {} and got["tier_changed"] is False, (
        f"added {sorted(got['added'])}, removed {sorted(got['removed'])}; both refusals land "
        "on high_risk_annex_iii with the same duties on the same dates"
    )
    assert got["requires_reassessment"] is False, (
        "requires_reassessment is a tier change OR an obligation appearing OR one vanishing — "
        "and here none of the three happened. A changed answer alone does not trigger it: "
        "sending a release manager back through a full reassessment because a reason id moved "
        "is how a control gets switched off"
    )

    # A question asked in only one trail is not a changed answer.
    plain = lesson.classification_record(sys_("demand-forecaster"))
    grew = lesson.classification_record(sys_("demand-forecaster",
                                             annex_iii_area="education (Annex III point 3(b))"))
    got = lesson.diff_records(plain, grew)
    assert got["changed_answers"] == ["q3_annex_iii_area"], (
        f"got {got['changed_answers']}; q4 appears in only one trail and q5 and q6 in only "
        "the other, so neither is a CHANGED answer. Compare the questions the two trails "
        "share — the earlier question that changed is what sent the interview down a "
        "different path, and it is the one worth reporting"
    )
    assert got["requires_reassessment"] is True


def test_diff_identity_and_system_mismatch():
    record = lesson.classification_record(sys_("cv-ranker"))
    same = lesson.diff_records(record, copy.deepcopy(record))
    assert same["added"] == {} and same["removed"] == {} and same["changed_answers"] == [], (
        f"a record diffed against itself moves nothing, got {same}"
    )
    assert same["tier_changed"] is False and same["requires_reassessment"] is False
    assert same["system_id"] == "cv-ranker"

    other = lesson.classification_record(sys_("house-lm"))
    try:
        lesson.diff_records(record, other)
    except ValueError:
        pass
    except Exception as exc:
        raise AssertionError(
            f"diffing two different systems raised {type(exc).__name__}; raise ValueError"
        ) from None
    else:
        raise AssertionError(
            "diffing two different systems returned a report. It would look entirely "
            "plausible and mean nothing: compare system_id first and raise ValueError"
        )


# --- exercise 6: the panel --------------------------------------------------------------

def test_disagreement_groups_and_majority():
    got = lesson.disagreement_report(copy.deepcopy(lesson.PANEL))
    assert set(got) == {"system_id", "unanimous", "tiers", "majority_tier", "pivot_question",
                        "pivot_answers"}, (
        f"keys were {sorted(got)}; expected exactly the six documented names"
    )
    assert got["system_id"] == "shift-allocator" and got["unanimous"] is False
    assert got["tiers"] == {"annex_iii_derogated": ["analyst_a"],
                            "high_risk_annex_iii": ["analyst_b", "analyst_c"],
                            "minimal": ["analyst_d"]}, (
        f"got {got['tiers']}; group by the tier reached, analyst ids sorted within each, and "
        "include no tier nobody reached — a pre-seeded dict of all seven tiers reports six "
        "empty verdicts"
    )
    assert got["majority_tier"] == "high_risk_annex_iii", (
        f"got {got['majority_tier']!r}; two of the four reached it"
    )

    # On the shipped PANEL the most COMMON tier is also the most SEVERE tier anyone reached,
    # so "take the severest" scores the same as "count them". Here it does not: three of the
    # four say minimal, one says high-risk. TIER_SEVERITY is the TIE-BREAK, not the rule.
    lopsided = {"a1": copy.deepcopy(lesson.PANEL["analyst_d"]),
                "a2": copy.deepcopy(lesson.PANEL["analyst_d"]),
                "a3": copy.deepcopy(lesson.PANEL["analyst_d"]),
                "a4": copy.deepcopy(lesson.PANEL["analyst_b"])}
    got = lesson.disagreement_report(lopsided)
    assert got["tiers"] == {"minimal": ["a1", "a2", "a3"],
                            "high_risk_annex_iii": ["a4"]}, f"got {got['tiers']}"
    assert got["majority_tier"] == "minimal", (
        f"got {got['majority_tier']!r}; three of the four analysts reached 'minimal' and one "
        "reached 'high_risk_annex_iii'. majority_tier is the tier the MOST analysts reached; "
        "severity only breaks a tie. Returning the severest tier anyone reached is a "
        "different report — it is the answer to 'what is the worst case', and it makes a "
        "lone dissenter outvote a panel"
    )
    assert got["unanimous"] is False and got["pivot_question"] == "q3_annex_iii_area"

    # Insertion order deliberately the reverse of sorted order: on the shipped PANEL the two
    # happen to coincide, so a report that never sorts passes by accident.
    unsorted_panel = {"zulu": copy.deepcopy(lesson.PANEL["analyst_b"]),
                      "alpha": copy.deepcopy(lesson.PANEL["analyst_c"])}
    got = lesson.disagreement_report(unsorted_panel)
    assert got["tiers"] == {"high_risk_annex_iii": ["alpha", "zulu"]}, (
        f"got {got['tiers']}; sort the analyst ids within each tier. Leaving them in "
        "insertion order makes the report depend on who filled the form in first, which is "
        "the same defect the tie-break rule exists to prevent"
    )


def test_disagreement_pivot_is_the_earliest_split():
    got = lesson.disagreement_report(copy.deepcopy(lesson.PANEL))
    assert got["pivot_question"] == "q3_annex_iii_area", (
        f"got {got['pivot_question']!r}. Three of the four analysts are arguing about "
        "profiling at q4, but analyst_d never reaches q4: they answered q3 differently. Walk "
        "QUESTION_IDS in order and stop at the FIRST question the panel splits on — "
        "everything after it is downstream of an unsettled question"
    )
    assert set(got["pivot_answers"]) == set(lesson.PANEL), (
        f"pivot_answers covered {sorted(got['pivot_answers'])}; every analyst was asked q3"
    )
    assert got["pivot_answers"]["analyst_d"] is None, (
        "analyst_d's answer to q3 is None, and that is the answer that splits the panel"
    )

    # Same panel minus the analyst who split at q3: the pivot moves DOWN to q4, not up.
    later = {k: copy.deepcopy(v) for k, v in lesson.PANEL.items() if k != "analyst_d"}
    got = lesson.disagreement_report(later)
    assert got["pivot_question"] == "q4_article_6_3", (
        f"got {got['pivot_question']!r}; with analyst_d gone, q1 to q3 are unanimous and the "
        "first split is the derogation reason at q4"
    )
    assert set(got["pivot_answers"]) == {"analyst_a", "analyst_b", "analyst_c"}
    assert got["pivot_answers"]["analyst_a"] == lesson.REASON_GRANTED, (
        "the pivot answers are the recorded trail answers, so for q4 they are reason ids — "
        "which is what tells the panel what to go and settle"
    )


def test_disagreement_tie_breaks_by_severity_not_insertion_order():
    a, b = copy.deepcopy(lesson.PANEL["analyst_a"]), copy.deepcopy(lesson.PANEL["analyst_b"])
    for panel in ({"x_first": a, "y_second": b}, {"y_second": b, "x_first": a}):
        got = lesson.disagreement_report(panel)
        assert got["majority_tier"] == "high_risk_annex_iii", (
            f"with insertion order {list(panel)} the majority came out "
            f"{got['majority_tier']!r}. One analyst each: break the tie by TIER_SEVERITY, "
            "which ranks high_risk_annex_iii above annex_iii_derogated. Counter.most_common() "
            "breaks ties by insertion order, which makes the verdict depend on who filled "
            "the form in first"
        )
    # A three-way tie, to prove the rule is severity and not 'the first one seen'.
    three = {"p": copy.deepcopy(lesson.PANEL["analyst_d"]),
             "q": copy.deepcopy(lesson.PANEL["analyst_a"]),
             "r": copy.deepcopy(lesson.PANEL["analyst_b"])}
    assert lesson.disagreement_report(three)["majority_tier"] == "high_risk_annex_iii", (
        "in a three-way tie the most severe tier wins"
    )


def test_disagreement_unanimous_and_system_mismatch():
    solo = {"only": copy.deepcopy(lesson.PANEL["analyst_a"])}
    got = lesson.disagreement_report(solo)
    assert got["unanimous"] is True, "one analyst cannot disagree with anyone"
    assert got["pivot_question"] is None and got["pivot_answers"] == {}, (
        f"got pivot {got['pivot_question']!r} / {got['pivot_answers']}; with no split there "
        "is no pivot, and pivot_answers is empty rather than a copy of everyone's answers"
    )
    assert got["majority_tier"] == "annex_iii_derogated" and \
        got["tiers"] == {"annex_iii_derogated": ["only"]}

    agreeing = {"m": copy.deepcopy(lesson.PANEL["analyst_b"]),
                "n": copy.deepcopy(lesson.PANEL["analyst_b"])}
    got = lesson.disagreement_report(agreeing)
    assert got["unanimous"] is True and got["pivot_question"] is None, (
        "two identical descriptions split on nothing"
    )

    # The same tier for two different reasons: agreement on the outcome, not on the reasoning.
    right_answer_wrong_reason = {"b": copy.deepcopy(lesson.PANEL["analyst_b"]),
                                 "c": copy.deepcopy(lesson.PANEL["analyst_c"])}
    got = lesson.disagreement_report(right_answer_wrong_reason)
    assert got["unanimous"] is True, (
        f"got unanimous={got['unanimous']!r}; both analysts reached high_risk_annex_iii. "
        "unanimous is about the TIER — do not compute it as 'there is no pivot', because a "
        "panel can split on a question and still land on the same tier"
    )
    assert got["pivot_question"] == "q4_article_6_3", (
        f"got pivot {got['pivot_question']!r}; they agree on the tier and disagree on WHY — "
        "one says the system profiles, the other says it poses a significant risk. A pivot "
        "under a unanimous verdict is worth reporting: the two reasons imply different fixes"
    )
    assert set(got["pivot_answers"].values()) == {lesson.REASON_PROFILING_OVERRIDE,
                                                  lesson.REASON_SIGNIFICANT_RISK}
    assert got["tiers"] == {"high_risk_annex_iii": ["b", "c"]}, (
        f"got {got['tiers']}; the analyst ids within a tier are sorted"
    )

    try:
        lesson.disagreement_report({"a": sys_("credit-scorer"), "b": sys_("house-lm")})
    except ValueError:
        pass
    except Exception as exc:
        raise AssertionError(
            f"a panel over two different systems raised {type(exc).__name__}; raise ValueError"
        ) from None
    else:
        raise AssertionError(
            "two descriptions of two DIFFERENT systems are not a panel. Without the id check "
            "the report is a confident answer to a question nobody asked"
        )


RUBRIC = [
    (test_derogation_shape_and_not_claimed, 3,
     "an unclaimed derogation fails as REASON_NOT_CLAIMED with no documentation gaps — the "
     "Article 6(4) duty is created by the claim"),
    (test_derogation_profiling_overrides_everything, 4,
     "test performs_profiling BEFORE the conditions, the risk limb and the documentation; it "
     "is unconditional, and still report the condition the provider met"),
    (test_derogation_condition_order_and_none, 3,
     "walk DEROGATION_CONDITIONS in order and take the FIRST truthy field; test the risk limb "
     "and the conditions before the paperwork"),
    (test_derogation_documentation_gaps, 4,
     "four duties, sorted: falsy assessment_documented, a justification blank after .strip(), "
     "documented_on not STRICTLY before placed_on_market (or either missing), and falsy "
     "registered_in_eu_database"),
    (test_ask_tier_minimality_and_order, 4,
     "return as soon as a question decides; ask in QUESTIONS order and never ask a question "
     "after the tier is settled"),
    (test_ask_tier_exactly_one_decisive_and_it_is_last, 3,
     "exactly one Answer has decisive=True, and it is the last entry in the trail"),
    (test_ask_tier_q3_is_never_decisive_and_q4_records_the_reason, 3,
     "q3 records the area itself and is never decisive; q4 records "
     "derogation_assessment(system)['reason'] and always is"),
    (test_ask_tier_q6_is_always_decisive, 2,
     "q6 is always decisive: True -> transparency_only, False -> minimal, so 'minimal' is an "
     "answered question rather than a fall-through"),
    (test_obligations_shape_and_cumulation, 3,
     "read the Article 50 duties off the system's own flags, not off the tier, and return one "
     "date per obligation"),
    (test_obligations_derogated_is_not_minimal, 3,
     "the annex_iii_derogated tier owes art6_4_derogation_documentation and "
     "art49_2_registration"),
    (test_obligations_prohibited_stops, 2,
     "on the prohibited tier, add art5_prohibited_practice and return before anything else"),
    (test_obligations_legacy_marking_boundary, 3,
     "art50_mark_synthetic takes ART50_MARK_LEGACY only when placed_on_market is STRICTLY "
     "before GENERAL_APPLICATION; a missing date falls back to the standard one"),
    (test_obligations_binding_now_and_next_deadline, 3,
     "binding_now is date <= as_of; next_deadline is the earliest date STRICTLY after as_of, "
     "or None"),
    (test_record_shape_and_json_round_trip, 3,
     "serialise each Answer to a plain dict and store as_of as an ISO string, so the record "
     "round-trips through JSON"),
    (test_record_seal_is_recomputable_and_covers_every_field, 4,
     "hash canonical_bytes of the record WITHOUT record_hash, then add the field; the digest "
     "must cover every other key"),
    (test_record_derogation_present_only_when_asked, 2,
     "store the whole derogation_assessment dict when q4 was asked, and None when it was not"),
    (test_diff_reports_added_and_removed_with_their_dates, 4,
     "added maps to the date in `after`, removed to the date in `before`; changed_answers "
     "covers questions in BOTH trails whose answer differs"),
    (test_diff_catches_a_change_the_tier_hides, 3,
     "requires_reassessment is True on any obligation appearing OR vanishing, not only on a "
     "tier change"),
    (test_diff_identity_and_system_mismatch, 2,
     "raise ValueError when the two records' system_id differ"),
    (test_disagreement_groups_and_majority, 3,
     "group analysts by the tier they reached, ids sorted, tiers nobody reached omitted"),
    (test_disagreement_pivot_is_the_earliest_split, 4,
     "walk QUESTION_IDS in order and take the first question whose answers differ among the "
     "analysts who were ASKED it"),
    (test_disagreement_tie_breaks_by_severity_not_insertion_order, 2,
     "break a tie with TIER_SEVERITY order, never with Counter.most_common()"),
    (test_disagreement_unanimous_and_system_mismatch, 2,
     "no split means pivot_question None and pivot_answers {}; differing ids raise ValueError"),
]
